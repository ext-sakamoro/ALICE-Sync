#![allow(
    clippy::significant_drop_tightening,
    clippy::significant_drop_in_scrutinee,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc
)]

//! Transport Abstraction + UDP/TCP/Channel implementations
//!
//! Provides a unified `SyncTransport` trait for sending and receiving
//! serialized messages over any network substrate.

use std::net::SocketAddr;
use tokio::net::{TcpListener, TcpStream, UdpSocket};
use tokio::sync::mpsc;

use crate::protocol::Message;
use crate::reliability::{
    DeliveryFailure, PacketError, ReassemblyBuffer, ReliableEndpoint, Tokens, HEADER_SIZE,
    MAX_PAYLOAD,
};

/// Received envelope: payload + source address
#[derive(Debug, Clone)]
pub struct Envelope {
    /// Deserialized message
    pub message: Message,
    /// Source peer address
    pub from: SocketAddr,
}

/// Errors from the transport layer
#[derive(Debug)]
pub enum TransportError {
    Io(std::io::Error),
    Codec(String),
    Closed,
    /// A received message refused by the session's handshake (ALICE-Law G7)
    Handshake(crate::protocol::HandshakeRefused),
    /// No certificate is held for the address, so it cannot be dialled (the secure transport dials only a
    /// device whose certificate it holds)
    NoPeerCert(std::net::SocketAddr),
    /// A packet to `peer` was never acknowledged, or was pushed out of the send window: what it carried
    /// may not have arrived
    Undelivered {
        peer: SocketAddr,
        failure: DeliveryFailure,
    },
}

impl From<std::io::Error> for TransportError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl std::fmt::Display for TransportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "transport I/O: {e}"),
            Self::Codec(s) => write!(f, "transport codec: {s}"),
            Self::Closed => write!(f, "transport closed"),
            Self::Handshake(r) => write!(f, "handshake refused: {r}"),
            Self::NoPeerCert(a) => write!(f, "no peer certificate for {a}"),
            Self::Undelivered { peer, failure } => write!(f, "not delivered to {peer}: {failure}"),
        }
    }
}

impl std::error::Error for TransportError {}

/// Result type for transport operations
pub type TransportResult<T> = std::result::Result<T, TransportError>;

// ============================================================================
// SyncTransport trait
// ============================================================================

/// Abstract transport for P2P synchronization.
///
/// Implementations handle the actual network I/O.
/// Messages are serialized with bitcode for minimal bandwidth.
/// Abstract transport for P2P synchronization.
///
/// Implementations handle the actual network I/O.
/// Messages are serialized with bitcode for minimal bandwidth.
///
/// All async methods return `Send` futures for `tokio::spawn` compatibility.
pub trait SyncTransport: Send + Sync {
    /// Send a message to a specific peer
    fn send_to(
        &self,
        message: &Message,
        addr: SocketAddr,
    ) -> impl std::future::Future<Output = TransportResult<()>> + Send;

    /// Receive the next message (blocking until available)
    fn recv(&self) -> impl std::future::Future<Output = TransportResult<Envelope>> + Send;

    /// Broadcast a message to all known peers
    fn broadcast(
        &self,
        message: &Message,
        peers: &[SocketAddr],
    ) -> impl std::future::Future<Output = TransportResult<()>> + Send {
        async move {
            for peer in peers {
                self.send_to(message, *peer).await?;
            }
            Ok(())
        }
    }

    /// Send application bytes to a specific peer; the receiver gets them as [`Frame::Bytes`] from
    /// [`Self::recv_frame`] and never decodes them as a Sync message
    ///
    /// The default refuses: a transport that cannot carry bytes beside Sync messages returns
    /// [`TransportError::Io`] of kind [`std::io::ErrorKind::Unsupported`] and sends nothing.
    fn send_bytes_to(
        &self,
        bytes: &[u8],
        addr: SocketAddr,
    ) -> impl std::future::Future<Output = TransportResult<()>> + Send {
        let _ = (bytes, addr);
        async {
            Err(TransportError::Io(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "this transport does not carry application bytes",
            )))
        }
    }

    /// Receive the next Sync message or block of application bytes, in the order they arrived
    ///
    /// The default is [`Self::recv`] (a transport without bytes only ever returns
    /// [`Frame::Message`]).
    fn recv_frame(&self) -> impl std::future::Future<Output = TransportResult<Frame>> + Send {
        async { self.recv().await.map(Frame::Message) }
    }

    /// Local bound address
    fn local_addr(&self) -> TransportResult<SocketAddr>;
}

/// What [`SyncTransport::recv_frame`] received
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Frame {
    /// A Sync message
    Message(Envelope),
    /// Application bytes, as sent with [`SyncTransport::send_bytes_to`]
    Bytes {
        /// The bytes
        bytes: Vec<u8>,
        /// Source peer address
        from: SocketAddr,
    },
}

impl Frame {
    /// The peer the frame came from
    #[must_use]
    pub const fn from(&self) -> SocketAddr {
        match self {
            Self::Message(envelope) => envelope.from,
            Self::Bytes { from, .. } => *from,
        }
    }
}

// ============================================================================
// UDP Transport
// ============================================================================

/// How often [`UdpTransport::recv`] sends the due acknowledgements and retransmissions while it waits
const MAINTENANCE: std::time::Duration = std::time::Duration::from_millis(25);

/// How many peer addresses a [`UdpTransport`] keeps state for by default ([`UdpOptions`])
pub const MAX_UDP_PEERS: usize = 1024;

/// Options of [`UdpTransport::bind_with`]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UdpOptions {
    /// How many peer addresses have state (at least 1): a packet from or to a new address beyond that pushes
    /// out the least recently used peer, whose unacknowledged packets are returned from `recv` as
    /// [`TransportError::Undelivered`] ([`DeliveryFailure::Abandoned`])
    pub max_peers: usize,
    /// Where the transport's link tokens start ([`crate::reliability::Tokens::starting_at`]): a caller that
    /// numbers its process incarnations passes a start past the previous incarnation's tokens
    pub token_start: u64,
}

impl Default for UdpOptions {
    fn default() -> Self {
        Self {
            max_peers: MAX_UDP_PEERS,
            token_start: 0,
        }
    }
}

/// The peers' state and the transport's link tokens, under one lock
struct Peers {
    links: std::collections::BTreeMap<SocketAddr, PeerLink>,
    tokens: Tokens,
}

/// One peer's reliability state and reassembly
struct PeerLink {
    reliability: ReliableEndpoint,
    reassembly: ReassemblyBuffer,
    /// when a packet was last sent to or received from the peer
    last_used: std::time::Instant,
}

impl PeerLink {
    fn new(now: std::time::Instant) -> Self {
        Self {
            reliability: ReliableEndpoint::new(),
            reassembly: ReassemblyBuffer::new(),
            last_used: now,
        }
    }
}

/// The link of `addr`, made when it has none; a new link beyond `max` pushes out the least recently used one,
/// whose unacknowledged packets are returned as delivery failures
fn link_of<'a>(
    peers: &'a mut std::collections::BTreeMap<SocketAddr, PeerLink>,
    addr: SocketAddr,
    max: usize,
    evicted: &mut Vec<(SocketAddr, DeliveryFailure)>,
) -> &'a mut PeerLink {
    let now = crate::clock::now();
    if !peers.contains_key(&addr) {
        while peers.len() >= max {
            let Some(oldest) = peers
                .iter()
                .min_by_key(|(a, l)| (l.last_used, **a))
                .map(|(a, _)| *a)
            else {
                break;
            };
            if let Some(mut gone) = peers.remove(&oldest) {
                evicted.extend(gone.reliability.abandon().into_iter().map(|f| (oldest, f)));
            }
        }
    }
    let link = peers.entry(addr).or_insert_with(|| PeerLink::new(now));
    link.last_used = now;
    link
}

/// UDP-based transport with reliability layer.
///
/// Best for: low-latency real-time sync (games, `IoT`, live collaboration).
///
/// Each peer address has its own reliability state. While [`SyncTransport::recv`] waits it acknowledges
/// what was received (also when this side sends nothing else) and retransmits what was not acknowledged; a
/// packet that is never acknowledged, or that is pushed out of the send window, is returned from `recv` as
/// [`TransportError::Undelivered`] (a session reports it as [`crate::session::SessionEvent::DeliveryFailed`]).
/// A datagram that is not a packet of this wire version is dropped; the first from each address is returned
/// as [`TransportError::Handshake`] with [`crate::protocol::HandshakeRefused::WireMismatch`].
///
/// The traffic to a peer goes on a link the peer opens ([`crate::reliability`]): the first message to an
/// address waits for the peer's answer, which `recv` takes, so a side that only sends must still run `recv`.
/// A peer that dropped the link (its peer limit, or a restart) asks for a new one, and what it had not
/// acknowledged is sent again on it. Plain UDP does not authenticate a peer: a datagram forged with its
/// address can reopen its link (the secure transport authenticates every datagram).
pub struct UdpTransport {
    socket: UdpSocket,
    peers: tokio::sync::Mutex<Peers>,
    failures: tokio::sync::Mutex<std::collections::VecDeque<(SocketAddr, DeliveryFailure)>>,
    /// Peers already reported as sending another wire version, oldest first (at most `max_peers`)
    other_wire: tokio::sync::Mutex<std::collections::VecDeque<SocketAddr>>,
    /// How many peer addresses have state (links, and the other-wire reports)
    max_peers: usize,
}

impl UdpTransport {
    /// Bind to a local address
    pub async fn bind(addr: &str) -> TransportResult<Self> {
        Self::bind_with(addr, UdpOptions::default()).await
    }

    /// Bind with [`UdpOptions`]
    pub async fn bind_with(addr: &str, options: UdpOptions) -> TransportResult<Self> {
        let socket = UdpSocket::bind(addr).await?;
        Ok(Self {
            socket,
            peers: tokio::sync::Mutex::new(Peers {
                links: std::collections::BTreeMap::new(),
                tokens: Tokens::starting_at(options.token_start),
            }),
            failures: tokio::sync::Mutex::new(std::collections::VecDeque::new()),
            other_wire: tokio::sync::Mutex::new(std::collections::VecDeque::new()),
            max_peers: options.max_peers.max(1),
        })
    }

    /// How many peer addresses have reliability state now (at most the limit)
    pub async fn peer_count(&self) -> usize {
        self.peers.lock().await.links.len()
    }

    /// Data packets of links this transport did not hold (answered with a reopen request), over all peers
    pub async fn stale_frames(&self) -> u64 {
        self.peers
            .lock()
            .await
            .links
            .values()
            .map(|l| l.reliability.stale_frames)
            .sum()
    }

    async fn note_evicted(&self, evicted: Vec<(SocketAddr, DeliveryFailure)>) {
        if !evicted.is_empty() {
            self.failures.lock().await.extend(evicted);
        }
    }

    /// Acknowledged and lost packet counts over all peers, and the loss rate
    pub async fn reliability_stats(&self) -> (u64, u64, f64) {
        let peers = self.peers.lock().await;
        let (acked, lost) = peers.links.values().fold((0, 0), |(a, l), p| {
            (
                a + p.reliability.packets_acked,
                l + p.reliability.packets_lost,
            )
        });
        let total = acked + lost;
        #[allow(clippy::cast_precision_loss)]
        let rate = if total == 0 {
            0.0
        } else {
            lost as f64 / total as f64
        };
        (acked, lost, rate)
    }

    /// Collect and send retransmissions to `peer` now (`recv` also does this while it waits)
    pub async fn retransmit(&self, peer: SocketAddr) -> TransportResult<()> {
        let packets = {
            let mut peers = self.peers.lock().await;
            let Some(link) = peers.links.get_mut(&peer) else {
                return Ok(());
            };
            let packets = link.reliability.collect_retransmits();
            self.note_failures(peer, &mut link.reliability).await;
            packets
        };
        for packet in packets {
            self.socket.send_to(&packet, peer).await?;
        }
        Ok(())
    }

    async fn note_failures(&self, peer: SocketAddr, reliability: &mut ReliableEndpoint) {
        let found = reliability.take_failures();
        if !found.is_empty() {
            let mut q = self.failures.lock().await;
            q.extend(found.into_iter().map(|f| (peer, f)));
        }
    }

    /// Send every peer's due acknowledgements and retransmissions, and note its delivery failures
    async fn maintain(&self) -> TransportResult<()> {
        let mut out = Vec::new();
        {
            let mut peers = self.peers.lock().await;
            for (addr, link) in peers.links.iter_mut() {
                for p in link.reliability.take_acks() {
                    out.push((*addr, p));
                }
                for p in link.reliability.collect_retransmits() {
                    out.push((*addr, p));
                }
                link.reassembly.gc();
                self.note_failures(*addr, &mut link.reliability).await;
            }
        }
        for (addr, p) in out {
            self.socket.send_to(&p, addr).await?;
        }
        Ok(())
    }
}

impl SyncTransport for UdpTransport {
    async fn send_to(&self, message: &Message, addr: SocketAddr) -> TransportResult<()> {
        let payload = message.to_compact_bytes();
        let packets = {
            let mut peers = self.peers.lock().await;
            let Peers { links, tokens } = &mut *peers;
            let mut evicted = Vec::new();
            let link = link_of(links, addr, self.max_peers, &mut evicted);
            let packets: Vec<Vec<u8>> = if payload.len() <= MAX_PAYLOAD - HEADER_SIZE {
                link.reliability.wrap_outgoing(&payload, tokens)
            } else {
                let msg_id = link.reliability.next_message_id();
                crate::reliability::fragment_payload(msg_id, &payload)
                    .iter()
                    .flat_map(|frag| link.reliability.wrap_outgoing(frag, tokens))
                    .collect()
            };
            self.note_failures(addr, &mut link.reliability).await;
            self.note_evicted(evicted).await;
            packets
        };
        for packet in packets {
            self.socket.send_to(&packet, addr).await?;
        }
        Ok(())
    }

    async fn recv(&self) -> TransportResult<Envelope> {
        let mut buf = vec![0u8; 65535];
        let mut next = tokio::time::Instant::from_std(crate::clock::now()) + MAINTENANCE;
        loop {
            if let Some((peer, failure)) = self.failures.lock().await.pop_front() {
                return Err(TransportError::Undelivered { peer, failure });
            }
            let (len, from) = tokio::select! {
                r = self.socket.recv_from(&mut buf) => r?,
                () = tokio::time::sleep_until(next) => {
                    next = tokio::time::Instant::from_std(crate::clock::now()) + MAINTENANCE;
                    self.maintain().await?;
                    continue;
                }
            };
            let packet = &buf[..len];
            match crate::reliability::classify(packet) {
                Ok(_) => {}
                Err(PacketError::OtherWire) => {
                    let mut seen = self.other_wire.lock().await;
                    if seen.contains(&from) {
                        continue;
                    }
                    if seen.len() >= self.max_peers {
                        seen.pop_front();
                    }
                    seen.push_back(from);
                    return Err(TransportError::Handshake(
                        crate::protocol::HandshakeRefused::WireMismatch { peer: from },
                    ));
                }
                // this wire's magic with an unknown kind or a wrong length: dropped, no state, no report
                Err(PacketError::Malformed) => continue,
            }
            let (complete, replies) = {
                let mut peers = self.peers.lock().await;
                let Peers { links, tokens } = &mut *peers;
                let mut evicted = Vec::new();
                let link = link_of(links, from, self.max_peers, &mut evicted);
                self.note_evicted(evicted).await;
                let mut got = link.reliability.unwrap_incoming(packet, tokens);
                // what was received is acknowledged now (also by the maintenance while recv waits): the
                // sender then holds as little unacknowledged as it can if this side drops the link later
                if got.payload.is_some() {
                    got.replies.extend(link.reliability.take_acks());
                }
                self.note_failures(from, &mut link.reliability).await;
                // a whole message decodes as it is; otherwise it is a fragment
                let complete = got.payload.and_then(|payload| {
                    if Message::from_compact_bytes(&payload).is_some() {
                        Some(payload)
                    } else {
                        link.reassembly.feed(&payload)
                    }
                });
                (complete, got.replies)
            };
            for reply in replies {
                self.socket.send_to(&reply, from).await?;
            }
            if let Some(msg) = complete.and_then(|c| Message::from_compact_bytes(&c)) {
                return Ok(Envelope { message: msg, from });
            }
            // Could not decode — skip
        }
    }

    fn local_addr(&self) -> TransportResult<SocketAddr> {
        self.socket.local_addr().map_err(TransportError::Io)
    }
}

// ============================================================================
// TCP Transport
// ============================================================================

/// TCP-based transport for reliable ordered delivery.
///
/// Best for: business applications, configuration sync, event sourcing.
pub struct TcpTransport {
    /// Incoming connection listener
    listener: TcpListener,
    /// Connected peers: addr → write half
    peers: tokio::sync::Mutex<std::collections::BTreeMap<SocketAddr, TcpPeerWriter>>,
    /// Received message channel
    rx: tokio::sync::Mutex<mpsc::Receiver<Envelope>>,
    /// Sender for spawned read tasks
    tx: mpsc::Sender<Envelope>,
}

struct TcpPeerWriter {
    writer: tokio::sync::Mutex<tokio::io::WriteHalf<TcpStream>>,
}

impl TcpTransport {
    /// Bind and start listening
    pub async fn bind(addr: &str) -> TransportResult<Self> {
        let listener = TcpListener::bind(addr).await?;
        let (tx, rx) = mpsc::channel(1024);

        Ok(Self {
            listener,
            peers: tokio::sync::Mutex::new(std::collections::BTreeMap::new()),
            rx: tokio::sync::Mutex::new(rx),
            tx,
        })
    }

    /// Accept a new incoming connection
    pub async fn accept(&self) -> TransportResult<SocketAddr> {
        let (stream, addr) = self.listener.accept().await?;
        self.add_peer_stream(stream, addr).await;
        Ok(addr)
    }

    /// Connect to a remote peer
    pub async fn connect(&self, addr: SocketAddr) -> TransportResult<()> {
        let stream = TcpStream::connect(addr).await?;
        let local = stream.local_addr()?;
        self.add_peer_stream(stream, addr).await;
        let _ = local;
        Ok(())
    }

    async fn add_peer_stream(&self, stream: TcpStream, addr: SocketAddr) {
        use tokio::io::AsyncReadExt;

        let (read_half, write_half) = tokio::io::split(stream);

        self.peers.lock().await.insert(
            addr,
            TcpPeerWriter {
                writer: tokio::sync::Mutex::new(write_half),
            },
        );

        // Spawn read task
        let tx = self.tx.clone();
        tokio::spawn(async move {
            let mut reader = read_half;
            let mut len_buf = [0u8; 4];

            loop {
                // Read length-prefixed message: [len:4][payload:len]
                if reader.read_exact(&mut len_buf).await.is_err() {
                    break;
                }
                let len = u32::from_le_bytes(len_buf) as usize;
                if len > 1_000_000 {
                    break; // sanity limit
                }

                let mut payload = vec![0u8; len];
                if reader.read_exact(&mut payload).await.is_err() {
                    break;
                }

                if let Some(msg) = Message::from_compact_bytes(&payload) {
                    let _ = tx
                        .send(Envelope {
                            message: msg,
                            from: addr,
                        })
                        .await;
                }
            }
        });
    }
}

impl SyncTransport for TcpTransport {
    async fn send_to(&self, message: &Message, addr: SocketAddr) -> TransportResult<()> {
        use tokio::io::AsyncWriteExt;

        let peers = self.peers.lock().await;
        let peer = peers.get(&addr).ok_or(TransportError::Closed)?;

        let payload = message.to_compact_bytes();
        let len = (payload.len() as u32).to_le_bytes();

        let mut writer = peer.writer.lock().await;
        writer.write_all(&len).await?;
        writer.write_all(&payload).await?;
        writer.flush().await?;
        Ok(())
    }

    async fn recv(&self) -> TransportResult<Envelope> {
        self.rx
            .lock()
            .await
            .recv()
            .await
            .ok_or(TransportError::Closed)
    }

    fn local_addr(&self) -> TransportResult<SocketAddr> {
        self.listener.local_addr().map_err(TransportError::Io)
    }
}

// ============================================================================
// Channel Transport (in-process, for testing)
// ============================================================================

/// In-process transport using tokio mpsc channels.
///
/// No actual networking — used for unit/integration testing.
pub struct ChannelTransport {
    local_addr: SocketAddr,
    tx: mpsc::Sender<(Envelope, SocketAddr)>,
    rx: tokio::sync::Mutex<mpsc::Receiver<(Envelope, SocketAddr)>>,
}

impl ChannelTransport {
    /// Create a pair of connected channel transports for testing
    #[must_use]
    pub fn pair() -> (Self, Self) {
        let addr_a: SocketAddr = "127.0.0.1:10001".parse().unwrap();
        let addr_b: SocketAddr = "127.0.0.1:10002".parse().unwrap();

        let (tx_a, rx_b) = mpsc::channel(256);
        let (tx_b, rx_a) = mpsc::channel(256);

        (
            Self {
                local_addr: addr_a,
                tx: tx_a,
                rx: tokio::sync::Mutex::new(rx_a),
            },
            Self {
                local_addr: addr_b,
                tx: tx_b,
                rx: tokio::sync::Mutex::new(rx_b),
            },
        )
    }
}

impl SyncTransport for ChannelTransport {
    async fn send_to(&self, message: &Message, addr: SocketAddr) -> TransportResult<()> {
        let envelope = Envelope {
            message: message.clone(),
            from: self.local_addr,
        };
        self.tx
            .send((envelope, addr))
            .await
            .map_err(|_| TransportError::Closed)?;
        Ok(())
    }

    async fn recv(&self) -> TransportResult<Envelope> {
        let (envelope, _) = self
            .rx
            .lock()
            .await
            .recv()
            .await
            .ok_or(TransportError::Closed)?;
        Ok(envelope)
    }

    fn local_addr(&self) -> TransportResult<SocketAddr> {
        Ok(self.local_addr)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::Message;

    #[tokio::test]
    async fn test_channel_transport_pair() {
        let (a, b) = ChannelTransport::pair();

        let msg = Message::Bye;
        let addr_b = b.local_addr().unwrap();

        a.send_to(&msg, addr_b).await.unwrap();
        let envelope = b.recv().await.unwrap();
        assert!(matches!(envelope.message, Message::Bye));
    }

    #[tokio::test]
    async fn test_channel_transport_hello() {
        use crate::{NodeId, WorldHash};

        let (a, b) = ChannelTransport::pair();
        let addr_b = b.local_addr().unwrap();

        let hello = Message::Hello {
            node_id: NodeId(42),
            seq: 100,
            world_hash: WorldHash(0xBEEF),
            semantics_id: crate::SYNC_SEMANTICS_ID,
        };

        a.send_to(&hello, addr_b).await.unwrap();
        let env = b.recv().await.unwrap();

        match env.message {
            Message::Hello {
                node_id,
                seq,
                world_hash,
                ..
            } => {
                assert_eq!(node_id.0, 42);
                assert_eq!(seq, 100);
                assert_eq!(world_hash.0, 0xBEEF);
            }
            _ => panic!("expected Hello"),
        }
    }

    #[tokio::test]
    async fn test_channel_transport_bidirectional() {
        let (a, b) = ChannelTransport::pair();
        let addr_a = a.local_addr().unwrap();
        let addr_b = b.local_addr().unwrap();

        // A → B
        a.send_to(&Message::Ack { seq: 1 }, addr_b).await.unwrap();
        let env = b.recv().await.unwrap();
        assert!(matches!(env.message, Message::Ack { seq: 1 }));

        // B → A
        b.send_to(&Message::Ack { seq: 2 }, addr_a).await.unwrap();
        let env = a.recv().await.unwrap();
        assert!(matches!(env.message, Message::Ack { seq: 2 }));
    }
}
