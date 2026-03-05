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
use crate::reliability::{ReassemblyBuffer, ReliableEndpoint, HEADER_SIZE, MAX_PAYLOAD};

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

    /// Local bound address
    fn local_addr(&self) -> TransportResult<SocketAddr>;
}

// ============================================================================
// UDP Transport
// ============================================================================

/// UDP-based transport with reliability layer.
///
/// Best for: low-latency real-time sync (games, `IoT`, live collaboration).
pub struct UdpTransport {
    socket: UdpSocket,
    reliability: tokio::sync::Mutex<ReliableEndpoint>,
    reassembly: tokio::sync::Mutex<ReassemblyBuffer>,
}

impl UdpTransport {
    /// Bind to a local address
    pub async fn bind(addr: &str) -> TransportResult<Self> {
        let socket = UdpSocket::bind(addr).await?;
        Ok(Self {
            socket,
            reliability: tokio::sync::Mutex::new(ReliableEndpoint::new()),
            reassembly: tokio::sync::Mutex::new(ReassemblyBuffer::new()),
        })
    }

    /// Get a reference to the reliability endpoint (for stats)
    pub async fn reliability_stats(&self) -> (u64, u64, f64) {
        let rel = self.reliability.lock().await;
        (rel.packets_acked, rel.packets_lost, rel.loss_rate())
    }

    /// Collect and send retransmissions
    pub async fn retransmit(&self, peer: SocketAddr) -> TransportResult<()> {
        let retransmits = self.reliability.lock().await.collect_retransmits();
        for packet in retransmits {
            self.socket.send_to(&packet, peer).await?;
        }
        Ok(())
    }
}

impl SyncTransport for UdpTransport {
    async fn send_to(&self, message: &Message, addr: SocketAddr) -> TransportResult<()> {
        let payload = message.to_compact_bytes();

        if payload.len() <= MAX_PAYLOAD - HEADER_SIZE {
            // Single packet
            let packet = self.reliability.lock().await.wrap_outgoing(&payload);
            self.socket.send_to(&packet, addr).await?;
        } else {
            // Fragment and send each fragment as a reliable packet
            let msg_id = self.reliability.lock().await.local_seq();
            let fragments = crate::reliability::fragment_payload(msg_id, &payload);
            for frag in fragments {
                let packet = self.reliability.lock().await.wrap_outgoing(&frag);
                self.socket.send_to(&packet, addr).await?;
            }
        }
        Ok(())
    }

    async fn recv(&self) -> TransportResult<Envelope> {
        let mut buf = vec![0u8; 65535];
        loop {
            let (len, from) = self.socket.recv_from(&mut buf).await?;
            let packet = &buf[..len];

            let unwrap_result = self.reliability.lock().await.unwrap_incoming(packet);
            let Some(payload) = unwrap_result else {
                continue; // duplicate or invalid
            };

            // Check if this is a fragment (payload starts with fragment header)
            // Fragments have payload > MAX_PAYLOAD threshold, but we detect by trying
            // bitcode decode first (fast path for non-fragmented)
            if let Some(msg) = Message::from_compact_bytes(&payload) {
                return Ok(Envelope { message: msg, from });
            }

            // Try fragment reassembly
            let reassemble_result = self.reassembly.lock().await.feed(&payload);
            if let Some(complete) = reassemble_result {
                if let Some(msg) = Message::from_compact_bytes(&complete) {
                    return Ok(Envelope { message: msg, from });
                }
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
    peers: tokio::sync::Mutex<rustc_hash::FxHashMap<SocketAddr, TcpPeerWriter>>,
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
            peers: tokio::sync::Mutex::new(rustc_hash::FxHashMap::default()),
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
        };

        a.send_to(&hello, addr_b).await.unwrap();
        let env = b.recv().await.unwrap();

        match env.message {
            Message::Hello {
                node_id,
                seq,
                world_hash,
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
