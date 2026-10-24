//! A [`SyncTransport`] over [`alice_channel`] (feature `secure-transport`): every peer is a device holding a
//! certificate the policy accepts, and every message travels sealed under keys from a Noise handshake.
//!
//! [`SecureTransport`] works like ALICE-Sync's `TcpTransport` (a listener, one connection per peer, a
//! reader task per connection feeding one receive queue), with the channel in place of the bare stream:
//!
//! - a connection becomes a peer only after the handshake: the peer proved a device key certified by a
//!   master key the [`PeerPolicy`] knows, and (on the dialling side) the accepting side confirmed it on
//!   the new keys
//! - [`SyncTransport::send_to`] dials an address it has no connection to, so a node can reply to a peer
//!   it has never accepted from
//! - a message of any size up to [`MAX_MESSAGE`] is sent as one or more channel messages; the first
//!   carries the total length (4 bytes, little-endian), and the rest follow in order on the same
//!   connection, under the same send lock, so messages to one peer never interleave
//! - a connection that fails (I/O, a refused or undecodable channel message, a length over the limit, or
//!   a send that did not finish) is dropped; the next `send_to` to a dialled address dials again
//!
//! An accepted peer is known by its remote socket address (as `TcpTransport` does): replies go to the
//! address in [`Envelope::from`]. [`SecureTransport::peer_cert`] gives the certificate behind an address.

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crate::protocol::Message;
use crate::transport::{Envelope, SyncTransport, TransportError, TransportResult};
use alice_channel::{
    AsyncRecvHalf, AsyncSecureChannel, AsyncSendHalf, ChannelError, Credential, PeerPolicy,
    MAX_PAYLOAD,
};
use alice_identity::device_cert::DeviceCert;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, Mutex};

/// The largest encoded Sync message sent or accepted, in bytes (the limit of ALICE-Sync's `TcpTransport`)
pub const MAX_MESSAGE: usize = 1_000_000;

/// How long a handshake may take, from the TCP connection to an open channel, unless told otherwise
pub const DEFAULT_HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

/// Bytes of the total length at the head of a message's first channel message
const LEN_PREFIX: usize = 4;

/// One connected peer: the sending half of its channel and the certificate it proved
struct Peer {
    /// Tells this connection from a later one to the same address
    conn: u64,
    cert: DeviceCert,
    tx: Mutex<AsyncSendHalf<TcpStream>>,
}

type Peers = Mutex<BTreeMap<SocketAddr, Arc<Peer>>>;

/// A Sync transport whose peers are certified devices and whose messages are sealed
pub struct SecureTransport<P> {
    listener: TcpListener,
    cred: Arc<Credential>,
    policy: Arc<P>,
    handshake_timeout: Duration,
    peers: Arc<Peers>,
    /// Held while dialling, so that two sends to a new address open one connection
    dialling: Mutex<()>,
    next_conn: AtomicU64,
    refused: AtomicU64,
    rx: Mutex<mpsc::Receiver<Envelope>>,
    tx: mpsc::Sender<Envelope>,
}

fn channel_error(e: ChannelError) -> TransportError {
    match e {
        ChannelError::Io(kind) => TransportError::Io(kind.into()),
        ChannelError::Closed => TransportError::Closed,
        other => TransportError::Codec(format!("secure channel: {other}")),
    }
}

fn timed_out() -> TransportError {
    TransportError::Io(std::io::ErrorKind::TimedOut.into())
}

impl<P: PeerPolicy + Send + Sync + 'static> SecureTransport<P> {
    /// Bind and start listening, with this device's credential and the policy that decides which peers
    /// are accepted
    ///
    /// # Errors
    /// [`TransportError::Io`] when the address cannot be bound
    pub async fn bind(addr: &str, cred: Credential, policy: P) -> TransportResult<Self> {
        Self::bind_with_timeout(addr, cred, policy, DEFAULT_HANDSHAKE_TIMEOUT).await
    }

    /// As [`Self::bind`], with the time a handshake may take in either direction
    ///
    /// # Errors
    /// [`TransportError::Io`] when the address cannot be bound
    pub async fn bind_with_timeout(
        addr: &str,
        cred: Credential,
        policy: P,
        handshake_timeout: Duration,
    ) -> TransportResult<Self> {
        let listener = TcpListener::bind(addr).await?;
        let (tx, rx) = mpsc::channel(1024);
        Ok(Self {
            listener,
            cred: Arc::new(cred),
            policy: Arc::new(policy),
            handshake_timeout,
            peers: Arc::new(Mutex::new(BTreeMap::new())),
            dialling: Mutex::new(()),
            next_conn: AtomicU64::new(0),
            refused: AtomicU64::new(0),
            rx: Mutex::new(rx),
            tx,
        })
    }

    /// Accept the next connection and complete its handshake
    ///
    /// # Errors
    /// [`TransportError::Io`] when the listener fails or the handshake takes longer than the timeout;
    /// [`TransportError::Codec`] when the peer is refused (unknown master key, refused certificate, or a
    /// broken handshake); [`TransportError::Closed`] when it hangs up during the handshake. A failed
    /// handshake adds no peer.
    pub async fn accept(&self) -> TransportResult<SocketAddr> {
        let (stream, addr) = self.listener.accept().await?;
        let opened = tokio::time::timeout(
            self.handshake_timeout,
            AsyncSecureChannel::accept(stream, &self.cred, &*self.policy),
        )
        .await;
        match opened {
            Ok(Ok(channel)) => {
                self.add_peer(addr, channel).await?;
                Ok(addr)
            }
            Ok(Err(e)) => {
                self.refused.fetch_add(1, Ordering::Relaxed);
                Err(channel_error(e))
            }
            Err(_) => {
                self.refused.fetch_add(1, Ordering::Relaxed);
                Err(timed_out())
            }
        }
    }

    /// Accept connections until the listener fails, on a task of its own; refused handshakes are
    /// counted in [`Self::refused_count`] and do not stop the loop
    #[must_use]
    pub fn serve(self: Arc<Self>) -> tokio::task::JoinHandle<TransportError> {
        tokio::spawn(async move {
            loop {
                match self.accept().await {
                    Ok(_) => {}
                    // a refused or timed-out peer is that peer's failure, not the listener's
                    Err(TransportError::Codec(_) | TransportError::Closed) => {}
                    Err(TransportError::Io(e)) if e.kind() == std::io::ErrorKind::TimedOut => {}
                    Err(e) => return e,
                }
            }
        })
    }

    /// Dial a peer and complete the handshake; a connection already open to `addr` is replaced
    ///
    /// # Errors
    /// As [`Self::accept`]
    pub async fn connect(&self, addr: SocketAddr) -> TransportResult<()> {
        let _dialling = self.dialling.lock().await;
        self.dial(addr).await
    }

    async fn dial(&self, addr: SocketAddr) -> TransportResult<()> {
        let opened = tokio::time::timeout(self.handshake_timeout, async {
            let stream = TcpStream::connect(addr).await.map_err(TransportError::Io)?;
            AsyncSecureChannel::connect_with_timeout(
                stream,
                &self.cred,
                &*self.policy,
                self.handshake_timeout,
            )
            .await
            .map_err(channel_error)
        })
        .await
        .map_err(|_| timed_out())??;
        self.add_peer(addr, opened).await
    }

    /// The certificate of the peer connected at `addr`, as verified in its handshake
    pub async fn peer_cert(&self, addr: SocketAddr) -> Option<DeviceCert> {
        self.peers.lock().await.get(&addr).map(|p| p.cert.clone())
    }

    /// The addresses of the connected peers, in order
    pub async fn peers(&self) -> Vec<SocketAddr> {
        self.peers.lock().await.keys().copied().collect()
    }

    /// How many incoming handshakes were refused or timed out
    #[must_use]
    pub fn refused_count(&self) -> u64 {
        self.refused.load(Ordering::Relaxed)
    }

    async fn add_peer(
        &self,
        addr: SocketAddr,
        channel: AsyncSecureChannel<TcpStream>,
    ) -> TransportResult<()> {
        let cert = channel.peer_cert().clone();
        let (tx, rx) = channel.into_split().map_err(channel_error)?;
        let conn = self.next_conn.fetch_add(1, Ordering::Relaxed);
        self.peers.lock().await.insert(
            addr,
            Arc::new(Peer {
                conn,
                cert,
                tx: Mutex::new(tx),
            }),
        );
        tokio::spawn(read_loop(
            rx,
            addr,
            conn,
            self.tx.clone(),
            self.peers.clone(),
        ));
        Ok(())
    }

    async fn peer_for_send(&self, addr: SocketAddr) -> TransportResult<Arc<Peer>> {
        if let Some(p) = self.peers.lock().await.get(&addr) {
            return Ok(p.clone());
        }
        let _dialling = self.dialling.lock().await;
        // another send may have dialled while this one waited
        if let Some(p) = self.peers.lock().await.get(&addr) {
            return Ok(p.clone());
        }
        self.dial(addr).await?;
        self.peers
            .lock()
            .await
            .get(&addr)
            .cloned()
            .ok_or(TransportError::Closed)
    }
}

/// Remove the peer at `addr` if it is still the connection `conn` (a newer one to the same address stays)
async fn drop_peer(peers: &Peers, addr: SocketAddr, conn: u64) {
    let mut peers = peers.lock().await;
    if peers.get(&addr).is_some_and(|p| p.conn == conn) {
        peers.remove(&addr);
    }
}

/// Read one Sync message: the first channel message carries the total length, the rest follow in order
async fn read_message(rx: &mut AsyncRecvHalf<TcpStream>) -> Result<Vec<u8>, TransportError> {
    let first = rx.recv().await.map_err(channel_error)?;
    let Some((len, head)) = first.split_first_chunk::<LEN_PREFIX>() else {
        return Err(TransportError::Codec(
            "a message without its length".to_owned(),
        ));
    };
    let total = u32::from_le_bytes(*len) as usize;
    if total > MAX_MESSAGE {
        return Err(TransportError::Codec(format!(
            "a message of {total} bytes, over {MAX_MESSAGE}"
        )));
    }
    if head.len() > total {
        return Err(TransportError::Codec(
            "a message longer than its length".to_owned(),
        ));
    }
    let mut bytes = Vec::with_capacity(total);
    bytes.extend_from_slice(head);
    while bytes.len() < total {
        let part = rx.recv().await.map_err(channel_error)?;
        if part.is_empty() || bytes.len() + part.len() > total {
            return Err(TransportError::Codec(
                "a message part that does not fit its length".to_owned(),
            ));
        }
        bytes.extend_from_slice(&part);
    }
    Ok(bytes)
}

async fn read_loop(
    mut rx: AsyncRecvHalf<TcpStream>,
    addr: SocketAddr,
    conn: u64,
    out: mpsc::Sender<Envelope>,
    peers: Arc<Peers>,
) {
    // any failure ends the connection: a channel out of step or a peer breaking the framing is not read on
    while let Ok(bytes) = read_message(&mut rx).await {
        // as TcpTransport: a sealed message from a verified peer that is not a Sync message is skipped
        if let Some(message) = Message::from_compact_bytes(&bytes) {
            if out
                .send(Envelope {
                    message,
                    from: addr,
                })
                .await
                .is_err()
            {
                break;
            }
        }
    }
    drop_peer(&peers, addr, conn).await;
}

impl<P: PeerPolicy + Send + Sync + 'static> SyncTransport for SecureTransport<P> {
    async fn send_to(&self, message: &Message, addr: SocketAddr) -> TransportResult<()> {
        let bytes = message.to_compact_bytes();
        if bytes.len() > MAX_MESSAGE {
            return Err(TransportError::Codec(format!(
                "a message of {} bytes, over {MAX_MESSAGE}",
                bytes.len()
            )));
        }
        let total = u32::try_from(bytes.len())
            .map_err(|_| TransportError::Codec("a message too long for its length".to_owned()))?;
        let peer = self.peer_for_send(addr).await?;
        let mut tx = peer.tx.lock().await;
        let split = bytes.len().min(MAX_PAYLOAD - LEN_PREFIX);
        let mut first = Vec::with_capacity(LEN_PREFIX + split);
        first.extend_from_slice(&total.to_le_bytes());
        first.extend_from_slice(&bytes[..split]);
        let mut sent = tx.send(&first).await;
        for part in bytes[split..].chunks(MAX_PAYLOAD) {
            if sent.is_err() {
                break;
            }
            sent = tx.send(part).await;
        }
        drop(tx);
        if let Err(e) = sent {
            // the peer may hold half a message: the connection is not used again
            drop_peer(&self.peers, addr, peer.conn).await;
            return Err(channel_error(e));
        }
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
