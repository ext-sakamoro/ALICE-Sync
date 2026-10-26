//! A [`SyncTransport`] over [`alice_channel`] (feature `secure-transport`): every peer is a device holding a
//! certificate the policy accepts, and every message travels sealed under keys from a Noise handshake.
//!
//! [`SecureTransport`] works like ALICE-Sync's `TcpTransport` (a listener, one connection per peer, a
//! reader task per connection feeding one receive queue), with the channel in place of the bare stream:
//!
//! - a connection becomes a peer only after the handshake: the peer proved a device key certified by a
//!   master key the [`PeerPolicy`] knows, and (on the dialling side) the accepting side confirmed it on
//!   the new keys
//! - the handshake is Noise IK: the dialling side must hold the certificate of the device it dials, and
//!   takes that device's Noise static key from it. The transport keeps a book of certificates by address
//!   ([`SecureTransport::add_peer_cert`], or [`SecureTransport::connect`] with the certificate); every dial
//!   checks the book's certificate against the policy as it stands, and an address without a certificate is
//!   never dialled ([`TransportError::NoPeerCert`])
//! - [`SyncTransport::send_to`] dials an address in the book it has no connection to, so a node can reply
//!   to a peer it has never accepted from; [`SecureTransport::dialer`] makes a transport that only dials
//!   and opens no listener
//! - a message of any size up to [`MAX_MESSAGE`] is sent as one or more channel messages; the first
//!   carries the kind (1 byte: [`KIND_MESSAGE`] for a Sync message, [`KIND_BYTES`] for application
//!   bytes) and the total length (4 bytes, little-endian), and the rest follow in order on the same
//!   connection, under the same send lock, so messages to one peer never interleave
//! - a connection that fails (I/O, a refused or undecodable channel message, a kind other than the two,
//!   a message of the message kind that is not a Sync message ([`SecureTransport::malformed_count`]),
//!   a length over the limit, or a send that did not finish) is dropped; the next `send_to` to a
//!   dialled address dials again
//!
//! An accepted peer is known by its remote socket address (as `TcpTransport` does): replies go to the
//! address in [`Envelope::from`]. [`SecureTransport::recv_verified`] gives each received message with
//! the certificate of the connection it came on; [`SecureTransport::peer_cert`] gives the certificate
//! behind an address while it is connected. [`SecureTransport::revalidate`] checks the open connections
//! against the policy as it stands now and closes the ones it refuses.

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crate::protocol::Message;
use crate::transport::{Envelope, Frame, SyncTransport, TransportError, TransportResult};
use alice_channel::{
    AsyncRecvHalf, AsyncSecureChannel, AsyncSendHalf, ChannelError, Credential, PeerPolicy,
    MAX_PAYLOAD,
};
use alice_identity::artifact::Revocation;
use alice_identity::device_cert::DeviceCert;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, Mutex};
use tokio::task::AbortHandle;

/// The largest encoded Sync message or block of application bytes sent or accepted, in bytes (the limit
/// of ALICE-Sync's `TcpTransport`)
pub const MAX_MESSAGE: usize = 1_000_000;

/// How long a handshake may take, from the TCP connection to an open channel, unless told otherwise
pub const DEFAULT_HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

/// The kind byte of a message carrying an encoded Sync [`Message`]
pub const KIND_MESSAGE: u8 = 0;

/// The kind byte of a message carrying application bytes, never decoded as a Sync [`Message`]
pub const KIND_BYTES: u8 = 1;

/// Bytes at the head of a message's first channel message: the kind and the total length
const HEAD: usize = 5;

/// What a received message carries
#[derive(Debug, Clone)]
pub enum Body {
    /// A Sync message (kind [`KIND_MESSAGE`])
    Message(Message),
    /// Application bytes (kind [`KIND_BYTES`]), as sent with [`SecureTransport::send_bytes_to`]
    Bytes(Vec<u8>),
}

/// A received message with the certificate of the connection it came on
#[derive(Debug, Clone)]
pub struct VerifiedEnvelope {
    pub body: Body,
    /// The peer's address, where replies go
    pub from: SocketAddr,
    /// The certificate the peer proved in the handshake of this connection
    pub peer: Arc<DeviceCert>,
}

/// One connected peer: the sending half of its channel, the certificate it proved, and its reader
struct Peer {
    /// Tells this connection from a later one to the same address
    conn: u64,
    cert: Arc<DeviceCert>,
    tx: Mutex<AsyncSendHalf<TcpStream>>,
    reader: AbortHandle,
}

type Peers = Mutex<BTreeMap<SocketAddr, Arc<Peer>>>;

/// A Sync transport whose peers are certified devices and whose messages are sealed
pub struct SecureTransport<P> {
    /// `None` for a transport that only dials
    listener: Option<TcpListener>,
    cred: Arc<Credential>,
    policy: Arc<P>,
    handshake_timeout: Duration,
    peers: Arc<Peers>,
    /// The certificate to dial each address with
    book: Mutex<BTreeMap<SocketAddr, DeviceCert>>,
    /// Held while dialling, so that two sends to a new address open one connection
    dialling: Mutex<()>,
    next_conn: AtomicU64,
    refused: AtomicU64,
    skipped_bytes: AtomicU64,
    malformed: Arc<AtomicU64>,
    rx: Mutex<mpsc::Receiver<VerifiedEnvelope>>,
    tx: mpsc::Sender<VerifiedEnvelope>,
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

fn no_listener() -> TransportError {
    TransportError::Io(std::io::ErrorKind::NotConnected.into())
}

/// Whether the policy as it stands at `now` still accepts `cert`: the subject's master key is known, the
/// certificate verifies under it at `now`, and no revocation list held revokes it
fn still_accepted(cert: &DeviceCert, policy: &impl PeerPolicy, now: u64) -> bool {
    let Some(master) = policy.master_of(cert.subject()) else {
        return false;
    };
    if cert.verify(&master, now).is_err() {
        return false;
    }
    !matches!(policy.revocation(), Revocation::Checked(list) if list.revokes(cert))
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
        Ok(Self::new(Some(listener), cred, policy, handshake_timeout))
    }

    /// A transport that only dials: it opens no listener, and [`Self::accept`], [`Self::serve`] and
    /// [`SyncTransport::local_addr`] return [`TransportError::Io`] with
    /// [`std::io::ErrorKind::NotConnected`]; replies come back on the connections it dialled
    #[must_use]
    pub fn dialer(cred: Credential, policy: P) -> Self {
        Self::new(None, cred, policy, DEFAULT_HANDSHAKE_TIMEOUT)
    }

    fn new(
        listener: Option<TcpListener>,
        cred: Credential,
        policy: P,
        handshake_timeout: Duration,
    ) -> Self {
        let (tx, rx) = mpsc::channel(1024);
        Self {
            listener,
            cred: Arc::new(cred),
            policy: Arc::new(policy),
            handshake_timeout,
            peers: Arc::new(Mutex::new(BTreeMap::new())),
            book: Mutex::new(BTreeMap::new()),
            dialling: Mutex::new(()),
            next_conn: AtomicU64::new(0),
            refused: AtomicU64::new(0),
            skipped_bytes: AtomicU64::new(0),
            malformed: Arc::new(AtomicU64::new(0)),
            rx: Mutex::new(rx),
            tx,
        }
    }

    /// Accept the next connection and complete its handshake
    ///
    /// # Errors
    /// [`TransportError::Io`] when the listener fails, there is no listener
    /// ([`std::io::ErrorKind::NotConnected`]), or the handshake takes longer than the timeout;
    /// [`TransportError::Codec`] when the peer is refused (unknown master key, refused certificate, or a
    /// broken handshake); [`TransportError::Closed`] when it hangs up during the handshake. A failed
    /// handshake adds no peer.
    pub async fn accept(&self) -> TransportResult<SocketAddr> {
        let listener = self.listener.as_ref().ok_or_else(no_listener)?;
        let (stream, addr) = listener.accept().await?;
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
    /// counted in [`Self::refused_count`] and do not stop the loop. Without a listener the task ends at
    /// once with [`std::io::ErrorKind::NotConnected`].
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

    /// Put `cert` in the book as the certificate to dial `addr` with, replacing the one held for it; an open
    /// connection to `addr` is kept
    pub async fn add_peer_cert(&self, addr: SocketAddr, cert: DeviceCert) {
        self.book.lock().await.insert(addr, cert);
    }

    /// The certificate the book holds for `addr`
    pub async fn book_cert(&self, addr: SocketAddr) -> Option<DeviceCert> {
        self.book.lock().await.get(&addr).cloned()
    }

    /// Put `cert` in the book for `addr` (as [`Self::add_peer_cert`]), dial it and complete the handshake;
    /// a connection already open to `addr` is replaced
    ///
    /// # Errors
    /// As [`Self::accept`]; [`TransportError::Codec`] when the policy refuses `cert` (nothing is written)
    pub async fn connect(&self, addr: SocketAddr, cert: DeviceCert) -> TransportResult<()> {
        self.add_peer_cert(addr, cert).await;
        let _dialling = self.dialling.lock().await;
        self.dial(addr).await
    }

    async fn dial(&self, addr: SocketAddr) -> TransportResult<()> {
        let cert = self
            .book_cert(addr)
            .await
            .ok_or(TransportError::NoPeerCert(addr))?;
        // checked before the TCP connection: a refused certificate reaches nothing on the other side
        alice_channel::check_cert(&cert, &*self.policy).map_err(channel_error)?;
        let opened = tokio::time::timeout(self.handshake_timeout, async {
            let stream = TcpStream::connect(addr).await.map_err(TransportError::Io)?;
            AsyncSecureChannel::connect_with_timeout(
                stream,
                &self.cred,
                &*self.policy,
                &cert,
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
        self.peers
            .lock()
            .await
            .get(&addr)
            .map(|p| DeviceCert::clone(&p.cert))
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

    /// How many received blocks of application bytes [`SyncTransport::recv`] passed over (it returns
    /// Sync messages only; [`Self::recv_verified`] returns both)
    #[must_use]
    pub fn skipped_bytes_count(&self) -> u64 {
        self.skipped_bytes.load(Ordering::Relaxed)
    }

    /// How many connections were closed because a verified peer sent a message of the message kind that
    /// does not decode as a Sync message (a protocol error of an authenticated peer)
    #[must_use]
    pub fn malformed_count(&self) -> u64 {
        self.malformed.load(Ordering::Relaxed)
    }

    /// Close the connection to `addr`, on this side and (as its reader sees the end) on the other;
    /// `false` when no peer is connected there
    pub async fn close(&self, addr: SocketAddr) -> bool {
        let Some(peer) = self.peers.lock().await.remove(&addr) else {
            return false;
        };
        peer.reader.abort();
        true
    }

    /// Check every open connection against the policy as it stands now ([`PeerPolicy::now`], the master
    /// keys and the revocation lists it holds) and close the ones it no longer accepts; returns their
    /// addresses, in order
    pub async fn revalidate(&self) -> Vec<SocketAddr> {
        let now = self.policy.now();
        let mut closed = Vec::new();
        self.peers.lock().await.retain(|addr, peer| {
            let keep = still_accepted(&peer.cert, &*self.policy, now);
            if !keep {
                peer.reader.abort();
                closed.push(*addr);
            }
            keep
        });
        closed
    }

    /// The next received message with the certificate of the connection it came on
    ///
    /// # Errors
    /// [`TransportError::Closed`] when the receive queue is closed
    pub async fn recv_verified(&self) -> TransportResult<VerifiedEnvelope> {
        self.rx
            .lock()
            .await
            .recv()
            .await
            .ok_or(TransportError::Closed)
    }

    /// Send application bytes to `addr` (dialling it as [`SyncTransport::send_to`] does); the receiver
    /// gets them as [`Body::Bytes`] and never decodes them as a Sync message
    ///
    /// # Errors
    /// [`TransportError::Codec`] when `bytes` is longer than [`MAX_MESSAGE`] (nothing is sent); otherwise
    /// as [`SyncTransport::send_to`]
    pub async fn send_bytes_to(&self, bytes: &[u8], addr: SocketAddr) -> TransportResult<()> {
        self.send_framed(KIND_BYTES, bytes, addr).await
    }

    async fn add_peer(
        &self,
        addr: SocketAddr,
        channel: AsyncSecureChannel<TcpStream>,
    ) -> TransportResult<()> {
        let cert = Arc::new(channel.peer_cert().clone());
        let (tx, rx) = channel.into_split().map_err(channel_error)?;
        let conn = self.next_conn.fetch_add(1, Ordering::Relaxed);
        // the reader is started under the lock, so it cannot end and drop the peer before it is inserted
        let mut peers = self.peers.lock().await;
        let reader = tokio::spawn(read_loop(
            rx,
            addr,
            conn,
            cert.clone(),
            self.tx.clone(),
            self.peers.clone(),
            self.malformed.clone(),
        ))
        .abort_handle();
        peers.insert(
            addr,
            Arc::new(Peer {
                conn,
                cert,
                tx: Mutex::new(tx),
                reader,
            }),
        );
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

    async fn send_framed(&self, kind: u8, bytes: &[u8], addr: SocketAddr) -> TransportResult<()> {
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
        let split = bytes.len().min(MAX_PAYLOAD - HEAD);
        let mut first = Vec::with_capacity(HEAD + split);
        first.push(kind);
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
}

/// Remove the peer at `addr` if it is still the connection `conn` (a newer one to the same address stays)
async fn drop_peer(peers: &Peers, addr: SocketAddr, conn: u64) {
    let mut peers = peers.lock().await;
    if peers.get(&addr).is_some_and(|p| p.conn == conn) {
        peers.remove(&addr);
    }
}

/// Split the head of a message's first channel message: its kind, total length, and the bytes after
fn parse_head(first: &[u8]) -> Result<(u8, usize, &[u8]), TransportError> {
    let Some(([kind, len @ ..], head)) = first.split_first_chunk::<HEAD>() else {
        return Err(TransportError::Codec(
            "a message without its kind and length".to_owned(),
        ));
    };
    if *kind != KIND_MESSAGE && *kind != KIND_BYTES {
        return Err(TransportError::Codec(format!("a message of kind {kind}")));
    }
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
    Ok((*kind, total, head))
}

/// Read one message: the first channel message carries the kind and the total length, the rest follow
/// in order
async fn read_message(rx: &mut AsyncRecvHalf<TcpStream>) -> Result<(u8, Vec<u8>), TransportError> {
    let first = rx.recv().await.map_err(channel_error)?;
    let (kind, total, head) = parse_head(&first)?;
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
    Ok((kind, bytes))
}

async fn read_loop(
    mut rx: AsyncRecvHalf<TcpStream>,
    addr: SocketAddr,
    conn: u64,
    cert: Arc<DeviceCert>,
    out: mpsc::Sender<VerifiedEnvelope>,
    peers: Arc<Peers>,
    malformed: Arc<AtomicU64>,
) {
    // any failure ends the connection: a channel out of step or a peer breaking the framing is not read on
    while let Ok((kind, bytes)) = read_message(&mut rx).await {
        let body = if kind == KIND_BYTES {
            Body::Bytes(bytes)
        } else if let Some(message) = Message::from_compact_bytes(&bytes) {
            Body::Message(message)
        } else {
            // a verified peer that sends something else under the message kind breaks the protocol: the
            // connection ends here, counted
            malformed.fetch_add(1, Ordering::Relaxed);
            break;
        };
        let envelope = VerifiedEnvelope {
            body,
            from: addr,
            peer: cert.clone(),
        };
        if out.send(envelope).await.is_err() {
            break;
        }
    }
    drop_peer(&peers, addr, conn).await;
}

impl<P: PeerPolicy + Send + Sync + 'static> SyncTransport for SecureTransport<P> {
    async fn send_to(&self, message: &Message, addr: SocketAddr) -> TransportResult<()> {
        self.send_framed(KIND_MESSAGE, &message.to_compact_bytes(), addr)
            .await
    }

    /// The next received Sync message; application bytes received before it are passed over and counted
    /// in [`SecureTransport::skipped_bytes_count`]
    async fn recv(&self) -> TransportResult<Envelope> {
        loop {
            let v = self.recv_verified().await?;
            match v.body {
                Body::Message(message) => {
                    return Ok(Envelope {
                        message,
                        from: v.from,
                    })
                }
                Body::Bytes(_) => {
                    self.skipped_bytes.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }

    async fn send_bytes_to(&self, bytes: &[u8], addr: SocketAddr) -> TransportResult<()> {
        self.send_framed(KIND_BYTES, bytes, addr).await
    }

    /// The next received Sync message or block of application bytes (nothing is passed over)
    async fn recv_frame(&self) -> TransportResult<Frame> {
        let v = self.recv_verified().await?;
        Ok(match v.body {
            Body::Message(message) => Frame::Message(Envelope {
                message,
                from: v.from,
            }),
            Body::Bytes(bytes) => Frame::Bytes {
                bytes,
                from: v.from,
            },
        })
    }

    fn local_addr(&self) -> TransportResult<SocketAddr> {
        self.listener
            .as_ref()
            .ok_or_else(no_listener)?
            .local_addr()
            .map_err(TransportError::Io)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn first(kind: u8, total: u32, head: &[u8]) -> Vec<u8> {
        let mut f = vec![kind];
        f.extend_from_slice(&total.to_le_bytes());
        f.extend_from_slice(head);
        f
    }

    #[test]
    fn the_two_kinds_are_read_and_no_other() {
        for kind in [KIND_MESSAGE, KIND_BYTES] {
            let f = first(kind, 3, &[7, 8]);
            let (k, total, head) = parse_head(&f).unwrap();
            assert_eq!((k, total, head), (kind, 3, &[7, 8][..]));
        }
        for kind in 2..=u8::MAX {
            assert!(
                matches!(
                    parse_head(&first(kind, 1, &[0])),
                    Err(TransportError::Codec(_))
                ),
                "kind {kind}"
            );
        }
    }

    #[test]
    fn a_head_shorter_than_kind_and_length_is_refused() {
        for n in 0..HEAD {
            assert!(parse_head(&first(KIND_BYTES, 0, &[])[..n]).is_err(), "{n}");
        }
        assert!(parse_head(&first(KIND_BYTES, 0, &[])).is_ok());
    }

    #[test]
    fn a_length_over_the_limit_is_refused_and_the_limit_is_not() {
        let max = u32::try_from(MAX_MESSAGE).unwrap();
        assert!(parse_head(&first(KIND_BYTES, max, &[])).is_ok());
        assert!(parse_head(&first(KIND_BYTES, max + 1, &[])).is_err());
        assert!(parse_head(&first(KIND_BYTES, 1, &[1, 2])).is_err());
    }
}
