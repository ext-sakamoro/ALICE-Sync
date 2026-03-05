#![allow(
    clippy::significant_drop_tightening,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc
)]

//! Async Session Driver — High-Level P2P Sync Orchestrator
//!
//! Provides a builder-pattern API for creating complete sync sessions
//! that compose transport, discovery, pub/sub, and sync mode selection.
//!
//! # Sync Modes
//!
//! | Mode | Best For | Characteristics |
//! |------|----------|-----------------|
//! | `Lockstep` | RTS, turn-based, < 4 players | Waits for all inputs |
//! | `Rollback` | Fighting games, FPS, action | Predicts + corrects |
//! | `Crdt` | Collaboration, IoT, databases | Eventual consistency |
//! | `EventSourcing` | Audit logs, financial, legal | Append-only + replay |
//! | `Snapshot` | Late-join, recovery | Full state transfer |

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{broadcast, RwLock};

use crate::channel::PubSub;
use crate::crdt::{CrdtMergeable, LwwMap};
use crate::discovery::Discovery;
use crate::protocol::Message;
use crate::transport::{Envelope, SyncTransport, TransportError};
use crate::{NodeId, WorldHash};

/// Synchronization mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncMode {
    /// Lockstep: all inputs collected before advancing.
    /// Deterministic, high latency tolerance.
    Lockstep,
    /// Rollback: predict and advance immediately, correct on mismatch.
    /// Low perceived latency, CPU cost on rollback.
    Rollback,
    /// CRDT: conflict-free replicated data types.
    /// Eventual consistency, no coordination needed.
    Crdt,
    /// Event Sourcing: append-only event log with replay.
    /// Full audit trail, deterministic rebuild.
    EventSourcing,
    /// Snapshot: periodic full state transfer.
    /// Simple, good for late-join and recovery.
    Snapshot,
}

/// Session configuration
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// Sync mode to use
    pub mode: SyncMode,
    /// Local node ID
    pub node_id: NodeId,
    /// Bind address for transport
    pub bind_addr: String,
    /// Tick rate (ticks per second) — 0 means event-driven (no fixed tick)
    pub tick_rate: u32,
    /// Hash check interval (every N ticks)
    pub hash_check_interval: u64,
    /// Maximum peers (0 = unlimited)
    pub max_peers: usize,
    /// Enable LAN discovery
    pub enable_discovery: bool,
    /// Human-readable instance name
    pub instance_name: String,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            mode: SyncMode::Crdt,
            node_id: NodeId(1),
            bind_addr: "0.0.0.0:0".to_string(),
            tick_rate: 60,
            hash_check_interval: 100,
            max_peers: 0,
            instance_name: "alice-sync-node".to_string(),
            enable_discovery: true,
        }
    }
}

/// Session state (shared between driver and user code)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    /// Session is being configured
    Building,
    /// Session is active and syncing
    Running,
    /// Session is paused (no tick processing)
    Paused,
    /// Session has been shut down
    Stopped,
}

/// Session statistics
#[derive(Debug, Clone, Default)]
pub struct SessionStats {
    /// Total ticks processed
    pub ticks: u64,
    /// Total messages sent
    pub messages_sent: u64,
    /// Total messages received
    pub messages_received: u64,
    /// Total bytes sent
    pub bytes_sent: u64,
    /// Total bytes received
    pub bytes_received: u64,
    /// Current peer count
    pub peer_count: usize,
    /// Current world hash
    pub world_hash: u64,
    /// Sync mode
    pub mode: Option<SyncMode>,
}

/// Events emitted by the session
#[derive(Debug, Clone)]
pub enum SessionEvent {
    /// A peer connected
    PeerJoined { addr: SocketAddr, node_id: NodeId },
    /// A peer disconnected
    PeerLeft { addr: SocketAddr },
    /// A sync message was received
    MessageReceived(Envelope),
    /// Hash divergence detected
    Divergence {
        peer: SocketAddr,
        local: WorldHash,
        remote: WorldHash,
    },
    /// Session state changed
    StateChanged(SessionState),
    /// Tick completed
    Tick { seq: u64 },
}

/// Builder for constructing a `SyncSession`.
pub struct SessionBuilder {
    config: SessionConfig,
}

impl SessionBuilder {
    /// Create a new session builder with default config.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: SessionConfig::default(),
        }
    }

    /// Set the sync mode.
    #[must_use]
    pub const fn mode(mut self, mode: SyncMode) -> Self {
        self.config.mode = mode;
        self
    }

    /// Set the local node ID.
    #[must_use]
    pub const fn node_id(mut self, id: u64) -> Self {
        self.config.node_id = NodeId(id);
        self
    }

    /// Set the bind address.
    #[must_use]
    pub fn bind(mut self, addr: &str) -> Self {
        self.config.bind_addr = addr.to_string();
        self
    }

    /// Set the tick rate (ticks per second). 0 = event-driven.
    #[must_use]
    pub const fn tick_rate(mut self, rate: u32) -> Self {
        self.config.tick_rate = rate;
        self
    }

    /// Set the hash check interval.
    #[must_use]
    pub const fn hash_check_interval(mut self, interval: u64) -> Self {
        self.config.hash_check_interval = interval;
        self
    }

    /// Set maximum peer count (0 = unlimited).
    #[must_use]
    pub const fn max_peers(mut self, max: usize) -> Self {
        self.config.max_peers = max;
        self
    }

    /// Enable or disable LAN discovery.
    #[must_use]
    pub const fn discovery(mut self, enable: bool) -> Self {
        self.config.enable_discovery = enable;
        self
    }

    /// Set instance name.
    #[must_use]
    pub fn name(mut self, name: &str) -> Self {
        self.config.instance_name = name.to_string();
        self
    }

    /// Build the session (does not start it yet).
    pub fn build<T: SyncTransport + 'static>(self, transport: T) -> SyncSession<T> {
        let local_addr = transport
            .local_addr()
            .unwrap_or_else(|_| "0.0.0.0:0".parse().unwrap());

        let discovery = Arc::new(Discovery::new(local_addr, &self.config.instance_name));
        let pubsub = Arc::new(PubSub::new());
        let (event_tx, _) = broadcast::channel(1024);

        let replica_id = self.config.node_id.0;
        SyncSession {
            config: self.config,
            transport: Arc::new(transport),
            discovery,
            pubsub,
            state: Arc::new(RwLock::new(SessionState::Building)),
            stats: Arc::new(RwLock::new(SessionStats::default())),
            event_tx,
            crdt_state: Arc::new(RwLock::new(LwwMap::new(replica_id))),
        }
    }
}

impl Default for SessionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// A live P2P sync session.
///
/// Composes transport, discovery, pub/sub, and sync mode into
/// a single orchestrated session.
pub struct SyncSession<T: SyncTransport> {
    config: SessionConfig,
    transport: Arc<T>,
    discovery: Arc<Discovery>,
    pubsub: Arc<PubSub>,
    state: Arc<RwLock<SessionState>>,
    stats: Arc<RwLock<SessionStats>>,
    event_tx: broadcast::Sender<SessionEvent>,
    /// CRDT shared state (used when mode = Crdt)
    crdt_state: Arc<RwLock<LwwMap<String, Vec<u8>>>>,
}

impl<T: SyncTransport + 'static> SyncSession<T> {
    /// Get a reference to the transport.
    #[must_use]
    pub const fn transport(&self) -> &Arc<T> {
        &self.transport
    }

    /// Get a reference to the discovery system.
    #[must_use]
    pub const fn discovery(&self) -> &Arc<Discovery> {
        &self.discovery
    }

    /// Get a reference to the pub/sub router.
    #[must_use]
    pub const fn pubsub(&self) -> &Arc<PubSub> {
        &self.pubsub
    }

    /// Get the current session config.
    #[must_use]
    pub const fn config(&self) -> &SessionConfig {
        &self.config
    }

    /// Get current session state.
    pub async fn state(&self) -> SessionState {
        *self.state.read().await
    }

    /// Get current session stats.
    pub async fn stats(&self) -> SessionStats {
        self.stats.read().await.clone()
    }

    /// Subscribe to session events.
    #[must_use]
    pub fn subscribe_events(&self) -> broadcast::Receiver<SessionEvent> {
        self.event_tx.subscribe()
    }

    /// Add a manual peer.
    pub async fn add_peer(&self, addr: SocketAddr) {
        use std::collections::HashMap;
        self.discovery
            .register_peer(addr, &format!("peer-{addr}"), HashMap::new())
            .await;
    }

    /// Send a hello handshake to a peer.
    pub async fn handshake(
        &self,
        peer: SocketAddr,
        world_hash: WorldHash,
    ) -> Result<(), TransportError> {
        let hello = Message::Hello {
            node_id: self.config.node_id,
            seq: 0,
            world_hash,
        };
        self.transport.send_to(&hello, peer).await?;
        let mut stats = self.stats.write().await;
        stats.messages_sent += 1;
        Ok(())
    }

    /// Broadcast a message to all known peers.
    pub async fn broadcast(&self, message: &Message) -> Result<(), TransportError> {
        let peers = self.discovery.peer_addrs().await;
        self.transport.broadcast(message, &peers).await?;
        let mut stats = self.stats.write().await;
        stats.messages_sent += peers.len() as u64;
        Ok(())
    }

    /// Receive the next message from any peer.
    pub async fn recv(&self) -> Result<Envelope, TransportError> {
        let envelope = self.transport.recv().await?;
        // Update discovery timestamp
        self.discovery.peer_seen(&envelope.from).await;
        let mut stats = self.stats.write().await;
        stats.messages_received += 1;
        Ok(envelope)
    }

    /// Set a CRDT key-value (for CRDT mode).
    pub async fn crdt_set(&self, key: String, value: Vec<u8>) {
        self.crdt_state.write().await.insert(key, value);
    }

    /// Get a CRDT value by key.
    pub async fn crdt_get(&self, key: &str) -> Option<Vec<u8>> {
        let key_owned = key.to_string();
        self.crdt_state.read().await.get(&key_owned).cloned()
    }

    /// Merge remote CRDT state.
    pub async fn crdt_merge(&self, remote: &LwwMap<String, Vec<u8>>) {
        self.crdt_state.write().await.merge(remote);
    }

    /// Start the session tick loop.
    ///
    /// This spawns background tasks for:
    /// 1. Message receive loop (dispatches to event channel)
    /// 2. Fixed-rate tick loop (if `tick_rate` > 0)
    /// 3. Discovery GC (if discovery enabled)
    ///
    /// Returns immediately. Use `subscribe_events()` to process events.
    pub async fn start(&self) {
        *self.state.write().await = SessionState::Running;
        let _ = self
            .event_tx
            .send(SessionEvent::StateChanged(SessionState::Running));

        // Receive loop
        let transport = Arc::clone(&self.transport);
        let discovery = Arc::clone(&self.discovery);
        let event_tx = self.event_tx.clone();
        let stats = Arc::clone(&self.stats);
        let state = Arc::clone(&self.state);

        tokio::spawn(async move {
            loop {
                if *state.read().await == SessionState::Stopped {
                    break;
                }

                match transport.recv().await {
                    Ok(envelope) => {
                        discovery.peer_seen(&envelope.from).await;
                        stats.write().await.messages_received += 1;
                        let _ = event_tx.send(SessionEvent::MessageReceived(envelope));
                    }
                    Err(TransportError::Closed) => break,
                    Err(_) => {}
                }
            }
        });

        // Tick loop
        if self.config.tick_rate > 0 {
            let tick_interval = Duration::from_micros(1_000_000 / u64::from(self.config.tick_rate));
            let event_tx = self.event_tx.clone();
            let stats = Arc::clone(&self.stats);
            let state = Arc::clone(&self.state);

            tokio::spawn(async move {
                let mut interval = tokio::time::interval(tick_interval);
                let mut tick_seq = 0u64;

                loop {
                    interval.tick().await;

                    let current_state = *state.read().await;
                    if current_state == SessionState::Stopped {
                        break;
                    }
                    if current_state == SessionState::Paused {
                        continue;
                    }

                    tick_seq += 1;
                    stats.write().await.ticks = tick_seq;
                    let _ = event_tx.send(SessionEvent::Tick { seq: tick_seq });
                }
            });
        }

        // Discovery GC
        if self.config.enable_discovery {
            self.discovery.start_mdns_gc();
        }
    }

    /// Pause the session (stops tick processing, keeps connections alive).
    pub async fn pause(&self) {
        *self.state.write().await = SessionState::Paused;
        let _ = self
            .event_tx
            .send(SessionEvent::StateChanged(SessionState::Paused));
    }

    /// Resume a paused session.
    pub async fn resume(&self) {
        *self.state.write().await = SessionState::Running;
        let _ = self
            .event_tx
            .send(SessionEvent::StateChanged(SessionState::Running));
    }

    /// Stop the session.
    pub async fn stop(&self) {
        // Send Bye to all peers
        let peers = self.discovery.peer_addrs().await;
        for peer in &peers {
            let _ = self.transport.send_to(&Message::Bye, *peer).await;
        }

        *self.state.write().await = SessionState::Stopped;
        let _ = self
            .event_tx
            .send(SessionEvent::StateChanged(SessionState::Stopped));
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::ChannelTransport;

    #[tokio::test]
    async fn test_session_builder_defaults() {
        let (a, _b) = ChannelTransport::pair();
        let session = SessionBuilder::new().build(a);

        assert_eq!(session.config().mode, SyncMode::Crdt);
        assert_eq!(session.config().tick_rate, 60);
        assert_eq!(session.config().node_id, NodeId(1));
    }

    #[tokio::test]
    async fn test_session_builder_custom() {
        let (a, _b) = ChannelTransport::pair();
        let session = SessionBuilder::new()
            .mode(SyncMode::Rollback)
            .node_id(42)
            .tick_rate(120)
            .max_peers(8)
            .name("test-node")
            .discovery(false)
            .build(a);

        assert_eq!(session.config().mode, SyncMode::Rollback);
        assert_eq!(session.config().tick_rate, 120);
        assert_eq!(session.config().node_id, NodeId(42));
        assert_eq!(session.config().max_peers, 8);
        assert_eq!(session.config().instance_name, "test-node");
        assert!(!session.config().enable_discovery);
    }

    #[tokio::test]
    async fn test_session_state_transitions() {
        let (a, _b) = ChannelTransport::pair();
        let session = SessionBuilder::new().tick_rate(0).discovery(false).build(a);

        assert_eq!(session.state().await, SessionState::Building);

        session.start().await;
        assert_eq!(session.state().await, SessionState::Running);

        session.pause().await;
        assert_eq!(session.state().await, SessionState::Paused);

        session.resume().await;
        assert_eq!(session.state().await, SessionState::Running);

        session.stop().await;
        assert_eq!(session.state().await, SessionState::Stopped);
    }

    #[tokio::test]
    async fn test_session_add_peer() {
        let (a, _b) = ChannelTransport::pair();
        let session = SessionBuilder::new().discovery(false).build(a);

        let peer_addr: SocketAddr = "127.0.0.1:9999".parse().unwrap();
        session.add_peer(peer_addr).await;

        assert_eq!(session.discovery().peer_count().await, 1);
    }

    #[tokio::test]
    async fn test_session_crdt_operations() {
        let (a, _b) = ChannelTransport::pair();
        let session = SessionBuilder::new()
            .mode(SyncMode::Crdt)
            .node_id(1)
            .build(a);

        session
            .crdt_set("key1".to_string(), b"value1".to_vec())
            .await;
        session
            .crdt_set("key2".to_string(), b"value2".to_vec())
            .await;

        assert_eq!(session.crdt_get("key1").await, Some(b"value1".to_vec()));
        assert_eq!(session.crdt_get("key2").await, Some(b"value2".to_vec()));
        assert_eq!(session.crdt_get("key3").await, None);
    }

    #[tokio::test]
    async fn test_session_send_receive() {
        let (a, b) = ChannelTransport::pair();
        let addr_b = b.local_addr().unwrap();

        let session_a = SessionBuilder::new()
            .node_id(1)
            .discovery(false)
            .tick_rate(0)
            .build(a);

        let session_b = SessionBuilder::new()
            .node_id(2)
            .discovery(false)
            .tick_rate(0)
            .build(b);

        session_a
            .handshake(addr_b, WorldHash(0x1234))
            .await
            .unwrap();

        let envelope = session_b.recv().await.unwrap();
        match envelope.message {
            Message::Hello {
                node_id,
                world_hash,
                ..
            } => {
                assert_eq!(node_id, NodeId(1));
                assert_eq!(world_hash.0, 0x1234);
            }
            _ => panic!("expected Hello"),
        }

        let stats = session_a.stats().await;
        assert_eq!(stats.messages_sent, 1);
    }

    #[tokio::test]
    async fn test_session_broadcast() {
        let (a, b) = ChannelTransport::pair();
        let addr_b = b.local_addr().unwrap();

        let session_a = SessionBuilder::new()
            .node_id(1)
            .discovery(false)
            .tick_rate(0)
            .build(a);

        // Register peer
        session_a.add_peer(addr_b).await;

        session_a.broadcast(&Message::Bye).await.unwrap();

        let envelope = b.recv().await.unwrap();
        assert!(matches!(envelope.message, Message::Bye));
    }

    #[tokio::test]
    async fn test_session_pubsub_integration() {
        let (a, _b) = ChannelTransport::pair();
        let session = SessionBuilder::new().build(a);

        let peer: SocketAddr = "127.0.0.1:9999".parse().unwrap();
        let mut rx = session.pubsub().subscribe("test/topic", peer).await;

        session
            .pubsub()
            .publish("test/topic", b"hello".to_vec(), None)
            .await;

        let msg = rx.recv().await.unwrap();
        assert_eq!(msg.payload, b"hello");
    }
}
