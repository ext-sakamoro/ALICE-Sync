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
//! | Mode | Best For | Implementation |
//! |------|----------|----------------|
//! | `Lockstep` | RTS, turn-based, < 4 players | Wraps [`LockstepSession`] |
//! | `Rollback` | Fighting games, FPS, action | Wraps [`RollbackSession`] |
//! | `Crdt` | Collaboration, IoT, databases | `LwwMap` eventual consistency |
//! | `EventSourcing` | Audit logs, financial, legal | Append-only log + replay |
//! | `Snapshot` | Late-join, recovery | Full state transfer |

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{broadcast, Mutex, RwLock};

use crate::channel::PubSub;
use crate::crdt::{CrdtMergeable, LwwMap};
use crate::discovery::Discovery;
use crate::input_sync::{InputFrame, LockstepSession, RollbackAction, RollbackSession, SyncResult};
use crate::protocol::Message;
use crate::transport::{Envelope, SyncTransport, TransportError};
use crate::{NodeId, WorldHash};

/// Synchronization mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncMode {
    /// Lockstep: all inputs collected before advancing.
    /// Uses [`LockstepSession`] from `input_sync`.
    Lockstep,
    /// Rollback: predict and advance immediately, correct on mismatch.
    /// Uses [`RollbackSession`] from `input_sync`.
    Rollback,
    /// CRDT: conflict-free replicated data types.
    /// Uses `LwwMap` for eventual consistency.
    Crdt,
    /// Event Sourcing: append-only event log with replay.
    /// Full audit trail, deterministic rebuild.
    EventSourcing,
    /// Snapshot: periodic full state transfer.
    /// Simple, good for late-join and recovery.
    Snapshot,
}

/// Event log entry for `EventSourcing` mode.
#[derive(Debug, Clone)]
pub struct EventEntry {
    /// Sequence number (auto-incremented, starts at 1)
    pub seq: u64,
    /// Serialized event data
    pub data: Vec<u8>,
    /// Node that originated this event
    pub origin: NodeId,
}

/// Snapshot data for Snapshot mode.
#[derive(Debug, Clone)]
pub struct SnapshotData {
    /// Frame/tick this snapshot was taken at
    pub frame: u64,
    /// Serialized state data
    pub data: Vec<u8>,
    /// State checksum for verification
    pub checksum: u64,
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
    /// Number of players (used by Lockstep and Rollback modes)
    pub player_count: u8,
    /// Local player ID (used by Rollback mode)
    pub local_player: u8,
    /// Maximum rollback frames (used by Rollback mode)
    pub max_rollback: u64,
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
            player_count: 2,
            local_player: 0,
            max_rollback: 8,
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

// ============================================================================
// Mode-specific internal state
// ============================================================================

/// Internal state that varies by sync mode.
/// Not exposed publicly — access through mode-specific methods on `SyncSession`.
enum ModeState {
    Lockstep(Arc<Mutex<LockstepSession>>),
    Rollback(Arc<Mutex<RollbackSession>>),
    Crdt(Arc<RwLock<LwwMap<String, Vec<u8>>>>),
    EventSourcing(Arc<RwLock<Vec<EventEntry>>>),
    Snapshot(Arc<RwLock<Option<SnapshotData>>>),
}

// ============================================================================
// Session Builder
// ============================================================================

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

    /// Set number of players (for Lockstep / Rollback modes).
    #[must_use]
    pub const fn player_count(mut self, count: u8) -> Self {
        self.config.player_count = count;
        self
    }

    /// Set local player ID (for Rollback mode).
    #[must_use]
    pub const fn local_player(mut self, id: u8) -> Self {
        self.config.local_player = id;
        self
    }

    /// Set maximum rollback frames (for Rollback mode).
    #[must_use]
    pub const fn max_rollback(mut self, frames: u64) -> Self {
        self.config.max_rollback = frames;
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

        let mode_state = match self.config.mode {
            SyncMode::Lockstep => ModeState::Lockstep(Arc::new(Mutex::new(LockstepSession::new(
                self.config.player_count,
            )))),
            SyncMode::Rollback => ModeState::Rollback(Arc::new(Mutex::new(RollbackSession::new(
                self.config.player_count,
                self.config.local_player,
                self.config.max_rollback,
            )))),
            SyncMode::Crdt => {
                ModeState::Crdt(Arc::new(RwLock::new(LwwMap::new(self.config.node_id.0))))
            }
            SyncMode::EventSourcing => ModeState::EventSourcing(Arc::new(RwLock::new(Vec::new()))),
            SyncMode::Snapshot => ModeState::Snapshot(Arc::new(RwLock::new(None))),
        };

        SyncSession {
            config: self.config,
            transport: Arc::new(transport),
            discovery,
            pubsub,
            state: Arc::new(RwLock::new(SessionState::Building)),
            stats: Arc::new(RwLock::new(SessionStats::default())),
            event_tx,
            mode_state,
        }
    }
}

impl Default for SessionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Sync Session
// ============================================================================

/// A live P2P sync session.
///
/// Composes transport, discovery, pub/sub, and sync mode into
/// a single orchestrated session. Use mode-specific methods
/// (`lockstep_*`, `rollback_*`, `crdt_*`, `event_*`, `snapshot_*`)
/// to interact with the session's sync logic.
pub struct SyncSession<T: SyncTransport> {
    config: SessionConfig,
    transport: Arc<T>,
    discovery: Arc<Discovery>,
    pubsub: Arc<PubSub>,
    state: Arc<RwLock<SessionState>>,
    stats: Arc<RwLock<SessionStats>>,
    event_tx: broadcast::Sender<SessionEvent>,
    mode_state: ModeState,
}

impl<T: SyncTransport + 'static> SyncSession<T> {
    // ========================================================================
    // Generic accessors
    // ========================================================================

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

    // ========================================================================
    // Peer management
    // ========================================================================

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
        self.discovery.peer_seen(&envelope.from).await;
        let mut stats = self.stats.write().await;
        stats.messages_received += 1;
        Ok(envelope)
    }

    // ========================================================================
    // Lockstep mode methods
    // ========================================================================

    /// Add a local or remote input (Lockstep mode).
    /// No-op if session is not in Lockstep mode.
    pub async fn lockstep_add_input(&self, input: InputFrame) {
        if let ModeState::Lockstep(ref session) = self.mode_state {
            session.lock().await.add_local_input(input);
        }
    }

    /// Add a remote player's input (Lockstep mode).
    /// No-op if session is not in Lockstep mode.
    pub async fn lockstep_add_remote(&self, input: InputFrame) {
        if let ModeState::Lockstep(ref session) = self.mode_state {
            session.lock().await.add_remote_input(input);
        }
    }

    /// Check if all inputs are ready for the next frame (Lockstep mode).
    /// Returns `false` if session is not in Lockstep mode.
    pub async fn lockstep_ready(&self) -> bool {
        if let ModeState::Lockstep(ref session) = self.mode_state {
            session.lock().await.ready_to_advance()
        } else {
            false
        }
    }

    /// Advance to the next frame, returning all players' inputs (Lockstep mode).
    /// Returns `None` if not ready or session is not in Lockstep mode.
    pub async fn lockstep_advance(&self) -> Option<Vec<InputFrame>> {
        if let ModeState::Lockstep(ref session) = self.mode_state {
            session.lock().await.advance()
        } else {
            None
        }
    }

    /// Record a checksum for verification (Lockstep mode).
    pub async fn lockstep_record_checksum(&self, frame: u64, checksum: u64) {
        if let ModeState::Lockstep(ref session) = self.mode_state {
            session.lock().await.record_checksum(frame, checksum);
        }
    }

    /// Verify a remote checksum (Lockstep mode).
    pub async fn lockstep_verify_checksum(&self, frame: u64, remote: u64) -> SyncResult {
        if let ModeState::Lockstep(ref session) = self.mode_state {
            session.lock().await.verify_checksum(frame, remote)
        } else {
            SyncResult::Ok
        }
    }

    /// Current confirmed frame (Lockstep mode). Returns 0 if wrong mode.
    pub async fn lockstep_confirmed_frame(&self) -> u64 {
        if let ModeState::Lockstep(ref session) = self.mode_state {
            session.lock().await.confirmed_frame()
        } else {
            0
        }
    }

    // ========================================================================
    // Rollback mode methods
    // ========================================================================

    /// Add local input and get all players' inputs for the frame (Rollback mode).
    /// Remote players' inputs are predicted if not yet received.
    /// Returns empty vec if session is not in Rollback mode.
    pub async fn rollback_add_local(&self, input: InputFrame) -> Vec<InputFrame> {
        if let ModeState::Rollback(ref session) = self.mode_state {
            session.lock().await.add_local_input(input)
        } else {
            Vec::new()
        }
    }

    /// Add a remote player's confirmed input (Rollback mode).
    /// Returns the action the game should take (None, Rollback, or Desync).
    pub async fn rollback_add_remote(&self, input: InputFrame) -> RollbackAction {
        if let ModeState::Rollback(ref session) = self.mode_state {
            session.lock().await.add_remote_input(input)
        } else {
            RollbackAction::None
        }
    }

    /// Save a state snapshot for potential rollback (Rollback mode).
    pub async fn rollback_save_snapshot(&self, frame: u64, state: Vec<u8>, checksum: u64) {
        if let ModeState::Rollback(ref session) = self.mode_state {
            session.lock().await.save_snapshot(frame, state, checksum);
        }
    }

    /// Get a snapshot for rollback to a specific frame (Rollback mode).
    pub async fn rollback_get_snapshot(&self, frame: u64) -> Option<Vec<u8>> {
        if let ModeState::Rollback(ref session) = self.mode_state {
            session.lock().await.get_snapshot(frame).map(<[u8]>::to_vec)
        } else {
            None
        }
    }

    /// Get inputs for a frame during re-simulation after rollback (Rollback mode).
    pub async fn rollback_inputs_for_frame(&self, frame: u64) -> Vec<InputFrame> {
        if let ModeState::Rollback(ref session) = self.mode_state {
            session.lock().await.inputs_for_frame(frame)
        } else {
            Vec::new()
        }
    }

    /// How many frames ahead of confirmation (Rollback mode). Returns 0 if wrong mode.
    pub async fn rollback_frames_ahead(&self) -> u64 {
        if let ModeState::Rollback(ref session) = self.mode_state {
            session.lock().await.frames_ahead()
        } else {
            0
        }
    }

    /// Current confirmed frame (Rollback mode). Returns 0 if wrong mode.
    pub async fn rollback_confirmed_frame(&self) -> u64 {
        if let ModeState::Rollback(ref session) = self.mode_state {
            session.lock().await.confirmed_frame()
        } else {
            0
        }
    }

    /// Verify a remote checksum (Rollback mode).
    pub async fn rollback_verify_checksum(&self, frame: u64, remote: u64) -> SyncResult {
        if let ModeState::Rollback(ref session) = self.mode_state {
            session.lock().await.verify_checksum(frame, remote)
        } else {
            SyncResult::Ok
        }
    }

    // ========================================================================
    // CRDT mode methods
    // ========================================================================

    /// Set a CRDT key-value (CRDT mode).
    pub async fn crdt_set(&self, key: String, value: Vec<u8>) {
        if let ModeState::Crdt(ref state) = self.mode_state {
            state.write().await.insert(key, value);
        }
    }

    /// Get a CRDT value by key (CRDT mode).
    pub async fn crdt_get(&self, key: &str) -> Option<Vec<u8>> {
        if let ModeState::Crdt(ref state) = self.mode_state {
            let key_owned = key.to_string();
            state.read().await.get(&key_owned).cloned()
        } else {
            None
        }
    }

    /// Merge remote CRDT state (CRDT mode).
    pub async fn crdt_merge(&self, remote: &LwwMap<String, Vec<u8>>) {
        if let ModeState::Crdt(ref state) = self.mode_state {
            state.write().await.merge(remote);
        }
    }

    // ========================================================================
    // EventSourcing mode methods
    // ========================================================================

    /// Append an event to the log (`EventSourcing` mode).
    /// The event is tagged with this node's ID and auto-assigned a sequence number.
    pub async fn event_append(&self, data: Vec<u8>) {
        self.event_append_from(data, self.config.node_id).await;
    }

    /// Append an event with an explicit origin node (`EventSourcing` mode).
    pub async fn event_append_from(&self, data: Vec<u8>, origin: NodeId) {
        if let ModeState::EventSourcing(ref log) = self.mode_state {
            let mut log = log.write().await;
            let seq = log.len() as u64 + 1;
            log.push(EventEntry { seq, data, origin });
        }
    }

    /// Get all events since a given sequence number (exclusive).
    /// Returns events where `entry.seq > since_seq`.
    pub async fn event_log_since(&self, since_seq: u64) -> Vec<EventEntry> {
        if let ModeState::EventSourcing(ref log) = self.mode_state {
            log.read()
                .await
                .iter()
                .filter(|e| e.seq > since_seq)
                .cloned()
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Total number of events in the log (`EventSourcing` mode).
    pub async fn event_count(&self) -> usize {
        if let ModeState::EventSourcing(ref log) = self.mode_state {
            log.read().await.len()
        } else {
            0
        }
    }

    /// Last sequence number in the log. Returns 0 if empty or wrong mode.
    pub async fn event_last_seq(&self) -> u64 {
        if let ModeState::EventSourcing(ref log) = self.mode_state {
            log.read().await.last().map_or(0, |e| e.seq)
        } else {
            0
        }
    }

    // ========================================================================
    // Snapshot mode methods
    // ========================================================================

    /// Save a state snapshot (Snapshot mode).
    /// Replaces any previous snapshot.
    pub async fn snapshot_save(&self, frame: u64, data: Vec<u8>, checksum: u64) {
        if let ModeState::Snapshot(ref store) = self.mode_state {
            *store.write().await = Some(SnapshotData {
                frame,
                data,
                checksum,
            });
        }
    }

    /// Get the current snapshot (Snapshot mode).
    pub async fn snapshot_get(&self) -> Option<SnapshotData> {
        if let ModeState::Snapshot(ref store) = self.mode_state {
            store.read().await.clone()
        } else {
            None
        }
    }

    /// Get the current snapshot's frame number. Returns `None` if no snapshot or wrong mode.
    pub async fn snapshot_frame(&self) -> Option<u64> {
        if let ModeState::Snapshot(ref store) = self.mode_state {
            store.read().await.as_ref().map(|s| s.frame)
        } else {
            None
        }
    }

    // ========================================================================
    // Session lifecycle
    // ========================================================================

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
        assert_eq!(session.config().player_count, 2);
        assert_eq!(session.config().local_player, 0);
        assert_eq!(session.config().max_rollback, 8);
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
            .player_count(4)
            .local_player(1)
            .max_rollback(12)
            .build(a);

        assert_eq!(session.config().mode, SyncMode::Rollback);
        assert_eq!(session.config().tick_rate, 120);
        assert_eq!(session.config().node_id, NodeId(42));
        assert_eq!(session.config().max_peers, 8);
        assert_eq!(session.config().instance_name, "test-node");
        assert!(!session.config().enable_discovery);
        assert_eq!(session.config().player_count, 4);
        assert_eq!(session.config().local_player, 1);
        assert_eq!(session.config().max_rollback, 12);
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

    // ========================================================================
    // Lockstep mode tests
    // ========================================================================

    #[tokio::test]
    async fn test_session_lockstep_basic() {
        let (a, _b) = ChannelTransport::pair();
        let session = SessionBuilder::new()
            .mode(SyncMode::Lockstep)
            .player_count(2)
            .build(a);

        // Player 0 input
        session
            .lockstep_add_input(InputFrame::new(1, 0).with_movement(1, 0, 0))
            .await;
        assert!(!session.lockstep_ready().await);

        // Player 1 input
        session
            .lockstep_add_remote(InputFrame::new(1, 1).with_movement(0, 0, -1))
            .await;
        assert!(session.lockstep_ready().await);

        // Advance
        let inputs = session.lockstep_advance().await.unwrap();
        assert_eq!(inputs.len(), 2);
        assert_eq!(inputs[0].movement[0], 1);
        assert_eq!(inputs[1].movement[2], -1);
        assert_eq!(session.lockstep_confirmed_frame().await, 1);
    }

    #[tokio::test]
    async fn test_session_lockstep_checksum() {
        let (a, _b) = ChannelTransport::pair();
        let session = SessionBuilder::new()
            .mode(SyncMode::Lockstep)
            .player_count(2)
            .build(a);

        session.lockstep_record_checksum(1, 0xABCD).await;
        assert_eq!(
            session.lockstep_verify_checksum(1, 0xABCD).await,
            SyncResult::Ok
        );
        assert_eq!(
            session.lockstep_verify_checksum(1, 0xDEAD).await,
            SyncResult::Desync {
                frame: 1,
                local: 0xABCD,
                remote: 0xDEAD
            }
        );
    }

    // ========================================================================
    // Rollback mode tests
    // ========================================================================

    #[tokio::test]
    async fn test_session_rollback_basic() {
        let (a, _b) = ChannelTransport::pair();
        let session = SessionBuilder::new()
            .mode(SyncMode::Rollback)
            .player_count(2)
            .local_player(0)
            .max_rollback(8)
            .build(a);

        // Local input for frame 1
        let inputs = session
            .rollback_add_local(InputFrame::new(1, 0).with_movement(1, 0, 0))
            .await;
        assert_eq!(inputs.len(), 2);

        // Remote input matches prediction → no rollback
        let action = session.rollback_add_remote(InputFrame::new(1, 1)).await;
        assert_eq!(action, RollbackAction::None);
    }

    #[tokio::test]
    async fn test_session_rollback_mismatch() {
        let (a, _b) = ChannelTransport::pair();
        let session = SessionBuilder::new()
            .mode(SyncMode::Rollback)
            .player_count(2)
            .local_player(0)
            .max_rollback(8)
            .build(a);

        // Advance 2 frames
        session
            .rollback_add_local(InputFrame::new(1, 0).with_movement(1, 0, 0))
            .await;
        session
            .rollback_add_local(InputFrame::new(2, 0).with_movement(1, 0, 0))
            .await;

        // Remote for frame 1 differs from prediction → rollback
        let action = session
            .rollback_add_remote(InputFrame::new(1, 1).with_movement(5, 5, 5))
            .await;
        assert_eq!(action, RollbackAction::Rollback { to_frame: 1 });
    }

    #[tokio::test]
    async fn test_session_rollback_snapshot() {
        let (a, _b) = ChannelTransport::pair();
        let session = SessionBuilder::new()
            .mode(SyncMode::Rollback)
            .player_count(2)
            .local_player(0)
            .build(a);

        session
            .rollback_save_snapshot(1, vec![1, 2, 3], 0xAAAA)
            .await;
        let snap = session.rollback_get_snapshot(1).await.unwrap();
        assert_eq!(snap, vec![1, 2, 3]);
        assert_eq!(
            session.rollback_verify_checksum(1, 0xAAAA).await,
            SyncResult::Ok
        );
    }

    #[tokio::test]
    async fn test_session_rollback_frames_ahead() {
        let (a, _b) = ChannelTransport::pair();
        let session = SessionBuilder::new()
            .mode(SyncMode::Rollback)
            .player_count(2)
            .local_player(0)
            .build(a);

        session.rollback_add_local(InputFrame::new(1, 0)).await;
        session.rollback_add_local(InputFrame::new(2, 0)).await;
        session.rollback_add_local(InputFrame::new(3, 0)).await;

        assert_eq!(session.rollback_frames_ahead().await, 3);

        session.rollback_add_remote(InputFrame::new(1, 1)).await;
        assert_eq!(session.rollback_confirmed_frame().await, 1);
        assert_eq!(session.rollback_frames_ahead().await, 2);
    }

    // ========================================================================
    // EventSourcing mode tests
    // ========================================================================

    #[tokio::test]
    async fn test_session_event_sourcing_basic() {
        let (a, _b) = ChannelTransport::pair();
        let session = SessionBuilder::new()
            .mode(SyncMode::EventSourcing)
            .node_id(1)
            .build(a);

        session.event_append(b"event-1".to_vec()).await;
        session.event_append(b"event-2".to_vec()).await;
        session.event_append(b"event-3".to_vec()).await;

        assert_eq!(session.event_count().await, 3);
        assert_eq!(session.event_last_seq().await, 3);
    }

    #[tokio::test]
    async fn test_session_event_sourcing_replay() {
        let (a, _b) = ChannelTransport::pair();
        let session = SessionBuilder::new()
            .mode(SyncMode::EventSourcing)
            .node_id(1)
            .build(a);

        session.event_append(b"a".to_vec()).await;
        session.event_append(b"b".to_vec()).await;
        session.event_append(b"c".to_vec()).await;

        // Get events since seq 1 (exclusive) → seq 2, 3
        let events = session.event_log_since(1).await;
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].seq, 2);
        assert_eq!(events[0].data, b"b");
        assert_eq!(events[1].seq, 3);
        assert_eq!(events[1].data, b"c");

        // Get all events (since 0)
        let all = session.event_log_since(0).await;
        assert_eq!(all.len(), 3);
    }

    #[tokio::test]
    async fn test_session_event_sourcing_origin() {
        let (a, _b) = ChannelTransport::pair();
        let session = SessionBuilder::new()
            .mode(SyncMode::EventSourcing)
            .node_id(1)
            .build(a);

        session.event_append(b"local".to_vec()).await;
        session
            .event_append_from(b"remote".to_vec(), NodeId(99))
            .await;

        let events = session.event_log_since(0).await;
        assert_eq!(events[0].origin, NodeId(1));
        assert_eq!(events[1].origin, NodeId(99));
    }

    // ========================================================================
    // Snapshot mode tests
    // ========================================================================

    #[tokio::test]
    async fn test_session_snapshot_basic() {
        let (a, _b) = ChannelTransport::pair();
        let session = SessionBuilder::new().mode(SyncMode::Snapshot).build(a);

        assert!(session.snapshot_get().await.is_none());
        assert!(session.snapshot_frame().await.is_none());

        session.snapshot_save(42, vec![10, 20, 30], 0xDEAD).await;

        let snap = session.snapshot_get().await.unwrap();
        assert_eq!(snap.frame, 42);
        assert_eq!(snap.data, vec![10, 20, 30]);
        assert_eq!(snap.checksum, 0xDEAD);
        assert_eq!(session.snapshot_frame().await, Some(42));
    }

    #[tokio::test]
    async fn test_session_snapshot_overwrites() {
        let (a, _b) = ChannelTransport::pair();
        let session = SessionBuilder::new().mode(SyncMode::Snapshot).build(a);

        session.snapshot_save(1, vec![1], 0x1).await;
        session.snapshot_save(2, vec![2], 0x2).await;

        let snap = session.snapshot_get().await.unwrap();
        assert_eq!(snap.frame, 2);
        assert_eq!(snap.data, vec![2]);
    }

    // ========================================================================
    // Wrong-mode safety tests
    // ========================================================================

    #[tokio::test]
    async fn test_wrong_mode_returns_defaults() {
        let (a, _b) = ChannelTransport::pair();
        let session = SessionBuilder::new().mode(SyncMode::Crdt).build(a);

        // Lockstep methods on CRDT session → safe defaults
        assert!(!session.lockstep_ready().await);
        assert!(session.lockstep_advance().await.is_none());
        assert_eq!(session.lockstep_confirmed_frame().await, 0);

        // Rollback methods on CRDT session → safe defaults
        assert!(session
            .rollback_add_local(InputFrame::new(1, 0))
            .await
            .is_empty());
        assert_eq!(
            session.rollback_add_remote(InputFrame::new(1, 1)).await,
            RollbackAction::None
        );
        assert_eq!(session.rollback_frames_ahead().await, 0);

        // EventSourcing on CRDT → safe defaults
        assert_eq!(session.event_count().await, 0);
        assert!(session.event_log_since(0).await.is_empty());

        // Snapshot on CRDT → safe defaults
        assert!(session.snapshot_get().await.is_none());
    }
}
