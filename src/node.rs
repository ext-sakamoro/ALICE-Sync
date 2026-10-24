/*
    ALICE-Sync
    Copyright (C) 2026 Moroya Sakamoto

    License: LicenseRef-Proprietary (see LICENSE-Proprietary)
*/

//! Node management - participants in the P2P network
//!
//! v0.3: Zero-copy event access via slice references

use crate::authorship::{Author, AuthorshipError, SignedEvents, TrustPolicy, Verifier};
use crate::converge::Converge;
use crate::event::Origin;
use crate::{Event, EventStream, Result, SyncError, World, WorldHash};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Unique node identifier
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    bitcode::Encode,
    bitcode::Decode,
)]
pub struct NodeId(pub u64);

/// Node connection state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeState {
    Disconnected,
    Connecting,
    Synced,
    Diverged,
}

/// Peer information
#[derive(Debug, Clone)]
pub struct Peer {
    pub id: NodeId,
    pub state: NodeState,
    pub last_seq: u64,
    pub world_hash: Option<WorldHash>,
}

/// A node in the ALICE-Sync network
#[derive(Debug)]
pub struct Node {
    pub id: NodeId,
    /// the origin this node's events carry: its author's, or [`Origin::of_node`] without one
    origin: Origin,
    /// the signing side of this node's events; without one, other nodes do not accept them
    author: Option<Author>,
    /// the authors' checkpoints and events received from other nodes
    verifier: Verifier,
    world: World,
    /// the keyed records the world is derived from (ALICE-Law G13: the same world for every
    /// arrival order of the same events)
    converge: Converge,
    events: EventStream,
    peers: BTreeMap<NodeId, Peer>,
    state: NodeState,
}

impl Node {
    /// Create a new node
    #[must_use]
    pub fn new(id: NodeId) -> Self {
        Self {
            id,
            origin: Origin::of_node(id.0),
            author: None,
            verifier: Verifier::new(),
            world: World::new(0),
            converge: Converge::new(),
            events: EventStream::new(),
            peers: BTreeMap::new(),
            state: NodeState::Disconnected,
        }
    }

    /// Create a node with a specific world seed
    #[must_use]
    pub fn with_seed(id: NodeId, seed: u64) -> Self {
        Self {
            id,
            origin: Origin::of_node(id.0),
            author: None,
            verifier: Verifier::new(),
            world: World::new(seed),
            converge: Converge::new(),
            events: EventStream::new(),
            peers: BTreeMap::new(),
            state: NodeState::Disconnected,
        }
    }

    /// A node whose events `author` signs: they carry the author's origin, and other nodes accept them
    /// under the author's checkpoints ([`crate::authorship`])
    #[must_use]
    pub fn with_author(id: NodeId, author: Author) -> Self {
        let mut node = Self::new(id);
        node.origin = author.origin();
        node.author = Some(author);
        node
    }

    /// The origin this node's events carry
    #[must_use]
    pub const fn origin(&self) -> Origin {
        self.origin
    }

    /// Get current world hash (O(1))
    #[inline(always)]
    #[must_use]
    pub const fn world_hash(&self) -> WorldHash {
        self.world.hash()
    }

    /// Get current state
    #[inline(always)]
    #[must_use]
    pub const fn state(&self) -> NodeState {
        self.state
    }

    /// Generate a local event and apply it.
    ///
    /// # Errors
    ///
    /// Returns an error if the event fails to apply to the world state.
    ///
    /// # Panics
    ///
    /// Panics if the event stream is empty after pushing (should never happen).
    pub fn emit(&mut self, event: Event) -> Result<()> {
        self.events.push(event, self.origin);
        let event_with_id = self.events.all().last().unwrap();
        if let Some(author) = &mut self.author {
            author
                .extend(event_with_id)
                .map_err(SyncError::Authorship)?;
        }
        self.converge.apply(&mut self.world, event_with_id)
    }

    /// The signed events this node holds past `since` (per origin, the sequence number a peer already
    /// has; 0 for an origin it does not name): its own, with a checkpoint at its last event, and those it
    /// accepted from other authors, with their checkpoints
    ///
    /// # Errors
    /// [`AuthorshipError::NoAuthor`] for a node without an author, [`AuthorshipError::Credential`]
    pub fn signed_events_since(
        &mut self,
        since: &BTreeMap<Origin, u64>,
    ) -> std::result::Result<SignedEvents, AuthorshipError> {
        let author = self.author.as_mut().ok_or(AuthorshipError::NoAuthor)?;
        author.sign_now()?;
        let from = since.get(&self.origin).copied().unwrap_or(0);
        let mut out = SignedEvents {
            events: self.events.since(crate::event::SeqNum(from)).to_vec(),
            checkpoints: author
                .checkpoints()
                .range(from + 1..)
                .map(|(_, c)| c.clone())
                .collect(),
        };
        let others = self.verifier.since(since);
        out.events.extend(others.events);
        out.checkpoints.extend(others.checkpoints);
        Ok(out)
    }

    /// Take signed events from another node: each author's events apply once its verified checkpoint
    /// covers them ([`crate::authorship`]); returns how many applied. This node's own events in the batch
    /// are passed over (it applied them when it emitted them).
    ///
    /// # Errors
    /// As [`Verifier::receive`]
    pub fn receive(
        &mut self,
        batch: &SignedEvents,
        policy: &impl TrustPolicy,
    ) -> std::result::Result<usize, AuthorshipError> {
        let own = self.origin;
        let theirs = SignedEvents {
            events: batch
                .events
                .iter()
                .filter(|e| e.origin != own)
                .cloned()
                .collect(),
            checkpoints: batch
                .checkpoints
                .iter()
                .filter(|c| c.origin != own)
                .cloned()
                .collect(),
        };
        let (events, converge, world) = (&mut self.events, &mut self.converge, &mut self.world);
        self.verifier.receive(&theirs, policy, |e| {
            events.observe(e.timestamp);
            converge.apply(world, e).map_err(|err| err.to_string())
        })
    }

    /// Add a peer
    pub fn add_peer(&mut self, peer_id: NodeId) {
        self.peers.insert(
            peer_id,
            Peer {
                id: peer_id,
                state: NodeState::Connecting,
                last_seq: 0,
                world_hash: None,
            },
        );
    }

    /// Get events since a peer's last known sequence (ZERO-COPY: returns slice).
    ///
    /// # Errors
    ///
    /// Returns `SyncError::UnknownNode` if the peer is not registered.
    #[inline]
    pub fn events_for_peer(&self, peer_id: NodeId) -> Result<&[Event]> {
        let peer = self
            .peers
            .get(&peer_id)
            .ok_or(SyncError::UnknownNode(peer_id))?;
        let seq = crate::event::SeqNum(peer.last_seq);
        Ok(self.events.since(seq))
    }

    /// Update peer's sync state.
    ///
    /// # Errors
    ///
    /// Returns `SyncError::UnknownNode` if the peer is not registered,
    /// or `SyncError::StateDivergence` if world hashes don't match.
    pub fn update_peer(
        &mut self,
        peer_id: NodeId,
        last_seq: u64,
        world_hash: WorldHash,
    ) -> Result<()> {
        let local_hash = self.world_hash();
        let peer = self
            .peers
            .get_mut(&peer_id)
            .ok_or(SyncError::UnknownNode(peer_id))?;
        peer.last_seq = last_seq;
        peer.world_hash = Some(world_hash);

        if world_hash != local_hash {
            peer.state = NodeState::Diverged;
            return Err(SyncError::StateDivergence {
                local: local_hash,
                remote: world_hash,
            });
        }

        peer.state = NodeState::Synced;
        Ok(())
    }

    /// Get all peers
    pub fn peers(&self) -> impl Iterator<Item = &Peer> {
        self.peers.values()
    }

    /// Get world reference
    #[inline(always)]
    #[must_use]
    pub const fn world(&self) -> &World {
        &self.world
    }

    /// Get event stream reference
    #[inline(always)]
    #[must_use]
    pub const fn events(&self) -> &EventStream {
        &self.events
    }

    /// Statistics
    #[must_use]
    pub fn stats(&self) -> NodeStats {
        NodeStats {
            events_count: self.events.len(),
            events_bytes: self.events.total_bytes(),
            peers_count: self.peers.len(),
            synced_peers: self
                .peers
                .values()
                .filter(|p| p.state == NodeState::Synced)
                .count(),
            entity_count: self.world.entity_count(),
        }
    }
}

/// Node statistics
#[derive(Debug, Clone)]
pub struct NodeStats {
    pub events_count: usize,
    pub events_bytes: usize,
    pub peers_count: usize,
    pub synced_peers: usize,
    pub entity_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::authorship::testing;
    use crate::EventKind;

    #[test]
    fn test_node_sync() {
        let mut node_a = Node::with_author(NodeId(1), testing::author(1));
        let mut node_b = Node::with_author(NodeId(2), testing::author(2));

        node_a
            .emit(Event::new(EventKind::Spawn {
                entity: 1,
                kind: 0,
                pos: [0, 0, 0],
            }))
            .unwrap();

        node_a
            .emit(Event::new(EventKind::Motion {
                entity: 1,
                delta: [1000, 0, 0],
            }))
            .unwrap();

        let batch = node_a
            .signed_events_since(&std::collections::BTreeMap::new())
            .unwrap();
        node_b.receive(&batch, &testing::trusting(&[1])).unwrap();

        assert_eq!(node_a.world_hash(), node_b.world_hash());
    }

    #[test]
    fn test_zero_copy_events() {
        let mut node = Node::new(NodeId(1));
        node.add_peer(NodeId(2));

        // Emit some events
        for i in 0..100u32 {
            node.emit(Event::new(EventKind::Spawn {
                entity: i,
                kind: 0,
                pos: [0, 0, 0],
            }))
            .unwrap();
        }

        // Get events for peer (zero-copy slice)
        let events = node.events_for_peer(NodeId(2)).unwrap();
        assert_eq!(events.len(), 100);

        // Verify it's a slice, not a copy
        let ptr1 = events.as_ptr();
        let ptr2 = node.events().all().as_ptr();
        assert_eq!(ptr1, ptr2); // Same memory!
    }

    #[test]
    fn test_many_entities_sync() {
        let mut node_a = Node::with_author(NodeId(1), testing::author(1));
        let mut node_b = Node::with_author(NodeId(2), testing::author(2));

        for i in 0..1000u32 {
            node_a
                .emit(Event::new(EventKind::Spawn {
                    entity: i,
                    kind: 0,
                    pos: [i as i16, 0, 0],
                }))
                .unwrap();
        }

        let batch = node_a
            .signed_events_since(&std::collections::BTreeMap::new())
            .unwrap();
        node_b.receive(&batch, &testing::trusting(&[1])).unwrap();

        assert_eq!(node_a.world_hash(), node_b.world_hash());
        assert_eq!(node_a.world().entity_count(), 1000);
    }

    // --- Node 追加テスト ---

    #[test]
    fn test_node_initial_state() {
        let node = Node::new(NodeId(42));
        assert_eq!(node.id.0, 42);
        assert_eq!(node.state(), NodeState::Disconnected);
        assert_eq!(node.events().len(), 0);
        assert_eq!(node.world().entity_count(), 0);
    }

    #[test]
    fn test_node_with_seed() {
        let node = Node::with_seed(NodeId(1), 0xCAFE);
        // with_seed は正常に生成できる
        assert_eq!(node.id.0, 1);
    }

    #[test]
    fn test_node_stats() {
        let mut node = Node::new(NodeId(1));
        node.add_peer(NodeId(2));

        node.emit(Event::new(EventKind::Spawn {
            entity: 1,
            kind: 0,
            pos: [0, 0, 0],
        }))
        .unwrap();

        let stats = node.stats();
        assert_eq!(stats.events_count, 1);
        assert_eq!(stats.peers_count, 1);
        assert_eq!(stats.synced_peers, 0);
        assert_eq!(stats.entity_count, 1);
        assert!(stats.events_bytes > 0);
    }

    #[test]
    fn test_node_update_peer_diverged() {
        let mut node = Node::new(NodeId(1));
        node.add_peer(NodeId(2));

        // ノードに何かスポーンさせて hash != zero にする
        node.emit(Event::new(EventKind::Spawn {
            entity: 1,
            kind: 0,
            pos: [0, 0, 0],
        }))
        .unwrap();

        // 間違ったハッシュを渡すと StateDivergence エラー
        let wrong_hash = WorldHash(0xDEAD);
        let result = node.update_peer(NodeId(2), 1, wrong_hash);
        assert!(result.is_err());
        match result {
            Err(crate::SyncError::StateDivergence { .. }) => {}
            _ => panic!("expected StateDivergence"),
        }
    }

    #[test]
    fn test_node_update_peer_synced() {
        let mut node = Node::new(NodeId(1));
        node.add_peer(NodeId(2));

        // ハッシュが一致すれば Synced になる
        let local_hash = node.world_hash();
        node.update_peer(NodeId(2), 0, local_hash).unwrap();

        let synced = node
            .peers()
            .filter(|p| p.state == NodeState::Synced)
            .count();
        assert_eq!(synced, 1);
    }

    #[test]
    fn test_node_events_for_peer_unknown_returns_error() {
        let node = Node::new(NodeId(1));
        let result = node.events_for_peer(NodeId(99));
        assert!(result.is_err());
        match result {
            Err(crate::SyncError::UnknownNode(id)) => assert_eq!(id.0, 99),
            _ => panic!("expected UnknownNode"),
        }
    }

    #[test]
    fn test_node_update_peer_unknown_returns_error() {
        let mut node = Node::new(NodeId(1));
        let result = node.update_peer(NodeId(99), 0, WorldHash(0));
        assert!(matches!(result, Err(crate::SyncError::UnknownNode(_))));
    }

    #[test]
    fn test_node_peers_iterator() {
        let mut node = Node::new(NodeId(1));
        node.add_peer(NodeId(2));
        node.add_peer(NodeId(3));

        let count = node.peers().count();
        assert_eq!(count, 2);
    }
}
