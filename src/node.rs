/*
    ALICE-Sync
    Copyright (C) 2026 Moroya Sakamoto

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

//! Node management - participants in the P2P network
//!
//! v0.3: Zero-copy event access via slice references

use crate::{Event, EventStream, Result, SyncError, World, WorldHash};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Unique node identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, bitcode::Encode, bitcode::Decode)]
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
    world: World,
    events: EventStream,
    peers: HashMap<NodeId, Peer>,
    state: NodeState,
}

impl Node {
    /// Create a new node
    pub fn new(id: NodeId) -> Self {
        Self {
            id,
            world: World::new(0),
            events: EventStream::new(),
            peers: HashMap::new(),
            state: NodeState::Disconnected,
        }
    }

    /// Create a node with a specific world seed
    pub fn with_seed(id: NodeId, seed: u64) -> Self {
        Self {
            id,
            world: World::new(seed),
            events: EventStream::new(),
            peers: HashMap::new(),
            state: NodeState::Disconnected,
        }
    }

    /// Get current world hash (O(1))
    #[inline(always)]
    pub fn world_hash(&self) -> WorldHash {
        self.world.hash()
    }

    /// Get current state
    #[inline(always)]
    pub fn state(&self) -> NodeState {
        self.state
    }

    /// Generate a local event and apply it
    pub fn emit(&mut self, event: Event) -> Result<()> {
        self.events.push(event.clone(), self.id.0);
        let event_with_id = self.events.all().last().unwrap().clone();
        self.world.apply(&event_with_id)?;
        Ok(())
    }

    /// Apply an event (from local or remote)
    pub fn apply_event(&mut self, event: &Event) -> Result<()> {
        let mut event = event.clone();
        if event.seq.0 == 0 {
            self.events.push(event.clone(), event.origin);
            event = self.events.all().last().unwrap().clone();
        }
        self.world.apply(&event)
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

    /// Get events since a peer's last known sequence (ZERO-COPY: returns slice)
    #[inline]
    pub fn events_for_peer(&self, peer_id: NodeId) -> Result<&[Event]> {
        let peer = self.peers.get(&peer_id).ok_or(SyncError::UnknownNode(peer_id))?;
        let seq = crate::event::SeqNum(peer.last_seq);
        Ok(self.events.since(seq))
    }

    /// Update peer's sync state
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
    pub fn world(&self) -> &World {
        &self.world
    }

    /// Get event stream reference
    #[inline(always)]
    pub fn events(&self) -> &EventStream {
        &self.events
    }

    /// Statistics
    pub fn stats(&self) -> NodeStats {
        NodeStats {
            events_count: self.events.len(),
            events_bytes: self.events.total_bytes(),
            peers_count: self.peers.len(),
            synced_peers: self.peers.values().filter(|p| p.state == NodeState::Synced).count(),
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
    use crate::EventKind;

    #[test]
    fn test_node_sync() {
        let mut node_a = Node::new(NodeId(1));
        let mut node_b = Node::new(NodeId(2));

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

        for e in node_a.events().all() {
            node_b.apply_event(e).unwrap();
        }

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
        let mut node_a = Node::new(NodeId(1));
        let mut node_b = Node::new(NodeId(2));

        for i in 0..1000u32 {
            node_a
                .emit(Event::new(EventKind::Spawn {
                    entity: i,
                    kind: 0,
                    pos: [i as i16, 0, 0],
                }))
                .unwrap();
        }

        for e in node_a.events().all() {
            node_b.apply_event(e).unwrap();
        }

        assert_eq!(node_a.world_hash(), node_b.world_hash());
        assert_eq!(node_a.world().entity_count(), 1000);
    }
}
