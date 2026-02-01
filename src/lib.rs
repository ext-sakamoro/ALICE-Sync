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

//! # ALICE-Sync
//!
//! P2P synchronization via event diffing, not data transfer.
//!
//! > "Don't send data. Send the delta."
//!
//! ## v0.6 "Demon Mode" Optimizations
//!
//! 1. **Sort-Based Batching**: Events sorted by entity_id for sequential access
//! 2. **Coalescing**: Multiple motions to same entity merged before apply
//! 3. **Auto-SIMD Detection**: Contiguous 8-slot ranges trigger Vertical SIMD
//! 4. **Prefetcher Friendly**: 1.4x faster than random access pattern
//!
//! ## v0.5 "Zen Mode" Optimizations
//!
//! 1. **Complete SoA World**: Entity data decomposed into component arrays
//! 2. **Vertical SIMD**: Process 8 entities' x-coordinates in one instruction
//! 3. **7x Memory Reduction**: Motion reads 12 bytes/entity instead of 84 bytes
//! 4. **100% Cache Efficiency**: Only touch data you need
//! 5. **8-wide Parallel Hash**: Compute 8 entity hashes simultaneously

pub mod arena;
pub mod event;
pub mod fixed_point;
pub mod node;
pub mod protocol;
pub mod world;
pub mod world_soa;

pub use arena::{Arena, Handle};
pub use event::{
    Event, EventId, EventKind, EventMeta, EventStream, EventType,
    MotionData, SpawnData, DespawnData, PropertyData, InputData, TickData, CustomData,
    SeqNum, SoAStats,
};
pub use fixed_point::{Fixed, Vec3Fixed, Vec3Simd};
pub use node::{Node, NodeId, NodeState};
pub use protocol::{Message, Protocol};
pub use world::{Entity, EntityProps, World, WorldHash, WorldState, MAX_PROPS};
pub use world_soa::{WorldStorage, WorldSoA, Slot};

/// ALICE-Sync version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Result type for ALICE-Sync operations
pub type Result<T> = std::result::Result<T, SyncError>;

/// Error types for ALICE-Sync
#[derive(Debug, Clone, PartialEq)]
pub enum SyncError {
    CausalityViolation { expected: u64, got: u64 },
    StateDivergence { local: WorldHash, remote: WorldHash },
    UnknownNode(NodeId),
    NetworkError(String),
    SerializationError(String),
}

impl std::fmt::Display for SyncError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CausalityViolation { expected, got } => {
                write!(f, "Causality violation: expected seq {}, got {}", expected, got)
            }
            Self::StateDivergence { local, remote } => {
                write!(
                    f,
                    "State divergence: local {:016x} != remote {:016x}",
                    local.0, remote.0
                )
            }
            Self::UnknownNode(id) => write!(f, "Unknown node: {:?}", id),
            Self::NetworkError(msg) => write!(f, "Network error: {}", msg),
            Self::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
        }
    }
}

impl std::error::Error for SyncError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_sync() {
        let mut node_a = Node::new(NodeId(1));
        let mut node_b = Node::new(NodeId(2));

        let spawn = Event::new(EventKind::Spawn {
            entity: 1,
            kind: 0,
            pos: [0, 0, 0],
        });
        let motion = Event::new(EventKind::Motion {
            entity: 1,
            delta: [1000, 0, 0],
        });

        node_a.apply_event(&spawn).unwrap();
        node_a.apply_event(&motion).unwrap();

        node_b.apply_event(&spawn).unwrap();
        node_b.apply_event(&motion).unwrap();

        assert_eq!(node_a.world_hash(), node_b.world_hash());
    }

    #[test]
    fn test_bandwidth_efficiency() {
        let motion = Event::new(EventKind::Motion {
            entity: 42,
            delta: [100, 200, 300],
        });

        let compact = motion.to_compact_bytes();
        println!("Motion event (bitcode): {} bytes", compact.len());
        assert!(compact.len() < 40);
    }

    #[test]
    fn test_entity_is_copy() {
        let entity = Entity {
            id: 1,
            kind: 0,
            position: Vec3Fixed::ZERO,
            properties: EntityProps::EMPTY,
        };

        // Entity is Copy - this compiles
        let copy = entity;
        assert_eq!(copy.id, entity.id);
    }

    #[test]
    fn test_simd_position_update() {
        let mut pos = Vec3Fixed::from_f32(1.0, 2.0, 3.0);
        let delta = Vec3Simd::new(Fixed::ONE.0, Fixed::ONE.0, Fixed::ONE.0);

        delta.add_to_vec3(&mut pos);

        assert!((pos.x.to_f32() - 2.0).abs() < 0.01);
        assert!((pos.y.to_f32() - 3.0).abs() < 0.01);
        assert!((pos.z.to_f32() - 4.0).abs() < 0.01);
    }
}
