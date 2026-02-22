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

// Justified pedantic suppression for SIMD/fixed-point/networking code:
// - inline_always: SIMD hot paths, fixed-point arithmetic, hash mixing
// - cast_*: intentional type narrowing in network serialization and SIMD
// - similar_names: dx/dy/dz, pos_x/pos_y/pos_z standard in spatial code
// - too_many_lines: complex SIMD batching with multiple code paths
// - module_name_repetitions: bridge module types mirror crate names
#![allow(
    clippy::inline_always,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::similar_names,
    clippy::module_name_repetitions,
    clippy::too_many_lines
)]

//! # ALICE-Sync
//!
//! P2P synchronization via event diffing, not data transfer.
//!
//! > "Don't send data. Send the delta."
//!
//! ## Quick Start
//!
//! ```rust
//! use alice_sync::{Node, NodeId, Event, EventKind};
//!
//! // Two nodes process the same events → identical world hash
//! let mut node_a = Node::new(NodeId(1));
//! let mut node_b = Node::new(NodeId(2));
//!
//! let spawn = Event::new(EventKind::Spawn { entity: 1, kind: 0, pos: [0, 0, 0] });
//! let motion = Event::new(EventKind::Motion { entity: 1, delta: [100, 0, 0] });
//!
//! node_a.apply_event(&spawn).unwrap();
//! node_a.apply_event(&motion).unwrap();
//! node_b.apply_event(&spawn).unwrap();
//! node_b.apply_event(&motion).unwrap();
//!
//! assert_eq!(node_a.world_hash(), node_b.world_hash());
//! ```
//!
//! ## Modules
//!
//! ### Core
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`arena`] | Generational arena allocator with O(1) insert/remove |
//! | [`event`] | Event types, `SoA` event storage, bitcode serialization |
//! | [`fixed_point`] | Q16.16 fixed-point arithmetic, SIMD `Vec3Simd` |
//! | [`node`] | P2P node with causal event ordering |
//! | [`protocol`] | Handshake / Sync / Ack / `HashCheck` wire protocol |
//! | [`world`] | `AoS` entity world with O(1) incremental hashing |
//! | [`world_soa`] | `SoA` world with Demon Mode batch processing + 8-wide SIMD |
//! | [`input_sync`] | Lockstep and Rollback input synchronization sessions |
//!
//! ### Feature-Gated Bridges
//!
//! | Module | Feature | Description |
//! |--------|---------|-------------|
//! | `physics_bridge` | `physics` | ALICE-Physics deterministic simulation bridge |
//! | `telemetry` | `telemetry` | ALICE-DB time-series sync metrics |
//! | `cache_bridge` | `cache` | Markov oracle entity prefetching via ALICE-Cache |
//! | `auth_bridge` | `auth` | Ed25519 ZKP peer authentication via ALICE-Auth |
//! | `codec_bridge` | `codec` | Wavelet + rANS event stream compression via ALICE-Codec |
//! | `analytics_bridge` | `analytics` | DDSketch/HLL/CMS probabilistic telemetry via ALICE-Analytics |
//! | `cloud_bridge` | `cloud` | Star-topology multi-device spatial sync hub |
//!
//! ## Cargo Features
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | `std` | Yes | Standard library support |
//! | `async` | No | Tokio async runtime integration |
//! | `simd` | No | Explicit SIMD acceleration hints |
//! | `python` | No | `PyO3` + `NumPy` zero-copy bindings |
//! | `physics` | No | ALICE-Physics bridge (`InputFrame` ↔ `FrameInput`, rollback) |
//! | `telemetry` | No | ALICE-DB time-series sync telemetry |
//! | `cache` | No | Markov-oracle entity prefetching via ALICE-Cache |
//! | `auth` | No | Ed25519 ZKP peer authentication via ALICE-Auth |
//! | `codec` | No | Wavelet + rANS event compression via ALICE-Codec |
//! | `analytics` | No | Probabilistic sketch telemetry via ALICE-Analytics |
//! | `cloud` | No | Cloud-side multi-device spatial synchronization |
//! | `all-bridges` | No | Enable all integration bridges at once |
//!
//! ## Optimization History
//!
//! - **v0.6 "Demon Mode"**: Sort-based batching, coalescing, auto-SIMD detection
//! - **v0.5 "Zen Mode"**: Complete `SoA` world, vertical 8-wide SIMD, 7x memory reduction
//! - **v0.4 "Ultra Mode"**: Branchless hashing, `SoA` event storage
//! - **v0.3 "God Mode"**: XOR rolling hash, Vec direct indexing, Copy entities
//! - **v0.2**: Fixed-point determinism, arena allocation, bitcode serialization
//! - **v0.1**: Core event system, basic world state

#[cfg(feature = "analytics")]
pub mod analytics_bridge;
pub mod arena;
#[cfg(feature = "auth")]
pub mod auth_bridge;
#[cfg(feature = "cache")]
pub mod cache_bridge;
#[cfg(feature = "cloud")]
pub mod cloud_bridge;
#[cfg(feature = "codec")]
pub mod codec_bridge;
pub mod event;
pub mod fixed_point;
pub mod input_sync;
pub mod node;
#[cfg(feature = "physics")]
pub mod physics_bridge;
pub mod protocol;
#[cfg(feature = "python")]
mod python;
#[cfg(feature = "telemetry")]
pub mod telemetry;
pub mod world;
pub mod world_soa;

pub use arena::{Arena, Handle};
pub use event::{
    CustomData, DespawnData, Event, EventId, EventKind, EventMeta, EventStream, EventType,
    InputData, MotionData, PropertyData, SeqNum, SoAStats, SpawnData, TickData,
};
pub use fixed_point::{Fixed, Vec3Fixed, Vec3Simd};
pub use input_sync::{
    InputBuffer, InputFrame, LockstepSession, RollbackAction, RollbackSession, SyncResult,
};
pub use node::{Node, NodeId, NodeState};
#[cfg(feature = "physics")]
pub use physics_bridge::{
    physics_checksum_to_world_hash, physics_input_to_sync, sync_input_to_physics,
    sync_inputs_to_physics, world_hash_to_physics_checksum, PhysicsRollbackSession,
};
pub use protocol::{Message, Protocol};
pub use world::{Entity, EntityProps, World, WorldHash, WorldState, MAX_PROPS};
pub use world_soa::{Slot, WorldSoA, WorldStorage};

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
                write!(f, "Causality violation: expected seq {expected}, got {got}")
            }
            Self::StateDivergence { local, remote } => {
                write!(
                    f,
                    "State divergence: local {:016x} != remote {:016x}",
                    local.0, remote.0
                )
            }
            Self::UnknownNode(id) => write!(f, "Unknown node: {id:?}"),
            Self::NetworkError(msg) => write!(f, "Network error: {msg}"),
            Self::SerializationError(msg) => write!(f, "Serialization error: {msg}"),
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
