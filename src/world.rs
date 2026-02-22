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

//! World state - deterministic simulation with O(1) incremental hashing
//!
//! v0.4 "Ultra Mode" optimizations:
//! 1. `HashMap` eliminated: Direct Vec indexing for entity lookup
//! 2. Entity is Copy: Fixed-size properties, no heap allocation
//! 3. SIMD: Vec3 operations use wide crate
//! 4. Raw integer mixing: No Hasher trait overhead
//! 5. Branchless property hash: Zero conditional branches
//! 6. Bounds check elimination: unsafe `get_unchecked`

use crate::arena::{Arena, Handle};
use crate::fixed_point::{Vec3Fixed, Vec3Simd};
use crate::{Event, EventKind, Result};
use serde::{Deserialize, Serialize};

/// `WyHash` mixing constant (proven avalanche properties)
const WYHASH_K: u64 = 0x517c_c1b7_2722_0a95;

/// World state hash (64-bit XOR rolling hash)
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Default,
    Serialize,
    Deserialize,
    bitcode::Encode,
    bitcode::Decode,
)]
pub struct WorldHash(pub u64);

impl WorldHash {
    #[inline(always)]
    #[must_use]
    pub const fn zero() -> Self {
        Self(0)
    }

    #[inline(always)]
    #[must_use]
    pub const fn xor(self, other: u64) -> Self {
        Self(self.0 ^ other)
    }
}

// ============================================================================
// Entity Properties - Fixed size, no heap allocation
// ============================================================================

/// Maximum number of properties per entity (compile-time constant)
pub const MAX_PROPS: usize = 16;

/// Fixed-size property storage (no `HashMap`, no allocation)
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct EntityProps {
    /// Property values indexed by property ID (0-15)
    pub values: [i32; MAX_PROPS],
    /// Bitmask of active properties (bit N = property N is set)
    pub active: u16,
}

impl EntityProps {
    pub const EMPTY: Self = Self {
        values: [0; MAX_PROPS],
        active: 0,
    };

    /// Set a property value
    #[inline(always)]
    pub fn set(&mut self, prop: u16, value: i32) {
        let idx = (prop as usize) & (MAX_PROPS - 1); // Mask to valid range
        self.values[idx] = value;
        self.active |= 1 << idx;
    }

    /// Get a property value (None if not set)
    #[inline(always)]
    #[must_use]
    pub fn get(&self, prop: u16) -> Option<i32> {
        let idx = (prop as usize) & (MAX_PROPS - 1);
        if self.active & (1 << idx) != 0 {
            Some(self.values[idx])
        } else {
            None
        }
    }

    /// Hash all active properties (truly branchless)
    /// Uses arithmetic masking instead of conditional branches
    #[inline(always)]
    #[must_use]
    pub fn hash_bits(&self) -> u64 {
        let mut h = self.active as u64;

        // Branchless: multiply by 0 or 1 based on active bit
        // This eliminates branch prediction misses
        for i in 0..MAX_PROPS {
            let is_active = ((self.active >> i) & 1) as u64;
            let val = (self.values[i] as u64).wrapping_mul(is_active);
            h ^= val.rotate_left((i * 4) as u32);
        }
        h
    }
}

// ============================================================================
// Entity - Now Copy trait (no heap allocation)
// ============================================================================

/// Entity in the world (fixed-point, deterministic, Copy)
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Entity {
    pub id: u32,
    pub kind: u16,
    pub position: Vec3Fixed,
    pub properties: EntityProps, // Fixed-size, no HashMap!
}

impl Entity {
    /// Compute hash of this entity (raw integer mixing, no Hasher overhead)
    ///
    /// Uses WyHash-style mixing: pure register operations, zero memory access,
    /// no state machine, single-pass avalanche.
    #[inline(always)]
    #[must_use]
    pub fn compute_hash(&self) -> u64 {
        // Start with entity ID
        let mut h = self.id as u64;

        // Mix in kind with rotation (avoid clustering)
        h ^= (self.kind as u64).rotate_left(32);

        // Mix in position coordinates with different rotations
        // Each rotation offset spreads bits to avoid cancellation
        h ^= (self.position.x.0 as u64).rotate_left(5);
        h ^= (self.position.y.0 as u64).rotate_left(14);
        h ^= (self.position.z.0 as u64).rotate_left(23);

        // Mix in properties (already branchless)
        h ^= self.properties.hash_bits();

        // Final avalanche: WyHash-style strong mixing
        // This ensures all input bits affect all output bits
        h = (h ^ (h >> 30)).wrapping_mul(WYHASH_K);
        h ^ (h >> 27)
    }
}

// ============================================================================
// World State - Vec-based direct indexing (no HashMap)
// ============================================================================

/// World state container with O(1) entity lookup
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorldState {
    /// All entities in arena (cache-friendly)
    pub entities: Arena<Entity>,
    /// Direct ID -> Handle mapping (Vec, not `HashMap`!)
    /// Index = `entity_id`, Value = Handle (or `Handle::INVALID`)
    pub id_map: Vec<Handle>,
    /// Current simulation frame
    pub frame: u64,
    /// Random seed (for deterministic procedural generation)
    pub seed: u64,
}

impl WorldState {
    /// Ensure `id_map` can hold the given `entity_id`
    #[inline]
    fn ensure_capacity(&mut self, entity_id: u32) {
        let required = entity_id as usize + 1;
        if self.id_map.len() < required {
            self.id_map.resize(required, Handle::INVALID);
        }
    }

    /// Get handle for `entity_id` (O(1) array access)
    #[inline(always)]
    #[must_use]
    pub fn get_handle(&self, entity_id: u32) -> Option<Handle> {
        self.id_map
            .get(entity_id as usize)
            .copied()
            .filter(super::arena::Handle::is_valid)
    }

    /// Set handle for `entity_id`
    #[inline(always)]
    pub fn set_handle(&mut self, entity_id: u32, handle: Handle) {
        self.ensure_capacity(entity_id);
        self.id_map[entity_id as usize] = handle;
    }

    /// Remove handle for `entity_id`
    #[inline(always)]
    pub fn remove_handle(&mut self, entity_id: u32) -> Option<Handle> {
        if let Some(slot) = self.id_map.get_mut(entity_id as usize) {
            let old = *slot;
            if old.is_valid() {
                *slot = Handle::INVALID;
                return Some(old);
            }
        }
        None
    }
}

// ============================================================================
// World - Deterministic simulation with O(1) everything
// ============================================================================

/// The deterministic world simulation with O(1) hashing
#[derive(Debug, Clone)]
pub struct World {
    state: WorldState,
    /// XOR rolling hash (updated incrementally)
    current_hash: WorldHash,
}

impl World {
    /// Create a new world with a seed
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            state: WorldState {
                seed,
                ..Default::default()
            },
            current_hash: WorldHash::zero(),
        }
    }

    /// Apply an event to the world (deterministic, O(1) hash update).
    /// Safe version with bounds checking.
    ///
    /// # Errors
    ///
    /// Currently infallible but returns `Result` for future error handling.
    #[inline]
    pub fn apply(&mut self, event: &Event) -> Result<()> {
        // SAFETY: We use the safe path by default
        // For maximum performance, use apply_unchecked after validating handles
        match &event.kind {
            EventKind::Motion { entity, delta } => {
                if let Some(handle) = self.state.get_handle(*entity) {
                    // SAFETY: handle was just validated by get_handle
                    unsafe { self.apply_motion_unchecked(handle, *delta) };
                }
            }

            EventKind::Spawn { entity, kind, pos } => {
                let new_entity = Entity {
                    id: *entity,
                    kind: *kind,
                    position: Vec3Fixed::from_i16_array(*pos),
                    properties: EntityProps::EMPTY,
                };

                // XOR in new entity hash
                let entity_hash = new_entity.compute_hash();
                self.current_hash = self.current_hash.xor(entity_hash);

                let handle = self.state.entities.insert(new_entity);
                self.state.set_handle(*entity, handle);
            }

            EventKind::Despawn { entity } => {
                if let Some(handle) = self.state.remove_handle(*entity) {
                    // SAFETY: handle was just validated by remove_handle
                    unsafe {
                        let e = self.state.entities.get_unchecked(handle);
                        let entity_hash = e.compute_hash();
                        self.current_hash = self.current_hash.xor(entity_hash);
                    }
                    self.state.entities.remove(handle);
                }
            }

            EventKind::Property {
                entity,
                prop,
                value,
            } => {
                if let Some(handle) = self.state.get_handle(*entity) {
                    // SAFETY: handle was just validated by get_handle
                    unsafe { self.apply_property_unchecked(handle, *prop, *value) };
                }
            }

            EventKind::Input { .. } | EventKind::Custom { .. } => {
                // Input/Custom events affect game logic, not world state directly
            }

            EventKind::Tick { frame } => {
                self.state.frame = *frame;
                self.tick_simulation();
            }
        }

        Ok(())
    }

    /// Apply motion without bounds checks (internal use)
    ///
    /// # Safety
    /// Caller must ensure handle is valid and points to an occupied entity
    #[inline(always)]
    unsafe fn apply_motion_unchecked(&mut self, handle: Handle, delta: [i16; 3]) {
        let e = self.state.entities.get_unchecked_mut(handle);

        // 1. XOR out old hash
        let old_hash = e.compute_hash();
        self.current_hash = self.current_hash.xor(old_hash);

        // 2. Update position using SIMD
        let delta_simd = Vec3Simd::from_i16_array(delta);
        delta_simd.add_to_vec3(&mut e.position);

        // 3. XOR in new hash
        let new_hash = e.compute_hash();
        self.current_hash = self.current_hash.xor(new_hash);
    }

    /// Apply property change without bounds checks (internal use)
    ///
    /// # Safety
    /// Caller must ensure handle is valid and points to an occupied entity
    #[inline(always)]
    unsafe fn apply_property_unchecked(&mut self, handle: Handle, prop: u16, value: i32) {
        let e = self.state.entities.get_unchecked_mut(handle);

        // XOR out old hash
        let old_hash = e.compute_hash();
        self.current_hash = self.current_hash.xor(old_hash);

        // Update property (O(1), no HashMap)
        e.properties.set(prop, value);

        // XOR in new hash
        let new_hash = e.compute_hash();
        self.current_hash = self.current_hash.xor(new_hash);
    }

    /// Deterministic simulation tick
    #[inline]
    #[allow(clippy::unused_self)]
    fn tick_simulation(&mut self) {
        // Physics, AI, etc. would go here
        // All operations must use fixed-point for determinism
    }

    /// Get world hash in O(1)
    #[inline(always)]
    #[must_use]
    pub fn hash(&self) -> WorldHash {
        self.current_hash
    }

    /// Full hash recalculation (for verification, O(N))
    pub fn recalculate_hash(&mut self) -> WorldHash {
        let mut hash = 0u64;
        for (_, entity) in self.state.entities.iter() {
            hash ^= entity.compute_hash();
        }
        self.current_hash = WorldHash(hash);
        self.current_hash
    }

    /// Get current state
    #[inline(always)]
    #[must_use]
    pub fn state(&self) -> &WorldState {
        &self.state
    }

    /// Get entity count
    #[inline(always)]
    #[must_use]
    pub fn entity_count(&self) -> usize {
        self.state.entities.len()
    }

    /// Get current frame
    #[inline(always)]
    #[must_use]
    pub fn frame(&self) -> u64 {
        self.state.frame
    }

    /// Get entity by ID (O(1) lookup)
    #[inline(always)]
    #[must_use]
    pub fn get_entity(&self, id: u32) -> Option<&Entity> {
        self.state
            .get_handle(id)
            .and_then(|h| self.state.entities.get(h))
    }

    /// Get entity by ID (mutable, O(1) lookup)
    #[inline(always)]
    pub fn get_entity_mut(&mut self, id: u32) -> Option<&mut Entity> {
        self.state
            .get_handle(id)
            .and_then(|h| self.state.entities.get_mut(h))
    }

    /// Iterate all entities
    pub fn iter_entities(&self) -> impl Iterator<Item = &Entity> {
        self.state.entities.iter().map(|(_, e)| e)
    }

    // ========================================================================
    // Batch Processing (SoA Optimized)
    // ========================================================================

    /// Apply motion events in batch (SIMD-friendly, minimal branching)
    ///
    /// All motion events are processed in a tight loop with:
    /// - Hot instruction cache
    /// - Minimal branch misprediction
    /// - Consecutive memory access patterns
    #[inline]
    pub fn apply_motions_batch(&mut self, motions: &[crate::MotionData]) {
        for m in motions {
            // Read packed fields (unaligned access on packed struct)
            let entity_id = m.entity;
            let dx = m.delta_x;
            let dy = m.delta_y;
            let dz = m.delta_z;

            if let Some(handle) = self.state.get_handle(entity_id) {
                // SAFETY: handle was just validated
                unsafe {
                    let e = self.state.entities.get_unchecked_mut(handle);

                    // XOR out old hash
                    let old_hash = e.compute_hash();
                    self.current_hash = self.current_hash.xor(old_hash);

                    // Direct fixed-point addition (no SIMD conversion overhead)
                    e.position.x.0 = e.position.x.0.wrapping_add((dx as i32) << 6);
                    e.position.y.0 = e.position.y.0.wrapping_add((dy as i32) << 6);
                    e.position.z.0 = e.position.z.0.wrapping_add((dz as i32) << 6);

                    // XOR in new hash
                    let new_hash = e.compute_hash();
                    self.current_hash = self.current_hash.xor(new_hash);
                }
            }
        }
    }

    /// Apply spawn events in batch
    #[inline]
    pub fn apply_spawns_batch(&mut self, spawns: &[crate::SpawnData]) {
        for s in spawns {
            let new_entity = Entity {
                id: s.entity,
                kind: s.kind,
                position: Vec3Fixed::from_i16_array(s.pos),
                properties: EntityProps::EMPTY,
            };

            let entity_hash = new_entity.compute_hash();
            self.current_hash = self.current_hash.xor(entity_hash);

            let handle = self.state.entities.insert(new_entity);
            self.state.set_handle(s.entity, handle);
        }
    }

    /// Apply event stream using batch processing (Run-Length optimization).
    ///
    /// Groups consecutive events of the same type and processes them together,
    /// maximizing instruction cache utilization and minimizing branch misprediction.
    ///
    /// # Errors
    ///
    /// Returns an error if any individual event fails to apply.
    pub fn apply_stream_batched(&mut self, stream: &crate::EventStream) -> Result<()> {
        let meta = stream.metadata();
        if meta.is_empty() {
            return Ok(());
        }

        let mut head = 0;
        let len = meta.len();

        while head < len {
            let current_type = meta[head].event_type;

            // Find run-length: consecutive events of the same type
            let mut end = head + 1;
            while end < len && meta[end].event_type == current_type {
                end += 1;
            }

            // Batch process based on type
            match current_type {
                crate::EventType::Motion => {
                    let start_idx = meta[head].index as usize;
                    let end_idx = meta[end - 1].index as usize + 1;
                    self.apply_motions_batch(&stream.motions[start_idx..end_idx]);
                }
                crate::EventType::Spawn => {
                    let start_idx = meta[head].index as usize;
                    let end_idx = meta[end - 1].index as usize + 1;
                    self.apply_spawns_batch(&stream.spawns[start_idx..end_idx]);
                }
                crate::EventType::Despawn => {
                    // Despawn is lightweight, process individually
                    for m in &meta[head..end] {
                        let idx = m.index as usize;
                        let entity = stream.despawns[idx].entity;
                        if let Some(handle) = self.state.remove_handle(entity) {
                            unsafe {
                                let e = self.state.entities.get_unchecked(handle);
                                let entity_hash = e.compute_hash();
                                self.current_hash = self.current_hash.xor(entity_hash);
                            }
                            self.state.entities.remove(handle);
                        }
                    }
                }
                crate::EventType::Property => {
                    for m in &meta[head..end] {
                        let idx = m.index as usize;
                        let p = &stream.properties[idx];
                        if let Some(handle) = self.state.get_handle(p.entity) {
                            unsafe {
                                self.apply_property_unchecked(handle, p.prop, p.value);
                            }
                        }
                    }
                }
                _ => {
                    // Input, Tick, Custom - process via standard path
                    for i in head..end {
                        let event = &stream.all()[i];
                        self.apply(event)?;
                    }
                }
            }

            head = end;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_props() {
        let mut props = EntityProps::EMPTY;
        assert_eq!(props.get(0), None);

        props.set(0, 100);
        assert_eq!(props.get(0), Some(100));

        props.set(5, 500);
        assert_eq!(props.get(5), Some(500));
        assert_eq!(props.get(0), Some(100));
    }

    #[test]
    fn test_incremental_hash() {
        let mut world = World::new(42);

        world
            .apply(&Event::new(EventKind::Spawn {
                entity: 1,
                kind: 0,
                pos: [0, 0, 0],
            }))
            .unwrap();

        let hash_after_spawn = world.hash();

        world
            .apply(&Event::new(EventKind::Motion {
                entity: 1,
                delta: [1000, 0, 0],
            }))
            .unwrap();

        let hash_after_move = world.hash();
        assert_ne!(hash_after_spawn, hash_after_move);

        // Verify incremental == full recalc
        let recalc = world.recalculate_hash();
        assert_eq!(world.hash(), recalc);
    }

    #[test]
    fn test_deterministic_world() {
        let mut world_a = World::new(42);
        let mut world_b = World::new(42);

        let events = vec![
            Event::new(EventKind::Spawn {
                entity: 1,
                kind: 0,
                pos: [0, 0, 0],
            }),
            Event::new(EventKind::Motion {
                entity: 1,
                delta: [1000, 2000, 3000],
            }),
            Event::new(EventKind::Property {
                entity: 1,
                prop: 0,
                value: 100,
            }),
        ];

        for e in &events {
            world_a.apply(e).unwrap();
            world_b.apply(e).unwrap();
        }

        assert_eq!(world_a.hash(), world_b.hash());

        let e_a = world_a.get_entity(1).unwrap();
        let e_b = world_b.get_entity(1).unwrap();
        assert_eq!(e_a.position, e_b.position);
        assert_eq!(e_a.properties.get(0), e_b.properties.get(0));
    }

    #[test]
    fn test_xor_rollback() {
        let mut world = World::new(0);

        world
            .apply(&Event::new(EventKind::Spawn {
                entity: 1,
                kind: 0,
                pos: [0, 0, 0],
            }))
            .unwrap();

        world
            .apply(&Event::new(EventKind::Despawn { entity: 1 }))
            .unwrap();

        assert_eq!(world.hash(), WorldHash::zero());
    }

    #[test]
    fn test_direct_indexing() {
        let mut world = World::new(0);

        // Spawn entities with non-sequential IDs
        for id in [10, 100, 1000] {
            world
                .apply(&Event::new(EventKind::Spawn {
                    entity: id,
                    kind: 0,
                    pos: [id as i16, 0, 0],
                }))
                .unwrap();
        }

        // All should be retrievable
        assert!(world.get_entity(10).is_some());
        assert!(world.get_entity(100).is_some());
        assert!(world.get_entity(1000).is_some());
        assert!(world.get_entity(999).is_none());
    }
}
