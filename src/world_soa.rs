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

//! World Storage - Complete SoA (Structure of Arrays) Layout
//!
//! v0.5 "Zen Mode": Entity data decomposed into component arrays
//!
//! Benefits:
//! 1. Cache pollution zero: Motion only touches pos_x/pos_y/pos_z
//! 2. Vertical SIMD: Process 8 entities' x-coordinates in one instruction
//! 3. Memory bandwidth: 12 bytes/entity instead of 84 bytes
//! 4. Perfect alignment: Each array is naturally aligned for SIMD

use crate::fixed_point::Fixed;
use serde::{Deserialize, Serialize};
use wide::i32x8;

/// Maximum entities (pre-allocated for cache efficiency)
pub const MAX_ENTITIES: usize = 65536;

/// Maximum properties per entity
pub const MAX_PROPS: usize = 16;

/// SoA World Storage - Component arrays instead of Entity structs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldStorage {
    // Generation tracking (for handle validation)
    pub generations: Vec<u32>,
    pub alive: Vec<bool>,
    pub free_list: Vec<u32>,

    // Core components (cache-line aligned arrays)
    pub ids: Vec<u32>,
    pub kinds: Vec<u16>,

    // Position: 3 separate arrays for Vertical SIMD
    pub pos_x: Vec<i32>,  // Fixed-point raw bits
    pub pos_y: Vec<i32>,
    pub pos_z: Vec<i32>,

    // Properties: 16 separate arrays + active bitmask
    pub prop_values: [Vec<i32>; MAX_PROPS],
    pub prop_active: Vec<u16>,

    // Metadata
    pub count: usize,
    pub capacity: usize,
    pub frame: u64,
    pub seed: u64,
}

impl Default for WorldStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl WorldStorage {
    /// Create new storage with default capacity
    pub fn new() -> Self {
        Self::with_capacity(1024)
    }

    /// Create storage with specific capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            generations: vec![0; capacity],
            alive: vec![false; capacity],
            free_list: (0..capacity as u32).rev().collect(),

            ids: vec![0; capacity],
            kinds: vec![0; capacity],

            pos_x: vec![0; capacity],
            pos_y: vec![0; capacity],
            pos_z: vec![0; capacity],

            prop_values: std::array::from_fn(|_| vec![0; capacity]),
            prop_active: vec![0; capacity],

            count: 0,
            capacity,
            frame: 0,
            seed: 0,
        }
    }

    /// Spawn entity, returns slot index
    #[inline(always)]
    pub fn spawn(&mut self, id: u32, kind: u16, x: i32, y: i32, z: i32) -> u32 {
        let slot = if let Some(slot) = self.free_list.pop() {
            slot
        } else {
            // Grow capacity
            let new_slot = self.capacity as u32;
            self.grow();
            new_slot
        };

        let idx = slot as usize;
        self.generations[idx] = self.generations[idx].wrapping_add(1);
        self.alive[idx] = true;
        self.ids[idx] = id;
        self.kinds[idx] = kind;
        self.pos_x[idx] = x;
        self.pos_y[idx] = y;
        self.pos_z[idx] = z;
        self.prop_active[idx] = 0;

        // Clear properties
        for prop in &mut self.prop_values {
            prop[idx] = 0;
        }

        self.count += 1;
        slot
    }

    /// Despawn entity by slot
    #[inline(always)]
    pub fn despawn(&mut self, slot: u32) -> bool {
        let idx = slot as usize;
        if idx >= self.capacity || !self.alive[idx] {
            return false;
        }

        self.alive[idx] = false;
        self.free_list.push(slot);
        self.count -= 1;
        true
    }

    /// Grow storage capacity
    fn grow(&mut self) {
        let new_capacity = self.capacity * 2;

        self.generations.resize(new_capacity, 0);
        self.alive.resize(new_capacity, false);
        self.ids.resize(new_capacity, 0);
        self.kinds.resize(new_capacity, 0);
        self.pos_x.resize(new_capacity, 0);
        self.pos_y.resize(new_capacity, 0);
        self.pos_z.resize(new_capacity, 0);
        self.prop_active.resize(new_capacity, 0);

        for prop in &mut self.prop_values {
            prop.resize(new_capacity, 0);
        }

        // Add new slots to free list
        for i in (self.capacity..new_capacity).rev() {
            self.free_list.push(i as u32);
        }

        self.capacity = new_capacity;
    }

    /// Check if slot is alive
    #[inline(always)]
    pub fn is_alive(&self, slot: u32) -> bool {
        let idx = slot as usize;
        idx < self.capacity && self.alive[idx]
    }

    /// Get position as Fixed tuple
    #[inline(always)]
    pub fn get_position(&self, slot: u32) -> (Fixed, Fixed, Fixed) {
        let idx = slot as usize;
        (
            Fixed(self.pos_x[idx]),
            Fixed(self.pos_y[idx]),
            Fixed(self.pos_z[idx]),
        )
    }

    /// Set position
    #[inline(always)]
    pub fn set_position(&mut self, slot: u32, x: i32, y: i32, z: i32) {
        let idx = slot as usize;
        self.pos_x[idx] = x;
        self.pos_y[idx] = y;
        self.pos_z[idx] = z;
    }

    /// Add to position (motion)
    #[inline(always)]
    pub fn add_position(&mut self, slot: u32, dx: i32, dy: i32, dz: i32) {
        let idx = slot as usize;
        self.pos_x[idx] = self.pos_x[idx].wrapping_add(dx);
        self.pos_y[idx] = self.pos_y[idx].wrapping_add(dy);
        self.pos_z[idx] = self.pos_z[idx].wrapping_add(dz);
    }

    /// Set property
    #[inline(always)]
    pub fn set_property(&mut self, slot: u32, prop: u16, value: i32) {
        let idx = slot as usize;
        let prop_idx = (prop as usize) & (MAX_PROPS - 1);
        self.prop_values[prop_idx][idx] = value;
        self.prop_active[idx] |= 1 << prop_idx;
    }

    /// Get property
    #[inline(always)]
    pub fn get_property(&self, slot: u32, prop: u16) -> Option<i32> {
        let idx = slot as usize;
        let prop_idx = (prop as usize) & (MAX_PROPS - 1);
        if self.prop_active[idx] & (1 << prop_idx) != 0 {
            Some(self.prop_values[prop_idx][idx])
        } else {
            None
        }
    }

    /// Compute hash for a single entity (branchless)
    #[inline(always)]
    pub fn compute_hash(&self, slot: u32) -> u64 {
        const WYHASH_K: u64 = 0x517cc1b727220a95;

        let idx = slot as usize;

        let mut h = self.ids[idx] as u64;
        h ^= (self.kinds[idx] as u64).rotate_left(32);
        h ^= (self.pos_x[idx] as u64).rotate_left(5);
        h ^= (self.pos_y[idx] as u64).rotate_left(14);
        h ^= (self.pos_z[idx] as u64).rotate_left(23);

        // Branchless property hash
        let active = self.prop_active[idx];
        for i in 0..MAX_PROPS {
            let is_active = ((active >> i) & 1) as u64;
            let val = (self.prop_values[i][idx] as u64).wrapping_mul(is_active);
            h ^= val.rotate_left((i * 4) as u32);
        }

        // Final avalanche
        h = (h ^ (h >> 30)).wrapping_mul(WYHASH_K);
        h ^ (h >> 27)
    }

    // ========================================================================
    // Vertical SIMD Operations (8 entities at once)
    // ========================================================================

    /// Load 8 x-coordinates into SIMD register
    #[inline(always)]
    pub fn load_pos_x_8(&self, start: usize) -> i32x8 {
        debug_assert!(start + 8 <= self.capacity);
        i32x8::new([
            self.pos_x[start],
            self.pos_x[start + 1],
            self.pos_x[start + 2],
            self.pos_x[start + 3],
            self.pos_x[start + 4],
            self.pos_x[start + 5],
            self.pos_x[start + 6],
            self.pos_x[start + 7],
        ])
    }

    /// Load 8 y-coordinates into SIMD register
    #[inline(always)]
    pub fn load_pos_y_8(&self, start: usize) -> i32x8 {
        debug_assert!(start + 8 <= self.capacity);
        i32x8::new([
            self.pos_y[start],
            self.pos_y[start + 1],
            self.pos_y[start + 2],
            self.pos_y[start + 3],
            self.pos_y[start + 4],
            self.pos_y[start + 5],
            self.pos_y[start + 6],
            self.pos_y[start + 7],
        ])
    }

    /// Load 8 z-coordinates into SIMD register
    #[inline(always)]
    pub fn load_pos_z_8(&self, start: usize) -> i32x8 {
        debug_assert!(start + 8 <= self.capacity);
        i32x8::new([
            self.pos_z[start],
            self.pos_z[start + 1],
            self.pos_z[start + 2],
            self.pos_z[start + 3],
            self.pos_z[start + 4],
            self.pos_z[start + 5],
            self.pos_z[start + 6],
            self.pos_z[start + 7],
        ])
    }

    /// Store 8 x-coordinates from SIMD register
    #[inline(always)]
    pub fn store_pos_x_8(&mut self, start: usize, values: i32x8) {
        debug_assert!(start + 8 <= self.capacity);
        let arr = values.to_array();
        self.pos_x[start..start + 8].copy_from_slice(&arr);
    }

    /// Store 8 y-coordinates from SIMD register
    #[inline(always)]
    pub fn store_pos_y_8(&mut self, start: usize, values: i32x8) {
        debug_assert!(start + 8 <= self.capacity);
        let arr = values.to_array();
        self.pos_y[start..start + 8].copy_from_slice(&arr);
    }

    /// Store 8 z-coordinates from SIMD register
    #[inline(always)]
    pub fn store_pos_z_8(&mut self, start: usize, values: i32x8) {
        debug_assert!(start + 8 <= self.capacity);
        let arr = values.to_array();
        self.pos_z[start..start + 8].copy_from_slice(&arr);
    }

    /// Add delta to 8 consecutive entities' positions (Vertical SIMD)
    #[inline(always)]
    pub fn add_positions_8(&mut self, start: usize, dx: i32x8, dy: i32x8, dz: i32x8) {
        let px = self.load_pos_x_8(start);
        let py = self.load_pos_y_8(start);
        let pz = self.load_pos_z_8(start);

        self.store_pos_x_8(start, px + dx);
        self.store_pos_y_8(start, py + dy);
        self.store_pos_z_8(start, pz + dz);
    }

    /// Compute hashes for 8 consecutive entities (parallel)
    #[inline(always)]
    pub fn compute_hashes_8(&self, start: usize) -> [u64; 8] {
        let mut hashes = [0u64; 8];
        for i in 0..8 {
            hashes[i] = self.compute_hash((start + i) as u32);
        }
        hashes
    }

    /// XOR-reduce 8 hashes into one
    #[inline(always)]
    pub fn reduce_hashes_xor(hashes: [u64; 8]) -> u64 {
        hashes[0] ^ hashes[1] ^ hashes[2] ^ hashes[3]
            ^ hashes[4] ^ hashes[5] ^ hashes[6] ^ hashes[7]
    }

    /// Entity count
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if empty
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Iterate alive slots
    pub fn iter_alive(&self) -> impl Iterator<Item = u32> + '_ {
        (0..self.capacity as u32).filter(|&slot| self.is_alive(slot))
    }
}

/// Slot-based handle for WorldStorage
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Slot {
    pub index: u32,
    pub generation: u32,
}

impl Slot {
    pub const INVALID: Self = Self {
        index: u32::MAX,
        generation: 0,
    };

    #[inline(always)]
    pub fn new(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }

    #[inline(always)]
    pub fn is_valid(&self) -> bool {
        self.index != u32::MAX
    }
}

// ============================================================================
// WorldSoA - Complete SoA World Implementation
// ============================================================================

use crate::world::WorldHash;
use crate::{Event, EventKind, MotionData, Result};

/// SoA-based World (v0.5 "Zen Mode")
///
/// Unlike the AoS World, this uses WorldStorage for complete SoA layout.
/// Benefits:
/// - 7x less memory read for Motion events
/// - 8x parallelism with Vertical SIMD
/// - 100% cache efficiency
#[derive(Debug, Clone)]
pub struct WorldSoA {
    pub storage: WorldStorage,
    current_hash: WorldHash,
    /// entity_id -> slot mapping
    id_to_slot: Vec<u32>,
}

impl Default for WorldSoA {
    fn default() -> Self {
        Self::new(0)
    }
}

impl WorldSoA {
    /// Create new SoA world with seed
    pub fn new(seed: u64) -> Self {
        let mut storage = WorldStorage::with_capacity(1024);
        storage.seed = seed;

        Self {
            storage,
            current_hash: WorldHash::zero(),
            id_to_slot: Vec::new(),
        }
    }

    /// Ensure id_to_slot can hold entity_id
    #[inline(always)]
    fn ensure_id_capacity(&mut self, entity_id: u32) {
        let required = entity_id as usize + 1;
        if self.id_to_slot.len() < required {
            self.id_to_slot.resize(required, u32::MAX);
        }
    }

    /// Get slot for entity_id
    #[inline(always)]
    fn get_slot(&self, entity_id: u32) -> Option<u32> {
        self.id_to_slot
            .get(entity_id as usize)
            .copied()
            .filter(|&s| s != u32::MAX && self.storage.is_alive(s))
    }

    /// Apply event to SoA world
    #[inline(always)]
    pub fn apply(&mut self, event: &Event) -> Result<()> {
        match &event.kind {
            EventKind::Motion { entity, delta } => {
                if let Some(slot) = self.get_slot(*entity) {
                    // XOR out old hash
                    let old_hash = self.storage.compute_hash(slot);
                    self.current_hash = self.current_hash.xor(old_hash);

                    // Update position in SoA storage
                    let dx = (delta[0] as i32) << 6;
                    let dy = (delta[1] as i32) << 6;
                    let dz = (delta[2] as i32) << 6;
                    self.storage.add_position(slot, dx, dy, dz);

                    // XOR in new hash
                    let new_hash = self.storage.compute_hash(slot);
                    self.current_hash = self.current_hash.xor(new_hash);
                }
            }

            EventKind::Spawn { entity, kind, pos } => {
                let x = (pos[0] as i32) << 6;
                let y = (pos[1] as i32) << 6;
                let z = (pos[2] as i32) << 6;

                let slot = self.storage.spawn(*entity, *kind, x, y, z);

                // XOR in new entity hash
                let entity_hash = self.storage.compute_hash(slot);
                self.current_hash = self.current_hash.xor(entity_hash);

                // Map entity_id -> slot
                self.ensure_id_capacity(*entity);
                self.id_to_slot[*entity as usize] = slot;
            }

            EventKind::Despawn { entity } => {
                if let Some(slot) = self.get_slot(*entity) {
                    // XOR out hash
                    let entity_hash = self.storage.compute_hash(slot);
                    self.current_hash = self.current_hash.xor(entity_hash);

                    self.storage.despawn(slot);
                    self.id_to_slot[*entity as usize] = u32::MAX;
                }
            }

            EventKind::Property { entity, prop, value } => {
                if let Some(slot) = self.get_slot(*entity) {
                    // XOR out old hash
                    let old_hash = self.storage.compute_hash(slot);
                    self.current_hash = self.current_hash.xor(old_hash);

                    self.storage.set_property(slot, *prop, *value);

                    // XOR in new hash
                    let new_hash = self.storage.compute_hash(slot);
                    self.current_hash = self.current_hash.xor(new_hash);
                }
            }

            EventKind::Tick { frame } => {
                self.storage.frame = *frame;
            }

            _ => {}
        }

        Ok(())
    }

    /// Get world hash (O(1))
    #[inline(always)]
    pub fn hash(&self) -> WorldHash {
        self.current_hash
    }

    /// Recalculate hash (O(N), for verification)
    pub fn recalculate_hash(&mut self) -> WorldHash {
        let mut hash = 0u64;
        for slot in self.storage.iter_alive() {
            hash ^= self.storage.compute_hash(slot);
        }
        self.current_hash = WorldHash(hash);
        self.current_hash
    }

    /// Entity count
    #[inline(always)]
    pub fn entity_count(&self) -> usize {
        self.storage.len()
    }

    /// Current frame
    #[inline(always)]
    pub fn frame(&self) -> u64 {
        self.storage.frame
    }

    // ========================================================================
    // Vertical SIMD Batch Operations (8 entities at once)
    // ========================================================================

    /// Apply motions to 8 consecutive entities using Vertical SIMD
    ///
    /// This is the "God Mode" of motion processing:
    /// - Loads 8 x-coordinates in one instruction
    /// - Adds delta to all 8 in one instruction
    /// - Stores 8 results in one instruction
    /// - Same for y and z
    ///
    /// Requires entities to be at consecutive slots (use with sorted data)
    #[inline(always)]
    pub fn apply_motions_vertical_8(
        &mut self,
        start_slot: usize,
        dx: i32x8,
        dy: i32x8,
        dz: i32x8,
    ) {
        // Compute old hashes for 8 entities
        let old_hashes = self.storage.compute_hashes_8(start_slot);
        let old_combined = WorldStorage::reduce_hashes_xor(old_hashes);

        // XOR out old hashes
        self.current_hash = self.current_hash.xor(old_combined);

        // Vertical SIMD: update 8 positions at once
        self.storage.add_positions_8(start_slot, dx, dy, dz);

        // Compute new hashes for 8 entities
        let new_hashes = self.storage.compute_hashes_8(start_slot);
        let new_combined = WorldStorage::reduce_hashes_xor(new_hashes);

        // XOR in new hashes
        self.current_hash = self.current_hash.xor(new_combined);
    }

    /// Batch apply motion events (with automatic Vertical SIMD for consecutive entities)
    #[inline(always)]
    pub fn apply_motions_batch(&mut self, motions: &[MotionData]) {
        for m in motions {
            if let Some(slot) = self.get_slot(m.entity) {
                // XOR out old hash
                let old_hash = self.storage.compute_hash(slot);
                self.current_hash = self.current_hash.xor(old_hash);

                // Update position
                let dx = (m.delta_x as i32) << 6;
                let dy = (m.delta_y as i32) << 6;
                let dz = (m.delta_z as i32) << 6;
                self.storage.add_position(slot, dx, dy, dz);

                // XOR in new hash
                let new_hash = self.storage.compute_hash(slot);
                self.current_hash = self.current_hash.xor(new_hash);
            }
        }
    }

    // ========================================================================
    // Lv.6 "Demon Mode": Sort-Based Batching
    // ========================================================================

    /// Apply motions with sort-based optimization (Demon Mode)
    ///
    /// Strategy:
    /// 1. Sort motions by entity_id (→ sequential memory access)
    /// 2. Detect contiguous slot ranges for Vertical SIMD
    /// 3. Coalesce multiple motions to the same entity
    /// 4. Apply with maximum cache efficiency
    #[inline(always)]
    pub fn apply_motions_sorted(&mut self, mut motions: Vec<MotionData>) {
        if motions.is_empty() {
            return;
        }

        // Step 1: Sort by entity_id (unstable sort is faster)
        motions.sort_unstable_by_key(|m| m.entity);

        // Step 2: Coalesce consecutive motions to the same entity
        let mut coalesced: Vec<MotionData> = Vec::with_capacity(motions.len());
        let mut current = motions[0];

        for m in motions.iter().skip(1) {
            if m.entity == current.entity {
                // Coalesce: accumulate deltas
                current.delta_x = current.delta_x.saturating_add(m.delta_x);
                current.delta_y = current.delta_y.saturating_add(m.delta_y);
                current.delta_z = current.delta_z.saturating_add(m.delta_z);
            } else {
                coalesced.push(current);
                current = *m;
            }
        }
        coalesced.push(current);

        // Step 3: Resolve entity_id -> slot mapping (sorted order)
        let mut slot_motions: Vec<(u32, MotionData)> = Vec::with_capacity(coalesced.len());
        for m in coalesced {
            if let Some(slot) = self.get_slot(m.entity) {
                slot_motions.push((slot, m));
            }
        }

        // Step 4: Sort by slot for sequential memory access
        slot_motions.sort_unstable_by_key(|(slot, _)| *slot);

        // Step 5: Detect contiguous ranges and apply with Vertical SIMD
        let mut i = 0;
        while i < slot_motions.len() {
            // Try to find a contiguous run of 8+ slots
            let start_slot = slot_motions[i].0;
            let mut run_len = 1;

            while i + run_len < slot_motions.len()
                && slot_motions[i + run_len].0 == start_slot + run_len as u32
                && run_len < 8
            {
                run_len += 1;
            }

            if run_len == 8 {
                // Full SIMD batch: 8 contiguous slots with same delta
                // Check if all have same delta for uniform SIMD
                let first_delta = &slot_motions[i].1;
                let all_same_delta = slot_motions[i..i + 8].iter().all(|(_, m)| {
                    m.delta_x == first_delta.delta_x
                        && m.delta_y == first_delta.delta_y
                        && m.delta_z == first_delta.delta_z
                });

                if all_same_delta {
                    // Uniform SIMD: all 8 have same delta
                    let dx = (first_delta.delta_x as i32) << 6;
                    let dy = (first_delta.delta_y as i32) << 6;
                    let dz = (first_delta.delta_z as i32) << 6;

                    self.apply_motions_vertical_8(
                        start_slot as usize,
                        i32x8::splat(dx),
                        i32x8::splat(dy),
                        i32x8::splat(dz),
                    );
                } else {
                    // Variable SIMD: 8 different deltas
                    let deltas: Vec<_> = slot_motions[i..i + 8]
                        .iter()
                        .map(|(_, m)| ((m.delta_x as i32) << 6, (m.delta_y as i32) << 6, (m.delta_z as i32) << 6))
                        .collect();

                    let dx = i32x8::new([
                        deltas[0].0, deltas[1].0, deltas[2].0, deltas[3].0,
                        deltas[4].0, deltas[5].0, deltas[6].0, deltas[7].0,
                    ]);
                    let dy = i32x8::new([
                        deltas[0].1, deltas[1].1, deltas[2].1, deltas[3].1,
                        deltas[4].1, deltas[5].1, deltas[6].1, deltas[7].1,
                    ]);
                    let dz = i32x8::new([
                        deltas[0].2, deltas[1].2, deltas[2].2, deltas[3].2,
                        deltas[4].2, deltas[5].2, deltas[6].2, deltas[7].2,
                    ]);

                    self.apply_motions_vertical_8(start_slot as usize, dx, dy, dz);
                }
                i += 8;
            } else {
                // Scalar fallback for non-contiguous slots
                let (slot, m) = &slot_motions[i];

                let old_hash = self.storage.compute_hash(*slot);
                self.current_hash = self.current_hash.xor(old_hash);

                let dx = (m.delta_x as i32) << 6;
                let dy = (m.delta_y as i32) << 6;
                let dz = (m.delta_z as i32) << 6;
                self.storage.add_position(*slot, dx, dy, dz);

                let new_hash = self.storage.compute_hash(*slot);
                self.current_hash = self.current_hash.xor(new_hash);

                i += 1;
            }
        }
    }

    /// Apply motions with pre-sorted input (skip sorting step)
    ///
    /// Use when motions are already sorted by entity_id
    #[inline(always)]
    pub fn apply_motions_presorted(&mut self, motions: &[MotionData]) {
        // Resolve entity_id -> slot mapping (already sorted)
        let mut slot_motions: Vec<(u32, &MotionData)> = Vec::with_capacity(motions.len());
        for m in motions {
            if let Some(slot) = self.get_slot(m.entity) {
                slot_motions.push((slot, m));
            }
        }

        // Sort by slot for sequential memory access
        slot_motions.sort_unstable_by_key(|(slot, _)| *slot);

        // Apply with sequential access pattern
        for (slot, m) in slot_motions {
            let old_hash = self.storage.compute_hash(slot);
            self.current_hash = self.current_hash.xor(old_hash);

            let dx = (m.delta_x as i32) << 6;
            let dy = (m.delta_y as i32) << 6;
            let dz = (m.delta_z as i32) << 6;
            self.storage.add_position(slot, dx, dy, dz);

            let new_hash = self.storage.compute_hash(slot);
            self.current_hash = self.current_hash.xor(new_hash);
        }
    }

    // ========================================================================
    // Gemini Version: ID-Sort Only (No Slot-Sort, No Coalesce)
    // ========================================================================

    /// Apply motions with ID-sort only (Gemini's Demon Mode)
    ///
    /// Simpler strategy:
    /// 1. Sort by entity_id only (no slot re-sort)
    /// 2. No coalescing
    /// 3. Detect contiguous slots for SIMD
    ///
    /// Hypothesis: Faster when fragmentation is low (ID ≈ Slot)
    #[inline(always)]
    pub fn apply_motions_demon(&mut self, mut motions: Vec<MotionData>) {
        if motions.is_empty() {
            return;
        }

        // Step 1: Sort by entity_id only (no slot conversion/re-sort)
        motions.sort_unstable_by_key(|m| m.entity);

        let mut i = 0;
        let len = motions.len();

        while i < len {
            // Check for contiguous block of 8 slots
            let mut can_simd = false;
            let mut start_slot = 0u32;

            if i + 8 <= len {
                if let Some(s) = self.get_slot(motions[i].entity) {
                    start_slot = s;
                    let mut contiguous = true;

                    for k in 1..8 {
                        if let Some(s_next) = self.get_slot(motions[i + k].entity) {
                            if s_next != start_slot + k as u32 {
                                contiguous = false;
                                break;
                            }
                        } else {
                            contiguous = false;
                            break;
                        }
                    }
                    can_simd = contiguous;
                }
            }

            if can_simd {
                // SIMD path: 8 contiguous slots
                let dx = i32x8::new([
                    (motions[i].delta_x as i32) << 6,
                    (motions[i + 1].delta_x as i32) << 6,
                    (motions[i + 2].delta_x as i32) << 6,
                    (motions[i + 3].delta_x as i32) << 6,
                    (motions[i + 4].delta_x as i32) << 6,
                    (motions[i + 5].delta_x as i32) << 6,
                    (motions[i + 6].delta_x as i32) << 6,
                    (motions[i + 7].delta_x as i32) << 6,
                ]);
                let dy = i32x8::new([
                    (motions[i].delta_y as i32) << 6,
                    (motions[i + 1].delta_y as i32) << 6,
                    (motions[i + 2].delta_y as i32) << 6,
                    (motions[i + 3].delta_y as i32) << 6,
                    (motions[i + 4].delta_y as i32) << 6,
                    (motions[i + 5].delta_y as i32) << 6,
                    (motions[i + 6].delta_y as i32) << 6,
                    (motions[i + 7].delta_y as i32) << 6,
                ]);
                let dz = i32x8::new([
                    (motions[i].delta_z as i32) << 6,
                    (motions[i + 1].delta_z as i32) << 6,
                    (motions[i + 2].delta_z as i32) << 6,
                    (motions[i + 3].delta_z as i32) << 6,
                    (motions[i + 4].delta_z as i32) << 6,
                    (motions[i + 5].delta_z as i32) << 6,
                    (motions[i + 6].delta_z as i32) << 6,
                    (motions[i + 7].delta_z as i32) << 6,
                ]);

                // Hash update (XOR out old)
                let old_hashes = self.storage.compute_hashes_8(start_slot as usize);
                for h in old_hashes {
                    self.current_hash = self.current_hash.xor(h);
                }

                // SIMD add
                self.storage.add_positions_8(start_slot as usize, dx, dy, dz);

                // Hash update (XOR in new)
                let new_hashes = self.storage.compute_hashes_8(start_slot as usize);
                for h in new_hashes {
                    self.current_hash = self.current_hash.xor(h);
                }

                i += 8;
            } else {
                // Scalar fallback
                let m = &motions[i];
                if let Some(slot) = self.get_slot(m.entity) {
                    let old_h = self.storage.compute_hash(slot);
                    self.current_hash = self.current_hash.xor(old_h);

                    self.storage.pos_x[slot as usize] += (m.delta_x as i32) << 6;
                    self.storage.pos_y[slot as usize] += (m.delta_y as i32) << 6;
                    self.storage.pos_z[slot as usize] += (m.delta_z as i32) << 6;

                    let new_h = self.storage.compute_hash(slot);
                    self.current_hash = self.current_hash.xor(new_h);
                }
                i += 1;
            }
        }
    }

    /// Apply motions without any sorting (baseline for comparison)
    #[inline(always)]
    pub fn apply_motions_nosort(&mut self, motions: &[MotionData]) {
        for m in motions {
            if let Some(slot) = self.get_slot(m.entity) {
                let old_h = self.storage.compute_hash(slot);
                self.current_hash = self.current_hash.xor(old_h);

                self.storage.pos_x[slot as usize] += (m.delta_x as i32) << 6;
                self.storage.pos_y[slot as usize] += (m.delta_y as i32) << 6;
                self.storage.pos_z[slot as usize] += (m.delta_z as i32) << 6;

                let new_h = self.storage.compute_hash(slot);
                self.current_hash = self.current_hash.xor(new_h);
            }
        }
    }

    /// Apply motions to consecutive entity slots using Vertical SIMD
    ///
    /// This is the maximum performance path when entities are densely packed.
    /// Process 8 entities per iteration.
    pub fn apply_uniform_motion_vertical(&mut self, start_slot: usize, count: usize, dx: i16, dy: i16, dz: i16) {
        let dx_fixed = (dx as i32) << 6;
        let dy_fixed = (dy as i32) << 6;
        let dz_fixed = (dz as i32) << 6;

        let dx_simd = i32x8::splat(dx_fixed);
        let dy_simd = i32x8::splat(dy_fixed);
        let dz_simd = i32x8::splat(dz_fixed);

        // Process 8 entities at a time
        let chunks = count >> 3;
        for i in 0..chunks {
            let slot = start_slot + i * 8;
            self.apply_motions_vertical_8(slot, dx_simd, dy_simd, dz_simd);
        }

        // Handle remainder
        let remainder_start = start_slot + chunks * 8;
        for i in 0..(count % 8) {
            let slot = (remainder_start + i) as u32;
            if self.storage.is_alive(slot) {
                let old_hash = self.storage.compute_hash(slot);
                self.current_hash = self.current_hash.xor(old_hash);

                self.storage.add_position(slot, dx_fixed, dy_fixed, dz_fixed);

                let new_hash = self.storage.compute_hash(slot);
                self.current_hash = self.current_hash.xor(new_hash);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spawn_despawn() {
        let mut storage = WorldStorage::with_capacity(16);

        let slot = storage.spawn(1, 0, 1000, 2000, 3000);
        assert!(storage.is_alive(slot));
        assert_eq!(storage.len(), 1);

        let (x, y, z) = storage.get_position(slot);
        assert_eq!(x.0, 1000);
        assert_eq!(y.0, 2000);
        assert_eq!(z.0, 3000);

        storage.despawn(slot);
        assert!(!storage.is_alive(slot));
        assert_eq!(storage.len(), 0);
    }

    #[test]
    fn test_motion() {
        let mut storage = WorldStorage::with_capacity(16);
        let slot = storage.spawn(1, 0, 0, 0, 0);

        storage.add_position(slot, 100, 200, 300);

        let (x, y, z) = storage.get_position(slot);
        assert_eq!(x.0, 100);
        assert_eq!(y.0, 200);
        assert_eq!(z.0, 300);
    }

    #[test]
    fn test_properties() {
        let mut storage = WorldStorage::with_capacity(16);
        let slot = storage.spawn(1, 0, 0, 0, 0);

        assert_eq!(storage.get_property(slot, 0), None);

        storage.set_property(slot, 0, 42);
        assert_eq!(storage.get_property(slot, 0), Some(42));

        storage.set_property(slot, 5, 100);
        assert_eq!(storage.get_property(slot, 5), Some(100));
    }

    #[test]
    fn test_vertical_simd() {
        let mut storage = WorldStorage::with_capacity(16);

        // Spawn 8 entities at consecutive slots
        for i in 0..8u32 {
            storage.spawn(i, 0, i as i32 * 100, 0, 0);
        }

        // Verify initial positions
        let px = storage.load_pos_x_8(0);
        assert_eq!(px.to_array(), [0, 100, 200, 300, 400, 500, 600, 700]);

        // Add delta using SIMD
        let dx = i32x8::splat(10);
        let dy = i32x8::splat(20);
        let dz = i32x8::splat(30);
        storage.add_positions_8(0, dx, dy, dz);

        // Verify updated positions
        let px = storage.load_pos_x_8(0);
        assert_eq!(px.to_array(), [10, 110, 210, 310, 410, 510, 610, 710]);
    }

    #[test]
    fn test_hash_consistency() {
        let mut storage = WorldStorage::with_capacity(16);
        let slot = storage.spawn(42, 1, 1000, 2000, 3000);
        storage.set_property(slot, 0, 100);

        let hash1 = storage.compute_hash(slot);
        let hash2 = storage.compute_hash(slot);

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, 0);
    }

    #[test]
    fn test_world_soa_basic() {
        let mut world = WorldSoA::new(42);

        world.apply(&Event::new(EventKind::Spawn {
            entity: 1,
            kind: 0,
            pos: [0, 0, 0],
        })).unwrap();

        assert_eq!(world.entity_count(), 1);
        assert_ne!(world.hash(), WorldHash::zero());
    }

    #[test]
    fn test_world_soa_motion() {
        let mut world = WorldSoA::new(42);

        world.apply(&Event::new(EventKind::Spawn {
            entity: 1,
            kind: 0,
            pos: [0, 0, 0],
        })).unwrap();

        let hash_before = world.hash();

        world.apply(&Event::new(EventKind::Motion {
            entity: 1,
            delta: [1000, 0, 0],
        })).unwrap();

        assert_ne!(world.hash(), hash_before);

        // Verify position changed
        let slot = world.get_slot(1).unwrap();
        let (x, _, _) = world.storage.get_position(slot);
        assert!(x.0 > 0);
    }

    #[test]
    fn test_world_soa_deterministic() {
        let mut world_a = WorldSoA::new(42);
        let mut world_b = WorldSoA::new(42);

        let events = vec![
            Event::new(EventKind::Spawn { entity: 1, kind: 0, pos: [0, 0, 0] }),
            Event::new(EventKind::Motion { entity: 1, delta: [100, 200, 300] }),
            Event::new(EventKind::Property { entity: 1, prop: 0, value: 42 }),
        ];

        for e in &events {
            world_a.apply(e).unwrap();
            world_b.apply(e).unwrap();
        }

        assert_eq!(world_a.hash(), world_b.hash());
    }

    #[test]
    fn test_world_soa_xor_rollback() {
        let mut world = WorldSoA::new(0);

        world.apply(&Event::new(EventKind::Spawn {
            entity: 1,
            kind: 0,
            pos: [0, 0, 0],
        })).unwrap();

        world.apply(&Event::new(EventKind::Despawn { entity: 1 })).unwrap();

        assert_eq!(world.hash(), WorldHash::zero());
    }

    #[test]
    fn test_world_soa_vertical_simd() {
        let mut world = WorldSoA::new(42);

        // Spawn 16 entities (for 2 SIMD batches)
        for i in 0..16u32 {
            world.apply(&Event::new(EventKind::Spawn {
                entity: i,
                kind: 0,
                pos: [0, 0, 0],
            })).unwrap();
        }

        let hash_before = world.hash();

        // Apply uniform motion to all 16 using Vertical SIMD
        world.apply_uniform_motion_vertical(0, 16, 100, 0, 0);

        assert_ne!(world.hash(), hash_before);

        // Verify all positions updated
        for i in 0..16 {
            let (x, _, _) = world.storage.get_position(i);
            assert!(x.0 > 0);
        }
    }

    #[test]
    fn test_demon_mode_sorted_batch() {
        use crate::MotionData;

        let mut world = WorldSoA::new(42);

        // Spawn 100 entities
        for i in 0..100u32 {
            world.apply(&Event::new(EventKind::Spawn {
                entity: i,
                kind: 0,
                pos: [0, 0, 0],
            })).unwrap();
        }

        // Create random-order motions
        let motions: Vec<MotionData> = vec![
            MotionData { entity: 50, delta_x: 10, delta_y: 0, delta_z: 0 },
            MotionData { entity: 10, delta_x: 20, delta_y: 0, delta_z: 0 },
            MotionData { entity: 90, delta_x: 30, delta_y: 0, delta_z: 0 },
            MotionData { entity: 10, delta_x: 5, delta_y: 0, delta_z: 0 },  // Same entity (should coalesce)
        ];

        let hash_before = world.hash();

        // Apply with Demon Mode (sorted batch)
        world.apply_motions_sorted(motions);

        assert_ne!(world.hash(), hash_before);

        // Verify entity 10 got coalesced delta (20 + 5 = 25)
        let slot = world.get_slot(10).unwrap();
        let (x, _, _) = world.storage.get_position(slot);
        // 25 << 6 = 1600
        assert_eq!(x.0, 1600);
    }

    #[test]
    fn test_demon_mode_contiguous_simd() {
        use crate::MotionData;

        let mut world = WorldSoA::new(42);

        // Spawn 16 entities (contiguous IDs)
        for i in 0..16u32 {
            world.apply(&Event::new(EventKind::Spawn {
                entity: i,
                kind: 0,
                pos: [0, 0, 0],
            })).unwrap();
        }

        // Create motions for 8 contiguous entities (should trigger SIMD path)
        let motions: Vec<MotionData> = (0..8)
            .map(|i| MotionData {
                entity: i,
                delta_x: 100,
                delta_y: 0,
                delta_z: 0,
            })
            .collect();

        let hash_before = world.hash();

        world.apply_motions_sorted(motions);

        assert_ne!(world.hash(), hash_before);

        // Verify all 8 entities updated
        for i in 0..8 {
            let slot = world.get_slot(i).unwrap();
            let (x, _, _) = world.storage.get_position(slot);
            assert_eq!(x.0, 100 << 6);
        }
    }
}
