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

//! Event system - the core of ALICE-Sync
//!
//! v0.4 "Ultra Mode": `SoA` (Structure of Arrays) event storage
//! - Events grouped by type for cache-friendly batch processing
//! - Eliminates enum tag overhead in hot loops
//! - SIMD-friendly memory layout

use crate::fixed_point::Vec3Fixed;
use bitcode::{Decode, Encode};
use serde::{Deserialize, Serialize};

/// Unique event identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Encode, Decode)]
pub struct EventId(pub u64);

/// Sequence number for causal ordering
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Encode, Decode,
)]
pub struct SeqNum(pub u64);

/// Event kinds - all state changes expressible in few bytes
/// Uses i16 for network-efficient coordinates (6 bytes vs 12 bytes for f32x3)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Encode, Decode)]
pub enum EventKind {
    /// Entity motion: `entity_id` + delta vector
    /// Network: 4 + 6 = 10 bytes (was 16 with f32)
    Motion { entity: u32, delta: [i16; 3] },

    /// Entity spawn: `entity_id` + type + position
    /// Network: 4 + 2 + 6 = 12 bytes (was 22 with f32)
    Spawn {
        entity: u32,
        kind: u16,
        pos: [i16; 3],
    },

    /// Entity despawn: `entity_id`
    /// Network: 4 bytes
    Despawn { entity: u32 },

    /// Property change: `entity_id` + `property_id` + value
    /// Network: 4 + 2 + 4 = 10 bytes
    Property { entity: u32, prop: u16, value: i32 },

    /// Input event: `player_id` + `input_code`
    /// Network: 2 + 4 = 6 bytes
    Input { player: u16, code: u32 },

    /// Time tick: deterministic frame advance
    /// Network: 8 bytes
    Tick { frame: u64 },

    /// Custom event with compact payload
    /// Network: 2 + 16 = 18 bytes
    Custom { type_id: u16, payload: [u8; 16] },
}

impl EventKind {
    /// Convert i16 delta to `Vec3Fixed`
    #[inline]
    #[must_use]
    pub fn delta_to_fixed(delta: [i16; 3]) -> Vec3Fixed {
        Vec3Fixed::from_i16_array(delta)
    }

    /// Convert `Vec3Fixed` to i16 delta
    #[inline]
    #[must_use]
    pub fn fixed_to_delta(v: Vec3Fixed) -> [i16; 3] {
        v.to_i16_array()
    }

    /// Create motion event from `Vec3Fixed`
    #[inline]
    #[must_use]
    pub fn motion(entity: u32, delta: Vec3Fixed) -> Self {
        Self::Motion {
            entity,
            delta: delta.to_i16_array(),
        }
    }

    /// Create spawn event from `Vec3Fixed`
    #[inline]
    #[must_use]
    pub fn spawn(entity: u32, kind: u16, pos: Vec3Fixed) -> Self {
        Self::Spawn {
            entity,
            kind,
            pos: pos.to_i16_array(),
        }
    }
}

/// A single event with metadata
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Encode, Decode)]
pub struct Event {
    /// Unique identifier
    pub id: EventId,
    /// Sequence number for ordering
    pub seq: SeqNum,
    /// Origin node
    pub origin: u64,
    /// Timestamp (logical clock)
    pub timestamp: u64,
    /// The actual event data
    pub kind: EventKind,
}

impl Event {
    /// Create a new event (ID and seq assigned later)
    #[must_use]
    pub fn new(kind: EventKind) -> Self {
        Self {
            id: EventId(0),
            seq: SeqNum(0),
            origin: 0,
            timestamp: 0,
            kind,
        }
    }

    /// Serialize to bytes using bincode (for compatibility)
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap_or_default()
    }

    /// Serialize to compact bytes using bitcode
    #[must_use]
    pub fn to_compact_bytes(&self) -> Vec<u8> {
        bitcode::encode(self)
    }

    /// Deserialize from bincode bytes
    #[must_use]
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        bincode::deserialize(bytes).ok()
    }

    /// Deserialize from bitcode bytes
    #[must_use]
    pub fn from_compact_bytes(bytes: &[u8]) -> Option<Self> {
        bitcode::decode(bytes).ok()
    }

    /// Get approximate size in bytes (compact format)
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        // Header: id(8) + seq(8) + origin(8) + timestamp(8) = 32
        // But with varint encoding, typically 8-16 bytes
        let header = 12; // Estimated compressed header
        let payload = match &self.kind {
            EventKind::Spawn { .. } => 12,
            EventKind::Despawn { .. } => 4,
            EventKind::Motion { .. } | EventKind::Property { .. } => 10,
            EventKind::Input { .. } => 6,
            EventKind::Tick { .. } => 8,
            EventKind::Custom { .. } => 18,
        };
        header + payload
    }
}

// ============================================================================
// SoA (Structure of Arrays) Event Data
// ============================================================================

/// Motion event data (cache-friendly, SIMD-ready)
/// Layout: `[entity:4][dx:2][dy:2][dz:2]` = 10 bytes + 2 padding = 12 bytes
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Encode, Decode)]
#[repr(C)]
pub struct MotionData {
    pub entity: u32,
    pub delta_x: i16,
    pub delta_y: i16,
    pub delta_z: i16,
}

/// Spawn event data (packed)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Encode, Decode)]
#[repr(C)]
pub struct SpawnData {
    pub entity: u32,
    pub kind: u16,
    pub pos: [i16; 3],
}

/// Despawn event data (minimal)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Encode, Decode)]
#[repr(C)]
pub struct DespawnData {
    pub entity: u32,
}

/// Property event data
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Encode, Decode)]
#[repr(C)]
pub struct PropertyData {
    pub entity: u32,
    pub prop: u16,
    pub value: i32,
}

/// Input event data
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Encode, Decode)]
#[repr(C)]
pub struct InputData {
    pub player: u16,
    pub code: u32,
}

/// Tick event data
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Encode, Decode)]
#[repr(C)]
pub struct TickData {
    pub frame: u64,
}

/// Custom event data
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Encode, Decode)]
#[repr(C)]
pub struct CustomData {
    pub type_id: u16,
    pub payload: [u8; 16],
}

/// Event type tag (for ordered replay)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Encode, Decode)]
#[repr(u8)]
pub enum EventType {
    Motion = 0,
    Spawn = 1,
    Despawn = 2,
    Property = 3,
    Input = 4,
    Tick = 5,
    Custom = 6,
}

/// Event metadata (sequence, origin, type)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Encode, Decode)]
#[repr(C)]
pub struct EventMeta {
    pub seq: u64,
    pub origin: u64,
    pub event_type: EventType,
    pub index: u32, // Index into the type-specific array
}

// ============================================================================
// SoA EventStream - Cache-friendly batch processing
// ============================================================================

/// Stream of events with `SoA` layout for maximum cache efficiency
///
/// Events are stored by type, enabling:
/// - SIMD-friendly batch processing
/// - Better instruction cache utilization
/// - Reduced branch misprediction
#[derive(Debug, Clone, Default)]
pub struct EventStream {
    // Type-specific arrays (hot data)
    pub motions: Vec<MotionData>,
    pub spawns: Vec<SpawnData>,
    pub despawns: Vec<DespawnData>,
    pub properties: Vec<PropertyData>,
    pub inputs: Vec<InputData>,
    pub ticks: Vec<TickData>,
    pub customs: Vec<CustomData>,

    // Metadata for ordered replay
    meta: Vec<EventMeta>,

    // Legacy: unified event list for API compatibility
    events: Vec<Event>,

    next_seq: u64,
    next_id: u64,
}

impl EventStream {
    /// Create a new event stream
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a new event, assigning ID and sequence
    pub fn push(&mut self, mut event: Event, origin: u64) -> EventId {
        event.id = EventId(self.next_id);
        event.seq = SeqNum(self.next_seq);
        event.origin = origin;
        event.timestamp = self.next_seq;

        // Store in SoA arrays
        let (event_type, index) = match &event.kind {
            EventKind::Motion { entity, delta } => {
                let idx = self.motions.len() as u32;
                self.motions.push(MotionData {
                    entity: *entity,
                    delta_x: delta[0],
                    delta_y: delta[1],
                    delta_z: delta[2],
                });
                (EventType::Motion, idx)
            }
            EventKind::Spawn { entity, kind, pos } => {
                let idx = self.spawns.len() as u32;
                self.spawns.push(SpawnData {
                    entity: *entity,
                    kind: *kind,
                    pos: *pos,
                });
                (EventType::Spawn, idx)
            }
            EventKind::Despawn { entity } => {
                let idx = self.despawns.len() as u32;
                self.despawns.push(DespawnData { entity: *entity });
                (EventType::Despawn, idx)
            }
            EventKind::Property {
                entity,
                prop,
                value,
            } => {
                let idx = self.properties.len() as u32;
                self.properties.push(PropertyData {
                    entity: *entity,
                    prop: *prop,
                    value: *value,
                });
                (EventType::Property, idx)
            }
            EventKind::Input { player, code } => {
                let idx = self.inputs.len() as u32;
                self.inputs.push(InputData {
                    player: *player,
                    code: *code,
                });
                (EventType::Input, idx)
            }
            EventKind::Tick { frame } => {
                let idx = self.ticks.len() as u32;
                self.ticks.push(TickData { frame: *frame });
                (EventType::Tick, idx)
            }
            EventKind::Custom { type_id, payload } => {
                let idx = self.customs.len() as u32;
                self.customs.push(CustomData {
                    type_id: *type_id,
                    payload: *payload,
                });
                (EventType::Custom, idx)
            }
        };

        // Store metadata for ordered replay
        self.meta.push(EventMeta {
            seq: self.next_seq,
            origin,
            event_type,
            index,
        });

        self.next_id += 1;
        self.next_seq += 1;

        let id = event.id;
        self.events.push(event);
        id
    }

    /// Get events since a sequence number
    #[must_use]
    pub fn since(&self, seq: SeqNum) -> &[Event] {
        let start = self.events.partition_point(|e| e.seq < seq);
        &self.events[start..]
    }

    /// Get all events
    #[must_use]
    pub fn all(&self) -> &[Event] {
        &self.events
    }

    /// Get all motion events (for batch SIMD processing)
    #[inline(always)]
    #[must_use]
    pub fn all_motions(&self) -> &[MotionData] {
        &self.motions
    }

    /// Get all spawn events
    #[inline(always)]
    #[must_use]
    pub fn all_spawns(&self) -> &[SpawnData] {
        &self.spawns
    }

    /// Get event metadata for ordered replay
    #[inline(always)]
    #[must_use]
    pub fn metadata(&self) -> &[EventMeta] {
        &self.meta
    }

    /// Current sequence number
    #[must_use]
    pub fn current_seq(&self) -> SeqNum {
        SeqNum(self.next_seq.saturating_sub(1))
    }

    /// Total events count
    #[must_use]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Check if empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Total bytes if serialized (compact)
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.events.iter().map(Event::size_bytes).sum()
    }

    /// Serialize all events to compact format
    #[must_use]
    pub fn to_compact_bytes(&self) -> Vec<u8> {
        bitcode::encode(&self.events)
    }

    /// Get `SoA` statistics
    #[must_use]
    pub fn soa_stats(&self) -> SoAStats {
        SoAStats {
            motions: self.motions.len(),
            spawns: self.spawns.len(),
            despawns: self.despawns.len(),
            properties: self.properties.len(),
            inputs: self.inputs.len(),
            ticks: self.ticks.len(),
            customs: self.customs.len(),
        }
    }
}

/// `SoA` storage statistics
#[derive(Debug, Clone, Copy)]
pub struct SoAStats {
    pub motions: usize,
    pub spawns: usize,
    pub despawns: usize,
    pub properties: usize,
    pub inputs: usize,
    pub ticks: usize,
    pub customs: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_compact_size() {
        let motion = Event::new(EventKind::Motion {
            entity: 1,
            delta: [100, 0, 0], // i16
        });

        let compact = motion.to_compact_bytes();
        let bincode_size = motion.to_bytes().len();

        println!(
            "Compact: {} bytes, Bincode: {} bytes",
            compact.len(),
            bincode_size
        );
        // Compact should be significantly smaller
        assert!(compact.len() <= bincode_size);
    }

    #[test]
    fn test_event_roundtrip() {
        let original = Event::new(EventKind::Spawn {
            entity: 42,
            kind: 1,
            pos: [1000, 2000, 3000],
        });

        let bytes = original.to_compact_bytes();
        let decoded = Event::from_compact_bytes(&bytes).unwrap();

        assert_eq!(original.kind, decoded.kind);
    }

    #[test]
    fn test_fixed_point_conversion() {
        let v = Vec3Fixed::from_f32(1.5, -2.0, 0.5);
        let delta = EventKind::fixed_to_delta(v);
        let back = EventKind::delta_to_fixed(delta);

        // Precision loss is acceptable for network
        assert!((v.x.to_f32() - back.x.to_f32()).abs() < 0.1);
    }
}
