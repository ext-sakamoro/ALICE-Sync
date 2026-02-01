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

//! P2P Protocol - minimal bandwidth communication
//!
//! Uses bitcode for compact serialization

use bitcode::{Decode, Encode};
use crate::{Event, NodeId, WorldHash};
use serde::{Deserialize, Serialize};

/// Protocol message types
#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub enum Message {
    /// Initial handshake
    Hello {
        node_id: NodeId,
        seq: u64,
        world_hash: WorldHash,
    },

    /// Request events since sequence number
    RequestSync { since_seq: u64 },

    /// Batch of events (compact encoded)
    Events { events: Vec<Event> },

    /// Periodic hash check
    HashCheck { seq: u64, world_hash: WorldHash },

    /// Hash mismatch
    Diverged {
        my_hash: WorldHash,
        your_hash: WorldHash,
    },

    /// Acknowledgment
    Ack { seq: u64 },

    /// Goodbye
    Bye,
}

impl Message {
    /// Serialize to bytes (bincode, compatible)
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap_or_default()
    }

    /// Serialize to compact bytes (bitcode)
    pub fn to_compact_bytes(&self) -> Vec<u8> {
        bitcode::encode(self)
    }

    /// Deserialize from bincode bytes
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        bincode::deserialize(bytes).ok()
    }

    /// Deserialize from bitcode bytes
    pub fn from_compact_bytes(bytes: &[u8]) -> Option<Self> {
        bitcode::decode(bytes).ok()
    }

    /// Get message size in bytes
    pub fn size_bytes(&self) -> usize {
        self.to_compact_bytes().len()
    }
}

/// Protocol handler
#[derive(Debug)]
pub struct Protocol {
    node_id: NodeId,
    hash_check_interval: u64,
    events_since_check: u64,
}

impl Protocol {
    /// Create a new protocol handler
    pub fn new(node_id: NodeId) -> Self {
        Self {
            node_id,
            hash_check_interval: 100,
            events_since_check: 0,
        }
    }

    /// Create hello message
    pub fn hello(&self, seq: u64, world_hash: WorldHash) -> Message {
        Message::Hello {
            node_id: self.node_id,
            seq,
            world_hash,
        }
    }

    /// Create sync request
    pub fn request_sync(&self, since_seq: u64) -> Message {
        Message::RequestSync { since_seq }
    }

    /// Create events message
    pub fn events(&self, events: Vec<Event>) -> Message {
        Message::Events { events }
    }

    /// Check if hash check is due
    pub fn should_hash_check(&self) -> bool {
        self.events_since_check >= self.hash_check_interval
    }

    /// Create hash check message
    pub fn hash_check(&mut self, seq: u64, world_hash: WorldHash) -> Message {
        self.events_since_check = 0;
        Message::HashCheck { seq, world_hash }
    }

    /// Record event processed
    pub fn event_processed(&mut self) {
        self.events_since_check += 1;
    }

    /// Create ack message
    pub fn ack(&self, seq: u64) -> Message {
        Message::Ack { seq }
    }
}

/// Bandwidth statistics
#[derive(Debug, Clone, Default)]
pub struct BandwidthStats {
    pub messages_sent: u64,
    pub messages_received: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub events_synced: u64,
}

impl BandwidthStats {
    /// Average bytes per event
    pub fn bytes_per_event(&self) -> f64 {
        if self.events_synced == 0 {
            0.0
        } else {
            self.bytes_sent as f64 / self.events_synced as f64
        }
    }

    /// Compression ratio vs raw data
    pub fn compression_ratio(&self, raw_world_size: u64) -> f64 {
        if self.bytes_sent == 0 {
            0.0
        } else {
            raw_world_size as f64 / self.bytes_sent as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EventKind;

    #[test]
    fn test_message_compact_size() {
        let hello = Message::Hello {
            node_id: NodeId(1),
            seq: 1000,
            world_hash: WorldHash::zero(),
        };

        let compact = hello.to_compact_bytes();
        let bincode_size = hello.to_bytes().len();

        println!("Hello - Compact: {} bytes, Bincode: {} bytes", compact.len(), bincode_size);
        assert!(compact.len() <= bincode_size);
    }

    #[test]
    fn test_events_message_compact() {
        let events = vec![
            Event::new(EventKind::Motion {
                entity: 1,
                delta: [100, 0, 0],
            }),
            Event::new(EventKind::Motion {
                entity: 2,
                delta: [0, 100, 0],
            }),
        ];

        let msg = Message::Events { events };
        let compact = msg.to_compact_bytes();
        let bincode_size = msg.to_bytes().len();

        println!(
            "Events(2) - Compact: {} bytes, Bincode: {} bytes",
            compact.len(),
            bincode_size
        );
        assert!(compact.len() <= bincode_size);
    }
}
