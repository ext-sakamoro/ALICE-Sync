//! UDP Reliability Layer
//!
//! Game-grade lightweight reliability on top of unreliable datagrams.
//!
//! # Design
//!
//! - **Sequence numbers**: `u16` wrapping (65536 cycle)
//! - **ACK bitfield**: 32-bit trailing ACK (covers last 33 packets)
//! - **Retransmission**: RTT-based adaptive timeout
//! - **Fragmentation**: MTU-safe split/reassemble for large payloads
//!
//! # Packet Header (10 bytes)
//!
//! ```text
//! [seq:2][ack:2][ack_bits:4][payload_len:2]
//! ```

#![allow(
    clippy::missing_const_for_fn,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::missing_panics_doc,
    clippy::branches_sharing_code
)]

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Maximum payload per UDP datagram (conservative MTU - IP/UDP headers)
pub const MAX_PAYLOAD: usize = 1200;

/// Reliability header size in bytes
pub const HEADER_SIZE: usize = 10;

/// Maximum fragment count per message
pub const MAX_FRAGMENTS: u8 = 255;

/// Packet header on the wire
#[derive(Debug, Clone, Copy)]
pub struct PacketHeader {
    /// Sequence number (sender's outgoing counter)
    pub seq: u16,
    /// Most recent received sequence (remote ack)
    pub ack: u16,
    /// Bitfield of previous 32 acks (bit 0 = ack-1, bit 1 = ack-2, ...)
    pub ack_bits: u32,
    /// Payload length in bytes
    pub payload_len: u16,
}

impl PacketHeader {
    /// Serialize to 10 bytes (little-endian)
    #[must_use]
    pub fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0..2].copy_from_slice(&self.seq.to_le_bytes());
        buf[2..4].copy_from_slice(&self.ack.to_le_bytes());
        buf[4..8].copy_from_slice(&self.ack_bits.to_le_bytes());
        buf[8..10].copy_from_slice(&self.payload_len.to_le_bytes());
        buf
    }

    /// Deserialize from 10 bytes
    #[must_use]
    pub fn from_bytes(buf: &[u8; HEADER_SIZE]) -> Self {
        Self {
            seq: u16::from_le_bytes([buf[0], buf[1]]),
            ack: u16::from_le_bytes([buf[2], buf[3]]),
            ack_bits: u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]),
            payload_len: u16::from_le_bytes([buf[8], buf[9]]),
        }
    }
}

/// Sent packet metadata for ACK tracking
#[derive(Debug)]
struct SentPacket {
    seq: u16,
    send_time: Instant,
    payload: Vec<u8>,
    acked: bool,
    retransmit_count: u8,
}

/// Received packet tracking for deduplication
#[derive(Debug)]
struct RecvTracker {
    /// Most recent received sequence
    most_recent: u16,
    /// Bitfield of received packets (relative to `most_recent`)
    recv_bits: u32,
    /// Whether we have received any packet yet
    initialized: bool,
}

impl RecvTracker {
    fn new() -> Self {
        Self {
            most_recent: 0,
            recv_bits: 0,
            initialized: false,
        }
    }

    /// Record a received sequence number. Returns `false` if duplicate.
    fn record(&mut self, seq: u16) -> bool {
        if !self.initialized {
            self.most_recent = seq;
            self.initialized = true;
            return true;
        }

        let diff = seq_diff(seq, self.most_recent);

        if diff == 0 {
            return false; // duplicate of most_recent
        }

        if diff > 0 {
            // Newer packet: shift bits and update most_recent
            let shift = diff.min(32) as u32;
            self.recv_bits = if shift >= 32 {
                0
            } else {
                self.recv_bits << shift
            };
            // Mark the old most_recent in the bitfield
            if shift <= 32 {
                self.recv_bits |= 1 << (shift - 1);
            }
            self.most_recent = seq;
            true
        } else {
            // Older packet: check if already received
            let bit = (-diff) as u32;
            if bit > 32 {
                return false; // too old
            }
            let mask = 1 << (bit - 1);
            if self.recv_bits & mask != 0 {
                return false; // duplicate
            }
            self.recv_bits |= mask;
            true
        }
    }

    /// Build ack header from current state
    fn ack_header(&self) -> (u16, u32) {
        (self.most_recent, self.recv_bits)
    }
}

/// RTT estimator (exponential moving average)
#[derive(Debug)]
pub struct RttEstimator {
    /// Smoothed RTT
    srtt: Duration,
    /// RTT variance
    rttvar: Duration,
    /// Retransmission timeout
    rto: Duration,
    /// Sample count
    samples: u32,
}

impl Default for RttEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl RttEstimator {
    #[must_use]
    pub fn new() -> Self {
        Self {
            srtt: Duration::from_millis(100),
            rttvar: Duration::from_millis(50),
            rto: Duration::from_millis(250),
            samples: 0,
        }
    }

    /// Update with a new RTT sample
    pub fn update(&mut self, rtt: Duration) {
        if self.samples == 0 {
            self.srtt = rtt;
            self.rttvar = rtt / 2;
        } else {
            // RFC 6298 algorithm
            let diff = rtt.abs_diff(self.srtt);
            self.rttvar = self.rttvar * 3 / 4 + diff / 4;
            self.srtt = self.srtt * 7 / 8 + rtt / 8;
        }
        self.rto = self.srtt + self.rttvar * 4;
        // Clamp RTO to [10ms, 5s]
        self.rto = self.rto.max(Duration::from_millis(10));
        self.rto = self.rto.min(Duration::from_secs(5));
        self.samples += 1;
    }

    /// Current smoothed RTT
    #[must_use]
    pub fn srtt(&self) -> Duration {
        self.srtt
    }

    /// Current retransmission timeout
    #[must_use]
    pub fn rto(&self) -> Duration {
        self.rto
    }
}

/// Reliability endpoint — one per peer connection.
///
/// Handles sequencing, ACK tracking, retransmission, and RTT estimation.
#[derive(Debug)]
pub struct ReliableEndpoint {
    /// Outgoing sequence counter
    local_seq: u16,
    /// Sent packets awaiting ACK
    sent_packets: VecDeque<SentPacket>,
    /// Receive tracking for dedup + ack generation
    recv_tracker: RecvTracker,
    /// RTT estimator
    pub rtt: RttEstimator,
    /// Maximum unacked packets in flight
    max_in_flight: usize,
    /// Packet loss counter
    pub packets_lost: u64,
    /// Packet acked counter
    pub packets_acked: u64,
}

impl ReliableEndpoint {
    /// Create a new reliability endpoint
    #[must_use]
    pub fn new() -> Self {
        Self {
            local_seq: 0,
            sent_packets: VecDeque::with_capacity(256),
            recv_tracker: RecvTracker::new(),
            rtt: RttEstimator::new(),
            max_in_flight: 256,
            packets_lost: 0,
            packets_acked: 0,
        }
    }

    /// Current outgoing sequence number (for fragment message IDs)
    #[must_use]
    pub fn local_seq(&self) -> u16 {
        self.local_seq
    }

    /// Wrap a payload into a reliable packet (header + payload).
    ///
    /// Returns the serialized packet bytes ready for sending.
    pub fn wrap_outgoing(&mut self, payload: &[u8]) -> Vec<u8> {
        let seq = self.local_seq;
        self.local_seq = self.local_seq.wrapping_add(1);

        let (ack, ack_bits) = self.recv_tracker.ack_header();

        let header = PacketHeader {
            seq,
            ack,
            ack_bits,
            payload_len: payload.len() as u16,
        };

        self.sent_packets.push_back(SentPacket {
            seq,
            send_time: Instant::now(),
            payload: payload.to_vec(),
            acked: false,
            retransmit_count: 0,
        });

        // Trim old sent packets
        while self.sent_packets.len() > self.max_in_flight {
            if let Some(old) = self.sent_packets.pop_front() {
                if !old.acked {
                    self.packets_lost += 1;
                }
            }
        }

        let mut packet = Vec::with_capacity(HEADER_SIZE + payload.len());
        packet.extend_from_slice(&header.to_bytes());
        packet.extend_from_slice(payload);
        packet
    }

    /// Process an incoming packet. Returns the payload if not a duplicate.
    ///
    /// Also processes ACK information from the remote peer.
    pub fn unwrap_incoming(&mut self, packet: &[u8]) -> Option<Vec<u8>> {
        if packet.len() < HEADER_SIZE {
            return None;
        }

        let header_bytes: [u8; HEADER_SIZE] = packet[..HEADER_SIZE].try_into().ok()?;
        let header = PacketHeader::from_bytes(&header_bytes);

        // Process ACKs from remote
        self.process_acks(header.ack, header.ack_bits);

        // Check for duplicate
        if !self.recv_tracker.record(header.seq) {
            return None; // duplicate
        }

        let payload_end = HEADER_SIZE + header.payload_len as usize;
        if packet.len() < payload_end {
            return None; // truncated
        }

        Some(packet[HEADER_SIZE..payload_end].to_vec())
    }

    /// Process ACK information from a received packet header
    fn process_acks(&mut self, ack: u16, ack_bits: u32) {
        let now = Instant::now();

        for sent in &mut self.sent_packets {
            if sent.acked {
                continue;
            }

            let diff = seq_diff(ack, sent.seq);
            let is_acked = if diff == 0 {
                true
            } else if diff > 0 && diff <= 32 {
                ack_bits & (1 << (diff - 1)) != 0
            } else {
                false
            };

            if is_acked {
                sent.acked = true;
                self.packets_acked += 1;
                // Update RTT only for first transmission (not retransmits)
                if sent.retransmit_count == 0 {
                    let rtt = now.duration_since(sent.send_time);
                    self.rtt.update(rtt);
                }
            }
        }

        // Remove acked packets from front
        while self.sent_packets.front().is_some_and(|p| p.acked) {
            self.sent_packets.pop_front();
        }
    }

    /// Collect packets that need retransmission (timed out, not yet acked).
    pub fn collect_retransmits(&mut self) -> Vec<Vec<u8>> {
        let now = Instant::now();
        let rto = self.rtt.rto();
        let mut retransmits = Vec::new();

        for sent in &mut self.sent_packets {
            if sent.acked {
                continue;
            }
            if now.duration_since(sent.send_time) >= rto {
                sent.send_time = now; // reset timer
                sent.retransmit_count += 1;

                if sent.retransmit_count > 10 {
                    sent.acked = true; // give up
                    self.packets_lost += 1;
                    continue;
                }

                // Re-wrap with current ack state
                let (ack, ack_bits) = self.recv_tracker.ack_header();
                let header = PacketHeader {
                    seq: sent.seq,
                    ack,
                    ack_bits,
                    payload_len: sent.payload.len() as u16,
                };

                let mut packet = Vec::with_capacity(HEADER_SIZE + sent.payload.len());
                packet.extend_from_slice(&header.to_bytes());
                packet.extend_from_slice(&sent.payload);
                retransmits.push(packet);
            }
        }

        retransmits
    }

    /// Current packet loss rate (0.0 - 1.0)
    #[must_use]
    pub fn loss_rate(&self) -> f64 {
        let total = self.packets_acked + self.packets_lost;
        if total == 0 {
            0.0
        } else {
            self.packets_lost as f64 / total as f64
        }
    }
}

impl Default for ReliableEndpoint {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Fragment / Reassemble
// ============================================================================

/// Fragment header (3 bytes, prepended to each fragment payload)
///
/// ```text
/// [message_id:2][fragment_index:1 (high 4 bits = total, low 4 bits = index)]
/// ```
/// Supports up to 15 fragments per message (~18 KB with 1200 MTU).
#[derive(Debug, Clone, Copy)]
pub struct FragmentHeader {
    pub message_id: u16,
    pub fragment_count: u8,
    pub fragment_index: u8,
}

impl FragmentHeader {
    /// Serialize to 4 bytes
    #[must_use]
    pub fn to_bytes(&self) -> [u8; 4] {
        [
            self.message_id.to_le_bytes()[0],
            self.message_id.to_le_bytes()[1],
            self.fragment_count,
            self.fragment_index,
        ]
    }

    /// Deserialize from 4 bytes
    #[must_use]
    pub fn from_bytes(buf: &[u8; 4]) -> Self {
        Self {
            message_id: u16::from_le_bytes([buf[0], buf[1]]),
            fragment_count: buf[2],
            fragment_index: buf[3],
        }
    }
}

/// Fragment a large payload into MTU-safe chunks.
///
/// Each fragment includes a 4-byte fragment header.
/// Returns empty vec if payload fits in a single packet.
#[must_use]
pub fn fragment_payload(message_id: u16, payload: &[u8]) -> Vec<Vec<u8>> {
    let max_frag_payload = MAX_PAYLOAD - 4; // subtract fragment header

    if payload.len() <= MAX_PAYLOAD {
        // No fragmentation needed — return single unfragmented payload
        return vec![payload.to_vec()];
    }

    let fragment_count = payload.len().div_ceil(max_frag_payload);
    assert!(fragment_count <= MAX_FRAGMENTS as usize);

    let mut fragments = Vec::with_capacity(fragment_count);
    for (i, chunk) in payload.chunks(max_frag_payload).enumerate() {
        let header = FragmentHeader {
            message_id,
            fragment_count: fragment_count as u8,
            fragment_index: i as u8,
        };
        let mut frag = Vec::with_capacity(4 + chunk.len());
        frag.extend_from_slice(&header.to_bytes());
        frag.extend_from_slice(chunk);
        fragments.push(frag);
    }

    fragments
}

/// Fragment entry: (total count, per-index payloads)
type FragmentEntry = (u8, Vec<Option<Vec<u8>>>);

/// Reassembly buffer for fragmented messages
#[derive(Debug)]
pub struct ReassemblyBuffer {
    /// Pending messages: `message_id` → (`fragment_count`, received fragments)
    pending: rustc_hash::FxHashMap<u16, FragmentEntry>,
    /// Timeout for incomplete messages
    timeout: Duration,
    /// Creation time for each pending message
    timestamps: rustc_hash::FxHashMap<u16, Instant>,
}

impl Default for ReassemblyBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl ReassemblyBuffer {
    #[must_use]
    pub fn new() -> Self {
        Self {
            pending: rustc_hash::FxHashMap::default(),
            timeout: Duration::from_secs(5),
            timestamps: rustc_hash::FxHashMap::default(),
        }
    }

    /// Feed a fragment. Returns the complete reassembled payload if all
    /// fragments have been received.
    pub fn feed(&mut self, fragment: &[u8]) -> Option<Vec<u8>> {
        if fragment.len() < 4 {
            return None;
        }

        let header_bytes: [u8; 4] = fragment[..4].try_into().ok()?;
        let header = FragmentHeader::from_bytes(&header_bytes);
        let payload = &fragment[4..];

        let entry = self.pending.entry(header.message_id).or_insert_with(|| {
            (
                header.fragment_count,
                vec![None; header.fragment_count as usize],
            )
        });

        self.timestamps
            .entry(header.message_id)
            .or_insert_with(Instant::now);

        let idx = header.fragment_index as usize;
        if idx < entry.1.len() {
            entry.1[idx] = Some(payload.to_vec());
        }

        // Check if all fragments received
        if entry.1.iter().all(Option::is_some) {
            let complete: Vec<u8> = entry
                .1
                .iter()
                .flat_map(|f| f.as_ref().unwrap().iter().copied())
                .collect();
            self.pending.remove(&header.message_id);
            self.timestamps.remove(&header.message_id);
            Some(complete)
        } else {
            None
        }
    }

    /// Remove timed-out incomplete messages
    pub fn gc(&mut self) {
        let now = Instant::now();
        let timeout = self.timeout;
        let expired: Vec<u16> = self
            .timestamps
            .iter()
            .filter(|(_, t)| now.duration_since(**t) > timeout)
            .map(|(id, _)| *id)
            .collect();
        for id in expired {
            self.pending.remove(&id);
            self.timestamps.remove(&id);
        }
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Signed sequence difference handling u16 wrap-around.
/// Returns positive if `a` is ahead of `b`.
fn seq_diff(a: u16, b: u16) -> i32 {
    let diff = a.wrapping_sub(b) as i16;
    diff as i32
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packet_header_roundtrip() {
        let header = PacketHeader {
            seq: 1234,
            ack: 5678,
            ack_bits: 0xDEAD_BEEF,
            payload_len: 100,
        };
        let bytes = header.to_bytes();
        let restored = PacketHeader::from_bytes(&bytes);
        assert_eq!(restored.seq, 1234);
        assert_eq!(restored.ack, 5678);
        assert_eq!(restored.ack_bits, 0xDEAD_BEEF);
        assert_eq!(restored.payload_len, 100);
    }

    #[test]
    fn test_reliable_endpoint_basic() {
        let mut sender = ReliableEndpoint::new();
        let mut receiver = ReliableEndpoint::new();

        // Sender sends a packet
        let packet = sender.wrap_outgoing(b"hello");
        assert!(packet.len() >= HEADER_SIZE + 5);

        // Receiver processes it
        let payload = receiver.unwrap_incoming(&packet).unwrap();
        assert_eq!(payload, b"hello");

        // Receiver sends an ACK-bearing packet back
        let ack_packet = receiver.wrap_outgoing(b"ack");
        let ack_payload = sender.unwrap_incoming(&ack_packet).unwrap();
        assert_eq!(ack_payload, b"ack");

        // Sender should have 1 acked packet
        assert_eq!(sender.packets_acked, 1);
    }

    #[test]
    fn test_duplicate_rejection() {
        let mut sender = ReliableEndpoint::new();
        let mut receiver = ReliableEndpoint::new();

        let packet = sender.wrap_outgoing(b"once");
        assert!(receiver.unwrap_incoming(&packet).is_some());
        assert!(receiver.unwrap_incoming(&packet).is_none()); // duplicate
    }

    #[test]
    fn test_seq_wrap_around() {
        assert_eq!(seq_diff(0, 65535), 1); // 0 is 1 ahead of 65535
        assert_eq!(seq_diff(65535, 0), -1); // 65535 is 1 behind 0
        assert_eq!(seq_diff(100, 100), 0);
        assert_eq!(seq_diff(105, 100), 5);
    }

    #[test]
    fn test_rtt_estimator() {
        let mut rtt = RttEstimator::new();
        rtt.update(Duration::from_millis(50));
        assert!(rtt.srtt() <= Duration::from_millis(100));
        rtt.update(Duration::from_millis(60));
        rtt.update(Duration::from_millis(55));
        // RTO should be reasonable
        assert!(rtt.rto() < Duration::from_secs(1));
    }

    #[test]
    fn test_fragment_small_payload() {
        let fragments = fragment_payload(1, b"small");
        assert_eq!(fragments.len(), 1);
        assert_eq!(fragments[0], b"small"); // no fragment header for single
    }

    #[test]
    fn test_fragment_large_payload() {
        let data = vec![0xABu8; 3000]; // larger than MAX_PAYLOAD
        let fragments = fragment_payload(42, &data);
        assert!(fragments.len() > 1);

        // Reassemble
        let mut reasm = ReassemblyBuffer::new();
        let mut result = None;
        for frag in &fragments {
            if let Some(complete) = reasm.feed(frag) {
                result = Some(complete);
            }
        }
        assert_eq!(result.unwrap(), data);
    }

    #[test]
    fn test_reassembly_out_of_order() {
        let data = vec![0xCDu8; 3000];
        let mut fragments = fragment_payload(7, &data);
        fragments.reverse(); // out of order

        let mut reasm = ReassemblyBuffer::new();
        let mut result = None;
        for frag in &fragments {
            if let Some(complete) = reasm.feed(frag) {
                result = Some(complete);
            }
        }
        assert_eq!(result.unwrap(), data);
    }

    #[test]
    fn test_recv_tracker_ordering() {
        let mut tracker = RecvTracker::new();

        // First packet
        assert!(tracker.record(1));
        assert_eq!(tracker.most_recent, 1);

        // Sequential
        assert!(tracker.record(2));
        assert!(tracker.record(3));
        assert_eq!(tracker.most_recent, 3);

        // Out of order (older)
        assert!(tracker.record(0)); // This should set bit for seq 0

        // Duplicate
        assert!(!tracker.record(2));
    }

    #[test]
    fn test_loss_rate() {
        let endpoint = ReliableEndpoint::new();
        assert!(endpoint.loss_rate().abs() < f64::EPSILON);
    }
}
