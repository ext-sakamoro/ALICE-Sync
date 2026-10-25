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
//! - **Receive window**: the last [`RECV_WINDOW`] sequence numbers are remembered, so a retransmission
//!   that comes after newer packets is still taken once and a repeat is refused
//! - **Acknowledgement-only packets** ([`ReliableEndpoint::take_acks`]): a receiver acknowledges what it
//!   got even when it has no data to send; such a packet has an empty payload, takes no sequence number
//!   and is never delivered (an older peer decodes it as nothing and still reads its acknowledgements).
//!   A packet received far behind the newest is acknowledged by its own sequence number
//! - **Delivery failures** ([`ReliableEndpoint::take_failures`]): a packet that is never acknowledged
//!   after [`MAX_RETRANSMITS`] tries, or that is pushed out of the send window, is reported, not counted as
//!   acknowledged
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

/// How many sequence numbers, the newest received one included, the receiver remembers
pub const RECV_WINDOW: u16 = 1024;

/// How many times an unacknowledged packet is sent again before it is reported as given up
pub const MAX_RETRANSMITS: u8 = 10;

const RECV_WORDS: usize = (RECV_WINDOW / 64) as usize;

/// A packet the sender stopped trying to deliver
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeliveryFailure {
    /// Never acknowledged after [`MAX_RETRANSMITS`] retransmissions
    GaveUp { seq: u16 },
    /// Pushed out of the send window (too many packets in flight) before it was acknowledged
    Evicted { seq: u16 },
}

impl core::fmt::Display for DeliveryFailure {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::GaveUp { seq } => write!(f, "packet {seq} never acknowledged"),
            Self::Evicted { seq } => write!(f, "packet {seq} pushed out of the send window"),
        }
    }
}

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

/// Received packet tracking for deduplication: a bit per sequence number of the last [`RECV_WINDOW`]
#[derive(Debug)]
struct RecvTracker {
    /// Most recent received sequence
    most_recent: u16,
    /// slot `seq % RECV_WINDOW` is set when `seq`, within the window, has been received
    seen: [u64; RECV_WORDS],
    /// Whether we have received any packet yet
    initialized: bool,
}

impl RecvTracker {
    fn new() -> Self {
        Self {
            most_recent: 0,
            seen: [0; RECV_WORDS],
            initialized: false,
        }
    }

    const fn slot(seq: u16) -> (usize, u64) {
        let i = seq % RECV_WINDOW;
        ((i / 64) as usize, 1 << (i % 64))
    }

    fn has(&self, seq: u16) -> bool {
        let (w, b) = Self::slot(seq);
        self.seen[w] & b != 0
    }

    fn set(&mut self, seq: u16) {
        let (w, b) = Self::slot(seq);
        self.seen[w] |= b;
    }

    /// Whether `seq` lies in the window ending at the newest received sequence number
    fn in_window(&self, seq: u16) -> bool {
        let back = seq_diff(self.most_recent, seq);
        (0..i32::from(RECV_WINDOW)).contains(&back)
    }

    /// Record a received sequence number. Returns `false` if it was received before or is older than the
    /// window.
    fn record(&mut self, seq: u16) -> bool {
        if !self.initialized {
            self.most_recent = seq;
            self.initialized = true;
            self.set(seq);
            return true;
        }
        let diff = seq_diff(seq, self.most_recent);
        if diff > 0 {
            // newer: the slots the window moves over held sequence numbers that have now left it
            if diff >= i32::from(RECV_WINDOW) {
                self.seen = [0; RECV_WORDS];
            } else {
                for k in 1..=diff {
                    let (w, b) = Self::slot(self.most_recent.wrapping_add(k as u16));
                    self.seen[w] &= !b;
                }
            }
            self.most_recent = seq;
            self.set(seq);
            return true;
        }
        if !self.in_window(seq) || self.has(seq) {
            return false;
        }
        self.set(seq);
        true
    }

    /// The acknowledgement header with `reference` as its newest acknowledged sequence number: bit `i`
    /// stands for `reference - 1 - i`
    fn ack_header_for(&self, reference: u16) -> (u16, u32) {
        let mut bits = 0u32;
        for i in 0..32u16 {
            let seq = reference.wrapping_sub(1 + i);
            if self.in_window(seq) && self.has(seq) {
                bits |= 1 << i;
            }
        }
        (reference, bits)
    }

    /// Build ack header from current state
    fn ack_header(&self) -> (u16, u32) {
        self.ack_header_for(self.most_recent)
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
    /// Received sequence numbers to acknowledge by name in the next acknowledgement-only packets
    acks_due: std::collections::BTreeSet<u16>,
    /// Packets given up or evicted, not yet taken
    failures: Vec<DeliveryFailure>,
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
            acks_due: std::collections::BTreeSet::new(),
            failures: Vec::new(),
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
            send_time: crate::clock::now(),
            payload: payload.to_vec(),
            acked: false,
            retransmit_count: 0,
        });

        // Trim old sent packets: one never acknowledged is reported
        while self.sent_packets.len() > self.max_in_flight {
            if let Some(old) = self.sent_packets.pop_front() {
                if !old.acked {
                    self.packets_lost += 1;
                    self.failures
                        .push(DeliveryFailure::Evicted { seq: old.seq });
                }
            }
        }
        // this packet's header acknowledges the newest received one and the 32 before it
        let newest = self.recv_tracker.most_recent;
        self.acks_due
            .retain(|r| !(0..33).contains(&seq_diff(newest, *r)));

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

        let payload_end = HEADER_SIZE + header.payload_len as usize;
        if packet.len() < payload_end {
            return None; // truncated
        }

        // Process ACKs from remote
        self.process_acks(header.ack, header.ack_bits);

        // an acknowledgement-only packet carries no data and takes no sequence number
        if header.payload_len == 0 {
            return None;
        }

        let fresh = self.recv_tracker.record(header.seq);
        // a repeat is acknowledged again too: the sender did not see the earlier acknowledgement
        if fresh || self.recv_tracker.in_window(header.seq) {
            self.acks_due.insert(header.seq);
        }
        if !fresh {
            return None; // duplicate or too old
        }

        Some(packet[HEADER_SIZE..payload_end].to_vec())
    }

    /// Process ACK information from a received packet header
    fn process_acks(&mut self, ack: u16, ack_bits: u32) {
        let now = crate::clock::now();

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

    /// Collect packets that need retransmission (timed out, not yet acked). A packet that has been sent
    /// [`MAX_RETRANSMITS`] times more without an acknowledgement is dropped and reported
    /// ([`Self::take_failures`]).
    pub fn collect_retransmits(&mut self) -> Vec<Vec<u8>> {
        self.retransmit(false)
    }

    /// As [`Self::collect_retransmits`] for every unacknowledged packet now, whatever its timer (for a
    /// caller that knows the path changed)
    pub fn collect_retransmits_now(&mut self) -> Vec<Vec<u8>> {
        self.retransmit(true)
    }

    fn retransmit(&mut self, all: bool) -> Vec<Vec<u8>> {
        let now = crate::clock::now();
        let rto = self.rtt.rto();
        let mut retransmits = Vec::new();
        let mut gave_up = Vec::new();

        for sent in &mut self.sent_packets {
            if sent.acked {
                continue;
            }
            if all || now.duration_since(sent.send_time) >= rto {
                sent.send_time = now; // reset timer
                sent.retransmit_count += 1;

                if sent.retransmit_count > MAX_RETRANSMITS {
                    gave_up.push(sent.seq);
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

        if !gave_up.is_empty() {
            self.sent_packets.retain(|p| !gave_up.contains(&p.seq));
            for seq in gave_up {
                self.packets_lost += 1;
                self.failures.push(DeliveryFailure::GaveUp { seq });
            }
        }
        retransmits
    }

    /// Acknowledgement-only packets for what was received since the last packet this side sent: one with
    /// the newest received sequence number, and one per packet received far behind it (each names its own
    /// sequence number). They take no sequence number and are not retransmitted
    pub fn take_acks(&mut self) -> Vec<Vec<u8>> {
        let due = std::mem::take(&mut self.acks_due);
        let newest = self.recv_tracker.most_recent;
        let mut refs: Vec<u16> = Vec::new();
        if due.iter().any(|r| (0..33).contains(&seq_diff(newest, *r))) {
            refs.push(newest);
        }
        refs.extend(
            due.into_iter()
                .filter(|r| !(0..33).contains(&seq_diff(newest, *r))),
        );
        refs.into_iter()
            .map(|reference| {
                let (ack, ack_bits) = self.recv_tracker.ack_header_for(reference);
                let header = PacketHeader {
                    seq: self.local_seq.wrapping_sub(1),
                    ack,
                    ack_bits,
                    payload_len: 0,
                };
                header.to_bytes().to_vec()
            })
            .collect()
    }

    /// The packets given up or evicted since the last call
    pub fn take_failures(&mut self) -> Vec<DeliveryFailure> {
        std::mem::take(&mut self.failures)
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
    pending: std::collections::BTreeMap<u16, FragmentEntry>,
    /// Timeout for incomplete messages
    timeout: Duration,
    /// Creation time for each pending message
    timestamps: std::collections::BTreeMap<u16, Instant>,
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
            pending: std::collections::BTreeMap::new(),
            timeout: Duration::from_secs(5),
            timestamps: std::collections::BTreeMap::new(),
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
            .or_insert_with(crate::clock::now);

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
        let now = crate::clock::now();
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
