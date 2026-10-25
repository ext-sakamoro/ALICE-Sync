//! UDP Reliability Layer
//!
//! Game-grade lightweight reliability on top of unreliable datagrams.
//!
//! # Design
//!
//! - **Sequence numbers**: `u64`, never wrapping (a peer would need 2^64 packets)
//! - **ACK bitfield**: 32-bit trailing ACK (covers the 33 packets up to the acknowledged one)
//! - **Retransmission**: RTT-based adaptive timeout
//! - **Fragmentation**: MTU-safe split/reassemble for large payloads, with a `u64` message id
//! - **Receive window**: the last [`RECV_WINDOW`] sequence numbers are remembered, so a retransmission
//!   that comes after newer packets is still taken once and a repeat is refused
//! - **Acknowledgement-only packets** ([`ReliableEndpoint::take_acks`]): a receiver acknowledges what it
//!   got even when it has no data to send; such a packet is its own kind, carries no sequence number and
//!   is never delivered. A packet received far behind the newest is acknowledged by its own sequence
//!   number
//! - **Delivery failures** ([`ReliableEndpoint::take_failures`]): a packet that is never acknowledged
//!   after [`MAX_RETRANSMITS`] tries, or that is pushed out of the send window, is reported, not counted as
//!   acknowledged
//!
//! # Packets (wire version 2)
//!
//! Every packet starts with [`WIRE_MAGIC`] (10 bytes: `AS`, a zero byte, the version, four zero bytes,
//! `0xff 0xff`), then its kind:
//!
//! ```text
//! data              [magic:10][1][seq:8][ack:8][ack_bits:4][payload_len:2][payload]
//! acknowledgement   [magic:10][2][ack:8][ack_bits:4]
//! ```
//!
//! all integers little-endian, and the datagram is exactly its kind's length (the payload length included).
//! A packet without the magic is not of this version and is refused by [`classify`] as
//! [`PacketError::OtherWire`]; the transport reports such a peer. No version-1 packet starts with the magic:
//! version 1 headers are `[seq:2][ack:2][ack_bits:4][payload_len:2]` with the datagram 10 + payload_len
//! bytes, so bytes 8 and 9 of the magic (payload length 0xffff) would make a version-1 datagram longer than
//! UDP carries. A datagram with the magic whose kind is unknown or whose length is not its kind's is
//! [`PacketError::Malformed`] and is dropped without a report.

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

/// The first bytes of every packet of this wire version: `AS`, a zero byte, the version (2), four zero bytes
/// and `0xff 0xff` (where a version-1 header holds a payload length no version-1 datagram can have)
pub const WIRE_MAGIC: [u8; 10] = *b"AS\x00\x02\x00\x00\x00\x00\xff\xff";

const M: usize = WIRE_MAGIC.len();

/// The kind byte of a data packet
pub const KIND_DATA: u8 = 1;

/// The kind byte of an acknowledgement-only packet
pub const KIND_ACK: u8 = 2;

/// Header size of a data packet in bytes (magic, kind, seq, ack, ack bits, payload length)
pub const HEADER_SIZE: usize = M + 1 + 8 + 8 + 4 + 2;

/// Size of an acknowledgement-only packet in bytes (magic, kind, ack, ack bits)
pub const ACK_SIZE: usize = M + 1 + 8 + 4;

/// Maximum fragment count per message
pub const MAX_FRAGMENTS: u8 = 255;

/// How many sequence numbers, the newest received one included, the receiver remembers
pub const RECV_WINDOW: u64 = 1024;

/// How many times an unacknowledged packet is sent again before it is reported as given up
pub const MAX_RETRANSMITS: u8 = 10;

const RECV_WORDS: usize = (RECV_WINDOW / 64) as usize;

/// A packet the sender stopped trying to deliver
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeliveryFailure {
    /// Never acknowledged after [`MAX_RETRANSMITS`] retransmissions
    GaveUp { seq: u64 },
    /// Pushed out of the send window (too many packets in flight) before it was acknowledged
    Evicted { seq: u64 },
    /// In flight when the endpoint stopped delivering ([`ReliableEndpoint::abandon`]: the transport dropped the
    /// peer's state at its peer limit)
    Abandoned { seq: u64 },
}

impl core::fmt::Display for DeliveryFailure {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::GaveUp { seq } => write!(f, "packet {seq} never acknowledged"),
            Self::Evicted { seq } => write!(f, "packet {seq} pushed out of the send window"),
            Self::Abandoned { seq } => write!(f, "packet {seq} abandoned with the peer's state"),
        }
    }
}

/// What a received datagram is
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Packet<'a> {
    /// A data packet: its header and payload
    Data(PacketHeader, &'a [u8]),
    /// An acknowledgement-only packet: the newest acknowledged sequence number and the 32 before it
    Ack { ack: u64, ack_bits: u32 },
}

/// Why a datagram is not a packet of this wire version
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PacketError {
    /// It does not start with [`WIRE_MAGIC`]: another wire version (an older peer)
    OtherWire,
    /// It starts with the magic, but its kind is unknown or its length is not its kind's
    Malformed,
}

/// The header of a data packet
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PacketHeader {
    /// Sequence number (sender's outgoing counter)
    pub seq: u64,
    /// Most recent received sequence (remote ack)
    pub ack: u64,
    /// Bitfield of the 32 acks before `ack` (bit 0 = ack-1, bit 1 = ack-2, ...)
    pub ack_bits: u32,
    /// Payload length in bytes
    pub payload_len: u16,
}

impl PacketHeader {
    /// The bytes of a data packet's header
    #[must_use]
    pub fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];
        buf[..M].copy_from_slice(&WIRE_MAGIC);
        buf[M] = KIND_DATA;
        buf[M + 1..M + 9].copy_from_slice(&self.seq.to_le_bytes());
        buf[M + 9..M + 17].copy_from_slice(&self.ack.to_le_bytes());
        buf[M + 17..M + 21].copy_from_slice(&self.ack_bits.to_le_bytes());
        buf[M + 21..M + 23].copy_from_slice(&self.payload_len.to_le_bytes());
        buf
    }

    /// Read a data packet's header (`None` without the magic or the data kind)
    #[must_use]
    pub fn from_bytes(buf: &[u8; HEADER_SIZE]) -> Option<Self> {
        if buf[..M] != WIRE_MAGIC || buf[M] != KIND_DATA {
            return None;
        }
        Some(Self {
            seq: u64_at(buf, M + 1),
            ack: u64_at(buf, M + 9),
            ack_bits: u32_at(buf, M + 17),
            payload_len: u16::from_le_bytes([buf[M + 21], buf[M + 22]]),
        })
    }
}

fn u64_at(b: &[u8], at: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[at..at + 8]);
    u64::from_le_bytes(a)
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    let mut a = [0u8; 4];
    a.copy_from_slice(&b[at..at + 4]);
    u32::from_le_bytes(a)
}

/// Read a datagram as a packet of this wire version
///
/// # Errors
/// [`PacketError::OtherWire`] without the magic, [`PacketError::Malformed`] for an unknown kind or a
/// datagram whose length is not its kind's (a data packet is exactly its header and its payload length)
pub fn classify(packet: &[u8]) -> Result<Packet<'_>, PacketError> {
    if packet.len() <= M || packet[..M] != WIRE_MAGIC {
        return Err(PacketError::OtherWire);
    }
    match packet[M] {
        KIND_DATA if packet.len() >= HEADER_SIZE => {
            let payload_len = u16::from_le_bytes([packet[M + 21], packet[M + 22]]);
            if packet.len() != HEADER_SIZE + usize::from(payload_len) {
                return Err(PacketError::Malformed);
            }
            Ok(Packet::Data(
                PacketHeader {
                    seq: u64_at(packet, M + 1),
                    ack: u64_at(packet, M + 9),
                    ack_bits: u32_at(packet, M + 17),
                    payload_len,
                },
                &packet[HEADER_SIZE..],
            ))
        }
        KIND_ACK if packet.len() == ACK_SIZE => Ok(Packet::Ack {
            ack: u64_at(packet, M + 1),
            ack_bits: u32_at(packet, M + 9),
        }),
        _ => Err(PacketError::Malformed),
    }
}

/// The bytes of an acknowledgement-only packet
#[must_use]
pub fn ack_packet(ack: u64, ack_bits: u32) -> [u8; ACK_SIZE] {
    let mut buf = [0u8; ACK_SIZE];
    buf[..M].copy_from_slice(&WIRE_MAGIC);
    buf[M] = KIND_ACK;
    buf[M + 1..M + 9].copy_from_slice(&ack.to_le_bytes());
    buf[M + 9..M + 13].copy_from_slice(&ack_bits.to_le_bytes());
    buf
}

/// Sent packet metadata for ACK tracking
#[derive(Debug)]
struct SentPacket {
    seq: u64,
    send_time: Instant,
    payload: Vec<u8>,
    acked: bool,
    retransmit_count: u8,
}

/// Received packet tracking for deduplication: a bit per sequence number of the last [`RECV_WINDOW`]
#[derive(Debug)]
struct RecvTracker {
    /// Most recent received sequence
    most_recent: u64,
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

    const fn slot(seq: u64) -> (usize, u64) {
        let i = seq % RECV_WINDOW;
        ((i / 64) as usize, 1 << (i % 64))
    }

    fn has(&self, seq: u64) -> bool {
        let (w, b) = Self::slot(seq);
        self.seen[w] & b != 0
    }

    fn set(&mut self, seq: u64) {
        let (w, b) = Self::slot(seq);
        self.seen[w] |= b;
    }

    /// Whether `seq` lies in the window ending at the newest received sequence number
    fn in_window(&self, seq: u64) -> bool {
        self.initialized && seq <= self.most_recent && self.most_recent - seq < RECV_WINDOW
    }

    /// Record a received sequence number. Returns `false` if it was received before or is older than the
    /// window.
    fn record(&mut self, seq: u64) -> bool {
        if !self.initialized {
            self.most_recent = seq;
            self.initialized = true;
            self.set(seq);
            return true;
        }
        if seq > self.most_recent {
            // newer: the slots the window moves over held sequence numbers that have now left it
            let diff = seq - self.most_recent;
            if diff >= RECV_WINDOW {
                self.seen = [0; RECV_WORDS];
            } else {
                for k in 1..=diff {
                    let (w, b) = Self::slot(self.most_recent + k);
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

    /// The acknowledgement with `reference` as its newest acknowledged sequence number: bit `i` stands
    /// for `reference - 1 - i`
    fn ack_for(&self, reference: u64) -> (u64, u32) {
        let mut bits = 0u32;
        for i in 0..32u64 {
            let Some(seq) = reference.checked_sub(1 + i) else {
                break;
            };
            if self.in_window(seq) && self.has(seq) {
                bits |= 1 << i;
            }
        }
        (reference, bits)
    }

    /// Build ack header from current state
    fn ack_header(&self) -> (u64, u32) {
        self.ack_for(self.most_recent)
    }
}

/// Whether `seq` is acknowledged by the acknowledgement `(ack, ack_bits)`
fn acknowledges(ack: u64, ack_bits: u32, seq: u64) -> bool {
    if seq == ack {
        return true;
    }
    seq < ack && ack - seq <= 32 && ack_bits & (1 << (ack - seq - 1)) != 0
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
    local_seq: u64,
    /// The next fragment message id
    next_message: u64,
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
    acks_due: std::collections::BTreeSet<u64>,
    /// Packets given up or evicted, not yet taken
    failures: Vec<DeliveryFailure>,
}

/// The `ack` of a header from an endpoint that has received nothing: with no bits set it acknowledges only
/// `u64::MAX`, which no endpoint reaches
const NO_ACK: u64 = u64::MAX;

impl ReliableEndpoint {
    /// Create a new reliability endpoint
    #[must_use]
    pub fn new() -> Self {
        Self {
            local_seq: 0,
            next_message: 0,
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

    /// The sequence number the next data packet takes
    #[must_use]
    pub fn local_seq(&self) -> u64 {
        self.local_seq
    }

    /// A fresh message id for the fragments of one message
    pub fn next_message_id(&mut self) -> u64 {
        let id = self.next_message;
        self.next_message += 1;
        id
    }

    fn header_ack(&self) -> (u64, u32) {
        if self.recv_tracker.initialized {
            self.recv_tracker.ack_header()
        } else {
            (NO_ACK, 0)
        }
    }

    /// Wrap a payload into a reliable data packet (header + payload).
    ///
    /// Returns the serialized packet bytes ready for sending.
    pub fn wrap_outgoing(&mut self, payload: &[u8]) -> Vec<u8> {
        let seq = self.local_seq;
        self.local_seq += 1;

        let (ack, ack_bits) = self.header_ack();

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
        if self.recv_tracker.initialized {
            self.acks_due
                .retain(|r| !(*r <= newest && newest - r <= 32));
        }

        let mut packet = Vec::with_capacity(HEADER_SIZE + payload.len());
        packet.extend_from_slice(&header.to_bytes());
        packet.extend_from_slice(payload);
        packet
    }

    /// Process an incoming packet of this wire version. Returns the payload of a data packet not seen
    /// before; `None` for a repeat, an acknowledgement-only packet, or a datagram that is not a packet of
    /// this version (use [`classify`] to tell those apart).
    pub fn unwrap_incoming(&mut self, packet: &[u8]) -> Option<Vec<u8>> {
        match classify(packet).ok()? {
            Packet::Ack { ack, ack_bits } => {
                self.process_acks(ack, ack_bits);
                None
            }
            Packet::Data(header, payload) => {
                self.process_acks(header.ack, header.ack_bits);
                let fresh = self.recv_tracker.record(header.seq);
                // a repeat is acknowledged again too: the sender did not see the earlier acknowledgement
                if fresh || self.recv_tracker.in_window(header.seq) {
                    self.acks_due.insert(header.seq);
                }
                fresh.then(|| payload.to_vec())
            }
        }
    }

    /// Process ACK information from a received packet
    fn process_acks(&mut self, ack: u64, ack_bits: u32) {
        let now = crate::clock::now();

        for sent in &mut self.sent_packets {
            if sent.acked || !acknowledges(ack, ack_bits, sent.seq) {
                continue;
            }
            sent.acked = true;
            self.packets_acked += 1;
            // Update RTT only for first transmission (not retransmits)
            if sent.retransmit_count == 0 {
                let rtt = now.duration_since(sent.send_time);
                self.rtt.update(rtt);
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
        let (ack, ack_bits) = self.header_ack();
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
    /// sequence number). They carry no sequence number and are not retransmitted
    pub fn take_acks(&mut self) -> Vec<Vec<u8>> {
        let due = std::mem::take(&mut self.acks_due);
        if !self.recv_tracker.initialized {
            return Vec::new();
        }
        let newest = self.recv_tracker.most_recent;
        let near = |r: &u64| *r <= newest && newest - r <= 32;
        let mut refs: Vec<u64> = Vec::new();
        if due.iter().any(near) {
            refs.push(newest);
        }
        refs.extend(due.iter().copied().filter(|r| !near(r)));
        refs.into_iter()
            .map(|reference| {
                let (ack, ack_bits) = self.recv_tracker.ack_for(reference);
                ack_packet(ack, ack_bits).to_vec()
            })
            .collect()
    }

    /// Stop trying to deliver every packet not yet acknowledged, and return them as
    /// [`DeliveryFailure::Abandoned`] (after the failures not yet taken); the endpoint keeps no packet in flight
    pub fn abandon(&mut self) -> Vec<DeliveryFailure> {
        let mut out = std::mem::take(&mut self.failures);
        for sent in self.sent_packets.drain(..) {
            if !sent.acked {
                self.packets_lost += 1;
                out.push(DeliveryFailure::Abandoned { seq: sent.seq });
            }
        }
        out
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

/// Fragment header (10 bytes, prepended to each fragment payload)
///
/// ```text
/// [message_id:8][fragment_count:1][fragment_index:1]
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FragmentHeader {
    pub message_id: u64,
    pub fragment_count: u8,
    pub fragment_index: u8,
}

/// Bytes of a [`FragmentHeader`]
pub const FRAGMENT_HEADER_SIZE: usize = 10;

impl FragmentHeader {
    /// Serialize to 10 bytes
    #[must_use]
    pub fn to_bytes(&self) -> [u8; FRAGMENT_HEADER_SIZE] {
        let mut b = [0u8; FRAGMENT_HEADER_SIZE];
        b[..8].copy_from_slice(&self.message_id.to_le_bytes());
        b[8] = self.fragment_count;
        b[9] = self.fragment_index;
        b
    }

    /// Deserialize from 10 bytes
    #[must_use]
    pub fn from_bytes(buf: &[u8; FRAGMENT_HEADER_SIZE]) -> Self {
        Self {
            message_id: u64_at(buf, 0),
            fragment_count: buf[8],
            fragment_index: buf[9],
        }
    }
}

/// Fragment a large payload into MTU-safe chunks.
///
/// Each fragment includes a [`FRAGMENT_HEADER_SIZE`]-byte fragment header.
/// Returns the payload whole if it fits in a single packet.
#[must_use]
pub fn fragment_payload(message_id: u64, payload: &[u8]) -> Vec<Vec<u8>> {
    let max_frag_payload = MAX_PAYLOAD - FRAGMENT_HEADER_SIZE;

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
        let mut frag = Vec::with_capacity(FRAGMENT_HEADER_SIZE + chunk.len());
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
    pending: std::collections::BTreeMap<u64, FragmentEntry>,
    /// Timeout for incomplete messages
    timeout: Duration,
    /// Creation time for each pending message
    timestamps: std::collections::BTreeMap<u64, Instant>,
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
        if fragment.len() < FRAGMENT_HEADER_SIZE {
            return None;
        }

        let header_bytes: [u8; FRAGMENT_HEADER_SIZE] =
            fragment[..FRAGMENT_HEADER_SIZE].try_into().ok()?;
        let header = FragmentHeader::from_bytes(&header_bytes);
        let payload = &fragment[FRAGMENT_HEADER_SIZE..];

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
        let expired: Vec<u64> = self
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packet_header_roundtrip() {
        let header = PacketHeader {
            seq: 0x0123_4567_89AB_CDEF,
            ack: 5678,
            ack_bits: 0xDEAD_BEEF,
            payload_len: 100,
        };
        let bytes = header.to_bytes();
        assert_eq!(&bytes[..M], &WIRE_MAGIC);
        assert_eq!(bytes[M], KIND_DATA);
        assert_eq!(PacketHeader::from_bytes(&bytes), Some(header));
    }

    /// No version-1 datagram (10-byte header, then its payload length of bytes) starts with the magic, also
    /// one whose first bytes spell the version-2 prefix (seq 0x5341, ack 0x0200)
    #[test]
    fn test_no_version_1_datagram_has_the_magic() {
        for len in 0..=1200usize {
            for bits_low in [0u8, 1, 2, 0xff] {
                let mut p = vec![0u8; 10 + len];
                p[0..2].copy_from_slice(&0x5341u16.to_le_bytes());
                p[2..4].copy_from_slice(&0x0200u16.to_le_bytes());
                p[4] = bits_low;
                p[8..10].copy_from_slice(&(len as u16).to_le_bytes());
                assert_eq!(&p[..4], b"AS\x00\x02");
                assert_eq!(classify(&p), Err(PacketError::OtherWire), "len {len}");
            }
        }
        // the payload length the magic implies cannot fit a datagram
        assert!(10 + usize::from(u16::from_le_bytes([WIRE_MAGIC[8], WIRE_MAGIC[9]])) > 65_507);
    }

    #[test]
    fn test_classify_refuses_other_wires() {
        // a version-1 data header: [seq:2][ack:2][ack_bits:4][payload_len:2]
        let old = [1u8, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0x55];
        assert_eq!(classify(&old), Err(PacketError::OtherWire));
        assert_eq!(classify(b""), Err(PacketError::OtherWire));
        let mut other_version = ack_packet(3, 0);
        other_version[3] = 1;
        assert_eq!(classify(&other_version), Err(PacketError::OtherWire));
        let mut unknown_kind = ack_packet(3, 0);
        unknown_kind[M] = 9;
        assert_eq!(classify(&unknown_kind), Err(PacketError::Malformed));
        assert_eq!(
            classify(&ack_packet(3, 0)[..ACK_SIZE - 1]),
            Err(PacketError::Malformed)
        );
    }

    #[test]
    fn test_data_carries_its_payload() {
        let mut ep = ReliableEndpoint::new();
        let p = ep.wrap_outgoing(b"abc");
        assert!(matches!(classify(&p), Ok(Packet::Data(_, b"abc"))));
        assert_eq!(classify(&p[..p.len() - 1]), Err(PacketError::Malformed));
        let longer = [&p[..], &[0u8]].concat();
        assert_eq!(
            classify(&longer),
            Err(PacketError::Malformed),
            "a trailing byte"
        );
        let ack_longer = [&ack_packet(3, 0)[..], &[0u8]].concat();
        assert_eq!(classify(&ack_longer), Err(PacketError::Malformed));
        let empty = ep.wrap_outgoing(b"");
        assert!(matches!(classify(&empty), Ok(Packet::Data(_, b""))));
    }

    #[test]
    fn test_ack_packet_has_no_seq() {
        let a = ack_packet(0x1_0000_0000, 0b101);
        assert_eq!(a.len(), ACK_SIZE);
        assert_eq!(
            classify(&a),
            Ok(Packet::Ack {
                ack: 0x1_0000_0000,
                ack_bits: 0b101
            })
        );
    }

    #[test]
    fn test_reliable_endpoint_basic() {
        let mut sender = ReliableEndpoint::new();
        let mut receiver = ReliableEndpoint::new();

        let packet = sender.wrap_outgoing(b"hello");
        assert_eq!(packet.len(), HEADER_SIZE + 5);
        assert_eq!(receiver.unwrap_incoming(&packet).unwrap(), b"hello");

        let reply = receiver.wrap_outgoing(b"ack");
        assert_eq!(sender.unwrap_incoming(&reply).unwrap(), b"ack");
        assert_eq!(sender.packets_acked, 1);
    }

    #[test]
    fn test_first_header_acknowledges_nothing() {
        let mut a = ReliableEndpoint::new();
        let mut b = ReliableEndpoint::new();
        let first = a.wrap_outgoing(b"x");
        // b sends before receiving anything: its header must not acknowledge a's seq 0
        let early = b.wrap_outgoing(b"y");
        a.unwrap_incoming(&early).unwrap();
        assert_eq!(a.packets_acked, 0);
        b.unwrap_incoming(&first).unwrap();
        a.unwrap_incoming(&b.wrap_outgoing(b"z")).unwrap();
        assert_eq!(a.packets_acked, 1);
    }

    #[test]
    fn test_duplicate_rejection() {
        let mut sender = ReliableEndpoint::new();
        let mut receiver = ReliableEndpoint::new();

        let packet = sender.wrap_outgoing(b"once");
        assert!(receiver.unwrap_incoming(&packet).is_some());
        assert!(receiver.unwrap_incoming(&packet).is_none());
    }

    #[test]
    fn test_acknowledges() {
        assert!(acknowledges(100, 0, 100));
        assert!(acknowledges(100, 1, 99));
        assert!(!acknowledges(100, 1, 98));
        assert!(acknowledges(100, 1 << 31, 68));
        assert!(!acknowledges(100, u32::MAX, 67));
        assert!(!acknowledges(100, u32::MAX, 101));
        assert!(acknowledges(5, 0b1_0000, 0));
    }

    #[test]
    fn test_rtt_estimator() {
        let mut rtt = RttEstimator::new();
        rtt.update(Duration::from_millis(50));
        assert!(rtt.srtt() <= Duration::from_millis(100));
        rtt.update(Duration::from_millis(60));
        rtt.update(Duration::from_millis(55));
        assert!(rtt.rto() < Duration::from_secs(1));
    }

    #[test]
    fn test_fragment_small_payload() {
        let fragments = fragment_payload(1, b"small");
        assert_eq!(fragments.len(), 1);
        assert_eq!(fragments[0], b"small");
    }

    #[test]
    fn test_fragment_large_payload() {
        let data = vec![0xABu8; 3000];
        let fragments = fragment_payload(42, &data);
        assert!(fragments.len() > 1);

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
    fn test_fragment_ids_past_u16_do_not_collide() {
        let a = vec![1u8; 3000];
        let b = vec![2u8; 3000];
        let fa = fragment_payload(7, &a);
        let fb = fragment_payload(7 + (1 << 16), &b);
        let mut reasm = ReassemblyBuffer::new();
        assert!(reasm.feed(&fa[0]).is_none());
        assert!(reasm.feed(&fb[0]).is_none());
        let mut got = Vec::new();
        for f in fa[1..].iter().chain(&fb[1..]) {
            if let Some(m) = reasm.feed(f) {
                got.push(m);
            }
        }
        assert_eq!(got, vec![a, b]);
    }

    #[test]
    fn test_reassembly_out_of_order() {
        let data = vec![0xCDu8; 3000];
        let mut fragments = fragment_payload(7, &data);
        fragments.reverse();

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
        assert!(tracker.record(1));
        assert_eq!(tracker.most_recent, 1);
        assert!(tracker.record(2));
        assert!(tracker.record(3));
        assert_eq!(tracker.most_recent, 3);
        assert!(tracker.record(0));
        assert!(!tracker.record(2));
    }

    /// Sequence numbers past 2^16 and 2^32 are neither repeats of small ones nor older than them
    #[test]
    fn test_recv_tracker_past_u16_and_u32() {
        for base in [(1u64 << 16) - 3, (1u64 << 32) - 3] {
            let mut t = RecvTracker::new();
            for s in base..base + 6 {
                assert!(t.record(s), "{s}");
            }
            assert_eq!(t.most_recent, base + 5);
            assert!(!t.record(base + 1), "repeat across the boundary");
            assert!(!t.record(base - (RECV_WINDOW - 4)), "older than the window");
            let (ack, bits) = t.ack_header();
            assert_eq!((ack, bits & 0b1_1111), (base + 5, 0b1_1111));
        }
    }

    #[test]
    fn test_endpoint_past_u16_and_u32() {
        for base in [(1u64 << 16) - 2, (1u64 << 32) - 2] {
            let mut a = ReliableEndpoint::new();
            let mut b = ReliableEndpoint::new();
            a.local_seq = base;
            let ps: Vec<_> = (0..4u8).map(|i| a.wrap_outgoing(&[i])).collect();
            for (i, p) in ps.iter().enumerate() {
                assert_eq!(b.unwrap_incoming(p), Some(vec![i as u8]));
            }
            assert_eq!(b.unwrap_incoming(&ps[0]), None);
            for ack in b.take_acks() {
                a.unwrap_incoming(&ack);
            }
            assert_eq!(a.packets_acked, 4);
            assert!(a.sent_packets.is_empty());
        }
    }

    /// A sequence number exactly the window behind the newest shares its slot: it is never acknowledged
    #[test]
    fn test_ack_far_behind_ignores_the_slot_of_the_newest() {
        let mut t = RecvTracker::new();
        assert!(t.record(1100));
        assert!(t.record(100));
        let (ack, bits) = t.ack_for(100);
        assert_eq!(
            (ack, bits),
            (100, 0),
            "76 = 1100 - RECV_WINDOW was never received"
        );
    }

    /// A sequence number older than the window is refused although its slot is free
    #[test]
    fn test_older_than_the_window_with_a_free_slot() {
        let mut t = RecvTracker::new();
        assert!(t.record(2000));
        assert!(!t.record(100));
        assert!(!t.record(2000 - RECV_WINDOW));
        assert!(t.record(2001 - RECV_WINDOW));
    }

    /// What a data packet's header acknowledges is not acknowledged again by an acknowledgement-only one
    #[test]
    fn test_data_header_covers_the_due_acks() {
        let mut a = ReliableEndpoint::new();
        let mut b = ReliableEndpoint::new();
        b.unwrap_incoming(&a.wrap_outgoing(b"x")).unwrap();
        let reply = b.wrap_outgoing(b"y");
        assert!(b.take_acks().is_empty());
        a.unwrap_incoming(&reply).unwrap();
        assert_eq!(a.packets_acked, 1);
    }

    /// A repeat is acknowledged again: the sender did not see the first acknowledgement
    #[test]
    fn test_a_repeat_is_acknowledged_again() {
        let mut a = ReliableEndpoint::new();
        let mut b = ReliableEndpoint::new();
        let p = a.wrap_outgoing(b"x");
        b.unwrap_incoming(&p).unwrap();
        assert_eq!(b.take_acks().len(), 1);
        assert_eq!(b.unwrap_incoming(&p), None);
        let again = b.take_acks();
        assert_eq!(again.len(), 1);
        a.unwrap_incoming(&again[0]);
        assert_eq!(a.packets_acked, 1);
    }

    /// A packet is sent [`MAX_RETRANSMITS`] more times, then given up
    #[test]
    fn test_retransmits_then_gives_up() {
        let mut a = ReliableEndpoint::new();
        a.wrap_outgoing(b"void");
        let mut sent = 0;
        for _ in 0..20 {
            sent += a.collect_retransmits_now().len();
        }
        assert_eq!(sent, usize::from(MAX_RETRANSMITS));
        assert_eq!(a.take_failures(), vec![DeliveryFailure::GaveUp { seq: 0 }]);
    }

    #[test]
    fn test_loss_rate() {
        let endpoint = ReliableEndpoint::new();
        assert!(endpoint.loss_rate().abs() < f64::EPSILON);
    }
}
