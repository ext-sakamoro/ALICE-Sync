//! UDP Reliability Layer
//!
//! Game-grade lightweight reliability on top of unreliable datagrams.
//!
//! # Design
//!
//! - **Links and epochs**: an endpoint's traffic to its peer is a link the peer opens: the sender sends an
//!   open request with a token (the next value of its own [`Tokens`]), the receiver answers with the link's
//!   [`Epoch`] (its own next token and the sender's). Data and acknowledgements carry the epoch; a receiver
//!   delivers only data of the epoch it holds for that address, and a sender takes only acknowledgements of
//!   its current epoch. A receiver that holds no link of that epoch (it dropped the peer's state, or it
//!   restarted) answers with a reopen request, and the sender opens a new link and sends what was not
//!   acknowledged again under the new epoch, so a link that was re-created never has its data acknowledged
//!   as repeats of the old one
//! - **Sequence numbers**: `u64`, never wrapping (a peer would need 2^64 packets), counted per endpoint
//!   across its links
//! - **ACK bitfield**: 32-bit trailing ACK (covers the 33 packets up to the acknowledged one)
//! - **Retransmission**: RTT-based adaptive timeout, also of an open request not yet answered
//! - **Fragmentation**: MTU-safe split/reassemble for large payloads, with a `u64` message id
//! - **Receive window**: the last [`RECV_WINDOW`] sequence numbers of the link are remembered, so a
//!   retransmission that comes after newer packets is still taken once and a repeat is refused
//! - **Acknowledgement-only packets** ([`ReliableEndpoint::take_acks`]): the receiver acknowledges what it
//!   got; a packet received far behind the newest is acknowledged by its own sequence number
//! - **Delivery failures** ([`ReliableEndpoint::take_failures`]): a packet that is never acknowledged
//!   after [`MAX_RETRANSMITS`] tries (an open request never answered included), that is pushed out of the
//!   send window, or that was in flight when the endpoint was abandoned, is reported, not counted as
//!   acknowledged
//!
//! # Packets (wire version 2)
//!
//! Every packet starts with [`WIRE_MAGIC`] (10 bytes: `AS`, a zero byte, the version, four zero bytes,
//! `0xff 0xff`), then its kind:
//!
//! ```text
//! data              [magic:10][1][epoch:16][seq:8][payload_len:2][payload]
//! acknowledgement   [magic:10][2][epoch:16][ack:8][ack_bits:4]
//! open              [magic:10][3][token:8]
//! accept            [magic:10][4][epoch:16]
//! reopen            [magic:10][5][epoch:16]
//! ```
//!
//! An epoch is the receiver's token then the sender's (8 bytes each). All integers little-endian, and the
//! datagram is exactly its kind's length (the payload length included). A packet without the magic is not
//! of this version and is refused by [`classify`] as [`PacketError::OtherWire`]; the transport reports such
//! a peer. No version-1 packet starts with the magic: version 1 headers are
//! `[seq:2][ack:2][ack_bits:4][payload_len:2]` with the datagram 10 + payload_len bytes, so bytes 8 and 9 of
//! the magic (payload length 0xffff) would make a version-1 datagram longer than UDP carries. A datagram
//! with the magic whose kind is unknown or whose length is not its kind's is [`PacketError::Malformed`] and
//! is dropped without a report.
//!
//! # Limits
//!
//! - Plain UDP is not authenticated: a datagram forged with a peer's address can open, reopen or reset a
//!   link. The secure transport authenticates every datagram
//! - Delivery is at least once across a lost link: a packet the receiver delivered whose acknowledgement
//!   was lost is sent again under the new epoch when the receiver dropped the link (or restarted) before
//!   acknowledging it
//! - Tokens restart with the process. When both peers restart at once while a datagram of the old link is
//!   still in flight, the new epoch can equal the old one; [`Tokens::starting_at`] takes a caller-chosen
//!   start (an incarnation number) that keeps the epochs of two incarnations apart

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

/// The kind byte of an open request
pub const KIND_OPEN: u8 = 3;

/// The kind byte of the answer to an open request
pub const KIND_ACCEPT: u8 = 4;

/// The kind byte of a reopen request
pub const KIND_REOPEN: u8 = 5;

/// Header size of a data packet in bytes (magic, kind, epoch, seq, payload length)
pub const HEADER_SIZE: usize = M + 1 + 16 + 8 + 2;

/// Size of an acknowledgement-only packet in bytes (magic, kind, epoch, ack, ack bits)
pub const ACK_SIZE: usize = M + 1 + 16 + 8 + 4;

/// Size of an open request in bytes (magic, kind, token)
pub const OPEN_SIZE: usize = M + 1 + 8;

/// Size of an accept or a reopen request in bytes (magic, kind, epoch)
pub const EPOCH_PACKET_SIZE: usize = M + 1 + 16;

/// Maximum fragment count per message
pub const MAX_FRAGMENTS: u8 = 255;

/// How many sequence numbers, the newest received one included, the receiver remembers
pub const RECV_WINDOW: u64 = 1024;

/// How many times an unacknowledged packet is sent again before it is reported as given up
pub const MAX_RETRANSMITS: u8 = 10;

const RECV_WORDS: usize = (RECV_WINDOW / 64) as usize;

/// A link: the receiver's token and the sender's
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Epoch {
    /// The token the receiver gave the link when it accepted it
    pub receiver: u64,
    /// The token of the sender's open request
    pub sender: u64,
}

impl Epoch {
    fn to_bytes(self) -> [u8; 16] {
        let mut b = [0u8; 16];
        b[..8].copy_from_slice(&self.receiver.to_le_bytes());
        b[8..].copy_from_slice(&self.sender.to_le_bytes());
        b
    }

    fn at(b: &[u8], at: usize) -> Self {
        Self {
            receiver: u64_at(b, at),
            sender: u64_at(b, at + 8),
        }
    }
}

/// The tokens of one transport (every link it opens and every link it accepts takes the next one): never
/// the same value twice while it lives
#[derive(Debug, Clone)]
pub struct Tokens {
    next: u64,
}

impl Tokens {
    /// Tokens from 0
    #[must_use]
    pub fn new() -> Self {
        Self::starting_at(0)
    }

    /// Tokens from `start`: a caller that numbers its incarnations passes `incarnation << 32` (or any start
    /// past the tokens of the previous incarnation), so the epochs of two incarnations differ
    #[must_use]
    pub fn starting_at(start: u64) -> Self {
        Self { next: start }
    }

    /// The next token
    pub fn take(&mut self) -> u64 {
        let t = self.next;
        self.next = self.next.wrapping_add(1);
        t
    }
}

impl Default for Tokens {
    fn default() -> Self {
        Self::new()
    }
}

/// A packet the sender stopped trying to deliver
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeliveryFailure {
    /// Never acknowledged after [`MAX_RETRANSMITS`] retransmissions (or its link was never opened)
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
    /// An acknowledgement-only packet: the link, the newest acknowledged sequence number and the 32 before it
    Ack {
        epoch: Epoch,
        ack: u64,
        ack_bits: u32,
    },
    /// An open request with the sender's token
    Open { token: u64 },
    /// The answer to an open request: the link's epoch
    Accept { epoch: Epoch },
    /// The receiver holds no link of this epoch
    Reopen { epoch: Epoch },
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
    /// The link
    pub epoch: Epoch,
    /// Sequence number (sender's outgoing counter)
    pub seq: u64,
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
        buf[M + 1..M + 17].copy_from_slice(&self.epoch.to_bytes());
        buf[M + 17..M + 25].copy_from_slice(&self.seq.to_le_bytes());
        buf[M + 25..M + 27].copy_from_slice(&self.payload_len.to_le_bytes());
        buf
    }

    /// Read a data packet's header (`None` without the magic or the data kind)
    #[must_use]
    pub fn from_bytes(buf: &[u8; HEADER_SIZE]) -> Option<Self> {
        if buf[..M] != WIRE_MAGIC || buf[M] != KIND_DATA {
            return None;
        }
        Some(Self {
            epoch: Epoch::at(buf, M + 1),
            seq: u64_at(buf, M + 17),
            payload_len: u16::from_le_bytes([buf[M + 25], buf[M + 26]]),
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
    let n = packet.len();
    match packet[M] {
        KIND_DATA if n >= HEADER_SIZE => {
            let payload_len = u16::from_le_bytes([packet[M + 25], packet[M + 26]]);
            if n != HEADER_SIZE + usize::from(payload_len) {
                return Err(PacketError::Malformed);
            }
            Ok(Packet::Data(
                PacketHeader {
                    epoch: Epoch::at(packet, M + 1),
                    seq: u64_at(packet, M + 17),
                    payload_len,
                },
                &packet[HEADER_SIZE..],
            ))
        }
        KIND_ACK if n == ACK_SIZE => Ok(Packet::Ack {
            epoch: Epoch::at(packet, M + 1),
            ack: u64_at(packet, M + 17),
            ack_bits: u32_at(packet, M + 25),
        }),
        KIND_OPEN if n == OPEN_SIZE => Ok(Packet::Open {
            token: u64_at(packet, M + 1),
        }),
        KIND_ACCEPT if n == EPOCH_PACKET_SIZE => Ok(Packet::Accept {
            epoch: Epoch::at(packet, M + 1),
        }),
        KIND_REOPEN if n == EPOCH_PACKET_SIZE => Ok(Packet::Reopen {
            epoch: Epoch::at(packet, M + 1),
        }),
        _ => Err(PacketError::Malformed),
    }
}

/// The bytes of an acknowledgement-only packet of the link `epoch`
#[must_use]
pub fn ack_packet(epoch: Epoch, ack: u64, ack_bits: u32) -> [u8; ACK_SIZE] {
    let mut buf = [0u8; ACK_SIZE];
    buf[..M].copy_from_slice(&WIRE_MAGIC);
    buf[M] = KIND_ACK;
    buf[M + 1..M + 17].copy_from_slice(&epoch.to_bytes());
    buf[M + 17..M + 25].copy_from_slice(&ack.to_le_bytes());
    buf[M + 25..M + 29].copy_from_slice(&ack_bits.to_le_bytes());
    buf
}

/// The bytes of an open request with the sender's `token`
#[must_use]
pub fn open_packet(token: u64) -> [u8; OPEN_SIZE] {
    let mut buf = [0u8; OPEN_SIZE];
    buf[..M].copy_from_slice(&WIRE_MAGIC);
    buf[M] = KIND_OPEN;
    buf[M + 1..].copy_from_slice(&token.to_le_bytes());
    buf
}

fn epoch_packet(kind: u8, epoch: Epoch) -> [u8; EPOCH_PACKET_SIZE] {
    let mut buf = [0u8; EPOCH_PACKET_SIZE];
    buf[..M].copy_from_slice(&WIRE_MAGIC);
    buf[M] = kind;
    buf[M + 1..].copy_from_slice(&epoch.to_bytes());
    buf
}

/// The bytes of the answer to an open request: the link's `epoch`
#[must_use]
pub fn accept_packet(epoch: Epoch) -> [u8; EPOCH_PACKET_SIZE] {
    epoch_packet(KIND_ACCEPT, epoch)
}

/// The bytes of a reopen request: the receiver holds no link of `epoch`
#[must_use]
pub fn reopen_packet(epoch: Epoch) -> [u8; EPOCH_PACKET_SIZE] {
    epoch_packet(KIND_REOPEN, epoch)
}

/// Sent packet metadata for ACK tracking
#[derive(Debug)]
struct SentPacket {
    seq: u64,
    send_time: Instant,
    payload: Vec<u8>,
    acked: bool,
    retransmit_count: u8,
    /// whether it went out under the current epoch (a packet waiting for its link is not on the wire)
    on_wire: bool,
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
    #[cfg(test)]
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

/// What a received datagram gave: the payload of a data packet not seen before, and the packets to send
/// back (an accept, a reopen request, the data of a link just accepted)
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Incoming {
    /// The payload of a data packet not seen before on the current link
    pub payload: Option<Vec<u8>>,
    /// Packets to send to the peer now
    pub replies: Vec<Vec<u8>>,
}

/// The link the peer opened to this endpoint
#[derive(Debug)]
struct Inbound {
    epoch: Epoch,
    tracker: RecvTracker,
    /// Received sequence numbers to acknowledge by name in the next acknowledgement-only packets
    acks_due: std::collections::BTreeSet<u64>,
}

/// Reliability endpoint — one per peer address.
///
/// Handles links, sequencing, ACK tracking, retransmission, and RTT estimation. The tokens of the
/// transport the endpoint belongs to ([`Tokens`]) are passed to the calls that open or accept a link.
#[derive(Debug)]
pub struct ReliableEndpoint {
    /// The token of this endpoint's open request (the current link's, once accepted)
    token: Option<u64>,
    /// The link this endpoint sends on, once the peer accepted it
    epoch: Option<Epoch>,
    /// When the open request last went out and how many times it was sent again
    open_sent: Option<(Instant, u8)>,
    /// Outgoing sequence counter
    local_seq: u64,
    /// The next fragment message id
    next_message: u64,
    /// Sent packets awaiting ACK (and packets waiting for their link)
    sent_packets: VecDeque<SentPacket>,
    /// The peer's link to this endpoint
    inbound: Option<Inbound>,
    /// RTT estimator
    pub rtt: RttEstimator,
    /// Maximum unacked packets in flight
    max_in_flight: usize,
    /// Packet loss counter
    pub packets_lost: u64,
    /// Packet acked counter
    pub packets_acked: u64,
    /// Data packets of a link this endpoint does not hold (dropped, answered with a reopen request)
    pub stale_frames: u64,
    /// Acknowledgements and accepts of a link that is not this endpoint's current one (ignored)
    pub stale_acks: u64,
    /// Packets given up or evicted, not yet taken
    failures: Vec<DeliveryFailure>,
}

impl ReliableEndpoint {
    /// Create a new reliability endpoint
    #[must_use]
    pub fn new() -> Self {
        Self {
            token: None,
            epoch: None,
            open_sent: None,
            local_seq: 0,
            next_message: 0,
            sent_packets: VecDeque::with_capacity(256),
            inbound: None,
            rtt: RttEstimator::new(),
            max_in_flight: 256,
            packets_lost: 0,
            packets_acked: 0,
            stale_frames: 0,
            stale_acks: 0,
            failures: Vec::new(),
        }
    }

    /// The sequence number the next data packet takes
    #[must_use]
    pub fn local_seq(&self) -> u64 {
        self.local_seq
    }

    /// The link this endpoint sends on, once the peer accepted it
    #[must_use]
    pub fn epoch(&self) -> Option<Epoch> {
        self.epoch
    }

    /// The link the peer opened to this endpoint
    #[must_use]
    pub fn inbound_epoch(&self) -> Option<Epoch> {
        self.inbound.as_ref().map(|i| i.epoch)
    }

    /// A fresh message id for the fragments of one message
    pub fn next_message_id(&mut self) -> u64 {
        let id = self.next_message;
        self.next_message += 1;
        id
    }

    fn data_packet(epoch: Epoch, seq: u64, payload: &[u8]) -> Vec<u8> {
        let header = PacketHeader {
            epoch,
            seq,
            payload_len: payload.len() as u16,
        };
        let mut packet = Vec::with_capacity(HEADER_SIZE + payload.len());
        packet.extend_from_slice(&header.to_bytes());
        packet.extend_from_slice(payload);
        packet
    }

    /// A new open request (a fresh token): the endpoint has no link to send on
    fn open(&mut self, tokens: &mut Tokens) -> Vec<u8> {
        let token = tokens.take();
        self.token = Some(token);
        self.epoch = None;
        self.open_sent = Some((crate::clock::now(), 0));
        open_packet(token).to_vec()
    }

    /// Take a payload for the peer: the packets to send now, a data packet on an accepted link, an open
    /// request when the endpoint has none (the payload goes out once the peer accepts), or nothing while an
    /// open request is in flight.
    pub fn wrap_outgoing(&mut self, payload: &[u8], tokens: &mut Tokens) -> Vec<Vec<u8>> {
        let seq = self.local_seq;
        self.local_seq += 1;
        let now = crate::clock::now();
        self.sent_packets.push_back(SentPacket {
            seq,
            send_time: now,
            payload: payload.to_vec(),
            acked: false,
            retransmit_count: 0,
            on_wire: self.epoch.is_some(),
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

        match self.epoch {
            Some(epoch) => vec![Self::data_packet(epoch, seq, payload)],
            None if self.open_sent.is_none() => vec![self.open(tokens)],
            None => Vec::new(),
        }
    }

    /// Process a received datagram (see [`Incoming`]). A datagram that is not a packet of this wire version
    /// gives nothing (use [`classify`] to tell those apart).
    pub fn unwrap_incoming(&mut self, packet: &[u8], tokens: &mut Tokens) -> Incoming {
        let mut out = Incoming::default();
        let Ok(packet) = classify(packet) else {
            return out;
        };
        match packet {
            Packet::Data(header, payload) => match self.inbound.as_mut() {
                Some(inb) if inb.epoch == header.epoch => {
                    let fresh = inb.tracker.record(header.seq);
                    // a repeat is acknowledged again too: the sender did not see the earlier acknowledgement
                    if fresh || inb.tracker.in_window(header.seq) {
                        inb.acks_due.insert(header.seq);
                    }
                    out.payload = fresh.then(|| payload.to_vec());
                }
                _ => {
                    self.stale_frames += 1;
                    out.replies.push(reopen_packet(header.epoch).to_vec());
                }
            },
            Packet::Ack {
                epoch,
                ack,
                ack_bits,
            } => {
                if self.epoch == Some(epoch) {
                    self.process_acks(ack, ack_bits);
                } else {
                    self.stale_acks += 1;
                }
            }
            Packet::Open { token } => {
                let epoch = match &self.inbound {
                    // a repeated request (its accept was lost, so no data came on the link): the same link.
                    // The same token after data came is a sender that started again: a new link
                    Some(inb) if inb.epoch.sender == token && !inb.tracker.initialized => inb.epoch,
                    _ => {
                        let epoch = Epoch {
                            receiver: tokens.take(),
                            sender: token,
                        };
                        self.inbound = Some(Inbound {
                            epoch,
                            tracker: RecvTracker::new(),
                            acks_due: std::collections::BTreeSet::new(),
                        });
                        epoch
                    }
                };
                out.replies.push(accept_packet(epoch).to_vec());
            }
            Packet::Accept { epoch } => {
                if self.epoch.is_none() && self.token == Some(epoch.sender) {
                    self.epoch = Some(epoch);
                    self.open_sent = None;
                    let now = crate::clock::now();
                    for sent in &mut self.sent_packets {
                        if !sent.acked && !sent.on_wire {
                            sent.on_wire = true;
                            sent.send_time = now;
                            out.replies
                                .push(Self::data_packet(epoch, sent.seq, &sent.payload));
                        }
                    }
                } else if self.epoch != Some(epoch) {
                    self.stale_acks += 1;
                }
            }
            Packet::Reopen { epoch } => {
                if self.epoch == Some(epoch) {
                    self.epoch = None;
                    self.open_sent = None;
                    // what was not acknowledged goes out again on the new link; each reopen counts as one
                    // more try, so a peer that keeps refusing the link ends in a reported failure
                    let mut gave_up = Vec::new();
                    for sent in &mut self.sent_packets {
                        if sent.acked {
                            continue;
                        }
                        sent.on_wire = false;
                        sent.retransmit_count += 1;
                        if sent.retransmit_count > MAX_RETRANSMITS {
                            gave_up.push(sent.seq);
                        }
                    }
                    self.give_up(&gave_up);
                    if self.sent_packets.iter().any(|p| !p.acked) {
                        out.replies.push(self.open(tokens));
                    } else {
                        self.token = None;
                    }
                } else {
                    self.stale_acks += 1;
                }
            }
        }
        out
    }

    fn give_up(&mut self, seqs: &[u64]) {
        if seqs.is_empty() {
            return;
        }
        self.sent_packets.retain(|p| !seqs.contains(&p.seq));
        for &seq in seqs {
            self.packets_lost += 1;
            self.failures.push(DeliveryFailure::GaveUp { seq });
        }
    }

    /// Process ACK information from a received packet
    fn process_acks(&mut self, ack: u64, ack_bits: u32) {
        let now = crate::clock::now();

        for sent in &mut self.sent_packets {
            if sent.acked || !sent.on_wire || !acknowledges(ack, ack_bits, sent.seq) {
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

    /// Collect packets that need retransmission (timed out, not yet acked), and the open request when it
    /// has not been answered in time. A packet (or an open request) sent [`MAX_RETRANSMITS`] times more
    /// without an answer is dropped and reported ([`Self::take_failures`]).
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
        let Some(epoch) = self.epoch else {
            // no link yet: the open request is what is sent again
            if let (Some(token), Some((at, tries))) = (self.token, self.open_sent) {
                if all || now.duration_since(at) >= rto {
                    if tries >= MAX_RETRANSMITS {
                        let waiting: Vec<u64> = self
                            .sent_packets
                            .iter()
                            .filter(|p| !p.acked)
                            .map(|p| p.seq)
                            .collect();
                        self.give_up(&waiting);
                        self.token = None;
                        self.open_sent = None;
                    } else {
                        self.open_sent = Some((now, tries + 1));
                        retransmits.push(open_packet(token).to_vec());
                    }
                }
            }
            return retransmits;
        };
        let mut gave_up = Vec::new();

        for sent in &mut self.sent_packets {
            if sent.acked || !sent.on_wire {
                continue;
            }
            if all || now.duration_since(sent.send_time) >= rto {
                sent.send_time = now; // reset timer
                sent.retransmit_count += 1;

                if sent.retransmit_count > MAX_RETRANSMITS {
                    gave_up.push(sent.seq);
                    continue;
                }
                retransmits.push(Self::data_packet(epoch, sent.seq, &sent.payload));
            }
        }

        self.give_up(&gave_up);
        retransmits
    }

    /// Acknowledgement-only packets for what was received since the last call: one with the newest
    /// received sequence number, and one per packet received far behind it (each names its own sequence
    /// number). They are not retransmitted
    pub fn take_acks(&mut self) -> Vec<Vec<u8>> {
        let Some(inb) = self.inbound.as_mut() else {
            return Vec::new();
        };
        let due = std::mem::take(&mut inb.acks_due);
        if !inb.tracker.initialized {
            return Vec::new();
        }
        let newest = inb.tracker.most_recent;
        let near = |r: &u64| *r <= newest && newest - r <= 32;
        let mut refs: Vec<u64> = Vec::new();
        if due.iter().any(near) {
            refs.push(newest);
        }
        refs.extend(due.iter().copied().filter(|r| !near(r)));
        refs.into_iter()
            .map(|reference| {
                let (ack, ack_bits) = inb.tracker.ack_for(reference);
                ack_packet(inb.epoch, ack, ack_bits).to_vec()
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

    /// One side of a link: an endpoint and the tokens of its transport
    struct Side {
        ep: ReliableEndpoint,
        tokens: Tokens,
    }

    impl Side {
        fn new(start: u64) -> Self {
            Self {
                ep: ReliableEndpoint::new(),
                tokens: Tokens::starting_at(start),
            }
        }

        fn take(&mut self, p: &[u8]) -> Incoming {
            self.ep.unwrap_incoming(p, &mut self.tokens)
        }
    }

    /// Carry `packets` from `from` to `to` and every reply back and forth until none is left; the payloads
    /// delivered to each side
    fn settle(
        from: &mut Side,
        to: &mut Side,
        packets: Vec<Vec<u8>>,
    ) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
        let (mut at_to, mut at_from) = (Vec::new(), Vec::new());
        let mut towards_to = packets;
        let mut towards_from: Vec<Vec<u8>> = Vec::new();
        while !towards_to.is_empty() || !towards_from.is_empty() {
            for p in std::mem::take(&mut towards_to) {
                let r = to.take(&p);
                at_to.extend(r.payload);
                towards_from.extend(r.replies);
            }
            for p in std::mem::take(&mut towards_from) {
                let r = from.take(&p);
                at_from.extend(r.payload);
                towards_to.extend(r.replies);
            }
        }
        (at_to, at_from)
    }

    fn send(from: &mut Side, to: &mut Side, payload: &[u8]) -> Vec<Vec<u8>> {
        let p = from.ep.wrap_outgoing(payload, &mut from.tokens);
        settle(from, to, p).0
    }

    fn ack(from: &mut Side, to: &mut Side) {
        let acks = to.ep.take_acks();
        settle(to, from, acks);
    }

    #[test]
    fn test_packet_header_roundtrip() {
        let header = PacketHeader {
            epoch: Epoch {
                receiver: 7,
                sender: 0x0123_4567_89AB_CDEF,
            },
            seq: 0x0123_4567_89AB_CDEF,
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
        let e = Epoch {
            receiver: 1,
            sender: 2,
        };
        let old = [1u8, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0x55];
        assert_eq!(classify(&old), Err(PacketError::OtherWire));
        assert_eq!(classify(b""), Err(PacketError::OtherWire));
        let mut other_version = ack_packet(e, 3, 0);
        other_version[3] = 1;
        assert_eq!(classify(&other_version), Err(PacketError::OtherWire));
        let mut unknown_kind = ack_packet(e, 3, 0);
        unknown_kind[M] = 9;
        assert_eq!(classify(&unknown_kind), Err(PacketError::Malformed));
        for p in [
            ack_packet(e, 3, 0).to_vec(),
            open_packet(4).to_vec(),
            accept_packet(e).to_vec(),
            reopen_packet(e).to_vec(),
        ] {
            assert!(classify(&p).is_ok());
            assert_eq!(classify(&p[..p.len() - 1]), Err(PacketError::Malformed));
            let longer = [&p[..], &[0u8]].concat();
            assert_eq!(
                classify(&longer),
                Err(PacketError::Malformed),
                "a trailing byte"
            );
        }
    }

    #[test]
    fn test_the_kinds_round_trip() {
        let e = Epoch {
            receiver: 0x1_0000_0000,
            sender: u64::MAX,
        };
        assert_eq!(
            classify(&ack_packet(e, 5, 0b101)),
            Ok(Packet::Ack {
                epoch: e,
                ack: 5,
                ack_bits: 0b101
            })
        );
        assert_eq!(classify(&open_packet(9)), Ok(Packet::Open { token: 9 }));
        assert_eq!(classify(&accept_packet(e)), Ok(Packet::Accept { epoch: e }));
        assert_eq!(classify(&reopen_packet(e)), Ok(Packet::Reopen { epoch: e }));
    }

    #[test]
    fn test_data_carries_its_payload() {
        let e = Epoch {
            receiver: 1,
            sender: 2,
        };
        let p = ReliableEndpoint::data_packet(e, 0, b"abc");
        assert!(matches!(classify(&p), Ok(Packet::Data(h, b"abc")) if h.epoch == e));
        assert_eq!(classify(&p[..p.len() - 1]), Err(PacketError::Malformed));
        let longer = [&p[..], &[0u8]].concat();
        assert_eq!(
            classify(&longer),
            Err(PacketError::Malformed),
            "a trailing byte"
        );
        let empty = ReliableEndpoint::data_packet(e, 0, b"");
        assert!(matches!(classify(&empty), Ok(Packet::Data(_, b""))));
    }

    /// The first payload opens the link: open, accept, then the data, which arrives once
    #[test]
    fn test_the_first_payload_opens_the_link() {
        let (mut a, mut b) = (Side::new(0), Side::new(100));
        let first = a.ep.wrap_outgoing(b"hello", &mut a.tokens);
        assert_eq!(first.len(), 1);
        assert_eq!(classify(&first[0]), Ok(Packet::Open { token: 0 }));
        // a second payload waits for the same link
        assert!(a.ep.wrap_outgoing(b"again", &mut a.tokens).is_empty());
        let (at_b, _) = settle(&mut a, &mut b, first);
        assert_eq!(at_b, vec![b"hello".to_vec(), b"again".to_vec()]);
        let e = Epoch {
            receiver: 100,
            sender: 0,
        };
        assert_eq!((a.ep.epoch(), b.ep.inbound_epoch()), (Some(e), Some(e)));
        // now data goes out directly
        let next = a.ep.wrap_outgoing(b"third", &mut a.tokens);
        assert!(matches!(classify(&next[0]), Ok(Packet::Data(h, _)) if h.epoch == e));
        ack(&mut a, &mut b);
        assert_eq!(a.ep.packets_acked, 2);
    }

    /// A lost accept: the open request is sent again, and the repeat gets the same link
    #[test]
    fn test_a_lost_accept_gets_the_same_link() {
        let (mut a, mut b) = (Side::new(0), Side::new(100));
        let open = a.ep.wrap_outgoing(b"x", &mut a.tokens);
        let lost = b.take(&open[0]);
        let again = a.ep.collect_retransmits_now();
        assert_eq!(again, open);
        let r = b.take(&again[0]);
        assert_eq!(r.replies, lost.replies, "the same accept");
        let (at_b, _) = settle(&mut b, &mut a, r.replies);
        assert!(at_b.is_empty());
        assert_eq!(b.ep.inbound_epoch(), a.ep.epoch());
    }

    #[test]
    fn test_duplicate_rejection() {
        let (mut a, mut b) = (Side::new(0), Side::new(100));
        assert_eq!(send(&mut a, &mut b, b"once"), vec![b"once".to_vec()]);
        let again = a.ep.collect_retransmits_now();
        assert_eq!(again.len(), 1);
        assert_eq!(b.take(&again[0]).payload, None);
    }

    /// The receiver dropped its link (the transport's peer limit) and the sender keeps its own: the next data
    /// is refused with a reopen request, the sender opens a new link and the payload arrives, not acked as a
    /// repeat of the old link's sequence numbers
    #[test]
    fn test_a_receiver_that_dropped_the_link_gets_the_data() {
        let (mut a, mut b) = (Side::new(0), Side::new(100));
        for i in 0..10u8 {
            assert_eq!(send(&mut a, &mut b, &[i]), vec![vec![i]]);
        }
        ack(&mut a, &mut b);
        // the receiver's state for a is gone; its tokens move on
        b.ep = ReliableEndpoint::new();
        assert_eq!(
            send(&mut a, &mut b, b"new message"),
            vec![b"new message".to_vec()]
        );
        assert_eq!(b.ep.stale_frames, 1);
        assert_ne!(a.ep.epoch().unwrap().receiver, 100);
        ack(&mut a, &mut b);
        assert_eq!(a.ep.packets_acked, 11);
        assert!(a.ep.take_failures().is_empty());
    }

    /// The sender dropped its link and starts again (or its process restarted): the receiver gives the new
    /// open request a new link, and the first new data (sequence number 0 again) arrives
    #[test]
    fn test_a_sender_that_started_again_gets_its_data_through() {
        let (mut a, mut b) = (Side::new(0), Side::new(100));
        for i in 0..10u8 {
            send(&mut a, &mut b, &[i]);
        }
        ack(&mut a, &mut b);
        // a restarted process: a new endpoint and tokens from the start again
        a = Side::new(0);
        assert_eq!(
            send(&mut a, &mut b, b"after restart"),
            vec![b"after restart".to_vec()]
        );
        ack(&mut a, &mut b);
        assert_eq!(a.ep.packets_acked, 1);
    }

    /// A restarted receiver (no state, tokens from the start): a delayed data packet of the old link is
    /// refused, the reopen request makes the sender open a new link (its token moved on, so the epoch
    /// differs from the old one although the receiver's token is the same again) and the payload arrives
    #[test]
    fn test_a_restarted_receiver_refuses_a_delayed_old_frame() {
        let (mut a, mut b) = (Side::new(0), Side::new(100));
        send(&mut a, &mut b, b"before");
        ack(&mut a, &mut b);
        let old = a.ep.epoch().unwrap();
        let delayed = a.ep.wrap_outgoing(b"in flight", &mut a.tokens);
        b = Side::new(100);
        let r = b.take(&delayed[0]);
        assert_eq!(r.payload, None, "the old link's frame is not delivered");
        assert_eq!(b.ep.stale_frames, 1);
        let (_, at_b) = settle(&mut b, &mut a, r.replies);
        assert_eq!(at_b, vec![b"in flight".to_vec()]);
        let new = a.ep.epoch().unwrap();
        assert_eq!(
            new.receiver, old.receiver,
            "the restarted receiver's tokens start the same"
        );
        assert_ne!(new, old);
        // the old frame again (delayed twice) is still refused
        assert_eq!(b.take(&delayed[0]).payload, None);
    }

    /// A restarted sender: a delayed data packet of the sender's old link reaching the receiver after the
    /// new link was accepted is refused (its epoch is not the receiver's current one)
    #[test]
    fn test_a_delayed_frame_from_before_a_sender_restart_is_refused() {
        let (mut a, mut b) = (Side::new(0), Side::new(100));
        send(&mut a, &mut b, b"before");
        let delayed = a.ep.wrap_outgoing(b"old", &mut a.tokens);
        a = Side::new(0);
        assert_eq!(send(&mut a, &mut b, b"new"), vec![b"new".to_vec()]);
        let r = b.take(&delayed[0]);
        assert_eq!(r.payload, None);
        assert_eq!(b.ep.stale_frames, 1);
    }

    /// Both ends restart, the sender with tokens past its previous incarnation's: a delayed data packet of
    /// the old link has the same receiver token as the new link but not the same sender token, and is refused
    #[test]
    fn test_both_restarted_the_sender_token_keeps_the_links_apart() {
        let (mut a, mut b) = (Side::new(0), Side::new(100));
        for i in 0..6u8 {
            send(&mut a, &mut b, &[i]);
        }
        let delayed = a.ep.wrap_outgoing(b"old", &mut a.tokens); // seq 6 of the old link
        let (mut a, mut b) = (Side::new(1 << 32), Side::new(100));
        assert_eq!(send(&mut a, &mut b, b"new"), vec![b"new".to_vec()]);
        let (old, new) = (Epoch::at(&delayed[0], M + 1), a.ep.epoch().unwrap());
        assert_eq!(old.receiver, new.receiver);
        assert_ne!(old.sender, new.sender);
        assert_eq!(
            b.take(&delayed[0]).payload,
            None,
            "the old link's packet is refused"
        );
        assert_eq!(b.ep.stale_frames, 1);
    }

    /// An acknowledgement of another link (a stale one) is ignored and counted
    #[test]
    fn test_a_stale_ack_is_ignored() {
        let (mut a, mut b) = (Side::new(0), Side::new(100));
        send(&mut a, &mut b, b"x");
        let e = a.ep.epoch().unwrap();
        let stale = Epoch {
            receiver: e.receiver + 1,
            ..e
        };
        a.take(&ack_packet(stale, 0, 0));
        assert_eq!((a.ep.packets_acked, a.ep.stale_acks), (0, 1));
        a.take(&ack_packet(e, 0, 0));
        assert_eq!(a.ep.packets_acked, 1);
    }

    /// A reopen request for a link that is not the sender's current one is ignored
    #[test]
    fn test_a_stale_reopen_is_ignored() {
        let (mut a, mut b) = (Side::new(0), Side::new(100));
        send(&mut a, &mut b, b"x");
        let e = a.ep.epoch().unwrap();
        let r = a.take(&reopen_packet(Epoch { sender: 99, ..e }));
        assert!(r.replies.is_empty());
        assert_eq!(a.ep.epoch(), Some(e));
    }

    /// An open request never answered: after MAX_RETRANSMITS more tries every waiting payload is given up
    #[test]
    fn test_an_open_never_answered_gives_up() {
        let mut a = Side::new(0);
        a.ep.wrap_outgoing(b"one", &mut a.tokens);
        a.ep.wrap_outgoing(b"two", &mut a.tokens);
        let mut sent = 0;
        for _ in 0..20 {
            sent += a.ep.collect_retransmits_now().len();
        }
        assert_eq!(sent, usize::from(MAX_RETRANSMITS));
        assert_eq!(
            a.ep.take_failures(),
            vec![
                DeliveryFailure::GaveUp { seq: 0 },
                DeliveryFailure::GaveUp { seq: 1 }
            ]
        );
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
            let (mut a, mut b) = (Side::new(0), Side::new(100));
            a.ep.local_seq = base;
            for i in 0..4u8 {
                assert_eq!(send(&mut a, &mut b, &[i]), vec![vec![i]]);
            }
            let again = a.ep.collect_retransmits_now();
            assert!(again.iter().all(|p| b.take(p).payload.is_none()));
            ack(&mut a, &mut b);
            assert_eq!(a.ep.packets_acked, 4);
            assert!(a.ep.sent_packets.is_empty());
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

    /// A repeat is acknowledged again: the sender did not see the first acknowledgement
    #[test]
    fn test_a_repeat_is_acknowledged_again() {
        let (mut a, mut b) = (Side::new(0), Side::new(100));
        send(&mut a, &mut b, b"x");
        assert_eq!(b.ep.take_acks().len(), 1);
        let again = a.ep.collect_retransmits_now();
        assert_eq!(b.take(&again[0]).payload, None);
        let acks = b.ep.take_acks();
        assert_eq!(acks.len(), 1);
        a.take(&acks[0]);
        assert_eq!(a.ep.packets_acked, 1);
    }

    /// A packet is sent [`MAX_RETRANSMITS`] more times, then given up
    #[test]
    fn test_retransmits_then_gives_up() {
        let (mut a, mut b) = (Side::new(0), Side::new(100));
        send(&mut a, &mut b, b"x");
        b.ep = ReliableEndpoint::new();
        let _ = b;
        a.ep.wrap_outgoing(b"void", &mut a.tokens);
        let mut sent = 0;
        for _ in 0..20 {
            sent += a.ep.collect_retransmits_now().len();
        }
        // the first packet and "void" are each sent MAX_RETRANSMITS more times
        assert_eq!(sent, 2 * usize::from(MAX_RETRANSMITS));
        assert_eq!(
            a.ep.take_failures(),
            vec![
                DeliveryFailure::GaveUp { seq: 0 },
                DeliveryFailure::GaveUp { seq: 1 }
            ]
        );
    }

    #[test]
    fn test_loss_rate() {
        let endpoint = ReliableEndpoint::new();
        assert!(endpoint.loss_rate().abs() < f64::EPSILON);
    }
}
