//! The snapshot decoded as its bytes arrive (see [the parent module](super), "Reading").
//!
//! This is the only decoder: [`AnonymousRelayCore::restore`] feeds it the whole slice, and
//! [`AnonymousRelayCore::restore_from_pieces`] feeds it each piece, so every read makes the same
//! checks in the same order. It is a state machine over the fields of the format: each step names
//! the next field ([`At`]) and its size; the bytes of a field that a piece cuts are kept until the
//! field is complete, then the field is checked and the step after it chosen. A check runs as soon
//! as the last byte of the fields it needs is there, which is where the one-slice read of earlier
//! versions ran it, so an error comes out of the [`SnapshotReader::feed`] that hands over that byte.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Arc;

use super::super::{
    AnonymousRelayCore, ConnId, Limits, Logged, Payload, Room, RoomAlias, RoomToken, SenderTag,
};
use super::{SnapshotError, SNAPSHOT_MAGIC, SNAPSHOT_VERSION};

/// The longest run of fixed-size fields read as one step (a room's token, last activity and next
/// index)
const FIELD: usize = 48;

/// Where the reader is: a run of fixed-size fields, a payload, or past the end
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum At {
    Field(Step),
    /// The payload of a held-back entry or of a logged one, whose length was read before it
    Payload(Entry),
    /// The snapshot is complete: any further byte is trailing
    End,
}

/// A step of fixed-size fields. Fields with no check between them are read as one step; a field a
/// check needs on its own (so that it is made before the bytes after it are read) is a step of its
/// own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Step {
    Magic,
    Version,
    /// The `i`-th limit, in the order of the format
    Limit(u8),
    RoomCount,
    Alias,
    /// Token, last activity, next index
    RoomHead,
    MemberCount,
    /// Connection and tag
    Member,
    NextSeqCount,
    /// Tag and sequence number
    NextSeq,
    HeldTagCount,
    HeldTag,
    HeldCount,
    /// Sequence number and payload length
    HeldEntry,
    RateCount,
    /// Tag, window start, publishes
    Rate,
    LogCount,
    /// Index, tag, sequence number and payload length
    LogEntry,
}

/// Whose payload is being read
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Entry {
    Held,
    Log,
}

impl Step {
    /// The bytes of the step
    const fn size(self) -> usize {
        match self {
            Self::Magic | Self::RoomCount => 4,
            Self::Version => 2,
            // max_publishes_per_window is the one u32 among the limits
            Self::Limit(5) => 4,
            Self::Limit(_) => 8,
            Self::Alias => 32,
            Self::RoomHead => 48,
            Self::MemberCount
            | Self::NextSeqCount
            | Self::HeldTagCount
            | Self::HeldCount
            | Self::RateCount
            | Self::LogCount => 4,
            Self::Member | Self::NextSeq => 24,
            Self::HeldTag => 16,
            Self::HeldEntry => 12,
            Self::Rate => 28,
            Self::LogEntry => 36,
        }
    }
}

/// The room being read and the bookkeeping its checks use
#[derive(Debug)]
struct RoomInProgress {
    alias: RoomAlias,
    token: RoomToken,
    last_active: u64,
    next_index: u64,
    /// The tags the room keeps state for, added to as entries are read
    tags: BTreeSet<SenderTag>,
    members: BTreeMap<ConnId, SenderTag>,
    member_tags: BTreeSet<SenderTag>,
    next_seq: BTreeMap<SenderTag, u64>,
    held: BTreeMap<SenderTag, BTreeMap<u64, Payload>>,
    rate: BTreeMap<SenderTag, (u64, u32)>,
    log: VecDeque<Logged>,
    /// Per tag, the sequence number of its latest logged payload
    last_logged: BTreeMap<SenderTag, u64>,
    /// Entries left in the list being read (for the held-back payloads: the tags left)
    left: u32,
    /// The held-back list being read: its tag, that tag's next number, its payloads, entries left
    held_tag: SenderTag,
    held_next: u64,
    held_payloads: BTreeMap<u64, Payload>,
    held_left: u32,
    /// The log: the index of its first entry, the entries it has, the entries read
    first_index: u64,
    should_keep: u64,
    logged: u64,
    /// The fixed fields of the entry whose payload is being read
    entry_seq: u64,
    entry_index: u64,
    entry_from: SenderTag,
}

impl RoomInProgress {
    fn new(alias: RoomAlias) -> Self {
        Self {
            alias,
            token: RoomToken::from_raw([0; 32]),
            last_active: 0,
            next_index: 0,
            tags: BTreeSet::new(),
            members: BTreeMap::new(),
            member_tags: BTreeSet::new(),
            next_seq: BTreeMap::new(),
            held: BTreeMap::new(),
            rate: BTreeMap::new(),
            log: VecDeque::new(),
            last_logged: BTreeMap::new(),
            left: 0,
            held_tag: SenderTag::from_raw([0; 16]),
            held_next: 0,
            held_payloads: BTreeMap::new(),
            held_left: 0,
            first_index: 0,
            should_keep: 0,
            logged: 0,
            entry_seq: 0,
            entry_index: 0,
            entry_from: SenderTag::from_raw([0; 16]),
        }
    }

    /// One more tag the room keeps state for, refused past `max_tags_per_room`
    fn add_tag(&mut self, tag: SenderTag, limits: &Limits) -> Result<(), SnapshotError> {
        self.tags.insert(tag);
        if self.tags.len() > limits.max_tags_per_room {
            return Err(SnapshotError::Inconsistent);
        }
        Ok(())
    }
}

/// Reads a snapshot from pieces handed to it one at a time, for a host that cannot pull the pieces
/// from an iterator (its storage reads are asynchronous: it reads a piece, hands it over with
/// [`feed`](Self::feed), waits for the next read, and so on), then [`finish`](Self::finish) gives the
/// core.
///
/// Fed the bytes of a snapshot in pieces cut anywhere (empty pieces included), it gives the core or
/// the [`SnapshotError`] that [`AnonymousRelayCore::restore`] (made with [`new`](Self::new)) or
/// [`AnonymousRelayCore::restore_with_limits`] (made with [`with_limits`](Self::with_limits)) gives
/// for the pieces concatenated: it is the decoder those use.
///
/// # When an error comes out
///
/// Every check is made as soon as the bytes it needs are there, so an error comes out of the
/// [`feed`](Self::feed) that hands over the last byte of the field that fails it, before more bytes
/// are read: a limit exceeded in the head, a count over its limit, a payload length over
/// `max_payload` (before the payload's bytes), a key out of order, an unreachable state, and
/// [`SnapshotError::Trailing`] for the first byte after the snapshot (in the same piece as its end or
/// a later one). [`SnapshotError::Truncated`] comes only from [`finish`](Self::finish), when the bytes
/// fed end inside the snapshot (no bytes at all included).
///
/// After an error the reader is spent: every later [`feed`](Self::feed), with bytes or without, returns
/// the same error, and [`finish`](Self::finish) returns it.
///
/// # Memory
///
/// The reader does not keep the pieces it is fed. Besides the core it is building (the rooms read so
/// far, which become the core), it holds the bytes of one field a piece cuts: at most 48 bytes of
/// fixed-size fields, or one payload, which grows as its bytes arrive and never past its length (at
/// most `max_payload`), then becomes part of the core. What it reserves for a payload is at most twice
/// the bytes of it that have arrived, never more than its length. Its other bookkeeping is the per-connection room count across rooms and, for the
/// room being read, the sets of tags its checks use, all bounded by the limits.
#[derive(Debug)]
pub struct SnapshotReader {
    /// The limits the core runs under, `None` for the stored ones
    run_under: Option<Limits>,
    at: At,
    /// The bytes of the current fixed-size step received so far
    field: [u8; FIELD],
    field_len: usize,
    /// The current payload's length and the bytes of it received so far
    payload_len: usize,
    payload: Vec<u8>,
    /// The stored limits, filled as they are read
    stored: Limits,
    /// The limits the state is checked against and the core runs under, once the head is read
    limits: Limits,
    rooms: BTreeMap<RoomAlias, Arc<Room>>,
    rooms_of_conn: BTreeMap<ConnId, usize>,
    rooms_left: u32,
    /// The room being read (the one whose alias was read last; before the first, an empty one)
    room: RoomInProgress,
    failed: Option<SnapshotError>,
}

impl Default for SnapshotReader {
    fn default() -> Self {
        Self::new()
    }
}

const NO_LIMITS: Limits = Limits {
    max_payload: 0,
    max_log: 0,
    max_pending: 0,
    max_rooms: 0,
    max_members: 0,
    max_publishes_per_window: 0,
    window: 0,
    max_rooms_per_conn: 0,
    max_tags_per_room: 0,
};

fn u32_at(b: &[u8], at: usize) -> u32 {
    let mut a = [0u8; 4];
    a.copy_from_slice(&b[at..at + 4]);
    u32::from_be_bytes(a)
}

fn u64_at(b: &[u8], at: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[at..at + 8]);
    u64::from_be_bytes(a)
}

fn bytes_at<const N: usize>(b: &[u8], at: usize) -> [u8; N] {
    let mut a = [0u8; N];
    a.copy_from_slice(&b[at..at + N]);
    a
}

/// A count, refused when it is over `limit`
fn count(n: u32, limit: usize) -> Result<u32, SnapshotError> {
    if usize::try_from(n).map_or(true, |n| n > limit) {
        return Err(SnapshotError::Inconsistent);
    }
    Ok(n)
}

/// A limit as this platform's `usize`
fn limit(v: u64) -> Result<usize, SnapshotError> {
    usize::try_from(v).map_err(|_| SnapshotError::LimitTooLarge)
}

impl SnapshotReader {
    /// A reader that runs the core under the limits stored in the snapshot, as
    /// [`AnonymousRelayCore::restore`] does.
    #[must_use]
    pub fn new() -> Self {
        Self::under(None)
    }

    /// A reader that checks the state against `limits` and runs the core under them instead of the
    /// limits stored in the snapshot, as [`AnonymousRelayCore::restore_with_limits`] does.
    #[must_use]
    pub fn with_limits(limits: Limits) -> Self {
        Self::under(Some(limits))
    }

    fn under(run_under: Option<Limits>) -> Self {
        Self {
            run_under,
            at: At::Field(Step::Magic),
            field: [0; FIELD],
            field_len: 0,
            payload_len: 0,
            payload: Vec::new(),
            stored: NO_LIMITS,
            limits: NO_LIMITS,
            rooms: BTreeMap::new(),
            rooms_of_conn: BTreeMap::new(),
            rooms_left: 0,
            room: RoomInProgress::new(RoomAlias::from_raw([0; 32])),
            failed: None,
        }
    }

    /// Read the next piece of the snapshot (any length, empty included). The piece is not kept: only
    /// the bytes of a field it cuts are, until the next piece completes the field.
    ///
    /// # Errors
    ///
    /// A [`SnapshotError`] as soon as the bytes fed so far fail a check (see
    /// [when an error comes out](Self#when-an-error-comes-out)); never
    /// [`SnapshotError::Truncated`]. After an error, the same error, whatever is fed.
    pub fn feed(&mut self, piece: &[u8]) -> Result<(), SnapshotError> {
        if let Some(e) = self.failed {
            return Err(e);
        }
        let r = self.consume(piece);
        if let Err(e) = r {
            self.failed = Some(e);
            // what was being built is of no further use
            self.rooms = BTreeMap::new();
            self.rooms_of_conn = BTreeMap::new();
            self.room = RoomInProgress::new(RoomAlias::from_raw([0; 32]));
            self.payload = Vec::new();
        }
        r
    }

    /// The core the bytes fed hold.
    ///
    /// # Errors
    ///
    /// The error a [`feed`](Self::feed) returned, or [`SnapshotError::Truncated`] when the bytes fed
    /// end inside the snapshot.
    pub fn finish(self) -> Result<AnonymousRelayCore, SnapshotError> {
        if let Some(e) = self.failed {
            return Err(e);
        }
        if self.at != At::End {
            return Err(SnapshotError::Truncated);
        }
        Ok(AnonymousRelayCore {
            limits: self.limits,
            rooms: self.rooms,
        })
    }

    fn consume(&mut self, mut input: &[u8]) -> Result<(), SnapshotError> {
        while !input.is_empty() {
            match self.at {
                At::Field(step) => {
                    let n = step.size();
                    if self.field_len == 0 && input.len() >= n {
                        // the whole step is in this piece: read it in place
                        let (f, rest) = input.split_at(n);
                        input = rest;
                        self.on_field(step, f)?;
                    } else {
                        let k = (n - self.field_len).min(input.len());
                        self.field[self.field_len..self.field_len + k].copy_from_slice(&input[..k]);
                        self.field_len += k;
                        input = &input[k..];
                        if self.field_len == n {
                            self.field_len = 0;
                            let f = self.field;
                            self.on_field(step, &f[..n])?;
                        }
                    }
                }
                At::Payload(entry) => {
                    let k = (self.payload_len - self.payload.len()).min(input.len());
                    self.grow_payload(k);
                    self.payload.extend_from_slice(&input[..k]);
                    input = &input[k..];
                    if self.payload.len() == self.payload_len {
                        self.on_payload(entry)?;
                    }
                }
                At::End => return Err(SnapshotError::Trailing),
            }
        }
        Ok(())
    }

    /// Room for `k` more payload bytes that are here: the capacity at least doubles (so a payload
    /// fed a byte at a time is copied a bounded number of times) but never passes the payload's
    /// length, so what is reserved is at most twice the bytes that have arrived
    fn grow_payload(&mut self, k: usize) {
        let need = self.payload.len() + k;
        if need > self.payload.capacity() {
            let target = self
                .payload
                .capacity()
                .saturating_mul(2)
                .max(need)
                .min(self.payload_len);
            self.payload.reserve_exact(target - self.payload.len());
        }
    }

    /// Start reading a payload of `n` bytes (checked against its limit); a payload of no bytes is
    /// complete at once
    fn start_payload(&mut self, n: usize, entry: Entry) -> Result<(), SnapshotError> {
        self.payload_len = n;
        self.payload = Vec::new();
        self.at = At::Payload(entry);
        if n == 0 {
            self.on_payload(entry)?;
        }
        Ok(())
    }

    /// A payload length, refused over `max_payload`
    fn payload_length(&self, n: u32) -> Result<usize, SnapshotError> {
        let n = usize::try_from(n).map_err(|_| SnapshotError::Inconsistent)?;
        if n > self.limits.max_payload {
            return Err(SnapshotError::Inconsistent);
        }
        Ok(n)
    }

    /// The next room, or the end
    fn next_room(&mut self) {
        if self.rooms_left == 0 {
            self.at = At::End;
        } else {
            self.rooms_left -= 1;
            self.at = At::Field(Step::Alias);
        }
    }

    /// The fixed-size step `f` (exactly its size) is complete: check it and choose the next step
    fn on_field(&mut self, step: Step, f: &[u8]) -> Result<(), SnapshotError> {
        let limits = self.limits;
        match step {
            Step::Magic => {
                if f != SNAPSHOT_MAGIC {
                    return Err(SnapshotError::BadMagic);
                }
                self.at = At::Field(Step::Version);
            }
            Step::Version => {
                let version = u16::from_be_bytes(bytes_at(f, 0));
                if version != SNAPSHOT_VERSION {
                    return Err(SnapshotError::UnknownVersion(version));
                }
                self.at = At::Field(Step::Limit(0));
            }
            Step::Limit(i) => {
                let s = &mut self.stored;
                match i {
                    0 => s.max_payload = limit(u64_at(f, 0))?,
                    1 => s.max_log = limit(u64_at(f, 0))?,
                    2 => s.max_pending = limit(u64_at(f, 0))?,
                    3 => s.max_rooms = limit(u64_at(f, 0))?,
                    4 => s.max_members = limit(u64_at(f, 0))?,
                    5 => s.max_publishes_per_window = u32_at(f, 0),
                    6 => s.window = u64_at(f, 0),
                    7 => s.max_rooms_per_conn = limit(u64_at(f, 0))?,
                    _ => s.max_tags_per_room = limit(u64_at(f, 0))?,
                }
                if i < 8 {
                    self.at = At::Field(Step::Limit(i + 1));
                } else {
                    self.limits = self.run_under.unwrap_or(self.stored);
                    self.at = At::Field(Step::RoomCount);
                }
            }
            Step::RoomCount => {
                self.rooms_left = count(u32_at(f, 0), limits.max_rooms)?;
                self.next_room();
            }
            Step::Alias => {
                let alias = RoomAlias::from_raw(bytes_at(f, 0));
                if self
                    .rooms
                    .last_key_value()
                    .is_some_and(|(last, _)| *last >= alias)
                {
                    return Err(SnapshotError::NotCanonical);
                }
                self.room = RoomInProgress::new(alias);
                self.at = At::Field(Step::RoomHead);
            }
            Step::RoomHead => {
                let room = &mut self.room;
                room.token = RoomToken::from_raw(bytes_at(f, 0));
                room.last_active = u64_at(f, 32);
                room.next_index = u64_at(f, 40);
                self.at = At::Field(Step::MemberCount);
            }
            Step::MemberCount => {
                let n = count(
                    u32_at(f, 0),
                    limits.max_members.min(limits.max_tags_per_room),
                )?;
                self.room.left = n;
                self.at = if n == 0 {
                    At::Field(Step::NextSeqCount)
                } else {
                    At::Field(Step::Member)
                };
            }
            Step::Member => {
                let conn = ConnId::from_raw(u64_at(f, 0));
                let tag = SenderTag::from_raw(bytes_at(f, 8));
                let room = &mut self.room;
                if room
                    .members
                    .last_key_value()
                    .is_some_and(|(last, _)| *last >= conn)
                {
                    return Err(SnapshotError::NotCanonical);
                }
                if !room.member_tags.insert(tag) {
                    return Err(SnapshotError::Inconsistent);
                }
                room.add_tag(tag, &limits)?;
                let in_rooms = self.rooms_of_conn.entry(conn).or_insert(0);
                *in_rooms += 1;
                if *in_rooms > limits.max_rooms_per_conn {
                    return Err(SnapshotError::Inconsistent);
                }
                room.members.insert(conn, tag);
                room.left -= 1;
                if room.left == 0 {
                    self.at = At::Field(Step::NextSeqCount);
                }
            }
            Step::NextSeqCount => {
                let n = count(u32_at(f, 0), limits.max_tags_per_room)?;
                self.room.left = n;
                self.at = if n == 0 {
                    At::Field(Step::HeldTagCount)
                } else {
                    At::Field(Step::NextSeq)
                };
            }
            Step::NextSeq => {
                let tag = SenderTag::from_raw(bytes_at(f, 0));
                let seq = u64_at(f, 16);
                let room = &mut self.room;
                if room
                    .next_seq
                    .last_key_value()
                    .is_some_and(|(last, _)| *last >= tag)
                {
                    return Err(SnapshotError::NotCanonical);
                }
                room.add_tag(tag, &limits)?;
                room.next_seq.insert(tag, seq);
                room.left -= 1;
                if room.left == 0 {
                    self.at = At::Field(Step::HeldTagCount);
                }
            }
            Step::HeldTagCount => {
                let n = count(u32_at(f, 0), limits.max_tags_per_room)?;
                self.room.left = n;
                self.at = if n == 0 {
                    At::Field(Step::RateCount)
                } else {
                    At::Field(Step::HeldTag)
                };
            }
            Step::HeldTag => {
                let tag = SenderTag::from_raw(bytes_at(f, 0));
                let room = &mut self.room;
                if room
                    .held
                    .last_key_value()
                    .is_some_and(|(last, _)| *last >= tag)
                {
                    return Err(SnapshotError::NotCanonical);
                }
                room.add_tag(tag, &limits)?;
                room.held_tag = tag;
                room.held_next = room.next_seq.get(&tag).copied().unwrap_or(0);
                room.held_payloads = BTreeMap::new();
                self.at = At::Field(Step::HeldCount);
            }
            Step::HeldCount => {
                let n = count(u32_at(f, 0), limits.max_pending)?;
                self.room.held_left = n;
                if n == 0 {
                    self.end_held_tag();
                } else {
                    self.at = At::Field(Step::HeldEntry);
                }
            }
            Step::HeldEntry => {
                self.room.entry_seq = u64_at(f, 0);
                let n = self.payload_length(u32_at(f, 8))?;
                self.start_payload(n, Entry::Held)?;
            }
            Step::RateCount => {
                let n = count(u32_at(f, 0), limits.max_tags_per_room)?;
                self.room.left = n;
                self.at = if n == 0 {
                    At::Field(Step::LogCount)
                } else {
                    At::Field(Step::Rate)
                };
            }
            Step::Rate => {
                let tag = SenderTag::from_raw(bytes_at(f, 0));
                let start = u64_at(f, 16);
                let n = u32_at(f, 24);
                let room = &mut self.room;
                if room
                    .rate
                    .last_key_value()
                    .is_some_and(|(last, _)| *last >= tag)
                {
                    return Err(SnapshotError::NotCanonical);
                }
                if n > limits.max_publishes_per_window {
                    return Err(SnapshotError::Inconsistent);
                }
                room.add_tag(tag, &limits)?;
                room.rate.insert(tag, (start, n));
                room.left -= 1;
                if room.left == 0 {
                    self.at = At::Field(Step::LogCount);
                }
            }
            Step::LogCount => {
                // the log holds the last payloads, at most min(next index, max_log), ending just
                // before the next index (fewer than that after max_log was raised by
                // `restore_with_limits`: the log then grows to the new limit)
                let should_keep = u64::from(u32_at(f, 0));
                let room = &mut self.room;
                if should_keep
                    > room
                        .next_index
                        .min(u64::try_from(limits.max_log).unwrap_or(u64::MAX))
                {
                    return Err(SnapshotError::Inconsistent);
                }
                room.should_keep = should_keep;
                room.first_index = room.next_index - should_keep;
                room.logged = 0;
                if should_keep == 0 {
                    self.end_room()?;
                } else {
                    self.at = At::Field(Step::LogEntry);
                }
            }
            Step::LogEntry => {
                let room = &mut self.room;
                room.entry_index = u64_at(f, 0);
                room.entry_from = SenderTag::from_raw(bytes_at(f, 8));
                room.entry_seq = u64_at(f, 24);
                let n = self.payload_length(u32_at(f, 32))?;
                self.start_payload(n, Entry::Log)?;
            }
        }
        Ok(())
    }

    /// The current payload is complete: check its entry and choose the next step
    fn on_payload(&mut self, entry: Entry) -> Result<(), SnapshotError> {
        let payload: Payload = Arc::new(core::mem::take(&mut self.payload));
        match entry {
            Entry::Held => {
                let room = &mut self.room;
                let seq = room.entry_seq;
                if room
                    .held_payloads
                    .last_key_value()
                    .is_some_and(|(last, _)| *last >= seq)
                {
                    return Err(SnapshotError::NotCanonical);
                }
                // held numbers lie after the next expected one and below u64::MAX (refused on
                // publish)
                if seq <= room.held_next || seq == u64::MAX {
                    return Err(SnapshotError::Inconsistent);
                }
                room.held_payloads.insert(seq, payload);
                room.held_left -= 1;
                if room.held_left == 0 {
                    self.end_held_tag();
                } else {
                    self.at = At::Field(Step::HeldEntry);
                }
            }
            Entry::Log => {
                let room = &mut self.room;
                let (index, from, seq) = (room.entry_index, room.entry_from, room.entry_seq);
                if index != room.first_index + room.logged {
                    return Err(SnapshotError::Inconsistent);
                }
                // within a tag, a payload follows the previous one, or the sequence started again
                // at 0 (the tag made way and came back)
                if let Some(prev) = room.last_logged.insert(from, seq) {
                    if seq != 0 && prev.checked_add(1) != Some(seq) {
                        return Err(SnapshotError::Inconsistent);
                    }
                }
                room.log.push_back(Logged {
                    index,
                    from,
                    seq,
                    payload,
                });
                room.logged += 1;
                if room.logged == room.should_keep {
                    self.end_room()?;
                } else {
                    self.at = At::Field(Step::LogEntry);
                }
            }
        }
        Ok(())
    }

    /// The held-back list of a tag is read: keep it, then the next tag or the rate windows
    fn end_held_tag(&mut self) {
        let room = &mut self.room;
        let payloads = core::mem::take(&mut room.held_payloads);
        room.held.insert(room.held_tag, payloads);
        room.left -= 1;
        self.at = if room.left == 0 {
            At::Field(Step::RateCount)
        } else {
            At::Field(Step::HeldTag)
        };
    }

    /// The room is read: check what spans its lists, keep it, then the next room or the end
    fn end_room(&mut self) -> Result<(), SnapshotError> {
        let r = core::mem::replace(
            &mut self.room,
            RoomInProgress::new(RoomAlias::from_raw([0; 32])),
        );
        // a tag the room tracks: its latest logged payload is the one just before its next number
        // (the entry and that payload are made by the same forward, and the log drops oldest first)
        for (tag, next) in &r.next_seq {
            if let Some(last) = r.last_logged.get(tag) {
                if last.checked_add(1) != Some(*next) {
                    return Err(SnapshotError::Inconsistent);
                }
            }
        }
        self.rooms.insert(
            r.alias,
            Arc::new(Room {
                token: r.token,
                members: r.members,
                next_seq: r.next_seq,
                held: r.held,
                rate: r.rate,
                log: r.log,
                next_index: r.next_index,
                last_active: r.last_active,
            }),
        );
        self.next_room();
        Ok(())
    }
}

impl AnonymousRelayCore {
    /// The core the whole of `bytes` holds, read under `run_under` (stored limits when `None`)
    pub(super) fn read_whole(
        bytes: &[u8],
        run_under: Option<Limits>,
    ) -> Result<Self, SnapshotError> {
        let mut r = SnapshotReader::under(run_under);
        r.feed(bytes)?;
        r.finish()
    }

    /// The core the pieces hold, read under `run_under` (stored limits when `None`); one piece is held
    /// at a time
    pub(super) fn read_pieces<I>(
        pieces: I,
        run_under: Option<Limits>,
    ) -> Result<Self, SnapshotError>
    where
        I: IntoIterator,
        I::Item: AsRef<[u8]>,
    {
        let mut r = SnapshotReader::under(run_under);
        for piece in pieces {
            r.feed(piece.as_ref())?;
        }
        r.finish()
    }
}
