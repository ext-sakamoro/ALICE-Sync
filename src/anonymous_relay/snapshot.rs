//! The anonymous relay's state as bytes, for a host that may evict and restart the process holding
//! the core: [`AnonymousRelayCore::snapshot`] writes it, [`AnonymousRelayCore::restore`] reads it back.
//!
//! A restored core behaves as the core that wrote the snapshot: given the same frames afterwards it
//! sends the same frames, and its own snapshot is the same bytes. Everything the core holds is kept:
//! the limits, and for each room its token, the bound connections and their tags, each tag's next
//! sequence number, the payloads held back waiting for a gap, each tag's rate window (start and count),
//! the log with its running index, and the time the room was last active.
//!
//! # Connections
//!
//! The bindings ([`ConnId`] to the tag it publishes as) are kept. A [`ConnId`] is the host's handle for
//! one transport connection, and a host that restarts its process may keep the connections themselves
//! open (a host that hibernates keeps its sockets). The host restores with the same numbering for the
//! connections that are still open, and calls [`AnonymousRelayCore::close`] for each one that is gone
//! (otherwise the room keeps routing to it, and its tag stays taken). It must not give a new
//! connection a number a restored binding still uses.
//!
//! # Clock
//!
//! The times in the state (each room's last activity, each rate window's start) are in the caller's
//! clock units, as passed to `bind` and `publish`. They are stored as they are: after a restore the
//! host passes times on the same clock, not one restarted at zero.
//!
//! # Format (version 1)
//!
//! Integers big-endian, fixed width; every map in ascending key order (the order the core iterates
//! in), so a state has exactly one encoding and the same state always gives the same bytes.
//!
//! | bytes | content |
//! |---|---|
//! | 4 | magic [`SNAPSHOT_MAGIC`] (`ARLY`) |
//! | 2 | format version [`SNAPSHOT_VERSION`] (`u16`) |
//! | 68 | the limits: `max_payload`, `max_log`, `max_pending`, `max_rooms`, `max_members` (`u64` each), `max_publishes_per_window` (`u32`), `window`, `max_rooms_per_conn`, `max_tags_per_room` (`u64` each) |
//! | 4 | number of rooms (`u32`) |
//! | rest | each room, by ascending alias |
//!
//! A room: alias (32 bytes), token (32), `last_active` (`u64`), the log's next index (`u64`), then five
//! lists, each a `u32` count and its entries:
//!
//! - members: connection (`u64`) and tag (16 bytes), by ascending connection;
//! - next sequence numbers: tag and `u64`, by ascending tag;
//! - held-back payloads: tag, then a `u32` count of `(seq: u64, payload)`, by ascending sequence number
//!   (a tag may have an empty list: it still counts as a tag the room keeps state for);
//! - rate windows: tag, window start (`u64`) and publishes in it (`u32`), by ascending tag;
//! - log: `(index: u64, tag, seq: u64, payload)`, oldest first.
//!
//! A payload is a `u32` length and the bytes.
//!
//! Counts and payload lengths are `u32`. A state with a list or a payload longer than `u32::MAX`
//! (entries or bytes) is not written at all: [`AnonymousRelayCore::snapshot`] returns
//! [`SnapshotError::Unrepresentable`] naming it, before writing any byte.
//!
//! # Reading
//!
//! [`AnonymousRelayCore::restore`] takes exactly one snapshot: other magic, another version, bytes that
//! end early, bytes after the end, keys out of order or repeated, or a state the core could not have
//! reached (a log whose indices do not run up to the next index or that does not hold the last
//! `min(next index, max_log)` payloads, two connections of a room on one tag, a
//! held-back payload at or before its tag's next sequence number, a limit that does not fit this
//! platform's `usize`) is a [`SnapshotError`], never a panic. Nothing is allocated for a count or a
//! length before the bytes it names are there, so a corrupted count cannot make a decode reserve more
//! space than the input holds.

use std::collections::{BTreeMap, VecDeque};

use super::{AnonymousRelayCore, ConnId, Limits, LogItem, Room, RoomAlias, RoomToken, SenderTag};

/// The first four bytes of a snapshot.
pub const SNAPSHOT_MAGIC: [u8; 4] = *b"ARLY";

/// The snapshot format this build writes, and the only one it reads.
pub const SNAPSHOT_VERSION: u16 = 1;

/// Why bytes are not a snapshot this build reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SnapshotError {
    /// The bytes do not start with [`SNAPSHOT_MAGIC`].
    BadMagic,
    /// The format version is not [`SNAPSHOT_VERSION`].
    UnknownVersion(u16),
    /// The bytes end inside the snapshot.
    Truncated,
    /// Bytes follow the snapshot.
    Trailing,
    /// Keys of a map are not strictly ascending (out of order or repeated).
    NotCanonical,
    /// The state is not one the core could have reached.
    Inconsistent,
    /// A limit does not fit this platform's `usize`.
    LimitTooLarge,
    /// [`AnonymousRelayCore::snapshot`]: a count or a payload length in the state is over `u32::MAX`, so
    /// the format cannot hold the state; `what` names the list or the payload.
    Unrepresentable {
        /// The list or payload that is too long.
        what: &'static str,
    },
}

impl core::fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BadMagic => f.write_str("not an anonymous relay snapshot (bad magic)"),
            Self::UnknownVersion(v) => write!(
                f,
                "anonymous relay snapshot version {v} is not read by this build (it reads {SNAPSHOT_VERSION})"
            ),
            Self::Truncated => f.write_str("anonymous relay snapshot ends early"),
            Self::Trailing => f.write_str("bytes follow the anonymous relay snapshot"),
            Self::NotCanonical => {
                f.write_str("anonymous relay snapshot has keys out of order or repeated")
            }
            Self::Inconsistent => {
                f.write_str("anonymous relay snapshot holds a state the relay cannot reach")
            }
            Self::LimitTooLarge => {
                f.write_str("anonymous relay snapshot has a limit too large for this platform")
            }
            Self::Unrepresentable { what } => write!(
                f,
                "anonymous relay state cannot be snapshotted: {what} is longer than u32::MAX"
            ),
        }
    }
}

impl std::error::Error for SnapshotError {}

impl AnonymousRelayCore {
    /// The whole state as bytes (see [the format](self)); the same state always gives the same bytes.
    ///
    /// # Errors
    ///
    /// [`SnapshotError::Unrepresentable`] when a count or a payload length in the state is over
    /// `u32::MAX`; the state is checked before any byte is written, so no partial snapshot is made.
    pub fn snapshot(&self) -> Result<Vec<u8>, SnapshotError> {
        self.check_representable()?;
        let Self { limits, rooms } = self;
        let mut out = Vec::new();
        out.extend_from_slice(&SNAPSHOT_MAGIC);
        out.extend_from_slice(&SNAPSHOT_VERSION.to_be_bytes());
        let Limits {
            max_payload,
            max_log,
            max_pending,
            max_rooms,
            max_members,
            max_publishes_per_window,
            window,
            max_rooms_per_conn,
            max_tags_per_room,
        } = *limits;
        for v in [max_payload, max_log, max_pending, max_rooms, max_members] {
            put_usize(&mut out, v);
        }
        out.extend_from_slice(&max_publishes_per_window.to_be_bytes());
        out.extend_from_slice(&window.to_be_bytes());
        put_usize(&mut out, max_rooms_per_conn);
        put_usize(&mut out, max_tags_per_room);
        put_len(&mut out, rooms.len(), ROOMS)?;
        for (alias, room) in rooms {
            let Room {
                token,
                members,
                next_seq,
                held,
                rate,
                log,
                next_index,
                last_active,
            } = room;
            out.extend_from_slice(alias.as_ref());
            out.extend_from_slice(token.as_ref());
            out.extend_from_slice(&last_active.to_be_bytes());
            out.extend_from_slice(&next_index.to_be_bytes());
            put_len(&mut out, members.len(), MEMBERS)?;
            for (conn, tag) in members {
                out.extend_from_slice(&conn.as_u64().to_be_bytes());
                out.extend_from_slice(tag.as_ref());
            }
            put_len(&mut out, next_seq.len(), NEXT_SEQ)?;
            for (tag, seq) in next_seq {
                out.extend_from_slice(tag.as_ref());
                out.extend_from_slice(&seq.to_be_bytes());
            }
            put_len(&mut out, held.len(), HELD_TAGS)?;
            for (tag, payloads) in held {
                out.extend_from_slice(tag.as_ref());
                put_len(&mut out, payloads.len(), HELD_PAYLOADS)?;
                for (seq, payload) in payloads {
                    out.extend_from_slice(&seq.to_be_bytes());
                    put_bytes(&mut out, payload)?;
                }
            }
            put_len(&mut out, rate.len(), RATE)?;
            for (tag, (start, count)) in rate {
                out.extend_from_slice(tag.as_ref());
                out.extend_from_slice(&start.to_be_bytes());
                out.extend_from_slice(&count.to_be_bytes());
            }
            put_len(&mut out, log.len(), LOG)?;
            for item in log {
                let LogItem {
                    index,
                    from,
                    seq,
                    payload,
                } = item;
                out.extend_from_slice(&index.to_be_bytes());
                out.extend_from_slice(from.as_ref());
                out.extend_from_slice(&seq.to_be_bytes());
                put_bytes(&mut out, payload)?;
            }
        }
        Ok(out)
    }

    /// A core in the state `bytes` holds, as written by [`snapshot`](Self::snapshot).
    ///
    /// The host restores with the same [`ConnId`] numbering for connections still open and calls
    /// [`close`](Self::close) for each one that is gone (see [the module documentation](self)).
    ///
    /// # Errors
    ///
    /// See [`SnapshotError`].
    pub fn restore(bytes: &[u8]) -> Result<Self, SnapshotError> {
        let mut r = Reader(bytes);
        if r.take::<4>()? != SNAPSHOT_MAGIC {
            return Err(SnapshotError::BadMagic);
        }
        let version = u16::from_be_bytes(r.take()?);
        if version != SNAPSHOT_VERSION {
            return Err(SnapshotError::UnknownVersion(version));
        }
        let max_payload = r.usize()?;
        let max_log = r.usize()?;
        let max_pending = r.usize()?;
        let max_rooms = r.usize()?;
        let max_members = r.usize()?;
        let max_publishes_per_window = r.u32()?;
        let window = r.u64()?;
        let max_rooms_per_conn = r.usize()?;
        let max_tags_per_room = r.usize()?;
        let limits = Limits {
            max_payload,
            max_log,
            max_pending,
            max_rooms,
            max_members,
            max_publishes_per_window,
            window,
            max_rooms_per_conn,
            max_tags_per_room,
        };
        let mut rooms = BTreeMap::new();
        let mut rooms_of_conn: BTreeMap<ConnId, usize> = BTreeMap::new();
        for _ in 0..r.u32()? {
            let alias = RoomAlias::from_raw(r.take()?);
            if rooms
                .last_key_value()
                .is_some_and(|(last, _)| *last >= alias)
            {
                return Err(SnapshotError::NotCanonical);
            }
            let room = read_room(&mut r, &limits)?;
            for conn in room.members.keys() {
                let n = rooms_of_conn.entry(*conn).or_insert(0);
                *n += 1;
                if *n > limits.max_rooms_per_conn {
                    return Err(SnapshotError::Inconsistent);
                }
            }
            rooms.insert(alias, room);
        }
        if !r.0.is_empty() {
            return Err(SnapshotError::Trailing);
        }
        if rooms.len() > limits.max_rooms {
            return Err(SnapshotError::Inconsistent);
        }
        Ok(Self { limits, rooms })
    }
}

/// One room, checked for the invariants the core keeps
fn read_room(r: &mut Reader<'_>, limits: &Limits) -> Result<Room, SnapshotError> {
    let token = RoomToken::from_raw(r.take()?);
    let last_active = r.u64()?;
    let next_index = r.u64()?;
    let mut members: BTreeMap<ConnId, SenderTag> = BTreeMap::new();
    for _ in 0..r.u32()? {
        let conn = ConnId::from_raw(r.u64()?);
        let tag = SenderTag::from_raw(r.take()?);
        if members
            .last_key_value()
            .is_some_and(|(last, _)| *last >= conn)
        {
            return Err(SnapshotError::NotCanonical);
        }
        if members.values().any(|t| *t == tag) {
            return Err(SnapshotError::Inconsistent);
        }
        members.insert(conn, tag);
    }
    let mut next_seq: BTreeMap<SenderTag, u64> = BTreeMap::new();
    for _ in 0..r.u32()? {
        let tag = SenderTag::from_raw(r.take()?);
        let seq = r.u64()?;
        if next_seq
            .last_key_value()
            .is_some_and(|(last, _)| *last >= tag)
        {
            return Err(SnapshotError::NotCanonical);
        }
        next_seq.insert(tag, seq);
    }
    let mut held: BTreeMap<SenderTag, BTreeMap<u64, Vec<u8>>> = BTreeMap::new();
    for _ in 0..r.u32()? {
        let tag = SenderTag::from_raw(r.take()?);
        if held.last_key_value().is_some_and(|(last, _)| *last >= tag) {
            return Err(SnapshotError::NotCanonical);
        }
        let next = next_seq.get(&tag).copied().unwrap_or(0);
        let mut payloads: BTreeMap<u64, Vec<u8>> = BTreeMap::new();
        for _ in 0..r.u32()? {
            let seq = r.u64()?;
            let payload = r.bytes()?;
            if payloads
                .last_key_value()
                .is_some_and(|(last, _)| *last >= seq)
            {
                return Err(SnapshotError::NotCanonical);
            }
            if seq <= next || payload.len() > limits.max_payload {
                return Err(SnapshotError::Inconsistent);
            }
            payloads.insert(seq, payload);
        }
        if payloads.len() > limits.max_pending {
            return Err(SnapshotError::Inconsistent);
        }
        held.insert(tag, payloads);
    }
    let mut rate: BTreeMap<SenderTag, (u64, u32)> = BTreeMap::new();
    for _ in 0..r.u32()? {
        let tag = SenderTag::from_raw(r.take()?);
        let start = r.u64()?;
        let count = r.u32()?;
        if rate.last_key_value().is_some_and(|(last, _)| *last >= tag) {
            return Err(SnapshotError::NotCanonical);
        }
        rate.insert(tag, (start, count));
    }
    let mut log: VecDeque<LogItem> = VecDeque::new();
    for _ in 0..r.u32()? {
        let index = r.u64()?;
        let from = SenderTag::from_raw(r.take()?);
        let seq = r.u64()?;
        let payload = r.bytes()?;
        // indices run one by one
        if log
            .back()
            .is_some_and(|last| last.index.checked_add(1) != Some(index))
        {
            return Err(SnapshotError::Inconsistent);
        }
        log.push_back(LogItem {
            index,
            from,
            seq,
            payload,
        });
    }
    // the log ends just before the next index and holds the last min(next index, max_log) items
    let ends_right = log
        .back()
        .is_none_or(|last| last.index.checked_add(1) == Some(next_index));
    let kept = u64::try_from(log.len()).unwrap_or(u64::MAX);
    let should_keep = next_index.min(u64::try_from(limits.max_log).unwrap_or(u64::MAX));
    if !ends_right || kept != should_keep || members.len() > limits.max_members {
        return Err(SnapshotError::Inconsistent);
    }
    let room = Room {
        token,
        members,
        next_seq,
        held,
        rate,
        log,
        next_index,
        last_active,
    };
    if room.tag_count() > limits.max_tags_per_room {
        return Err(SnapshotError::Inconsistent);
    }
    Ok(room)
}

fn put_usize(out: &mut Vec<u8>, v: usize) {
    // usize is at most 64 bits on every target Rust supports
    out.extend_from_slice(&u64::try_from(v).unwrap_or(u64::MAX).to_be_bytes());
}

const ROOMS: &str = "the rooms";
const MEMBERS: &str = "a room's members";
const NEXT_SEQ: &str = "a room's sequence numbers";
const HELD_TAGS: &str = "a room's tags with held-back payloads";
const HELD_PAYLOADS: &str = "a tag's held-back payloads";
const RATE: &str = "a room's rate windows";
const LOG: &str = "a room's log";
const PAYLOAD: &str = "a payload";

/// A count or length as the format's `u32`, or the error naming what is too long
fn count(n: usize, what: &'static str) -> Result<u32, SnapshotError> {
    u32::try_from(n).map_err(|_| SnapshotError::Unrepresentable { what })
}

impl AnonymousRelayCore {
    /// Every count and payload length of the state fits the format (checked before writing a byte)
    fn check_representable(&self) -> Result<(), SnapshotError> {
        count(self.rooms.len(), ROOMS)?;
        for room in self.rooms.values() {
            count(room.members.len(), MEMBERS)?;
            count(room.next_seq.len(), NEXT_SEQ)?;
            count(room.held.len(), HELD_TAGS)?;
            for payloads in room.held.values() {
                count(payloads.len(), HELD_PAYLOADS)?;
                for p in payloads.values() {
                    count(p.len(), PAYLOAD)?;
                }
            }
            count(room.rate.len(), RATE)?;
            count(room.log.len(), LOG)?;
            for item in &room.log {
                count(item.payload.len(), PAYLOAD)?;
            }
        }
        Ok(())
    }
}

fn put_len(out: &mut Vec<u8>, n: usize, what: &'static str) -> Result<(), SnapshotError> {
    out.extend_from_slice(&count(n, what)?.to_be_bytes());
    Ok(())
}

fn put_bytes(out: &mut Vec<u8>, p: &[u8]) -> Result<(), SnapshotError> {
    put_len(out, p.len(), PAYLOAD)?;
    out.extend_from_slice(p);
    Ok(())
}

struct Reader<'a>(&'a [u8]);

impl Reader<'_> {
    fn take<const N: usize>(&mut self) -> Result<[u8; N], SnapshotError> {
        let (head, rest) = self
            .0
            .split_first_chunk::<N>()
            .ok_or(SnapshotError::Truncated)?;
        self.0 = rest;
        Ok(*head)
    }

    fn u32(&mut self) -> Result<u32, SnapshotError> {
        Ok(u32::from_be_bytes(self.take()?))
    }

    fn u64(&mut self) -> Result<u64, SnapshotError> {
        Ok(u64::from_be_bytes(self.take()?))
    }

    fn usize(&mut self) -> Result<usize, SnapshotError> {
        usize::try_from(self.u64()?).map_err(|_| SnapshotError::LimitTooLarge)
    }

    /// A length-prefixed byte string; the length is checked against the bytes present before anything
    /// is allocated
    fn bytes(&mut self) -> Result<Vec<u8>, SnapshotError> {
        let n = usize::try_from(self.u32()?).map_err(|_| SnapshotError::Truncated)?;
        if n > self.0.len() {
            return Err(SnapshotError::Truncated);
        }
        let (p, rest) = self.0.split_at(n);
        self.0 = rest;
        Ok(p.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::{count, SnapshotError, LOG, PAYLOAD};

    /// A length over `u32::MAX` is refused, naming what is too long; `u32::MAX` itself fits. A state
    /// that long cannot be built in a test, so the check is tested where every count and length passes.
    #[test]
    fn count_over_u32_is_unrepresentable() {
        assert_eq!(count(0, LOG), Ok(0));
        assert_eq!(count(u32::MAX as usize, LOG), Ok(u32::MAX));
        #[cfg(target_pointer_width = "64")]
        {
            let over = u32::MAX as usize + 1;
            assert_eq!(
                count(over, PAYLOAD),
                Err(SnapshotError::Unrepresentable { what: PAYLOAD })
            );
            assert_eq!(
                count(usize::MAX, LOG),
                Err(SnapshotError::Unrepresentable { what: LOG })
            );
        }
    }
}
