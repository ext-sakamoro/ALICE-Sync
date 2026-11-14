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
//! The bindings ([`ConnId`](super::ConnId) to the tag it publishes as) are kept. A [`ConnId`](super::ConnId) is the host's handle for
//! one transport connection, and a host that restarts its process may keep the connections themselves
//! open (a host that hibernates keeps its sockets). The host restores with the same numbering for the
//! connections that are still open, and closes each one that is gone (otherwise the room keeps routing
//! to it, and its tag stays taken): [`AnonymousRelayCore::bound_conns`] lists the restored bindings, and
//! [`AnonymousRelayCore::close_all_except`] closes every one not in the set the host still holds. It
//! must not give a new connection a number a restored binding still uses.
//!
//! # Limits
//!
//! [`AnonymousRelayCore::restore`] runs the core under the limits stored in the snapshot.
//! [`AnonymousRelayCore::restore_with_limits`] runs it under limits the host passes instead (for example
//! a new configuration deployed while the process was evicted): every check below is made against the
//! new limits, so a state over any of them (a room with more members than the new `max_members`, a log
//! longer than the new `max_log`, a payload over the new `max_payload`, a rate window counting more
//! than the new `max_publishes_per_window`, ...) is refused with the same [`SnapshotError`] `restore`
//! gives for a state over its stored limits; nothing is trimmed. After a restore under raised limits
//! the core follows them (a payload the old `max_payload` refused is now accepted, the log grows to the
//! new `max_log`), and its snapshot stores the new limits.
//!
//! # In pieces
//!
//! A host that stores the snapshot over several steps in bounded pieces, while the core keeps running,
//! uses [`AnonymousRelayCore::snapshot_writer`]: the [`SnapshotWriter`] takes the state at the moment it
//! is made, and its pieces ([`SnapshotWriter::next_chunk`]), concatenated, are the bytes
//! [`AnonymousRelayCore::snapshot`] returns at that moment, whatever the core does between the pieces.
//! It does not copy the state: the rooms are shared with the core, which copies a room the writer still
//! holds when it changes it (the room's fields, not its payloads, which stay shared). Besides the piece
//! it appends, the writer holds per room one reference and at most [`SnapshotWriter::BUFFER`] bytes.
//! [`AnonymousRelayCore::restore_from_pieces`] and [`AnonymousRelayCore::restore_from_pieces_with_limits`]
//! read the pieces as they come, without joining them, and give the same core or the same
//! [`SnapshotError`] as [`AnonymousRelayCore::restore`] and [`AnonymousRelayCore::restore_with_limits`]
//! on the concatenated bytes, wherever the pieces are cut. The format has no length or checksum over
//! the whole, so nothing is computed ahead of the first byte.
//!
//! A host whose storage reads are asynchronous (it reads a piece, waits for the read to complete, then
//! reads the next) cannot hand an iterator over the pieces; it hands each piece as it arrives to a
//! [`SnapshotReader`] ([`SnapshotReader::feed`]), then takes the core from
//! [`SnapshotReader::finish`]. The reader is the decoder `restore` and `restore_from_pieces` use, so
//! the result is theirs; it keeps none of the pieces, only the bytes of one field a piece cuts (at most
//! one payload). See [`SnapshotReader`] for when each error comes out.
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
//! [`AnonymousRelayCore::restore`] takes exactly one snapshot. These are a [`SnapshotError`], never a
//! panic: other magic, another version, bytes that end early, bytes after the end, keys out of order or
//! repeated, a limit that does not fit this platform's `usize`, and any state the core cannot reach:
//!
//! - a count over its limit (rooms over `max_rooms`, members over `max_members`, held-back payloads of a
//!   tag over `max_pending`, the tags a room keeps state for over `max_tags_per_room`, a connection in
//!   more rooms than `max_rooms_per_conn`), a payload over `max_payload`, a rate window counting more
//!   than `max_publishes_per_window`;
//! - two connections of a room on one tag; a held-back payload at or before its tag's next sequence
//!   number, or at `u64::MAX` (a publish of that number is refused);
//! - a log longer than `min(next index, max_log)`, or whose indices do not run up to the next index
//!   (a log shorter than that is a state the core reaches after `max_log` was raised, see below); within one tag, a logged payload that neither follows the tag's previous one
//!   (`seq + 1`) nor starts its sequence again at 0 (a tag that made way and came back); for a tag the
//!   room tracks, a latest logged payload other than the one just before its next sequence number.
//!
//! A count is checked against its limit when it is read and each entry as it is read, so the work a
//! decode does is bounded by the limits and the bytes present, never by a count alone. Nothing is
//! allocated for a count before its entries are there, and a payload's buffer is never more than
//! twice the bytes of it present (nor more than its length).
//!
//! The counters at the top of their range are states the core reaches and handles: a tag whose next
//! sequence number is `u64::MAX` has used every number, and a room whose next log index is `u64::MAX`
//! has forwarded all it can; a publish that would need a number past either is refused with
//! [`Refusal::Exhausted`](super::Refusal::Exhausted), so a restored core cannot be made to overflow.

use super::{AnonymousRelayCore, Limits, Logged, Room};

mod reader;
mod stream;

pub use reader::SnapshotReader;
pub use stream::SnapshotWriter;

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
            } = &**room;
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
                let Logged {
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
    /// The host restores with the same [`ConnId`](super::ConnId) numbering for connections still open and calls
    /// [`close`](Self::close) for each one that is gone (see [the module documentation](self)).
    ///
    /// # Errors
    ///
    /// See [`SnapshotError`].
    pub fn restore(bytes: &[u8]) -> Result<Self, SnapshotError> {
        Self::read_whole(bytes, None)
    }
}

impl AnonymousRelayCore {
    /// A core in the state `bytes` holds, run under `limits` instead of the limits stored in the snapshot.
    ///
    /// The state is checked against `limits`: a state over any of them is refused (see
    /// [the module documentation](self)). The bytes are read and checked as by
    /// [`restore`](Self::restore) otherwise.
    ///
    /// # Errors
    ///
    /// See [`SnapshotError`].
    pub fn restore_with_limits(bytes: &[u8], limits: Limits) -> Result<Self, SnapshotError> {
        Self::read_whole(bytes, Some(limits))
    }
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
