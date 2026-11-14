//! The snapshot written and read in pieces (see [the parent module](super), "In pieces").

use std::num::NonZeroUsize;
use std::sync::Arc;

use super::super::{AnonymousRelayCore, ConnId, Limits, Room, RoomAlias, SenderTag};
use super::{put_usize, read, SnapshotError, Source, SNAPSHOT_MAGIC, SNAPSHOT_VERSION};

/// Where the writer is in the format
#[derive(Clone, Copy, Debug)]
enum Stage {
    /// Magic, version, limits and the room count
    Head,
    /// Room `.0`, at the part `.1`
    Room(usize, Part),
    Done,
}

/// Where the writer is within a room; each list remembers the last key written (the next entry is the
/// first key after it), the log its position
#[derive(Clone, Copy, Debug)]
enum Part {
    /// Alias, token, last activity, next index and the member count
    Start,
    Members(Option<ConnId>),
    NextSeq(Option<SenderTag>),
    HeldTags(Option<SenderTag>),
    Held(SenderTag, Option<u64>),
    Rate(Option<SenderTag>),
    Log(usize),
}

/// The state of an [`AnonymousRelayCore`] at one moment, written out in pieces of a bounded size
/// (see [`AnonymousRelayCore::snapshot_writer`]).
#[derive(Debug)]
pub struct SnapshotWriter {
    limits: Limits,
    /// The rooms at the moment the writer was made, shared with the core (copied on write by it)
    rooms: Vec<(RoomAlias, Arc<Room>)>,
    stage: Stage,
    /// The fixed-size fields of the current step (at most [`SnapshotWriter::BUFFER`] bytes) and how
    /// many of them are written
    buf: Vec<u8>,
    buf_at: usize,
    /// The payload of the current step (shared with the room) and how many of its bytes are written
    payload: Option<Arc<Vec<u8>>>,
    payload_at: usize,
    high_water: usize,
}

impl SnapshotWriter {
    /// The most bytes the writer holds of its own at any time besides the piece it appends: the
    /// fixed-size fields of one step (a room's head: alias, token, last activity, next index and the
    /// member count, 84 bytes; the snapshot's head is 78). Payloads are not copied: they are appended
    /// to the piece from the room's own bytes.
    pub const BUFFER: usize = 84;

    fn new(core: &AnonymousRelayCore) -> Self {
        Self {
            limits: core.limits,
            rooms: core
                .rooms
                .iter()
                .map(|(alias, room)| (*alias, Arc::clone(room)))
                .collect(),
            stage: Stage::Head,
            buf: Vec::with_capacity(Self::BUFFER),
            buf_at: 0,
            payload: None,
            payload_at: 0,
            high_water: 0,
        }
    }

    /// Append the next piece, at most `max` bytes, to `out`; whether bytes are left to write after it.
    ///
    /// Every call before the last appends exactly `max` bytes; the last appends the rest (at least
    /// one byte, unless the writer was already done) and returns `false`. A call after that appends
    /// nothing and returns `false`.
    pub fn next_chunk(&mut self, max: NonZeroUsize, out: &mut Vec<u8>) -> bool {
        let mut room = max.get();
        while room > 0 && self.ready() {
            if self.buf_at < self.buf.len() {
                let n = room.min(self.buf.len() - self.buf_at);
                out.extend_from_slice(&self.buf[self.buf_at..self.buf_at + n]);
                self.buf_at += n;
                room -= n;
            } else if let Some(p) = &self.payload {
                let n = room.min(p.len() - self.payload_at);
                out.extend_from_slice(&p[self.payload_at..self.payload_at + n]);
                self.payload_at += n;
                room -= n;
            }
        }
        self.ready()
    }

    /// Whether bytes are left: makes the next step when the current one is written, until one has bytes
    /// or the snapshot is done
    fn ready(&mut self) -> bool {
        loop {
            if self.buf_at < self.buf.len() {
                return true;
            }
            if let Some(p) = &self.payload {
                if self.payload_at < p.len() {
                    return true;
                }
                self.payload = None;
            }
            if matches!(self.stage, Stage::Done) {
                return false;
            }
            self.buf.clear();
            self.buf_at = 0;
            self.payload_at = 0;
            self.step();
            self.high_water = self.high_water.max(self.buf.len());
        }
    }

    /// The most bytes the writer has held of its own at once so far (never over [`Self::BUFFER`]).
    #[doc(hidden)]
    #[must_use]
    pub fn buffer_high_water(&self) -> usize {
        self.high_water
    }

    fn put_u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }

    fn put_u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }

    /// A count; the writer was made only after every count was checked to fit
    fn put_count(&mut self, n: usize) {
        self.put_u32(u32::try_from(n).unwrap_or(u32::MAX));
    }

    /// A payload: its length into the buffer, its bytes shared
    fn put_payload(&mut self, p: &Arc<Vec<u8>>) {
        self.put_count(p.len());
        self.payload = Some(Arc::clone(p));
    }

    /// Write the fields of the step at `self.stage` into the buffer (and the payload) and move on; a
    /// step may write nothing (the end of a held-back list)
    fn step(&mut self) {
        match self.stage {
            Stage::Head => {
                self.buf.extend_from_slice(&SNAPSHOT_MAGIC);
                self.buf.extend_from_slice(&SNAPSHOT_VERSION.to_be_bytes());
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
                } = self.limits;
                for v in [max_payload, max_log, max_pending, max_rooms, max_members] {
                    put_usize(&mut self.buf, v);
                }
                self.put_u32(max_publishes_per_window);
                self.put_u64(window);
                put_usize(&mut self.buf, max_rooms_per_conn);
                put_usize(&mut self.buf, max_tags_per_room);
                self.put_count(self.rooms.len());
                self.stage = self.room_start(0);
            }
            Stage::Room(i, part) => {
                let room = Arc::clone(&self.rooms[i].1);
                self.stage = Stage::Room(i, self.room_step(i, &room, part));
                if let Stage::Room(i, Part::Log(k)) = self.stage {
                    if k > room.log.len() {
                        self.stage = self.room_start(i + 1);
                    }
                }
            }
            Stage::Done => {}
        }
    }

    /// The stage that starts room `i`, or the end
    fn room_start(&self, i: usize) -> Stage {
        if i < self.rooms.len() {
            Stage::Room(i, Part::Start)
        } else {
            Stage::Done
        }
    }

    /// Write one step of room `i` at `part`; the part after it (`Log(len + 1)` when the room is done)
    fn room_step(&mut self, i: usize, room: &Room, part: Part) -> Part {
        match part {
            Part::Start => {
                self.buf.extend_from_slice(self.rooms[i].0.as_ref());
                self.buf.extend_from_slice(room.token.as_ref());
                self.put_u64(room.last_active);
                self.put_u64(room.next_index);
                self.put_count(room.members.len());
                Part::Members(None)
            }
            Part::Members(last) => {
                let next = match last {
                    None => room.members.iter().next(),
                    Some(c) => room.members.range(after(c)).next(),
                };
                if let Some((conn, tag)) = next {
                    self.put_u64(conn.as_u64());
                    self.buf.extend_from_slice(tag.as_ref());
                    Part::Members(Some(*conn))
                } else {
                    self.put_count(room.next_seq.len());
                    Part::NextSeq(None)
                }
            }
            Part::NextSeq(last) => {
                let next = match last {
                    None => room.next_seq.iter().next(),
                    Some(t) => room.next_seq.range(after(t)).next(),
                };
                if let Some((tag, seq)) = next {
                    self.buf.extend_from_slice(tag.as_ref());
                    self.put_u64(*seq);
                    Part::NextSeq(Some(*tag))
                } else {
                    self.put_count(room.held.len());
                    Part::HeldTags(None)
                }
            }
            Part::HeldTags(last) => {
                let next = match last {
                    None => room.held.iter().next(),
                    Some(t) => room.held.range(after(t)).next(),
                };
                if let Some((tag, payloads)) = next {
                    self.buf.extend_from_slice(tag.as_ref());
                    self.put_count(payloads.len());
                    Part::Held(*tag, None)
                } else {
                    self.put_count(room.rate.len());
                    Part::Rate(None)
                }
            }
            Part::Held(tag, last) => {
                let payloads = room.held.get(&tag);
                let next = payloads.and_then(|p| match last {
                    None => p.iter().next(),
                    Some(s) => p.range(after(s)).next(),
                });
                if let Some((seq, payload)) = next {
                    self.put_u64(*seq);
                    self.put_payload(payload);
                    Part::Held(tag, Some(*seq))
                } else {
                    Part::HeldTags(Some(tag))
                }
            }
            Part::Rate(last) => {
                let next = match last {
                    None => room.rate.iter().next(),
                    Some(t) => room.rate.range(after(t)).next(),
                };
                if let Some((tag, (start, n))) = next {
                    self.buf.extend_from_slice(tag.as_ref());
                    self.put_u64(*start);
                    self.put_u32(*n);
                    Part::Rate(Some(*tag))
                } else {
                    self.put_count(room.log.len());
                    Part::Log(0)
                }
            }
            Part::Log(k) => {
                if let Some(item) = room.log.get(k) {
                    self.put_u64(item.index);
                    self.buf.extend_from_slice(item.from.as_ref());
                    self.put_u64(item.seq);
                    self.put_payload(&item.payload);
                }
                // past the last item: `Log(len + 1)` ends the room
                Part::Log(k + 1)
            }
        }
    }
}

/// The keys after `k`
const fn after<K>(k: K) -> (std::ops::Bound<K>, std::ops::Bound<K>) {
    (std::ops::Bound::Excluded(k), std::ops::Bound::Unbounded)
}

/// The pieces of a snapshot as a [`Source`]: one piece is held at a time
struct Pieces<I: Iterator> {
    pieces: I,
    cur: Option<I::Item>,
    at: usize,
}

impl<I> Pieces<I>
where
    I: Iterator,
    I::Item: AsRef<[u8]>,
{
    /// The unread bytes of the current piece, moving to the next non-empty piece when it is used up;
    /// empty when no byte is left
    fn rest(&mut self) -> &[u8] {
        loop {
            let len = self.cur.as_ref().map_or(0, |p| p.as_ref().len());
            if self.at < len {
                break;
            }
            match self.pieces.next() {
                Some(p) => {
                    self.cur = Some(p);
                    self.at = 0;
                }
                None => {
                    self.cur = None;
                    self.at = 0;
                    break;
                }
            }
        }
        self.cur
            .as_ref()
            .map_or(&[][..], |p| &p.as_ref()[self.at..])
    }

    /// Fill `dst` from the pieces, or [`SnapshotError::Truncated`] when they end first
    fn fill(&mut self, mut dst: &mut [u8]) -> Result<(), SnapshotError> {
        while !dst.is_empty() {
            let rest = self.rest();
            if rest.is_empty() {
                return Err(SnapshotError::Truncated);
            }
            let n = rest.len().min(dst.len());
            let (head, tail) = dst.split_at_mut(n);
            head.copy_from_slice(&rest[..n]);
            self.at += n;
            dst = tail;
        }
        Ok(())
    }
}

impl<I> Source for Pieces<I>
where
    I: Iterator,
    I::Item: AsRef<[u8]>,
{
    fn take<const N: usize>(&mut self) -> Result<[u8; N], SnapshotError> {
        let mut out = [0u8; N];
        self.fill(&mut out)?;
        Ok(out)
    }

    /// Grown as the bytes arrive (never reserved for the whole length up front), then trimmed to the
    /// length
    fn payload(&mut self, n: usize) -> Result<Vec<u8>, SnapshotError> {
        let mut p = Vec::new();
        while p.len() < n {
            let rest = self.rest();
            if rest.is_empty() {
                return Err(SnapshotError::Truncated);
            }
            let k = rest.len().min(n - p.len());
            p.extend_from_slice(&rest[..k]);
            self.at += k;
        }
        p.shrink_to_fit();
        Ok(p)
    }

    fn at_end(&mut self) -> bool {
        self.rest().is_empty()
    }
}

impl AnonymousRelayCore {
    /// A writer of the state as it is now, in pieces (see [`SnapshotWriter`]): its pieces, concatenated,
    /// are the bytes [`snapshot`](Self::snapshot) returns now, whatever the core does afterwards.
    ///
    /// The writer shares the rooms with the core instead of copying them; a room the core changes
    /// while the writer holds it is copied by the core then, without its payloads (they stay shared).
    ///
    /// # Errors
    ///
    /// As [`snapshot`](Self::snapshot): [`SnapshotError::Unrepresentable`] when a count or a payload
    /// length is over `u32::MAX`, before any piece is made.
    pub fn snapshot_writer(&self) -> Result<SnapshotWriter, SnapshotError> {
        self.check_representable()?;
        Ok(SnapshotWriter::new(self))
    }

    /// [`restore`](Self::restore) from a snapshot given in pieces, read as they come (the pieces are
    /// never joined; one is held at a time). The result, the core or the [`SnapshotError`], is the one
    /// `restore` gives for the pieces concatenated, wherever they are cut; empty pieces are skipped.
    ///
    /// # Errors
    ///
    /// See [`SnapshotError`].
    pub fn restore_from_pieces<I>(pieces: I) -> Result<Self, SnapshotError>
    where
        I: IntoIterator,
        I::Item: AsRef<[u8]>,
    {
        read(
            Pieces {
                pieces: pieces.into_iter(),
                cur: None,
                at: 0,
            },
            None,
        )
    }

    /// [`restore_with_limits`](Self::restore_with_limits) from a snapshot given in pieces, as
    /// [`restore_from_pieces`](Self::restore_from_pieces) reads them.
    ///
    /// # Errors
    ///
    /// See [`SnapshotError`].
    pub fn restore_from_pieces_with_limits<I>(
        pieces: I,
        limits: Limits,
    ) -> Result<Self, SnapshotError>
    where
        I: IntoIterator,
        I::Item: AsRef<[u8]>,
    {
        read(
            Pieces {
                pieces: pieces.into_iter(),
                cur: None,
                at: 0,
            },
            Some(limits),
        )
    }
}
