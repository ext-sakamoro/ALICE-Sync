//! Anonymous room relay core: the relay learns no device ids. Frames in, frames out, no I/O.
//!
//! This is the relay for groups whose members must stay unknown to whoever runs it. The other relay,
//! [`crate::relay`], keeps each room as a list of named members (device ids) and sees who is in which room;
//! pick it only where that is acceptable (for example a game room whose players are known anyway).
//!
//! - A room is named by a [`RoomAlias`] (32 bytes, such as the exporter value a group derives for the epoch)
//!   and entered with a [`RoomToken`] (32 bytes, a second value the members derive for the same epoch). The
//!   first bind of an alias opens the room under the token it presents; every later bind must present the
//!   same token. A group moves to a new room when its epoch changes, and the old room's token does not
//!   open the new one, so a member that joined later cannot enter or read an earlier room.
//! - A connection is a [`ConnId`] the server gives each transport connection; the relay routes by it and
//!   knows nothing else about the peer.
//! - A publisher is known by a [`SenderTag`] (16 bytes) it chose for the room, for example at random per
//!   epoch, and keeps across reconnects within the room. Per tag, payloads leave in sequence order: a payload
//!   ahead of the next expected one is held back (up to [`Limits::max_pending`]) and the publisher is asked
//!   to resend the gap; a duplicate is dropped. Sequence numbers run
//!   from 0 to `u64::MAX - 1`; a publish past that, or past a room's last log index, is refused as
//!   [`Refusal::Exhausted`]. A tag held by another bound connection is refused. Payloads
//!   are opaque (sealed end to end by the members): a member that takes another member's tag after it
//!   leaves can only disorder that tag's sequence, never forge a payload.
//! - Forwarded payloads are kept in a per-room log, capped by count, for a member that reconnects.
//! - A bound connection keeps its tag (a bind under another tag is refused), so a new tag takes a new
//!   connection, which the server admits at its own rate (for example a per-source handshake gate).
//! - Limits: payload size, members per room, publishes per tag per window, rooms per connection, tags a room
//!   keeps state for (a tag nobody holds and with nothing held back makes way, and its sequence starts again),
//!   and rooms; when the room count is reached, the empty room idle longest makes way for a new one (its log
//!   goes with it). A room that made way can be opened again under another token by whoever names its alias
//!   first: members that then get `BadToken` move to their next epoch's room.
//!
//! All maps are ordered, so the output for a given input sequence is the same on every build.
//!
//! The whole state can be written out and read back ([`AnonymousRelayCore::snapshot`],
//! [`AnonymousRelayCore::restore`]), so a host that may evict and restart the process holding the core
//! keeps every room across the restart; see [`snapshot`] for the format and what the host must do about
//! connections.

use std::collections::{BTreeMap, VecDeque};

pub mod snapshot;

pub use snapshot::{SnapshotError, SNAPSHOT_MAGIC, SNAPSHOT_VERSION};

fn hex(f: &mut core::fmt::Formatter<'_>, bytes: &[u8]) -> core::fmt::Result {
    for b in bytes {
        write!(f, "{b:02x}")?;
    }
    Ok(())
}

/// A room's name: 32 bytes the members derive for the epoch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RoomAlias([u8; 32]);

impl RoomAlias {
    /// An alias from its bytes.
    #[must_use]
    pub const fn from_raw(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
}

impl AsRef<[u8]> for RoomAlias {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl core::fmt::Display for RoomAlias {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        hex(f, &self.0)
    }
}

/// What a member presents to enter a room: 32 bytes the members derive for the epoch.
///
/// It has no `Display`, and its `Debug` shows no byte, so it does not end up in a log.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct RoomToken([u8; 32]);

impl RoomToken {
    /// A token from its bytes.
    #[must_use]
    pub const fn from_raw(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
}

impl AsRef<[u8]> for RoomToken {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl core::fmt::Debug for RoomToken {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("RoomToken(..)")
    }
}

/// A publisher's name within one room, chosen by the publisher.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SenderTag([u8; 16]);

impl SenderTag {
    /// A tag from its bytes.
    #[must_use]
    pub const fn from_raw(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }
}

impl AsRef<[u8]> for SenderTag {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl core::fmt::Display for SenderTag {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        hex(f, &self.0)
    }
}

/// A transport connection, numbered by the server that runs the relay.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ConnId(u64);

impl ConnId {
    /// A connection id from its number.
    #[must_use]
    pub const fn from_raw(id: u64) -> Self {
        Self(id)
    }

    /// The number.
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl core::fmt::Display for ConnId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// What the relay accepts, set by whoever runs it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Limits {
    /// Largest payload accepted, in bytes.
    pub max_payload: usize,
    /// Payloads kept in a room's log.
    pub max_log: usize,
    /// Payloads held back per sender tag while waiting for a gap to fill.
    pub max_pending: usize,
    /// Rooms held at once.
    pub max_rooms: usize,
    /// Connections bound to one room at once.
    pub max_members: usize,
    /// Publishes accepted per sender tag in one window (a connection keeps its tag, so a new tag needs a new
    /// connection, which the server admits at its own rate).
    pub max_publishes_per_window: u32,
    /// The window, in the caller's clock units.
    pub window: u64,
    /// Rooms one connection may be bound to at once.
    pub max_rooms_per_conn: usize,
    /// Sender tags a room keeps state for (sequence, held payloads); a tag no connection holds and with
    /// nothing held back makes way for a new one.
    pub max_tags_per_room: usize,
}

/// Why the relay refused a frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Refusal {
    /// The room is not open on this relay.
    NoRoom,
    /// The token is not the room's.
    BadToken,
    /// The connection is not bound to the room.
    NotBound,
    /// Another bound connection holds the sender tag.
    TagInUse,
    /// The payload is larger than [`Limits::max_payload`].
    TooLarge,
    /// The publisher is more than [`Limits::max_pending`] payloads ahead of the next expected one.
    TooFarAhead,
    /// [`Limits::max_rooms`] rooms are held and none is empty.
    TooManyRooms,
    /// [`Limits::max_members`] connections are bound to the room.
    RoomFull,
    /// The sender tag reached [`Limits::max_publishes_per_window`] in this window.
    RateLimited,
    /// The connection is bound to [`Limits::max_rooms_per_conn`] rooms.
    ConnRoomLimit,
    /// The connection is bound to the room under another tag.
    TagChange,
    /// The room keeps [`Limits::max_tags_per_room`] tags and none can make way.
    TooManyTags,
    /// The publish would need a sequence number or a log index past `u64::MAX`: a tag's last sequence
    /// number is `u64::MAX - 1`, and a room forwards at most `u64::MAX` payloads.
    Exhausted,
}

/// A frame the relay sends.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Out {
    /// `to` is bound to `room`.
    Bound {
        /// The connection.
        to: ConnId,
        /// Room.
        room: RoomAlias,
    },
    /// A payload for `to`, published under `from` with `seq`.
    Forward {
        /// Recipient.
        to: ConnId,
        /// Room.
        room: RoomAlias,
        /// The publisher's tag.
        from: SenderTag,
        /// The publisher's sequence number.
        seq: u64,
        /// The payload as published.
        payload: Vec<u8>,
    },
    /// Ask `to` to publish `from..=to_seq` again.
    Resend {
        /// The publisher asked.
        to: ConnId,
        /// Room.
        room: RoomAlias,
        /// First missing sequence number.
        from: u64,
        /// Last missing sequence number.
        to_seq: u64,
    },
    /// A refused frame.
    Refused {
        /// Sender of the refused frame.
        to: ConnId,
        /// Room named by the frame.
        room: RoomAlias,
        /// Why.
        reason: Refusal,
    },
}

/// One logged payload.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LogItem {
    /// Running index in the room's log.
    pub index: u64,
    /// The publisher's tag.
    pub from: SenderTag,
    /// The publisher's sequence number.
    pub seq: u64,
    /// The payload as published.
    pub payload: Vec<u8>,
}

/// A range read from a room's log.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LogRange {
    /// Index of the oldest payload still kept.
    pub first_index: u64,
    /// The payloads in the range that are still kept.
    pub items: Vec<LogItem>,
}

#[derive(Clone, Debug)]
struct Room {
    token: RoomToken,
    members: BTreeMap<ConnId, SenderTag>,
    next_seq: BTreeMap<SenderTag, u64>,
    held: BTreeMap<SenderTag, BTreeMap<u64, Vec<u8>>>,
    rate: BTreeMap<SenderTag, (u64, u32)>,
    log: VecDeque<LogItem>,
    next_index: u64,
    last_active: u64,
}

/// The anonymous relay's state: its limits and the open rooms.
#[derive(Clone, Debug)]
pub struct AnonymousRelayCore {
    limits: Limits,
    rooms: BTreeMap<RoomAlias, Room>,
}

/// Whether two tokens are equal, in time that does not depend on where they differ
fn same(a: &RoomToken, b: &RoomToken) -> bool {
    a.0.iter()
        .zip(b.0.iter())
        .fold(0u8, |acc, (x, y)| acc | (x ^ y))
        == 0
}

impl AnonymousRelayCore {
    /// A relay with no rooms.
    #[must_use]
    pub const fn new(limits: Limits) -> Self {
        Self {
            limits,
            rooms: BTreeMap::new(),
        }
    }

    /// Bind connection `conn` to `room` under `token`, publishing as `tag`; the first bind of an alias
    /// opens the room under its token.
    pub fn bind(
        &mut self,
        conn: ConnId,
        room: RoomAlias,
        token: RoomToken,
        tag: SenderTag,
        now: u64,
    ) -> Vec<Out> {
        let refuse = |reason| {
            vec![Out::Refused {
                to: conn,
                room,
                reason,
            }]
        };
        let joining = self
            .rooms
            .get(&room)
            .is_none_or(|r| !r.members.contains_key(&conn));
        if joining
            && self
                .rooms
                .values()
                .filter(|r| r.members.contains_key(&conn))
                .count()
                >= self.limits.max_rooms_per_conn
        {
            return refuse(Refusal::ConnRoomLimit);
        }
        if !self.rooms.contains_key(&room) {
            if self.rooms.len() >= self.limits.max_rooms && !self.make_way() {
                return refuse(Refusal::TooManyRooms);
            }
            self.rooms.insert(
                room,
                Room {
                    token,
                    members: BTreeMap::new(),
                    next_seq: BTreeMap::new(),
                    held: BTreeMap::new(),
                    rate: BTreeMap::new(),
                    log: VecDeque::new(),
                    next_index: 0,
                    last_active: now,
                },
            );
        }
        let max_members = self.limits.max_members;
        let max_tags = self.limits.max_tags_per_room;
        let Some(r) = self.rooms.get_mut(&room) else {
            return refuse(Refusal::NoRoom);
        };
        if !same(&r.token, &token) {
            return refuse(Refusal::BadToken);
        }
        if r.members.get(&conn).is_some_and(|held| *held != tag) {
            return refuse(Refusal::TagChange);
        }
        if r.members.iter().any(|(c, t)| *t == tag && *c != conn) {
            return refuse(Refusal::TagInUse);
        }
        if !r.members.contains_key(&conn) && r.members.len() >= max_members {
            return refuse(Refusal::RoomFull);
        }
        if !r.knows(tag) && r.tag_count() >= max_tags {
            r.prune_tags();
            if r.tag_count() >= max_tags {
                return refuse(Refusal::TooManyTags);
            }
        }
        r.members.insert(conn, tag);
        r.last_active = now;
        vec![Out::Bound { to: conn, room }]
    }

    /// Drop the empty room idle longest; whether one was dropped
    fn make_way(&mut self) -> bool {
        let oldest = self
            .rooms
            .iter()
            .filter(|(_, r)| r.members.is_empty())
            .min_by_key(|(alias, r)| (r.last_active, **alias))
            .map(|(alias, _)| *alias);
        oldest.is_some_and(|alias| self.rooms.remove(&alias).is_some())
    }

    /// Handle a payload published by connection `conn` with sequence number `seq`.
    pub fn publish(
        &mut self,
        conn: ConnId,
        room: RoomAlias,
        seq: u64,
        payload: Vec<u8>,
        now: u64,
    ) -> Vec<Out> {
        let refuse = |reason| {
            vec![Out::Refused {
                to: conn,
                room,
                reason,
            }]
        };
        let limits = self.limits;
        let Some(r) = self.rooms.get_mut(&room) else {
            return refuse(Refusal::NoRoom);
        };
        let Some(&from) = r.members.get(&conn) else {
            return refuse(Refusal::NotBound);
        };
        if payload.len() > limits.max_payload {
            return refuse(Refusal::TooLarge);
        }
        // the sequence space is 0..=u64::MAX - 1 and a room forwards at most u64::MAX payloads: refuse,
        // before anything changes, a publish that would need a number past either
        if seq == u64::MAX {
            return refuse(Refusal::Exhausted);
        }
        let next = r.next_seq.get(&from).copied().unwrap_or(0);
        if seq == next && r.next_index.checked_add(r.run_from(from, seq)).is_none() {
            return refuse(Refusal::Exhausted);
        }
        let window = limits.window.max(1);
        let slot = r.rate.entry(from).or_insert((now, 0));
        if now >= slot.0.saturating_add(window) {
            *slot = (now, 0);
        }
        if slot.1 >= limits.max_publishes_per_window {
            return refuse(Refusal::RateLimited);
        }
        slot.1 += 1;
        r.last_active = now;
        if seq < next {
            return Vec::new();
        }
        if seq > next {
            let held = r.held.entry(from).or_default();
            if held.contains_key(&seq) {
                return Vec::new();
            }
            if held.len() >= limits.max_pending {
                return refuse(Refusal::TooFarAhead);
            }
            held.insert(seq, payload);
            return vec![Out::Resend {
                to: conn,
                room,
                from: next,
                to_seq: seq - 1,
            }];
        }
        // seq == next: forward it and everything held right after it (the check above keeps the log
        // index within u64; seq is below u64::MAX and so are held numbers, so `s + 1` does not wrap)
        let mut out = Vec::new();
        let mut s = seq;
        let mut p = payload;
        loop {
            r.forward(limits.max_log, conn, from, room, s, p, &mut out);
            let Some(after) = s.checked_add(1) else {
                break;
            };
            s = after;
            match r.held.get_mut(&from).and_then(|h| h.remove(&s)) {
                Some(next_p) => p = next_p,
                None => break,
            }
        }
        r.next_seq.insert(from, s);
        out
    }

    /// Read log indices `from..to` (end exclusive) of a room, for a bound connection.
    ///
    /// # Errors
    ///
    /// The room is not open or the connection is not bound to it.
    pub fn log_range(
        &self,
        conn: ConnId,
        room: RoomAlias,
        from: u64,
        to: u64,
    ) -> Result<LogRange, Refusal> {
        let r = self.rooms.get(&room).ok_or(Refusal::NoRoom)?;
        if !r.members.contains_key(&conn) {
            return Err(Refusal::NotBound);
        }
        let first_index = r.log.front().map_or(r.next_index, |i| i.index);
        let items = r
            .log
            .iter()
            .filter(|i| i.index >= from && i.index < to)
            .cloned()
            .collect();
        Ok(LogRange { first_index, items })
    }

    /// How many sender tags `room` keeps state for (0 for a room not open).
    #[must_use]
    pub fn tags_held(&self, room: RoomAlias) -> usize {
        self.rooms.get(&room).map_or(0, |r| {
            let mut tags: std::collections::BTreeSet<SenderTag> =
                r.next_seq.keys().copied().collect();
            tags.extend(r.held.keys().copied());
            tags.extend(r.members.values().copied());
            tags.len()
        })
    }

    /// Unbind connection `conn` from every room (its transport closed); the rooms and their logs stay.
    pub fn close(&mut self, conn: ConnId) {
        for r in self.rooms.values_mut() {
            r.members.remove(&conn);
        }
    }
}

impl Room {
    /// The tags this room keeps state for: held by a connection, or with a sequence, held payloads or a rate
    fn tags(&self) -> std::collections::BTreeSet<SenderTag> {
        let mut tags: std::collections::BTreeSet<SenderTag> =
            self.members.values().copied().collect();
        tags.extend(self.next_seq.keys().copied());
        tags.extend(self.held.keys().copied());
        tags.extend(self.rate.keys().copied());
        tags
    }

    /// How many payloads a publish of `seq` by `tag` forwards: itself and the held ones right after it
    fn run_from(&self, tag: SenderTag, seq: u64) -> u64 {
        let mut run = 1u64;
        if let Some(held) = self.held.get(&tag) {
            let mut want = seq.checked_add(1);
            for (&s, _) in held.range(seq.saturating_add(1)..) {
                if Some(s) != want {
                    break;
                }
                run += 1;
                want = s.checked_add(1);
            }
        }
        run
    }

    fn tag_count(&self) -> usize {
        self.tags().len()
    }

    fn knows(&self, tag: SenderTag) -> bool {
        self.members.values().any(|t| *t == tag)
            || self.next_seq.contains_key(&tag)
            || self.held.contains_key(&tag)
            || self.rate.contains_key(&tag)
    }

    /// Forget the tags no connection holds and with nothing held back (their sequence and rate go with them:
    /// a member that comes back under such a tag starts its sequence again at 0)
    fn prune_tags(&mut self) {
        let held_by_member: std::collections::BTreeSet<SenderTag> =
            self.members.values().copied().collect();
        let gone: Vec<SenderTag> = self
            .tags()
            .into_iter()
            .filter(|t| {
                !held_by_member.contains(t) && self.held.get(t).is_none_or(BTreeMap::is_empty)
            })
            .collect();
        for t in gone {
            self.next_seq.remove(&t);
            self.held.remove(&t);
            self.rate.remove(&t);
        }
    }

    /// Send one payload to every bound connection but the publisher's, and log it
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &mut self,
        max_log: usize,
        publisher: ConnId,
        from: SenderTag,
        room: RoomAlias,
        seq: u64,
        payload: Vec<u8>,
        out: &mut Vec<Out>,
    ) {
        for &to in self.members.keys() {
            if to != publisher {
                out.push(Out::Forward {
                    to,
                    room,
                    from,
                    seq,
                    payload: payload.clone(),
                });
            }
        }
        self.log.push_back(LogItem {
            index: self.next_index,
            from,
            seq,
            payload,
        });
        self.next_index += 1;
        while self.log.len() > max_log {
            self.log.pop_front();
        }
    }
}
