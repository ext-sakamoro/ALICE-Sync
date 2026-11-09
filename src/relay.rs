//! Room relay core: frames in, frames out, no I/O and no simulation.
//!
//! A room is a fixed set of members (device ids). A member publishes payloads numbered by its own
//! sequence; the relay forwards each payload, byte for byte, to every other member. Per publisher,
//! payloads leave in sequence order: a payload ahead of the next expected one is held back (up to
//! [`Limits::max_pending`]) and the publisher is asked to resend the missing range; a duplicate is
//! dropped. Payloads are opaque (sealed end to end by the members); the relay reads only the
//! sequence number. Forwarded payloads are kept in a per-room log, in arrival order with a running
//! index, capped by count, so a reconnecting member or a late joiner can read a range.
//!
//! All maps are ordered (`BTreeMap` / `BTreeSet`), so the output for a given input sequence is the
//! same on every build.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// A member's device id: ALICE-Identity's [`DeviceId`](alice_identity::device_cert::DeviceId), the
/// 16 bytes its certificate carries (derived from the subject and the device key, the same value as the
/// member's [`crate::event::Origin`] and the key its revocation is listed under).
pub type PeerId = alice_identity::device_cert::DeviceId;

/// A room id.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RoomId(u64);

impl RoomId {
    /// A room id from its number.
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

impl core::fmt::Display for RoomId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Per-room limits set by whoever runs the relay.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Limits {
    /// Largest payload accepted, in bytes.
    pub max_payload: usize,
    /// Payloads kept in the log.
    pub max_log: usize,
    /// Payloads held back per publisher while waiting for a gap to fill.
    pub max_pending: usize,
}

/// Why the relay refused a frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Refusal {
    /// The room is not open on this relay.
    NoRoom,
    /// The sender is not a member of the room.
    NotMember,
    /// The payload is larger than [`Limits::max_payload`].
    TooLarge,
    /// The sender is more than [`Limits::max_pending`] payloads ahead of the next expected one.
    TooFarAhead,
}

/// Why a room could not be opened.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OpenError {
    /// Fewer than two members.
    TooFewMembers,
    /// The same member listed twice.
    DuplicateMember,
    /// A room with this id is already open.
    AlreadyOpen,
}

/// A frame the relay sends.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Out {
    /// A payload for `to`, published by `from` with `seq`.
    Forward {
        /// Recipient.
        to: PeerId,
        /// Room.
        room: RoomId,
        /// Publisher.
        from: PeerId,
        /// Publisher's sequence number.
        seq: u64,
        /// The payload as published.
        payload: Vec<u8>,
    },
    /// Ask `to` to publish `from..=to_seq` again.
    Resend {
        /// The publisher asked.
        to: PeerId,
        /// Room.
        room: RoomId,
        /// First missing sequence number.
        from: u64,
        /// Last missing sequence number.
        to_seq: u64,
    },
    /// A refused frame.
    Refused {
        /// Sender of the refused frame.
        to: PeerId,
        /// Room named by the frame.
        room: RoomId,
        /// Why.
        reason: Refusal,
    },
}

/// One logged payload.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LogItem {
    /// Running index in the room's log.
    pub index: u64,
    /// Publisher.
    pub from: PeerId,
    /// Publisher's sequence number.
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
    members: BTreeSet<PeerId>,
    limits: Limits,
    next_seq: BTreeMap<PeerId, u64>,
    held: BTreeMap<PeerId, BTreeMap<u64, Vec<u8>>>,
    log: VecDeque<LogItem>,
    next_index: u64,
}

/// The relay's state: the open rooms.
#[derive(Clone, Debug, Default)]
pub struct RelayCore {
    rooms: BTreeMap<RoomId, Room>,
}

impl RelayCore {
    /// A relay with no rooms.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Open a room with a fixed member list.
    ///
    /// # Errors
    ///
    /// Fewer than two members, a duplicate member, or the id already open.
    pub fn open_room(
        &mut self,
        id: RoomId,
        members: &[PeerId],
        limits: Limits,
    ) -> Result<(), OpenError> {
        let set: BTreeSet<PeerId> = members.iter().copied().collect();
        if set.len() != members.len() {
            return Err(OpenError::DuplicateMember);
        }
        if set.len() < 2 {
            return Err(OpenError::TooFewMembers);
        }
        if self.rooms.contains_key(&id) {
            return Err(OpenError::AlreadyOpen);
        }
        self.rooms.insert(
            id,
            Room {
                members: set,
                limits,
                next_seq: BTreeMap::new(),
                held: BTreeMap::new(),
                log: VecDeque::new(),
                next_index: 0,
            },
        );
        Ok(())
    }

    /// Handle a payload published by `from` with sequence number `seq`.
    pub fn publish(&mut self, from: PeerId, room: RoomId, seq: u64, payload: Vec<u8>) -> Vec<Out> {
        let refuse = |reason| {
            vec![Out::Refused {
                to: from,
                room,
                reason,
            }]
        };
        let Some(r) = self.rooms.get_mut(&room) else {
            return refuse(Refusal::NoRoom);
        };
        if !r.members.contains(&from) {
            return refuse(Refusal::NotMember);
        }
        if payload.len() > r.limits.max_payload {
            return refuse(Refusal::TooLarge);
        }
        let next = r.next_seq.get(&from).copied().unwrap_or(0);
        if seq < next {
            return Vec::new();
        }
        let mut out = Vec::new();
        if seq > next {
            let held = r.held.entry(from).or_default();
            if held.contains_key(&seq) {
                return Vec::new();
            }
            if held.len() >= r.limits.max_pending {
                return refuse(Refusal::TooFarAhead);
            }
            held.insert(seq, payload);
            out.push(Out::Resend {
                to: from,
                room,
                from: next,
                to_seq: seq - 1,
            });
            return out;
        }
        // seq == next: forward it and everything held right after it
        let mut s = seq;
        let mut p = payload;
        loop {
            Self::forward(r, from, room, s, p, &mut out);
            s += 1;
            match r.held.get_mut(&from).and_then(|h| h.remove(&s)) {
                Some(next_p) => p = next_p,
                None => break,
            }
        }
        r.next_seq.insert(from, s);
        out
    }

    fn forward(
        r: &mut Room,
        from: PeerId,
        room: RoomId,
        seq: u64,
        payload: Vec<u8>,
        out: &mut Vec<Out>,
    ) {
        for &to in &r.members {
            if to != from {
                out.push(Out::Forward {
                    to,
                    room,
                    from,
                    seq,
                    payload: payload.clone(),
                });
            }
        }
        r.log.push_back(LogItem {
            index: r.next_index,
            from,
            seq,
            payload,
        });
        r.next_index += 1;
        while r.log.len() > r.limits.max_log {
            r.log.pop_front();
        }
    }

    /// Read log indices `from..to` (end exclusive) of a room, for a member.
    ///
    /// # Errors
    ///
    /// The room is not open or the reader is not a member.
    pub fn log_range(
        &self,
        reader: PeerId,
        room: RoomId,
        from: u64,
        to: u64,
    ) -> Result<LogRange, Refusal> {
        let r = self.rooms.get(&room).ok_or(Refusal::NoRoom)?;
        if !r.members.contains(&reader) {
            return Err(Refusal::NotMember);
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
}
