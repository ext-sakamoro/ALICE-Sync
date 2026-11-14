//! The anonymous relay's wire frames ([`crate::anonymous_relay`]): what a client and the relay send each
//! other inside the relay hop's channel. No frame carries a device id.
//!
//! Layout: one tag byte (`Bind` 0x41, `Bound` 0x42, `Publish` 0x43, `Forward` 0x44, `Resend` 0x45,
//! `Refused` 0x46, `LogRange` 0x47, `Log` 0x48, `Close` 0x49), then the fields in order. A room is its 32-byte
//! alias, a token 32 bytes, a sender tag 16 bytes, integers big-endian fixed width, a payload a u32 length
//! then the bytes (at most [`MAX_FRAME`]), a refusal one byte (`NoRoom` 1, `BadToken` 2, `NotBound` 3,
//! `TagInUse` 4, `TooLarge` 5, `TooFarAhead` 6, `TooManyRooms` 7, `RoomFull` 8, `RateLimited` 9,
//! `ConnRoomLimit` 10, `TagChange` 11, `TooManyTags` 12, `Exhausted` 13). Decoding
//! is strict: the bytes must be exactly one frame.
//!
//! The tags lie outside those of the named relay's frames ([`crate::relay_wire`], [`NAMED_RELAY_TAGS`]): a
//! named relay frame given here is refused as [`WireError::NamedRelayFrame`], and a frame of this module
//! given to the named decoder is its `UnknownTag`, so neither relay takes the other's frames.

use crate::anonymous_relay::{Refusal, RoomAlias, RoomToken, SenderTag};

/// The largest payload a frame carries, bytes.
pub const MAX_FRAME: usize = crate::relay_wire::MAX_FRAME;

/// The tags of the named relay's frames ([`crate::relay_wire`]).
pub const NAMED_RELAY_TAGS: core::ops::RangeInclusive<u8> = 1..=8;

const BIND: u8 = 0x41;
const BOUND: u8 = 0x42;
const PUBLISH: u8 = 0x43;
const FORWARD: u8 = 0x44;
const RESEND: u8 = 0x45;
const REFUSED: u8 = 0x46;
const LOG_RANGE: u8 = 0x47;
const LOG: u8 = 0x48;
const CLOSE: u8 = 0x49;

/// One wire frame.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Frame {
    /// Client: bind this connection to the room `room`, entering with `token` and publishing as `tag`.
    Bind {
        /// Room.
        room: RoomAlias,
        /// The room's token.
        token: RoomToken,
        /// The publisher's tag in the room.
        tag: SenderTag,
    },
    /// Relay: the connection is bound to the room.
    Bound {
        /// Room.
        room: RoomAlias,
    },
    /// Client: publish `payload` with this publisher's sequence number `seq`.
    Publish {
        /// Room.
        room: RoomAlias,
        /// The publisher's sequence number.
        seq: u64,
        /// The payload (sealed end to end).
        payload: Vec<u8>,
    },
    /// Relay: a payload published under `from`.
    Forward {
        /// Room.
        room: RoomAlias,
        /// The publisher's tag.
        from: SenderTag,
        /// The publisher's sequence number.
        seq: u64,
        /// The payload as published.
        payload: Vec<u8>,
    },
    /// Relay: publish `from..=to_seq` again.
    Resend {
        /// Room.
        room: RoomAlias,
        /// First missing sequence number.
        from: u64,
        /// Last missing sequence number.
        to_seq: u64,
    },
    /// Relay: a frame was refused.
    Refused {
        /// Room named by the frame.
        room: RoomAlias,
        /// Why.
        reason: Refusal,
    },
    /// Client: read the room's log from `from_index` up to (not including) `to_index`.
    LogRange {
        /// Room.
        room: RoomAlias,
        /// First index asked for.
        from_index: u64,
        /// End of the range (exclusive).
        to_index: u64,
    },
    /// Relay: the logged payloads of a range: `(index, publisher tag, seq, payload)`.
    Log {
        /// Room.
        room: RoomAlias,
        /// Index of the oldest payload still kept.
        first_index: u64,
        /// The kept payloads of the range.
        items: Vec<(u64, SenderTag, u64, Vec<u8>)>,
    },
    /// Client: leave the room on this connection.
    Close {
        /// Room.
        room: RoomAlias,
    },
}

/// Why bytes are not a frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WireError {
    /// The bytes end inside the frame.
    Truncated,
    /// Bytes follow the frame.
    Trailing,
    /// A frame of the named relay ([`crate::relay_wire`]): this relay does not take it.
    NamedRelayFrame(u8),
    /// The tag is not a frame's.
    UnknownTag(u8),
    /// The refusal byte is not a refusal's.
    UnknownRefusal(u8),
    /// A payload length is over [`MAX_FRAME`].
    TooLarge,
}

const fn refusal_code(r: Refusal) -> u8 {
    match r {
        Refusal::NoRoom => 1,
        Refusal::BadToken => 2,
        Refusal::NotBound => 3,
        Refusal::TagInUse => 4,
        Refusal::TooLarge => 5,
        Refusal::TooFarAhead => 6,
        Refusal::TooManyRooms => 7,
        Refusal::RoomFull => 8,
        Refusal::RateLimited => 9,
        Refusal::ConnRoomLimit => 10,
        Refusal::TagChange => 11,
        Refusal::TooManyTags => 12,
        Refusal::Exhausted => 13,
    }
}

const fn refusal_of(code: u8) -> Option<Refusal> {
    Some(match code {
        1 => Refusal::NoRoom,
        2 => Refusal::BadToken,
        3 => Refusal::NotBound,
        4 => Refusal::TagInUse,
        5 => Refusal::TooLarge,
        6 => Refusal::TooFarAhead,
        7 => Refusal::TooManyRooms,
        8 => Refusal::RoomFull,
        9 => Refusal::RateLimited,
        10 => Refusal::ConnRoomLimit,
        11 => Refusal::TagChange,
        12 => Refusal::TooManyTags,
        13 => Refusal::Exhausted,
        _ => return None,
    })
}

fn put_payload(out: &mut Vec<u8>, p: &[u8]) {
    out.extend_from_slice(&u32::try_from(p.len()).unwrap_or(u32::MAX).to_be_bytes());
    out.extend_from_slice(p);
}

impl Frame {
    /// The frame as bytes.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        match self {
            Self::Bind { room, token, tag } => {
                out.push(BIND);
                out.extend_from_slice(room.as_ref());
                out.extend_from_slice(token.as_ref());
                out.extend_from_slice(tag.as_ref());
            }
            Self::Bound { room } => {
                out.push(BOUND);
                out.extend_from_slice(room.as_ref());
            }
            Self::Publish { room, seq, payload } => {
                out.push(PUBLISH);
                out.extend_from_slice(room.as_ref());
                out.extend_from_slice(&seq.to_be_bytes());
                put_payload(&mut out, payload);
            }
            Self::Forward {
                room,
                from,
                seq,
                payload,
            } => {
                out.push(FORWARD);
                out.extend_from_slice(room.as_ref());
                out.extend_from_slice(from.as_ref());
                out.extend_from_slice(&seq.to_be_bytes());
                put_payload(&mut out, payload);
            }
            Self::Resend { room, from, to_seq } => {
                out.push(RESEND);
                out.extend_from_slice(room.as_ref());
                out.extend_from_slice(&from.to_be_bytes());
                out.extend_from_slice(&to_seq.to_be_bytes());
            }
            Self::Refused { room, reason } => {
                out.push(REFUSED);
                out.extend_from_slice(room.as_ref());
                out.push(refusal_code(*reason));
            }
            Self::LogRange {
                room,
                from_index,
                to_index,
            } => {
                out.push(LOG_RANGE);
                out.extend_from_slice(room.as_ref());
                out.extend_from_slice(&from_index.to_be_bytes());
                out.extend_from_slice(&to_index.to_be_bytes());
            }
            Self::Log {
                room,
                first_index,
                items,
            } => {
                out.push(LOG);
                out.extend_from_slice(room.as_ref());
                out.extend_from_slice(&first_index.to_be_bytes());
                out.extend_from_slice(
                    &u32::try_from(items.len()).unwrap_or(u32::MAX).to_be_bytes(),
                );
                for (index, from, seq, payload) in items {
                    out.extend_from_slice(&index.to_be_bytes());
                    out.extend_from_slice(from.as_ref());
                    out.extend_from_slice(&seq.to_be_bytes());
                    put_payload(&mut out, payload);
                }
            }
            Self::Close { room } => {
                out.push(CLOSE);
                out.extend_from_slice(room.as_ref());
            }
        }
        out
    }

    /// Read exactly one frame from `bytes`.
    ///
    /// # Errors
    ///
    /// See [`WireError`].
    pub fn decode(bytes: &[u8]) -> Result<Self, WireError> {
        let mut r = Reader(bytes);
        let tag = r.u8()?;
        let frame = match tag {
            BIND => Self::Bind {
                room: r.room()?,
                token: RoomToken::from_raw(r.take()?),
                tag: SenderTag::from_raw(r.take()?),
            },
            BOUND => Self::Bound { room: r.room()? },
            PUBLISH => Self::Publish {
                room: r.room()?,
                seq: r.u64()?,
                payload: r.payload()?,
            },
            FORWARD => Self::Forward {
                room: r.room()?,
                from: SenderTag::from_raw(r.take()?),
                seq: r.u64()?,
                payload: r.payload()?,
            },
            RESEND => Self::Resend {
                room: r.room()?,
                from: r.u64()?,
                to_seq: r.u64()?,
            },
            REFUSED => {
                let room = r.room()?;
                let code = r.u8()?;
                Self::Refused {
                    room,
                    reason: refusal_of(code).ok_or(WireError::UnknownRefusal(code))?,
                }
            }
            LOG_RANGE => Self::LogRange {
                room: r.room()?,
                from_index: r.u64()?,
                to_index: r.u64()?,
            },
            LOG => {
                let room = r.room()?;
                let first_index = r.u64()?;
                let n = r.u32()?;
                let mut items = Vec::new();
                for _ in 0..n {
                    items.push((
                        r.u64()?,
                        SenderTag::from_raw(r.take()?),
                        r.u64()?,
                        r.payload()?,
                    ));
                }
                Self::Log {
                    room,
                    first_index,
                    items,
                }
            }
            CLOSE => Self::Close { room: r.room()? },
            t if NAMED_RELAY_TAGS.contains(&t) => return Err(WireError::NamedRelayFrame(t)),
            t => return Err(WireError::UnknownTag(t)),
        };
        if r.0.is_empty() {
            Ok(frame)
        } else {
            Err(WireError::Trailing)
        }
    }
}

struct Reader<'a>(&'a [u8]);

impl Reader<'_> {
    fn take<const N: usize>(&mut self) -> Result<[u8; N], WireError> {
        let (head, rest) = self
            .0
            .split_first_chunk::<N>()
            .ok_or(WireError::Truncated)?;
        self.0 = rest;
        Ok(*head)
    }

    fn u8(&mut self) -> Result<u8, WireError> {
        Ok(self.take::<1>()?[0])
    }

    fn u32(&mut self) -> Result<u32, WireError> {
        Ok(u32::from_be_bytes(self.take()?))
    }

    fn u64(&mut self) -> Result<u64, WireError> {
        Ok(u64::from_be_bytes(self.take()?))
    }

    fn room(&mut self) -> Result<RoomAlias, WireError> {
        Ok(RoomAlias::from_raw(self.take()?))
    }

    fn payload(&mut self) -> Result<Vec<u8>, WireError> {
        let n = usize::try_from(self.u32()?).map_err(|_| WireError::TooLarge)?;
        if n > MAX_FRAME {
            return Err(WireError::TooLarge);
        }
        if n > self.0.len() {
            return Err(WireError::Truncated);
        }
        let (p, rest) = self.0.split_at(n);
        self.0 = rest;
        Ok(p.to_vec())
    }
}
