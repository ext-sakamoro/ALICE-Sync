//! The relay's wire frames (what a client and the relay send each other inside the relay hop's
//! encrypted channel; the relay reads these frames, never the payloads).
//!
//! Layout: one tag byte (`Bind` 1, `Publish` 2, `Forward` 3, `Resend` 4, `Refused` 5, `LogRange` 6,
//! `Log` 7, `Close` 8), then the fields in order. Integers are big-endian fixed width, a room is
//! its u64, a peer id its 16 bytes (the device id), a payload a u32 length then the bytes (at most [`MAX_FRAME`]),
//! a refusal one byte (`NoRoom` 1, `NotMember` 2, `TooLarge` 3, `TooFarAhead` 4). Decoding is
//! strict: the bytes must be exactly one frame.

use crate::relay::{PeerId, Refusal, RoomId};

/// The largest payload a frame carries, bytes.
pub const MAX_FRAME: usize = 1 << 20;

/// One wire frame.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Frame {
    /// Client: bind this connection to a room it is a member of.
    Bind {
        /// Room.
        room: RoomId,
    },
    /// Client: publish a sealed payload with the client's sequence number.
    Publish {
        /// Room.
        room: RoomId,
        /// The publisher's sequence number.
        seq: u64,
        /// Sealed payload.
        payload: Vec<u8>,
    },
    /// Relay: a payload another member published.
    Forward {
        /// Room.
        room: RoomId,
        /// Publisher.
        from: PeerId,
        /// The publisher's sequence number.
        seq: u64,
        /// Sealed payload, as published.
        payload: Vec<u8>,
    },
    /// Relay: publish `from..=to_seq` again.
    Resend {
        /// Room.
        room: RoomId,
        /// First missing sequence number.
        from: u64,
        /// Last missing sequence number.
        to_seq: u64,
    },
    /// Relay: a frame was refused.
    Refused {
        /// Room named by the refused frame.
        room: RoomId,
        /// Why.
        reason: Refusal,
    },
    /// Client: read the room's log from `from_index` up to (not including) `to_index`.
    LogRange {
        /// Room.
        room: RoomId,
        /// First index asked for.
        from_index: u64,
        /// End of the range (exclusive).
        to_index: u64,
    },
    /// Relay: the logged payloads of a range: `(index, publisher, seq, payload)`.
    Log {
        /// Room.
        room: RoomId,
        /// Index of the oldest payload still kept.
        first_index: u64,
        /// The kept payloads of the range.
        items: Vec<(u64, PeerId, u64, Vec<u8>)>,
    },
    /// Client: leave the room on this connection.
    Close {
        /// Room.
        room: RoomId,
    },
}

/// Why bytes are not a frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WireError {
    /// The bytes end inside the frame.
    Truncated,
    /// Bytes follow the frame.
    Trailing,
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
        Refusal::NotMember => 2,
        Refusal::TooLarge => 3,
        Refusal::TooFarAhead => 4,
    }
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
        let room = |out: &mut Vec<u8>, r: &RoomId| out.extend_from_slice(&r.as_u64().to_be_bytes());
        match self {
            Self::Bind { room: r } => {
                out.push(1);
                room(&mut out, r);
            }
            Self::Publish {
                room: r,
                seq,
                payload,
            } => {
                out.push(2);
                room(&mut out, r);
                out.extend_from_slice(&seq.to_be_bytes());
                put_payload(&mut out, payload);
            }
            Self::Forward {
                room: r,
                from,
                seq,
                payload,
            } => {
                out.push(3);
                room(&mut out, r);
                out.extend_from_slice(from.as_ref());
                out.extend_from_slice(&seq.to_be_bytes());
                put_payload(&mut out, payload);
            }
            Self::Resend {
                room: r,
                from,
                to_seq,
            } => {
                out.push(4);
                room(&mut out, r);
                out.extend_from_slice(&from.to_be_bytes());
                out.extend_from_slice(&to_seq.to_be_bytes());
            }
            Self::Refused { room: r, reason } => {
                out.push(5);
                room(&mut out, r);
                out.push(refusal_code(*reason));
            }
            Self::LogRange {
                room: r,
                from_index,
                to_index,
            } => {
                out.push(6);
                room(&mut out, r);
                out.extend_from_slice(&from_index.to_be_bytes());
                out.extend_from_slice(&to_index.to_be_bytes());
            }
            Self::Log {
                room: r,
                first_index,
                items,
            } => {
                out.push(7);
                room(&mut out, r);
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
            Self::Close { room: r } => {
                out.push(8);
                room(&mut out, r);
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
            1 => Self::Bind { room: r.room()? },
            2 => Self::Publish {
                room: r.room()?,
                seq: r.u64()?,
                payload: r.payload()?,
            },
            3 => Self::Forward {
                room: r.room()?,
                from: r.peer()?,
                seq: r.u64()?,
                payload: r.payload()?,
            },
            4 => Self::Resend {
                room: r.room()?,
                from: r.u64()?,
                to_seq: r.u64()?,
            },
            5 => {
                let room = r.room()?;
                let code = r.u8()?;
                let reason = match code {
                    1 => Refusal::NoRoom,
                    2 => Refusal::NotMember,
                    3 => Refusal::TooLarge,
                    4 => Refusal::TooFarAhead,
                    c => return Err(WireError::UnknownRefusal(c)),
                };
                Self::Refused { room, reason }
            }
            6 => Self::LogRange {
                room: r.room()?,
                from_index: r.u64()?,
                to_index: r.u64()?,
            },
            7 => {
                let room = r.room()?;
                let first_index = r.u64()?;
                let n = r.u32()?;
                let mut items = Vec::new();
                for _ in 0..n {
                    items.push((r.u64()?, r.peer()?, r.u64()?, r.payload()?));
                }
                Self::Log {
                    room,
                    first_index,
                    items,
                }
            }
            8 => Self::Close { room: r.room()? },
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

    fn room(&mut self) -> Result<RoomId, WireError> {
        Ok(RoomId::from_raw(self.u64()?))
    }

    fn peer(&mut self) -> Result<PeerId, WireError> {
        Ok(PeerId::from_raw(self.take()?))
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
