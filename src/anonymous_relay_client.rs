//! The anonymous relay's client: what a member sends to the relay and what it makes of the relay's frames.
//! Frames in, frames out, no I/O ([`RelayClient`]).
//!
//! A [`RelayClient`] is one member of one room: the room's alias and token for the epoch, the sender tag it
//! chose for the room, and the next sequence number it will publish under that tag. It builds the frames
//! of [`crate::anonymous_relay_wire`] and reads the relay's answers ([`RelayClient::on_frame`]); the host
//! carries the frames over whatever transport it has (the blocking [`RelayConnection`] of the feature
//! `anonymous-relay-client`, or its own).
//!
//! # Payloads kept for a resend
//!
//! The relay acknowledges no publish: [`crate::anonymous_relay::Out`] has no frame that tells a publisher
//! a payload was taken. What it does send is a [`Frame::Resend`] when a publish arrives ahead of the next
//! sequence number it expects (an earlier publish was lost, for example on a connection that closed), and
//! its log, which a member can read. So the client keeps every payload it published until it knows the
//! relay has it, and answers a resend from what it kept. A kept payload is released when:
//!
//! - a [`Frame::Resend`] asks from sequence number `from`: the relay forwarded everything below `from`
//!   (it forwards a tag's payloads only in order), so every kept payload below `from` is released;
//! - a [`Frame::Log`] holds an item under this client's tag with sequence number `s`: the relay logged it,
//!   so every kept payload up to `s` is released;
//! - the host calls [`RelayClient::release_through`] because it learned by other means (for example an
//!   end-to-end receipt from the other members) that the relay forwarded up to that number.
//!
//! The number of kept payloads is bounded ([`RelayClient::with_max_unacked`], default
//! [`DEFAULT_MAX_UNACKED`]). When the bound is reached, [`RelayClient::publish`] refuses the next payload
//! with [`ClientError::Full`] and builds no frame (no sequence number is used): dropping the oldest kept
//! payload instead would leave a payload the relay may still ask for and the client can no longer give,
//! which stops that tag's delivery at the relay for good. A host that gets `Full` reads the log
//! ([`RelayClient::read_log`]) or waits for the other members' receipts, and publishes again.
//!
//! After every bind (a new connection, or a bind again on the same one) [`Event::Bound`] carries the kept
//! payloads as publish frames to send again: a payload that was lost with the old connection reaches the
//! relay, and one it already has is dropped there as a duplicate.
//!
//! A resend that asks for a payload no longer kept is [`ClientError::Unavailable`]. It happens when the
//! relay forgot the tag (a tag no connection holds makes way when the room keeps too many, and its sequence
//! starts again at 0) while the client kept counting: the host starts again under a new tag.
//!
//! # Refusals
//!
//! [`classify`] maps each [`Refusal`] to what a client does about it ([`Action`]); [`Event::Refused`]
//! carries both.

use std::collections::VecDeque;

use crate::anonymous_relay::{LogItem, Refusal, RoomAlias, RoomToken, SenderTag};
use crate::anonymous_relay_wire::Frame;

#[cfg(feature = "anonymous-relay-client")]
mod connection;
#[cfg(feature = "anonymous-relay-client")]
mod http;

#[cfg(feature = "anonymous-relay-client")]
pub use connection::{RelayConnection, RelayConnectionError, MAX_FRAME_BYTES};
#[cfg(feature = "anonymous-relay-client")]
pub use http::{fetch_relay_cert, FetchCertError, MAX_RELAY_CERT_RESPONSE_HEAD};

/// How many published payloads a [`RelayClient`] keeps for a resend unless set otherwise
/// ([`RelayClient::with_max_unacked`]).
pub const DEFAULT_MAX_UNACKED: usize = 64;

/// What a client does about a refusal ([`classify`]).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Action {
    /// Send the same frame again later, on the same connection: the condition passes with time (a rate
    /// window, a tag another connection still holds, a gap the relay is waiting for).
    RetryLater,
    /// Bind the room again ([`RelayClient::bind`]), then go on: the relay does not hold this connection
    /// in the room (the room was dropped, or the binding was).
    Rebind,
    /// Move to the next epoch's room (a new alias, token and tag): this room is not the group's any more,
    /// or its numbers are used up.
    NextEpoch,
    /// Reach the members another way (another relay, a direct route): this relay cannot take the room or
    /// the member now.
    FallBack,
    /// A frame this client should never have sent, or limits that do not match the relay's: a defect or a
    /// configuration error to report, not to retry.
    Internal,
}

/// What a client does about `reason`.
///
/// | Refusal | Action | Why |
/// |---|---|---|
/// | `NoRoom` | `Rebind` | the room is not open (the relay dropped it or restarted without it); the first bind opens it again |
/// | `BadToken` | `NextEpoch` | another token holds the alias: the group moved on, or the room made way and was opened by someone else |
/// | `NotBound` | `Rebind` | the relay does not hold this connection in the room |
/// | `TagInUse` | `RetryLater` | another connection holds the tag, usually this member's previous one not yet closed by the relay |
/// | `TooLarge` | `Internal` | the client checks the size before it publishes, so the relay's limit is smaller than the client's (or a log read's budget is smaller than a payload) |
/// | `TooFarAhead` | `RetryLater` | the relay holds back as many payloads as it may while it waits for a gap; it asks for the gap with a resend |
/// | `TooManyRooms` | `FallBack` | the relay is full of rooms in use |
/// | `RoomFull` | `FallBack` | the room has as many members as the relay allows |
/// | `RateLimited` | `RetryLater` | the tag reached its publishes for the window |
/// | `ConnRoomLimit` | `Internal` | a client binds one room per connection |
/// | `TagChange` | `Internal` | a client keeps its tag on a connection |
/// | `TooManyTags` | `RetryLater` | the room keeps as many tags as it may; a tag makes way when no connection holds it |
/// | `Exhausted` | `NextEpoch` | the tag's sequence numbers or the room's log indices are used up |
///
/// The match has no catch-all arm, so a refusal added to [`Refusal`] does not build until it is
/// classified here.
#[must_use]
pub const fn classify(reason: Refusal) -> Action {
    match reason {
        Refusal::NoRoom | Refusal::NotBound => Action::Rebind,
        Refusal::BadToken | Refusal::Exhausted => Action::NextEpoch,
        Refusal::TagInUse | Refusal::TooFarAhead | Refusal::RateLimited | Refusal::TooManyTags => {
            Action::RetryLater
        }
        Refusal::TooLarge | Refusal::ConnRoomLimit | Refusal::TagChange => Action::Internal,
        Refusal::TooManyRooms | Refusal::RoomFull => Action::FallBack,
    }
}

/// What a relay frame meant ([`RelayClient::on_frame`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Event {
    /// The connection is bound to the room. `replay` are the kept payloads as publish frames, in sequence
    /// order, to send now (the relay drops the ones it already has).
    Bound {
        /// The publish frames to send again.
        replay: Vec<Frame>,
    },
    /// A payload another member published.
    Forward {
        /// The publisher's tag.
        from: SenderTag,
        /// The publisher's sequence number.
        seq: u64,
        /// The payload as published.
        payload: Vec<u8>,
    },
    /// The relay asked for payloads again: the publish frames to send, in sequence order.
    Resend(Vec<Frame>),
    /// A page of the room's log.
    LogPage {
        /// Index of the oldest payload the relay still keeps.
        first_index: u64,
        /// The payloads of the page, in index order.
        items: Vec<LogItem>,
        /// The read of the next page (from the last index + 1), or `None` when this page was empty and
        /// the log is read to its end.
        next: Option<Frame>,
    },
    /// The relay refused a frame.
    Refused {
        /// Why.
        reason: Refusal,
        /// What to do about it ([`classify`]).
        action: Action,
    },
}

/// Why the client built no frame or could not use one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ClientError {
    /// The payload is longer than the client's `max_payload`.
    TooLarge {
        /// The payload's length.
        len: usize,
        /// The limit.
        max: usize,
    },
    /// The tag's sequence numbers are used up (the last one is `u64::MAX - 1`, as the relay counts).
    Exhausted,
    /// The client keeps as many payloads for a resend as it may; see the module documentation.
    Full {
        /// The bound.
        limit: usize,
    },
    /// The relay asked for payloads the client no longer keeps (`from..=to_seq`); the host starts again
    /// under a new tag.
    Unavailable {
        /// First sequence number asked for.
        from: u64,
        /// Last sequence number asked for.
        to_seq: u64,
    },
    /// The frame names another room.
    OtherRoom(RoomAlias),
    /// A frame only a client sends (`Bind`, `Publish`, `LogRange`, `Close`), or a resend whose range is
    /// empty: not an answer of the relay.
    Unexpected,
}

impl core::fmt::Display for ClientError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::TooLarge { len, max } => write!(f, "payload of {len} bytes over the limit {max}"),
            Self::Exhausted => f.write_str("the tag's sequence numbers are used up"),
            Self::Full { limit } => write!(f, "{limit} payloads kept for a resend already"),
            Self::Unavailable { from, to_seq } => {
                write!(f, "the relay asked for {from}..={to_seq}, no longer kept")
            }
            Self::OtherRoom(room) => write!(f, "a frame for another room {room}"),
            Self::Unexpected => f.write_str("not a frame the relay sends"),
        }
    }
}

impl std::error::Error for ClientError {}

/// One member of one room of the anonymous relay, without I/O.
#[derive(Clone, Debug)]
pub struct RelayClient {
    room: RoomAlias,
    token: RoomToken,
    tag: SenderTag,
    next_seq: u64,
    max_payload: usize,
    max_unacked: usize,
    /// Published and not known to be at the relay, in sequence order
    unacked: VecDeque<(u64, Vec<u8>)>,
}

impl RelayClient {
    /// A member of `room` entering with `token`, publishing as `tag` from sequence number `next_seq` (0 for
    /// a new tag; the host keeps the tag and the number across restarts within the room), with payloads of
    /// at most `max_payload` bytes (at most the relay's [`Limits::max_payload`](crate::anonymous_relay::Limits::max_payload)).
    #[must_use]
    pub const fn new(
        room: RoomAlias,
        token: RoomToken,
        tag: SenderTag,
        next_seq: u64,
        max_payload: usize,
    ) -> Self {
        Self {
            room,
            token,
            tag,
            next_seq,
            max_payload,
            max_unacked: DEFAULT_MAX_UNACKED,
            unacked: VecDeque::new(),
        }
    }

    /// The same client keeping at most `max_unacked` payloads for a resend (see the module documentation).
    #[must_use]
    pub const fn with_max_unacked(mut self, max_unacked: usize) -> Self {
        self.max_unacked = max_unacked;
        self
    }

    /// The room.
    #[must_use]
    pub const fn room(&self) -> RoomAlias {
        self.room
    }

    /// The tag this client publishes as.
    #[must_use]
    pub const fn tag(&self) -> SenderTag {
        self.tag
    }

    /// The sequence number the next publish takes (what the host keeps across a restart).
    #[must_use]
    pub const fn next_seq(&self) -> u64 {
        self.next_seq
    }

    /// The payloads kept for a resend: sequence number and payload, in sequence order.
    pub fn unacked(&self) -> impl Iterator<Item = (u64, &[u8])> {
        self.unacked.iter().map(|(s, p)| (*s, p.as_slice()))
    }

    /// The frame that binds the connection to the room.
    #[must_use]
    pub const fn bind(&self) -> Frame {
        Frame::Bind {
            room: self.room,
            token: self.token,
            tag: self.tag,
        }
    }

    /// The frame that publishes `payload` under the next sequence number; the payload is kept until the
    /// relay is known to have it.
    ///
    /// # Errors
    ///
    /// [`ClientError::TooLarge`] over `max_payload`, [`ClientError::Exhausted`] when the tag's numbers are
    /// used up, [`ClientError::Full`] when the client keeps as many payloads as it may. No frame is built
    /// and no sequence number is used in each case.
    pub fn publish(&mut self, payload: Vec<u8>) -> Result<Frame, ClientError> {
        if payload.len() > self.max_payload {
            return Err(ClientError::TooLarge {
                len: payload.len(),
                max: self.max_payload,
            });
        }
        // the relay's sequence space is 0..=u64::MAX - 1 (it refuses u64::MAX as Exhausted)
        let seq = self.next_seq;
        if seq == u64::MAX {
            return Err(ClientError::Exhausted);
        }
        if self.unacked.len() >= self.max_unacked {
            return Err(ClientError::Full {
                limit: self.max_unacked,
            });
        }
        self.next_seq = seq + 1;
        let frame = self.publish_frame(seq, payload.clone());
        self.unacked.push_back((seq, payload));
        Ok(frame)
    }

    const fn publish_frame(&self, seq: u64, payload: Vec<u8>) -> Frame {
        Frame::Publish {
            room: self.room,
            seq,
            payload,
        }
    }

    /// The frame that reads the room's log from index `from` to its end, one page per answer
    /// ([`Event::LogPage`] carries the read of the next page).
    #[must_use]
    pub const fn read_log(&self, from: u64) -> Frame {
        Frame::LogRange {
            room: self.room,
            from_index: from,
            to_index: u64::MAX,
        }
    }

    /// The frame that leaves the room on this connection.
    #[must_use]
    pub const fn leave(&self) -> Frame {
        Frame::Close { room: self.room }
    }

    /// Release the kept payloads up to and including sequence number `seq` (the relay is known to have
    /// them).
    pub fn release_through(&mut self, seq: u64) {
        while self.unacked.front().is_some_and(|(s, _)| *s <= seq) {
            self.unacked.pop_front();
        }
    }

    /// The kept payloads `from..=to_seq` as publish frames, or `None` when one of them is not kept
    fn kept_range(&self, from: u64, to_seq: u64) -> Option<Vec<Frame>> {
        let mut frames = Vec::new();
        let mut want = from;
        for (s, p) in self
            .unacked
            .iter()
            .filter(|(s, _)| (from..=to_seq).contains(s))
        {
            if *s != want {
                return None;
            }
            frames.push(self.publish_frame(*s, p.clone()));
            want = s.checked_add(1)?;
        }
        // every number of the range was kept (`want` went past `to_seq`, and the range is not empty)
        (want > to_seq && !frames.is_empty()).then_some(frames)
    }

    /// What a frame from the relay means.
    ///
    /// # Errors
    ///
    /// [`ClientError::OtherRoom`] for a frame of another room, [`ClientError::Unexpected`] for a frame only
    /// a client sends or an empty resend range, [`ClientError::Unavailable`] for a resend of payloads no
    /// longer kept.
    pub fn on_frame(&mut self, frame: Frame) -> Result<Event, ClientError> {
        let room = match &frame {
            Frame::Bound { room }
            | Frame::Forward { room, .. }
            | Frame::Resend { room, .. }
            | Frame::Refused { room, .. }
            | Frame::Log { room, .. } => *room,
            Frame::Bind { .. }
            | Frame::Publish { .. }
            | Frame::LogRange { .. }
            | Frame::Close { .. } => return Err(ClientError::Unexpected),
        };
        if room != self.room {
            return Err(ClientError::OtherRoom(room));
        }
        match frame {
            Frame::Bound { .. } => Ok(Event::Bound {
                replay: self
                    .unacked
                    .iter()
                    .map(|(s, p)| self.publish_frame(*s, p.clone()))
                    .collect(),
            }),
            Frame::Forward {
                from, seq, payload, ..
            } => Ok(Event::Forward { from, seq, payload }),
            Frame::Resend { from, to_seq, .. } => {
                if from > to_seq {
                    return Err(ClientError::Unexpected);
                }
                let frames = self
                    .kept_range(from, to_seq)
                    .ok_or(ClientError::Unavailable { from, to_seq })?;
                // the relay forwards a tag's payloads in order: it has every one below `from`
                if let Some(below) = from.checked_sub(1) {
                    self.release_through(below);
                }
                Ok(Event::Resend(frames))
            }
            Frame::Refused { reason, .. } => Ok(Event::Refused {
                reason,
                action: classify(reason),
            }),
            Frame::Log {
                first_index, items, ..
            } => {
                let own = items
                    .iter()
                    .filter(|(_, tag, _, _)| *tag == self.tag)
                    .map(|(_, _, seq, _)| *seq)
                    .max();
                if let Some(seq) = own {
                    self.release_through(seq);
                }
                let next = match items.last() {
                    Some((index, ..)) => index.checked_add(1).map(|from| self.read_log(from)),
                    None => None,
                };
                let items = items
                    .into_iter()
                    .map(|(index, from, seq, payload)| LogItem {
                        index,
                        from,
                        seq,
                        payload,
                    })
                    .collect();
                Ok(Event::LogPage {
                    first_index,
                    items,
                    next,
                })
            }
            // client frames were refused above
            Frame::Bind { .. }
            | Frame::Publish { .. }
            | Frame::LogRange { .. }
            | Frame::Close { .. } => Err(ClientError::Unexpected),
        }
    }
}
