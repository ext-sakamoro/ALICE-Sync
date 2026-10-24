/*
    ALICE-Sync
    Copyright (C) 2026 Moroya Sakamoto

    License: LicenseRef-Proprietary (see LICENSE-Proprietary)
*/

//! `Event::to_bytes` / `Message::to_bytes` in the alice-serial format: each struct is an array of
//! its fields in declaration order, each enum an array of its variant number and then its fields,
//! unsigned integers as `U64`, signed ones as `I64`, fixed byte arrays and signatures as `Bytes`
//!
//! Decoding is strict: the exact number of fields, a known variant number, every integer inside
//! its type's range and every byte array of its exact length, or nothing

use alice_serial::Value;

use crate::authorship::Checkpoint;
use crate::event::{Event, EventId, EventKind, Origin, SeqNum};
use crate::protocol::Message;
use crate::{NodeId, SyncError, WorldHash};

pub(crate) fn encode(value: &Value) -> Result<Vec<u8>, SyncError> {
    alice_serial::encode(value).map_err(|e| SyncError::SerializationError(e.to_string()))
}

pub(crate) fn decode(bytes: &[u8]) -> Option<Value> {
    alice_serial::decode(bytes).ok()
}

fn u(n: impl Into<u64>) -> Value {
    Value::U64(n.into())
}

fn i(n: impl Into<i64>) -> Value {
    Value::I64(n.into())
}

fn fields(v: &Value, n: usize) -> Option<&[Value]> {
    match v {
        Value::Array(a) if a.len() == n => Some(a),
        _ => None,
    }
}

fn to_u<T: TryFrom<u64>>(v: &Value) -> Option<T> {
    match v {
        Value::U64(n) => T::try_from(*n).ok(),
        _ => None,
    }
}

fn to_i<T: TryFrom<i64>>(v: &Value) -> Option<T> {
    match v {
        Value::I64(n) => T::try_from(*n).ok(),
        _ => None,
    }
}

fn to_array<const N: usize>(v: &Value) -> Option<[u8; N]> {
    match v {
        Value::Bytes(b) => b.as_slice().try_into().ok(),
        _ => None,
    }
}

fn i16s(v: &Value) -> Option<[i16; 3]> {
    let f = fields(v, 3)?;
    Some([to_i(&f[0])?, to_i(&f[1])?, to_i(&f[2])?])
}

fn i16s_value(a: [i16; 3]) -> Value {
    Value::Array(a.iter().map(|&x| i(x)).collect())
}

pub(crate) fn event_value(e: &Event) -> Value {
    let kind = match &e.kind {
        EventKind::Motion { entity, delta } => vec![u(0u8), u(*entity), i16s_value(*delta)],
        EventKind::Spawn { entity, kind, pos } => {
            vec![u(1u8), u(*entity), u(*kind), i16s_value(*pos)]
        }
        EventKind::Despawn { entity } => vec![u(2u8), u(*entity)],
        EventKind::Property {
            entity,
            prop,
            value,
        } => vec![u(3u8), u(*entity), u(*prop), i(*value)],
        EventKind::Input { player, code } => vec![u(4u8), u(*player), u(*code)],
        EventKind::Tick { frame } => vec![u(5u8), u(*frame)],
        EventKind::Custom { type_id, payload } => {
            vec![u(6u8), u(*type_id), Value::Bytes(payload.to_vec())]
        }
    };
    Value::Array(vec![
        u(e.id.0),
        u(e.seq.0),
        Value::Bytes(e.origin.0.to_vec()),
        u(e.timestamp),
        Value::Array(kind),
    ])
}

fn event_kind_from(v: &Value) -> Option<EventKind> {
    let Value::Array(a) = v else {
        return None;
    };
    let (tag, f) = a.split_first()?;
    Some(match (to_u::<u8>(tag)?, f.len()) {
        (0, 2) => EventKind::Motion {
            entity: to_u(&f[0])?,
            delta: i16s(&f[1])?,
        },
        (1, 3) => EventKind::Spawn {
            entity: to_u(&f[0])?,
            kind: to_u(&f[1])?,
            pos: i16s(&f[2])?,
        },
        (2, 1) => EventKind::Despawn {
            entity: to_u(&f[0])?,
        },
        (3, 3) => EventKind::Property {
            entity: to_u(&f[0])?,
            prop: to_u(&f[1])?,
            value: to_i(&f[2])?,
        },
        (4, 2) => EventKind::Input {
            player: to_u(&f[0])?,
            code: to_u(&f[1])?,
        },
        (5, 1) => EventKind::Tick {
            frame: to_u(&f[0])?,
        },
        (6, 2) => EventKind::Custom {
            type_id: to_u(&f[0])?,
            payload: to_array(&f[1])?,
        },
        _ => return None,
    })
}

pub(crate) fn event_from(v: &Value) -> Option<Event> {
    let f = fields(v, 5)?;
    Some(Event {
        id: EventId(to_u(&f[0])?),
        seq: SeqNum(to_u(&f[1])?),
        origin: Origin(to_array(&f[2])?),
        timestamp: to_u(&f[3])?,
        kind: event_kind_from(&f[4])?,
    })
}

fn checkpoint_value(c: &Checkpoint) -> Value {
    Value::Array(vec![
        Value::Bytes(c.origin.0.to_vec()),
        u(c.seq),
        Value::Bytes(c.head.to_vec()),
        Value::Bytes(c.signature.clone()),
    ])
}

fn checkpoint_from(v: &Value) -> Option<Checkpoint> {
    let f = fields(v, 4)?;
    let Value::Bytes(signature) = &f[3] else {
        return None;
    };
    Some(Checkpoint {
        origin: Origin(to_array(&f[0])?),
        seq: to_u(&f[1])?,
        head: to_array(&f[2])?,
        signature: signature.clone(),
    })
}

fn list<T>(v: &Value, item: impl Fn(&Value) -> Option<T>) -> Option<Vec<T>> {
    match v {
        Value::Array(a) => a.iter().map(item).collect(),
        _ => None,
    }
}

pub(crate) fn message_value(m: &Message) -> Value {
    Value::Array(match m {
        Message::Hello {
            node_id,
            seq,
            world_hash,
            semantics_id,
        } => vec![
            u(0u8),
            u(node_id.0),
            u(*seq),
            u(world_hash.0),
            Value::Bytes(semantics_id.to_vec()),
        ],
        Message::RequestSync { since_seq } => vec![u(1u8), u(*since_seq)],
        Message::Events {
            events,
            checkpoints,
        } => vec![
            u(2u8),
            Value::Array(events.iter().map(event_value).collect()),
            Value::Array(checkpoints.iter().map(checkpoint_value).collect()),
        ],
        Message::HashCheck { seq, world_hash } => vec![u(3u8), u(*seq), u(world_hash.0)],
        Message::Diverged { my_hash, your_hash } => vec![u(4u8), u(my_hash.0), u(your_hash.0)],
        Message::Ack { seq } => vec![u(5u8), u(*seq)],
        Message::Bye => vec![u(6u8)],
    })
}

pub(crate) fn message_from(v: &Value) -> Option<Message> {
    let Value::Array(a) = v else {
        return None;
    };
    let (tag, f) = a.split_first()?;
    Some(match (to_u::<u8>(tag)?, f.len()) {
        (0, 4) => Message::Hello {
            node_id: NodeId(to_u(&f[0])?),
            seq: to_u(&f[1])?,
            world_hash: WorldHash(to_u(&f[2])?),
            semantics_id: to_array(&f[3])?,
        },
        (1, 1) => Message::RequestSync {
            since_seq: to_u(&f[0])?,
        },
        (2, 2) => Message::Events {
            events: list(&f[0], event_from)?,
            checkpoints: list(&f[1], checkpoint_from)?,
        },
        (3, 2) => Message::HashCheck {
            seq: to_u(&f[0])?,
            world_hash: WorldHash(to_u(&f[1])?),
        },
        (4, 2) => Message::Diverged {
            my_hash: WorldHash(to_u(&f[0])?),
            your_hash: WorldHash(to_u(&f[1])?),
        },
        (5, 1) => Message::Ack { seq: to_u(&f[0])? },
        (6, 0) => Message::Bye,
        _ => return None,
    })
}
