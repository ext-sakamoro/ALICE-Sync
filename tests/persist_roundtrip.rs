//! `alice_sync::persist`: every Law-bearing type round-trips through the
//! stamped format, and bytes written under another identifier are refused
//! with an error that names both identifiers (ALICE-Law G8).
//!
//! oracle: the format table in `src/persist.rs` (magic, version, 32-byte
//! identifier at bytes 6..38, type tag at byte 38); a round trip must give
//! back the value and re-encode to the same bytes.

use alice_sync::fixed_point::{Fixed, Vec3Fixed};
use alice_sync::persist::{
    decode_state, decode_state_strict, encode_state, FormatStatus, IdStatus, Persist, PersistError,
    FORMAT_VERSION, MAGIC,
};
use alice_sync::{Entity, Event, EventKind, World, WorldState, SYNC_STATE_SEMANTICS_ID};

const ID: std::ops::Range<usize> = 6..38;

/// `encode_state(world().state())` as format version 1 wrote it (alice-sync a555bd4, the last
/// release of that format; SHA-256 aded0fa98b4703ac2875f7a5234f04d24acef8547c98d0c3e2b84b0255bef663)
const WORLD_STATE_V1: &[u8] = include_bytes!("data/persist/world_state_v1.bin");

fn world() -> World {
    let mut w = World::new(7);
    for i in 0..6u32 {
        let k = i as i16;
        w.apply(&Event::new(EventKind::Spawn {
            entity: i,
            kind: (i % 3) as u16,
            pos: [k * 10, -k, 300 - k],
        }))
        .unwrap();
    }
    w.apply(&Event::new(EventKind::Property {
        entity: 2,
        prop: 5,
        value: -77,
    }))
    .unwrap();
    // free slots on the free list, then one reused
    w.apply(&Event::new(EventKind::Despawn { entity: 1 }))
        .unwrap();
    w.apply(&Event::new(EventKind::Despawn { entity: 4 }))
        .unwrap();
    w.apply(&Event::new(EventKind::Spawn {
        entity: 9,
        kind: 1,
        pos: [i16::MIN, i16::MAX, 0],
    }))
    .unwrap();
    w
}

fn roundtrip<T: Persist + core::fmt::Debug>(value: &T) -> T {
    let bytes = encode_state(value);
    assert_eq!(&bytes[..4], &MAGIC);
    assert_eq!(&bytes[4..6], &FORMAT_VERSION.to_le_bytes());
    assert_eq!(
        &bytes[ID], &SYNC_STATE_SEMANTICS_ID,
        "the writer stamps this build's identifier"
    );
    let d = decode_state::<T>(&bytes).expect("own bytes decode");
    assert_eq!(d.status, IdStatus::Current);
    assert_eq!(
        encode_state(&d.value),
        bytes,
        "re-encoding gives the same bytes"
    );
    d.value
}

fn foreign(mut bytes: Vec<u8>) -> Vec<u8> {
    bytes[ID.start] ^= 0x01;
    bytes
}

fn assert_mismatch<T: Persist + core::fmt::Debug>(value: &T) {
    let bytes = foreign(encode_state(value));
    let stored: [u8; 32] = bytes[ID].try_into().unwrap();
    match decode_state::<T>(&bytes) {
        Err(PersistError::Semantics(m)) => {
            assert_eq!(m.stored, stored);
            assert_eq!(m.expected, SYNC_STATE_SEMANTICS_ID);
            let text = m.to_string();
            assert!(text.contains(&hex(&stored)) && text.contains(&hex(&SYNC_STATE_SEMANTICS_ID)));
        }
        other => panic!("expected a semantics mismatch, got {other:?}"),
    }
}

fn hex(b: &[u8]) -> String {
    b.iter().fold(String::new(), |mut s, x| {
        use std::fmt::Write;
        let _ = write!(s, "{x:02x}");
        s
    })
}

#[test]
fn fixed_and_vec3_round_trip() {
    for raw in [0, 1, -1, i32::MAX, i32::MIN, 65_536] {
        assert_eq!(roundtrip(&Fixed(raw)), Fixed(raw));
    }
    let v = Vec3Fixed::from_f32(1.25, -2.5, 0.0);
    assert_eq!(roundtrip(&v), v);
}

#[test]
fn entity_round_trips() {
    let w = world();
    let mut compared = 0;
    for e in w.iter_entities() {
        let back: Entity = roundtrip(e);
        assert_eq!(back.compute_hash(), e.compute_hash());
        assert_eq!(back.id, e.id);
        compared += 1;
    }
    assert_eq!(compared, 5);
}

#[test]
fn world_state_round_trips_with_its_live_entities_and_id_map() {
    let w = world();
    let back: WorldState = roundtrip(w.state());
    assert_eq!(back.frame, w.state().frame);
    assert_eq!(back.entities.len(), 5);
    for id in [0u32, 2, 3, 5, 9] {
        let h = back.get_handle(id).expect("live entity keeps its handle");
        assert_eq!(back.entities.get(h).unwrap().id, id);
    }
    assert!(back.get_handle(1).is_none() && back.get_handle(4).is_none());
}

#[cfg(feature = "async")]
#[test]
fn register_of_a_position_round_trips() {
    use alice_sync::crdt::LwwRegister;
    let mut r = LwwRegister::new(Vec3Fixed::from_f32(3.0, 4.0, -5.0), 9);
    r.set_at(Vec3Fixed::from_f32(-1.0, 0.5, 2.0), 41);
    let back = roundtrip(&r);
    assert_eq!(back.value, r.value);
    assert_eq!(
        (back.timestamp, back.replica_id),
        (r.timestamp, r.replica_id)
    );
    assert_mismatch(&r);
}

/// Each type refuses bytes stamped with another identifier, naming both.
#[test]
fn another_identifier_is_refused_naming_both() {
    assert_mismatch(&Fixed(5));
    assert_mismatch(&Vec3Fixed::from_f32(1.0, 2.0, 3.0));
    let w = world();
    assert_mismatch(w.iter_entities().next().unwrap());
    assert_mismatch(w.state());
}

#[test]
fn header_and_framing_errors() {
    let bytes = encode_state(&Fixed(5));
    let mut bad = bytes.clone();
    bad[0] = b'X';
    assert_eq!(decode_state::<Fixed>(&bad), Err(PersistError::BadMagic));
    let mut bad = bytes.clone();
    bad[4] = 4;
    assert_eq!(
        decode_state::<Fixed>(&bad),
        Err(PersistError::UnsupportedVersion(4))
    );
    // version 1 is an earlier format: read, marked legacy, and refused by the strict reader
    // (a version 1 header carries a full identifier, read through the table of earlier ones)
    let mut v1 = bytes.clone();
    v1[4] = 1;
    v1[ID].copy_from_slice(&alice_sync::persist::PERSISTED_FULL_SYNC_SEMANTICS_IDS[0].full);
    let d = decode_state::<Fixed>(&v1).unwrap();
    assert_eq!((d.value, d.format), (Fixed(5), FormatStatus::Legacy(1)));
    assert_eq!(
        decode_state_strict::<Fixed>(&v1),
        Err(PersistError::LegacyFormat(1))
    );
    assert_eq!(
        decode_state_strict::<Fixed>(&bytes).unwrap().format,
        FormatStatus::Current
    );
    assert_eq!(
        decode_state::<Vec3Fixed>(&bytes),
        Err(PersistError::WrongType {
            stored: 1,
            expected: 2
        })
    );
    assert_eq!(
        decode_state::<Fixed>(&bytes[..bytes.len() - 1]),
        Err(PersistError::Truncated)
    );
    let mut long = bytes.clone();
    long.push(0);
    assert_eq!(
        decode_state::<Fixed>(&long),
        Err(PersistError::TrailingBytes(1))
    );
    assert_eq!(decode_state::<Fixed>(&[]), Err(PersistError::Truncated));
}

/// A version 1 world whose free list or id map would break the unchecked
/// accessors is refused, not decoded, each with the reason of the check that
/// caught it (the fixture written by the last release of that format).
#[test]
fn malformed_world_state_is_refused() {
    let bytes = WORLD_STATE_V1.to_vec();
    assert_eq!(&bytes[4..6], &1u16.to_le_bytes());
    // walk the slots (format table in src/persist.rs) to find the one free
    // slot's next-free field and the free-list head that follows the slots
    let body = 39;
    let n = u32::from_le_bytes(bytes[body..body + 4].try_into().unwrap()) as usize;
    let mut at = body + 4;
    let mut free_next_at = Vec::new();
    for _ in 0..n {
        let state = bytes[at];
        if state == 0 {
            free_next_at.push(at + 1 + 4);
        }
        at += 1 + 4 + if state == 1 { 4 + 2 + 12 + 64 + 2 } else { 4 };
    }
    assert_eq!(free_next_at.len(), 1, "the scenario leaves one free slot");
    let head_at = at;
    let refused = |bad: &[u8]| match decode_state::<WorldState>(bad) {
        Err(PersistError::Malformed(why)) => why,
        other => panic!("expected Malformed, got {other:?}"),
    };
    let with = |pos: usize, v: u32| {
        let mut bad = bytes.clone();
        bad[pos..pos + 4].copy_from_slice(&v.to_le_bytes());
        bad
    };
    assert_eq!(
        refused(&with(head_at, 0)),
        "the free list reaches an occupied slot"
    );
    assert_eq!(
        refused(&with(head_at, n as u32)),
        "the free list points past the last slot"
    );
    // the head is right, but the free slot links on to an occupied one
    assert_eq!(
        refused(&with(free_next_at[0], 0)),
        "the free list reaches an occupied slot"
    );
    // the free slot links to itself
    let own = u32::from_le_bytes(bytes[head_at..head_at + 4].try_into().unwrap());
    assert_eq!(
        refused(&with(free_next_at[0], own)),
        "the free list has a cycle"
    );
    assert_eq!(
        refused(&with(head_at, u32::MAX)),
        "a free slot is not on the free list"
    );
    // an id map handle with a stale generation
    let first_gen = head_at + 4 + 4 + 4;
    let g = u32::from_le_bytes(bytes[first_gen..first_gen + 4].try_into().unwrap());
    assert_eq!(
        refused(&with(first_gen, g.wrapping_add(1))),
        "an id map handle does not point to an occupied slot of its generation"
    );
}

/// The version 1 fixture is read as legacy: the same live entities, frame and seed as the world
/// that wrote it, and re-encoded it gives today's bytes for that world.
#[test]
fn a_version_1_world_is_read_as_legacy() {
    let d = decode_state::<WorldState>(WORLD_STATE_V1).unwrap();
    assert_eq!(d.format, FormatStatus::Legacy(1));
    assert_eq!(d.status, IdStatus::Current);
    let w = world();
    assert_eq!(
        (d.value.frame, d.value.seed),
        (w.state().frame, w.state().seed)
    );
    let mut ids = Vec::new();
    for (id, h) in d.value.ids() {
        let e = d.value.entities.get(h).unwrap();
        assert_eq!(e.id, id);
        let original = w
            .state()
            .entities
            .get(w.state().get_handle(id).unwrap())
            .unwrap();
        assert_eq!(e.compute_hash(), original.compute_hash());
        ids.extend([id]);
    }
    assert_eq!(ids, [0, 2, 3, 5, 9]);
    assert_eq!(encode_state(&d.value), encode_state(w.state()));
    assert!(matches!(
        decode_state_strict::<WorldState>(WORLD_STATE_V1),
        Err(PersistError::LegacyFormat(1))
    ));
}

/// The current world body: the live count, the entity bodies in strictly ascending id order, the
/// frame and the seed
#[test]
fn a_version_2_world_out_of_id_order_is_refused() {
    let bytes = encode_state(world().state());
    let body = 39;
    let n = u32::from_le_bytes(bytes[body..body + 4].try_into().unwrap()) as usize;
    assert_eq!(n, 5);
    assert_eq!(bytes.len(), body + 4 + n * ENTITY_BODY + 8 + 8);
    let entity = |i: usize| body + 4 + i * ENTITY_BODY;
    let refused = |bad: &[u8]| match decode_state::<WorldState>(bad) {
        Err(PersistError::Malformed(why)) => why,
        other => panic!("expected Malformed, got {other:?}"),
    };
    // swap the first two entity bodies: ids 2 then 0
    let mut swapped = bytes.clone();
    let (a, b) = (entity(0), entity(1));
    let first = swapped[a..a + ENTITY_BODY].to_vec();
    swapped.copy_within(b..b + ENTITY_BODY, a);
    swapped[b..b + ENTITY_BODY].copy_from_slice(&first);
    assert_eq!(
        refused(&swapped),
        "entities are not in strictly ascending id order"
    );
    // the same id twice
    let mut twice = bytes.clone();
    twice.copy_within(a..a + 4, b);
    assert_eq!(
        refused(&twice),
        "entities are not in strictly ascending id order"
    );
    // a count above the limit is malformed; one within it but above the bodies that follow is
    // truncated, not allocated for
    let mut more = bytes.clone();
    more[body..body + 4].copy_from_slice(&u32::MAX.to_le_bytes());
    assert_eq!(refused(&more), "more live entities than MAX_LIVE_ENTITIES");
    let limit = u32::try_from(alice_sync::world::MAX_LIVE_ENTITIES).unwrap();
    more[body..body + 4].copy_from_slice(&(limit + 1).to_le_bytes());
    assert_eq!(refused(&more), "more live entities than MAX_LIVE_ENTITIES");
    more[body..body + 4].copy_from_slice(&limit.to_le_bytes());
    assert!(matches!(
        decode_state::<WorldState>(&more),
        Err(PersistError::Truncated)
    ));
}

/// id, kind, position, 16 property values and the active mask
const ENTITY_BODY: usize = 4 + 2 + 12 + 64 + 2;
