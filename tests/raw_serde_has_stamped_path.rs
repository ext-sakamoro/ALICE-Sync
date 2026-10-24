//! Every type with a raw serde form (feature `serde-raw`, which carries no Law identifier) also has a
//! stamped path to durable bytes: `persist::encode_state` for the `Persist` types, and
//! `WorldSoA::to_state` into one of them for `WorldStorage`. The list is the raw-only types of
//! scripts/serialize_types.toml; a type added there without a stamped path fails to compile here.

use alice_sync::fixed_point::{Fixed, Vec3Fixed};
use alice_sync::persist::{decode_state, encode_state, Persist};
use alice_sync::{Entity, Event, EventKind, WorldSoA, WorldState};

/// The stamped writer reaches `T` and reads it back
fn stamped<T: Persist + core::fmt::Debug>(value: &T) -> Vec<u8> {
    let bytes = encode_state(value);
    decode_state::<T>(&bytes).expect("stamped bytes read back");
    bytes
}

#[test]
fn every_raw_serde_type_has_a_stamped_path() {
    let mut covered = Vec::new();
    stamped(&Fixed(5));
    covered.push("Fixed");
    stamped(&Vec3Fixed::from_i16_array([1, 2, 3]));
    covered.push("Vec3Fixed");
    stamped(&Entity::default());
    covered.push("Entity");
    stamped(&WorldState::default());
    covered.push("WorldState");
    // WorldStorage is the column store of WorldSoA: stored through its WorldState
    let mut soa = WorldSoA::new(3);
    soa.apply(&Event::new(EventKind::Spawn {
        entity: 9,
        kind: 1,
        pos: [1, 2, 3],
    }))
    .unwrap();
    let _storage: &alice_sync::WorldStorage = soa.storage();
    let bytes = stamped(&soa.to_state());
    let back = WorldSoA::from_state(&decode_state::<WorldState>(&bytes).unwrap().value).unwrap();
    assert_eq!(back.hash(), soa.hash());
    covered.push("WorldStorage");

    // the raw-only list of scripts/serialize_types.toml
    let manifest = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/serialize_types.toml"),
    )
    .unwrap();
    let raw_only = manifest
        .lines()
        .find_map(|l| l.trim().strip_prefix("raw_only = "))
        .expect("raw_only in scripts/serialize_types.toml");
    assert_eq!(
        raw_only.trim().parse::<usize>().unwrap(),
        covered.len(),
        "a raw serde type without a stamped path in this test"
    );
}
