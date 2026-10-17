//! `SYNC_SEMANTICS_ID` is the fold of its pins, every pin is the source it
//! claims to be, and every pin moves the identifier.
//!
//! oracle: the fold is recomputed here from the documented procedure (sort
//! by name, length-prefixed name, 32 digest bytes, SHA-256), the golden
//! entries are read from `tests/determinism_golden.rs` (the source, not a
//! copy), and the upstream entries are compared with the upstream crates'
//! own constants when the bridge feature is on.

use alice_sync::semantics::{PREVIOUS_SYNC_SEMANTICS_IDS, SYNC_SEMANTICS_ID, SYNC_SEMANTICS_PINS};
use sha2::{Digest, Sha256};

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn fold(pins: &[(&str, [u8; 32])]) -> [u8; 32] {
    let mut sorted = pins.to_vec();
    sorted.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
    assert!(!sorted.is_empty(), "empty table");
    for w in sorted.windows(2) {
        assert_ne!(w[0].0, w[1].0, "repeated name");
    }
    let mut h = Sha256::new();
    for (name, digest) in &sorted {
        assert!(!name.is_empty() && name.is_ascii(), "bad name {name:?}");
        h.update(u32::try_from(name.len()).unwrap().to_be_bytes());
        h.update(name.as_bytes());
        h.update(digest);
    }
    h.finalize().into()
}

/// The hex in the module documentation, the constant and the fold agree.
#[test]
fn the_identifier_is_the_fold_of_its_pins() {
    let folded = fold(&SYNC_SEMANTICS_PINS);
    assert_eq!(
        hex(&SYNC_SEMANTICS_ID),
        hex(&folded),
        "SYNC_SEMANTICS_ID is not the fold of its pins"
    );
    let doc = include_str!("../src/semantics.rs");
    assert!(
        doc.contains(&format!("//! {}", hex(&SYNC_SEMANTICS_ID))),
        "the module doc does not show the identifier's hex"
    );
}

/// The golden entries are the digests `tests/determinism_golden.rs` records.
#[test]
fn golden_pins_are_the_recorded_goldens() {
    let src = include_str!("determinism_golden.rs");
    let mut compared = 0;
    for (name, constant) in [
        ("fixed", "GOLDEN_FIXED"),
        ("world", "GOLDEN_WORLD"),
        ("crdt", "GOLDEN_CRDT"),
    ] {
        let key = format!("pub const {constant}: &str = \"");
        let start = src
            .find(&key)
            .unwrap_or_else(|| panic!("{constant} not found"))
            + key.len();
        let recorded = &src[start..start + 64];
        let pin = SYNC_SEMANTICS_PINS
            .iter()
            .find(|(n, _)| *n == name)
            .expect(name)
            .1;
        assert_eq!(hex(&pin), recorded, "pin {name} differs from {constant}");
        compared += 1;
    }
    assert_eq!(compared, 3);
    let names: Vec<_> = SYNC_SEMANTICS_PINS.iter().map(|(n, _)| *n).collect();
    assert_eq!(
        names,
        ["fixed", "world", "crdt", "alice-physics", "alice-analytics"]
    );
}

#[cfg(feature = "physics")]
#[test]
fn physics_pin_is_the_resolved_physics_identifier() {
    let pin = SYNC_SEMANTICS_PINS
        .iter()
        .find(|(n, _)| *n == "alice-physics")
        .unwrap()
        .1;
    assert_eq!(hex(&pin), hex(&alice_physics::PHYSICS_SEMANTICS_ID));
}

#[cfg(feature = "analytics")]
#[test]
fn analytics_pin_is_the_resolved_analytics_identifier() {
    let pin = SYNC_SEMANTICS_PINS
        .iter()
        .find(|(n, _)| *n == "alice-analytics")
        .unwrap()
        .1;
    assert_eq!(hex(&pin), hex(&alice_analytics::SEMANTICS_ID));
}

/// Changing any single pin (each golden and each upstream identifier) moves
/// the identifier.
#[test]
fn every_pin_moves_the_identifier() {
    for i in 0..SYNC_SEMANTICS_PINS.len() {
        let mut changed = SYNC_SEMANTICS_PINS;
        changed[i].1[31] ^= 1;
        assert_ne!(
            fold(&changed),
            SYNC_SEMANTICS_ID,
            "pin {} does not move the identifier",
            changed[i].0
        );
    }
}

#[test]
fn the_previous_list_does_not_contain_the_current_identifier() {
    assert!(!PREVIOUS_SYNC_SEMANTICS_IDS.contains(&SYNC_SEMANTICS_ID));
}
