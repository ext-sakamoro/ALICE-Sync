//! Oracle: stored state carries the state identifier, the fold of the pins its bytes' meaning
//! depends on, while the Hello handshake carries the full identifier that also folds the bridges
//! (alice-physics, alice-analytics). A change of a bridge pin alone keeps stored worlds readable
//! and refuses the peer; a change of a state pin refuses both. Format 1 and 2 files, stamped with
//! a full identifier, are read through the table of earlier full identifiers, each proven by
//! recomputing both folds from its recorded pins.

use alice_sync::persist::{
    decode_state, encode_state, FormatStatus, IdStatus, PersistError, FORMAT_VERSION,
    PERSISTED_FULL_SYNC_SEMANTICS_IDS,
};
use alice_sync::semantics::{check_semantics, SYNC_STATE_PIN_NAMES};
use alice_sync::world::{World, WorldState};
use alice_sync::{Fixed, SYNC_SEMANTICS_ID, SYNC_SEMANTICS_PINS, SYNC_STATE_SEMANTICS_ID};
use sha2::{Digest, Sha256};

/// The fold of `src/semantics.rs`, written from its documentation: sort by name, then for each
/// entry the name length (u32 big-endian), the name and the 32 digest bytes, all under SHA-256
fn fold(pins: &[(&str, [u8; 32])]) -> [u8; 32] {
    let mut sorted = pins.to_vec();
    sorted.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
    let mut h = Sha256::new();
    for (name, digest) in sorted {
        h.update((name.len() as u32).to_be_bytes());
        h.update(name.as_bytes());
        h.update(digest);
    }
    h.finalize().into()
}

fn state_pins(pins: &[(&'static str, [u8; 32])]) -> Vec<(&'static str, [u8; 32])> {
    pins.iter()
        .copied()
        .filter(|(n, _)| SYNC_STATE_PIN_NAMES.contains(n))
        .collect()
}

/// `bytes` with its header identifier (bytes 6..38) replaced
fn stamped(bytes: &[u8], id: &[u8; 32]) -> Vec<u8> {
    let mut b = bytes.to_vec();
    b[6..38].copy_from_slice(id);
    b
}

#[test]
fn the_state_pins_are_the_stored_types_pins_and_no_bridge() {
    assert_eq!(SYNC_STATE_PIN_NAMES, ["crdt", "fixed", "receive", "world"]);
    for n in SYNC_STATE_PIN_NAMES {
        assert!(SYNC_SEMANTICS_PINS.iter().any(|(p, _)| *p == n), "{n}");
    }
}

#[test]
fn the_state_identifier_is_the_fold_of_the_state_pins() {
    assert_eq!(
        SYNC_STATE_SEMANTICS_ID,
        fold(&state_pins(&SYNC_SEMANTICS_PINS))
    );
    assert_eq!(SYNC_SEMANTICS_ID, fold(&SYNC_SEMANTICS_PINS));
    assert_ne!(SYNC_STATE_SEMANTICS_ID, SYNC_SEMANTICS_ID);
}

#[test]
fn stored_state_carries_the_state_identifier_in_format_3() {
    let bytes = encode_state(&Fixed::from_int(3));
    assert_eq!(FORMAT_VERSION, 3);
    assert_eq!(&bytes[4..6], &3u16.to_le_bytes());
    assert_eq!(&bytes[6..38], &SYNC_STATE_SEMANTICS_ID);
    let d = decode_state::<Fixed>(&bytes).unwrap();
    assert_eq!(d.status, IdStatus::Current);
    assert_eq!(d.format, FormatStatus::Current);
}

#[test]
fn every_earlier_full_identifier_is_proven_from_its_recorded_pins() {
    assert!(!PERSISTED_FULL_SYNC_SEMANTICS_IDS.is_empty());
    for e in PERSISTED_FULL_SYNC_SEMANTICS_IDS {
        assert_eq!(e.full, fold(e.pins), "full fold");
        assert_eq!(e.state, fold(&state_pins(e.pins)), "state fold");
    }
}

#[test]
fn a_bridge_only_pin_change_keeps_stored_state_and_refuses_the_peer() {
    let mut pins = SYNC_SEMANTICS_PINS.to_vec();
    for (n, d) in &mut pins {
        if *n == "alice-physics" {
            d[0] ^= 1;
        }
    }
    let full = fold(&pins);
    let state = fold(&state_pins(&pins));
    assert_ne!(full, SYNC_SEMANTICS_ID);
    assert_eq!(state, SYNC_STATE_SEMANTICS_ID);
    // a world stored by that build reads here
    let bytes = stamped(&encode_state(World::new(7).state()), &state);
    assert!(decode_state::<WorldState>(&bytes).is_ok());
    // its Hello does not
    assert!(check_semantics(&full).is_err());
}

#[test]
fn a_state_pin_change_refuses_stored_state_and_the_peer() {
    let mut pins = SYNC_SEMANTICS_PINS.to_vec();
    for (n, d) in &mut pins {
        if *n == "world" {
            d[0] ^= 1;
        }
    }
    let full = fold(&pins);
    let state = fold(&state_pins(&pins));
    assert_ne!(state, SYNC_STATE_SEMANTICS_ID);
    let bytes = stamped(&encode_state(World::new(7).state()), &state);
    assert!(matches!(
        decode_state::<WorldState>(&bytes),
        Err(PersistError::Semantics(_))
    ));
    assert!(check_semantics(&full).is_err());
}

#[test]
fn a_format_2_file_with_an_earlier_full_identifier_reads_as_legacy() {
    // format 2 kept the body of today's format; only the header identifier differs
    let e = &PERSISTED_FULL_SYNC_SEMANTICS_IDS[0];
    let mut bytes = stamped(&encode_state(&Fixed::from_int(5)), &e.full);
    bytes[4..6].copy_from_slice(&2u16.to_le_bytes());
    let d = decode_state::<Fixed>(&bytes).unwrap();
    assert_eq!(d.value, Fixed::from_int(5));
    assert_eq!(d.format, FormatStatus::Legacy(2));
    // the same file stamped with an identifier the table does not hold is refused
    let mut unknown = e.full;
    unknown[0] ^= 1;
    let bytes = stamped(&bytes, &unknown);
    assert!(matches!(
        decode_state::<Fixed>(&bytes),
        Err(PersistError::Semantics(_))
    ));
}

#[test]
fn a_format_3_file_never_consults_the_full_identifier_table() {
    // a format 3 header holding an earlier full identifier is not a state identifier
    let e = &PERSISTED_FULL_SYNC_SEMANTICS_IDS[0];
    let bytes = stamped(&encode_state(&Fixed::from_int(5)), &e.full);
    assert!(matches!(
        decode_state::<Fixed>(&bytes),
        Err(PersistError::Semantics(_))
    ));
}

#[test]
fn a_format_3_file_stamped_with_the_full_identifier_is_refused_naming_both() {
    let bytes = stamped(&encode_state(&Fixed::from_int(5)), &SYNC_SEMANTICS_ID);
    match decode_state::<Fixed>(&bytes) {
        Err(PersistError::Semantics(m)) => {
            assert_eq!(m.stored, SYNC_SEMANTICS_ID);
            assert_eq!(m.expected, SYNC_STATE_SEMANTICS_ID);
        }
        other => panic!("expected Semantics, got {other:?}"),
    }
}

#[test]
fn the_strict_reader_refuses_formats_1_and_2() {
    use alice_sync::persist::decode_state_strict;
    let e = &PERSISTED_FULL_SYNC_SEMANTICS_IDS[0];
    let mut bytes = stamped(&encode_state(&Fixed::from_int(5)), &e.full);
    bytes[4..6].copy_from_slice(&2u16.to_le_bytes());
    assert_eq!(
        decode_state_strict::<Fixed>(&bytes).map(|d| d.value),
        Err(PersistError::LegacyFormat(2))
    );
}
