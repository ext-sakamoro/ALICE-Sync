//! Content hash of ALICE-Sync's Law-bearing semantics: [`SYNC_SEMANTICS_ID`].
//!
//! ALICE-Sync's deterministic state is integer Q16.16 (`Fixed`), a world
//! whose rolling hash is updated per event, and CRDTs whose merge is defined
//! by fixed tie-break rules. The goldens in `tests/determinism_golden.rs` pin
//! the bits each of those produces for a fixed scenario. This module folds
//! those digests, plus the identifiers of the Law-bearing crates this crate
//! bridges, into one 32-byte value. Two builds with the same identifier
//! produce the same bits on every pinned scenario; a build that changes any
//! of them, or that bridges a producer with a different identifier, has a
//! different value.
//!
//! # Hex form
//!
//! ```text
//! c844437ee0ad82dfae3ccbeb9bf1c0bfe8111b83060bd9d4305c2d8111d56095
//! ```
//!
//! # Entries
//!
//! [`SYNC_SEMANTICS_PINS`]: one entry per golden (`"fixed"`, `"world"`,
//! `"crdt"`, the SHA-256 recorded in `tests/determinism_golden.rs`), plus one
//! per bridged Law-bearing crate: `"alice-physics"` (its
//! `PHYSICS_SEMANTICS_ID`, the `physics` bridge) and `"alice-analytics"` (its
//! `SEMANTICS_ID`, the `analytics` bridge), at the versions this release
//! resolves.
//!
//! # Fold
//!
//! The same procedure as the `SEMANTICS_ID` fold of `alice-det-math` and
//! `PHYSICS_SEMANTICS_ID` of `alice-physics`: sort the entries by name
//! (ascending byte order), reject an empty table, a repeated name, an empty
//! name or a name that is not ASCII, then SHA-256 over, for each entry in
//! that order, the name length as 4 bytes big-endian (`u32`), the name bytes
//! and the 32 digest bytes.
//!
//! # Independent of cargo features
//!
//! The table and the identifier are constants that no `cfg` gates: the
//! upstream entries are recorded as hex, so a build without the `physics` or
//! `analytics` feature reports the same identifier. With the feature on,
//! `tests/sync_semantics_id.rs` checks the recorded hex against the
//! upstream crate's constant, so resolving another upstream identifier is a
//! failing test, not a silent change.
//!
//! # Version and toolchain
//!
//! The identifier is a constant of this crate version; a dependency bump or
//! a toolchain change that moved any pinned bits fails a golden, and moving
//! the golden moves the identifier. Earlier identifiers whose pinned bits
//! are still reproduced are listed in [`PREVIOUS_SYNC_SEMANTICS_IDS`].

/// Decodes 64 hex digits into 32 bytes at compile time; anything else is a
/// compile error.
pub(crate) const fn hex32(s: &str) -> [u8; 32] {
    const fn nibble(c: u8) -> u8 {
        match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            _ => panic!("not a lower-case hex digit"),
        }
    }
    let b = s.as_bytes();
    assert!(b.len() == 64, "a digest is 64 hex digits");
    let mut out = [0u8; 32];
    let mut i = 0;
    while i < 32 {
        out[i] = nibble(b[2 * i]) * 16 + nibble(b[2 * i + 1]);
        i += 1;
    }
    out
}

/// The pins folded into [`SYNC_SEMANTICS_ID`], as `(name, digest)`.
pub const SYNC_SEMANTICS_PINS: [(&str, [u8; 32]); 5] = [
    (
        "fixed",
        hex32("dc65b542978e3eb841faee4a7202f7416cee5cd6ce4fdd64cebc18a97ba31f9d"),
    ),
    (
        "world",
        hex32("875410bef5edafaaff33e1e74a2c879b987e394f784f5ef361b6d828d531f593"),
    ),
    (
        "crdt",
        hex32("54f05313b39a7b671f53bf5c87fb4e37cb1b64f40b402ef98517047c4a730bc8"),
    ),
    (
        "alice-physics",
        hex32("79ca52976f404dfb453908970ebe45f584970ff9344a56a06aae3fc9c3c38108"),
    ),
    (
        "alice-analytics",
        hex32("d2209b30f6f1f45baa1b638bcdfee34ac64773b2e63b9c083b2e77afc691398e"),
    ),
];

/// The semantics identifier of this crate: the fold of
/// [`SYNC_SEMANTICS_PINS`] (see the module documentation).
pub const SYNC_SEMANTICS_ID: [u8; 32] =
    hex32("c844437ee0ad82dfae3ccbeb9bf1c0bfe8111b83060bd9d4305c2d8111d56095");

/// Earlier values of [`SYNC_SEMANTICS_ID`] whose pinned bits this release
/// still reproduces (none yet: this is the first identifier).
pub const PREVIOUS_SYNC_SEMANTICS_IDS: &[[u8; 32]] = &[];
