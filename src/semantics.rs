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
//! bc5f1786af86ae3bd7176dfa46bc0d95b44d618776daeb6019aaccb843390264
//! ```
//!
//! # Entries
//!
//! [`SYNC_SEMANTICS_PINS`]: one entry per golden (`"fixed"`, `"world"`,
//! `"crdt"`, `"receive"`, the SHA-256 recorded in `tests/determinism_golden.rs`;
//! `"receive"` is the world a node builds from events of several peers received
//! out of order, ALICE-Law G13), plus one
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
pub const SYNC_SEMANTICS_PINS: [(&str, [u8; 32]); 6] = [
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
        "receive",
        hex32("b0f4dd9b848374e314f1d88ae220e926d8a3d05d49d08b3a77d8d9673bb2b242"),
    ),
    (
        "alice-physics",
        hex32("a3f7eb247574fcbac11839aeaf64f1bae826721f2a2e12e8b30fd22db70612c1"),
    ),
    (
        "alice-analytics",
        hex32("d2209b30f6f1f45baa1b638bcdfee34ac64773b2e63b9c083b2e77afc691398e"),
    ),
];

/// The semantics identifier of this crate: the fold of
/// [`SYNC_SEMANTICS_PINS`] (see the module documentation).
pub const SYNC_SEMANTICS_ID: [u8; 32] =
    hex32("bc5f1786af86ae3bd7176dfa46bc0d95b44d618776daeb6019aaccb843390264");

/// Earlier values of [`SYNC_SEMANTICS_ID`] whose pinned bits this release
/// still reproduces (none yet: this is the first identifier).
pub const PREVIOUS_SYNC_SEMANTICS_IDS: &[[u8; 32]] = &[];

/// How a stored or received identifier relates to this build's
/// [`SYNC_SEMANTICS_ID`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdStatus {
    /// The identifier is this build's [`SYNC_SEMANTICS_ID`].
    Current,
    /// The identifier is listed in [`PREVIOUS_SYNC_SEMANTICS_IDS`]: an earlier
    /// identifier whose pinned bits this build still reproduces.
    VerifiedPrevious([u8; 32]),
}

/// A stored or received identifier that is neither [`SYNC_SEMANTICS_ID`] nor
/// listed in [`PREVIOUS_SYNC_SEMANTICS_IDS`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SemanticsMismatch {
    /// The identifier the bytes or the peer carried.
    pub stored: [u8; 32],
    /// This build's [`SYNC_SEMANTICS_ID`].
    pub expected: [u8; 32],
}

impl core::fmt::Display for SemanticsMismatch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "semantics identifier mismatch: stored ")?;
        for b in self.stored {
            write!(f, "{b:02x}")?;
        }
        write!(f, ", expected ")?;
        for b in self.expected {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

impl std::error::Error for SemanticsMismatch {}

/// Compares an identifier carried by stored bytes or a peer with this build's.
///
/// # Errors
///
/// [`SemanticsMismatch`] naming both identifiers when `stored` is neither
/// [`SYNC_SEMANTICS_ID`] nor in [`PREVIOUS_SYNC_SEMANTICS_IDS`].
pub fn check_semantics(stored: &[u8; 32]) -> Result<IdStatus, SemanticsMismatch> {
    check_semantics_against(stored, &SYNC_SEMANTICS_ID, PREVIOUS_SYNC_SEMANTICS_IDS)
}

/// [`check_semantics`] against an explicit current identifier and list of
/// earlier ones (the list is empty in this release, so the previous-identifier
/// path is tested through this function).
pub(crate) fn check_semantics_against(
    stored: &[u8; 32],
    current: &[u8; 32],
    previous: &[[u8; 32]],
) -> Result<IdStatus, SemanticsMismatch> {
    if stored == current {
        Ok(IdStatus::Current)
    } else if previous.contains(stored) {
        Ok(IdStatus::VerifiedPrevious(*stored))
    } else {
        Err(SemanticsMismatch {
            stored: *stored,
            expected: *current,
        })
    }
}
