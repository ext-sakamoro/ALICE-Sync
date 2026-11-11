//! Durable form of ALICE-Sync's Law-bearing state, stamped with
//! [`SYNC_STATE_SEMANTICS_ID`] (the Hello handshake carries [`crate::SYNC_SEMANTICS_ID`]).
//!
//! The fixed-point types ([`Fixed`], [`Vec3Fixed`]) and the state built from
//! them ([`Entity`], [`WorldState`], `LwwRegister` of those) and the replicated
//! sets and maps of [`crate::crdt`] (`OrSet`, `LwwMap`) are the values
//! whose bits the crate's semantics identifier names. Their raw `serde`
//! implementations are behind the opt-in `serde-raw` feature and carry no
//! identifier, so they are not a durable format. To store or hand over such a
//! value, encode it here:
//!
//! ```
//! use alice_sync::fixed_point::Vec3Fixed;
//! use alice_sync::persist::{decode_state, encode_state, IdStatus};
//!
//! let v = Vec3Fixed::from_f32(1.0, -2.5, 0.25);
//! let bytes = encode_state(&v);
//! let decoded = decode_state::<Vec3Fixed>(&bytes).unwrap();
//! assert_eq!(decoded.value, v);
//! assert_eq!(decoded.status, IdStatus::Current);
//! ```
//!
//! # Format (version 3)
//!
//! All integers little-endian.
//!
//! | bytes | content |
//! |---|---|
//! | 4 | magic `ASYN` |
//! | 2 | format version, `3` |
//! | 32 | the writer's `SYNC_STATE_SEMANTICS_ID` |
//! | 1 | type tag (`1` Fixed, `2` Vec3Fixed, `3` Entity, `4` WorldState, `5` LwwRegister, `6` u64, `7` bool, `8` Option, `9` OrSet, `10` LwwMap) |
//! | rest | the value's body |
//!
//! Bodies: `Fixed` is its raw `i32`; `Vec3Fixed` is `x`, `y`, `z`; `Entity` is
//! `id: u32`, `kind: u16`, the position, the 16 property values (`i32`) and the
//! active mask (`u16`); `WorldState` is the number of live entities (`u32`),
//! their bodies in ascending id order, `frame: u64` and `seed: u64`;
//! `LwwRegister<V>` is `timestamp: u64`, `replica_id: u64`, then the tag and
//! body of `V`. `u64` is its 8 bytes; `bool` is one byte, 0 or 1; `Option<V>` is
//! 0, or 1 then the tag and body of `V`. `OrSet<V>` is the number of live values
//! (`u32`), each value (tag and body, strictly ascending) with its tags (a count
//! then `(replica: u64, n: u64)` pairs, strictly ascending, never none), then the
//! add counters (a count, then `(replica, n)` pairs by replica) and the tombstoned
//! tags (as a value's tags). `LwwMap<K, V>` is the number of keys (`u32`), then
//! each key (tag and body, strictly ascending) and its `LwwRegister<Option<V>>`
//! body: the replicated entries only. The writing replica and its clock are local
//! to a device, so two devices holding the same map write the same bytes; a map
//! read back has replica 0 and its clock at its latest timestamp, and
//! `LwwMap::with_replica` names the device that writes through it next.
//!
//! A world is stored as its logical content only: which entities live with
//! which values, the frame and the seed. Where the entities sit in memory (arena
//! slots, generations, the free list) depends on the order in which they were
//! spawned and removed, so two nodes holding the same world after receiving the
//! same events in different orders would otherwise write different bytes.
//! Decoding rebuilds the world with the entities placed in id order, so handles
//! ([`crate::arena::Handle`]) are meaningful only inside the process that made
//! them and are not kept across an encode and decode.
//!
//! Version 2 had today's body and carried the writer's full `SYNC_SEMANTICS_ID`. Version 1 also
//! carried it and stored the arena layout (slots, generations, free list, a dense id map). Their
//! headers are still parsed as [`FormatStatus::Legacy`], but the state identifier their full
//! identifier maps to is not this release's and is not reproduced (the `crdt` pin moved), so both
//! readers refuse them with [`PersistError::Semantics`] naming the format version and both state
//! identifiers.
//!
//! # Reading
//!
//! [`decode_state`] compares the identifier of a version 3 file with this build's through
//! [`check_state_semantics`]: its own state identifier gives [`IdStatus::Current`], one in
//! [`PREVIOUS_SYNC_STATE_SEMANTICS_IDS`](crate::PREVIOUS_SYNC_STATE_SEMANTICS_IDS) gives
//! [`IdStatus::VerifiedPrevious`], and any other identifier is [`PersistError::Semantics`] naming
//! the file's format version and both identifiers. A version 1 or 2 file carries a full identifier: it is looked up in
//! [`PERSISTED_FULL_SYNC_SEMANTICS_IDS`] and the state identifier it maps to is checked the same
//! way; a full identifier the table does not hold is [`PersistError::Semantics`]. The full
//! identifier itself is never compared here. A decoded `WorldState` is checked
//! for the invariants the world's unchecked accessors rely on (ids strictly
//! ascending in version 2; the arena and id map in version 1), so a malformed
//! file is an error rather than undefined behaviour. The memory a decode
//! reserves follows the length of the bytes, never a count or an id read from
//! them.

use crate::arena::{Arena, Handle};
use crate::crdt::{LwwMap, LwwRegister, OrSet, ReplicaId};
use crate::fixed_point::{Fixed, Vec3Fixed};
pub use crate::semantics::IdStatus;
use crate::semantics::{check_state_semantics, SemanticsMismatch, SYNC_STATE_SEMANTICS_ID};
use crate::world::{Entity, EntityProps, WorldState, MAX_LIVE_ENTITIES, MAX_PROPS};
use std::collections::{BTreeMap, BTreeSet};

/// The first four bytes of every encoded value.
pub const MAGIC: [u8; 4] = *b"ASYN";

/// The format version this release writes.
pub const FORMAT_VERSION: u16 = 3;

/// Earlier format versions this release still reads, as [`FormatStatus::Legacy`].
pub const LEGACY_FORMAT_VERSIONS: [u16; 2] = [1, 2];

/// An earlier full `SYNC_SEMANTICS_ID` that format 1 and 2 files carry, with the state identifier
/// it implies and the pins it was folded from (a test recomputes both folds from `pins`)
#[derive(Debug)]
pub struct PersistedFullId {
    /// The full identifier stored in the header
    pub full: [u8; 32],
    /// The fold of the state pins among `pins`
    pub state: [u8; 32],
    /// The pins `full` is the fold of
    pub pins: &'static [(&'static str, [u8; 32])],
}

/// The full identifiers format 1 and 2 files may carry. Read by [`decode_state`] only; the Hello
/// handshake never consults it.
pub const PERSISTED_FULL_SYNC_SEMANTICS_IDS: &[PersistedFullId] = &[PersistedFullId {
    full: crate::semantics::hex32(
        "705744d7b16a6be1f9510f9fa05ae9571a21f09660592e0bdf7841139173b772",
    ),
    state: crate::semantics::hex32(
        "2591de396e984ec245198762324c0e540c7d476d64c122e53ced71225791efc6",
    ),
    pins: &[
        (
            "fixed",
            crate::semantics::hex32(
                "dc65b542978e3eb841faee4a7202f7416cee5cd6ce4fdd64cebc18a97ba31f9d",
            ),
        ),
        (
            "world",
            crate::semantics::hex32(
                "875410bef5edafaaff33e1e74a2c879b987e394f784f5ef361b6d828d531f593",
            ),
        ),
        (
            "crdt",
            crate::semantics::hex32(
                "54f05313b39a7b671f53bf5c87fb4e37cb1b64f40b402ef98517047c4a730bc8",
            ),
        ),
        (
            "receive",
            crate::semantics::hex32(
                "b0f4dd9b848374e314f1d88ae220e926d8a3d05d49d08b3a77d8d9673bb2b242",
            ),
        ),
        (
            "alice-physics",
            crate::semantics::hex32(
                "79ca52976f404dfb453908970ebe45f584970ff9344a56a06aae3fc9c3c38108",
            ),
        ),
        (
            "alice-analytics",
            crate::semantics::hex32(
                "d2209b30f6f1f45baa1b638bcdfee34ac64773b2e63b9c083b2e77afc691398e",
            ),
        ),
    ],
}];

const HEADER_LEN: usize = 4 + 2 + 32;

/// Why stored bytes could not be decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PersistError {
    /// The bytes do not start with [`MAGIC`].
    BadMagic,
    /// A format version this release does not read.
    UnsupportedVersion(u16),
    /// An earlier format version, refused by [`decode_state_strict`].
    LegacyFormat(u16),
    /// The stored semantics identifier is neither this build's nor a verified
    /// earlier one. `format` is the file's format version; for a version 1 or 2
    /// file `mismatch.stored` is the state identifier its full identifier maps to
    /// in [`PERSISTED_FULL_SYNC_SEMANTICS_IDS`], or the full identifier itself
    /// when the table does not hold it.
    Semantics {
        /// The format version in the file's header.
        format: u16,
        /// The stored identifier and this build's state identifier.
        mismatch: SemanticsMismatch,
    },
    /// The bytes hold another type than the one requested.
    WrongType {
        /// The tag in the bytes.
        stored: u8,
        /// The tag of the requested type.
        expected: u8,
    },
    /// The bytes end before the value does.
    Truncated,
    /// Bytes remain after the value.
    TrailingBytes(usize),
    /// The value violates an invariant of its type.
    Malformed(&'static str),
}

impl core::fmt::Display for PersistError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BadMagic => write!(f, "not an ALICE-Sync state (bad magic)"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported state format version {v}"),
            Self::LegacyFormat(v) => write!(
                f,
                "state format version {v} is an earlier format (current {FORMAT_VERSION})"
            ),
            Self::Semantics { format, mismatch } => {
                write!(f, "state format version {format}: {mismatch}")
            }
            Self::WrongType { stored, expected } => {
                write!(f, "state holds type tag {stored}, expected {expected}")
            }
            Self::Truncated => write!(f, "state bytes are truncated"),
            Self::TrailingBytes(n) => write!(f, "{n} bytes remain after the state"),
            Self::Malformed(why) => write!(f, "malformed state: {why}"),
        }
    }
}

impl std::error::Error for PersistError {}

/// Which format version the bytes were written in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormatStatus {
    /// [`FORMAT_VERSION`].
    Current,
    /// One of [`LEGACY_FORMAT_VERSIONS`].
    Legacy(u16),
}

/// A decoded value with the relation of its stored identifier to this build.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decoded<T> {
    /// The value.
    pub value: T,
    /// Whether it was written under this build's identifier or a verified
    /// earlier one.
    pub status: IdStatus,
    /// Whether it was written in the current format or an earlier one.
    pub format: FormatStatus,
}

mod sealed {
    pub trait Sealed {}
}

/// A Law-bearing type with a durable encoding (implemented by this crate
/// only).
pub trait Persist: sealed::Sealed + Sized {
    #[doc(hidden)]
    const TAG: u8;
    #[doc(hidden)]
    fn write_body(&self, out: &mut Vec<u8>);
    #[doc(hidden)]
    fn read_body(r: &mut Reader<'_>) -> Result<Self, PersistError>;
    /// The body as format version 1 laid it out (the same as the current body
    /// for every type but the world)
    #[doc(hidden)]
    fn read_body_v1(r: &mut Reader<'_>) -> Result<Self, PersistError> {
        Self::read_body(r)
    }
}

/// Encodes `value` with the header and this build's `SYNC_STATE_SEMANTICS_ID`.
#[must_use]
pub fn encode_state<T: Persist>(value: &T) -> Vec<u8> {
    encode_with_id(value, &SYNC_STATE_SEMANTICS_ID)
}

pub(crate) fn encode_with_id<T: Persist>(value: &T, id: &[u8; 32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_LEN + 1 + 16);
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(id);
    out.push(T::TAG);
    value.write_body(&mut out);
    out
}

/// Decodes a value written by [`encode_state`].
///
/// # Errors
///
/// [`PersistError`]: a bad magic or version, an identifier that is neither
/// this build's nor a verified earlier one, another type's tag, truncated or
/// trailing bytes, or a value that violates its type's invariants.
pub fn decode_state<T: Persist>(bytes: &[u8]) -> Result<Decoded<T>, PersistError> {
    decode_with(bytes, check_stored_id)
}

/// The check of [`decode_state`]: a version 3 identifier is a state identifier; an earlier
/// version's is a full identifier, read through [`PERSISTED_FULL_SYNC_SEMANTICS_IDS`]
fn check_stored_id(id: &[u8; 32], format: FormatStatus) -> Result<IdStatus, SemanticsMismatch> {
    match format {
        FormatStatus::Current => check_state_semantics(id),
        FormatStatus::Legacy(_) => match PERSISTED_FULL_SYNC_SEMANTICS_IDS
            .iter()
            .find(|e| e.full == *id)
        {
            Some(e) => check_state_semantics(&e.state),
            None => Err(SemanticsMismatch {
                stored: *id,
                expected: SYNC_STATE_SEMANTICS_ID,
            }),
        },
    }
}

/// Decodes a value written in the current format only.
///
/// # Errors
///
/// As [`decode_state`], and [`PersistError::LegacyFormat`] for bytes in an
/// earlier format.
pub fn decode_state_strict<T: Persist>(bytes: &[u8]) -> Result<Decoded<T>, PersistError> {
    let decoded = decode_state(bytes)?;
    match decoded.format {
        FormatStatus::Current => Ok(decoded),
        FormatStatus::Legacy(v) => Err(PersistError::LegacyFormat(v)),
    }
}

pub(crate) fn decode_with<T: Persist>(
    bytes: &[u8],
    check: impl FnOnce(&[u8; 32], FormatStatus) -> Result<IdStatus, SemanticsMismatch>,
) -> Result<Decoded<T>, PersistError> {
    let mut r = Reader { bytes };
    if r.take(4)? != MAGIC {
        return Err(PersistError::BadMagic);
    }
    let version = r.u16()?;
    let format = if version == FORMAT_VERSION {
        FormatStatus::Current
    } else if LEGACY_FORMAT_VERSIONS.contains(&version) {
        FormatStatus::Legacy(version)
    } else {
        return Err(PersistError::UnsupportedVersion(version));
    };
    let id: [u8; 32] = r
        .take(32)?
        .try_into()
        .map_err(|_| PersistError::Truncated)?;
    let status = check(&id, format).map_err(|mismatch| PersistError::Semantics {
        format: version,
        mismatch,
    })?;
    let value = read_tagged::<T>(&mut r, format)?;
    if !r.bytes.is_empty() {
        return Err(PersistError::TrailingBytes(r.bytes.len()));
    }
    Ok(Decoded {
        value,
        status,
        format,
    })
}

fn read_tagged<T: Persist>(r: &mut Reader<'_>, format: FormatStatus) -> Result<T, PersistError> {
    let tag = r.u8()?;
    if tag != T::TAG {
        return Err(PersistError::WrongType {
            stored: tag,
            expected: T::TAG,
        });
    }
    match format {
        // format 2 has today's body; only format 1 laid the world out differently
        FormatStatus::Current | FormatStatus::Legacy(2) => T::read_body(r),
        FormatStatus::Legacy(_) => T::read_body_v1(r),
    }
}

/// A cursor over the bytes being decoded.
#[doc(hidden)]
pub struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], PersistError> {
        if self.bytes.len() < n {
            return Err(PersistError::Truncated);
        }
        let (head, tail) = self.bytes.split_at(n);
        self.bytes = tail;
        Ok(head)
    }
    fn u8(&mut self) -> Result<u8, PersistError> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, PersistError> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }
    fn u32(&mut self) -> Result<u32, PersistError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn i32(&mut self) -> Result<i32, PersistError> {
        Ok(self.u32()?.cast_signed())
    }
    fn u64(&mut self) -> Result<u64, PersistError> {
        let b = self.take(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(b);
        Ok(u64::from_le_bytes(a))
    }
}

impl sealed::Sealed for Fixed {}
impl Persist for Fixed {
    const TAG: u8 = 1;
    fn write_body(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.0.to_le_bytes());
    }
    fn read_body(r: &mut Reader<'_>) -> Result<Self, PersistError> {
        Ok(Self(r.i32()?))
    }
}

impl sealed::Sealed for Vec3Fixed {}
impl Persist for Vec3Fixed {
    const TAG: u8 = 2;
    fn write_body(&self, out: &mut Vec<u8>) {
        for c in [self.x, self.y, self.z] {
            c.write_body(out);
        }
    }
    fn read_body(r: &mut Reader<'_>) -> Result<Self, PersistError> {
        Ok(Self {
            x: Fixed::read_body(r)?,
            y: Fixed::read_body(r)?,
            z: Fixed::read_body(r)?,
        })
    }
}

impl sealed::Sealed for Entity {}
impl Persist for Entity {
    const TAG: u8 = 3;
    fn write_body(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.id.to_le_bytes());
        out.extend_from_slice(&self.kind.to_le_bytes());
        self.position.write_body(out);
        for v in self.properties.values {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&self.properties.active.to_le_bytes());
    }
    fn read_body(r: &mut Reader<'_>) -> Result<Self, PersistError> {
        let id = r.u32()?;
        let kind = r.u16()?;
        let position = Vec3Fixed::read_body(r)?;
        let mut values = [0i32; MAX_PROPS];
        for v in &mut values {
            *v = r.i32()?;
        }
        let active = r.u16()?;
        Ok(Self {
            id,
            kind,
            position,
            properties: EntityProps { values, active },
        })
    }
}

impl sealed::Sealed for WorldState {}
impl Persist for WorldState {
    const TAG: u8 = 4;
    fn write_body(&self, out: &mut Vec<u8>) {
        // the live entities in id order
        let live: Vec<&Entity> = self
            .ids()
            .into_iter()
            .filter_map(|(_, h)| self.entities.get(h))
            .collect();
        out.extend_from_slice(&len_u32(live.len()).to_le_bytes());
        for e in live {
            e.write_body(out);
        }
        out.extend_from_slice(&self.frame.to_le_bytes());
        out.extend_from_slice(&self.seed.to_le_bytes());
    }
    fn read_body(r: &mut Reader<'_>) -> Result<Self, PersistError> {
        let n = r.u32()? as usize;
        if n > MAX_LIVE_ENTITIES {
            return Err(PersistError::Malformed(
                "more live entities than MAX_LIVE_ENTITIES",
            ));
        }
        // no more entities can follow than the bytes hold bodies for
        let mut world = Self::default();
        world.entities = Arena::with_capacity(n.min(r.bytes.len() / ENTITY_BODY_LEN));
        let mut previous: Option<u32> = None;
        for _ in 0..n {
            let e = Entity::read_body(r)?;
            if previous.is_some_and(|p| e.id <= p) {
                return Err(PersistError::Malformed(
                    "entities are not in strictly ascending id order",
                ));
            }
            previous = Some(e.id);
            let id = e.id;
            let h = world.entities.insert(e);
            world.set_handle(id, h);
        }
        world.frame = r.u64()?;
        world.seed = r.u64()?;
        Ok(world)
    }
    fn read_body_v1(r: &mut Reader<'_>) -> Result<Self, PersistError> {
        let n = r.u32()? as usize;
        let mut slots = Vec::with_capacity(n.min(r.bytes.len()));
        for _ in 0..n {
            let state = r.u8()?;
            let generation = r.u32()?;
            let slot = match state {
                1 => (Some(Entity::read_body(r)?), u32::MAX, generation),
                0 => (None, r.u32()?, generation),
                _ => return Err(PersistError::Malformed("slot state is neither 0 nor 1")),
            };
            slots.extend([slot]);
        }
        let free_head = r.u32()?;
        let entities = Arena::from_parts(slots, free_head).map_err(PersistError::Malformed)?;
        // version 1 stored the id map densely: entry i is the handle of entity id i, or invalid
        let m = r.u32()?;
        let mut world = Self::default();
        world.entities = entities;
        for id in 0..m {
            let index = r.u32()?;
            let generation = r.u32()?;
            let h = Handle { index, generation };
            if h.is_valid() {
                if world.entities.get(h).is_none() {
                    return Err(PersistError::Malformed(
                        "an id map handle does not point to an occupied slot of its generation",
                    ));
                }
                world.set_handle(id, h);
            }
        }
        world.frame = r.u64()?;
        world.seed = r.u64()?;
        Ok(world)
    }
}

/// The bytes of one entity body: id, kind, position, the property values and the active mask
const ENTITY_BODY_LEN: usize = 4 + 2 + 3 * 4 + MAX_PROPS * 4 + 2;

impl<V: Persist + Clone> sealed::Sealed for LwwRegister<V> {}
impl<V: Persist + Clone> Persist for LwwRegister<V> {
    const TAG: u8 = 5;
    fn write_body(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.timestamp.to_le_bytes());
        out.extend_from_slice(&self.replica_id.to_le_bytes());
        out.push(V::TAG);
        self.value.write_body(out);
    }
    fn read_body(r: &mut Reader<'_>) -> Result<Self, PersistError> {
        let timestamp = r.u64()?;
        let replica_id = r.u64()?;
        let value = read_tagged::<V>(r, FormatStatus::Current)?;
        Ok(Self {
            value,
            timestamp,
            replica_id,
        })
    }
}

impl sealed::Sealed for u64 {}
impl Persist for u64 {
    const TAG: u8 = 6;
    fn write_body(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }
    fn read_body(r: &mut Reader<'_>) -> Result<Self, PersistError> {
        r.u64()
    }
}

impl sealed::Sealed for bool {}
impl Persist for bool {
    const TAG: u8 = 7;
    fn write_body(&self, out: &mut Vec<u8>) {
        out.push(u8::from(*self));
    }
    fn read_body(r: &mut Reader<'_>) -> Result<Self, PersistError> {
        match r.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(PersistError::Malformed("a bool that is neither 0 nor 1")),
        }
    }
}

impl<V: Persist> sealed::Sealed for Option<V> {}
impl<V: Persist> Persist for Option<V> {
    const TAG: u8 = 8;
    fn write_body(&self, out: &mut Vec<u8>) {
        match self {
            None => out.push(0),
            Some(v) => {
                out.push(1);
                out.push(V::TAG);
                v.write_body(out);
            }
        }
    }
    fn read_body(r: &mut Reader<'_>) -> Result<Self, PersistError> {
        match r.u8()? {
            0 => Ok(None),
            1 => Ok(Some(read_tagged::<V>(r, FormatStatus::Current)?)),
            _ => Err(PersistError::Malformed("an option that is neither 0 nor 1")),
        }
    }
}

/// A count read before a sequence: refused when even one byte per element would not fit
fn count(r: &mut Reader<'_>) -> Result<usize, PersistError> {
    let n = r.u32()? as usize;
    if n > r.bytes.len() {
        return Err(PersistError::Truncated);
    }
    Ok(n)
}

fn write_tag_pairs(out: &mut Vec<u8>, pairs: &BTreeSet<(ReplicaId, u64)>) {
    out.extend_from_slice(&len_u32(pairs.len()).to_le_bytes());
    for (replica, n) in pairs {
        out.extend_from_slice(&replica.to_le_bytes());
        out.extend_from_slice(&n.to_le_bytes());
    }
}

/// Pairs in strictly ascending order (the canonical form a `BTreeSet` writes)
fn read_tag_pairs(r: &mut Reader<'_>) -> Result<BTreeSet<(ReplicaId, u64)>, PersistError> {
    let n = count(r)?;
    let mut out = BTreeSet::new();
    let mut last = None;
    for _ in 0..n {
        let p = (r.u64()?, r.u64()?);
        if last.is_some_and(|l| l >= p) {
            return Err(PersistError::Malformed("tags not in ascending order"));
        }
        last = Some(p);
        out.insert(p);
    }
    Ok(out)
}

impl<V: Persist + Clone + Ord> sealed::Sealed for OrSet<V> {}
impl<V: Persist + Clone + Ord> Persist for OrSet<V> {
    const TAG: u8 = 9;
    fn write_body(&self, out: &mut Vec<u8>) {
        let (elements, counters, tombstones) = self.parts();
        out.extend_from_slice(&len_u32(elements.len()).to_le_bytes());
        for (v, tags) in elements {
            out.push(V::TAG);
            v.write_body(out);
            write_tag_pairs(out, tags);
        }
        out.extend_from_slice(&len_u32(counters.len()).to_le_bytes());
        for (replica, n) in counters {
            out.extend_from_slice(&replica.to_le_bytes());
            out.extend_from_slice(&n.to_le_bytes());
        }
        write_tag_pairs(out, tombstones);
    }
    fn read_body(r: &mut Reader<'_>) -> Result<Self, PersistError> {
        let n = count(r)?;
        let mut elements = BTreeMap::new();
        for _ in 0..n {
            let v = read_tagged::<V>(r, FormatStatus::Current)?;
            if elements
                .last_key_value()
                .is_some_and(|(last, _)| *last >= v)
            {
                return Err(PersistError::Malformed("set values not in ascending order"));
            }
            let tags = read_tag_pairs(r)?;
            if tags.is_empty() {
                return Err(PersistError::Malformed("a set value with no tag"));
            }
            elements.insert(v, tags);
        }
        let n = count(r)?;
        let mut counters = BTreeMap::new();
        for _ in 0..n {
            let (replica, c) = (r.u64()?, r.u64()?);
            if counters
                .last_key_value()
                .is_some_and(|(last, _)| *last >= replica)
            {
                return Err(PersistError::Malformed("counters not in ascending order"));
            }
            counters.insert(replica, c);
        }
        let tombstones = read_tag_pairs(r)?;
        Ok(Self::from_parts(elements, counters, tombstones))
    }
}

impl<K: Persist + Clone + Ord, V: Persist + Clone> sealed::Sealed for LwwMap<K, V> {}
impl<K: Persist + Clone + Ord, V: Persist + Clone> Persist for LwwMap<K, V> {
    const TAG: u8 = 10;
    fn write_body(&self, out: &mut Vec<u8>) {
        // the replicated entries only: the writing replica and its clock are local
        let entries = self.registers();
        out.extend_from_slice(&len_u32(entries.len()).to_le_bytes());
        for (k, reg) in entries {
            out.push(K::TAG);
            k.write_body(out);
            reg.write_body(out);
        }
    }
    fn read_body(r: &mut Reader<'_>) -> Result<Self, PersistError> {
        let n = count(r)?;
        let mut entries = BTreeMap::new();
        for _ in 0..n {
            let k = read_tagged::<K>(r, FormatStatus::Current)?;
            if entries.last_key_value().is_some_and(|(last, _)| *last >= k) {
                return Err(PersistError::Malformed("map keys not in ascending order"));
            }
            entries.insert(k, LwwRegister::<Option<V>>::read_body(r)?);
        }
        Ok(Self::from_registers(entries))
    }
}

fn len_u32(n: usize) -> u32 {
    u32::try_from(n).expect("more than u32::MAX elements cannot be encoded")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantics::check_semantics_against;

    /// An identifier listed as a verified earlier one is accepted with that
    /// status (the release's own list is empty, so the list is passed here).
    #[test]
    fn a_previous_identifier_is_accepted_as_verified_previous() {
        let earlier = [7u8; 32];
        let bytes = encode_with_id(&Fixed(42), &earlier);
        let d = decode_with::<Fixed>(&bytes, |id, _| {
            check_semantics_against(id, &SYNC_STATE_SEMANTICS_ID, &[earlier])
        })
        .unwrap();
        assert_eq!(d.value, Fixed(42));
        assert_eq!(d.status, IdStatus::VerifiedPrevious(earlier));
        // the same bytes against the release's own (empty) list are refused
        assert!(matches!(
            decode_state::<Fixed>(&bytes),
            Err(PersistError::Semantics { mismatch: m, .. }) if m.stored == earlier
        ));
    }

    /// A version 1 world whose free list or id map would break the unchecked
    /// accessors is refused, not decoded, each with the reason of the check that
    /// caught it (the fixture written by the last release of that format). `decode_state` refuses
    /// every version 1 file at its identifier before the body, so the body checks are read here past
    /// an identifier check that accepts.
    #[test]
    fn malformed_world_state_is_refused() {
        let bytes = include_bytes!("../tests/data/persist/world_state_v1.bin").to_vec();
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
        let refused =
            |bad: &[u8]| match decode_with::<WorldState>(bad, |_, _| Ok(IdStatus::Current)) {
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
}
