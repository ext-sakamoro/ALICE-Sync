//! Durable form of ALICE-Sync's Law-bearing state, stamped with
//! [`SYNC_SEMANTICS_ID`].
//!
//! The fixed-point types ([`Fixed`], [`Vec3Fixed`]) and the state built from
//! them ([`Entity`], [`WorldState`], and with the `async` feature
//! `LwwRegister` of those) are the values
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
//! # Format (version 2)
//!
//! All integers little-endian.
//!
//! | bytes | content |
//! |---|---|
//! | 4 | magic `ASYN` |
//! | 2 | format version, `2` |
//! | 32 | the writer's `SYNC_SEMANTICS_ID` |
//! | 1 | type tag (`1` Fixed, `2` Vec3Fixed, `3` Entity, `4` WorldState, `5` LwwRegister) |
//! | rest | the value's body |
//!
//! Bodies: `Fixed` is its raw `i32`; `Vec3Fixed` is `x`, `y`, `z`; `Entity` is
//! `id: u32`, `kind: u16`, the position, the 16 property values (`i32`) and the
//! active mask (`u16`); `WorldState` is the number of live entities (`u32`),
//! their bodies in ascending id order, `frame: u64` and `seed: u64`;
//! `LwwRegister<V>` is `timestamp: u64`, `replica_id: u64`, then the tag and
//! body of `V`.
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
//! Version 1 stored the arena layout (slots, generations, free list, a dense id
//! map). It is still read, as [`FormatStatus::Legacy`]; [`decode_state_strict`]
//! refuses it.
//!
//! # Reading
//!
//! [`decode_state`] compares the stored identifier with this build's through
//! [`check_semantics`]: its own identifier gives [`IdStatus::Current`], one in
//! [`PREVIOUS_SYNC_SEMANTICS_IDS`](crate::PREVIOUS_SYNC_SEMANTICS_IDS) gives
//! [`IdStatus::VerifiedPrevious`], and any other identifier is
//! [`PersistError::Semantics`] naming both. A decoded `WorldState` is checked
//! for the invariants the world's unchecked accessors rely on (ids strictly
//! ascending in version 2; the arena and id map in version 1), so a malformed
//! file is an error rather than undefined behaviour. The memory a decode
//! reserves follows the length of the bytes, never a count or an id read from
//! them.

use std::collections::BTreeMap;

use crate::arena::{Arena, Handle};
#[cfg(feature = "async")]
use crate::crdt::LwwRegister;
use crate::fixed_point::{Fixed, Vec3Fixed};
pub use crate::semantics::IdStatus;
use crate::semantics::{check_semantics, SemanticsMismatch, SYNC_SEMANTICS_ID};
use crate::world::{Entity, EntityProps, WorldState, MAX_LIVE_ENTITIES, MAX_PROPS};

/// The first four bytes of every encoded value.
pub const MAGIC: [u8; 4] = *b"ASYN";

/// The format version this release writes.
pub const FORMAT_VERSION: u16 = 2;

/// Earlier format versions this release still reads, as [`FormatStatus::Legacy`].
pub const LEGACY_FORMAT_VERSIONS: [u16; 1] = [1];

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
    /// earlier one.
    Semantics(SemanticsMismatch),
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
            Self::Semantics(m) => write!(f, "{m}"),
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

impl From<SemanticsMismatch> for PersistError {
    fn from(m: SemanticsMismatch) -> Self {
        Self::Semantics(m)
    }
}

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

/// Encodes `value` with the header and this build's `SYNC_SEMANTICS_ID`.
#[must_use]
pub fn encode_state<T: Persist>(value: &T) -> Vec<u8> {
    encode_with_id(value, &SYNC_SEMANTICS_ID)
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
    decode_with(bytes, check_semantics)
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
    check: impl FnOnce(&[u8; 32]) -> Result<IdStatus, SemanticsMismatch>,
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
    let status = check(&id)?;
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
        FormatStatus::Current => T::read_body(r),
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
        // the live entities in id order: the id map holds exactly them, ordered by id
        let live: Vec<&Entity> = self
            .id_map
            .values()
            .filter(|h| h.is_valid())
            .filter_map(|&h| self.entities.get(h))
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
        let mut entities = Arena::with_capacity(n.min(r.bytes.len() / ENTITY_BODY_LEN));
        let mut id_map = BTreeMap::new();
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
            id_map.insert(id, entities.insert(e));
        }
        let frame = r.u64()?;
        let seed = r.u64()?;
        Ok(Self {
            entities,
            id_map,
            frame,
            seed,
        })
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
        let mut id_map = BTreeMap::new();
        for id in 0..m {
            let index = r.u32()?;
            let generation = r.u32()?;
            let h = Handle { index, generation };
            if h.is_valid() {
                if entities.get(h).is_none() {
                    return Err(PersistError::Malformed(
                        "an id map handle does not point to an occupied slot of its generation",
                    ));
                }
                id_map.insert(id, h);
            }
        }
        let frame = r.u64()?;
        let seed = r.u64()?;
        Ok(Self {
            entities,
            id_map,
            frame,
            seed,
        })
    }
}

/// The bytes of one entity body: id, kind, position, the property values and the active mask
const ENTITY_BODY_LEN: usize = 4 + 2 + 3 * 4 + MAX_PROPS * 4 + 2;

#[cfg(feature = "async")]
impl<V: Persist + Clone> sealed::Sealed for LwwRegister<V> {}
#[cfg(feature = "async")]
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
        let d = decode_with::<Fixed>(&bytes, |id| {
            check_semantics_against(id, &SYNC_SEMANTICS_ID, &[earlier])
        })
        .unwrap();
        assert_eq!(d.value, Fixed(42));
        assert_eq!(d.status, IdStatus::VerifiedPrevious(earlier));
        // the same bytes against the release's own (empty) list are refused
        assert!(matches!(
            decode_state::<Fixed>(&bytes),
            Err(PersistError::Semantics(m)) if m.stored == earlier
        ));
    }
}
