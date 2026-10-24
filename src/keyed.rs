/*
    ALICE-Sync
    Copyright (C) 2026 Moroya Sakamoto

    License: LicenseRef-Proprietary (see LICENSE-Proprietary)
*/

//! A lookup table for keys another node chooses (entity ids).
//!
//! An entity id is chosen by whichever node spawns the entity, so a table indexed by it must not
//! size itself by the id (a dense table of `u32::MAX` entries) and must not let the chooser make
//! lookups slow (a hash with a fixed, known seed lets a peer pick ids that all collide). This map
//! hashes with SipHash under a key drawn per map.
//!
//! The key changes only where entries sit in the table, never what a lookup returns. Nothing that
//! reaches a world hash, a stored byte or another node iterates the table in its own order: those
//! paths take a copy sorted by key ([`sorted`]). `tests/keyed_lookup.rs` checks that worlds built
//! with different keys give bit-identical stored bytes and world hashes.

#![allow(
    clippy::disallowed_types,
    clippy::disallowed_methods,
    reason = "the one keyed lookup table of the crate: lookups only; ordered output goes through \
              `sorted` (ALICE-Law G10 allowance in scripts/entropy_guard.py, G13 lines in \
              scripts/order_allowlist.toml)"
)]

use std::collections::hash_map::RandomState;
use std::collections::HashMap;

/// A lookup table hashed under a key drawn per table
pub(crate) type KeyedMap<K, V> = HashMap<K, V, RandomState>;

/// An empty table with a fresh key
pub(crate) fn keyed_map<K, V>() -> KeyedMap<K, V> {
    HashMap::with_hasher(RandomState::new())
}

/// The entries in ascending key order: the only way an entry list leaves the table
pub(crate) fn sorted<K: Ord + Copy, V: Copy>(map: &KeyedMap<K, V>) -> Vec<(K, V)> {
    let mut entries: Vec<(K, V)> = map.iter().map(|(&k, &v)| (k, v)).collect();
    entries.sort_unstable_by_key(|&(k, _)| k);
    entries
}

/// Ids below this are looked up in a dense table, which grows up to it and never past; larger ids
/// are looked up in a keyed table
pub(crate) const DENSE_IDS: u32 = 4 * crate::world::MAX_LIVE_ENTITIES as u32;

/// A value of an [`IdIndex`] with a reserved value marking an empty dense entry
pub(crate) trait Slot: Copy + PartialEq {
    /// Never stored as a value
    const EMPTY: Self;
}

impl Slot for crate::arena::Handle {
    const EMPTY: Self = Self::INVALID;
}

impl Slot for u32 {
    const EMPTY: Self = Self::MAX;
}

/// An index from entity id to `V`: a dense table for ids below its bound, a keyed table above
#[derive(Debug, Clone)]
pub(crate) struct IdIndex<V> {
    bound: u32,
    dense: Vec<V>,
    keyed: KeyedMap<u32, V>,
    len: usize,
}

impl<V: Slot> Default for IdIndex<V> {
    fn default() -> Self {
        Self::with_dense_bound(DENSE_IDS)
    }
}

impl<V: Slot> IdIndex<V> {
    /// An empty index whose dense table covers ids below `bound` (at most [`DENSE_IDS`])
    pub(crate) fn with_dense_bound(bound: u32) -> Self {
        Self {
            bound: bound.min(DENSE_IDS),
            dense: Vec::new(),
            keyed: keyed_map(),
            len: 0,
        }
    }

    #[inline]
    pub(crate) fn get(&self, id: u32) -> Option<V> {
        // the dense table never reaches past the bound, so an id it holds is below it
        match self.dense.get(id as usize) {
            Some(&v) => (v != V::EMPTY).then_some(v),
            None if id >= self.bound => self.keyed.get(&id).copied(),
            None => None,
        }
    }

    /// Set the entry of `id`, returning the one it replaces (`value` must not be [`Slot::EMPTY`])
    pub(crate) fn insert(&mut self, id: u32, value: V) -> Option<V> {
        debug_assert!(value != V::EMPTY, "the empty marker is not a value");
        let old = if id < self.bound {
            let i = id as usize;
            if i >= self.dense.len() {
                // grow exactly, at least doubling, never past the bound
                let want = (i + 1).max(2 * self.dense.len()).min(self.bound as usize);
                self.dense.reserve_exact(want - self.dense.len());
                self.dense.resize(want, V::EMPTY);
            }
            let old = std::mem::replace(&mut self.dense[i], value);
            (old != V::EMPTY).then_some(old)
        } else {
            self.keyed.insert(id, value)
        };
        if old.is_none() {
            self.len += 1;
        }
        old
    }

    pub(crate) fn remove(&mut self, id: u32) -> Option<V> {
        let old = if id < self.bound {
            self.dense
                .get_mut(id as usize)
                .map(|v| std::mem::replace(v, V::EMPTY))
                .filter(|&v| v != V::EMPTY)
        } else {
            self.keyed.remove(&id)
        };
        if old.is_some() {
            self.len -= 1;
        }
        old
    }

    pub(crate) const fn len(&self) -> usize {
        self.len
    }

    /// The entries in ascending id order (every keyed id is above every dense one)
    pub(crate) fn sorted(&self) -> Vec<(u32, V)> {
        let mut out = Vec::with_capacity(self.len);
        out.extend(
            self.dense
                .iter()
                .enumerate()
                .filter(|(_, &v)| v != V::EMPTY)
                .map(|(i, &v)| (u32::try_from(i).unwrap_or(u32::MAX), v)),
        );
        out.extend(sorted(&self.keyed));
        out
    }
}

#[cfg(feature = "serde-raw")]
pub(crate) mod serde_sorted {
    //! `serde-raw` form of an [`IdIndex`]: the entries as a sequence in ascending id order, so the
    //! raw bytes do not depend on the table's key or on where the dense part ends

    use super::{IdIndex, Slot};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub(crate) fn serialize<V, S>(index: &IdIndex<V>, s: S) -> Result<S::Ok, S::Error>
    where
        V: Slot + Serialize,
        S: Serializer,
    {
        index.sorted().serialize(s)
    }

    pub(crate) fn deserialize<'de, V, D>(d: D) -> Result<IdIndex<V>, D::Error>
    where
        V: Slot + Deserialize<'de>,
        D: Deserializer<'de>,
    {
        let entries: Vec<(u32, V)> = Vec::deserialize(d)?;
        let mut index = IdIndex::default();
        for (id, v) in entries {
            index.insert(id, v);
        }
        Ok(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_dense_part_grows_lazily_and_never_past_its_bound() {
        let mut ix: IdIndex<u32> = IdIndex::with_dense_bound(100);
        assert_eq!(ix.dense.capacity(), 0);
        ix.insert(3, 1);
        assert!(ix.dense.len() >= 4 && ix.dense.len() <= 100);
        ix.insert(99, 2);
        assert_eq!(ix.dense.len(), 100);
        assert!(
            ix.dense.capacity() <= 100,
            "capacity {}",
            ix.dense.capacity()
        );
        ix.insert(100, 3);
        ix.insert(u32::MAX, 4);
        assert_eq!(ix.dense.len(), 100);
        assert_eq!(ix.len(), 4);
        assert_eq!(ix.sorted(), [(3, 1), (99, 2), (100, 3), (u32::MAX, 4)]);
        assert_eq!(ix.remove(99), Some(2));
        assert_eq!(ix.remove(99), None);
        // an emptied or never-set dense entry reads as absent, not as the empty marker
        assert_eq!((ix.get(99), ix.get(50), ix.get(0)), (None, None, None));
        assert_eq!(ix.remove(u32::MAX), Some(4));
        assert_eq!((ix.len(), ix.get(3), ix.get(100)), (2, Some(1), Some(3)));
        assert_eq!(ix.insert(3, 9), Some(1));
        assert_eq!(ix.len(), 2);
    }

    #[test]
    fn the_bound_is_capped_by_dense_ids() {
        let ix: IdIndex<u32> = IdIndex::with_dense_bound(u32::MAX);
        assert_eq!(ix.bound, DENSE_IDS);
    }
}
