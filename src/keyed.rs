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

#[cfg(feature = "serde-raw")]
pub(crate) mod serde_sorted {
    //! `serde-raw` form of a [`KeyedMap`]: the entries as a sequence in ascending key order, so the
    //! raw bytes do not depend on the table's key

    use super::{keyed_map, sorted, KeyedMap};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub(crate) fn serialize<K, V, S>(map: &KeyedMap<K, V>, s: S) -> Result<S::Ok, S::Error>
    where
        K: Ord + Copy + Serialize,
        V: Copy + Serialize,
        S: Serializer,
    {
        sorted(map).serialize(s)
    }

    pub(crate) fn deserialize<'de, K, V, D>(d: D) -> Result<KeyedMap<K, V>, D::Error>
    where
        K: Eq + std::hash::Hash + Deserialize<'de>,
        V: Deserialize<'de>,
        D: Deserializer<'de>,
    {
        let entries: Vec<(K, V)> = Vec::deserialize(d)?;
        let mut map = keyed_map();
        map.extend(entries);
        Ok(map)
    }
}
