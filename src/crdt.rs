#![allow(clippy::missing_const_for_fn)]

//! Conflict-Free Replicated Data Types (CRDTs)
//!
//! Eventually consistent data structures that converge without coordination.
//! No central server, no locking, no conflict resolution — mathematically
//! guaranteed to converge when all updates are delivered.
//!
//! # Types
//!
//! | CRDT | Use Case |
//! |------|----------|
//! | [`LwwRegister`] | Single-value state (position, config, status) |
//! | [`GCounter`] | Monotonic counter (views, events, sensor readings) |
//! | [`PnCounter`] | Increment/decrement counter (inventory, score) |
//! | [`OrSet`] | Set with add/remove (tags, members, permissions) |
//! | [`LwwMap`] | Key-value store (user profiles, device config) |
//!
//! # Sync Protocol
//!
//! All CRDTs implement [`CrdtMergeable`] — a single `merge()` call
//! integrates remote state. No ordering constraints, no causal tracking.
//! Just merge whenever updates arrive.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Node identity for CRDT attribution
pub type ReplicaId = u64;

/// Lamport timestamp for LWW ordering
pub type LamportTs = u64;

/// Trait for mergeable CRDTs
pub trait CrdtMergeable {
    /// Merge remote state into local state.
    /// Commutative, associative, idempotent — order doesn't matter.
    fn merge(&mut self, other: &Self);
}

// ============================================================================
// LWW-Register (Last-Writer-Wins)
// ============================================================================

/// Last-Writer-Wins Register.
///
/// Stores a single value. Concurrent writes resolve by highest timestamp.
/// Ties broken by highest replica ID.
///
/// Use for: entity position, device status, configuration values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LwwRegister<V: Clone> {
    pub value: V,
    pub timestamp: LamportTs,
    pub replica_id: ReplicaId,
}

impl<V: Clone> LwwRegister<V> {
    /// Create a new register
    #[must_use]
    pub fn new(value: V, replica_id: ReplicaId) -> Self {
        Self {
            value,
            timestamp: 0,
            replica_id,
        }
    }

    /// Set a new value (increments timestamp)
    pub fn set(&mut self, value: V) {
        self.timestamp += 1;
        self.value = value;
    }

    /// Set with explicit timestamp
    pub fn set_at(&mut self, value: V, timestamp: LamportTs) {
        self.timestamp = timestamp;
        self.value = value;
    }
}

impl<V: Clone> CrdtMergeable for LwwRegister<V> {
    fn merge(&mut self, other: &Self) {
        if other.timestamp > self.timestamp
            || (other.timestamp == self.timestamp && other.replica_id > self.replica_id)
        {
            self.value = other.value.clone();
            self.timestamp = other.timestamp;
            self.replica_id = other.replica_id;
        }
    }
}

// ============================================================================
// G-Counter (Grow-only Counter)
// ============================================================================

/// Grow-only Counter.
///
/// Each replica increments its own slot. Total = sum of all slots.
/// Monotonically increasing — cannot decrement.
///
/// Use for: event counts, sensor readings, page views.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GCounter {
    counts: BTreeMap<ReplicaId, u64>,
}

impl Default for GCounter {
    fn default() -> Self {
        Self::new()
    }
}

impl GCounter {
    #[must_use]
    pub fn new() -> Self {
        Self {
            counts: BTreeMap::new(),
        }
    }

    /// Increment this replica's count
    pub fn increment(&mut self, replica_id: ReplicaId) {
        *self.counts.entry(replica_id).or_insert(0) += 1;
    }

    /// Increment by a specific amount
    pub fn increment_by(&mut self, replica_id: ReplicaId, amount: u64) {
        *self.counts.entry(replica_id).or_insert(0) += amount;
    }

    /// Total count across all replicas
    #[must_use]
    pub fn value(&self) -> u64 {
        self.counts.values().sum()
    }

    /// This replica's local count
    #[must_use]
    pub fn local_count(&self, replica_id: ReplicaId) -> u64 {
        self.counts.get(&replica_id).copied().unwrap_or(0)
    }
}

impl CrdtMergeable for GCounter {
    fn merge(&mut self, other: &Self) {
        for (&replica, &count) in &other.counts {
            let entry = self.counts.entry(replica).or_insert(0);
            *entry = (*entry).max(count);
        }
    }
}

// ============================================================================
// PN-Counter (Positive-Negative Counter)
// ============================================================================

/// Positive-Negative Counter.
///
/// Supports both increment and decrement by combining two G-Counters.
/// Value = `P.value()` - `N.value()`
///
/// Use for: inventory levels, bidirectional scores, resource pools.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PnCounter {
    p: GCounter,
    n: GCounter,
}

impl Default for PnCounter {
    fn default() -> Self {
        Self::new()
    }
}

impl PnCounter {
    #[must_use]
    pub fn new() -> Self {
        Self {
            p: GCounter::new(),
            n: GCounter::new(),
        }
    }

    /// Increment the counter
    pub fn increment(&mut self, replica_id: ReplicaId) {
        self.p.increment(replica_id);
    }

    /// Decrement the counter
    pub fn decrement(&mut self, replica_id: ReplicaId) {
        self.n.increment(replica_id);
    }

    /// Current value (can be negative)
    #[must_use]
    pub fn value(&self) -> i64 {
        self.p.value() as i64 - self.n.value() as i64
    }
}

impl CrdtMergeable for PnCounter {
    fn merge(&mut self, other: &Self) {
        self.p.merge(&other.p);
        self.n.merge(&other.n);
    }
}

// ============================================================================
// OR-Set (Observed-Remove Set)
// ============================================================================

/// Observed-Remove Set.
///
/// Add and remove elements. Concurrent add + remove of the same element
/// resolves to "add wins" (the element stays).
///
/// Use for: tag collections, member lists, permission sets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrSet<V: Clone + Ord> {
    /// Elements: value → set of unique tags (`replica_id`, counter)
    elements: BTreeMap<V, BTreeSet<(ReplicaId, u64)>>,
    /// Per-replica tag counter
    counters: BTreeMap<ReplicaId, u64>,
    /// Tombstones: tags that have been removed
    tombstones: BTreeSet<(ReplicaId, u64)>,
}

impl<V: Clone + Ord> Default for OrSet<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V: Clone + Ord> OrSet<V> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            elements: BTreeMap::new(),
            counters: BTreeMap::new(),
            tombstones: BTreeSet::new(),
        }
    }

    /// Add an element
    pub fn add(&mut self, replica_id: ReplicaId, value: V) {
        let counter = self.counters.entry(replica_id).or_insert(0);
        *counter += 1;
        let tag = (replica_id, *counter);

        self.elements.entry(value).or_default().insert(tag);
    }

    /// Remove an element (tombstones all current tags for this value)
    pub fn remove(&mut self, value: &V) {
        if let Some(tags) = self.elements.remove(value) {
            for tag in tags {
                self.tombstones.insert(tag);
            }
        }
    }

    /// Check if the set contains a value
    #[must_use]
    pub fn contains(&self, value: &V) -> bool {
        self.elements
            .get(value)
            .is_some_and(|tags| !tags.is_empty())
    }

    /// Get all elements in the set
    #[must_use]
    pub fn values(&self) -> Vec<&V> {
        self.elements
            .iter()
            .filter(|(_, tags)| !tags.is_empty())
            .map(|(v, _)| v)
            .collect()
    }

    /// Number of elements
    #[must_use]
    pub fn len(&self) -> usize {
        self.elements
            .values()
            .filter(|tags| !tags.is_empty())
            .count()
    }

    /// Check if empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<V: Clone + Ord> CrdtMergeable for OrSet<V> {
    fn merge(&mut self, other: &Self) {
        // Merge elements: union of all tags, minus tombstoned tags
        for (value, other_tags) in &other.elements {
            let local_tags = self.elements.entry(value.clone()).or_default();
            for &tag in other_tags {
                if !self.tombstones.contains(&tag) {
                    local_tags.insert(tag);
                }
            }
        }

        // Merge tombstones
        for &tag in &other.tombstones {
            self.tombstones.insert(tag);
            // Remove tombstoned tags from elements
            for tags in self.elements.values_mut() {
                tags.remove(&tag);
            }
        }

        // Merge counters
        for (&replica, &count) in &other.counters {
            let entry = self.counters.entry(replica).or_insert(0);
            *entry = (*entry).max(count);
        }
    }
}

// ============================================================================
// LWW-Map (Last-Writer-Wins Map)
// ============================================================================

/// Last-Writer-Wins Map.
///
/// Key-value store where each key is an independent LWW-Register.
/// Concurrent writes to the same key resolve by timestamp.
///
/// Use for: user profiles, device config, distributed settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LwwMap<K: Clone + Ord, V: Clone> {
    entries: BTreeMap<K, LwwRegister<Option<V>>>,
    replica_id: ReplicaId,
    clock: LamportTs,
}

impl<K: Clone + Ord, V: Clone> LwwMap<K, V> {
    /// Create a new map
    #[must_use]
    pub fn new(replica_id: ReplicaId) -> Self {
        Self {
            entries: BTreeMap::new(),
            replica_id,
            clock: 0,
        }
    }

    /// Insert or update a key-value pair
    pub fn insert(&mut self, key: K, value: V) {
        self.clock += 1;
        let register = self
            .entries
            .entry(key)
            .or_insert_with(|| LwwRegister::new(None, self.replica_id));
        register.set_at(Some(value), self.clock);
        register.replica_id = self.replica_id;
    }

    /// Remove a key (tombstone — sets value to None with higher timestamp)
    pub fn remove(&mut self, key: &K) {
        self.clock += 1;
        if let Some(register) = self.entries.get_mut(key) {
            register.set_at(None, self.clock);
            register.replica_id = self.replica_id;
        }
    }

    /// Get a value by key
    #[must_use]
    pub fn get(&self, key: &K) -> Option<&V> {
        self.entries.get(key).and_then(|r| r.value.as_ref())
    }

    /// Check if key exists (not tombstoned)
    #[must_use]
    pub fn contains_key(&self, key: &K) -> bool {
        self.entries.get(key).is_some_and(|r| r.value.is_some())
    }

    /// Get all live key-value pairs
    #[must_use]
    pub fn entries(&self) -> Vec<(&K, &V)> {
        self.entries
            .iter()
            .filter_map(|(k, r)| r.value.as_ref().map(|v| (k, v)))
            .collect()
    }

    /// Number of live entries
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.values().filter(|r| r.value.is_some()).count()
    }

    /// Check if empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<K: Clone + Ord, V: Clone> CrdtMergeable for LwwMap<K, V> {
    fn merge(&mut self, other: &Self) {
        for (key, other_reg) in &other.entries {
            let local_reg = self
                .entries
                .entry(key.clone())
                .or_insert_with(|| LwwRegister::new(None, self.replica_id));
            local_reg.merge(other_reg);
        }
        self.clock = self.clock.max(other.clock);
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- LWW-Register ---

    #[test]
    fn test_lww_register_basic() {
        let mut reg = LwwRegister::new(0i32, 1);
        reg.set(42);
        assert_eq!(reg.value, 42);
        assert_eq!(reg.timestamp, 1);
    }

    #[test]
    fn test_lww_register_merge_higher_ts_wins() {
        let mut a = LwwRegister::new(10, 1);
        a.set(20); // ts=1

        let mut b = LwwRegister::new(10, 2);
        b.set(30);
        b.set(40); // ts=2

        a.merge(&b);
        assert_eq!(a.value, 40);
    }

    #[test]
    fn test_lww_register_merge_tie_breaks_by_replica() {
        let mut a = LwwRegister::new(0, 1);
        a.set_at(10, 5);

        let mut b = LwwRegister::new(0, 2);
        b.set_at(20, 5); // same ts, higher replica

        a.merge(&b);
        assert_eq!(a.value, 20); // replica 2 wins
    }

    // --- G-Counter ---

    #[test]
    fn test_gcounter_basic() {
        let mut c = GCounter::new();
        c.increment(1);
        c.increment(1);
        c.increment(2);
        assert_eq!(c.value(), 3);
        assert_eq!(c.local_count(1), 2);
    }

    #[test]
    fn test_gcounter_merge() {
        let mut a = GCounter::new();
        a.increment_by(1, 5);

        let mut b = GCounter::new();
        b.increment_by(2, 3);

        a.merge(&b);
        assert_eq!(a.value(), 8); // 5 + 3
    }

    #[test]
    fn test_gcounter_merge_max() {
        let mut a = GCounter::new();
        a.increment_by(1, 10);

        let mut b = GCounter::new();
        b.increment_by(1, 7); // stale count for replica 1

        a.merge(&b);
        assert_eq!(a.value(), 10); // max(10, 7)
    }

    // --- PN-Counter ---

    #[test]
    fn test_pn_counter_basic() {
        let mut c = PnCounter::new();
        c.increment(1);
        c.increment(1);
        c.decrement(1);
        assert_eq!(c.value(), 1);
    }

    #[test]
    fn test_pn_counter_negative() {
        let mut c = PnCounter::new();
        c.decrement(1);
        c.decrement(1);
        assert_eq!(c.value(), -2);
    }

    #[test]
    fn test_pn_counter_merge() {
        let mut a = PnCounter::new();
        a.increment(1);
        a.increment(1);

        let mut b = PnCounter::new();
        b.decrement(2);

        a.merge(&b);
        assert_eq!(a.value(), 1); // 2 - 1
    }

    // --- OR-Set ---

    #[test]
    fn test_or_set_add_remove() {
        let mut s = OrSet::new();
        s.add(1, "hello".to_string());
        assert!(s.contains(&"hello".to_string()));

        s.remove(&"hello".to_string());
        assert!(!s.contains(&"hello".to_string()));
    }

    #[test]
    fn test_or_set_add_wins_over_concurrent_remove() {
        let mut a = OrSet::new();
        a.add(1, "x".to_string());

        // B clones A, then removes
        let mut b = a.clone();
        b.remove(&"x".to_string());

        // A adds again concurrently
        a.add(1, "x".to_string());

        // Merge: add wins because A's new tag is not in B's tombstones
        a.merge(&b);
        assert!(a.contains(&"x".to_string()));
    }

    #[test]
    fn test_or_set_merge() {
        let mut a = OrSet::new();
        a.add(1, 10);
        a.add(1, 20);

        let mut b = OrSet::new();
        b.add(2, 20);
        b.add(2, 30);

        a.merge(&b);
        assert_eq!(a.len(), 3); // {10, 20, 30}
        assert!(a.contains(&10));
        assert!(a.contains(&20));
        assert!(a.contains(&30));
    }

    // --- LWW-Map ---

    #[test]
    fn test_lww_map_basic() {
        let mut m: LwwMap<String, i32> = LwwMap::new(1);
        m.insert("key".to_string(), 42);
        assert_eq!(m.get(&"key".to_string()), Some(&42));
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn test_lww_map_remove() {
        let mut m: LwwMap<String, i32> = LwwMap::new(1);
        m.insert("key".to_string(), 42);
        m.remove(&"key".to_string());
        assert_eq!(m.get(&"key".to_string()), None);
        assert_eq!(m.len(), 0);
    }

    #[test]
    fn test_lww_map_merge() {
        let mut a: LwwMap<String, i32> = LwwMap::new(1);
        a.insert("x".to_string(), 10);

        let mut b: LwwMap<String, i32> = LwwMap::new(2);
        b.insert("y".to_string(), 20);

        a.merge(&b);
        assert_eq!(a.get(&"x".to_string()), Some(&10));
        assert_eq!(a.get(&"y".to_string()), Some(&20));
        assert_eq!(a.len(), 2);
    }

    #[test]
    fn test_lww_map_merge_conflict() {
        let mut a: LwwMap<String, i32> = LwwMap::new(1);
        a.insert("key".to_string(), 10); // clock=1

        let mut b: LwwMap<String, i32> = LwwMap::new(2);
        b.insert("key".to_string(), 20);
        b.insert("key".to_string(), 30); // clock=2

        a.merge(&b);
        assert_eq!(a.get(&"key".to_string()), Some(&30)); // higher ts wins
    }

    // --- CrdtMergeable commutativity ---

    #[test]
    fn test_merge_commutativity() {
        let mut a = GCounter::new();
        a.increment_by(1, 5);

        let mut b = GCounter::new();
        b.increment_by(2, 3);

        let mut ab = a.clone();
        ab.merge(&b);

        let mut ba = b.clone();
        ba.merge(&a);

        assert_eq!(ab.value(), ba.value());
    }

    #[test]
    fn test_merge_idempotency() {
        let mut a = GCounter::new();
        a.increment_by(1, 5);

        let b = a.clone();
        a.merge(&b);
        a.merge(&b);
        a.merge(&b);

        assert_eq!(a.value(), 5); // unchanged
    }
}
