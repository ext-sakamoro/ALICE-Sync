//! ALICE-Sync × ALICE-Cache Bridge
//!
//! Entity state prefetching via Markov oracle.
//! When entity E is accessed, the cache oracle predicts which entities
//! will be accessed next and prefetches their states.

use crate::{Entity, WorldHash};
use alice_cache::AliceCache;

/// Cached world state for fast entity lookup and predictive prefetching.
///
/// Wraps `AliceCache` with entity-aware access patterns so the Markov oracle
/// learns entity access sequences (e.g., entity 1 → entity 2 → entity 5)
/// and prefetches predicted-next entities before they're needed.
pub struct SyncCache {
    /// Entity state cache (entity_id → serialized state).
    cache: AliceCache<u32, CachedEntity>,
    /// Last accessed entity (for oracle training).
    last_accessed: Option<u32>,
    /// Cache hits counter.
    hits: u64,
    /// Cache misses counter.
    misses: u64,
}

/// Minimal cached entity snapshot.
#[derive(Clone, Debug)]
pub struct CachedEntity {
    /// Entity ID.
    pub id: u32,
    /// Entity kind.
    pub kind: u16,
    /// Position as fixed-point triple (from Vec3Fixed).
    pub position: [i32; 3],
    /// World hash at cache time (staleness detection).
    pub snapshot_hash: u64,
}

impl SyncCache {
    /// Create a new sync cache with the given capacity (max entities).
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: AliceCache::new(capacity),
            last_accessed: None,
            hits: 0,
            misses: 0,
        }
    }

    /// Cache an entity's state.
    ///
    /// Also trains the Markov oracle: if entity A was accessed just before
    /// entity B, the oracle learns A → B transition for future prefetching.
    pub fn put(&mut self, entity: &Entity, world_hash: WorldHash) {
        let id = entity.id;
        let cached = CachedEntity {
            id,
            kind: entity.kind,
            position: [
                entity.position.x.0,
                entity.position.y.0,
                entity.position.z.0,
            ],
            snapshot_hash: world_hash.0,
        };
        self.cache.put(id, cached);

        // Train oracle: last_accessed → id
        if let Some(prev) = self.last_accessed {
            // Access prev then id to train the Markov oracle
            let _ = self.cache.get(&prev);
            let _ = self.cache.get(&id);
        }
        self.last_accessed = Some(id);
    }

    /// Get a cached entity state.
    ///
    /// Returns `None` if the entity is not in cache or the snapshot is stale.
    pub fn get(&mut self, entity_id: u32, current_hash: WorldHash) -> Option<CachedEntity> {
        match self.cache.get(&entity_id) {
            Some(cached) if cached.snapshot_hash == current_hash.0 => {
                self.hits += 1;
                self.last_accessed = Some(entity_id);
                Some(cached)
            }
            _ => {
                self.misses += 1;
                None
            }
        }
    }

    /// Check if the oracle predicts entity `next` should be prefetched
    /// after accessing entity `current`.
    pub fn should_prefetch(&self, current: u32, next: u32) -> bool {
        self.cache.should_prefetch(&current, &next)
    }

    /// Cache hit rate.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Total cache hits.
    pub fn hits(&self) -> u64 {
        self.hits
    }
    /// Total cache misses.
    pub fn misses(&self) -> u64 {
        self.misses
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Entity, EntityProps, Vec3Fixed};

    fn make_entity(id: u32) -> Entity {
        Entity {
            id,
            kind: 0,
            position: Vec3Fixed::from_f32(id as f32, 0.0, 0.0),
            properties: EntityProps::EMPTY,
        }
    }

    #[test]
    fn test_sync_cache_put_get() {
        let mut cache = SyncCache::new(100);
        let hash = WorldHash(12345);

        let e1 = make_entity(1);
        cache.put(&e1, hash);

        // Hit: same hash
        let cached = cache.get(1, hash);
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().id, 1);
        assert_eq!(cache.hits(), 1);

        // Miss: stale hash
        let stale = cache.get(1, WorldHash(99999));
        assert!(stale.is_none());
        assert_eq!(cache.misses(), 1);

        // Miss: unknown entity
        let unknown = cache.get(999, hash);
        assert!(unknown.is_none());
    }

    #[test]
    fn test_oracle_training() {
        let mut cache = SyncCache::new(100);
        let hash = WorldHash(1);

        // Train pattern: entity 1 → entity 2 repeatedly
        for _ in 0..100 {
            cache.put(&make_entity(1), hash);
            cache.put(&make_entity(2), hash);
        }

        // Oracle should predict 2 after 1
        assert!(cache.should_prefetch(1, 2));
    }
}
