use crate::semantic::types::HybridSearchResult;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Cache statistics for the query cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub size: usize,
    pub generation: u64,
}

#[derive(Debug, Clone)]
struct CachedEntry {
    result: HybridSearchResult,
    generation: u64,
    inserted: Instant,
    last_access: u64,
}

/// A cache for search results, keyed by query parameters and using generation-based invalidation.
#[derive(Debug)]
pub struct QueryCache {
    entries: Mutex<HashMap<[u8; 32], CachedEntry>>,
    generation: AtomicU64,
    capacity: usize,
    ttl: Duration,
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
    access_clock: AtomicU64,
}

impl Default for QueryCache {
    fn default() -> Self {
        Self::new(256, Duration::from_secs(300))
    }
}

impl QueryCache {
    /// Create a new QueryCache with the specified capacity and TTL.
    pub fn new(capacity: usize, ttl: Duration) -> Self {
        Self {
            entries: Mutex::new(HashMap::with_capacity(capacity)),
            generation: AtomicU64::new(0),
            capacity,
            ttl,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            access_clock: AtomicU64::new(0),
        }
    }

    /// Retrieve a cached result if it is fresh and matches the current generation.
    pub fn get(&self, key: [u8; 32]) -> Option<HybridSearchResult> {
        let current_gen = self.generation.load(Ordering::Relaxed);
        let mut entries = self.entries.lock().unwrap_or_else(|poisoned| {
            log::warn!("QueryCache lock was poisoned; recovering cached state");
            poisoned.into_inner()
        });

        if let Some(entry) = entries.get_mut(&key) {
            if entry.generation == current_gen && entry.inserted.elapsed() <= self.ttl {
                entry.last_access = self.access_clock.fetch_add(1, Ordering::Relaxed);
                self.hits.fetch_add(1, Ordering::Relaxed);
                return Some(entry.result.clone());
            }
            // If invalid due to generation or TTL, we can remove it
            entries.remove(&key);
        }

        self.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Insert a result into the cache.
    pub fn insert(&self, key: [u8; 32], result: HybridSearchResult) {
        self.insert_with_capacity(key, result, self.capacity);
    }

    /// Insert with a request-specific upper bound, capped by the cache's capacity.
    pub fn insert_with_capacity(
        &self,
        key: [u8; 32],
        result: HybridSearchResult,
        requested_capacity: usize,
    ) {
        let capacity = requested_capacity.min(self.capacity);
        if capacity == 0 {
            return;
        }

        let current_gen = self.generation.load(Ordering::Relaxed);
        let mut entries = self.entries.lock().unwrap_or_else(|poisoned| {
            log::warn!("QueryCache lock was poisoned; recovering cached state");
            poisoned.into_inner()
        });

        while entries.len() >= capacity && !entries.contains_key(&key) {
            if let Some((&lru_key, _)) = entries.iter().min_by_key(|(_, entry)| entry.last_access) {
                entries.remove(&lru_key);
                self.evictions.fetch_add(1, Ordering::Relaxed);
            } else {
                break;
            }
        }

        let last_access = self.access_clock.fetch_add(1, Ordering::Relaxed);
        entries.insert(
            key,
            CachedEntry {
                result,
                generation: current_gen,
                inserted: Instant::now(),
                last_access,
            },
        );
    }

    /// Invalidate all entries by bumping the generation counter.
    pub fn invalidate(&self) {
        self.generation.fetch_add(1, Ordering::Relaxed);
    }

    /// Clear all entries and bump generation.
    pub fn clear(&self) {
        self.invalidate();
        self.entries
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
    }

    /// Compute a collision-resistant cache key from all query parameters.
    pub fn compute_key(
        query: &str,
        inputs_hash: [u8; 32],
        mode: &str,
        grouping: &str,
        limit: usize,
        offset: usize,
    ) -> [u8; 32] {
        let mut hasher = Sha256::new();

        let update_hash = |hasher: &mut Sha256, bytes: &[u8]| {
            hasher.update((bytes.len() as u64).to_le_bytes());
            hasher.update(bytes);
        };

        update_hash(&mut hasher, query.as_bytes());
        update_hash(&mut hasher, &inputs_hash);
        update_hash(&mut hasher, mode.as_bytes());
        update_hash(&mut hasher, grouping.as_bytes());
        update_hash(&mut hasher, &(limit as u64).to_le_bytes());
        update_hash(&mut hasher, &(offset as u64).to_le_bytes());

        hasher.finalize().into()
    }

    /// Retrieve cache statistics.
    pub fn stats(&self) -> QueryCacheStats {
        let size = self
            .entries
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len();
        QueryCacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            size,
            generation: self.generation.load(Ordering::Relaxed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic::types::SearchMode;
    use std::thread;

    fn result(total_count: u32) -> HybridSearchResult {
        HybridSearchResult {
            results: Vec::new(),
            total_count,
            group_count: None,
            search_mode: SearchMode::LexicalOnly,
            semantic_available: false,
            fallback_reason: None,
            latency_ms: 0,
            confidence: None,
            profile: None,
            telemetry: None,
        }
    }

    fn key(value: u8) -> [u8; 32] {
        [value; 32]
    }

    #[test]
    fn test_insert_and_get() {
        let cache = QueryCache::new(10, Duration::from_secs(60));
        let key = QueryCache::compute_key("query", key(123), "Hybrid", "None", 10, 0);

        cache.insert(key, result(1));
        assert_eq!(cache.get(key).map(|r| r.total_count), Some(1));

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn test_ttl_expiry() {
        let cache = QueryCache::new(10, Duration::from_millis(10));
        cache.insert(key(1), result(1));
        thread::sleep(Duration::from_millis(20));

        assert!(cache.get(key(1)).is_none());
    }

    #[test]
    fn test_generation_invalidation() {
        let cache = QueryCache::new(10, Duration::from_secs(60));
        cache.insert(key(1), result(1));

        assert_eq!(cache.get(key(1)).map(|r| r.total_count), Some(1));
        cache.invalidate();
        assert!(cache.get(key(1)).is_none()); // Old generation
    }

    #[test]
    fn test_capacity_eviction() {
        let cache = QueryCache::new(2, Duration::from_secs(60));

        cache.insert(key(1), result(1));
        thread::sleep(Duration::from_millis(5));
        cache.insert(key(2), result(2));
        thread::sleep(Duration::from_millis(5));
        cache.insert(key(3), result(3)); // Should evict 1

        assert!(cache.get(key(1)).is_none());
        assert_eq!(cache.get(key(2)).map(|r| r.total_count), Some(2));
        assert_eq!(cache.get(key(3)).map(|r| r.total_count), Some(3));
        assert_eq!(cache.stats().evictions, 1);
    }

    #[test]
    fn zero_capacity_never_stores_an_entry() {
        let cache = QueryCache::new(0, Duration::from_secs(60));
        cache.insert(key(1), result(1));
        assert!(cache.get(key(1)).is_none());
        assert_eq!(cache.stats().size, 0);
    }

    #[test]
    fn reads_refresh_lru_order() {
        let cache = QueryCache::new(2, Duration::from_secs(60));
        cache.insert(key(1), result(1));
        cache.insert(key(2), result(2));
        assert!(cache.get(key(1)).is_some());
        cache.insert(key(3), result(3));

        assert!(cache.get(key(1)).is_some());
        assert!(cache.get(key(2)).is_none());
        assert!(cache.get(key(3)).is_some());
    }
}
