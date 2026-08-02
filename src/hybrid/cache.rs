use serde::{Deserialize, Serialize};
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
    result_json: String,
    generation: u64,
    inserted: Instant,
}

/// A cache for search results, keyed by query parameters and using generation-based invalidation.
#[derive(Debug)]
pub struct QueryCache {
    entries: Mutex<HashMap<u64, CachedEntry>>,
    generation: AtomicU64,
    capacity: usize,
    ttl: Duration,
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
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
        }
    }

    /// Retrieve a cached serialized result if it exists, is fresh, and matches the current generation.
    pub fn get(&self, key: u64) -> Option<String> {
        let current_gen = self.generation.load(Ordering::Relaxed);
        let mut entries = self.entries.lock().unwrap();

        if let Some(entry) = entries.get(&key) {
            if entry.generation == current_gen && entry.inserted.elapsed() <= self.ttl {
                self.hits.fetch_add(1, Ordering::Relaxed);
                return Some(entry.result_json.clone());
            }
            // If invalid due to generation or TTL, we can remove it
            entries.remove(&key);
        }

        self.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Insert a new serialized result into the cache.
    pub fn insert(&self, key: u64, result_json: String) {
        let current_gen = self.generation.load(Ordering::Relaxed);
        let mut entries = self.entries.lock().unwrap();

        if entries.len() >= self.capacity {
            // Simple eviction: remove the oldest entry
            if let Some((&oldest_key, _)) = entries.iter().min_by_key(|(_, entry)| entry.inserted) {
                entries.remove(&oldest_key);
                self.evictions.fetch_add(1, Ordering::Relaxed);
            }
        }

        entries.insert(
            key,
            CachedEntry {
                result_json,
                generation: current_gen,
                inserted: Instant::now(),
            },
        );
    }

    /// Invalidate all entries by bumping the generation counter.
    pub fn invalidate(&self) {
        self.generation.fetch_add(1, Ordering::Relaxed);
    }

    /// Compute a cache key based on query parameters using FNV-1a.
    pub fn compute_key(
        query: &str,
        filters_hash: u64,
        mode: &str,
        grouping: &str,
        limit: usize,
        offset: usize,
    ) -> u64 {
        let mut hash: u64 = 0xcbf29ce484222325;

        let update_hash = |h: &mut u64, bytes: &[u8]| {
            for &byte in bytes {
                *h ^= byte as u64;
                *h = h.wrapping_mul(0x100000001b3);
            }
        };

        update_hash(&mut hash, query.as_bytes());
        update_hash(&mut hash, &filters_hash.to_le_bytes());
        update_hash(&mut hash, mode.as_bytes());
        update_hash(&mut hash, grouping.as_bytes());
        update_hash(&mut hash, &limit.to_le_bytes());
        update_hash(&mut hash, &offset.to_le_bytes());

        hash
    }

    /// Retrieve cache statistics.
    pub fn stats(&self) -> QueryCacheStats {
        let size = self.entries.lock().unwrap().len();
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
    use std::thread;

    #[test]
    fn test_insert_and_get() {
        let cache = QueryCache::new(10, Duration::from_secs(60));
        let key = QueryCache::compute_key("query", 123, "Hybrid", "None", 10, 0);

        cache.insert(key, "{}".to_string());
        assert_eq!(cache.get(key), Some("{}".to_string()));

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn test_ttl_expiry() {
        let cache = QueryCache::new(10, Duration::from_millis(10));
        cache.insert(1, "{}".to_string());
        thread::sleep(Duration::from_millis(20));

        assert_eq!(cache.get(1), None);
    }

    #[test]
    fn test_generation_invalidation() {
        let cache = QueryCache::new(10, Duration::from_secs(60));
        cache.insert(1, "{}".to_string());

        assert_eq!(cache.get(1), Some("{}".to_string()));
        cache.invalidate();
        assert_eq!(cache.get(1), None); // Old generation
    }

    #[test]
    fn test_capacity_eviction() {
        let cache = QueryCache::new(2, Duration::from_secs(60));

        cache.insert(1, "1".to_string());
        thread::sleep(Duration::from_millis(5));
        cache.insert(2, "2".to_string());
        thread::sleep(Duration::from_millis(5));
        cache.insert(3, "3".to_string()); // Should evict 1

        assert_eq!(cache.get(1), None);
        assert_eq!(cache.get(2), Some("2".to_string()));
        assert_eq!(cache.get(3), Some("3".to_string()));
        assert_eq!(cache.stats().evictions, 1);
    }
}
