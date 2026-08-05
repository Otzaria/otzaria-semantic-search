use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// Cache statistics for the embedding cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub size: usize,
    pub capacity: usize,
}

#[derive(Debug, Clone)]
struct CachedEmbedding {
    vector: Vec<f32>,
    last_access: u64,
}

/// An LRU-like cache for embedding vectors of recently embedded texts.
#[derive(Debug)]
pub struct EmbeddingCache {
    entries: Mutex<HashMap<String, CachedEmbedding>>,
    max_entries: usize,
    hits: AtomicU64,
    misses: AtomicU64,
    access_clock: AtomicU64,
}

impl EmbeddingCache {
    /// Create a new EmbeddingCache with the specified capacity.
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Mutex::new(HashMap::with_capacity(max_entries)),
            max_entries,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            access_clock: AtomicU64::new(0),
        }
    }

    /// Retrieve a cloned vector from the cache if it exists.
    pub fn get(&self, text: &str) -> Option<Vec<f32>> {
        let mut entries = self.entries.lock().unwrap_or_else(|poisoned| {
            log::warn!("EmbeddingCache lock was poisoned; recovering cached state");
            poisoned.into_inner()
        });
        if let Some(entry) = entries.get_mut(text) {
            entry.last_access = self.access_clock.fetch_add(1, Ordering::Relaxed);
            self.hits.fetch_add(1, Ordering::Relaxed);
            Some(entry.vector.clone())
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Insert a new embedding vector into the cache.
    pub fn insert(&self, text: &str, vector: Vec<f32>) {
        if self.max_entries == 0 {
            return;
        }

        let mut entries = self.entries.lock().unwrap_or_else(|poisoned| {
            log::warn!("EmbeddingCache lock was poisoned; recovering cached state");
            poisoned.into_inner()
        });

        if entries.len() >= self.max_entries && !entries.contains_key(text) {
            if let Some((lru_key, _)) = entries.iter().min_by_key(|(_, entry)| entry.last_access) {
                let lru_key = lru_key.clone();
                entries.remove(&lru_key);
            }
        }

        entries.insert(
            text.to_string(),
            CachedEmbedding {
                vector,
                last_access: self.access_clock.fetch_add(1, Ordering::Relaxed),
            },
        );
    }

    /// Clear all entries from the cache (e.g. on model change).
    pub fn invalidate_all(&self) {
        self.entries
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
    }

    /// Retrieve cache statistics.
    pub fn stats(&self) -> EmbeddingCacheStats {
        let size = self
            .entries
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len();
        EmbeddingCacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            size,
            capacity: self.max_entries,
        }
    }
}

/// Compute a 64-bit FNV-1a hash of a text string.
pub fn compute_text_hash(text: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in text.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_insert_and_get() {
        let cache = EmbeddingCache::new(10);
        let vec = vec![0.1, 0.2, 0.3];

        cache.insert("hello world", vec.clone());
        assert_eq!(cache.get("hello world"), Some(vec));

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn test_capacity_eviction() {
        let cache = EmbeddingCache::new(2);

        cache.insert("one", vec![1.0]);
        cache.insert("two", vec![2.0]);
        cache.insert("three", vec![3.0]); // Should evict one

        assert_eq!(cache.get("one"), None);
        assert_eq!(cache.get("two"), Some(vec![2.0]));
        assert_eq!(cache.get("three"), Some(vec![3.0]));
    }

    #[test]
    fn test_invalidation() {
        let cache = EmbeddingCache::new(10);
        cache.insert("one", vec![1.0]);
        cache.insert("two", vec![2.0]);

        assert_eq!(cache.stats().size, 2);
        cache.invalidate_all();
        assert_eq!(cache.stats().size, 0);
    }

    #[test]
    fn test_concurrent_access() {
        let cache = std::sync::Arc::new(EmbeddingCache::new(1000));
        let mut handles = vec![];

        for i in 0..10 {
            let cache_clone = cache.clone();
            handles.push(thread::spawn(move || {
                for j in 0..100 {
                    let key = format!("{i}:{j}");
                    cache_clone.insert(&key, vec![j as f32]);
                    cache_clone.get(&key);
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let stats = cache.stats();
        assert_eq!(stats.size, 1000);
        assert_eq!(stats.hits, 1000);
    }

    #[test]
    fn reads_refresh_lru_order() {
        let cache = EmbeddingCache::new(2);
        cache.insert("one", vec![1.0]);
        cache.insert("two", vec![2.0]);
        assert!(cache.get("one").is_some());
        cache.insert("three", vec![3.0]);

        assert!(cache.get("one").is_some());
        assert!(cache.get("two").is_none());
        assert!(cache.get("three").is_some());
    }

    #[test]
    fn zero_capacity_never_stores_an_embedding() {
        let cache = EmbeddingCache::new(0);
        cache.insert("query", vec![1.0]);
        assert!(cache.get("query").is_none());
        assert_eq!(cache.stats().size, 0);
    }
}
