use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;
use std::time::Instant;
use serde::{Deserialize, Serialize};

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
    inserted: Instant,
}

/// An LRU-like cache for embedding vectors of recently embedded texts.
#[derive(Debug)]
pub struct EmbeddingCache {
    entries: RwLock<HashMap<u64, CachedEmbedding>>,
    max_entries: usize,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl EmbeddingCache {
    /// Create a new EmbeddingCache with the specified capacity.
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: RwLock::new(HashMap::with_capacity(max_entries)),
            max_entries,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Retrieve a cloned vector from the cache if it exists.
    pub fn get(&self, text_hash: u64) -> Option<Vec<f32>> {
        let read_guard = self.entries.read().unwrap();
        if let Some(entry) = read_guard.get(&text_hash) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            Some(entry.vector.clone())
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Insert a new embedding vector into the cache.
    pub fn insert(&self, text_hash: u64, vector: Vec<f32>) {
        let mut write_guard = self.entries.write().unwrap();
        
        if write_guard.len() >= self.max_entries {
            // Evict the oldest entry (approximation of LRU via oldest insertion time)
            if let Some((&oldest_key, _)) = write_guard
                .iter()
                .min_by_key(|(_, entry)| entry.inserted)
            {
                write_guard.remove(&oldest_key);
            }
        }

        write_guard.insert(
            text_hash,
            CachedEmbedding {
                vector,
                inserted: Instant::now(),
            },
        );
    }

    /// Clear all entries from the cache (e.g. on model change).
    pub fn invalidate_all(&self) {
        let mut write_guard = self.entries.write().unwrap();
        write_guard.clear();
    }

    /// Retrieve cache statistics.
    pub fn stats(&self) -> EmbeddingCacheStats {
        let size = self.entries.read().unwrap().len();
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
        let hash = compute_text_hash("hello world");
        let vec = vec![0.1, 0.2, 0.3];
        
        cache.insert(hash, vec.clone());
        assert_eq!(cache.get(hash), Some(vec));
        
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn test_capacity_eviction() {
        let cache = EmbeddingCache::new(2);
        
        cache.insert(1, vec![1.0]);
        thread::sleep(std::time::Duration::from_millis(5));
        cache.insert(2, vec![2.0]);
        thread::sleep(std::time::Duration::from_millis(5));
        cache.insert(3, vec![3.0]); // Should evict 1
        
        assert_eq!(cache.get(1), None);
        assert_eq!(cache.get(2), Some(vec![2.0]));
        assert_eq!(cache.get(3), Some(vec![3.0]));
    }

    #[test]
    fn test_invalidation() {
        let cache = EmbeddingCache::new(10);
        cache.insert(1, vec![1.0]);
        cache.insert(2, vec![2.0]);
        
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
                    cache_clone.insert(i * 100 + j, vec![j as f32]);
                    cache_clone.get(i * 100 + j);
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
}
