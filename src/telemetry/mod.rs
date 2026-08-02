use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// A per-query record of search execution metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchTelemetry {
    pub query_type: String,
    pub search_mode: String,
    pub fusion_strategy: String,
    pub alpha: f32,
    pub lexical_candidates: u32,
    pub semantic_candidates: u32,
    pub fused_candidates: u32,
    pub cache_hit: bool,
    pub latency_ms: u64,
    pub embedding_latency_ms: Option<u64>,
    pub fusion_latency_ms: u64,
    pub confidence: Option<f32>,
    pub profile: String,
}

/// A snapshot of aggregated telemetry metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetrySnapshot {
    pub total_searches: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub embedding_calls: u64,
    pub avg_latency_ms: f64,
    pub cache_hit_rate: f64,
    pub strategy_distribution: HashMap<String, u64>,
}

/// Thread-safe collector for aggregating search telemetry.
pub struct TelemetryCollector {
    total_searches: AtomicU64,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
    embedding_calls: AtomicU64,
    total_latency_us: AtomicU64,
    strategy_counts: Mutex<HashMap<String, u64>>,
}

impl Default for TelemetryCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl TelemetryCollector {
    pub fn new() -> Self {
        Self {
            total_searches: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            embedding_calls: AtomicU64::new(0),
            total_latency_us: AtomicU64::new(0),
            strategy_counts: Mutex::new(HashMap::new()),
        }
    }

    /// Records metrics from a single search execution.
    pub fn record_search(&self, telemetry: &SearchTelemetry) {
        self.total_searches.fetch_add(1, Ordering::Relaxed);

        if telemetry.cache_hit {
            self.cache_hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.cache_misses.fetch_add(1, Ordering::Relaxed);
            if telemetry.embedding_latency_ms.is_some() {
                self.embedding_calls.fetch_add(1, Ordering::Relaxed);
            }
        }

        // Convert latency to microseconds to retain precision in total
        let latency_us = telemetry.latency_ms.saturating_mul(1000);
        self.total_latency_us
            .fetch_add(latency_us, Ordering::Relaxed);

        // Update strategy counts
        if let Ok(mut counts) = self.strategy_counts.lock() {
            *counts.entry(telemetry.fusion_strategy.clone()).or_insert(0) += 1;
        }
    }

    /// Reads all counters atomically and computes a snapshot.
    pub fn snapshot(&self) -> TelemetrySnapshot {
        let total = self.total_searches.load(Ordering::Relaxed);
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let misses = self.cache_misses.load(Ordering::Relaxed);
        let embeddings = self.embedding_calls.load(Ordering::Relaxed);
        let total_us = self.total_latency_us.load(Ordering::Relaxed);

        let avg_latency_ms = if total > 0 {
            (total_us as f64 / total as f64) / 1000.0
        } else {
            0.0
        };

        let total_cache_ops = hits + misses;
        let cache_hit_rate = if total_cache_ops > 0 {
            hits as f64 / total_cache_ops as f64
        } else {
            0.0
        };

        let strategy_distribution = self
            .strategy_counts
            .lock()
            .map(|guard| guard.clone())
            .unwrap_or_default();

        TelemetrySnapshot {
            total_searches: total,
            cache_hits: hits,
            cache_misses: misses,
            embedding_calls: embeddings,
            avg_latency_ms,
            cache_hit_rate,
            strategy_distribution,
        }
    }

    /// Zeros out all counters and clears the strategy distribution.
    pub fn reset(&self) {
        self.total_searches.store(0, Ordering::Relaxed);
        self.cache_hits.store(0, Ordering::Relaxed);
        self.cache_misses.store(0, Ordering::Relaxed);
        self.embedding_calls.store(0, Ordering::Relaxed);
        self.total_latency_us.store(0, Ordering::Relaxed);

        if let Ok(mut counts) = self.strategy_counts.lock() {
            counts.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    fn make_telemetry(cache_hit: bool, latency_ms: u64, strategy: &str) -> SearchTelemetry {
        SearchTelemetry {
            query_type: "Test".into(),
            search_mode: "Hybrid".into(),
            fusion_strategy: strategy.into(),
            alpha: 0.5,
            lexical_candidates: 10,
            semantic_candidates: 10,
            fused_candidates: 10,
            cache_hit,
            latency_ms,
            embedding_latency_ms: if cache_hit {
                None
            } else {
                Some(latency_ms / 2)
            },
            fusion_latency_ms: 1,
            confidence: None,
            profile: "Balanced".into(),
        }
    }

    #[test]
    fn test_record_and_snapshot() {
        let collector = TelemetryCollector::new();

        collector.record_search(&make_telemetry(false, 100, "RRF"));
        collector.record_search(&make_telemetry(true, 10, "Weighted"));
        collector.record_search(&make_telemetry(true, 10, "Weighted"));

        let snapshot = collector.snapshot();
        assert_eq!(snapshot.total_searches, 3);
        assert_eq!(snapshot.cache_hits, 2);
        assert_eq!(snapshot.cache_misses, 1);
        assert_eq!(snapshot.embedding_calls, 1);
        assert_eq!(snapshot.avg_latency_ms, 40.0); // (100 + 10 + 10) / 3

        // Cache hit rate = 2 / 3
        assert!((snapshot.cache_hit_rate - 0.666).abs() < 0.01);

        assert_eq!(snapshot.strategy_distribution.get("RRF"), Some(&1));
        assert_eq!(snapshot.strategy_distribution.get("Weighted"), Some(&2));
    }

    #[test]
    fn test_reset() {
        let collector = TelemetryCollector::new();
        collector.record_search(&make_telemetry(false, 100, "RRF"));

        collector.reset();
        let snapshot = collector.snapshot();

        assert_eq!(snapshot.total_searches, 0);
        assert_eq!(snapshot.cache_hits, 0);
        assert_eq!(snapshot.cache_misses, 0);
        assert_eq!(snapshot.embedding_calls, 0);
        assert_eq!(snapshot.avg_latency_ms, 0.0);
        assert_eq!(snapshot.cache_hit_rate, 0.0);
        assert!(snapshot.strategy_distribution.is_empty());
    }

    #[test]
    fn test_concurrent_recording() {
        use std::sync::Arc;

        let collector = Arc::new(TelemetryCollector::new());
        let mut handles = vec![];

        for _ in 0..10 {
            let collector = Arc::clone(&collector);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    collector.record_search(&make_telemetry(false, 5, "Adaptive"));
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let snapshot = collector.snapshot();
        assert_eq!(snapshot.total_searches, 1000);
        assert_eq!(snapshot.cache_misses, 1000);
        assert_eq!(snapshot.embedding_calls, 1000);
        assert_eq!(snapshot.strategy_distribution.get("Adaptive"), Some(&1000));
    }
}
