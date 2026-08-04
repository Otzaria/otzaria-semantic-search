//! Measurement helpers for hybrid-search benchmarks.
//!
//! Provides query sets, timing helpers, percentile aggregation and serial
//! throughput estimates. The caller supplies the search closure and corpus.
//!
//! # Usage
//!
//! ```no_run
//! use otzaria_semantic_search::benchmark::{aggregate, measure};
//! use std::time::Duration;
//!
//! let (_, latency) = measure(|| 2 + 2);
//! let result = aggregate("example", 1_000, "short", vec![latency, Duration::from_micros(5)]);
//! println!("{}", result.summary());
//! ```

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Configuration for a benchmark run.
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Corpus sizes to test. Each creates a synthetic index of that many vectors.
    pub corpus_sizes: Vec<usize>,
    /// Query sets to run against each corpus.
    pub query_sets: Vec<QuerySet>,
    /// Number of iterations per (corpus, query) pair. Results are aggregated.
    pub iterations: usize,
    /// Warm-up iterations discarded before measurement.
    pub warmup: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            corpus_sizes: vec![100, 1_000, 10_000],
            query_sets: vec![
                QuerySet::short(),
                QuerySet::conceptual(),
                QuerySet::exact_reference(),
                QuerySet::phrase(),
            ],
            iterations: 50,
            warmup: 5,
        }
    }
}

/// A named set of queries of a particular type.
#[derive(Debug, Clone)]
pub struct QuerySet {
    /// Human-readable name for the set.
    pub name: String,
    /// The queries to run.
    pub queries: Vec<String>,
    /// Expected query classification.
    pub expected_type: String,
}

impl QuerySet {
    /// Short Hebrew queries (1-2 words).
    pub fn short() -> Self {
        Self {
            name: "short".to_string(),
            queries: vec![
                "תשובה".to_string(),
                "שבת קודש".to_string(),
                "ברכה".to_string(),
                "כשרות".to_string(),
                "תפילה".to_string(),
            ],
            expected_type: "Short".to_string(),
        }
    }

    /// Long conceptual queries (5+ words).
    pub fn conceptual() -> Self {
        Self {
            name: "conceptual".to_string(),
            queries: vec![
                "מה המשמעות של החיים ביקום לפי הקבלה".to_string(),
                "כיצד מתקיימת בחירה חופשית לפי הרמבם".to_string(),
                "מדוע נבחר אברהם אבינו מכל בני דורו".to_string(),
                "הקשר בין גלות וגאולה במשנת חכמי ספרד".to_string(),
            ],
            expected_type: "Conceptual".to_string(),
        }
    }

    /// Exact reference queries with numbers or specific format.
    pub fn exact_reference() -> Self {
        Self {
            name: "exact_reference".to_string(),
            queries: vec![
                "בראשית א:א".to_string(),
                "ברכות דף כ".to_string(),
                "שולחן ערוך אורח חיים סימן א".to_string(),
                "רמבם הלכות תשובה פרק ג".to_string(),
            ],
            expected_type: "ExactReference".to_string(),
        }
    }

    /// Quoted phrase queries.
    pub fn phrase() -> Self {
        Self {
            name: "phrase".to_string(),
            queries: vec![
                "\"בראשית ברא אלהים\"".to_string(),
                "\"ואהבת לרעך כמוך\"".to_string(),
                "\"שמע ישראל\"".to_string(),
                "\"לא תרצח\"".to_string(),
            ],
            expected_type: "ExactReference".to_string(),
        }
    }
}

/// Results from one benchmark configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Name identifying this benchmark.
    pub name: String,
    /// Number of vectors in the corpus.
    pub corpus_size: usize,
    /// Query set name.
    pub query_set: String,
    /// Number of iterations measured (excluding warmup).
    pub iterations: usize,
    /// Median latency.
    pub p50_us: u64,
    /// 95th percentile latency.
    pub p95_us: u64,
    /// 99th percentile latency.
    pub p99_us: u64,
    /// Minimum latency.
    pub min_us: u64,
    /// Maximum latency.
    pub max_us: u64,
    /// Mean latency.
    pub mean_us: u64,
    /// Queries per second (throughput).
    pub throughput_qps: f64,
    /// Total wall-clock time for all iterations.
    pub total_duration_ms: u64,
}

impl BenchmarkResult {
    /// Human-readable summary line.
    pub fn summary(&self) -> String {
        format!(
            "{name:<30} corpus={corpus:>6}  p50={p50:>6}μs  p95={p95:>6}μs  \
             p99={p99:>6}μs  qps={qps:>8.1}",
            name = self.name,
            corpus = self.corpus_size,
            p50 = self.p50_us,
            p95 = self.p95_us,
            p99 = self.p99_us,
            qps = self.throughput_qps,
        )
    }
}

/// Compute percentile from a sorted slice of durations.
fn percentile(sorted: &[Duration], pct: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let idx = ((sorted.len() as f64 - 1.0) * pct / 100.0).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Aggregate raw latency measurements into a [`BenchmarkResult`].
pub fn aggregate(
    name: &str,
    corpus_size: usize,
    query_set: &str,
    mut latencies: Vec<Duration>,
) -> BenchmarkResult {
    latencies.sort_unstable();
    let iterations = latencies.len();
    let total: Duration = latencies.iter().sum();

    BenchmarkResult {
        name: name.to_string(),
        corpus_size,
        query_set: query_set.to_string(),
        iterations,
        p50_us: percentile(&latencies, 50.0).as_micros() as u64,
        p95_us: percentile(&latencies, 95.0).as_micros() as u64,
        p99_us: percentile(&latencies, 99.0).as_micros() as u64,
        min_us: latencies.first().map_or(0, |d| d.as_micros() as u64),
        max_us: latencies.last().map_or(0, |d| d.as_micros() as u64),
        mean_us: if iterations > 0 {
            (total.as_micros() as u64) / iterations as u64
        } else {
            0
        },
        throughput_qps: if total.as_secs_f64() > 0.0 {
            iterations as f64 / total.as_secs_f64()
        } else {
            0.0
        },
        total_duration_ms: total.as_millis() as u64,
    }
}

/// Measures the latency of a single closure invocation.
pub fn measure<F, R>(f: F) -> (R, Duration)
where
    F: FnOnce() -> R,
{
    let start = Instant::now();
    let result = f();
    (result, start.elapsed())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_of_empty_is_zero() {
        assert_eq!(percentile(&[], 50.0), Duration::ZERO);
    }

    #[test]
    fn percentile_of_single_element() {
        let sorted = vec![Duration::from_micros(100)];
        assert_eq!(percentile(&sorted, 50.0), Duration::from_micros(100));
        assert_eq!(percentile(&sorted, 99.0), Duration::from_micros(100));
    }

    #[test]
    fn percentile_of_sorted_sequence() {
        let sorted: Vec<Duration> = (1..=100)
            .map(|i| Duration::from_micros(i * 10))
            .collect();
        assert_eq!(percentile(&sorted, 50.0), Duration::from_micros(500));
        assert_eq!(percentile(&sorted, 95.0), Duration::from_micros(950));
        assert_eq!(percentile(&sorted, 99.0), Duration::from_micros(990));
    }

    #[test]
    fn aggregate_produces_valid_results() {
        let latencies: Vec<Duration> = (1..=100)
            .map(|i| Duration::from_micros(i * 10))
            .collect();
        let result = aggregate("test_bench", 1000, "short", latencies);

        assert_eq!(result.name, "test_bench");
        assert_eq!(result.corpus_size, 1000);
        assert_eq!(result.iterations, 100);
        assert!(result.p50_us > 0);
        assert!(result.p95_us > result.p50_us);
        assert!(result.throughput_qps > 0.0);
    }

    #[test]
    fn measure_captures_elapsed_time() {
        let (result, duration) = measure(|| {
            std::thread::sleep(Duration::from_millis(10));
            42
        });
        assert_eq!(result, 42);
        assert!(duration >= Duration::from_millis(5));
    }

    #[test]
    fn query_set_constructors_produce_non_empty_sets() {
        assert!(!QuerySet::short().queries.is_empty());
        assert!(!QuerySet::conceptual().queries.is_empty());
        assert!(!QuerySet::exact_reference().queries.is_empty());
        assert!(!QuerySet::phrase().queries.is_empty());
    }

    #[test]
    fn benchmark_result_summary_is_readable() {
        let result = BenchmarkResult {
            name: "test".to_string(),
            corpus_size: 1000,
            query_set: "short".to_string(),
            iterations: 100,
            p50_us: 500,
            p95_us: 950,
            p99_us: 990,
            min_us: 100,
            max_us: 1000,
            mean_us: 550,
            throughput_qps: 1818.18,
            total_duration_ms: 55,
        };
        let summary = result.summary();
        assert!(summary.contains("test"));
        assert!(summary.contains("1000"));
        assert!(summary.contains("500"));
    }
}
