//! Reproducible measurements of the semantic retrieval hot path.
//!
//! ```text
//! cargo bench
//! cargo bench -- --vectors 1000000 --dim 256
//! ```
//!
//! Measures [`VectorStore`] directly, so it needs no embedding backend and runs
//! in a default build.
//!
//! Deliberately not Criterion: this exists so a performance claim about this
//! crate can be re-run and checked, and pulling a benchmarking framework (and its
//! dependency tree) into a crate destined for a mobile build is a poor trade for
//! statistical polish. `harness = false`, so this is a plain program.
//!
//! # What it is for
//!
//! Whether brute-force scanning can serve the real library at all, and what an
//! ANN backend (roadmap P4) has to beat. The full-library figure is *extrapolated*
//! from the measured rate — the linear cost of a brute-force scan is exactly what
//! makes the extrapolation sound, and exactly why it stops holding the moment a
//! real index is introduced. Run it before and after any change to
//! `VectorStore::search`.
//!
//! Numbers are single-threaded and machine-specific. Compare runs on one machine;
//! never quote them as absolutes.

use otzaria_semantic_search::semantic::store::{VectorStore, VectorStoreConfig};
use otzaria_semantic_search::semantic::types::VectorMetadata;
use std::time::{Duration, Instant};

/// Lines in the library snapshot the roadmap sizes against.
const LIBRARY_LINE_COUNT: u64 = 6_058_210;

struct Options {
    vectors: usize,
    dim: usize,
    top_k: usize,
    queries: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            // Large enough for the per-query cost to dominate setup, small enough
            // to stay under a gigabyte of vectors (200k × 1024 × 4B ≈ 780 MiB).
            vectors: 200_000,
            dim: 1024,
            top_k: 40,
            queries: 20,
        }
    }
}

fn parse_options() -> Options {
    let mut options = Options::default();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut index = 0;

    while index < args.len() {
        let (flag, value) = (args[index].as_str(), args.get(index + 1));
        let parsed = |name: &str| -> usize {
            value
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(|| panic!("{name} needs a positive integer, got {value:?}"))
        };
        match flag {
            "--vectors" => options.vectors = parsed("--vectors"),
            "--dim" => options.dim = parsed("--dim"),
            "--top-k" => options.top_k = parsed("--top-k"),
            "--queries" => options.queries = parsed("--queries"),
            // cargo bench passes its own flags; ignore what is not ours.
            _ => {
                index += 1;
                continue;
            }
        }
        index += 2;
    }
    options
}

/// A deterministic pseudo-random unit-ish vector.
///
/// Deterministic on purpose: two runs must measure the same data. The values only
/// have to be spread out — a store full of identical vectors would make every
/// candidate tie and exercise the tie-break instead of the scan.
fn vector(seed: usize, dim: usize) -> Vec<f32> {
    let mut state = (seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..dim)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let scaled = (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32;
            scaled / 8_388_608.0 - 1.0
        })
        .collect()
}

fn metadata(index: usize) -> VectorMetadata {
    VectorMetadata {
        semantic_id: format!("{index:010x}"),
        source_book_key: format!("otzaria/book{}.txt", index % 5_000),
        source_doc_key: String::new(),
        line_id: index as u64,
        section_id: (index / 20) as u64,
        line_hash: index as u64,
        chunk_hash: String::new(),
        content_hash: 1,
        reference: String::new(),
        segment: (index % 500) as u64,
        is_pdf: false,
        title: String::new(),
        facets: vec!["/מקרא/תורה".to_string(), "/era/תנך".to_string()],
    }
}

fn median(mut durations: Vec<Duration>) -> Duration {
    durations.sort_unstable();
    durations[durations.len() / 2]
}

fn main() {
    let options = parse_options();
    let Options {
        vectors,
        dim,
        top_k,
        queries,
    } = options;

    println!("otzaria-semantic-search — vector search benchmark");
    println!("  vectors: {vectors}\n  dim:     {dim}\n  top_k:   {top_k}\n  queries: {queries}");
    println!(
        "  vector payload: ~{:.0} MiB",
        (vectors * dim * 4) as f64 / (1024.0 * 1024.0)
    );

    let dir = std::env::temp_dir().join("otzaria_bench_vector_search");
    let store = VectorStore::open_or_create(VectorStoreConfig {
        db_path: dir.clone(),
        embedding_dim: dim as u32,
        collection_name: "bench".to_string(),
    })
    .expect("store");

    let started = Instant::now();
    let batch_size = 10_000;
    for start in (0..vectors).step_by(batch_size) {
        let batch: Vec<(VectorMetadata, Vec<f32>)> = (start..(start + batch_size).min(vectors))
            .map(|index| (metadata(index), vector(index, dim)))
            .collect();
        store.insert_batch(&batch).expect("insert");
    }
    let insert_elapsed = started.elapsed();
    println!(
        "\ninsert: {insert_elapsed:.2?} for {vectors} vectors ({:.0} vectors/s)",
        vectors as f64 / insert_elapsed.as_secs_f64()
    );
    assert_eq!(store.vector_count(), vectors);

    let query = vector(usize::MAX / 3, dim);
    // Warm up: first touch of a freshly filled map pays page faults that have
    // nothing to do with the scan.
    for _ in 0..3 {
        store.search(&query, top_k, None).expect("search");
    }

    let mut unfiltered = Vec::with_capacity(queries);
    for round in 0..queries {
        let query = vector(round + 1, dim);
        let started = Instant::now();
        let hits = store.search(&query, top_k, None).expect("search");
        unfiltered.push(started.elapsed());
        assert_eq!(hits.len(), top_k.min(vectors));
    }

    let per_query = median(unfiltered);
    let dims_per_second = (vectors * dim) as f64 / per_query.as_secs_f64();
    println!("search (no filter): {per_query:.2?} median per query");
    println!("  → {:.2} G dot-product dims/s", dims_per_second / 1e9);

    // The question this benchmark exists to answer.
    let full_library = Duration::from_secs_f64(
        (LIBRARY_LINE_COUNT as f64 * dim as f64) / dims_per_second.max(f64::MIN_POSITIVE),
    );
    println!(
        "  → extrapolated to the {LIBRARY_LINE_COUNT}-line library: {full_library:.2?} per query"
    );
    println!("    (linear, single-threaded, brute force — the figure an ANN backend must beat)");

    let filters = otzaria_semantic_search::semantic::types::SearchFilters {
        facets: Some(vec!["/era/תנך".to_string()]),
        ..Default::default()
    };
    let mut filtered = Vec::with_capacity(queries);
    for round in 0..queries {
        let query = vector(round + 1, dim);
        let started = Instant::now();
        store
            .search(&query, top_k, Some(&filters))
            .expect("filtered search");
        filtered.push(started.elapsed());
    }
    println!(
        "search (facet filter matching everything): {:.2?} median per query",
        median(filtered)
    );

    let _ = std::fs::remove_dir_all(&dir);
}
