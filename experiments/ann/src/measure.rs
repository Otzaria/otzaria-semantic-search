use crate::data::{Config, Filter, key_id};
use crate::{Result, path_str};
use otzaria_semantic_search::cancellation::CancellationToken;
use otzaria_semantic_search::errors::VectorStoreError;
use otzaria_semantic_search::semantic::oxv::reader::Segment;
use otzaria_semantic_search::semantic::oxv::scan::{ScanRequest, scan_segments};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::Path;
use std::time::Instant;
use usearch::Index;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Hits {
    pub keys: Vec<u64>,
    pub scores: Vec<f32>,
}

pub enum Backend {
    Oxv(Segment),
    Ann(Index),
}

impl Backend {
    pub fn open(config: &Config, dir: &Path, backend: &str, expansion: usize) -> Result<Self> {
        match backend {
            "raw" => Ok(Self::Oxv(Segment::open(&dir.join("raw.oxv"))?)),
            "decoded" => Ok(Self::Oxv(Segment::open(&dir.join("decoded.oxv"))?)),
            "oxv" => Ok(Self::Oxv(Segment::open(&dir.join("encoded.oxv"))?)),
            "usearch-load" | "usearch-view" => {
                let index = config.ann(expansion)?;
                let path = dir.join("index.usearch");
                if backend == "usearch-load" {
                    index.load(path_str(&path)?)?;
                } else {
                    index.view(path_str(&path)?)?;
                }
                index.change_expansion_search(expansion);
                index.reserve_capacity_and_threads(index.capacity(), 1)?;
                Ok(Self::Ann(index))
            }
            _ => Err("backend must be raw, decoded, oxv, usearch-load, or usearch-view".into()),
        }
    }

    pub fn search(
        &self,
        config: &Config,
        query: &[f32],
        filter: Filter,
        request: &ScanRequest<'_>,
        cancel: &CancellationToken,
    ) -> Result<Hits> {
        if cancel.is_cancelled() {
            return Err(VectorStoreError::Cancelled.into());
        }
        let hits = match self {
            Self::Oxv(segment) => {
                let found = scan_segments(&[segment], query, request, cancel)?;
                Hits {
                    keys: found.iter().map(|hit| key_id(hit.key)).collect(),
                    scores: found.iter().map(|hit| hit.score).collect(),
                }
            }
            Self::Ann(index) => {
                let top_k = request.top_k.min(index.size());
                let found = if matches!(filter, Filter::All) {
                    index.search(query, top_k)?
                } else {
                    // A predicate during graph traversal, not a top-k result postfilter.
                    index.filtered_search(query, top_k, |key| filter.admits(config, key))?
                };
                Hits {
                    keys: found.keys,
                    scores: found
                        .distances
                        .into_iter()
                        .map(|distance| 1.0 - distance)
                        .collect(),
                }
            }
        };
        // USearch has no abortable search API here: cancellation discards the
        // result only after the native call returns. This is a production blocker.
        if cancel.is_cancelled() {
            return Err(VectorStoreError::Cancelled.into());
        }
        Ok(hits)
    }
}

pub fn measure(
    dir: &Path,
    backend_name: &str,
    filter_name: &str,
    threads: usize,
    expansion: usize,
    repeats: usize,
) -> Result<Value> {
    let config = Config::read(dir)?;
    let filter = Filter::parse(filter_name)?;
    let queries: Vec<_> = (0..config.queries).map(|id| config.query(id)).collect();
    let books = filter.books();
    let request = ScanRequest {
        top_k: config.top_k,
        books: books.as_ref(),
        threads,
        window_penalty: 0.0,
    };
    let eligible_vectors: usize = (0..crate::data::BOOKS)
        .filter(|book| filter.admits_book(*book))
        .map(|book| {
            config
                .per_book()
                .min(config.vectors.saturating_sub(book * config.per_book()))
        })
        .sum();
    let cancel = CancellationToken::new();
    let before = memory();
    let started = Instant::now();
    let backend = Backend::open(&config, dir, backend_name, expansion)?;
    let open_us = started.elapsed().as_secs_f64() * 1e6;
    let after_open = memory();
    let started = Instant::now();
    let first = backend.search(&config, &queries[0], filter, &request, &cancel)?;
    let first_query_us = started.elapsed().as_secs_f64() * 1e6;
    let after_first = memory();
    for query in queries.iter().take(3) {
        std::hint::black_box(backend.search(&config, query, filter, &request, &cancel)?);
    }
    let mut samples = Vec::with_capacity(config.queries * repeats);
    let mut hits = Vec::with_capacity(config.queries);
    for round in 0..repeats {
        for (query_id, query) in queries.iter().enumerate() {
            let started = Instant::now();
            let found = backend.search(&config, query, filter, &request, &cancel)?;
            samples.push(started.elapsed().as_secs_f64() * 1e6);
            // Check membership outside the timed interval. Short result lists are
            // kept in recall's denominator; they are never padded with exact hits.
            if found
                .keys
                .iter()
                .any(|&key| key == 0 || key > config.vectors as u64 || !filter.admits(&config, key))
            {
                return Err("a backend returned an invalid or filtered-out key".into());
            }
            if found
                .keys
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len()
                != found.keys.len()
            {
                return Err("a backend returned a duplicate key".into());
            }
            if round == 0 {
                hits.push(found);
            } else if hits[query_id] != found {
                return Err("repeated queries produced different keys, scores, or order".into());
            }
        }
    }
    let ann_memory = match &backend {
        Backend::Ann(index) => Some(index.memory_usage()),
        _ => None,
    };
    Ok(json!({
        "schema": 1, "backend": backend_name, "filter": filter_name, "threads": threads,
        "eligible_vectors": eligible_vectors,
        "oxv_effective_threads": if matches!(&backend, Backend::Oxv(_)) { Some(threads.max(1).min(eligible_vectors.div_ceil(4096).max(1))) } else { None },
        "expansion_search": expansion, "repeats": repeats, "config": config,
        "cold_definition": "fresh process and freshly opened handle; standalone invocation does not evict file caches; see runner cache_advice when present",
        "open_us": open_us, "first_query_us": first_query_us,
        "first_query_keys": first.keys, "warm_us": summary(&samples),
        "samples_us": samples, "queries": hits,
        "memory_before_open": before, "memory_after_open": after_open,
        "memory_after_first": after_first, "memory_after_warm": memory(),
        "ann_memory_usage_bytes": ann_memory,
    }))
}

pub fn summary(values: &[f64]) -> Value {
    assert!(!values.is_empty());
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let percentile = |fraction: f64| {
        sorted[((sorted.len() as f64 * fraction).ceil() as usize)
            .saturating_sub(1)
            .min(sorted.len() - 1)]
    };
    json!({ "n": sorted.len(), "min": sorted[0], "p50": percentile(0.50), "p95": percentile(0.95), "p99": percentile(0.99), "max": sorted[sorted.len() - 1] })
}

pub fn memory() -> Value {
    // Linux measurements only. Null elsewhere is preferable to a fabricated
    // cross-platform RAM figure. VmHWM includes allocations earlier in this process.
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
        return Value::Null;
    };
    let field = |name: &str| -> Option<u64> {
        status.lines().find_map(|line| {
            line.strip_prefix(name)
                .and_then(|value| value.split_whitespace().next()?.parse::<u64>().ok())
                .map(|kb| kb * 1024)
        })
    };
    let stat = std::fs::read_to_string("/proc/self/stat").ok();
    // The parenthesized executable name can contain spaces/parentheses. Fields
    // after its final ')' start at state (field 3); faults are fields 10 and 12.
    let fields: Vec<_> = stat
        .as_deref()
        .and_then(|s| s.rsplit_once(')'))
        .map(|(_, values)| values.split_whitespace().collect())
        .unwrap_or_default();
    let faults = |index: usize| {
        fields
            .get(index)
            .and_then(|value| value.parse::<u64>().ok())
    };
    json!({ "rss_bytes": field("VmRSS:"), "peak_rss_bytes": field("VmHWM:"), "anon_bytes": field("RssAnon:"), "file_bytes": field("RssFile:"), "virtual_bytes": field("VmSize:"), "minor_faults": faults(7), "major_faults": faults(9) })
}

pub fn thread_probe(samples: usize) -> Value {
    let mut timings = Vec::new();
    for threads in [1, 2, 4, 8] {
        let mut values = Vec::with_capacity(samples);
        for _ in 0..samples {
            let started = Instant::now();
            std::thread::scope(|scope| {
                for _ in 1..threads {
                    scope.spawn(|| std::hint::black_box(()));
                }
                std::hint::black_box(());
            });
            values.push(started.elapsed().as_secs_f64() * 1e6);
        }
        timings.push(json!({ "threads_including_caller": threads, "empty_scope_us": summary(&values), "samples_us": values }));
    }
    json!({ "kind": "empty-scope launch/join proxy, not an attributed OXV CPU profile", "timings": timings })
}
