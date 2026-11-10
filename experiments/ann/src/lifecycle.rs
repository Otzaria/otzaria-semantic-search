//! Lifecycle checks use the real OXV installer and compactor. The mutable ANN
//! graph is a disposable derivative; no production generation contains it.
use crate::data::{self, Config, Filter, key_id};
use crate::measure::{Backend, Hits};
use crate::{Result, native_checks, path_str};
use otzaria_semantic_search::cancellation::CancellationToken;
use otzaria_semantic_search::distribution::package::PackageKind;
use otzaria_semantic_search::errors::VectorStoreError;
use otzaria_semantic_search::semantic::oxv::scan::ScanRequest;
use otzaria_semantic_search::semantic::segment_set::{CompactionPolicy, SegmentSet, compact};
use serde_json::{Value, json};
use std::cell::Cell;
use std::collections::HashSet;
use std::path::Path;
use std::time::Instant;

fn exact(set: &SegmentSet, config: &Config, filter: Filter) -> Result<Vec<Hits>> {
    let books = filter.books();
    let request = ScanRequest {
        top_k: config.top_k,
        books: books.as_ref(),
        threads: 1,
        window_penalty: 0.0,
    };
    (0..config.queries.min(20))
        .map(|id| {
            let hits = set.scan(&config.query(id), &request, &CancellationToken::new())?;
            Ok(Hits {
                keys: hits.iter().map(|hit| key_id(hit.key)).collect(),
                scores: hits.iter().map(|hit| hit.score).collect(),
            })
        })
        .collect()
}

fn recall(expected: &[u64], actual: &[u64]) -> Option<f64> {
    if expected.is_empty() {
        return None;
    }
    let actual: HashSet<_> = actual.iter().copied().collect();
    Some(expected.iter().filter(|key| actual.contains(key)).count() as f64 / expected.len() as f64)
}

pub fn run(dir: &Path) -> Result<Value> {
    let config = Config::read(dir)?;
    let set_path = dir.join("lifecycle-set");
    let old = SegmentSet::open(&set_path)?;
    if old.info().library_version != 1 {
        return Err(
            "lifecycle is destructive only to its own fixture and must run once after build".into(),
        );
    }
    let codec = old.codec().clone();
    let filters = [
        Filter::All,
        Filter::FivePercent,
        Filter::OnePercent,
        Filter::Empty,
    ];
    let mut before = Vec::new();
    for filter in filters {
        before.push(exact(&old, &config, filter)?);
    }
    let ann = config.ann(256)?;
    ann.load(path_str(&dir.join("index.usearch"))?)?;
    ann.change_expansion_search(256);
    let deleted: Vec<_> = (1..=config.vectors as u64)
        .filter(|key| key % 50 == 0)
        .collect();
    let added_count = (config.vectors * 6 / 100).max(1);
    let added: Vec<_> =
        (config.vectors as u64 + 1..=config.vectors as u64 + added_count as u64).collect();
    let deleted_set: HashSet<_> = deleted.iter().copied().collect();
    let is_live = |key: u64| {
        key > 0 && key <= config.vectors as u64 + added_count as u64 && !deleted_set.contains(&key)
    };
    let delta_path = dir.join("lifecycle-delta.oxv");
    let delta = data::write_segment(
        &config,
        &delta_path,
        &codec,
        &added,
        &deleted,
        PackageKind::Delta,
        None,
    )?;
    let started = Instant::now();
    data::install(&config, &set_path, &delta_path, &delta, &codec)?;
    let oxv_install_us = started.elapsed().as_secs_f64() * 1e6;
    let current = SegmentSet::open(&set_path)?;
    let live_count = config.vectors - deleted.len() + added_count;
    if current.info().slots_live as usize != live_count || current.generation() <= old.generation()
    {
        return Err("OXV delta did not produce the expected live generation".into());
    }
    let started = Instant::now();
    for &key in &deleted {
        if ann.remove(key)? != 1 {
            return Err("ANN removal missed a key".into());
        }
    }
    ann.reserve_capacity_and_threads(config.vectors + added_count, 1)?;
    let mut decoded = vec![0f32; config.dim];
    let mut encoded = vec![0u8; codec.bytes_per_vector()];
    for &key in &added {
        let encoded_info = codec.encode(&config.vector(key), &mut encoded);
        codec.decode(&encoded, encoded_info.scale, &mut decoded);
        ann.add(key, &decoded)?;
    }
    let ann_mutation_us = started.elapsed().as_secs_f64() * 1e6;
    native_checks::verify_contents(&ann, &current)
        .map_err(|error| format!("ANN after delta: {error}"))?;

    let mut after_delta = Vec::new();
    let mut recall_by_filter = Vec::new();
    for (index, filter) in filters.into_iter().enumerate() {
        if exact(&old, &config, filter)? != before[index] {
            return Err("a published delta changed an open OXV generation's answers".into());
        }
        let expected = exact(&current, &config, filter)?;
        let mut values = Vec::new();
        for (id, expected) in expected.iter().enumerate() {
            let found = if matches!(filter, Filter::All) {
                ann.search(&config.query(id), config.top_k)?
            } else {
                ann.filtered_search(&config.query(id), config.top_k, |key| {
                    filter.admits(&config, key)
                })?
            };
            if found
                .keys
                .iter()
                .any(|&key| !is_live(key) || !filter.admits(&config, key))
            {
                return Err("ANN returned a tombstoned or filtered-out key after delta".into());
            }
            if let Some(value) = recall(&expected.keys, &found.keys) {
                values.push(value);
            }
        }
        recall_by_filter.push(json!({ "filter": format!("{filter:?}"), "queries": values.len(), "mean_recall_vs_oxv": if values.is_empty() { None } else { Some(values.iter().sum::<f64>() / values.len() as f64) } }));
        after_delta.push(expected);
    }

    // Exact distances and strictly better neighbors must survive compaction.
    // USearch may reorder ties or select different keys tied at the top-k cutoff;
    // full key-to-vector validation above and below guards the underlying content.
    let native_exact_before: Vec<_> = (0..config.queries.min(20))
        .map(|id| ann.exact_search(&config.query(id), config.top_k))
        .collect::<std::result::Result<_, _>>()?;
    let started = Instant::now();
    ann.compact()?;
    let ann_compact_us = started.elapsed().as_secs_f64() * 1e6;
    native_checks::verify_contents(&ann, &current)
        .map_err(|error| format!("ANN after compaction: {error}"))?;
    for (id, expected) in native_exact_before.iter().enumerate() {
        let query = config.query(id);
        let actual = ann.exact_search(&query, config.top_k)?;
        native_checks::verify_exact_parity(
            &ann,
            &query,
            expected,
            &actual,
            config.top_k.min(live_count),
            &is_live,
        )
        .map_err(|error| format!("ANN compaction query {id}: {error}"))?;
    }
    let started = Instant::now();
    let compacted = compact(
        &set_path,
        &CompactionPolicy {
            force: true,
            refresh_hints: false,
            ..CompactionPolicy::default()
        },
        None,
        &CancellationToken::new(),
    )?;
    let oxv_compact_us = started.elapsed().as_secs_f64() * 1e6;
    let fresh = SegmentSet::open(&set_path)?;
    if !compacted.compacted || fresh.generation() <= current.generation() {
        return Err("compaction did not publish a new generation".into());
    }
    for (index, filter) in filters.into_iter().enumerate() {
        if exact(&fresh, &config, filter)? != after_delta[index]
            || exact(&current, &config, filter)? != after_delta[index]
            || exact(&old, &config, filter)? != before[index]
        {
            return Err("OXV compaction changed exact scores/order or a retained snapshot".into());
        }
    }
    let compacted_path = dir.join("lifecycle-compacted.usearch");
    ann.save(path_str(&compacted_path)?)?;
    for viewed in [false, true] {
        let restored = config.ann(256)?;
        if viewed {
            restored.view(path_str(&compacted_path)?)?;
        } else {
            restored.load(path_str(&compacted_path)?)?;
        }
        native_checks::verify_contents(&restored, &current)
            .map_err(|error| format!("ANN restore (view={viewed}): {error}"))?;
        for key in 1..=config.vectors as u64 + added_count as u64 {
            if restored.contains(key) != is_live(key) {
                return Err("ANN key mapping changed on compact/save/restore".into());
            }
        }
        for (id, expected) in native_exact_before.iter().enumerate() {
            let query = config.query(id);
            let actual = restored.exact_search(&query, config.top_k)?;
            native_checks::verify_exact_parity(
                &restored,
                &query,
                expected,
                &actual,
                config.top_k.min(live_count),
                &is_live,
            )
            .map_err(|error| format!("ANN restore (view={viewed}) query {id}: {error}"))?;
        }
    }

    let cancelled = CancellationToken::new();
    cancelled.cancel();
    let request = ScanRequest {
        top_k: config.top_k,
        books: None,
        threads: 1,
        window_penalty: 0.0,
    };
    let oxv_pre_cancelled = matches!(
        fresh.scan(&config.query(0), &request, &cancelled),
        Err(VectorStoreError::Cancelled)
    );
    let backend = Backend::Ann(ann);
    let ann_pre_cancelled = backend
        .search(&config, &config.query(0), Filter::All, &request, &cancelled)
        .err()
        .is_some_and(|error| {
            matches!(
                error.downcast_ref::<VectorStoreError>(),
                Some(VectorStoreError::Cancelled)
            )
        });
    if !oxv_pre_cancelled || !ann_pre_cancelled {
        return Err("pre-cancelled search returned results".into());
    }
    let Backend::Ann(ann) = backend else {
        unreachable!()
    };
    let token = CancellationToken::new();
    let predicate_calls = Cell::new(0usize);
    let cancelled_at = Cell::new(None);
    // Cancellation is injected after native traversal starts, without sleeps or
    // racing a fast query. Returning false is NOT native cancellation: traversal
    // continues; only its completed result can be discarded by our wrapper.
    let ignored_result = ann.filtered_search(&config.query(0), config.top_k, |_key| {
        predicate_calls.set(predicate_calls.get() + 1);
        if predicate_calls.get() == 1 {
            token.cancel();
            cancelled_at.set(Some(Instant::now()));
        }
        !token.is_cancelled()
    })?;
    let cancellation_return_us = cancelled_at
        .get()
        .map(|start| start.elapsed().as_secs_f64() * 1e6);
    if !token.is_cancelled() {
        return Err("in-flight cancellation probe did not enter native search".into());
    }
    drop(ignored_result);

    // A bad derivative must never make the exact source unreadable. Demonstrate
    // the fallback without silently using it in the ANN performance measurements.
    let corrupt_path = dir.join("lifecycle-invalid.usearch");
    std::fs::write(&corrupt_path, b"invalid experiment-only ANN cache")?;
    let refused = config.ann(128)?.load(path_str(&corrupt_path)?).is_err();
    let fallback_hits = exact(&fresh, &config, Filter::All)?;
    if !refused || fallback_hits != after_delta[0] {
        return Err("a refused ANN derivative did not leave exact fallback available".into());
    }

    Ok(json!({
        "status": "passed", "vectors_before": config.vectors, "deleted": deleted.len(), "added": added_count,
        "live_after": live_count, "oxv_generation_before": old.generation(), "oxv_generation_delta": current.generation(),
        "oxv_generation_compacted": fresh.generation(), "oxv_install_us": oxv_install_us,
        "ann_mutation_us": ann_mutation_us, "oxv_compact_us": oxv_compact_us, "ann_compact_us": ann_compact_us,
        "oxv_bytes_after_delta": current.info().bytes_on_disk, "oxv_bytes_after_compact": fresh.info().bytes_on_disk,
        "ann_bytes_after_compact": std::fs::metadata(compacted_path)?.len(),
        "recall_after_delta": recall_by_filter,
        "exact_oxv_compaction_score_and_order_parity": true, "retained_oxv_snapshot_parity": true,
        "ann_exact_compaction_and_restore_parity": true, "ann_all_key_membership_after_load_and_view": true,
        "ann_exact_tie_policy": "equal_distance_permutations_and_verified_cutoff_substitutions",
        "ann_all_live_vectors_match_oxv": true,
        "ann_vector_validation_stages": ["after_delta", "after_compact", "load", "view"],
        "pre_cancelled_oxv": oxv_pre_cancelled, "pre_cancelled_ann_wrapper": ann_pre_cancelled,
        "native_ann_abort_supported": false,
        "ann_cancellation_predicate_calls_after_request": predicate_calls.get().saturating_sub(1),
        "ann_cancellation_request_to_native_return_us": cancellation_return_us,
        "cancelled_result_discarded": true, "corrupt_ann_refused_exact_fallback_available": true,
        "uncovered_production_contracts": ["window/alias penalties before top-k", "foreign links and moved-text resolver", "crash-safe ANN generation publication", "cancellable native ANN search/build/compaction", "real corpus and multi-platform CPU coverage"],
    }))
}
