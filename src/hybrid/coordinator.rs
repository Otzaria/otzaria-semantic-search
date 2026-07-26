//! Hybrid search coordinator.
//!
//! Merges results from the lexical (Tantivy/BM25) and semantic (embedding)
//! search paths into one ranked set, with score normalization, dynamic weighting
//! and post-fusion grouping.
//!
//! # Modes
//!
//! | requested | semantic path healthy | result |
//! |---|---|---|
//! | [`SearchMode::LexicalOnly`] | not consulted | lexical results only |
//! | [`SearchMode::Hybrid`] | yes | fused results |
//! | [`SearchMode::Hybrid`] | no | lexical results, mode reported as `LexicalOnly` |
//! | [`SearchMode::SemanticOnly`] | yes | semantic results only |
//! | [`SearchMode::SemanticOnly`] | no | empty, with a `fallback_reason` |
//!
//! A `SemanticOnly` request is honoured rather than quietly answered with BM25:
//! the caller excluded the lexical path, so handing back lexical hits labelled as
//! a semantic search would misrepresent them. Every degradation is visible
//! through [`HybridSearchResult::search_mode`] and
//! [`HybridSearchResult::fallback_reason`], which is what lets the caller decide
//! whether to retry in another mode.

use crate::errors::SemanticSearchError;
use crate::hybrid::fusion::{normalize_bm25_scores, normalize_semantic_scores};
use crate::hybrid::grouping::group_results;
use crate::hybrid::ranking::{analyze_query, compute_alpha, BonusConfig};
use crate::semantic::engine::SemanticEngine;
use crate::semantic::types::{
    BookForIndexing, FusedCandidate, GroupingMode, HybridMergedSibling, HybridResultItem,
    HybridSearchResult, IndexDiff, LexicalCandidate, ResultSource, SearchFilters, SearchMode,
    SemanticCandidate, SemanticStatus,
};
use std::collections::HashMap;
use std::sync::RwLock;

/// Saturation constant for BM25 normalization (`score / (k + score)`).
///
/// A heuristic starting point, not a measured one: at `k = 1` the curve
/// saturates so fast that most real BM25 scores land near 1.0 and lose their
/// ordering information after normalization. Calibrating it — or replacing
/// weighted fusion with RRF — is roadmap P5.
const BM25_SATURATION_K: f32 = 1.0;

/// Upper bound on semantic candidates fetched for one query.
///
/// The store allocates a heap proportional to `top_k`, so an unvalidated
/// `limit`/`offset` from the caller would size an allocation from user input.
/// Hitting the cap is logged rather than silently truncating.
const MAX_SEMANTIC_CANDIDATES: usize = 10_000;

/// Configuration for hybrid search execution.
#[derive(Debug, Clone)]
pub struct HybridSearchParams {
    pub limit: usize,
    pub offset: usize,
    pub grouping: Option<GroupingMode>,
    pub filters: Option<SearchFilters>,
    /// Forces a retrieval mode. `None` means [`SearchMode::Hybrid`].
    pub force_mode: Option<SearchMode>,
}

impl Default for HybridSearchParams {
    fn default() -> Self {
        Self {
            limit: 20,
            offset: 0,
            grouping: None,
            filters: None,
            force_mode: None,
        }
    }
}

/// Main hybrid search coordinator.
pub struct HybridCoordinator {
    semantic_engine: RwLock<Option<SemanticEngine>>,
    bonus_config: BonusConfig,
}

impl HybridCoordinator {
    /// Create a new coordinator. Passing `None` disables the semantic path.
    pub fn new(semantic_engine: Option<SemanticEngine>) -> Self {
        Self {
            semantic_engine: RwLock::new(semantic_engine),
            bonus_config: BonusConfig::default(),
        }
    }

    /// Primary search entry point. Coordinates BM25 and semantic candidates.
    pub fn search(
        &self,
        query: &str,
        lexical_candidates: Vec<LexicalCandidate>,
        params: &HybridSearchParams,
    ) -> Result<HybridSearchResult, SemanticSearchError> {
        let start_time = std::time::Instant::now();
        let requested = params.force_mode.unwrap_or(SearchMode::Hybrid);

        // Recover rather than propagate a poisoned lock: a panic in one query
        // must not disable the semantic path for the rest of the session.
        let semantic_guard = self
            .semantic_engine
            .read()
            .unwrap_or_else(|e| e.into_inner());

        let semantic = if requested == SearchMode::LexicalOnly {
            // Not consulted — this is the caller's choice, not a degradation.
            SemanticOutcome::skipped()
        } else {
            match semantic_guard.as_ref() {
                None => SemanticOutcome::failed("no semantic engine is configured".to_string()),
                Some(engine) => {
                    match engine.search(query, self.semantic_top_k(params), params.filters.as_ref())
                    {
                        Ok(candidates) => SemanticOutcome::ok(candidates),
                        Err(e) => {
                            log::warn!(
                                "Semantic search path failed: {e}. Serving the lexical results."
                            );
                            SemanticOutcome::failed(e.to_string())
                        }
                    }
                }
            }
        };

        let mode = match requested {
            SearchMode::LexicalOnly => SearchMode::LexicalOnly,
            // Honoured whether or not the semantic path worked; when it did not,
            // the result set is empty and `fallback_reason` says why.
            SearchMode::SemanticOnly => SearchMode::SemanticOnly,
            // Degrade to lexical rather than failing the whole query.
            SearchMode::Hybrid if semantic.healthy => SearchMode::Hybrid,
            SearchMode::Hybrid => SearchMode::LexicalOnly,
        };

        // In semantic-only mode the lexical candidates the caller supplied are
        // deliberately discarded.
        let lexical_candidates = if mode == SearchMode::SemanticOnly {
            Vec::new()
        } else {
            lexical_candidates
        };

        // Weighting follows the mode that actually ran, so a single-source score
        // is not scaled down by the missing side's weight.
        let alpha = match mode {
            SearchMode::LexicalOnly => 1.0,
            SearchMode::SemanticOnly => 0.0,
            SearchMode::Hybrid => compute_alpha(&analyze_query(query)),
        };

        let fused = self.fuse_candidates(lexical_candidates, semantic.candidates, alpha, mode);

        let (results, total_count, group_count) = match params.grouping {
            Some(grouping_mode) => {
                let grouped = group_results(fused, grouping_mode);
                let group_count = grouped.len() as u32;
                let total: u32 = grouped.iter().map(|g| g.group_count).sum();

                let results = grouped
                    .into_iter()
                    .skip(params.offset)
                    .take(params.limit)
                    .map(|group| {
                        let merged = group
                            .siblings
                            .into_iter()
                            .map(|s| HybridMergedSibling {
                                title: s.title,
                                reference: s.reference,
                                id: s.line_id,
                                segment: s.segment,
                                is_pdf: s.is_pdf,
                                file_path: s.file_path,
                            })
                            .collect();
                        into_result_item(group.representative, group.group_count, merged)
                    })
                    .collect();

                (results, total, Some(group_count))
            }
            None => {
                let total = fused.len() as u32;
                let results = fused
                    .into_iter()
                    .skip(params.offset)
                    .take(params.limit)
                    .map(|candidate| into_result_item(candidate, 1, Vec::new()))
                    .collect();

                (results, total, None)
            }
        };

        Ok(HybridSearchResult {
            results,
            total_count,
            group_count,
            search_mode: mode,
            semantic_available: semantic.healthy,
            fallback_reason: semantic.failure,
            latency_ms: start_time.elapsed().as_millis() as u64,
        })
    }

    /// How many semantic candidates to fetch for one page of results.
    ///
    /// Over-fetches relative to `limit` because fusion, grouping and dedup all
    /// discard candidates, so the page must be filled from a wider window.
    fn semantic_top_k(&self, params: &HybridSearchParams) -> usize {
        let requested = params
            .offset
            .saturating_add(params.limit.saturating_mul(2))
            .max(1);

        if requested > MAX_SEMANTIC_CANDIDATES {
            log::warn!(
                "Semantic candidate window capped at {MAX_SEMANTIC_CANDIDATES} \
                 (requested {requested} for limit={} offset={})",
                params.limit,
                params.offset
            );
            return MAX_SEMANTIC_CANDIDATES;
        }
        requested
    }

    /// Fuse lexical and semantic candidates into one ranked list.
    ///
    /// Candidates are merged on `line_id`, the global Tantivy document id — see
    /// [`FusedCandidate`] for the invariant that requires.
    fn fuse_candidates(
        &self,
        lexical: Vec<LexicalCandidate>,
        semantic: Vec<SemanticCandidate>,
        alpha: f32,
        mode: SearchMode,
    ) -> Vec<FusedCandidate> {
        let bm25_scores: Vec<f32> = lexical.iter().map(|l| l.bm25_score).collect();
        let norm_bm25 = normalize_bm25_scores(&bm25_scores, BM25_SATURATION_K);

        let sem_scores: Vec<f32> = semantic.iter().map(|s| s.similarity_score).collect();
        let norm_sem = normalize_semantic_scores(&sem_scores);

        let mut fused_map: HashMap<u64, FusedCandidate> =
            HashMap::with_capacity(lexical.len() + semantic.len());

        for (candidate, &normalized) in lexical.into_iter().zip(norm_bm25.iter()) {
            let item = FusedCandidate {
                title: candidate.title,
                reference: candidate.reference,
                text: candidate.text,
                line_id: candidate.line_id,
                section_id: candidate.section_id,
                line_hash: candidate.line_hash,
                segment: candidate.segment,
                is_pdf: candidate.is_pdf,
                file_path: candidate.file_path,
                needs_hydration: false,
                source: ResultSource::Lexical,
                raw_bm25_score: Some(candidate.bm25_score),
                normalized_bm25: Some(normalized),
                raw_semantic_score: None,
                normalized_semantic: None,
                fused_score: alpha * normalized,
                lexical_weight: alpha,
                semantic_weight: 1.0 - alpha,
            };
            fused_map.insert(item.line_id, item);
        }

        for (candidate, &normalized) in semantic.into_iter().zip(norm_sem.iter()) {
            let line_id = candidate.metadata.line_id;
            let contribution = (1.0 - alpha) * normalized;

            match fused_map.get_mut(&line_id) {
                // Found by both engines: keep the lexical text and record both
                // scores. Provenance must survive fusion.
                Some(existing) => {
                    existing.source = ResultSource::Both;
                    existing.raw_semantic_score = Some(candidate.similarity_score);
                    existing.normalized_semantic = Some(normalized);
                    existing.fused_score += contribution;
                    if mode == SearchMode::Hybrid {
                        existing.fused_score += self.bonus_config.agreement_bonus;
                    }
                }
                // Semantic-only: the vector store holds metadata but no line
                // body, so the text has to be hydrated from Tantivy by id.
                None => {
                    let metadata = candidate.metadata;
                    fused_map.insert(
                        line_id,
                        FusedCandidate {
                            title: metadata.title,
                            reference: metadata.reference,
                            text: String::new(),
                            line_id,
                            section_id: metadata.section_id,
                            line_hash: metadata.line_hash,
                            segment: metadata.segment,
                            is_pdf: metadata.is_pdf,
                            file_path: metadata.source_book_key,
                            needs_hydration: true,
                            source: ResultSource::Semantic,
                            raw_bm25_score: None,
                            normalized_bm25: None,
                            raw_semantic_score: Some(candidate.similarity_score),
                            normalized_semantic: Some(normalized),
                            fused_score: contribution,
                            lexical_weight: alpha,
                            semantic_weight: 1.0 - alpha,
                        },
                    );
                }
            }
        }

        let mut results: Vec<FusedCandidate> = fused_map.into_values().collect();
        // Ties break on line_id so pagination is stable across calls; `HashMap`
        // iteration order is not.
        results.sort_by(|a, b| {
            b.fused_score
                .total_cmp(&a.fused_score)
                .then_with(|| a.line_id.cmp(&b.line_id))
        });
        results
    }

    /// Compare Tantivy's per-book content hashes against the semantic index.
    ///
    /// `None` when no semantic engine is configured — there is nothing to index.
    pub fn semantic_index_diff(&self, tantivy_books: &HashMap<String, u64>) -> Option<IndexDiff> {
        let guard = self
            .semantic_engine
            .read()
            .unwrap_or_else(|e| e.into_inner());
        guard
            .as_ref()
            .map(|engine| engine.diff_against_tantivy(tantivy_books))
    }

    /// Discard the semantic index and start over for the current configuration.
    ///
    /// The recovery path out of [`SemanticSearchError::IncompatibleIndex`]:
    /// without it a status reporting `needs_full_reindex` would be a dead end.
    /// Returns the number of vectors discarded, or `None` if there is no engine.
    pub fn reset_semantic_index(&self) -> Result<Option<u32>, SemanticSearchError> {
        let mut guard = self
            .semantic_engine
            .write()
            .unwrap_or_else(|e| e.into_inner());
        match guard.as_mut() {
            Some(engine) => engine.reset_index().map(Some),
            None => Ok(None),
        }
    }

    /// Index books into the semantic index, replacing anything held for them.
    ///
    /// Returns the number of chunks written, or `None` if there is no engine.
    /// Takes the write lock for the duration, so queries run against the
    /// pre-existing index until it returns.
    pub fn index_books(
        &self,
        books: &[BookForIndexing],
    ) -> Result<Option<u32>, SemanticSearchError> {
        let mut guard = self
            .semantic_engine
            .write()
            .unwrap_or_else(|e| e.into_inner());
        match guard.as_mut() {
            Some(engine) => engine.index_books(books).map(Some),
            None => Ok(None),
        }
    }

    /// Retrieve semantic engine status.
    pub fn status(&self) -> SemanticStatus {
        let guard = self
            .semantic_engine
            .read()
            .unwrap_or_else(|e| e.into_inner());

        match guard.as_ref() {
            Some(engine) => engine.status(),
            None => SemanticStatus {
                available: false,
                model_loaded: false,
                indexed_book_count: 0,
                vector_count: 0,
                model_id: "none".to_string(),
                embedding_dim: 0,
                embedding_backend: None,
                vector_backend: "none".to_string(),
                vectors_persisted: false,
                needs_full_reindex: None,
                last_error: Some("Semantic engine disabled".to_string()),
            },
        }
    }
}

/// Outcome of consulting the semantic path for one query.
struct SemanticOutcome {
    candidates: Vec<SemanticCandidate>,
    /// Whether the semantic path ran and returned successfully. Finding nothing
    /// still counts as healthy.
    healthy: bool,
    /// Why it did not run, when it was expected to.
    failure: Option<String>,
}

impl SemanticOutcome {
    fn ok(candidates: Vec<SemanticCandidate>) -> Self {
        Self {
            candidates,
            healthy: true,
            failure: None,
        }
    }

    fn failed(reason: String) -> Self {
        Self {
            candidates: Vec::new(),
            healthy: false,
            failure: Some(reason),
        }
    }

    /// The caller asked for lexical-only, so nothing was expected of the
    /// semantic path and there is nothing to report.
    fn skipped() -> Self {
        Self {
            candidates: Vec::new(),
            healthy: false,
            failure: None,
        }
    }
}

/// Convert a fused candidate into the frontend-facing result item.
fn into_result_item(
    candidate: FusedCandidate,
    merged_count: u32,
    merged: Vec<HybridMergedSibling>,
) -> HybridResultItem {
    HybridResultItem {
        title: candidate.title.clone(),
        reference: candidate.reference.clone(),
        text: candidate.text.clone(),
        id: candidate.line_id,
        segment: candidate.segment,
        is_pdf: candidate.is_pdf,
        file_path: candidate.file_path.clone(),
        merged_count,
        merged,
        lexical_score: candidate.raw_bm25_score,
        semantic_score: candidate.raw_semantic_score,
        fused_score: candidate.fused_score,
        needs_hydration: candidate.needs_hydration,
        source: candidate.source,
        provenance: Some(candidate),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic::embedding::mock;
    use crate::semantic::engine::SemanticConfig;
    use crate::semantic::store::VectorStoreConfig;
    use crate::semantic::types::{BookForIndexing, BookLine};
    use std::path::PathBuf;

    const LINE_ONE: &str = "בראשית ברא אלהים את השמים ואת הארץ";
    const LINE_TWO: &str = "והארץ היתה תהו ובהו וחשך על פני תהום";
    const LINE_THREE: &str = "ויאמר אלהים יהי אור ויהי אור מאיר";

    struct TempDir(PathBuf);
    impl TempDir {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "otzaria_coordinator_test_{name}_{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            let _ = std::fs::create_dir_all(&path);
            Self(path)
        }
        fn path(&self) -> &std::path::Path {
            &self.0
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn mock_book() -> BookForIndexing {
        BookForIndexing {
            source_book_key: "otzaria/tanach/genesis.txt".to_string(),
            title: "בראשית".to_string(),
            content_hash: 987654,
            is_pdf: false,
            topics: "/מקרא/תורה".to_string(),
            extra_facets: vec!["/author/משה רבנו".to_string(), "/era/תנך".to_string()],
            lines: vec![
                BookLine {
                    line_id: 1,
                    section_id: 100,
                    text: LINE_ONE.to_string(),
                    line_hash: 11111,
                    reference: "בראשית א:א".to_string(),
                    segment: 1,
                },
                BookLine {
                    line_id: 2,
                    section_id: 100,
                    text: LINE_TWO.to_string(),
                    line_hash: 22222,
                    reference: "בראשית א:ב".to_string(),
                    segment: 2,
                },
                BookLine {
                    line_id: 3,
                    section_id: 101,
                    text: LINE_THREE.to_string(),
                    line_hash: 33333,
                    reference: "בראשית א:ג".to_string(),
                    segment: 3,
                },
            ],
        }
    }

    /// A coordinator over an indexed 3-line book.
    fn indexed_coordinator(dir: &TempDir) -> HybridCoordinator {
        let model_path = dir.path().join("model.gguf");
        mock::write_stub_gguf(&model_path, 3).unwrap();
        let root = dir.path().join("semantic");

        let mut engine = SemanticEngine::open(SemanticConfig {
            root_dir: root.clone(),
            model_path,
            embedding_dim: 64,
            store: VectorStoreConfig {
                db_path: root.join("vectors"),
                embedding_dim: 64,
                collection_name: "chunks".to_string(),
            },
            ..Default::default()
        })
        .unwrap();
        engine.index_book(&mock_book()).unwrap();

        HybridCoordinator::new(Some(engine))
    }

    /// A coordinator whose engine exists but has no model loaded, so every
    /// semantic query fails.
    fn broken_coordinator(dir: &TempDir) -> HybridCoordinator {
        let root = dir.path().join("semantic");
        let engine = SemanticEngine::open(SemanticConfig {
            root_dir: root.clone(),
            // Deliberately absent: load_model will fail.
            model_path: dir.path().join("absent.gguf"),
            embedding_dim: 64,
            store: VectorStoreConfig {
                db_path: root.join("vectors"),
                embedding_dim: 64,
                collection_name: "chunks".to_string(),
            },
            ..Default::default()
        })
        .unwrap();
        HybridCoordinator::new(Some(engine))
    }

    fn lexical(line_id: u64, text: &str, bm25: f32) -> LexicalCandidate {
        LexicalCandidate {
            title: "בראשית".to_string(),
            reference: format!("בראשית א:{line_id}"),
            text: text.to_string(),
            line_id,
            section_id: 100,
            line_hash: line_id * 11111,
            segment: line_id,
            is_pdf: false,
            file_path: "otzaria/tanach/genesis.txt".to_string(),
            bm25_score: bm25,
        }
    }

    // ── mode contract ──

    #[test]
    fn hybrid_mode_merges_both_sources_and_keeps_provenance() {
        let dir = TempDir::new("hybrid");
        let coordinator = indexed_coordinator(&dir);

        let result = coordinator
            .search(
                LINE_ONE,
                vec![lexical(1, LINE_ONE, 15.5)],
                &HybridSearchParams::default(),
            )
            .unwrap();

        assert_eq!(result.search_mode, SearchMode::Hybrid);
        assert!(result.semantic_available);
        assert!(result.fallback_reason.is_none());

        let both = result
            .results
            .iter()
            .find(|r| r.id == 1)
            .expect("line 1 was found by both engines");
        assert_eq!(both.source, ResultSource::Both);
        assert!(both.lexical_score.is_some());
        assert!(both.semantic_score.is_some());
        assert!(!both.needs_hydration, "the lexical path supplied the text");
        assert_eq!(both.text, LINE_ONE);

        // The semantic-only hits are present too, flagged for hydration.
        let semantic_only: Vec<_> = result
            .results
            .iter()
            .filter(|r| r.source == ResultSource::Semantic)
            .collect();
        assert!(!semantic_only.is_empty());
        for item in semantic_only {
            assert!(item.needs_hydration);
            assert!(item.text.is_empty());
        }
    }

    /// The bug: only `LexicalOnly` was special-cased, so a `SemanticOnly`
    /// request ran as Hybrid and returned lexical results.
    #[test]
    fn semantic_only_mode_returns_no_lexical_results() {
        let dir = TempDir::new("semantic_only");
        let coordinator = indexed_coordinator(&dir);

        // A lexical candidate for a line that is NOT in the semantic index, so
        // its presence in the output can only come from the lexical path.
        let result = coordinator
            .search(
                LINE_ONE,
                vec![lexical(999, "שורה שרק המנוע הלקסיקלי מכיר", 42.0)],
                &HybridSearchParams {
                    force_mode: Some(SearchMode::SemanticOnly),
                    ..Default::default()
                },
            )
            .unwrap();

        assert_eq!(result.search_mode, SearchMode::SemanticOnly);
        assert!(result.semantic_available);
        assert!(
            !result.results.is_empty(),
            "the semantic hits must be there"
        );
        assert!(
            result.results.iter().all(|r| r.id != 999),
            "a lexical-only candidate must not appear in semantic-only mode"
        );
        assert!(result
            .results
            .iter()
            .all(|r| r.source == ResultSource::Semantic));
        assert!(
            result.results.iter().all(|r| r.lexical_score.is_none()),
            "no BM25 score may leak into a semantic-only result"
        );
    }

    #[test]
    fn semantic_only_scores_are_not_scaled_down_by_a_missing_lexical_side() {
        let dir = TempDir::new("semantic_only_scores");
        let coordinator = indexed_coordinator(&dir);

        let result = coordinator
            .search(
                LINE_ONE,
                vec![],
                &HybridSearchParams {
                    force_mode: Some(SearchMode::SemanticOnly),
                    ..Default::default()
                },
            )
            .unwrap();

        let top = &result.results[0];
        assert_eq!(top.id, 1, "the exact line should rank first");
        assert!(
            top.fused_score > 0.9,
            "a self-match must score near 1.0 in semantic-only mode, got {}",
            top.fused_score
        );
        let provenance = top.provenance.as_ref().unwrap();
        assert_eq!(provenance.lexical_weight, 0.0);
        assert_eq!(provenance.semantic_weight, 1.0);
    }

    #[test]
    fn lexical_only_mode_never_consults_the_semantic_path() {
        let dir = TempDir::new("lexical_only");
        let coordinator = indexed_coordinator(&dir);

        let result = coordinator
            .search(
                LINE_ONE,
                vec![lexical(1, LINE_ONE, 15.5)],
                &HybridSearchParams {
                    force_mode: Some(SearchMode::LexicalOnly),
                    ..Default::default()
                },
            )
            .unwrap();

        assert_eq!(result.search_mode, SearchMode::LexicalOnly);
        assert_eq!(result.results.len(), 1);
        assert_eq!(result.results[0].source, ResultSource::Lexical);
        assert!(
            result.fallback_reason.is_none(),
            "lexical-only was requested; that is not a degradation"
        );
        // Nothing was scaled away by a semantic weight that never applied.
        let provenance = result.results[0].provenance.as_ref().unwrap();
        assert_eq!(provenance.lexical_weight, 1.0);
    }

    #[test]
    fn lexical_only_scores_survive_normalization_ordering() {
        let dir = TempDir::new("lexical_order");
        let coordinator = indexed_coordinator(&dir);

        let result = coordinator
            .search(
                LINE_ONE,
                vec![
                    lexical(1, LINE_ONE, 2.0),
                    lexical(2, LINE_TWO, 30.0),
                    lexical(3, LINE_THREE, 9.0),
                ],
                &HybridSearchParams {
                    force_mode: Some(SearchMode::LexicalOnly),
                    ..Default::default()
                },
            )
            .unwrap();

        let ids: Vec<u64> = result.results.iter().map(|r| r.id).collect();
        assert_eq!(ids, vec![2, 3, 1], "higher BM25 must rank higher");
    }

    // ── graceful degradation ──

    #[test]
    fn hybrid_falls_back_to_lexical_when_the_semantic_path_fails() {
        let dir = TempDir::new("degradation");
        let coordinator = broken_coordinator(&dir);

        let result = coordinator
            .search(
                LINE_ONE,
                vec![lexical(1, LINE_ONE, 15.5)],
                &HybridSearchParams::default(),
            )
            .unwrap();

        assert_eq!(
            result.search_mode,
            SearchMode::LexicalOnly,
            "the reported mode must be the one that actually ran"
        );
        assert!(!result.semantic_available);
        assert!(
            result.fallback_reason.is_some(),
            "a silent degradation is indistinguishable from agreement"
        );
        assert_eq!(result.results.len(), 1, "BM25 results still come through");
    }

    #[test]
    fn semantic_only_reports_the_failure_instead_of_serving_lexical_results() {
        let dir = TempDir::new("semantic_only_broken");
        let coordinator = broken_coordinator(&dir);

        let result = coordinator
            .search(
                LINE_ONE,
                vec![lexical(1, LINE_ONE, 15.5)],
                &HybridSearchParams {
                    force_mode: Some(SearchMode::SemanticOnly),
                    ..Default::default()
                },
            )
            .unwrap();

        assert_eq!(result.search_mode, SearchMode::SemanticOnly);
        assert!(!result.semantic_available);
        assert!(result.fallback_reason.is_some());
        assert!(
            result.results.is_empty(),
            "lexical results must not be passed off as semantic ones"
        );
    }

    #[test]
    fn a_coordinator_without_a_semantic_engine_still_serves_lexical_search() {
        let coordinator = HybridCoordinator::new(None);

        let result = coordinator
            .search(
                LINE_ONE,
                vec![lexical(1, LINE_ONE, 15.5)],
                &HybridSearchParams::default(),
            )
            .unwrap();

        assert_eq!(result.search_mode, SearchMode::LexicalOnly);
        assert!(!result.semantic_available);
        assert_eq!(
            result.fallback_reason.as_deref(),
            Some("no semantic engine is configured")
        );
        assert_eq!(result.results.len(), 1);

        let status = coordinator.status();
        assert!(!status.available);
        assert!(!status.model_loaded);
    }

    #[test]
    fn an_empty_query_does_not_panic_and_degrades_cleanly() {
        let dir = TempDir::new("empty_query");
        let coordinator = indexed_coordinator(&dir);

        let result = coordinator
            .search(
                "",
                vec![lexical(1, LINE_ONE, 1.0)],
                &HybridSearchParams::default(),
            )
            .unwrap();

        // The semantic side cannot embed an empty query; the lexical side still works.
        assert_eq!(result.search_mode, SearchMode::LexicalOnly);
        assert!(result.fallback_reason.is_some());
        assert_eq!(result.results.len(), 1);
    }

    #[test]
    fn a_search_with_no_candidates_at_all_returns_an_empty_result_not_an_error() {
        let coordinator = HybridCoordinator::new(None);
        let result = coordinator
            .search("שאילתה ללא תוצאות", vec![], &HybridSearchParams::default())
            .unwrap();

        assert!(result.results.is_empty());
        assert_eq!(result.total_count, 0);
        assert!(result.group_count.is_none());
    }

    // ── pagination and grouping ──

    #[test]
    fn pagination_is_stable_and_does_not_repeat_or_skip_results() {
        let dir = TempDir::new("pagination");
        let coordinator = indexed_coordinator(&dir);
        let lexical_candidates = vec![
            lexical(1, LINE_ONE, 10.0),
            lexical(2, LINE_TWO, 8.0),
            lexical(3, LINE_THREE, 6.0),
        ];

        let page = |offset: usize| {
            coordinator
                .search(
                    LINE_ONE,
                    lexical_candidates.clone(),
                    &HybridSearchParams {
                        limit: 2,
                        offset,
                        ..Default::default()
                    },
                )
                .unwrap()
        };

        let first = page(0);
        let second = page(2);

        assert_eq!(first.total_count, 3);
        assert_eq!(second.total_count, 3);
        assert_eq!(first.results.len(), 2);
        assert_eq!(second.results.len(), 1);

        let mut seen: Vec<u64> = first.results.iter().map(|r| r.id).collect();
        seen.extend(second.results.iter().map(|r| r.id));
        seen.sort_unstable();
        assert_eq!(seen, vec![1, 2, 3], "every result appears exactly once");

        // Repeating the same page yields the same order.
        let first_again = page(0);
        let ids: Vec<u64> = first.results.iter().map(|r| r.id).collect();
        let ids_again: Vec<u64> = first_again.results.iter().map(|r| r.id).collect();
        assert_eq!(ids, ids_again);
    }

    #[test]
    fn an_offset_past_the_end_returns_no_results() {
        let dir = TempDir::new("offset_past_end");
        let coordinator = indexed_coordinator(&dir);

        let result = coordinator
            .search(
                LINE_ONE,
                vec![lexical(1, LINE_ONE, 10.0)],
                &HybridSearchParams {
                    limit: 10,
                    offset: 1000,
                    ..Default::default()
                },
            )
            .unwrap();

        assert!(result.results.is_empty());
        assert!(result.total_count > 0, "the total is unaffected by paging");
    }

    #[test]
    fn grouping_by_section_collapses_siblings_and_reports_both_counts() {
        let dir = TempDir::new("grouping");
        let coordinator = indexed_coordinator(&dir);

        let result = coordinator
            .search(
                LINE_ONE,
                vec![lexical(1, LINE_ONE, 10.0), lexical(2, LINE_TWO, 8.0)],
                &HybridSearchParams {
                    grouping: Some(GroupingMode::SameSection),
                    ..Default::default()
                },
            )
            .unwrap();

        // Lines 1 and 2 share section 100; line 3 is in section 101.
        assert_eq!(result.group_count, Some(2));
        assert_eq!(result.total_count, 3, "the candidate total is preserved");

        let big_group = result
            .results
            .iter()
            .find(|r| r.merged_count > 1)
            .expect("section 100 should have collapsed");
        assert_eq!(big_group.merged_count, 2);
        assert_eq!(big_group.merged.len(), 1);
    }

    #[test]
    fn filters_narrow_the_semantic_side_of_a_hybrid_search() {
        let dir = TempDir::new("coordinator_filters");
        let coordinator = indexed_coordinator(&dir);

        let result = coordinator
            .search(
                LINE_ONE,
                vec![],
                &HybridSearchParams {
                    filters: Some(SearchFilters {
                        book_paths: Some(vec!["some/other/book.txt".to_string()]),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            )
            .unwrap();

        assert!(result.semantic_available);
        assert!(
            result.results.is_empty(),
            "the filter excludes the only indexed book"
        );
    }

    #[test]
    fn empty_filter_lists_do_not_suppress_results() {
        let dir = TempDir::new("coordinator_empty_filters");
        let coordinator = indexed_coordinator(&dir);

        let result = coordinator
            .search(
                LINE_ONE,
                vec![],
                &HybridSearchParams {
                    filters: Some(SearchFilters {
                        book_paths: Some(vec![]),
                        facets: Some(vec![]),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            )
            .unwrap();

        assert!(!result.results.is_empty());
    }

    #[test]
    fn the_semantic_candidate_window_is_capped() {
        let coordinator = HybridCoordinator::new(None);
        let huge = HybridSearchParams {
            limit: usize::MAX,
            offset: usize::MAX,
            ..Default::default()
        };
        assert_eq!(coordinator.semantic_top_k(&huge), MAX_SEMANTIC_CANDIDATES);

        let normal = HybridSearchParams {
            limit: 20,
            offset: 40,
            ..Default::default()
        };
        assert_eq!(coordinator.semantic_top_k(&normal), 80);

        // Never zero: a zero window would make the store return nothing.
        let nothing = HybridSearchParams {
            limit: 0,
            offset: 0,
            ..Default::default()
        };
        assert_eq!(coordinator.semantic_top_k(&nothing), 1);
    }

    #[test]
    fn status_passes_the_engine_state_through() {
        let dir = TempDir::new("status");
        let coordinator = indexed_coordinator(&dir);

        let status = coordinator.status();
        assert!(status.available);
        assert!(status.model_loaded);
        assert_eq!(status.vector_count, 3);
        assert_eq!(status.indexed_book_count, 1);
        assert_eq!(status.embedding_backend.as_deref(), Some("mock-hash-v1"));
        assert!(!status.vectors_persisted);
    }
}
