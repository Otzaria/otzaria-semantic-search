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

use crate::cancellation::CancellationToken;
use crate::config::feature_flags::FeatureFlags;
use crate::config::profiles::{FusionStrategy, RankingProfile, SearchProfile};
use crate::errors::SemanticSearchError;
use crate::hybrid::fusion::{
    best_first, normalize_bm25_adaptive, normalize_bm25_scores, normalize_semantic_with_threshold,
};
use crate::hybrid::grouping::group_results;
use crate::hybrid::ranking::{
    analyze_query, compute_alpha_with, compute_phrase_match_bonus, compute_rare_term_bonus,
    QueryFeatures,
};
use crate::semantic::engine::SemanticEngine;
use crate::semantic::official_index::{OfficialSemanticIndex, ReloadOutcome};
use crate::semantic::resolve::{CandidateResolver, NoResolver, ResolvedLine, VectorHit};
use crate::semantic::segment_set::SetInfo;
use crate::semantic::types::{
    BookForIndexing, FusedCandidate, GroupingMode, HybridMergedSibling, HybridResultItem,
    HybridSearchResult, IndexDiff, IndexingSummary, LexicalCandidate, ResultSource, SearchFilters,
    SearchMode, SemanticCandidate, SemanticStatus, VectorMetadata,
};
use crate::telemetry::SearchTelemetry;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::{Mutex, RwLock};

/// Upper bound on semantic candidates fetched for one query.
///
/// The store allocates a heap proportional to `top_k`, so an unvalidated
/// `limit`/`offset` from the caller would size an allocation from user input.
/// Hitting the cap is logged rather than silently truncating.
const MAX_SEMANTIC_CANDIDATES: usize = 10_000;

/// Normalized cosine threshold below which semantic candidates contribute zero.
///
/// Without a threshold, an orthogonal vector (cosine ≈ 0) normalizes to 0.5 and
/// gives irrelevant semantic results a mid-range score. This maps everything
/// below the threshold to 0. The value 0.55 corresponds to a raw cosine of
/// approximately 0.1; below that the embedding model considers the texts
/// essentially unrelated.
///
/// The threshold a search applies is its profile's
/// [`semantic_threshold`](RankingProfile::semantic_threshold), which a host can now pass
/// per search ([`HybridSearchParams::ranking`]). This constant stands in only for a
/// threshold that is not a number — which no preset has, the feature flags ignore, and
/// [`RankingProfile::validate`] refuses in a profile passed with a search.
const SEMANTIC_RELEVANCE_THRESHOLD: f32 = 0.55;

/// Minimum normalized query length (in characters) to store a result in the
/// query cache. Single-character queries from live typing pollute the cache
/// with entries that are almost never reused.
const MIN_CACHEABLE_QUERY_LEN: usize = 2;

/// Configuration for hybrid search execution.
#[derive(Debug, Clone)]
pub struct HybridSearchParams {
    pub limit: usize,
    pub offset: usize,
    pub grouping: Option<GroupingMode>,
    pub filters: Option<SearchFilters>,
    /// Forces a retrieval mode. `None` means [`SearchMode::Hybrid`].
    pub force_mode: Option<SearchMode>,
    pub profile: Option<crate::config::profiles::SearchProfile>,
    pub feature_flags: Option<crate::config::feature_flags::FeatureFlags>,
    /// Every ranking parameter for this search, in place of the preset `profile` names.
    ///
    /// What lets a host tune the ranking — the fusion strategy and RRF's `k`, alpha per
    /// query type or one fixed alpha, BM25's `k`, the semantic threshold, the bonuses —
    /// without a release of this crate. It is
    /// [validated](crate::config::profiles::RankingProfile::validate) before anything runs,
    /// and a parameter out of range fails the search with
    /// [`SemanticSearchError::InvalidRankingParameter`] rather than ranking by a value
    /// nobody chose. `profile` is then not consulted; `feature_flags` still apply on top,
    /// as they do to a preset.
    ///
    /// `None`, the default, ranks by the preset exactly as before. The defaults are
    /// unmeasured placeholders: [`RankingProfile`] says what calibrating them needs.
    pub ranking: Option<RankingProfile>,
}

impl Default for HybridSearchParams {
    fn default() -> Self {
        Self {
            limit: 20,
            offset: 0,
            grouping: None,
            filters: None,
            force_mode: None,
            profile: None,
            feature_flags: None,
            ranking: None,
        }
    }
}

/// Which semantic path a coordinator is serving.
///
/// Two variants because they are two different things, not two configurations of one.
/// [`Self::Official`] is the application path: an artifact built elsewhere, verified,
/// opened read-only, with no library to compare against and nothing to re-index.
/// [`Self::SelfBuilt`] is the builder and prototype path, which owns a
/// [`SemanticEngine`] and can still write.
///
/// The read side is served identically by both, which is why hybrid search does not care
/// which one it has. Every build-side request goes through an accessor only the self-built
/// variant satisfies, so an official artifact refuses it by name — a
/// [`SemanticSearchError::ReadOnlyIndex`] rather than the `None` that means "no semantic
/// path at all".
// Both variants are large and the difference between them is 216 bytes, of which exactly
// one exists per coordinator, behind a lock, read by reference. Boxing to even them out
// would buy that once and pay an indirection on every query.
#[allow(clippy::large_enum_variant)]
pub enum SemanticSide {
    SelfBuilt(SemanticEngine),
    Official(OfficialSemanticIndex),
}

impl From<SemanticEngine> for SemanticSide {
    fn from(engine: SemanticEngine) -> Self {
        Self::SelfBuilt(engine)
    }
}

impl From<OfficialSemanticIndex> for SemanticSide {
    fn from(index: OfficialSemanticIndex) -> Self {
        Self::Official(index)
    }
}

impl SemanticSide {
    fn embed_query(&self, query: &str) -> Result<Vec<f32>, SemanticSearchError> {
        match self {
            Self::SelfBuilt(engine) => engine.embed_query(query),
            Self::Official(index) => index.embed_query(query),
        }
    }

    /// The semantic candidates for one query vector.
    ///
    /// A self-built index holds its lines' metadata and answers alone. An official vector
    /// set holds keys and the records where they were built, so its hits go through the
    /// host's `resolver`, and every candidate carries the live line's id, section, line
    /// hash and facets — never the vectors'.
    ///
    /// Under a filter the scan reads the books the resolver admits, and weighs besides the
    /// vectors it names as [unreached](CandidateResolver::unreached): those of texts that
    /// moved into an admitted book, at their own scores, beside the scan's hits.
    fn candidates(
        &self,
        query_vector: &[f32],
        top_k: usize,
        filters: Option<&SearchFilters>,
        resolver: &dyn CandidateResolver,
        cancel: &CancellationToken,
        telemetry: &mut SearchTelemetry,
    ) -> Result<Vec<SemanticCandidate>, SemanticSearchError> {
        let index = match self {
            Self::SelfBuilt(engine) => {
                return engine.search_vector(query_vector, top_k, filters, cancel)
            }
            Self::Official(index) => index,
        };
        let books = resolver.admissible_books(filters)?;
        let unreached = match &books {
            Some(_) => resolver.unreached(filters, index.generation(), cancel)?,
            None => Vec::new(),
        };
        let scan_started = std::time::Instant::now();
        let hits =
            index.search_hits_with(query_vector, top_k, books.as_ref(), &unreached, cancel)?;
        telemetry.scan_ms = Some(scan_started.elapsed().as_millis() as u64);
        telemetry.semantic_hits = hits.len().min(u32::MAX as usize) as u32;
        cancel.checkpoint()?;

        let resolve_started = std::time::Instant::now();
        let lines = resolver.resolve(&hits, filters, cancel)?;
        telemetry.resolve_ms = Some(resolve_started.elapsed().as_millis() as u64);
        let (candidates, unresolved) = candidates_of(&hits, lines);
        telemetry.semantic_unresolved = unresolved;
        Ok(candidates)
    }

    /// The generation of the vectors a search reads, folded into the query cache's key: a
    /// reloaded set answers differently.
    fn vectors_generation(&self) -> u64 {
        match self {
            Self::SelfBuilt(_) => 0,
            Self::Official(index) => index.generation(),
        }
    }

    fn status(&self) -> SemanticStatus {
        match self {
            Self::SelfBuilt(engine) => engine.status(),
            Self::Official(index) => index.status(),
        }
    }

    /// The engine behind a self-built index, or a refusal naming the operation.
    fn builder(
        &mut self,
        operation: &'static str,
    ) -> Result<&mut SemanticEngine, SemanticSearchError> {
        match self {
            Self::SelfBuilt(engine) => Ok(engine),
            Self::Official(_) => Err(SemanticSearchError::ReadOnlyIndex { operation }),
        }
    }

    /// As [`Self::builder`], for a build-side question that only reads — so asking it does
    /// not need the writer's lock and does not stall searches.
    fn builder_ref(&self, operation: &'static str) -> Result<&SemanticEngine, SemanticSearchError> {
        match self {
            Self::SelfBuilt(engine) => Ok(engine),
            Self::Official(_) => Err(SemanticSearchError::ReadOnlyIndex { operation }),
        }
    }
}

/// The candidates the lines a resolver returned for `hits` make, and how many hits resolved
/// to none.
///
/// In the scan's order — best hit first, ties by key — and within one hit in the order the
/// resolver gave its lines, best placed first; whatever order it returned them in. The
/// position is the rank a fusion by rank reads, and a resolver may well return a hit's lines
/// late: the application's appends those of text that moved books after all the others.
fn candidates_of(
    hits: &[VectorHit],
    mut lines: Vec<ResolvedLine>,
) -> (Vec<SemanticCandidate>, u32) {
    // Stable, so each hit keeps its lines' order.
    lines.sort_by_key(|line| line.hit);
    let mut resolved = vec![false; hits.len()];
    let candidates: Vec<SemanticCandidate> = lines
        .into_iter()
        .filter_map(|line| {
            let Some(hit) = hits.get(line.hit as usize) else {
                log::warn!(
                    "The resolver returned a line for hit {} of {}; it is skipped",
                    line.hit,
                    hits.len()
                );
                return None;
            };
            resolved[line.hit as usize] = true;
            let key = hit.key.to_hex();
            Some(SemanticCandidate {
                metadata: VectorMetadata {
                    semantic_id: key.clone(),
                    source_doc_key: format!("{}#{}", line.file_path, line.segment),
                    source_book_key: line.file_path,
                    line_id: line.line_id,
                    section_id: line.section_id,
                    line_hash: line.line_hash,
                    chunk_hash: key,
                    content_hash: 0,
                    reference: line.reference,
                    segment: line.segment,
                    is_pdf: line.is_pdf,
                    title: line.title,
                    facets: line.facets.to_vec(),
                },
                similarity_score: hit.score,
            })
        })
        .collect();
    let unresolved = resolved.iter().filter(|resolved| !**resolved).count() as u32;
    (candidates, unresolved)
}

/// Main hybrid search coordinator.
pub struct HybridCoordinator {
    semantic: RwLock<Option<SemanticSide>>,
    /// Held for the whole of [`HybridCoordinator::index_books`].
    ///
    /// The engine lock is released between books so searches can run, which also
    /// means two indexing runs could interleave — each dropping and re-inserting
    /// the other's books, with the manifest committed by whichever finished last.
    /// This serializes them instead. Distinct from the engine lock on purpose: it
    /// excludes other *writers* without excluding readers.
    indexing: Mutex<()>,
    query_cache: crate::hybrid::cache::QueryCache,
    embedding_cache: crate::semantic::embedding_cache::EmbeddingCache,
    telemetry: crate::telemetry::TelemetryCollector,
    normalizer: crate::hybrid::hebrew_normalizer::HebrewNormalizer,
    metadata_ranker: crate::hybrid::metadata_ranker::MetadataRanker,
}

struct FusionContext<'a> {
    alpha: f32,
    mode: SearchMode,
    profile: &'a RankingProfile,
    query_features: &'a QueryFeatures,
    query_facets: &'a [String],
}

impl HybridCoordinator {
    /// Create a coordinator over a self-built index. Passing `None` disables the semantic
    /// path.
    ///
    /// The builder and prototype path — see [`SemanticSide`]. The application uses
    /// [`Self::with_official_index`].
    pub fn new(semantic_engine: Option<SemanticEngine>) -> Self {
        Self::with_semantic_side(semantic_engine.map(SemanticSide::from))
    }

    /// Create a coordinator over an installed official artifact: read-only, with every
    /// build-side operation refused by name.
    pub fn with_official_index(index: OfficialSemanticIndex) -> Self {
        Self::with_semantic_side(Some(SemanticSide::from(index)))
    }

    fn with_semantic_side(semantic: Option<SemanticSide>) -> Self {
        Self {
            semantic: RwLock::new(semantic),
            indexing: Mutex::new(()),
            query_cache: crate::hybrid::cache::QueryCache::new(
                100,
                std::time::Duration::from_secs(300),
            ),
            embedding_cache: crate::semantic::embedding_cache::EmbeddingCache::new(500),
            telemetry: crate::telemetry::TelemetryCollector::new(),
            normalizer: crate::hybrid::hebrew_normalizer::HebrewNormalizer::new(),
            metadata_ranker: crate::hybrid::metadata_ranker::MetadataRanker::default(),
        }
    }

    /// Primary search entry point. Coordinates BM25 and semantic candidates.
    ///
    /// [`Self::search_cancellable`] with a token nobody cancels, so never
    /// [`SemanticSearchError::Cancelled`].
    ///
    /// Searches through [`NoResolver`]: right for a self-built index, which resolves its
    /// own hits. An official vector set needs the host's resolver —
    /// [`Self::search_cancellable`] — and without one its semantic side contributes
    /// nothing and says why.
    pub fn search(
        &self,
        query: &str,
        lexical_candidates: Vec<LexicalCandidate>,
        params: &HybridSearchParams,
    ) -> Result<HybridSearchResult, SemanticSearchError> {
        self.search_cancellable(
            query,
            lexical_candidates,
            params,
            &NoResolver,
            &CancellationToken::new(),
        )
    }

    /// As [`Self::search`], or [`SemanticSearchError::Cancelled`] once `cancel` is
    /// cancelled.
    ///
    /// The token is looked at before anything else — before either cache is consulted —
    /// after the query is embedded, throughout the vector scan, before fusion, and after
    /// it; [`crate::cancellation`] says why each one is there.
    ///
    /// A cancelled search returns `Cancelled` from every mode. It is never degraded to the
    /// lexical results the way a failed semantic path is, because nobody wants the answer
    /// any more, and it leaves this coordinator exactly as it found it: the result is not
    /// cached, the query's embedding is not cached either (both caches are written only
    /// after the last checkpoint), and the search is not counted in the telemetry.
    ///
    /// `resolver` ties an official vector set's hits to the host's live lines — see
    /// [`CandidateResolver`]; a self-built index never asks it. A resolver that fails is a
    /// semantic failure like any other: the search degrades to its lexical results, and
    /// [`HybridSearchResult::fallback_reason`] says why.
    pub fn search_cancellable(
        &self,
        query: &str,
        lexical_candidates: Vec<LexicalCandidate>,
        params: &HybridSearchParams,
        resolver: &dyn CandidateResolver,
        cancel: &CancellationToken,
    ) -> Result<HybridSearchResult, SemanticSearchError> {
        cancel.checkpoint()?;
        let start_time = std::time::Instant::now();
        let requested = params.force_mode.unwrap_or(SearchMode::Hybrid);
        let flags = params.feature_flags.clone().unwrap_or_default();
        let ranking_profile = match &params.ranking {
            // Refused here, before anything is looked up or computed: a parameter out of
            // range would otherwise rank, silently, by whatever the arithmetic made of it.
            Some(ranking) => {
                ranking.validate()?;
                let mut ranking = ranking.clone();
                flags.apply(&mut ranking);
                ranking
            }
            None => {
                let selected_profile = params.profile.unwrap_or(SearchProfile::Balanced);
                FeatureFlags::resolve(selected_profile, &flags)
            }
        };
        let normalized_query = self.normalizer.normalize(query);
        let query_features = analyze_query(&normalized_query);
        let requested_alpha = ranking_profile
            .alpha_override
            .unwrap_or_else(|| {
                compute_alpha_with(&query_features, &ranking_profile.alpha_by_query_type)
            })
            .clamp(0.0, 1.0);
        let telemetry_per_query = flags.telemetry_per_query.unwrap_or(true);

        let mut telemetry_record = crate::telemetry::SearchTelemetry {
            query_type: query_features.estimated_type.to_string(),
            search_mode: requested.to_string(),
            fusion_strategy: ranking_profile.fusion_strategy.to_string(),
            alpha: requested_alpha,
            lexical_candidates: lexical_candidates.len().min(u32::MAX as usize) as u32,
            semantic_candidates: 0,
            fused_candidates: 0,
            cache_lookup: false,
            cache_hit: false,
            latency_ms: 0,
            embedding_latency_ms: None,
            fusion_latency_ms: 0,
            confidence: None,
            profile: ranking_profile.profile.to_string(),
            semantic_hits: 0,
            semantic_unresolved: 0,
            scan_ms: None,
            resolve_ms: None,
        };

        // §1.1 + §3.4: Do not read from or write to the cache for empty or
        // very short queries. Empty queries cannot be embedded and will always
        // degrade to lexical-only; caching that would poison future lookups.
        // Single-character queries from live typing are almost never reused.
        let cacheable = ranking_profile.query_cache_enabled
            && normalized_query.chars().count() >= MIN_CACHEABLE_QUERY_LEN;
        telemetry_record.cache_lookup = cacheable;

        // Recover rather than propagate a poisoned lock: a panic in one query
        // must not disable the semantic path for the rest of the session.
        let semantic_guard = self.semantic.read().unwrap_or_else(|e| e.into_inner());

        // The lexical candidates are inputs, not state owned by this coordinator.
        // Hashing them prevents a cache hit after Tantivy produced a new window — and the
        // two generations, one after an index commit or a vector set reload: a
        // semantic-only search has no lexical input to change with them.
        let inputs_hash = hash_search_inputs(
            &params.filters,
            &lexical_candidates,
            &ranking_profile,
            &params.feature_flags,
            [
                resolver.generation(),
                semantic_guard
                    .as_ref()
                    .map_or(0, SemanticSide::vectors_generation),
            ],
        );
        let cache_key = crate::hybrid::cache::QueryCache::compute_key(
            query,
            inputs_hash,
            &requested.to_string(),
            &format!("{:?}", params.grouping),
            params.limit,
            params.offset,
        );
        if cacheable {
            if let Some(mut cached_result) = self.query_cache.get(cache_key) {
                let latency_ms = start_time.elapsed().as_millis() as u64;
                cached_result.latency_ms = latency_ms;
                telemetry_record.cache_hit = true;
                telemetry_record.search_mode = cached_result.search_mode.to_string();
                if cached_result.search_mode != SearchMode::Hybrid {
                    telemetry_record.fusion_strategy = "SingleSource".to_string();
                }
                telemetry_record.semantic_candidates = cached_result
                    .telemetry
                    .as_ref()
                    .map_or(0, |record| record.semantic_candidates);
                telemetry_record.fused_candidates = cached_result
                    .telemetry
                    .as_ref()
                    .map_or(cached_result.total_count, |record| record.fused_candidates);
                telemetry_record.latency_ms = latency_ms;
                telemetry_record.confidence = cached_result.confidence;

                if ranking_profile.telemetry_enabled {
                    self.telemetry.record_search(&telemetry_record);
                }
                cached_result.telemetry = (ranking_profile.telemetry_enabled
                    && telemetry_per_query)
                    .then_some(telemetry_record);
                return Ok(cached_result);
            }
        }

        let skip_semantic_for_exact = requested == SearchMode::Hybrid && requested_alpha >= 1.0;

        // A vector this query had to embed, kept out of the embedding cache until the last
        // checkpoint has passed: a cancelled search writes to neither cache.
        let mut fresh_embedding = None;

        let semantic = if requested == SearchMode::LexicalOnly || skip_semantic_for_exact {
            // Not consulted — this is the caller's choice (or an optimization),
            // not a degradation.
            SemanticOutcome::skipped()
        } else {
            match semantic_guard.as_ref() {
                None => SemanticOutcome::failed("no semantic index is configured".to_string()),
                Some(side) => {
                    let embedding_start = std::time::Instant::now();
                    let cached_vector = ranking_profile
                        .embedding_cache_enabled
                        .then(|| self.embedding_cache.get(&normalized_query))
                        .flatten();

                    let query_vector = match cached_vector {
                        Some(vector) => Ok(vector),
                        None => {
                            let result = side.embed_query(&normalized_query);
                            telemetry_record.embedding_latency_ms =
                                Some(embedding_start.elapsed().as_millis() as u64);
                            if ranking_profile.embedding_cache_enabled {
                                if let Ok(vector) = &result {
                                    fresh_embedding = Some(vector.clone());
                                }
                            }
                            result
                        }
                    };
                    // Embedding is the one stage nothing can interrupt, so look again
                    // before paying for the scan.
                    cancel.checkpoint()?;

                    match query_vector.and_then(|vector| {
                        side.candidates(
                            &vector,
                            self.semantic_top_k(params, &ranking_profile),
                            params.filters.as_ref(),
                            resolver,
                            cancel,
                            &mut telemetry_record,
                        )
                    }) {
                        Ok(candidates) => SemanticOutcome::ok(candidates),
                        // Abandoned, not failed: nothing to degrade to, because nobody is
                        // waiting for the lexical results either.
                        Err(SemanticSearchError::Cancelled) => {
                            return Err(SemanticSearchError::Cancelled)
                        }
                        Err(error) => {
                            log::warn!(
                                "Semantic search path failed: {error}. Serving the lexical results."
                            );
                            SemanticOutcome::failed(error.to_string())
                        }
                    }
                }
            }
        };
        // Fusion is cheap next to the scan behind it, but its result would only be thrown
        // away.
        cancel.checkpoint()?;

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
            SearchMode::Hybrid => requested_alpha,
        };

        let sem_len = semantic.candidates.len().min(u32::MAX as usize) as u32;
        let fusion_start = std::time::Instant::now();
        let query_facets = params
            .filters
            .as_ref()
            .and_then(|filters| filters.facets.as_deref())
            .unwrap_or_default();
        let fused = self.fuse_candidates(
            lexical_candidates,
            semantic.candidates,
            FusionContext {
                alpha,
                mode,
                profile: &ranking_profile,
                query_features: &query_features,
                query_facets,
            },
        );
        let fused_count = fused.len().min(u32::MAX as usize) as u32;

        let scores: Vec<f32> = fused.iter().map(|c| c.fused_score).collect();
        let confidence = crate::hybrid::fusion::compute_confidence(&scores);

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
                    .collect::<Vec<_>>();

                (results, total, Some(group_count))
            }
            None => {
                let total = fused.len() as u32;
                let results = fused
                    .into_iter()
                    .skip(params.offset)
                    .take(params.limit)
                    .map(|candidate| into_result_item(candidate, 1, Vec::new()))
                    .collect::<Vec<_>>();

                (results, total, None)
            }
        };

        let fusion_latency_ms = fusion_start.elapsed().as_millis() as u64;
        let latency_ms = start_time.elapsed().as_millis() as u64;

        // §2.4: Populate telemetry fields BEFORE cloning into the result, so
        // the client-visible telemetry carries the final values instead of the
        // initial defaults (zeroes).
        telemetry_record.search_mode = mode.to_string();
        if mode != SearchMode::Hybrid {
            telemetry_record.fusion_strategy = "SingleSource".to_string();
        }
        telemetry_record.alpha = alpha;
        telemetry_record.fusion_latency_ms = fusion_latency_ms;
        telemetry_record.fused_candidates = fused_count;
        telemetry_record.latency_ms = latency_ms;
        telemetry_record.confidence = confidence;
        telemetry_record.semantic_candidates = sem_len;

        let final_result = HybridSearchResult {
            results,
            total_count,
            group_count,
            search_mode: mode,
            semantic_available: semantic.healthy,
            fallback_reason: semantic.failure,
            latency_ms,
            confidence,
            profile: Some(ranking_profile.profile.to_string()),
            telemetry: (ranking_profile.telemetry_enabled && telemetry_per_query)
                .then_some(telemetry_record.clone()),
        };

        // The last look, with the result in hand and nothing recorded yet. A search
        // cancelled while it fused is not counted, not cached, and not handed back to be
        // hydrated — hydration costs the caller a lookup per result.
        cancel.checkpoint()?;

        if ranking_profile.telemetry_enabled {
            self.telemetry.record_search(&telemetry_record);
        }

        if let Some(vector) = fresh_embedding {
            self.embedding_cache.insert(&normalized_query, vector);
        }

        // §1.1 + §3.4: Only cache queries with enough substance to be reused.
        if cacheable {
            // Keep the complete telemetry record internally even when the
            // caller opted out of per-query telemetry.  A later cache hit
            // still needs the original candidate counts for aggregate stats.
            let mut cached_result = final_result.clone();
            cached_result.telemetry = Some(telemetry_record.clone());
            self.query_cache.insert_with_capacity(
                cache_key,
                cached_result,
                flags.query_cache_capacity.unwrap_or(100),
            );
        }

        Ok(final_result)
    }

    /// How many semantic candidates to fetch for one page of results.
    ///
    /// Over-fetches relative to `limit` because fusion, grouping and dedup all
    /// discard candidates, so the page must be filled from a wider window.
    fn semantic_top_k(&self, params: &HybridSearchParams, profile: &RankingProfile) -> usize {
        let multiplier = if profile.candidate_window_multiplier.is_finite() {
            profile.candidate_window_multiplier.clamp(1.0, 10.0)
        } else {
            2.0
        };
        let window = ((params.limit as f64) * multiplier as f64)
            .ceil()
            .min(usize::MAX as f64) as usize;
        let requested = params.offset.saturating_add(window).max(1);

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
    /// Candidates are merged on the book and the line together — `(file_path, line_id)` —
    /// never on the id alone: an index that was updated book by book can give two books
    /// the same id range (its ids encode a catalogue position that moves), and a merge on
    /// the id would fuse one book's lexical hit with another book's semantic one. See
    /// [`FusedCandidate`].
    fn fuse_candidates(
        &self,
        lexical: Vec<LexicalCandidate>,
        semantic: Vec<SemanticCandidate>,
        context: FusionContext<'_>,
    ) -> Vec<FusedCandidate> {
        let FusionContext {
            alpha,
            mode,
            profile,
            query_features,
            query_facets,
        } = context;
        let mut lexical_by_id: HashMap<(String, u64), (usize, LexicalCandidate)> = HashMap::new();
        for (rank, candidate) in lexical.into_iter().enumerate() {
            match lexical_by_id.entry((candidate.file_path.clone(), candidate.line_id)) {
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert((rank, candidate));
                }
                std::collections::hash_map::Entry::Occupied(mut entry) => {
                    if candidate
                        .bm25_score
                        .total_cmp(&entry.get().1.bm25_score)
                        .is_gt()
                    {
                        entry.insert((rank, candidate));
                    }
                }
            }
        }
        let mut lexical: Vec<(usize, LexicalCandidate)> = lexical_by_id.into_values().collect();
        lexical.sort_by_key(|(rank, _)| *rank);

        let mut semantic_by_id: HashMap<(String, u64), (usize, SemanticCandidate)> = HashMap::new();
        for (rank, candidate) in semantic.into_iter().enumerate() {
            let line = (
                candidate.metadata.source_book_key.clone(),
                candidate.metadata.line_id,
            );
            match semantic_by_id.entry(line) {
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert((rank, candidate));
                }
                std::collections::hash_map::Entry::Occupied(mut entry) => {
                    if candidate
                        .similarity_score
                        .total_cmp(&entry.get().1.similarity_score)
                        .is_gt()
                    {
                        entry.insert((rank, candidate));
                    }
                }
            }
        }
        let mut semantic: Vec<(usize, SemanticCandidate)> = semantic_by_id.into_values().collect();
        semantic.sort_by_key(|(rank, _)| *rank);

        let bm25_scores: Vec<f32> = lexical
            .iter()
            .map(|(_, candidate)| candidate.bm25_score)
            .collect();
        let norm_bm25 = match profile.fusion_strategy {
            FusionStrategy::Adaptive => {
                normalize_bm25_adaptive(&bm25_scores, profile.bm25_saturation_k)
            }
            FusionStrategy::Weighted | FusionStrategy::RRF { .. } => {
                normalize_bm25_scores(&bm25_scores, profile.bm25_saturation_k)
            }
        };

        let sem_scores: Vec<f32> = semantic
            .iter()
            .map(|(_, candidate)| candidate.similarity_score)
            .collect();
        let threshold = if profile.semantic_threshold.is_finite() {
            profile.semantic_threshold.clamp(0.0, 1.0)
        } else {
            SEMANTIC_RELEVANCE_THRESHOLD
        };
        let norm_sem = normalize_semantic_with_threshold(&sem_scores, threshold);
        let rrf_k = match profile.fusion_strategy {
            FusionStrategy::RRF { k } if mode == SearchMode::Hybrid => Some(k.max(1)),
            _ => None,
        };

        let mut fused_map: HashMap<(String, u64), FusedCandidate> =
            HashMap::with_capacity(lexical.len() + semantic.len());

        for ((_, candidate), &normalized) in lexical.into_iter().zip(norm_bm25.iter()) {
            let lexical_rank = fused_map.len() as u32 + 1;
            let fused_score = rrf_k
                .map(|k| 1.0 / (k as f32 + lexical_rank as f32))
                .unwrap_or(alpha * normalized);
            let line_id = candidate.line_id;

            fused_map.insert(
                (candidate.file_path.clone(), line_id),
                FusedCandidate {
                    title: candidate.title,
                    reference: candidate.reference,
                    text: candidate.text,
                    line_id,
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
                    fused_score,
                    lexical_weight: alpha,
                    semantic_weight: 1.0 - alpha,
                },
            );
        }

        for (semantic_rank, ((_, candidate), &normalized)) in
            semantic.into_iter().zip(norm_sem.iter()).enumerate()
        {
            // RRF ignores score magnitudes, so a threshold only has meaning if
            // candidates below it are excluded. Weighted/adaptive fusion keeps
            // them at zero to preserve semantic-only paging and grouping.
            if normalized <= 0.0 && rrf_k.is_some() {
                continue;
            }
            let line_id = candidate.metadata.line_id;
            let contribution = rrf_k
                .map(|k| 1.0 / (k as f32 + semantic_rank as f32 + 1.0))
                .unwrap_or((1.0 - alpha) * normalized);
            let metadata_bonus = if profile.metadata_ranking_enabled && rrf_k.is_none() {
                self.metadata_ranker
                    .compute_signal(
                        &candidate.metadata.source_book_key,
                        &candidate.metadata.facets,
                        query_facets,
                    )
                    .total
            } else {
                0.0
            };

            match fused_map.get_mut(&(candidate.metadata.source_book_key.clone(), line_id)) {
                // Found by both engines: keep the lexical text and record both
                // scores. Provenance must survive fusion.
                Some(existing) => {
                    existing.source = ResultSource::Both;
                    existing.raw_semantic_score = Some(candidate.similarity_score);
                    existing.normalized_semantic = Some(normalized);
                    existing.fused_score += contribution + metadata_bonus;
                    if mode == SearchMode::Hybrid && rrf_k.is_none() {
                        existing.fused_score += profile.agreement_bonus.max(0.0);
                    }
                }
                // Semantic-only: the vector store holds metadata but no line
                // body, so the text has to be hydrated from Tantivy by id.
                None => {
                    let metadata = candidate.metadata;
                    fused_map.insert(
                        (metadata.source_book_key.clone(), line_id),
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
                            fused_score: contribution + metadata_bonus,
                            lexical_weight: alpha,
                            semantic_weight: 1.0 - alpha,
                        },
                    );
                }
            }
        }

        let mut results: Vec<FusedCandidate> = fused_map.into_values().collect();
        if rrf_k.is_none() {
            let mut section_counts: HashMap<(String, u64), usize> = HashMap::new();
            for candidate in &results {
                *section_counts
                    .entry((candidate.file_path.clone(), candidate.section_id))
                    .or_default() += 1;
            }

            for candidate in &mut results {
                candidate.fused_score += profile.phrase_match_bonus.max(0.0)
                    * compute_phrase_match_bonus(&candidate.text, &query_features.quoted_phrases);
                candidate.fused_score += profile.rare_term_bonus.max(0.0)
                    * compute_rare_term_bonus(&candidate.text, &query_features.rare_tokens);
                if section_counts
                    .get(&(candidate.file_path.clone(), candidate.section_id))
                    .copied()
                    .unwrap_or(0)
                    > 1
                {
                    candidate.fused_score += profile.section_coverage_bonus.max(0.0);
                }
            }
        }

        // Ties break on the line, then its book, so pagination is stable across calls;
        // `HashMap` iteration order is not.
        let sort_results = |results: &mut Vec<FusedCandidate>| results.sort_by(best_first);
        sort_results(&mut results);

        if rrf_k.is_none() && profile.duplicate_penalty > 0.0 {
            let mut seen = std::collections::HashSet::new();
            for candidate in &mut results {
                if candidate.line_hash != 0 && !seen.insert(candidate.line_hash) {
                    candidate.fused_score -= profile.duplicate_penalty;
                }
            }
            sort_results(&mut results);
        }
        results
    }

    /// Compare the library's per-book fingerprints against the semantic index.
    ///
    /// `Ok(None)` when no semantic index is configured — there is nothing to index. An
    /// installed official artifact is a refusal, not `None`: asking which books need
    /// indexing presumes this device indexes, and the answer would be a list nobody may
    /// act on.
    pub fn semantic_index_diff(
        &self,
        books: &HashMap<String, crate::semantic::types::ContentFingerprint>,
    ) -> Result<Option<IndexDiff>, SemanticSearchError> {
        let guard = self.semantic.read().unwrap_or_else(|e| e.into_inner());
        match guard.as_ref() {
            None => Ok(None),
            Some(side) => Ok(Some(side.builder_ref("semantic_index_diff")?.diff(books))),
        }
    }

    /// Discard the semantic index and start over for the current configuration.
    ///
    /// The recovery path out of [`SemanticSearchError::IncompatibleIndex`]:
    /// without it a status reporting `needs_full_reindex` would be a dead end.
    /// Returns the number of vectors discarded, or `None` if there is no engine.
    pub fn reset_semantic_index(&self) -> Result<Option<u32>, SemanticSearchError> {
        let _indexing = self.indexing.lock().unwrap_or_else(|e| e.into_inner());
        let mut guard = self.semantic.write().unwrap_or_else(|e| e.into_inner());
        let result = match guard.as_mut() {
            Some(side) => side
                .builder("reset_semantic_index")
                .and_then(SemanticEngine::reset_index)
                .map(Some),
            None => Ok(None),
        };
        if result.is_ok() {
            self.query_cache.invalidate();
        }
        result
    }

    /// Remove books that disappeared from the library.
    ///
    /// This consumes the keys reported by [`IndexDiff::removed_books`]. It shares
    /// the indexing mutex with indexing and reset, so destructive lifecycle
    /// operations cannot land between two books of an active batch.
    pub fn remove_semantic_books(
        &self,
        source_book_keys: &[String],
    ) -> Result<Option<u32>, SemanticSearchError> {
        let _indexing = self.indexing.lock().unwrap_or_else(|e| e.into_inner());
        let mut guard = self.semantic.write().unwrap_or_else(|e| e.into_inner());
        let result = match guard.as_mut() {
            Some(side) => side
                .builder("remove_semantic_books")?
                .remove_books(source_book_keys)
                .map(Some),
            None => Ok(None),
        };
        if matches!(&result, Ok(Some(removed)) if *removed > 0) {
            self.query_cache.invalidate();
        }
        result
    }

    /// Index books into the semantic index, replacing anything held for them.
    ///
    /// Returns `None` if there is no semantic engine. Two concurrent calls do not
    /// interleave — the second waits for the first.
    ///
    /// # Searches block while a book is being indexed
    ///
    /// Indexing needs `&mut SemanticEngine` and searching needs `&`, so the two
    /// cannot overlap: a query issued during indexing waits. The wait is bounded
    /// to **one book** — the lock is taken and released per book rather than held
    /// across the whole set — so a full-library index stays interruptible instead
    /// of blocking search for its entire duration. It is still a stall, not
    /// concurrency.
    ///
    /// Making indexing genuinely concurrent with search needs more than a lock
    /// change: either finer-grained interior mutability inside the engine, or building
    /// into a staging index and swapping it in atomically.
    ///
    /// It is no longer a blocker for the application, though. The app installs a
    /// prebuilt read-only index and never indexes, so the only caller that can be
    /// blocked here is the artifact builder (S4b) — a batch tool with no UI thread.
    ///
    /// # The manifest is written once, not per book
    ///
    /// Releasing the engine lock between books must not mean committing the
    /// manifest between books: every write serializes every record, so per-book
    /// saves move `O(B²)` bytes and ask for `B` `fsync`s. The current vector store
    /// is volatile, so a mid-run manifest checkpoint cannot preserve useful work:
    /// its vectors disappear on restart. The manifest is therefore written once
    /// at the end, and once on an error path to keep in-process state coherent.
    /// A persistent store must add an append-only journal or incremental
    /// checkpoint format before claiming crash-resumable indexing.
    pub fn index_books(
        &self,
        books: &[BookForIndexing],
    ) -> Result<Option<IndexingSummary>, SemanticSearchError> {
        if books.is_empty() {
            let guard = self.semantic.read().unwrap_or_else(|e| e.into_inner());
            return match guard.as_ref() {
                None => Ok(None),
                // Refused even with nothing to do: the request itself is the error, and
                // reporting "indexed 0 books" would tell the caller it may index.
                Some(side) => side
                    .builder_ref("index_books")
                    .map(|_| Some(IndexingSummary::default())),
            };
        }

        let _indexing = self.indexing.lock().unwrap_or_else(|e| e.into_inner());

        let mut summary = IndexingSummary::default();
        let mut dirty = false;

        for book in books {
            let mut guard = self.semantic.write().unwrap_or_else(|e| e.into_inner());
            let Some(side) = guard.as_mut() else {
                return Ok(None);
            };
            let engine = side.builder("index_books")?;

            match engine.index_book_deferred(book) {
                Ok(outcome) => {
                    dirty |= outcome.did_work();
                    summary.record(outcome);
                }
                Err(indexing_error) => {
                    // Commit what did land. Losing the manifest here would strand
                    // vectors the store already holds: nothing on disk would name
                    // them, so the next run would re-embed those books and the old
                    // vectors would sit there unreferenced.
                    if let Err(flush_error) = engine.flush_manifest() {
                        log::warn!(
                            "Could not commit the manifest after an indexing failure \
                             ({flush_error}); completed changes may be re-indexed"
                        );
                    }
                    self.query_cache.invalidate();
                    return Err(indexing_error);
                }
            }
        }

        if dirty {
            let mut guard = self.semantic.write().unwrap_or_else(|e| e.into_inner());
            let Some(side) = guard.as_mut() else {
                return Ok(None);
            };
            side.builder("index_books")?.flush_manifest()?;
            self.query_cache.invalidate();
        }
        Ok(Some(summary))
    }

    /// Whether a semantic index is configured at all — of either kind.
    pub fn has_semantic_index(&self) -> bool {
        self.semantic
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .is_some()
    }

    /// Retrieve semantic status, in the same shape for both kinds of index.
    pub fn status(&self) -> SemanticStatus {
        let guard = self.semantic.read().unwrap_or_else(|e| e.into_inner());

        match guard.as_ref() {
            Some(side) => side.status(),
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

    /// Open the generation an install or a compaction made live, keeping the model, and
    /// drop every cached result — they were computed from the generation before.
    ///
    /// `Ok(None)` when no semantic index is configured; a configuration error for a
    /// self-built one, which has no vector set to reload. Searches wait while the new generation
    /// opens, which reads its small sections and maps its vectors: tens of milliseconds.
    pub fn reload_semantic_vectors(&self) -> Result<Option<ReloadOutcome>, SemanticSearchError> {
        let mut guard = self.semantic.write().unwrap_or_else(|e| e.into_inner());
        let outcome = match guard.as_mut() {
            None => return Ok(None),
            Some(SemanticSide::SelfBuilt(_)) => {
                return Err(SemanticSearchError::Config(
                    "reload_semantic_vectors reloads an official vector set, and this \
                     coordinator serves a self-built index"
                        .to_string(),
                ))
            }
            Some(SemanticSide::Official(index)) => index.reload_vectors()?,
        };
        if matches!(outcome, ReloadOutcome::Reloaded { .. }) {
            self.query_cache.invalidate();
        }
        Ok(Some(outcome))
    }

    /// What the official vector set holds — generation, library version, segments,
    /// whether it wants compacting — or `None` without one.
    pub fn vector_set_info(&self) -> Option<SetInfo> {
        let guard = self.semantic.read().unwrap_or_else(|e| e.into_inner());
        match guard.as_ref() {
            Some(SemanticSide::Official(index)) => Some(index.set_info().clone()),
            _ => None,
        }
    }

    pub fn get_telemetry_snapshot(&self) -> crate::telemetry::TelemetrySnapshot {
        self.telemetry.snapshot()
    }

    pub fn reset_telemetry(&self) {
        self.telemetry.reset();
    }

    pub fn embedding_cache_stats(&self) -> crate::semantic::embedding_cache::EmbeddingCacheStats {
        self.embedding_cache.stats()
    }

    pub fn metadata_ranker(&self) -> &crate::hybrid::metadata_ranker::MetadataRanker {
        &self.metadata_ranker
    }

    pub fn clear_query_cache(&self) {
        self.query_cache.clear();
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

/// Hash every input that can change a search result. Length-prefixing fields
/// prevents adjacent strings from producing the same byte stream.
fn hash_search_inputs(
    filters: &Option<SearchFilters>,
    lexical: &[LexicalCandidate],
    profile: &RankingProfile,
    flags: &Option<FeatureFlags>,
    generations: [u64; 2],
) -> [u8; 32] {
    fn feed(hasher: &mut Sha256, bytes: &[u8]) {
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
    }

    let mut hasher = Sha256::new();
    for generation in generations {
        feed(&mut hasher, &generation.to_le_bytes());
    }
    feed(&mut hasher, format!("{filters:?}").as_bytes());
    feed(&mut hasher, format!("{profile:?}").as_bytes());
    feed(&mut hasher, format!("{flags:?}").as_bytes());

    for candidate in lexical {
        feed(&mut hasher, &candidate.line_id.to_le_bytes());
        feed(&mut hasher, &candidate.section_id.to_le_bytes());
        feed(&mut hasher, &candidate.line_hash.to_le_bytes());
        feed(&mut hasher, &candidate.segment.to_le_bytes());
        feed(&mut hasher, &candidate.bm25_score.to_bits().to_le_bytes());
        feed(&mut hasher, &[u8::from(candidate.is_pdf)]);
        feed(&mut hasher, candidate.title.as_bytes());
        feed(&mut hasher, candidate.reference.as_bytes());
        feed(&mut hasher, candidate.text.as_bytes());
        feed(&mut hasher, candidate.file_path.as_bytes());
    }

    hasher.finalize().into()
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
            // The clock alone collided: macOS ticks coarser than a test takes to start.
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "otzaria_coordinator_test_{name}_{}_{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos(),
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
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
            content_fingerprint: 987654,
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
        let coordinator = semantic_coordinator(dir);
        coordinator.index_books(&[mock_book()]).unwrap().unwrap();
        coordinator
    }

    /// How many times the coordinator's engine has written its manifest.
    fn manifest_save_count(coordinator: &HybridCoordinator) -> u32 {
        let guard = coordinator.semantic.read().unwrap();
        guard
            .as_ref()
            .expect("the coordinator must have a semantic index")
            .builder_ref("manifest_save_count")
            .expect("the coordinator must have an engine")
            .manifest_save_count()
    }

    /// A coordinator over an empty but working semantic engine.
    fn semantic_coordinator(dir: &TempDir) -> HybridCoordinator {
        let model_path = mock::write_stub_onnx_package(&dir.path().join("model"));
        let root = dir.path().join("semantic");

        let engine = SemanticEngine::open(SemanticConfig {
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

        HybridCoordinator::new(Some(engine))
    }

    /// A coordinator whose engine exists but has no model loaded, so every
    /// semantic query fails.
    fn broken_coordinator(dir: &TempDir) -> HybridCoordinator {
        let root = dir.path().join("semantic");
        let engine = SemanticEngine::open(SemanticConfig {
            root_dir: root.clone(),
            // Deliberately absent: load_model will fail.
            model_path: dir.path().join("absent.onnx"),
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
                &HybridSearchParams {
                    feature_flags: Some(FeatureFlags {
                        semantic_threshold_override: Some(0.0),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
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
    fn a_coordinator_without_a_semantic_index_still_serves_lexical_search() {
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
            Some("no semantic index is configured")
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
    fn semantic_only_mode_discards_lexical_candidates_and_honors_mode() {
        let dir = TempDir::new("semantic_only_mode");
        let coordinator = indexed_coordinator(&dir);

        let result = coordinator
            .search(
                LINE_ONE,
                vec![lexical(999, "תורה, בראשית א, א", 100.0)],
                &HybridSearchParams {
                    force_mode: Some(SearchMode::SemanticOnly),
                    ..Default::default()
                },
            )
            .unwrap();

        assert_eq!(result.search_mode, SearchMode::SemanticOnly);
        assert!(result.semantic_available);
        // The supplied lexical candidate (id 999) must be discarded in SemanticOnly mode.
        assert!(result.results.iter().all(|r| r.id != 999));
        assert!(result
            .results
            .iter()
            .all(|r| r.source == ResultSource::Semantic));
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
                    feature_flags: Some(FeatureFlags {
                        semantic_threshold_override: Some(0.0),
                        ..Default::default()
                    }),
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
        let profile = RankingProfile::default();
        let huge = HybridSearchParams {
            limit: usize::MAX,
            offset: usize::MAX,
            ..Default::default()
        };
        assert_eq!(
            coordinator.semantic_top_k(&huge, &profile),
            MAX_SEMANTIC_CANDIDATES
        );

        let normal = HybridSearchParams {
            limit: 20,
            offset: 40,
            ..Default::default()
        };
        assert_eq!(coordinator.semantic_top_k(&normal, &profile), 80);

        // Never zero: a zero window would make the store return nothing.
        let nothing = HybridSearchParams {
            limit: 0,
            offset: 0,
            ..Default::default()
        };
        assert_eq!(coordinator.semantic_top_k(&nothing, &profile), 1);
    }

    /// Indexing and searching cannot overlap — indexing needs `&mut`, searching
    /// `&` — so this pins down what must hold anyway: a search issued while
    /// indexing runs still completes and returns a correct result, and neither
    /// side deadlocks.
    ///
    /// What it deliberately does *not* assert is that searches observe the index
    /// growing book by book. The per-book lock granularity makes that true, but
    /// observing it depends on the reader being scheduled between two of the
    /// writer's acquisitions, which no amount of test structure can guarantee —
    /// on a loaded or single-core runner it would fail for reasons unrelated to
    /// the code. Asserting it would buy a flaky test, not a stronger guarantee.
    #[test]
    fn searches_during_indexing_succeed_and_do_not_deadlock() {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        use std::sync::Arc;

        let dir = TempDir::new("interleave");
        let model_path = mock::write_stub_onnx_package(&dir.path().join("model"));
        let root = dir.path().join("semantic");

        let engine = SemanticEngine::open(SemanticConfig {
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
        let coordinator = Arc::new(HybridCoordinator::new(Some(engine)));

        let books: Vec<BookForIndexing> = (0..40)
            .map(|i| {
                let mut book = mock_book();
                book.source_book_key = format!("otzaria/book{i:02}.txt");
                book.content_fingerprint = i;
                book
            })
            .collect();

        let indexing_done = Arc::new(AtomicBool::new(false));
        let searches = Arc::new(AtomicUsize::new(0));

        let reader = {
            let coordinator = Arc::clone(&coordinator);
            let indexing_done = Arc::clone(&indexing_done);
            let searches = Arc::clone(&searches);
            std::thread::spawn(move || {
                loop {
                    let result = coordinator
                        .search(LINE_ONE, vec![], &HybridSearchParams::default())
                        .expect("a search during indexing must still succeed");
                    // Whatever it sees, it must be internally consistent: never a
                    // result without provenance, never a NaN score.
                    for item in &result.results {
                        assert!(item.provenance.is_some());
                        assert!(item.fused_score.is_finite());
                    }
                    searches.fetch_add(1, Ordering::Relaxed);
                    if indexing_done.load(Ordering::Relaxed) {
                        break;
                    }
                    std::thread::yield_now();
                }
            })
        };

        let summary = coordinator.index_books(&books).unwrap().unwrap();
        indexing_done.store(true, Ordering::Relaxed);
        reader
            .join()
            .expect("the reader must not deadlock or panic");

        assert_eq!(summary.books_indexed, 40);
        assert_eq!(summary.chunks_written, 120);
        assert!(
            searches.load(Ordering::Relaxed) > 0,
            "at least one search must have completed alongside indexing"
        );
        assert_eq!(coordinator.status().vector_count, 120);
    }

    /// Releasing the engine lock between books must not mean committing the
    /// manifest between books. Every write serializes every record, so per-book
    /// saves are `O(B²)` bytes and `B` `fsync`s — invisible in a three-book test
    /// and fatal over a real library.
    #[test]
    fn indexing_through_the_coordinator_does_not_write_the_manifest_per_book() {
        let count_writes = |books: u64, name: &str| -> u32 {
            let dir = TempDir::new(name);
            let coordinator = semantic_coordinator(&dir);
            let library: Vec<BookForIndexing> = (0..books)
                .map(|i| {
                    let mut book = mock_book();
                    book.source_book_key = format!("otzaria/book{i:03}.txt");
                    book.content_fingerprint = i + 1;
                    book
                })
                .collect();

            coordinator.index_books(&library).unwrap().unwrap();
            manifest_save_count(&coordinator)
        };

        let few = count_writes(4, "coordinator_writes_few");
        let many = count_writes(40, "coordinator_writes_many");
        assert_eq!(
            few, many,
            "ten times the books must not cost ten times the manifest writes \
             (measured {few} and {many})"
        );
    }

    /// The volatile store makes intermediate manifest checkpoints actively
    /// misleading: after a crash their vectors are gone anyway. Even a long run
    /// therefore has one final manifest commit.
    #[test]
    fn a_long_indexing_run_writes_one_final_manifest() {
        let dir = TempDir::new("single_final_manifest");
        // Warm up first: loading the model records its identity, which is a
        // one-off write and would otherwise be counted below.
        let coordinator = indexed_coordinator(&dir);

        let library: Vec<BookForIndexing> = (1..=225)
            .map(|i| {
                let mut book = mock_book();
                book.source_book_key = format!("otzaria/book{i:04}.txt");
                book.content_fingerprint = i + 1;
                book
            })
            .collect();

        let before = manifest_save_count(&coordinator);
        coordinator.index_books(&library).unwrap().unwrap();
        let after = manifest_save_count(&coordinator);

        assert_eq!(
            after - before,
            1,
            "a whole-manifest checkpoint inside the batch would reintroduce superlinear I/O"
        );
    }

    #[test]
    fn a_batch_of_skipped_books_does_not_rewrite_the_manifest() {
        let dir = TempDir::new("skip_without_manifest_write");
        let coordinator = indexed_coordinator(&dir);
        let before = manifest_save_count(&coordinator);

        let summary = coordinator
            .index_books(&[mock_book(), mock_book()])
            .unwrap()
            .unwrap();

        assert_eq!(summary.books_skipped, 2);
        assert_eq!(manifest_save_count(&coordinator), before);
    }

    /// Two indexing runs must not interleave: they would drop and re-insert each
    /// other's books, and whichever finished last would commit the manifest.
    #[test]
    fn concurrent_indexing_runs_are_serialized() {
        use std::sync::Arc;

        let dir = TempDir::new("concurrent_indexing");
        let coordinator = Arc::new(semantic_coordinator(&dir));

        let batch = |prefix: &str, offset: u64| -> Vec<BookForIndexing> {
            (0..20)
                .map(|i| {
                    let mut book = mock_book();
                    book.source_book_key = format!("otzaria/{prefix}{i:02}.txt");
                    book.content_fingerprint = offset + i + 1;
                    book
                })
                .collect()
        };

        let first = batch("alpha", 0);
        let second = batch("beta", 1_000);

        let handle = {
            let coordinator = Arc::clone(&coordinator);
            std::thread::spawn(move || coordinator.index_books(&first).unwrap().unwrap())
        };
        let second_summary = coordinator.index_books(&second).unwrap().unwrap();
        let first_summary = handle.join().expect("indexing must not deadlock");

        assert_eq!(first_summary.books_indexed, 20);
        assert_eq!(second_summary.books_indexed, 20);
        // Both batches are present and complete: 40 books × 3 lines.
        let status = coordinator.status();
        assert_eq!(status.indexed_book_count, 40);
        assert_eq!(status.vector_count, 120);
    }

    #[test]
    fn reset_waits_for_an_active_indexing_lifecycle() {
        use std::sync::{mpsc, Arc};
        use std::time::Duration;

        let dir = TempDir::new("reset_serialization");
        let coordinator = Arc::new(indexed_coordinator(&dir));
        let indexing_guard = coordinator
            .indexing
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let (started_tx, started_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();

        let resetter = {
            let coordinator = Arc::clone(&coordinator);
            std::thread::spawn(move || {
                started_tx.send(()).unwrap();
                let result = coordinator.reset_semantic_index();
                done_tx.send(result).unwrap();
            })
        };

        started_rx.recv().unwrap();
        assert!(
            done_rx.recv_timeout(Duration::from_millis(50)).is_err(),
            "reset must wait on the same lifecycle mutex as indexing"
        );
        drop(indexing_guard);
        assert_eq!(
            done_rx
                .recv_timeout(Duration::from_secs(2))
                .expect("reset should continue after indexing releases the mutex")
                .unwrap(),
            Some(3)
        );
        resetter.join().unwrap();
    }

    #[test]
    fn indexing_an_empty_book_list_is_a_no_op() {
        let dir = TempDir::new("index_nothing");
        let coordinator = indexed_coordinator(&dir);

        let summary = coordinator.index_books(&[]).unwrap().unwrap();
        assert_eq!(summary.books_processed(), 0);
        assert_eq!(coordinator.status().vector_count, 3);

        // And with no engine at all it reports absence rather than success.
        assert!(HybridCoordinator::new(None)
            .index_books(&[])
            .unwrap()
            .is_none());
    }

    #[test]
    fn a_query_cache_hit_is_observable_and_recorded() {
        let coordinator = HybridCoordinator::new(None);
        let candidates = vec![lexical(1, LINE_ONE, 10.0)];

        let first = coordinator
            .search(LINE_ONE, candidates.clone(), &HybridSearchParams::default())
            .unwrap();
        let second = coordinator
            .search(LINE_ONE, candidates, &HybridSearchParams::default())
            .unwrap();

        assert!(!first.telemetry.unwrap().cache_hit);
        assert!(second.telemetry.unwrap().cache_hit);
        let snapshot = coordinator.get_telemetry_snapshot();
        assert_eq!(snapshot.total_searches, 2);
        assert_eq!(snapshot.cache_hits, 1);
        assert_eq!(snapshot.cache_misses, 1);
    }

    #[test]
    fn changed_lexical_inputs_cannot_reuse_a_cached_result() {
        let coordinator = HybridCoordinator::new(None);
        let first = coordinator
            .search(
                LINE_ONE,
                vec![lexical(1, LINE_ONE, 10.0)],
                &HybridSearchParams::default(),
            )
            .unwrap();
        let second = coordinator
            .search(
                LINE_ONE,
                vec![lexical(2, LINE_TWO, 9.0)],
                &HybridSearchParams::default(),
            )
            .unwrap();

        assert_eq!(first.results[0].id, 1);
        assert_eq!(second.results[0].id, 2);
        assert!(!second.telemetry.unwrap().cache_hit);
    }

    #[test]
    fn indexing_invalidates_cached_search_results() {
        let dir = TempDir::new("cache_invalidation");
        let coordinator = semantic_coordinator(&dir);
        let params = HybridSearchParams {
            force_mode: Some(SearchMode::LexicalOnly),
            ..Default::default()
        };
        let candidates = vec![lexical(1, LINE_ONE, 10.0)];

        coordinator
            .search(LINE_ONE, candidates.clone(), &params)
            .unwrap();
        assert!(
            coordinator
                .search(LINE_ONE, candidates.clone(), &params)
                .unwrap()
                .telemetry
                .unwrap()
                .cache_hit
        );

        coordinator.index_books(&[mock_book()]).unwrap().unwrap();
        assert!(
            !coordinator
                .search(LINE_ONE, candidates, &params)
                .unwrap()
                .telemetry
                .unwrap()
                .cache_hit
        );
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

    // ── cancellation ──

    /// What a caller can compare two results by: the ranked ids, where each came from, and
    /// the exact fused score.
    fn ranked(result: &HybridSearchResult) -> Vec<(u64, ResultSource, u32)> {
        result
            .results
            .iter()
            .map(|item| (item.id, item.source, item.fused_score.to_bits()))
            .collect()
    }

    /// A search per keystroke abandons most of its queries before they start. One already
    /// cancelled costs nothing, in every mode: neither cache is consulted or written, the
    /// query is not embedded, nothing is scanned and nothing is counted.
    #[test]
    fn a_search_cancelled_before_it_starts_costs_nothing_and_leaves_nothing() {
        use crate::cancellation::probe;

        let dir = TempDir::new("cancel_before_start");
        let coordinator = indexed_coordinator(&dir);
        let cancel = CancellationToken::new();
        cancel.cancel();

        for mode in [
            SearchMode::Hybrid,
            SearchMode::SemanticOnly,
            SearchMode::LexicalOnly,
        ] {
            let params = HybridSearchParams {
                force_mode: Some(mode),
                ..Default::default()
            };
            let (result, checkpoints) = probe::checkpoints_of(|| {
                coordinator.search_cancellable(
                    LINE_TWO,
                    vec![lexical(2, LINE_TWO, 9.0)],
                    &params,
                    &NoResolver,
                    &cancel,
                )
            });
            assert!(
                matches!(result, Err(SemanticSearchError::Cancelled)),
                "{mode}: {result:?}"
            );
            assert!(checkpoints.is_empty(), "{mode}: nothing may be scanned");
        }

        let embeddings = coordinator.embedding_cache_stats();
        assert_eq!(
            (embeddings.hits, embeddings.misses, embeddings.size),
            (0, 0, 0),
            "the embedding cache is looked at only just before embedding"
        );
        assert_eq!(coordinator.get_telemetry_snapshot().total_searches, 0);

        // Nor was a result cached: the same search, uncancelled, is computed afresh.
        let fresh = coordinator
            .search(
                LINE_TWO,
                vec![lexical(2, LINE_TWO, 9.0)],
                &HybridSearchParams::default(),
            )
            .unwrap();
        assert!(!fresh.telemetry.unwrap().cache_hit);
    }

    /// Cancelled from another thread once the scan has begun — the query embedded already
    /// — the search stops inside the scan. It is not served as a lexical fallback, and the
    /// embedding it paid for is not cached: a cancelled search writes nothing.
    #[test]
    fn a_search_cancelled_during_the_scan_is_neither_degraded_nor_cached() {
        use crate::cancellation::probe;

        let dir = TempDir::new("cancel_mid_scan");
        let coordinator = indexed_coordinator(&dir);
        let modes = [SearchMode::Hybrid, SearchMode::SemanticOnly];
        let params = |mode| HybridSearchParams {
            force_mode: Some(mode),
            ..Default::default()
        };

        for mode in modes {
            let cancel = CancellationToken::new();
            let (result, checkpoints) = probe::cancelling_at(&cancel, 0, || {
                coordinator.search_cancellable(
                    LINE_TWO,
                    vec![lexical(2, LINE_TWO, 9.0)],
                    &params(mode),
                    &NoResolver,
                    &cancel,
                )
            });
            assert!(
                matches!(result, Err(SemanticSearchError::Cancelled)),
                "{mode}: {result:?}"
            );
            assert_eq!(checkpoints, [0], "{mode}: the scan had started");
        }
        assert_eq!(
            coordinator.embedding_cache_stats().size,
            0,
            "a cancelled search must not cache the embedding it computed"
        );
        assert_eq!(coordinator.get_telemetry_snapshot().total_searches, 0);

        // Nothing was poisoned or half-written: the same searches now complete, and agree
        // with a coordinator that never saw the cancelled ones.
        let reference_dir = TempDir::new("cancel_mid_scan_reference");
        let reference = indexed_coordinator(&reference_dir);
        for mode in modes {
            let after = coordinator
                .search(LINE_TWO, vec![lexical(2, LINE_TWO, 9.0)], &params(mode))
                .unwrap();
            let expected = reference
                .search(LINE_TWO, vec![lexical(2, LINE_TWO, 9.0)], &params(mode))
                .unwrap();
            assert_eq!(after.search_mode, mode);
            assert!(!after.telemetry.as_ref().unwrap().cache_hit);
            assert_eq!(ranked(&after), ranked(&expected), "{mode}");
        }
        assert_eq!(
            coordinator.embedding_cache_stats().size,
            1,
            "an uncancelled search caches its embedding as it always did"
        );
    }

    /// A token nobody cancels is no token: the answer, the caches and the counts are those
    /// of [`HybridCoordinator::search`].
    #[test]
    fn a_token_nobody_cancels_changes_nothing() {
        use crate::cancellation::probe;

        let dir = TempDir::new("cancel_never");
        let coordinator = indexed_coordinator(&dir);
        let reference_dir = TempDir::new("cancel_never_reference");
        let reference = indexed_coordinator(&reference_dir);
        let candidates = vec![lexical(1, LINE_ONE, 10.0), lexical(2, LINE_TWO, 8.0)];

        let (with_token, checkpoints) = probe::checkpoints_of(|| {
            coordinator
                .search_cancellable(
                    LINE_ONE,
                    candidates.clone(),
                    &HybridSearchParams::default(),
                    &NoResolver,
                    &CancellationToken::new(),
                )
                .unwrap()
        });
        let without = reference
            .search(LINE_ONE, candidates, &HybridSearchParams::default())
            .unwrap();

        assert_eq!(checkpoints, [0], "the three-vector store is one interval");
        assert_eq!(with_token.search_mode, without.search_mode);
        assert_eq!(ranked(&with_token), ranked(&without));
        assert_eq!(
            coordinator.embedding_cache_stats().size,
            reference.embedding_cache_stats().size
        );
        assert_eq!(
            coordinator.get_telemetry_snapshot().total_searches,
            reference.get_telemetry_snapshot().total_searches
        );
    }

    // ── ranking parameters per search ──

    /// The fusion exactly as it stood before its numbers became parameters: the body of
    /// `fuse_candidates` at 1865ba0, verbatim, with `self`'s metadata ranker passed in and the
    /// two constants it read at the values they had.
    ///
    /// Frozen on purpose. It is what "the defaults reproduce the ranking" is measured against,
    /// so it must not follow the code it checks; a deliberate change to the default ranking —
    /// calibrated numbers, say — replaces it in the same commit.
    fn fuse_before_profile_parameters(
        metadata_ranker: &crate::hybrid::metadata_ranker::MetadataRanker,
        lexical: Vec<LexicalCandidate>,
        semantic: Vec<SemanticCandidate>,
        context: FusionContext<'_>,
    ) -> Vec<FusedCandidate> {
        const BM25_SATURATION_K: f32 = 10.0;
        const SEMANTIC_RELEVANCE_THRESHOLD: f32 = 0.55;

        let FusionContext {
            alpha,
            mode,
            profile,
            query_features,
            query_facets,
        } = context;
        let mut lexical_by_id: HashMap<(String, u64), (usize, LexicalCandidate)> = HashMap::new();
        for (rank, candidate) in lexical.into_iter().enumerate() {
            match lexical_by_id.entry((candidate.file_path.clone(), candidate.line_id)) {
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert((rank, candidate));
                }
                std::collections::hash_map::Entry::Occupied(mut entry) => {
                    if candidate
                        .bm25_score
                        .total_cmp(&entry.get().1.bm25_score)
                        .is_gt()
                    {
                        entry.insert((rank, candidate));
                    }
                }
            }
        }
        let mut lexical: Vec<(usize, LexicalCandidate)> = lexical_by_id.into_values().collect();
        lexical.sort_by_key(|(rank, _)| *rank);

        let mut semantic_by_id: HashMap<(String, u64), (usize, SemanticCandidate)> = HashMap::new();
        for (rank, candidate) in semantic.into_iter().enumerate() {
            let line = (
                candidate.metadata.source_book_key.clone(),
                candidate.metadata.line_id,
            );
            match semantic_by_id.entry(line) {
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert((rank, candidate));
                }
                std::collections::hash_map::Entry::Occupied(mut entry) => {
                    if candidate
                        .similarity_score
                        .total_cmp(&entry.get().1.similarity_score)
                        .is_gt()
                    {
                        entry.insert((rank, candidate));
                    }
                }
            }
        }
        let mut semantic: Vec<(usize, SemanticCandidate)> = semantic_by_id.into_values().collect();
        semantic.sort_by_key(|(rank, _)| *rank);

        let bm25_scores: Vec<f32> = lexical
            .iter()
            .map(|(_, candidate)| candidate.bm25_score)
            .collect();
        let norm_bm25 = match profile.fusion_strategy {
            FusionStrategy::Adaptive => normalize_bm25_adaptive(&bm25_scores, BM25_SATURATION_K),
            FusionStrategy::Weighted | FusionStrategy::RRF { .. } => {
                normalize_bm25_scores(&bm25_scores, BM25_SATURATION_K)
            }
        };

        let sem_scores: Vec<f32> = semantic
            .iter()
            .map(|(_, candidate)| candidate.similarity_score)
            .collect();
        let threshold = if profile.semantic_threshold.is_finite() {
            profile.semantic_threshold.clamp(0.0, 1.0)
        } else {
            SEMANTIC_RELEVANCE_THRESHOLD
        };
        let norm_sem = normalize_semantic_with_threshold(&sem_scores, threshold);
        let rrf_k = match profile.fusion_strategy {
            FusionStrategy::RRF { k } if mode == SearchMode::Hybrid => Some(k.max(1)),
            _ => None,
        };

        let mut fused_map: HashMap<(String, u64), FusedCandidate> =
            HashMap::with_capacity(lexical.len() + semantic.len());

        for ((_, candidate), &normalized) in lexical.into_iter().zip(norm_bm25.iter()) {
            let lexical_rank = fused_map.len() as u32 + 1;
            let fused_score = rrf_k
                .map(|k| 1.0 / (k as f32 + lexical_rank as f32))
                .unwrap_or(alpha * normalized);
            let line_id = candidate.line_id;

            fused_map.insert(
                (candidate.file_path.clone(), line_id),
                FusedCandidate {
                    title: candidate.title,
                    reference: candidate.reference,
                    text: candidate.text,
                    line_id,
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
                    fused_score,
                    lexical_weight: alpha,
                    semantic_weight: 1.0 - alpha,
                },
            );
        }

        for (semantic_rank, ((_, candidate), &normalized)) in
            semantic.into_iter().zip(norm_sem.iter()).enumerate()
        {
            // RRF ignores score magnitudes, so a threshold only has meaning if
            // candidates below it are excluded. Weighted/adaptive fusion keeps
            // them at zero to preserve semantic-only paging and grouping.
            if normalized <= 0.0 && rrf_k.is_some() {
                continue;
            }
            let line_id = candidate.metadata.line_id;
            let contribution = rrf_k
                .map(|k| 1.0 / (k as f32 + semantic_rank as f32 + 1.0))
                .unwrap_or((1.0 - alpha) * normalized);
            let metadata_bonus = if profile.metadata_ranking_enabled && rrf_k.is_none() {
                metadata_ranker
                    .compute_signal(
                        &candidate.metadata.source_book_key,
                        &candidate.metadata.facets,
                        query_facets,
                    )
                    .total
            } else {
                0.0
            };

            match fused_map.get_mut(&(candidate.metadata.source_book_key.clone(), line_id)) {
                // Found by both engines: keep the lexical text and record both
                // scores. Provenance must survive fusion.
                Some(existing) => {
                    existing.source = ResultSource::Both;
                    existing.raw_semantic_score = Some(candidate.similarity_score);
                    existing.normalized_semantic = Some(normalized);
                    existing.fused_score += contribution + metadata_bonus;
                    if mode == SearchMode::Hybrid && rrf_k.is_none() {
                        existing.fused_score += profile.agreement_bonus.max(0.0);
                    }
                }
                // Semantic-only: the vector store holds metadata but no line
                // body, so the text has to be hydrated from Tantivy by id.
                None => {
                    let metadata = candidate.metadata;
                    fused_map.insert(
                        (metadata.source_book_key.clone(), line_id),
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
                            fused_score: contribution + metadata_bonus,
                            lexical_weight: alpha,
                            semantic_weight: 1.0 - alpha,
                        },
                    );
                }
            }
        }

        let mut results: Vec<FusedCandidate> = fused_map.into_values().collect();
        if rrf_k.is_none() {
            let mut section_counts: HashMap<(String, u64), usize> = HashMap::new();
            for candidate in &results {
                *section_counts
                    .entry((candidate.file_path.clone(), candidate.section_id))
                    .or_default() += 1;
            }

            for candidate in &mut results {
                candidate.fused_score += profile.phrase_match_bonus.max(0.0)
                    * compute_phrase_match_bonus(&candidate.text, &query_features.quoted_phrases);
                candidate.fused_score += profile.rare_term_bonus.max(0.0)
                    * compute_rare_term_bonus(&candidate.text, &query_features.rare_tokens);
                if section_counts
                    .get(&(candidate.file_path.clone(), candidate.section_id))
                    .copied()
                    .unwrap_or(0)
                    > 1
                {
                    candidate.fused_score += profile.section_coverage_bonus.max(0.0);
                }
            }
        }

        // Ties break on the line, then its book, so pagination is stable across calls;
        // `HashMap` iteration order is not.
        let sort_results = |results: &mut Vec<FusedCandidate>| {
            results.sort_by(|a, b| {
                b.fused_score
                    .total_cmp(&a.fused_score)
                    .then_with(|| a.line_id.cmp(&b.line_id))
                    .then_with(|| a.file_path.cmp(&b.file_path))
            });
        };
        sort_results(&mut results);

        if rrf_k.is_none() && profile.duplicate_penalty > 0.0 {
            let mut seen = std::collections::HashSet::new();
            for candidate in &mut results {
                if candidate.line_hash != 0 && !seen.insert(candidate.line_hash) {
                    candidate.fused_score -= profile.duplicate_penalty;
                }
            }
            sort_results(&mut results);
        }
        results
    }

    /// The alpha a search asked for before the table was a parameter: `compute_alpha` as it
    /// was, then the same clamp.
    fn alpha_before_profile_parameters(profile: &RankingProfile, features: &QueryFeatures) -> f32 {
        use crate::hybrid::ranking::QueryType;
        profile
            .alpha_override
            .unwrap_or(match features.estimated_type {
                QueryType::ExactReference if features.has_quoted_phrase => 1.0,
                QueryType::ExactReference => 0.85,
                QueryType::Short => 0.7,
                QueryType::Mixed => 0.5,
                QueryType::Conceptual => 0.3,
                QueryType::Unknown => 0.5,
            })
            .clamp(0.0, 1.0)
    }

    /// Every field of every fused candidate, each float as its bits.
    fn fused_bits(fused: &[FusedCandidate]) -> Vec<String> {
        let bits = |value: Option<f32>| {
            value.map_or("-".to_string(), |value| format!("{:08x}", value.to_bits()))
        };
        fused
            .iter()
            .map(|c| {
                format!(
                    "{} {:?} f={} rb={} nb={} rs={} ns={} lw={} sw={} h={} | {:?} {:?} {:?} {} {} {} {} {:?}",
                    c.line_id,
                    c.source,
                    bits(Some(c.fused_score)),
                    bits(c.raw_bm25_score),
                    bits(c.normalized_bm25),
                    bits(c.raw_semantic_score),
                    bits(c.normalized_semantic),
                    bits(Some(c.lexical_weight)),
                    bits(Some(c.semantic_weight)),
                    c.needs_hydration,
                    c.title,
                    c.reference,
                    c.text,
                    c.section_id,
                    c.line_hash,
                    c.segment,
                    c.is_pdf,
                    c.file_path
                )
            })
            .collect()
    }

    fn semantic_hit(
        line_id: u64,
        similarity_score: f32,
        book: &str,
        section_id: u64,
        line_hash: u64,
    ) -> SemanticCandidate {
        SemanticCandidate {
            metadata: crate::semantic::types::VectorMetadata {
                semantic_id: format!("{book}#{line_id}"),
                source_book_key: book.to_string(),
                source_doc_key: format!("{book}#{line_id}"),
                line_id,
                section_id,
                line_hash,
                chunk_hash: String::new(),
                content_hash: 0,
                reference: format!("הפניה {line_id}"),
                segment: line_id,
                is_pdf: false,
                title: "ספר".to_string(),
                facets: vec!["/מקרא/תורה".to_string(), "/era/תנך".to_string()],
            },
            similarity_score,
        }
    }

    /// A resolver may return a hit's lines after a worse hit's — the application's appends
    /// those of text that moved books after all the others — and a candidate's position is
    /// its rank to a fusion by rank. So the candidates come in the scan's order whatever the
    /// resolver's, each hit's lines in the order it gave them.
    #[test]
    fn semantic_candidates_follow_the_scan_whatever_order_the_resolver_returns() {
        let hit = |n: u32, score: f32| VectorHit {
            score,
            key: crate::semantic::chunk_key::ChunkKey::of(&format!("[PASSAGE] {n}")),
            records: Vec::new(),
            seg: 0,
            slot: n,
        };
        let hits = [hit(0, 0.9), hit(1, 0.8), hit(2, 0.7), hit(3, 0.6)];
        let line = |hit: u32, book: &str, line_id: u64| ResolvedLine {
            hit,
            line_id,
            file_path: book.to_string(),
            section_id: 0,
            line_hash: 0,
            segment: line_id,
            is_pdf: false,
            facets: Vec::<String>::new().into(),
            title: String::new(),
            reference: String::new(),
        };
        // Hit 0's second line arrives last, as moved text does; hit 3 resolves nowhere.
        let lines = vec![
            line(0, "b.txt", 5),
            line(1, "a.txt", 9),
            line(2, "c.txt", 3),
            line(0, "a.txt", 7),
        ];
        let (candidates, unresolved) = candidates_of(&hits, lines);
        let order: Vec<(&str, u64)> = candidates
            .iter()
            .map(|candidate| {
                (
                    candidate.metadata.source_book_key.as_str(),
                    candidate.metadata.line_id,
                )
            })
            .collect();
        assert_eq!(
            order,
            [("b.txt", 5), ("a.txt", 7), ("a.txt", 9), ("c.txt", 3)]
        );
        assert_eq!(unresolved, 1);
    }

    /// Two books whose lines share an id are two lines, in every mode that fuses. An index
    /// updated book by book can give two books the same ids, and a merge on the id alone
    /// fused one book's lexical hit with the other book's semantic one — a result shown
    /// under one book with the other's score.
    #[test]
    fn two_books_sharing_a_line_id_are_never_merged() {
        let coordinator = HybridCoordinator::new(None);
        let genesis = "otzaria/tanach/genesis.txt";
        let berachot = "otzaria/mishna/berachot.txt";
        let profile = RankingProfile::from_profile(SearchProfile::Balanced);
        let features = analyze_query(LINE_ONE);
        let facets: [String; 0] = [];
        for (mode, alpha) in [(SearchMode::Hybrid, 0.3), (SearchMode::SemanticOnly, 0.0)] {
            let fused = coordinator.fuse_candidates(
                vec![lexical(7, LINE_ONE, 12.0)],
                vec![
                    semantic_hit(7, 0.9, berachot, 3, 70),
                    semantic_hit(7, 0.8, genesis, 100, 77777),
                ],
                FusionContext {
                    alpha,
                    mode,
                    profile: &profile,
                    query_features: &features,
                    query_facets: &facets,
                },
            );
            let line = |book: &str| {
                fused
                    .iter()
                    .find(|candidate| candidate.file_path == book && candidate.line_id == 7)
                    .unwrap_or_else(|| panic!("{mode}: line 7 of {book} is a result of its own"))
            };
            assert_eq!(line(berachot).raw_bm25_score, None, "{mode}");
            assert_eq!(line(berachot).raw_semantic_score, Some(0.9), "{mode}");
            assert_eq!(line(berachot).section_id, 3, "{mode}");
            assert_eq!(line(genesis).raw_semantic_score, Some(0.8), "{mode}");
            if mode == SearchMode::Hybrid {
                assert_eq!(fused.len(), 2);
                assert_eq!(line(genesis).raw_bm25_score, Some(12.0));
            }
        }
    }

    /// The defaults reproduce the ranking this crate produced before they were parameters,
    /// bit for bit. Over every preset; every fusion strategy, and RRF's `k` of 0, which the
    /// fusion still treats as 1; a threshold that is not a number and thresholds either side
    /// of the candidates; every mode at the alphas it runs with; and candidates that reach
    /// the edges — the same line twice on either side, NaN and infinite BM25 scores, negative
    /// and NaN cosines, line hashes shared for the duplicate penalty, facets for the metadata
    /// bonus. And the alpha each preset asks for, for every query type.
    #[test]
    fn the_default_parameters_rank_exactly_as_the_constants_did() {
        let coordinator = HybridCoordinator::new(None);
        let genesis = "otzaria/tanach/genesis.txt";
        let semantic_sets: [Vec<SemanticCandidate>; 2] = [
            vec![],
            vec![
                semantic_hit(1, 0.93, genesis, 100, 11111),
                semantic_hit(2, 0.41, genesis, 100, 22222),
                semantic_hit(5, 0.12, "otzaria/b.txt", 7, 11111),
                semantic_hit(6, -0.3, "otzaria/b.txt", 7, 0),
                semantic_hit(7, f32::NAN, "otzaria/b.txt", 8, 9),
                semantic_hit(1, 0.95, genesis, 100, 11111),
                semantic_hit(8, 0.1, "otzaria/b.txt", 8, 10),
                semantic_hit(9, 1.0, "otzaria/c.txt", 1, 12),
                semantic_hit(10, 0.55, "otzaria/c.txt", 1, 13),
            ],
        ];
        let lexical_sets: [Vec<LexicalCandidate>; 3] = [
            vec![],
            vec![
                lexical(1, LINE_ONE, 45.0),
                lexical(2, LINE_TWO, 25.0),
                lexical(3, LINE_THREE, 0.0),
                lexical(4, "בראשית ברא", f32::INFINITY),
                lexical(11, "שורה", f32::NAN),
                lexical(2, LINE_TWO, 25.0),
                lexical(12, "שורה אחרת", 4.0),
            ],
            vec![lexical(1, LINE_ONE, 2.0), lexical(3, LINE_THREE, 9.0)],
        ];
        let strategies = [
            None,
            Some(FusionStrategy::Weighted),
            Some(FusionStrategy::RRF { k: 0 }),
            Some(FusionStrategy::RRF { k: 5 }),
            Some(FusionStrategy::Adaptive),
        ];
        let queries = ["\"בראשית ברא\" אלהים", "בראשית ברא אלהים את", "מה"];
        let facets = ["/era/תנך".to_string()];

        let mut compared = 0;
        for preset in [
            SearchProfile::Fast,
            SearchProfile::Balanced,
            SearchProfile::Best,
        ] {
            for threshold in [None, Some(f32::NAN), Some(0.7), Some(0.0)] {
                for strategy in strategies {
                    let mut profile = RankingProfile::from_profile(preset);
                    if let Some(threshold) = threshold {
                        profile.semantic_threshold = threshold;
                    }
                    if let Some(strategy) = strategy {
                        profile.fusion_strategy = strategy;
                    }
                    for query in queries {
                        let features = analyze_query(query);
                        for (mode, alpha) in [
                            (SearchMode::Hybrid, 0.3),
                            (SearchMode::Hybrid, 0.85),
                            (SearchMode::SemanticOnly, 0.0),
                            (SearchMode::LexicalOnly, 1.0),
                        ] {
                            for lexical in &lexical_sets {
                                for semantic in &semantic_sets {
                                    let context = || FusionContext {
                                        alpha,
                                        mode,
                                        profile: &profile,
                                        query_features: &features,
                                        query_facets: &facets,
                                    };
                                    let now = coordinator.fuse_candidates(
                                        lexical.clone(),
                                        semantic.clone(),
                                        context(),
                                    );
                                    let before = fuse_before_profile_parameters(
                                        coordinator.metadata_ranker(),
                                        lexical.clone(),
                                        semantic.clone(),
                                        context(),
                                    );
                                    assert_eq!(
                                        fused_bits(&now),
                                        fused_bits(&before),
                                        "{preset} {threshold:?} {strategy:?} {query:?} {mode} {alpha}"
                                    );
                                    compared += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(compared, 3 * 4 * 5 * 3 * 4 * 3 * 2);

        for preset in [
            SearchProfile::Fast,
            SearchProfile::Balanced,
            SearchProfile::Best,
        ] {
            let profile = RankingProfile::from_profile(preset);
            for query in [
                "\"בראשית ברא\"",
                "ברכות 20",
                "שלום עולם",
                "ברכות דף כ",
                LINE_ONE,
                "",
            ] {
                let features = analyze_query(query);
                let now = profile
                    .alpha_override
                    .unwrap_or_else(|| compute_alpha_with(&features, &profile.alpha_by_query_type))
                    .clamp(0.0, 1.0);
                assert_eq!(
                    now.to_bits(),
                    alpha_before_profile_parameters(&profile, &features).to_bits(),
                    "{preset} {query:?}"
                );
            }
        }
    }

    /// Passing a preset as a ranking profile is naming the preset: the same results, bit for
    /// bit, in the same mode, at the same alpha, in every mode.
    #[test]
    fn a_preset_passed_as_a_ranking_profile_ranks_exactly_as_the_preset() {
        let dir = TempDir::new("preset_as_ranking");
        let coordinator = indexed_coordinator(&dir);
        let lexical_sets = [
            vec![lexical(1, LINE_ONE, 15.5)],
            vec![
                lexical(1, LINE_ONE, 2.0),
                lexical(2, LINE_TWO, 30.0),
                lexical(3, LINE_THREE, 9.0),
            ],
        ];

        for preset in [
            SearchProfile::Fast,
            SearchProfile::Balanced,
            SearchProfile::Best,
        ] {
            for mode in [
                SearchMode::Hybrid,
                SearchMode::SemanticOnly,
                SearchMode::LexicalOnly,
            ] {
                for query in [LINE_ONE, "בראשית ברא", "\"בראשית ברא\"", "ויאמר אלהים יהי"]
                {
                    for candidates in &lexical_sets {
                        let run = |params: HybridSearchParams| {
                            coordinator.clear_query_cache();
                            coordinator
                                .search(query, candidates.clone(), &params)
                                .unwrap()
                        };
                        let named = run(HybridSearchParams {
                            force_mode: Some(mode),
                            profile: Some(preset),
                            ..Default::default()
                        });
                        let passed = run(HybridSearchParams {
                            force_mode: Some(mode),
                            ranking: Some(RankingProfile::from_profile(preset)),
                            ..Default::default()
                        });

                        let case = format!("{preset} {mode} {query:?}");
                        assert_eq!(passed.search_mode, named.search_mode, "{case}");
                        assert_eq!(ranked(&passed), ranked(&named), "{case}");
                        assert_eq!(passed.total_count, named.total_count, "{case}");
                        let (passed, named) = (passed.telemetry.unwrap(), named.telemetry.unwrap());
                        assert_eq!(passed.alpha.to_bits(), named.alpha.to_bits(), "{case}");
                        assert_eq!(passed.fusion_strategy, named.fusion_strategy, "{case}");
                        assert_eq!(passed.profile, named.profile, "{case}");
                    }
                }
            }
        }
    }

    /// Each parameter passed with a search is the one that search ranks by.
    #[test]
    fn every_parameter_passed_with_a_search_is_the_one_it_ranks_by() {
        use crate::config::profiles::QueryTypeAlphas;

        let dir = TempDir::new("ranking_takes_effect");
        let coordinator = indexed_coordinator(&dir);
        let search = |query: &str, candidates: Vec<LexicalCandidate>, mode, ranking| {
            coordinator
                .search(
                    query,
                    candidates,
                    &HybridSearchParams {
                        force_mode: Some(mode),
                        ranking: Some(ranking),
                        ..Default::default()
                    },
                )
                .unwrap()
        };
        let balanced = RankingProfile::default();
        let score = |result: &HybridSearchResult, id: u64| {
            result
                .results
                .iter()
                .find(|item| item.id == id)
                .expect("the line is in the result")
                .fused_score
        };

        // The fusion strategy and RRF's `k`: line 1 tops both sides, so it gets 1 / (k + 1)
        // from each.
        for k in [5, 60] {
            let result = search(
                LINE_ONE,
                vec![lexical(1, LINE_ONE, 15.5)],
                SearchMode::Hybrid,
                RankingProfile {
                    fusion_strategy: FusionStrategy::RRF { k },
                    ..balanced.clone()
                },
            );
            let expected = 2.0 / (k as f32 + 1.0);
            assert!((score(&result, 1) - expected).abs() < 1e-6, "k = {k}");
            assert_eq!(
                result.telemetry.unwrap().fusion_strategy,
                format!("RRF(k={k})")
            );
        }

        // One fixed alpha, and alpha per query type: the second line is conceptual.
        let fixed = search(
            LINE_TWO,
            vec![lexical(2, LINE_TWO, 9.0)],
            SearchMode::Hybrid,
            RankingProfile {
                alpha_override: Some(0.25),
                ..balanced.clone()
            },
        );
        assert_eq!(fixed.telemetry.unwrap().alpha, 0.25);
        let per_type = search(
            LINE_TWO,
            vec![lexical(2, LINE_TWO, 9.0)],
            SearchMode::Hybrid,
            RankingProfile {
                alpha_by_query_type: QueryTypeAlphas {
                    conceptual: 0.9,
                    ..QueryTypeAlphas::default()
                },
                ..balanced.clone()
            },
        );
        assert_eq!(per_type.telemetry.unwrap().alpha, 0.9);

        // A quoted phrase skips the semantic path at the default alpha of 1.0 only.
        let quoted = "\"בראשית ברא\"";
        let skipped = search(
            quoted,
            vec![lexical(1, LINE_ONE, 15.5)],
            SearchMode::Hybrid,
            balanced.clone(),
        );
        assert_eq!(skipped.search_mode, SearchMode::LexicalOnly);
        let consulted = search(
            quoted,
            vec![lexical(1, LINE_ONE, 15.5)],
            SearchMode::Hybrid,
            RankingProfile {
                alpha_by_query_type: QueryTypeAlphas {
                    quoted_phrase: 0.6,
                    ..QueryTypeAlphas::default()
                },
                ..balanced.clone()
            },
        );
        assert_eq!(consulted.search_mode, SearchMode::Hybrid);
        assert_eq!(consulted.telemetry.unwrap().alpha, 0.6);

        // BM25's `k`: the lexical score is normalized against it.
        for k in [2.0, 10.0, 50.0] {
            let result = search(
                LINE_ONE,
                vec![lexical(1, LINE_ONE, 15.5)],
                SearchMode::LexicalOnly,
                RankingProfile {
                    bm25_saturation_k: k,
                    ..balanced.clone()
                },
            );
            let provenance = result.results[0].provenance.as_ref().unwrap();
            assert_eq!(
                provenance.normalized_bm25,
                Some(crate::hybrid::fusion::normalize_bm25_scores(&[15.5], k)[0]),
                "k = {k}"
            );
        }

        // The semantic threshold: a semantic score below it contributes nothing.
        let normalized = |threshold: f32| {
            search(
                LINE_ONE,
                vec![],
                SearchMode::SemanticOnly,
                RankingProfile {
                    semantic_threshold: threshold,
                    ..balanced.clone()
                },
            )
            .results
            .iter()
            .map(|item| {
                (
                    item.id,
                    item.provenance
                        .as_ref()
                        .unwrap()
                        .normalized_semantic
                        .unwrap(),
                )
            })
            .collect::<Vec<_>>()
        };
        let open = normalized(0.0);
        let strict = normalized(0.9);
        assert!(
            open.iter().any(|&(_, score)| score > 0.0 && score < 0.9),
            "the fixture needs a semantic score between the two thresholds: {open:?}"
        );
        assert!(
            strict
                .iter()
                .all(|&(_, score)| score == 0.0 || score >= 0.9),
            "{strict:?}"
        );
        assert!(strict.iter().any(|&(id, score)| id == 1 && score >= 0.9));

        // The agreement bonus: added to the line both sides found, and to no other.
        let with_bonus = |bonus: f32| {
            search(
                LINE_ONE,
                vec![
                    lexical(1, LINE_ONE, 15.5),
                    lexical(999, "שורה לקסיקלית בלבד", 3.0),
                ],
                SearchMode::Hybrid,
                RankingProfile {
                    agreement_bonus: bonus,
                    ..balanced.clone()
                },
            )
        };
        let (without, with) = (with_bonus(0.0), with_bonus(0.3));
        assert!(((score(&with, 1) - score(&without, 1)) - 0.3).abs() < 1e-6);
        assert_eq!(score(&with, 999), score(&without, 999));
    }

    /// A parameter out of its range fails the search before it runs — named, never clamped
    /// — and the search leaves nothing behind.
    #[test]
    fn a_ranking_parameter_out_of_range_fails_the_search_and_names_it() {
        let dir = TempDir::new("ranking_refused");
        let coordinator = indexed_coordinator(&dir);
        let params = |ranking| HybridSearchParams {
            ranking: Some(ranking),
            ..Default::default()
        };

        let refused = coordinator.search(
            LINE_TWO,
            vec![lexical(2, LINE_TWO, 9.0)],
            &params(RankingProfile {
                bm25_saturation_k: -1.0,
                ..RankingProfile::default()
            }),
        );
        match refused {
            Err(SemanticSearchError::InvalidRankingParameter {
                parameter: "bm25_saturation_k",
                value,
                ..
            }) => assert_eq!(value, "-1"),
            other => panic!("expected the parameter named, got {other:?}"),
        }
        let embeddings = coordinator.embedding_cache_stats();
        assert_eq!((embeddings.misses, embeddings.size), (0, 0));
        assert_eq!(coordinator.get_telemetry_snapshot().total_searches, 0);

        // A value the feature flags would have clamped is refused here, not clamped.
        assert!(matches!(
            coordinator.search(
                LINE_TWO,
                vec![],
                &params(RankingProfile {
                    semantic_threshold: 1.5,
                    ..RankingProfile::default()
                }),
            ),
            Err(SemanticSearchError::InvalidRankingParameter {
                parameter: "semantic_threshold",
                ..
            })
        ));

        // And a valid profile runs.
        assert!(coordinator
            .search(LINE_TWO, vec![], &params(RankingProfile::default()))
            .is_ok());
    }

    /// Results ranked by different parameters are different results: the query cache keys on
    /// the whole profile, so one ranking never answers for another.
    #[test]
    fn the_query_cache_never_answers_one_ranking_with_another() {
        let coordinator = HybridCoordinator::new(None);
        let candidates = vec![lexical(1, LINE_ONE, 10.0)];
        let ranked_by = |k: f32| {
            coordinator
                .search(
                    LINE_ONE,
                    candidates.clone(),
                    &HybridSearchParams {
                        ranking: Some(RankingProfile {
                            bm25_saturation_k: k,
                            ..RankingProfile::default()
                        }),
                        ..Default::default()
                    },
                )
                .unwrap()
        };

        let first = ranked_by(10.0);
        let other = ranked_by(2.0);
        assert!(!other.telemetry.as_ref().unwrap().cache_hit);
        assert_ne!(first.results[0].fused_score, other.results[0].fused_score);
        assert!(ranked_by(10.0).telemetry.unwrap().cache_hit);
    }
}
