//! Hybrid search coordinator.
//!
//! Merges results from lexical (Tantivy/BM25) and semantic (embedding/zvec)
//! search paths into a single unified result set with dynamic ranking,
//! score normalization, and post-fusion grouping.

use crate::errors::SemanticSearchError;
use crate::hybrid::fusion::{normalize_bm25_scores, normalize_semantic_scores};
use crate::hybrid::grouping::group_results;
use crate::hybrid::ranking::{analyze_query, compute_alpha, BonusConfig};
use crate::semantic::engine::SemanticEngine;
use crate::semantic::types::{
    FusedCandidate, GroupingMode, HybridMergedSibling, HybridResultItem, HybridSearchResult,
    LexicalCandidate, ResultSource, SearchFilters, SearchMode, SemanticCandidate, SemanticStatus,
};
use std::collections::HashMap;
use std::sync::RwLock;

/// Configuration for hybrid search execution.
#[derive(Debug, Clone)]
pub struct HybridSearchParams {
    pub limit: usize,
    pub offset: usize,
    pub grouping: Option<GroupingMode>,
    pub filters: Option<SearchFilters>,
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

/// Main Hybrid Search Coordinator.
pub struct HybridCoordinator {
    semantic_engine: RwLock<Option<SemanticEngine>>,
    bonus_config: BonusConfig,
}

impl HybridCoordinator {
    /// Create a new HybridCoordinator.
    pub fn new(semantic_engine: Option<SemanticEngine>) -> Self {
        Self {
            semantic_engine: RwLock::new(semantic_engine),
            bonus_config: BonusConfig::default(),
        }
    }

    /// Primary search entry point. Coordinates BM25 & Semantic candidates.
    pub fn search(
        &self,
        query: &str,
        lexical_candidates: Vec<LexicalCandidate>,
        params: &HybridSearchParams,
    ) -> Result<HybridSearchResult, SemanticSearchError> {
        let start_time = std::time::Instant::now();

        let semantic_lock = self.semantic_engine.read().unwrap();
        let (semantic_candidates, mode) = match (params.force_mode, semantic_lock.as_ref()) {
            (Some(SearchMode::LexicalOnly), _) | (_, None) => (Vec::new(), SearchMode::LexicalOnly),
            (_, Some(engine)) => {
                match engine.search(query, params.limit * 3, params.filters.as_ref()) {
                    Ok(cands) => (cands, SearchMode::Hybrid),
                    Err(e) => {
                        log::warn!("Semantic search path failed: {e}. Falling back to BM25.");
                        (Vec::new(), SearchMode::LexicalOnly)
                    }
                }
            }
        };

        let features = analyze_query(query);
        let alpha = compute_alpha(&features);

        let fused = self.fuse_candidates(lexical_candidates, semantic_candidates, alpha);

        let (final_items, total_count, group_count) = if let Some(g_mode) = params.grouping {
            let grouped = group_results(fused, g_mode);
            let g_count = grouped.len() as u32;
            let total = grouped.iter().map(|g| g.group_count).sum::<u32>();

            let paginated = grouped.into_iter().skip(params.offset).take(params.limit);

            let items = paginated
                .map(|g| HybridResultItem {
                    title: g.representative.title.clone(),
                    reference: g.representative.reference.clone(),
                    text: g.representative.text.clone(),
                    id: g.representative.line_id,
                    segment: g.representative.segment,
                    is_pdf: g.representative.is_pdf,
                    file_path: g.representative.file_path.clone(),
                    merged_count: g.group_count,
                    merged: g
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
                        .collect(),
                    lexical_score: g.representative.raw_bm25_score,
                    semantic_score: g.representative.raw_semantic_score,
                    fused_score: g.representative.fused_score,
                    source: g.representative.source,
                    provenance: Some(g.representative),
                })
                .collect();

            (items, total, Some(g_count))
        } else {
            let total = fused.len() as u32;
            let paginated = fused.into_iter().skip(params.offset).take(params.limit);

            let items = paginated
                .map(|c| HybridResultItem {
                    title: c.title.clone(),
                    reference: c.reference.clone(),
                    text: c.text.clone(),
                    id: c.line_id,
                    segment: c.segment,
                    is_pdf: c.is_pdf,
                    file_path: c.file_path.clone(),
                    merged_count: 1,
                    merged: Vec::new(),
                    lexical_score: c.raw_bm25_score,
                    semantic_score: c.raw_semantic_score,
                    fused_score: c.fused_score,
                    source: c.source,
                    provenance: Some(c),
                })
                .collect();

            (items, total, None)
        };

        let latency_ms = start_time.elapsed().as_millis() as u64;

        Ok(HybridSearchResult {
            results: final_items,
            total_count,
            group_count,
            search_mode: mode,
            semantic_available: semantic_lock.is_some(),
            latency_ms,
        })
    }

    /// Fuse lexical and semantic candidates.
    fn fuse_candidates(
        &self,
        lexical: Vec<LexicalCandidate>,
        semantic: Vec<SemanticCandidate>,
        alpha: f32,
    ) -> Vec<FusedCandidate> {
        let bm25_scores: Vec<f32> = lexical.iter().map(|l| l.bm25_score).collect();
        let norm_bm25 = normalize_bm25_scores(&bm25_scores, 1.0);

        let sem_scores: Vec<f32> = semantic.iter().map(|s| s.similarity_score).collect();
        let norm_sem = normalize_semantic_scores(&sem_scores);

        let mut fused_map: HashMap<u64, FusedCandidate> = HashMap::new();

        // Add lexical candidates
        for (cand, &nbm25) in lexical.into_iter().zip(norm_bm25.iter()) {
            let item = FusedCandidate {
                title: cand.title,
                reference: cand.reference,
                text: cand.text,
                line_id: cand.line_id,
                section_id: cand.section_id,
                line_hash: cand.line_hash,
                segment: cand.segment,
                is_pdf: cand.is_pdf,
                file_path: cand.file_path,
                source: ResultSource::Lexical,
                raw_bm25_score: Some(cand.bm25_score),
                normalized_bm25: Some(nbm25),
                raw_semantic_score: None,
                normalized_semantic: None,
                fused_score: alpha * nbm25,
                lexical_weight: alpha,
                semantic_weight: 1.0 - alpha,
            };
            fused_map.insert(item.line_id, item);
        }

        // Merge semantic candidates
        for (cand, &nsem) in semantic.into_iter().zip(norm_sem.iter()) {
            let line_id = cand.metadata.line_id;
            let sem_contrib = (1.0 - alpha) * nsem;

            if let Some(existing) = fused_map.get_mut(&line_id) {
                existing.source = ResultSource::Both;
                existing.raw_semantic_score = Some(cand.similarity_score);
                existing.normalized_semantic = Some(nsem);
                existing.fused_score += sem_contrib + self.bonus_config.exact_match_bonus;
            } else {
                let item = FusedCandidate {
                    title: cand.metadata.title,
                    reference: cand.metadata.reference,
                    text: String::new(),
                    line_id,
                    section_id: cand.metadata.section_id,
                    line_hash: cand.metadata.line_hash,
                    segment: cand.metadata.segment,
                    is_pdf: cand.metadata.is_pdf,
                    file_path: cand.metadata.source_book_key,
                    source: ResultSource::Semantic,
                    raw_bm25_score: None,
                    normalized_bm25: None,
                    raw_semantic_score: Some(cand.similarity_score),
                    normalized_semantic: Some(nsem),
                    fused_score: sem_contrib,
                    lexical_weight: alpha,
                    semantic_weight: 1.0 - alpha,
                };
                fused_map.insert(line_id, item);
            }
        }

        let mut results: Vec<FusedCandidate> = fused_map.into_values().collect();
        results.sort_by(|a, b| {
            b.fused_score
                .partial_cmp(&a.fused_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results
    }

    /// Retrieve semantic engine status.
    pub fn status(&self) -> SemanticStatus {
        let lock = self.semantic_engine.read().unwrap();
        match lock.as_ref() {
            Some(engine) => engine.status(),
            None => SemanticStatus {
                available: false,
                model_loaded: false,
                indexed_book_count: 0,
                vector_count: 0,
                model_id: "none".to_string(),
                embedding_dim: 0,
                last_error: Some("Semantic engine disabled".to_string()),
            },
        }
    }
}
