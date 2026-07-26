//! Clean high-level API for Flutter / flutter_rust_bridge.
//!
//! Provides Flutter with domain-level operations for hybrid search and semantic index
//! lifecycle management.

use crate::hybrid::coordinator::{HybridCoordinator, HybridSearchParams};
use crate::semantic::engine::SemanticEngine;
use crate::semantic::types::{
    HybridSearchResult, IndexDiff, LexicalCandidate, SearchFilters, SemanticStatus,
};
use std::collections::HashMap;
use std::sync::Arc;

/// Parameters for a hybrid search API call.
/// Groups all optional parameters to avoid too_many_arguments clippy lint.
#[derive(Debug, Clone, Default)]
pub struct SearchRequest {
    pub query: String,
    pub lexical_candidates: Vec<LexicalCandidate>,
    pub limit: Option<u32>,
    pub offset: Option<u32>,
    pub grouping: Option<crate::semantic::types::GroupingMode>,
    pub filters: Option<SearchFilters>,
    pub force_mode: Option<crate::semantic::types::SearchMode>,
}

/// Opaque wrapper for HybridCoordinator handle.
pub struct OtzariaHybridEngine {
    coordinator: Arc<HybridCoordinator>,
}

impl OtzariaHybridEngine {
    /// Initialize hybrid search engine with optional semantic engine.
    pub fn new(coordinator: HybridCoordinator) -> Self {
        Self {
            coordinator: Arc::new(coordinator),
        }
    }

    /// Perform a hybrid search combining BM25 candidates and semantic vectors.
    pub fn search(&self, request: SearchRequest) -> Result<HybridSearchResult, String> {
        let params = HybridSearchParams {
            limit: request.limit.unwrap_or(20) as usize,
            offset: request.offset.unwrap_or(0) as usize,
            grouping: request.grouping,
            filters: request.filters,
            force_mode: request.force_mode,
        };

        self.coordinator
            .search(&request.query, request.lexical_candidates, &params)
            .map_err(|e| e.to_string())
    }

    /// Query current status of the semantic search sidecar.
    pub fn get_semantic_status(&self) -> SemanticStatus {
        self.coordinator.status()
    }

    /// Get the diff between Tantivy's book hashes and the semantic index.
    /// Used to determine which books need re-indexing.
    pub fn get_semantic_index_diff(
        &self,
        semantic_engine: &SemanticEngine,
        tantivy_books: &HashMap<String, u64>,
    ) -> IndexDiff {
        semantic_engine.diff_against_tantivy(tantivy_books)
    }
}
