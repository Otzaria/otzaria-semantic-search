//! Clean high-level API for Flutter / flutter_rust_bridge.
//!
//! Provides Flutter with domain-level operations for hybrid search and semantic index
//! lifecycle management.

use crate::hybrid::coordinator::{HybridCoordinator, HybridSearchParams};
use crate::semantic::types::{
    GroupingMode, HybridSearchResult, LexicalCandidate, SearchFilters, SearchMode, SemanticStatus,
};
use std::sync::Arc;

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
    pub fn search(
        &self,
        query: String,
        lexical_candidates: Vec<LexicalCandidate>,
        limit: Option<u32>,
        offset: Option<u32>,
        grouping: Option<GroupingMode>,
        filters: Option<SearchFilters>,
        force_mode: Option<SearchMode>,
    ) -> Result<HybridSearchResult, String> {
        let params = HybridSearchParams {
            limit: limit.unwrap_or(20) as usize,
            offset: offset.unwrap_or(0) as usize,
            grouping,
            filters,
            force_mode,
        };

        self.coordinator
            .search(&query, lexical_candidates, &params)
            .map_err(|e| e.to_string())
    }

    /// Query current status of the semantic search sidecar.
    pub fn get_semantic_status(&self) -> SemanticStatus {
        self.coordinator.status()
    }
}
