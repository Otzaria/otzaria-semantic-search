//! Clean high-level API for Flutter / flutter_rust_bridge.
//!
//! Provides domain-level operations for hybrid search and semantic index
//! lifecycle management. Flutter never sees GGUF, chunking, the vector backend,
//! the manifest or the fusion implementation.
//!
//! # Scope
//!
//! This is the seam the bridge will be generated over, not the finished bridge.
//! Progress streams, cancellation and model-download management are roadmap P7;
//! what is here is what the correctness work needs to be reachable — searching,
//! status, the index diff, indexing and the reset that recovers from an
//! incompatible index.

use crate::hybrid::coordinator::{HybridCoordinator, HybridSearchParams};
use crate::semantic::types::{
    BookForIndexing, HybridSearchResult, IndexDiff, LexicalCandidate, SearchFilters, SemanticStatus,
};
use std::collections::HashMap;
use std::sync::Arc;

/// Parameters for a hybrid search API call.
/// Groups all optional parameters to avoid a too-many-arguments signature.
#[derive(Debug, Clone, Default)]
pub struct SearchRequest {
    pub query: String,
    pub lexical_candidates: Vec<LexicalCandidate>,
    pub limit: Option<u32>,
    pub offset: Option<u32>,
    pub grouping: Option<crate::semantic::types::GroupingMode>,
    pub filters: Option<SearchFilters>,
    /// Retrieval mode. `None` means hybrid.
    ///
    /// Distinct from the app's lexical query mode (exact/advanced/fuzzy): that
    /// describes how the *query* is interpreted, this describes where the
    /// candidates come from.
    pub force_mode: Option<crate::semantic::types::SearchMode>,
}

/// Opaque handle over the hybrid coordinator.
#[derive(Clone)]
pub struct OtzariaHybridEngine {
    coordinator: Arc<HybridCoordinator>,
}

impl OtzariaHybridEngine {
    /// Initialize the hybrid search engine.
    pub fn new(coordinator: HybridCoordinator) -> Self {
        Self {
            coordinator: Arc::new(coordinator),
        }
    }

    /// Perform a hybrid search combining BM25 candidates and semantic vectors.
    ///
    /// A semantic failure never fails the call: the result reports the mode that
    /// actually ran and why, so the caller can surface a degraded state instead
    /// of an error. See [`HybridSearchResult`].
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

    /// Query the current status of the semantic sidecar.
    pub fn get_semantic_status(&self) -> SemanticStatus {
        self.coordinator.status()
    }

    /// Diff Tantivy's per-book content hashes against the semantic index, to
    /// determine what needs indexing.
    ///
    /// `None` when no semantic engine is configured.
    pub fn get_semantic_index_diff(
        &self,
        tantivy_books: &HashMap<String, u64>,
    ) -> Option<IndexDiff> {
        self.coordinator.semantic_index_diff(tantivy_books)
    }

    /// Index books into the semantic index, replacing anything held for them.
    ///
    /// Returns the number of chunks written, or `None` if the semantic path is
    /// disabled. Synchronous and potentially long-running — the caller owns
    /// scheduling it off the UI thread until the progress API lands in P7.
    pub fn index_books(&self, books: &[BookForIndexing]) -> Result<Option<u32>, String> {
        self.coordinator
            .index_books(books)
            .map_err(|e| e.to_string())
    }

    /// Discard the semantic index and start over.
    ///
    /// Required when [`SemanticStatus::needs_full_reindex`] is set: the stored
    /// vectors were built with an incompatible configuration and cannot be
    /// queried or extended until they are dropped. Returns the number of vectors
    /// discarded, or `None` if there is no semantic engine.
    pub fn reset_semantic_index(&self) -> Result<Option<u32>, String> {
        self.coordinator
            .reset_semantic_index()
            .map_err(|e| e.to_string())
    }
}
