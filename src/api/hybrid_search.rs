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
    BookForIndexing, ContentFingerprint, HybridSearchResult, IndexDiff, IndexingSummary,
    LexicalCandidate, SearchFilters, SemanticStatus,
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
    pub profile: Option<crate::config::profiles::SearchProfile>,
    pub feature_flags: Option<crate::config::feature_flags::FeatureFlags>,
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
            profile: request.profile,
            feature_flags: request.feature_flags,
        };

        self.coordinator
            .search(&request.query, request.lexical_candidates, &params)
            .map_err(|e| e.to_string())
    }

    /// Query the current status of the semantic sidecar.
    pub fn get_semantic_status(&self) -> SemanticStatus {
        self.coordinator.status()
    }

    /// Retrieve the current telemetry snapshot.
    pub fn get_telemetry_snapshot(&self) -> crate::telemetry::TelemetrySnapshot {
        self.coordinator.get_telemetry_snapshot()
    }

    /// Reset the telemetry data.
    pub fn reset_telemetry(&self) {
        self.coordinator.reset_telemetry()
    }

    /// Clear the query cache.
    pub fn clear_query_cache(&self) {
        self.coordinator.clear_query_cache()
    }

    /// Diff the library's per-book fingerprints against the semantic index.
    ///
    /// Prefer this form: the caller decides what a book's fingerprint is, which is
    /// the only way a PDF can ever be reported as up to date.
    ///
    /// * text book → [`ContentFingerprint::from_lexical_hash`] of the lexical
    ///   engine's `contentHash`, which already folds in the metadata it indexes;
    /// * PDF → [`ContentFingerprint::canonical`], which folds the caller's own
    ///   authoritative source revision together with the title, category path
    ///   and facets. It must cover extracted text, line/section structure and
    ///   extraction/OCR version. A size/mtime signature alone cannot prove the
    ///   index is current — a
    ///   corrected author changes every vector and no byte of the file — and
    ///   [`ContentFingerprint::content_only`] is how to say so;
    /// * nothing → [`ContentFingerprint::Unverifiable`].
    ///
    /// The last two land the book in [`IndexDiff::unverifiable_books`].
    ///
    /// Across the FFI boundary prefer
    /// [`Self::get_semantic_index_diff_from_lexical_hashes`] or a plain
    /// `u64` DTO: an enum Dart can construct is an enum Dart can construct wrongly.
    ///
    /// `None` when no semantic engine is configured.
    pub fn get_semantic_index_diff(
        &self,
        books: &HashMap<String, ContentFingerprint>,
    ) -> Option<IndexDiff> {
        self.coordinator.semantic_index_diff(books)
    }

    /// Diff raw lexical `contentHash` values against the semantic index.
    ///
    /// Convenience for a caller that has nothing but Tantivy's hashes. Every PDF
    /// then lands in [`IndexDiff::unverifiable_books`] on every call, because the
    /// lexical engine records `contentHash = 0` for them and that cannot prove
    /// anything — see [`SemanticEngine::diff_against_tantivy`](crate::semantic::engine::SemanticEngine::diff_against_tantivy).
    pub fn get_semantic_index_diff_from_lexical_hashes(
        &self,
        tantivy_books: &HashMap<String, u64>,
    ) -> Option<IndexDiff> {
        let fingerprints = tantivy_books
            .iter()
            .map(|(key, &hash)| (key.clone(), ContentFingerprint::from_lexical_hash(hash)))
            .collect();
        self.coordinator.semantic_index_diff(&fingerprints)
    }

    /// Index books into the semantic index, replacing anything held for them.
    ///
    /// Returns what happened per category (indexed / skipped / empty), or `None`
    /// if the semantic path is disabled. Synchronous and potentially
    /// long-running; searches stall for at most one book at a time, the manifest
    /// is committed once rather than per book, and two concurrent calls
    /// are serialized — see [`HybridCoordinator::index_books`]. Scheduling it off
    /// the UI thread is the caller's job until the progress API lands in P7.
    pub fn index_books(
        &self,
        books: &[BookForIndexing],
    ) -> Result<Option<IndexingSummary>, String> {
        self.coordinator
            .index_books(books)
            .map_err(|e| e.to_string())
    }

    /// Remove books reported by [`IndexDiff::removed_books`].
    ///
    /// Returns the number of semantic vectors removed, or `None` when the
    /// semantic path is disabled.
    pub fn remove_semantic_books(
        &self,
        source_book_keys: &[String],
    ) -> Result<Option<u32>, String> {
        self.coordinator
            .remove_semantic_books(source_book_keys)
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
