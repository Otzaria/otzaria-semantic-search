//! Clean high-level API for Flutter / flutter_rust_bridge.
//!
//! Provides domain-level operations for hybrid search and semantic index
//! lifecycle management. Flutter never sees the model, chunking, the vector backend,
//! the manifest or the fusion implementation.
//!
//! # Scope
//!
//! This is the seam the bridge will be generated over, not the finished bridge.
//! What is here is what the correctness work needs to be reachable — searching,
//! status, the index diff, indexing and the reset that recovers from an
//! incompatible index.
//!
//! The indexing operations are **prototype and build-side**: tests and the future
//! artifact builder use them. The application path is installing a prebuilt
//! read-only index, so no progress stream and no cancel/resume will be added here
//! — see `docs/PRODUCT_CONTRACT.md` §4. Model download management is the host
//! application's job (§5).
//!
//! A *search*, on the other hand, can be abandoned:
//! [`OtzariaHybridEngine::search_cancellable`]. That is not the indexing cancel/resume
//! ruled out above — it is what lets a search per keystroke drop the queries the next
//! keystroke made obsolete. See [`crate::cancellation`].
//!
//! On a coordinator built over an installed official artifact
//! ([`HybridCoordinator::with_official_index`]) every one of those operations refuses
//! by name, with
//! [`SemanticSearchError::ReadOnlyIndex`](crate::errors::SemanticSearchError::ReadOnlyIndex).
//! They are still on this type because the builder needs them and it has no other seam;
//! what the *application* may call is search and status. Dropping them from the surface
//! the app links against belongs to the FFI layer, in S5.

use crate::cancellation::CancellationToken;
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
    /// Every ranking parameter for this search, in place of the preset `profile` names:
    /// how the host tunes the fusion strategy, RRF's `k`, alpha, BM25's `k`, the semantic
    /// threshold and the bonuses without a release of this crate. `None` is the preset.
    ///
    /// Validated first; a parameter out of range fails the search, naming the parameter.
    /// See [`HybridSearchParams::ranking`], and
    /// [`RankingProfile`](crate::config::profiles::RankingProfile) for why the defaults are
    /// still placeholders.
    pub ranking: Option<crate::config::profiles::RankingProfile>,
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
    ///
    /// Cannot be cancelled; [`Self::search_cancellable`] can.
    pub fn search(&self, request: SearchRequest) -> Result<HybridSearchResult, String> {
        self.search_cancellable(request, &CancellationToken::new())
            .map_err(|e| e.to_string())
    }

    /// As [`Self::search`], abandoned once `cancel` is cancelled.
    ///
    /// For a search per keystroke: keep a clone of the token, and cancel it when the next
    /// keystroke's search starts. The superseded search then stops at its next checkpoint
    /// — within a fraction of a millisecond of scanning, or once the query embedding in
    /// progress finishes — with
    /// [`SemanticSearchError::Cancelled`](crate::errors::SemanticSearchError::Cancelled),
    /// having cached nothing and logged nothing. See [`crate::cancellation`].
    ///
    /// The error is the engine's own, not its message as in [`Self::search`], because the
    /// one outcome a caller must tell apart has to be matched, not parsed: `Cancelled` is
    /// the caller's own doing, to be dropped silently, while every other error is exactly
    /// what [`Self::search`] would have reported as text. Hydrating semantic-only results
    /// happens after this returns, in the caller, so a caller that cancels from another
    /// thread should look at the token again before it hydrates.
    pub fn search_cancellable(
        &self,
        request: SearchRequest,
        cancel: &CancellationToken,
    ) -> Result<HybridSearchResult, crate::errors::SemanticSearchError> {
        let params = HybridSearchParams {
            limit: request.limit.unwrap_or(20) as usize,
            offset: request.offset.unwrap_or(0) as usize,
            grouping: request.grouping,
            filters: request.filters,
            force_mode: request.force_mode,
            profile: request.profile,
            feature_flags: request.feature_flags,
            ranking: request.ranking,
        };

        // The facade serves a self-built index, which resolves its own hits.
        self.coordinator.search_cancellable(
            &request.query,
            request.lexical_candidates,
            &params,
            &crate::semantic::resolve::NoResolver,
            cancel,
        )
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
    /// `Ok(None)` when no semantic index is configured. An error when the index is an
    /// installed official artifact: the question only makes sense for an index this
    /// device builds.
    pub fn get_semantic_index_diff(
        &self,
        books: &HashMap<String, ContentFingerprint>,
    ) -> Result<Option<IndexDiff>, String> {
        self.coordinator
            .semantic_index_diff(books)
            .map_err(|e| e.to_string())
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
    ) -> Result<Option<IndexDiff>, String> {
        let fingerprints = tantivy_books
            .iter()
            .map(|(key, &hash)| (key.clone(), ContentFingerprint::from_lexical_hash(hash)))
            .collect();
        self.coordinator
            .semantic_index_diff(&fingerprints)
            .map_err(|e| e.to_string())
    }

    /// Index books into the semantic index, replacing anything held for them.
    ///
    /// Returns what happened per category (indexed / skipped / empty), or `None`
    /// if the semantic path is disabled. Synchronous and potentially
    /// long-running; searches stall for at most one book at a time, the manifest
    /// is committed once rather than per book, and two concurrent calls
    /// are serialized — see [`HybridCoordinator::index_books`].
    ///
    /// Build-side and test-side only: the application installs a prebuilt index
    /// rather than calling this, and an official artifact refuses it. Scheduling it off
    /// any UI thread is the caller's job, and stays that way — there is no progress API
    /// coming.
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
