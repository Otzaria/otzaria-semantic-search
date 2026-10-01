//! What a vector backend provides, in two halves: what the runtime may do, and what
//! only a builder may do.
//!
//! The split is the contract, not a convenience. The official index is built on a build
//! machine and opened read-only on the user's device, so the runtime path is handed a
//! [`VectorSearchBackend`] — which has no `insert`, no `remove` and no `clear` to call.
//! That is a property of the type rather than a rule a caller has to remember, and it is
//! why [`OfficialSemanticIndex`](crate::semantic::official_index::OfficialSemanticIndex)
//! cannot write to an artifact even by mistake.
//!
//! [`VectorStoreBackend`] adds the mutations, and is what a builder and the prototype
//! indexing path in [`SemanticEngine`](crate::semantic::engine::SemanticEngine) get.
//!
//! Note what neither trait implies: neither implementation is an approximate-nearest-
//! neighbour index. Both scan every stored vector. Whether a full scan meets the latency
//! and memory budget at library scale is what S2b measures.

use crate::cancellation::CancellationToken;
use crate::errors::VectorStoreError;
use crate::semantic::types::{SearchFilters, SemanticCandidate, VectorMetadata};

/// The read side of a vector backend: everything a query needs, and nothing more.
pub trait VectorSearchBackend: Send + Sync {
    /// Identifier of the backend that owns the payload format.
    ///
    /// Recorded as `store.backend_id` in an artifact's identity — see
    /// [`StoreIdentity`](crate::semantic::versioning::StoreIdentity) — so one backend
    /// never reads a payload another one wrote.
    fn backend_id(&self) -> &'static str;

    /// Whether stored vectors survive process restart.
    fn is_persistent(&self) -> bool;

    /// Dimensionality every vector must have.
    fn embedding_dim(&self) -> u32;

    /// Number of vectors currently stored.
    fn count(&self) -> u32;

    /// Search for the top-k most similar vectors to a query.
    ///
    /// [`Self::search_cancellable`] with a token nobody cancels — the same scan, the same
    /// answer.
    fn search(
        &self,
        query_vector: &[f32],
        top_k: usize,
        filters: Option<&SearchFilters>,
    ) -> Result<Vec<SemanticCandidate>, VectorStoreError> {
        self.search_cancellable(query_vector, top_k, filters, &CancellationToken::new())
    }

    /// Search for the top-k most similar vectors to a query, or stop with
    /// [`VectorStoreError::Cancelled`] once `cancel` is cancelled.
    ///
    /// The method a backend implements, and it has no default on purpose. The scan is the
    /// expensive part of a query — every stored vector, with nothing to narrow it — so it
    /// is where a cancel has to be noticed: an implementation looks at the token before its
    /// first record and every
    /// [`SCAN_CHECK_INTERVAL`](crate::cancellation::SCAN_CHECK_INTERVAL) records after
    /// that, not just once before it starts. A default that only looked first would let a
    /// new backend compile and still finish every abandoned scan.
    fn search_cancellable(
        &self,
        query_vector: &[f32],
        top_k: usize,
        filters: Option<&SearchFilters>,
        cancel: &CancellationToken,
    ) -> Result<Vec<SemanticCandidate>, VectorStoreError>;

    /// Book keys that have vectors stored, in a deterministic order.
    fn book_keys(&self) -> Vec<String>;

    /// How many vectors are stored for one book. `0` for a book the store does not hold.
    fn book_vector_count(&self, source_book_key: &str) -> usize;
}

/// The write side: what a builder does, and what the application never does.
pub trait VectorStoreBackend: VectorSearchBackend {
    /// Validate, normalize, and insert or replace a batch of vectors.
    fn insert_batch(
        &self,
        records: Vec<(VectorMetadata, Vec<f32>)>,
    ) -> Result<u32, VectorStoreError>;

    /// Remove all vectors belonging to a book.
    fn remove_by_book(&self, source_book_key: &str) -> Result<u32, VectorStoreError>;

    /// Remove all vectors.
    fn clear(&self) -> Result<u32, VectorStoreError>;

    /// Flush the store to durable storage.
    ///
    /// A no-op for a volatile backend, returning `Ok` so a caller can place its commit
    /// points correctly — never a claim that anything was persisted. Ask
    /// [`VectorSearchBackend::is_persistent`] for that.
    fn commit(&self) -> Result<(), VectorStoreError>;
}
