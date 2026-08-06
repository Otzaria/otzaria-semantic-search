use crate::errors::VectorStoreError;
use crate::semantic::types::{SearchFilters, SemanticCandidate, VectorMetadata};

/// Trait defining the contract for vector storage backends.
///
/// Both the in-memory backend and the persistent Zevc backend implement this
/// trait. The current [`SemanticEngine`](crate::semantic::engine::SemanticEngine)
/// still selects the in-memory store directly; this is the common contract a
/// future backend-selection layer can use.
///
/// Note what this trait does *not* imply: neither implementation is an
/// approximate-nearest-neighbour index. Both scan every stored vector. The
/// official read-only backend that the product contract calls for — and the
/// measurement that decides whether a full scan can meet the budget at library
/// scale — is stage S2.
pub trait VectorStoreBackend: Send + Sync {
    /// Stable identifier persisted in the manifest.
    fn backend_id(&self) -> &'static str;

    /// Whether stored vectors survive process restart.
    fn is_persistent(&self) -> bool;

    /// Dimensionality every vector must have.
    fn embedding_dim(&self) -> u32;

    /// Number of vectors currently stored.
    fn count(&self) -> u32;

    /// Validate, normalize, and insert or replace a batch of vectors.
    fn insert_batch(
        &self,
        records: Vec<(VectorMetadata, Vec<f32>)>,
    ) -> Result<u32, VectorStoreError>;

    /// Search for the top-k most similar vectors to a query.
    fn search(
        &self,
        query_vector: &[f32],
        top_k: usize,
        filters: Option<&SearchFilters>,
    ) -> Result<Vec<SemanticCandidate>, VectorStoreError>;

    /// Remove all vectors belonging to a book.
    fn remove_by_book(&self, source_book_key: &str) -> Result<u32, VectorStoreError>;

    /// Remove all vectors.
    fn clear(&self) -> Result<u32, VectorStoreError>;

    /// List all book keys that have vectors stored.
    fn book_keys(&self) -> Vec<String>;
}
