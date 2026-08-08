//! Vector store: similarity search over the semantic index, isolated from Tantivy —
//! its own directory, lifecycle and failure domain.
//!
//! This backend keeps vectors in memory and scans them exhaustively (`O(N·D)`). It
//! is **not persistent**, which [`VectorStore::is_persistent`] reports so callers
//! never treat a stale manifest as a populated index.
//!
//! The official path does not use this backend: it opens an installed artifact through
//! [`ReadOnlyZevcStore`](crate::semantic::zevc_store::ReadOnlyZevcStore), which persists
//! and cannot be written to. This one is what
//! [`SemanticEngine::open`](crate::semantic::engine::SemanticEngine::open) starts from —
//! the prototype and test default — and any backend can take its place through
//! [`SemanticEngine::with_store`](crate::semantic::engine::SemanticEngine::with_store).
//!
//! Whether the official backend also needs to be *approximate* is a question the S2b
//! measurement answers — a full scan may or may not fit the latency and memory budget
//! once the dimension and precision are chosen in S1.
//! [`VectorSearchBackend`](crate::semantic::store_backend::VectorSearchBackend) is the
//! seam either answer slots into.

use crate::errors::VectorStoreError;
// One threshold for the whole crate, so no two layers can disagree about it.
use crate::semantic::embedding::MIN_VECTOR_NORM;
use crate::semantic::types::{SearchFilters, SemanticCandidate, VectorMetadata};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::path::PathBuf;
use std::sync::RwLock;

/// Recorded in the manifest, so one backend never reads another's index.
pub const BACKEND_ID: &str = "in-memory-v1";

#[derive(Debug, Clone)]
pub struct VectorStoreConfig {
    pub db_path: PathBuf,
    pub embedding_dim: u32,
    pub collection_name: String,
}

impl Default for VectorStoreConfig {
    fn default() -> Self {
        Self {
            db_path: PathBuf::from("semantic_db"),
            embedding_dim: 1024,
            collection_name: "otzaria_chunks".to_string(),
        }
    }
}

/// A stored vector record. Vectors are L2-normalized on the way in, so cosine
/// similarity at search time is a single dot product.
#[derive(Debug, Clone)]
pub struct StoredVectorRecord {
    pub metadata: VectorMetadata,
    /// Pre-normalized vector (L2 norm = 1.0).
    pub vector: Vec<f32>,
}

/// Both maps are guarded by one lock: with a lock each, insert took them in the
/// opposite order from delete — a lock-order inversion that can deadlock under
/// concurrent indexing. One lock also makes a batch or a delete atomic for readers.
#[derive(Default)]
struct StoreState {
    /// `semantic_id` → record.
    records: HashMap<String, StoredVectorRecord>,
    /// `source_book_key` → its `semantic_id`s. Invariant: an id appears exactly once,
    /// under exactly one book, and only while `records` holds it.
    book_index: HashMap<String, Vec<String>>,
}

/// Bounded-heap entry for top-k selection.
///
/// The id is borrowed from the record: the scan visits every vector, so an owned `String`
/// here would be one allocation per stored vector per query — cost the full scan does not
/// require. The records are behind the read lock for the whole scan, so they outlive it.
struct ScoredEntry<'a> {
    score: f32,
    semantic_id: &'a str,
}

impl PartialEq for ScoredEntry<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for ScoredEntry<'_> {}

impl PartialOrd for ScoredEntry<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredEntry<'_> {
    /// Orders candidates **worst first**, making a max-heap a bounded min-heap: `peek`
    /// yields the candidate to evict. Among equal scores a larger `semantic_id` is
    /// "greater" — randomized `HashMap` order would otherwise return a different top-k
    /// every run. `total_cmp` stays total if a NaN slips through.
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .total_cmp(&self.score)
            .then_with(|| self.semantic_id.cmp(other.semantic_id))
    }
}

pub struct VectorStore {
    config: VectorStoreConfig,
    state: RwLock<StoreState>,
}

impl VectorStore {
    /// Open or initialize a vector store at the configured directory.
    pub fn open_or_create(config: VectorStoreConfig) -> Result<Self, VectorStoreError> {
        if config.embedding_dim == 0 {
            return Err(VectorStoreError::OpenFailed {
                reason: "embedding_dim must be greater than zero".to_string(),
            });
        }

        if let Err(e) = std::fs::create_dir_all(&config.db_path) {
            return Err(VectorStoreError::OpenFailed {
                reason: format!("Failed to create DB directory: {e}"),
            });
        }

        let store = Self {
            config,
            state: RwLock::new(StoreState::default()),
        };

        log::info!(
            "VectorStore ({BACKEND_ID}, persistent={}) initialized at: {}",
            store.is_persistent(),
            store.config.db_path.display()
        );
        Ok(store)
    }

    pub fn backend_id(&self) -> &'static str {
        BACKEND_ID
    }

    /// Whether stored vectors survive process restart. `false` for the in-memory
    /// backend, and a caller must not trust an on-disk manifest claiming indexed books
    /// while it is: the vectors those records describe are gone.
    pub fn is_persistent(&self) -> bool {
        false
    }

    /// Dimensionality every vector in this store must have.
    pub fn embedding_dim(&self) -> u32 {
        self.config.embedding_dim
    }

    /// Insert or replace a batch of vector records, L2-normalizing each. Either the
    /// whole batch is applied or none of it: everything is validated up front, so a
    /// rejected vector cannot leave the store half-written.
    ///
    /// The store re-checks that every vector is *retrievable* even though the runtime
    /// already did, because this is a public entry point that also accepts vectors the
    /// runtime never saw, and an unscorable vector leaves a record that exists but can
    /// never be returned. Rejected: a non-finite component (scores `NaN`), a zero norm
    /// (no direction), and a finite vector whose squares overflow `f32` (norm `inf`,
    /// reciprocal `0`, so normalization silently yields zeros). The threshold has one
    /// definition, `MIN_VECTOR_NORM`, compared with the same `<=` in both places.
    pub fn insert_batch(
        &self,
        batch: &[(VectorMetadata, Vec<f32>)],
    ) -> Result<(), VectorStoreError> {
        for (meta, vector) in batch {
            if vector.len() as u32 != self.config.embedding_dim {
                return Err(VectorStoreError::DimensionMismatch {
                    store_dim: self.config.embedding_dim,
                    vector_dim: vector.len() as u32,
                });
            }

            let reject = |reason: String| VectorStoreError::InsertFailed {
                reason: format!(
                    "vector for {} {reason}; it could never be matched by a search",
                    meta.semantic_id
                ),
            };

            if let Some(position) = vector.iter().position(|x| !x.is_finite()) {
                return Err(reject(format!(
                    "has a non-finite component at {position} ({})",
                    vector[position]
                )));
            }
            let norm = l2_norm(vector);
            if !norm.is_finite() {
                return Err(reject(format!(
                    "has a non-finite norm ({norm}); its magnitudes overflowed f32"
                )));
            }
            // `<=`: a vector exactly on the threshold is rejected, as in every layer.
            if norm <= MIN_VECTOR_NORM {
                return Err(reject(format!(
                    "has a norm of {norm}, at or below the minimum {MIN_VECTOR_NORM}, so it \
                     has no usable direction"
                )));
            }
        }

        let mut state = self.write_state();

        for (meta, vector) in batch {
            let id = meta.semantic_id.clone();
            let record = StoredVectorRecord {
                metadata: meta.clone(),
                vector: l2_normalize_vec(vector),
            };

            match state.records.insert(id.clone(), record) {
                None => state
                    .book_index
                    .entry(meta.source_book_key.clone())
                    .or_default()
                    .push(id),
                // Already listed; only a move between books needs fixing. The id is
                // derived from the book key, but a stale entry would leak vectors past
                // a book delete.
                Some(previous) => {
                    let previous_book = previous.metadata.source_book_key;
                    if previous_book != meta.source_book_key {
                        if let Some(ids) = state.book_index.get_mut(&previous_book) {
                            ids.retain(|existing| existing != &id);
                            if ids.is_empty() {
                                state.book_index.remove(&previous_book);
                            }
                        }
                        state
                            .book_index
                            .entry(meta.source_book_key.clone())
                            .or_default()
                            .push(id);
                    }
                }
            }
        }

        log::debug!("VectorStore inserted {} vectors", batch.len());
        Ok(())
    }

    /// Search for nearest neighbours by cosine similarity: a bounded min-heap for
    /// `O(N log k)` top-k selection, results sorted by descending similarity with ties
    /// broken by `semantic_id` for reproducibility.
    pub fn search(
        &self,
        query_vector: &[f32],
        top_k: usize,
        filters: Option<&SearchFilters>,
    ) -> Result<Vec<SemanticCandidate>, VectorStoreError> {
        if query_vector.len() as u32 != self.config.embedding_dim {
            return Err(VectorStoreError::DimensionMismatch {
                store_dim: self.config.embedding_dim,
                vector_dim: query_vector.len() as u32,
            });
        }
        if top_k == 0 {
            return Ok(Vec::new());
        }

        // An incomparable query yields nothing rather than NaN-scoring everything.
        let norm = l2_norm(query_vector);
        if !norm.is_finite() || norm <= MIN_VECTOR_NORM {
            return Ok(Vec::new());
        }
        if query_vector.iter().any(|x| !x.is_finite()) {
            return Ok(Vec::new());
        }
        let query = l2_normalize_vec(query_vector);

        // Group facet paths by dimension once per scan: `matches` per record would
        // recompile them for every stored vector. `None` skips the check entirely.
        let filters = filters.and_then(SearchFilters::compile);

        let state = self.read_state();

        let mut heap: BinaryHeap<ScoredEntry> = BinaryHeap::with_capacity(top_k + 1);
        for record in state.records.values() {
            if let Some(compiled) = filters.as_ref() {
                if !compiled.matches(&record.metadata) {
                    continue;
                }
            }

            // Both sides are pre-normalized, so the dot product is the cosine.
            let score = dot_product(&query, &record.vector);
            if score.is_nan() {
                continue;
            }

            let entry = ScoredEntry {
                score,
                semantic_id: &record.metadata.semantic_id,
            };
            if heap.len() < top_k {
                heap.push(entry);
            } else if let Some(weakest) = heap.peek() {
                // Worst-first ordering, so `<` reads as "better than the weakest kept".
                if entry < *weakest {
                    heap.pop();
                    heap.push(entry);
                }
            }
        }

        let mut candidates: Vec<SemanticCandidate> = heap
            .into_iter()
            .filter_map(|entry| {
                state
                    .records
                    .get(entry.semantic_id)
                    .map(|record| SemanticCandidate {
                        metadata: record.metadata.clone(),
                        similarity_score: entry.score,
                    })
            })
            .collect();

        candidates.sort_by(|a, b| {
            b.similarity_score
                .total_cmp(&a.similarity_score)
                .then_with(|| a.metadata.semantic_id.cmp(&b.metadata.semantic_id))
        });

        Ok(candidates)
    }

    /// Remove every vector belonging to a book. Returns how many were removed.
    pub fn delete_book(&self, source_book_key: &str) -> Result<u32, VectorStoreError> {
        let mut state = self.write_state();

        let mut deleted = 0u32;
        if let Some(ids) = state.book_index.remove(source_book_key) {
            for id in ids {
                if state.records.remove(&id).is_some() {
                    deleted += 1;
                }
            }
        }

        log::info!("Deleted {deleted} vectors for book: {source_book_key}");
        Ok(deleted)
    }

    /// Remove every vector in the store, for a rebuild from scratch (incompatible
    /// configuration, model change, corrupted manifest).
    pub fn clear(&self) -> Result<u32, VectorStoreError> {
        let mut state = self.write_state();
        let removed = state.records.len() as u32;
        state.records.clear();
        state.book_index.clear();
        log::info!("Cleared {removed} vectors from the store");
        Ok(removed)
    }

    pub fn vector_count(&self) -> usize {
        self.read_state().records.len()
    }

    pub fn book_vector_count(&self, source_book_key: &str) -> usize {
        self.read_state()
            .book_index
            .get(source_book_key)
            .map_or(0, |ids| ids.len())
    }

    /// List book keys deterministically for backend-neutral lifecycle code.
    pub fn book_keys(&self) -> Vec<String> {
        let mut keys: Vec<String> = self.read_state().book_index.keys().cloned().collect();
        keys.sort();
        keys
    }

    pub fn contains(&self, semantic_id: &str) -> bool {
        self.read_state().records.contains_key(semantic_id)
    }

    /// Flush state to durable storage. A no-op for the in-memory backend — see
    /// [`VectorStore::is_persistent`] — returning `Ok` so callers can place commit
    /// points correctly, not to claim anything was persisted.
    pub fn commit(&self) -> Result<(), VectorStoreError> {
        Ok(())
    }

    /// Acquire the write lock, recovering from poisoning: refusing every later access
    /// would take the semantic path down for the whole session over one bad batch.
    /// Recovery is sound because the two maps cannot be left disagreeing — each record
    /// enters or leaves both in one uninterrupted step, and validation happens before
    /// the lock is taken, so an aborted batch only ever leaves *fewer* vectors.
    fn write_state(&self) -> std::sync::RwLockWriteGuard<'_, StoreState> {
        self.state.write().unwrap_or_else(|poisoned| {
            log::warn!("VectorStore lock was poisoned; recovering state");
            poisoned.into_inner()
        })
    }

    /// Acquire the read lock, recovering from poisoning. See [`Self::write_state`].
    fn read_state(&self) -> std::sync::RwLockReadGuard<'_, StoreState> {
        self.state.read().unwrap_or_else(|poisoned| {
            log::warn!("VectorStore lock was poisoned; recovering state");
            poisoned.into_inner()
        })
    }
}

impl crate::semantic::store_backend::VectorSearchBackend for VectorStore {
    fn backend_id(&self) -> &'static str {
        VectorStore::backend_id(self)
    }

    fn is_persistent(&self) -> bool {
        VectorStore::is_persistent(self)
    }

    fn embedding_dim(&self) -> u32 {
        VectorStore::embedding_dim(self)
    }

    fn count(&self) -> u32 {
        VectorStore::vector_count(self).min(u32::MAX as usize) as u32
    }

    fn search(
        &self,
        query_vector: &[f32],
        top_k: usize,
        filters: Option<&SearchFilters>,
    ) -> Result<Vec<SemanticCandidate>, VectorStoreError> {
        VectorStore::search(self, query_vector, top_k, filters)
    }

    fn book_keys(&self) -> Vec<String> {
        VectorStore::book_keys(self)
    }

    fn book_vector_count(&self, source_book_key: &str) -> usize {
        VectorStore::book_vector_count(self, source_book_key)
    }
}

impl crate::semantic::store_backend::VectorStoreBackend for VectorStore {
    fn insert_batch(
        &self,
        records: Vec<(VectorMetadata, Vec<f32>)>,
    ) -> Result<u32, VectorStoreError> {
        VectorStore::insert_batch(self, &records)?;
        Ok(records.len().min(u32::MAX as usize) as u32)
    }

    fn remove_by_book(&self, source_book_key: &str) -> Result<u32, VectorStoreError> {
        VectorStore::delete_book(self, source_book_key)
    }

    fn clear(&self) -> Result<u32, VectorStoreError> {
        VectorStore::clear(self)
    }

    fn commit(&self) -> Result<(), VectorStoreError> {
        VectorStore::commit(self)
    }
}

/// Dot product of two equal-length vectors, split across several accumulators: one
/// running sum forces a serial float-add dependency chain per element, while
/// independent lanes pipeline and vectorize. This is the innermost loop of every
/// semantic query, run once per stored vector.
#[inline]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    const LANES: usize = 8;

    let mut acc = [0.0f32; LANES];
    let mut a_chunks = a.chunks_exact(LANES);
    let mut b_chunks = b.chunks_exact(LANES);

    for (x, y) in a_chunks.by_ref().zip(b_chunks.by_ref()) {
        for lane in 0..LANES {
            acc[lane] += x[lane] * y[lane];
        }
    }

    let mut sum = acc.iter().sum::<f32>();
    for (x, y) in a_chunks.remainder().iter().zip(b_chunks.remainder().iter()) {
        sum += x * y;
    }
    sum
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Return a new L2-normalized copy. A vector whose norm is indistinguishable from
/// zero is returned unchanged; callers check the norm themselves.
fn l2_normalize_vec(v: &[f32]) -> Vec<f32> {
    let norm = l2_norm(v);
    if norm <= MIN_VECTOR_NORM {
        return v.to_vec();
    }
    let inv = 1.0 / norm;
    v.iter().map(|x| x * inv).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_metadata(id: &str, book: &str) -> VectorMetadata {
        VectorMetadata {
            semantic_id: id.to_string(),
            source_book_key: book.to_string(),
            source_doc_key: format!("{book}:{id}"),
            line_id: 1,
            section_id: 10,
            line_hash: 100,
            chunk_hash: "hash".to_string(),
            content_hash: 555,
            reference: "Ref 1".to_string(),
            segment: 0,
            is_pdf: false,
            title: "Test Book".to_string(),
            facets: vec![
                "/מקרא/תורה".to_string(),
                "/author/Author 1".to_string(),
                "/era/Rishonim".to_string(),
            ],
        }
    }

    struct TempDir(PathBuf);
    impl TempDir {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "otzaria_test_{name}_{}",
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

    fn store(dir: &TempDir, dim: u32) -> VectorStore {
        VectorStore::open_or_create(VectorStoreConfig {
            db_path: dir.path().to_path_buf(),
            embedding_dim: dim,
            collection_name: "test".to_string(),
        })
        .unwrap()
    }

    #[test]
    fn test_vector_store_crud() {
        let dir = TempDir::new("store_crud");
        let store = store(&dir, 4);

        store
            .insert_batch(&[
                (
                    sample_metadata("id1", "book1.txt"),
                    vec![1.0, 0.0, 0.0, 0.0],
                ),
                (
                    sample_metadata("id2", "book1.txt"),
                    vec![0.0, 1.0, 0.0, 0.0],
                ),
            ])
            .unwrap();

        assert_eq!(store.vector_count(), 2);

        let hits = store.search(&[1.0, 0.0, 0.0, 0.0], 10, None).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].metadata.semantic_id, "id1");
        assert!((hits[0].similarity_score - 1.0).abs() < 1e-5);

        assert_eq!(store.delete_book("book1.txt").unwrap(), 2);
        assert_eq!(store.vector_count(), 0);
    }

    #[test]
    fn vectors_are_normalized_on_insert_so_scores_are_cosines() {
        let dir = TempDir::new("normalize_on_insert");
        let store = store(&dir, 4);

        // Same direction as the query, but 10× the magnitude.
        store
            .insert_batch(&[(
                sample_metadata("scaled", "book.txt"),
                vec![10.0, 0.0, 0.0, 0.0],
            )])
            .unwrap();

        let hits = store.search(&[0.5, 0.0, 0.0, 0.0], 5, None).unwrap();
        assert_eq!(hits.len(), 1);
        assert!(
            (hits[0].similarity_score - 1.0).abs() < 1e-5,
            "magnitude must not affect similarity, got {}",
            hits[0].similarity_score
        );
    }

    #[test]
    fn reinserting_the_same_chunk_replaces_it_without_duplicating_the_book_index() {
        let dir = TempDir::new("reinsert");
        let store = store(&dir, 4);
        let meta = sample_metadata("id1", "book1.txt");

        for _ in 0..3 {
            store
                .insert_batch(&[(meta.clone(), vec![1.0, 0.0, 0.0, 0.0])])
                .unwrap();
        }

        assert_eq!(store.vector_count(), 1);
        assert_eq!(
            store.book_vector_count("book1.txt"),
            1,
            "the book index must not accumulate duplicate ids"
        );

        // The delete must still find and remove it exactly once.
        assert_eq!(store.delete_book("book1.txt").unwrap(), 1);
        assert_eq!(store.vector_count(), 0);
        assert_eq!(store.book_vector_count("book1.txt"), 0);
    }

    #[test]
    fn delete_book_only_removes_that_books_vectors() {
        let dir = TempDir::new("delete_scope");
        let store = store(&dir, 4);

        store
            .insert_batch(&[
                (sample_metadata("a1", "a.txt"), vec![1.0, 0.0, 0.0, 0.0]),
                (sample_metadata("b1", "b.txt"), vec![0.0, 1.0, 0.0, 0.0]),
                (sample_metadata("b2", "b.txt"), vec![0.0, 0.0, 1.0, 0.0]),
            ])
            .unwrap();

        assert_eq!(store.delete_book("b.txt").unwrap(), 2);
        assert_eq!(store.vector_count(), 1);
        assert!(store.contains("a1"));
        assert!(!store.contains("b1"));
        assert_eq!(store.delete_book("does-not-exist.txt").unwrap(), 0);
    }

    #[test]
    fn clear_empties_the_store() {
        let dir = TempDir::new("clear");
        let store = store(&dir, 4);
        store
            .insert_batch(&[
                (sample_metadata("a1", "a.txt"), vec![1.0, 0.0, 0.0, 0.0]),
                (sample_metadata("b1", "b.txt"), vec![0.0, 1.0, 0.0, 0.0]),
            ])
            .unwrap();

        assert_eq!(store.clear().unwrap(), 2);
        assert_eq!(store.vector_count(), 0);
        assert_eq!(store.book_vector_count("a.txt"), 0);
        assert!(store
            .search(&[1.0, 0.0, 0.0, 0.0], 5, None)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn dimension_mismatch_is_rejected_on_both_insert_and_search() {
        let dir = TempDir::new("dims");
        let store = store(&dir, 4);

        let err = store
            .insert_batch(&[(sample_metadata("id1", "book.txt"), vec![1.0, 0.0])])
            .unwrap_err();
        assert!(matches!(
            err,
            VectorStoreError::DimensionMismatch {
                store_dim: 4,
                vector_dim: 2
            }
        ));
        assert_eq!(
            store.vector_count(),
            0,
            "a rejected insert must store nothing"
        );

        assert!(matches!(
            store.search(&[1.0, 0.0], 5, None).unwrap_err(),
            VectorStoreError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn a_batch_with_one_bad_vector_is_rejected_as_a_whole() {
        let dir = TempDir::new("atomic_batch");
        let store = store(&dir, 4);

        let result = store.insert_batch(&[
            (
                sample_metadata("good", "book.txt"),
                vec![1.0, 0.0, 0.0, 0.0],
            ),
            (sample_metadata("bad", "book.txt"), vec![1.0]),
        ]);

        assert!(result.is_err());
        assert_eq!(
            store.vector_count(),
            0,
            "the valid record must not be applied when its batch is rejected"
        );
    }

    /// A stored NaN would leave a record that exists but can never be returned.
    #[test]
    fn non_finite_vectors_are_refused_by_the_store() {
        let dir = TempDir::new("non_finite_insert");
        let store = store(&dir, 4);

        for bad in [
            vec![f32::NAN, 0.0, 0.0, 0.0],
            vec![f32::INFINITY, 0.0, 0.0, 0.0],
            vec![f32::NEG_INFINITY, 1.0, 0.0, 0.0],
            // No direction: matches nothing, so it can never be retrieved.
            vec![0.0, 0.0, 0.0, 0.0],
            // Finite, but the squares overflow f32: norm `inf`, so all zeros.
            vec![1e30, 1e30, 1e30, 1e30],
            vec![f32::MAX, f32::MAX, 0.0, 0.0],
            // Exactly on the threshold: one constant, one comparison, refused in both.
            vec![MIN_VECTOR_NORM, 0.0, 0.0, 0.0],
        ] {
            let result = store.insert_batch(&[(sample_metadata("bad", "book.txt"), bad.clone())]);
            assert!(
                matches!(result, Err(VectorStoreError::InsertFailed { .. })),
                "{bad:?} must be refused, got {result:?}"
            );
        }
        assert_eq!(store.vector_count(), 0);

        // The whole batch is refused, so nothing is half-applied.
        let result = store.insert_batch(&[
            (
                sample_metadata("good", "book.txt"),
                vec![1.0, 0.0, 0.0, 0.0],
            ),
            (
                sample_metadata("bad", "book.txt"),
                vec![f32::NAN, 0.0, 0.0, 0.0],
            ),
        ]);
        assert!(result.is_err());
        assert_eq!(store.vector_count(), 0);
    }

    /// A large magnitude that still squares within `f32` is perfectly usable.
    #[test]
    fn a_large_but_representable_vector_is_accepted() {
        let dir = TempDir::new("large_vector");
        let store = store(&dir, 4);

        store
            .insert_batch(&[(
                sample_metadata("large", "book.txt"),
                vec![1e18, 0.0, 0.0, 0.0],
            )])
            .unwrap();
        assert_eq!(store.vector_count(), 1);

        let hits = store.search(&[1.0, 0.0, 0.0, 0.0], 5, None).unwrap();
        assert!((hits[0].similarity_score - 1.0).abs() < 1e-5);
    }

    #[test]
    fn a_non_finite_query_returns_nothing_rather_than_an_arbitrary_set() {
        let dir = TempDir::new("non_finite_query");
        let store = store(&dir, 4);
        store
            .insert_batch(&[(sample_metadata("id1", "book.txt"), vec![1.0, 0.0, 0.0, 0.0])])
            .unwrap();

        for query in [
            vec![f32::NAN, 0.0, 0.0, 0.0],
            vec![f32::INFINITY, 0.0, 0.0, 0.0],
            vec![1e30, 1e30, 1e30, 1e30],
        ] {
            assert!(
                store.search(&query, 5, None).unwrap().is_empty(),
                "query {query:?} must not return results"
            );
        }
    }

    #[test]
    fn zero_norm_query_returns_no_results_instead_of_arbitrary_ones() {
        let dir = TempDir::new("zero_query");
        let store = store(&dir, 4);
        store
            .insert_batch(&[(sample_metadata("id1", "book.txt"), vec![1.0, 0.0, 0.0, 0.0])])
            .unwrap();

        assert!(store
            .search(&[0.0, 0.0, 0.0, 0.0], 5, None)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn top_k_zero_returns_nothing() {
        let dir = TempDir::new("top_k_zero");
        let store = store(&dir, 4);
        store
            .insert_batch(&[(sample_metadata("id1", "book.txt"), vec![1.0, 0.0, 0.0, 0.0])])
            .unwrap();

        assert!(store
            .search(&[1.0, 0.0, 0.0, 0.0], 0, None)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn test_top_k_bounds() {
        let dir = TempDir::new("top_k");
        let store = store(&dir, 4);

        let batch: Vec<(VectorMetadata, Vec<f32>)> = (0..20)
            .map(|i| {
                let mut v = vec![0.0; 4];
                v[i % 4] = 1.0;
                (sample_metadata(&format!("id{i:02}"), "book.txt"), v)
            })
            .collect();
        store.insert_batch(&batch).unwrap();

        let hits = store.search(&[1.0, 0.0, 0.0, 0.0], 3, None).unwrap();
        assert_eq!(hits.len(), 3);
    }

    /// The heap must keep the *best* k, not merely k of them: a reversed eviction
    /// comparison returns the right number of results, all of them the wrong ones.
    #[test]
    fn top_k_returns_the_best_candidates_not_just_k_of_them() {
        let dir = TempDir::new("top_k_best");
        let store = store(&dir, 2);

        // 20 vectors fanned out from the query: 0 closest, 19 furthest, inserted
        // worst-first so "keep whatever came first" also fails.
        let batch: Vec<(VectorMetadata, Vec<f32>)> = (0..20)
            .rev()
            .map(|i| {
                let angle = i as f32 * 0.05;
                (
                    sample_metadata(&format!("id{i:02}"), "book.txt"),
                    vec![angle.cos(), angle.sin()],
                )
            })
            .collect();
        store.insert_batch(&batch).unwrap();

        let hits = store.search(&[1.0, 0.0], 5, None).unwrap();
        let ids: Vec<&str> = hits
            .iter()
            .map(|h| h.metadata.semantic_id.as_str())
            .collect();
        assert_eq!(
            ids,
            vec!["id00", "id01", "id02", "id03", "id04"],
            "the five nearest vectors must be the ones returned"
        );

        let worst_kept = hits.last().unwrap().similarity_score;
        let all = store.search(&[1.0, 0.0], 20, None).unwrap();
        for discarded in &all[5..] {
            assert!(
                discarded.similarity_score <= worst_kept,
                "{} ({}) should not have been discarded",
                discarded.metadata.semantic_id,
                discarded.similarity_score
            );
        }
    }

    #[test]
    fn results_are_sorted_by_descending_similarity() {
        let dir = TempDir::new("ordering");
        let store = store(&dir, 4);

        store
            .insert_batch(&[
                (sample_metadata("far", "book.txt"), vec![0.0, 1.0, 0.0, 0.0]),
                (
                    sample_metadata("exact", "book.txt"),
                    vec![1.0, 0.0, 0.0, 0.0],
                ),
                (
                    sample_metadata("near", "book.txt"),
                    vec![1.0, 0.3, 0.0, 0.0],
                ),
            ])
            .unwrap();

        let hits = store.search(&[1.0, 0.0, 0.0, 0.0], 3, None).unwrap();
        let ids: Vec<&str> = hits
            .iter()
            .map(|h| h.metadata.semantic_id.as_str())
            .collect();
        assert_eq!(ids, vec!["exact", "near", "far"]);

        for pair in hits.windows(2) {
            assert!(pair[0].similarity_score >= pair[1].similarity_score);
        }
    }

    /// Randomized `HashMap` order means tie-breaking has to be explicit.
    #[test]
    fn tied_scores_break_deterministically_and_reproducibly() {
        let dir = TempDir::new("ties");
        let store = store(&dir, 4);

        // 12 records, all at exactly the same distance from the query.
        let batch: Vec<(VectorMetadata, Vec<f32>)> = (0..12)
            .map(|i| {
                (
                    sample_metadata(&format!("id{i:02}"), "book.txt"),
                    vec![1.0, 0.0, 0.0, 0.0],
                )
            })
            .collect();
        store.insert_batch(&batch).unwrap();

        let first = store.search(&[1.0, 0.0, 0.0, 0.0], 4, None).unwrap();
        let ids: Vec<String> = first
            .iter()
            .map(|h| h.metadata.semantic_id.clone())
            .collect();
        assert_eq!(ids, vec!["id00", "id01", "id02", "id03"]);

        for _ in 0..5 {
            let again = store.search(&[1.0, 0.0, 0.0, 0.0], 4, None).unwrap();
            let again_ids: Vec<String> = again
                .iter()
                .map(|h| h.metadata.semantic_id.clone())
                .collect();
            assert_eq!(again_ids, ids, "repeated identical queries must agree");
        }
    }

    #[test]
    fn filters_are_applied_during_search() {
        let dir = TempDir::new("search_filters");
        let store = store(&dir, 4);

        let mut pdf_meta = sample_metadata("pdf", "scan.pdf");
        pdf_meta.is_pdf = true;
        store
            .insert_batch(&[
                (
                    sample_metadata("text", "book.txt"),
                    vec![1.0, 0.0, 0.0, 0.0],
                ),
                (pdf_meta, vec![1.0, 0.0, 0.0, 0.0]),
            ])
            .unwrap();

        let all = store.search(&[1.0, 0.0, 0.0, 0.0], 10, None).unwrap();
        assert_eq!(all.len(), 2);

        let no_pdf = SearchFilters {
            include_pdf: Some(false),
            ..Default::default()
        };
        let filtered = store
            .search(&[1.0, 0.0, 0.0, 0.0], 10, Some(&no_pdf))
            .unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].metadata.semantic_id, "text");

        let by_book = SearchFilters {
            book_paths: Some(vec!["scan.pdf".to_string()]),
            ..Default::default()
        };
        let filtered = store
            .search(&[1.0, 0.0, 0.0, 0.0], 10, Some(&by_book))
            .unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].metadata.semantic_id, "pdf");
    }

    /// [`SearchFilters`]: an empty filter list must not empty the result set.
    #[test]
    fn empty_filter_lists_do_not_remove_results() {
        let dir = TempDir::new("empty_filters");
        let store = store(&dir, 4);
        store
            .insert_batch(&[(sample_metadata("id1", "book.txt"), vec![1.0, 0.0, 0.0, 0.0])])
            .unwrap();

        let empty_everything = SearchFilters {
            book_paths: Some(vec![]),
            facets: Some(vec![]),
            include_pdf: Some(true),
        };
        let hits = store
            .search(&[1.0, 0.0, 0.0, 0.0], 10, Some(&empty_everything))
            .unwrap();
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn filters_that_match_nothing_yield_no_results_not_an_error() {
        let dir = TempDir::new("filter_no_match");
        let store = store(&dir, 4);
        store
            .insert_batch(&[(sample_metadata("id1", "book.txt"), vec![1.0, 0.0, 0.0, 0.0])])
            .unwrap();

        let filters = SearchFilters {
            facets: Some(vec!["/תלמוד/בבלי".to_string()]),
            ..Default::default()
        };
        assert!(store
            .search(&[1.0, 0.0, 0.0, 0.0], 10, Some(&filters))
            .unwrap()
            .is_empty());
    }

    #[test]
    fn empty_store_searches_cleanly() {
        let dir = TempDir::new("empty_store");
        let store = store(&dir, 4);
        assert_eq!(store.vector_count(), 0);
        assert!(store
            .search(&[1.0, 0.0, 0.0, 0.0], 10, None)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn backend_reports_that_it_does_not_persist() {
        let dir = TempDir::new("persistence_flag");
        let store = store(&dir, 4);
        assert_eq!(store.backend_id(), "in-memory-v1");
        assert!(
            !store.is_persistent(),
            "the in-memory backend must not claim durability"
        );
        store.commit().unwrap();
    }

    #[test]
    fn zero_dimension_config_is_rejected() {
        let dir = TempDir::new("zero_dim");
        let result = VectorStore::open_or_create(VectorStoreConfig {
            db_path: dir.path().to_path_buf(),
            embedding_dim: 0,
            collection_name: "test".to_string(),
        });
        assert!(matches!(result, Err(VectorStoreError::OpenFailed { .. })));
    }

    #[test]
    fn dot_product_matches_the_naive_sum_including_non_multiples_of_the_lane_width() {
        // Lengths around the 8-lane split, where a chunked dot product drops one.
        for len in [0usize, 1, 7, 8, 9, 15, 16, 17, 1024, 1025] {
            let a: Vec<f32> = (0..len).map(|i| (i as f32 * 0.37).sin()).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32 * 0.11).cos()).collect();

            let naive: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let fast = dot_product(&a, &b);

            assert!(
                (naive - fast).abs() <= 1e-4 * naive.abs().max(1.0),
                "len {len}: naive {naive} vs chunked {fast}"
            );
        }
    }

    #[test]
    fn l2_normalize_vec_leaves_a_zero_vector_alone() {
        let zero = vec![0.0f32; 4];
        assert_eq!(l2_normalize_vec(&zero), zero);

        let normalized = l2_normalize_vec(&[3.0, 4.0]);
        assert!((l2_norm(&normalized) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn concurrent_inserts_and_deletes_do_not_deadlock() {
        use std::sync::Arc;

        let dir = TempDir::new("concurrent");
        let store = Arc::new(store(&dir, 4));

        let mut handles = Vec::new();
        for worker in 0..4 {
            let store = Arc::clone(&store);
            handles.push(std::thread::spawn(move || {
                let book = format!("book{worker}.txt");
                for i in 0..200 {
                    let meta = sample_metadata(&format!("w{worker}_{i}"), &book);
                    store
                        .insert_batch(&[(meta, vec![1.0, 0.0, 0.0, 0.0])])
                        .unwrap();
                    if i % 20 == 0 {
                        store.delete_book(&book).unwrap();
                        let _ = store.search(&[1.0, 0.0, 0.0, 0.0], 5, None).unwrap();
                    }
                }
            }));
        }
        for handle in handles {
            handle.join().expect("worker thread panicked or deadlocked");
        }

        // The two maps stayed in agreement.
        for worker in 0..4 {
            let book = format!("book{worker}.txt");
            let indexed = store.book_vector_count(&book);
            assert_eq!(
                indexed, 19,
                "book {book} should hold the 19 records added after the last delete"
            );
        }
        assert_eq!(store.vector_count(), 4 * 19);
    }
}
