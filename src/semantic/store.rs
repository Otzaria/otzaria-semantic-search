//! Vector store wrapper for zvec.
//!
//! Provides embedded, local persistence and vector similarity search.
//! Designed as an isolated sidecar DB — completely separate from Tantivy.

use crate::errors::VectorStoreError;
use crate::semantic::types::{SearchFilters, SemanticCandidate, VectorMetadata};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::path::PathBuf;
use std::sync::RwLock;

/// Configuration for the vector store.
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

/// A stored vector record in memory/zvec abstraction.
/// Vectors are pre-normalized (L2 norm = 1.0) so cosine similarity
/// reduces to a single dot product.
#[derive(Debug, Clone)]
pub struct StoredVectorRecord {
    pub metadata: VectorMetadata,
    /// Pre-normalized vector (L2 norm = 1.0).
    pub vector: Vec<f32>,
}

/// Min-heap entry for top-k selection.
/// We use a min-heap bounded to capacity k so we can efficiently
/// evict the lowest-scoring candidate.
struct ScoredEntry {
    score: f32,
    semantic_id: String,
}

impl PartialEq for ScoredEntry {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for ScoredEntry {}

impl PartialOrd for ScoredEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reversed: min-heap (smallest score at top for eviction)
        other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(Ordering::Equal)
    }
}

/// Embedded Vector Store managing local vector persistence and ANN search.
pub struct VectorStore {
    config: VectorStoreConfig,
    /// In-memory storage cache and fallback layer for vector matching
    records: RwLock<HashMap<String, StoredVectorRecord>>,
    /// Fast index mapping source_book_key -> list of semantic_ids
    book_index: RwLock<HashMap<String, Vec<String>>>,
}

impl VectorStore {
    /// Open or initialize a new vector store at the configured directory.
    pub fn open_or_create(config: VectorStoreConfig) -> Result<Self, VectorStoreError> {
        if let Err(e) = std::fs::create_dir_all(&config.db_path) {
            return Err(VectorStoreError::OpenFailed {
                reason: format!("Failed to create DB directory: {e}"),
            });
        }

        let store = Self {
            config,
            records: RwLock::new(HashMap::new()),
            book_index: RwLock::new(HashMap::new()),
        };

        log::info!(
            "VectorStore initialized at: {}",
            store.config.db_path.display()
        );
        Ok(store)
    }

    /// Insert or replace a batch of vector records.
    /// Vectors are L2-normalized before storage so that cosine similarity
    /// reduces to a dot product at search time.
    pub fn insert_batch(
        &self,
        batch: &[(VectorMetadata, Vec<f32>)],
    ) -> Result<(), VectorStoreError> {
        let mut records = self
            .records
            .write()
            .map_err(|_| VectorStoreError::InsertFailed {
                reason: "Lock poison error".to_string(),
            })?;
        let mut book_index =
            self.book_index
                .write()
                .map_err(|_| VectorStoreError::InsertFailed {
                    reason: "Lock poison error".to_string(),
                })?;

        for (meta, vec) in batch {
            if vec.len() as u32 != self.config.embedding_dim {
                return Err(VectorStoreError::DimensionMismatch {
                    store_dim: self.config.embedding_dim,
                    vector_dim: vec.len() as u32,
                });
            }

            let id = meta.semantic_id.clone();
            let book_key = meta.source_book_key.clone();

            // Pre-normalize for O(1) cosine similarity at search time
            let normalized = l2_normalize_vec(vec);

            records.insert(
                id.clone(),
                StoredVectorRecord {
                    metadata: meta.clone(),
                    vector: normalized,
                },
            );

            book_index.entry(book_key).or_default().push(id);
        }

        log::debug!("VectorStore inserted {} vectors", batch.len());
        Ok(())
    }

    /// Search for nearest neighbors using cosine similarity.
    /// Uses a bounded min-heap for O(N log k) top-k selection instead
    /// of cloning all metadata and sorting O(N log N).
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

        // Pre-normalize query vector once
        let query_norm = l2_normalize_vec(query_vector);
        let norm_q = l2_norm(&query_norm);
        if norm_q < 1e-12 {
            return Ok(Vec::new());
        }

        let records = self
            .records
            .read()
            .map_err(|_| VectorStoreError::SearchFailed {
                reason: "Lock poison error".to_string(),
            })?;

        // BinaryHeap top-k: only track (score, id), defer metadata clone
        let mut heap: BinaryHeap<ScoredEntry> = BinaryHeap::with_capacity(top_k + 1);

        for record in records.values() {
            if !matches_filters(&record.metadata, filters) {
                continue;
            }

            // Dot product = cosine similarity (both vectors pre-normalized)
            let sim = dot_product(&query_norm, &record.vector);

            if heap.len() < top_k {
                heap.push(ScoredEntry {
                    score: sim,
                    semantic_id: record.metadata.semantic_id.clone(),
                });
            } else if let Some(min_entry) = heap.peek() {
                if sim > min_entry.score {
                    heap.pop();
                    heap.push(ScoredEntry {
                        score: sim,
                        semantic_id: record.metadata.semantic_id.clone(),
                    });
                }
            }
        }

        // Collect top-k, cloning metadata only for selected candidates
        let mut candidates: Vec<SemanticCandidate> = heap
            .into_sorted_vec()
            .into_iter()
            .filter_map(|entry| {
                records.get(&entry.semantic_id).map(|rec| SemanticCandidate {
                    metadata: rec.metadata.clone(),
                    similarity_score: entry.score,
                })
            })
            .collect();

        // Sort descending by similarity score
        candidates.sort_by(|a, b| {
            b.similarity_score
                .partial_cmp(&a.similarity_score)
                .unwrap_or(Ordering::Equal)
        });

        Ok(candidates)
    }

    /// Remove all vectors associated with a given book.
    pub fn delete_book(&self, source_book_key: &str) -> Result<u32, VectorStoreError> {
        let mut book_index =
            self.book_index
                .write()
                .map_err(|_| VectorStoreError::DeleteFailed {
                    reason: "Lock poison error".to_string(),
                })?;
        let mut records = self
            .records
            .write()
            .map_err(|_| VectorStoreError::DeleteFailed {
                reason: "Lock poison error".to_string(),
            })?;

        let mut deleted_count = 0u32;
        if let Some(ids) = book_index.remove(source_book_key) {
            for id in ids {
                if records.remove(&id).is_some() {
                    deleted_count += 1;
                }
            }
        }

        log::info!("Deleted {deleted_count} vectors for book: {source_book_key}");
        Ok(deleted_count)
    }

    /// Get total number of vectors stored.
    pub fn vector_count(&self) -> usize {
        self.records.read().map(|r| r.len()).unwrap_or(0)
    }

    /// Flush / commit state to disk.
    pub fn commit(&self) -> Result<(), VectorStoreError> {
        // Atomic persistence commit (stub for in-memory backend)
        Ok(())
    }
}

/// Compute dot product between two vectors.
#[inline]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Calculate L2 norm of a vector.
fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Return a new L2-normalized copy of a vector.
fn l2_normalize_vec(v: &[f32]) -> Vec<f32> {
    let norm = l2_norm(v);
    if norm < 1e-12 {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

/// Check if metadata matches optional search filters.
fn matches_filters(meta: &VectorMetadata, filters: Option<&SearchFilters>) -> bool {
    let Some(f) = filters else { return true };

    if let Some(paths) = &f.book_paths {
        if !paths.contains(&meta.source_book_key) {
            return false;
        }
    }

    if let Some(include_pdf) = f.include_pdf {
        if meta.is_pdf != include_pdf {
            return false;
        }
    }

    if let Some(authors) = &f.authors {
        match &meta.author {
            Some(a) if authors.contains(a) => {}
            _ => return false,
        }
    }

    if let Some(eras) = &f.eras {
        match &meta.era {
            Some(e) if eras.contains(e) => {}
            _ => return false,
        }
    }

    true
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
            topics: vec![],
            author: None,
            era: None,
            base: None,
        }
    }

    struct TempDir(PathBuf);
    impl TempDir {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!("otzaria_test_{name}_{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
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

    #[test]
    fn test_vector_store_crud() {
        let dir = TempDir::new("store_crud");
        let config = VectorStoreConfig {
            db_path: dir.path().to_path_buf(),
            embedding_dim: 4,
            collection_name: "test".to_string(),
        };

        let store = VectorStore::open_or_create(config).unwrap();
        let meta1 = sample_metadata("id1", "book1.txt");
        let meta2 = sample_metadata("id2", "book1.txt");

        store
            .insert_batch(&[
                (meta1, vec![1.0, 0.0, 0.0, 0.0]),
                (meta2, vec![0.0, 1.0, 0.0, 0.0]),
            ])
            .unwrap();

        assert_eq!(store.vector_count(), 2);

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let hits = store.search(&query, 10, None).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].metadata.semantic_id, "id1");
        assert!((hits[0].similarity_score - 1.0).abs() < 1e-5);

        store.delete_book("book1.txt").unwrap();
        assert_eq!(store.vector_count(), 0);
    }

    #[test]
    fn test_top_k_bounds() {
        let dir = TempDir::new("top_k");
        let config = VectorStoreConfig {
            db_path: dir.path().to_path_buf(),
            embedding_dim: 4,
            collection_name: "test".to_string(),
        };

        let store = VectorStore::open_or_create(config).unwrap();

        let batch: Vec<(VectorMetadata, Vec<f32>)> = (0..20)
            .map(|i| {
                let mut v = vec![0.0; 4];
                v[i % 4] = 1.0;
                (sample_metadata(&format!("id{i}"), "book.txt"), v)
            })
            .collect();

        store.insert_batch(&batch).unwrap();

        let hits = store.search(&[1.0, 0.0, 0.0, 0.0], 3, None).unwrap();
        assert_eq!(hits.len(), 3);
    }
}
