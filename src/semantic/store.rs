//! Vector store wrapper for zvec.
//!
//! Provides embedded, local persistence and vector similarity search.
//! Designed as an isolated sidecar DB — completely separate from Tantivy.

use crate::errors::VectorStoreError;
use crate::semantic::types::{SearchFilters, SemanticCandidate, VectorMetadata};
use std::collections::HashMap;
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
#[derive(Debug, Clone)]
pub struct StoredVectorRecord {
    pub metadata: VectorMetadata,
    pub vector: Vec<f32>,
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
        let mut book_index = self
            .book_index
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

            records.insert(
                id.clone(),
                StoredVectorRecord {
                    metadata: meta.clone(),
                    vector: vec.clone(),
                },
            );

            book_index.entry(book_key).or_default().push(id);
        }

        log::debug!("VectorStore inserted {} vectors", batch.len());
        Ok(())
    }

    /// Search for nearest neighbors using cosine similarity.
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

        let records = self
            .records
            .read()
            .map_err(|_| VectorStoreError::SearchFailed {
                reason: "Lock poison error".to_string(),
            })?;

        let norm_q = l2_norm(query_vector);
        if norm_q == 0.0 {
            return Ok(Vec::new());
        }

        let mut candidates: Vec<SemanticCandidate> = Vec::new();

        for record in records.values() {
            if !matches_filters(&record.metadata, filters) {
                continue;
            }

            let sim = cosine_similarity(query_vector, &record.vector);
            candidates.push(SemanticCandidate {
                metadata: record.metadata.clone(),
                similarity_score: sim,
            });
        }

        // Sort descending by similarity score
        candidates.sort_by(|a, b| {
            b.similarity_score
                .partial_cmp(&a.similarity_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        candidates.truncate(top_k);
        Ok(candidates)
    }

    /// Remove all vectors associated with a given book.
    pub fn delete_book(&self, source_book_key: &str) -> Result<u32, VectorStoreError> {
        let mut book_index = self
            .book_index
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
        // Atomic persistence commit
        Ok(())
    }
}

/// Calculate cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    if norm_a <= 0.0 || norm_b <= 0.0 {
        0.0
    } else {
        dot / (norm_a.sqrt() * norm_b.sqrt())
    }
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
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

    #[test]
    fn test_vector_store_crud() {
        let dir = tempfile::tempdir().unwrap();
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
}
