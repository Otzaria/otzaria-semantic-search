//! Semantic search engine orchestrator.
//!
//! Integrates chunking, embedding, vector storage, and manifest tracking into
//! a cohesive semantic indexing and retrieval subsystem.

use crate::errors::SemanticSearchError;
use crate::semantic::chunker::{Chunker, ChunkerConfig};
use crate::semantic::embedding::{EmbeddingConfig, EmbeddingRuntime};
use crate::semantic::manifest::{ManifestConfig, SemanticManifest};
use crate::semantic::store::{VectorStore, VectorStoreConfig};
use crate::semantic::types::{
    BookForIndexing, IndexDiff, SearchFilters, SemanticCandidate, SemanticStatus, VectorMetadata,
};
use std::collections::HashMap;
use std::path::PathBuf;

/// Master configuration for the SemanticEngine.
#[derive(Debug, Clone)]
pub struct SemanticConfig {
    pub root_dir: PathBuf,
    pub embedding_model_id: String,
    pub embedding_dim: u32,
    pub model_path: PathBuf,
    pub chunking: ChunkerConfig,
    pub store: VectorStoreConfig,
}

impl Default for SemanticConfig {
    fn default() -> Self {
        let root = PathBuf::from("semantic_db");
        Self {
            root_dir: root.clone(),
            embedding_model_id: "EMD123/Otzaria-Embedding-V1-Flash-0.6B".to_string(),
            embedding_dim: 1024,
            model_path: PathBuf::from("models/otzaria-embedding-v1-flash-q4.gguf"),
            chunking: ChunkerConfig::default(),
            store: VectorStoreConfig {
                db_path: root.join("zvec"),
                embedding_dim: 1024,
                collection_name: "chunks".to_string(),
            },
        }
    }
}

/// The main semantic search engine sidecar.
pub struct SemanticEngine {
    config: SemanticConfig,
    manifest: SemanticManifest,
    chunker: Chunker,
    store: VectorStore,
    runtime: Option<EmbeddingRuntime>,
    last_error: Option<String>,
}

impl SemanticEngine {
    /// Initialize or open existing semantic search engine.
    pub fn open(config: SemanticConfig) -> Result<Self, SemanticSearchError> {
        let manifest_cfg = ManifestConfig {
            embedding_model_id: config.embedding_model_id.clone(),
            embedding_dim: config.embedding_dim,
            pooling: "last-token".to_string(),
            model_quantization: "Q4".to_string(),
            vector_precision: "f32".to_string(),
            chunking_version: config.chunking.chunking_version,
            normalization_version: 1,
        };

        let manifest = match SemanticManifest::load(&config.root_dir) {
            Ok(m) => {
                let mismatches = m.validate(&manifest_cfg);
                if !mismatches.is_empty() {
                    log::warn!(
                        "Manifest mismatch detected: {:?}. Re-index recommended.",
                        mismatches
                    );
                }
                m
            }
            Err(_) => SemanticManifest::new(&manifest_cfg),
        };

        let store = VectorStore::open_or_create(config.store.clone())?;
        let chunker = Chunker::new(config.chunking.clone());

        let engine = Self {
            config,
            manifest,
            chunker,
            store,
            runtime: None,
            last_error: None,
        };

        Ok(engine)
    }

    /// Load embedding runtime model into memory.
    pub fn load_model(&mut self) -> Result<(), SemanticSearchError> {
        if self.runtime.is_none() {
            let embed_config = EmbeddingConfig {
                model_path: self.config.model_path.clone(),
                embedding_dim: self.config.embedding_dim,
                pooling: "last-token".to_string(),
                ..Default::default()
            };
            let mut rt = EmbeddingRuntime::new(embed_config);
            if let Err(e) = rt.load() {
                self.last_error = Some(e.to_string());
                return Err(SemanticSearchError::EmbeddingRuntime(e));
            }
            self.runtime = Some(rt);
        }
        Ok(())
    }

    /// Unload model from memory to conserve resources.
    pub fn unload_model(&mut self) {
        self.runtime = None;
        log::info!("Unloaded embedding model from memory");
    }

    /// Index a book into the semantic vector store.
    pub fn index_book(&mut self, book: &BookForIndexing) -> Result<u32, SemanticSearchError> {
        if self.runtime.is_none() {
            self.load_model()?;
        }

        let chunks = self.chunker.chunk_book(book);
        if chunks.is_empty() {
            return Ok(0);
        }

        let Some(runtime) = self.runtime.as_ref() else {
            return Err(SemanticSearchError::Config(
                "Embedding runtime model failed to initialize".to_string(),
            ));
        };

        let mut batch: Vec<(VectorMetadata, Vec<f32>)> = Vec::with_capacity(chunks.len());
        for chunk in &chunks {
            let vec = runtime.embed_one(&chunk.embedding_text)?;
            let meta = VectorMetadata::from(chunk);
            batch.push((meta, vec));
        }

        self.store.insert_batch(&batch)?;

        self.manifest.mark_book_indexed(
            book.source_book_key.clone(),
            book.content_hash,
            chunks.len() as u32,
            self.config.chunking.chunking_version,
            1,
        );

        self.manifest.save(&self.config.root_dir)?;
        Ok(chunks.len() as u32)
    }

    /// Remove a book from the semantic index.
    pub fn remove_book(&mut self, source_book_key: &str) -> Result<u32, SemanticSearchError> {
        let count = self.store.delete_book(source_book_key)?;
        self.manifest.remove_book(source_book_key);
        self.manifest.save(&self.config.root_dir)?;
        Ok(count)
    }

    /// Execute vector similarity search.
    pub fn search(
        &self,
        query: &str,
        top_k: usize,
        filters: Option<&SearchFilters>,
    ) -> Result<Vec<SemanticCandidate>, SemanticSearchError> {
        let Some(runtime) = &self.runtime else {
            return Err(SemanticSearchError::Config(
                "Embedding model is not loaded".to_string(),
            ));
        };

        let query_vec = runtime.embed_one(query)?;
        let results = self.store.search(&query_vec, top_k, filters)?;
        Ok(results)
    }

    /// Check differences between Tantivy book hashes and semantic index.
    pub fn diff_against_tantivy(&self, tantivy_books: &HashMap<String, u64>) -> IndexDiff {
        let mut new_books = Vec::new();
        let mut changed_books = Vec::new();

        for (book_key, &content_hash) in tantivy_books {
            if self.manifest.book_needs_reindex(
                book_key,
                content_hash,
                self.config.chunking.chunking_version,
                1,
            ) {
                if self.manifest.books.contains_key(book_key) {
                    changed_books.push(book_key.clone());
                } else {
                    new_books.push(book_key.clone());
                }
            }
        }

        let mut removed_books = Vec::new();
        for book_key in self.manifest.books.keys() {
            if !tantivy_books.contains_key(book_key) {
                removed_books.push(book_key.clone());
            }
        }

        IndexDiff {
            new_books,
            changed_books,
            removed_books,
            model_mismatch: false,
            chunking_mismatch: false,
            normalization_mismatch: false,
        }
    }

    /// Retrieve current operational status.
    pub fn status(&self) -> SemanticStatus {
        SemanticStatus {
            available: true,
            model_loaded: self.runtime.is_some(),
            indexed_book_count: self.manifest.book_count() as u32,
            vector_count: self.store.vector_count() as u32,
            model_id: self.config.embedding_model_id.clone(),
            embedding_dim: self.config.embedding_dim,
            last_error: self.last_error.clone(),
        }
    }
}
