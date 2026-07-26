//! Semantic search engine orchestrator.
//!
//! Ties chunking, embedding, vector storage and manifest tracking into one
//! indexing and retrieval subsystem, with its own lifecycle: a failure here must
//! never take the lexical path down with it.
//!
//! # Index compatibility
//!
//! Vectors are only comparable to other vectors produced by the same model, the
//! same backend, the same pooling and the same dimensionality. When the on-disk
//! manifest disagrees with the current configuration, the engine does not
//! silently carry on — [`SemanticEngine::search`] and
//! [`SemanticEngine::index_book`] refuse with
//! [`SemanticSearchError::IncompatibleIndex`] (so the coordinator falls back to
//! BM25) until [`SemanticEngine::reset_index`] rebuilds from scratch.

use crate::errors::{ManifestError, SemanticSearchError};
use crate::semantic::chunker::{Chunker, ChunkerConfig};
use crate::semantic::embedding::{EmbeddingConfig, EmbeddingRuntime};
use crate::semantic::manifest::{
    describe_mismatches, ManifestConfig, ManifestMismatch, SemanticManifest,
};
use crate::semantic::store::{VectorStore, VectorStoreConfig};
use crate::semantic::types::{
    BookForIndexing, IndexDiff, SearchFilters, SemanticCandidate, SemanticChunk, SemanticStatus,
    VectorMetadata,
};
use std::collections::HashMap;
use std::path::PathBuf;

/// Master configuration for the [`SemanticEngine`].
#[derive(Debug, Clone)]
pub struct SemanticConfig {
    pub root_dir: PathBuf,
    pub embedding_model_id: String,
    pub embedding_dim: u32,
    pub model_path: PathBuf,
    /// Pooling strategy the model requires (e.g. `"last-token"`).
    pub pooling: String,
    /// Quantization of the model weights (e.g. `"Q4"`). Distinct from
    /// [`SemanticConfig::vector_precision`] — that is the stored vectors.
    pub model_quantization: String,
    /// Precision vectors are stored at (e.g. `"f32"`).
    pub vector_precision: String,
    /// Version of the text preprocessing applied before embedding. Bump it when
    /// preprocessing changes, so existing vectors are recognised as stale.
    pub normalization_version: u32,
    /// Texts handed to the embedding backend per inference call.
    pub embedding_batch_size: usize,
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
            pooling: "last-token".to_string(),
            model_quantization: "Q4".to_string(),
            vector_precision: "f32".to_string(),
            normalization_version: 1,
            embedding_batch_size: 32,
            chunking: ChunkerConfig::default(),
            store: VectorStoreConfig {
                db_path: root.join("vectors"),
                embedding_dim: 1024,
                collection_name: "chunks".to_string(),
            },
        }
    }
}

impl SemanticConfig {
    /// Reject configurations that cannot produce a working index.
    ///
    /// Catches the dimension disagreement in particular: `embedding_dim` and
    /// `store.embedding_dim` are set independently, and a mismatch would surface
    /// only as a `DimensionMismatch` on the first insert, i.e. mid-index.
    pub fn validate(&self) -> Result<(), SemanticSearchError> {
        if self.embedding_dim == 0 {
            return Err(SemanticSearchError::Config(
                "embedding_dim must be greater than zero".to_string(),
            ));
        }
        if self.store.embedding_dim != self.embedding_dim {
            return Err(SemanticSearchError::Config(format!(
                "embedding_dim ({}) does not match store.embedding_dim ({}); \
                 every stored vector must have the model's dimensionality",
                self.embedding_dim, self.store.embedding_dim
            )));
        }
        if self.embedding_batch_size == 0 {
            return Err(SemanticSearchError::Config(
                "embedding_batch_size must be greater than zero".to_string(),
            ));
        }
        if self.embedding_model_id.trim().is_empty() {
            return Err(SemanticSearchError::Config(
                "embedding_model_id must not be empty".to_string(),
            ));
        }
        Ok(())
    }
}

/// The main semantic search engine sidecar.
pub struct SemanticEngine {
    config: SemanticConfig,
    manifest: SemanticManifest,
    chunker: Chunker,
    store: VectorStore,
    runtime: Option<EmbeddingRuntime>,
    /// Non-empty when the persisted index was built with a configuration this
    /// one cannot use. While non-empty the semantic path is refused rather than
    /// answered with vectors from a different space.
    incompatibilities: Vec<ManifestMismatch>,
    last_error: Option<String>,
}

impl SemanticEngine {
    /// Initialize or open an existing semantic search index.
    ///
    /// Never fails because of a bad manifest: a corrupt or foreign-version file
    /// is moved aside and a fresh index is started, with the reason kept in
    /// [`SemanticStatus::last_error`]. It does fail on an invalid configuration
    /// or an unusable store directory, which are the caller's bugs to fix.
    pub fn open(config: SemanticConfig) -> Result<Self, SemanticSearchError> {
        config.validate()?;

        let store = VectorStore::open_or_create(config.store.clone())?;
        let chunker = Chunker::new(config.chunking.clone());

        let mut engine = Self {
            manifest: SemanticManifest::new(&manifest_config(&config, None, None)),
            config,
            chunker,
            store,
            runtime: None,
            incompatibilities: Vec::new(),
            last_error: None,
        };
        engine.open_manifest()?;

        Ok(engine)
    }

    /// Load the manifest, deciding between reuse, pruning and starting fresh.
    fn open_manifest(&mut self) -> Result<(), SemanticSearchError> {
        let expected = manifest_config(&self.config, None, None);

        match SemanticManifest::load(&self.config.root_dir) {
            Ok(manifest) => {
                let mismatches = manifest.validate(&expected);
                self.manifest = manifest;

                if !mismatches.is_empty() {
                    log::warn!(
                        "Semantic index is incompatible with the current configuration \
                         ({}). The semantic path is disabled until reset_index() and a \
                         re-index; lexical search is unaffected.",
                        describe_mismatches(&mismatches)
                    );
                    self.incompatibilities = mismatches;
                    return Ok(());
                }

                // The manifest survived the restart but the vectors it describes
                // did not. Left alone it would claim books are indexed while
                // every query returns nothing.
                if !self.store.is_persistent() && self.manifest.book_count() > 0 {
                    let dropped = self.manifest.clear_books();
                    log::info!(
                        "Vector backend '{}' does not persist; dropped {dropped} stale book \
                         record(s) from the manifest so they will be re-indexed",
                        self.store.backend_id()
                    );
                    self.manifest.save(&self.config.root_dir)?;
                }
                Ok(())
            }

            // First run: nothing to reconcile.
            Err(ManifestError::NotFound { .. }) => {
                self.manifest = SemanticManifest::new(&expected);
                self.manifest.save(&self.config.root_dir)?;
                Ok(())
            }

            // Unusable file. Keep it for diagnosis, start clean, and remember
            // why — an index that silently reset itself is hard to debug.
            Err(e) => {
                let tag = match e {
                    ManifestError::UnsupportedFormatVersion { .. } => "unsupported-version",
                    _ => "corrupt",
                };
                log::warn!("Cannot use the semantic manifest ({e}); starting a fresh index");

                match SemanticManifest::quarantine(&self.config.root_dir, tag) {
                    Ok(path) => {
                        self.last_error = Some(format!(
                            "manifest was unusable ({e}); moved to {} and the index was reset",
                            path.display()
                        ));
                    }
                    Err(move_err) => {
                        log::warn!("Could not move the unusable manifest aside: {move_err}");
                        self.last_error = Some(format!(
                            "manifest was unusable ({e}) and could not be moved aside \
                             ({move_err}); the index was reset"
                        ));
                    }
                }

                // The store is empty on a fresh open, but be explicit: a future
                // persistent backend must not keep vectors whose manifest is gone.
                self.store.clear()?;
                self.manifest = SemanticManifest::new(&expected);
                self.manifest.save(&self.config.root_dir)?;
                Ok(())
            }
        }
    }

    /// Load the embedding model into memory.
    ///
    /// Also completes manifest validation: the model's file checksum and backend
    /// are only knowable once it is loaded, so a model swapped behind an
    /// unchanged model id is detected here rather than at open.
    pub fn load_model(&mut self) -> Result<(), SemanticSearchError> {
        if self.runtime.is_some() {
            return Ok(());
        }

        let mut runtime = EmbeddingRuntime::new(EmbeddingConfig {
            model_path: self.config.model_path.clone(),
            embedding_dim: self.config.embedding_dim,
            pooling: self.config.pooling.clone(),
            batch_size: self.config.embedding_batch_size,
            ..Default::default()
        });

        if let Err(e) = runtime.load() {
            self.last_error = Some(e.to_string());
            return Err(SemanticSearchError::EmbeddingRuntime(e));
        }

        let checksum = runtime.model_checksum().map(str::to_string);
        let backend = runtime.backend().map(|b| b.id().to_string());
        self.runtime = Some(runtime);

        let expected = manifest_config(&self.config, checksum.clone(), backend.clone());
        let mismatches = self.manifest.validate(&expected);
        if !mismatches.is_empty() {
            log::warn!(
                "The loaded model does not match the one this index was built with ({}). \
                 The semantic path is disabled until reset_index() and a re-index.",
                describe_mismatches(&mismatches)
            );
            self.incompatibilities = mismatches;
            return Ok(());
        }

        // Compatible: record the identity so a later session can compare.
        let was_unknown =
            self.manifest.model_checksum.is_none() || self.manifest.embedding_backend.is_none();
        if was_unknown {
            self.manifest.set_model_identity(checksum, backend);
            self.manifest.save(&self.config.root_dir)?;
        }
        Ok(())
    }

    /// Unload the model to free memory. The index is untouched.
    pub fn unload_model(&mut self) {
        self.runtime = None;
        log::info!("Unloaded embedding model from memory");
    }

    /// Index a book, replacing anything previously indexed for it.
    ///
    /// Returns the number of chunks written. Saves the manifest, so a crash
    /// afterwards cannot leave the book looking indexed when it is not — use
    /// [`SemanticEngine::index_books`] to index many books with one manifest
    /// write.
    pub fn index_book(&mut self, book: &BookForIndexing) -> Result<u32, SemanticSearchError> {
        let written = self.index_book_inner(book)?;
        self.manifest.save(&self.config.root_dir)?;
        Ok(written)
    }

    /// Index several books, writing the manifest once at the end.
    ///
    /// Serializing the whole manifest after every book turns a full library
    /// index into quadratic I/O. On failure, everything indexed so far is still
    /// committed before the error propagates, so a retry resumes instead of
    /// restarting.
    pub fn index_books(&mut self, books: &[BookForIndexing]) -> Result<u32, SemanticSearchError> {
        let mut total = 0u32;
        for book in books {
            match self.index_book_inner(book) {
                Ok(written) => total = total.saturating_add(written),
                Err(e) => {
                    // Persist the progress made before giving up.
                    if let Err(save_err) = self.manifest.save(&self.config.root_dir) {
                        log::warn!(
                            "Could not save the manifest after an indexing failure: {save_err}"
                        );
                    }
                    return Err(e);
                }
            }
        }
        self.manifest.save(&self.config.root_dir)?;
        Ok(total)
    }

    /// Index one book without saving the manifest.
    fn index_book_inner(&mut self, book: &BookForIndexing) -> Result<u32, SemanticSearchError> {
        self.ensure_index_usable()?;

        let chunks = self.chunker.chunk_book(book);

        // Only pay for the model when there is something to embed. A book with
        // no embeddable lines still needs its stale vectors dropped below, and
        // that cleanup must not depend on a model being available.
        if !chunks.is_empty() {
            self.load_model()?;
            // Loading the model can itself uncover an incompatibility — the file
            // checksum is only knowable once it is read.
            self.ensure_index_usable()?;
        }

        // Every mutation happens after embedding succeeds. Deleting first would
        // leave a book with no vectors but an unchanged manifest entry if the
        // model failed halfway through.
        let vectors = self.embed_chunks(&chunks)?;

        let removed = self.store.delete_book(&book.source_book_key)?;
        if removed > 0 {
            log::debug!(
                "Replaced {removed} existing vector(s) for {}",
                book.source_book_key
            );
        }

        if chunks.is_empty() {
            // A book with no embeddable lines owns no vectors and no record.
            // Without this, lines deleted from a book would keep their vectors.
            self.store.commit()?;
            self.manifest.remove_book(&book.source_book_key);
            return Ok(0);
        }

        let batch: Vec<(VectorMetadata, Vec<f32>)> = chunks
            .iter()
            .map(VectorMetadata::from)
            .zip(vectors)
            .collect();

        if let Err(e) = self.store.insert_batch(&batch) {
            // Do not leave a manifest entry describing vectors that are not there.
            self.manifest.remove_book(&book.source_book_key);
            self.last_error = Some(e.to_string());
            return Err(e.into());
        }
        self.store.commit()?;

        self.manifest.mark_book_indexed(
            book.source_book_key.clone(),
            book.content_hash,
            chunks.len() as u32,
            self.config.chunking.chunking_version,
            self.config.normalization_version,
        );

        Ok(chunks.len() as u32)
    }

    /// Embed the chunks' texts in backend-sized batches.
    fn embed_chunks(&self, chunks: &[SemanticChunk]) -> Result<Vec<Vec<f32>>, SemanticSearchError> {
        if chunks.is_empty() {
            return Ok(Vec::new());
        }

        let Some(runtime) = self.runtime.as_ref() else {
            return Err(SemanticSearchError::Config(
                "Embedding runtime model failed to initialize".to_string(),
            ));
        };

        let texts: Vec<&str> = chunks.iter().map(|c| c.embedding_text.as_str()).collect();
        let vectors = runtime.embed_batch(&texts)?;

        if vectors.len() != chunks.len() {
            return Err(SemanticSearchError::Config(format!(
                "embedding backend returned {} vectors for {} chunks",
                vectors.len(),
                chunks.len()
            )));
        }
        Ok(vectors)
    }

    /// Remove a book from the semantic index. Returns how many vectors went.
    pub fn remove_book(&mut self, source_book_key: &str) -> Result<u32, SemanticSearchError> {
        let count = self.store.delete_book(source_book_key)?;
        self.store.commit()?;
        self.manifest.remove_book(source_book_key);
        self.manifest.save(&self.config.root_dir)?;
        Ok(count)
    }

    /// Drop the whole index and start a fresh manifest for the current
    /// configuration. Returns how many vectors were discarded.
    ///
    /// This is the recovery path out of [`SemanticSearchError::IncompatibleIndex`]:
    /// afterwards the engine is usable again and every book needs re-indexing.
    pub fn reset_index(&mut self) -> Result<u32, SemanticSearchError> {
        let removed = self.store.clear()?;
        self.store.commit()?;

        let checksum = self
            .runtime
            .as_ref()
            .and_then(|r| r.model_checksum())
            .map(str::to_string);
        let backend = self
            .runtime
            .as_ref()
            .and_then(|r| r.backend())
            .map(|b| b.id().to_string());

        self.manifest = SemanticManifest::new(&manifest_config(&self.config, checksum, backend));
        self.manifest.save(&self.config.root_dir)?;
        self.incompatibilities.clear();
        self.last_error = None;

        log::info!("Semantic index reset; {removed} vector(s) discarded");
        Ok(removed)
    }

    /// Execute a vector similarity search.
    pub fn search(
        &self,
        query: &str,
        top_k: usize,
        filters: Option<&SearchFilters>,
    ) -> Result<Vec<SemanticCandidate>, SemanticSearchError> {
        self.ensure_index_usable()?;

        let Some(runtime) = &self.runtime else {
            return Err(SemanticSearchError::Config(
                "Embedding model is not loaded".to_string(),
            ));
        };

        let query_vec = runtime.embed_one(query)?;
        Ok(self.store.search(&query_vec, top_k, filters)?)
    }

    /// Compare Tantivy's per-book content hashes against the semantic index.
    ///
    /// When a configuration change invalidated the index, every known book is
    /// reported as needing work — an incremental update cannot repair a change
    /// of model or chunking.
    pub fn diff_against_tantivy(&self, tantivy_books: &HashMap<String, u64>) -> IndexDiff {
        let model_mismatch = self
            .incompatibilities
            .iter()
            .any(ManifestMismatch::invalidates_vectors);
        let chunking_mismatch = self
            .incompatibilities
            .iter()
            .any(|m| matches!(m, ManifestMismatch::ChunkingVersion { .. }));
        let normalization_mismatch = self
            .incompatibilities
            .iter()
            .any(|m| matches!(m, ManifestMismatch::NormalizationVersion { .. }));
        let full_rebuild = model_mismatch || chunking_mismatch || normalization_mismatch;

        let mut new_books = Vec::new();
        let mut changed_books = Vec::new();

        for (book_key, &content_hash) in tantivy_books {
            let known = self.manifest.books.contains_key(book_key);
            let needs_work = full_rebuild
                || self.manifest.book_needs_reindex(
                    book_key,
                    content_hash,
                    self.config.chunking.chunking_version,
                    self.config.normalization_version,
                );

            if needs_work {
                if known {
                    changed_books.push(book_key.clone());
                } else {
                    new_books.push(book_key.clone());
                }
            }
        }

        let mut removed_books: Vec<String> = self
            .manifest
            .books
            .keys()
            .filter(|book_key| !tantivy_books.contains_key(*book_key))
            .cloned()
            .collect();

        // Deterministic order: the caller may show these lists or drive progress
        // from them, and `HashMap` iteration order changes between runs.
        new_books.sort_unstable();
        changed_books.sort_unstable();
        removed_books.sort_unstable();

        IndexDiff {
            new_books,
            changed_books,
            removed_books,
            model_mismatch,
            chunking_mismatch,
            normalization_mismatch,
        }
    }

    /// Retrieve current operational status.
    pub fn status(&self) -> SemanticStatus {
        let vector_count = self.store.vector_count() as u32;
        let needs_full_reindex = (!self.incompatibilities.is_empty())
            .then(|| describe_mismatches(&self.incompatibilities));

        SemanticStatus {
            available: needs_full_reindex.is_none() && self.runtime.is_some() && vector_count > 0,
            model_loaded: self.runtime.is_some(),
            indexed_book_count: self.manifest.book_count() as u32,
            vector_count,
            model_id: self.config.embedding_model_id.clone(),
            embedding_dim: self.config.embedding_dim,
            embedding_backend: self
                .runtime
                .as_ref()
                .and_then(|r| r.backend())
                .map(|b| b.id().to_string()),
            vector_backend: self.store.backend_id().to_string(),
            vectors_persisted: self.store.is_persistent(),
            needs_full_reindex,
            last_error: self.last_error.clone(),
        }
    }

    /// The configuration this engine was opened with.
    pub fn config(&self) -> &SemanticConfig {
        &self.config
    }

    /// Mismatches that currently disable the semantic path, if any.
    pub fn incompatibilities(&self) -> &[ManifestMismatch] {
        &self.incompatibilities
    }

    /// Refuse the operation when the persisted index cannot be used.
    fn ensure_index_usable(&self) -> Result<(), SemanticSearchError> {
        if self.incompatibilities.is_empty() {
            return Ok(());
        }
        Err(SemanticSearchError::IncompatibleIndex {
            details: describe_mismatches(&self.incompatibilities),
        })
    }
}

/// Build the manifest-comparison config from the engine configuration.
///
/// `model_checksum` and `embedding_backend` are `None` until a model is loaded;
/// see [`ManifestConfig`].
fn manifest_config(
    config: &SemanticConfig,
    model_checksum: Option<String>,
    embedding_backend: Option<String>,
) -> ManifestConfig {
    ManifestConfig {
        embedding_model_id: config.embedding_model_id.clone(),
        model_checksum,
        embedding_backend,
        embedding_dim: config.embedding_dim,
        pooling: config.pooling.clone(),
        model_quantization: config.model_quantization.clone(),
        vector_precision: config.vector_precision.clone(),
        vector_backend: crate::semantic::store::BACKEND_ID.to_string(),
        chunking_version: config.chunking.chunking_version,
        normalization_version: config.normalization_version,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::EmbeddingError;
    use crate::semantic::embedding::mock;
    use crate::semantic::types::BookLine;

    struct TempDir(PathBuf);
    impl TempDir {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "otzaria_engine_test_{name}_{}",
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

    /// A config rooted at `dir` with a valid stub model and a small dimension so
    /// the tests stay fast.
    fn config_at(dir: &TempDir) -> SemanticConfig {
        let model_path = dir.path().join("model.gguf");
        mock::write_stub_gguf(&model_path, 3).unwrap();

        let root = dir.path().join("semantic");
        SemanticConfig {
            root_dir: root.clone(),
            model_path,
            embedding_dim: 64,
            store: VectorStoreConfig {
                db_path: root.join("vectors"),
                embedding_dim: 64,
                collection_name: "chunks".to_string(),
            },
            ..Default::default()
        }
    }

    fn book(key: &str, content_hash: u64, lines: &[(u64, u64, &str)]) -> BookForIndexing {
        BookForIndexing {
            source_book_key: key.to_string(),
            title: "ספר בדיקה".to_string(),
            content_hash,
            is_pdf: false,
            topics: vec!["/מקרא/תורה".to_string()],
            author: Some("מחבר".to_string()),
            era: Some("תנך".to_string()),
            base: None,
            lines: lines
                .iter()
                .map(|&(line_id, section_id, text)| BookLine {
                    line_id,
                    section_id,
                    text: text.to_string(),
                    line_hash: line_id * 1000,
                    reference: format!("הפניה {line_id}"),
                    segment: line_id,
                })
                .collect(),
        }
    }

    fn three_line_book() -> BookForIndexing {
        book(
            "otzaria/tanach/genesis.txt",
            111,
            &[
                (1, 100, "בראשית ברא אלהים את השמים ואת הארץ"),
                (2, 100, "והארץ היתה תהו ובהו וחשך על פני תהום"),
                (3, 101, "ויאמר אלהים יהי אור ויהי אור מאיר"),
            ],
        )
    }

    // ── configuration ──

    #[test]
    fn a_dimension_disagreement_between_model_and_store_is_rejected_at_open() {
        let dir = TempDir::new("dim_disagreement");
        let mut config = config_at(&dir);
        config.store.embedding_dim = 128; // model says 64

        match SemanticEngine::open(config) {
            Err(SemanticSearchError::Config(msg)) => {
                assert!(msg.contains("embedding_dim"), "unhelpful message: {msg}");
            }
            Err(other) => panic!("expected a config error, got {other}"),
            Ok(_) => panic!("a dimension disagreement must not be accepted"),
        }
    }

    #[test]
    fn other_invalid_configurations_are_rejected() {
        let dir = TempDir::new("invalid_config");

        let mut zero_dim = config_at(&dir);
        zero_dim.embedding_dim = 0;
        zero_dim.store.embedding_dim = 0;
        assert!(SemanticEngine::open(zero_dim).is_err());

        let mut zero_batch = config_at(&dir);
        zero_batch.embedding_batch_size = 0;
        assert!(SemanticEngine::open(zero_batch).is_err());

        let mut no_model_id = config_at(&dir);
        no_model_id.embedding_model_id = "  ".to_string();
        assert!(SemanticEngine::open(no_model_id).is_err());
    }

    // ── indexing ──

    #[test]
    fn indexing_writes_one_vector_per_embeddable_line() {
        let dir = TempDir::new("index_basic");
        let mut engine = SemanticEngine::open(config_at(&dir)).unwrap();

        let written = engine.index_book(&three_line_book()).unwrap();
        assert_eq!(written, 3);

        let status = engine.status();
        assert_eq!(status.indexed_book_count, 1);
        assert_eq!(status.vector_count, 3);
        assert!(status.model_loaded);
        assert!(status.available);
        assert_eq!(status.embedding_backend.as_deref(), Some("mock-hash-v1"));
        assert!(!status.vectors_persisted);
        assert!(status.needs_full_reindex.is_none());
    }

    #[test]
    fn indexing_is_idempotent_for_unchanged_content() {
        let dir = TempDir::new("idempotent");
        let mut engine = SemanticEngine::open(config_at(&dir)).unwrap();
        let book = three_line_book();

        engine.index_book(&book).unwrap();
        engine.index_book(&book).unwrap();

        assert_eq!(engine.status().vector_count, 3);
        assert_eq!(engine.status().indexed_book_count, 1);
    }

    /// The bug: a re-index inserted the new chunks without removing the old
    /// ones, so a line deleted from a book kept its vector and kept being
    /// returned as a result.
    #[test]
    fn reindexing_a_shrunk_book_drops_the_vectors_of_deleted_lines() {
        let dir = TempDir::new("reindex_shrink");
        let mut engine = SemanticEngine::open(config_at(&dir)).unwrap();

        engine.index_book(&three_line_book()).unwrap();
        assert_eq!(engine.status().vector_count, 3);

        // The third line was deleted from the book and its content hash changed.
        let shrunk = book(
            "otzaria/tanach/genesis.txt",
            222,
            &[
                (1, 100, "בראשית ברא אלהים את השמים ואת הארץ"),
                (2, 100, "והארץ היתה תהו ובהו וחשך על פני תהום"),
            ],
        );
        assert_eq!(engine.index_book(&shrunk).unwrap(), 2);

        assert_eq!(
            engine.status().vector_count,
            2,
            "the deleted line's vector must be gone"
        );
        assert_eq!(engine.status().indexed_book_count, 1);

        // And the manifest agrees the book is now current.
        let mut tantivy = HashMap::new();
        tantivy.insert("otzaria/tanach/genesis.txt".to_string(), 222u64);
        assert!(engine.diff_against_tantivy(&tantivy).is_up_to_date());
    }

    #[test]
    fn reindexing_a_book_that_lost_every_line_removes_it_from_the_index() {
        let dir = TempDir::new("reindex_emptied");
        let mut engine = SemanticEngine::open(config_at(&dir)).unwrap();

        engine.index_book(&three_line_book()).unwrap();
        let emptied = book("otzaria/tanach/genesis.txt", 333, &[]);
        assert_eq!(engine.index_book(&emptied).unwrap(), 0);

        let status = engine.status();
        assert_eq!(status.vector_count, 0, "no orphan vectors may survive");
        assert_eq!(
            status.indexed_book_count, 0,
            "a book with nothing to index owns no manifest record"
        );
    }

    /// Cleaning up a book that has nothing left to embed must not require a
    /// working model — otherwise a missing model file would strand its vectors.
    #[test]
    fn indexing_a_book_with_no_embeddable_lines_needs_no_model() {
        let dir = TempDir::new("empty_book_no_model");
        let mut config = config_at(&dir);
        config.model_path = dir.path().join("absent.gguf");

        let mut engine = SemanticEngine::open(config).unwrap();
        // Lines too short to embed at all.
        let blank = book("blank.txt", 1, &[(1, 1, "א"), (2, 1, "   ")]);

        assert_eq!(engine.index_book(&blank).unwrap(), 0);
        assert!(!engine.status().model_loaded);
        assert_eq!(engine.status().indexed_book_count, 0);
    }

    #[test]
    fn reindexing_one_book_leaves_other_books_alone() {
        let dir = TempDir::new("reindex_isolation");
        let mut engine = SemanticEngine::open(config_at(&dir)).unwrap();

        engine.index_book(&three_line_book()).unwrap();
        engine
            .index_book(&book(
                "otzaria/mishna/berachot.txt",
                777,
                &[(10, 1, "מאימתי קורין את שמע בערבית משעה שהכהנים נכנסין")],
            ))
            .unwrap();
        assert_eq!(engine.status().vector_count, 4);

        let shrunk = book(
            "otzaria/tanach/genesis.txt",
            222,
            &[(1, 100, "בראשית ברא אלהים את השמים ואת הארץ")],
        );
        engine.index_book(&shrunk).unwrap();

        assert_eq!(engine.status().vector_count, 2);
        assert_eq!(engine.status().indexed_book_count, 2);
    }

    #[test]
    fn removing_a_book_drops_its_vectors_and_its_record() {
        let dir = TempDir::new("remove_book");
        let mut engine = SemanticEngine::open(config_at(&dir)).unwrap();
        engine.index_book(&three_line_book()).unwrap();

        assert_eq!(engine.remove_book("otzaria/tanach/genesis.txt").unwrap(), 3);
        assert_eq!(engine.status().vector_count, 0);
        assert_eq!(engine.status().indexed_book_count, 0);

        // Removing an unknown book is a no-op, not an error.
        assert_eq!(engine.remove_book("not/indexed.txt").unwrap(), 0);
    }

    #[test]
    fn index_books_writes_the_manifest_once_and_indexes_everything() {
        let dir = TempDir::new("batch_index");
        let mut engine = SemanticEngine::open(config_at(&dir)).unwrap();

        let books: Vec<BookForIndexing> = (0..5)
            .map(|i| {
                book(
                    &format!("otzaria/book{i}.txt"),
                    i as u64,
                    &[
                        (1, 1, "שורה ראשונה עם מספיק תווים כדי לעמוד בפני עצמה"),
                        (2, 1, "שורה שנייה עם מספיק תווים כדי לעמוד בפני עצמה"),
                    ],
                )
            })
            .collect();

        assert_eq!(engine.index_books(&books).unwrap(), 10);
        assert_eq!(engine.status().vector_count, 10);
        assert_eq!(engine.status().indexed_book_count, 5);

        // The single manifest write covered every book.
        let reloaded = SemanticManifest::load(&engine.config.root_dir).unwrap();
        assert_eq!(reloaded.book_count(), 5);
        assert_eq!(reloaded.total_chunk_count(), 10);
    }

    #[test]
    fn indexing_uses_the_batch_embedding_path_and_matches_single_embeddings() {
        let dir = TempDir::new("batch_equivalence");
        let mut engine = SemanticEngine::open(config_at(&dir)).unwrap();
        let book = three_line_book();
        engine.index_book(&book).unwrap();

        // Query with the exact text of a line: a batched embedding must be
        // identical to the single-text one, or the top hit would not be exact.
        let hits = engine.search(&book.lines[1].text, 3, None).unwrap();
        assert!(!hits.is_empty());
        assert!(
            (hits[0].similarity_score - 1.0).abs() < 1e-5,
            "expected an exact self-match, got {}",
            hits[0].similarity_score
        );
        assert_eq!(hits[0].metadata.line_id, 2);
    }

    // ── search ──

    #[test]
    fn search_before_a_model_is_loaded_fails_without_touching_the_index() {
        let dir = TempDir::new("search_no_model");
        let engine = SemanticEngine::open(config_at(&dir)).unwrap();

        assert!(matches!(
            engine.search("בריאת העולם", 5, None),
            Err(SemanticSearchError::Config(_))
        ));
        assert!(!engine.status().available);
    }

    #[test]
    fn search_applies_filters() {
        let dir = TempDir::new("search_filters");
        let mut engine = SemanticEngine::open(config_at(&dir)).unwrap();
        engine.index_book(&three_line_book()).unwrap();

        let query = "בראשית ברא אלהים את השמים ואת הארץ";
        assert_eq!(engine.search(query, 10, None).unwrap().len(), 3);

        let other_book = SearchFilters {
            book_paths: Some(vec!["otzaria/mishna/berachot.txt".to_string()]),
            ..Default::default()
        };
        assert!(engine
            .search(query, 10, Some(&other_book))
            .unwrap()
            .is_empty());

        let by_topic = SearchFilters {
            topics: Some(vec!["/מקרא".to_string()]),
            ..Default::default()
        };
        assert_eq!(engine.search(query, 10, Some(&by_topic)).unwrap().len(), 3);
    }

    #[test]
    fn a_whitespace_only_query_fails_instead_of_returning_arbitrary_results() {
        let dir = TempDir::new("blank_query");
        let mut engine = SemanticEngine::open(config_at(&dir)).unwrap();
        engine.index_book(&three_line_book()).unwrap();

        assert!(matches!(
            engine.search("   ", 5, None),
            Err(SemanticSearchError::EmbeddingRuntime(
                EmbeddingError::InferenceFailed { .. }
            ))
        ));
    }

    // ── model loading ──

    #[test]
    fn a_missing_model_file_fails_loudly_and_is_recorded() {
        let dir = TempDir::new("missing_model");
        let mut config = config_at(&dir);
        config.model_path = dir.path().join("absent.gguf");

        let mut engine = SemanticEngine::open(config).unwrap();
        assert!(matches!(
            engine.load_model(),
            Err(SemanticSearchError::EmbeddingRuntime(
                EmbeddingError::ModelNotFound { .. }
            ))
        ));

        let status = engine.status();
        assert!(!status.model_loaded);
        assert!(!status.available);
        assert!(status.last_error.is_some());
    }

    #[test]
    fn loading_the_model_records_its_identity_in_the_manifest() {
        let dir = TempDir::new("model_identity");
        let config = config_at(&dir);
        let mut engine = SemanticEngine::open(config.clone()).unwrap();
        engine.load_model().unwrap();

        let manifest = SemanticManifest::load(&config.root_dir).unwrap();
        assert_eq!(manifest.embedding_backend.as_deref(), Some("mock-hash-v1"));
        assert_eq!(
            manifest.model_checksum.as_ref().map(String::len),
            Some(64),
            "the model's SHA-256 must be recorded"
        );
        assert!(engine.incompatibilities().is_empty());
    }

    /// Same model id, different weights behind it. Only the checksum catches
    /// this, and it must be caught: the old vectors are from another space.
    #[test]
    fn a_swapped_model_file_disables_the_semantic_path() {
        let dir = TempDir::new("swapped_model");
        let config = config_at(&dir);

        {
            let mut engine = SemanticEngine::open(config.clone()).unwrap();
            engine.index_book(&three_line_book()).unwrap();
        }

        // Replace the model file's contents, keeping its id and path.
        let mut bytes = std::fs::read(&config.model_path).unwrap();
        bytes.extend_from_slice(b"different weights entirely");
        std::fs::write(&config.model_path, bytes).unwrap();

        let mut engine = SemanticEngine::open(config).unwrap();
        engine.load_model().unwrap();

        assert!(
            engine
                .incompatibilities()
                .iter()
                .any(|m| matches!(m, ManifestMismatch::ModelChecksum { .. })),
            "got {:?}",
            engine.incompatibilities()
        );
        assert!(matches!(
            engine.search("בריאה", 5, None),
            Err(SemanticSearchError::IncompatibleIndex { .. })
        ));
        assert!(engine.status().needs_full_reindex.is_some());
    }

    // ── reopen ──

    /// With a non-persistent store the vectors are gone after a restart. A
    /// manifest still claiming the books are indexed is the dangerous state:
    /// the diff reports "nothing to do" while every query comes back empty.
    #[test]
    fn reopening_prunes_book_records_whose_vectors_did_not_survive() {
        let dir = TempDir::new("reopen_prune");
        let config = config_at(&dir);

        {
            let mut engine = SemanticEngine::open(config.clone()).unwrap();
            engine.index_book(&three_line_book()).unwrap();
            assert_eq!(engine.status().indexed_book_count, 1);
            assert_eq!(engine.status().vector_count, 3);
        }

        let engine = SemanticEngine::open(config).unwrap();
        let status = engine.status();
        assert_eq!(status.vector_count, 0, "the in-memory store starts empty");
        assert_eq!(
            status.indexed_book_count, 0,
            "the manifest must not claim books whose vectors are gone"
        );
        assert!(!status.available);
        assert!(
            status.needs_full_reindex.is_none(),
            "an empty index is not an incompatible one"
        );

        // The diff therefore asks for the book again instead of reporting done.
        let mut tantivy = HashMap::new();
        tantivy.insert("otzaria/tanach/genesis.txt".to_string(), 111u64);
        let diff = engine.diff_against_tantivy(&tantivy);
        assert!(!diff.is_up_to_date());
        assert_eq!(diff.new_books, vec!["otzaria/tanach/genesis.txt"]);
        assert!(diff.changed_books.is_empty());
        assert!(!diff.needs_full_rebuild());
    }

    #[test]
    fn reopening_preserves_the_configuration_metadata_and_stays_compatible() {
        let dir = TempDir::new("reopen_config");
        let config = config_at(&dir);

        let created_at = {
            let mut engine = SemanticEngine::open(config.clone()).unwrap();
            engine.load_model().unwrap();
            engine.manifest.created_at
        };

        let engine = SemanticEngine::open(config.clone()).unwrap();
        assert!(
            engine.incompatibilities().is_empty(),
            "reopening with the same configuration must not look incompatible"
        );
        assert_eq!(
            engine.manifest.created_at, created_at,
            "the manifest must be reused, not recreated"
        );
        assert_eq!(
            engine.manifest.embedding_model_id,
            config.embedding_model_id
        );
        assert!(engine.status().last_error.is_none());
    }

    #[test]
    fn reopening_and_reindexing_restores_a_working_index() {
        let dir = TempDir::new("reopen_reindex");
        let config = config_at(&dir);

        {
            let mut engine = SemanticEngine::open(config.clone()).unwrap();
            engine.index_book(&three_line_book()).unwrap();
        }

        let mut engine = SemanticEngine::open(config).unwrap();
        engine.index_book(&three_line_book()).unwrap();

        let status = engine.status();
        assert!(status.available);
        assert_eq!(status.vector_count, 3);
        assert_eq!(status.indexed_book_count, 1);
    }

    // ── manifest compatibility ──

    #[test]
    fn a_changed_dimension_disables_the_semantic_path_until_reset() {
        let dir = TempDir::new("dim_change");
        let config = config_at(&dir);

        {
            let mut engine = SemanticEngine::open(config.clone()).unwrap();
            engine.index_book(&three_line_book()).unwrap();
        }

        let mut smaller = config.clone();
        smaller.embedding_dim = 32;
        smaller.store.embedding_dim = 32;

        let mut engine = SemanticEngine::open(smaller).unwrap();
        assert!(engine
            .incompatibilities()
            .iter()
            .any(|m| matches!(m, ManifestMismatch::Dimensions { .. })));

        // Both reading and writing are refused, with a reason.
        assert!(matches!(
            engine.search("בריאה", 5, None),
            Err(SemanticSearchError::IncompatibleIndex { .. })
        ));
        assert!(matches!(
            engine.index_book(&three_line_book()),
            Err(SemanticSearchError::IncompatibleIndex { .. })
        ));

        let status = engine.status();
        assert!(!status.available);
        let reason = status
            .needs_full_reindex
            .expect("a reason must be reported");
        assert!(reason.contains("Dimensions"), "unhelpful reason: {reason}");

        // Reset is the way out.
        engine.reset_index().unwrap();
        assert!(engine.incompatibilities().is_empty());
        assert_eq!(engine.index_book(&three_line_book()).unwrap(), 3);
        assert!(engine.status().available);
        assert!(engine.status().needs_full_reindex.is_none());
    }

    #[test]
    fn a_changed_chunking_version_forces_every_book_to_be_reindexed() {
        let dir = TempDir::new("chunking_change");
        let config = config_at(&dir);

        {
            let mut engine = SemanticEngine::open(config.clone()).unwrap();
            engine.index_book(&three_line_book()).unwrap();
        }

        let mut bumped = config;
        bumped.chunking.chunking_version = 2;
        let engine = SemanticEngine::open(bumped).unwrap();

        let mut tantivy = HashMap::new();
        tantivy.insert("otzaria/tanach/genesis.txt".to_string(), 111u64);
        let diff = engine.diff_against_tantivy(&tantivy);

        assert!(diff.chunking_mismatch);
        assert!(!diff.model_mismatch);
        assert!(diff.needs_full_rebuild());
        assert!(!diff.is_up_to_date());
        assert_eq!(diff.books_to_index(), 1);
    }

    #[test]
    fn a_changed_model_id_is_reported_as_a_model_mismatch() {
        let dir = TempDir::new("model_id_change");
        let config = config_at(&dir);

        {
            let mut engine = SemanticEngine::open(config.clone()).unwrap();
            engine.index_book(&three_line_book()).unwrap();
        }

        let mut renamed = config;
        renamed.embedding_model_id = "EMD123/Some-Other-Model".to_string();
        let engine = SemanticEngine::open(renamed).unwrap();

        let diff = engine.diff_against_tantivy(&HashMap::new());
        assert!(diff.model_mismatch);
        assert!(diff.needs_full_rebuild());
    }

    #[test]
    fn a_corrupt_manifest_is_quarantined_and_the_index_starts_fresh() {
        let dir = TempDir::new("corrupt_manifest");
        let config = config_at(&dir);

        {
            let mut engine = SemanticEngine::open(config.clone()).unwrap();
            engine.index_book(&three_line_book()).unwrap();
        }

        // Simulate a truncated write (the pre-fsync failure mode).
        std::fs::write(
            SemanticManifest::file_path(&config.root_dir),
            b"{\"format_version\": 2, \"embedding_mod",
        )
        .unwrap();

        let mut engine = SemanticEngine::open(config.clone()).unwrap();
        let status = engine.status();
        assert_eq!(status.indexed_book_count, 0);
        assert_eq!(status.vector_count, 0);
        assert!(
            status.last_error.is_some(),
            "a silent reset is untraceable; the reason must be reported"
        );
        assert!(engine.incompatibilities().is_empty());

        // The broken file is kept for diagnosis, not deleted.
        let quarantined: Vec<_> = std::fs::read_dir(&config.root_dir)
            .unwrap()
            .filter_map(Result::ok)
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|name| name.contains("corrupt"))
            .collect();
        assert_eq!(quarantined.len(), 1, "found {quarantined:?}");

        // And the engine is fully usable again.
        assert_eq!(engine.index_book(&three_line_book()).unwrap(), 3);
        assert!(engine.status().available);
    }

    #[test]
    fn a_manifest_from_another_format_version_is_quarantined_too() {
        let dir = TempDir::new("old_format");
        let config = config_at(&dir);
        std::fs::create_dir_all(&config.root_dir).unwrap();
        std::fs::write(
            SemanticManifest::file_path(&config.root_dir),
            b"{\"format_version\": 1}",
        )
        .unwrap();

        let engine = SemanticEngine::open(config.clone()).unwrap();
        assert!(engine.status().last_error.is_some());
        assert!(engine.incompatibilities().is_empty());
        assert!(SemanticManifest::load(&config.root_dir).is_ok());

        let quarantined = std::fs::read_dir(&config.root_dir)
            .unwrap()
            .filter_map(Result::ok)
            .any(|e| {
                e.file_name()
                    .to_string_lossy()
                    .contains("unsupported-version")
            });
        assert!(quarantined);
    }

    #[test]
    fn reset_index_clears_vectors_and_book_records() {
        let dir = TempDir::new("reset");
        let mut engine = SemanticEngine::open(config_at(&dir)).unwrap();
        engine.index_book(&three_line_book()).unwrap();

        assert_eq!(engine.reset_index().unwrap(), 3);
        let status = engine.status();
        assert_eq!(status.vector_count, 0);
        assert_eq!(status.indexed_book_count, 0);
        assert!(status.last_error.is_none());

        // Persisted too, not just in memory.
        let manifest = SemanticManifest::load(&engine.config.root_dir).unwrap();
        assert_eq!(manifest.book_count(), 0);
    }

    // ── diff ──

    #[test]
    fn diff_classifies_new_changed_and_removed_books() {
        let dir = TempDir::new("diff");
        let mut engine = SemanticEngine::open(config_at(&dir)).unwrap();

        engine
            .index_book(&book(
                "kept.txt",
                1,
                &[(1, 1, "שורה ארוכה דיה כדי לעמוד בפני עצמה")],
            ))
            .unwrap();
        engine
            .index_book(&book(
                "changed.txt",
                2,
                &[(1, 1, "שורה ארוכה דיה כדי לעמוד בפני עצמה")],
            ))
            .unwrap();
        engine
            .index_book(&book(
                "gone.txt",
                3,
                &[(1, 1, "שורה ארוכה דיה כדי לעמוד בפני עצמה")],
            ))
            .unwrap();

        let mut tantivy = HashMap::new();
        tantivy.insert("kept.txt".to_string(), 1u64);
        tantivy.insert("changed.txt".to_string(), 99u64);
        tantivy.insert("brand-new.txt".to_string(), 4u64);

        let diff = engine.diff_against_tantivy(&tantivy);
        assert_eq!(diff.new_books, vec!["brand-new.txt"]);
        assert_eq!(diff.changed_books, vec!["changed.txt"]);
        assert_eq!(diff.removed_books, vec!["gone.txt"]);
        assert_eq!(diff.books_to_index(), 2);
        assert!(!diff.needs_full_rebuild());
    }

    #[test]
    fn diff_output_order_is_deterministic() {
        let dir = TempDir::new("diff_order");
        let engine = SemanticEngine::open(config_at(&dir)).unwrap();

        let mut tantivy = HashMap::new();
        for name in ["c.txt", "a.txt", "d.txt", "b.txt"] {
            tantivy.insert(name.to_string(), 1u64);
        }

        let diff = engine.diff_against_tantivy(&tantivy);
        assert_eq!(diff.new_books, vec!["a.txt", "b.txt", "c.txt", "d.txt"]);
    }

    #[test]
    fn an_empty_index_reports_every_book_as_new() {
        let dir = TempDir::new("diff_empty");
        let engine = SemanticEngine::open(config_at(&dir)).unwrap();

        let mut tantivy = HashMap::new();
        tantivy.insert("a.txt".to_string(), 1u64);
        let diff = engine.diff_against_tantivy(&tantivy);

        assert_eq!(diff.new_books.len(), 1);
        assert!(diff.changed_books.is_empty());
        assert!(!diff.is_up_to_date());
    }

    #[test]
    fn unload_model_keeps_the_index_but_stops_serving_queries() {
        let dir = TempDir::new("unload");
        let mut engine = SemanticEngine::open(config_at(&dir)).unwrap();
        engine.index_book(&three_line_book()).unwrap();

        engine.unload_model();
        let status = engine.status();
        assert!(!status.model_loaded);
        assert!(
            !status.available,
            "without a model the query cannot be embedded"
        );
        assert_eq!(status.vector_count, 3, "the index itself is untouched");
        assert!(engine.search("בריאה", 5, None).is_err());

        // Re-loading brings it back.
        engine.load_model().unwrap();
        assert!(engine.status().available);
        assert!(engine.search("בריאה", 5, None).is_ok());
    }
}
