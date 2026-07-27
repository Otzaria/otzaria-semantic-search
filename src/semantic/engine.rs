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
    describe_mismatches, BookIndexNeed, ManifestConfig, ManifestMismatch, SemanticManifest,
};
use crate::semantic::store::{VectorStore, VectorStoreConfig};
use crate::semantic::types::{
    BookForIndexing, ContentFingerprint, IndexDiff, IndexOutcome, IndexingSummary, SearchFilters,
    SemanticCandidate, SemanticChunk, SemanticStatus, VectorMetadata,
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
                    // Only records that claim vectors are stale. A `chunk_count == 0`
                    // marker describes no vectors, so nothing about it was lost —
                    // dropping it would just make a scanned PDF get reprocessed on
                    // every startup.
                    let dropped = self.manifest.clear_books_with_vectors();
                    if dropped > 0 {
                        log::info!(
                            "Vector backend '{}' does not persist; dropped {dropped} stale book \
                             record(s) so they will be re-indexed, and kept {} empty-book \
                             marker(s)",
                            self.store.backend_id(),
                            self.manifest.empty_book_count()
                        );
                        self.manifest.save(&self.config.root_dir)?;
                    }
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
    /// The returned [`IndexOutcome`] says what happened: vectors written, the book
    /// already current and skipped, or nothing embeddable. It is deliberately not
    /// a bare chunk count — a skipped book has chunks in the index but wrote none,
    /// and reporting the former as the latter made "chunks written" untrue.
    ///
    /// Saves the manifest, so a crash afterwards cannot leave the book looking
    /// indexed when it is not. That costs a full serialize and `fsync` per call:
    /// for more than one book use [`SemanticEngine::index_books`], or
    /// [`SemanticEngine::index_book_deferred`] with an explicit
    /// [`SemanticEngine::flush_manifest`] when the caller needs to release its
    /// lock between books.
    pub fn index_book(
        &mut self,
        book: &BookForIndexing,
    ) -> Result<IndexOutcome, SemanticSearchError> {
        let outcome = self.index_book_inner(book)?;
        if outcome.did_work() {
            self.manifest.save(&self.config.root_dir)?;
        }
        Ok(outcome)
    }

    /// Index one book and leave the manifest unsaved.
    ///
    /// For a caller driving the loop itself — [`HybridCoordinator::index_books`]
    /// does, so it can release the engine lock between books. Whoever calls this
    /// **must** call [`SemanticEngine::flush_manifest`] afterwards, including on
    /// the error path: until then the vectors are in the store and nothing on
    /// disk says so.
    ///
    /// The alternative, saving per book, is not merely slower — it is quadratic.
    /// The manifest holds every book, so writing it after each of `B` books moves
    /// `O(B²)` bytes and asks for `B` `fsync`s. Over a library of thousands of
    /// books that is the difference between seconds and hours.
    ///
    /// [`HybridCoordinator::index_books`]: crate::hybrid::coordinator::HybridCoordinator::index_books
    pub fn index_book_deferred(
        &mut self,
        book: &BookForIndexing,
    ) -> Result<IndexOutcome, SemanticSearchError> {
        self.index_book_inner(book)
    }

    /// Persist the manifest.
    ///
    /// The commit point for [`SemanticEngine::index_book_deferred`]. Idempotent
    /// and cheap to over-call — but not free, so a caller should batch changes
    /// rather than call it per book.
    pub fn flush_manifest(&mut self) -> Result<(), SemanticSearchError> {
        self.manifest.save(&self.config.root_dir)?;
        Ok(())
    }

    /// Index several books, writing the manifest once at the end.
    ///
    /// Serializing the whole manifest after every book turns a full library
    /// index into quadratic I/O. On failure, everything indexed so far is still
    /// committed before the error propagates, so a retry resumes instead of
    /// restarting.
    pub fn index_books(
        &mut self,
        books: &[BookForIndexing],
    ) -> Result<IndexingSummary, SemanticSearchError> {
        let mut summary = IndexingSummary::default();
        let mut dirty = false;
        for book in books {
            match self.index_book_inner(book) {
                Ok(outcome) => {
                    dirty |= outcome.did_work();
                    summary.record(outcome);
                }
                Err(e) => {
                    // The failing call can also have removed a stale manifest
                    // entry before its store insertion failed, so flush even
                    // when every earlier outcome was a skip.
                    if let Err(save_err) = self.manifest.save(&self.config.root_dir) {
                        log::warn!(
                            "Could not save the manifest after an indexing failure: {save_err}"
                        );
                    }
                    return Err(e);
                }
            }
        }
        if dirty {
            self.manifest.save(&self.config.root_dir)?;
        }
        Ok(summary)
    }

    /// Index one book without saving the manifest.
    fn index_book_inner(
        &mut self,
        book: &BookForIndexing,
    ) -> Result<IndexOutcome, SemanticSearchError> {
        self.ensure_index_usable()?;

        let chunks = self.chunker.chunk_book(book);
        let line_fingerprint = book.line_fingerprint();

        if self.book_is_already_current(book, chunks.len(), line_fingerprint) {
            log::debug!(
                "{} is unchanged; skipping {} chunk(s) of inference",
                book.source_book_key,
                chunks.len()
            );
            return Ok(IndexOutcome::Skipped {
                chunks: chunks.len() as u32,
            });
        }

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
            // No vectors, but the book *was* processed. Recording that — rather
            // than dropping the entry — is what stops a scanned PDF or a book of
            // headings from being handed over again on every startup.
            self.store.commit()?;
            self.manifest.mark_book_indexed(
                book.source_book_key.clone(),
                book.content_fingerprint,
                line_fingerprint,
                0,
                self.config.chunking.chunking_version,
                self.config.normalization_version,
            );
            return Ok(IndexOutcome::Empty);
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
            book.content_fingerprint,
            line_fingerprint,
            chunks.len() as u32,
            self.config.chunking.chunking_version,
            self.config.normalization_version,
        );

        Ok(IndexOutcome::Indexed {
            chunks: chunks.len() as u32,
        })
    }

    /// Whether this exact book is already in the index, so embedding it again
    /// would produce identical vectors.
    ///
    /// Checked against the book's own lines rather than the lexical content hash,
    /// which is why it also works for PDFs — the lexical index has no fingerprint
    /// for them, so [`SemanticEngine::diff_against_tantivy`] has to report every
    /// PDF as needing attention. This is what makes that cheap: re-chunking, then
    /// no inference at all when nothing changed.
    ///
    /// The stored vector count is part of the comparison, so a manifest entry can
    /// never license a skip when the vectors it describes are absent.
    fn book_is_already_current(
        &self,
        book: &BookForIndexing,
        chunk_count: usize,
        line_fingerprint: u64,
    ) -> bool {
        let Some(entry) = self.manifest.book(&book.source_book_key) else {
            return false;
        };

        entry.line_fingerprint == line_fingerprint
            && entry.content_hash == book.content_fingerprint
            && entry.chunk_count as usize == chunk_count
            && entry.chunking_version == self.config.chunking.chunking_version
            && entry.normalization_version == self.config.normalization_version
            && self.store.book_vector_count(&book.source_book_key) == chunk_count
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
        self.remove_books(&[source_book_key.to_owned()])
    }

    /// Remove several books, committing the store and manifest once.
    ///
    /// This is the application path for [`IndexDiff::removed_books`]. Keeping it
    /// batched avoids rewriting the whole manifest for every deleted library
    /// book.
    pub fn remove_books(
        &mut self,
        source_book_keys: &[String],
    ) -> Result<u32, SemanticSearchError> {
        let mut removed_vectors = 0u32;
        let mut dirty = false;

        for source_book_key in source_book_keys {
            let count = self.store.delete_book(source_book_key)?;
            removed_vectors = removed_vectors.saturating_add(count);
            let removed_record = self.manifest.remove_book(source_book_key).is_some();
            dirty |= count > 0 || removed_record;
        }

        if dirty {
            self.store.commit()?;
            self.manifest.save(&self.config.root_dir)?;
        }
        Ok(removed_vectors)
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

    /// Compare the library's per-book fingerprints against the semantic index.
    ///
    /// This is the primary form: the caller decides what a book's fingerprint is.
    ///
    /// * A text book's lexical `contentHash` →
    ///   [`ContentFingerprint::from_lexical_hash`]. It already covers the
    ///   metadata the lexical engine indexes, so it can reach "nothing to do".
    /// * A PDF → [`ContentFingerprint::canonical`], folding the caller's
    ///   authoritative source revision (including extracted text structure and
    ///   extraction/OCR version) together with the title, category path and
    ///   facets. **Without the metadata a match proves
    ///   too little**: renaming a book or correcting its author changes what is
    ///   stored in every one of its vectors while leaving the file untouched, so
    ///   a file-only signature would report "up to date" over a stale index.
    ///   A size/mtime signature is not authoritative;
    ///   [`ContentFingerprint::content_only`] says exactly that and lands the
    ///   book in [`IndexDiff::unverifiable_books`] instead of claiming more.
    /// * Nothing at all → [`ContentFingerprint::Unverifiable`], also
    ///   [`IndexDiff::unverifiable_books`].
    ///
    /// When a configuration change invalidated the index, every known book is
    /// reported as changed — an incremental update cannot repair a change of model
    /// or chunking.
    pub fn diff(&self, books: &HashMap<String, ContentFingerprint>) -> IndexDiff {
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
        let mut unverifiable_books = Vec::new();

        for (book_key, &fingerprint) in books {
            let need = self.manifest.book_index_need(
                book_key,
                fingerprint,
                self.config.chunking.chunking_version,
                self.config.normalization_version,
            );

            match need {
                _ if full_rebuild && need.is_known() => changed_books.push(book_key.clone()),
                _ if full_rebuild => new_books.push(book_key.clone()),
                BookIndexNeed::Missing => new_books.push(book_key.clone()),
                BookIndexNeed::Changed => changed_books.push(book_key.clone()),
                BookIndexNeed::Unverifiable => unverifiable_books.push(book_key.clone()),
                BookIndexNeed::UpToDate => {}
            }
        }

        let mut removed_books: Vec<String> = self
            .manifest
            .books
            .keys()
            .filter(|book_key| !books.contains_key(*book_key))
            .cloned()
            .collect();

        // Deterministic order: the caller may show these lists or drive progress
        // from them, and `HashMap` iteration order changes between runs.
        new_books.sort_unstable();
        changed_books.sort_unstable();
        unverifiable_books.sort_unstable();
        removed_books.sort_unstable();

        IndexDiff {
            new_books,
            changed_books,
            unverifiable_books,
            removed_books,
            model_mismatch,
            chunking_mismatch,
            normalization_mismatch,
        }
    }

    /// Convenience wrapper over [`SemanticEngine::diff`] for raw lexical hashes.
    ///
    /// `tantivy_books` maps a book key to its raw `contentHash`, and `0` is the
    /// lexical engine's "no fingerprint" marker — it records that for every PDF,
    /// because their extracted text never enters the library database. Comparing
    /// it as a hash would make `0 == 0` mean "unchanged", so a replaced or
    /// re-scanned PDF would never be re-indexed.
    ///
    /// Consequently **every PDF appears in [`IndexDiff::unverifiable_books`] on
    /// every call**, and `is_up_to_date()` never returns `true` for a library that
    /// contains one. That is honest rather than useful: the index genuinely cannot
    /// tell. Producing such a book costs the caller a fresh text extraction, so a
    /// caller that has its own PDF signature should call [`SemanticEngine::diff`]
    /// with it and get a diff that can actually reach "nothing to do".
    pub fn diff_against_tantivy(&self, tantivy_books: &HashMap<String, u64>) -> IndexDiff {
        let fingerprints = tantivy_books
            .iter()
            .map(|(key, &hash)| (key.clone(), ContentFingerprint::from_lexical_hash(hash)))
            .collect();
        self.diff(&fingerprints)
    }

    /// How many times this session has written the manifest.
    ///
    /// A cost, not a statistic: each write serializes every book record and
    /// `fsync`s. Exposed so the indexing loop's write count can be asserted rather
    /// than assumed — the quadratic version of that loop passed every other test.
    pub fn manifest_save_count(&self) -> u32 {
        self.manifest.save_count()
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

    fn book(key: &str, content_fingerprint: u64, lines: &[(u64, u64, &str)]) -> BookForIndexing {
        BookForIndexing {
            source_book_key: key.to_string(),
            title: "ספר בדיקה".to_string(),
            content_fingerprint,
            is_pdf: false,
            topics: "/מקרא/תורה".to_string(),
            extra_facets: vec!["/author/מחבר".to_string(), "/era/תנך".to_string()],
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

        let outcome = engine.index_book(&three_line_book()).unwrap();
        assert_eq!(outcome, IndexOutcome::Indexed { chunks: 3 });
        assert_eq!(outcome.chunks_written(), 3);

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
        assert_eq!(
            engine.index_book(&shrunk).unwrap(),
            IndexOutcome::Indexed { chunks: 2 }
        );

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
    fn reindexing_a_book_that_lost_every_line_drops_its_vectors_but_keeps_its_record() {
        let dir = TempDir::new("reindex_emptied");
        let mut engine = SemanticEngine::open(config_at(&dir)).unwrap();

        engine.index_book(&three_line_book()).unwrap();
        let emptied = book("otzaria/tanach/genesis.txt", 333, &[]);
        assert_eq!(engine.index_book(&emptied).unwrap(), IndexOutcome::Empty);

        let status = engine.status();
        assert_eq!(status.vector_count, 0, "no orphan vectors may survive");
        assert_eq!(
            status.indexed_book_count, 1,
            "the book was processed; the record is what stops it being reprocessed"
        );

        // And the diff agrees there is nothing left to do for it.
        let mut tantivy = HashMap::new();
        tantivy.insert("otzaria/tanach/genesis.txt".to_string(), 333u64);
        assert!(engine.diff_against_tantivy(&tantivy).is_up_to_date());
    }

    /// A scanned PDF yields no text. Without a marker it would be reported as new
    /// and reprocessed on every startup — the problem the lexical indexer solves
    /// with its own empty-book marker.
    #[test]
    fn a_book_that_yields_nothing_embeddable_is_not_offered_again() {
        let dir = TempDir::new("empty_book_marker");
        let mut engine = SemanticEngine::open(config_at(&dir)).unwrap();

        // Lines too short to embed: the book is processed but produces no chunks.
        let headings = book(
            "headings.txt",
            4242,
            &[(1, 1, "א"), (2, 1, "ב"), (3, 1, "ג")],
        );
        assert_eq!(engine.index_book(&headings).unwrap(), IndexOutcome::Empty);

        let mut tantivy = HashMap::new();
        tantivy.insert("headings.txt".to_string(), 4242u64);

        let diff = engine.diff_against_tantivy(&tantivy);
        assert!(
            diff.is_up_to_date(),
            "an empty book must not be reported as needing work: {diff:?}"
        );
        assert!(diff.new_books.is_empty());
        assert!(diff.changed_books.is_empty());

        // It survives a restart too, since it never had vectors to lose.
        drop(engine);
        let engine = SemanticEngine::open(config_at(&dir)).unwrap();
        assert_eq!(engine.status().indexed_book_count, 1);
        assert!(engine.diff_against_tantivy(&tantivy).is_up_to_date());
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

        assert_eq!(engine.index_book(&blank).unwrap(), IndexOutcome::Empty);
        assert!(!engine.status().model_loaded);
        assert_eq!(
            engine.status().indexed_book_count,
            1,
            "the empty-book marker is written without needing a model"
        );
    }

    /// The lexical engine records `contentHash = 0` for every PDF, because their
    /// extracted text is not in the library database. Compared as an ordinary
    /// hash, `0 == 0` means "unchanged" and a replaced PDF would keep its old
    /// vectors forever.
    #[test]
    fn a_changed_pdf_is_reindexed_even_though_its_content_hash_never_changes() {
        let dir = TempDir::new("pdf_reindex");
        let mut engine = SemanticEngine::open(config_at(&dir)).unwrap();

        let mut scan = book(
            "otzaria/scans/responsa.pdf",
            0, // exactly what Tantivy reports for a PDF
            &[
                (1, 1, "עמוד ראשון של השאלות והתשובות הסרוקות"),
                (2, 1, "עמוד שני של השאלות והתשובות הסרוקות"),
            ],
        );
        scan.is_pdf = true;
        engine.index_book(&scan).unwrap();
        assert_eq!(engine.status().vector_count, 2);

        // The PDF is re-scanned: different text, same contentHash of 0.
        let mut rescanned = scan.clone();
        rescanned.lines[1].text = "עמוד שני לאחר סריקה מחדש באיכות טובה יותר".to_string();
        rescanned.lines.push(BookLine {
            line_id: 3,
            section_id: 1,
            text: "עמוד שלישי שלא זוהה בסריקה הראשונה".to_string(),
            line_hash: 3000,
            reference: "עמוד 3".to_string(),
            segment: 3,
        });
        assert_eq!(rescanned.content_fingerprint, 0);

        // The diff must ask for it — it cannot prove anything from a hash of 0.
        let mut tantivy = HashMap::new();
        tantivy.insert("otzaria/scans/responsa.pdf".to_string(), 0u64);
        let diff = engine.diff_against_tantivy(&tantivy);
        assert!(
            !diff.is_up_to_date(),
            "a book with no usable fingerprint can never be declared current"
        );
        // Reported as unverifiable, not as changed: it is not *known* to have
        // changed, and producing it costs the caller a fresh text extraction.
        assert_eq!(diff.unverifiable_books, vec!["otzaria/scans/responsa.pdf"]);
        assert!(diff.changed_books.is_empty());
        assert_eq!(diff.books_to_index(), 1);

        assert_eq!(
            engine.index_book(&rescanned).unwrap(),
            IndexOutcome::Indexed { chunks: 3 }
        );
        assert_eq!(
            engine.status().vector_count,
            3,
            "the re-scanned text must have replaced the old vectors"
        );

        // The new text is findable and the replaced text is not.
        let hits = engine
            .search("עמוד שלישי שלא זוהה בסריקה הראשונה", 5, None)
            .unwrap();
        assert!((hits[0].similarity_score - 1.0).abs() < 1e-5);
        assert_eq!(hits[0].metadata.line_id, 3);
    }

    /// The failure mode a file-only signature cannot catch: the PDF on disk is
    /// byte-identical, but the library corrected its author. Every vector carries
    /// that author, so "unchanged" would keep serving the old one.
    #[test]
    fn correcting_a_pdfs_metadata_is_reported_as_a_change() {
        let dir = TempDir::new("pdf_metadata_change");
        let mut engine = SemanticEngine::open(config_at(&dir)).unwrap();

        const KEY: &str = "otzaria/scans/responsa.pdf";
        const SIGNATURE: u64 = 0xBEEF_CAFE;

        // The caller folds its authoritative extraction revision together with
        // the metadata.
        let fingerprint_of = |book: &BookForIndexing| {
            ContentFingerprint::canonical(
                SIGNATURE,
                &book.title,
                &book.topics,
                &book.extra_facets,
                book.is_pdf,
            )
        };

        let mut scan = book(
            KEY,
            0,
            &[
                (1, 1, "עמוד ראשון של השאלות והתשובות הסרוקות"),
                (2, 1, "עמוד שני של השאלות והתשובות הסרוקות"),
            ],
        );
        scan.is_pdf = true;
        scan.extra_facets = vec!["/author/מחבר ראשון".to_string()];
        scan.content_fingerprint = fingerprint_of(&scan).as_raw();
        engine.index_book(&scan).unwrap();

        // Nothing changed: the same book reports as current.
        let current = HashMap::from([(KEY.to_string(), fingerprint_of(&scan))]);
        assert!(
            engine.diff(&current).is_up_to_date(),
            "a canonical fingerprint is what lets a PDF reach 'nothing to do'"
        );

        // Same extracted source revision, corrected author.
        let mut corrected = scan.clone();
        corrected.extra_facets = vec!["/author/מחבר מדויק".to_string()];
        corrected.content_fingerprint = fingerprint_of(&corrected).as_raw();

        let after = HashMap::from([(KEY.to_string(), fingerprint_of(&corrected))]);
        let diff = engine.diff(&after);
        assert_eq!(
            diff.changed_books,
            vec![KEY.to_string()],
            "a metadata-only correction must be visible at diff time, before the \
             lines are loaded"
        );
        assert!(diff.unverifiable_books.is_empty());

        // And re-indexing really replaces the stored facets.
        assert_eq!(
            engine.index_book(&corrected).unwrap(),
            IndexOutcome::Indexed { chunks: 2 }
        );
        let hits = engine
            .search("עמוד ראשון של השאלות והתשובות הסרוקות", 5, None)
            .unwrap();
        assert!(hits[0]
            .metadata
            .facets
            .contains(&"/author/מחבר מדויק".to_string()));
        assert!(!hits[0]
            .metadata
            .facets
            .contains(&"/author/מחבר ראשון".to_string()));
    }

    /// Facet order is meaningless, so reordering must not cost a re-embed.
    #[test]
    fn reordering_a_books_facets_does_not_trigger_re_embedding() {
        let dir = TempDir::new("facet_order");
        let mut engine = SemanticEngine::open(config_at(&dir)).unwrap();

        let mut original = book(
            "otzaria/commentary.txt",
            42,
            &[(1, 1, "שורה ארוכה דיה כדי לעמוד בפני עצמה בבדיקה")],
        );
        original.extra_facets = vec![
            "/author/רבי אחד".to_string(),
            "/author/רבי שני".to_string(),
            "/era/ראשונים".to_string(),
        ];
        engine.index_book(&original).unwrap();

        let mut reordered = original.clone();
        reordered.extra_facets = vec![
            "/era/ראשונים".to_string(),
            "/author/רבי שני".to_string(),
            "/author/רבי אחד".to_string(),
        ];
        assert_eq!(
            engine.index_book(&reordered).unwrap(),
            IndexOutcome::Skipped { chunks: 1 },
            "the same facets in another order describe the same book"
        );
    }

    /// The manifest holds every book, so writing it per book moves `O(B²)` bytes.
    /// This is the assertion that catches a loop that reverted to doing so.
    #[test]
    fn the_manifest_write_count_does_not_grow_with_the_number_of_books() {
        let index_books = |count: u64, name: &str| -> u32 {
            let dir = TempDir::new(name);
            let mut engine = SemanticEngine::open(config_at(&dir)).unwrap();
            let books: Vec<BookForIndexing> = (0..count)
                .map(|i| {
                    book(
                        &format!("otzaria/book{i}.txt"),
                        i + 1,
                        &[(1, 1, "שורה ארוכה דיה כדי לעמוד בפני עצמה בבדיקה")],
                    )
                })
                .collect();
            engine.index_books(&books).unwrap();
            engine.manifest_save_count()
        };

        let few = index_books(3, "manifest_writes_few");
        let many = index_books(30, "manifest_writes_many");
        assert_eq!(
            few, many,
            "ten times the books must not cost ten times the manifest writes \
             (measured {few} and {many})"
        );
        // Three, all of them one-offs: the fresh manifest written at open, the
        // model identity recorded the first time a model loads, and the batch
        // commit. None of them scales with the library.
        assert!(many <= 3, "unexpectedly many manifest writes: {many}");
    }

    /// Reporting every PDF as "needs attention" is only acceptable because
    /// handing one back costs re-chunking and no inference when it is unchanged.
    #[test]
    fn an_unchanged_book_is_skipped_without_re_embedding() {
        let dir = TempDir::new("skip_unchanged");
        let mut engine = SemanticEngine::open(config_at(&dir)).unwrap();

        let mut scan = book(
            "otzaria/scans/unchanged.pdf",
            0,
            &[(1, 1, "עמוד סרוק עם מספיק תווים כדי להיות מוטמע")],
        );
        scan.is_pdf = true;
        engine.index_book(&scan).unwrap();

        // Drop the model: a genuine skip must not need one.
        engine.unload_model();
        assert_eq!(
            engine.index_book(&scan).unwrap(),
            IndexOutcome::Skipped { chunks: 1 },
            "an unchanged book reports what the index holds, and that it wrote nothing"
        );
        assert!(
            !engine.status().model_loaded,
            "no inference means no model load"
        );
        assert_eq!(engine.status().vector_count, 1);

        // Whereas a changed line does load the model and re-embed.
        let mut changed = scan.clone();
        changed.lines[0].text = "עמוד סרוק אחר לגמרי עם מספיק תווים להטמעה".to_string();
        engine.index_book(&changed).unwrap();
        assert!(engine.status().model_loaded);
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

        let summary = engine.index_books(&books).unwrap();
        assert_eq!(summary.books_indexed, 5);
        assert_eq!(summary.books_skipped, 0);
        assert_eq!(summary.books_empty, 0);
        assert_eq!(summary.chunks_written, 10);
        assert_eq!(summary.books_processed(), 5);
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
            facets: Some(vec!["/מקרא".to_string()]),
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
        assert_eq!(
            engine.index_book(&three_line_book()).unwrap(),
            IndexOutcome::Indexed { chunks: 3 }
        );
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
        assert_eq!(
            engine.index_book(&three_line_book()).unwrap(),
            IndexOutcome::Indexed { chunks: 3 }
        );
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
