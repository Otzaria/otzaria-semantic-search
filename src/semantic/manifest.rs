//! Semantic index manifest and versioning.
//!
//! The manifest records the exact configuration the semantic index was built
//! with — embedding model and its file checksum, embedding backend, vector
//! backend, dimensions, pooling, precision, chunking and normalization versions
//! — plus a per-book indexing record.
//!
//! On startup the manifest is compared against the current configuration. Any
//! mismatch means the stored vectors live in a different space from the ones the
//! current configuration would produce, so they cannot be searched or added to.
//! [`SemanticEngine`](crate::semantic::engine::SemanticEngine) disables the
//! semantic path in that case — queries fall back to BM25 — until the index is
//! reset and rebuilt.
//!
//! # Atomic persistence
//!
//! The manifest is written to a `.tmp` file which is `fsync`ed and then renamed
//! over the target. A crash therefore leaves either the previous manifest or the
//! new one, never a half-written file — and [`SemanticManifest::load`] recovers
//! the leftovers a crash mid-`save` can strand. Durability of the rename itself
//! is enforced on Unix and best-effort elsewhere; [`SemanticManifest::save`]
//! documents exactly which.
//!
//! Writing it is not cheap — the whole document is serialized every time — so a
//! bulk index commits once rather than per book. With the current volatile vector
//! store, intermediate checkpoints cannot preserve work across a restart. See
//! [`SemanticEngine::index_book_deferred`](crate::semantic::engine::SemanticEngine::index_book_deferred).

use crate::errors::ManifestError;
use crate::semantic::types::ContentFingerprint;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
use std::num::NonZeroU64;
use std::path::{Path, PathBuf};

/// Current manifest format version. Bump when the schema changes.
const MANIFEST_FORMAT_VERSION: u32 = 3;

/// Manifest file name.
const MANIFEST_FILENAME: &str = "semantic_manifest.json";

/// Where the previous manifest is parked while a retried rename is in flight.
///
/// Only [`SemanticManifest::save`]'s fallback path creates it, and only for the
/// duration of one rename. Its existence alongside a missing target means a crash
/// caught that window, and [`SemanticManifest::load`] recovers from it.
const MANIFEST_PREVIOUS_SUFFIX: &str = "previous";

/// Full semantic index manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticManifest {
    /// Format version of this manifest.
    pub format_version: u32,

    // ── Model metadata ──
    /// Embedding model identifier (e.g. "EMD123/Otzaria-Embedding-V1-Flash-0.6B").
    pub embedding_model_id: String,
    /// SHA-256 of the model file, once a model has been loaded.
    ///
    /// Guards the case the model id alone cannot catch: same id, different
    /// weights behind the same path.
    pub model_checksum: Option<String>,
    /// Identifier of the inference backend that produced the vectors, once a
    /// model has been loaded (e.g. `"mock-hash-v1"`).
    pub embedding_backend: Option<String>,
    /// Embedding vector dimensionality (e.g. 1024).
    pub embedding_dim: u32,
    /// Pooling strategy used (e.g. "last-token").
    pub pooling: String,
    /// Model quantization level (e.g. "Q4").
    pub model_quantization: String,
    /// Vector storage precision in the store (e.g. "f32", "f16").
    pub vector_precision: String,
    /// Identifier of the vector storage backend that holds the vectors.
    pub vector_backend: String,

    // ── Algorithm versions ──
    /// Chunking algorithm version. Increment when chunking logic changes.
    pub chunking_version: u32,
    /// Semantic normalization version. Increment when text preprocessing changes.
    pub normalization_version: u32,

    // ── Timestamps ──
    /// When this manifest was first created (Unix timestamp).
    pub created_at: u64,
    /// When this manifest was last updated (Unix timestamp).
    pub updated_at: u64,

    // ── Per-book tracking ──
    /// Per-book indexing records, keyed by `source_book_key` (file path).
    pub books: HashMap<String, BookManifestEntry>,

    /// How many times this instance has been written to disk.
    ///
    /// Not part of the format — `serde(skip)` — and not a statistic. Writing the
    /// manifest costs a full serialize plus an `fsync`, so "how many times did
    /// that happen" is a correctness property of the indexing loop, and the only
    /// way to assert it is to count. See
    /// [`SemanticManifest::save_count`].
    #[serde(skip)]
    saves: u32,
}

/// Per-book entry in the manifest.
///
/// An entry with `chunk_count == 0` is meaningful, not a leftover: it records
/// that the book *was* processed and yielded nothing embeddable — a scanned PDF
/// with no text layer, or a book of headings only. Without it the book would be
/// reported as new on every startup and reprocessed forever. The lexical indexer
/// keeps an empty-book marker for exactly this reason.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BookManifestEntry {
    /// Stable book identifier (file path).
    pub source_book_key: String,
    /// The raw fingerprint the caller vouched for at indexing time — the lexical
    /// `contentHash` for a text book, whatever
    /// [`ContentFingerprint::canonical`] produced for a PDF.
    ///
    /// `0` means there was none, in which case `line_fingerprint` is the only
    /// thing that can decide whether anything changed.
    pub content_hash: u64,
    /// Fingerprint computed by this crate from the book itself — its lines and the
    /// metadata stored in every vector.
    ///
    /// What settles a book whose `content_hash` could not prove it was current:
    /// one recorded as `0`, or one whose fingerprint covers the content but not
    /// the metadata. See
    /// [`BookForIndexing::line_fingerprint`](crate::semantic::types::BookForIndexing::line_fingerprint).
    pub line_fingerprint: u64,
    /// Number of semantic chunks generated for this book. `0` is a valid,
    /// deliberate value — see the type-level note.
    pub chunk_count: u32,
    /// When this book was last indexed (Unix timestamp).
    pub indexed_at: u64,
    /// Chunking version used for this specific book.
    pub chunking_version: u32,
    /// Normalization version used for this specific book.
    pub normalization_version: u32,
}

/// The configuration a manifest is validated against.
///
/// `model_checksum` and `embedding_backend` are `Option` because they are only
/// known once a model has actually been loaded. While they are `None`,
/// [`SemanticManifest::validate`] leaves those dimensions unchecked instead of
/// reporting a false mismatch; the engine re-validates after loading the model.
#[derive(Debug, Clone)]
pub struct ManifestConfig {
    pub embedding_model_id: String,
    pub model_checksum: Option<String>,
    pub embedding_backend: Option<String>,
    pub embedding_dim: u32,
    pub pooling: String,
    pub model_quantization: String,
    pub vector_precision: String,
    pub vector_backend: String,
    pub chunking_version: u32,
    pub normalization_version: u32,
}

impl SemanticManifest {
    /// Create a new manifest from the given configuration.
    pub fn new(config: &ManifestConfig) -> Self {
        let now = current_unix_timestamp();
        Self {
            format_version: MANIFEST_FORMAT_VERSION,
            embedding_model_id: config.embedding_model_id.clone(),
            model_checksum: config.model_checksum.clone(),
            embedding_backend: config.embedding_backend.clone(),
            embedding_dim: config.embedding_dim,
            pooling: config.pooling.clone(),
            model_quantization: config.model_quantization.clone(),
            vector_precision: config.vector_precision.clone(),
            vector_backend: config.vector_backend.clone(),
            chunking_version: config.chunking_version,
            normalization_version: config.normalization_version,
            created_at: now,
            updated_at: now,
            books: HashMap::new(),
            saves: 0,
        }
    }

    /// How many times this instance has written itself to disk.
    ///
    /// Resets to zero when a manifest is loaded, since the counter describes this
    /// process's writes rather than the file's history.
    pub fn save_count(&self) -> u32 {
        self.saves
    }

    /// Load a manifest from the given directory.
    ///
    /// Returns [`ManifestError::NotFound`] when there is nothing to load (a
    /// first run), [`ManifestError::ParseFailed`] when the file is unreadable or
    /// malformed, and [`ManifestError::UnsupportedFormatVersion`] when it was
    /// written by a different schema version. The caller decides how to react;
    /// see [`SemanticEngine::open`](crate::semantic::engine::SemanticEngine::open),
    /// which quarantines an unusable file and starts fresh.
    pub fn load(dir: &Path) -> Result<Self, ManifestError> {
        let path = Self::file_path(dir);
        if !path.exists() {
            // A crash inside `save` can leave the manifest somewhere other than
            // its own name. Recovering it is what makes the "previous or new,
            // never none" guarantee hold: without this the state reads as a first
            // run and the whole index is silently rebuilt.
            //
            // Order matters. `.previous` is a manifest that was *already* in
            // service, so it is preferred over `.tmp`, which was only ever a
            // candidate. `.tmp` is still worth having: `save` `fsync`s it before
            // renaming, so one that outlived a crash is a complete document, and a
            // partial one fails to parse below and is discarded.
            let candidates = [Self::previous_path(dir), Self::tmp_path(dir)];
            let recovered = candidates
                .iter()
                .find(|candidate| Self::is_recoverable(candidate));

            let Some(recovered) = recovered else {
                for unusable in candidates.iter().filter(|c| c.exists()) {
                    // Left where it is — this path deletes no evidence — but not
                    // silently: a leftover nobody mentions is a leftover nobody
                    // investigates.
                    log::warn!(
                        "Ignoring {}: it is not a readable manifest of format version \
                         {MANIFEST_FORMAT_VERSION}",
                        unusable.display()
                    );
                }
                return Err(ManifestError::NotFound {
                    path: path.display().to_string(),
                });
            };

            log::warn!(
                "No manifest at {}, but a usable copy exists at {}; a previous save was \
                 interrupted. Recovering it.",
                path.display(),
                recovered.display()
            );
            std::fs::rename(recovered, &path).map_err(|e| ManifestError::WriteFailed {
                reason: format!(
                    "Failed to recover the manifest {} to {}: {e}",
                    recovered.display(),
                    path.display()
                ),
            })?;
            // Recovery changes a directory entry just like `save` does. Without
            // flushing the directory, reporting a successful recovery would not
            // mean the live name survives a power loss.
            sync_directory(dir)?;
        }

        let content = std::fs::read_to_string(&path).map_err(|e| ManifestError::ParseFailed {
            reason: format!("Failed to read manifest: {e}"),
        })?;

        // Read the version before the full document, so a schema change reports
        // a version mismatch rather than a confusing field-level parse error.
        let probe: FormatVersionProbe =
            serde_json::from_str(&content).map_err(|e| ManifestError::ParseFailed {
                reason: format!("Failed to parse manifest JSON: {e}"),
            })?;
        if probe.format_version != MANIFEST_FORMAT_VERSION {
            return Err(ManifestError::UnsupportedFormatVersion {
                found: probe.format_version,
                supported: MANIFEST_FORMAT_VERSION,
            });
        }

        serde_json::from_str(&content).map_err(|e| ManifestError::ParseFailed {
            reason: format!("Failed to parse manifest JSON: {e}"),
        })
    }

    /// Save the manifest to the given directory.
    ///
    /// Atomic: the payload is written to a temp file, `fsync`ed, then renamed over
    /// the target. A crash at any point leaves a readable manifest — the previous
    /// one, the new one, or a recoverable copy [`Self::load`] knows how to find —
    /// never a partial file and never none.
    ///
    /// Durable, with one platform caveat. The rename itself only becomes durable
    /// once the *directory* entry is flushed, and that is not uniformly possible:
    ///
    /// * On Unix the directory is opened and `fsync`ed, and a failure fails the
    ///   save. Returning `Ok` from it would be a lie — the caller would record
    ///   progress that a power loss can still undo.
    /// * On other platforms (Windows has no directory `fsync`) the rename is left
    ///   as the filesystem's own guarantee. `save` returning `Ok` there means the
    ///   data reached the disk and the rename was issued, not that the rename
    ///   survives a power loss.
    ///
    /// The `.previous` file that appears if a rename has to be retried is part of
    /// the atomicity guarantee, not debris: see the fallback below.
    pub fn save(&mut self, dir: &Path) -> Result<(), ManifestError> {
        self.updated_at = current_unix_timestamp();

        let target = Self::file_path(dir);
        let tmp = Self::tmp_path(dir);

        std::fs::create_dir_all(dir).map_err(|e| ManifestError::WriteFailed {
            reason: format!("Failed to create manifest directory: {e}"),
        })?;

        let content =
            serde_json::to_string_pretty(self).map_err(|e| ManifestError::WriteFailed {
                reason: format!("Failed to serialize manifest: {e}"),
            })?;

        // Scope the handle so it is closed before the rename (required on Windows).
        {
            let mut file = std::fs::File::create(&tmp).map_err(|e| ManifestError::WriteFailed {
                reason: format!("Failed to create temp manifest: {e}"),
            })?;
            file.write_all(content.as_bytes())
                .map_err(|e| ManifestError::WriteFailed {
                    reason: format!("Failed to write temp manifest: {e}"),
                })?;
            // Without this the rename can be durable before the data is, leaving
            // an empty or partial manifest after a power loss.
            file.sync_all().map_err(|e| ManifestError::WriteFailed {
                reason: format!("Failed to flush temp manifest: {e}"),
            })?;
        }

        let mut parked: Option<PathBuf> = None;
        if let Err(first) = rename(&tmp, &target) {
            // `rename` replaces an existing file on both Unix and Windows, but a
            // Windows file lock (search indexer, antivirus, another handle) can
            // still deny it.
            //
            // The retry must not simply delete the target: a crash in that window
            // would leave no manifest at all, which is worse than either version.
            // Parking it under `.previous` keeps a recoverable copy throughout,
            // and `load` recovers from it if a crash lands mid-swap.
            log::warn!("Atomic manifest rename failed ({first}); retrying via a parked copy");
            let previous = Self::previous_path(dir);
            let _ = std::fs::remove_file(&previous);

            if rename(&target, &previous).is_ok() {
                parked = Some(previous.clone());
            }

            if let Err(second) = rename(&tmp, &target) {
                // Put the old manifest back rather than leaving the directory
                // without one. If even that fails the parked copy stays where it
                // is, which is exactly what `load` knows how to recover.
                if let Some(previous) = &parked {
                    if let Err(restore) = rename(previous, &target) {
                        log::error!(
                            "Could not restore the parked manifest {} to {} ({restore}); it \
                             stays parked and will be recovered on the next open",
                            previous.display(),
                            target.display()
                        );
                    }
                }
                return Err(ManifestError::WriteFailed {
                    reason: format!(
                        "Failed to rename temp manifest to final: {second} \
                         (first attempt: {first})"
                    ),
                });
            }
        }

        // The rename is only durable once the directory entry itself is flushed;
        // otherwise a power loss can resurrect the old name. This must happen
        // *before* the parked copy is discarded, so a crash in between still
        // leaves something recoverable.
        sync_directory(dir)?;

        if let Some(previous) = parked {
            match std::fs::remove_file(&previous) {
                Ok(()) => {
                    // Persist the cleanup too. A resurrected `.previous` is
                    // harmless while the live target exists, but leaving the
                    // operation half-durable makes later recovery ambiguous.
                    sync_directory(dir)?;
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => {
                    log::warn!(
                        "The new manifest is durable, but its parked predecessor {} \
                         could not be removed: {e}",
                        previous.display()
                    );
                }
            }
        }

        self.saves = self.saves.saturating_add(1);
        log::debug!("Manifest saved to {}", target.display());
        Ok(())
    }

    /// Path of the parked copy used by [`Self::save`]'s fallback.
    pub fn previous_path(dir: &Path) -> PathBuf {
        dir.join(format!("{MANIFEST_FILENAME}.{MANIFEST_PREVIOUS_SUFFIX}"))
    }

    /// Path of the temp file [`Self::save`] writes before renaming.
    pub fn tmp_path(dir: &Path) -> PathBuf {
        dir.join(format!("{MANIFEST_FILENAME}.tmp"))
    }

    /// Whether a leftover file is a manifest worth recovering.
    ///
    /// Parses it and checks the format version, so a half-written temp file or one
    /// from another schema is passed over rather than promoted into place.
    fn is_recoverable(candidate: &Path) -> bool {
        let Ok(content) = std::fs::read_to_string(candidate) else {
            return false;
        };
        match serde_json::from_str::<Self>(&content) {
            Ok(manifest) => manifest.format_version == MANIFEST_FORMAT_VERSION,
            Err(_) => false,
        }
    }

    /// Move an unusable manifest aside so a fresh one can be written without
    /// destroying evidence, and return where it went.
    ///
    /// `tag` describes why (e.g. `"corrupt"`, `"incompatible"`).
    pub fn quarantine(dir: &Path, tag: &str) -> Result<PathBuf, ManifestError> {
        let source = Self::file_path(dir);
        let quarantined = dir.join(format!(
            "{MANIFEST_FILENAME}.{tag}.{}",
            current_unix_timestamp()
        ));

        std::fs::rename(&source, &quarantined).map_err(|e| ManifestError::WriteFailed {
            reason: format!(
                "Failed to move {} aside to {}: {e}",
                source.display(),
                quarantined.display()
            ),
        })?;

        log::warn!(
            "Quarantined unusable manifest ({tag}) to {}",
            quarantined.display()
        );
        Ok(quarantined)
    }

    /// Validate this manifest against the current configuration.
    /// Returns a list of mismatches (empty = everything matches).
    pub fn validate(&self, config: &ManifestConfig) -> Vec<ManifestMismatch> {
        let mut mismatches = Vec::new();

        if self.embedding_model_id != config.embedding_model_id {
            mismatches.push(ManifestMismatch::ModelId {
                manifest: self.embedding_model_id.clone(),
                config: config.embedding_model_id.clone(),
            });
        }

        // Only compare a checksum/backend the caller actually knows. Nothing to
        // compare on either side is not a mismatch — it is recorded on save.
        if let (Some(manifest), Some(config)) = (&self.model_checksum, &config.model_checksum) {
            if manifest != config {
                mismatches.push(ManifestMismatch::ModelChecksum {
                    manifest: manifest.clone(),
                    config: config.clone(),
                });
            }
        }

        if let (Some(manifest), Some(config)) = (&self.embedding_backend, &config.embedding_backend)
        {
            if manifest != config {
                mismatches.push(ManifestMismatch::EmbeddingBackend {
                    manifest: manifest.clone(),
                    config: config.clone(),
                });
            }
        }

        if self.embedding_dim != config.embedding_dim {
            mismatches.push(ManifestMismatch::Dimensions {
                manifest: self.embedding_dim,
                config: config.embedding_dim,
            });
        }

        if self.pooling != config.pooling {
            mismatches.push(ManifestMismatch::Pooling {
                manifest: self.pooling.clone(),
                config: config.pooling.clone(),
            });
        }

        if self.model_quantization != config.model_quantization {
            mismatches.push(ManifestMismatch::ModelQuantization {
                manifest: self.model_quantization.clone(),
                config: config.model_quantization.clone(),
            });
        }

        if self.vector_precision != config.vector_precision {
            mismatches.push(ManifestMismatch::VectorPrecision {
                manifest: self.vector_precision.clone(),
                config: config.vector_precision.clone(),
            });
        }

        if self.vector_backend != config.vector_backend {
            mismatches.push(ManifestMismatch::VectorBackend {
                manifest: self.vector_backend.clone(),
                config: config.vector_backend.clone(),
            });
        }

        if self.chunking_version != config.chunking_version {
            mismatches.push(ManifestMismatch::ChunkingVersion {
                manifest: self.chunking_version,
                config: config.chunking_version,
            });
        }

        if self.normalization_version != config.normalization_version {
            mismatches.push(ManifestMismatch::NormalizationVersion {
                manifest: self.normalization_version,
                config: config.normalization_version,
            });
        }

        mismatches
    }

    /// Record the identity of the model that produced this index.
    ///
    /// Called once a model is loaded, so a later session can detect that the
    /// file behind the same model id changed.
    pub fn set_model_identity(&mut self, checksum: Option<String>, backend: Option<String>) {
        if checksum.is_some() {
            self.model_checksum = checksum;
        }
        if backend.is_some() {
            self.embedding_backend = backend;
        }
    }

    /// What, if anything, a book needs — from the lexical fingerprint alone.
    ///
    /// This is the decision available at diff time, before the book's lines have
    /// been loaded. A book whose fingerprint cannot prove everything reports
    /// [`BookIndexNeed::Unverifiable`] rather than being assumed current: only a
    /// [`ContentFingerprint::Canonical`] match reaches
    /// [`BookIndexNeed::UpToDate`], because only that kind of fingerprint moves
    /// when the metadata stored in the vectors changes.
    pub fn book_index_need(
        &self,
        source_book_key: &str,
        fingerprint: ContentFingerprint,
        chunking_version: u32,
        normalization_version: u32,
    ) -> BookIndexNeed {
        let Some(entry) = self.books.get(source_book_key) else {
            return BookIndexNeed::Missing;
        };

        if entry.chunking_version != chunking_version
            || entry.normalization_version != normalization_version
        {
            return BookIndexNeed::Changed;
        }

        // A stored `0` is the "no fingerprint" marker, so it can never equal an
        // incoming hash. Belt and braces: `ContentFingerprint` already carries a
        // `NonZeroU64`, but this entry was deserialized from disk and the guard
        // costs nothing.
        let recorded = match NonZeroU64::new(entry.content_hash) {
            Some(recorded) => recorded,
            None => return BookIndexNeed::Unverifiable,
        };

        match fingerprint {
            ContentFingerprint::Canonical(hash) if recorded == hash => BookIndexNeed::UpToDate,
            // The text is provably unchanged, but this fingerprint says nothing
            // about title, category or facets — all of which are stored in the
            // vectors. The lines have to settle it.
            ContentFingerprint::ContentOnly(hash) if recorded == hash => {
                BookIndexNeed::Unverifiable
            }
            ContentFingerprint::Canonical(_) | ContentFingerprint::ContentOnly(_) => {
                BookIndexNeed::Changed
            }
            // Nothing to compare (a PDF whose caller has no signature of its
            // own). Only the lines themselves can settle it.
            ContentFingerprint::Unverifiable => BookIndexNeed::Unverifiable,
        }
    }

    /// Check if a specific book needs re-indexing.
    pub fn book_needs_reindex(
        &self,
        source_book_key: &str,
        fingerprint: ContentFingerprint,
        chunking_version: u32,
        normalization_version: u32,
    ) -> bool {
        self.book_index_need(
            source_book_key,
            fingerprint,
            chunking_version,
            normalization_version,
        )
        .needs_work()
    }

    /// The recorded entry for a book, if it has one.
    pub fn book(&self, source_book_key: &str) -> Option<&BookManifestEntry> {
        self.books.get(source_book_key)
    }

    /// Record that a book has been indexed.
    ///
    /// `chunk_count == 0` is recorded, not skipped: it is the marker that says
    /// the book was processed and has nothing to embed.
    pub fn mark_book_indexed(
        &mut self,
        source_book_key: String,
        content_hash: u64,
        line_fingerprint: u64,
        chunk_count: u32,
        chunking_version: u32,
        normalization_version: u32,
    ) {
        let entry = BookManifestEntry {
            source_book_key: source_book_key.clone(),
            content_hash,
            line_fingerprint,
            chunk_count,
            indexed_at: current_unix_timestamp(),
            chunking_version,
            normalization_version,
        };
        self.books.insert(source_book_key, entry);
    }

    /// Remove a book from the manifest.
    pub fn remove_book(&mut self, source_book_key: &str) -> Option<BookManifestEntry> {
        self.books.remove(source_book_key)
    }

    /// Drop every per-book record, keeping the configuration metadata.
    ///
    /// For a full rebuild. Returns how many records were dropped.
    pub fn clear_books(&mut self) -> usize {
        let dropped = self.books.len();
        self.books.clear();
        dropped
    }

    /// Drop only the records that describe stored vectors, keeping the
    /// empty-book markers.
    ///
    /// Used when the vectors are known to be gone — a non-persistent store after
    /// a restart. Keeping a record that claims vectors would make the manifest
    /// report "up to date" while every query came back empty. A `chunk_count == 0`
    /// record describes no vectors, so nothing about it was lost and dropping it
    /// would only force the same book to be reprocessed every session.
    ///
    /// Returns how many records were dropped.
    pub fn clear_books_with_vectors(&mut self) -> usize {
        let before = self.books.len();
        self.books.retain(|_, entry| entry.chunk_count == 0);
        before - self.books.len()
    }

    /// Number of books recorded as processed but holding no vectors.
    pub fn empty_book_count(&self) -> usize {
        self.books
            .values()
            .filter(|entry| entry.chunk_count == 0)
            .count()
    }

    /// Get the total number of indexed books.
    pub fn book_count(&self) -> usize {
        self.books.len()
    }

    /// Get the total number of chunks across all books.
    pub fn total_chunk_count(&self) -> u32 {
        self.books.values().map(|b| b.chunk_count).sum()
    }

    /// Get manifest file path for a given directory.
    pub fn file_path(dir: &Path) -> PathBuf {
        dir.join(MANIFEST_FILENAME)
    }
}

/// What a book needs, as far as the lexical fingerprint can tell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BookIndexNeed {
    /// No record: the book has never been processed.
    Missing,
    /// The content hash or an algorithm version changed.
    Changed,
    /// The lexical index has no fingerprint for this book (a PDF), so it cannot
    /// be declared current. Its lines have to be compared — see
    /// [`BookManifestEntry::line_fingerprint`].
    Unverifiable,
    /// Recorded, current, and provably so.
    UpToDate,
}

impl BookIndexNeed {
    /// Whether the caller has to hand this book over for processing.
    ///
    /// `Unverifiable` counts: the book must be examined, even though the
    /// examination often concludes that nothing changed.
    pub fn needs_work(&self) -> bool {
        !matches!(self, Self::UpToDate)
    }

    /// Whether this book was ever recorded.
    pub fn is_known(&self) -> bool {
        !matches!(self, Self::Missing)
    }
}

/// Minimal view used to read `format_version` before the full document.
#[derive(Deserialize)]
struct FormatVersionProbe {
    format_version: u32,
}

/// Types of mismatches between manifest and current config.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ManifestMismatch {
    ModelId { manifest: String, config: String },
    ModelChecksum { manifest: String, config: String },
    EmbeddingBackend { manifest: String, config: String },
    Dimensions { manifest: u32, config: u32 },
    Pooling { manifest: String, config: String },
    ModelQuantization { manifest: String, config: String },
    VectorPrecision { manifest: String, config: String },
    VectorBackend { manifest: String, config: String },
    ChunkingVersion { manifest: u32, config: u32 },
    NormalizationVersion { manifest: u32, config: u32 },
}

impl ManifestMismatch {
    /// Whether this mismatch means the vectors themselves are invalid (as
    /// opposed to only the chunk boundaries or text preprocessing).
    ///
    /// Feeds [`IndexDiff::model_mismatch`](crate::semantic::types::IndexDiff).
    pub fn invalidates_vectors(&self) -> bool {
        matches!(
            self,
            Self::ModelId { .. }
                | Self::ModelChecksum { .. }
                | Self::EmbeddingBackend { .. }
                | Self::Dimensions { .. }
                | Self::Pooling { .. }
                | Self::ModelQuantization { .. }
                | Self::VectorPrecision { .. }
                | Self::VectorBackend { .. }
        )
    }
}

impl std::fmt::Display for ManifestMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ModelId { manifest, config } => {
                write!(f, "Model ID: manifest='{manifest}', config='{config}'")
            }
            Self::ModelChecksum { manifest, config } => {
                write!(
                    f,
                    "Model checksum: manifest='{manifest}', config='{config}'"
                )
            }
            Self::EmbeddingBackend { manifest, config } => {
                write!(
                    f,
                    "Embedding backend: manifest='{manifest}', config='{config}'"
                )
            }
            Self::Dimensions { manifest, config } => {
                write!(f, "Dimensions: manifest={manifest}, config={config}")
            }
            Self::Pooling { manifest, config } => {
                write!(f, "Pooling: manifest='{manifest}', config='{config}'")
            }
            Self::ModelQuantization { manifest, config } => {
                write!(
                    f,
                    "Model quantization: manifest='{manifest}', config='{config}'"
                )
            }
            Self::VectorPrecision { manifest, config } => {
                write!(
                    f,
                    "Vector precision: manifest='{manifest}', config='{config}'"
                )
            }
            Self::VectorBackend { manifest, config } => {
                write!(
                    f,
                    "Vector backend: manifest='{manifest}', config='{config}'"
                )
            }
            Self::ChunkingVersion { manifest, config } => {
                write!(f, "Chunking version: manifest={manifest}, config={config}")
            }
            Self::NormalizationVersion { manifest, config } => {
                write!(
                    f,
                    "Normalization version: manifest={manifest}, config={config}"
                )
            }
        }
    }
}

/// Render a mismatch list as one human-readable line.
pub fn describe_mismatches(mismatches: &[ManifestMismatch]) -> String {
    mismatches
        .iter()
        .map(|m| m.to_string())
        .collect::<Vec<_>>()
        .join("; ")
}

/// Flush the directory entry so a rename inside it survives a power loss.
///
/// Unix only. Windows has no equivalent — a directory cannot be opened as a file
/// there — so the rename is left as the filesystem's own guarantee rather than
/// pretending to a durability the platform does not offer. On Unix a failure is
/// returned: silently downgrading to "probably durable" is what makes a crash
/// surprising later.
#[cfg(unix)]
fn sync_directory(dir: &Path) -> Result<(), ManifestError> {
    #[cfg(test)]
    if failpoints::next_dir_sync_fails() {
        return Err(ManifestError::WriteFailed {
            reason: format!(
                "injected directory sync failure for {} (manifest failpoint)",
                dir.display()
            ),
        });
    }

    let handle = std::fs::File::open(dir).map_err(|e| ManifestError::WriteFailed {
        reason: format!(
            "Failed to open {} to flush its directory entry: {e}",
            dir.display()
        ),
    })?;
    handle.sync_all().map_err(|e| ManifestError::WriteFailed {
        reason: format!(
            "Failed to flush the directory entry for {}: {e}",
            dir.display()
        ),
    })
}

/// See the Unix implementation. Nothing to do here; documented, not silent.
#[cfg(not(unix))]
fn sync_directory(_dir: &Path) -> Result<(), ManifestError> {
    Ok(())
}

/// `std::fs::rename`, with a test-only failure injection point.
///
/// [`SemanticManifest::save`]'s fallback exists for a Windows file lock denying a
/// rename, which is not a state a test can reach by arranging files on disk. The
/// alternative to injecting it is an untested recovery path — and an untested
/// recovery path is the one that does not work when it is finally needed.
fn rename(from: &Path, to: &Path) -> std::io::Result<()> {
    #[cfg(test)]
    if failpoints::next_rename_fails() {
        return Err(std::io::Error::other(
            "injected rename failure (manifest failpoint)",
        ));
    }
    std::fs::rename(from, to)
}

/// Failure injection for the save paths that cannot be reached otherwise.
///
/// A schedule rather than a count, because the interesting case is not "renames
/// fail" but "*this* rename fails": the restore path only runs when the retry
/// fails **after** the old manifest was parked, which is the first and third
/// rename failing while the second succeeds.
#[cfg(test)]
mod failpoints {
    use std::cell::RefCell;

    thread_local! {
        /// Pending outcomes for the next renames, consumed front to back.
        /// Thread-local, so a test using it cannot disturb tests running in
        /// parallel.
        static RENAME_SCHEDULE: RefCell<std::collections::VecDeque<bool>> =
            const { RefCell::new(std::collections::VecDeque::new()) };
        /// Pending outcomes for the next directory syncs.
        static DIR_SYNC_SCHEDULE: RefCell<std::collections::VecDeque<bool>> =
            const { RefCell::new(std::collections::VecDeque::new()) };
    }

    /// Schedule which of the next renames fail: `[true, false, true]` fails the
    /// first, lets the second through, fails the third.
    pub fn schedule_rename_failures(schedule: &[bool]) {
        RENAME_SCHEDULE.with(|queue| *queue.borrow_mut() = schedule.iter().copied().collect());
    }

    /// Schedule which of the next directory syncs fail.
    pub fn schedule_dir_sync_failures(schedule: &[bool]) {
        DIR_SYNC_SCHEDULE.with(|queue| *queue.borrow_mut() = schedule.iter().copied().collect());
    }

    pub fn next_rename_fails() -> bool {
        RENAME_SCHEDULE.with(|queue| queue.borrow_mut().pop_front().unwrap_or(false))
    }

    #[allow(dead_code)]
    pub fn next_dir_sync_fails() -> bool {
        DIR_SYNC_SCHEDULE.with(|queue| queue.borrow_mut().pop_front().unwrap_or(false))
    }

    /// Drop any unconsumed schedule, so one test cannot leak into the next on the
    /// same thread.
    pub fn reset() {
        schedule_rename_failures(&[]);
        schedule_dir_sync_failures(&[]);
    }
}

/// Get current Unix timestamp in seconds.
fn current_unix_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> ManifestConfig {
        ManifestConfig {
            embedding_model_id: "EMD123/Otzaria-Embedding-V1-Flash-0.6B".to_string(),
            model_checksum: None,
            embedding_backend: None,
            embedding_dim: 1024,
            pooling: "last-token".to_string(),
            model_quantization: "Q4".to_string(),
            vector_precision: "f32".to_string(),
            vector_backend: "in-memory-v1".to_string(),
            chunking_version: 1,
            normalization_version: 1,
        }
    }

    struct TempDir(PathBuf);
    impl TempDir {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "otzaria_manifest_test_{name}_{}",
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

    #[test]
    fn new_manifest_has_correct_fields() {
        let config = test_config();
        let manifest = SemanticManifest::new(&config);

        assert_eq!(manifest.format_version, MANIFEST_FORMAT_VERSION);
        assert_eq!(manifest.embedding_model_id, config.embedding_model_id);
        assert_eq!(manifest.embedding_dim, 1024);
        assert_eq!(manifest.pooling, "last-token");
        assert_eq!(manifest.chunking_version, 1);
        assert_eq!(manifest.vector_backend, "in-memory-v1");
        assert!(manifest.books.is_empty());
    }

    #[test]
    fn validate_matching_config_returns_empty() {
        let config = test_config();
        let manifest = SemanticManifest::new(&config);
        assert!(manifest.validate(&config).is_empty());
    }

    #[test]
    fn validate_detects_model_mismatch() {
        let config = test_config();
        let manifest = SemanticManifest::new(&config);

        let mut changed = config;
        changed.embedding_model_id = "different-model".to_string();

        let mismatches = manifest.validate(&changed);
        assert_eq!(mismatches.len(), 1);
        assert!(matches!(mismatches[0], ManifestMismatch::ModelId { .. }));
        assert!(mismatches[0].invalidates_vectors());
    }

    #[test]
    fn validate_detects_dimension_mismatch() {
        let config = test_config();
        let manifest = SemanticManifest::new(&config);

        let mut changed = config;
        changed.embedding_dim = 768;

        let mismatches = manifest.validate(&changed);
        assert_eq!(mismatches.len(), 1);
        assert!(matches!(mismatches[0], ManifestMismatch::Dimensions { .. }));
    }

    /// These four dimensions were declared in the manifest but never actually
    /// compared, so switching quantization, precision or storage backend left a
    /// stale index looking valid.
    #[test]
    fn validate_detects_quantization_precision_and_backend_changes() {
        let config = test_config();
        let manifest = SemanticManifest::new(&config);

        let cases: Vec<(&str, ManifestConfig)> = vec![
            (
                "quantization",
                ManifestConfig {
                    model_quantization: "Q8".to_string(),
                    ..config.clone()
                },
            ),
            (
                "precision",
                ManifestConfig {
                    vector_precision: "f16".to_string(),
                    ..config.clone()
                },
            ),
            (
                "vector backend",
                ManifestConfig {
                    vector_backend: "zvec-v1".to_string(),
                    ..config.clone()
                },
            ),
            (
                "pooling",
                ManifestConfig {
                    pooling: "mean".to_string(),
                    ..config.clone()
                },
            ),
        ];

        for (name, changed) in cases {
            let mismatches = manifest.validate(&changed);
            assert_eq!(mismatches.len(), 1, "{name} should produce one mismatch");
            assert!(
                mismatches[0].invalidates_vectors(),
                "{name} invalidates stored vectors"
            );
        }
    }

    #[test]
    fn validate_detects_a_changed_model_file_behind_the_same_id() {
        let mut config = test_config();
        config.model_checksum = Some("aaaa".to_string());
        config.embedding_backend = Some("mock-hash-v1".to_string());
        let manifest = SemanticManifest::new(&config);
        assert!(manifest.validate(&config).is_empty());

        let mut swapped = config.clone();
        swapped.model_checksum = Some("bbbb".to_string());
        let mismatches = manifest.validate(&swapped);
        assert_eq!(mismatches.len(), 1);
        assert!(matches!(
            mismatches[0],
            ManifestMismatch::ModelChecksum { .. }
        ));

        let mut other_backend = config;
        other_backend.embedding_backend = Some("gguf-candle-v1".to_string());
        let mismatches = manifest.validate(&other_backend);
        assert_eq!(mismatches.len(), 1);
        assert!(matches!(
            mismatches[0],
            ManifestMismatch::EmbeddingBackend { .. }
        ));
    }

    /// Before a model is loaded the checksum and backend are unknown. Unknown
    /// must not read as "changed", or every startup would look incompatible.
    #[test]
    fn unknown_checksum_or_backend_is_not_a_mismatch() {
        let mut recorded = test_config();
        recorded.model_checksum = Some("aaaa".to_string());
        recorded.embedding_backend = Some("mock-hash-v1".to_string());
        let manifest = SemanticManifest::new(&recorded);

        // Config side unknown (model not loaded yet).
        assert!(manifest.validate(&test_config()).is_empty());

        // Manifest side unknown (index built before a model was ever loaded).
        let fresh = SemanticManifest::new(&test_config());
        assert!(fresh.validate(&recorded).is_empty());
    }

    #[test]
    fn validate_reports_every_mismatch_at_once() {
        let config = test_config();
        let manifest = SemanticManifest::new(&config);

        let changed = ManifestConfig {
            embedding_model_id: "other".to_string(),
            embedding_dim: 256,
            chunking_version: 9,
            normalization_version: 4,
            ..config
        };

        let mismatches = manifest.validate(&changed);
        assert_eq!(mismatches.len(), 4);
        assert!(!describe_mismatches(&mismatches).is_empty());
    }

    #[test]
    fn chunking_and_normalization_changes_do_not_invalidate_the_vector_space() {
        let config = test_config();
        let manifest = SemanticManifest::new(&config);

        for changed in [
            ManifestConfig {
                chunking_version: 2,
                ..config.clone()
            },
            ManifestConfig {
                normalization_version: 2,
                ..config.clone()
            },
        ] {
            let mismatches = manifest.validate(&changed);
            assert_eq!(mismatches.len(), 1);
            assert!(
                !mismatches[0].invalidates_vectors(),
                "chunking/normalization changes require re-chunking, not a new vector space"
            );
        }
    }

    /// A canonical fingerprint, the kind that can reach "up to date".
    fn hash(value: u64) -> ContentFingerprint {
        ContentFingerprint::from_lexical_hash(value)
    }

    #[test]
    fn book_tracking_lifecycle() {
        let config = test_config();
        let mut manifest = SemanticManifest::new(&config);

        assert_eq!(
            manifest.book_index_need("book_a", hash(12345), 1, 1),
            BookIndexNeed::Missing
        );
        manifest.mark_book_indexed("book_a".to_string(), 12345, 777, 100, 1, 1);
        assert_eq!(
            manifest.book_index_need("book_a", hash(12345), 1, 1),
            BookIndexNeed::UpToDate
        );
        assert!(!manifest.book_needs_reindex("book_a", hash(12345), 1, 1));

        // Changed content, chunking or normalization all force a re-index.
        for (name, need) in [
            (
                "content",
                manifest.book_index_need("book_a", hash(99999), 1, 1),
            ),
            (
                "chunking",
                manifest.book_index_need("book_a", hash(12345), 2, 1),
            ),
            (
                "normalization",
                manifest.book_index_need("book_a", hash(12345), 1, 2),
            ),
        ] {
            assert_eq!(need, BookIndexNeed::Changed, "{name} must force a re-index");
            assert!(need.needs_work());
            assert!(need.is_known());
        }

        assert_eq!(manifest.total_chunk_count(), 100);
        assert_eq!(manifest.book("book_a").unwrap().line_fingerprint, 777);
        assert!(manifest.remove_book("book_a").is_some());
        assert_eq!(
            manifest.book_index_need("book_a", hash(12345), 1, 1),
            BookIndexNeed::Missing
        );
    }

    /// The lexical engine records `contentHash = 0` for PDFs. Treating it as a
    /// hash means `0 == 0` reads as "unchanged", so a replaced PDF would never be
    /// re-indexed. It has to report as unverifiable instead.
    #[test]
    fn a_book_without_a_lexical_fingerprint_is_never_declared_up_to_date() {
        let config = test_config();
        let mut manifest = SemanticManifest::new(&config);
        manifest.mark_book_indexed("scan.pdf".to_string(), 0, 777, 12, 1, 1);

        let need = manifest.book_index_need("scan.pdf", ContentFingerprint::Unverifiable, 1, 1);
        assert_eq!(need, BookIndexNeed::Unverifiable);
        assert!(
            need.needs_work(),
            "the caller has to hand the book over so its lines can be compared"
        );
        assert!(need.is_known(), "but it is not a new book");

        // A version change still takes precedence: nothing about the lines can
        // rescue an index built by different chunking.
        assert_eq!(
            manifest.book_index_need("scan.pdf", ContentFingerprint::Unverifiable, 2, 1),
            BookIndexNeed::Changed
        );
    }

    /// A fingerprint that covers the content and nothing else cannot license a
    /// skip. Renaming a book or correcting its author changes what every one of
    /// its vectors carries while leaving the file's bytes — and therefore a
    /// size/mtime signature — identical.
    #[test]
    fn a_content_only_fingerprint_never_reaches_up_to_date() {
        let config = test_config();
        let mut manifest = SemanticManifest::new(&config);
        let signature = 0xBEEF_CAFE;
        manifest.mark_book_indexed("scan.pdf".to_string(), signature, 777, 12, 1, 1);

        assert_eq!(
            manifest.book_index_need(
                "scan.pdf",
                ContentFingerprint::content_only(signature),
                1,
                1
            ),
            BookIndexNeed::Unverifiable,
            "the text is provably unchanged; the metadata is not, so the lines decide"
        );

        // A different signature is a different matter: the content really did
        // change, and that is known rather than merely unproven.
        assert_eq!(
            manifest.book_index_need("scan.pdf", ContentFingerprint::content_only(999), 1, 1),
            BookIndexNeed::Changed
        );

        // The canonical form is what reaches "nothing to do".
        assert_eq!(
            manifest.book_index_need(
                "scan.pdf",
                ContentFingerprint::from_lexical_hash(signature),
                1,
                1
            ),
            BookIndexNeed::UpToDate
        );
    }

    /// A recorded `0` is the "no fingerprint" marker. Nothing may ever match it —
    /// including a caller that somehow presents zero as a hash, which the type
    /// makes impossible and this checks anyway, because the record came off disk.
    #[test]
    fn a_recorded_zero_fingerprint_matches_nothing() {
        let config = test_config();
        let mut manifest = SemanticManifest::new(&config);
        manifest.mark_book_indexed("scan.pdf".to_string(), 0, 777, 12, 1, 1);

        for fingerprint in [
            ContentFingerprint::from_lexical_hash(0),
            ContentFingerprint::content_only(0),
            ContentFingerprint::from_lexical_hash(1),
            ContentFingerprint::canonical(1, "כותרת", "/מקרא", &[], true),
        ] {
            assert_ne!(
                manifest.book_index_need("scan.pdf", fingerprint, 1, 1),
                BookIndexNeed::UpToDate,
                "{fingerprint:?} must not match a record that has no fingerprint"
            );
        }
    }

    /// A book that yielded nothing embeddable must stay recorded, or every
    /// startup reports it as new and reprocesses it.
    #[test]
    fn an_empty_book_record_is_a_valid_up_to_date_record() {
        let config = test_config();
        let mut manifest = SemanticManifest::new(&config);
        manifest.mark_book_indexed("headings-only.txt".to_string(), 42, 777, 0, 1, 1);

        assert_eq!(
            manifest.book_index_need("headings-only.txt", hash(42), 1, 1),
            BookIndexNeed::UpToDate
        );
        assert_eq!(manifest.book_count(), 1);
        assert_eq!(manifest.empty_book_count(), 1);
        assert_eq!(manifest.total_chunk_count(), 0);
    }

    /// When the vectors are gone, records that claimed vectors are stale — but an
    /// empty-book marker described no vectors, so nothing about it was lost.
    #[test]
    fn clearing_lost_vectors_keeps_the_empty_book_markers() {
        let config = test_config();
        let mut manifest = SemanticManifest::new(&config);
        manifest.mark_book_indexed("with-vectors.txt".to_string(), 1, 111, 10, 1, 1);
        manifest.mark_book_indexed("scanned.pdf".to_string(), 0, 222, 0, 1, 1);

        assert_eq!(manifest.clear_books_with_vectors(), 1);
        assert_eq!(manifest.book_count(), 1);
        assert!(manifest.book("scanned.pdf").is_some());
        assert!(manifest.book("with-vectors.txt").is_none());

        // The surviving marker still answers "nothing to do" for its own lines.
        assert_eq!(
            manifest.book_index_need("scanned.pdf", ContentFingerprint::Unverifiable, 1, 1),
            BookIndexNeed::Unverifiable
        );
    }

    #[test]
    fn clear_books_keeps_configuration_but_drops_records() {
        let config = test_config();
        let mut manifest = SemanticManifest::new(&config);
        manifest.mark_book_indexed("a".to_string(), 1, 11, 10, 1, 1);
        manifest.mark_book_indexed("b".to_string(), 2, 22, 20, 1, 1);

        assert_eq!(manifest.clear_books(), 2);
        assert_eq!(manifest.book_count(), 0);
        assert_eq!(manifest.total_chunk_count(), 0);
        assert!(
            manifest.validate(&config).is_empty(),
            "configuration metadata must survive"
        );
    }

    #[test]
    fn save_and_load_roundtrip() {
        let dir = TempDir::new("roundtrip");
        let mut config = test_config();
        config.model_checksum = Some("deadbeef".to_string());
        config.embedding_backend = Some("mock-hash-v1".to_string());

        let mut manifest = SemanticManifest::new(&config);
        manifest.mark_book_indexed("book_a".to_string(), 12345, 777, 100, 1, 1);
        manifest.save(dir.path()).unwrap();

        let loaded = SemanticManifest::load(dir.path()).unwrap();
        assert_eq!(loaded.embedding_model_id, manifest.embedding_model_id);
        assert_eq!(loaded.embedding_dim, manifest.embedding_dim);
        assert_eq!(loaded.model_checksum.as_deref(), Some("deadbeef"));
        assert_eq!(loaded.embedding_backend.as_deref(), Some("mock-hash-v1"));
        assert_eq!(loaded.vector_backend, "in-memory-v1");
        assert_eq!(loaded.books.len(), 1);
        assert_eq!(loaded.book("book_a").unwrap().line_fingerprint, 777);
        assert!(loaded.validate(&config).is_empty());
    }

    #[test]
    fn load_reports_a_missing_manifest_distinctly_from_a_broken_one() {
        let dir = TempDir::new("missing");
        assert!(matches!(
            SemanticManifest::load(dir.path()),
            Err(ManifestError::NotFound { .. })
        ));
    }

    #[test]
    fn load_rejects_corrupt_json() {
        let dir = TempDir::new("corrupt");
        std::fs::write(
            SemanticManifest::file_path(dir.path()),
            b"{ this is not json",
        )
        .unwrap();

        assert!(matches!(
            SemanticManifest::load(dir.path()),
            Err(ManifestError::ParseFailed { .. })
        ));
    }

    #[test]
    fn load_rejects_a_truncated_but_syntactically_valid_manifest() {
        let dir = TempDir::new("truncated");
        // Valid JSON, right format version, missing every other field.
        std::fs::write(
            SemanticManifest::file_path(dir.path()),
            format!("{{\"format_version\": {MANIFEST_FORMAT_VERSION}}}").as_bytes(),
        )
        .unwrap();

        assert!(matches!(
            SemanticManifest::load(dir.path()),
            Err(ManifestError::ParseFailed { .. })
        ));
    }

    #[test]
    fn load_rejects_other_format_versions() {
        let dir = TempDir::new("format_version");

        for version in [1u32, MANIFEST_FORMAT_VERSION + 1] {
            std::fs::write(
                SemanticManifest::file_path(dir.path()),
                format!("{{\"format_version\": {version}}}").as_bytes(),
            )
            .unwrap();

            match SemanticManifest::load(dir.path()) {
                Err(ManifestError::UnsupportedFormatVersion { found, supported }) => {
                    assert_eq!(found, version);
                    assert_eq!(supported, MANIFEST_FORMAT_VERSION);
                }
                other => panic!("expected a format-version error for {version}, got {other:?}"),
            }
        }
    }

    #[test]
    fn quarantine_moves_the_file_aside_and_preserves_its_bytes() {
        let dir = TempDir::new("quarantine");
        let path = SemanticManifest::file_path(dir.path());
        std::fs::write(&path, b"{ broken").unwrap();

        let moved = SemanticManifest::quarantine(dir.path(), "corrupt").unwrap();
        assert!(!path.exists(), "the unusable manifest must be moved away");
        assert!(moved.exists());
        assert_eq!(std::fs::read(&moved).unwrap(), b"{ broken");
        assert!(moved
            .file_name()
            .unwrap()
            .to_string_lossy()
            .contains("corrupt"));
    }

    #[test]
    fn quarantine_of_a_missing_file_is_an_error_not_a_panic() {
        let dir = TempDir::new("quarantine_missing");
        assert!(SemanticManifest::quarantine(dir.path(), "corrupt").is_err());
    }

    #[test]
    fn save_replaces_an_existing_manifest_and_leaves_no_temp_file() {
        let dir = TempDir::new("replace");
        let config = test_config();

        let mut first = SemanticManifest::new(&config);
        first.mark_book_indexed("a".to_string(), 1, 11, 10, 1, 1);
        first.save(dir.path()).unwrap();

        let mut second = SemanticManifest::new(&config);
        second.mark_book_indexed("b".to_string(), 2, 22, 20, 1, 1);
        second.save(dir.path()).unwrap();

        let loaded = SemanticManifest::load(dir.path()).unwrap();
        assert_eq!(loaded.books.len(), 1);
        assert!(loaded.books.contains_key("b"));

        let tmp = dir.path().join(format!("{MANIFEST_FILENAME}.tmp"));
        assert!(
            !tmp.exists(),
            "the temp file must be renamed, not left behind"
        );
    }

    /// The retry path must never be a window with no manifest on disk: a crash
    /// there would lose the index state entirely, which is worse than keeping
    /// either version.
    #[test]
    fn a_readable_manifest_exists_at_every_point_of_a_rewrite() {
        let dir = TempDir::new("always_readable");
        let config = test_config();

        let mut manifest = SemanticManifest::new(&config);
        manifest.mark_book_indexed("a".to_string(), 1, 11, 10, 1, 1);
        manifest.save(dir.path()).unwrap();

        // Rewrite repeatedly; after each one the manifest must still parse and no
        // temporary or side-stepped file may be left behind.
        for round in 2..6u64 {
            let mut next = SemanticManifest::new(&config);
            next.mark_book_indexed(format!("book{round}"), round, round * 11, 10, 1, 1);
            next.save(dir.path()).unwrap();

            let loaded = SemanticManifest::load(dir.path()).unwrap();
            assert_eq!(loaded.book_count(), 1);
            assert!(loaded.book(&format!("book{round}")).is_some());

            assert!(
                !dir.path().join(format!("{MANIFEST_FILENAME}.tmp")).exists(),
                "the temp file must not survive a successful save"
            );
            assert!(
                !SemanticManifest::previous_path(dir.path()).exists(),
                "the parked copy must not survive a successful save"
            );
        }
    }

    /// Simulates a crash inside `save`'s fallback window: the manifest is parked
    /// under `.previous` and the target is gone. Without recovery this reads as a
    /// first run and the entire index is silently rebuilt.
    #[test]
    fn a_manifest_parked_by_an_interrupted_save_is_recovered() {
        let dir = TempDir::new("parked_recovery");
        let config = test_config();

        let mut manifest = SemanticManifest::new(&config);
        manifest.mark_book_indexed("book_a".to_string(), 12345, 777, 100, 1, 1);
        manifest.save(dir.path()).unwrap();

        // Reproduce the window exactly: target parked, nothing in its place.
        std::fs::rename(
            SemanticManifest::file_path(dir.path()),
            SemanticManifest::previous_path(dir.path()),
        )
        .unwrap();
        assert!(!SemanticManifest::file_path(dir.path()).exists());

        let recovered = SemanticManifest::load(dir.path())
            .expect("a parked manifest must be recovered, not reported as a first run");
        assert_eq!(recovered.book_count(), 1);
        assert!(recovered.book("book_a").is_some());

        // Recovery is a move, so the file is back in place for the next open and
        // the parked copy is gone.
        assert!(SemanticManifest::file_path(dir.path()).exists());
        assert!(!SemanticManifest::previous_path(dir.path()).exists());
        assert_eq!(SemanticManifest::load(dir.path()).unwrap().book_count(), 1);
    }

    /// A `.tmp` that outlived a crash is a complete manifest: `save` `fsync`s it
    /// before renaming. With nothing else on disk it is better than declaring a
    /// first run and silently rebuilding the whole index.
    #[test]
    fn a_temp_file_left_by_an_interrupted_save_is_recovered() {
        let dir = TempDir::new("tmp_recovery");
        let mut manifest = SemanticManifest::new(&test_config());
        manifest.mark_book_indexed("book_a".to_string(), 12345, 777, 100, 1, 1);
        manifest.save(dir.path()).unwrap();

        // The crash window: the payload is written and flushed, the rename never
        // happened.
        std::fs::rename(
            SemanticManifest::file_path(dir.path()),
            SemanticManifest::tmp_path(dir.path()),
        )
        .unwrap();

        let recovered = SemanticManifest::load(dir.path())
            .expect("a flushed temp file must be recovered, not treated as a first run");
        assert_eq!(recovered.book_count(), 1);
        assert!(SemanticManifest::file_path(dir.path()).exists());
        assert!(!SemanticManifest::tmp_path(dir.path()).exists());
    }

    /// A half-written `.tmp` — the crash landed inside `write_all` — must not be
    /// promoted. Parsing it before the rename is what separates the two cases.
    #[test]
    fn a_half_written_temp_file_is_not_recovered() {
        let dir = TempDir::new("tmp_partial");
        std::fs::write(
            SemanticManifest::tmp_path(dir.path()),
            br#"{"format_version": 3, "embedding_model_id": "#,
        )
        .unwrap();

        assert!(matches!(
            SemanticManifest::load(dir.path()),
            Err(ManifestError::NotFound { .. })
        ));
        assert!(!SemanticManifest::file_path(dir.path()).exists());
    }

    /// The parked copy is preferred: it was a manifest in service, while a `.tmp`
    /// was only ever a candidate.
    #[test]
    fn a_parked_copy_wins_over_a_temp_file() {
        let dir = TempDir::new("recovery_order");
        let config = test_config();

        let mut in_service = SemanticManifest::new(&config);
        in_service.mark_book_indexed("parked_book".to_string(), 1, 1, 5, 1, 1);
        in_service.save(dir.path()).unwrap();
        std::fs::rename(
            SemanticManifest::file_path(dir.path()),
            SemanticManifest::previous_path(dir.path()),
        )
        .unwrap();

        let mut candidate = SemanticManifest::new(&config);
        candidate.mark_book_indexed("tmp_book".to_string(), 2, 2, 7, 1, 1);
        candidate.save(dir.path()).unwrap();
        std::fs::rename(
            SemanticManifest::file_path(dir.path()),
            SemanticManifest::tmp_path(dir.path()),
        )
        .unwrap();

        let recovered = SemanticManifest::load(dir.path()).unwrap();
        assert!(recovered.book("parked_book").is_some());
        assert!(recovered.book("tmp_book").is_none());
    }

    /// A file that contains only a matching version probe is not recoverable.
    /// Recovery must validate the whole schema before preferring `.previous`
    /// over a complete `.tmp`.
    #[test]
    fn an_incomplete_parked_copy_does_not_hide_a_complete_temp_file() {
        let dir = TempDir::new("recovery_requires_full_schema");
        std::fs::write(
            SemanticManifest::previous_path(dir.path()),
            format!("{{\"format_version\": {MANIFEST_FORMAT_VERSION}}}"),
        )
        .unwrap();

        let mut candidate = SemanticManifest::new(&test_config());
        candidate.mark_book_indexed("tmp_book".to_string(), 2, 2, 7, 1, 1);
        std::fs::write(
            SemanticManifest::tmp_path(dir.path()),
            serde_json::to_vec_pretty(&candidate).unwrap(),
        )
        .unwrap();

        let recovered = SemanticManifest::load(dir.path()).unwrap();
        assert!(recovered.book("tmp_book").is_some());
        assert!(
            SemanticManifest::previous_path(dir.path()).exists(),
            "the rejected copy is evidence and must not be deleted"
        );
    }

    // ── the save fallback, reached by injecting the failure it exists for ──

    /// A denied rename (a Windows file lock, in practice) must still leave a
    /// correct manifest in place, with no leftovers.
    #[test]
    fn a_denied_rename_is_retried_through_a_parked_copy() {
        let dir = TempDir::new("rename_retry");
        let config = test_config();
        failpoints::reset();

        let mut first = SemanticManifest::new(&config);
        first.mark_book_indexed("old_book".to_string(), 1, 1, 3, 1, 1);
        first.save(dir.path()).unwrap();

        let mut second = SemanticManifest::new(&config);
        second.mark_book_indexed("new_book".to_string(), 2, 2, 4, 1, 1);
        // Fail the direct rename; let the park and the retry through.
        failpoints::schedule_rename_failures(&[true]);
        second
            .save(dir.path())
            .expect("the fallback exists so that a denied rename still succeeds");
        failpoints::reset();

        let loaded = SemanticManifest::load(dir.path()).unwrap();
        assert!(
            loaded.book("new_book").is_some(),
            "the new manifest is live"
        );
        assert!(loaded.book("old_book").is_none());
        assert!(
            !SemanticManifest::previous_path(dir.path()).exists(),
            "the parked copy is transient, not debris"
        );
        assert!(!SemanticManifest::tmp_path(dir.path()).exists());
    }

    /// When the retry fails too, the old manifest goes back. The one outcome that
    /// is not allowed is a directory with no manifest at all.
    #[test]
    fn a_failed_retry_restores_the_previous_manifest() {
        let dir = TempDir::new("rename_restore");
        let config = test_config();
        failpoints::reset();

        let mut first = SemanticManifest::new(&config);
        first.mark_book_indexed("old_book".to_string(), 1, 1, 3, 1, 1);
        first.save(dir.path()).unwrap();

        let mut second = SemanticManifest::new(&config);
        second.mark_book_indexed("new_book".to_string(), 2, 2, 4, 1, 1);
        // Fail the first rename, park successfully, fail the retry, restore.
        failpoints::schedule_rename_failures(&[true, false, true, false]);
        let error = second.save(dir.path());
        failpoints::reset();

        assert!(
            matches!(error, Err(ManifestError::WriteFailed { .. })),
            "a save that did not happen must not report success"
        );
        let loaded = SemanticManifest::load(dir.path())
            .expect("the previous manifest must be back in place");
        assert!(loaded.book("old_book").is_some());
        assert!(loaded.book("new_book").is_none());
    }

    /// Even when every rename is denied, `load` finds a manifest. Here the park
    /// itself fails, so the target was never moved.
    #[test]
    fn a_manifest_remains_readable_when_every_rename_is_denied() {
        let dir = TempDir::new("rename_all_denied");
        let config = test_config();
        failpoints::reset();

        let mut first = SemanticManifest::new(&config);
        first.mark_book_indexed("old_book".to_string(), 1, 1, 3, 1, 1);
        first.save(dir.path()).unwrap();

        let mut second = SemanticManifest::new(&config);
        second.mark_book_indexed("new_book".to_string(), 2, 2, 4, 1, 1);
        failpoints::schedule_rename_failures(&[true, true, true, true]);
        assert!(second.save(dir.path()).is_err());
        failpoints::reset();

        assert!(SemanticManifest::load(dir.path())
            .unwrap()
            .book("old_book")
            .is_some());
    }

    /// A directory `fsync` that fails means the rename may not survive a power
    /// loss, so on Unix the save reports failure — the caller must not record
    /// progress it might lose. The *contents* are still correct: this is a
    /// durability failure, not an atomicity one.
    #[cfg(unix)]
    #[test]
    fn a_failed_directory_sync_fails_the_save_on_unix() {
        let dir = TempDir::new("dir_sync_failure");
        let config = test_config();
        failpoints::reset();

        let mut manifest = SemanticManifest::new(&config);
        manifest.mark_book_indexed("book_a".to_string(), 1, 1, 3, 1, 1);
        failpoints::schedule_dir_sync_failures(&[true]);
        let error = manifest.save(dir.path());
        failpoints::reset();

        assert!(
            matches!(error, Err(ManifestError::WriteFailed { .. })),
            "returning Ok here would promise durability the platform did not give"
        );
        assert_eq!(
            manifest.save_count(),
            0,
            "a failed durability step is not a successful save"
        );
        assert_eq!(
            SemanticManifest::load(dir.path()).unwrap().book_count(),
            1,
            "the manifest itself is intact — only its durability is unproven"
        );
    }

    /// If fallback parked the old manifest, it cannot be deleted until the new
    /// live name is durable. That copy is the recovery guarantee for this exact
    /// failure window.
    #[cfg(unix)]
    #[test]
    fn a_directory_sync_failure_keeps_the_parked_manifest() {
        let dir = TempDir::new("fallback_dir_sync_failure");
        let config = test_config();
        failpoints::reset();

        let mut first = SemanticManifest::new(&config);
        first.mark_book_indexed("old_book".to_string(), 1, 1, 3, 1, 1);
        first.save(dir.path()).unwrap();

        let mut second = SemanticManifest::new(&config);
        second.mark_book_indexed("new_book".to_string(), 2, 2, 4, 1, 1);
        failpoints::schedule_rename_failures(&[true]);
        failpoints::schedule_dir_sync_failures(&[true]);
        let error = second.save(dir.path());
        failpoints::reset();

        assert!(matches!(error, Err(ManifestError::WriteFailed { .. })));
        assert_eq!(second.save_count(), 0);
        let parked = SemanticManifest::previous_path(dir.path());
        assert!(parked.exists(), "the durable old copy must be retained");
        let old: SemanticManifest =
            serde_json::from_slice(&std::fs::read(parked).unwrap()).unwrap();
        assert!(old.book("old_book").is_some());
    }

    /// With neither file present it really is a first run.
    #[test]
    fn no_manifest_and_no_parked_copy_is_still_a_first_run() {
        let dir = TempDir::new("genuinely_missing");
        assert!(matches!(
            SemanticManifest::load(dir.path()),
            Err(ManifestError::NotFound { .. })
        ));
    }

    /// A leftover copy is parsed before it is promoted, so an unusable one is
    /// passed over rather than renamed into place. It is also left where it is:
    /// nothing on this path deletes evidence.
    #[test]
    fn a_corrupt_parked_copy_is_not_promoted_into_place() {
        let dir = TempDir::new("parked_corrupt");
        let parked = SemanticManifest::previous_path(dir.path());
        std::fs::write(&parked, b"{ broken").unwrap();

        assert!(
            matches!(
                SemanticManifest::load(dir.path()),
                Err(ManifestError::NotFound { .. })
            ),
            "an unusable leftover is not a manifest, so there is none"
        );
        assert!(parked.exists(), "the corrupt copy must be left as evidence");
        assert!(
            !SemanticManifest::file_path(dir.path()).exists(),
            "it must not have been promoted to the live name"
        );
    }

    #[test]
    fn save_creates_a_missing_directory() {
        let dir = TempDir::new("nested");
        let nested = dir.path().join("a").join("b");

        let mut manifest = SemanticManifest::new(&test_config());
        manifest.save(&nested).unwrap();
        assert!(SemanticManifest::load(&nested).is_ok());
    }
}
