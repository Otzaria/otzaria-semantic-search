//! Semantic index manifest and versioning.
//!
//! The manifest tracks the exact configuration used to build the semantic index:
//! embedding model, dimensions, chunking algorithm version, and per-book status.
//!
//! On startup, the manifest is compared against the current configuration.
//! Any mismatch disables the semantic search path (graceful degradation to BM25)
//! until the index is rebuilt.
//!
//! # Atomic Persistence
//!
//! The manifest is always written atomically: write to a `.tmp` file, then
//! rename over the target. This prevents partial/corrupt manifests if the
//! app crashes mid-write.

use crate::errors::ManifestError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Current manifest format version. Bump when the schema changes.
const MANIFEST_FORMAT_VERSION: u32 = 1;

/// Manifest file name.
const MANIFEST_FILENAME: &str = "semantic_manifest.json";

/// Full semantic index manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticManifest {
    /// Format version of this manifest (for forward compatibility).
    pub format_version: u32,

    // ── Model metadata ──
    /// Embedding model identifier (e.g. "EMD123/Otzaria-Embedding-V1-Flash-0.6B").
    pub embedding_model_id: String,
    /// Model file checksum (SHA256 of the GGUF file) for integrity verification.
    pub model_checksum: Option<String>,
    /// Embedding vector dimensionality (e.g. 1024).
    pub embedding_dim: u32,
    /// Pooling strategy used (e.g. "last-token").
    pub pooling: String,
    /// Model quantization level (e.g. "Q4").
    pub model_quantization: String,
    /// Vector storage precision in the store (e.g. "f32", "f16").
    pub vector_precision: String,

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
}

/// Per-book entry in the manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BookManifestEntry {
    /// Stable book identifier (file path).
    pub source_book_key: String,
    /// Content hash at the time of indexing (matches Tantivy contentHash).
    pub content_hash: u64,
    /// Number of semantic chunks generated for this book.
    pub chunk_count: u32,
    /// When this book was last indexed (Unix timestamp).
    pub indexed_at: u64,
    /// Chunking version used for this specific book.
    pub chunking_version: u32,
    /// Normalization version used for this specific book.
    pub normalization_version: u32,
}

/// Configuration to compare against the manifest.
#[derive(Debug, Clone)]
pub struct ManifestConfig {
    pub embedding_model_id: String,
    pub embedding_dim: u32,
    pub pooling: String,
    pub model_quantization: String,
    pub vector_precision: String,
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
            model_checksum: None,
            embedding_dim: config.embedding_dim,
            pooling: config.pooling.clone(),
            model_quantization: config.model_quantization.clone(),
            vector_precision: config.vector_precision.clone(),
            chunking_version: config.chunking_version,
            normalization_version: config.normalization_version,
            created_at: now,
            updated_at: now,
            books: HashMap::new(),
        }
    }

    /// Load a manifest from the given directory.
    pub fn load(dir: &Path) -> Result<Self, ManifestError> {
        let path = dir.join(MANIFEST_FILENAME);
        if !path.exists() {
            return Err(ManifestError::NotFound {
                path: path.display().to_string(),
            });
        }

        let content = std::fs::read_to_string(&path).map_err(|e| ManifestError::ParseFailed {
            reason: format!("Failed to read manifest: {e}"),
        })?;

        let manifest: Self =
            serde_json::from_str(&content).map_err(|e| ManifestError::ParseFailed {
                reason: format!("Failed to parse manifest JSON: {e}"),
            })?;

        Ok(manifest)
    }

    /// Save the manifest to the given directory (atomic write).
    pub fn save(&mut self, dir: &Path) -> Result<(), ManifestError> {
        self.updated_at = current_unix_timestamp();

        let target = dir.join(MANIFEST_FILENAME);
        let tmp = dir.join(format!("{MANIFEST_FILENAME}.tmp"));

        // Ensure the directory exists
        std::fs::create_dir_all(dir).map_err(|e| ManifestError::WriteFailed {
            reason: format!("Failed to create manifest directory: {e}"),
        })?;

        // Write to temp file
        let content =
            serde_json::to_string_pretty(self).map_err(|e| ManifestError::WriteFailed {
                reason: format!("Failed to serialize manifest: {e}"),
            })?;

        std::fs::write(&tmp, content.as_bytes()).map_err(|e| ManifestError::WriteFailed {
            reason: format!("Failed to write temp manifest: {e}"),
        })?;

        // Atomic rename (remove target first if present on Windows to prevent error 183 / Access Denied)
        if target.exists() {
            let _ = std::fs::remove_file(&target);
        }

        std::fs::rename(&tmp, &target).map_err(|e| ManifestError::WriteFailed {
            reason: format!("Failed to rename temp manifest to final: {e}"),
        })?;

        log::info!("Manifest saved to {}", target.display());
        Ok(())
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

    /// Check if a specific book needs re-indexing.
    pub fn book_needs_reindex(
        &self,
        source_book_key: &str,
        content_hash: u64,
        chunking_version: u32,
        normalization_version: u32,
    ) -> bool {
        match self.books.get(source_book_key) {
            None => true, // New book
            Some(entry) => {
                entry.content_hash != content_hash
                    || entry.chunking_version != chunking_version
                    || entry.normalization_version != normalization_version
            }
        }
    }

    /// Record that a book has been indexed.
    pub fn mark_book_indexed(
        &mut self,
        source_book_key: String,
        content_hash: u64,
        chunk_count: u32,
        chunking_version: u32,
        normalization_version: u32,
    ) {
        let entry = BookManifestEntry {
            source_book_key: source_book_key.clone(),
            content_hash,
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

/// Types of mismatches between manifest and current config.
#[derive(Debug, Clone)]
pub enum ManifestMismatch {
    ModelId { manifest: String, config: String },
    Dimensions { manifest: u32, config: u32 },
    Pooling { manifest: String, config: String },
    ChunkingVersion { manifest: u32, config: u32 },
    NormalizationVersion { manifest: u32, config: u32 },
}

impl std::fmt::Display for ManifestMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ModelId { manifest, config } => {
                write!(f, "Model ID: manifest='{manifest}', config='{config}'")
            }
            Self::Dimensions { manifest, config } => {
                write!(f, "Dimensions: manifest={manifest}, config={config}")
            }
            Self::Pooling { manifest, config } => {
                write!(f, "Pooling: manifest='{manifest}', config='{config}'")
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
            embedding_dim: 1024,
            pooling: "last-token".to_string(),
            model_quantization: "Q4".to_string(),
            vector_precision: "f32".to_string(),
            chunking_version: 1,
            normalization_version: 1,
        }
    }

    #[test]
    fn new_manifest_has_correct_fields() {
        let config = test_config();
        let manifest = SemanticManifest::new(&config);

        assert_eq!(manifest.embedding_model_id, config.embedding_model_id);
        assert_eq!(manifest.embedding_dim, 1024);
        assert_eq!(manifest.pooling, "last-token");
        assert_eq!(manifest.chunking_version, 1);
        assert!(manifest.books.is_empty());
    }

    #[test]
    fn validate_matching_config_returns_empty() {
        let config = test_config();
        let manifest = SemanticManifest::new(&config);
        let mismatches = manifest.validate(&config);
        assert!(mismatches.is_empty());
    }

    #[test]
    fn validate_detects_model_mismatch() {
        let config = test_config();
        let manifest = SemanticManifest::new(&config);

        let mut changed_config = config;
        changed_config.embedding_model_id = "different-model".to_string();

        let mismatches = manifest.validate(&changed_config);
        assert_eq!(mismatches.len(), 1);
        assert!(matches!(mismatches[0], ManifestMismatch::ModelId { .. }));
    }

    #[test]
    fn validate_detects_dimension_mismatch() {
        let config = test_config();
        let manifest = SemanticManifest::new(&config);

        let mut changed_config = config;
        changed_config.embedding_dim = 768;

        let mismatches = manifest.validate(&changed_config);
        assert_eq!(mismatches.len(), 1);
        assert!(matches!(mismatches[0], ManifestMismatch::Dimensions { .. }));
    }

    #[test]
    fn book_tracking_lifecycle() {
        let config = test_config();
        let mut manifest = SemanticManifest::new(&config);

        // New book needs indexing
        assert!(manifest.book_needs_reindex("book_a", 12345, 1, 1));

        // Index it
        manifest.mark_book_indexed("book_a".to_string(), 12345, 100, 1, 1);

        // Same content doesn't need reindex
        assert!(!manifest.book_needs_reindex("book_a", 12345, 1, 1));

        // Changed content needs reindex
        assert!(manifest.book_needs_reindex("book_a", 99999, 1, 1));

        // Changed chunking version needs reindex
        assert!(manifest.book_needs_reindex("book_a", 12345, 2, 1));

        // Remove book
        manifest.remove_book("book_a");
        assert!(manifest.book_needs_reindex("book_a", 12345, 1, 1));
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
    fn save_and_load_roundtrip() {
        let dir = TempDir::new("roundtrip");
        let config = test_config();
        let mut manifest = SemanticManifest::new(&config);
        manifest.mark_book_indexed("book_a".to_string(), 12345, 100, 1, 1);

        manifest.save(dir.path()).unwrap();
        let loaded = SemanticManifest::load(dir.path()).unwrap();

        assert_eq!(loaded.embedding_model_id, manifest.embedding_model_id);
        assert_eq!(loaded.embedding_dim, manifest.embedding_dim);
        assert_eq!(loaded.books.len(), 1);
        assert!(loaded.books.contains_key("book_a"));
    }
}
