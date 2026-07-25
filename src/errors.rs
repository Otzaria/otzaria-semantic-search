//! Centralized error types for the semantic search subsystem.
//!
//! Design: errors are categorized by subsystem so callers can decide
//! whether to propagate, log, or gracefully degrade.

use thiserror::Error;

/// Top-level error for the semantic search subsystem.
#[derive(Error, Debug)]
pub enum SemanticSearchError {
    #[error("Embedding runtime error: {0}")]
    EmbeddingRuntime(#[from] EmbeddingError),

    #[error("Vector store error: {0}")]
    VectorStore(#[from] VectorStoreError),

    #[error("Manifest error: {0}")]
    Manifest(#[from] ManifestError),

    #[error("Chunking error: {0}")]
    Chunking(#[from] ChunkingError),

    #[error("Fusion error: {0}")]
    Fusion(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}

/// Errors from the embedding model runtime.
#[derive(Error, Debug)]
pub enum EmbeddingError {
    #[error("Model file not found: {path}")]
    ModelNotFound { path: String },

    #[error("Tokenizer file not found: {path}")]
    TokenizerNotFound { path: String },

    #[error("Model loading failed: {reason}")]
    LoadFailed { reason: String },

    #[error("Inference failed: {reason}")]
    InferenceFailed { reason: String },

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: u32, actual: u32 },

    #[error("Model not loaded — call load_model() first")]
    NotLoaded,
}

/// Errors from the vector store (zvec).
#[derive(Error, Debug)]
pub enum VectorStoreError {
    #[error("Store not initialized at path: {path}")]
    NotInitialized { path: String },

    #[error("Store open failed: {reason}")]
    OpenFailed { reason: String },

    #[error("Insert failed: {reason}")]
    InsertFailed { reason: String },

    #[error("Search failed: {reason}")]
    SearchFailed { reason: String },

    #[error("Delete failed: {reason}")]
    DeleteFailed { reason: String },

    #[error("Commit failed: {reason}")]
    CommitFailed { reason: String },

    #[error("Dimension mismatch: store has {store_dim}, vector has {vector_dim}")]
    DimensionMismatch { store_dim: u32, vector_dim: u32 },

    #[error("Store is corrupted: {reason}")]
    Corrupted { reason: String },
}

/// Errors from the manifest/versioning system.
#[derive(Error, Debug)]
pub enum ManifestError {
    #[error("Manifest file not found at: {path}")]
    NotFound { path: String },

    #[error("Manifest parse failed: {reason}")]
    ParseFailed { reason: String },

    #[error("Model mismatch: manifest has '{manifest_model}', config has '{config_model}'")]
    ModelMismatch {
        manifest_model: String,
        config_model: String,
    },

    #[error("Dimension mismatch: manifest has {manifest_dim}, config has {config_dim}")]
    DimensionMismatch {
        manifest_dim: u32,
        config_dim: u32,
    },

    #[error("Chunking version mismatch: manifest has {manifest_ver}, config has {config_ver}")]
    ChunkingVersionMismatch {
        manifest_ver: u32,
        config_ver: u32,
    },

    #[error("Write failed: {reason}")]
    WriteFailed { reason: String },
}

/// Errors from the chunking subsystem.
#[derive(Error, Debug)]
pub enum ChunkingError {
    #[error("Invalid section structure: {reason}")]
    InvalidStructure { reason: String },

    #[error("Chunk mapping failed for line {line_id}: {reason}")]
    MappingFailed { line_id: u64, reason: String },
}

/// Result type alias for semantic search operations.
pub type SemanticResult<T> = std::result::Result<T, SemanticSearchError>;
