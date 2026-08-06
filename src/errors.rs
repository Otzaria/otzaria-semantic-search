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

    /// The on-disk semantic index was built with a configuration that is
    /// incompatible with the current one (different model, dimensions,
    /// chunking, …). The semantic path stays disabled until
    /// `SemanticEngine::reset_index` is called and the books are re-indexed.
    #[error("Semantic index is incompatible with the current configuration: {details}")]
    IncompatibleIndex { details: String },

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

    /// The file exists but is not a usable GGUF container. Guards against a
    /// truncated download or a placeholder file being accepted as a model.
    #[error("Not a valid GGUF model file ({path}): {reason}")]
    InvalidModelFile { path: String, reason: String },

    /// No inference backend is compiled in. A default build has none by choice, so
    /// a production binary can never fall back to the hash-based stand-in embedder;
    /// real inference is opt-in through `--features llama-backend`, which compiles
    /// llama.cpp and ggml through cmake.
    #[error("No embedding backend is available in this build: {reason}")]
    BackendUnavailable { reason: String },

    #[error("Inference failed: {reason}")]
    InferenceFailed { reason: String },

    /// A configured pooling strategy this build does not implement.
    ///
    /// Refused rather than ignored: pooling decides what a vector *is* and is
    /// recorded in the manifest as part of the index's identity, so a typo that
    /// fell through used to produce vectors pooled one way while the manifest
    /// claimed another — with no error anywhere.
    #[error("Unknown pooling strategy '{found}' (supported: {supported})")]
    UnknownPooling { found: String, supported: String },

    /// A pooling strategy this crate has a name for but no backend performs.
    ///
    /// Distinct from [`Self::UnknownPooling`], which is a spelling nothing can
    /// even parse. This one parses, round-trips through the manifest, and used to
    /// be accepted: `pooling = "mean"` validated, the manifest was written with
    /// `"pooling": "mean"`, and only the later model load failed. Correcting the
    /// configuration afterwards then produced a *different* failure that outlived
    /// the typo — a pooling mismatch against the manifest the typo had written,
    /// pointing at the index instead of at the value that caused it, and
    /// recoverable only by discarding the index. Refusing it while it is still
    /// only a configuration keeps the diagnosis where the mistake is.
    #[error("No embedding backend implements pooling '{pooling}' (implemented: {implemented})")]
    PoolingNotImplemented {
        pooling: String,
        implemented: String,
    },

    /// The loaded backend pools differently from the configuration it was loaded
    /// for.
    ///
    /// A real backend reports what its model requires, not what it was asked for.
    /// Carrying on would store vectors under a pooling label that does not
    /// describe them, and the mislabelling only becomes visible as bad search
    /// results.
    #[error(
        "Pooling mismatch: configuration says '{configured}', backend '{backend}' pools '{actual}'"
    )]
    PoolingMismatch {
        backend: String,
        configured: String,
        actual: String,
    },

    /// The backend has no tokenizer to answer with.
    ///
    /// Distinct from a failed tokenization: the hash stand-in has no token ids at
    /// all, and inventing plausible ones would turn the parity check against a
    /// reference tokenizer — the `golden` tests, which assert exact token-id equality
    /// — into a comparison between two fabrications.
    #[error("Backend '{backend}' cannot tokenize: {reason}")]
    TokenizationUnsupported { backend: String, reason: String },

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: u32, actual: u32 },

    #[error("Model not loaded — call load_model() first")]
    NotLoaded,
}

/// Errors from the vector store, whichever backend is in use.
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

    /// The manifest was written by a different schema version. Not recoverable
    /// by parsing — the index has to be rebuilt.
    #[error("Unsupported manifest format version {found} (this build supports {supported})")]
    UnsupportedFormatVersion { found: u32, supported: u32 },

    #[error("Model mismatch: manifest has '{manifest_model}', config has '{config_model}'")]
    ModelMismatch {
        manifest_model: String,
        config_model: String,
    },

    #[error("Dimension mismatch: manifest has {manifest_dim}, config has {config_dim}")]
    DimensionMismatch { manifest_dim: u32, config_dim: u32 },

    #[error("Chunking version mismatch: manifest has {manifest_ver}, config has {config_ver}")]
    ChunkingVersionMismatch { manifest_ver: u32, config_ver: u32 },

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
