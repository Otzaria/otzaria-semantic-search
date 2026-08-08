//! Centralized error types for the semantic search subsystem.
//!
//! Design: errors are categorized by subsystem so callers can decide
//! whether to propagate, log, or gracefully degrade.

use crate::semantic::versioning::{describe_identity_mismatches, IdentityField, IdentityMismatch};
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

    #[error("Semantic artifact error: {0}")]
    Artifact(#[from] ArtifactError),

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

    /// A build-side operation was asked of an installed official artifact.
    ///
    /// Not "no semantic index": there is one, it is open, and it is read-only. The two
    /// have to stay distinguishable — a caller that reads a refusal as "nothing
    /// configured" would go on to offer indexing as the fix, and indexing the library on
    /// the device is exactly what the product contract rules out.
    #[error(
        "'{operation}' is not available: the semantic index is an installed official \
         artifact, opened read-only. Producing one is a build-machine operation"
    )]
    ReadOnlyIndex { operation: &'static str },

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

/// Why an official artifact was refused.
///
/// Every variant is a refusal to proceed, never a degradation: a package that fails
/// any of these checks is the wrong package or a damaged one, and there is nothing the
/// device can repair. The distinctions are kept because the host application has to
/// tell the user which of them happened — `incompatible` is answered by fetching the
/// matching artifact, `corrupt` by downloading this one again.
#[derive(Error, Debug)]
pub enum ArtifactError {
    /// `manifest.json` or `payloads.json` is missing, unreadable or not the JSON it
    /// claims to be.
    #[error("Artifact metadata is unusable ({path}): {reason}")]
    MetadataUnusable { path: String, reason: String },

    /// The metadata is a document this build does not read. Refused rather than
    /// parsed leniently: filling in a field the writer never recorded would be a
    /// guess presented as agreement.
    #[error("Unsupported artifact metadata version {found} (this build reads {supported})")]
    UnsupportedMetadataVersion { found: u32, supported: u32 },

    /// An identity field exists but carries no value — a blank string, a zero
    /// version. Checked before any comparison, because two unfilled identities agree
    /// with each other.
    #[error("Artifact identity is incomplete: {field} {reason}")]
    IncompleteIdentity {
        field: IdentityField,
        reason: String,
    },

    /// The artifact describes a different corpus, model or store format than this
    /// installation. Lists every disagreement, not the first.
    #[error(
        "Artifact does not match this installation: {}",
        describe_identity_mismatches(mismatches)
    )]
    IdentityMismatch { mismatches: Vec<IdentityMismatch> },

    /// The artifact's metadata digest is not the one that was published for it.
    ///
    /// This is the only check that distinguishes *the official artifact* from a
    /// self-consistent impostor: `payloads.json` travels inside the package, so a
    /// payload replaced together with its checksum passes every other check here.
    #[error("Artifact digest is {actual}, but {expected} was published for it")]
    UnexpectedArtifactDigest { expected: String, actual: String },

    /// A package with no payload. Not an empty index — an incomplete package.
    #[error("Artifact has no checksummed payload files")]
    NoPayload,

    /// A payload name that is not a portable single file name, or one that would
    /// overwrite the metadata. What blocks `../` escaping the package directory — and
    /// what keeps a package written on one platform readable on another.
    #[error("Unsafe artifact payload name {name:?}: {reason}")]
    UnsafePayloadName { name: String, reason: String },

    #[error("Artifact payload {payload:?} has no valid SHA-256 in payloads.json")]
    MalformedPayloadChecksum { payload: String },

    #[error("Artifact payload {payload:?} is missing")]
    PayloadMissing { payload: String },

    /// A symlink or a directory where a payload should be. Refused rather than
    /// followed: the checksum would then describe a file outside the package.
    #[error("Artifact payload {payload:?} is not a regular file")]
    PayloadNotRegularFile { payload: String },

    #[error(
        "Artifact payload {payload:?} failed its checksum (expected {expected}, got {actual})"
    )]
    PayloadChecksumFailed {
        payload: String,
        expected: String,
        actual: String,
    },

    /// The manifest and the payload describe different packages — declared sizes or
    /// counts that the files do not support. A manifest is a claim about the payload,
    /// so it has to be checked against it and not only for being present.
    #[error("Artifact manifest disagrees with its payload: {reason}")]
    ManifestDisagreesWithPayload { reason: String },

    /// The install target cannot be used — no parent directory, an existing
    /// non-directory, or a path inside the package itself.
    #[error("Invalid install target: {reason}")]
    InvalidInstallTarget { reason: String },

    /// A crash interrupted an install and the leftovers could not be resolved.
    ///
    /// Distinct from [`Self::Io`] because the caller's next step is different: the
    /// previous artifact may be sitting under a recovery name, and overwriting it
    /// would destroy the only good copy on the device.
    #[error("An interrupted install could not be recovered: {reason}")]
    InterruptedInstall { reason: String },

    #[error("Artifact IO error ({context}): {source}")]
    Io {
        context: String,
        #[source]
        source: std::io::Error,
    },
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
