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

    /// `#[source]` and not `#[from]`: the conversion is written out below this enum, so
    /// that a cancelled scan becomes [`Self::Cancelled`] instead of this.
    #[error("Vector store error: {0}")]
    VectorStore(#[source] VectorStoreError),

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

    /// The caller cancelled the search through its
    /// [`CancellationToken`](crate::cancellation::CancellationToken) before it finished.
    ///
    /// Not a failure, and not to be shown as one: the caller asked for it, because a newer
    /// query superseded this one. The search logged nothing and left nothing behind — no
    /// cached result, no cached embedding, no telemetry — so dropping it is all a caller
    /// has to do. See [`crate::cancellation`].
    #[error("The search was cancelled before it finished")]
    Cancelled,

    /// A ranking parameter passed with a search is one the ranking is not defined for: not
    /// a number, negative, or outside its range. Refused before the search runs, rather
    /// than clamped into something nobody asked for — see
    /// [`RankingProfile::validate`](crate::config::profiles::RankingProfile::validate).
    ///
    /// `parameter` is the field's path in the profile, `alpha_by_query_type.short` say.
    #[error("Ranking parameter {parameter} is {value}, and it must be {requirement}")]
    InvalidRankingParameter {
        parameter: &'static str,
        value: String,
        requirement: &'static str,
    },

    /// The application's resolver could not tie the vectors a scan returned to live lines
    /// — its index could not be read, say. The semantic side of that one search fails, and
    /// the search degrades to its lexical results as it does for any other semantic failure.
    ///
    /// Distinct from a hit that resolves nowhere, which is not an error: a vector whose
    /// text no live line holds any more is skipped, and counted.
    #[error("The semantic results could not be resolved to live lines: {reason}")]
    Resolution { reason: String },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}

/// By hand rather than `#[from]`, because of one variant: a scan abandoned through its
/// token is [`SemanticSearchError::Cancelled`] whichever layer noticed it. Wrapped as a
/// store error it would read as a fault of the store, and a caller matching on
/// `Cancelled` — the one outcome it must tell apart from a failure — would miss it.
impl From<VectorStoreError> for SemanticSearchError {
    fn from(error: VectorStoreError) -> Self {
        match error {
            VectorStoreError::Cancelled => Self::Cancelled,
            other => Self::VectorStore(other),
        }
    }
}

/// A cancelled resolution is a cancelled search, as a cancelled scan is; any other failure
/// of the index is [`SemanticSearchError::Resolution`].
impl From<crate::semantic::resolve::ResolveError> for SemanticSearchError {
    fn from(error: crate::semantic::resolve::ResolveError) -> Self {
        match error {
            crate::semantic::resolve::ResolveError::Cancelled => Self::Cancelled,
            crate::semantic::resolve::ResolveError::Index { reason } => Self::Resolution { reason },
        }
    }
}

/// Errors from the embedding model runtime.
#[derive(Error, Debug)]
pub enum EmbeddingError {
    #[error("Model file not found: {path}")]
    ModelNotFound { path: String },

    /// An ONNX package without its tokenizer. The tokenizer is part of the package —
    /// it decides which token ids the graph sees, and the package checksum covers it —
    /// so it is looked for in one place only, and never substituted.
    #[error(
        "Tokenizer file not found: {path} (an ONNX model is a package, and its \
         tokenizer.json must sit beside the graph file)"
    )]
    TokenizerNotFound { path: String },

    #[error("Model loading failed: {reason}")]
    LoadFailed { reason: String },

    /// Not a usable ONNX model: a path that names no ONNX graph at all — one that does
    /// not end in `.onnx`, such as a GGUF file, whose support was removed — whether or not
    /// a file is there, or an ONNX package that is not whole, for which `path` is always
    /// the graph and the reason names whichever file of the package was at fault. Guards
    /// against a truncated download or a placeholder file being accepted as a model.
    #[error("Not a valid ONNX model file ({path}): {reason}")]
    InvalidModelFile { path: String, reason: String },

    /// No inference backend is compiled in. A default build has none by choice, so a
    /// production binary can never fall back to the hash-based stand-in embedder; real
    /// inference is opt-in through `--features onnx-backend`.
    #[error("No embedding backend is available in this build: {reason}")]
    BackendUnavailable { reason: String },

    /// The ONNX backend is compiled in, but the ONNX Runtime shared library it loads when
    /// a model does could not be loaded: none found, not a runtime, too old, refused, or a
    /// different one already running in this process.
    ///
    /// Not [`Self::BackendUnavailable`], whose message says the *build* lacks a backend:
    /// the fix here is a file put in place, a path or an environment variable, never a
    /// rebuild, and a message saying otherwise sends the reader to the wrong one. The
    /// reason says where the backend looked, in order — the path the application passes
    /// ([`EmbeddingDeployment::onnx_runtime`](crate::semantic::embedding::EmbeddingDeployment::onnx_runtime)),
    /// `OTZARIA_ONNX_RUNTIME`, the platform's file name beside the graph — and which of
    /// them a refused library came from.
    #[error("ONNX Runtime could not be loaded: {reason}")]
    OnnxRuntimeUnavailable { reason: String },

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

    /// The scan stopped at a checkpoint because its token was cancelled. Not a fault of
    /// the store: it reaches the caller as [`SemanticSearchError::Cancelled`].
    #[error("Search cancelled before the scan finished")]
    Cancelled,
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

    /// An identity field names a recipe version this build does not implement.
    ///
    /// Distinct from [`Self::IncompleteIdentity`], which catches a field nobody filled in.
    /// This one is filled in, plausible, and describes behaviour that exists nowhere: the
    /// three recipe versions are versions of *this crate's code*, so a number with no
    /// implementation behind it means the vectors were built by something else or the
    /// identity was written by hand. Being lenient would let an artifact declare a recipe
    /// and an installation agree to it, with neither running it.
    #[error(
        "{field} is {found}, and this build implements {supported}: that recipe exists \
         nowhere in this code"
    )]
    UnsupportedRecipeVersion {
        field: &'static str,
        found: u32,
        supported: String,
    },

    /// The chunker configuration in hand and the identity an artifact declares disagree
    /// about a value they both carry.
    ///
    /// One fact in two places is a fact that drifts. `embedding_text_version` has to be in
    /// the configuration, because that is what selects the code path, and in the identity,
    /// because an installation compares identities and never sees a configuration.
    #[error(
        "{field} is {configured} in the chunker configuration and {declared} in the model \
         identity"
    )]
    RecipeDisagreesWithIdentity {
        field: &'static str,
        configured: u32,
        declared: u32,
    },

    /// The artifact describes a different line recipe, model or store format than this
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

    /// A delta that is not the next step for the vector set it was offered to.
    ///
    /// `field` names what disagreed: `delta.from_library_version` for a delta that starts
    /// past the set's library version, or overlaps it — a gap, which applying would paper
    /// over with vectors the set never had — and `delta.codec_params` for one quantized in
    /// another codec epoch, whose bytes mean something else. A delta the set has already
    /// absorbed is not this error; it is reported as already applied.
    #[error("The delta does not apply to this vector set: {field} — {reason}")]
    DeltaDoesNotApply { field: &'static str, reason: String },

    /// A release under the id of a segment the set holds with other bytes, which a generation
    /// that opens still stands on: a version that is installed — the same identity, kind,
    /// library versions and keys — published again, embedded or anchored anew. A sound
    /// release, and a sound set: the set keeps the bytes it serves, and the release does not
    /// replace them (`docs/ARTIFACT_CONTRACT.md` §2.4 says what a host can do).
    #[error(
        "The release's segment {id} is a version this set has installed, published again with \
         other bytes: it holds SHA-256 {installed_sha256} and serves it, and this release is \
         {offered_sha256}"
    )]
    SegmentIdTaken {
        id: String,
        installed_sha256: String,
        offered_sha256: String,
    },

    /// The device lacks the free space an install or a compaction needs.
    ///
    /// `available` is what the filesystem reported, or — where it could not be asked — what
    /// the operation managed to write before the device filled up. Nothing was installed or
    /// replaced, and the partial output was removed.
    #[error(
        "Not enough free space: the operation needs {needed} byte(s) and {available} are \
         available"
    )]
    InsufficientSpace { needed: u64, available: u64 },

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

/// Why a build machine could not produce — or could not vouch for — an artifact.
///
/// Deliberately **not** a variant of [`SemanticSearchError`]: packing never runs on a
/// user's device, and an error the application cannot encounter has no business in the
/// enum the application matches on.
///
/// The distinctions here are the ones a build log has to make. "The corpus could not be
/// read" is a broken build input; "this line has a vector and no document" is a broken
/// pairing between the vectors and the catalogue they claim to describe — and the second
/// one is the fault that produces confident, wrong search results if it ships.
#[derive(Error, Debug)]
pub enum PackError {
    /// The corpus source itself failed. Distinct from [`Self::LineNotInCorpus`]: that is
    /// a fault in the vectors, this is a fault in what they are being checked against.
    #[error("The corpus could not be read: {reason}")]
    Corpus { reason: String },

    /// The output path is not somewhere a whole package can be written: it is not a
    /// directory, or it is one that already holds files.
    ///
    /// A non-empty directory is refused rather than merged into: what is already there is
    /// evidence of an earlier build, and a package written over it would describe files
    /// this run never wrote.
    #[error("Cannot build into {path}: {reason}")]
    UnusableOutput { path: String, reason: String },

    #[error("The vector input is malformed: {reason}")]
    MalformedInput { reason: String },

    /// Two books, or two entries of one book, claim the same line: built, it would be
    /// recorded twice, at two positions.
    #[error("line_id {line_id} appears more than once in the input")]
    DuplicateLineId { line_id: u64 },

    /// A book lists a line the corpus holds no document for.
    #[error("line_id {line_id} is listed under a book and the corpus holds no such line")]
    LineNotInCorpus { line_id: u64 },

    /// The vectors and the plan do not describe the same set of lines.
    ///
    /// Checked in **both** directions, because they are different faults and neither is
    /// visible any other way:
    ///
    /// * *missing* — lines the recipe embeds that got no vector. A library missing most of
    ///   itself still produces a package whose counts, checksums and identity all agree,
    ///   so one good vector out of six million would otherwise build successfully.
    /// * *unexpected* — vectors for lines the recipe does **not** embed. Distinct from
    ///   [`Self::LineNotInCorpus`], which is a line the corpus has never heard of: this one
    ///   exists and is answerable, it simply should not have been embedded. A line too
    ///   short to carry meaning acquiring a vector means the package was built by a recipe
    ///   other than the one it declares.
    #[error(
        "The vectors cover {covered} line(s) and the plan expects {expected}: \
         {missing} have no vector{}, and {unexpected} vector(s) name a line the recipe \
         does not embed{}",
        describe_first(*first_missing),
        describe_first(*first_unexpected)
    )]
    CoverageMismatch {
        expected: usize,
        covered: usize,
        missing: usize,
        unexpected: usize,
        /// Smallest missing id, so two runs over the same fault name the same line.
        first_missing: Option<u64>,
        /// Smallest unexpected id, for the same reason.
        first_unexpected: Option<u64>,
    },

    #[error("There is nothing to pack: the input holds no vectors")]
    NoVectors,

    /// A plan record's text is not the text its digest names.
    ///
    /// The digest travels beside the text precisely so that the machine holding no
    /// corpus can still tell that what reached it is what was exported. Without the
    /// comparison a plan file damaged in transit — truncated mid-record, re-encoded,
    /// edited — would be embedded happily, and the resulting vector would describe a
    /// passage the library does not contain while carrying a `chunk_hash` that says
    /// otherwise.
    #[error(
        "The plan's text for record {record} hashes to {actual}, and the plan declares \
         {declared}: what reached this worker is not what was exported"
    )]
    PlanTextChanged {
        /// The record's position in the plan.
        record: u64,
        declared: String,
        actual: String,
    },

    /// The chunker configuration a build was handed is not the one the artifact declares.
    ///
    /// `chunking_identity` is a one-way hash of the whole configuration, so nothing can
    /// recover the recipe from an artifact — a builder has to be *given* the recipe, and
    /// this is the only thing that can establish it was given the right one. Without the
    /// check, `expected_line_ids` would apply one recipe while the artifact declared
    /// another, and coverage would certify a set nobody built.
    #[error(
        "The chunker configuration hashes to {actual} and the model declares \
         chunking_identity {declared}: the vectors would be built by a recipe other than \
         the one the artifact announces"
    )]
    RecipeMismatch { declared: u64, actual: u64 },

    /// The model identity a build declares disagrees with the model file it loaded.
    ///
    /// This is what turns the declaration into a checked claim. A file of finished floats
    /// can be labelled with any model; a build that loads the model itself cannot, because
    /// the checksum, the backend, the width, the pooling and the effective token cap are
    /// all readable from the thing that is about to produce the vectors.
    #[error(
        "The declared model identity does not match the model file: {field} is declared \
         as {declared:?}, and the loaded model reports {loaded:?}"
    )]
    ModelDisagreesWithFile {
        field: &'static str,
        declared: String,
        loaded: String,
    },

    /// The loaded backend does not produce semantic vectors.
    ///
    /// Nothing downstream can tell: hash vectors have a plausible norm, a plausible
    /// dimension and plausible neighbours, and an artifact built from them passes every
    /// structural check in this crate. The identity records which backend it was, so a
    /// mismatched runtime would refuse it — but a runtime built with the same stand-in
    /// would open it and answer nonsense with full confidence.
    #[error(
        "Backend '{backend}' reports that its vectors are not semantic; an artifact built \
         from them would look entirely normal and answer nothing"
    )]
    NonSemanticBackend { backend: String },

    /// The recipe embeds no line of this corpus. Refused before the model is asked for a
    /// single vector, because the alternative is an empty artifact that verifies.
    #[error("The recipe embeds no line of the {books} book(s) in this corpus")]
    NothingToEmbed { books: usize },

    #[error("Embedding error: {0}")]
    Embedding(#[from] EmbeddingError),

    #[error("Artifact error: {0}")]
    Artifact(#[from] ArtifactError),

    #[error("Vector store error: {0}")]
    VectorStore(#[from] VectorStoreError),

    #[error("Pack IO error ({context}): {source}")]
    Io {
        context: String,
        #[source]
        source: std::io::Error,
    },
}

/// Name an example line in a coverage rejection, or say nothing when that side is clean.
fn describe_first(line_id: Option<u64>) -> String {
    line_id.map_or(String::new(), |line_id| format!(" (first: line {line_id})"))
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
