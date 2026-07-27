//! Embedding model runtime.
//!
//! # Backend status
//!
//! Real GGUF inference is **not implemented yet** (roadmap P2). This module
//! provides the runtime shell around it: model-file validation, checksumming,
//! batch API, pooling/normalization contract and backend selection.
//!
//! Two build configurations exist, and the difference is deliberate:
//!
//! | build | backend | behaviour of [`EmbeddingRuntime::load`] |
//! |---|---|---|
//! | default (production) | none | `Err(EmbeddingError::BackendUnavailable)` |
//! | `--features mock-embedding` (and in-crate tests) | [`EmbeddingBackendKind::MockHash`] | `Ok` |
//!
//! The mock backend is a deterministic hash of the input text. It is **not a
//! semantic model**: similarity between two of its vectors carries no meaning
//! beyond token overlap. Gating it behind a non-default feature is what keeps a
//! release build from silently serving fake vectors — a production binary fails
//! loudly instead.

use crate::errors::EmbeddingError;
use std::io::Read;
use std::path::{Path, PathBuf};

/// GGUF container magic, little-endian `b"GGUF"`.
const GGUF_MAGIC: [u8; 4] = *b"GGUF";

/// GGUF container versions this crate is willing to open.
///
/// v1 is excluded deliberately: it stored `tensor_count` and
/// `metadata_kv_count` as `u32`, and the widening to `u64` came in v2 (see the
/// GGUF spec's version history). Reading a v1 header with the v2 layout would
/// misparse it, so claiming v1 support while parsing v2 fields would be worse
/// than refusing it. No v1 model is in circulation for this crate's purpose;
/// supporting it, if ever needed, means a real per-version parser in the
/// backend (roadmap P2).
const GGUF_SUPPORTED_VERSIONS: std::ops::RangeInclusive<u32> = 2..=3;

/// Size of the GGUF header: magic (4) + version (4) + `tensor_count` (8) +
/// `metadata_kv_count` (8).
const GGUF_HEADER_BYTES: usize = 24;

/// Sanity ceiling for the header's declared counts.
///
/// A real embedding model has hundreds of tensors and tens of metadata entries.
/// A wildly larger count means the bytes are not a GGUF header — a truncated
/// download, or another format that happens to start with the right four bytes.
const GGUF_MAX_DECLARED_COUNT: u64 = 1 << 24;

/// Smallest number of bytes a metadata key/value pair can occupy: key length
/// (8) + at least one key byte + value type (4) + at least one value byte.
const GGUF_MIN_METADATA_ENTRY_BYTES: u64 = 14;

/// Smallest number of bytes a tensor descriptor can occupy: name length (8) +
/// at least one name byte + dimension count (4) + one dimension (8) + type (4)
/// + offset (8).
const GGUF_MIN_TENSOR_INFO_BYTES: u64 = 33;

/// Read buffer for hashing the model file. Large enough that hashing a
/// multi-hundred-megabyte model is bound by the hash, not by syscalls.
const HASH_BUFFER_BYTES: usize = 1 << 20;

/// Below this L2 norm a vector carries no direction and cannot be normalized.
const MIN_VECTOR_NORM: f32 = 1e-12;

/// Configuration for embedding runtime loading.
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    pub model_path: PathBuf,
    pub embedding_dim: u32,
    pub max_tokens: usize,
    /// Number of texts handed to the backend per inference call.
    pub batch_size: usize,
    pub pooling: String,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("models/otzaria-embedding-v1-flash-q4.gguf"),
            embedding_dim: 1024,
            max_tokens: 512,
            batch_size: 32,
            pooling: "last-token".to_string(),
        }
    }
}

/// Identifies which embedding implementation a runtime has loaded.
///
/// Recorded in the semantic manifest so an index built by one backend is never
/// silently queried through another — vectors from different backends live in
/// different spaces and are not comparable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingBackendKind {
    /// Deterministic hash-based stand-in. Test/development only; requires the
    /// `mock-embedding` feature. See the module docs.
    MockHash,
    // A real GGUF backend is added in roadmap P2.
}

impl EmbeddingBackendKind {
    /// Stable identifier persisted in the manifest.
    pub fn id(&self) -> &'static str {
        match self {
            Self::MockHash => "mock-hash-v1",
        }
    }

    /// Whether vectors produced by this backend carry semantic meaning.
    pub fn is_semantic(&self) -> bool {
        match self {
            Self::MockHash => false,
        }
    }
}

/// Local embedding runtime.
pub struct EmbeddingRuntime {
    config: EmbeddingConfig,
    backend: Option<EmbeddingBackendKind>,
    /// SHA-256 of the model file, computed by [`Self::load`].
    model_checksum: Option<String>,
}

impl EmbeddingRuntime {
    /// Initialize runtime with configuration. No file access happens here.
    pub fn new(config: EmbeddingConfig) -> Self {
        Self {
            config,
            backend: None,
            model_checksum: None,
        }
    }

    /// Load the model from disk.
    ///
    /// Validates the GGUF container and computes the file's SHA-256 in a single
    /// pass, then selects a backend. Fails with
    /// [`EmbeddingError::BackendUnavailable`] in builds that have no inference
    /// backend compiled in.
    pub fn load(&mut self) -> Result<(), EmbeddingError> {
        if !self.config.model_path.exists() {
            return Err(EmbeddingError::ModelNotFound {
                path: self.config.model_path.display().to_string(),
            });
        }

        let checksum = validate_and_checksum_gguf(&self.config.model_path)?;

        #[cfg(any(test, feature = "mock-embedding"))]
        {
            log::warn!(
                "Embedding backend = MOCK (deterministic hashing). \
                 Vectors are NOT semantic; this build is not fit for production. \
                 Model file: {}",
                self.config.model_path.display()
            );
            self.model_checksum = Some(checksum);
            self.backend = Some(EmbeddingBackendKind::MockHash);
            Ok(())
        }

        #[cfg(not(any(test, feature = "mock-embedding")))]
        {
            let _ = checksum;
            Err(EmbeddingError::BackendUnavailable {
                reason: format!(
                    "real GGUF inference is not implemented yet (roadmap P2); \
                     model file {} validated but cannot be executed",
                    self.config.model_path.display()
                ),
            })
        }
    }

    /// Check if a backend is currently loaded.
    pub fn is_loaded(&self) -> bool {
        self.backend.is_some()
    }

    /// The loaded backend, or `None` before a successful [`Self::load`].
    pub fn backend(&self) -> Option<EmbeddingBackendKind> {
        self.backend
    }

    /// SHA-256 of the loaded model file, or `None` before a successful load.
    pub fn model_checksum(&self) -> Option<&str> {
        self.model_checksum.as_deref()
    }

    /// Embed a single text into an L2-normalized vector.
    ///
    /// Convenience wrapper over [`Self::embed_batch`]; indexing should call the
    /// batch form directly so the backend sees whole batches.
    pub fn embed_one(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let mut out = self.embed_batch(&[text])?;
        out.pop().ok_or_else(|| EmbeddingError::InferenceFailed {
            reason: "backend returned no vector for a single input".to_string(),
        })
    }

    /// Embed a batch of texts into L2-normalized vectors, one per input, in
    /// input order.
    ///
    /// Inputs are split into chunks of [`EmbeddingConfig::batch_size`] before
    /// reaching the backend. Every returned vector is verified to have the
    /// configured dimensionality and a non-zero norm, so a degenerate vector
    /// can never silently enter the index.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let Some(backend) = self.backend else {
            return Err(EmbeddingError::NotLoaded);
        };

        let dim = self.config.embedding_dim;
        let mut results = Vec::with_capacity(texts.len());

        for group in texts.chunks(self.batch_size()) {
            let raw = self.infer(backend, group)?;

            if raw.len() != group.len() {
                return Err(EmbeddingError::InferenceFailed {
                    reason: format!(
                        "backend returned {} vectors for {} inputs",
                        raw.len(),
                        group.len()
                    ),
                });
            }

            for mut vector in raw {
                normalize_validated(&mut vector, dim)?;
                results.push(vector);
            }
        }

        Ok(results)
    }

    /// Dispatch one already-sized batch to the loaded backend.
    ///
    /// Returns raw, unnormalized vectors; [`Self::embed_batch`] owns validation
    /// and normalization so every backend gets the same treatment.
    fn infer(
        &self,
        backend: EmbeddingBackendKind,
        group: &[&str],
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        match backend {
            EmbeddingBackendKind::MockHash => {
                #[cfg(any(test, feature = "mock-embedding"))]
                {
                    Ok(group
                        .iter()
                        .map(|t| mock::hash_embedding(t, self.config.embedding_dim))
                        .collect())
                }
                #[cfg(not(any(test, feature = "mock-embedding")))]
                {
                    let _ = group;
                    Err(EmbeddingError::BackendUnavailable {
                        reason: "mock backend is not compiled into this build".to_string(),
                    })
                }
            }
        }
    }

    /// Expected embedding dimensionality.
    pub fn dim(&self) -> u32 {
        self.config.embedding_dim
    }

    /// Pooling strategy this runtime is configured for.
    pub fn pooling(&self) -> &str {
        &self.config.pooling
    }

    /// Maximum number of texts sent to the backend per inference call.
    pub fn batch_size(&self) -> usize {
        self.config.batch_size.max(1)
    }
}

/// L2-normalize a vector in place after checking it can be compared at all.
///
/// Rejects anything that would enter the index as an unsearchable point:
///
/// * **wrong dimensionality** — it could not be stored;
/// * **a non-finite component** — `NaN` poisons its own norm, and
///   `NaN < MIN_VECTOR_NORM` is `false`, so a norm test alone waves it through;
/// * **a non-finite norm** — which also catches a *finite* vector whose squares
///   overflow `f32` (components around `1e30`): the norm becomes `inf`, the
///   reciprocal `0`, and the vector silently normalizes to all zeros;
/// * **no direction** — a zero vector matches nothing.
///
/// Getting this wrong is quiet rather than loud: the book is recorded as indexed,
/// then every one of its vectors scores `NaN` and is discarded during search. The
/// result is an index that looks complete and returns nothing for that book.
pub fn normalize_validated(vector: &mut [f32], expected_dim: u32) -> Result<(), EmbeddingError> {
    if vector.len() as u32 != expected_dim {
        return Err(EmbeddingError::DimensionMismatch {
            expected: expected_dim,
            actual: vector.len() as u32,
        });
    }

    if let Some(position) = vector.iter().position(|x| !x.is_finite()) {
        return Err(EmbeddingError::InferenceFailed {
            reason: format!(
                "vector component {position} is not finite ({}); such a vector \
                 can never be matched",
                vector[position]
            ),
        });
    }

    let norm = l2_normalize(vector);
    if !norm.is_finite() {
        return Err(EmbeddingError::InferenceFailed {
            reason: format!("vector norm is not finite ({norm}); the magnitudes overflowed f32"),
        });
    }
    if norm < MIN_VECTOR_NORM {
        return Err(EmbeddingError::InferenceFailed {
            reason: "vector has no direction (zero norm) — empty or unrepresentable input"
                .to_string(),
        });
    }

    // Normalization only shrinks magnitudes (|x| <= norm), so this cannot fire
    // once the input and the norm are finite. Cheap enough to keep as a backstop
    // against a future backend normalizing on its own.
    debug_assert!(vector.iter().all(|x| x.is_finite()));
    Ok(())
}

/// L2-normalize in place and return the norm the vector had beforehand.
///
/// A norm indistinguishable from zero leaves the vector untouched — the caller
/// must treat the returned norm as a failure signal rather than dividing by it.
/// Prefer [`normalize_validated`], which also rejects vectors that cannot be
/// compared.
pub fn l2_normalize(vec: &mut [f32]) -> f32 {
    let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > MIN_VECTOR_NORM {
        let inv = 1.0 / norm;
        for val in vec.iter_mut() {
            *val *= inv;
        }
    }
    norm
}

/// Validate a GGUF container and return the file's SHA-256, reading the file
/// exactly once.
///
/// Checks the complete header — magic, version, and both declared counts, which
/// have to be present and plausible. Full metadata and tensor parsing belongs to
/// the real backend (roadmap P2); what this buys is that a placeholder, a
/// truncated download, or another format that happens to begin with `GGUF` is
/// rejected rather than accepted as a model.
///
/// One pass over the file: a multi-hundred-megabyte model is read once, not once
/// to validate and again to hash.
pub fn validate_and_checksum_gguf(path: &Path) -> Result<String, EmbeddingError> {
    use sha2::{Digest, Sha256};

    let invalid = |reason: String| EmbeddingError::InvalidModelFile {
        path: path.display().to_string(),
        reason,
    };

    let mut file = std::fs::File::open(path).map_err(|e| EmbeddingError::LoadFailed {
        reason: format!("cannot open model file {}: {e}", path.display()),
    })?;

    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; HASH_BUFFER_BYTES];
    let mut header = Vec::with_capacity(GGUF_HEADER_BYTES);
    let mut total_bytes = 0u64;

    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|e| EmbeddingError::LoadFailed {
                reason: format!("cannot read model file {}: {e}", path.display()),
            })?;
        if read == 0 {
            break;
        }
        if header.len() < GGUF_HEADER_BYTES {
            let wanted = (GGUF_HEADER_BYTES - header.len()).min(read);
            header.extend_from_slice(&buffer[..wanted]);
        }
        hasher.update(&buffer[..read]);
        total_bytes += read as u64;
    }

    if header.len() < GGUF_HEADER_BYTES {
        return Err(invalid(format!(
            "file is {total_bytes} bytes; a GGUF header needs {GGUF_HEADER_BYTES}"
        )));
    }
    if header[..4] != GGUF_MAGIC {
        return Err(invalid(format!(
            "expected magic {:?}, found {:?}",
            String::from_utf8_lossy(&GGUF_MAGIC),
            String::from_utf8_lossy(&header[..4])
        )));
    }

    let field = |offset: usize| -> u64 {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&header[offset..offset + 8]);
        u64::from_le_bytes(bytes)
    };

    let version = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
    if !GGUF_SUPPORTED_VERSIONS.contains(&version) {
        return Err(invalid(format!(
            "unsupported GGUF version {version} (supported: {}..={})",
            GGUF_SUPPORTED_VERSIONS.start(),
            GGUF_SUPPORTED_VERSIONS.end()
        )));
    }

    let tensor_count = field(8);
    let metadata_kv_count = field(16);
    if tensor_count > GGUF_MAX_DECLARED_COUNT || metadata_kv_count > GGUF_MAX_DECLARED_COUNT {
        return Err(invalid(format!(
            "implausible header counts (tensors: {tensor_count}, metadata: \
             {metadata_kv_count}); the file is probably not GGUF or is corrupt"
        )));
    }

    // A download cut short keeps its header intact, so the header alone cannot
    // detect truncation. What can: the header *declares* how many tensors and
    // metadata entries follow, and each of those has a minimum size. A file
    // smaller than that floor is provably incomplete.
    //
    // This is a lower bound, not a parse — the real weights dwarf it. It catches
    // the interrupted-download case without opening the format, which is the
    // backend's job (roadmap P2).
    let minimum_bytes = GGUF_HEADER_BYTES as u64
        + metadata_kv_count.saturating_mul(GGUF_MIN_METADATA_ENTRY_BYTES)
        + tensor_count.saturating_mul(GGUF_MIN_TENSOR_INFO_BYTES);
    if total_bytes < minimum_bytes {
        return Err(invalid(format!(
            "file is {total_bytes} bytes but its header declares {tensor_count} tensor(s) \
             and {metadata_kv_count} metadata entry/entries, which need at least \
             {minimum_bytes} bytes — the download is incomplete"
        )));
    }

    let digest = hasher.finalize();
    Ok(hex_encode(&digest))
}

/// Lower-case hex encoding.
fn hex_encode(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(out, "{b:02x}");
    }
    out
}

/// Deterministic stand-in embedder. See the module docs — this is not a model.
#[cfg(any(test, feature = "mock-embedding"))]
pub mod mock {
    use sha2::{Digest, Sha256};

    /// Feature-hash `text` into a `dim`-sized vector.
    ///
    /// Deterministic across runs and platforms, which is all the rest of the
    /// pipeline needs from it. Whitespace-only input yields the zero vector;
    /// [`super::EmbeddingRuntime::embed_batch`] rejects that rather than
    /// storing a directionless vector.
    pub fn hash_embedding(text: &str, dim: u32) -> Vec<f32> {
        let dim = dim.max(1) as usize;
        let mut vec = vec![0.0f32; dim];

        for (idx, word) in text.split_whitespace().enumerate() {
            let hash = Sha256::digest(word.as_bytes());

            let bucket1 = (hash[0] as usize | ((hash[1] as usize) << 8)) % dim;
            let bucket2 = (hash[2] as usize | ((hash[3] as usize) << 8)) % dim;

            let val1 = if hash[4] % 2 == 0 { 1.0f32 } else { -1.0f32 };
            let val2 = if hash[5] % 2 == 0 { 0.5f32 } else { -0.5f32 };

            let decay = (idx + 1) as f32;
            vec[bucket1] += val1 / decay;
            vec[bucket2] += val2 / decay;
        }

        vec
    }

    /// Write a minimal well-formed GGUF header (magic + version + empty tensor
    /// and metadata counts) so tests exercise real container validation instead
    /// of a placeholder file.
    pub fn write_stub_gguf(path: &std::path::Path, version: u32) -> std::io::Result<()> {
        let mut bytes = Vec::with_capacity(super::GGUF_HEADER_BYTES);
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&version.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        bytes.extend_from_slice(&0u64.to_le_bytes()); // metadata_kv_count
        debug_assert_eq!(bytes.len(), super::GGUF_HEADER_BYTES);
        std::fs::write(path, bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempDir(PathBuf);
    impl TempDir {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "otzaria_embed_test_{name}_{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            let _ = std::fs::create_dir_all(&path);
            Self(path)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn loaded_runtime(dir: &TempDir, dim: u32) -> EmbeddingRuntime {
        let model = dir.path().join("model.gguf");
        mock::write_stub_gguf(&model, 3).unwrap();
        let mut rt = EmbeddingRuntime::new(EmbeddingConfig {
            model_path: model,
            embedding_dim: dim,
            batch_size: 4,
            ..Default::default()
        });
        rt.load().unwrap();
        rt
    }

    #[test]
    fn test_embedding_normalization() {
        let mut vec = vec![3.0, 4.0];
        let norm = l2_normalize(&mut vec);
        assert!((norm - 5.0).abs() < 1e-5);
        assert!((vec[0] - 0.6).abs() < 1e-5);
        assert!((vec[1] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn zero_vector_normalization_reports_zero_norm_and_does_not_divide() {
        let mut vec = vec![0.0, 0.0];
        assert_eq!(l2_normalize(&mut vec), 0.0);
        assert_eq!(vec, vec![0.0, 0.0]);
    }

    #[test]
    fn load_rejects_missing_model() {
        let dir = TempDir::new("missing");
        let mut rt = EmbeddingRuntime::new(EmbeddingConfig {
            model_path: dir.path().join("nope.gguf"),
            ..Default::default()
        });
        assert!(matches!(
            rt.load(),
            Err(EmbeddingError::ModelNotFound { .. })
        ));
        assert!(!rt.is_loaded());
    }

    #[test]
    fn load_rejects_placeholder_that_is_not_gguf() {
        let dir = TempDir::new("placeholder");
        let model = dir.path().join("fake.gguf");
        // The exact placeholder the old integration test relied on.
        std::fs::write(&model, b"GGUF_MOCK").unwrap();

        let mut rt = EmbeddingRuntime::new(EmbeddingConfig {
            model_path: model,
            ..Default::default()
        });
        // b"GGUF_MOCK" has the right magic but version bytes "_MOC".
        assert!(matches!(
            rt.load(),
            Err(EmbeddingError::InvalidModelFile { .. })
        ));
        assert!(!rt.is_loaded());
    }

    #[test]
    fn load_rejects_wrong_magic_and_truncated_files() {
        let dir = TempDir::new("magic");

        let wrong = dir.path().join("wrong.gguf");
        std::fs::write(&wrong, b"NOTGGUF_and_more_bytes_to_fill_the_header").unwrap();
        let mut rt = EmbeddingRuntime::new(EmbeddingConfig {
            model_path: wrong,
            ..Default::default()
        });
        assert!(matches!(
            rt.load(),
            Err(EmbeddingError::InvalidModelFile { .. })
        ));

        let short = dir.path().join("short.gguf");
        std::fs::write(&short, b"GGUF").unwrap();
        let mut rt = EmbeddingRuntime::new(EmbeddingConfig {
            model_path: short,
            ..Default::default()
        });
        assert!(matches!(
            rt.load(),
            Err(EmbeddingError::InvalidModelFile { .. })
        ));
    }

    /// The whole header has to be there. A file holding only the magic and a
    /// version is 8 bytes of coincidence, not a model.
    #[test]
    fn load_rejects_a_header_that_stops_after_the_version() {
        let dir = TempDir::new("partial_header");
        let model = dir.path().join("partial.gguf");

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        assert_eq!(bytes.len(), 8);
        std::fs::write(&model, &bytes).unwrap();

        match validate_and_checksum_gguf(&model) {
            Err(EmbeddingError::InvalidModelFile { reason, .. }) => {
                assert!(reason.contains("header"), "unhelpful reason: {reason}");
            }
            other => panic!("an 8-byte file must be rejected, got {other:?}"),
        }

        // Truncated part-way through the counts, too.
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&[0u8; 4]);
        assert_eq!(bytes.len(), GGUF_HEADER_BYTES - 4);
        std::fs::write(&model, &bytes).unwrap();
        assert!(matches!(
            validate_and_checksum_gguf(&model),
            Err(EmbeddingError::InvalidModelFile { .. })
        ));
    }

    /// A corrupt or misidentified file can carry the right magic and a garbage
    /// tensor count. Real models have hundreds of tensors, not billions.
    #[test]
    fn load_rejects_implausible_header_counts() {
        let dir = TempDir::new("implausible");
        let model = dir.path().join("garbage.gguf");

        for (tensors, metadata) in [(u64::MAX, 0u64), (0, u64::MAX), (1 << 40, 1 << 40)] {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(b"GGUF");
            bytes.extend_from_slice(&3u32.to_le_bytes());
            bytes.extend_from_slice(&tensors.to_le_bytes());
            bytes.extend_from_slice(&metadata.to_le_bytes());
            std::fs::write(&model, &bytes).unwrap();

            assert!(
                matches!(
                    validate_and_checksum_gguf(&model),
                    Err(EmbeddingError::InvalidModelFile { .. })
                ),
                "tensors={tensors} metadata={metadata} must be rejected"
            );
        }
    }

    #[test]
    fn an_empty_but_well_formed_container_is_accepted() {
        // Zero tensors and zero metadata entries: nothing is declared, so nothing
        // is missing. This is what the test stubs write.
        let dir = TempDir::new("plausible");
        let model = dir.path().join("real.gguf");
        mock::write_stub_gguf(&model, 3).unwrap();
        assert!(validate_and_checksum_gguf(&model).is_ok());
    }

    #[test]
    fn load_rejects_unsupported_gguf_versions() {
        let dir = TempDir::new("version");
        let model = dir.path().join("other-version.gguf");

        // v1 stored the counts as u32; parsing it with the v2 layout would
        // misread the header, so it is refused rather than misparsed.
        for version in [0, 1, GGUF_SUPPORTED_VERSIONS.end() + 1] {
            mock::write_stub_gguf(&model, version).unwrap();
            let mut rt = EmbeddingRuntime::new(EmbeddingConfig {
                model_path: model.clone(),
                ..Default::default()
            });
            assert!(
                matches!(rt.load(), Err(EmbeddingError::InvalidModelFile { .. })),
                "version {version} must be refused"
            );
        }

        for version in GGUF_SUPPORTED_VERSIONS {
            mock::write_stub_gguf(&model, version).unwrap();
            assert!(
                validate_and_checksum_gguf(&model).is_ok(),
                "version {version} must be accepted"
            );
        }
    }

    /// A download cut short keeps its header, so the header alone cannot detect
    /// it. The declared counts can: each tensor and metadata entry has a minimum
    /// size, and a file below that floor is provably incomplete.
    #[test]
    fn load_rejects_a_truncated_download_whose_header_survived() {
        let dir = TempDir::new("truncated_download");
        let model = dir.path().join("interrupted.gguf");

        let header = |tensors: u64, metadata: u64| {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(b"GGUF");
            bytes.extend_from_slice(&3u32.to_le_bytes());
            bytes.extend_from_slice(&tensors.to_le_bytes());
            bytes.extend_from_slice(&metadata.to_le_bytes());
            bytes
        };

        // A realistic embedding model header, with almost none of the body.
        let mut interrupted = header(291, 24);
        interrupted.extend_from_slice(&[0u8; 128]);
        std::fs::write(&model, &interrupted).unwrap();

        match validate_and_checksum_gguf(&model) {
            Err(EmbeddingError::InvalidModelFile { reason, .. }) => {
                assert!(reason.contains("incomplete"), "unhelpful reason: {reason}");
            }
            other => panic!("a truncated download must be rejected, got {other:?}"),
        }

        // The same header with a body large enough to hold what it declares.
        let mut complete = header(291, 24);
        complete.resize(24 + 291 * 33 + 24 * 14 + 4096, 0u8);
        std::fs::write(&model, &complete).unwrap();
        assert!(validate_and_checksum_gguf(&model).is_ok());
    }

    #[test]
    fn load_computes_model_checksum_and_detects_a_changed_file() {
        let dir = TempDir::new("checksum");
        let model = dir.path().join("model.gguf");

        mock::write_stub_gguf(&model, 3).unwrap();
        let mut rt = EmbeddingRuntime::new(EmbeddingConfig {
            model_path: model.clone(),
            ..Default::default()
        });
        rt.load().unwrap();
        let first = rt.model_checksum().unwrap().to_string();
        assert_eq!(first.len(), 64, "sha256 hex is 64 chars");

        // Same bytes → same checksum (stable across loads).
        let mut again = EmbeddingRuntime::new(EmbeddingConfig {
            model_path: model.clone(),
            ..Default::default()
        });
        again.load().unwrap();
        assert_eq!(again.model_checksum().unwrap(), first);

        // Different bytes behind the same path → different checksum.
        let mut bytes = std::fs::read(&model).unwrap();
        bytes.extend_from_slice(b"extra tensor payload");
        std::fs::write(&model, &bytes).unwrap();
        let mut changed = EmbeddingRuntime::new(EmbeddingConfig {
            model_path: model,
            ..Default::default()
        });
        changed.load().unwrap();
        assert_ne!(changed.model_checksum().unwrap(), first);
    }

    #[test]
    fn checksum_is_computed_over_the_whole_file_across_buffer_boundaries() {
        let dir = TempDir::new("big");
        let a = dir.path().join("a.gguf");
        let b = dir.path().join("b.gguf");

        // Two files identical for the first HASH_BUFFER_BYTES + 1 bytes, both
        // carrying a valid header so the checksum is what is under test.
        let mut base = Vec::with_capacity(HASH_BUFFER_BYTES + 64);
        base.extend_from_slice(b"GGUF");
        base.extend_from_slice(&3u32.to_le_bytes());
        base.extend_from_slice(&8u64.to_le_bytes()); // tensor_count
        base.extend_from_slice(&2u64.to_le_bytes()); // metadata_kv_count
        base.resize(HASH_BUFFER_BYTES + 1, 7u8);

        let mut with_tail = base.clone();
        with_tail.push(9u8);
        std::fs::write(&a, &base).unwrap();
        std::fs::write(&b, &with_tail).unwrap();

        assert_ne!(
            validate_and_checksum_gguf(&a).unwrap(),
            validate_and_checksum_gguf(&b).unwrap(),
            "a trailing byte past the read buffer must change the checksum"
        );
    }

    #[test]
    fn embed_before_load_fails() {
        let rt = EmbeddingRuntime::new(EmbeddingConfig::default());
        assert!(matches!(
            rt.embed_one("שלום"),
            Err(EmbeddingError::NotLoaded)
        ));
        assert!(matches!(
            rt.embed_batch(&["שלום"]),
            Err(EmbeddingError::NotLoaded)
        ));
    }

    #[test]
    fn embeddings_are_normalized_and_have_configured_dim() {
        let dir = TempDir::new("normalized");
        let rt = loaded_runtime(&dir, 64);

        let v = rt.embed_one("בראשית ברא אלהים את השמים ואת הארץ").unwrap();
        assert_eq!(v.len(), 64);
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm was {norm}");
    }

    #[test]
    fn batch_and_single_paths_agree_and_preserve_order() {
        let dir = TempDir::new("batch_order");
        let rt = loaded_runtime(&dir, 32);

        // More texts than batch_size (4) so several backend calls are made.
        let texts = [
            "בראשית ברא אלהים",
            "והארץ היתה תהו ובהו",
            "ויאמר אלהים יהי אור",
            "ויהי אור",
            "ויקרא אלהים לאור יום",
            "ולחשך קרא לילה",
        ];
        let batched = rt.embed_batch(&texts).unwrap();
        assert_eq!(batched.len(), texts.len());

        for (i, text) in texts.iter().enumerate() {
            let single = rt.embed_one(text).unwrap();
            assert_eq!(
                batched[i], single,
                "batch element {i} must equal the single-text result"
            );
        }
    }

    #[test]
    fn empty_and_whitespace_only_text_is_rejected_not_stored_as_zero_vector() {
        let dir = TempDir::new("degenerate");
        let rt = loaded_runtime(&dir, 16);

        for text in ["", "   ", "\t\n  "] {
            assert!(
                matches!(
                    rt.embed_one(text),
                    Err(EmbeddingError::InferenceFailed { .. })
                ),
                "text {text:?} must not yield a vector"
            );
        }
    }

    /// A norm test alone is not enough: `NaN < MIN_VECTOR_NORM` is `false`, so a
    /// poisoned vector would be stored and then silently dropped at search time,
    /// leaving a book that is recorded as indexed but unsearchable.
    #[test]
    fn non_finite_vectors_are_rejected() {
        let cases: Vec<(&str, Vec<f32>)> = vec![
            ("NaN component", vec![f32::NAN, 1.0, 0.0, 0.0]),
            ("infinite component", vec![f32::INFINITY, 1.0, 0.0, 0.0]),
            (
                "negative infinite component",
                vec![f32::NEG_INFINITY, 1.0, 0.0, 0.0],
            ),
            ("all NaN", vec![f32::NAN; 4]),
            // Finite components whose squares overflow f32: the norm becomes
            // `inf`, the reciprocal `0`, and the vector normalizes to all zeros.
            ("finite but overflowing", vec![1e30, 1e30, 1e30, 1e30]),
        ];

        for (name, mut vector) in cases {
            let result = normalize_validated(&mut vector, 4);
            assert!(
                matches!(result, Err(EmbeddingError::InferenceFailed { .. })),
                "{name} must be rejected, got {result:?}"
            );
        }
    }

    #[test]
    fn a_zero_vector_is_rejected_for_having_no_direction() {
        let mut zero = vec![0.0f32; 4];
        assert!(matches!(
            normalize_validated(&mut zero, 4),
            Err(EmbeddingError::InferenceFailed { .. })
        ));
    }

    #[test]
    fn a_wrong_dimension_vector_is_rejected_before_anything_else() {
        let mut short = vec![f32::NAN, 1.0];
        assert!(
            matches!(
                normalize_validated(&mut short, 4),
                Err(EmbeddingError::DimensionMismatch {
                    expected: 4,
                    actual: 2
                })
            ),
            "the dimension is the more specific diagnosis"
        );
    }

    #[test]
    fn a_healthy_vector_is_normalized_in_place() {
        let mut vector = vec![3.0f32, 4.0, 0.0, 0.0];
        normalize_validated(&mut vector, 4).unwrap();

        let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        assert!((vector[0] - 0.6).abs() < 1e-6);
        assert!((vector[1] - 0.8).abs() < 1e-6);
        assert!(vector.iter().all(|x| x.is_finite()));
    }

    /// A very small but representable vector still has a direction and must be
    /// kept — the guard is against zero and non-finite, not against "small".
    #[test]
    fn a_tiny_but_representable_vector_is_kept() {
        let mut vector = vec![1e-6f32, 0.0, 0.0, 0.0];
        normalize_validated(&mut vector, 4).unwrap();
        assert!((vector[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn empty_batch_is_a_no_op() {
        let dir = TempDir::new("empty_batch");
        let rt = loaded_runtime(&dir, 16);
        assert!(rt.embed_batch(&[]).unwrap().is_empty());
    }

    #[test]
    fn backend_is_reported_and_marked_non_semantic() {
        let dir = TempDir::new("backend_kind");
        let rt = loaded_runtime(&dir, 16);
        let backend = rt.backend().expect("loaded");
        assert_eq!(backend, EmbeddingBackendKind::MockHash);
        assert_eq!(backend.id(), "mock-hash-v1");
        assert!(
            !backend.is_semantic(),
            "the stand-in backend must never claim to be semantic"
        );
    }

    #[test]
    fn identical_text_embeds_identically_across_runtimes() {
        let dir = TempDir::new("determinism");
        let a = loaded_runtime(&dir, 48);
        let b = loaded_runtime(&dir, 48);
        assert_eq!(
            a.embed_one("תלמוד תורה כנגד כולם").unwrap(),
            b.embed_one("תלמוד תורה כנגד כולם").unwrap()
        );
    }
}
