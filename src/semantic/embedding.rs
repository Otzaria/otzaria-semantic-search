//! Embedding model runtime: the shell *around* inference, never inference
//! itself — the configuration's checks, the batch API, and the single place a
//! vector's dimension, finiteness and norm are checked before it can enter the
//! index. What an implementation must provide, and which one a build gets, live in
//! [`backend`](crate::semantic::backend); what an ONNX model *is* on disk, and how it
//! is validated and checksummed, in [`model_package`](crate::semantic::model_package).

use crate::errors::EmbeddingError;
use crate::semantic::backend::{
    ensure_pooling_is_implemented, max_tokens_past_any_encoder, select_backend_for,
    EmbeddingBackend, Pooling,
};
use crate::semantic::model_package::{ensure_onnx_model_path, validate_model};
use std::io::Read;
use std::path::PathBuf;

/// Read buffer for hashing, large enough that hashing is bound by the hash rather
/// than by syscalls.
pub(crate) const HASH_BUFFER_BYTES: usize = 1 << 20;

/// At or below this L2 norm a vector carries no direction and cannot be
/// normalized.
///
/// One threshold for the whole crate, compared with the same `<=` at every layer:
/// the runtime, the store's guard and the query path. Two copies would drift, and
/// a vector one layer rejects while another normalizes is exactly the record that
/// exists but can never be found.
pub(crate) const MIN_VECTOR_NORM: f32 = 1e-12;

/// Configuration for embedding runtime loading.
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    pub model_path: PathBuf,
    /// Dimensionality every stored vector must have, checked against the loaded
    /// backend's [`EmbeddingBackend::dim`]: a real backend reads its width from
    /// the model file, and a silent disagreement fills the index with wrong-width
    /// vectors.
    pub embedding_dim: u32,
    /// Token cap *requested* for a single input — the total sequence length, special
    /// tokens included. This layer never enforces it — it has no tokenizer; it is a
    /// contract the backend implements, spelled out at
    /// [`EmbeddingBackend::max_tokens`]. Here it only travels: the backend reports it
    /// back and the manifest records it, so a change is detected as an incompatibility
    /// instead of silently changing what gets embedded.
    pub max_tokens: usize,
    /// Number of texts handed to the backend per inference call.
    pub batch_size: usize,
    /// How the backend must collapse token states into one vector. Typed rather
    /// than a free string: it is part of the index's identity in the manifest.
    pub pooling: Pooling,
}

/// The production model: the Meivin Round 2 int8 graph, as its identity
/// (`config/models/meivin-round2-onnx/model.json`) declares it, at a path relative to the
/// working directory. `defaults_are_the_production_identity` holds the two together.
impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("models/meivin-round2-onnx/seforim-embed-round2-int8.onnx"),
            embedding_dim: 256,
            max_tokens: 256,
            batch_size: 32,
            pooling: Pooling::InGraph,
        }
    }
}

impl EmbeddingConfig {
    /// Reject a configuration no backend could serve, before the model file is
    /// opened. Also called by [`select_backend_for`], reachable without `load`.
    ///
    /// `batch_size` is deliberately absent: zero there is recoverable and means
    /// "one text per call" ([`EmbeddingRuntime::batch_size`] clamps it), whereas a
    /// zero dimensionality or token cap makes every embedding degenerate. A model path
    /// that names no ONNX graph — a GGUF, say — is refused here, as
    /// [`EmbeddingError::InvalidModelFile`] (see
    /// [`ensure_onnx_model_path`]), since no backend reads anything else. The pooling
    /// check asks whether a backend *performs* the strategy, and belongs here because the
    /// value is persisted as the index's identity — see [`ensure_pooling_is_implemented`].
    /// So is the token cap, which is held to
    /// [`ONNX_MAX_TOKENS_CEILING`](crate::semantic::backend::ONNX_MAX_TOKENS_CEILING)
    /// here for the same reason.
    pub fn validate(&self) -> Result<(), EmbeddingError> {
        if self.embedding_dim == 0 {
            return Err(EmbeddingError::LoadFailed {
                reason: "embedding_dim is 0; a vector with no components cannot be \
                         stored or compared"
                    .to_string(),
            });
        }
        // 2, not 1: the cap is the total sequence length, and every backend spends at
        // least one token of it on a special token — a BERT tokenizer's [CLS] and
        // [SEP]. A cap of 1 leaves no budget for content and embeds every text as a bare
        // special token: a plausible-looking vector carrying nothing of the input. A
        // backend whose specials take more refuses the cap at load, where it knows how
        // many it adds.
        if self.max_tokens < 2 {
            return Err(EmbeddingError::LoadFailed {
                reason: format!(
                    "max_tokens is {}; the cap counts the special tokens the model adds \
                     around the text, so at least 2 are needed for any content to reach the \
                     model",
                    self.max_tokens
                ),
            });
        }
        ensure_onnx_model_path(&self.model_path)?;
        if let Some(reason) = max_tokens_past_any_encoder("max_tokens", self.max_tokens) {
            return Err(EmbeddingError::LoadFailed { reason });
        }
        ensure_pooling_is_implemented(self.pooling)?;
        Ok(())
    }
}

/// Where this machine keeps what a backend loads besides the model — today, the ONNX
/// Runtime shared library.
///
/// Deployment, not identity. [`EmbeddingConfig`] says what the vectors are, and is persisted
/// as an index's identity; this says where the code that computes them lives, which differs
/// between the build machine and every device while their vectors stay comparable. So
/// nothing here reaches a manifest, an artifact's identity, a chunking identity or a backend
/// id, and changing it invalidates nothing. A type of its own, held beside the configuration
/// rather than inside it, so that no code deriving an identity from an [`EmbeddingConfig`]
/// can pick it up.
///
/// A host passes it through
/// [`SemanticConfig::deployment`](crate::semantic::engine::SemanticConfig::deployment) or
/// [`OfficialIndexConfig::deployment`](crate::semantic::official_index::OfficialIndexConfig::deployment);
/// [`EmbeddingRuntime::with_deployment`] hands it to the backend. The default passes
/// nothing, which leaves every lookup at its own default — what the build-machine tools run
/// with.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EmbeddingDeployment {
    /// The ONNX Runtime shared library an ONNX model runs on (`onnxruntime.dll`,
    /// `libonnxruntime.so`, `libonnxruntime.dylib`), for an application that ships the
    /// runtime itself — on macOS signed inside the application's bundle, which is what
    /// library validation accepts.
    ///
    /// `Some` is the first place looked and, once set, the only one: a path that cannot be
    /// opened or loaded is [`EmbeddingError::OnnxRuntimeUnavailable`] naming it, never a
    /// fall-back to `OTZARIA_ONNX_RUNTIME` or to the file beside the graph, which would run a
    /// runtime the application did not choose. An empty path is refused for the same reason:
    /// it names no file. `None` looks at `OTZARIA_ONNX_RUNTIME`, then beside the graph. A
    /// relative path is resolved against the current directory, so pass an absolute one.
    ///
    /// A process holds one runtime: once one has loaded, a different path is refused rather
    /// than ignored, because ONNX Runtime cannot be unloaded or replaced. Ignored by the
    /// stand-in, which runs nothing. `docs/ONNX_BACKEND.md` §3 has the whole lookup.
    pub onnx_runtime: Option<PathBuf>,
}

/// Local embedding runtime: owns the *policy* around a backend, while the backend
/// owns inference. Batching, the returned-count check, dimension/finiteness/norm
/// validation and L2 normalization happen here exactly once.
pub struct EmbeddingRuntime {
    config: EmbeddingConfig,
    /// Handed to the backend when it is built; never compared with anything.
    deployment: EmbeddingDeployment,
    /// `None` until a successful [`Self::load`]. Boxed rather than generic: a type
    /// parameter would push a build-time choice into every signature above this
    /// one, up through `SemanticEngine` and the coordinator.
    backend: Option<Box<dyn EmbeddingBackend>>,
    /// The model's `model_checksum`, computed by [`Self::load`] — see
    /// [`Self::model_checksum`] for what it covers per format.
    model_checksum: Option<String>,
    /// SHA-256 of the package's `tokenizer.json`, read by the same validation.
    tokenizer_checksum: Option<String>,
}

impl EmbeddingRuntime {
    /// Initialize runtime with configuration, deployed as the defaults describe (see
    /// [`EmbeddingDeployment`]). No file access happens here.
    pub fn new(config: EmbeddingConfig) -> Self {
        Self::with_deployment(config, EmbeddingDeployment::default())
    }

    /// Initialize runtime with configuration and where this machine keeps what the backend
    /// loads besides the model — the path a host that ships ONNX Runtime passes. No file
    /// access happens here; [`Self::load`] hands `deployment` to the backend it builds.
    pub fn with_deployment(config: EmbeddingConfig, deployment: EmbeddingDeployment) -> Self {
        Self {
            config,
            deployment,
            backend: None,
            model_checksum: None,
            tokenizer_checksum: None,
        }
    }

    /// Load the model from disk.
    ///
    /// Nothing is installed unless every step succeeds: a failed load leaves the
    /// runtime exactly as unloaded as it was, rather than half-loaded with a
    /// checksum recorded for a backend that never materialized.
    pub fn load(&mut self) -> Result<(), EmbeddingError> {
        self.config.validate()?;

        if !self.config.model_path.exists() {
            return Err(EmbeddingError::ModelNotFound {
                path: self.config.model_path.display().to_string(),
            });
        }

        // The ONNX graph, with the package around it: every file of it whole, and the
        // checksum that names them all.
        let validated = validate_model(&self.config.model_path)?;
        let backend = select_backend_for(&self.config, &self.deployment)?;
        let tokenizer = validated
            .files()
            .iter()
            .find(|file| file.relpath == crate::semantic::model_package::ONNX_TOKENIZER_FILE)
            .map(|file| file.sha256.clone());
        self.adopt(backend, Some(validated.checksum().to_string()))?;
        self.tokenizer_checksum = tokenizer;
        Ok(())
    }

    /// Install a backend after checking it agrees with this configuration.
    ///
    /// A real backend reads width, context length and pooling from the model file,
    /// and each disagreement is a silent corruption if let through: a wrong
    /// dimensionality fails mid-index, a wrong pooling mislabels stored vectors in
    /// the manifest, a zero cap truncates every input to nothing. Shared by
    /// [`Self::load`] and the test-only constructor.
    fn adopt(
        &mut self,
        backend: Box<dyn EmbeddingBackend>,
        checksum: Option<String>,
    ) -> Result<(), EmbeddingError> {
        if backend.dim() != self.config.embedding_dim {
            return Err(EmbeddingError::DimensionMismatch {
                expected: self.config.embedding_dim,
                actual: backend.dim(),
            });
        }
        if backend.pooling() != self.config.pooling {
            return Err(EmbeddingError::PoolingMismatch {
                backend: backend.id().to_string(),
                configured: self.config.pooling.to_string(),
                actual: backend.pooling().to_string(),
            });
        }
        if backend.max_tokens() == 0 {
            return Err(EmbeddingError::LoadFailed {
                reason: format!(
                    "backend '{}' reports a token cap of 0, so every input would \
                     truncate to nothing",
                    backend.id()
                ),
            });
        }

        // An index built from non-semantic vectors looks entirely normal from the
        // outside; nothing downstream can tell from the vectors alone.
        if !backend.is_semantic() {
            log::warn!(
                "Embedding backend '{}' reports that its vectors are NOT semantic \
                 (deterministic hashing); this build is not fit for production. \
                 Model file: {}",
                backend.id(),
                self.config.model_path.display()
            );
        }

        self.model_checksum = checksum;
        self.backend = Some(backend);
        Ok(())
    }

    /// Install a ready-made backend, for tests that drive the runtime's validation
    /// with backends misbehaving on purpose. Deliberately not public: a public
    /// constructor would let a release build inject a fake and serve its vectors as
    /// semantic, which `tests/production_backend_gate.rs` exists to keep refused.
    #[cfg(test)]
    fn with_backend(
        config: EmbeddingConfig,
        backend: Box<dyn EmbeddingBackend>,
    ) -> Result<Self, EmbeddingError> {
        let mut runtime = Self::new(config);
        runtime.adopt(backend, None)?;
        Ok(runtime)
    }

    /// Check if a backend is currently loaded.
    pub fn is_loaded(&self) -> bool {
        self.backend.is_some()
    }

    /// The loaded backend, or `None` before a successful [`Self::load`].
    pub fn backend(&self) -> Option<&dyn EmbeddingBackend> {
        self.backend.as_deref()
    }

    /// Identifier of the loaded backend, recorded by the manifest and the status
    /// report. Two backends do not produce comparable vectors even from the same
    /// weights, so the id is what tells a later session that the index it is about
    /// to query was built by something else.
    pub fn backend_id(&self) -> Option<&'static str> {
        self.backend.as_ref().map(|backend| backend.id())
    }

    /// Whether the loaded backend's vectors carry semantic meaning. `false` before
    /// a model is loaded too: nothing has produced a meaningful vector then either.
    pub fn backend_is_semantic(&self) -> bool {
        self.backend
            .as_ref()
            .is_some_and(|backend| backend.is_semantic())
    }

    /// The loaded model's `model_checksum`, or `None` before a successful load: the
    /// package checksum, the SHA-256 of a canonical manifest listing the graph, every
    /// external-data file it names and `tokenizer.json`, each with its size and SHA-256 —
    /// see [`model_package`](crate::semantic::model_package). Nothing else in the
    /// directory is covered, because nothing else reaches a vector.
    ///
    /// Lowercase hex, 64 digits, and a statement that the bytes behind the model path are
    /// the ones an index was built with — not a download verification.
    pub fn model_checksum(&self) -> Option<&str> {
        self.model_checksum.as_deref()
    }

    /// SHA-256 of the loaded package's `tokenizer.json`, or `None` before a successful
    /// load — the half of a model family's identity a package can be checked against
    /// directly: two packages of one family share their tokenizer byte for byte.
    pub fn tokenizer_checksum(&self) -> Option<&str> {
        self.tokenizer_checksum.as_deref()
    }

    /// Convenience wrapper over [`Self::embed_batch`]; indexing should call the
    /// batch form directly so the backend sees whole batches.
    pub fn embed_one(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let mut out = self.embed_batch(&[text])?;
        out.pop().ok_or_else(|| EmbeddingError::InferenceFailed {
            reason: "backend returned no vector for a single input".to_string(),
        })
    }

    /// Embed a batch into L2-normalized vectors, one per input, in input order.
    ///
    /// **The single normalization and validation choke point** — every vector this
    /// crate produces passes through here, which is what lets
    /// [`EmbeddingBackend::embed_batch_raw`] return raw vectors. A backend that
    /// normalized or screened its own output would hide a degenerate vector behind
    /// a plausible unit norm before this code could see it.
    /// [`VectorStore::insert_batch`](crate::semantic::store::VectorStore::insert_batch)
    /// re-checks the same invariant against the same `MIN_VECTOR_NORM`, because it
    /// is public and accepts vectors that never came through here.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let Some(backend) = self.backend.as_deref() else {
            return Err(EmbeddingError::NotLoaded);
        };

        let dim = self.config.embedding_dim;
        let mut results = Vec::with_capacity(texts.len());

        for group in texts.chunks(self.batch_size()) {
            let raw = backend.embed_batch_raw(group)?;

            // Vectors get their metadata by position upstream, so a short batch
            // would attach one line's text to another line's reference.
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

    /// Expected embedding dimensionality: the configured value, which `adopt` has
    /// proven equal to the backend's [`EmbeddingBackend::dim`] and which is also
    /// answerable before a model is loaded.
    pub fn dim(&self) -> u32 {
        self.config.embedding_dim
    }

    /// Pooling strategy this runtime is configured for, which `adopt` has proven
    /// the loaded backend performs.
    pub fn pooling(&self) -> Pooling {
        self.config.pooling
    }

    /// The token cap in force: the loaded backend's effective one, or the requested
    /// one before a model is loaded. A backend reports the cap it applies, which is what
    /// counts once one is loaded. This layer never truncates — the cap is the backend's
    /// contract, see [`EmbeddingBackend::max_tokens`].
    pub fn max_tokens(&self) -> usize {
        self.backend
            .as_ref()
            .map_or(self.config.max_tokens, |backend| backend.max_tokens())
    }

    /// Maximum number of texts sent to the backend per inference call.
    pub fn batch_size(&self) -> usize {
        self.config.batch_size.max(1)
    }
}

/// What a CPU worker records as its device: the architecture, and — where the ONNX backend
/// is compiled — the int8 kernels ONNX Runtime gives this CPU and whether the backend makes
/// them exact (the x86 gate), so a shard says which products its vectors came from.
pub fn cpu_description() -> String {
    let arch = std::env::consts::ARCH;
    match onnx_int8_kernels() {
        Some(kernels) => format!("cpu {arch}: {kernels}"),
        None => format!("cpu {arch}"),
    }
}

#[cfg(all(
    feature = "onnx-backend",
    any(
        all(
            target_os = "macos",
            any(target_arch = "aarch64", target_arch = "x86_64")
        ),
        all(
            target_os = "linux",
            target_env = "gnu",
            any(target_arch = "aarch64", target_arch = "x86_64")
        ),
        all(
            target_os = "windows",
            target_env = "msvc",
            any(target_arch = "aarch64", target_arch = "x86_64")
        )
    )
))]
fn onnx_int8_kernels() -> Option<String> {
    Some(crate::semantic::onnx_backend::cpu_int8_kernels())
}

#[cfg(not(all(
    feature = "onnx-backend",
    any(
        all(
            target_os = "macos",
            any(target_arch = "aarch64", target_arch = "x86_64")
        ),
        all(
            target_os = "linux",
            target_env = "gnu",
            any(target_arch = "aarch64", target_arch = "x86_64")
        ),
        all(
            target_os = "windows",
            target_env = "msvc",
            any(target_arch = "aarch64", target_arch = "x86_64")
        )
    )
)))]
fn onnx_int8_kernels() -> Option<String> {
    None
}

/// L2-normalize a vector in place after checking it can be compared at all.
///
/// **Unconditional, and deliberately not versioned.** Unit length is what makes a dot
/// product a cosine, and every store applies it on insertion and on query regardless of
/// what any manifest says. `normalization_version` is about the *text* that reaches the
/// model — see [`TextNormalizationRecipe`](crate::semantic::recipe::TextNormalizationRecipe)
/// — and giving that name to this would produce a "version" three code paths ignore.
///
/// Two of the four guards are not redundant with a norm test: `NaN <
/// MIN_VECTOR_NORM` is `false`, so a poisoned component passes one; and a
/// *finite* vector whose squares overflow `f32` (components around `1e30`) gets an
/// `inf` norm, a `0` reciprocal, and normalizes silently to all zeros.
///
/// The failure mode is quiet — the book is recorded as indexed while every one of
/// its vectors scores `NaN` and is discarded at search time.
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
    // `<=`, matching the `>` in `l2_normalize` and the store's guard. With `<`,
    // the vector exactly on the threshold was neither rejected nor normalized and
    // scored as its own magnitude rather than as a cosine.
    if norm <= MIN_VECTOR_NORM {
        return Err(EmbeddingError::InferenceFailed {
            reason: format!(
                "vector norm {norm} is at or below the minimum {MIN_VECTOR_NORM}; it has no \
                 usable direction"
            ),
        });
    }

    // Cannot fire once input and norm are finite, since normalization only shrinks
    // magnitudes; kept as a cheap backstop.
    debug_assert!(vector.iter().all(|x| x.is_finite()));
    Ok(())
}

/// L2-normalize in place and return the norm the vector had beforehand. A norm
/// indistinguishable from zero leaves the vector untouched, so the caller must
/// treat the returned norm as a failure signal. Prefer [`normalize_validated`].
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

/// EOF inside a structure the file itself declared is proof of truncation; an I/O
/// error is a fact about the disk. Only the first condemns the model.
pub(crate) enum ReadError {
    Eof { at: u64 },
    Io(std::io::Error),
}

/// Buffered forward reader that hashes everything it passes over, so validating
/// and checksumming a model of hundreds of megabytes reads it once — the ONNX
/// package walk in [`model_package`](crate::semantic::model_package).
pub(crate) struct HashingReader {
    inner: std::io::BufReader<std::fs::File>,
    hasher: sha2::Sha256,
    consumed: u64,
}

impl HashingReader {
    pub(crate) fn new(file: std::fs::File) -> Self {
        use sha2::Digest;
        Self {
            inner: std::io::BufReader::with_capacity(HASH_BUFFER_BYTES, file),
            hasher: sha2::Sha256::new(),
            consumed: 0,
        }
    }

    /// Bytes read — and therefore hashed — so far.
    pub(crate) fn consumed(&self) -> u64 {
        self.consumed
    }

    fn read_exact_hashed(&mut self, buf: &mut [u8]) -> std::io::Result<()> {
        use sha2::Digest;
        self.inner.read_exact(buf)?;
        self.hasher.update(&*buf);
        self.consumed += buf.len() as u64;
        Ok(())
    }

    pub(crate) fn fill(&mut self, buf: &mut [u8]) -> Result<(), ReadError> {
        match self.read_exact_hashed(buf) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                Err(ReadError::Eof { at: self.consumed })
            }
            Err(e) => Err(ReadError::Io(e)),
        }
    }

    /// Skip `len` bytes, hashing them.
    pub(crate) fn skip(&mut self, mut len: u64) -> Result<(), ReadError> {
        const CHUNK: usize = 64 << 10;
        let mut scratch = vec![0u8; (len.min(CHUNK as u64)) as usize];
        while len > 0 {
            let wanted = len.min(scratch.len() as u64) as usize;
            self.fill(&mut scratch[..wanted])?;
            len -= wanted as u64;
        }
        Ok(())
    }

    /// Hash whatever is left; returns the digest and the total byte count.
    pub(crate) fn finish(mut self) -> std::io::Result<(String, u64)> {
        use sha2::Digest;
        let mut buffer = vec![0u8; HASH_BUFFER_BYTES];
        loop {
            let read = self.inner.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            self.hasher.update(&buffer[..read]);
            self.consumed += read as u64;
        }
        Ok((hex_encode(&self.hasher.finalize()), self.consumed))
    }
}

/// Lower-case hex encoding.
pub(crate) fn hex_encode(bytes: &[u8]) -> String {
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

    /// Feature-hash `text` into a `dim`-sized vector, deterministically across runs
    /// and platforms. Whitespace-only input yields the zero vector, which
    /// [`super::EmbeddingRuntime::embed_batch`] rejects.
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

    /// Just enough of the protobuf wire format to hand-encode ONNX fixtures.
    ///
    /// Written independently of the walk in
    /// [`model_package`](crate::semantic::model_package), so a fixture cannot share the
    /// walk's mistakes about the format.
    pub mod proto {
        pub const VARINT: u8 = 0;
        pub const FIXED64: u8 = 1;
        pub const LEN: u8 = 2;
        pub const START_GROUP: u8 = 3;
        pub const END_GROUP: u8 = 4;
        pub const FIXED32: u8 = 5;

        pub fn varint(out: &mut Vec<u8>, mut value: u64) {
            loop {
                let low = (value & 0x7f) as u8;
                value >>= 7;
                if value == 0 {
                    out.push(low);
                    return;
                }
                out.push(low | 0x80);
            }
        }

        pub fn key(out: &mut Vec<u8>, field: u32, wire: u8) {
            varint(out, (u64::from(field) << 3) | u64::from(wire));
        }

        pub fn uint(out: &mut Vec<u8>, field: u32, value: u64) {
            key(out, field, VARINT);
            varint(out, value);
        }

        /// An `int64`: a negative value is written as its ten-byte two's complement.
        pub fn int(out: &mut Vec<u8>, field: u32, value: i64) {
            uint(out, field, value as u64);
        }

        pub fn bytes(out: &mut Vec<u8>, field: u32, payload: &[u8]) {
            key(out, field, LEN);
            varint(out, payload.len() as u64);
            out.extend_from_slice(payload);
        }

        pub fn string(out: &mut Vec<u8>, field: u32, value: &str) {
            bytes(out, field, value.as_bytes());
        }
    }

    /// ONNX messages for fixtures, built from [`proto`] with the field numbers of
    /// `onnx/onnx.proto3`.
    pub mod onnx {
        use super::proto;

        /// `TensorProto.DataType.FLOAT`.
        pub const FLOAT: u64 = 1;
        /// `TensorProto.DataType.INT64`.
        pub const INT64: u64 = 7;

        /// One dimension of a declared shape.
        pub enum Dim<'a> {
            Fixed(i64),
            Named(&'a str),
        }

        /// A `ValueInfoProto` declaring a tensor of `elem_type` and `dims`.
        pub fn value_info(name: &str, elem_type: u64, dims: &[Dim<'_>]) -> Vec<u8> {
            let mut shape = Vec::new();
            for dim in dims {
                let mut dimension = Vec::new();
                match dim {
                    Dim::Fixed(value) => proto::int(&mut dimension, 1, *value),
                    Dim::Named(param) => proto::string(&mut dimension, 2, param),
                }
                proto::bytes(&mut shape, 1, &dimension);
            }
            let mut tensor_type = Vec::new();
            proto::uint(&mut tensor_type, 1, elem_type);
            proto::bytes(&mut tensor_type, 2, &shape);
            let mut type_proto = Vec::new();
            proto::bytes(&mut type_proto, 1, &tensor_type);

            let mut value_info = Vec::new();
            proto::string(&mut value_info, 1, name);
            proto::bytes(&mut value_info, 2, &type_proto);
            value_info
        }

        /// An `OperatorSetIdProto`.
        pub fn opset(domain: &str, version: u64) -> Vec<u8> {
            let mut opset = Vec::new();
            proto::string(&mut opset, 1, domain);
            proto::uint(&mut opset, 2, version);
            opset
        }

        /// A `StringStringEntryProto`.
        pub fn entry(key: &str, value: &str) -> Vec<u8> {
            let mut entry = Vec::new();
            proto::string(&mut entry, 1, key);
            proto::string(&mut entry, 2, value);
            entry
        }

        /// A float `TensorProto` of `dims` whose data lives in `location`, at `offset`
        /// for `length` bytes when given.
        pub fn external_tensor(
            name: &str,
            dims: &[i64],
            location: &str,
            offset: Option<u64>,
            length: Option<u64>,
        ) -> Vec<u8> {
            let mut tensor = Vec::new();
            for dim in dims {
                proto::int(&mut tensor, 1, *dim);
            }
            proto::uint(&mut tensor, 2, FLOAT);
            proto::string(&mut tensor, 8, name);
            proto::bytes(&mut tensor, 13, &entry("location", location));
            if let Some(offset) = offset {
                proto::bytes(&mut tensor, 13, &entry("offset", &offset.to_string()));
            }
            if let Some(length) = length {
                proto::bytes(&mut tensor, 13, &entry("length", &length.to_string()));
            }
            proto::uint(&mut tensor, 14, 1); // data_location = EXTERNAL
            tensor
        }

        /// A `GraphProto` with the given inputs, outputs and initializers.
        pub fn graph(
            name: &str,
            inputs: &[Vec<u8>],
            outputs: &[Vec<u8>],
            initializers: &[Vec<u8>],
        ) -> Vec<u8> {
            let mut graph = Vec::new();
            proto::string(&mut graph, 2, name);
            for initializer in initializers {
                proto::bytes(&mut graph, 5, initializer);
            }
            for input in inputs {
                proto::bytes(&mut graph, 11, input);
            }
            for output in outputs {
                proto::bytes(&mut graph, 12, output);
            }
            graph
        }

        /// A `ModelProto`, its fields in the order a protobuf serializer writes them.
        pub fn model(ir_version: u64, graph: &[u8], opsets: &[Vec<u8>]) -> Vec<u8> {
            let mut model = Vec::new();
            proto::uint(&mut model, 1, ir_version);
            proto::string(&mut model, 2, "otzaria-stub");
            proto::bytes(&mut model, 7, graph);
            for opset in opsets {
                proto::bytes(&mut model, 8, opset);
            }
            model
        }

        /// The inputs every stub graph declares: the two a sentence encoder takes.
        pub fn encoder_inputs() -> Vec<Vec<u8>> {
            ["input_ids", "attention_mask"]
                .into_iter()
                .map(|name| {
                    value_info(name, INT64, &[Dim::Fixed(1), Dim::Named("sequence_length")])
                })
                .collect()
        }

        /// The graph [`super::write_stub_onnx_package`] writes: IR 8, opset 17, the
        /// inputs `input_ids` and `attention_mask` (`int64[1, sequence_length]`) and the
        /// output `sentence_embedding` (`float[1, 8]`). No nodes, so no runtime would
        /// run it; the validator's structural checks all pass.
        pub fn stub_graph() -> Vec<u8> {
            stub_graph_named("otzaria-stub-encoder")
        }

        /// [`stub_graph`] under another graph name: just as valid, in other bytes — what
        /// a second model, or the same model's file replaced, is to a checksum.
        pub fn stub_graph_named(name: &str) -> Vec<u8> {
            let output = value_info("sentence_embedding", FLOAT, &[Dim::Fixed(1), Dim::Fixed(8)]);
            model(
                8,
                &graph(name, &encoder_inputs(), &[output], &[]),
                &[opset("", 17)],
            )
        }
    }

    /// A `tokenizer.json` in the shape of a Hugging Face tokenizer, and a whole JSON
    /// object — which is all the validator asks of it.
    pub const STUB_TOKENIZER_JSON: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"[UNK]":0,"[CLS]":1,"[SEP]":2,"[QUERY]":3,"[PASSAGE]":4},"unk_token":"[UNK]"}}"#;

    /// SHA-256 of [`STUB_TOKENIZER_JSON`]: the tokenizer checksum of every stub package.
    pub fn stub_tokenizer_checksum() -> String {
        use sha2::Digest;
        format!("{:x}", sha2::Sha256::digest(STUB_TOKENIZER_JSON.as_bytes()))
    }

    /// Write a minimal valid ONNX package into `dir` — [`onnx::stub_graph`] as
    /// `model.onnx`, and [`STUB_TOKENIZER_JSON`] beside it — and return the graph's path,
    /// which is what a configuration names.
    ///
    /// # Panics
    ///
    /// If `dir` cannot be written: a fixture for tests, not a library path.
    pub fn write_stub_onnx_package(dir: &std::path::Path) -> std::path::PathBuf {
        std::fs::create_dir_all(dir).expect("the fixture directory must be writable");
        let graph = dir.join("model.onnx");
        std::fs::write(&graph, onnx::stub_graph()).expect("the stub graph must be writable");
        std::fs::write(dir.join("tokenizer.json"), STUB_TOKENIZER_JSON)
            .expect("the stub tokenizer must be writable");
        graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    struct TempDir(PathBuf);
    impl TempDir {
        fn new(name: &str) -> Self {
            // The clock alone collided: macOS ticks coarser than a test takes to start.
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "otzaria_embed_test_{name}_{}_{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos(),
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
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
        let model = mock::write_stub_onnx_package(dir.path());
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
            model_path: dir.path().join("nope.onnx"),
            ..Default::default()
        });
        assert!(matches!(
            rt.load(),
            Err(EmbeddingError::ModelNotFound { .. })
        ));
        assert!(!rt.is_loaded());
    }

    #[test]
    fn load_rejects_a_placeholder_that_is_not_a_graph() {
        let dir = TempDir::new("placeholder");
        let model = mock::write_stub_onnx_package(dir.path());
        std::fs::write(&model, b"ONNX_MOCK").unwrap();

        let mut rt = EmbeddingRuntime::new(EmbeddingConfig {
            model_path: model,
            ..Default::default()
        });
        assert!(matches!(
            rt.load(),
            Err(EmbeddingError::InvalidModelFile { .. })
        ));
        assert!(!rt.is_loaded());
    }

    #[test]
    fn load_computes_model_checksum_and_detects_a_changed_file() {
        let dir = TempDir::new("checksum");
        let model = mock::write_stub_onnx_package(dir.path());

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

        // Different bytes behind the same path → different checksum: here the
        // tokenizer's, which decides the ids the graph sees as surely as the graph does.
        let tokenizer = dir.path().join("tokenizer.json");
        let changed_tokenizer = mock::STUB_TOKENIZER_JSON.replace("[UNK]\":0", "[UNK]\":9");
        assert_ne!(changed_tokenizer, mock::STUB_TOKENIZER_JSON);
        std::fs::write(&tokenizer, changed_tokenizer).unwrap();
        let mut changed = EmbeddingRuntime::new(EmbeddingConfig {
            model_path: model,
            ..Default::default()
        });
        changed.load().unwrap();
        assert_ne!(changed.model_checksum().unwrap(), first);
    }

    /// Every file of the package is hashed whole, past the read buffer too: two packages
    /// identical but for one byte beyond the first `HASH_BUFFER_BYTES + 1` of an
    /// external-data file must not share a checksum.
    #[test]
    fn checksum_is_computed_over_whole_files_across_buffer_boundaries() {
        use mock::onnx::{self, Dim};

        let dir = TempDir::new("big");
        let package = |name: &str, tail: &[u8]| -> PathBuf {
            let root = dir.path().join(name);
            std::fs::create_dir_all(&root).unwrap();
            let mut weights = vec![7u8; HASH_BUFFER_BYTES + 1];
            weights.extend_from_slice(tail);
            std::fs::write(root.join("weights.bin"), weights).unwrap();
            let output = onnx::value_info(
                "sentence_embedding",
                onnx::FLOAT,
                &[Dim::Fixed(1), Dim::Fixed(8)],
            );
            let tensor = onnx::external_tensor("w", &[8], "weights.bin", None, None);
            let graph = onnx::model(
                8,
                &onnx::graph("big", &onnx::encoder_inputs(), &[output], &[tensor]),
                &[onnx::opset("", 17)],
            );
            std::fs::write(root.join("model.onnx"), graph).unwrap();
            std::fs::write(root.join("tokenizer.json"), mock::STUB_TOKENIZER_JSON).unwrap();
            root.join("model.onnx")
        };

        let base = validate_model(&package("base", b"")).unwrap();
        let with_tail = validate_model(&package("with_tail", b"\x09")).unwrap();
        assert_ne!(
            base.checksum(),
            with_tail.checksum(),
            "a trailing byte past the read buffer must change the checksum"
        );
    }

    /// An ONNX package loads through the same `load`, and what it records is the
    /// package checksum — not the graph's own hash, which would miss a changed tokenizer.
    #[test]
    fn load_accepts_an_onnx_package_and_records_its_package_checksum() {
        let dir = TempDir::new("onnx_package");
        let graph = mock::write_stub_onnx_package(dir.path());

        let mut rt = EmbeddingRuntime::new(EmbeddingConfig {
            model_path: graph.clone(),
            embedding_dim: 16,
            max_tokens: 256,
            pooling: Pooling::InGraph,
            ..Default::default()
        });
        rt.load()
            .expect("the stub package loads through the stand-in");

        let package = crate::semantic::model_package::validate_onnx_package(&graph).unwrap();
        assert_eq!(rt.model_checksum(), Some(package.checksum()));
        let graph_alone = {
            use sha2::Digest;
            hex_encode(&sha2::Sha256::digest(std::fs::read(&graph).unwrap()))
        };
        assert_ne!(
            rt.model_checksum(),
            Some(graph_alone.as_str()),
            "the checksum covers the package, not the graph alone"
        );
        assert_eq!(rt.pooling(), Pooling::InGraph);
        assert_eq!(rt.backend_id(), Some("mock-hash-v1"));
        assert_eq!(rt.embed_one("שלום עולם").unwrap().len(), 16);
    }

    #[test]
    fn load_refuses_an_onnx_package_without_its_tokenizer() {
        let dir = TempDir::new("onnx_no_tokenizer");
        let graph = mock::write_stub_onnx_package(dir.path());
        std::fs::remove_file(dir.path().join("tokenizer.json")).unwrap();

        let mut rt = EmbeddingRuntime::new(EmbeddingConfig {
            model_path: graph,
            embedding_dim: 16,
            pooling: Pooling::InGraph,
            ..Default::default()
        });
        assert!(matches!(
            rt.load(),
            Err(EmbeddingError::TokenizerNotFound { .. })
        ));
        assert!(!rt.is_loaded());
        assert!(rt.model_checksum().is_none());
    }

    /// A GGUF container behind an `.onnx` name is read as a graph, because the path
    /// decides, and is refused by what it is.
    #[test]
    fn a_gguf_named_like_an_onnx_graph_is_refused_by_what_it_is() {
        let dir = TempDir::new("gguf_as_onnx");
        let graph = mock::write_stub_onnx_package(dir.path());
        std::fs::write(
            &graph,
            b"GGUF\x03\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00",
        )
        .unwrap();

        let mut rt = EmbeddingRuntime::new(EmbeddingConfig {
            model_path: graph,
            pooling: Pooling::InGraph,
            ..Default::default()
        });
        match rt.load() {
            Err(error @ EmbeddingError::InvalidModelFile { .. }) => {
                let message = error.to_string();
                assert!(
                    message.contains("ONNX") && message.contains("GGUF"),
                    "{message}"
                );
            }
            other => panic!("expected InvalidModelFile, got {other:?}"),
        }
    }

    /// A model path that names no ONNX graph is refused as an invalid model before the
    /// file is opened — whether or not it is there, which `ModelNotFound` for the absent
    /// one would disprove — and nothing is installed.
    #[test]
    fn load_refuses_a_model_path_that_names_no_onnx_graph_before_reading_it() {
        let dir = TempDir::new("not_onnx_path");
        let present = dir.path().join("model.gguf");
        std::fs::write(&present, b"GGUF\x03\x00\x00\x00").unwrap();
        for model_path in [
            present,
            dir.path().join("absent.gguf"),
            dir.path().join("model"),
        ] {
            let mut rt = EmbeddingRuntime::new(EmbeddingConfig {
                model_path: model_path.clone(),
                ..Default::default()
            });
            match rt.load() {
                Err(EmbeddingError::InvalidModelFile { path, reason }) => {
                    assert_eq!(path, model_path.display().to_string());
                    assert!(reason.contains("does not end in .onnx"), "{reason}");
                }
                other => panic!(
                    "{} must be refused by its name, got {other:?}",
                    model_path.display()
                ),
            }
            assert!(!rt.is_loaded());
            assert!(rt.model_checksum().is_none());
        }
    }

    /// The defaults are the production model's, as its identity file declares it, so a
    /// default configuration is one the official artifact can be opened with — and the
    /// default model path names an ONNX graph.
    #[test]
    fn defaults_are_the_production_identity() {
        let identity: serde_json::Value = serde_json::from_str(include_str!(
            "../../config/models/meivin-round2-onnx/model.json"
        ))
        .unwrap();
        let config = EmbeddingConfig::default();
        assert_eq!(
            u64::from(config.embedding_dim),
            identity["embedding_dim"].as_u64().unwrap()
        );
        assert_eq!(
            config.max_tokens as u64,
            identity["max_tokens"].as_u64().unwrap()
        );
        assert_eq!(
            config.pooling.as_str(),
            identity["pooling"].as_str().unwrap()
        );
        assert!(crate::semantic::model_package::names_an_onnx_graph(
            &config.model_path
        ));
        assert!(config.validate().is_ok());
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

        // More texts than batch_size (4), so several backend calls are made.
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
    /// poisoned vector would be stored and then dropped at search time.
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
            // Squares overflow f32: the norm becomes `inf` and the vector would
            // normalize to all zeros.
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

    /// The vector exactly on the threshold: with `<` here and `>` in
    /// `l2_normalize` it was neither rejected nor normalized, and entered the index
    /// scoring as its own magnitude instead of as a cosine.
    #[test]
    fn a_vector_exactly_at_the_minimum_norm_is_rejected() {
        let mut boundary = vec![MIN_VECTOR_NORM, 0.0, 0.0, 0.0];
        assert_eq!(
            boundary.iter().map(|x| x * x).sum::<f32>().sqrt(),
            MIN_VECTOR_NORM,
            "the fixture must sit exactly on the threshold for this to mean anything"
        );

        let result = normalize_validated(&mut boundary, 4);
        assert!(
            matches!(result, Err(EmbeddingError::InferenceFailed { .. })),
            "a vector at the threshold must be rejected, not stored unnormalized: \
             {result:?}"
        );

        // Just above it, the result is a unit vector.
        let mut above = vec![MIN_VECTOR_NORM * 1_000.0, 0.0, 0.0, 0.0];
        normalize_validated(&mut above, 4).unwrap();
        assert!((above[0] - 1.0).abs() < 1e-6);
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

    /// The guard is against zero and non-finite, not against "small".
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

    /// The id is what the manifest records, so it is pinned to a literal:
    /// changing it invalidates every index already built by this backend.
    #[test]
    fn backend_is_reported_and_marked_non_semantic() {
        let dir = TempDir::new("backend_kind");
        let rt = loaded_runtime(&dir, 16);

        let backend = rt.backend().expect("loaded");
        assert_eq!(backend.id(), "mock-hash-v1");
        assert!(
            !backend.is_semantic(),
            "the stand-in backend must never claim to be semantic"
        );

        // The same facts through the accessors the manifest and status paths use,
        // which must not disagree with the backend itself.
        assert_eq!(rt.backend_id(), Some("mock-hash-v1"));
        assert!(!rt.backend_is_semantic());

        let unloaded = EmbeddingRuntime::new(EmbeddingConfig::default());
        assert!(unloaded.backend().is_none());
        assert_eq!(unloaded.backend_id(), None);
        assert!(!unloaded.backend_is_semantic());
    }

    // ── the backend contract, exercised with backends that misbehave on purpose ──

    /// A backend whose output is under the test's control: every field is a lever
    /// on something the runtime is supposed to catch, and no honest backend can be
    /// made to produce these.
    struct FakeBackend {
        reported_dim: u32,
        pooling: Pooling,
        max_tokens: usize,
        /// Returned verbatim for every input.
        vector: Vec<f32>,
        /// 1 is correct; 0 and 2 are the bugs the returned-count check exists for.
        vectors_per_input: usize,
    }

    impl FakeBackend {
        fn healthy(dim: u32) -> Self {
            // Not unit length: the runtime must normalize whatever it is handed.
            let mut vector = vec![0.0f32; dim as usize];
            if let Some(first) = vector.first_mut() {
                *first = 7.0;
            }
            if dim > 1 {
                vector[1] = -24.0;
            }
            Self {
                reported_dim: dim,
                pooling: Pooling::InGraph,
                max_tokens: 512,
                vector,
                vectors_per_input: 1,
            }
        }

        fn returning(dim: u32, vector: Vec<f32>) -> Self {
            Self {
                vector,
                ..Self::healthy(dim)
            }
        }

        fn boxed(self) -> Box<dyn EmbeddingBackend> {
            Box::new(self)
        }
    }

    impl EmbeddingBackend for FakeBackend {
        fn id(&self) -> &'static str {
            "fake-test-backend"
        }
        fn is_semantic(&self) -> bool {
            true
        }
        fn dim(&self) -> u32 {
            self.reported_dim
        }
        fn max_tokens(&self) -> usize {
            self.max_tokens
        }
        fn pooling(&self) -> Pooling {
            self.pooling
        }
        fn tokenize(&self, _text: &str) -> Result<Vec<u32>, EmbeddingError> {
            Ok(vec![1, 2, 3])
        }
        fn embed_batch_raw(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
            Ok(
                std::iter::repeat_n(self.vector.clone(), texts.len() * self.vectors_per_input)
                    .collect(),
            )
        }
    }

    /// A real backend reads its width from the model, so a 512-dimension model
    /// behind a 1024 configuration can really happen. Caught at load, because the
    /// alternative is wrong-width vectors reaching the store mid-index.
    #[test]
    fn a_backend_whose_dimension_disagrees_with_the_configuration_is_refused() {
        let config = EmbeddingConfig {
            embedding_dim: 64,
            ..Default::default()
        };

        match EmbeddingRuntime::with_backend(config.clone(), FakeBackend::healthy(32).boxed()) {
            Err(EmbeddingError::DimensionMismatch { expected, actual }) => {
                assert_eq!((expected, actual), (64, 32));
            }
            Err(other) => panic!("expected a dimension mismatch, got {other:?}"),
            Ok(_) => panic!("a backend narrower than the configuration must be refused"),
        }

        // Wider is equally refused.
        assert!(matches!(
            EmbeddingRuntime::with_backend(config.clone(), FakeBackend::healthy(128).boxed()),
            Err(EmbeddingError::DimensionMismatch { .. })
        ));

        // Agreement loads, so the check is not refusing everything.
        let runtime = EmbeddingRuntime::with_backend(config, FakeBackend::healthy(64).boxed())
            .expect("a backend that agrees must load");
        assert!(runtime.is_loaded());
        assert_eq!(runtime.dim(), 64);
    }

    /// A *valid* configuration against a backend that pools otherwise — the case no
    /// configuration-time guard can pre-empt, since nothing about the configuration
    /// is wrong. A backend that mean-pools token states itself behind an `in-graph`
    /// configuration lands here, and this is why [`Pooling::Mean`] must stay a
    /// variant.
    #[test]
    fn a_backend_that_pools_differently_from_the_configuration_is_refused() {
        let config = EmbeddingConfig {
            embedding_dim: 16,
            pooling: Pooling::InGraph,
            ..Default::default()
        };
        let backend = FakeBackend {
            pooling: Pooling::Mean,
            ..FakeBackend::healthy(16)
        };

        match EmbeddingRuntime::with_backend(config, backend.boxed()) {
            Err(EmbeddingError::PoolingMismatch {
                configured, actual, ..
            }) => {
                assert_eq!(configured, "in-graph");
                assert_eq!(actual, "mean");
            }
            Err(other) => panic!("expected a pooling mismatch, got {other:?}"),
            Ok(_) => panic!("a backend that pools differently must be refused"),
        }
    }

    /// Refused *before the model file is opened*, so it cannot reach a backend or a
    /// manifest. The model path here does not exist — a `ModelNotFound` would prove
    /// the check ran too late.
    #[test]
    fn load_refuses_a_pooling_no_backend_implements_before_reading_the_model() {
        let dir = TempDir::new("pooling_unimplemented");
        let absent = dir.path().join("not-even-there.onnx");

        let mut rt = EmbeddingRuntime::new(EmbeddingConfig {
            model_path: absent,
            embedding_dim: 16,
            pooling: Pooling::Mean,
            ..Default::default()
        });

        match rt.load() {
            Err(EmbeddingError::PoolingNotImplemented {
                pooling,
                implemented,
            }) => {
                assert_eq!(pooling, "mean");
                assert!(
                    implemented.contains("in-graph"),
                    "unhelpful reason: {implemented}"
                );
            }
            other => panic!("a pooling nothing performs must be refused, got {other:?}"),
        }
        assert!(!rt.is_loaded(), "a refused backend must not be installed");
        assert!(
            rt.model_checksum().is_none(),
            "nothing may be recorded for a backend that was never installed"
        );
    }

    /// A cap of zero truncates every input to nothing.
    #[test]
    fn a_backend_with_a_zero_token_cap_is_refused() {
        let backend = FakeBackend {
            max_tokens: 0,
            ..FakeBackend::healthy(16)
        };
        assert!(matches!(
            EmbeddingRuntime::with_backend(
                EmbeddingConfig {
                    embedding_dim: 16,
                    ..Default::default()
                },
                backend.boxed()
            ),
            Err(EmbeddingError::LoadFailed { .. })
        ));
    }

    #[test]
    fn a_nonsensical_configuration_is_refused_before_the_model_is_read() {
        let dir = TempDir::new("bad_config");
        let absent = dir.path().join("not-even-there.onnx");

        for config in [
            EmbeddingConfig {
                model_path: absent.clone(),
                embedding_dim: 0,
                ..Default::default()
            },
            EmbeddingConfig {
                model_path: absent,
                max_tokens: 0,
                ..Default::default()
            },
        ] {
            let mut rt = EmbeddingRuntime::new(config);
            // `LoadFailed`, not `ModelNotFound`, proves the file was never touched.
            assert!(matches!(rt.load(), Err(EmbeddingError::LoadFailed { .. })));
        }
    }

    /// A cap past any encoder's context is a configuration nothing can serve, and is
    /// refused as one — before the file is opened, as the paths here do not exist —
    /// naming the ceiling.
    #[test]
    fn validate_bounds_the_token_cap_past_any_encoder() {
        let dir = TempDir::new("cap_ceiling");
        let onnx = |max_tokens: usize| EmbeddingConfig {
            model_path: dir.path().join("absent.onnx"),
            pooling: Pooling::InGraph,
            max_tokens,
            ..Default::default()
        };
        for cap in [
            crate::semantic::backend::ONNX_MAX_TOKENS_CEILING + 1,
            u32::MAX as usize,
            usize::MAX,
        ] {
            match onnx(cap).validate() {
                Err(EmbeddingError::LoadFailed { reason }) => assert!(
                    reason.contains(&format!("max_tokens is {cap}"))
                        && reason.contains(
                            &crate::semantic::backend::ONNX_MAX_TOKENS_CEILING.to_string()
                        ),
                    "{reason}"
                ),
                other => panic!("a cap of {cap} must be refused, got {other:?}"),
            }
            let mut rt = EmbeddingRuntime::new(onnx(cap));
            assert!(matches!(rt.load(), Err(EmbeddingError::LoadFailed { .. })));
        }
        assert!(onnx(crate::semantic::backend::ONNX_MAX_TOKENS_CEILING)
            .validate()
            .is_ok());
    }

    /// Pins the guard to `EmbeddingConfig::validate` rather than to the order of
    /// steps inside `load`, so every entry point that validates gets it.
    #[test]
    fn validate_refuses_a_pooling_no_backend_implements() {
        let unimplemented = EmbeddingConfig {
            pooling: Pooling::Mean,
            ..Default::default()
        };
        assert!(matches!(
            unimplemented.validate(),
            Err(EmbeddingError::PoolingNotImplemented { .. })
        ));

        // The default must stay valid, or nothing loads at all.
        assert!(EmbeddingConfig::default().validate().is_ok());
    }

    /// The choke point from the other side. `FakeBackend` returns non-unit vectors
    /// on purpose, or this would be indistinguishable from doing nothing.
    #[test]
    fn embed_batch_normalizes_whatever_the_backend_returns() {
        let config = EmbeddingConfig {
            embedding_dim: 4,
            batch_size: 2,
            ..Default::default()
        };
        let rt = EmbeddingRuntime::with_backend(config, FakeBackend::healthy(4).boxed()).unwrap();

        // More inputs than batch_size, so several backend calls happen.
        let vectors = rt.embed_batch(&["a", "b", "c"]).unwrap();
        assert_eq!(vectors.len(), 3);
        for vector in &vectors {
            let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-6,
                "the runtime must normalize a raw vector, got norm {norm}"
            );
        }
    }

    #[test]
    fn embed_batch_rejects_every_unusable_vector_a_backend_can_return() {
        let config = EmbeddingConfig {
            embedding_dim: 4,
            ..Default::default()
        };

        let cases: Vec<(&str, Vec<f32>)> = vec![
            ("a NaN component", vec![f32::NAN, 1.0, 0.0, 0.0]),
            ("an infinite component", vec![f32::INFINITY, 1.0, 0.0, 0.0]),
            ("all zeros", vec![0.0; 4]),
            // Squares overflow f32: the norm becomes `inf` and the vector would
            // normalize to all zeros.
            ("finite but overflowing", vec![1e30; 4]),
            (
                "exactly at the minimum norm",
                vec![MIN_VECTOR_NORM, 0.0, 0.0, 0.0],
            ),
        ];

        for (name, vector) in cases {
            let rt = EmbeddingRuntime::with_backend(
                config.clone(),
                FakeBackend::returning(4, vector).boxed(),
            )
            .unwrap();
            let result = rt.embed_batch(&["שורה כלשהי"]);
            assert!(
                matches!(result, Err(EmbeddingError::InferenceFailed { .. })),
                "{name} must be rejected, got {result:?}"
            );
        }

        // A wrong *length* is diagnosed as the dimension problem it is, even though
        // the reported dim agreed at load time: a backend can be inconsistent with
        // itself, and this is the last line before the store.
        let rt = EmbeddingRuntime::with_backend(
            config,
            FakeBackend::returning(4, vec![1.0, 2.0]).boxed(),
        )
        .unwrap();
        assert!(matches!(
            rt.embed_batch(&["שורה כלשהי"]),
            Err(EmbeddingError::DimensionMismatch {
                expected: 4,
                actual: 2
            })
        ));
    }

    /// Vectors are paired with chunk metadata by position, so a wrong-length batch
    /// would mislabel results rather than lose them.
    #[test]
    fn embed_batch_rejects_a_backend_that_returns_the_wrong_number_of_vectors() {
        let config = EmbeddingConfig {
            embedding_dim: 4,
            batch_size: 8,
            ..Default::default()
        };

        for (name, per_input) in [("too few", 0usize), ("too many", 2)] {
            let backend = FakeBackend {
                vectors_per_input: per_input,
                ..FakeBackend::healthy(4)
            };
            let rt = EmbeddingRuntime::with_backend(config.clone(), backend.boxed()).unwrap();

            match rt.embed_batch(&["א", "ב", "ג"]) {
                Err(EmbeddingError::InferenceFailed { reason }) => {
                    assert!(
                        reason.contains("vectors for"),
                        "{name}: unhelpful reason: {reason}"
                    );
                }
                other => panic!("{name} must be rejected, got {other:?}"),
            }
        }
    }

    /// The request reaches the backend, and the backend's effective answer is what
    /// the runtime reports.
    #[test]
    fn the_requested_token_cap_reaches_the_backend_and_the_backends_answer_wins() {
        let dir = TempDir::new("max_tokens");
        let model = mock::write_stub_onnx_package(dir.path());

        let mut rt = EmbeddingRuntime::new(EmbeddingConfig {
            model_path: model,
            embedding_dim: 16,
            max_tokens: 128,
            ..Default::default()
        });
        // Before loading, the request is all there is to report.
        assert_eq!(rt.max_tokens(), 128);
        rt.load().unwrap();
        assert_eq!(
            rt.max_tokens(),
            128,
            "the selected backend must have been built with the requested cap"
        );

        // A backend that clamps is reported as it is, not as the caller asked.
        let clamped = FakeBackend {
            max_tokens: 32,
            ..FakeBackend::healthy(16)
        };
        let rt = EmbeddingRuntime::with_backend(
            EmbeddingConfig {
                embedding_dim: 16,
                max_tokens: 4096,
                ..Default::default()
            },
            clamped.boxed(),
        )
        .unwrap();
        assert_eq!(rt.max_tokens(), 32);
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
