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

/// How much of the file's start is read to parse the metadata and tensor
/// descriptors.
///
/// The descriptor region of a real model is a few megabytes at most — the
/// tokenizer vocabulary dominates it. This is the ceiling on believing the
/// declared sizes: a supported GGUF whose descriptor region exceeds it is
/// rejected. Silently accepting a file whose structure was not validated would
/// turn the validator into a magic-byte check.
const GGUF_MAX_DESCRIPTOR_REGION_BYTES: u64 = 64 << 20;

/// Default tensor-data alignment when the file declares none.
///
/// GGUF pads between the descriptors and the tensor data to `general.alignment`
/// bytes, 32 by default.
const GGUF_DEFAULT_ALIGNMENT: u64 = 32;

/// Read buffer for hashing the model file. Large enough that hashing a
/// multi-hundred-megabyte model is bound by the hash, not by syscalls.
const HASH_BUFFER_BYTES: usize = 1 << 20;

/// At or below this L2 norm a vector carries no direction and cannot be
/// normalized.
///
/// One threshold for the whole crate, used with the same `<=` comparison at every
/// layer — the embedding runtime, the store's own guard, and the query path.
/// Two copies would drift, and a vector that one layer rejects while another
/// normalizes is exactly the record that exists but can never be found.
pub(crate) const MIN_VECTOR_NORM: f32 = 1e-12;

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
    // `<=`, matching the `>` in `l2_normalize` and the store's own guard. At
    // exactly the threshold a `<` test rejected nothing and `l2_normalize`
    // normalized nothing, so the one vector on the boundary passed through
    // unnormalized — scoring as its own magnitude rather than as a cosine.
    if norm <= MIN_VECTOR_NORM {
        return Err(EmbeddingError::InferenceFailed {
            reason: format!(
                "vector norm {norm} is at or below the minimum {MIN_VECTOR_NORM}; it has no \
                 usable direction"
            ),
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
/// # What is checked
///
/// 1. **The header, before anything else is read.** Magic, version, and both
///    declared counts. A file that is not GGUF at all is rejected after 24 bytes
///    rather than after hashing however many gigabytes it happens to be.
/// 2. **The descriptor region is parsed**: every metadata key/value pair and
///    every tensor descriptor, which yields where the tensor data starts and,
///    per tensor, the offset it claims within that data.
/// 3. **The file is long enough to hold what its own descriptors describe.** Each
///    tensor's size is bounded below by one bit per element — true of every ggml
///    type, including the most aggressive ternary quantizations — so this is a
///    real lower bound rather than a guess.
///
/// # What is not
///
/// The bound is a lower bound. Deriving each tensor's *exact* length needs the
/// ggml type table with the block size of every quantization, and getting one
/// entry of that wrong rejects a valid model — a worse failure than accepting an
/// invalid one, since it makes the feature unavailable rather than late-failing
/// with a clear error. So a download cut off inside the final tensor can still
/// pass. What cannot pass: a placeholder, another format that begins with
/// `GGUF`, a header-only stub, or a download that stopped anywhere before the
/// last tensor's data.
///
/// If the descriptors cannot be parsed — an unknown value type from a future
/// spec revision, or a descriptor region larger than this code is willing to read
/// — the structural check is skipped and only the coarse floor applies. Failing there would mean rejecting a model this
/// crate merely failed to understand.
///
/// **This is not download verification.** A checksum computed from the file
/// cannot attest to the file; only comparing it against a published SHA-256 can,
/// which belongs with model distribution (roadmap P2/P9). What this checksum is
/// for is detecting that the bytes behind a model path *changed* between
/// sessions, which would silently invalidate every stored vector.
///
/// One pass over the file: a multi-hundred-megabyte model is read once, not once
/// to validate and again to hash.
pub fn validate_and_checksum_gguf(path: &Path) -> Result<String, EmbeddingError> {
    let invalid = |reason: String| EmbeddingError::InvalidModelFile {
        path: path.display().to_string(),
        reason,
    };
    let unreadable = |e: std::io::Error| EmbeddingError::LoadFailed {
        reason: format!("cannot read model file {}: {e}", path.display()),
    };

    let file = std::fs::File::open(path).map_err(|e| EmbeddingError::LoadFailed {
        reason: format!("cannot open model file {}: {e}", path.display()),
    })?;
    let mut reader = HashingReader::new(file);

    // ── 1. the header, on its own ──
    let mut header = [0u8; GGUF_HEADER_BYTES];
    if let Err(e) = reader.read_exact_hashed(&mut header) {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            return Err(invalid(format!(
                "file is {} bytes; a GGUF header needs {GGUF_HEADER_BYTES}",
                reader.consumed()
            )));
        }
        return Err(unreadable(e));
    }

    if header[..4] != GGUF_MAGIC {
        return Err(invalid(format!(
            "expected magic {:?}, found {:?}",
            String::from_utf8_lossy(&GGUF_MAGIC),
            String::from_utf8_lossy(&header[..4])
        )));
    }

    let version = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
    if !GGUF_SUPPORTED_VERSIONS.contains(&version) {
        return Err(invalid(format!(
            "unsupported GGUF version {version} (supported: {}..={})",
            GGUF_SUPPORTED_VERSIONS.start(),
            GGUF_SUPPORTED_VERSIONS.end()
        )));
    }

    let field = |offset: usize| -> u64 {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&header[offset..offset + 8]);
        u64::from_le_bytes(bytes)
    };
    let header = GgufHeader {
        tensor_count: field(8),
        metadata_kv_count: field(16),
    };
    if header.tensor_count > GGUF_MAX_DECLARED_COUNT
        || header.metadata_kv_count > GGUF_MAX_DECLARED_COUNT
    {
        return Err(invalid(format!(
            "implausible header counts (tensors: {}, metadata: {}); the file is \
             probably not GGUF or is corrupt",
            header.tensor_count, header.metadata_kv_count
        )));
    }
    if header.tensor_count == 0 {
        return Err(invalid(
            "the GGUF declares no tensors; it is a container header, not an embedding model"
                .to_string(),
        ));
    }

    // ── 2. the descriptors, hashed as they are parsed ──
    let layout = read_gguf_layout(&mut reader, &header);
    let parsed = match layout {
        LayoutOutcome::Parsed(parsed) => parsed,
        LayoutOutcome::Truncated { at, wanted } => {
            return Err(invalid(format!(
                "the file ends inside its own descriptors — {wanted} at byte {at}; the \
                 download is incomplete"
            )));
        }
        LayoutOutcome::Unparsed(reason) => {
            return Err(invalid(format!(
                "the GGUF descriptor table is invalid or unsupported: {reason}"
            )));
        }
    };

    // ── 3. the rest of the file, hashed ──
    let (checksum, total_bytes) = reader.finish().map_err(unreadable)?;

    let required = parsed.data_start.saturating_add(parsed.min_data_bytes);
    if total_bytes < required {
        return Err(invalid(format!(
            "file is {total_bytes} bytes, but its {} tensor descriptor(s) place \
             data up to at least byte {required} (tensor data starts at \
             {}) — the download is incomplete",
            header.tensor_count, parsed.data_start
        )));
    }

    Ok(checksum)
}

/// The counts a GGUF header declares.
struct GgufHeader {
    tensor_count: u64,
    metadata_kv_count: u64,
}

/// Where the descriptors say the tensor data lives.
struct GgufLayout {
    /// First byte of the tensor data blob, after alignment padding.
    data_start: u64,
    /// Highest `tensor offset + lower bound on that tensor's size`, relative to
    /// [`Self::data_start`].
    min_data_bytes: u64,
}

/// Result of parsing the descriptor region.
enum LayoutOutcome {
    Parsed(GgufLayout),
    /// The file ended mid-descriptor: provably incomplete.
    Truncated {
        at: u64,
        wanted: &'static str,
    },
    /// Invalid under the supported schema, or beyond this validator's explicit
    /// resource limits. In either case the model is not safe to accept.
    Unparsed(String),
}

/// Parse the metadata and tensor descriptors, hashing every byte on the way
/// through.
///
/// Reads strictly forward and never seeks, so the caller's single pass over the
/// file is preserved.
fn read_gguf_layout(reader: &mut HashingReader, header: &GgufHeader) -> LayoutOutcome {
    const MAX_METADATA_KEY_BYTES: u64 = u16::MAX as u64;
    const MAX_TENSOR_NAME_BYTES: u64 = 64;
    /// GGUF tensors have at most four dimensions.
    const MAX_TENSOR_DIMS: u32 = 4;
    const ALIGNMENT_KEY: &[u8] = b"general.alignment";

    macro_rules! read {
        ($call:expr, $what:literal) => {
            match $call {
                Ok(value) => value,
                Err(ReadError::Eof { at }) => return LayoutOutcome::Truncated { at, wanted: $what },
                Err(ReadError::Io(e)) => {
                    return LayoutOutcome::Unparsed(format!("read failed: {e}"))
                }
            }
        };
    }

    let mut alignment = GGUF_DEFAULT_ALIGNMENT;

    for index in 0..header.metadata_kv_count {
        if reader.consumed() > GGUF_MAX_DESCRIPTOR_REGION_BYTES {
            return LayoutOutcome::Unparsed(format!(
                "metadata exceeds {GGUF_MAX_DESCRIPTOR_REGION_BYTES} bytes at entry {index}"
            ));
        }

        let key_len = read!(reader.read_u64(), "a metadata key length");
        if key_len > MAX_METADATA_KEY_BYTES {
            return LayoutOutcome::Unparsed(format!("metadata key {index} claims {key_len} bytes"));
        }
        // Keys are short enough to keep; the alignment is read from one of them.
        let key = read!(reader.read_bytes(key_len), "a metadata key");
        let value_type = read!(reader.read_u32(), "a metadata value type");

        if key == ALIGNMENT_KEY {
            if value_type != GgufValueType::UInt32 as u32 {
                return LayoutOutcome::Unparsed("general.alignment is not a uint32".to_string());
            }
            let declared = read!(reader.read_u32(), "the declared alignment") as u64;
            // The GGUF contract requires a multiple of eight, not a power of two.
            if declared < 8 || declared % 8 != 0 {
                return LayoutOutcome::Unparsed(format!(
                    "general.alignment is {declared}, expected a multiple of 8"
                ));
            }
            alignment = declared;
            continue;
        }

        match skip_gguf_value(reader, value_type, 0) {
            SkipOutcome::Done => {}
            SkipOutcome::Truncated { at, wanted } => {
                return LayoutOutcome::Truncated { at, wanted }
            }
            SkipOutcome::Unparsed(reason) => return LayoutOutcome::Unparsed(reason),
        }
    }

    let mut min_data_bytes = 0u64;
    for index in 0..header.tensor_count {
        if reader.consumed() > GGUF_MAX_DESCRIPTOR_REGION_BYTES {
            return LayoutOutcome::Unparsed(format!(
                "tensor descriptors exceed {GGUF_MAX_DESCRIPTOR_REGION_BYTES} bytes at \
                 tensor {index}"
            ));
        }

        let name_len = read!(reader.read_u64(), "a tensor name length");
        if name_len > MAX_TENSOR_NAME_BYTES {
            return LayoutOutcome::Unparsed(format!("tensor {index} name claims {name_len} bytes"));
        }
        read!(reader.skip(name_len), "a tensor name");

        let dim_count = read!(reader.read_u32(), "a tensor dimension count");
        if dim_count == 0 || dim_count > MAX_TENSOR_DIMS {
            return LayoutOutcome::Unparsed(format!(
                "tensor {index} claims {dim_count} dimensions"
            ));
        }
        let mut elements = 1u64;
        for _ in 0..dim_count {
            let extent = read!(reader.read_u64(), "a tensor dimension");
            elements = elements.saturating_mul(extent.max(1));
        }
        let _ggml_type = read!(reader.read_u32(), "a tensor type");
        let offset = read!(reader.read_u64(), "a tensor data offset");
        if offset % alignment != 0 {
            return LayoutOutcome::Unparsed(format!(
                "tensor {index} offset {offset} is not aligned to {alignment} bytes"
            ));
        }

        // One bit per element. Every ggml type stores more than that — the most
        // aggressive quantizations in existence sit around 1.6 bits per weight —
        // so this cannot over-reject, whatever type table a future model uses.
        let floor = elements.div_ceil(8).max(1);
        min_data_bytes = min_data_bytes.max(offset.saturating_add(floor));
    }

    let descriptors_end = reader.consumed();
    let data_start = descriptors_end.next_multiple_of(alignment);

    LayoutOutcome::Parsed(GgufLayout {
        data_start,
        min_data_bytes,
    })
}

/// GGUF metadata value types, in spec order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
enum GgufValueType {
    UInt8 = 0,
    Int8 = 1,
    UInt16 = 2,
    Int16 = 3,
    UInt32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    UInt64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GgufValueType {
    fn from_raw(raw: u32) -> Option<Self> {
        Some(match raw {
            0 => Self::UInt8,
            1 => Self::Int8,
            2 => Self::UInt16,
            3 => Self::Int16,
            4 => Self::UInt32,
            5 => Self::Int32,
            6 => Self::Float32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::UInt64,
            11 => Self::Int64,
            12 => Self::Float64,
            _ => return None,
        })
    }

    /// Fixed encoded width, or `None` for the variable-length types.
    fn fixed_width(self) -> Option<u64> {
        Some(match self {
            Self::UInt8 | Self::Int8 | Self::Bool => 1,
            Self::UInt16 | Self::Int16 => 2,
            Self::UInt32 | Self::Int32 | Self::Float32 => 4,
            Self::UInt64 | Self::Int64 | Self::Float64 => 8,
            Self::String | Self::Array => return None,
        })
    }
}

/// Result of skipping one metadata value.
enum SkipOutcome {
    Done,
    Truncated { at: u64, wanted: &'static str },
    Unparsed(String),
}

/// Skip one metadata value, whatever its type.
///
/// `depth` guards the array-of-arrays case: the spec permits nesting, and a
/// corrupt file could describe it without end.
fn skip_gguf_value(reader: &mut HashingReader, raw_type: u32, depth: u32) -> SkipOutcome {
    const MAX_NESTING: u32 = 4;
    const MAX_ARRAY_LEN: u64 = 1 << 28;
    const MAX_STRING_BYTES: u64 = 1 << 26;

    if depth > MAX_NESTING {
        return SkipOutcome::Unparsed(format!("metadata arrays nested deeper than {MAX_NESTING}"));
    }

    let Some(value_type) = GgufValueType::from_raw(raw_type) else {
        // A type this parser does not know. Later bytes cannot be located, so the
        // structural check is abandoned rather than the file condemned.
        return SkipOutcome::Unparsed(format!("unknown metadata value type {raw_type}"));
    };

    macro_rules! read {
        ($call:expr, $what:literal) => {
            match $call {
                Ok(value) => value,
                Err(ReadError::Eof { at }) => return SkipOutcome::Truncated { at, wanted: $what },
                Err(ReadError::Io(e)) => return SkipOutcome::Unparsed(format!("read failed: {e}")),
            }
        };
    }

    if let Some(width) = value_type.fixed_width() {
        read!(reader.skip(width), "a metadata value");
        return SkipOutcome::Done;
    }

    match value_type {
        GgufValueType::String => {
            let len = read!(reader.read_u64(), "a metadata string length");
            if len > MAX_STRING_BYTES {
                return SkipOutcome::Unparsed(format!("metadata string claims {len} bytes"));
            }
            read!(reader.skip(len), "a metadata string");
            SkipOutcome::Done
        }
        GgufValueType::Array => {
            let element_type = read!(reader.read_u32(), "a metadata array element type");
            let count = read!(reader.read_u64(), "a metadata array length");
            if count > MAX_ARRAY_LEN {
                return SkipOutcome::Unparsed(format!("metadata array claims {count} elements"));
            }

            // Fixed-width elements are one arithmetic skip rather than `count`
            // round trips: a tokenizer's token-type array has ~150k entries.
            if let Some(width) = GgufValueType::from_raw(element_type).and_then(|t| t.fixed_width())
            {
                read!(
                    reader.skip(count.saturating_mul(width)),
                    "a metadata array body"
                );
                return SkipOutcome::Done;
            }

            for _ in 0..count {
                match skip_gguf_value(reader, element_type, depth + 1) {
                    SkipOutcome::Done => {}
                    other => return other,
                }
            }
            SkipOutcome::Done
        }
        _ => unreachable!("every other type has a fixed width"),
    }
}

/// A read failure, distinguishing "the file ended" from "the read failed".
///
/// The difference decides whether a model is rejected: hitting EOF inside a
/// structure the file itself declared is proof of truncation, whereas an I/O
/// error is a fact about the disk.
enum ReadError {
    Eof { at: u64 },
    Io(std::io::Error),
}

/// Buffered forward reader that hashes everything it passes over.
///
/// The point is the single pass: validating and checksumming a model that may be
/// hundreds of megabytes must not read it twice.
struct HashingReader {
    inner: std::io::BufReader<std::fs::File>,
    hasher: sha2::Sha256,
    consumed: u64,
}

impl HashingReader {
    fn new(file: std::fs::File) -> Self {
        use sha2::Digest;
        Self {
            inner: std::io::BufReader::with_capacity(HASH_BUFFER_BYTES, file),
            hasher: sha2::Sha256::new(),
            consumed: 0,
        }
    }

    /// Bytes read — and therefore hashed — so far.
    fn consumed(&self) -> u64 {
        self.consumed
    }

    fn read_exact_hashed(&mut self, buf: &mut [u8]) -> std::io::Result<()> {
        use sha2::Digest;
        self.inner.read_exact(buf)?;
        self.hasher.update(&*buf);
        self.consumed += buf.len() as u64;
        Ok(())
    }

    fn fill(&mut self, buf: &mut [u8]) -> Result<(), ReadError> {
        match self.read_exact_hashed(buf) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                Err(ReadError::Eof { at: self.consumed })
            }
            Err(e) => Err(ReadError::Io(e)),
        }
    }

    fn read_u32(&mut self) -> Result<u32, ReadError> {
        let mut bytes = [0u8; 4];
        self.fill(&mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_u64(&mut self) -> Result<u64, ReadError> {
        let mut bytes = [0u8; 8];
        self.fill(&mut bytes)?;
        Ok(u64::from_le_bytes(bytes))
    }

    /// Read `len` bytes and return them. For short fields only — the caller
    /// bounds `len` first.
    fn read_bytes(&mut self, len: u64) -> Result<Vec<u8>, ReadError> {
        let mut bytes = vec![0u8; len as usize];
        self.fill(&mut bytes)?;
        Ok(bytes)
    }

    /// Skip `len` bytes, hashing them.
    fn skip(&mut self, mut len: u64) -> Result<(), ReadError> {
        const CHUNK: usize = 64 << 10;
        let mut scratch = vec![0u8; (len.min(CHUNK as u64)) as usize];
        while len > 0 {
            let wanted = len.min(scratch.len() as u64) as usize;
            self.fill(&mut scratch[..wanted])?;
            len -= wanted as u64;
        }
        Ok(())
    }

    /// Hash whatever is left and return the digest with the total byte count.
    fn finish(mut self) -> std::io::Result<(String, u64)> {
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

    /// Write a minimal structurally valid GGUF with one scalar F32 tensor.
    ///
    /// It is only a container fixture, not a usable embedding model, but it makes
    /// tests exercise descriptor, alignment and payload validation rather than a
    /// magic-byte placeholder.
    pub fn write_stub_gguf(path: &std::path::Path, version: u32) -> std::io::Result<()> {
        let mut bytes = Vec::with_capacity(68);
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&version.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
        bytes.extend_from_slice(&0u64.to_le_bytes()); // metadata_kv_count

        bytes.extend_from_slice(&1u64.to_le_bytes()); // name length
        bytes.push(b'x');
        bytes.extend_from_slice(&1u32.to_le_bytes()); // one dimension
        bytes.extend_from_slice(&1u64.to_le_bytes()); // one element
        bytes.extend_from_slice(&0u32.to_le_bytes()); // F32
        bytes.extend_from_slice(&0u64.to_le_bytes()); // aligned data offset
        while bytes.len() % super::GGUF_DEFAULT_ALIGNMENT as usize != 0 {
            bytes.push(0);
        }
        bytes.extend_from_slice(&0f32.to_le_bytes());
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
    fn an_empty_container_is_not_an_embedding_model() {
        let dir = TempDir::new("empty_container");
        let model = dir.path().join("empty.gguf");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        std::fs::write(&model, bytes).unwrap();

        assert!(matches!(
            validate_and_checksum_gguf(&model),
            Err(EmbeddingError::InvalidModelFile { .. })
        ));
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

        // Padding arbitrary bytes until a guessed size floor is not enough:
        // descriptors themselves must parse.
        let mut padded_garbage = header(291, 24);
        padded_garbage.resize(16_000, 0u8);
        std::fs::write(&model, &padded_garbage).unwrap();
        assert!(matches!(
            validate_and_checksum_gguf(&model),
            Err(EmbeddingError::InvalidModelFile { .. })
        ));
    }

    /// Build a structurally valid GGUF: header, three metadata entries covering
    /// the variable-length types (a string, an array of strings, a `u32`), one
    /// tensor descriptor, alignment padding, then `data_bytes` of tensor data.
    ///
    /// The point is a file whose own descriptors let the validator prove whether
    /// its payload is present.
    fn gguf_with_one_tensor(dims: &[u64], data_bytes: usize) -> Vec<u8> {
        gguf_with_one_tensor_layout(dims, data_bytes, 32, 0)
    }

    fn gguf_with_one_tensor_layout(
        dims: &[u64],
        data_bytes: usize,
        alignment: u32,
        tensor_offset: u64,
    ) -> Vec<u8> {
        fn push_string(bytes: &mut Vec<u8>, value: &str) {
            bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
            bytes.extend_from_slice(value.as_bytes());
        }

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
        bytes.extend_from_slice(&3u64.to_le_bytes()); // metadata_kv_count

        // A plain string value.
        push_string(&mut bytes, "general.architecture");
        bytes.extend_from_slice(&8u32.to_le_bytes()); // String
        push_string(&mut bytes, "bert");

        // An array of strings — the shape a tokenizer vocabulary takes.
        push_string(&mut bytes, "tokenizer.ggml.tokens");
        bytes.extend_from_slice(&9u32.to_le_bytes()); // Array
        bytes.extend_from_slice(&8u32.to_le_bytes()); // of String
        bytes.extend_from_slice(&2u64.to_le_bytes()); // two of them
        push_string(&mut bytes, "אלף");
        push_string(&mut bytes, "בית");

        // The alignment the padding below uses.
        push_string(&mut bytes, "general.alignment");
        bytes.extend_from_slice(&4u32.to_le_bytes()); // UInt32
        bytes.extend_from_slice(&alignment.to_le_bytes());

        // One tensor at the requested offset in the data blob.
        push_string(&mut bytes, "blk.0.attn_q.weight");
        bytes.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for extent in dims {
            bytes.extend_from_slice(&extent.to_le_bytes());
        }
        bytes.extend_from_slice(&0u32.to_le_bytes()); // ggml type
        bytes.extend_from_slice(&tensor_offset.to_le_bytes());

        while bytes.len() % alignment as usize != 0 {
            bytes.push(0);
        }
        bytes.resize(bytes.len() + tensor_offset as usize + data_bytes, 0);
        bytes
    }

    /// The case a size floor cannot see: the descriptors are all there, and the
    /// weights they describe are not. A download interrupted after the tensor
    /// table looks perfectly well-formed until its own numbers are checked.
    #[test]
    fn a_download_cut_off_inside_the_weights_is_rejected() {
        let dir = TempDir::new("truncated_weights");
        let model = dir.path().join("cut_short.gguf");

        // 1024×1024 elements need at least 131072 bytes at one bit each. This file
        // carries a token amount of data and would sail past any floor derived
        // from the header counts alone (one tensor, three metadata entries).
        std::fs::write(&model, gguf_with_one_tensor(&[1024, 1024], 1_024)).unwrap();
        match validate_and_checksum_gguf(&model) {
            Err(EmbeddingError::InvalidModelFile { reason, .. }) => {
                assert!(reason.contains("incomplete"), "unhelpful reason: {reason}");
            }
            other => panic!("a download cut off inside the weights must be rejected: {other:?}"),
        }

        // With the data present it is accepted, so the bound is not simply
        // rejecting everything.
        std::fs::write(&model, gguf_with_one_tensor(&[1024, 1024], 131_072)).unwrap();
        assert!(
            validate_and_checksum_gguf(&model).is_ok(),
            "a file that holds what its descriptors describe must be accepted"
        );
    }

    /// The lower bound must be genuinely below every real quantization, or it
    /// rejects valid models. One bit per element is; the file above carries
    /// exactly that much and passes, and one byte less does not.
    #[test]
    fn the_size_bound_is_exactly_one_bit_per_element() {
        let dir = TempDir::new("bound_boundary");
        let model = dir.path().join("boundary.gguf");
        let elements = 8_192u64;
        let floor = elements as usize / 8;

        std::fs::write(&model, gguf_with_one_tensor(&[elements], floor)).unwrap();
        assert!(validate_and_checksum_gguf(&model).is_ok());

        std::fs::write(&model, gguf_with_one_tensor(&[elements], floor - 1)).unwrap();
        assert!(matches!(
            validate_and_checksum_gguf(&model),
            Err(EmbeddingError::InvalidModelFile { .. })
        ));
    }

    /// Supported GGUF versions have a closed metadata type enum. An unknown type
    /// means the descriptor table cannot be validated and must not be accepted.
    #[test]
    fn an_unknown_metadata_type_is_rejected() {
        let dir = TempDir::new("future_metadata");
        let model = dir.path().join("future.gguf");

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes()); // one tensor
        bytes.extend_from_slice(&1u64.to_le_bytes()); // one metadata entry
        bytes.extend_from_slice(&4u64.to_le_bytes());
        bytes.extend_from_slice(b"what");
        bytes.extend_from_slice(&9999u32.to_le_bytes()); // a type from the future
        bytes.resize(256, 0);
        std::fs::write(&model, &bytes).unwrap();

        assert!(matches!(
            validate_and_checksum_gguf(&model),
            Err(EmbeddingError::InvalidModelFile { .. })
        ));
    }

    #[test]
    fn alignment_may_be_any_multiple_of_eight() {
        let dir = TempDir::new("non_power_of_two_alignment");
        let model = dir.path().join("alignment24.gguf");
        std::fs::write(&model, gguf_with_one_tensor_layout(&[8], 1, 24, 0)).unwrap();
        assert!(validate_and_checksum_gguf(&model).is_ok());
    }

    #[test]
    fn tensor_offsets_must_honor_the_declared_alignment() {
        let dir = TempDir::new("misaligned_tensor_offset");
        let model = dir.path().join("misaligned.gguf");
        std::fs::write(&model, gguf_with_one_tensor_layout(&[8], 1, 24, 8)).unwrap();
        assert!(matches!(
            validate_and_checksum_gguf(&model),
            Err(EmbeddingError::InvalidModelFile { .. })
        ));
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

        // Two structurally valid files identical for the first
        // HASH_BUFFER_BYTES + 1 bytes, so only whole-file hashing can distinguish
        // the extra trailing byte.
        mock::write_stub_gguf(&a, 3).unwrap();
        let mut base = std::fs::read(&a).unwrap();
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

    /// The one vector on the boundary. With `<` on one side and `>` on the other
    /// it was neither rejected nor normalized, and went into the index scoring as
    /// its own magnitude instead of as a cosine.
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

        // Just above it, normalization happens and the result is a unit vector.
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
