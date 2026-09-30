//! The embedding backend contract: the trait an inference implementation must
//! satisfy ([`EmbeddingBackend`]), the pooling a backend and the configuration
//! must agree on ([`Pooling`]), and which implementation a build gets
//! ([`select_backend`]).
//!
//! A backend turns text into **raw, unnormalized** vectors and nothing else.
//! [`EmbeddingRuntime::embed_batch`](crate::semantic::embedding::EmbeddingRuntime::embed_batch)
//! is the single choke point for batching, the returned-count check,
//! dimension/finiteness/norm validation and normalization. A backend that
//! normalized or screened its own output would hide a degenerate vector behind a
//! plausible unit norm, and such a vector scores `NaN`, is dropped at search
//! time, and leaves a book recorded as indexed but never findable.
//! [`VectorStore::insert_batch`](crate::semantic::store::VectorStore::insert_batch)
//! re-checks the same invariant against the one shared threshold
//! (`embedding::MIN_VECTOR_NORM`), because it is public and sees vectors this
//! runtime never produced.
//!
//! | build | GGUF model | ONNX model |
//! |---|---|---|
//! | default | `Err(BackendUnavailable)` | `Err(BackendUnavailable)` |
//! | `--features llama-backend` | `LlamaCppBackend` | `Err(BackendUnavailable)` |
//! | `--features onnx-backend` | `Err(BackendUnavailable)` | `OnnxBackend` |
//! | `--features mock-embedding` (and in-crate tests) | `MockHashBackend` | `MockHashBackend` |
//!
//! A real backend's cell means `Ok`, or why that backend cannot serve the model.
//!
//! **The model's format picks the column, not trial and error**: a path ending in
//! `.onnx` is ONNX and every other path is GGUF
//! ([`ModelFormat::of`](crate::semantic::model_package::ModelFormat::of)), and only the
//! candidates serving that format are walked. The stand-in is gated so a release build
//! cannot serve fake vectors; real inference is gated because it links a large native
//! runtime. With a real backend and the stand-in both enabled the real one wins, since
//! `CANDIDATES` is ordered by preference.

use crate::errors::EmbeddingError;
use crate::semantic::embedding::EmbeddingConfig;
use crate::semantic::model_package::ModelFormat;

/// How a model's per-token hidden states are collapsed into one vector.
///
/// Closed rather than a string: pooling decides what the vector *is* and the
/// manifest records it as the index's identity, so a typo used to produce vectors
/// pooled one way while the manifest claimed another.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pooling {
    /// The final token's hidden state — what this crate's target model requires
    /// (`llama.cpp --pooling last`).
    LastToken,
    /// The mean of every token's hidden state. **Representable, not
    /// configurable**: no backend here performs it, so
    /// [`ensure_pooling_is_implemented`] refuses it. It exists because a
    /// single-variant enum could not express a configuration that *disagrees*
    /// with its backend, leaving [`EmbeddingError::PoolingMismatch`] and the
    /// load-time agreement check dead code.
    Mean,
    /// Nothing is pooled outside the model: its graph ends in the pooling (and any
    /// projection and normalization after it) and emits the finished sentence vector,
    /// which the backend takes as it is.
    ///
    /// A variant of its own rather than a claim to [`Self::Mean`] even when the graph
    /// happens to mean-pool inside: the manifest records who *performs* the pooling,
    /// and a backend that mean-pooled token states itself would produce different
    /// vectors from the same weights. Configurable — the ONNX backend serves exactly
    /// this — and the one strategy a token-level output can never satisfy.
    InGraph,
}

impl Pooling {
    /// What [`Self::parse`] searches, so a variant left out cannot be read back.
    /// The compiler misses that; `every_variant_is_listed_and_parseable` does not.
    pub const ALL: [Self; 3] = [Self::LastToken, Self::Mean, Self::InGraph];

    /// The exact string persisted in the manifest. These spellings are already on
    /// disk; changing one invalidates every existing index.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LastToken => "last-token",
            Self::Mean => "mean",
            Self::InGraph => "in-graph",
        }
    }

    /// Matched **exactly** — not case-insensitively, not trimmed. Accepting
    /// `"Last-Token"` would persist that spelling, and a later canonically
    /// spelled configuration would read it as a *different* pooling: semantic
    /// search disabled and a full re-index demanded over capitalization.
    pub fn parse(value: &str) -> Result<Self, EmbeddingError> {
        Self::ALL
            .into_iter()
            .find(|candidate| candidate.as_str() == value)
            .ok_or_else(|| EmbeddingError::UnknownPooling {
                found: value.to_string(),
                supported: Self::ALL
                    .iter()
                    .map(|p| p.as_str())
                    .collect::<Vec<_>>()
                    .join(", "),
            })
    }
}

impl std::fmt::Display for Pooling {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for Pooling {
    type Err = EmbeddingError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::parse(value)
    }
}

/// One inference implementation behind
/// [`EmbeddingRuntime`](crate::semantic::embedding::EmbeddingRuntime).
///
/// `Send + Sync` with `&self` methods, because the coordinator serves searches
/// through a shared reference: `&mut self` would force every query to take the
/// write lock and serialize search behind indexing. A backend needs its own
/// interior concurrency (a pool of contexts, or a worker per context) — one
/// context behind one `Mutex` satisfies the bound and reintroduces the same
/// serialization.
pub trait EmbeddingBackend: Send + Sync {
    /// Stable identifier, persisted in the manifest as `embedding_backend`. A
    /// change invalidates every stored vector, so it carries a version suffix.
    fn id(&self) -> &'static str;

    /// Whether the vectors carry semantic meaning at all. `false` for the hash
    /// stand-in, which nothing downstream could detect from the vectors alone.
    fn is_semantic(&self) -> bool;

    /// Dimensionality of the vectors this backend produces. Reported here because
    /// a real backend reads it from the model file and can disagree with the
    /// configuration; the runtime refuses that at load rather than letting
    /// wrong-width vectors reach the store mid-index.
    fn dim(&self) -> u32;

    /// The token cap this backend applies to a single input: the **total** sequence
    /// length, counting every special token the backend adds around the text —
    /// llama.cpp's appended EOS, a BERT tokenizer's `[CLS]` and `[SEP]`.
    ///
    /// A contract the *backend* implements, not something this layer enforces:
    /// the runtime has no tokenizer, and `Chunker`'s character limit is a
    /// different limit. The real backend honours it; the stand-in cannot and
    /// echoes the request. The three rules:
    ///
    /// * truncate, never fail — an over-long line must still be indexed;
    /// * keep the tail meaningful — with [`Pooling::LastToken`] the vector *is*
    ///   the final token's state, so keep leading tokens and still append EOS;
    /// * report the effective cap — clamp [`EmbeddingConfig::max_tokens`] to the
    ///   model's trained context length.
    ///
    /// The value is refused if zero at adoption and recorded in the manifest,
    /// which is what makes a change detectable rather than a silent re-embedding.
    fn max_tokens(&self) -> usize;

    /// The pooling this backend performs, compared against the configured one at
    /// load time. A real backend reports what its model requires.
    fn pooling(&self) -> Pooling;

    /// The model's own token ids for `text`, special tokens included. On the trait
    /// because stage 4 proves token-id parity against a reference tokenizer,
    /// which a vector comparison cannot do. A backend without a tokenizer returns
    /// [`EmbeddingError::TokenizationUnsupported`] rather than inventing ids.
    fn tokenize(&self, text: &str) -> Result<Vec<u32>, EmbeddingError>;

    /// Embed one already-sized batch into **raw, unnormalized** vectors:
    ///
    /// * one vector per input, in input order — the runtime pairs them
    ///   positionally with chunk metadata;
    /// * each [`Self::dim`] long;
    /// * not normalized, not screened for `NaN` or zero;
    /// * empty slice in, empty `Vec` out;
    /// * at most [`EmbeddingConfig::batch_size`] inputs.
    fn embed_batch_raw(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError>;
}

/// One backend's capabilities, stated without building it, plus its constructor.
/// The halves travel together because [`implemented_poolings`] needs the
/// capability before a model is loaded, and building a real backend to ask costs
/// a model load.
struct BackendCandidate {
    /// Mirrors what the built backend reports, so a capability can be attributed
    /// in an error message without constructing anything.
    /// `every_candidate_describes_the_backend_it_builds` holds the two together.
    id: &'static str,
    /// The model formats this candidate is asked to serve. [`select_backend`] walks
    /// only the candidates for the model path's format, so a backend never has to
    /// recognize — and never gets to half-load — a file of another format.
    formats: &'static [ModelFormat],
    /// A set, not one value: a real backend reports what the *loaded model*
    /// requires, so one implementation can serve several poolings. This only
    /// decides which configurations are worth attempting.
    poolings: &'static [Pooling],
    /// Builds the backend, or `None` in a build that did not compile it in.
    ///
    /// `None` means exactly one thing: *no such implementation here*. A real
    /// backend needs a third answer — compiled in, the right one, and still
    /// unable to load this model. A bare `Option` collapsed that into `None`, so
    /// [`select_backend`] walked *past* it: with `mock-embedding` also enabled, a
    /// broken model silently answered by hash vectors. `Some(Err(_))` is "mine to
    /// serve, and here is why I could not".
    construct: fn(&EmbeddingConfig) -> Constructed,
}

/// `None`: not in this build. `Some(Err(_))`: mine, and here is why it failed.
type Constructed = Option<Result<Box<dyn EmbeddingBackend>, EmbeddingError>>;

/// Every backend this crate implements, in preference order. **A real backend
/// must come before the stand-in serving the same format**, or enabling
/// `mock-embedding` on top of a real build indexes fake vectors unknowingly.
///
/// Deliberately not feature-gated: the table describes the implementations that
/// *exist*, which is what "a pooling no backend implements" means. Gating it
/// would make a default build reject `"last-token"` as unimplemented — reporting
/// a missing backend as a bad configuration value.
///
/// The stand-in has one row per format, each claiming what that format's real
/// backend claims, so a mock build runs the pooling agreement check for real.
const CANDIDATES: &[BackendCandidate] = &[
    BackendCandidate {
        id: "llama-cpp-qwen3-last-v1",
        formats: &[ModelFormat::Gguf],
        poolings: &[Pooling::LastToken],
        construct: llama_cpp_backend,
    },
    BackendCandidate {
        // `OnnxBackend::ID`, spelled out because this table is not feature-gated.
        id: "onnxruntime-sentence-v1",
        formats: &[ModelFormat::Onnx],
        poolings: &[Pooling::InGraph],
        construct: onnx_backend,
    },
    BackendCandidate {
        id: "mock-hash-v1",
        formats: &[ModelFormat::Gguf],
        poolings: &[Pooling::LastToken],
        construct: mock_hash_backend,
    },
    BackendCandidate {
        id: "mock-hash-v1",
        formats: &[ModelFormat::Onnx],
        poolings: &[Pooling::InGraph],
        construct: mock_hash_backend,
    },
];

/// Every pooling some backend performs, for a model of any format. Ordered by
/// [`Pooling::ALL`] so error messages are stable.
pub fn implemented_poolings() -> Vec<Pooling> {
    poolings_served(None)
}

/// Every pooling a backend performs for a `format` model, ordered by [`Pooling::ALL`].
pub fn implemented_poolings_for(format: ModelFormat) -> Vec<Pooling> {
    poolings_served(Some(format))
}

fn poolings_served(format: Option<ModelFormat>) -> Vec<Pooling> {
    Pooling::ALL
        .into_iter()
        .filter(|strategy| {
            candidates_for(format).any(|candidate| candidate.poolings.contains(strategy))
        })
        .collect()
}

/// The candidates serving `format`, or every candidate for `None`.
fn candidates_for(format: Option<ModelFormat>) -> impl Iterator<Item = &'static BackendCandidate> {
    CANDIDATES
        .iter()
        .filter(move |candidate| format.is_none_or(|format| candidate.formats.contains(&format)))
}

/// Refuse a pooling no backend implements while it is still only a configuration
/// value. `pooling = "mean"` parses, so without this it reached the manifest as
/// the index's identity; correcting the configuration then made the manifest
/// disagree with it, recoverable only by discarding the index.
///
/// Format-blind: it answers whether *any* backend performs `pooling`. A configuration
/// names a model, and [`ensure_pooling_is_implemented_for`] is the check that holds it
/// to that model's format.
///
/// # Errors
///
/// [`EmbeddingError::PoolingNotImplemented`], naming what is implemented.
pub fn ensure_pooling_is_implemented(pooling: Pooling) -> Result<(), EmbeddingError> {
    if implemented_poolings().contains(&pooling) {
        return Ok(());
    }
    Err(EmbeddingError::PoolingNotImplemented {
        pooling: pooling.to_string(),
        implemented: describe_implemented_poolings(None),
    })
}

/// [`ensure_pooling_is_implemented`], for a model of `format`: the check
/// [`EmbeddingConfig::validate`] and the engine's configuration make, because the
/// pooling reaches the manifest before any backend is asked.
///
/// Without it `pooling = "in-graph"` beside a GGUF model validated — an ONNX backend
/// implements it — was written into the index identity, and failed only at load, as a
/// mismatch against a manifest that then outlived the correction: the trap the
/// format-blind check closed for `"mean"`, reopened by a pooling only one format has.
///
/// # Errors
///
/// [`EmbeddingError::PoolingNotImplemented`] for a pooling nothing performs, naming what
/// `format` does implement — for GGUF, the message this crate has always given — and
/// [`EmbeddingError::PoolingNotForFormat`] for one only another format's backends
/// perform, naming both.
pub fn ensure_pooling_is_implemented_for(
    pooling: Pooling,
    format: ModelFormat,
) -> Result<(), EmbeddingError> {
    if implemented_poolings_for(format).contains(&pooling) {
        return Ok(());
    }
    if !implemented_poolings().contains(&pooling) {
        return Err(EmbeddingError::PoolingNotImplemented {
            pooling: pooling.to_string(),
            implemented: describe_implemented_poolings(Some(format)),
        });
    }
    let implemented_elsewhere: Vec<String> = ModelFormat::ALL
        .into_iter()
        .filter(|other| *other != format && implemented_poolings_for(*other).contains(&pooling))
        .map(|other| {
            format!(
                "{other} models ({})",
                backends_performing(pooling, Some(other))
            )
        })
        .collect();
    Err(EmbeddingError::PoolingNotForFormat {
        pooling: pooling.to_string(),
        format: format.to_string(),
        implemented: describe_implemented_poolings(Some(format)),
        implemented_elsewhere: implemented_elsewhere.join("; "),
    })
}

/// Attributed per backend, because "implemented: last-token" reads as a limit of
/// the build while "last-token (mock-hash-v1)" names the implementation.
fn describe_implemented_poolings(format: Option<ModelFormat>) -> String {
    let described: Vec<String> = Pooling::ALL
        .into_iter()
        .filter_map(|strategy| {
            let backends = backends_performing(strategy, format);
            (!backends.is_empty()).then(|| format!("{strategy} ({backends})"))
        })
        .collect();

    if described.is_empty() {
        "none".to_string()
    } else {
        described.join("; ")
    }
}

/// The ids of the backends performing `pooling` for `format`, in table order, each
/// once: the stand-in has a row per format.
fn backends_performing(pooling: Pooling, format: Option<ModelFormat>) -> String {
    let mut ids: Vec<&str> = Vec::new();
    for candidate in candidates_for(format) {
        if candidate.poolings.contains(&pooling) && !ids.contains(&candidate.id) {
            ids.push(candidate.id);
        }
    }
    ids.join(", ")
}

/// Choose the backend this build can offer for `config`, by walking the `CANDIDATES`
/// that serve its model's format rather than through `#[cfg]` blocks inside an
/// inference call.
///
/// `config` is validated here too: this function is public and reachable without
/// [`EmbeddingRuntime::load`](crate::semantic::embedding::EmbeddingRuntime::load),
/// so a direct caller could otherwise get a backend built for `max_tokens: 1`,
/// which embeds every text as a bare special token.
///
/// # Errors
///
/// [`EmbeddingError::BackendUnavailable`] when nothing serving the model's format is
/// compiled in — every default build, the guarantee
/// `tests/production_backend_gate.rs` holds — naming the feature that would serve it.
/// Otherwise whatever the first compiled-in candidate for the format, or
/// [`EmbeddingConfig::validate`], failed with.
pub fn select_backend(
    config: &EmbeddingConfig,
) -> Result<Box<dyn EmbeddingBackend>, EmbeddingError> {
    config.validate()?;
    let format = ModelFormat::of(&config.model_path);

    // `Some(Err(_))` stops the walk just as `Some(Ok(_))` does — see
    // `BackendCandidate::construct`.
    CANDIDATES
        .iter()
        .filter(|candidate| candidate.formats.contains(&format))
        .find_map(|candidate| (candidate.construct)(config))
        .unwrap_or_else(|| {
            // For GGUF this is, byte for byte, the message from before formats existed.
            Err(EmbeddingError::BackendUnavailable {
                reason: format!(
                    "this build has no inference backend compiled in (enable the \
                     `{}` feature for real {format} inference); model file {} \
                     validated but cannot be executed",
                    format.backend_feature(),
                    config.model_path.display()
                ),
            })
        })
}

/// Real GGUF inference, in a build that compiled it in.
///
/// `not(test)` because the in-crate suite drives this module with stub GGUF
/// containers, and a real backend ahead of the stand-in in [`CANDIDATES`] would
/// fail on every one of them. Integration tests link the library without
/// `cfg(test)` and so see the real table.
///
/// `not(target_arch = "arm")` because the llama crates are not dependencies
/// there; such a build takes the `None` arm below.
#[cfg(all(feature = "llama-backend", not(target_arch = "arm"), not(test)))]
fn llama_cpp_backend(config: &EmbeddingConfig) -> Constructed {
    use crate::semantic::llama_backend::{LlamaBackendConfig, LlamaCppBackend};

    // The constructor receives only an `EmbeddingConfig`, so llama.cpp's own
    // knobs come from the environment; typed callers use `LlamaCppBackend::open`.
    Some(LlamaBackendConfig::from_env_for(config).and_then(|tuning| {
        LlamaCppBackend::open(&config.model_path, config.max_tokens, &tuning)
            .map(|backend| Box::new(backend) as Box<dyn EmbeddingBackend>)
    }))
}

/// `None`, not `Some(Err(_))`: without the feature — or on a target the backend
/// is not built for — there is no such implementation at all.
#[cfg(not(all(feature = "llama-backend", not(target_arch = "arm"), not(test))))]
fn llama_cpp_backend(_config: &EmbeddingConfig) -> Constructed {
    None
}

// ── ONNX Runtime: the constructor pair ──────────────────────────────────────────
//
// The two `cfg`s below are exact complements and must stay so, and the `mod
// onnx_backend` line in `semantic/mod.rs` must carry the same feature/target
// condition as the first one. A target restriction goes into all three.

/// Real ONNX inference, in a build that compiled it in.
///
/// `not(test)` for the reason `llama_cpp_backend` has it: the in-crate suite drives
/// this module with stub packages that no real backend could load. Integration tests
/// link the library without `cfg(test)` and so see the real table.
///
/// The package's tokenizer is `tokenizer.json` beside the graph — see
/// [`onnx_tokenizer_path`](crate::semantic::model_package::onnx_tokenizer_path).
#[cfg(all(feature = "onnx-backend", not(test)))]
fn onnx_backend(config: &EmbeddingConfig) -> Constructed {
    use crate::semantic::model_package::onnx_tokenizer_path;
    use crate::semantic::onnx_backend::{OnnxBackend, OnnxBackendConfig};

    // As for llama: the table can pass only an `EmbeddingConfig`, so the runtime's own
    // knobs come from the environment; typed callers use `OnnxBackend::open`.
    Some(OnnxBackendConfig::from_env_for(config).and_then(|tuning| {
        OnnxBackend::open(
            &config.model_path,
            &onnx_tokenizer_path(&config.model_path),
            config.max_tokens,
            config.pooling,
            &tuning,
        )
        .map(|backend| Box::new(backend) as Box<dyn EmbeddingBackend>)
    }))
}

/// `None`, not `Some(Err(_))`: without the feature there is no such implementation.
#[cfg(not(all(feature = "onnx-backend", not(test))))]
fn onnx_backend(_config: &EmbeddingConfig) -> Constructed {
    None
}

// ── end of the ONNX Runtime constructor pair ────────────────────────────────────

#[cfg(any(test, feature = "mock-embedding"))]
fn mock_hash_backend(config: &EmbeddingConfig) -> Constructed {
    Some(Ok(Box::new(MockHashBackend::standing_in_for(
        ModelFormat::of(&config.model_path),
        config.embedding_dim,
        config.max_tokens,
    ))))
}

/// A default build has no stand-in, which is what makes [`select_backend`] fail
/// rather than quietly serve hash vectors.
#[cfg(not(any(test, feature = "mock-embedding")))]
fn mock_hash_backend(_config: &EmbeddingConfig) -> Constructed {
    None
}

/// Deterministic hash-based stand-in. **Not a model** — see the module docs. Its
/// vectors must not change: `tests/hybrid_integration_test.rs` asserts an exact
/// self-match similarity of 1.0.
#[cfg(any(test, feature = "mock-embedding"))]
pub struct MockHashBackend {
    dim: u32,
    max_tokens: usize,
    pooling: Pooling,
}

#[cfg(any(test, feature = "mock-embedding"))]
impl MockHashBackend {
    /// Already recorded in manifests written by this backend; changing it
    /// invalidates those indexes.
    pub const ID: &'static str = "mock-hash-v1";

    /// The stand-in for a GGUF model. `dim` is echoed back from
    /// [`EmbeddingBackend::dim`] so the load-time agreement check passes; a real
    /// backend reads it from the model.
    pub fn new(dim: u32, max_tokens: usize) -> Self {
        Self::standing_in_for(ModelFormat::Gguf, dim, max_tokens)
    }

    /// The stand-in for `format`'s real backend, claiming the pooling that one
    /// performs: [`Pooling::LastToken`] for GGUF, [`Pooling::InGraph`] for ONNX. Its
    /// vectors are the same hash either way.
    pub fn standing_in_for(format: ModelFormat, dim: u32, max_tokens: usize) -> Self {
        let pooling = match format {
            ModelFormat::Gguf => Pooling::LastToken,
            ModelFormat::Onnx => Pooling::InGraph,
        };
        Self {
            dim,
            max_tokens,
            pooling,
        }
    }
}

#[cfg(any(test, feature = "mock-embedding"))]
impl EmbeddingBackend for MockHashBackend {
    fn id(&self) -> &'static str {
        Self::ID
    }

    fn is_semantic(&self) -> bool {
        false
    }

    fn dim(&self) -> u32 {
        self.dim
    }

    /// Echoes the request: no tokenizer to truncate by, no context window to
    /// overflow.
    fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// A word-hash pools nothing. Claiming what the real backend for the model's
    /// format claims — [`Pooling::LastToken`] for GGUF — is what makes the default
    /// configuration load and the agreement check run for real instead of being
    /// trivially satisfied.
    fn pooling(&self) -> Pooling {
        self.pooling
    }

    /// Invented ids would turn stage 4's parity assertion into a comparison
    /// between two fabrications.
    fn tokenize(&self, _text: &str) -> Result<Vec<u32>, EmbeddingError> {
        Err(EmbeddingError::TokenizationUnsupported {
            backend: Self::ID.to_string(),
            reason: "the deterministic hash stand-in feature-hashes whitespace-separated \
                     words and has no tokenizer, so it has no token ids to report"
                .to_string(),
        })
    }

    fn embed_batch_raw(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        // Unnormalized and unchecked on purpose, zero vectors included: the
        // runtime is the only place that rejects those.
        Ok(texts
            .iter()
            .map(|text| crate::semantic::embedding::mock::hash_embedding(text, self.dim))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic::embedding::{mock, EmbeddingRuntime};

    /// Asserted here rather than discovered in the coordinator, hundreds of lines
    /// from the cause.
    #[test]
    fn a_backend_and_the_runtime_holding_it_are_send_and_sync() {
        fn require<T: Send + Sync>() {}
        require::<Box<dyn EmbeddingBackend>>();
        require::<MockHashBackend>();
        require::<EmbeddingRuntime>();
    }

    /// The exhaustive match is the point: a new variant fails to compile here
    /// until it is listed, since `ALL`'s type catches no omission.
    #[test]
    fn every_variant_is_listed_and_parseable() {
        fn listed(strategy: Pooling) -> bool {
            match strategy {
                Pooling::LastToken | Pooling::Mean | Pooling::InGraph => {
                    Pooling::ALL.contains(&strategy)
                }
            }
        }

        for strategy in [Pooling::LastToken, Pooling::Mean, Pooling::InGraph] {
            assert!(listed(strategy), "{strategy} is missing from Pooling::ALL");
            assert!(
                Pooling::parse(strategy.as_str()).is_ok(),
                "{strategy} cannot be parsed back from its own spelling"
            );
        }
    }

    #[test]
    fn pooling_round_trips_through_the_exact_string_the_manifest_stores() {
        assert_eq!(Pooling::LastToken.as_str(), "last-token");
        assert_eq!(Pooling::Mean.as_str(), "mean");
        assert_eq!(Pooling::InGraph.as_str(), "in-graph");

        for strategy in Pooling::ALL {
            let rendered = strategy.as_str();
            assert_eq!(
                Pooling::parse(rendered).unwrap(),
                strategy,
                "{rendered} must parse back to the variant that produced it"
            );
            // What a config reader and the manifest writer actually go through.
            assert_eq!(
                strategy.to_string().parse::<Pooling>().unwrap(),
                strategy,
                "{rendered} must survive Display → FromStr"
            );
        }
    }

    /// Or every existing index reports a pooling mismatch on upgrade.
    #[test]
    fn the_default_pooling_is_the_string_existing_manifests_hold() {
        assert_eq!(EmbeddingConfig::default().pooling, Pooling::LastToken);
        assert_eq!(EmbeddingConfig::default().pooling.as_str(), "last-token");
    }

    #[test]
    fn an_unknown_or_merely_misspelled_pooling_is_refused() {
        for wrong in [
            "",
            " ",
            "last_token",
            "lasttoken",
            "last token",
            "Last-Token",
            "LAST-TOKEN",
            " last-token ",
            "cls",
            "none",
            "in_graph",
            "ingraph",
            "In-Graph",
            "in graph",
        ] {
            match Pooling::parse(wrong) {
                Err(EmbeddingError::UnknownPooling { found, supported }) => {
                    assert_eq!(found, wrong);
                    assert!(
                        supported.contains("last-token"),
                        "the error must name what is accepted, got {supported:?}"
                    );
                }
                other => panic!("{wrong:?} must be refused, got {other:?}"),
            }
        }
    }

    /// `"mean"` stays representable — the agreement check needs a strategy to
    /// disagree with — while being unusable as a configuration value.
    #[test]
    fn a_pooling_no_backend_implements_is_representable_but_refused() {
        assert!(Pooling::ALL.contains(&Pooling::Mean));
        assert_eq!(Pooling::parse("mean").unwrap(), Pooling::Mean);

        // Refused as unimplemented, not as a bad spelling: only one of those
        // diagnoses means "fix the typo".
        assert!(!implemented_poolings().contains(&Pooling::Mean));
        match ensure_pooling_is_implemented(Pooling::Mean) {
            Err(EmbeddingError::PoolingNotImplemented {
                pooling,
                implemented,
            }) => {
                assert_eq!(pooling, "mean");
                assert!(
                    implemented.contains("last-token"),
                    "the error must name what can be used instead, got {implemented:?}"
                );
                assert!(
                    !implemented.contains("mean"),
                    "listing the refused strategy as available is worse than saying \
                     nothing, got {implemented:?}"
                );
            }
            other => panic!("a pooling nothing performs must be refused, got {other:?}"),
        }

        assert_eq!(
            implemented_poolings(),
            vec![Pooling::LastToken, Pooling::InGraph]
        );
        assert!(ensure_pooling_is_implemented(Pooling::LastToken).is_ok());
        assert!(ensure_pooling_is_implemented(Pooling::InGraph).is_ok());
    }

    #[test]
    fn each_format_implements_exactly_the_pooling_its_backends_perform() {
        assert_eq!(
            implemented_poolings_for(ModelFormat::Gguf),
            vec![Pooling::LastToken]
        );
        assert_eq!(
            implemented_poolings_for(ModelFormat::Onnx),
            vec![Pooling::InGraph]
        );
        assert!(ensure_pooling_is_implemented_for(Pooling::LastToken, ModelFormat::Gguf).is_ok());
        assert!(ensure_pooling_is_implemented_for(Pooling::InGraph, ModelFormat::Onnx).is_ok());
    }

    /// A pooling nothing performs keeps the exact message it always had for GGUF: the
    /// list of what GGUF backends implement is what it was before ONNX existed.
    #[test]
    fn a_pooling_nothing_performs_reads_as_it_always_has_for_gguf() {
        match ensure_pooling_is_implemented_for(Pooling::Mean, ModelFormat::Gguf) {
            Err(error @ EmbeddingError::PoolingNotImplemented { .. }) => assert_eq!(
                error.to_string(),
                "No embedding backend implements pooling 'mean' (implemented: last-token \
                 (llama-cpp-qwen3-last-v1, mock-hash-v1))"
            ),
            other => panic!("expected PoolingNotImplemented, got {other:?}"),
        }
        match ensure_pooling_is_implemented_for(Pooling::Mean, ModelFormat::Onnx) {
            Err(EmbeddingError::PoolingNotImplemented { implemented, .. }) => assert_eq!(
                implemented,
                "in-graph (onnxruntime-sentence-v1, mock-hash-v1)"
            ),
            other => panic!("expected PoolingNotImplemented, got {other:?}"),
        }
    }

    /// Both halves of the pairing are named, because either may be the one to change.
    #[test]
    fn a_pooling_the_models_format_does_not_serve_is_refused_by_name() {
        let refused = ensure_pooling_is_implemented_for(Pooling::InGraph, ModelFormat::Gguf)
            .expect_err("a GGUF backend pools last-token");
        let message = refused.to_string();
        match refused {
            EmbeddingError::PoolingNotForFormat {
                pooling,
                format,
                implemented,
                implemented_elsewhere,
            } => {
                assert_eq!(pooling, "in-graph");
                assert_eq!(format, "GGUF");
                assert_eq!(
                    implemented,
                    "last-token (llama-cpp-qwen3-last-v1, mock-hash-v1)"
                );
                assert_eq!(
                    implemented_elsewhere,
                    "ONNX models (onnxruntime-sentence-v1, mock-hash-v1)"
                );
                assert!(
                    message.contains("in-graph") && message.contains("GGUF"),
                    "{message}"
                );
            }
            other => panic!("expected PoolingNotForFormat, got {other:?}"),
        }

        match ensure_pooling_is_implemented_for(Pooling::LastToken, ModelFormat::Onnx) {
            Err(EmbeddingError::PoolingNotForFormat {
                format,
                implemented_elsewhere,
                ..
            }) => {
                assert_eq!(format, "ONNX");
                assert!(implemented_elsewhere.starts_with("GGUF models"));
            }
            other => panic!("expected PoolingNotForFormat, got {other:?}"),
        }
    }

    /// The table states what a backend pools without building it, and two
    /// statements of one fact can drift: an overstated row would accept a
    /// configuration the backend then refuses at load time.
    #[test]
    fn every_candidate_describes_the_backend_it_builds() {
        let mut constructed = 0usize;
        for candidate in CANDIDATES {
            assert!(
                !candidate.poolings.is_empty(),
                "{} declares no pooling, so no configuration could ever select it",
                candidate.id
            );
            assert!(
                !candidate.formats.is_empty(),
                "{} declares no format, so no model could ever reach it",
                candidate.id
            );

            // Built once per format it claims, from a model path of that format.
            for format in candidate.formats {
                let config = EmbeddingConfig {
                    model_path: match format {
                        ModelFormat::Gguf => "absent/model.gguf".into(),
                        ModelFormat::Onnx => "absent/model.onnx".into(),
                    },
                    embedding_dim: 8,
                    max_tokens: 64,
                    pooling: candidate.poolings[0],
                    ..Default::default()
                };
                let Some(built) = (candidate.construct)(&config) else {
                    continue; // not compiled into this build
                };
                // "compiled in, but cannot serve this config" — expected for a real
                // backend handed a nonexistent `model_path`, and no evidence about
                // the row's accuracy.
                let Ok(backend) = built else {
                    continue;
                };
                constructed += 1;

                assert_eq!(
                    backend.id(),
                    candidate.id,
                    "the table names a backend that reports itself as {}",
                    backend.id()
                );
                assert!(
                    candidate.poolings.contains(&backend.pooling()),
                    "{} pools {} for a {format} model, which its row does not declare",
                    candidate.id,
                    backend.pooling()
                );
            }
        }

        assert!(
            constructed > 0,
            "in-crate tests compile the stand-in, so at least one candidate must build"
        );
    }

    /// The format picks the rows walked: every format has a stand-in row, and a real
    /// backend for each format precedes the stand-in for it.
    #[test]
    fn every_format_is_served_and_a_real_backend_precedes_its_stand_in() {
        for format in ModelFormat::ALL {
            let rows: Vec<&str> = CANDIDATES
                .iter()
                .filter(|candidate| candidate.formats.contains(&format))
                .map(|candidate| candidate.id)
                .collect();
            assert_eq!(
                rows.last(),
                Some(&MockHashBackend::ID),
                "the stand-in must be the last resort for {format}, got {rows:?}"
            );
            assert!(
                rows.len() >= 2,
                "{format} has no real backend in the table: {rows:?}"
            );
        }
    }

    /// The no-backend arm cannot be checked here — `#[cfg(test)]` enables the
    /// stand-in by construction — so `tests/production_backend_gate.rs` has it.
    #[test]
    fn selection_yields_the_stand_in_and_reports_it_as_non_semantic() {
        let config = EmbeddingConfig {
            embedding_dim: 48,
            max_tokens: 128,
            ..Default::default()
        };
        let backend = select_backend(&config).expect("in-crate tests compile the stand-in");

        assert_eq!(backend.id(), "mock-hash-v1");
        assert!(
            !backend.is_semantic(),
            "the stand-in must never claim to be semantic"
        );
        assert_eq!(
            backend.dim(),
            48,
            "the backend must produce what the configuration asked for"
        );
        assert_eq!(backend.max_tokens(), 128);
        assert_eq!(backend.pooling(), Pooling::LastToken);
    }

    /// The stand-in serves an ONNX model too, claiming the pooling the real ONNX
    /// backend performs, so an ONNX configuration loads in a mock build.
    #[test]
    fn selection_yields_the_stand_in_for_an_onnx_model_claiming_in_graph_pooling() {
        let config = EmbeddingConfig {
            model_path: "absent/model.ONNX".into(),
            embedding_dim: 24,
            max_tokens: 256,
            pooling: Pooling::InGraph,
            ..Default::default()
        };
        let backend = select_backend(&config).expect("in-crate tests compile the stand-in");

        assert_eq!(backend.id(), "mock-hash-v1");
        assert!(!backend.is_semantic());
        assert_eq!(backend.dim(), 24);
        assert_eq!(backend.max_tokens(), 256);
        assert_eq!(backend.pooling(), Pooling::InGraph);

        // The same stand-in for a GGUF path claims last-token, as llama.cpp would.
        let gguf = select_backend(&EmbeddingConfig {
            embedding_dim: 24,
            ..Default::default()
        })
        .unwrap();
        assert_eq!(gguf.pooling(), Pooling::LastToken);
    }

    #[test]
    fn selection_refuses_a_configuration_no_backend_should_be_built_for() {
        let cases: Vec<(&str, EmbeddingConfig)> = vec![
            (
                "a zero token cap",
                EmbeddingConfig {
                    max_tokens: 0,
                    ..Default::default()
                },
            ),
            (
                // The cap counts the special tokens, so 1 leaves no content budget at all.
                "a token cap of one",
                EmbeddingConfig {
                    max_tokens: 1,
                    ..Default::default()
                },
            ),
            (
                "a zero dimensionality",
                EmbeddingConfig {
                    embedding_dim: 0,
                    ..Default::default()
                },
            ),
            (
                "a pooling nothing performs",
                EmbeddingConfig {
                    pooling: Pooling::Mean,
                    ..Default::default()
                },
            ),
            (
                "a pooling only another format's backends perform",
                EmbeddingConfig {
                    pooling: Pooling::InGraph,
                    ..Default::default()
                },
            ),
            (
                "an ONNX model asked to pool like a GGUF",
                EmbeddingConfig {
                    model_path: "absent/model.onnx".into(),
                    pooling: Pooling::LastToken,
                    ..Default::default()
                },
            ),
        ];

        for (name, config) in cases {
            let result = select_backend(&config);
            assert!(
                result.is_err(),
                "{name} must be refused, but a backend was built for it"
            );
        }
    }

    #[test]
    fn the_stand_in_refuses_to_tokenize_rather_than_inventing_ids() {
        let backend = MockHashBackend::new(16, 512);
        match backend.tokenize("בראשית ברא אלהים") {
            Err(EmbeddingError::TokenizationUnsupported { backend, reason }) => {
                assert_eq!(backend, "mock-hash-v1");
                assert!(
                    reason.contains("no tokenizer"),
                    "unhelpful reason: {reason}"
                );
            }
            other => panic!("the stand-in has no tokenizer; got {other:?}"),
        }
    }

    /// These vectors are a fixture the rest of the suite depends on, so the
    /// backend is pinned to the bytes `mock::hash_embedding` produces.
    #[test]
    fn the_stand_in_returns_exactly_the_documented_hash_and_does_not_normalize() {
        let texts = ["בראשית ברא אלהים", "ויאמר אלהים יהי אור", "ויהי אור"];
        let backend = MockHashBackend::new(32, 512);

        let produced = backend.embed_batch_raw(&texts).unwrap();
        assert_eq!(produced.len(), texts.len());
        for (vector, text) in produced.iter().zip(texts) {
            assert_eq!(
                *vector,
                mock::hash_embedding(text, 32),
                "the backend must be byte-identical to the documented hash"
            );
        }

        let norm = produced[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() > 1e-3,
            "the stand-in must return raw vectors, but got norm {norm}"
        );

        assert!(backend.embed_batch_raw(&[]).unwrap().is_empty());
    }

    #[test]
    fn the_stand_in_passes_a_degenerate_vector_through_for_the_runtime_to_reject() {
        let backend = MockHashBackend::new(8, 512);
        let produced = backend.embed_batch_raw(&["   "]).unwrap();
        assert_eq!(produced[0], vec![0.0f32; 8]);
    }
}
