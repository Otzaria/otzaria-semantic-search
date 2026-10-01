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
//! | build | ONNX model |
//! |---|---|
//! | default | `Err(BackendUnavailable)` |
//! | `--features onnx-backend` | `OnnxBackend` |
//! | `--features mock-embedding` (and in-crate tests) | `MockHashBackend` |
//!
//! A real backend's cell means `Ok`, or why that backend cannot serve the model.
//!
//! **An ONNX graph is the only model there is**: a path that does not end in `.onnx` is
//! refused by [`EmbeddingConfig::validate`] as
//! [`EmbeddingError::InvalidModelFile`] before any backend is asked — see
//! [`ensure_onnx_model_path`](crate::semantic::model_package::ensure_onnx_model_path).
//! The stand-in is gated so a release build cannot serve fake vectors; real inference is
//! gated because it brings a large native runtime into the build. With the real backend
//! and the stand-in both enabled the real one wins, since `CANDIDATES` is ordered by
//! preference.

use crate::errors::EmbeddingError;
use crate::semantic::embedding::{EmbeddingConfig, EmbeddingDeployment};

/// How a model's per-token hidden states are collapsed into one vector.
///
/// Closed rather than a string: pooling decides what the vector *is* and the
/// manifest records it as the index's identity, so a typo used to produce vectors
/// pooled one way while the manifest claimed another.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pooling {
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
    pub const ALL: [Self; 2] = [Self::Mean, Self::InGraph];

    /// The exact string persisted in the manifest. These spellings are already on
    /// disk; changing one invalidates every existing index.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Mean => "mean",
            Self::InGraph => "in-graph",
        }
    }

    /// Matched **exactly** — not case-insensitively, not trimmed. Accepting
    /// `"In-Graph"` would persist that spelling, and a later canonically
    /// spelled configuration would read it as a *different* pooling: semantic
    /// search disabled and a full re-index demanded over capitalization.
    ///
    /// `"last-token"`, the pooling of the GGUF backend this crate no longer has, is no
    /// longer a spelling: it reads as [`EmbeddingError::UnknownPooling`].
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
    /// length, counting every special token the backend adds around the text — a BERT
    /// tokenizer's `[CLS]` and `[SEP]`.
    ///
    /// A contract the *backend* implements, not something this layer enforces:
    /// the runtime has no tokenizer, and `Chunker`'s character limit is a
    /// different limit. The real backend honours it; the stand-in cannot and
    /// echoes the request. The three rules:
    ///
    /// * truncate, never fail — an over-long line must still be indexed;
    /// * cut the text, not its frame — keep leading tokens, and still add the special
    ///   tokens the model expects around them, inside the cap;
    /// * report the effective cap — the one actually applied, which a backend may
    ///   refuse at load rather than serve when the model cannot run it.
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
    ///
    /// Handed the deployment beside the configuration, never inside it: where this
    /// machine keeps a runtime is no part of what the vectors are. A backend with nothing
    /// to find ignores it.
    construct: fn(&EmbeddingConfig, &EmbeddingDeployment) -> Constructed,
}

/// `None`: not in this build. `Some(Err(_))`: mine, and here is why it failed.
type Constructed = Option<Result<Box<dyn EmbeddingBackend>, EmbeddingError>>;

/// Every backend this crate implements, in preference order. **The real backend
/// must come before the stand-in**, or enabling `mock-embedding` on top of a real
/// build indexes fake vectors unknowingly.
///
/// Deliberately not feature-gated: the table describes the implementations that
/// *exist*, which is what "a pooling no backend implements" means. Gating it
/// would make a default build reject `"in-graph"` as unimplemented — reporting
/// a missing backend as a bad configuration value.
///
/// The stand-in claims what the real backend claims, so a mock build runs the pooling
/// agreement check for real.
const CANDIDATES: &[BackendCandidate] = &[
    BackendCandidate {
        // `OnnxBackend::ID`, spelled out because this table is not feature-gated.
        id: "onnxruntime-sentence-v1",
        poolings: &[Pooling::InGraph],
        construct: onnx_backend,
    },
    BackendCandidate {
        id: "mock-hash-v1",
        poolings: &[Pooling::InGraph],
        construct: mock_hash_backend,
    },
];

/// Every pooling some backend performs. Ordered by [`Pooling::ALL`] so error messages
/// are stable.
pub fn implemented_poolings() -> Vec<Pooling> {
    Pooling::ALL
        .into_iter()
        .filter(|strategy| {
            CANDIDATES
                .iter()
                .any(|candidate| candidate.poolings.contains(strategy))
        })
        .collect()
}

/// Refuse a pooling no backend implements while it is still only a configuration
/// value — the check [`EmbeddingConfig::validate`] and the engine's configuration make,
/// because the pooling reaches the manifest before any backend is asked.
/// `pooling = "mean"` parses, so without this it reached the manifest as the index's
/// identity; correcting the configuration then made the manifest disagree with it,
/// recoverable only by discarding the index.
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
        implemented: describe_implemented_poolings(),
    })
}

/// The largest token cap an ONNX model may be configured with: past the context of any
/// sentence encoder (512 positions for Meivin Round 2; the long-context ones stop at 8192
/// or 32768).
///
/// A bound on configurations, not a claim about a graph — whether a graph runs a cap
/// below it is what the ONNX backend's load-time probe proves — and one that probe needs:
/// it is as long as the cap, so an unbounded cap (a `u32` of a few billion, which is what
/// a negative application setting arrives as) asked the allocator for hundreds of
/// gigabytes and aborted the process instead of failing the load.
pub const ONNX_MAX_TOKENS_CEILING: usize = 1 << 16;

/// Why a token cap of `max_tokens`, configured as `field`, is more than any backend
/// serves — or `None`. Refused while it is still a configuration, for the reason
/// [`ensure_pooling_is_implemented`] refuses a pooling: the cap is recorded as the
/// index's identity before any backend is asked. [`EmbeddingConfig::validate`], the
/// engine's configuration and the ONNX backend each refuse it in their own error type.
pub(crate) fn max_tokens_past_any_encoder(field: &str, max_tokens: usize) -> Option<String> {
    (max_tokens > ONNX_MAX_TOKENS_CEILING).then(|| {
        format!(
            "{field} is {max_tokens}, past the context of any ONNX sentence encoder: at most \
             {ONNX_MAX_TOKENS_CEILING} is accepted. Set it to the model's own cap — the \
             Meivin Round 2 identity declares 256 — and the load-time probe then proves the \
             graph runs it"
        )
    })
}

/// Attributed per backend, because "implemented: in-graph" reads as a limit of
/// the build while "in-graph (mock-hash-v1)" names the implementation.
fn describe_implemented_poolings() -> String {
    let described: Vec<String> = Pooling::ALL
        .into_iter()
        .filter_map(|strategy| {
            let backends = backends_performing(strategy);
            (!backends.is_empty()).then(|| format!("{strategy} ({backends})"))
        })
        .collect();

    if described.is_empty() {
        "none".to_string()
    } else {
        described.join("; ")
    }
}

/// The ids of the backends performing `pooling`, in table order, each once.
fn backends_performing(pooling: Pooling) -> String {
    let mut ids: Vec<&str> = Vec::new();
    for candidate in CANDIDATES {
        if candidate.poolings.contains(&pooling) && !ids.contains(&candidate.id) {
            ids.push(candidate.id);
        }
    }
    ids.join(", ")
}

/// Choose the backend this build can offer for `config`, by walking `CANDIDATES`
/// rather than through `#[cfg]` blocks inside an inference call.
///
/// `config` is validated here too: this function is public and reachable without
/// [`EmbeddingRuntime::load`](crate::semantic::embedding::EmbeddingRuntime::load),
/// so a direct caller could otherwise get a backend built for `max_tokens: 1`,
/// which embeds every text as a bare special token, or for a model path that names no
/// ONNX graph.
///
/// # Errors
///
/// [`EmbeddingError::BackendUnavailable`] when no backend is compiled in — every default
/// build, the guarantee `tests/production_backend_gate.rs` holds — naming the feature
/// that would serve the model, or, with that feature already on, saying that this target
/// has no ONNX backend (`no_backend_reason`).
/// Otherwise whatever the first compiled-in candidate, or
/// [`EmbeddingConfig::validate`], failed with.
///
/// Built for the default [`EmbeddingDeployment`]; [`select_backend_for`] takes another.
pub fn select_backend(
    config: &EmbeddingConfig,
) -> Result<Box<dyn EmbeddingBackend>, EmbeddingError> {
    select_backend_for(config, &EmbeddingDeployment::default())
}

/// [`select_backend`], for a model deployed as `deployment`: the backend is built to load
/// what it needs besides the model from where `deployment` says — the ONNX Runtime
/// library at [`EmbeddingDeployment::onnx_runtime`]. Which backend serves the model is
/// still the table's to decide, never the deployment's; a backend with nothing to load
/// ignores it.
///
/// # Errors
///
/// As [`select_backend`].
pub fn select_backend_for(
    config: &EmbeddingConfig,
    deployment: &EmbeddingDeployment,
) -> Result<Box<dyn EmbeddingBackend>, EmbeddingError> {
    config.validate()?;

    // `Some(Err(_))` stops the walk just as `Some(Ok(_))` does — see
    // `BackendCandidate::construct`.
    CANDIDATES
        .iter()
        .find_map(|candidate| (candidate.construct)(config, deployment))
        .unwrap_or_else(|| {
            Err(EmbeddingError::BackendUnavailable {
                reason: no_backend_reason(cfg!(feature = "onnx-backend"), &config.model_path),
            })
        })
}

/// Why nothing in this build serves the model — the reason [`select_backend`] gives when
/// no candidate answers.
///
/// It names `onnx-backend`, unless that feature is on (`onnx_feature_enabled`, the
/// caller's `cfg!`). Then no row answered because this target has no ONNX backend at all
/// — its crates are declared for desktop targets only — and telling the reader to enable
/// a feature that is already on would send them round a loop: the plugin's production
/// feature set enables it on phones too.
fn no_backend_reason(onnx_feature_enabled: bool, model_path: &std::path::Path) -> String {
    if onnx_feature_enabled {
        return format!(
            "this target has no ONNX backend in this version: `onnx-backend` is enabled, \
             but ONNX Runtime is loaded on desktop targets only (macOS, Linux with glibc, \
             Windows with MSVC; aarch64 and x86_64); model file {} validated but cannot be \
             executed here",
            model_path.display()
        );
    }
    format!(
        "this build has no inference backend compiled in (enable the \
         `onnx-backend` feature for real ONNX inference); model file {} \
         validated but cannot be executed",
        model_path.display()
    )
}

// ── ONNX Runtime: the constructor pair ──────────────────────────────────────────
//
// The two `cfg`s below are exact complements and must stay so, and the `mod
// onnx_backend` line in `semantic/mod.rs` must carry the same feature/target
// condition as the first one. A target restriction goes into all three.

/// Real ONNX inference, in a build that compiled it in.
///
/// `not(test)` because the in-crate suite drives this module with stub packages that no
/// real backend could load, and a real backend ahead of the stand-in in [`CANDIDATES`]
/// would fail on every one of them. Integration tests link the library without
/// `cfg(test)` and so see the real table.
///
/// The package's tokenizer is `tokenizer.json` beside the graph — see
/// [`onnx_tokenizer_path`](crate::semantic::model_package::onnx_tokenizer_path) — and the
/// runtime library is the one [`EmbeddingDeployment::onnx_runtime`] names, when it names
/// one.
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
    ),
    not(test)
))]
fn onnx_backend(config: &EmbeddingConfig, deployment: &EmbeddingDeployment) -> Constructed {
    use crate::semantic::model_package::onnx_tokenizer_path;
    use crate::semantic::onnx_backend::{OnnxBackend, OnnxBackendConfig};

    // The table passes no tuning, so the runtime's own knobs come from the environment;
    // typed callers use `OnnxBackend::open`.
    Some(OnnxBackendConfig::from_env_for(config).and_then(|tuning| {
        OnnxBackend::open(
            &config.model_path,
            &onnx_tokenizer_path(&config.model_path),
            config.max_tokens,
            config.pooling,
            &tuning,
            deployment.onnx_runtime.as_deref(),
        )
        .map(|backend| Box::new(backend) as Box<dyn EmbeddingBackend>)
    }))
}

/// `None`, not `Some(Err(_))`: without the feature there is no such implementation.
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
    ),
    not(test)
)))]
fn onnx_backend(_config: &EmbeddingConfig, _deployment: &EmbeddingDeployment) -> Constructed {
    None
}

// ── end of the ONNX Runtime constructor pair ────────────────────────────────────

/// The deployment is ignored: the stand-in loads nothing but the configuration.
#[cfg(any(test, feature = "mock-embedding"))]
fn mock_hash_backend(config: &EmbeddingConfig, _deployment: &EmbeddingDeployment) -> Constructed {
    Some(Ok(Box::new(MockHashBackend::new(
        config.embedding_dim,
        config.max_tokens,
    ))))
}

/// A default build has no stand-in, which is what makes [`select_backend`] fail
/// rather than quietly serve hash vectors.
#[cfg(not(any(test, feature = "mock-embedding")))]
fn mock_hash_backend(_config: &EmbeddingConfig, _deployment: &EmbeddingDeployment) -> Constructed {
    None
}

/// Deterministic hash-based stand-in. **Not a model** — see the module docs. Its
/// vectors must not change: `tests/hybrid_integration_test.rs` asserts an exact
/// self-match similarity of 1.0.
#[cfg(any(test, feature = "mock-embedding"))]
pub struct MockHashBackend {
    dim: u32,
    max_tokens: usize,
}

#[cfg(any(test, feature = "mock-embedding"))]
impl MockHashBackend {
    /// Already recorded in manifests written by this backend; changing it
    /// invalidates those indexes.
    pub const ID: &'static str = "mock-hash-v1";

    /// The stand-in for an ONNX model. `dim` is echoed back from
    /// [`EmbeddingBackend::dim`] so the load-time agreement check passes; a real
    /// backend reads it from the model.
    pub fn new(dim: u32, max_tokens: usize) -> Self {
        Self { dim, max_tokens }
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

    /// A word-hash pools nothing. Claiming what the real backend claims —
    /// [`Pooling::InGraph`] — is what makes the default configuration load and the
    /// agreement check run for real instead of being trivially satisfied.
    fn pooling(&self) -> Pooling {
        Pooling::InGraph
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
                Pooling::Mean | Pooling::InGraph => Pooling::ALL.contains(&strategy),
            }
        }

        for strategy in [Pooling::Mean, Pooling::InGraph] {
            assert!(listed(strategy), "{strategy} is missing from Pooling::ALL");
            assert!(
                Pooling::parse(strategy.as_str()).is_ok(),
                "{strategy} cannot be parsed back from its own spelling"
            );
        }
    }

    #[test]
    fn pooling_round_trips_through_the_exact_string_the_manifest_stores() {
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

    /// The default configuration is the production model's, and that model pools in its
    /// graph: `config/models/meivin-round2-onnx` declares `"in-graph"`.
    #[test]
    fn the_default_pooling_is_the_one_the_production_identity_declares() {
        assert_eq!(EmbeddingConfig::default().pooling, Pooling::InGraph);
        assert_eq!(EmbeddingConfig::default().pooling.as_str(), "in-graph");
    }

    #[test]
    fn an_unknown_or_merely_misspelled_pooling_is_refused() {
        for wrong in [
            "",
            " ",
            "in_graph",
            "ingraph",
            "In-Graph",
            "IN-GRAPH",
            " in-graph ",
            "in graph",
            "cls",
            "none",
            // The GGUF backend's pooling, gone with it.
            "last-token",
            "Last-Token",
        ] {
            match Pooling::parse(wrong) {
                Err(EmbeddingError::UnknownPooling { found, supported }) => {
                    assert_eq!(found, wrong);
                    assert!(
                        supported.contains("in-graph"),
                        "the error must name what is accepted, got {supported:?}"
                    );
                    assert!(
                        !supported.contains("last-token"),
                        "a pooling no longer representable must not be offered: {supported:?}"
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
                    implemented.contains("in-graph"),
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

        assert_eq!(implemented_poolings(), vec![Pooling::InGraph]);
        assert!(ensure_pooling_is_implemented(Pooling::InGraph).is_ok());
    }

    /// The refusal names each implementation beside the pooling it performs, so it reads
    /// as a fact about the backends and not as a limit of the build.
    #[test]
    fn a_pooling_nothing_performs_names_what_is_implemented_and_by_whom() {
        match ensure_pooling_is_implemented(Pooling::Mean) {
            Err(error @ EmbeddingError::PoolingNotImplemented { .. }) => assert_eq!(
                error.to_string(),
                "No embedding backend implements pooling 'mean' (implemented: in-graph \
                 (onnxruntime-sentence-v1, mock-hash-v1))"
            ),
            other => panic!("expected PoolingNotImplemented, got {other:?}"),
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

            let config = EmbeddingConfig {
                model_path: "absent/model.onnx".into(),
                embedding_dim: 8,
                max_tokens: 64,
                pooling: candidate.poolings[0],
                ..Default::default()
            };
            let Some(built) = (candidate.construct)(&config, &EmbeddingDeployment::default())
            else {
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
                "{} pools {}, which its row does not declare",
                candidate.id,
                backend.pooling()
            );
        }

        assert!(
            constructed > 0,
            "in-crate tests compile the stand-in, so at least one candidate must build"
        );
    }

    /// The stand-in is the last resort, and the real backend comes before it.
    #[test]
    fn the_real_backend_precedes_the_stand_in() {
        let rows: Vec<&str> = CANDIDATES.iter().map(|candidate| candidate.id).collect();
        assert_eq!(
            rows.last(),
            Some(&MockHashBackend::ID),
            "the stand-in must be the last resort, got {rows:?}"
        );
        assert!(
            rows.len() >= 2,
            "there is no real backend in the table: {rows:?}"
        );
    }

    /// What a build with no backend says: it names the feature when it is off; when it is
    /// on — which outside the desktop gate means a target with no ONNX backend, a phone —
    /// it says that, rather than asking for a feature already on. Built directly, because
    /// no target this suite runs on reaches that arm.
    #[test]
    fn a_build_with_no_backend_says_which_feature_or_why_not() {
        let onnx = std::path::Path::new("models/model.onnx");
        assert_eq!(
            no_backend_reason(false, onnx),
            "this build has no inference backend compiled in (enable the `onnx-backend` \
             feature for real ONNX inference); model file models/model.onnx validated but \
             cannot be executed"
        );
        let target = no_backend_reason(true, onnx);
        assert!(
            target.contains("this target has no ONNX backend in this version")
                && target.contains("desktop")
                && target.contains("models/model.onnx"),
            "{target}"
        );
        assert!(
            !target.contains("enable the"),
            "the feature is on; asking for it sends the reader round a loop: {target}"
        );
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
        assert_eq!(backend.pooling(), Pooling::InGraph);
    }

    /// The extension is matched in any ASCII case, so `.ONNX` is an ONNX graph too, and
    /// the stand-in serves it claiming the pooling the real backend performs.
    #[test]
    fn selection_yields_the_stand_in_for_an_onnx_model_in_any_case() {
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
                "a GGUF model",
                EmbeddingConfig {
                    model_path: "absent/model.gguf".into(),
                    ..Default::default()
                },
            ),
            (
                "a model path with no extension",
                EmbeddingConfig {
                    model_path: "absent/model".into(),
                    ..Default::default()
                },
            ),
            (
                "an ONNX token cap past any encoder's context",
                EmbeddingConfig {
                    model_path: "absent/model.onnx".into(),
                    pooling: Pooling::InGraph,
                    max_tokens: ONNX_MAX_TOKENS_CEILING + 1,
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

    /// A model path that names no ONNX graph is the configuration's mistake, refused as an
    /// invalid model before any backend — the stand-in included — is asked to serve it.
    #[test]
    fn selection_refuses_a_model_that_is_not_an_onnx_graph_as_an_invalid_model_file() {
        for path in [
            "models/model.gguf",
            "models/model",
            "models/model.onnx.gguf",
        ] {
            let config = EmbeddingConfig {
                model_path: path.into(),
                ..Default::default()
            };
            match select_backend(&config) {
                Err(EmbeddingError::InvalidModelFile {
                    path: refused,
                    reason,
                }) => {
                    assert_eq!(refused, path);
                    assert!(reason.contains(".onnx"), "{reason}");
                }
                Err(other) => panic!("{path} must be an invalid model file, got {other:?}"),
                Ok(backend) => panic!("{path} must be refused, got {}", backend.id()),
            }
        }
    }

    /// The ceiling is past any encoder's context, so every real cap is under it.
    #[test]
    fn the_token_cap_ceiling_is_past_any_encoder_and_names_the_fix() {
        for cap in [2, 256, 512, ONNX_MAX_TOKENS_CEILING] {
            assert_eq!(max_tokens_past_any_encoder("max_tokens", cap), None);
        }
        let refused = max_tokens_past_any_encoder("max_tokens", usize::MAX).expect("refused");
        assert!(
            refused.starts_with(&format!("max_tokens is {}", usize::MAX))
                && refused.contains(&ONNX_MAX_TOKENS_CEILING.to_string())
                && refused.contains("Set it to the model's own cap"),
            "the reason names the cap, the ceiling and the fix: {refused}"
        );
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
