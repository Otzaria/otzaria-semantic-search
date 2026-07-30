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
//! | build | backend | [`select_backend`] |
//! |---|---|---|
//! | default | none | `Err(EmbeddingError::BackendUnavailable)` |
//! | `--features llama-backend` | `LlamaCppBackend` | `Ok`, or why the model cannot be served |
//! | `--features mock-embedding` (and in-crate tests) | `MockHashBackend` | `Ok` |
//!
//! The stand-in is gated so a release build cannot serve fake vectors; real
//! inference is gated because it compiles llama.cpp through cmake. With both
//! enabled the real one wins, since `CANDIDATES` is ordered by preference.

use crate::errors::EmbeddingError;
use crate::semantic::embedding::EmbeddingConfig;

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
}

impl Pooling {
    /// What [`Self::parse`] searches, so a variant left out cannot be read back.
    /// The compiler misses that; `every_variant_is_listed_and_parseable` does not.
    pub const ALL: [Self; 2] = [Self::LastToken, Self::Mean];

    /// The exact string persisted in the manifest. These spellings are already on
    /// disk; changing one invalidates every existing index.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LastToken => "last-token",
            Self::Mean => "mean",
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

    /// The token cap this backend applies to a single input.
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
/// must come first**, or enabling `mock-embedding` on top of a real build indexes
/// fake vectors unknowingly.
///
/// Deliberately not feature-gated: the table describes the implementations that
/// *exist*, which is what "a pooling no backend implements" means. Gating it
/// would make a default build reject `"last-token"` as unimplemented — reporting
/// a missing backend as a bad configuration value.
const CANDIDATES: &[BackendCandidate] = &[
    BackendCandidate {
        id: "llama-cpp-qwen3-last-v1",
        poolings: &[Pooling::LastToken],
        construct: llama_cpp_backend,
    },
    BackendCandidate {
        id: "mock-hash-v1",
        poolings: &[Pooling::LastToken],
        construct: mock_hash_backend,
    },
];

/// Ordered by [`Pooling::ALL`] so error messages are stable.
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
/// value. `pooling = "mean"` parses, so without this it reached the manifest as
/// the index's identity; correcting the configuration then made the manifest
/// disagree with it, recoverable only by discarding the index.
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

/// Attributed per backend, because "implemented: last-token" reads as a limit of
/// the build while "last-token (mock-hash-v1)" names the implementation.
fn describe_implemented_poolings() -> String {
    let described: Vec<String> = Pooling::ALL
        .into_iter()
        .filter_map(|strategy| {
            let backends: Vec<&str> = CANDIDATES
                .iter()
                .filter(|candidate| candidate.poolings.contains(&strategy))
                .map(|candidate| candidate.id)
                .collect();
            (!backends.is_empty()).then(|| format!("{strategy} ({})", backends.join(", ")))
        })
        .collect();

    if described.is_empty() {
        "none".to_string()
    } else {
        described.join("; ")
    }
}

/// Choose the backend this build can offer for `config`, by walking `CANDIDATES`
/// rather than through `#[cfg]` blocks inside an inference call.
///
/// `config` is validated here too: this function is public and reachable without
/// [`EmbeddingRuntime::load`](crate::semantic::embedding::EmbeddingRuntime::load),
/// so a direct caller could otherwise get a backend built for `max_tokens: 0`.
///
/// # Errors
///
/// [`EmbeddingError::BackendUnavailable`] when nothing is compiled in — every
/// default build, the guarantee `tests/production_backend_gate.rs` holds.
/// Otherwise whatever the first compiled-in candidate, or
/// [`EmbeddingConfig::validate`], failed with.
pub fn select_backend(
    config: &EmbeddingConfig,
) -> Result<Box<dyn EmbeddingBackend>, EmbeddingError> {
    config.validate()?;

    // `Some(Err(_))` stops the walk just as `Some(Ok(_))` does — see
    // `BackendCandidate::construct`.
    CANDIDATES
        .iter()
        .find_map(|candidate| (candidate.construct)(config))
        .unwrap_or_else(|| {
            Err(EmbeddingError::BackendUnavailable {
                reason: format!(
                    "this build has no inference backend compiled in (enable the \
                     `llama-backend` feature for real GGUF inference); model file {} \
                     validated but cannot be executed",
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
#[cfg(all(feature = "llama-backend", not(test)))]
fn llama_cpp_backend(config: &EmbeddingConfig) -> Constructed {
    use crate::semantic::llama_backend::{LlamaBackendConfig, LlamaCppBackend};

    // The constructor receives only an `EmbeddingConfig`, so llama.cpp's own
    // knobs come from the environment; typed callers use `LlamaCppBackend::open`.
    Some(LlamaBackendConfig::from_env_for(config).and_then(|tuning| {
        LlamaCppBackend::open(&config.model_path, config.max_tokens, &tuning)
            .map(|backend| Box::new(backend) as Box<dyn EmbeddingBackend>)
    }))
}

/// `None`, not `Some(Err(_))`: without the feature there is no such
/// implementation at all.
#[cfg(not(all(feature = "llama-backend", not(test))))]
fn llama_cpp_backend(_config: &EmbeddingConfig) -> Constructed {
    None
}

#[cfg(any(test, feature = "mock-embedding"))]
fn mock_hash_backend(config: &EmbeddingConfig) -> Constructed {
    Some(Ok(Box::new(MockHashBackend::new(
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
}

#[cfg(any(test, feature = "mock-embedding"))]
impl MockHashBackend {
    /// Already recorded in manifests written by this backend; changing it
    /// invalidates those indexes.
    pub const ID: &'static str = "mock-hash-v1";

    /// `dim` is echoed back from [`EmbeddingBackend::dim`] so the load-time
    /// agreement check passes; a real backend reads it from the model.
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

    /// A word-hash pools nothing. Claiming [`Pooling::LastToken`] is what makes
    /// the default configuration load and the agreement check run for real
    /// instead of being trivially satisfied.
    fn pooling(&self) -> Pooling {
        Pooling::LastToken
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
                Pooling::LastToken | Pooling::Mean => Pooling::ALL.contains(&strategy),
            }
        }

        for strategy in [Pooling::LastToken, Pooling::Mean] {
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

        assert_eq!(implemented_poolings(), vec![Pooling::LastToken]);
        assert!(ensure_pooling_is_implemented(Pooling::LastToken).is_ok());
    }

    /// The table states what a backend pools without building it, and two
    /// statements of one fact can drift: an overstated row would accept a
    /// configuration the backend then refuses at load time.
    #[test]
    fn every_candidate_describes_the_backend_it_builds() {
        let config = EmbeddingConfig {
            embedding_dim: 8,
            max_tokens: 64,
            ..Default::default()
        };

        let mut constructed = 0usize;
        for candidate in CANDIDATES {
            let Some(built) = (candidate.construct)(&config) else {
                continue; // not compiled into this build
            };
            // "compiled in, but cannot serve this config" — expected for a real
            // backend handed a nonexistent default `model_path`, and no evidence
            // about the row's accuracy.
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
                "{} pools {} , which its row does not declare",
                candidate.id,
                backend.pooling()
            );
            assert!(
                !candidate.poolings.is_empty(),
                "{} declares no pooling, so no configuration could ever select it",
                candidate.id
            );
        }

        assert!(
            constructed > 0,
            "in-crate tests compile the stand-in, so at least one candidate must build"
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
        assert_eq!(backend.pooling(), Pooling::LastToken);
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
