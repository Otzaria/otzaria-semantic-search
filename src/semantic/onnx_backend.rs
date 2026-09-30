//! Real ONNX inference through ONNX Runtime: the `onnxruntime-sentence-v1` backend.
//!
//! **Not implemented yet.** This module fixes the contract the backend-selection table
//! in [`backend`](crate::semantic::backend) builds against — [`OnnxBackendConfig`],
//! [`OnnxBackend::open`] and the [`EmbeddingBackend`] impl — so that everything around
//! it can be written and tested now. Until the implementation lands,
//! [`OnnxBackend::open`] refuses every model with
//! [`EmbeddingError::BackendUnavailable`]: a build with `onnx-backend` on serves no ONNX
//! model at all rather than serving one wrongly.
//!
//! # What the backend id promises
//!
//! ```text
//! text                                   # already role-prefixed by the text recipe
//!   -> tokenizer.json                    # the package's own, never a substitute
//!   -> input_ids + attention_mask        # token_type_ids as zeros, only if declared
//!   -> truncate to max_tokens            # the total length, special tokens included
//!   -> the graph's first output          # [batch, dim]: pooled inside the graph
//!   -> return RAW                        # the runtime normalizes and validates
//! ```

use crate::errors::EmbeddingError;
use crate::semantic::backend::{EmbeddingBackend, Pooling};
use crate::semantic::embedding::EmbeddingConfig;
use std::path::Path;

/// Default intra-op threads per session: a cap, not a target — phones are big.LITTLE.
/// Four is the llama backend's measured cap, taken over until this backend has its own
/// measurement.
const DEFAULT_THREADS_CAP: usize = 4;

/// Default number of sessions, i.e. concurrent `embed_batch_raw` calls: one, because
/// the smallest target is a phone and every session holds its own activations.
const DEFAULT_SESSIONS: usize = 1;

/// Why [`OnnxBackend`] cannot run anything yet, for every error that says so.
const NOT_IMPLEMENTED: &str = "the `onnx-backend` feature is enabled, but this build's ONNX \
                               Runtime backend is a placeholder whose implementation has not \
                               landed yet";

/// Tuning for [`OnnxBackend`]. Separate from [`EmbeddingConfig`] for the reason
/// `LlamaBackendConfig` is: that type is persisted as an index's identity, while these
/// are deployment facts that must not change a stored vector, and so must not
/// invalidate an index when they change.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OnnxBackendConfig {
    /// ONNX Runtime intra-op threads per session. At least one.
    pub intra_threads: usize,
    /// Sessions in the pool: how many `embed_batch_raw` calls run at once. At least
    /// one.
    pub sessions: usize,
}

impl Default for OnnxBackendConfig {
    fn default() -> Self {
        Self {
            intra_threads: DEFAULT_THREADS_CAP.min(available_parallelism()),
            sessions: DEFAULT_SESSIONS,
        }
    }
}

impl OnnxBackendConfig {
    /// Environment variable overriding [`Self::intra_threads`].
    pub const ENV_THREADS: &'static str = "OTZARIA_ONNX_THREADS";
    /// Environment variable overriding [`Self::sessions`].
    pub const ENV_SESSIONS: &'static str = "OTZARIA_ONNX_SESSIONS";

    /// Defaults, with environment-variable overrides applied — the backend-selection
    /// table can hand a constructor nothing but an [`EmbeddingConfig`], exactly as for
    /// `LlamaBackendConfig::from_env_for`, whose signature this mirrors.
    ///
    /// # Errors
    ///
    /// [`EmbeddingError::LoadFailed`], naming the variable, if one is set to anything
    /// but a positive integer. Someone who exported `OTZARIA_ONNX_SESSIONS=two` is
    /// sizing memory, and quietly giving them the default sizes nothing; zero is refused
    /// rather than read as "let the runtime decide", because that is a guess about what
    /// was meant.
    pub fn from_env_for(_config: &EmbeddingConfig) -> Result<Self, EmbeddingError> {
        let mut selected = Self::default();
        read_positive(Self::ENV_THREADS, &mut selected.intra_threads)?;
        read_positive(Self::ENV_SESSIONS, &mut selected.sessions)?;
        Ok(selected)
    }
}

/// Override `slot` from `key` if it is set, refusing anything but a positive integer.
fn read_positive(key: &str, slot: &mut usize) -> Result<(), EmbeddingError> {
    let Ok(raw) = std::env::var(key) else {
        return Ok(());
    };
    match raw.trim().parse::<usize>() {
        Ok(value) if value > 0 => {
            *slot = value;
            Ok(())
        }
        _ => Err(EmbeddingError::LoadFailed {
            reason: format!("{key} is set to {raw:?}; it must be a positive whole number"),
        }),
    }
}

/// Machine parallelism, or 1 — `available_parallelism` fails on some sandboxed and
/// embedded targets, and 1 is the answer that cannot oversubscribe.
fn available_parallelism() -> usize {
    std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
}

/// Real ONNX inference: ONNX Runtime, the package's `tokenizer.json`, the graph's own
/// pooling. Construct with [`OnnxBackend::open`].
pub struct OnnxBackend {
    /// The graph's output width.
    dim: u32,
    /// The total sequence length every input is truncated to.
    max_tokens: usize,
}

impl OnnxBackend {
    /// Identifier recorded in the manifest as `embedding_backend`.
    ///
    /// It names the wiring — ONNX Runtime; `tokenizer.json` feeding `input_ids` and
    /// `attention_mask` (plus `token_type_ids` as zeros, only when the graph declares
    /// that input); the first output taken as the sentence vector — and a version bumped
    /// when that wiring changes what comes out for the same inputs. A change invalidates
    /// every stored vector.
    pub const ID: &'static str = "onnxruntime-sentence-v1";

    /// Load the graph at `graph` with the tokenizer at `tokenizer`, truncating every
    /// input to `max_tokens` tokens in total, special tokens included.
    ///
    /// `pooling` is what the configuration asks for; the backend serves only
    /// [`Pooling::InGraph`], and proves at load that the graph's first output is a
    /// finished `[batch, dim]` sentence vector.
    ///
    /// # Errors
    ///
    /// Today, always [`EmbeddingError::BackendUnavailable`]: the implementation has not
    /// landed.
    pub fn open(
        graph: &Path,
        tokenizer: &Path,
        max_tokens: usize,
        pooling: Pooling,
        tuning: &OnnxBackendConfig,
    ) -> Result<Self, EmbeddingError> {
        // Named in the signature because they are the contract; unused until the
        // implementation exists.
        let _ = (tokenizer, max_tokens, pooling, tuning);
        Err(EmbeddingError::BackendUnavailable {
            reason: format!(
                "{NOT_IMPLEMENTED}; model file {} cannot be executed",
                graph.display()
            ),
        })
    }
}

impl EmbeddingBackend for OnnxBackend {
    fn id(&self) -> &'static str {
        Self::ID
    }

    fn is_semantic(&self) -> bool {
        true
    }

    fn dim(&self) -> u32 {
        self.dim
    }

    fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    fn pooling(&self) -> Pooling {
        Pooling::InGraph
    }

    fn tokenize(&self, _text: &str) -> Result<Vec<u32>, EmbeddingError> {
        Err(EmbeddingError::BackendUnavailable {
            reason: NOT_IMPLEMENTED.to_string(),
        })
    }

    fn embed_batch_raw(&self, _texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        Err(EmbeddingError::BackendUnavailable {
            reason: NOT_IMPLEMENTED.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Every test that reads or writes the process environment holds this, because
    /// the test harness runs them on parallel threads of one process.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn the_defaults_are_one_session_and_at_most_four_threads() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::remove_var(OnnxBackendConfig::ENV_THREADS);
        std::env::remove_var(OnnxBackendConfig::ENV_SESSIONS);

        let tuning = OnnxBackendConfig::from_env_for(&EmbeddingConfig::default()).unwrap();
        assert_eq!(tuning, OnnxBackendConfig::default());
        assert_eq!(tuning.sessions, 1);
        assert_eq!(tuning.intra_threads, 4.min(available_parallelism()));
        assert!(tuning.intra_threads >= 1);
    }

    #[test]
    fn a_set_variable_overrides_the_default() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var(OnnxBackendConfig::ENV_THREADS, " 3 ");
        std::env::set_var(OnnxBackendConfig::ENV_SESSIONS, "2");
        let tuning = OnnxBackendConfig::from_env_for(&EmbeddingConfig::default());
        std::env::remove_var(OnnxBackendConfig::ENV_THREADS);
        std::env::remove_var(OnnxBackendConfig::ENV_SESSIONS);

        assert_eq!(
            tuning.unwrap(),
            OnnxBackendConfig {
                intra_threads: 3,
                sessions: 2
            }
        );
    }

    #[test]
    fn an_unparseable_or_zero_value_is_refused_and_named() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        for key in [
            OnnxBackendConfig::ENV_THREADS,
            OnnxBackendConfig::ENV_SESSIONS,
        ] {
            for wrong in ["two", "", "-1", "0", "1.5"] {
                std::env::set_var(key, wrong);
                let refused = OnnxBackendConfig::from_env_for(&EmbeddingConfig::default());
                std::env::remove_var(key);

                match refused {
                    Err(EmbeddingError::LoadFailed { reason }) => assert!(
                        reason.contains(key) && reason.contains(&format!("{wrong:?}")),
                        "the error must name the variable and its value: {reason}"
                    ),
                    other => panic!("{key}={wrong:?} must be refused, got {other:?}"),
                }
            }
        }
    }

    #[test]
    fn the_placeholder_refuses_to_open_anything() {
        let refused = OnnxBackend::open(
            Path::new("model.onnx"),
            Path::new("tokenizer.json"),
            256,
            Pooling::InGraph,
            &OnnxBackendConfig::default(),
        )
        .err()
        .expect("the placeholder must not produce a backend");

        match refused {
            EmbeddingError::BackendUnavailable { reason } => {
                assert!(reason.contains("model.onnx"), "unhelpful reason: {reason}");
            }
            other => panic!("expected BackendUnavailable, got {other:?}"),
        }
    }
}
