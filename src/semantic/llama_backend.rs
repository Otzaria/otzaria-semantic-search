//! Real GGUF inference through llama.cpp (delivered in PR #2).
//!
//! Behind the non-default `llama-backend` feature; a default build keeps failing
//! with [`EmbeddingError::BackendUnavailable`]. Vector measurements are recorded in
//! `docs/P2_REFERENCE_VECTORS.md` (machine-readable form:
//! `tests/data/golden_vectors.json`); memory, latency and build costs are in
//! `docs/P2_INFERENCE_SPIKE.md`. Everything numeric was measured against
//! `Otzaria-Embedding-V1-Flash-0.6B-Q4_K_M.gguf`, not taken from the model card.
//!
//! # Pipeline
//!
//! ```text
//! text
//!   -> tokenize(add_special = false, parse_special = false)   # content tokens only
//!   -> truncate to max_tokens - 1
//!   -> push EOS (151643)                                      # always the final token
//!   -> decode, several sequences per call
//!   -> pooling = LAST (the hidden state of the final token, i.e. the EOS)
//!   -> return RAW, unnormalized
//! ```
//!
//! Appending the EOS by hand rather than through `add_special` makes "EOS is the last
//! token" structural rather than conditional — see `truncate_with_eos`, and
//! `TokenizerContract` for the load-time proof that the two spellings agree.
//! `parse_special = false` is why the crate's one layout assumption exists; see the
//! `tokenizer` submodule. [`EmbeddingConfig::max_tokens`] is the **total** sequence
//! length, EOS included.
//!
//! Inference runs in a bounded pool of worker threads, one `LlamaContext` each, because
//! [`EmbeddingBackend::embed_batch_raw`] takes `&self` and a `Mutex<LlamaContext>` would
//! make a search wait for a whole indexing batch; see `ContextPool`, and
//! `DEFAULT_CONTEXTS` for why the default pool holds one. Tokenization needs no context
//! and runs fully parallel.
//!
//! On Apple platforms a live [`LlamaContext`] at process exit aborts ggml's Metal
//! teardown; `release_contexts_at_exit` defends against it and a host needs to do nothing.

use crate::errors::EmbeddingError;
use crate::semantic::backend::{EmbeddingBackend, Pooling};
use crate::semantic::embedding::EmbeddingConfig;

use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::token::LlamaToken;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

/// llama.cpp's hard ceiling on `n_seq_max` (`LLAMA_MAX_SEQ` in
/// `src/llama-cparams.h`). Exceeding it makes the context constructor throw, which
/// reaches Rust as an opaque null return; clamped here so the diagnosis stays local.
const LLAMA_MAX_SEQ: u32 = 256;

/// llama.cpp rounds `n_ctx` up to a multiple of this (`GGML_PAD(n_ctx, 256)`).
/// Applied here too, so [`LlamaBackendConfig::n_ctx`] reports the size that will
/// actually be allocated.
const N_CTX_PADDING: u32 = 256;

/// Default token budget for one context, and therefore for one decode call.
///
/// 512 tokens is one maximum-length chunk, or roughly ten typical Otzaria lines. At
/// 112 KiB of KV cache per token for this model (see `per_token_kv_bytes`) that is
/// 56.0 MiB — not this context's largest allocation; see `DEFAULT_N_UBATCH`.
const DEFAULT_N_CTX: u32 = 512;

/// Default micro-batch size: the most tokens llama.cpp pushes through the graph in
/// one pass, and therefore the size of a context's compute buffer — the single
/// largest allocation this backend makes.
///
/// llama.cpp reserves that buffer from a worst-case graph built at
/// `min(n_ctx, n_ubatch)` tokens (`llama-context.cpp:441`), and the graph holds an
/// `n_ubatch * n_vocab` logits tensor an embeddings-only context never reads. 256 rather
/// than `n_ctx` cuts the reserve from 446.24 to 283.87 MiB per context.
///
/// Lower would be cheaper still and just as fast — time is flat within run-to-run spread
/// at every value tested — but 256 is the largest reduction that leaves *every* golden
/// vector bit-identical (65/65; `docs/P2_REFERENCE_VECTORS.md` §3 concurs for
/// {2048, 512, 256}), and `n_ubatch` is a deployment knob deliberately outside the
/// manifest's index identity, so a value that moved stored vectors would recreate the
/// silent-reindex hazard §3 documents for `batch_size`.
///
/// Tables, and the caveat that Metal's lazy paging realized only ~30 MiB of peak RSS out
/// of the 162 MiB reserved per context: `docs/P2_INFERENCE_SPIKE.md` §4.
const DEFAULT_N_UBATCH: u32 = 256;

/// Default number of contexts, i.e. concurrent `embed_batch_raw` calls.
///
/// **One**, because the smallest target is a phone. Two measures 1.93× the throughput
/// but costs roughly half a gigabyte while decoding, taking peak RSS from ~1.20 GB to
/// ~1.68 GB, which on a 2–3 GB Android device is a background kill rather than a slow
/// search. So concurrency is opt-in through `OTZARIA_LLAMA_CONTEXTS`; one context is not
/// a cliff, since callers queue on the pool's condvar rather than failing. Raising it
/// lowers `n_threads` — see [`LlamaBackendConfig::clamped_to_machine`].
const DEFAULT_CONTEXTS: usize = 1;

/// Default ggml thread count per context.
///
/// Capped rather than set to the core count: phones are big.LITTLE, four is llama.cpp's
/// own default and the geometry the goldens were generated with, and four is where the
/// returns stop (one batch: 7.55 s at one thread, 2.18 s at four, 2.25 s at eight).
/// `n_threads` ∈ {1,2,4,8} measured bit-identical, so it cannot change a stored vector.
/// A *cap*, not a target — see [`LlamaBackendConfig::clamped_to_machine`].
const DEFAULT_THREADS_CAP: usize = 4;

/// Default ceiling on sequences per decode call, i.e. llama.cpp's `n_seq_max`.
///
/// Not free, though it looks it: llama.cpp sizes the output buffer as
/// `n_vocab * max(n_outputs, n_seq_max)` floats and reserves the logits half even for an
/// embeddings-only context (`llama-context.cpp:2028-2048`), which for this 151 669-token
/// vocabulary is 592.5 KiB per sequence slot per context — 149.11 MiB per context at
/// llama.cpp's ceiling of 256. 32 matches [`EmbeddingConfig::batch_size`]'s default, the
/// most texts the runtime ever hands over at once, and costs 18.64 MiB.
const DEFAULT_MAX_SEQUENCES: usize = 32;

/// Tuning for [`LlamaCppBackend`], all of it optional. Separate from [`EmbeddingConfig`]
/// because that type is persisted into the manifest as an index's identity, while these
/// are deployment facts that must not invalidate an index when they change.
/// [`LlamaCppBackend::open`] takes this directly; the backend-selection table can only
/// pass an [`EmbeddingConfig`], so that path goes through [`Self::from_env_for`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlamaBackendConfig {
    /// Token budget for one context, and therefore for one decode call. Raised to
    /// `max_tokens` if it is smaller — a context that cannot hold one maximum-length
    /// sequence would reject the inputs truncation exists to make embeddable — then
    /// rounded up to a multiple of 256, which llama.cpp does anyway.
    pub n_ctx: u32,
    /// Number of contexts, hence concurrent `embed_batch_raw` calls. Clamped to at least
    /// one. Roughly half a gigabyte each while decoding.
    pub contexts: usize,
    /// ggml threads per context. Clamped to at least one, **and clamped so that
    /// `contexts * n_threads` does not exceed the machine's parallelism** — see
    /// [`Self::clamped_to_machine`], which every constructor here ends with.
    pub n_threads: usize,
    /// Most sequences in one decode call (llama.cpp's `n_seq_max`). Clamped to at
    /// least one and to `n_ctx`. Costs 592.5 KiB of output buffer per slot per
    /// context for this vocabulary (`DEFAULT_MAX_SEQUENCES`), so raise it only
    /// alongside `EmbeddingConfig::batch_size`.
    pub max_sequences_per_decode: usize,
    /// Model layers to offload to a GPU. Zero — CPU only — matches the reference build the
    /// goldens were produced on, and is not a free switch: `docs/P2_REFERENCE_VECTORS.md`
    /// §5 measures CPU-vs-Metal agreement at worst cosine 0.99491.
    pub n_gpu_layers: u32,
    /// Whether llama.cpp's own logging is forwarded to the [`log`] facade. On by default:
    /// its explanation of a failed load is far better than "null result from llama cpp",
    /// and a library must not write to the host's stderr.
    pub forward_llama_logs: bool,
}

impl Default for LlamaBackendConfig {
    fn default() -> Self {
        Self {
            n_ctx: DEFAULT_N_CTX,
            contexts: DEFAULT_CONTEXTS.min(available_parallelism()),
            n_threads: DEFAULT_THREADS_CAP.min(available_parallelism()),
            max_sequences_per_decode: DEFAULT_MAX_SEQUENCES,
            n_gpu_layers: 0,
            forward_llama_logs: true,
        }
        .clamped_to_machine()
    }
}

impl LlamaBackendConfig {
    /// Environment variable overriding [`Self::n_ctx`].
    pub const ENV_N_CTX: &'static str = "OTZARIA_LLAMA_N_CTX";
    /// Environment variable overriding [`Self::contexts`].
    pub const ENV_CONTEXTS: &'static str = "OTZARIA_LLAMA_CONTEXTS";
    /// Environment variable overriding [`Self::n_threads`].
    pub const ENV_THREADS: &'static str = "OTZARIA_LLAMA_THREADS";
    /// Environment variable overriding [`Self::n_gpu_layers`].
    pub const ENV_GPU_LAYERS: &'static str = "OTZARIA_LLAMA_GPU_LAYERS";
    /// Environment variable overriding [`Self::max_sequences_per_decode`].
    pub const ENV_MAX_SEQUENCES: &'static str = "OTZARIA_LLAMA_MAX_SEQUENCES";

    /// Lower [`Self::n_threads`] until `contexts * n_threads` fits the machine.
    ///
    /// Each [`LlamaContext`] owns its own ggml thread pool, so the real demand is the
    /// product, and oversubscribing loses throughput silently: 4 contexts × 4 threads
    /// measured 2.1× *slower* than 4 × 2 on a 10-core machine
    /// (`docs/P2_INFERENCE_SPIKE.md` §4). Every constructor here ends with this call,
    /// which is idempotent.
    ///
    /// `contexts` is deliberately *not* clamped: it is chosen against memory rather than
    /// cores, and a pool larger than the core count still serves callers correctly.
    #[must_use]
    pub fn clamped_to_machine(mut self) -> Self {
        self.contexts = self.contexts.max(1);
        self.n_threads = self.n_threads.max(1);

        let cores = available_parallelism();
        if self.contexts.saturating_mul(self.n_threads) <= cores {
            return self;
        }
        // `max(1)`: llama.cpp rejects a context with zero threads.
        let fair = (cores / self.contexts).max(1);
        if fair < self.n_threads {
            log::debug!(
                "Embedding backend '{}': {} context(s) x {} thread(s) would ask for {} threads \
                 on {cores} core(s); lowering to {fair} thread(s) per context.",
                Self::ID_FOR_LOGS,
                self.contexts,
                self.n_threads,
                self.contexts.saturating_mul(self.n_threads),
            );
            self.n_threads = fair;
        }
        self
    }

    /// Named separately from [`LlamaCppBackend::ID`] because this `impl` is reachable
    /// without a backend.
    const ID_FOR_LOGS: &'static str = LlamaCppBackend::ID;

    /// Defaults for `config`, with environment-variable overrides applied — an
    /// operational escape hatch for the backend-selection table, which can hand a
    /// constructor nothing but an [`EmbeddingConfig`].
    ///
    /// # Errors
    ///
    /// [`EmbeddingError::LoadFailed`] if a variable is set but unparseable. Someone who
    /// exported `OTZARIA_LLAMA_CONTEXTS=four` is trying to bound memory, and quietly
    /// giving them the default bounds nothing.
    pub fn from_env_for(config: &EmbeddingConfig) -> Result<Self, EmbeddingError> {
        fn read<T: std::str::FromStr>(key: &str, slot: &mut T) -> Result<(), EmbeddingError> {
            let Ok(raw) = std::env::var(key) else {
                return Ok(());
            };
            *slot = raw
                .trim()
                .parse::<T>()
                .map_err(|_| EmbeddingError::LoadFailed {
                    reason: format!(
                        "{key} is set to {raw:?}, which is not a valid value for this setting"
                    ),
                })?;
            Ok(())
        }

        let mut selected = Self {
            // A context has to hold one whole sequence, so the cap is the floor here.
            n_ctx: DEFAULT_N_CTX.max(u32::try_from(config.max_tokens).unwrap_or(u32::MAX)),
            // `batch_size` is the most sequences one decode can be asked to hold, so
            // deriving it avoids paying for slots that can never be filled.
            max_sequences_per_decode: config.batch_size.max(1),
            ..Self::default()
        };
        read(Self::ENV_N_CTX, &mut selected.n_ctx)?;
        read(Self::ENV_CONTEXTS, &mut selected.contexts)?;
        read(Self::ENV_THREADS, &mut selected.n_threads)?;
        read(Self::ENV_GPU_LAYERS, &mut selected.n_gpu_layers)?;
        read(
            Self::ENV_MAX_SEQUENCES,
            &mut selected.max_sequences_per_decode,
        )?;

        // The top 255 values of the `u32` range have no multiple of `N_CTX_PADDING` above
        // them, so rounding up panicked in debug and wrapped to zero in release. Refused
        // here, where the error can still name the variable.
        if selected
            .n_ctx
            .checked_next_multiple_of(N_CTX_PADDING)
            .is_none()
        {
            return Err(EmbeddingError::LoadFailed {
                reason: format!(
                    "{} is set to {}, which cannot be rounded up to a multiple of \
                     {N_CTX_PADDING} tokens without overflowing a 32-bit count. A context that \
                     large could not be allocated on any device in any case; the largest usable \
                     value is {}.",
                    Self::ENV_N_CTX,
                    selected.n_ctx,
                    (u32::MAX / N_CTX_PADDING) * N_CTX_PADDING,
                ),
            });
        }

        Ok(selected.clamped_to_machine())
    }
}

/// Machine parallelism, or 1 — `available_parallelism` fails on some sandboxed and
/// embedded targets, and 1 is the answer that cannot oversubscribe.
fn available_parallelism() -> usize {
    std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
}

/// The process-wide llama.cpp backend. `LlamaBackend::init` refuses a second call and
/// frees the backend when dropped, so it cannot be owned per [`LlamaCppBackend`]; the
/// `Result` is stored so a failure is reported to every caller rather than retried.
static LLAMA: OnceLock<Result<LlamaBackend, String>> = OnceLock::new();

/// Whether llama.cpp's log callback has been installed.
static LOGS_FORWARDED: AtomicBool = AtomicBool::new(false);

fn shared_backend() -> Result<&'static LlamaBackend, EmbeddingError> {
    LLAMA
        .get_or_init(|| LlamaBackend::init().map_err(|e| e.to_string()))
        .as_ref()
        .map_err(|reason| EmbeddingError::LoadFailed {
            reason: format!("llama.cpp backend initialization failed: {reason}"),
        })
}

/// Send llama.cpp's logging to the [`log`] facade instead of stderr: a library linked
/// into a Flutter application must not write to the host's stderr, and llama.cpp's own
/// diagnosis of a failed load is far more actionable than the null pointer that reaches
/// Rust. Hand-written because 0.1.153's `send_logs_to_tracing` routes to `tracing`.
fn install_log_forwarding() {
    /// Forwards one log chunk. Must not panic: it runs on llama.cpp's threads, and
    /// unwinding across `extern "C"` is undefined behaviour.
    unsafe extern "C" fn forward(
        level: llama_cpp_sys_2::ggml_log_level,
        text: *const std::os::raw::c_char,
        _user_data: *mut std::os::raw::c_void,
    ) {
        if text.is_null() {
            return;
        }
        // SAFETY: a NUL-terminated C string valid for the duration of the call, and
        // nothing here retains the pointer.
        let message = unsafe { std::ffi::CStr::from_ptr(text) };
        let message = String::from_utf8_lossy(message.to_bytes());
        let message = message.trim_end_matches(['\n', '\r']);
        if message.is_empty() {
            return;
        }
        // INFO is mapped down: llama.cpp emits the loading progress indicator a dot
        // at a time, and a line per dot at INFO would drown the log.
        let level = match level {
            llama_cpp_sys_2::GGML_LOG_LEVEL_ERROR => log::Level::Error,
            llama_cpp_sys_2::GGML_LOG_LEVEL_WARN => log::Level::Warn,
            llama_cpp_sys_2::GGML_LOG_LEVEL_INFO => log::Level::Debug,
            _ => log::Level::Trace,
        };
        log::log!(target: "llama.cpp", level, "{message}");
    }

    // Once only: `ggml_log_set` is not documented as safe to race with logging from
    // another thread.
    if LOGS_FORWARDED.swap(true, Ordering::SeqCst) {
        return;
    }
    // SAFETY: `forward` matches `ggml_log_callback`, holds no state, and the
    // null user-data pointer it will be handed back is the one passed here.
    unsafe { llama_cpp_sys_2::llama_log_set(Some(forward), std::ptr::null_mut()) };
}

pub use tokenizer::RawVocab;

/// The one thing `llama-cpp-2` cannot do: tokenize with `parse_special = false`.
///
/// # Why this module exists
///
/// `parse_special = true` promotes characters that *look* like special-token markup to
/// the control token they name, so a line containing the literal text `<|endoftext|>`
/// becomes token 151643 instead of the eight tokens those characters are. The resulting
/// vector still passes every tolerance the goldens recommend (cosine 0.99287), so only
/// the token ids catch it.
///
/// [`LlamaModel::str_to_token`] hardcodes `true` at both of its `llama_tokenize` call
/// sites (`src/model.rs:327`, `:343`), 0.1.153 is the newest release, and the crate
/// offers no `parse_special` anywhere, no vocabulary or model pointer accessor (both
/// `pub(crate)`), and no re-export of `llama-cpp-sys-2`. `llama_tokenize` itself is a
/// stable C API; reaching it needs a `*const llama_vocab`.
///
/// # The one layout assumption in this crate
///
/// The pointer is read straight out of a `&LlamaModel`, which works because 0.1.153
/// declares
/// `#[repr(transparent)] pub struct LlamaModel { pub(crate) model: NonNull<llama_model> }`
/// (`src/model.rs:25-31`).
///
/// **That layout is an undocumented implementation detail, and upstream promises
/// nothing about it** — the opposite, if anything. `LlamaModel`'s entire doc comment is
/// "A safe wrapper around llama_model" and its field is `pub(crate)`; the same crate
/// explicitly disclaims `repr(transparent)` for sibling types (`src/token/data.rs:4-9`,
/// `src/token/logit_bias.rs:4-10`: *"Do not rely on `repr(transparent)` for this type.
/// It should be considered an implementation detail and may change across minor
/// versions"*); and `src/lib.rs:3-6` says *"this crate does not attempt to create a
/// stable API"*.
///
/// So **the exact `=0.1.153` requirement in `Cargo.toml` is what makes this sound, and
/// it is doing the real work**: it freezes the layout read out of the source above.
/// Every other guard is a backstop for someone editing that pin:
///
/// 1. **The version requirement.** The resolver cannot move `llama-cpp-2` underneath
///    this code; a bump is a human edit, and `Cargo.toml` says to read this module
///    first. **Prevention**, and the only guard that is.
/// 2. **A compile-time check** — `LAYOUT_IS_STILL_A_BARE_POINTER` fails the build if
///    `LlamaModel` stops being pointer-sized or pointer-aligned. That is all: any
///    same-sized replacement passes it (an `Arc<Inner>`, a `Box`, a `usize` handle). It
///    catches the likeliest accident, a second field, and nothing subtler.
/// 3. **Two load-time cross-checks** in [`LlamaCppBackend::open`]: vocabulary size and
///    EOS id from the derived pointer against the safe API. The embedding width is not
///    compared, and a *different* model sharing this tokenizer would pass both.
/// 4. **A tokenizer probe**, also in `open`: this path and `str_to_token` must produce
///    identical ids for text with no special markup. The strongest live-pointer check,
///    since a correct Hebrew tokenization cannot come out of a wrong vocabulary, and
///    the one that would catch a silently rewritten field.
///
/// **Guards 3 and 4 are detection after the fact, not prevention.** Both run only
/// *after* the derived pointer has been handed to `llama_model_get_vocab` in C; if the
/// layout assumption were false the undefined behaviour has already happened, and these
/// report it rather than avoiding it.
///
/// # Why the `unsafe` is sound
///
/// The lifetime is structural (the `Arc<LlamaModel>` below outlives the pointer), every
/// use is a `const` read, no `&mut` is ever taken, and llama.cpp's tokenize path is
/// `const` throughout `llama-vocab.cpp` with the tokenizer built eagerly at load, so
/// there is no lazy mutation to race. `llama_vocab::tokenize` answers a too-small buffer
/// with `-(size)` and **writes nothing**, so the retry in `RawVocab::tokenize` cannot be
/// an out-of-bounds write.
///
/// The alternative rejected: a second `vocab_only` model, whose pointer needs no layout
/// assumption, but which measured 35.0 MiB of RSS permanently for a duplicate
/// vocabulary plus two handles to one file that could disagree.
mod tokenizer {
    use super::{EmbeddingError, LlamaModel, LlamaToken};

    /// Guard 2 of the four described on this module: fails the build if [`LlamaModel`] is
    /// no longer pointer-sized and pointer-aligned.
    const LAYOUT_IS_STILL_A_BARE_POINTER: () = assert!(
        std::mem::size_of::<LlamaModel>() == std::mem::size_of::<*const std::ffi::c_void>()
            && std::mem::align_of::<LlamaModel>()
                == std::mem::align_of::<*const std::ffi::c_void>()
    );

    /// A model's vocabulary, borrowed for tokenization. Holds an `Arc` of the model, not
    /// just the pointer: the vocabulary is freed with the model, so making that ownership
    /// a field stops a refactor from dropping it while a `RawVocab` is in use.
    pub struct RawVocab {
        /// Keeps the vocabulary alive. Read through [`Self::model`].
        model: std::sync::Arc<LlamaModel>,
        vocab: *const llama_cpp_sys_2::llama_vocab,
    }

    // SAFETY: `vocab` points into `model`, which the `Arc` keeps alive, and `LlamaModel`
    // is itself `Send + Sync` (llama.cpp documents a loaded model as shareable across
    // threads). Every use is a read of an immutable vocabulary — `llama_tokenize` takes it
    // as `const`, with no lazy mutation — so the pointer is as shareable as the model.
    unsafe impl Send for RawVocab {}
    unsafe impl Sync for RawVocab {}

    impl std::fmt::Debug for RawVocab {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("RawVocab").finish_non_exhaustive()
        }
    }

    impl RawVocab {
        pub(super) fn new(model: std::sync::Arc<LlamaModel>) -> Result<Self, EmbeddingError> {
            // A `const` item is only evaluated if it is used.
            let () = LAYOUT_IS_STILL_A_BARE_POINTER;

            // SAFETY: guard 1 (the exact `=0.1.153` requirement) and guard 2 (the
            // assertion above) are what make this sound. In 0.1.153 `LlamaModel` is
            // `#[repr(transparent)]` over `NonNull<llama_model>`, itself
            // `#[repr(transparent)]` over `*const llama_model`, so a `&LlamaModel` and a
            // `&*const llama_model` describe the same bytes; the read is of an
            // initialized, non-null pointer held since construction. Guards 3 and 4 in
            // `LlamaCppBackend::open` then *detect* a pointer that is not this model's,
            // after the fact rather than instead of it — see this module's docs.
            let raw: *const llama_cpp_sys_2::llama_model = unsafe {
                *std::ptr::from_ref(model.as_ref()).cast::<*const llama_cpp_sys_2::llama_model>()
            };
            if raw.is_null() {
                return Err(EmbeddingError::LoadFailed {
                    reason: "llama.cpp model pointer is null immediately after a successful load; \
                             the llama-cpp-2 layout assumption in semantic::llama_backend::tokenizer \
                             no longer holds"
                        .to_string(),
                });
            }

            // SAFETY: `raw` is a live model pointer and `llama_model_get_vocab` is a pure
            // accessor returning a pointer into it.
            let vocab = unsafe { llama_cpp_sys_2::llama_model_get_vocab(raw) };
            if vocab.is_null() {
                return Err(EmbeddingError::InvalidModelFile {
                    path: "<loaded model>".to_string(),
                    reason: "the model carries no vocabulary, so it cannot be tokenized for"
                        .to_string(),
                });
            }

            Ok(Self { model, vocab })
        }

        pub(super) fn model(&self) -> &LlamaModel {
            &self.model
        }

        /// Guard 3: the vocabulary's size and EOS id as the derived pointer reports them,
        /// returned as data so [`super::LlamaCppBackend::open`] can compare them against
        /// the safe API and report the disagreement itself.
        pub(super) fn cross_check(&self) -> (i32, i32) {
            // SAFETY: `self.vocab` is live for as long as `self.model`, which the `Arc`
            // guarantees.
            unsafe {
                (
                    llama_cpp_sys_2::llama_vocab_n_tokens(self.vocab),
                    llama_cpp_sys_2::llama_vocab_eos(self.vocab),
                )
            }
        }

        /// Tokenize `text` with `parse_special = false`.
        ///
        /// `add_special` is exposed only so [`super::TokenizerContract::verify`] can
        /// compare the two spellings of the reference pipeline; production tokenization
        /// passes `false` and appends the EOS itself.
        ///
        /// Passes a pointer and a length rather than a `CString` as
        /// [`LlamaModel::str_to_token`] does, so an interior NUL byte in book text
        /// tokenizes instead of failing the call.
        pub(super) fn tokenize(
            &self,
            text: &str,
            add_special: bool,
        ) -> Result<Vec<LlamaToken>, EmbeddingError> {
            let Ok(byte_len) = i32::try_from(text.len()) else {
                return Err(EmbeddingError::InferenceFailed {
                    reason: format!(
                        "text is {} bytes; llama.cpp's tokenizer takes a 32-bit length",
                        text.len()
                    ),
                });
            };

            // Every token consumes at least one input byte, so this is an upper bound
            // rather than a guess; `+ 2` leaves room for `add_special`.
            let mut buffer = vec![LlamaToken(0); text.len().saturating_add(2)];
            let mut written = self.tokenize_into(text, byte_len, &mut buffer, add_special);

            if written < 0 {
                // A too-small buffer is answered with the negated required length.
                let needed = written.unsigned_abs() as usize;
                buffer = vec![LlamaToken(0); needed];
                written = self.tokenize_into(text, byte_len, &mut buffer, add_special);
            }

            let Ok(written) = usize::try_from(written) else {
                return Err(EmbeddingError::InferenceFailed {
                    reason: format!(
                        "llama.cpp's tokenizer failed on a {} byte input (returned {written})",
                        text.len()
                    ),
                });
            };
            if written > buffer.len() {
                return Err(EmbeddingError::InferenceFailed {
                    reason: format!(
                        "llama.cpp's tokenizer reported {written} tokens for a buffer of {}",
                        buffer.len()
                    ),
                });
            }
            buffer.truncate(written);
            Ok(buffer)
        }

        /// One `llama_tokenize` call. Negative means "buffer too small; this much is
        /// needed".
        fn tokenize_into(
            &self,
            text: &str,
            byte_len: i32,
            buffer: &mut [LlamaToken],
            add_special: bool,
        ) -> i32 {
            let capacity = i32::try_from(buffer.len()).unwrap_or(i32::MAX);
            // SAFETY: `text`'s pointer is valid for `byte_len` bytes — llama.cpp takes an
            // explicit length and does not require NUL termination — and `buffer` is
            // writable for `capacity` elements. `LlamaToken` is `#[repr(transparent)]`
            // over `llama_token`, so the cast reinterprets the same integers.
            unsafe {
                llama_cpp_sys_2::llama_tokenize(
                    self.vocab,
                    text.as_ptr().cast::<std::os::raw::c_char>(),
                    byte_len,
                    buffer.as_mut_ptr().cast::<llama_cpp_sys_2::llama_token>(),
                    capacity,
                    add_special,
                    false,
                )
            }
        }
    }
}

/// What this backend requires of a model's tokenizer, checked against the loaded model
/// rather than assumed: no BOS, exactly one EOS appended, `parse_special = false`. A
/// model that prepends a BOS instead scores cosine 0.9948 against the correct vector,
/// *above* the legitimate cross-implementation floor, so no vector tolerance can separate
/// the two and the check has to be on ids.
struct TokenizerContract;

impl TokenizerContract {
    /// `""` matters most: `add_special` on an empty string produces *only* the special
    /// tokens, so a prepended BOS cannot hide behind content. The rest carry no
    /// special-token markup, since the comparison below is against `str_to_token`.
    const PROBES: &'static [&'static str] = &["", " ", "תורה", "The quick brown fox", "1234"];

    /// Prove that `content_tokens(text) + [EOS]` is what llama.cpp itself produces for
    /// `text`, and that the vocabulary pointer is this model's.
    ///
    /// # Errors
    ///
    /// [`EmbeddingError::InvalidModelFile`] naming the probe and both token sequences. A
    /// caller cannot fix this, so it has to be diagnosable from a log alone.
    fn verify(
        vocab: &RawVocab,
        eos: LlamaToken,
        path: &std::path::Path,
    ) -> Result<(), EmbeddingError> {
        let invalid = |reason: String| EmbeddingError::InvalidModelFile {
            path: path.display().to_string(),
            reason,
        };

        for probe in Self::PROBES {
            let content = vocab.tokenize(probe, false)?;

            // Guard 4 of the `tokenizer` module: the raw and safe paths must agree.
            // These probes carry no special markup, so `parse_special` cannot make
            // them differ, and a wrong vocabulary pointer cannot make them agree.
            let via_safe_api = vocab
                .model()
                .str_to_token(probe, AddBos::Never)
                .map_err(|e| invalid(format!("tokenizing {probe:?} failed: {e}")))?;
            if via_safe_api != content {
                return Err(invalid(format!(
                    "the vocabulary reached directly disagrees with llama-cpp-2's own \
                     tokenizer on {probe:?} ({content:?} vs {via_safe_api:?}); the \
                     layout assumption in semantic::llama_backend::tokenizer no longer holds"
                )));
            }

            // `add_special` applies *both* `add_bos_token` and `add_eos_token` from the
            // GGUF, so this one comparison rejects a model that prepends a BOS and one
            // that appends no EOS.
            let mut expected = content.clone();
            expected.push(eos);
            let with_specials = vocab.tokenize(probe, true)?;
            if with_specials != expected {
                return Err(invalid(format!(
                    "this backend appends the EOS itself and requires a tokenizer that \
                     prepends no BOS and appends exactly one EOS ({}), but the model \
                     tokenizes {probe:?} as {with_specials:?} where {expected:?} was \
                     required — the vectors would be pooled over the wrong token",
                    eos.0
                )));
            }
        }

        Ok(())
    }
}

/// Real GGUF inference: llama.cpp, Qwen3-family embedding model, last-token pooling.
/// Construct with [`LlamaCppBackend::open`].
pub struct LlamaCppBackend {
    /// The vocabulary, and through it the model. Used by [`Self::tokenize`] unlocked.
    vocab: RawVocab,
    /// Worker threads, one context each. The contexts do not borrow from this struct,
    /// which is the point of the worker design.
    pool: ContextPool,
    /// The model's own embedding width, read from the file. `EmbeddingRuntime`'s adoption
    /// check compares it against the configured dimensionality.
    dim: u32,
    /// The requested token cap, clamped to the model's trained context length and to the
    /// context actually allocated.
    max_tokens: usize,
    /// The token appended to every sequence, and therefore the one pooling reads.
    eos: LlamaToken,
}

impl LlamaCppBackend {
    /// Identifier recorded in the manifest as `embedding_backend`.
    ///
    /// A change here invalidates every stored vector, so it carries only what changes the
    /// vectors *categorically*: the implementation (`llama-cpp`), the model family whose
    /// tokenizer and pooling this wiring implements (`qwen3`), the pooling (`last`), and a
    /// wiring version bumped when the output changes for the same inputs (`v1`).
    ///
    /// Deliberately **not** the llama.cpp build, even though `docs/P2_REFERENCE_VECTORS.md`
    /// §5 measures two builds agreeing only to cosine 0.99491: that is far tighter than
    /// any useful retrieval threshold, so including it would force a full re-index on a
    /// routine dependency bump. The build is frozen by the version pin plus `Cargo.lock`,
    /// the file by the manifest's `model_checksum`.
    pub const ID: &'static str = "llama-cpp-qwen3-last-v1";

    /// Load `model_path` and bring up a pool of inference contexts.
    ///
    /// `max_tokens` is a request; [`EmbeddingBackend::max_tokens`] reports what was
    /// granted after clamping.
    ///
    /// Everything is checked before returning rather than on the first search: that the
    /// derived vocabulary pointer agrees with the safe API (guard 3 of the `tokenizer`
    /// submodule), that the tokenizer convention is the one implemented here
    /// (`TokenizerContract`), that every ~0.5 GB context allocates, and that a probe
    /// embedding comes back finite and of the right width — the only thing that proves the
    /// file is an *embedding* model rather than a generative one.
    ///
    /// # Errors
    ///
    /// [`EmbeddingError::ModelNotFound`], [`EmbeddingError::InvalidModelFile`] for a file
    /// llama.cpp will not load or whose tokenizer or pooling is not the one implemented
    /// here, or [`EmbeddingError::LoadFailed`] for a context that will not allocate or an
    /// `n_ctx` whose padding cannot be expressed.
    ///
    /// **Never a panic.** The one arithmetic path that could reach one — `n_ctx` within
    /// 255 of `u32::MAX`, settable through [`LlamaBackendConfig::ENV_N_CTX`] — is a
    /// checked conversion here and a named error in `from_env_for`.
    pub fn open(
        model_path: &std::path::Path,
        max_tokens: usize,
        tuning: &LlamaBackendConfig,
    ) -> Result<Self, EmbeddingError> {
        if !model_path.exists() {
            return Err(EmbeddingError::ModelNotFound {
                path: model_path.display().to_string(),
            });
        }
        if tuning.forward_llama_logs {
            install_log_forwarding();
        }

        // Checked here too, because a caller building the struct literally never goes
        // through `from_env_for`.
        let requested = tuning.clone();
        let tuning = &requested.clone().clamped_to_machine();
        if tuning.n_threads != requested.n_threads {
            log::warn!(
                "Embedding backend '{}': {} context(s) x {} thread(s) is {} ggml threads on a \
                 {}-core machine, which measured 2.1x *slower* than not oversubscribing. Using \
                 {} thread(s) per context instead. Lower {} if you want more threads each.",
                Self::ID,
                requested.contexts,
                requested.n_threads,
                requested.contexts.saturating_mul(requested.n_threads),
                available_parallelism(),
                tuning.n_threads,
                LlamaBackendConfig::ENV_CONTEXTS,
            );
        }

        let backend = shared_backend()?;

        // ── the model ──
        let model_params = LlamaModelParams::default().with_n_gpu_layers(tuning.n_gpu_layers);
        let model =
            LlamaModel::load_from_file(backend, model_path, &model_params).map_err(|e| {
                EmbeddingError::InvalidModelFile {
                    path: model_path.display().to_string(),
                    reason: format!(
                        "llama.cpp could not load the model: {e} (llama.cpp's own explanation is \
                     logged under the 'llama.cpp' target)"
                    ),
                }
            })?;
        let model = Arc::new(model);

        let dim = u32::try_from(model.n_embd())
            .ok()
            .filter(|d| *d > 0)
            .ok_or_else(|| EmbeddingError::InvalidModelFile {
                path: model_path.display().to_string(),
                reason: format!(
                    "the model reports an embedding width of {}; it cannot be an embedding model",
                    model.n_embd()
                ),
            })?;

        // Informational rather than fatal: the checks below test the properties that
        // decide whether the vectors are right, and a Qwen3-family model of another size
        // should not be refused over a metadata string.
        let architecture = model
            .meta_val_str("general.architecture")
            .unwrap_or_else(|_| "<absent>".to_string());
        if architecture != "qwen3" {
            log::warn!(
                "Embedding backend '{}' was written against the qwen3 architecture, but {} \
                 declares '{architecture}'. The tokenizer and pooling checks below decide \
                 whether it can be served; this is a note, not a refusal.",
                Self::ID,
                model_path.display()
            );
        }

        // ── the tokenizer ──
        let vocab = RawVocab::new(Arc::clone(&model))?;
        let eos = model.token_eos();

        // Guard 3: a pointer that is not this model's cannot agree with the safe API.
        let (raw_vocab_size, raw_eos) = vocab.cross_check();
        if raw_vocab_size != model.n_vocab() || raw_eos != eos.0 {
            return Err(EmbeddingError::LoadFailed {
                reason: format!(
                    "the vocabulary reached directly reports {raw_vocab_size} tokens and EOS \
                     {raw_eos}, while llama-cpp-2 reports {} and {} for the same model; the \
                     layout assumption in semantic::llama_backend::tokenizer no longer holds",
                    model.n_vocab(),
                    eos.0
                ),
            });
        }
        TokenizerContract::verify(&vocab, eos, model_path)?;

        // ── sizing ──
        //
        // Clamped to the trained context length and reported, so a caller sees what it
        // got rather than what it asked for.
        let trained = model.n_ctx_train().max(1) as usize;
        let mut effective_max_tokens = max_tokens.clamp(1, trained);
        if effective_max_tokens < max_tokens {
            log::info!(
                "Embedding backend '{}': max_tokens {max_tokens} exceeds the model's trained \
                 context length {trained}; using {effective_max_tokens}.",
                Self::ID
            );
        }

        // A context must hold one whole sequence, so `n_ctx` is raised to the cap rather
        // than the cap lowered, which would silently shorten every chunk.
        //
        // `checked_next_multiple_of`: the top 255 `u32` values have no multiple of 256
        // above them, and the unchecked form panicked in debug and wrapped to zero in
        // release, then panicked in `n_seq_max`'s `clamp(1, 0)`.
        let requested_n_ctx = u32::try_from(effective_max_tokens)
            .unwrap_or(u32::MAX)
            .max(tuning.n_ctx)
            .max(1);
        let Some(n_ctx) = requested_n_ctx.checked_next_multiple_of(N_CTX_PADDING) else {
            return Err(EmbeddingError::LoadFailed {
                reason: format!(
                    "a context budget of {requested_n_ctx} tokens cannot be rounded up to a \
                     multiple of {N_CTX_PADDING}, which llama.cpp requires, without overflowing \
                     a 32-bit count; the largest usable value is {}",
                    (u32::MAX / N_CTX_PADDING) * N_CTX_PADDING,
                ),
            });
        };
        // Only shrinks where the cap exceeds a padded `u32` context; kept so the two
        // can never disagree.
        effective_max_tokens = effective_max_tokens.min(n_ctx as usize);

        // `kv_unified = true` below makes `n_ctx` one shared pool any sequence may draw
        // from. Without it llama.cpp gives each sequence a fixed `n_ctx / n_seq_max`
        // slice, which at `n_ctx` 512 and its ceiling of 256 sequences caps an input at
        // 2 tokens. `n_seq_max` itself is bounded by the request, by how many sequences
        // could physically fit, and by that ceiling.
        let n_seq_max = u32::try_from(tuning.max_sequences_per_decode)
            .unwrap_or(LLAMA_MAX_SEQ)
            .clamp(1, n_ctx.min(LLAMA_MAX_SEQ));
        let n_threads = i32::try_from(tuning.n_threads.max(1)).unwrap_or(i32::MAX);
        let n_ubatch = micro_batch_for(&model, n_ctx);
        let context_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(n_ctx))
            .with_n_batch(n_ctx)
            // Smaller than `n_batch`, so a decode *is* split: `DEFAULT_N_UBATCH` for the
            // ~162 MiB this saves, `micro_batch_for` for when splitting is legal.
            .with_n_ubatch(n_ubatch)
            .with_n_seq_max(n_seq_max)
            .with_kv_unified(true)
            .with_n_threads(n_threads)
            .with_n_threads_batch(n_threads)
            .with_embeddings(true)
            .with_pooling_type(LlamaPoolingType::Last);

        // ── the contexts ──
        let pool = ContextPool::spawn(
            &model,
            backend,
            &context_params,
            n_ctx,
            n_seq_max,
            dim,
            tuning.contexts.max(1),
        )?;

        let backend = Self {
            vocab,
            pool,
            dim,
            max_tokens: effective_max_tokens,
            eos,
        };

        // Check 5: only running the graph proves the model *embeds*.
        backend.verify_produces_embeddings(model_path)?;

        log::info!(
            "Embedding backend '{}' ready: {} — dim {dim}, max_tokens {effective_max_tokens} \
             (model trained for {trained}), n_ctx {n_ctx}, n_ubatch {n_ubatch}, \
             n_seq_max {n_seq_max}, {} context(s) x {n_threads} thread(s), {} GPU layer(s), \
             EOS {}, about {} MiB of KV cache per context",
            Self::ID,
            model_path.display(),
            tuning.contexts.max(1),
            tuning.n_gpu_layers,
            backend.eos.0,
            estimate_kv_mib(&model, n_ctx),
        );
        Ok(backend)
    }

    /// Run one probe through the whole path and insist the result is a usable vector.
    ///
    /// A generative GGUF loads perfectly well and then answers a pooled-embedding request
    /// with a null pointer; without this, that surfaces on the first indexing batch,
    /// after the manifest has been written. The magnitude check is loose on purpose: it
    /// asserts that something ran, since a buffer of zeros passes every other check.
    fn verify_produces_embeddings(&self, path: &std::path::Path) -> Result<(), EmbeddingError> {
        let invalid = |reason: String| EmbeddingError::InvalidModelFile {
            path: path.display().to_string(),
            reason,
        };

        let probed = self
            .embed_batch_raw(&["בראשית"])
            .map_err(|e| invalid(format!("the model produced no pooled embedding: {e}")))?;
        let Some(vector) = probed.first() else {
            return Err(invalid(
                "the model returned no vector for a one-word probe".to_string(),
            ));
        };
        if vector.len() as u32 != self.dim {
            return Err(invalid(format!(
                "the model reports an embedding width of {} but produced a {}-component vector",
                self.dim,
                vector.len()
            )));
        }
        let norm = vector
            .iter()
            .map(|x| f64::from(*x) * f64::from(*x))
            .sum::<f64>()
            .sqrt();
        if !norm.is_finite() || norm <= 0.0 {
            return Err(invalid(format!(
                "the model produced a vector of magnitude {norm} for a one-word probe; it is \
                 not usable as an embedding model"
            )));
        }
        log::debug!(
            "Embedding backend '{}': probe embedding has raw L2 norm {norm:.4} \
             (the golden corpus for this model spans 69.06–104.88)",
            Self::ID
        );
        Ok(())
    }

    /// Release every inference context now, without waiting for `Drop`.
    ///
    /// **A host does not need to call this**: `release_contexts_at_exit` handles process
    /// exit, including for a backend held in a `static`. This is the door for the two
    /// cases it cannot cover — freeing half a gigabyte per context while the process keeps
    /// running, and a host exiting through `_exit`, `abort` or another path that skips C
    /// `atexit` handlers.
    ///
    /// Afterwards [`EmbeddingBackend::embed_batch_raw`] fails with
    /// [`EmbeddingError::InferenceFailed`] while [`EmbeddingBackend::tokenize`] keeps
    /// working. Calling it twice is a no-op. Blocks for at most `EXIT_TEARDOWN_BUDGET`,
    /// since a context mid-decode only sees the request when that decode finishes.
    pub fn shutdown(&self) -> usize {
        self.pool.channels.release_all(EXIT_TEARDOWN_BUDGET)
    }
}

impl std::fmt::Debug for LlamaCppBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaCppBackend")
            .field("id", &Self::ID)
            .field("dim", &self.dim)
            .field("max_tokens", &self.max_tokens)
            .field("eos", &self.eos.0)
            .field("contexts", &self.pool.size())
            .finish()
    }
}

impl EmbeddingBackend for LlamaCppBackend {
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
        Pooling::LastToken
    }

    /// The exact sequence [`Self::embed_batch_raw`] will feed to the model. The goldens
    /// name this the primary correctness gate, so it must be the *same* computation
    /// inference uses — both go through `Self::token_ids`.
    fn tokenize(&self, text: &str) -> Result<Vec<u32>, EmbeddingError> {
        Ok(self
            .token_ids(text)?
            .into_iter()
            // Ids are non-negative for every real vocabulary, and the trait's `u32` could
            // not carry a negative sentinel anyway.
            .map(|token| token.0 as u32)
            .collect())
    }

    /// Embed one batch, raw and unnormalized, one vector per input in input order.
    ///
    /// The runtime checks the count but cannot check the order, and a reordering here
    /// would attach every book's text to another book's line unnoticed. Order is
    /// therefore structural: sequence `i` of a decode is the `i`th input and is read back
    /// by the same index. See `batched_vectors_come_back_in_input_order`.
    ///
    /// Neither normalized nor screened — `EmbeddingRuntime::embed_batch` does both.
    fn embed_batch_raw(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let sequences = texts
            .iter()
            .map(|text| self.token_ids(text))
            .collect::<Result<Vec<_>, _>>()?;
        self.pool.embed(sequences)
    }
}

impl LlamaCppBackend {
    /// Build the token sequence for `text`: content, truncated to `max_tokens - 1`, then
    /// the EOS. Shared by [`EmbeddingBackend::tokenize`] and
    /// [`EmbeddingBackend::embed_batch_raw`] so that what the goldens assert is what
    /// inference consumes. Never fails on account of length.
    fn token_ids(&self, text: &str) -> Result<Vec<LlamaToken>, EmbeddingError> {
        let content = self.vocab.tokenize(text, false)?;
        Ok(truncate_with_eos(content, self.max_tokens, self.eos))
    }
}

/// Apply the truncation convention: keep at most `max_tokens - 1` content tokens, then
/// append `eos`. A free function so the arithmetic can be tested without the 396 MB
/// model, which matters because the vector tolerances provably cannot see a truncation
/// bug — an off-by-one scores cosine 0.99838 against the correct vector.
///
/// The order is load-bearing: truncate, *then* push. Pushing first would drop the EOS for
/// any over-long input, which is the bug that makes last-token pooling read a content
/// token.
///
/// Nothing checks whether `content` already ends in an EOS, which is **only safe because
/// `RawVocab::tokenize` passes `parse_special = false`** and the tokenizer therefore
/// cannot emit a control token at all. If that ever becomes `true`, this needs a
/// `content.last() != Some(&eos)` guard, and the unit test will not say so: `content()`
/// never generates the EOS, so its "exactly one EOS" assertion is vacuous here.
fn truncate_with_eos(
    mut content: Vec<LlamaToken>,
    max_tokens: usize,
    eos: LlamaToken,
) -> Vec<LlamaToken> {
    // `max_tokens` counts the EOS, so the content budget is one less. `saturating_sub` for
    // a `max_tokens` of 0: at worst this yields `[eos]`, never an EOS-less sequence.
    content.truncate(max_tokens.saturating_sub(1));
    content.push(eos);
    content
}

// ─────────────────────────────── the pool ───────────────────────────────

/// What a worker thread can be asked to do.
enum Job {
    Embed {
        /// Token sequences, already truncated and EOS-terminated, in caller order.
        sequences: Vec<Vec<LlamaToken>>,
        /// Exactly one reply per `Job`, by protocol rather than by type.
        reply: Sender<Result<Vec<Vec<f32>>, EmbeddingError>>,
    },
    /// Drop this context now, acknowledge on the channel, and stop serving. The
    /// acknowledgement is sent *after* the [`LlamaContext`] has been dropped, so
    /// `release_contexts_at_exit` can know the GPU buffers are gone rather than hoping.
    Release(Sender<()>),
}

/// Every live worker's job channel, reachable from its [`ContextPool`] and from the
/// process-exit teardown. One shared owner rather than a clone in each place: a stray
/// `Sender` clone keeps a channel open, and an open channel keeps
/// [`ContextPool::drop`]'s `join` waiting forever.
struct WorkerChannels {
    /// Indexed by worker. `None` once that worker has been asked to release its context,
    /// which is how a second teardown, or one racing `Drop`, becomes a no-op.
    senders: Mutex<Vec<Option<Sender<Job>>>>,
}

impl WorkerChannels {
    fn sender(&self, index: usize) -> Option<Sender<Job>> {
        self.senders
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(index)?
            .clone()
    }

    /// Ask every worker to drop its context, and wait up to `budget` in total,
    /// returning the number that did not confirm. Bounded rather than a plain `join`
    /// because a worker mid-decode only sees the message when that decode finishes.
    fn release_all(&self, budget: std::time::Duration) -> usize {
        let mut waiting = Vec::new();
        {
            let mut senders = self
                .senders
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            for slot in senders.iter_mut() {
                let Some(jobs) = slot.take() else { continue };
                let (ack, acked) = mpsc::channel();
                if jobs.send(Job::Release(ack)).is_ok() {
                    waiting.push(acked);
                }
                // `jobs` is dropped here. `mpsc` still delivers the queued `Release`,
                // and a worker that misses it ends its loop on the closed channel.
            }
        }

        let deadline = std::time::Instant::now() + budget;
        waiting
            .into_iter()
            .filter(|acked| {
                let left = deadline.saturating_duration_since(std::time::Instant::now());
                acked.recv_timeout(left).is_err()
            })
            .count()
    }

    /// Close every channel without waiting. [`ContextPool::drop`] joins the threads
    /// itself, so it needs the channels shut rather than an acknowledgement.
    fn close_all(&self) {
        self.senders
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter_mut()
            .for_each(|slot| {
                slot.take();
            });
    }
}

// ──────────────────── teardown before the process exits ────────────────────

/// Every pool that still exists. `Weak`, so a pool dropped normally leaves nothing to
/// tear down and no memory held; entries are swept on each registration.
static LIVE_POOLS: Mutex<Vec<std::sync::Weak<WorkerChannels>>> = Mutex::new(Vec::new());

/// Whether `release_contexts_at_exit` has been registered with the C runtime.
static EXIT_TEARDOWN_INSTALLED: AtomicBool = AtomicBool::new(false);

/// How long the exit hook will wait, in total, for contexts to be released.
///
/// One batch in flight, generously: a 512-token decode measured about 0.75 s and a
/// full 32-text batch about 4 s. An exit that hangs is worse than one that aborts, so
/// the hook gives up and reports rather than waiting indefinitely.
const EXIT_TEARDOWN_BUDGET: std::time::Duration = std::time::Duration::from_secs(6);

/// Register `channels` for teardown at process exit, installing the hook once.
fn register_for_exit_teardown(channels: &Arc<WorkerChannels>) {
    if let Ok(mut live) = LIVE_POOLS.lock() {
        live.retain(|pool| pool.strong_count() > 0);
        live.push(Arc::downgrade(channels));
    }

    if EXIT_TEARDOWN_INSTALLED.swap(true, Ordering::SeqCst) {
        return;
    }

    extern "C" {
        /// `int atexit(void (*)(void))`. Declared here rather than depending on
        /// `libc` for one symbol; the C runtime is linked on every target.
        fn atexit(handler: extern "C" fn()) -> std::os::raw::c_int;
    }

    // SAFETY: the function passed matches `extern "C" fn()` exactly, it is registered
    // once (the swap above), and `release_contexts_at_exit` catches its own unwinds, so
    // nothing can propagate across the FFI boundary.
    let registered = unsafe { atexit(release_contexts_at_exit) };
    if registered != 0 {
        log::warn!(
            "Embedding backend '{}': the C runtime refused an exit handler ({registered}). On \
             macOS and iOS the process will abort in ggml's Metal teardown if the backend is \
             still alive at exit; call LlamaCppBackend::shutdown before exiting.",
            LlamaCppBackend::ID
        );
    }
}

/// Release every live context before the process's static destructors run.
///
/// ggml frees its Metal device from a C++ static destructor whose teardown asserts that
/// no residency sets remain registered (`GGML_ASSERT([rsets->data count] == 0) failed`
/// at `ggml/src/ggml-metal/ggml-metal-device.m:622`). A live [`LlamaContext`] still
/// holds them, so the process aborts with SIGABRT *after* every batch has succeeded.
/// `Drop` cannot cover the two shapes that matter: a backend held in a `static`, which
/// is never dropped and is what `otzaria_search_engine` does, and `process::exit` while
/// holding one. Both now exit 0.
///
/// An `atexit` handler runs in an ordinary calling context, not a signal handler, so
/// locks, allocation, channels and blocking are all permitted. The two real rules are
/// obeyed — nothing here calls `exit` or `_exit`, and the body is wrapped in
/// [`std::panic::catch_unwind`] so nothing unwinds across the `extern "C"` boundary.
/// Re-entry is a no-op, since each channel is `take`n.
///
/// It also runs early enough by the C++ ABI rather than by coincidence: ggml's Metal
/// device lives in a function-local `static` whose destructor is registered with
/// `__cxa_atexit` at backend registration, before any context exists, and the two handler
/// lists run interleaved in reverse order of registration — hence the registration from
/// [`ContextPool::spawn`], *after* the first context is up.
///
/// Nothing runs on `_exit`, `abort`, a signal or an Android process kill, but those paths
/// also skip the static destructor holding the assert.
extern "C" fn release_contexts_at_exit() {
    // Unwinding out of an `extern "C"` function is undefined behaviour, so the
    // possibility is removed rather than reasoned about.
    let _ = std::panic::catch_unwind(|| {
        let pools = match LIVE_POOLS.lock() {
            Ok(mut live) => std::mem::take(&mut *live),
            Err(_) => return,
        };
        let mut unconfirmed = 0usize;
        for pool in pools {
            if let Some(channels) = pool.upgrade() {
                unconfirmed += channels.release_all(EXIT_TEARDOWN_BUDGET);
            }
        }
        if unconfirmed > 0 {
            // Not fatal: on a non-Apple platform a context left alive at exit is
            // merely memory the kernel is about to reclaim.
            log::warn!(
                "Embedding backend '{}': {unconfirmed} inference context(s) did not confirm \
                 release within {EXIT_TEARDOWN_BUDGET:?} at process exit.",
                LlamaCppBackend::ID
            );
        }
    });
}

/// What the pool's mutex guards — note that no context and no model are here. It is
/// taken to move a `usize` and released before any inference starts.
struct PoolState {
    idle: Vec<usize>,
    /// Workers still able to serve. Decremented when one dies, so a pool that has lost
    /// every thread returns an error instead of blocking a caller forever.
    alive: usize,
}

/// A bounded set of worker threads, each owning one [`LlamaContext`].
///
/// Threads rather than a `Vec` of contexts behind a semaphore because `LlamaContext<'a>`
/// borrows the [`LlamaModel`] it was created from, so storing the two together is
/// self-referential and needs `unsafe`. Creating the context on the worker's own stack
/// keeps the borrow local to that frame, and respects `LlamaContext` being `!Sync`.
struct ContextPool {
    /// Shared with [`LIVE_POOLS`], so the process-exit teardown can reach the workers
    /// without owning the pool.
    channels: Arc<WorkerChannels>,
    /// One per worker, in the same order as [`WorkerChannels::senders`].
    threads: Vec<std::thread::JoinHandle<()>>,
    state: Mutex<PoolState>,
    /// Signalled when a worker is returned *or* dies; a waiter re-examines both.
    available: Condvar,
}

impl ContextPool {
    /// Create `count` workers and wait until every one has its context.
    ///
    /// Waiting is the point: a context is ~0.5 GB and allocation failure is a real
    /// outcome on a phone, so learning about it during the first search would turn a
    /// clear out-of-memory error into an intermittent one.
    ///
    /// Registers the pool for `release_contexts_at_exit` on the way out, which must happen
    /// *after* the first context exists for the ordering argument there to hold.
    #[allow(clippy::too_many_arguments)]
    fn spawn(
        model: &Arc<LlamaModel>,
        backend: &'static LlamaBackend,
        params: &LlamaContextParams,
        n_ctx: u32,
        n_seq_max: u32,
        dim: u32,
        count: usize,
    ) -> Result<Self, EmbeddingError> {
        let mut senders = Vec::with_capacity(count);
        let mut threads = Vec::with_capacity(count);
        let mut failure = None;

        for index in 0..count {
            let (jobs_tx, jobs_rx) = mpsc::channel::<Job>();
            let (ready_tx, ready_rx) = mpsc::channel::<Result<(), String>>();
            let model = Arc::clone(model);
            let params = params.clone();

            // A failed `spawn` takes the same `failure` path as a context that will not
            // allocate rather than returning `Err` here: `ContextPool` is not constructed
            // yet, so its `Drop` cannot run, and every *earlier* worker would be left
            // holding hundreds of megabytes with nothing owning it and no `join` — in the
            // failure that happens *because* memory is short.
            let thread = match std::thread::Builder::new()
                .name(format!("otzaria-embed-{index}"))
                .spawn(move || {
                    worker_loop(
                        &model, backend, params, n_ctx, n_seq_max, dim, &ready_tx, &jobs_rx,
                    );
                }) {
                Ok(thread) => thread,
                Err(e) => {
                    failure = Some(format!("could not start inference thread {index}: {e}"));
                    break;
                }
            };

            match ready_rx.recv() {
                Ok(Ok(())) => {
                    senders.push(Some(jobs_tx));
                    threads.push(thread);
                }
                Ok(Err(reason)) => {
                    failure = Some(reason);
                    break;
                }
                Err(_) => {
                    failure = Some(format!(
                        "inference thread {index} stopped before reporting readiness"
                    ));
                    break;
                }
            }
        }

        let alive = threads.len();
        let pool = Self {
            channels: Arc::new(WorkerChannels {
                senders: Mutex::new(senders),
            }),
            threads,
            state: Mutex::new(PoolState {
                idle: if failure.is_some() {
                    Vec::new()
                } else {
                    (0..alive).collect()
                },
                alive: if failure.is_some() { 0 } else { alive },
            }),
            available: Condvar::new(),
        };

        if let Some(reason) = failure {
            // Dropping the pool closes the channels and *joins*, so the memory is
            // provably gone before this function returns rather than whenever the
            // threads notice. Never registered for exit teardown on this path.
            drop(pool);
            return Err(EmbeddingError::LoadFailed {
                reason: format!("could not bring up an inference context pool: {reason}"),
            });
        }

        register_for_exit_teardown(&pool.channels);
        Ok(pool)
    }

    fn size(&self) -> usize {
        self.threads.len()
    }

    /// Run one batch on whichever context is free, waiting if none is.
    ///
    /// The lock is held for the checkout and the return, never across the decode, so
    /// two callers with two contexts make genuinely independent progress — the
    /// property `&self` on [`EmbeddingBackend::embed_batch_raw`] exists to allow.
    fn embed(&self, sequences: Vec<Vec<LlamaToken>>) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let index = self.check_out()?;
        let (reply_tx, reply_rx) = mpsc::channel();
        let job = Job::Embed {
            sequences,
            reply: reply_tx,
        };

        // A missing channel (the teardown released it, which a caller racing shutdown can
        // legitimately see), a failed send and a failed receive all mean this worker will
        // not answer. It is *not* returned to the pool: a pool of dead workers would look
        // permanently busy and block callers forever.
        let outcome = match self.channels.sender(index) {
            Some(jobs) => match jobs.send(job) {
                Ok(()) => reply_rx.recv().map_err(|_| ()),
                Err(_) => Err(()),
            },
            None => Err(()),
        };

        match outcome {
            Ok(result) => {
                self.check_in(index, true);
                result
            }
            Err(()) => {
                self.check_in(index, false);
                Err(EmbeddingError::InferenceFailed {
                    reason: format!(
                        "inference context {index} stopped responding; {} of {} contexts remain",
                        self.remaining(),
                        self.threads.len()
                    ),
                })
            }
        }
    }

    fn check_out(&self) -> Result<usize, EmbeddingError> {
        let mut state = self.state.lock().map_err(|_| Self::poisoned())?;
        loop {
            if let Some(index) = state.idle.pop() {
                return Ok(index);
            }
            if state.alive == 0 {
                return Err(EmbeddingError::InferenceFailed {
                    reason: "every inference context has stopped responding; the embedding \
                             backend has to be reloaded"
                        .to_string(),
                });
            }
            state = self.available.wait(state).map_err(|_| Self::poisoned())?;
        }
    }

    /// `notify_all` rather than `notify_one` for the retirement case: every waiter has
    /// to learn that `alive` reached zero, not just one of them.
    fn check_in(&self, index: usize, healthy: bool) {
        if let Ok(mut state) = self.state.lock() {
            if healthy {
                state.idle.push(index);
            } else {
                state.alive = state.alive.saturating_sub(1);
            }
            drop(state);
            self.available.notify_all();
        }
    }

    fn remaining(&self) -> usize {
        self.state.lock().map_or(0, |state| state.alive)
    }

    fn poisoned() -> EmbeddingError {
        EmbeddingError::InferenceFailed {
            reason: "the inference context pool's lock is poisoned; a worker panicked while \
                     holding it and the backend has to be reloaded"
                .to_string(),
        }
    }
}

impl Drop for ContextPool {
    /// Close the channels, then wait for the contexts to be freed.
    ///
    /// Joining is not politeness: each worker holds an `Arc<LlamaModel>` and a context
    /// borrowed from it, and `EmbeddingRuntime::load` can be called again, so leaving the
    /// threads running leaves hundreds of megabytes behind a dropped backend.
    ///
    /// Closing through [`WorkerChannels::close_all`] rather than by dropping a field is
    /// load-bearing: `self.channels` is an `Arc` whose fields are dropped only *after* this
    /// body returns, so a `Sender` left inside would keep a worker blocked in `recv` and
    /// the `join` below would never return.
    fn drop(&mut self) {
        self.channels.close_all();
        for thread in std::mem::take(&mut self.threads) {
            if let Err(panic) = thread.join() {
                // Logged rather than propagated: a panic while dropping would mask
                // whatever the caller was doing.
                log::error!("An inference worker panicked before shutting down: {panic:?}");
            }
        }
    }
}

/// One worker: create a context, then serve batches until the channel closes.
#[allow(clippy::too_many_arguments)]
fn worker_loop(
    model: &Arc<LlamaModel>,
    backend: &'static LlamaBackend,
    params: LlamaContextParams,
    n_ctx: u32,
    n_seq_max: u32,
    dim: u32,
    ready: &Sender<Result<(), String>>,
    jobs: &Receiver<Job>,
) {
    // The borrow of `model` lives entirely on this stack frame, which is why nothing
    // here is self-referential. See `ContextPool`.
    let mut context = match model.new_context(backend, params) {
        Ok(context) => context,
        Err(e) => {
            let _ = ready.send(Err(format!(
                "llama.cpp refused a context of {n_ctx} tokens: {e} (about {} MiB of KV cache, \
                 plus a compute buffer and an output buffer that are together several times \
                 larger; lower {} if the device cannot spare it)",
                estimate_kv_mib(model, n_ctx),
                LlamaBackendConfig::ENV_N_CTX
            )));
            return;
        }
    };

    // One allocation for this thread's lifetime: a batch can hold at most `n_ctx`
    // tokens, because that is the whole KV budget.
    let mut batch = LlamaBatch::new(n_ctx as usize, i32::try_from(n_seq_max).unwrap_or(i32::MAX));

    if ready.send(Ok(())).is_err() {
        return; // `spawn` gave up on us; nothing to serve.
    }

    // A break value rather than `while let`, so the context is dropped *before* the
    // acknowledgement: a teardown that got the ack early would have learned nothing.
    let release_ack = loop {
        match jobs.recv() {
            Ok(Job::Embed { sequences, reply }) => {
                let result = run_batch(&mut context, &mut batch, &sequences, n_ctx, n_seq_max, dim);
                // The caller may have gone away; not this thread's problem.
                let _ = reply.send(result);
            }
            Ok(Job::Release(ack)) => break Some(ack),
            // The pool was dropped; falling out drops the context just the same.
            Err(_) => break None,
        }
    };

    drop(batch);
    drop(context);
    if let Some(ack) = release_ack {
        // Sent after the context is gone, which is the contract. A failed send means
        // the teardown timed out and stopped listening.
        let _ = ack.send(());
    }
}

/// How much KV cache a context of `n_ctx` tokens needs, in MiB — for a log line and an
/// error message, where telling a 56 MiB request from a 1.8 GiB one is the distinction
/// a reader needs.
fn estimate_kv_mib(model: &LlamaModel, n_ctx: u32) -> u64 {
    per_token_kv_bytes(model).saturating_mul(u64::from(n_ctx)) / (1024 * 1024)
}

/// Bytes of KV cache one token costs: K and V, one row of `n_head_kv * head_dim` each
/// per layer, at f16. For this model, 112 KiB.
///
/// The head dimension is read from the file rather than derived as `n_embd / n_head`,
/// because llama.cpp only *seeds* it that way and then overrides it from
/// `ATTENTION_KEY_LENGTH` / `ATTENTION_VALUE_LENGTH` when the GGUF provides them
/// (`llama-model.cpp:1146-1151`). This model declares both as 128 where `1024 / 16` is 64,
/// so the derivation is wrong by 2× on the one model this function describes. When the
/// keys are absent the derivation is used and the result is a **lower bound**.
fn per_token_kv_bytes(model: &LlamaModel) -> u64 {
    let architecture = model
        .meta_val_str("general.architecture")
        .unwrap_or_default();
    let derived = || u64::try_from(model.n_embd()).unwrap_or(0) / u64::from(model.n_head().max(1));
    let declared = |key: &str| -> Option<u64> {
        model
            .meta_val_str(&format!("{architecture}.attention.{key}"))
            .ok()?
            .trim()
            .parse()
            .ok()
    };
    let head_dim_k = declared("key_length").unwrap_or_else(derived);
    let head_dim_v = declared("value_length").unwrap_or_else(derived);
    // 2 bytes per f16 element.
    u64::from(model.n_layer())
        .saturating_mul(u64::from(model.n_head_kv()))
        .saturating_mul(head_dim_k.saturating_add(head_dim_v))
        .saturating_mul(2)
}

/// The micro-batch size to run a context of `n_ctx` tokens at: `DEFAULT_N_UBATCH` when
/// llama.cpp will attend causally over this model, `n_ctx` when it will not.
///
/// The branch is not optional: for a non-causal model a decode larger than `n_ubatch`
/// **aborts the process** — `GGML_ASSERT((cparams.causal_attn || cparams.n_ubatch >=
/// n_tokens_all))` at `llama-context.cpp:1714` — and it would abort on the first real
/// batch rather than at load, since the probe in `verify_produces_embeddings` is two
/// tokens long.
///
/// `hparams.causal_attn` starts `true` (`llama-hparams.h:180`) and in 0.1.153 exactly two
/// things can make it false, both checked below: the GGUF key `{arch}.attention.causal`
/// (`llama-model.cpp:1045`), and the two architectures that hard-code it, `dream` and
/// `gemma-embedding`. The exact `=0.1.153` requirement is what makes "exactly two" a fact;
/// **a version bump should re-run `grep -rn 'causal_attn = false' llama.cpp/src/models/`.**
/// Anything that cannot be positively established falls back to `n_ctx`, so the failure
/// direction is wasted memory rather than an abort.
fn micro_batch_for(model: &LlamaModel, n_ctx: u32) -> u32 {
    /// Architectures the pinned llama.cpp hard-codes as non-causal.
    const NON_CAUSAL_ARCHITECTURES: [&str; 2] = ["dream", "gemma-embedding"];

    let architecture = model
        .meta_val_str("general.architecture")
        .unwrap_or_default();

    let causal = if NON_CAUSAL_ARCHITECTURES.contains(&architecture.as_str()) {
        false
    } else {
        match model.meta_val_str(&format!("{architecture}.attention.causal")) {
            // llama.cpp renders a GGUF boolean as "true"/"false"; the safe reading of
            // anything unanticipated is "do not assume causal".
            Ok(declared) => matches!(declared.trim(), "true" | "1"),
            // Absent, which is llama.cpp's default of causal.
            Err(_) => true,
        }
    };

    if causal {
        DEFAULT_N_UBATCH.min(n_ctx)
    } else {
        log::info!(
            "Embedding backend '{}': architecture '{architecture}' attends non-causally, so a \
             decode cannot be split into micro-batches (llama.cpp asserts \
             n_ubatch >= n_tokens for it). Using n_ubatch = n_ctx = {n_ctx}, which costs about \
             {} MiB more of compute buffer per context.",
            LlamaCppBackend::ID,
            u64::from(n_ctx - DEFAULT_N_UBATCH.min(n_ctx))
                .saturating_mul(u64::try_from(model.n_vocab()).unwrap_or(0))
                .saturating_mul(4)
                / (1024 * 1024),
        );
        n_ctx
    }
}

/// Decode `sequences` and collect one pooled vector per sequence, in order.
///
/// Splits into as many decode calls as the context budget requires. Not an
/// optimization: the runtime hands over up to `EmbeddingConfig::batch_size` texts,
/// which at the 512-token cap could be 16 384 tokens, so without grouping a large
/// batch of long chunks would simply fail to decode.
fn run_batch(
    context: &mut LlamaContext<'_>,
    batch: &mut LlamaBatch<'_>,
    sequences: &[Vec<LlamaToken>],
    n_ctx: u32,
    n_seq_max: u32,
    dim: u32,
) -> Result<Vec<Vec<f32>>, EmbeddingError> {
    let budget = n_ctx as usize;
    let max_sequences = n_seq_max as usize;
    let mut out = Vec::with_capacity(sequences.len());

    let mut group_start = 0usize;
    while group_start < sequences.len() {
        let mut group_end = group_start;
        let mut tokens = 0usize;
        while group_end < sequences.len() {
            let length = sequences[group_end].len();
            if length > budget {
                // Unreachable, since truncation caps every sequence at
                // `max_tokens <= n_ctx` — but the alternative is a silent wrong answer.
                return Err(EmbeddingError::InferenceFailed {
                    reason: format!(
                        "a sequence of {length} tokens does not fit a context of {budget}; \
                         truncation should have prevented this"
                    ),
                });
            }
            if group_end > group_start
                && (tokens + length > budget || group_end - group_start >= max_sequences)
            {
                break;
            }
            tokens += length;
            group_end += 1;
        }

        decode_group(
            context,
            batch,
            &sequences[group_start..group_end],
            dim,
            &mut out,
        )?;
        group_start = group_end;
    }

    Ok(out)
}

/// One `llama_decode` over one group of sequences.
fn decode_group(
    context: &mut LlamaContext<'_>,
    batch: &mut LlamaBatch<'_>,
    group: &[Vec<LlamaToken>],
    dim: u32,
    out: &mut Vec<Vec<f32>>,
) -> Result<(), EmbeddingError> {
    // Every group starts each sequence at position 0, so cells left by the previous group
    // would collide with the new ones and be attended to. This drops the cells' metadata
    // without zeroing the data buffers — bookkeeping, not a 56 MiB memset per decode.
    context
        .clear_kv_cache_seq(None, None, None)
        .map_err(|e| EmbeddingError::InferenceFailed {
            reason: format!("could not reset the KV cache between batches: {e}"),
        })?;

    batch.clear();
    for (seq_id, sequence) in group.iter().enumerate() {
        // `logits_all = false` still flags each sequence's final token, which is what makes
        // llama.cpp compute an output at all; pooling then writes one vector per sequence.
        let seq_id = i32::try_from(seq_id).map_err(|_| EmbeddingError::InferenceFailed {
            reason: "more sequences in one batch than a sequence id can hold".to_string(),
        })?;
        batch.add_sequence(sequence, seq_id, false).map_err(|e| {
            EmbeddingError::InferenceFailed {
                reason: format!(
                    "could not add a {}-token sequence to the batch: {e}",
                    sequence.len()
                ),
            }
        })?;
    }

    context
        .decode(batch)
        .map_err(|e| EmbeddingError::InferenceFailed {
            reason: format!(
                "llama.cpp could not decode a batch of {} sequence(s) / {} token(s): {e}",
                group.len(),
                batch.n_tokens()
            ),
        })?;

    // Read back by the same index the sequence was added under: the only place input
    // order could be lost.
    for (seq_id, sequence) in group.iter().enumerate() {
        let index = i32::try_from(seq_id).unwrap_or(i32::MAX);
        let vector =
            context
                .embeddings_seq_ith(index)
                .map_err(|e| EmbeddingError::InferenceFailed {
                    reason: format!(
                        "no pooled embedding for sequence {seq_id} of {} ({} tokens): {e}",
                        group.len(),
                        sequence.len()
                    ),
                })?;
        if vector.len() as u32 != dim {
            return Err(EmbeddingError::DimensionMismatch {
                expected: dim,
                actual: vector.len() as u32,
            });
        }
        // Copied because the slice borrows the context, which the next decode overwrites.
        out.push(vector.to_vec());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// This model's EOS; only its position matters to the arithmetic, not its value.
    const EOS: LlamaToken = LlamaToken(151_643);

    /// Serializes every test that reads or writes a tuning environment variable.
    /// `std::env` is process-global and cargo runs tests in threads, so without this a
    /// test that *sets* a variable fails a concurrent test that merely *reads* one.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// `from_env_for` under [`ENV_LOCK`], so a reader cannot forget to take it.
    fn tuning_for(config: &EmbeddingConfig) -> Result<LlamaBackendConfig, EmbeddingError> {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        LlamaBackendConfig::from_env_for(config)
    }

    fn content(len: usize) -> Vec<LlamaToken> {
        // Ids distinguishable from the EOS and from each other, so a wrong slice is
        // visible rather than plausible.
        (0..len).map(|i| LlamaToken(i as i32 + 1)).collect()
    }

    /// The one invariant that must never break: with last-token pooling, an EOS that is
    /// not last means the embedding describes an arbitrary interior word.
    ///
    /// Swept rather than spot-checked because the natural bug — `push` before
    /// `truncate` — is correct for every input short enough not to truncate.
    #[test]
    fn the_eos_survives_truncation_at_every_length() {
        for max_tokens in [1usize, 2, 3, 8, 511, 512, 513] {
            for length in 0..(max_tokens + 4) {
                let produced = truncate_with_eos(content(length), max_tokens, EOS);

                assert_eq!(
                    produced.last().copied(),
                    Some(EOS),
                    "max_tokens {max_tokens}, {length} content tokens: the EOS must be the \
                     final token or last-token pooling reads a content token"
                );
                assert!(
                    !produced.is_empty(),
                    "max_tokens {max_tokens}, {length} content tokens: an empty sequence \
                     cannot be decoded"
                );
                assert!(
                    produced.len() <= max_tokens.max(1),
                    "max_tokens {max_tokens}, {length} content tokens: produced {} tokens",
                    produced.len()
                );
                // A stray second EOS would change what is pooled for a text that
                // legitimately ends in one.
                assert_eq!(
                    produced.iter().filter(|t| **t == EOS).count(),
                    1,
                    "max_tokens {max_tokens}, {length} content tokens: {produced:?}"
                );
                // The tail is what is dropped, so the leading tokens must be untouched.
                let kept = produced.len() - 1;
                assert_eq!(produced[..kept], content(length)[..kept]);
            }
        }
    }

    /// `max_tokens` counts the EOS. The goldens pin both readings against the same
    /// 915-token text, so this asserts *which* one is implemented.
    #[test]
    fn max_tokens_is_the_total_sequence_length_including_the_eos() {
        // `over_512_trunc_total_512` — token_count 512 — is the convention here.
        let produced = truncate_with_eos(content(915), 512, EOS);
        assert_eq!(produced.len(), 512, "512 must mean 512 ids, EOS included");
        assert_eq!(produced[511], EOS);
        assert_eq!(
            produced[510],
            LlamaToken(511),
            "511 content tokens are kept"
        );

        // `over_512_trunc_content_512` — token_count 513 — is the convention this
        // backend does not implement.
        assert_ne!(produced.len(), 513);

        let exact = truncate_with_eos(content(511), 512, EOS);
        assert_eq!(exact.len(), 512);
        assert_eq!(exact[..511], content(511));
    }

    /// The bound the coordinator needs, asserted here so a backend that is not `Sync`
    /// fails to compile in this file rather than in `hybrid::coordinator`. `RawVocab`
    /// is named separately because it carries the hand-written `unsafe impl`.
    #[test]
    fn the_backend_is_send_and_sync() {
        fn require<T: Send + Sync>() {}
        require::<LlamaCppBackend>();
        require::<RawVocab>();
        require::<ContextPool>();
        require::<Box<dyn EmbeddingBackend>>();
    }

    #[test]
    fn the_default_tuning_is_bounded_rather_than_as_large_as_the_machine() {
        let tuning = LlamaBackendConfig::default();
        assert!(
            (1..=DEFAULT_CONTEXTS).contains(&tuning.contexts),
            "a pool of {} contexts is {} KiB/token of KV cache each, plus a compute buffer",
            tuning.contexts,
            112
        );
        assert!((1..=DEFAULT_THREADS_CAP).contains(&tuning.n_threads));

        // Pinned at exactly one, so raising it trips a test rather than being a quiet
        // edit: the second context buys 1.93x the throughput for ~0.5 GB of peak RSS.
        assert_eq!(
            tuning.contexts, 1,
            "the default context count is a memory budget, not a performance knob: \
             each additional context costs ~0.5 GB of peak RSS while decoding"
        );

        // The product, not the two knobs separately: this is what would have failed on
        // a 4-core phone before the clamp existed.
        assert!(
            tuning.contexts * tuning.n_threads <= available_parallelism(),
            "the default asks for {} ggml threads on {} core(s)",
            tuning.contexts * tuning.n_threads,
            available_parallelism()
        );
        assert_eq!(tuning.n_gpu_layers, 0, "the goldens were measured on CPU");
        assert_eq!(tuning.n_ctx, DEFAULT_N_CTX);
        // Not llama.cpp's ceiling of 256: each slot reserves `n_vocab` floats of logits
        // this backend never reads, measured at 149 MiB per context.
        assert!(
            tuning.max_sequences_per_decode <= DEFAULT_MAX_SEQUENCES,
            "{} sequence slots is {} MiB of unread logits per context",
            tuning.max_sequences_per_decode,
            tuning.max_sequences_per_decode * 151_669 * 4 / (1024 * 1024)
        );

        // And it follows `batch_size`: slots beyond it could never be filled.
        let derived = tuning_for(&EmbeddingConfig {
            batch_size: 4,
            ..Default::default()
        })
        .unwrap();
        assert_eq!(derived.max_sequences_per_decode, 4);
    }

    /// `n_ctx` has to be at least `max_tokens`, checked on the value `from_env_for`
    /// derives since that is the path the backend-selection table takes.
    #[test]
    fn the_derived_context_budget_holds_a_whole_sequence() {
        let config = EmbeddingConfig {
            max_tokens: 4096,
            ..Default::default()
        };
        let tuning = tuning_for(&config).unwrap();
        assert!(
            tuning.n_ctx >= 4096,
            "n_ctx {} cannot hold a {}-token sequence",
            tuning.n_ctx,
            config.max_tokens
        );
    }

    /// Someone who exports `OTZARIA_LLAMA_CONTEXTS=four` is trying to bound memory, and
    /// quietly using the default bounds nothing.
    #[test]
    fn an_unparseable_tuning_override_is_refused_rather_than_ignored() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let config = EmbeddingConfig::default();
        // SAFETY (2024 edition lint): single-threaded within this guard, and the
        // variable is removed before the guard is released.
        std::env::set_var(LlamaBackendConfig::ENV_CONTEXTS, "four");
        let refused = LlamaBackendConfig::from_env_for(&config);
        std::env::remove_var(LlamaBackendConfig::ENV_CONTEXTS);

        match refused {
            Err(EmbeddingError::LoadFailed { reason }) => {
                assert!(
                    reason.contains(LlamaBackendConfig::ENV_CONTEXTS) && reason.contains("four"),
                    "the error must name the variable and the value: {reason}"
                );
            }
            other => panic!("a bad override must be refused, got {other:?}"),
        }

        // And a good one is honoured, so the check is not refusing everything.
        std::env::set_var(LlamaBackendConfig::ENV_CONTEXTS, "3");
        let accepted = LlamaBackendConfig::from_env_for(&config);
        std::env::remove_var(LlamaBackendConfig::ENV_CONTEXTS);
        assert_eq!(accepted.unwrap().contexts, 3);
    }

    /// `contexts * n_threads` cannot exceed the machine, however it was asked for —
    /// without the clamp, raising the context count to buy throughput silently bought
    /// 2.1x less.
    #[test]
    fn the_thread_demand_is_clamped_to_the_machine_however_it_was_requested() {
        let cores = available_parallelism();

        // Absurd on any machine: the clamp is what makes this representable.
        let clamped = LlamaBackendConfig {
            contexts: 4,
            n_threads: 64,
            ..Default::default()
        }
        .clamped_to_machine();
        assert!(
            clamped.contexts * clamped.n_threads <= cores.max(4),
            "4 x 64 became {} x {} on {cores} core(s)",
            clamped.contexts,
            clamped.n_threads
        );
        assert!(
            clamped.n_threads >= 1,
            "a context needs at least one thread"
        );
        assert_eq!(
            clamped.contexts, 4,
            "contexts are honoured, threads give way"
        );

        // Idempotent, because `open` applies it again after `from_env_for` did.
        assert_eq!(clamped.clone().clamped_to_machine(), clamped);

        let floored = LlamaBackendConfig {
            contexts: 0,
            n_threads: 0,
            ..Default::default()
        }
        .clamped_to_machine();
        assert_eq!((floored.contexts, floored.n_threads), (1, 1));

        // And through the environment, the path the hazard was reported on.
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var(LlamaBackendConfig::ENV_CONTEXTS, "64");
        let from_env = LlamaBackendConfig::from_env_for(&EmbeddingConfig::default());
        std::env::remove_var(LlamaBackendConfig::ENV_CONTEXTS);
        let from_env = from_env.expect("64 contexts is a memory decision, not a parse error");
        assert_eq!(from_env.contexts, 64, "the request itself is honoured");
        assert_eq!(
            from_env.n_threads, 1,
            "64 contexts on {cores} core(s) leaves one thread each"
        );
    }

    /// `open` promises never to panic, and one reachable path broke it:
    /// `OTZARIA_LLAMA_N_CTX` in `4294967041..=4294967295` parses as a `u32` and has no
    /// multiple of 256 above it. Both the debug overflow and the release wrap-to-zero
    /// are now errors that name the variable.
    #[test]
    fn an_absurd_context_budget_is_refused_rather_than_overflowing() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let config = EmbeddingConfig::default();

        for value in [u32::MAX, u32::MAX - 254, 4_294_967_041] {
            std::env::set_var(LlamaBackendConfig::ENV_N_CTX, value.to_string());
            let refused = LlamaBackendConfig::from_env_for(&config);
            std::env::remove_var(LlamaBackendConfig::ENV_N_CTX);
            match refused {
                Err(EmbeddingError::LoadFailed { reason }) => assert!(
                    reason.contains(LlamaBackendConfig::ENV_N_CTX),
                    "the error must name the variable: {reason}"
                ),
                other => panic!("n_ctx {value} must be refused, got {other:?}"),
            }
        }

        // The largest expressible value is accepted, so the boundary is where it belongs.
        let largest = (u32::MAX / N_CTX_PADDING) * N_CTX_PADDING;
        std::env::set_var(LlamaBackendConfig::ENV_N_CTX, largest.to_string());
        let accepted = LlamaBackendConfig::from_env_for(&config);
        std::env::remove_var(LlamaBackendConfig::ENV_N_CTX);
        assert_eq!(accepted.expect("the largest padded n_ctx").n_ctx, largest);
    }
}

/// Verification against the golden reference vectors. **This is the parity gate**,
/// not a placeholder for one.
///
/// These need the real 396 MB GGUF, which is gitignored, so each is `#[ignore]`d
/// *and* skips loudly when `OTZARIA_TEST_MODEL` is unset — that keeps the ordinary
/// CI matrix green on machines with no model. The dedicated `golden-vectors` CI job
/// does fetch the model and run them; when its token secret is missing it fails
/// rather than reporting the skip as a pass. Run them locally with:
///
/// ```sh
/// OTZARIA_TEST_MODEL=./Otzaria-Embedding-V1-Flash-0.6B-Q4_K_M.gguf \
///   cargo test --lib --features llama-backend golden -- --ignored --nocapture
/// ```
///
/// Thresholds are read from the goldens' own `recommended_tolerances`, so the file
/// and the assertions cannot drift apart.
#[cfg(test)]
mod golden {
    use super::*;
    use std::path::{Path, PathBuf};

    /// The model, or `None` with a loud explanation. Skipping rather than failing because
    /// CI has no model, and a test that fails there teaches everyone to ignore it.
    fn model_path() -> Option<PathBuf> {
        match std::env::var("OTZARIA_TEST_MODEL") {
            Ok(path) if !path.trim().is_empty() => {
                let path = PathBuf::from(path.trim());
                if path.exists() {
                    return Some(path);
                }
                println!("SKIPPED: OTZARIA_TEST_MODEL points at {path:?}, which does not exist");
                None
            }
            _ => {
                println!(
                    "SKIPPED: OTZARIA_TEST_MODEL is not set. This test needs the 396 MB \
                     Otzaria-Embedding-V1-Flash-0.6B-Q4_K_M.gguf, which is gitignored."
                );
                None
            }
        }
    }

    fn goldens() -> serde_json::Value {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/golden_vectors.json");
        let raw = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        serde_json::from_str(&raw).expect("golden_vectors.json is not valid JSON")
    }

    /// Confirm the file behind `OTZARIA_TEST_MODEL` is the file the goldens describe, since
    /// the alternative is a wall of vector failures for a model that was the wrong one.
    fn assert_is_the_golden_model(path: &Path, header: &serde_json::Value) {
        use sha2::Digest;

        let expected_size = header["model_size_bytes"]
            .as_u64()
            .expect("model_size_bytes");
        let actual_size = std::fs::metadata(path).expect("stat model").len();
        assert_eq!(
            actual_size,
            expected_size,
            "{} is {actual_size} bytes; the goldens were produced from a {expected_size}-byte file",
            path.display()
        );

        let mut hasher = sha2::Sha256::new();
        let mut file = std::fs::File::open(path).expect("open model");
        let mut buffer = vec![0u8; 1 << 20];
        loop {
            let read = std::io::Read::read(&mut file, &mut buffer).expect("read model");
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
        let actual: String = hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        let expected = header["model_sha256"].as_str().expect("model_sha256");
        assert_eq!(
            actual,
            expected,
            "the model at {} is not the one the goldens were produced from",
            path.display()
        );
    }

    /// Standard, padded base64. Hand-rolled to keep a decoder out of the dependency tree.
    fn base64_decode(input: &str) -> Vec<u8> {
        const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = Vec::with_capacity(input.len() / 4 * 3);
        let (mut accumulator, mut bits) = (0u32, 0u32);
        for byte in input.bytes() {
            if byte == b'=' || byte.is_ascii_whitespace() {
                continue;
            }
            let value = ALPHABET
                .iter()
                .position(|c| *c == byte)
                .unwrap_or_else(|| panic!("{byte:?} is not a base64 character"));
            accumulator = (accumulator << 6) | value as u32;
            bits += 6;
            if bits >= 8 {
                bits -= 8;
                out.push(((accumulator >> bits) & 0xFF) as u8);
            }
        }
        out
    }

    /// Decode a golden vector payload: 1024 little-endian binary32 values.
    fn golden_vector(encoded: &str) -> Vec<f32> {
        let bytes = base64_decode(encoded);
        assert_eq!(
            bytes.len(),
            4096,
            "a golden vector is 1024 f32 = 4096 bytes"
        );
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Cosine in f64: an f32 accumulation over 1024 terms loses digits that matter when
    /// the answer is compared against 0.9999.
    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        let dot: f64 = a
            .iter()
            .zip(b)
            .map(|(x, y)| f64::from(*x) * f64::from(*y))
            .sum();
        let na: f64 = a
            .iter()
            .map(|x| f64::from(*x) * f64::from(*x))
            .sum::<f64>()
            .sqrt();
        let nb: f64 = b
            .iter()
            .map(|x| f64::from(*x) * f64::from(*x))
            .sum::<f64>()
            .sqrt();
        if na == 0.0 || nb == 0.0 {
            return 0.0;
        }
        dot / (na * nb)
    }

    fn l2_norm(v: &[f32]) -> f64 {
        v.iter()
            .map(|x| f64::from(*x) * f64::from(*x))
            .sum::<f64>()
            .sqrt()
    }

    fn normalized(v: &[f32]) -> Vec<f32> {
        let norm = l2_norm(v);
        v.iter().map(|x| (f64::from(*x) / norm) as f32).collect()
    }

    /// A single-context backend, so the memory a test holds is one KV cache.
    fn open_backend(path: &Path, max_tokens: usize) -> LlamaCppBackend {
        LlamaCppBackend::open(
            path,
            max_tokens,
            &LlamaBackendConfig {
                contexts: 1,
                ..Default::default()
            },
        )
        .expect("the golden model must load")
    }

    /// **The primary gate.** Exact token-id equality, then the vector measurements.
    ///
    /// Only `text` is read as input, per the goldens' `token_ids_comparison_rule`: feeding
    /// the golden ids back in would make the assertion a tautology. The ids carry the
    /// weight because a wrongly prepended BOS scores cosine 0.9947938 against its own
    /// golden while the worst *legitimate* agreement (PyTorch fp32) is 0.9947909 — the bug
    /// scores **higher** than the honest reference, so only the integers can separate them.
    #[test]
    #[ignore = "needs the 396 MB GGUF; set OTZARIA_TEST_MODEL and pass --ignored"]
    fn token_ids_match_the_reference_exactly_and_vectors_agree() {
        let Some(path) = model_path() else { return };
        let data = goldens();
        assert_is_the_golden_model(&path, &data["header"]);

        let tolerances = &data["header"]["recommended_tolerances"];
        let min_cosine = tolerances["cosine_min_any_llama_cpp_build"]
            .as_f64()
            .unwrap();
        let max_component = tolerances["max_abs_component_diff_normalized"]
            .as_f64()
            .unwrap();
        let max_norm_drift = tolerances["raw_l2_norm_relative_diff_max"]
            .as_f64()
            .unwrap();

        let records = data["vectors"].as_array().expect("vectors").clone();
        // Everything but `over_512_full` is measured at the production cap of 512; that
        // record's golden was produced untruncated, at 915 tokens.
        let at_512 = open_backend(&path, 512);
        assert_eq!(at_512.dim(), 1024);
        assert_eq!(at_512.max_tokens(), 512);
        assert_eq!(at_512.pooling(), Pooling::LastToken);
        assert!(at_512.is_semantic());
        assert_eq!(at_512.id(), "llama-cpp-qwen3-last-v1");

        let mut worst_cosine = (1.0f64, String::new());
        let mut worst_component = (0.0f64, String::new());
        let mut worst_norm = (0.0f64, String::new());
        let mut id_mismatches: Vec<String> = Vec::new();
        let mut checked = 0usize;

        println!(
            "\n{:<36} {:>5} {:>12} {:>10} {:>10}",
            "id", "toks", "cosine", "maxΔcomp", "Δnorm"
        );
        println!("{}", "-".repeat(78));

        for record in &records {
            let id = record["id"].as_str().unwrap();
            let text = record["text"].as_str().unwrap();
            let golden_ids: Vec<u32> = record["token_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect();

            // `over_512_trunc_content_512` is the convention this backend deliberately
            // does not implement, so asserting equality against it would be asserting
            // the wrong one.
            let rejected_convention = id == "over_512_trunc_content_512";
            let backend = if id == "over_512_full" {
                None // measured with its own cap, below
            } else {
                Some(&at_512)
            };
            let Some(backend) = backend else { continue };

            // Proves the exact bytes were read, invisible characters included.
            let text_digest: String = {
                use sha2::Digest;
                sha2::Sha256::digest(text.as_bytes())
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect()
            };
            assert_eq!(
                text_digest,
                record["text_utf8_sha256"].as_str().unwrap(),
                "{id}: the text read differs from the text the goldens were produced from"
            );

            let produced_ids = backend.tokenize(text).expect("tokenize");

            if rejected_convention {
                assert_ne!(
                    produced_ids, golden_ids,
                    "{id} pins the convention where max_tokens counts content only; this \
                     backend counts the total, so it must NOT reproduce these ids"
                );
                assert_eq!(
                    produced_ids.len(),
                    512,
                    "{id}: the implemented convention yields 512 ids, not {}",
                    produced_ids.len()
                );
                println!(
                    "{id:<36} {:>5}  (rejected convention, as designed)",
                    produced_ids.len()
                );
                continue;
            }

            if produced_ids != golden_ids {
                let first = produced_ids
                    .iter()
                    .zip(&golden_ids)
                    .position(|(a, b)| a != b)
                    .unwrap_or(produced_ids.len().min(golden_ids.len()));
                id_mismatches.push(format!(
                    "{id}: {} ids vs {} golden, first difference at {first}",
                    produced_ids.len(),
                    golden_ids.len()
                ));
                continue;
            }
            assert_eq!(
                produced_ids.last().copied(),
                Some(151_643),
                "{id}: the EOS must be the final token"
            );

            // One sequence per decode — the geometry the goldens were generated with.
            let raw = backend.embed_batch_raw(&[text]).expect("embed")[0].clone();
            let golden_norm = record["raw_l2_norm"].as_f64().unwrap();
            let produced_norm = l2_norm(&raw);
            let norm_drift = (produced_norm - golden_norm).abs() / golden_norm;

            let golden_unit =
                golden_vector(record["embedding_normalized_f32_b64"].as_str().unwrap());
            let produced_unit = normalized(&raw);
            let cos = cosine(&produced_unit, &golden_unit);
            let component = produced_unit
                .iter()
                .zip(&golden_unit)
                .map(|(a, b)| f64::from(*a - *b).abs())
                .fold(0.0f64, f64::max);

            println!(
                "{id:<36} {:>5} {cos:>12.9} {component:>10.3e} {norm_drift:>10.3e}",
                produced_ids.len()
            );

            if cos < worst_cosine.0 {
                worst_cosine = (cos, id.to_string());
            }
            if component > worst_component.0 {
                worst_component = (component, id.to_string());
            }
            if norm_drift > worst_norm.0 {
                worst_norm = (norm_drift, id.to_string());
            }
            checked += 1;
        }

        // `over_512_full`: no truncation, so a cap that clears its 915 tokens.
        drop(at_512);
        let untruncated = open_backend(&path, 1024);
        let full = records
            .iter()
            .find(|r| r["id"] == "over_512_full")
            .expect("over_512_full");
        let text = full["text"].as_str().unwrap();
        let golden_ids: Vec<u32> = full["token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let produced_ids = untruncated.tokenize(text).expect("tokenize");
        if produced_ids == golden_ids {
            let raw = untruncated.embed_batch_raw(&[text]).expect("embed")[0].clone();
            let golden_unit = golden_vector(full["embedding_normalized_f32_b64"].as_str().unwrap());
            let cos = cosine(&normalized(&raw), &golden_unit);
            let norm_drift = (l2_norm(&raw) - full["raw_l2_norm"].as_f64().unwrap()).abs()
                / full["raw_l2_norm"].as_f64().unwrap();
            println!(
                "{:<36} {:>5} {cos:>12.9} {:>10} {norm_drift:>10.3e}",
                "over_512_full",
                produced_ids.len(),
                ""
            );
            if cos < worst_cosine.0 {
                worst_cosine = (cos, "over_512_full".to_string());
            }
            if norm_drift > worst_norm.0 {
                worst_norm = (norm_drift, "over_512_full".to_string());
            }
            checked += 1;
        } else {
            id_mismatches.push(format!(
                "over_512_full: {} ids vs {} golden",
                produced_ids.len(),
                golden_ids.len()
            ));
        }

        println!("\n--- summary over {checked} records ---");
        println!(
            "worst cosine          {:.9}  ({})",
            worst_cosine.0, worst_cosine.1
        );
        println!(
            "worst max|Δcomponent| {:.4e}  ({})",
            worst_component.0, worst_component.1
        );
        println!(
            "worst relative Δnorm  {:.4e}  ({})",
            worst_norm.0, worst_norm.1
        );

        assert!(
            id_mismatches.is_empty(),
            "TOKEN ID MISMATCHES — the primary correctness gate:\n  {}",
            id_mismatches.join("\n  ")
        );
        assert!(
            worst_cosine.0 >= min_cosine,
            "worst cosine {:.9} on {} is below the goldens' {min_cosine}",
            worst_cosine.0,
            worst_cosine.1
        );
        assert!(
            worst_component.0 <= max_component,
            "worst component difference {:.4e} on {} exceeds {max_component}",
            worst_component.0,
            worst_component.1
        );
        assert!(
            worst_norm.0 <= max_norm_drift,
            "worst raw-norm drift {:.4e} on {} exceeds {max_norm_drift}",
            worst_norm.0,
            worst_norm.1
        );
        // Derived from the data rather than pinned, since the golden generator owns the
        // corpus. What must hold is that nothing was silently skipped.
        let rejected = records
            .iter()
            .filter(|r| r["id"] == "over_512_trunc_content_512")
            .count();
        assert_eq!(
            checked + rejected,
            records.len(),
            "{} of {} golden records were neither checked nor deliberately rejected",
            records.len() - checked - rejected,
            records.len()
        );
    }

    /// Batched inference: one vector per input, **in input order**, each equivalent to
    /// its single-sequence twin.
    ///
    /// Order has no other line of defence — `EmbeddingRuntime` pairs vectors positionally
    /// with chunk metadata and cannot check the pairing, so a transposition would attach
    /// every book's text to another book's line unnoticed. The assertion is therefore an
    /// argmax: each batched output's nearest neighbour must be its own freshly computed
    /// single-sequence vector.
    ///
    /// Bitwise equality with the single-sequence path is deliberately *not* asserted:
    /// llama.cpp does not provide it, with a reproduced worst cosine of 0.99668. The cause
    /// is not diagnosed; the magnitude is what is established.
    #[test]
    #[ignore = "needs the 396 MB GGUF; set OTZARIA_TEST_MODEL and pass --ignored"]
    fn batched_vectors_come_back_in_input_order() {
        let Some(path) = model_path() else { return };
        let data = goldens();
        let min_cosine = data["header"]["recommended_tolerances"]["batch_vs_single_cosine_min"]
            .as_f64()
            .unwrap();

        // Deliberately mixed: two near-identical texts (0.941 apart) beside two unrelated
        // ones (0.131), since the near-identical pair is what would hide a slightly wrong
        // ordering. The four relational ids are required by the ordering assertion; the
        // rest are filtered so a renamed coverage record cannot fail this test.
        const REQUIRED: [&str; 4] = [
            "near_identical_a",
            "unrelated_a",
            "near_identical_b",
            "unrelated_b",
        ];
        let preferred = [
            "near_identical_a",
            "unrelated_a",
            "near_identical_b",
            "space_only",
            "unrelated_b",
            "short_single_word",
            "medium_paragraph",
            "digits_only",
        ];
        let present = |id: &str| {
            data["vectors"]
                .as_array()
                .unwrap()
                .iter()
                .find(|r| r["id"] == id)
                .cloned()
        };
        for id in REQUIRED {
            assert!(
                present(id).is_some(),
                "the relational ordering assertion needs golden record {id}"
            );
        }
        let wanted: Vec<&str> = preferred
            .into_iter()
            .filter(|id| present(id).is_some())
            .collect();
        let owned: Vec<String> = wanted
            .iter()
            .map(|id| present(id).unwrap()["text"].as_str().unwrap().to_string())
            .collect();
        // A blank line is a real input with no golden: it tokenizes to nothing, so the
        // sequence is the EOS alone and `decode` runs on a single token.
        let mut texts: Vec<&str> = owned.iter().map(String::as_str).collect();
        let with_goldens = texts.len();
        texts.push("");

        let backend = open_backend(&path, 512);
        let batched = backend.embed_batch_raw(&texts).expect("batched embed");
        assert_eq!(batched.len(), texts.len(), "one vector per input");

        let singles: Vec<Vec<f32>> = texts
            .iter()
            .map(|t| {
                backend
                    .embed_batch_raw(&[t])
                    .expect("single embed")
                    .remove(0)
            })
            .collect();

        let unit_batched: Vec<Vec<f32>> = batched.iter().map(|v| normalized(v)).collect();
        let unit_singles: Vec<Vec<f32>> = singles.iter().map(|v| normalized(v)).collect();

        // The blank line only has to *work*: pinning its similarity to anything would be
        // pinning noise.
        let blank = &batched[with_goldens];
        assert_eq!(blank.len(), 1024, "a blank line must still yield a vector");
        assert!(
            blank.iter().all(|x| x.is_finite()) && l2_norm(blank) > 1.0,
            "a blank line produced an unusable vector of norm {}",
            l2_norm(blank)
        );
        println!("blank line: raw L2 norm {:.4}", l2_norm(blank));

        println!(
            "\n{:<24} {:>12} {:>12} {:>10}",
            "id", "cos(batch,1)", "cos(next)", "margin"
        );
        let mut worst = (1.0f64, String::new());
        for (index, id) in wanted.iter().enumerate() {
            // Every output against every single-sequence vector: the diagonal must win.
            let mut scored: Vec<(usize, f64)> = (0..with_goldens)
                .map(|other| (other, cosine(&unit_batched[index], &unit_singles[other])))
                .collect();
            scored.sort_by(|a, b| b.1.total_cmp(&a.1));
            let (best, best_cosine) = scored[0];
            let runner_up = scored[1].1;

            assert_eq!(
                best, index,
                "batched output {index} ({id}) is closest to input {best} \
                 ({}) — the batch was reordered",
                wanted[best]
            );
            println!(
                "{id:<24} {best_cosine:>12.9} {runner_up:>12.9} {:>10.6}",
                best_cosine - runner_up
            );
            if best_cosine < worst.0 {
                worst = (best_cosine, (*id).to_string());
            }
        }
        println!(
            "worst batch-vs-single cosine {:.9} ({}), goldens allow >= {min_cosine}",
            worst.0, worst.1
        );
        assert!(
            worst.0 >= min_cosine,
            "batched output for {} agrees with its single-sequence twin only to {:.9}",
            worst.1,
            worst.0
        );

        // The relational ordering, which a subtly wrong implementation cannot fake.
        let ordering = &data["relations"]["ordering_assertion"];
        let margin_min = ordering["margin_min"].as_f64().unwrap();
        let index_of = |id: &str| wanted.iter().position(|w| *w == id).unwrap();
        let near = cosine(
            &unit_batched[index_of("near_identical_a")],
            &unit_batched[index_of("near_identical_b")],
        );
        let unrelated = cosine(
            &unit_batched[index_of("unrelated_a")],
            &unit_batched[index_of("unrelated_b")],
        );
        println!(
            "relational ordering: near {near:.6} - unrelated {unrelated:.6} = {:.6} (>= {margin_min})",
            near - unrelated
        );
        assert!(
            near - unrelated >= margin_min,
            "near-identical {near:.6} and unrelated {unrelated:.6} are only {:.6} apart",
            near - unrelated
        );
    }

    /// Several threads inside `embed_batch_raw` at once, through `&self`, agreeing with
    /// the single-threaded answer — the shape `hybrid::coordinator` actually produces.
    /// Wall-clock figures are printed for information; the ratio depends on the machine
    /// and is not asserted.
    #[test]
    #[ignore = "needs the 396 MB GGUF; set OTZARIA_TEST_MODEL and pass --ignored"]
    fn concurrent_callers_through_a_shared_reference_agree_with_serial_ones() {
        let Some(path) = model_path() else { return };
        let data = goldens();
        let texts: Vec<String> = data["vectors"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|r| r["token_count"].as_u64().unwrap_or(0) <= 200)
            .map(|r| r["text"].as_str().unwrap().to_string())
            .collect();
        assert!(
            texts.len() >= 8,
            "need a few texts to keep two threads busy"
        );

        let backend = LlamaCppBackend::open(
            &path,
            512,
            &LlamaBackendConfig {
                contexts: 2,
                ..Default::default()
            },
        )
        .expect("open");
        println!("\npool: {backend:?}");

        let borrowed: Vec<&str> = texts.iter().map(String::as_str).collect();
        let started = std::time::Instant::now();
        let serial = backend.embed_batch_raw(&borrowed).expect("serial embed");
        let serial_time = started.elapsed();

        let started = std::time::Instant::now();
        let concurrent = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..4)
                .map(|_| {
                    let backend = &backend;
                    let borrowed = &borrowed;
                    // `&self` from four threads at once: what would not compile if the
                    // backend were not `Sync`, and serialize if the pool were one mutex.
                    scope
                        .spawn(move || backend.embed_batch_raw(borrowed).expect("concurrent embed"))
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("worker"))
                .collect::<Vec<_>>()
        });
        let concurrent_time = started.elapsed();

        println!(
            "{} texts: serial {serial_time:?}, 4 threads x the same batch {concurrent_time:?}",
            texts.len()
        );

        // The timings above cannot show whether the pool really parallelizes: two contexts
        // of four ggml threads want eight cores, and contention hides the effect. Pinning
        // each context to one thread removes that variable. Printed, never asserted: the
        // property is structural, and a wall-clock threshold would flake.
        let sample: Vec<&str> = borrowed.iter().take(8).copied().collect();
        for contexts in [1usize, 2] {
            let measured = LlamaCppBackend::open(
                &path,
                512,
                &LlamaBackendConfig {
                    contexts,
                    n_threads: 1,
                    ..Default::default()
                },
            )
            .expect("open");
            let started = std::time::Instant::now();
            std::thread::scope(|scope| {
                for _ in 0..2 {
                    let measured = &measured;
                    let sample = &sample;
                    scope.spawn(move || measured.embed_batch_raw(sample).expect("embed"));
                }
            });
            println!(
                "{contexts} context(s) x 1 thread, 2 concurrent batches of {}: {:?}",
                sample.len(),
                started.elapsed()
            );
        }

        for (thread, result) in concurrent.iter().enumerate() {
            assert_eq!(
                result.len(),
                serial.len(),
                "thread {thread} returned the wrong count"
            );
            for (index, vector) in result.iter().enumerate() {
                let cos = cosine(vector, &serial[index]);
                assert!(
                    cos >= 0.999_999,
                    "thread {thread}, text {index}: concurrent inference disagrees with serial \
                     inference (cosine {cos:.9}); the contexts are interfering"
                );
            }
        }
    }
}
