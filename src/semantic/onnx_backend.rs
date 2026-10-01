//! Real ONNX inference through ONNX Runtime: the `onnxruntime-sentence-v1` backend.
//!
//! Behind the non-default `onnx-backend` feature, on the desktop targets its crates are
//! declared for (`Cargo.toml` says which and why); anywhere else an ONNX model gets
//! [`EmbeddingError::BackendUnavailable`]. Where the backend is compiled in but the ONNX
//! Runtime library cannot be loaded, the error is
//! [`EmbeddingError::OnnxRuntimeUnavailable`] instead: the fix is a file, not a build.
//! Design and measurements are in `docs/ONNX_BACKEND.md`.
//!
//! # What the backend id promises
//!
//! ```text
//! text                                   # already role-prefixed by the text recipe
//!   -> tokenizer.json                    # the package's own, never a substitute
//!        padding off, add_special_tokens = true
//!   -> truncate to max_tokens            # on the right, content first: the total
//!                                        # length, and the specials survive
//!   -> input_ids + attention_mask (ones) # token_type_ids as zeros, only if declared
//!   -> one text per run, [1, len]        # never padded, never batched — see below
//!   -> the graph's first output          # [1, dim]: pooled inside the graph
//!   -> return RAW                        # the runtime normalizes and validates
//! ```
//!
//! [`EmbeddingBackend::tokenize`] and [`EmbeddingBackend::embed_batch_raw`] share one
//! path to the ids, so what the goldens compare is what the graph consumes.
//!
//! # ONNX Runtime is loaded, not linked
//!
//! The runtime is a shared library found at load time — the path the application passes
//! ([`EmbeddingDeployment::onnx_runtime`](crate::semantic::embedding::EmbeddingDeployment::onnx_runtime)),
//! else `OTZARIA_ONNX_RUNTIME`, else the platform's file name (`libonnxruntime.dylib`,
//! `libonnxruntime.so`, `onnxruntime.dll`) in the model package beside the graph, never
//! falling through from one that is set (`resolve_runtime_path`) — and loaded once per
//! process: `ort` holds one runtime and can neither unload nor replace it, so a second,
//! different library is refused rather than silently ignored (`RUNTIME`). Until that load
//! has succeeded no other `ort` function is called, because `ort`'s own fallback search
//! `expect`s: a missing library would be a panic instead of an error. And `ort` is handed
//! only a library the backend has already opened and checked itself, because a refusal
//! inside `ort` 2.0.0-rc.13 cannot be retried — its next call panics (`ensure_runtime`).
//! The runtime is code rather than model data and is not part of the package checksum.
//!
//! # One text per run
//!
//! Every text is its own `[1, len]` run, never padded into a batch, so a vector depends
//! on its text alone:
//!
//! * a dynamically quantized graph (`DynamicQuantizeLinear`) computes its scale over
//!   the whole input tensor, padding included — the same text alone and in a batch
//!   measured cosine 0.9938–0.9957;
//! * even in fp32, large padding moved components by ~8.6e-7;
//! * the production graph declares a batch of 1 anyway.
//!
//! A batch dimension is therefore only *inspected* — refused if it is fixed above 1 —
//! and the throughput a batch would buy comes from the session pool instead.
//!
//! # Concurrency
//!
//! Who embeds what decides how many sessions there are. The library's vectors are built on
//! the build machine only; the application opens a prebuilt artifact read-only
//! (`OfficialSemanticIndex`) and embeds nothing but queries, one at a time — so it needs
//! one session, the default. More are a build-machine knob, for callers that embed at the
//! same time.
//!
//! `ort` 2.0.0-rc.13's `Session::run` takes `&mut self` (upstream considers concurrent
//! `Run` on one session unsound), while [`EmbeddingBackend`] is `&self` and `Sync`. So
//! the backend owns a bounded pool of sessions (`Pool`), one lease per *text*: where one
//! caller embeds a batch while another embeds a query — the prototype engine indexing as
//! it searches — the batch holds a session for one inference at a time, and the query
//! waits for one text rather than for the batch. The queue is FIFO, so the batch cannot
//! take the session straight back. The tokenizer is `Sync` with a `&self` `encode` and is
//! shared without a lock.
//!
//! # Special tokens inside text
//!
//! The text recipe spells the role prefix as text — `"[PASSAGE] "`, `"[QUERY] "` — and
//! it becomes the learned token only because the tokenizer matches added special tokens
//! in its input. So they are matched, deliberately: the consequence is that a book
//! containing the literal string `[CLS]`, `[SEP]` or `[QUERY]` gets that token too. The
//! Python `tokenizers` package does exactly the same (`tests/data/onnx_fixture/
//! expected.json` holds its answers, and the tests assert them); the match is on the raw
//! text and case-sensitive, so `[query]` stays text. This is the opposite of the llama
//! backend's `parse_special = false`, and for the opposite reason: there a control token
//! inside a book is an accident, here the prefix *is* one.

use crate::errors::EmbeddingError;
use crate::semantic::backend::{max_tokens_past_the_format, EmbeddingBackend, Pooling};
use crate::semantic::embedding::EmbeddingConfig;
use crate::semantic::model_package::{onnx_package_root, ModelFormat};

use ort::logging::LogLevel;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::{HasSelectedOutputs, OutputSelector, RunOptions, Session};
use ort::value::{Tensor, TensorElementType, ValueType};
use tokenizers::{
    PostProcessor, Tokenizer, TruncationDirection, TruncationParams, TruncationStrategy,
};

use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex, PoisonError};

/// Default intra-op threads per session: a cap, not a target — phones are big.LITTLE.
///
/// Four is where the returns stop on the production graph: a 256-token input took
/// 27.9 ms at one thread, 20.1 at two, 16.4 at four and 16.2–16.6 at six to ten (Apple
/// M4, four performance cores). The thread count measured bit-identical from one to
/// eight, so it cannot change a stored vector.
const DEFAULT_THREADS_CAP: usize = 4;

/// Default number of sessions, i.e. concurrent inferences: one — what the application
/// needs, since it embeds only queries, one at a time, and never the library. Each
/// session holds its own copy of the weights: a second one adds 74 MiB of peak footprint
/// for the int8 graph (169 → 243 MiB) and 202 MiB for fp32 (387 → 589 MiB). More are a
/// build-machine knob, worth it only for callers that embed at the same time; and even
/// then one is not a cliff: runs are leased one text at a time, so a query waits for one
/// inference behind a batch (measured at most 14.7 ms), not for the batch.
const DEFAULT_SESSIONS: usize = 1;

/// Environment variable naming the ONNX Runtime shared library to load when the
/// application passes none ([`CONFIGURED_SETTING`]). Read by [`OnnxBackend::open`], not by
/// [`OnnxBackendConfig`]: it chooses the code that runs, not a size.
const RUNTIME_ENV: &str = "OTZARIA_ONNX_RUNTIME";

/// Where an application passes the runtime library, as refusals name it: the first place
/// looked, before [`RUNTIME_ENV`].
const CONFIGURED_SETTING: &str = "EmbeddingDeployment::onnx_runtime";

/// The runtime's file name on this platform, as Microsoft's releases spell it — what is
/// looked for beside the graph when neither the application nor [`RUNTIME_ENV`] names one.
#[cfg(target_os = "macos")]
const RUNTIME_FILE_NAME: &str = "libonnxruntime.dylib";
#[cfg(target_os = "linux")]
const RUNTIME_FILE_NAME: &str = "libonnxruntime.so";
#[cfg(target_os = "windows")]
const RUNTIME_FILE_NAME: &str = "onnxruntime.dll";

/// Graph optimization, fixed rather than left to the runtime's default because it is
/// part of the wiring the backend id names: `All` and `Disable`/`Level1` measured
/// maxabs 2.9e-6 apart on the same graph. `All` is also what the Python reference
/// (`onnxruntime.SessionOptions`) defaults to.
const GRAPH_OPTIMIZATION: GraphOptimizationLevel = GraphOptimizationLevel::All;

/// ONNX Runtime's session entry for exact int8 matrix products on x86
/// (`kOrtSessionOptionsAvx2PrecisionMode`), set to `"1"` on every session.
///
/// A dynamically quantized graph multiplies uint8 activations by int8 weights, and on x86
/// ONNX Runtime's kernels for that sum each pair of neighbouring products with `PMADDUBSW`,
/// in **saturating 16-bit** arithmetic: 255 × 127 × 2 = 64,770 is clamped to 32,767. Only
/// VNNI (or AMX) sums them in 32 bits, so the clamp is live on AVX2 and AVX-512 CPUs without
/// it, and — in Microsoft's Windows build — on SSE4.1 CPUs without AVX2. The production int8
/// graph's weights span the full ±127 and it saturates: on CI's x86-64 runner every one of
/// the 41 golden vectors moved, to cosine 0.9809 at worst, and emulating the clamp on the M4
/// reproduced its numbers case by case, to within 6e-4. With the entry set, the runtime
/// rewrites each int8 weight tensor as uint8 (`Avx2WeightS8ToU8Transformer`: `w + 128`, the
/// zero point moved with it — the same products exactly) so that its U8U8 kernels run,
/// which sum them without a 16-bit clamp: exactly.
///
/// The runtime applies it only where it finds its own U8S8 kernels unsafe
/// (`MlasPlatformU8S8Overflow`, x86 builds only); on ARM, and for an fp32 graph, it changes
/// nothing — the M4's optimized graph is byte-identical with and without it. Set
/// unconditionally, on the build machine and in the application alike, because it decides
/// what an int8 vector is on x86: a library embedded on one x86 CPU and queried on another
/// must have computed the same products. The cost is speed alone: ONNX Runtime documents
/// the U8U8 kernels as slower, and a VNNI CPU, whose int8 products were exact already, gives
/// up its VNNI kernels for the same result. `docs/ONNX_BACKEND.md` §0.1 has the mechanism.
const X64_QUANT_PRECISION: &str = "session.x64quantprecision";

/// The three inputs this backend knows how to feed.
const INPUT_IDS: &str = "input_ids";
const ATTENTION_MASK: &str = "attention_mask";
const TOKEN_TYPE_IDS: &str = "token_type_ids";

/// Words repeated into the load-time probe until the tokenizer truncates it to exactly
/// `max_tokens`. Three scripts, so no normalizer can strip all of them.
const PROBE_WORDS: [&str; 3] = ["בראשית", "the", "1"];

/// Tuning for [`OnnxBackend`]. Separate from [`EmbeddingConfig`] for the reason
/// `LlamaBackendConfig` is: that type is persisted as an index's identity, while these
/// are deployment facts that must not change a stored vector, and so must not
/// invalidate an index when they change. Measured not to: `docs/ONNX_BACKEND.md`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OnnxBackendConfig {
    /// ONNX Runtime intra-op threads per session. At least one.
    pub intra_threads: usize,
    /// Sessions in the pool: how many `embed_batch_raw` calls run at once. At least
    /// one. Each costs roughly the graph's size in memory. One in the application, which
    /// embeds only queries; more only on a build machine whose callers embed concurrently.
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

// ─────────────────────────────── the runtime ───────────────────────────────

/// The places a runtime library can come from, in the order [`resolve_runtime_path`]
/// looks at them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RuntimeSource {
    /// The application passed it: [`CONFIGURED_SETTING`].
    Configured,
    /// [`RUNTIME_ENV`] named it.
    Environment,
    /// [`RUNTIME_FILE_NAME`] in the model package, beside the graph.
    BesideGraph,
}

/// A runtime library, and the place it came from. Every refusal of a library names both,
/// because the fix is in that place: the application's setting, the variable, or the
/// model's folder.
#[derive(Debug, Clone, PartialEq, Eq)]
struct RuntimeLocation {
    /// Canonical, so two spellings of one file compare equal.
    path: PathBuf,
    source: RuntimeSource,
}

impl std::fmt::Display for RuntimeLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let path = self.path.display();
        match self.source {
            RuntimeSource::Configured => write!(f, "{path} (passed by the application)"),
            RuntimeSource::Environment => write!(f, "{path} (named by {RUNTIME_ENV})"),
            RuntimeSource::BesideGraph => write!(f, "{path} (beside the model)"),
        }
    }
}

/// The ONNX Runtime library this process has loaded, once one has.
struct LoadedRuntime {
    location: RuntimeLocation,
    /// Its version and build, as log lines name it: `1.28.0 (git-branch=HEAD,
    /// git-commit-id=…, build type=Release)`.
    description: String,
    /// The backend's own handle from `open_runtime_library`. Kept, never dropped, so the
    /// library's reference count never falls to zero under `ort`, which holds another.
    _library: libloading::Library,
}

/// What this process knows about its ONNX Runtime.
enum RuntimeSlot {
    /// Loaded and handed to `ort`; every backend in the process runs on it.
    Loaded(LoadedRuntime),
    /// `ort` refused a library after the backend's own checks had passed it. `ort` cannot
    /// retry within a process — see `ensure_runtime` — so every later load is refused.
    Unusable {
        location: RuntimeLocation,
        reason: String,
    },
}

/// `None` until a library has been handed to `ort`, so an attempt the backend's own checks
/// refused (a wrong path, a file that is not a runtime, one too old) touches nothing of
/// `ort`'s and can be retried with a correct one.
static RUNTIME: Mutex<Option<RuntimeSlot>> = Mutex::new(None);

/// Where the runtime library must come from for `graph`. The places, in order:
///
/// 1. `configured`, the path the application passed ([`CONFIGURED_SETTING`]);
/// 2. [`RUNTIME_ENV`] (`env_value`);
/// 3. [`RUNTIME_FILE_NAME`] in the package root, beside the graph.
///
/// The first place that is *set* decides, and a later one is never tried in its place: a
/// path the application passed, or the variable, that names nothing is refused rather than
/// skipped, because falling back would run a runtime nobody chose. Set but empty is refused
/// rather than read as unset, as for the tuning variables: it names no file, and falling
/// back would be a guess about what was meant. Every refusal says what each place looked
/// at held, in this order, so whoever reads it knows which one to fix.
///
/// Taken as arguments rather than read here so the rule is testable without mutating the
/// process environment. Returns the canonical path and the place it came from, which every
/// later refusal of the library names too.
fn resolve_runtime_path(
    configured: Option<&Path>,
    env_value: Option<std::ffi::OsString>,
    graph: &Path,
) -> Result<RuntimeLocation, EmbeddingError> {
    let unavailable = |reason: String| EmbeddingError::OnnxRuntimeUnavailable { reason };
    let beside = onnx_package_root(graph).join(RUNTIME_FILE_NAME);

    if let Some(requested) = configured {
        if requested.as_os_str().is_empty() {
            return Err(unavailable(format!(
                "the application passed an empty path as the ONNX Runtime library \
                 ({CONFIGURED_SETTING}), the first place looked; it must name the shared \
                 library ({RUNTIME_FILE_NAME}), or be None to look at {RUNTIME_ENV} and then \
                 beside the model"
            )));
        }
        let path = requested.canonicalize().map_err(|e| {
            unavailable(format!(
                "the application passed {} as the ONNX Runtime library ({CONFIGURED_SETTING}), \
                 which cannot be opened: {e}. It is the first place looked and, once passed, the \
                 only one: neither {RUNTIME_ENV} nor the model's folder is tried in its place",
                requested.display()
            ))
        })?;
        if let Some(passed_over) = env_value.filter(|value| !value.is_empty()) {
            log::info!(
                "Embedding backend '{}': {RUNTIME_ENV} is set to {}, but the application passed \
                 {}, which comes first",
                OnnxBackend::ID,
                Path::new(&passed_over).display(),
                path.display()
            );
        }
        return Ok(RuntimeLocation {
            path,
            source: RuntimeSource::Configured,
        });
    }

    if let Some(raw) = env_value {
        if raw.is_empty() {
            return Err(unavailable(format!(
                "the application passed no ONNX Runtime library ({CONFIGURED_SETTING}), and \
                 {RUNTIME_ENV}, the next place looked, is set but empty; it must name the shared \
                 library ({RUNTIME_FILE_NAME}), or be unset to use the one beside the model at {}",
                beside.display()
            )));
        }
        let requested = PathBuf::from(&raw);
        let path = requested.canonicalize().map_err(|e| {
            unavailable(format!(
                "the application passed no ONNX Runtime library ({CONFIGURED_SETTING}), and \
                 {RUNTIME_ENV}, the next place looked, is set to {}, which cannot be opened: {e}. \
                 Once set, the variable is the only place looked: the file beside the model is \
                 not tried in its place",
                requested.display()
            ))
        })?;
        return Ok(RuntimeLocation {
            path,
            source: RuntimeSource::Environment,
        });
    }

    let path = beside.canonicalize().map_err(|e| {
        unavailable(format!(
            "no ONNX Runtime shared library to load for {}. Looked, in order, at (1) the path \
             the application passes ({CONFIGURED_SETTING}): none passed; (2) {RUNTIME_ENV}: not \
             set; (3) {}, beside the model: {e}. Provide {RUNTIME_FILE_NAME} (ONNX Runtime 1.17 \
             or newer; the reference is Microsoft's official 1.28.0 release) in any one of them",
            graph.display(),
            beside.display()
        ))
    })?;
    Ok(RuntimeLocation {
        path,
        source: RuntimeSource::BesideGraph,
    })
}

/// Load the runtime `runtime` names unless this process already has — the same file again
/// is a no-op, from whichever place it came, and a different one is refused. Returns the
/// runtime's version and build, for the load log.
///
/// The library is opened and checked here first (`open_runtime_library`), and only then
/// handed to `ort::init_from`. The order is load-bearing: `ort` 2.0.0-rc.13 cannot survive
/// a refusal of its own. Its `OnceLock` runs the loader under `Once::call_once_force`, so a
/// load that *fails* still completes the `Once`, leaving the library slot marked
/// initialized with nothing in it — every later `init_from` then "succeeds" without
/// loading, and the next `ort` call panics on the missing `OrtGetApiBase` (strictly, reads
/// uninitialized memory) and poisons `ort`'s environment lock for the rest of the process.
/// So `ort` is only ever given a library it will accept — loadable, exporting
/// `OrtGetApiBase`, at least `1.{ort::MINOR_VERSION}` by `ort`'s own rule — and if it
/// refuses one anyway, that is remembered and every later load in the process is refused.
///
/// The first successful load also commits `ort`'s process-wide environment, so that the
/// runtime's log goes to the [`log`] facade — a library linked into a Flutter application
/// must not write to the host's stderr — and so that telemetry is off.
fn ensure_runtime(runtime: &RuntimeLocation) -> Result<String, EmbeddingError> {
    let unavailable = |reason: String| EmbeddingError::OnnxRuntimeUnavailable { reason };

    // Held across the load, so two first loads cannot race. Recovered from poisoning:
    // the slot only ever goes from empty to one complete value.
    let mut slot = RUNTIME.lock().unwrap_or_else(PoisonError::into_inner);
    match slot.as_ref() {
        Some(RuntimeSlot::Loaded(loaded)) if loaded.location.path == runtime.path => {
            return Ok(loaded.description.clone());
        }
        Some(RuntimeSlot::Loaded(loaded)) => {
            return Err(unavailable(format!(
                "this process already runs ONNX Runtime from {}, and a process can hold only \
                 one; {runtime} was requested. Ask for the library already loaded, or restart \
                 the process to switch",
                loaded.location
            )));
        }
        Some(RuntimeSlot::Unusable {
            location: refused,
            reason,
        }) => {
            return Err(unavailable(format!(
                "ONNX Runtime from {refused} was refused earlier in this process ({reason}), and \
                 the runtime binding cannot load another after that; restart the process with a \
                 working library"
            )));
        }
        None => {}
    }

    // Nothing of `ort`'s is touched if this fails.
    let (library, version) = open_runtime_library(runtime)?;

    // Absolute on purpose: for a relative path `ort` resolves against the executable's
    // directory through two `expect`s.
    let environment = match ort::init_from(&runtime.path) {
        Ok(environment) => environment,
        Err(e) => {
            let reason = e.to_string();
            *slot = Some(RuntimeSlot::Unusable {
                location: runtime.clone(),
                reason: reason.clone(),
            });
            return Err(unavailable(format!(
                "ONNX Runtime {version} at {runtime} passed this backend's checks but the \
                 runtime binding refused it: {reason}"
            )));
        }
    };
    // Recorded the moment `ort` holds the library, before anything else can fail: it
    // cannot be unloaded, and `ort` would silently ignore a later, different path.
    *slot = Some(RuntimeSlot::Loaded(LoadedRuntime {
        location: runtime.clone(),
        description: version.clone(),
        _library: library,
    }));

    let committed = environment
        .with_name("otzaria-semantic-search")
        .with_telemetry(false)
        .with_logger(Arc::new(forward_runtime_log))
        .commit();
    if !committed {
        log::debug!(
            "Embedding backend '{}': an ONNX Runtime environment was already configured in \
             this process; keeping it",
            OnnxBackend::ID
        );
    }
    // Created now rather than by the first session, so its logging level is lowered
    // before anything is logged: a custom logger is installed at VERBOSE, which would
    // hand every internal trace to the facade.
    let environment = ort::environment::Environment::current().map_err(|e| {
        unavailable(format!(
            "ONNX Runtime from {runtime} loaded, but its environment could not be created: {e}"
        ))
    })?;
    environment.set_log_level(LogLevel::Warning);

    let description = format!("{version} ({})", describe_build(ort::info()));
    if let Some(RuntimeSlot::Loaded(runtime)) = slot.as_mut() {
        runtime.description.clone_from(&description);
    }
    Ok(description)
}

/// Open `runtime` as an ONNX Runtime library and check it the way `ort::init_from` will —
/// so that `init_from` is never handed a library it refuses (see `ensure_runtime`).
/// Returns the handle, to be kept, and the runtime's version string
/// (`OrtGetApiBase()->GetVersionString()`), which `ort` reads but does not expose.
fn open_runtime_library(
    runtime: &RuntimeLocation,
) -> Result<(libloading::Library, String), EmbeddingError> {
    type GetApiBase = unsafe extern "system" fn() -> *const ort::sys::OrtApiBase;
    let unavailable = |reason: String| EmbeddingError::OnnxRuntimeUnavailable { reason };

    // SAFETY: loading a library runs its initializers, and there is no way to vet a file
    // before that: this is the library the process is about to run semantic search on,
    // chosen by the deployment (the application's path, `OTZARIA_ONNX_RUNTIME` or the
    // package's own file), and loading it is what `ort::init_from` does next in any case.
    let library = unsafe { libloading::Library::new(&runtime.path) }.map_err(|e| {
        unavailable(format!(
            "{runtime} is not a loadable shared library for this platform: {e}"
        ))
    })?;
    // SAFETY: the type is ONNX Runtime's C declaration of its entry point,
    // `const OrtApiBase* ORT_API_CALL OrtGetApiBase(void)`, exactly as `ort-sys` binds it.
    let get_api_base = unsafe { library.get::<GetApiBase>(b"OrtGetApiBase") }.map_err(|_| {
        unavailable(format!(
            "{runtime} loads but does not export OrtGetApiBase, so it is not ONNX Runtime"
        ))
    })?;
    // SAFETY: a call with no arguments into the loaded runtime, which returns null or a
    // pointer to a static table living as long as the library.
    let base = unsafe { get_api_base() };
    if base.is_null() {
        return Err(unavailable(format!(
            "{runtime}: OrtGetApiBase returned nothing"
        )));
    }
    // SAFETY: `base` is non-null and points at that table. `GetVersionString` returns a
    // NUL-terminated string the runtime owns ("do not deallocate"), valid while the
    // library is loaded; it is copied out before `library` can be dropped.
    let raw_version = unsafe { ((*base).GetVersionString)() };
    if raw_version.is_null() {
        return Err(unavailable(format!(
            "{runtime}: the runtime reports no version"
        )));
    }
    // SAFETY: non-null and NUL-terminated, per the contract above.
    let version = unsafe { std::ffi::CStr::from_ptr(raw_version) }
        .to_string_lossy()
        .into_owned();

    // `ort`'s own rule, verbatim: the second dotted field against its API level.
    let minor = version
        .split('.')
        .nth(1)
        .and_then(|field| field.parse::<u32>().ok())
        .unwrap_or(0);
    if minor < ort::MINOR_VERSION {
        return Err(unavailable(format!(
            "{runtime} is ONNX Runtime {version}; this backend needs 1.{} or newer (the \
             reference is Microsoft's official 1.28.0 release)",
            ort::MINOR_VERSION
        )));
    }
    Ok((library, version))
}

/// The fields of `ort::info()` worth a log line. Microsoft's 1.28.0 answers `ORT Build
/// Info: git-branch=HEAD, git-commit-id=da9b5e364c, fp8-kv-cache=1, build type=Release`;
/// other builds append `, cmake cxx flags: …`, which runs to hundreds of characters and
/// is cut.
fn describe_build(info: &str) -> String {
    let info = info.strip_prefix("ORT Build Info: ").unwrap_or(info);
    let head = info.split(", cmake").next().unwrap_or(info);
    head.trim().to_string()
}

/// Forwards one runtime log record to the [`log`] facade. Called from ONNX Runtime's
/// threads through `extern "C"`, so it must not panic, and does nothing that could.
fn forward_runtime_log(level: LogLevel, _category: &str, _id: &str, location: &str, message: &str) {
    // Info is mapped down, as for llama.cpp: session creation is chatty at that level.
    let level = match level {
        LogLevel::Verbose => log::Level::Trace,
        LogLevel::Info => log::Level::Debug,
        LogLevel::Warning => log::Level::Warn,
        LogLevel::Error | LogLevel::Fatal => log::Level::Error,
    };
    log::log!(target: "onnxruntime", level, "{message} ({location})");
}

// ─────────────────────────────── the graph ───────────────────────────────

/// How a graph's inputs and outputs are wired, established once at load.
#[derive(Debug, Clone)]
struct Wiring {
    /// Whether the graph declares `token_type_ids`, which is then fed as zeros.
    token_type_ids: bool,
    /// The first output's name: the only output requested from a run.
    output: String,
    /// Whether the graph fixes the batch at 1 (else it is dynamic). Reported only: the
    /// backend runs one text per call either way.
    batch_fixed_at_one: bool,
    /// The output width the graph declares, or `None` for a dynamic one — then the probe
    /// measures it.
    declared_dim: Option<u32>,
}

/// Check the graph's signature against what this backend can feed and read.
///
/// Every input ONNX Runtime reports is required: a graph input that also has an
/// initializer is an overridable initializer, which the runtime lists separately.
fn inspect_graph(session: &Session, graph: &Path) -> Result<Wiring, EmbeddingError> {
    let invalid = |reason: String| EmbeddingError::InvalidModelFile {
        path: graph.display().to_string(),
        reason,
    };

    let mut seen_ids = false;
    let mut seen_mask = false;
    let mut token_type_ids = false;
    let mut batch_fixed_at_one = false;
    for input in session.inputs() {
        let name = input.name();
        match name {
            INPUT_IDS => seen_ids = true,
            ATTENTION_MASK => seen_mask = true,
            TOKEN_TYPE_IDS => token_type_ids = true,
            other => {
                return Err(invalid(format!(
                    "the graph requires an input named '{other}' ({}), which this backend \
                     cannot feed: it provides {INPUT_IDS}, {ATTENTION_MASK} and, when \
                     declared, {TOKEN_TYPE_IDS} (as zeros). Re-export the graph without it",
                    input.dtype()
                )))
            }
        }
        let ValueType::Tensor { ty, shape, .. } = input.dtype() else {
            return Err(invalid(format!(
                "input '{name}' is {}, not a tensor",
                input.dtype()
            )));
        };
        if *ty != TensorElementType::Int64 {
            return Err(invalid(format!(
                "input '{name}' is {}; this backend feeds int64, as Hugging Face exports do",
                input.dtype()
            )));
        }
        let [batch, sequence] = shape[..] else {
            return Err(invalid(format!(
                "input '{name}' is {}; a [batch, sequence] tensor is required",
                input.dtype()
            )));
        };
        match batch {
            -1 => {}
            1 => batch_fixed_at_one = true,
            fixed => {
                return Err(invalid(format!(
                    "input '{name}' fixes the batch at {fixed}; this backend runs one text at a \
                     time and needs a batch dimension of 1 or a dynamic one"
                )))
            }
        }
        if sequence != -1 {
            return Err(invalid(format!(
                "input '{name}' fixes the sequence length at {sequence}; this backend feeds each \
                 text at its own length and needs a dynamic sequence dimension"
            )));
        }
    }
    for (seen, required) in [(seen_ids, INPUT_IDS), (seen_mask, ATTENTION_MASK)] {
        if !seen {
            return Err(invalid(format!(
                "the graph has no '{required}' input; this backend feeds {INPUT_IDS} and \
                 {ATTENTION_MASK}"
            )));
        }
    }

    let Some(first) = session.outputs().first() else {
        return Err(invalid("the graph declares no output".to_string()));
    };
    let ValueType::Tensor { ty, shape, .. } = first.dtype() else {
        return Err(invalid(format!(
            "the first output, '{}', is {}, not a tensor",
            first.name(),
            first.dtype()
        )));
    };
    if *ty != TensorElementType::Float32 {
        return Err(invalid(format!(
            "the first output, '{}', is {}; a float32 sentence vector is required",
            first.name(),
            first.dtype()
        )));
    }
    let declared_dim = match shape[..] {
        [batch, dim] => {
            if batch != -1 && batch != 1 {
                return Err(invalid(format!(
                    "the first output, '{}', is {}: a batch of {batch} for one text",
                    first.name(),
                    first.dtype()
                )));
            }
            match dim {
                -1 => None,
                width if width > 0 => Some(u32::try_from(width).map_err(|_| {
                    invalid(format!(
                        "the first output declares a width of {width}, which is not a \
                         plausible embedding"
                    ))
                })?),
                other => {
                    return Err(invalid(format!(
                        "the first output declares a width of {other}"
                    )))
                }
            }
        }
        [_, _, _] => {
            return Err(invalid(format!(
                "the first output, '{}', is {}: token-level hidden states, not a sentence \
                 vector. This backend serves pooling '{}' only — the graph must end in its own \
                 pooling and emit [batch, dim]; pooling token states in Rust is not implemented \
                 in v1. Re-export the graph with its pooling inside",
                first.name(),
                first.dtype(),
                Pooling::InGraph
            )))
        }
        _ => {
            return Err(invalid(format!(
                "the first output, '{}', is {}; a [batch, dim] sentence vector is required",
                first.name(),
                first.dtype()
            )))
        }
    };

    Ok(Wiring {
        token_type_ids,
        output: first.name().to_string(),
        batch_fixed_at_one,
        declared_dim,
    })
}

/// One session, configured the one way this backend's vectors are defined for.
///
/// Returns `ort::Result` so that `?` converts the builder's `Error<SessionBuilder>` —
/// which is neither `Send` nor `Sync` — into a plain `ort::Error` here, before it can
/// reach an error type that must be.
fn build_session(graph: &Path, intra_threads: usize) -> ort::Result<Session> {
    build_session_as(graph, intra_threads, true)
}

/// [`build_session`], with [`X64_QUANT_PRECISION`] on or off. The backend only ever asks
/// for on; off is for the golden tests, which report what the entry changes on the CPU
/// they run on, and need every other setting to be the backend's for that to mean anything.
fn build_session_as(
    graph: &Path,
    intra_threads: usize,
    exact_x86_int8: bool,
) -> ort::Result<Session> {
    Session::builder()?
        .with_optimization_level(GRAPH_OPTIMIZATION)?
        .with_config_entry(X64_QUANT_PRECISION, if exact_x86_int8 { "1" } else { "0" })?
        .with_intra_threads(intra_threads)?
        // Spinning keeps idle intra-op threads busy between runs: throughput on a
        // benchmark, battery on a phone, and no effect on a vector.
        .with_intra_op_spinning(false)?
        // Sequential execution has no inter-op pool; said explicitly so a changed
        // runtime default cannot start one.
        .with_parallel_execution(false)?
        .with_inter_threads(1)?
        .commit_from_file(graph)
}

/// What a run needs besides the session: which output to fetch. A `RunOptions` with a
/// selected output is `Send` but not `Sync`, so each session carries its own.
struct Worker {
    session: Session,
    /// Fetches the first output only, so a graph that also exports token states does
    /// not copy them out on every run.
    options: RunOptions<HasSelectedOutputs>,
}

impl Worker {
    fn new(session: Session, wiring: &Wiring) -> ort::Result<Self> {
        let options =
            RunOptions::new()?.with_outputs(OutputSelector::no_default().with(&wiring.output));
        Ok(Self { session, options })
    }

    /// Run one text's ids, `[1, len]`, and return the first output's single row.
    fn run(&mut self, wiring: &Wiring, ids: &[u32]) -> ort::Result<RunOutput> {
        let len = ids.len();
        let input_ids: Vec<i64> = ids.iter().map(|&id| i64::from(id)).collect();
        let mut inputs = ort::inputs![
            INPUT_IDS => Tensor::from_array(([1usize, len], input_ids))?,
            ATTENTION_MASK => Tensor::from_array(([1usize, len], vec![1i64; len]))?,
        ];
        if wiring.token_type_ids {
            inputs.push((
                TOKEN_TYPE_IDS.into(),
                Tensor::from_array(([1usize, len], vec![0i64; len]))?.into(),
            ));
        }

        let outputs = self.session.run_with_options(inputs, &self.options)?;
        let Some(value) = outputs.get(&wiring.output) else {
            return Ok(RunOutput::Missing);
        };
        let (shape, data) = value.try_extract_tensor::<f32>()?;
        // Copied out before the session is released: the outputs borrow it.
        Ok(RunOutput::Tensor {
            shape: shape.to_vec(),
            data: data.to_vec(),
        })
    }
}

/// What came back from one run, before it is checked.
enum RunOutput {
    Tensor {
        shape: Vec<i64>,
        data: Vec<f32>,
    },
    /// The runtime returned no value under the requested output's name.
    Missing,
}

// ─────────────────────────────── the pool ───────────────────────────────

/// What the pool's mutex guards. Every mutation is a single `push`, `pop` or counter
/// increment, so no panic can leave it half-updated, which is why a poisoned lock is
/// recovered rather than reported.
struct PoolState<T> {
    idle: Vec<T>,
    /// The ticket the next caller takes, and the ticket whose turn it is.
    next_ticket: u64,
    now_serving: u64,
}

/// A bounded set of sessions, lent out one inference at a time, first come first
/// served. Generic so the queueing can be tested without a runtime; the backend's is a
/// `Pool<Worker>`.
///
/// A checkout pool rather than a shared session, because `Session::run` needs `&mut`;
/// FIFO rather than whoever wakes first, because the caller that just returned a
/// session is already running and would otherwise take it straight back, starving a
/// query queued behind another caller's batch. Never fails and never blocks forever: every
/// [`Lease`] returns its item when dropped — during unwinding too — so the pool cannot
/// shrink, and a caller that finds every item lent out waits for one.
struct Pool<T> {
    state: Mutex<PoolState<T>>,
    /// Signalled when an item is returned *or* the turn advances; waiters re-check.
    turn: Condvar,
    size: usize,
}

impl<T> Pool<T> {
    fn new(items: Vec<T>) -> Self {
        let size = items.len();
        Self {
            state: Mutex::new(PoolState {
                idle: items,
                next_ticket: 0,
                now_serving: 0,
            }),
            turn: Condvar::new(),
            size,
        }
    }

    fn check_out(&self) -> Lease<'_, T> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let ticket = state.next_ticket;
        state.next_ticket = state.next_ticket.wrapping_add(1);
        loop {
            if state.now_serving == ticket {
                if let Some(item) = state.idle.pop() {
                    state.now_serving = state.now_serving.wrapping_add(1);
                    drop(state);
                    // The next ticket may find another idle item.
                    self.turn.notify_all();
                    return Lease {
                        pool: self,
                        item: Some(item),
                    };
                }
            }
            state = self
                .turn
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }

    fn check_in(&self, item: T) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.idle.push(item);
        drop(state);
        self.turn.notify_all();
    }
}

/// One item, borrowed from the pool until dropped.
struct Lease<'p, T> {
    pool: &'p Pool<T>,
    /// `Some` from construction until `drop`, the only place it is taken.
    item: Option<T>,
}

impl<T> Lease<'_, T> {
    fn get(&mut self) -> &mut T {
        match self.item.as_mut() {
            Some(item) => item,
            None => unreachable!("a lease holds its item until it is dropped"),
        }
    }
}

impl<T> Drop for Lease<'_, T> {
    fn drop(&mut self) {
        if let Some(item) = self.item.take() {
            self.pool.check_in(item);
        }
    }
}

// ─────────────────────────────── the backend ───────────────────────────────

/// Real ONNX inference: ONNX Runtime, the package's `tokenizer.json`, the graph's own
/// pooling. Construct with [`OnnxBackend::open`].
pub struct OnnxBackend {
    /// Padding off, truncation at `max_tokens`. `encode(&self)`, so shared unlocked.
    tokenizer: Tokenizer,
    pool: Pool<Worker>,
    wiring: Wiring,
    /// The graph's output width.
    dim: u32,
    /// The total sequence length every input is truncated to, special tokens included.
    max_tokens: usize,
    /// How many special tokens the tokenizer adds to every input.
    special_tokens: usize,
    intra_threads: usize,
    graph: PathBuf,
    runtime: RuntimeLocation,
}

impl OnnxBackend {
    /// Identifier recorded in the manifest as `embedding_backend`.
    ///
    /// It names the wiring — ONNX Runtime; `tokenizer.json` feeding `input_ids` and
    /// `attention_mask` (plus `token_type_ids` as zeros, only when the graph declares
    /// that input); the first output taken as the sentence vector — and a version bumped
    /// when that wiring changes what comes out for the same inputs. A change invalidates
    /// every stored vector.
    ///
    /// Also part of the wiring, and so covered by the same version: one text per run,
    /// graph optimization `All` (`GRAPH_OPTIMIZATION`), and exact int8 products on x86
    /// (`X64_QUANT_PRECISION`). Deliberately **not** the runtime's version or the thread
    /// count, as `LlamaCppBackend::ID` leaves out the llama.cpp build:
    /// `docs/ONNX_BACKEND.md` measures what they move.
    pub const ID: &'static str = "onnxruntime-sentence-v1";

    /// Load the graph at `graph` with the tokenizer at `tokenizer`, truncating every
    /// input to `max_tokens` tokens in total, special tokens included.
    ///
    /// `pooling` is what the configuration asks for; the backend serves only
    /// [`Pooling::InGraph`], and proves at load that the graph's first output is a
    /// finished `[batch, dim]` sentence vector.
    ///
    /// `runtime_library` is the ONNX Runtime library the application passes
    /// ([`EmbeddingDeployment::onnx_runtime`](crate::semantic::embedding::EmbeddingDeployment::onnx_runtime)):
    /// the first place looked, and once given the only one. `None` looks at
    /// `OTZARIA_ONNX_RUNTIME`, then for the platform's file name beside the graph (see the
    /// module docs). A deployment fact like `tuning`, and like it no part of [`Self::ID`].
    ///
    /// Everything is checked here rather than on the first search, cheapest first: the
    /// pooling, the tuning and a cap no encoder has — above
    /// [`ONNX_MAX_TOKENS_CEILING`](crate::semantic::backend::ONNX_MAX_TOKENS_CEILING),
    /// since the probe below is as long as the cap; that both files exist; that the
    /// tokenizer parses and leaves `max_tokens` room for content; the runtime library
    /// (see the module docs); the graph's inputs and first output; and finally a probe of
    /// exactly `max_tokens` tokens through **every** session, which must come back
    /// finite, `dim` long and identical across sessions — so a cap the graph cannot run,
    /// or a session that will not allocate, fails here and not in the middle of an
    /// index.
    ///
    /// # Errors
    ///
    /// * [`EmbeddingError::PoolingMismatch`] for a pooling other than `in-graph`;
    /// * [`EmbeddingError::ModelNotFound`] / [`EmbeddingError::TokenizerNotFound`];
    /// * [`EmbeddingError::LoadFailed`] for a zero thread or session count, a cap past
    ///   any encoder's context or one that leaves no room for content, or a session that
    ///   will not allocate;
    /// * [`EmbeddingError::OnnxRuntimeUnavailable`] when no ONNX Runtime can be loaded —
    ///   none found, a path passed or named that cannot be opened, too old, not a runtime,
    ///   or a different one already loaded — saying where the backend looked;
    /// * [`EmbeddingError::InvalidModelFile`] for a tokenizer or graph this backend
    ///   cannot serve, or a graph that fails the probe.
    pub fn open(
        graph: &Path,
        tokenizer: &Path,
        max_tokens: usize,
        pooling: Pooling,
        tuning: &OnnxBackendConfig,
        runtime_library: Option<&Path>,
    ) -> Result<Self, EmbeddingError> {
        let started = std::time::Instant::now();

        if pooling != Pooling::InGraph {
            return Err(EmbeddingError::PoolingMismatch {
                backend: Self::ID.to_string(),
                configured: pooling.to_string(),
                actual: Pooling::InGraph.to_string(),
            });
        }
        // Checked here too, because a caller building the struct literally never goes
        // through `from_env_for`.
        for (value, name) in [
            (tuning.intra_threads, "intra_threads"),
            (tuning.sessions, "sessions"),
        ] {
            if value == 0 {
                return Err(EmbeddingError::LoadFailed {
                    reason: format!(
                        "OnnxBackendConfig::{name} is 0; at least one is needed to run anything"
                    ),
                });
            }
        }
        // Checked here too, for a caller that never went through `EmbeddingConfig::validate`:
        // the probe below is as long as the cap (`ONNX_MAX_TOKENS_CEILING` says why).
        if let Some(reason) =
            max_tokens_past_the_format("max_tokens", max_tokens, ModelFormat::Onnx)
        {
            return Err(EmbeddingError::LoadFailed { reason });
        }
        if !graph.is_file() {
            return Err(EmbeddingError::ModelNotFound {
                path: graph.display().to_string(),
            });
        }
        if !tokenizer.is_file() {
            return Err(EmbeddingError::TokenizerNotFound {
                path: tokenizer.display().to_string(),
            });
        }

        // ── the tokenizer: pure Rust, so before any runtime is touched ──
        let (tokenizer_impl, special_tokens) = load_tokenizer(graph, tokenizer, max_tokens)?;

        // ── the runtime ──
        let runtime_location =
            resolve_runtime_path(runtime_library, std::env::var_os(RUNTIME_ENV), graph)?;
        let runtime = ensure_runtime(&runtime_location)?;

        let cores = available_parallelism();
        let demand = tuning.sessions.saturating_mul(tuning.intra_threads);
        if demand > cores {
            // Not clamped: the count was asked for explicitly, and oversubscription is
            // slow, not wrong.
            log::warn!(
                "Embedding backend '{}': {} session(s) x {} intra-op thread(s) is {demand} \
                 threads on a {cores}-core machine; runs that overlap will contend. Lower {} \
                 or {}.",
                Self::ID,
                tuning.sessions,
                tuning.intra_threads,
                OnnxBackendConfig::ENV_SESSIONS,
                OnnxBackendConfig::ENV_THREADS,
            );
        }

        // ── the sessions ──
        let invalid = |reason: String| EmbeddingError::InvalidModelFile {
            path: graph.display().to_string(),
            reason,
        };
        let first = build_session(graph, tuning.intra_threads).map_err(|e| {
            invalid(format!(
                "ONNX Runtime could not load the graph ({:?}): {}",
                e.code(),
                e.message()
            ))
        })?;
        let wiring = inspect_graph(&first, graph)?;

        let mut workers = Vec::with_capacity(tuning.sessions);
        let worker_failed = |e: ort::Error| EmbeddingError::LoadFailed {
            reason: format!("could not prepare an inference session: {e}"),
        };
        workers.push(Worker::new(first, &wiring).map_err(worker_failed)?);
        for index in 1..tuning.sessions {
            // The file is evidently valid by now, so a failure here is a resource one.
            let session = build_session(graph, tuning.intra_threads).map_err(|e| {
                EmbeddingError::LoadFailed {
                    reason: format!(
                        "session {} of {} for {} could not be created: {e} (each session holds \
                         its own copy of the weights; lower {})",
                        index + 1,
                        tuning.sessions,
                        graph.display(),
                        OnnxBackendConfig::ENV_SESSIONS
                    ),
                }
            })?;
            workers.push(Worker::new(session, &wiring).map_err(worker_failed)?);
        }

        let mut backend = Self {
            tokenizer: tokenizer_impl,
            pool: Pool::new(workers),
            dim: wiring.declared_dim.unwrap_or(0),
            wiring,
            max_tokens,
            special_tokens,
            intra_threads: tuning.intra_threads,
            graph: graph.to_path_buf(),
            runtime: runtime_location,
        };

        // ── the probe: only running the graph proves it serves this cap ──
        backend.dim = backend.probe_every_session()?;

        log::info!(
            "Embedding backend '{}' ready: {} — dim {} ({}), max_tokens {max_tokens} \
             ({special_tokens} special), batch dimension {} (one text per run), inputs \
             {INPUT_IDS} + {ATTENTION_MASK}{}, output '{}', {} session(s) x {} intra-op \
             thread(s), graph optimization {GRAPH_OPTIMIZATION:?}, {X64_QUANT_PRECISION} = 1; \
             ONNX Runtime {runtime} from {}; loaded in {} ms",
            Self::ID,
            backend.graph.display(),
            backend.dim,
            if backend.wiring.declared_dim.is_some() {
                "declared"
            } else {
                "measured"
            },
            if backend.wiring.batch_fixed_at_one {
                "fixed at 1"
            } else {
                "dynamic"
            },
            if backend.wiring.token_type_ids {
                " + token_type_ids (zeros)"
            } else {
                ""
            },
            backend.wiring.output,
            backend.pool.size,
            backend.intra_threads,
            backend.runtime,
            started.elapsed().as_millis(),
        );
        Ok(backend)
    }

    /// Run a probe of exactly `max_tokens` tokens through every session and return the
    /// width the graph produced.
    ///
    /// Every session rather than one: each allocates its own activations on its first
    /// run, and on a phone that allocation is where memory runs out — better here than
    /// on the first concurrent search. The sessions must also agree bit for bit, which is
    /// the cheapest proof that they run the same computation.
    fn probe_every_session(&self) -> Result<u32, EmbeddingError> {
        let invalid = |reason: String| EmbeddingError::InvalidModelFile {
            path: self.graph.display().to_string(),
            reason,
        };

        let words = PROBE_WORDS.len() * self.max_tokens;
        let text = PROBE_WORDS
            .iter()
            .cycle()
            .take(words)
            .copied()
            .collect::<Vec<_>>()
            .join(" ");
        let ids = self.token_ids(&text)?;
        if ids.len() != self.max_tokens {
            return Err(invalid(format!(
                "a {words}-word probe tokenized to {} ids where truncation to exactly {} was \
                 configured; the tokenizer does not truncate the way this backend requires",
                ids.len(),
                self.max_tokens
            )));
        }

        // Taken all at once: leases returned one by one would hand back the same session.
        let mut leases: Vec<Lease<'_, Worker>> =
            (0..self.pool.size).map(|_| self.pool.check_out()).collect();
        let mut reference: Option<Vec<f32>> = None;
        for (index, lease) in leases.iter_mut().enumerate() {
            let vector = self.run_checked(lease, &ids).map_err(|e| {
                invalid(format!(
                    "the load-time probe of exactly max_tokens = {} tokens failed on session {} \
                     of {}: {e}. If the model has fewer positions, lower max_tokens",
                    self.max_tokens,
                    index + 1,
                    self.pool.size
                ))
            })?;
            let norm = vector
                .iter()
                .map(|x| f64::from(*x) * f64::from(*x))
                .sum::<f64>()
                .sqrt();
            if !norm.is_finite() || norm <= 0.0 {
                return Err(invalid(format!(
                    "the probe produced a vector of magnitude {norm}; the graph is not usable \
                     as an embedding model"
                )));
            }
            match &reference {
                None => {
                    log::debug!(
                        "Embedding backend '{}': a {}-token probe has raw L2 norm {norm:.6}",
                        Self::ID,
                        ids.len()
                    );
                    reference = Some(vector);
                }
                Some(first) if first != &vector => {
                    return Err(EmbeddingError::LoadFailed {
                        reason: format!(
                            "session {} of {} computed a different probe vector from session 1; \
                             the sessions do not run the same computation",
                            index + 1,
                            self.pool.size
                        ),
                    })
                }
                Some(_) => {}
            }
        }
        drop(leases);

        let produced = reference.map_or(0, |v| v.len());
        u32::try_from(produced)
            .ok()
            .filter(|width| *width > 0)
            .ok_or_else(|| invalid(format!("the probe produced a {produced}-component vector")))
    }

    /// The exact ids fed to the graph for `text`: specials included, after truncation.
    /// Shared by [`EmbeddingBackend::tokenize`] and [`EmbeddingBackend::embed_batch_raw`]
    /// so that what the goldens assert is what inference consumes.
    fn token_ids(&self, text: &str) -> Result<Vec<u32>, EmbeddingError> {
        let encoding =
            self.tokenizer
                .encode(text, true)
                .map_err(|e| EmbeddingError::InferenceFailed {
                    reason: format!("tokenizing a {}-byte text failed: {e}", text.len()),
                })?;
        let ids = encoding.get_ids();
        if ids.len() > self.max_tokens {
            // Unreachable while truncation is configured — but the alternative is feeding
            // the graph a length the probe never proved.
            return Err(EmbeddingError::InferenceFailed {
                reason: format!(
                    "the tokenizer produced {} ids against a cap of {}; truncation should have \
                     prevented this",
                    ids.len(),
                    self.max_tokens
                ),
            });
        }
        Ok(ids.to_vec())
    }

    /// One run on a leased session, checked: exactly one row of the expected width.
    fn run_checked(
        &self,
        lease: &mut Lease<'_, Worker>,
        ids: &[u32],
    ) -> Result<Vec<f32>, EmbeddingError> {
        if ids.is_empty() {
            return Err(EmbeddingError::InferenceFailed {
                reason: "the text tokenized to no ids at all, not even special tokens; there is \
                         nothing to embed"
                    .to_string(),
            });
        }
        let output =
            lease
                .get()
                .run(&self.wiring, ids)
                .map_err(|e| EmbeddingError::InferenceFailed {
                    reason: format!(
                        "ONNX Runtime failed on a {}-token input ({:?}): {}",
                        ids.len(),
                        e.code(),
                        e.message()
                    ),
                })?;
        let (shape, data) = match output {
            RunOutput::Tensor { shape, data } => (shape, data),
            RunOutput::Missing => {
                return Err(EmbeddingError::InferenceFailed {
                    reason: format!("the run returned no output named '{}'", self.wiring.output),
                })
            }
        };
        let [1, width] = shape[..] else {
            return Err(EmbeddingError::InferenceFailed {
                reason: format!(
                    "the first output came back with shape {shape:?} for one text; [1, dim] was \
                     required"
                ),
            });
        };
        let width = u32::try_from(width).unwrap_or(0);
        // `self.dim` is 0 only during the probe of a graph with a dynamic width, where
        // the probe itself decides it.
        if self.dim != 0 && width != self.dim {
            return Err(EmbeddingError::DimensionMismatch {
                expected: self.dim,
                actual: width,
            });
        }
        if data.len() != width as usize {
            return Err(EmbeddingError::InferenceFailed {
                reason: format!(
                    "the output declares {width} components and holds {}",
                    data.len()
                ),
            });
        }
        Ok(data)
    }
}

/// Read `path` as a Hugging Face tokenizer and configure it the one way this backend
/// feeds a graph: padding off, special tokens matched in text, truncation on the right
/// to `max_tokens` in total. Returns it with the number of special tokens it adds.
///
/// Whatever padding or truncation the file carries is replaced, not trusted: padding
/// would feed `[PAD]` ids as content, and a stored truncation would cap every input at
/// the file's length rather than the configured one.
///
/// A refusal names `graph` as the model file, as every error about an ONNX package does,
/// and the tokenizer in its reason: the package is the unit that is valid or not.
fn load_tokenizer(
    graph: &Path,
    path: &Path,
    max_tokens: usize,
) -> Result<(Tokenizer, usize), EmbeddingError> {
    let invalid = |reason: String| EmbeddingError::InvalidModelFile {
        path: graph.display().to_string(),
        reason: format!("its tokenizer, {}, {reason}", path.display()),
    };
    let mut tokenizer = Tokenizer::from_file(path)
        .map_err(|e| invalid(format!("is not a Hugging Face tokenizer.json: {e}")))?;

    tokenizer.with_padding(None);
    // The default, said explicitly because the role prefix depends on it: `false` means
    // added special tokens are matched in the input (see the module docs).
    tokenizer.set_encode_special_tokens(false);

    // Checked before truncation is configured: the tokenizer subtracts this count from
    // the cap unchecked, so a smaller cap would wrap around.
    let special_tokens = tokenizer
        .get_post_processor()
        .map_or(0, |processor| processor.added_tokens(false));
    if max_tokens <= special_tokens {
        return Err(EmbeddingError::LoadFailed {
            reason: format!(
                "max_tokens is {max_tokens}, but the tokenizer adds {special_tokens} special \
                 token(s) to every input and the cap counts them, so no content would reach the \
                 model; at least {} is needed",
                special_tokens + 1
            ),
        });
    }
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: max_tokens,
            strategy: TruncationStrategy::LongestFirst,
            direction: TruncationDirection::Right,
            stride: 0,
        }))
        .map_err(|e| invalid(format!("refuses truncation to {max_tokens} tokens: {e}")))?;
    Ok((tokenizer, special_tokens))
}

impl std::fmt::Debug for OnnxBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxBackend")
            .field("id", &Self::ID)
            .field("graph", &self.graph)
            .field("dim", &self.dim)
            .field("max_tokens", &self.max_tokens)
            .field("special_tokens", &self.special_tokens)
            .field("wiring", &self.wiring)
            .field("sessions", &self.pool.size)
            .field("intra_threads", &self.intra_threads)
            .field("runtime", &self.runtime)
            .finish()
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

    /// The exact sequence [`Self::embed_batch_raw`] feeds the graph: specials included,
    /// after truncation.
    fn tokenize(&self, text: &str) -> Result<Vec<u32>, EmbeddingError> {
        self.token_ids(text)
    }

    /// Embed one batch, raw and unnormalized, one vector per input in input order.
    ///
    /// Order is structural: texts are run one at a time and pushed as they return. All
    /// of them are tokenized first, so a text the tokenizer rejects fails the batch
    /// before any inference is spent on it. Each run leases a session of its own — see
    /// the module docs on why a query must not wait for a whole batch.
    ///
    /// Neither normalized nor screened — `EmbeddingRuntime::embed_batch` does both.
    fn embed_batch_raw(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let sequences = texts
            .iter()
            .map(|text| self.token_ids(text))
            .collect::<Result<Vec<_>, _>>()?;

        let mut out = Vec::with_capacity(sequences.len());
        for ids in &sequences {
            let mut lease = self.pool.check_out();
            let vector = self.run_checked(&mut lease, ids)?;
            drop(lease);
            out.push(vector);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic::backend::ONNX_MAX_TOKENS_CEILING;
    use std::sync::Mutex;

    /// Every test that reads or writes the tuning variables holds this, because the test
    /// harness runs them on parallel threads of one process. `OTZARIA_ONNX_RUNTIME` is
    /// only ever read here: the rule it feeds is tested through `resolve_runtime_path`.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn fixture(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/onnx_fixture")
            .join(name)
    }

    /// Whether a runtime library is configured, saying so loudly when it is not. The
    /// tests that run a graph skip without one rather than fail, as the model-gated
    /// tests do: nothing is linked, so a machine without the library cannot run them.
    fn runtime_configured() -> bool {
        match std::env::var_os(RUNTIME_ENV) {
            Some(path) if !path.is_empty() => true,
            _ => {
                println!(
                    "SKIPPED: {RUNTIME_ENV} is not set. This test runs a graph and needs an ONNX \
                     Runtime shared library (1.17 or newer; Microsoft's 1.28.0 release is the \
                     reference), e.g. {RUNTIME_ENV}=/path/to/{RUNTIME_FILE_NAME}"
                );
                false
            }
        }
    }

    fn tuning(intra_threads: usize, sessions: usize) -> OnnxBackendConfig {
        OnnxBackendConfig {
            intra_threads,
            sessions,
        }
    }

    fn open_fixture(
        graph: &str,
        max_tokens: usize,
        tuning: &OnnxBackendConfig,
    ) -> Result<OnnxBackend, EmbeddingError> {
        OnnxBackend::open(
            &fixture(graph),
            &fixture("tokenizer.json"),
            max_tokens,
            Pooling::InGraph,
            tuning,
            None,
        )
    }

    fn expected() -> serde_json::Value {
        let path = fixture("expected.json");
        let raw = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        serde_json::from_str(&raw).expect("expected.json is not valid JSON")
    }

    fn ids_of(value: &serde_json::Value) -> Vec<u32> {
        value
            .as_array()
            .expect("an id list")
            .iter()
            .map(|id| u32::try_from(id.as_u64().expect("an id")).expect("a u32 id"))
            .collect()
    }

    fn l2_norm(v: &[f32]) -> f64 {
        v.iter()
            .map(|x| f64::from(*x) * f64::from(*x))
            .sum::<f64>()
            .sqrt()
    }

    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        let dot: f64 = a
            .iter()
            .zip(b)
            .map(|(x, y)| f64::from(*x) * f64::from(*y))
            .sum();
        dot / (l2_norm(a) * l2_norm(b))
    }

    /// Texts distinct enough that a reordering would show.
    const TEXTS: [&str; 6] = [
        "בראשית ברא אלהים",
        "the quick brown fox",
        "[PASSAGE] את השמים ואת הארץ",
        "[QUERY] lazy dog",
        "תורה",
        "",
    ];

    // ── configuration, refused before any runtime is touched ──

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

        assert_eq!(tuning.unwrap(), self::tuning(3, 2));
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

    /// Checked first, before any file is opened: the nonexistent paths below would
    /// otherwise be what the error reported.
    #[test]
    fn a_pooling_other_than_in_graph_is_refused_before_anything_is_read() {
        for pooling in [Pooling::LastToken, Pooling::Mean] {
            let refused = OnnxBackend::open(
                Path::new("absent/model.onnx"),
                Path::new("absent/tokenizer.json"),
                256,
                pooling,
                &OnnxBackendConfig::default(),
                None,
            );
            match refused {
                Err(EmbeddingError::PoolingMismatch {
                    backend,
                    configured,
                    actual,
                }) => {
                    assert_eq!(backend, OnnxBackend::ID);
                    assert_eq!(configured, pooling.as_str());
                    assert_eq!(actual, "in-graph");
                }
                other => panic!("pooling {pooling} must be refused, got {other:?}"),
            }
        }
    }

    #[test]
    fn a_literal_tuning_with_zero_threads_or_sessions_is_refused_and_named() {
        for (tuning, field) in [
            (self::tuning(0, 1), "intra_threads"),
            (self::tuning(1, 0), "sessions"),
        ] {
            match open_fixture("dynamic.onnx", 32, &tuning) {
                Err(EmbeddingError::LoadFailed { reason }) => {
                    assert!(reason.contains(field), "must name {field}: {reason}");
                }
                other => panic!("{tuning:?} must be refused, got {other:?}"),
            }
        }
    }

    #[test]
    fn a_missing_graph_or_tokenizer_is_reported_as_what_is_missing() {
        let missing_graph = OnnxBackend::open(
            &fixture("absent.onnx"),
            &fixture("tokenizer.json"),
            32,
            Pooling::InGraph,
            &tuning(1, 1),
            None,
        );
        assert!(
            matches!(&missing_graph, Err(EmbeddingError::ModelNotFound { path }) if path.ends_with("absent.onnx")),
            "{missing_graph:?}"
        );

        let missing_tokenizer = OnnxBackend::open(
            &fixture("dynamic.onnx"),
            &fixture("absent-tokenizer.json"),
            32,
            Pooling::InGraph,
            &tuning(1, 1),
            None,
        );
        assert!(
            matches!(&missing_tokenizer, Err(EmbeddingError::TokenizerNotFound { path }) if path.ends_with("absent-tokenizer.json")),
            "{missing_tokenizer:?}"
        );
    }

    /// The fixture's tokenizer adds `[CLS]` and `[SEP]`, and the cap counts them, so 2
    /// leaves no room and 3 is the smallest usable cap. Refused before truncation is
    /// configured, where the tokenizer's own arithmetic would underflow.
    #[test]
    fn a_cap_that_leaves_no_room_for_content_is_refused() {
        for cap in [0usize, 1, 2] {
            match load_tokenizer(&fixture("dynamic.onnx"), &fixture("tokenizer.json"), cap) {
                Err(EmbeddingError::LoadFailed { reason }) => assert!(
                    reason.contains(&format!("max_tokens is {cap}"))
                        && reason.contains("2 special")
                        && reason.contains("at least 3"),
                    "{reason}"
                ),
                other => panic!("a cap of {cap} must be refused, got {:?}", other.err()),
            }
        }
        let (_, specials) = load_tokenizer(&fixture("dynamic.onnx"), &fixture("tokenizer.json"), 3)
            .expect("3 fits");
        assert_eq!(specials, 2);
    }

    /// The load-time probe is `max_tokens` tokens long, so its cost is the cap's. A cap
    /// past any encoder's context — a mistyped setting, a `u32` from the application —
    /// overflowed the probe's arithmetic or asked the allocator for hundreds of gigabytes,
    /// which aborts the whole process instead of failing the load. Refused first, with no
    /// runtime needed.
    #[test]
    fn a_cap_past_any_encoders_context_is_refused_before_anything_is_loaded() {
        for cap in [
            ONNX_MAX_TOKENS_CEILING + 1,
            u32::MAX as usize,
            usize::MAX / 2,
            usize::MAX,
        ] {
            match open_fixture("dynamic.onnx", cap, &tuning(1, 1)) {
                Err(EmbeddingError::LoadFailed { reason }) => assert!(
                    reason.contains(&format!("max_tokens is {cap}"))
                        && reason.contains(&ONNX_MAX_TOKENS_CEILING.to_string()),
                    "{reason}"
                ),
                other => panic!("a cap of {cap} must be refused, got {:?}", other.err()),
            }
        }
        // The ceiling itself is only a bound on the probe, not a claim that a graph runs
        // it: `a_cap_the_graph_cannot_run_fails_at_load` is what refuses 49 for this one.
        assert!(load_tokenizer(
            &fixture("dynamic.onnx"),
            &fixture("tokenizer.json"),
            ONNX_MAX_TOKENS_CEILING
        )
        .is_ok());
    }

    /// Named as the package's fault: the graph is the model file, the reason says which
    /// of its files is wrong.
    #[test]
    fn a_file_that_is_not_a_tokenizer_is_refused_as_an_invalid_package() {
        // A graph is certainly not a tokenizer.
        match load_tokenizer(&fixture("static_batch.onnx"), &fixture("dynamic.onnx"), 32) {
            Err(EmbeddingError::InvalidModelFile { path, reason }) => {
                assert!(path.ends_with("static_batch.onnx"), "{path}");
                assert!(reason.contains("dynamic.onnx"), "{reason}");
                assert!(reason.contains("tokenizer"), "{reason}");
            }
            other => panic!("expected InvalidModelFile, got {:?}", other.err()),
        }
    }

    /// A tokenizer shaped like the production model's — Unigram, Metaspace, and a
    /// normalizer of NFKC plus regex `Replace` rules over niqqud, cantillation, bidi
    /// controls, quote marks and dashes — through this build's regex engine
    /// (`fancy-regex`), against the Python package's (onig) ids. Needs no runtime, so the
    /// engine choice is checked in every run, not only where the real model is.
    #[test]
    fn a_production_shaped_tokenizer_matches_the_python_reference_id_for_id() {
        let data = expected();
        let cases = data["unigram_cases"].as_array().expect("unigram_cases");
        assert!(cases.len() >= 10, "the regex rules need their cases");
        let tokenizer_path = fixture(data["unigram_tokenizer"].as_str().expect("its file"));
        for case in cases {
            let name = case["name"].as_str().unwrap();
            let cap = case["max_tokens"].as_u64().unwrap() as usize;
            let (tokenizer, specials) =
                load_tokenizer(&fixture("dynamic.onnx"), &tokenizer_path, cap).unwrap();
            assert_eq!(specials, 2);
            let produced = tokenizer
                .encode(case["text"].as_str().unwrap(), true)
                .unwrap()
                .get_ids()
                .to_vec();
            assert_eq!(produced, ids_of(&case["token_ids"]), "{name}");
        }
    }

    /// What text recipe 2 hands this tokenizer for a query typed with whitespace around it
    /// must tokenize exactly as the bare query does. The production-shaped tokenizer is
    /// the one that can tell: its Metaspace pre-tokenizer turns a doubled space after
    /// `[QUERY]` — and a trailing one — into a lone `▁` id, which the fixture's WordPiece
    /// tokenizer would silently drop.
    #[test]
    fn a_query_padded_with_whitespace_reaches_the_tokenizer_as_the_bare_query_does() {
        use crate::semantic::recipe::{query_input, EmbeddingTextRecipe, TextNormalizationRecipe};

        let (tokenizer, _) = load_tokenizer(
            &fixture("dynamic.onnx"),
            &fixture("tokenizer_unigram.json"),
            64,
        )
        .unwrap();
        let ids = |text: &str| tokenizer.encode(text, true).unwrap().get_ids().to_vec();
        let lone_metaspace = tokenizer
            .token_to_id("▁")
            .expect("the fixture has a bare ▁");
        let embedded = |query: &str| {
            query_input(
                EmbeddingTextRecipe::RolePrefixedLineOrNeighbourContext,
                TextNormalizationRecipe::AsSuppliedByCorpus,
                query,
            )
            .unwrap()
        };

        let bare = ids(&embedded("מצות תפילין"));
        assert!(!bare.contains(&lone_metaspace), "{bare:?}");
        for padded in [" מצות תפילין", "מצות תפילין ", "\u{00A0}מצות תפילין\t"]
        {
            assert_eq!(ids(&embedded(padded)), bare, "{padded:?}");
        }
        // The tokenizer fact the trim is for: the same query, prefixed untrimmed.
        assert!(ids("[QUERY]  מצות תפילין").contains(&lone_metaspace));
    }

    /// The passage side: a line the chunker's character cap ends on a space reaches the
    /// production-shaped tokenizer, under text recipe 2, without the lone `▁` that space
    /// would have become just before `[SEP]`.
    #[test]
    fn a_passage_capped_on_a_space_reaches_the_tokenizer_without_a_trailing_metaspace() {
        use crate::semantic::chunker::{Chunker, ChunkerConfig};
        use crate::semantic::types::{BookForIndexing, BookLine};

        let (tokenizer, _) = load_tokenizer(
            &fixture("dynamic.onnx"),
            &fixture("tokenizer_unigram.json"),
            64,
        )
        .unwrap();
        let ids = |text: &str| tokenizer.encode(text, true).unwrap().get_ids().to_vec();
        let lone_metaspace = tokenizer
            .token_to_id("▁")
            .expect("the fixture has a bare ▁");
        let sep = tokenizer.token_to_id("[SEP]").expect("[SEP]");

        let book = BookForIndexing {
            source_book_key: "book.txt".to_string(),
            title: "t".to_string(),
            content_fingerprint: 1,
            is_pdf: false,
            topics: String::new(),
            extra_facets: vec![],
            lines: vec![BookLine {
                line_id: 1,
                section_id: 1,
                segment: 1,
                reference: "t 1".to_string(),
                line_hash: 1,
                // 12 characters in, the cap below, is the space after "תפילין".
                text: "מצות תפילין בכל יום חוץ משבתות וימים טובים".to_string(),
            }],
        };
        let chunks = Chunker::new(ChunkerConfig {
            max_chunk_chars: 12,
            embedding_text_version: 2,
            ..ChunkerConfig::default()
        })
        .unwrap()
        .chunk_book(&book);
        let passage = ids(&chunks[0].embedding_text);
        assert_eq!(passage, ids("[PASSAGE] מצות תפילין"));
        assert_eq!(passage.last(), Some(&sep));
        assert_ne!(passage[passage.len() - 2], lone_metaspace, "{passage:?}");
        // The tokenizer fact the trim is for: the same passage with the space kept.
        let untrimmed = ids("[PASSAGE] מצות תפילין ");
        assert_eq!(
            untrimmed[untrimmed.len() - 2],
            lone_metaspace,
            "{untrimmed:?}"
        );
    }

    /// The stand-in's stub package reaches this backend first in a mock + ONNX build, so
    /// its tokenizer — WordLevel, no post-processor, hence no special tokens — must load
    /// with this crate's `tokenizers`, and the refusal come from the graph.
    #[test]
    fn the_stand_ins_stub_tokenizer_loads() {
        let dir = TempDir::new("stub_tokenizer");
        let graph = crate::semantic::embedding::mock::write_stub_onnx_package(&dir.0);
        let (tokenizer, specials) =
            load_tokenizer(&graph, &dir.0.join("tokenizer.json"), 256).expect("it loads");
        assert_eq!(specials, 0);
        assert!(!tokenizer
            .encode("[QUERY] x", true)
            .unwrap()
            .get_ids()
            .is_empty());
    }

    /// The tokenizer file sets padding to 16 and truncation to 512; neither may survive.
    #[test]
    fn the_files_own_padding_is_turned_off_and_its_truncation_replaced() {
        let (tokenizer, _) =
            load_tokenizer(&fixture("dynamic.onnx"), &fixture("tokenizer.json"), 8).unwrap();
        assert!(tokenizer.get_padding().is_none());
        let truncation = tokenizer
            .get_truncation()
            .expect("truncation is configured");
        assert_eq!(truncation.max_length, 8);
        assert_eq!(truncation.direction, TruncationDirection::Right);
        assert_eq!(truncation.stride, 0);

        let short = tokenizer.encode("the fox", true).unwrap();
        assert_eq!(
            short.get_ids(),
            [2, 7, 10, 3],
            "no [PAD] (0) may be appended"
        );
        let long = tokenizer
            .encode("the quick brown fox jumps over the lazy dog", true)
            .unwrap();
        assert_eq!(long.get_ids().len(), 8);
    }

    // ── the runtime library ──

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(name: &str) -> Self {
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "otzaria_onnx_backend_{name}_{}_{}",
                std::process::id(),
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A file at `path`, directories made, standing in for a runtime library where only the
    /// lookup is under test. Returns its canonical path.
    fn touch(path: &Path) -> PathBuf {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"not really a runtime").unwrap();
        path.canonicalize().unwrap()
    }

    fn located(path: PathBuf, source: RuntimeSource) -> RuntimeLocation {
        RuntimeLocation { path, source }
    }

    /// First hit wins: the application's path, then the variable, then the file beside the
    /// graph — each only when every place before it is unset.
    #[test]
    fn the_runtime_comes_from_the_application_then_the_variable_then_beside_the_graph() {
        let dir = TempDir::new("runtime_path");
        let graph = dir.0.join("model.onnx");
        std::fs::write(&graph, b"graph").unwrap();
        let beside = touch(&dir.0.join(RUNTIME_FILE_NAME));
        let named = touch(&dir.0.join("named").join(RUNTIME_FILE_NAME));
        let passed = touch(&dir.0.join("bundled").join(RUNTIME_FILE_NAME));
        let resolve = |configured: Option<&Path>, env: Option<&Path>| {
            resolve_runtime_path(configured, env.map(|path| path.into()), &graph).unwrap()
        };

        assert_eq!(
            resolve(None, None),
            located(beside, RuntimeSource::BesideGraph)
        );
        assert_eq!(
            resolve(None, Some(&named)),
            located(named.clone(), RuntimeSource::Environment)
        );
        for env in [Some(named.as_path()), None] {
            assert_eq!(
                resolve(Some(&passed), env),
                located(passed.clone(), RuntimeSource::Configured)
            );
        }
        // Two spellings of one file are one location, so the one-runtime rule sees one file.
        let spelled_otherwise = dir.0.join("named/../bundled").join(RUNTIME_FILE_NAME);
        assert_eq!(
            resolve(Some(&spelled_otherwise), None),
            located(passed, RuntimeSource::Configured)
        );
    }

    /// A place that is set decides even when it names nothing: refused, naming what it held,
    /// and never skipped in favour of a later place — although here both later places would
    /// resolve.
    #[test]
    fn a_place_that_is_set_but_names_nothing_is_refused_not_skipped() {
        let dir = TempDir::new("runtime_set_but_wrong");
        let graph = dir.0.join("model.onnx");
        std::fs::write(&graph, b"graph").unwrap();
        touch(&dir.0.join(RUNTIME_FILE_NAME));
        let named = touch(&dir.0.join("named").join(RUNTIME_FILE_NAME));
        let absent = dir.0.join("not-installed").join(RUNTIME_FILE_NAME);
        let refused = |configured: Option<&Path>, env: Option<&Path>| {
            let resolved = resolve_runtime_path(configured, env.map(|path| path.into()), &graph);
            match resolved {
                Err(EmbeddingError::OnnxRuntimeUnavailable { reason }) => reason,
                other => panic!("{configured:?} / {env:?} must be refused, got {other:?}"),
            }
        };

        // The application's path, with the variable set to a library that exists.
        let empty = refused(Some(Path::new("")), Some(&named));
        assert!(
            empty.contains(CONFIGURED_SETTING) && empty.contains("empty path"),
            "{empty}"
        );
        let wrong = refused(Some(&absent), Some(&named));
        assert!(
            wrong.contains(CONFIGURED_SETTING)
                && wrong.contains(&absent.display().to_string())
                && wrong.contains(&format!(
                    "neither {RUNTIME_ENV} nor the model's folder is tried"
                )),
            "{wrong}"
        );
        for reason in [&empty, &wrong] {
            assert!(
                !reason.contains(&named.display().to_string()),
                "the variable is not looked at: {reason}"
            );
        }

        // The variable, with a file beside the graph; the refusal says the application
        // passed nothing first.
        let empty = refused(None, Some(Path::new("")));
        assert!(
            empty.contains(CONFIGURED_SETTING)
                && empty.contains(&format!(
                    "{RUNTIME_ENV}, the next place looked, is set but empty"
                )),
            "{empty}"
        );
        let wrong = refused(None, Some(&absent));
        assert!(
            wrong.contains(CONFIGURED_SETTING)
                && wrong.contains(&format!("{RUNTIME_ENV}, the next place looked, is set to"))
                && wrong.contains(&absent.display().to_string())
                && wrong.contains("the file beside the model is not tried"),
            "{wrong}"
        );

        // Under a path the application passes, the variable is not read at all — not even
        // refused when it is empty.
        let passed = touch(&dir.0.join("bundled").join(RUNTIME_FILE_NAME));
        assert_eq!(
            resolve_runtime_path(Some(&passed), Some("".into()), &graph).unwrap(),
            located(passed, RuntimeSource::Configured)
        );
    }

    /// With nothing set and nothing beside the graph, the refusal goes through the three
    /// places in the order they are looked at: what a support message needs to say which
    /// one to fix.
    #[test]
    fn nothing_found_is_reported_place_by_place_in_lookup_order() {
        let dir = TempDir::new("no_runtime_anywhere");
        let graph = dir.0.join("model.onnx");
        std::fs::write(&graph, b"graph").unwrap();
        let reason = match resolve_runtime_path(None, None, &graph) {
            Err(EmbeddingError::OnnxRuntimeUnavailable { reason }) => reason,
            other => panic!("expected OnnxRuntimeUnavailable, got {other:?}"),
        };
        let beside = dir.0.join(RUNTIME_FILE_NAME).display().to_string();
        let at = |place: &str| {
            reason
                .find(place)
                .unwrap_or_else(|| panic!("{place} is not named: {reason}"))
        };
        assert!(
            at(CONFIGURED_SETTING) < at(RUNTIME_ENV) && at(RUNTIME_ENV) < at(&beside),
            "{reason}"
        );
    }

    /// Every refusal of a library names it through its location, so the place it came from
    /// travels with the path.
    #[test]
    fn a_location_says_which_place_it_came_from() {
        let path = PathBuf::from("/opt/ort").join(RUNTIME_FILE_NAME);
        let named = format!("named by {RUNTIME_ENV}");
        for (source, place) in [
            (RuntimeSource::Configured, "passed by the application"),
            (RuntimeSource::Environment, named.as_str()),
            (RuntimeSource::BesideGraph, "beside the model"),
        ] {
            assert_eq!(
                located(path.clone(), source).to_string(),
                format!("{} ({place})", path.display())
            );
        }
    }

    /// The backend is in this build; what is missing is the library it loads. Said as
    /// "no embedding backend is available in this build" — what a build without the
    /// feature says — it sends whoever reads `last_error` to rebuild an application that
    /// only needs a file put in place.
    #[test]
    fn a_runtime_that_cannot_be_loaded_is_reported_as_the_runtime_not_as_a_missing_backend() {
        let dir = TempDir::new("no_runtime");
        let graph = dir.0.join("model.onnx");
        std::fs::write(&graph, b"graph").unwrap();

        let missing = resolve_runtime_path(None, None, &graph).unwrap_err();
        assert!(
            matches!(missing, EmbeddingError::OnnxRuntimeUnavailable { .. }),
            "{missing:?}"
        );
        let missing = missing.to_string();
        assert!(
            missing.starts_with("ONNX Runtime could not be loaded: "),
            "{missing}"
        );
        // Still every way to provide one.
        assert!(missing.contains(CONFIGURED_SETTING), "{missing}");
        assert!(missing.contains(RUNTIME_ENV), "{missing}");
        assert!(
            missing.contains(&dir.0.join(RUNTIME_FILE_NAME).display().to_string()),
            "{missing}"
        );

        let not_a_runtime = located(
            fixture("expected.json").canonicalize().unwrap(),
            RuntimeSource::Configured,
        );
        for refused in [
            resolve_runtime_path(Some(Path::new("")), None, &graph).unwrap_err(),
            resolve_runtime_path(None, Some("".into()), &graph).unwrap_err(),
            ensure_runtime(&not_a_runtime).unwrap_err(),
        ] {
            let message = refused.to_string();
            assert!(
                message.starts_with("ONNX Runtime could not be loaded: "),
                "{message}"
            );
            assert!(!message.contains("No embedding backend"), "{message}");
        }
    }

    #[test]
    fn the_build_description_keeps_what_identifies_the_runtime() {
        assert_eq!(
            describe_build(
                "ORT Build Info: git-branch=rel-1.28.0, git-commit-id=abc123, build type=Release, \
                 cmake cxx flags: -O3 -DNDEBUG"
            ),
            "git-branch=rel-1.28.0, git-commit-id=abc123, build type=Release"
        );
        assert_eq!(
            describe_build(
                "ORT Build Info: git-branch=HEAD, git-commit-id=da9b5e364c, fp8-kv-cache=1, \
                 build type=Release"
            ),
            "git-branch=HEAD, git-commit-id=da9b5e364c, fp8-kv-cache=1, build type=Release"
        );
        assert_eq!(describe_build("something else"), "something else");
    }

    // ── the pool ──

    /// The queue is first come, first served — including against the caller that just
    /// returned the item and asks again, which is what keeps a query from waiting for a
    /// whole indexing batch.
    #[test]
    fn the_pool_serves_waiters_in_arrival_order_even_against_a_returning_caller() {
        let pool = Pool::new(vec![0u32]);
        let order = Mutex::new(Vec::<&str>::new());
        let tickets_taken = |pool: &Pool<u32>| {
            pool.state
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .next_ticket
        };
        let wait_for_tickets = |pool: &Pool<u32>, count: u64| {
            while tickets_taken(pool) < count {
                std::thread::yield_now();
            }
        };

        std::thread::scope(|scope| {
            let mut held = pool.check_out(); // ticket 0
            scope.spawn(|| {
                let _lease = pool.check_out(); // ticket 1
                order.lock().unwrap().push("first waiter");
            });
            wait_for_tickets(&pool, 2);
            scope.spawn(|| {
                let _lease = pool.check_out(); // ticket 2
                order.lock().unwrap().push("second waiter");
            });
            wait_for_tickets(&pool, 3);

            *held.get() += 1;
            drop(held);
            // Straight back into the queue, as an indexing batch does between texts.
            let _again = pool.check_out(); // ticket 3
            order.lock().unwrap().push("returning caller");
        });

        assert_eq!(
            order.into_inner().unwrap(),
            ["first waiter", "second waiter", "returning caller"]
        );
    }

    /// A caller that panics mid-inference still hands its session back, so the pool
    /// never shrinks and never leaves the next caller waiting forever.
    #[test]
    fn a_lease_dropped_by_a_panic_returns_its_item() {
        let pool = Pool::new(vec![1u32, 2]);
        let panicked = std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    let _lease = pool.check_out();
                    panic!("inference went wrong");
                })
                .join()
        });
        assert!(panicked.is_err());

        let mut a = pool.check_out();
        let mut b = pool.check_out();
        let mut both = [*a.get(), *b.get()];
        both.sort_unstable();
        assert_eq!(both, [1, 2], "both items are back");
    }

    #[test]
    fn the_backend_is_send_and_sync() {
        fn require<T: Send + Sync>() {}
        require::<OnnxBackend>();
        require::<Pool<Worker>>();
        require::<Tokenizer>();
        require::<Box<dyn EmbeddingBackend>>();
    }

    // ── against the fixture, with a runtime ──

    #[test]
    fn the_fixture_loads_and_reports_what_its_graph_declares() {
        if !runtime_configured() {
            return;
        }
        let backend = open_fixture("dynamic.onnx", 32, &tuning(1, 1)).expect("the fixture loads");
        assert_eq!(backend.id(), "onnxruntime-sentence-v1");
        assert!(backend.is_semantic());
        assert_eq!(backend.dim(), 4);
        assert_eq!(backend.max_tokens(), 32);
        assert_eq!(backend.pooling(), Pooling::InGraph);
        assert!(!backend.wiring.token_type_ids);
        assert!(!backend.wiring.batch_fixed_at_one);
        assert_eq!(backend.wiring.output, "sentence_embedding");
        assert_eq!(backend.special_tokens, 2);
        println!("{backend:?}");
    }

    /// **The fixture's parity gate**: token ids exactly, then vectors, against what the
    /// Python `tokenizers` and `onnxruntime` packages produced for the same inputs.
    ///
    /// The ids carry the special-token cases: a prefix spelled as text becomes the
    /// learned token, so does `[CLS]` inside a book, and `[query]` does not — each the
    /// same in Rust as in the Python reference.
    #[test]
    fn token_ids_and_vectors_match_the_python_reference() {
        if !runtime_configured() {
            return;
        }
        let data = expected();
        let cases = data["cases"].as_array().expect("cases");
        let mut backends: std::collections::BTreeMap<usize, OnnxBackend> = Default::default();
        let mut worst = (1.0f64, String::new());

        for case in cases {
            let name = case["name"].as_str().unwrap();
            let text = case["text"].as_str().unwrap();
            let cap = case["max_tokens"].as_u64().unwrap() as usize;
            let backend = backends
                .entry(cap)
                .or_insert_with(|| open_fixture("dynamic.onnx", cap, &tuning(1, 1)).unwrap());

            let produced = backend.tokenize(text).unwrap();
            assert_eq!(produced, ids_of(&case["token_ids"]), "{name}: token ids");

            let vector = backend.embed_batch_raw(&[text]).unwrap().remove(0);
            let reference: Vec<f32> = case["vector"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect();
            let max_abs = vector
                .iter()
                .zip(&reference)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let cos = cosine(&vector, &reference);
            println!(
                "{name:<30} {:>3} ids  cos {cos:.9}  max|Δ| {max_abs:.3e}",
                produced.len()
            );
            assert!(max_abs <= 1e-5, "{name}: max |Δ| {max_abs:e}");
            if cos < worst.0 {
                worst = (cos, name.to_string());
            }
        }
        assert!(
            worst.0 >= 0.999_999,
            "worst cosine {} on {}",
            worst.0,
            worst.1
        );
    }

    /// The cap is the total: an over-long text keeps `[CLS]`, the prefix and as much
    /// leading content as fits, then `[SEP]` — the right end is what is cut.
    #[test]
    fn an_over_long_input_is_truncated_on_the_right_keeping_the_specials() {
        if !runtime_configured() {
            return;
        }
        let backend = open_fixture("dynamic.onnx", 6, &tuning(1, 1)).unwrap();
        let long = "[PASSAGE] the quick brown fox jumps over the lazy dog";
        let ids = backend.tokenize(long).unwrap();
        assert_eq!(
            ids,
            [2, 6, 7, 8, 9, 3],
            "[CLS] [PASSAGE] the quick brown [SEP]"
        );

        // What is embedded is exactly that prefix: the vector of the long text equals
        // the vector of the text cut by hand to what fits.
        let embedded = backend
            .embed_batch_raw(&[long, "[PASSAGE] the quick brown"])
            .unwrap();
        assert_eq!(embedded[0], embedded[1]);

        // At the cap exactly, nothing is dropped.
        let exact = backend.tokenize("the quick brown fox").unwrap();
        assert_eq!(exact, [2, 7, 8, 9, 10, 3]);
    }

    #[test]
    fn every_vector_comes_back_in_input_order_and_equal_to_its_single_twin() {
        if !runtime_configured() {
            return;
        }
        let backend = open_fixture("dynamic.onnx", 32, &tuning(1, 1)).unwrap();
        let batched = backend.embed_batch_raw(&TEXTS).unwrap();
        assert_eq!(batched.len(), TEXTS.len());
        for (index, text) in TEXTS.iter().enumerate() {
            let single = backend.embed_batch_raw(&[text]).unwrap().remove(0);
            // Bit for bit: one text per run means a batch is only a loop.
            assert_eq!(batched[index], single, "text {index} ({text:?})");
            assert_eq!(single.len(), 4);
        }
        // Distinct texts, distinct vectors — or the order assertion above proves nothing.
        for (i, a) in batched.iter().enumerate() {
            for b in &batched[i + 1..] {
                assert_ne!(a, b);
            }
        }

        let reversed: Vec<&str> = TEXTS.iter().rev().copied().collect();
        let mut backwards = backend.embed_batch_raw(&reversed).unwrap();
        backwards.reverse();
        assert_eq!(backwards, batched, "order in is order out");
    }

    #[test]
    fn an_empty_batch_returns_no_vectors() {
        if !runtime_configured() {
            return;
        }
        let backend = open_fixture("dynamic.onnx", 32, &tuning(1, 1)).unwrap();
        assert!(backend.embed_batch_raw(&[]).unwrap().is_empty());
    }

    /// The backend returns the graph's output as it is; `EmbeddingRuntime` normalizes.
    /// The fixture's projection is deliberately not unit-length, so a normalizing
    /// backend would show here.
    #[test]
    fn the_vector_is_returned_raw() {
        if !runtime_configured() {
            return;
        }
        let backend = open_fixture("dynamic.onnx", 32, &tuning(1, 1)).unwrap();
        for vector in backend.embed_batch_raw(&TEXTS).unwrap() {
            let norm = l2_norm(&vector);
            assert!((norm - 1.0).abs() > 1e-3, "norm {norm} looks normalized");
        }
    }

    /// Several threads inside `embed_batch_raw` at once through `&self`, against one
    /// session and against two: the answers are the serial ones, bit for bit.
    #[test]
    fn concurrent_callers_through_a_shared_reference_agree_with_serial_ones() {
        if !runtime_configured() {
            return;
        }
        for sessions in [1usize, 2] {
            let backend = open_fixture("dynamic.onnx", 32, &tuning(1, sessions)).unwrap();
            let serial = backend.embed_batch_raw(&TEXTS).unwrap();
            let concurrent = std::thread::scope(|scope| {
                let handles: Vec<_> = (0..4)
                    .map(|_| {
                        let backend = &backend;
                        scope.spawn(move || {
                            (0..25)
                                .map(|_| backend.embed_batch_raw(&TEXTS).unwrap())
                                .collect::<Vec<_>>()
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .flat_map(|h| h.join().unwrap())
                    .collect::<Vec<_>>()
            });
            assert_eq!(concurrent.len(), 100);
            for result in &concurrent {
                assert_eq!(
                    result, &serial,
                    "{sessions} session(s): concurrent != serial"
                );
            }
        }
    }

    /// Deployment knobs, not identity: the thread count and the session count must not
    /// move a vector, and do not.
    #[test]
    fn threads_and_sessions_do_not_change_a_vector() {
        if !runtime_configured() {
            return;
        }
        let reference = open_fixture("dynamic.onnx", 32, &tuning(1, 1))
            .unwrap()
            .embed_batch_raw(&TEXTS)
            .unwrap();
        for (threads, sessions) in [(2, 1), (4, 1), (1, 2), (2, 2)] {
            let produced = open_fixture("dynamic.onnx", 32, &tuning(threads, sessions))
                .unwrap()
                .embed_batch_raw(&TEXTS)
                .unwrap();
            assert_eq!(
                produced, reference,
                "{threads} thread(s) x {sessions} session(s)"
            );
        }
    }

    #[test]
    fn a_graph_with_its_batch_fixed_at_one_serves_the_same_vectors() {
        if !runtime_configured() {
            return;
        }
        let dynamic = open_fixture("dynamic.onnx", 32, &tuning(1, 1)).unwrap();
        let fixed = open_fixture("static_batch.onnx", 32, &tuning(1, 1)).unwrap();
        assert!(fixed.wiring.batch_fixed_at_one);
        assert_eq!(
            fixed.embed_batch_raw(&TEXTS).unwrap(),
            dynamic.embed_batch_raw(&TEXTS).unwrap()
        );
    }

    /// `token_types.onnx` adds a type embedding whose row 0 is zero: its vectors equal
    /// `dynamic.onnx`'s exactly when the backend feeds zeros, and differ for anything
    /// else it could have fed.
    #[test]
    fn token_type_ids_are_fed_as_zeros_when_the_graph_declares_them() {
        if !runtime_configured() {
            return;
        }
        let without = open_fixture("dynamic.onnx", 32, &tuning(1, 1)).unwrap();
        let with = open_fixture("token_types.onnx", 32, &tuning(1, 1)).unwrap();
        assert!(with.wiring.token_type_ids);
        assert_eq!(
            with.embed_batch_raw(&TEXTS).unwrap(),
            without.embed_batch_raw(&TEXTS).unwrap()
        );
    }

    #[test]
    fn a_token_level_graph_is_refused_as_needing_rust_side_pooling() {
        if !runtime_configured() {
            return;
        }
        match open_fixture("token_level.onnx", 32, &tuning(1, 1)) {
            Err(EmbeddingError::InvalidModelFile { path, reason }) => {
                assert!(path.ends_with("token_level.onnx"), "{path}");
                assert!(
                    reason.contains("last_hidden_state"),
                    "names the output: {reason}"
                );
                assert!(reason.contains("token-level"), "{reason}");
                assert!(reason.contains("in-graph"), "{reason}");
                assert!(reason.contains("not implemented"), "{reason}");
            }
            other => panic!("a rank-3 output must be refused, got {other:?}"),
        }
    }

    #[test]
    fn an_input_the_backend_cannot_feed_is_refused_by_name() {
        if !runtime_configured() {
            return;
        }
        match open_fixture("extra_input.onnx", 32, &tuning(1, 1)) {
            Err(EmbeddingError::InvalidModelFile { reason, .. }) => {
                assert!(reason.contains("'position_ids'"), "{reason}");
            }
            other => panic!("position_ids must be refused, got {other:?}"),
        }
    }

    /// The fixture has 48 positions. 48 runs; 49 is out of the position table, and the
    /// load-time probe is what says so — not the first long chunk of an index.
    #[test]
    fn a_cap_the_graph_cannot_run_fails_at_load() {
        if !runtime_configured() {
            return;
        }
        let fits = open_fixture("dynamic.onnx", 48, &tuning(1, 1)).expect("48 positions fit");
        assert_eq!(fits.max_tokens(), 48);

        match open_fixture("dynamic.onnx", 49, &tuning(1, 2)) {
            Err(EmbeddingError::InvalidModelFile { reason, .. }) => {
                assert!(reason.contains("max_tokens = 49"), "{reason}");
                assert!(reason.contains("lower max_tokens"), "{reason}");
            }
            other => panic!("a cap of 49 must fail at load, got {other:?}"),
        }
    }

    #[test]
    fn a_file_that_is_not_an_onnx_graph_is_refused() {
        if !runtime_configured() {
            return;
        }
        // Named as a graph, as every path the selection table sends here is.
        let dir = TempDir::new("not_a_graph");
        let graph = dir.0.join("not_a_graph.onnx");
        std::fs::write(&graph, b"{\"this is\": \"not protobuf\"}").unwrap();
        match OnnxBackend::open(
            &graph,
            &fixture("tokenizer.json"),
            32,
            Pooling::InGraph,
            &tuning(1, 1),
            None,
        ) {
            Err(error @ EmbeddingError::InvalidModelFile { .. }) => {
                let message = error.to_string();
                assert!(
                    message.starts_with("Not a valid ONNX model file"),
                    "{message}"
                );
                assert!(message.contains("could not load the graph"), "{message}");
                assert!(message.contains(&graph.display().to_string()), "{message}");
            }
            other => panic!("expected InvalidModelFile, got {other:?}"),
        }
    }

    /// Whether or not a runtime is already loaded in this test process, a file that is
    /// not one is refused as unavailable and named, with the place it came from — never a
    /// panic, which is what `ort`'s own search would have been.
    #[test]
    fn a_file_that_is_not_a_runtime_library_is_refused_and_named() {
        let not_a_library = located(
            fixture("expected.json").canonicalize().unwrap(),
            RuntimeSource::Configured,
        );
        match ensure_runtime(&not_a_library) {
            Err(EmbeddingError::OnnxRuntimeUnavailable { reason }) => {
                assert!(reason.contains(&not_a_library.to_string()), "{reason}");
            }
            other => panic!("expected OnnxRuntimeUnavailable, got {other:?}"),
        }
    }

    /// One runtime per process: once one is loaded, asking for another file is an
    /// error that names both, with where each came from, and asking for the same one
    /// again is not — from whichever place.
    #[test]
    fn a_second_different_runtime_library_is_refused() {
        if !runtime_configured() {
            return;
        }
        open_fixture("dynamic.onnx", 32, &tuning(1, 1)).expect("loads the runtime");
        let loaded = match RUNTIME
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .as_ref()
        {
            Some(RuntimeSlot::Loaded(runtime)) => runtime.location.clone(),
            _ => panic!("a runtime is loaded"),
        };
        for source in [
            RuntimeSource::Configured,
            RuntimeSource::Environment,
            RuntimeSource::BesideGraph,
        ] {
            assert!(
                ensure_runtime(&located(loaded.path.clone(), source)).is_ok(),
                "the same library is a no-op, {source:?}"
            );
        }

        let other = located(
            fixture("tokenizer.json").canonicalize().unwrap(),
            RuntimeSource::Configured,
        );
        match ensure_runtime(&other) {
            Err(EmbeddingError::OnnxRuntimeUnavailable { reason }) => {
                assert!(reason.contains(&loaded.to_string()), "{reason}");
                assert!(reason.contains(&other.to_string()), "{reason}");
            }
            other => panic!("a second runtime must be refused, got {other:?}"),
        }
    }
}

/// Verification against the real model's golden vectors. **This is the parity gate** for
/// the production graphs, not a placeholder for one.
///
/// The goldens (written by `tools/generate_onnx_golden_vectors.py`) are the Python
/// reference's answers: the ids from the `tokenizers` package and the vector from the
/// `onnxruntime` package, for inputs spelled exactly as the tokenizer receives them, role
/// prefix included. There is one golden file per graph of the package — `GOLDEN_FILES` —
/// and the test takes the one whose `graph_sha256` is the graph's, so
/// `OTZARIA_TEST_ONNX_MODEL` may name either: the int8 graph, the default identity, or the
/// fp32 graph it was quantized from. A graph none of them describes fails, loudly.
///
/// The graphs are gated and never committed, so each test is `#[ignore]`d *and* skips
/// loudly when `OTZARIA_TEST_ONNX_MODEL` is unset — which keeps the ordinary matrix green
/// on machines with no model. Run them with:
///
/// ```sh
/// OTZARIA_ONNX_RUNTIME=/path/to/libonnxruntime.dylib \
/// OTZARIA_TEST_ONNX_MODEL=/path/to/seforim-embed-round2-int8.onnx \
///   cargo test --lib --features onnx-backend onnx_backend::golden -- --ignored --nocapture
/// ```
///
/// The tokenizer is the package's, `tokenizer.json` beside the graph.
#[cfg(test)]
mod golden {
    use super::*;
    use crate::semantic::model_package::onnx_tokenizer_path;
    use std::collections::HashMap;

    /// Names either graph of the package: `seforim-embed-round2-int8.onnx` or
    /// `seforim-embed-round2-fp32.onnx`.
    const MODEL_ENV: &str = "OTZARIA_TEST_ONNX_MODEL";

    /// How a graph computes, which is what decides how far a vector may drift between two
    /// correct machines.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Arithmetic {
        Fp32,
        /// Dynamic INT8 quantization (`DynamicQuantizeLinear`, `MatMulInteger`).
        DynamicInt8,
    }

    /// Every golden file under `tests/data/`, and the arithmetic of the graph it describes.
    /// A graph is found in them by its SHA-256, never by its name.
    const GOLDEN_FILES: [(&str, Arithmetic); 2] = [
        ("onnx_golden_vectors_int8.json", Arithmetic::DynamicInt8),
        ("onnx_golden_vectors.json", Arithmetic::Fp32),
    ];

    /// The agreement required between this backend and the Python reference, per case,
    /// once the token ids have been proven **exactly** equal — for fp32 anywhere, and for
    /// int8 on the kind of machine the goldens were produced on.
    ///
    /// fp32 is not held to 1.0 because the two sides need not run the same ONNX Runtime
    /// build or even the same instruction set: kernel choice alone (KleidiAI on or off,
    /// fused or unfused operators) measured maxabs 2.9e-6 on identical inputs, a cosine
    /// deficit around 1e-11. 0.99999 leaves that noise six orders of magnitude of room and
    /// still fails every wiring error the ids cannot see — an `attention_mask` or
    /// `token_type_ids` fed wrong, the wrong output taken, a half-precision execution
    /// provider. Truncation and prefix errors never reach this check: they change the ids.
    ///
    /// int8 on the goldens' own machine class runs the same int8 kernels as the reference,
    /// through the same ONNX Runtime release, and measured bit-identical — 41 of 41 — so
    /// the same bound applies there too.
    const MIN_COSINE: f64 = 0.999_99;

    /// The bound for int8 on another CPU family than the goldens'. Its vectors are not
    /// reproducible across int8 kernels: on the reference machine itself, turning KleidiAI
    /// off moved components by 1.07e-2 (cosine 0.99896) and an unfused graph by 1.06e-2
    /// (0.99908), and x86 runs MLAS's own int8 kernels — exact ones only because the backend
    /// sets `X64_QUANT_PRECISION`; without it CI's x86-64 runner clamped its products and
    /// fell to cosine 0.9809 here. Two correct int8 approximations,
    /// each within cosine 0.99911 of the fp32 graph on these cases, can lie up to twice
    /// that angle apart — cosine 0.9964 — so the bound sits below that, at 0.995, and still
    /// far above what a wiring error produces (a wrong output or mask gives a vector that
    /// is not this text's at all). A different graph cannot slip through it: the golden
    /// file is chosen by the graph's SHA-256.
    const MIN_COSINE_INT8_ACROSS_MACHINES: f64 = 0.995;

    /// One golden file, as chosen for a graph.
    struct Goldens {
        file: &'static str,
        arithmetic: Arithmetic,
        data: serde_json::Value,
    }

    impl Goldens {
        fn header(&self) -> &serde_json::Value {
            &self.data["header"]
        }

        /// The cosine every vector must reach here, and why.
        fn min_cosine(&self) -> (f64, &'static str) {
            match (self.arithmetic, same_machine_as(self.header())) {
                (Arithmetic::Fp32, _) => (MIN_COSINE, "fp32: kernels move a component ~1e-6"),
                (Arithmetic::DynamicInt8, true) => (
                    MIN_COSINE,
                    "int8 on the goldens' machine class: the same int8 kernels",
                ),
                (Arithmetic::DynamicInt8, false) => (
                    MIN_COSINE_INT8_ACROSS_MACHINES,
                    "int8 on another CPU family: other int8 kernels",
                ),
            }
        }
    }

    /// Whether this process runs on the kind of machine the goldens were produced on: int8
    /// kernels are chosen per CPU family.
    fn same_machine_as(header: &serde_json::Value) -> bool {
        header["reference"]["machine"]
            .as_str()
            .and_then(rust_platform_of)
            == Some((std::env::consts::OS, std::env::consts::ARCH))
    }

    /// The generator's `f"{platform.system()} {platform.machine()}"` as Rust spells the
    /// same platform (`std::env::consts::{OS, ARCH}`), or `None` for one it does not know.
    fn rust_platform_of(machine: &str) -> Option<(&'static str, &'static str)> {
        let mut parts = machine.split_whitespace();
        let os = match parts.next()? {
            "Darwin" => "macos",
            "Linux" => "linux",
            "Windows" => "windows",
            _ => return None,
        };
        let arch = match parts.next()? {
            "arm64" | "aarch64" => "aarch64",
            "x86_64" | "AMD64" => "x86_64",
            _ => return None,
        };
        parts.next().is_none().then_some((os, arch))
    }

    /// Selection by hash has a failure mode of its own: a graph none of the files knows is
    /// a failure naming the graphs they do know — never a skip, and never another graph's
    /// goldens. Runs without a model: the fixture graph is known to no golden file.
    #[test]
    #[should_panic(expected = "is a graph no golden file describes")]
    fn a_graph_no_golden_file_describes_fails_loudly() {
        let fixture =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/onnx_fixture/dynamic.onnx");
        let _ = goldens_for(&fixture);
    }

    /// The machine the tolerance depends on is read the way the generator writes it.
    #[test]
    fn the_goldens_machine_is_read_as_the_generator_writes_it() {
        assert_eq!(rust_platform_of("Darwin arm64"), Some(("macos", "aarch64")));
        assert_eq!(rust_platform_of("Linux x86_64"), Some(("linux", "x86_64")));
        assert_eq!(
            rust_platform_of("Linux aarch64"),
            Some(("linux", "aarch64"))
        );
        assert_eq!(
            rust_platform_of("Windows AMD64"),
            Some(("windows", "x86_64"))
        );
        for unknown in ["", "Darwin", "FreeBSD amd64", "Darwin arm64 extra"] {
            assert_eq!(rust_platform_of(unknown), None, "{unknown:?}");
        }
        // Both committed golden files record a machine this mapping knows.
        for (file, _) in GOLDEN_FILES {
            let path = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/data")
                .join(file);
            let data: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
            let machine = data["header"]["reference"]["machine"].as_str().unwrap();
            assert!(rust_platform_of(machine).is_some(), "{file}: {machine}");
        }
    }

    /// The graph, or `None` with a loud explanation. Skipping rather than failing because
    /// CI has no model, and a test that fails there teaches everyone to ignore it.
    fn model_path() -> Option<PathBuf> {
        let path = match std::env::var(MODEL_ENV) {
            Ok(path) if !path.trim().is_empty() => PathBuf::from(path.trim()),
            _ => {
                println!(
                    "SKIPPED: {MODEL_ENV} is not set. This test needs one of the model's graphs \
                     — seforim-embed-round2-int8.onnx (42 MB) or -fp32.onnx (168 MB) — with \
                     tokenizer.json beside it; they are gated and never committed."
                );
                return None;
            }
        };
        if !path.is_file() {
            println!("SKIPPED: {MODEL_ENV} points at {path:?}, which does not exist");
            return None;
        }
        match std::env::var_os(RUNTIME_ENV) {
            Some(runtime) if !runtime.is_empty() => Some(path),
            _ => {
                println!(
                    "SKIPPED: {RUNTIME_ENV} is not set; running the graph needs an ONNX Runtime \
                     shared library"
                );
                None
            }
        }
    }

    /// The golden file that describes `graph`, chosen by the graph's SHA-256.
    ///
    /// Fails rather than skips, for a missing golden file and for a graph no golden file
    /// describes alike: asking for this gate and getting a green tick without one would be
    /// the worst outcome.
    fn goldens_for(graph: &Path) -> Goldens {
        let graph_sha = sha256_of(graph);
        let mut known = Vec::new();
        for (file, arithmetic) in GOLDEN_FILES {
            let data = read_golden_file(file);
            let header = &data["header"];
            let described = header["graph_sha256"].as_str().unwrap_or_default();
            if described == graph_sha {
                return Goldens {
                    file,
                    arithmetic,
                    data,
                };
            }
            known.push(format!(
                "{file}: {} {described}",
                header["graph_file"].as_str().unwrap_or("?")
            ));
        }
        panic!(
            "{} (SHA-256 {graph_sha}) is a graph no golden file describes, so there is nothing \
             to check it against. The goldens know:\n  {}\nPoint {MODEL_ENV} at one of those, \
             or generate goldens for this graph with tools/generate_onnx_golden_vectors.py",
            graph.display(),
            known.join("\n  ")
        )
    }

    /// One file of `GOLDEN_FILES`, parsed. A missing or broken one fails: the gate was asked for.
    fn read_golden_file(file: &str) -> serde_json::Value {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data")
            .join(file);
        let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!(
                "cannot read {}: {e}. Generate it with tools/generate_onnx_golden_vectors.py",
                path.display()
            )
        });
        serde_json::from_str(&raw).unwrap_or_else(|e| panic!("{file} is not valid JSON: {e}"))
    }

    /// The fp32 graph's golden vectors, by case name, each with the ids it was computed
    /// from: what an int8 vector is held against to see what quantization costs on the CPU
    /// at hand. Reported, never a gate — that bound is for CI's numbers to decide.
    fn fp32_reference(dim: usize) -> HashMap<String, (Vec<u32>, Vec<f32>)> {
        let (file, _) = GOLDEN_FILES
            .iter()
            .find(|(_, arithmetic)| *arithmetic == Arithmetic::Fp32)
            .expect("an fp32 golden file");
        read_golden_file(file)["cases"]
            .as_array()
            .expect("cases")
            .iter()
            .map(|case| {
                let vector =
                    golden_vector(case["vector_f32_le_base64"].as_str().expect("vector"), dim);
                (
                    case["name"].as_str().expect("name").to_string(),
                    (golden_ids(case), vector),
                )
            })
            .collect()
    }

    /// A case's token ids, as the generator recorded them.
    fn golden_ids(case: &serde_json::Value) -> Vec<u32> {
        case["token_ids"]
            .as_array()
            .expect("token_ids")
            .iter()
            .map(|id| id.as_u64().expect("an id") as u32)
            .collect()
    }

    /// The lowest cosine among the cases that have one, with its case's name.
    fn worst_case<'n>(cosines: impl IntoIterator<Item = (Option<f64>, &'n str)>) -> String {
        cosines
            .into_iter()
            .filter_map(|(cos, name)| cos.map(|cos| (cos, name)))
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .map_or_else(
                || "no case comparable".to_string(),
                |(cos, name)| format!("{cos:.10} ({name})"),
            )
    }

    /// The median of `values`, which must not be empty.
    fn median(values: &[f64]) -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        let middle = sorted.len() / 2;
        if sorted.len().is_multiple_of(2) {
            (sorted[middle - 1] + sorted[middle]) / 2.0
        } else {
            sorted[middle]
        }
    }

    fn sha256_of(path: &Path) -> String {
        use sha2::Digest;

        let mut hasher = sha2::Sha256::new();
        let mut file = std::fs::File::open(path)
            .unwrap_or_else(|e| panic!("cannot open {}: {e}", path.display()));
        let mut buffer = vec![0u8; 1 << 20];
        loop {
            let read = std::io::Read::read(&mut file, &mut buffer).expect("read");
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
        hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }

    /// Confirm the rest of the package behind `OTZARIA_TEST_ONNX_MODEL` is the one the
    /// goldens were produced from — the graph already chose them by its hash — since the
    /// alternative is a wall of id failures for a tokenizer that was simply another one.
    fn assert_is_the_golden_package(graph: &Path, header: &serde_json::Value) {
        let tokenizer = onnx_tokenizer_path(graph);
        assert_eq!(
            sha256_of(&tokenizer),
            header["tokenizer_sha256"]
                .as_str()
                .expect("tokenizer_sha256"),
            "{} is not the tokenizer the goldens were produced with",
            tokenizer.display()
        );
        // And the package as the index identity names it (design D4): the same two files,
        // through the validator `EmbeddingRuntime::load` runs.
        if let Some(expected) = header["package_checksum"].as_str() {
            let validated = crate::semantic::model_package::validate_model(graph)
                .unwrap_or_else(|e| panic!("the golden package does not validate: {e}"));
            assert_eq!(
                validated.checksum(),
                expected,
                "the package checksum differs from the goldens' header"
            );
        }
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

    fn golden_vector(encoded: &str, dim: usize) -> Vec<f32> {
        let bytes = base64_decode(encoded);
        assert_eq!(bytes.len(), dim * 4, "a golden vector is {dim} f32 values");
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    }

    /// Cosine in f64, so the accumulation cannot be what fails a 0.99999 threshold.
    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        let dot: f64 = a
            .iter()
            .zip(b)
            .map(|(x, y)| f64::from(*x) * f64::from(*y))
            .sum();
        let norm = |v: &[f32]| {
            v.iter()
                .map(|x| f64::from(*x) * f64::from(*x))
                .sum::<f64>()
                .sqrt()
        };
        dot / (norm(a) * norm(b))
    }

    fn open_golden(graph: &Path, max_tokens: usize, sessions: usize) -> OnnxBackend {
        OnnxBackend::open(
            graph,
            &onnx_tokenizer_path(graph),
            max_tokens,
            Pooling::InGraph,
            &OnnxBackendConfig {
                sessions,
                ..OnnxBackendConfig::default()
            },
            None,
        )
        .expect("the golden model must load")
    }

    fn inputs(data: &serde_json::Value) -> Vec<String> {
        data["cases"]
            .as_array()
            .expect("cases")
            .iter()
            .map(|case| case["input"].as_str().expect("input").to_string())
            .collect()
    }

    /// **The primary gate.** Exact token-id equality, then the vectors.
    ///
    /// Only `input` is read as input, so feeding the golden ids back cannot make the
    /// assertion a tautology. The ids carry the weight: truncation, the role prefix and
    /// special-token matching are all decided before the graph runs, and none of them is
    /// reliably visible in a cosine.
    ///
    /// For the int8 graph every vector is also held against the fp32 graph's golden for
    /// the same case — reported, not asserted, and printed before any assertion can fail:
    /// "int8 on this CPU against fp32" is the number quality depends on, and it is what
    /// a cross-machine bound has to be decided from.
    #[test]
    #[ignore = "needs a Meivin graph (int8 or fp32); set OTZARIA_TEST_ONNX_MODEL and pass --ignored"]
    fn token_ids_match_the_reference_exactly_and_vectors_agree() {
        let Some(graph) = model_path() else { return };
        let goldens = goldens_for(&graph);
        let (data, header) = (&goldens.data, goldens.header());
        assert_is_the_golden_package(&graph, header);
        let (min_cosine, why) = goldens.min_cosine();

        let max_tokens = header["max_tokens"].as_u64().expect("max_tokens") as usize;
        let dim = header["dim"].as_u64().expect("dim") as usize;
        let backend = open_golden(&graph, max_tokens, 1);
        assert_eq!(backend.dim() as usize, dim);
        assert_eq!(backend.max_tokens(), max_tokens);
        println!("\n{backend:?}");
        println!(
            "goldens: {} ({:?}, from {} on {}); reference: onnxruntime {}, tokenizers {}",
            goldens.file,
            goldens.arithmetic,
            header["graph_file"],
            header["reference"]["machine"],
            header["onnxruntime_version"],
            header["tokenizers_version"]
        );
        println!("required cosine >= {min_cosine} ({why})");
        let fp32 = (goldens.arithmetic == Arithmetic::DynamicInt8).then(|| fp32_reference(dim));

        let cases = data["cases"].as_array().expect("cases");
        assert!(!cases.is_empty(), "the goldens hold no cases");
        let mut id_mismatches = Vec::new();
        let mut worst = (1.0f64, String::new());
        let mut identical = 0usize;
        let mut against_fp32 = Vec::new();
        println!(
            "\n{:<38} {:<8} {:>5} {:>14} {:>10}{}",
            "case",
            "role",
            "ids",
            "cosine",
            "max|Δ|",
            if fp32.is_some() {
                format!(" {:>14}", "vs fp32")
            } else {
                String::new()
            }
        );
        for case in cases {
            let name = case["name"].as_str().expect("name");
            let role = case["role"].as_str().expect("role");
            let input = case["input"].as_str().expect("input");
            match role {
                "passage" => assert!(input.starts_with("[PASSAGE] "), "{name}: {role}"),
                "query" => assert!(input.starts_with("[QUERY] "), "{name}: {role}"),
                "raw" => {}
                other => panic!("{name}: unknown role {other:?}"),
            }
            let golden_ids = golden_ids(case);

            // Proves the exact bytes were read, invisible characters included.
            if let Some(digest) = case["input_utf8_sha256"].as_str() {
                use sha2::Digest;
                let actual: String = sha2::Sha256::digest(input.as_bytes())
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect();
                assert_eq!(
                    actual, digest,
                    "{name}: the input read is not the golden input"
                );
            }

            let produced = backend.tokenize(input).expect("tokenize");
            if produced != golden_ids {
                let first = produced
                    .iter()
                    .zip(&golden_ids)
                    .position(|(a, b)| a != b)
                    .unwrap_or(produced.len().min(golden_ids.len()));
                id_mismatches.push(format!(
                    "{name}: {} ids vs {} golden, first difference at {first}",
                    produced.len(),
                    golden_ids.len()
                ));
                continue;
            }

            let vector = backend.embed_batch_raw(&[input]).expect("embed").remove(0);
            let reference =
                golden_vector(case["vector_f32_le_base64"].as_str().expect("vector"), dim);
            let cos = cosine(&vector, &reference);
            let max_abs = vector
                .iter()
                .zip(&reference)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            identical += usize::from(vector == reference);
            // The same case in the fp32 file, provided it is the same input: the ids say so.
            let versus_fp32 = fp32.as_ref().map(|fp32| match fp32.get(name) {
                Some((ids, vector_fp32)) if *ids == golden_ids => {
                    let cos = cosine(&vector, vector_fp32);
                    against_fp32.push((cos, name.to_string()));
                    format!(" {cos:>14.10}")
                }
                _ => format!(" {:>14}", "not comparable"),
            });
            println!(
                "{name:<38} {role:<8} {:>5} {cos:>14.10} {max_abs:>10.3e}{}",
                produced.len(),
                versus_fp32.unwrap_or_default()
            );
            if cos < worst.0 {
                worst = (cos, name.to_string());
            }
        }

        if fp32.is_some() {
            match against_fp32.iter().min_by(|a, b| a.0.total_cmp(&b.0)) {
                Some((cos, name)) => {
                    let values: Vec<f64> = against_fp32.iter().map(|(cos, _)| *cos).collect();
                    println!(
                        "int8 on this CPU against the fp32 graph's goldens: worst cosine {cos:.10} \
                         ({name}), median {:.10}, over {} of {} cases — reported, not a gate",
                        median(&values),
                        values.len(),
                        cases.len()
                    );
                }
                None => println!(
                    "int8 on this CPU against the fp32 graph's goldens: no case comparable \
                     (no fp32 golden with the same name and ids)"
                ),
            }
        }

        assert!(
            id_mismatches.is_empty(),
            "TOKEN ID MISMATCHES — the primary correctness gate:\n  {}",
            id_mismatches.join("\n  ")
        );
        println!(
            "worst cosine {:.10} ({}), required >= {min_cosine}; {identical} of {} vectors \
             bit-identical to the reference",
            worst.0,
            worst.1,
            cases.len()
        );
        assert!(
            worst.0 >= min_cosine,
            "worst cosine {:.10} on {} is below {min_cosine} ({why})",
            worst.0,
            worst.1
        );
    }

    /// What `X64_QUANT_PRECISION` changes on the CPU this runs on: the golden ids through
    /// two sessions built alike but for that entry — the backend's, and the runtime's
    /// default — compared and timed. Reported, never asserted, because the answer belongs
    /// to the CPU: nothing on ARM, on an x86 CPU with VNNI or for an fp32 graph; every int8
    /// vector on an AVX2 or AVX-512 CPU without VNNI, whose default kernels clamp. The
    /// timings are what the exact products cost there; each is the best of three passes
    /// after a warm-up, in alternation, and means something only when the tests run one at
    /// a time (`--test-threads=1`, as CI runs them).
    #[test]
    #[ignore = "needs a Meivin graph (int8 or fp32); set OTZARIA_TEST_ONNX_MODEL and pass --ignored"]
    fn what_the_x86_int8_entry_changes_on_this_cpu_is_reported() {
        let Some(graph) = model_path() else { return };
        let goldens = goldens_for(&graph);
        let dim = goldens.header()["dim"].as_u64().expect("dim") as usize;
        let cases = goldens.data["cases"].as_array().expect("cases");
        let names: Vec<&str> = cases
            .iter()
            .map(|case| case["name"].as_str().expect("name"))
            .collect();
        let ids: Vec<Vec<u32>> = cases.iter().map(golden_ids).collect();
        let references: Vec<Vec<f32>> = cases
            .iter()
            .map(|case| golden_vector(case["vector_f32_le_base64"].as_str().expect("vector"), dim))
            .collect();
        let fp32 = (goldens.arithmetic == Arithmetic::DynamicInt8).then(|| fp32_reference(dim));

        let runtime = resolve_runtime_path(None, std::env::var_os(RUNTIME_ENV), &graph)
            .expect("the runtime library");
        ensure_runtime(&runtime).expect("the runtime must load");
        let threads = OnnxBackendConfig::default().intra_threads;
        let mut variants: Vec<(&str, Wiring, Worker)> =
            [(true, "= 1, the backend's"), (false, "= 0, the default")]
                .into_iter()
                .map(|(exact, label)| {
                    let session = build_session_as(&graph, threads, exact).expect("a session");
                    let wiring =
                        inspect_graph(&session, &graph).expect("the golden graph's wiring");
                    let worker = Worker::new(session, &wiring).expect("a worker");
                    (label, wiring, worker)
                })
                .collect();

        let run_all =
            |wiring: &Wiring, worker: &mut Worker| -> (Vec<Vec<f32>>, std::time::Duration) {
                let started = std::time::Instant::now();
                let vectors = ids
                    .iter()
                    .map(|ids| match worker.run(wiring, ids).expect("a run") {
                        RunOutput::Tensor { data, .. } => data,
                        RunOutput::Missing => panic!("the run returned no '{}'", wiring.output),
                    })
                    .collect();
                (vectors, started.elapsed())
            };
        let mut vectors = Vec::new();
        let mut best = vec![std::time::Duration::MAX; variants.len()];
        for pass in 0..4 {
            for (index, (_, wiring, worker)) in variants.iter_mut().enumerate() {
                let (pass_vectors, elapsed) = run_all(wiring, worker);
                if pass == 0 {
                    vectors.push(pass_vectors);
                } else {
                    best[index] = best[index].min(elapsed);
                }
            }
        }

        println!(
            "\n{X64_QUANT_PRECISION} on this CPU ({}), {} texts, {threads} intra-op thread(s), \
             graph {}:",
            std::env::consts::ARCH,
            ids.len(),
            goldens.header()["graph_file"]
        );
        for ((label, _, _), (vectors, best)) in variants.iter().zip(vectors.iter().zip(&best)) {
            println!("  {X64_QUANT_PRECISION} {label}");
            let against_goldens = vectors
                .iter()
                .zip(&references)
                .map(|(vector, reference)| Some(cosine(vector, reference)));
            println!(
                "    worst cosine against {}: {}",
                goldens.file,
                worst_case(against_goldens.zip(names.iter().copied()))
            );
            if let Some(fp32) = &fp32 {
                let against_fp32 = vectors.iter().enumerate().map(|(i, vector)| {
                    fp32.get(names[i])
                        .filter(|(fp32_ids, _)| *fp32_ids == ids[i])
                        .map(|(_, reference)| cosine(vector, reference))
                });
                println!(
                    "    worst cosine against the fp32 graph's goldens: {}",
                    worst_case(against_fp32.zip(names.iter().copied()))
                );
            }
            println!("    {} texts in {best:.1?} (best of 3 passes)", ids.len());
        }
        let same = vectors[0]
            .iter()
            .zip(&vectors[1])
            .filter(|(a, b)| a == b)
            .count();
        println!(
            "  the two agree bit for bit on {same} of {} vectors{}",
            ids.len(),
            if same == ids.len() {
                " — the entry changes nothing on this CPU for this graph"
            } else {
                ""
            }
        );
    }

    /// A batch is one run per text, so batched and single answers must be *equal*, bit
    /// for bit, and in input order — `EmbeddingRuntime` pairs vectors with chunks by
    /// position and could not see a transposition.
    #[test]
    #[ignore = "needs a Meivin graph (int8 or fp32); set OTZARIA_TEST_ONNX_MODEL and pass --ignored"]
    fn batched_vectors_equal_single_ones_in_input_order() {
        let Some(graph) = model_path() else { return };
        let data = goldens_for(&graph).data;
        let max_tokens = data["header"]["max_tokens"].as_u64().expect("max_tokens") as usize;
        let owned = inputs(&data);
        let texts: Vec<&str> = owned.iter().map(String::as_str).collect();

        let backend = open_golden(&graph, max_tokens, 1);
        let batched = backend.embed_batch_raw(&texts).expect("batched");
        assert_eq!(batched.len(), texts.len());
        for (index, text) in texts.iter().enumerate() {
            let single = backend.embed_batch_raw(&[text]).expect("single").remove(0);
            assert_eq!(batched[index], single, "case {index}: batched != single");
        }
    }

    /// Several threads inside `embed_batch_raw` at once, through `&self`, over two
    /// sessions — the shape `hybrid::coordinator` produces — agreeing with the serial
    /// answer bit for bit. Timings are printed, never asserted.
    #[test]
    #[ignore = "needs a Meivin graph (int8 or fp32); set OTZARIA_TEST_ONNX_MODEL and pass --ignored"]
    fn concurrent_callers_through_a_shared_reference_agree_with_serial_ones() {
        let Some(graph) = model_path() else { return };
        let data = goldens_for(&graph).data;
        let max_tokens = data["header"]["max_tokens"].as_u64().expect("max_tokens") as usize;
        let owned = inputs(&data);
        let texts: Vec<&str> = owned.iter().map(String::as_str).collect();

        let backend = open_golden(&graph, max_tokens, 2);
        let started = std::time::Instant::now();
        let serial = backend.embed_batch_raw(&texts).expect("serial");
        let serial_time = started.elapsed();

        let started = std::time::Instant::now();
        let concurrent = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..4)
                .map(|_| {
                    let backend = &backend;
                    let texts = &texts;
                    scope.spawn(move || backend.embed_batch_raw(texts).expect("concurrent"))
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("worker"))
                .collect::<Vec<_>>()
        });
        println!(
            "{} texts: serial {serial_time:?}; 4 threads x the same batch over 2 sessions {:?}",
            texts.len(),
            started.elapsed()
        );
        for (thread, result) in concurrent.iter().enumerate() {
            assert_eq!(
                result, &serial,
                "thread {thread} disagrees with the serial answer"
            );
        }
    }
}
