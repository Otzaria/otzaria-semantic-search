//! Real ONNX inference through ONNX Runtime: the `onnxruntime-sentence-v1` backend.
//!
//! Behind the non-default `onnx-backend` feature, on the desktop targets its crates are
//! declared for (`Cargo.toml` says which and why); anywhere else an ONNX model gets
//! [`EmbeddingError::BackendUnavailable`]. Design and measurements are in
//! `docs/ONNX_BACKEND.md`.
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
//! The runtime is a shared library found at load time — `OTZARIA_ONNX_RUNTIME`, else
//! the platform's file name (`libonnxruntime.dylib`, `libonnxruntime.so`,
//! `onnxruntime.dll`) in the model package beside the graph — and loaded once per
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
//! `ort` 2.0.0-rc.13's `Session::run` takes `&mut self` (upstream considers concurrent
//! `Run` on one session unsound), while [`EmbeddingBackend`] is `&self` and `Sync`. So
//! the backend owns a bounded pool of sessions (`Pool`), one lease per *text*: an
//! indexing batch holds a session for one inference at a time, and a search query
//! queued behind it waits for one text rather than for the batch. The queue is FIFO, so
//! the batch cannot take the session straight back. The tokenizer is `Sync` with a
//! `&self` `encode` and is shared without a lock.
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
use crate::semantic::backend::{EmbeddingBackend, Pooling};
use crate::semantic::embedding::EmbeddingConfig;
use crate::semantic::model_package::onnx_package_root;

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
/// Four is the llama backend's measured cap, taken over until this backend has its own
/// measurement.
const DEFAULT_THREADS_CAP: usize = 4;

/// Default number of sessions, i.e. concurrent `embed_batch_raw` calls: one, because
/// the smallest target is a phone and every session holds its own activations.
const DEFAULT_SESSIONS: usize = 1;

/// Environment variable naming the ONNX Runtime shared library to load. Read by
/// [`OnnxBackend::open`], not by [`OnnxBackendConfig`]: it chooses the code that runs,
/// not a size.
const RUNTIME_ENV: &str = "OTZARIA_ONNX_RUNTIME";

/// The runtime's file name on this platform, as Microsoft's releases spell it — what is
/// looked for beside the graph when [`RUNTIME_ENV`] is unset.
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
    /// one. Each costs roughly the graph's size in memory.
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

/// The ONNX Runtime library this process has loaded, once one has.
struct LoadedRuntime {
    /// Canonical, so two spellings of one file compare equal.
    path: PathBuf,
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
    Unusable { path: PathBuf, reason: String },
}

/// `None` until a library has been handed to `ort`, so an attempt the backend's own checks
/// refused (a wrong path, a file that is not a runtime, one too old) touches nothing of
/// `ort`'s and can be retried with a correct one.
static RUNTIME: Mutex<Option<RuntimeSlot>> = Mutex::new(None);

/// Where the runtime library must come from for `graph`, in order: [`RUNTIME_ENV`]
/// (`env_value`), else [`RUNTIME_FILE_NAME`] in the package root. Taken as an argument
/// rather than read here so the rule is testable without mutating the process
/// environment. Returns the canonical path.
///
/// Set but empty is refused rather than read as unset, as for the tuning variables: it
/// names no file, and falling back would be a guess about what was meant.
fn resolve_runtime_path(
    env_value: Option<std::ffi::OsString>,
    graph: &Path,
) -> Result<PathBuf, EmbeddingError> {
    let unavailable = |reason: String| EmbeddingError::BackendUnavailable { reason };
    if let Some(raw) = env_value {
        let requested = PathBuf::from(&raw);
        if raw.is_empty() {
            return Err(unavailable(format!(
                "{RUNTIME_ENV} is set but empty; it must name the ONNX Runtime shared library \
                 ({RUNTIME_FILE_NAME}), or be unset to use the one beside the model"
            )));
        }
        return requested.canonicalize().map_err(|e| {
            unavailable(format!(
                "{RUNTIME_ENV} is set to {}, which cannot be opened: {e}",
                requested.display()
            ))
        });
    }

    let beside = onnx_package_root(graph).join(RUNTIME_FILE_NAME);
    beside.canonicalize().map_err(|_| {
        unavailable(format!(
            "no ONNX Runtime shared library to load for {}: set {RUNTIME_ENV} to the path of \
             {RUNTIME_FILE_NAME} (ONNX Runtime 1.17 or newer; the reference is Microsoft's \
             official 1.28.0 release), or place it beside the model at {}",
            graph.display(),
            beside.display()
        ))
    })
}

/// Load the runtime at `path` unless this process already has — the same file again is
/// a no-op, a different one is refused. Returns the runtime's version and build, for the
/// load log.
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
fn ensure_runtime(path: &Path) -> Result<String, EmbeddingError> {
    let unavailable = |reason: String| EmbeddingError::BackendUnavailable { reason };

    // Held across the load, so two first loads cannot race. Recovered from poisoning:
    // the slot only ever goes from empty to one complete value.
    let mut slot = RUNTIME.lock().unwrap_or_else(PoisonError::into_inner);
    match slot.as_ref() {
        Some(RuntimeSlot::Loaded(runtime)) if runtime.path == path => {
            return Ok(runtime.description.clone());
        }
        Some(RuntimeSlot::Loaded(runtime)) => {
            return Err(unavailable(format!(
                "this process already runs ONNX Runtime from {}, and a process can hold only \
                 one; {} was requested. Point {RUNTIME_ENV} at the library already loaded, or \
                 restart the process to switch",
                runtime.path.display(),
                path.display()
            )));
        }
        Some(RuntimeSlot::Unusable {
            path: refused,
            reason,
        }) => {
            return Err(unavailable(format!(
                "ONNX Runtime from {} was refused earlier in this process ({reason}), and the \
                 runtime binding cannot load another after that; restart the process with a \
                 working library",
                refused.display()
            )));
        }
        None => {}
    }

    // Nothing of `ort`'s is touched if this fails.
    let (library, version) = open_runtime_library(path)?;

    // Absolute on purpose: for a relative path `ort` resolves against the executable's
    // directory through two `expect`s.
    let environment = match ort::init_from(path) {
        Ok(environment) => environment,
        Err(e) => {
            let reason = e.to_string();
            *slot = Some(RuntimeSlot::Unusable {
                path: path.to_path_buf(),
                reason: reason.clone(),
            });
            return Err(unavailable(format!(
                "ONNX Runtime {version} at {} passed this backend's checks but the runtime \
                 binding refused it: {reason}",
                path.display()
            )));
        }
    };
    // Recorded the moment `ort` holds the library, before anything else can fail: it
    // cannot be unloaded, and `ort` would silently ignore a later, different path.
    *slot = Some(RuntimeSlot::Loaded(LoadedRuntime {
        path: path.to_path_buf(),
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
            "ONNX Runtime from {} loaded, but its environment could not be created: {e}",
            path.display()
        ))
    })?;
    environment.set_log_level(LogLevel::Warning);

    let description = format!("{version} ({})", describe_build(ort::info()));
    if let Some(RuntimeSlot::Loaded(runtime)) = slot.as_mut() {
        runtime.description.clone_from(&description);
    }
    Ok(description)
}

/// Open `path` as an ONNX Runtime library and check it the way `ort::init_from` will —
/// so that `init_from` is never handed a library it refuses (see `ensure_runtime`).
/// Returns the handle, to be kept, and the runtime's version string
/// (`OrtGetApiBase()->GetVersionString()`), which `ort` reads but does not expose.
fn open_runtime_library(path: &Path) -> Result<(libloading::Library, String), EmbeddingError> {
    type GetApiBase = unsafe extern "system" fn() -> *const ort::sys::OrtApiBase;
    let unavailable = |reason: String| EmbeddingError::BackendUnavailable { reason };

    // SAFETY: loading a library runs its initializers, and there is no way to vet a file
    // before that: this is the library the process is about to run semantic search on,
    // chosen by the deployment (`OTZARIA_ONNX_RUNTIME` or the package's own file), and
    // loading it is what `ort::init_from` does next in any case.
    let library = unsafe { libloading::Library::new(path) }.map_err(|e| {
        unavailable(format!(
            "{} is not a loadable shared library for this platform: {e}",
            path.display()
        ))
    })?;
    // SAFETY: the type is ONNX Runtime's C declaration of its entry point,
    // `const OrtApiBase* ORT_API_CALL OrtGetApiBase(void)`, exactly as `ort-sys` binds it.
    let get_api_base = unsafe { library.get::<GetApiBase>(b"OrtGetApiBase") }.map_err(|_| {
        unavailable(format!(
            "{} loads but does not export OrtGetApiBase, so it is not ONNX Runtime",
            path.display()
        ))
    })?;
    // SAFETY: a call with no arguments into the loaded runtime, which returns null or a
    // pointer to a static table living as long as the library.
    let base = unsafe { get_api_base() };
    if base.is_null() {
        return Err(unavailable(format!(
            "{}: OrtGetApiBase returned nothing",
            path.display()
        )));
    }
    // SAFETY: `base` is non-null and points at that table. `GetVersionString` returns a
    // NUL-terminated string the runtime owns ("do not deallocate"), valid while the
    // library is loaded; it is copied out before `library` can be dropped.
    let raw_version = unsafe { ((*base).GetVersionString)() };
    if raw_version.is_null() {
        return Err(unavailable(format!(
            "{}: the runtime reports no version",
            path.display()
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
            "{} is ONNX Runtime {version}; this backend needs 1.{} or newer (the reference is \
             Microsoft's official 1.28.0 release)",
            path.display(),
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
    Session::builder()?
        .with_optimization_level(GRAPH_OPTIMIZATION)?
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
/// query queued behind an indexing batch. Never fails and never blocks forever: every
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
    runtime: PathBuf,
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
    /// and graph optimization `All` (`GRAPH_OPTIMIZATION`). Deliberately **not** the
    /// runtime's version or the thread count, as `LlamaCppBackend::ID` leaves out the
    /// llama.cpp build: `docs/ONNX_BACKEND.md` measures what they move.
    pub const ID: &'static str = "onnxruntime-sentence-v1";

    /// Load the graph at `graph` with the tokenizer at `tokenizer`, truncating every
    /// input to `max_tokens` tokens in total, special tokens included.
    ///
    /// `pooling` is what the configuration asks for; the backend serves only
    /// [`Pooling::InGraph`], and proves at load that the graph's first output is a
    /// finished `[batch, dim]` sentence vector.
    ///
    /// Everything is checked here rather than on the first search, cheapest first: the
    /// pooling and the tuning; that both files exist; that the tokenizer parses and
    /// leaves `max_tokens` room for content; the runtime library (see the module docs);
    /// the graph's inputs and first output; and finally a probe of exactly `max_tokens`
    /// tokens through **every** session, which must come back finite, `dim` long and
    /// identical across sessions — so a cap the graph cannot run, or a session that will
    /// not allocate, fails here and not in the middle of an index.
    ///
    /// # Errors
    ///
    /// * [`EmbeddingError::PoolingMismatch`] for a pooling other than `in-graph`;
    /// * [`EmbeddingError::ModelNotFound`] / [`EmbeddingError::TokenizerNotFound`];
    /// * [`EmbeddingError::LoadFailed`] for a zero thread or session count, a cap that
    ///   leaves no room for content, or a session that will not allocate;
    /// * [`EmbeddingError::BackendUnavailable`] when no ONNX Runtime can be loaded — none
    ///   found, too old, not a runtime, or a different one already loaded;
    /// * [`EmbeddingError::InvalidModelFile`] for a tokenizer or graph this backend
    ///   cannot serve, or a graph that fails the probe.
    pub fn open(
        graph: &Path,
        tokenizer: &Path,
        max_tokens: usize,
        pooling: Pooling,
        tuning: &OnnxBackendConfig,
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
        let runtime_path = resolve_runtime_path(std::env::var_os(RUNTIME_ENV), graph)?;
        let runtime = ensure_runtime(&runtime_path)?;

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
            runtime: runtime_path,
        };

        // ── the probe: only running the graph proves it serves this cap ──
        backend.dim = backend.probe_every_session()?;

        log::info!(
            "Embedding backend '{}' ready: {} — dim {} ({}), max_tokens {max_tokens} \
             ({special_tokens} special), batch dimension {} (one text per run), inputs \
             {INPUT_IDS} + {ATTENTION_MASK}{}, output '{}', {} session(s) x {} intra-op \
             thread(s), graph optimization {GRAPH_OPTIMIZATION:?}; ONNX Runtime {runtime} from \
             {}; loaded in {} ms",
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
            backend.runtime.display(),
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

    #[test]
    fn the_runtime_comes_from_the_variable_first_then_from_beside_the_graph() {
        let dir = TempDir::new("runtime_path");
        let graph = dir.0.join("model.onnx");
        std::fs::write(&graph, b"graph").unwrap();

        // Neither: the error names both ways to provide one.
        match resolve_runtime_path(None, &graph) {
            Err(EmbeddingError::BackendUnavailable { reason }) => {
                assert!(reason.contains(RUNTIME_ENV), "{reason}");
                assert!(reason.contains(RUNTIME_FILE_NAME), "{reason}");
                assert!(reason.contains(&dir.0.display().to_string()), "{reason}");
            }
            other => panic!("expected BackendUnavailable, got {other:?}"),
        }

        // Beside the graph, when the variable is unset.
        let beside = dir.0.join(RUNTIME_FILE_NAME);
        std::fs::write(&beside, b"not really a runtime").unwrap();
        assert_eq!(
            resolve_runtime_path(None, &graph).unwrap(),
            beside.canonicalize().unwrap()
        );

        // The variable wins over the file beside the graph.
        let elsewhere = dir.0.join("elsewhere").join(RUNTIME_FILE_NAME);
        std::fs::create_dir_all(elsewhere.parent().unwrap()).unwrap();
        std::fs::write(&elsewhere, b"another").unwrap();
        assert_eq!(
            resolve_runtime_path(Some(elsewhere.clone().into_os_string()), &graph).unwrap(),
            elsewhere.canonicalize().unwrap()
        );

        // A variable naming nothing is refused, not skipped in favour of the file beside.
        for wrong in ["", "/definitely/not/here/libonnxruntime"] {
            match resolve_runtime_path(Some(wrong.into()), &graph) {
                Err(EmbeddingError::BackendUnavailable { reason }) => {
                    assert!(reason.contains(RUNTIME_ENV), "{reason}");
                }
                other => panic!("{RUNTIME_ENV}={wrong:?} must be refused, got {other:?}"),
            }
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
    /// not one is refused as unavailable and named — never a panic, which is what `ort`'s
    /// own search would have been.
    #[test]
    fn a_file_that_is_not_a_runtime_library_is_refused_and_named() {
        let not_a_library = fixture("expected.json").canonicalize().unwrap();
        match ensure_runtime(&not_a_library) {
            Err(EmbeddingError::BackendUnavailable { reason }) => {
                assert!(
                    reason.contains(&not_a_library.display().to_string()),
                    "{reason}"
                );
            }
            other => panic!("expected BackendUnavailable, got {other:?}"),
        }
    }

    /// One runtime per process: once one is loaded, asking for another file is an
    /// error that names both, and asking for the same one again is not.
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
            Some(RuntimeSlot::Loaded(runtime)) => runtime.path.clone(),
            _ => panic!("a runtime is loaded"),
        };
        assert!(
            ensure_runtime(&loaded).is_ok(),
            "the same library is a no-op"
        );

        let other = fixture("tokenizer.json").canonicalize().unwrap();
        match ensure_runtime(&other) {
            Err(EmbeddingError::BackendUnavailable { reason }) => {
                assert!(reason.contains(&loaded.display().to_string()), "{reason}");
                assert!(reason.contains(&other.display().to_string()), "{reason}");
            }
            other => panic!("a second runtime must be refused, got {other:?}"),
        }
    }
}

/// Verification against the real model's golden vectors. **This is the parity gate** for
/// the production graph, not a placeholder for one.
///
/// The goldens (`tests/data/onnx_golden_vectors.json`, written by
/// `tools/generate_onnx_golden_vectors.py`) are the Python reference's answers: the ids
/// from the `tokenizers` package and the vector from the `onnxruntime` package, for
/// inputs spelled exactly as the tokenizer receives them, role prefix included. They need
/// the gated 168 MB fp32 graph, so each test is `#[ignore]`d *and* skips loudly when
/// `OTZARIA_TEST_ONNX_MODEL` is unset — which keeps the ordinary matrix green on
/// machines with no model. Run them with:
///
/// ```sh
/// OTZARIA_ONNX_RUNTIME=/path/to/libonnxruntime.dylib \
/// OTZARIA_TEST_ONNX_MODEL=/path/to/seforim-embed-round2-fp32.onnx \
///   cargo test --lib --features onnx-backend onnx_backend::golden -- --ignored --nocapture
/// ```
///
/// The tokenizer is the package's, `tokenizer.json` beside the graph.
#[cfg(test)]
mod golden {
    use super::*;
    use crate::semantic::model_package::onnx_tokenizer_path;

    /// Names the fp32 graph, `seforim-embed-round2-fp32.onnx`.
    const MODEL_ENV: &str = "OTZARIA_TEST_ONNX_MODEL";

    /// The agreement required between this backend and the Python reference, per case,
    /// once the token ids have been proven **exactly** equal.
    ///
    /// Not 1.0, because the two sides need not run the same ONNX Runtime build or even the
    /// same instruction set: kernel choice alone (KleidiAI on or off, fused or unfused
    /// operators) measured maxabs 2.9e-6 on identical inputs, a cosine deficit around
    /// 1e-11. 0.99999 leaves that noise six orders of magnitude of room and still fails
    /// every wiring error the ids cannot see — an `attention_mask` or `token_type_ids`
    /// fed wrong, the wrong output taken, a half-precision execution provider, or the
    /// int8 graph standing in for the fp32 one (0.9986 on the author's own parity check).
    /// Truncation and prefix errors never reach this check: they change the ids.
    const MIN_COSINE: f64 = 0.999_99;

    /// The graph, or `None` with a loud explanation. Skipping rather than failing because
    /// CI has no model, and a test that fails there teaches everyone to ignore it.
    fn model_path() -> Option<PathBuf> {
        let path = match std::env::var(MODEL_ENV) {
            Ok(path) if !path.trim().is_empty() => PathBuf::from(path.trim()),
            _ => {
                println!(
                    "SKIPPED: {MODEL_ENV} is not set. This test needs the 168 MB \
                     seforim-embed-round2-fp32.onnx (with tokenizer.json beside it), which is \
                     gated and never committed."
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

    /// Missing goldens with a model present fail rather than skip: asking for this gate
    /// and getting a green tick without one would be the worst outcome.
    fn goldens() -> serde_json::Value {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/onnx_golden_vectors.json");
        let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!(
                "cannot read {}: {e}. Generate it with tools/generate_onnx_golden_vectors.py",
                path.display()
            )
        });
        serde_json::from_str(&raw).expect("onnx_golden_vectors.json is not valid JSON")
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

    /// Confirm the files behind `OTZARIA_TEST_ONNX_MODEL` are the ones the goldens were
    /// produced from, since the alternative is a wall of vector failures for a model that
    /// was simply the wrong one — the int8 graph beside the fp32 one, for instance.
    fn assert_is_the_golden_package(graph: &Path, header: &serde_json::Value) {
        let graph_sha = sha256_of(graph);
        assert_eq!(
            graph_sha,
            header["graph_sha256"].as_str().expect("graph_sha256"),
            "{} is not the graph the goldens were produced from ({})",
            graph.display(),
            header["graph_file"]
        );
        let tokenizer = onnx_tokenizer_path(graph);
        assert_eq!(
            sha256_of(&tokenizer),
            header["tokenizer_sha256"]
                .as_str()
                .expect("tokenizer_sha256"),
            "{} is not the tokenizer the goldens were produced with",
            tokenizer.display()
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

    fn golden_vector(encoded: &str, dim: usize) -> Vec<f32> {
        let bytes = base64_decode(encoded);
        assert_eq!(bytes.len(), dim * 4, "a golden vector is {dim} f32 values");
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
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
    #[test]
    #[ignore = "needs the 168 MB fp32 graph; set OTZARIA_TEST_ONNX_MODEL and pass --ignored"]
    fn token_ids_match_the_reference_exactly_and_vectors_agree() {
        let Some(graph) = model_path() else { return };
        let data = goldens();
        let header = &data["header"];
        assert_is_the_golden_package(&graph, header);

        let max_tokens = header["max_tokens"].as_u64().expect("max_tokens") as usize;
        let dim = header["dim"].as_u64().expect("dim") as usize;
        let backend = open_golden(&graph, max_tokens, 1);
        assert_eq!(backend.dim() as usize, dim);
        assert_eq!(backend.max_tokens(), max_tokens);
        println!("\n{backend:?}");
        println!(
            "reference: onnxruntime {}, tokenizers {}",
            header["onnxruntime_version"], header["tokenizers_version"]
        );

        let cases = data["cases"].as_array().expect("cases");
        assert!(!cases.is_empty(), "the goldens hold no cases");
        let mut id_mismatches = Vec::new();
        let mut worst = (1.0f64, String::new());
        println!(
            "\n{:<36} {:<8} {:>5} {:>14}",
            "case", "role", "ids", "cosine"
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
            let golden_ids: Vec<u32> = case["token_ids"]
                .as_array()
                .expect("token_ids")
                .iter()
                .map(|id| id.as_u64().expect("an id") as u32)
                .collect();

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
            println!("{name:<36} {role:<8} {:>5} {cos:>14.10}", produced.len());
            if cos < worst.0 {
                worst = (cos, name.to_string());
            }
        }

        assert!(
            id_mismatches.is_empty(),
            "TOKEN ID MISMATCHES — the primary correctness gate:\n  {}",
            id_mismatches.join("\n  ")
        );
        println!(
            "worst cosine {:.10} ({}), required >= {MIN_COSINE}",
            worst.0, worst.1
        );
        assert!(
            worst.0 >= MIN_COSINE,
            "worst cosine {:.10} on {} is below {MIN_COSINE}",
            worst.0,
            worst.1
        );
    }

    /// A batch is one run per text, so batched and single answers must be *equal*, bit
    /// for bit, and in input order — `EmbeddingRuntime` pairs vectors with chunks by
    /// position and could not see a transposition.
    #[test]
    #[ignore = "needs the 168 MB fp32 graph; set OTZARIA_TEST_ONNX_MODEL and pass --ignored"]
    fn batched_vectors_equal_single_ones_in_input_order() {
        let Some(graph) = model_path() else { return };
        let data = goldens();
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
    #[ignore = "needs the 168 MB fp32 graph; set OTZARIA_TEST_ONNX_MODEL and pass --ignored"]
    fn concurrent_callers_through_a_shared_reference_agree_with_serial_ones() {
        let Some(graph) = model_path() else { return };
        let data = goldens();
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
