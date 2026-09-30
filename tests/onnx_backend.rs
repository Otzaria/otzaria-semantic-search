//! The ONNX backend as a downstream crate reaches it: through `select_backend`, whose
//! constructor pair is compiled only outside `cfg(test)` and so is unreachable from the
//! in-crate suite, and through `EmbeddingRuntime`.
//!
//! Also the guard for the one thing in the ONNX gating a compiler cannot check on the
//! machine it runs on: that the target condition is spelled the same in every place that
//! carries it. `the_target_condition_is_spelled_identically_everywhere` runs in every
//! build.
//!
//! The tests that run a graph need an ONNX Runtime shared library — nothing is linked —
//! named by `OTZARIA_ONNX_RUNTIME`. Without one they skip loudly, as the model-gated
//! tests do; the refusals that happen before a runtime is touched run regardless.

/// Where the target condition lives. All six must be the same expression: `mod.rs` and
/// the constructor pair name the module that exists only where `Cargo.toml` declares its
/// crates, and a mismatch is a build break on exactly the targets nobody compiles
/// locally — or a backend compiled where its crates are absent.
#[test]
fn the_target_condition_is_spelled_identically_everywhere() {
    const MANIFEST: &str = include_str!("../Cargo.toml");
    const MOD_RS: &str = include_str!("../src/semantic/mod.rs");
    const BACKEND_RS: &str = include_str!("../src/semantic/backend.rs");
    const THIS_FILE: &str = include_str!("onnx_backend.rs");

    /// Whitespace and trailing commas are formatting, not meaning; `cargo fmt` moves both.
    fn normalized(text: &str) -> String {
        let squeezed: String = text.chars().filter(|c| !c.is_whitespace()).collect();
        squeezed.replace(",)", ")")
    }

    fn manifest_condition(dependency: &str) -> String {
        let suffix = format!(")'.dependencies.{dependency}]");
        let line = MANIFEST
            .lines()
            .find(|line| line.starts_with("[target.'cfg(") && line.ends_with(&suffix))
            .unwrap_or_else(|| {
                panic!("Cargo.toml no longer declares `{dependency}` under a target cfg")
            });
        normalized(&line["[target.'cfg(".len()..line.len() - suffix.len()])
    }

    let condition = manifest_condition("ort");
    assert_eq!(
        manifest_condition("tokenizers"),
        condition,
        "`ort` and `tokenizers` are declared for different targets in Cargo.toml"
    );
    assert!(
        condition.starts_with("any("),
        "the condition is expected to be a list of platforms: {condition}"
    );

    let feature = r#"feature="onnx-backend""#;
    for (file, source, attribute) in [
        (
            "src/semantic/mod.rs",
            MOD_RS,
            format!("#[cfg(all({feature},{condition}))]pubmodonnx_backend;"),
        ),
        (
            "src/semantic/backend.rs (the real constructor)",
            BACKEND_RS,
            format!("#[cfg(all({feature},{condition},not(test)))]fnonnx_backend("),
        ),
        (
            "src/semantic/backend.rs (the `None` constructor)",
            BACKEND_RS,
            format!("#[cfg(not(all({feature},{condition},not(test))))]fnonnx_backend("),
        ),
        (
            "tests/onnx_backend.rs",
            THIS_FILE,
            format!("#[cfg(all({feature},{condition}))]modwith_the_backend"),
        ),
    ] {
        assert!(
            normalized(source).contains(&attribute),
            "{file} does not gate on the target condition Cargo.toml declares the ONNX crates \
             under. Expected, modulo whitespace:\n{attribute}"
        );
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
mod with_the_backend {
    use otzaria_semantic_search::errors::EmbeddingError;
    use otzaria_semantic_search::semantic::backend::{select_backend, Pooling};
    use otzaria_semantic_search::semantic::embedding::{EmbeddingConfig, EmbeddingRuntime};
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;

    const RUNTIME_ENV: &str = "OTZARIA_ONNX_RUNTIME";
    const ENV_THREADS: &str = "OTZARIA_ONNX_THREADS";
    const ENV_SESSIONS: &str = "OTZARIA_ONNX_SESSIONS";

    /// `select_backend` reads the tuning variables, and one test here writes them; the
    /// harness runs tests on parallel threads of one process.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn lock_env() -> std::sync::MutexGuard<'static, ()> {
        ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn fixture(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/onnx_fixture")
            .join(name)
    }

    fn runtime_configured() -> bool {
        match std::env::var_os(RUNTIME_ENV) {
            Some(path) if !path.is_empty() => true,
            _ => {
                println!(
                    "SKIPPED: {RUNTIME_ENV} is not set; this test runs a graph and needs an ONNX \
                     Runtime shared library (1.17 or newer)"
                );
                false
            }
        }
    }

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(name: &str) -> Self {
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "otzaria_onnx_integration_{name}_{}_{}",
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

    /// A model package as a deployment lays it out: a graph and `tokenizer.json`.
    fn package(dir: &TempDir, graph: &str) -> PathBuf {
        let model = dir.0.join("model.onnx");
        std::fs::copy(fixture(graph), &model).unwrap();
        std::fs::copy(fixture("tokenizer.json"), dir.0.join("tokenizer.json")).unwrap();
        model
    }

    fn config_for(model_path: PathBuf) -> EmbeddingConfig {
        EmbeddingConfig {
            model_path,
            embedding_dim: 4,
            max_tokens: 32,
            batch_size: 4,
            pooling: Pooling::InGraph,
        }
    }

    /// The real table, not the stand-in: an `.onnx` path reaches `OnnxBackend` through
    /// the constructor pair, the tokenizer is found beside the graph, and the tuning
    /// comes from the environment.
    #[test]
    fn an_onnx_model_is_served_by_the_onnx_runtime_backend() {
        if !runtime_configured() {
            return;
        }
        let _guard = lock_env();
        let dir = TempDir::new("select");
        let backend = select_backend(&config_for(package(&dir, "dynamic.onnx")))
            .expect("the fixture package loads through the selection table");

        assert_eq!(backend.id(), "onnxruntime-sentence-v1");
        assert!(backend.is_semantic());
        assert_eq!(backend.dim(), 4);
        assert_eq!(backend.max_tokens(), 32);
        assert_eq!(backend.pooling(), Pooling::InGraph);

        let vectors = backend
            .embed_batch_raw(&["[QUERY] the fox", "[PASSAGE] בראשית ברא"])
            .unwrap();
        assert_eq!(vectors.len(), 2);
        assert_ne!(vectors[0], vectors[1]);
    }

    /// The selection-table path reads `OTZARIA_ONNX_THREADS`/`_SESSIONS`, and a bad value
    /// is an error naming it — not the default, and not a load that half happened.
    #[test]
    fn a_bad_tuning_variable_is_refused_through_the_selection_table() {
        let _guard = lock_env();
        let dir = TempDir::new("tuning");
        let config = config_for(package(&dir, "dynamic.onnx"));

        for key in [ENV_THREADS, ENV_SESSIONS] {
            std::env::set_var(key, "several");
            let refused = select_backend(&config).map(|_| ());
            std::env::remove_var(key);
            match refused {
                Err(EmbeddingError::LoadFailed { reason }) => {
                    assert!(reason.contains(key), "{reason}");
                }
                other => panic!("{key}=several must be refused, got {other:?}"),
            }
        }
    }

    /// With the stand-in compiled in too, an ONNX package the real backend rejects must
    /// produce the real backend's error — the `Some(Err)` rule that stops the walk —
    /// never hash vectors from the stand-in's ONNX row.
    #[test]
    fn a_package_the_real_backend_rejects_never_falls_through_to_the_stand_in() {
        let _guard = lock_env();
        let dir = TempDir::new("rejected");
        let model = dir.0.join("model.onnx");
        std::fs::copy(fixture("dynamic.onnx"), &model).unwrap();
        // No tokenizer.json beside it: refused before any runtime is needed.
        match select_backend(&config_for(model)) {
            Err(EmbeddingError::TokenizerNotFound { path }) => {
                assert!(path.ends_with("tokenizer.json"), "{path}");
            }
            Err(other) => panic!("expected TokenizerNotFound, got {other}"),
            Ok(backend) => panic!(
                "a package without a tokenizer must not load; got backend '{}'",
                backend.id()
            ),
        }
    }

    /// End to end through `EmbeddingRuntime::load`, which validates and checksums the
    /// model file before selecting a backend.
    ///
    /// **Ignored until the ONNX package validator lands** (design D3/D4, in a separate
    /// change): until then `load` validates every model file as GGUF and refuses an
    /// ONNX graph before the backend is reached. Remove the `#[ignore]` with that merge.
    #[test]
    #[ignore = "needs the ONNX package validator (design D3/D4); load() still validates every model as GGUF"]
    fn the_runtime_loads_an_onnx_package_and_normalizes_its_vectors() {
        if !runtime_configured() {
            return;
        }
        let _guard = lock_env();
        let dir = TempDir::new("runtime");
        let mut runtime = EmbeddingRuntime::new(config_for(package(&dir, "dynamic.onnx")));
        runtime
            .load()
            .expect("the fixture package loads end to end");

        assert_eq!(runtime.backend_id(), Some("onnxruntime-sentence-v1"));
        assert!(runtime.backend_is_semantic());
        assert!(runtime.model_checksum().is_some());
        let vectors = runtime.embed_batch(&["the fox", "תורה"]).unwrap();
        for vector in vectors {
            let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "the runtime normalizes: {norm}");
        }
    }
}
