//! The ONNX backend as a downstream crate reaches it: through `select_backend`, whose
//! constructor pair is compiled only outside `cfg(test)` and so is unreachable from the
//! in-crate suite, through `EmbeddingRuntime`, and through the two public paths a host
//! opens a session with — `SemanticEngine` and `OfficialSemanticIndex` — carrying the
//! runtime library the host passes.
//!
//! Also the guard for the one thing in the ONNX gating a compiler cannot check on the
//! machine it runs on: that the target condition is spelled the same in every place that
//! carries it. `the_target_condition_is_spelled_identically_everywhere` runs in every
//! build.
//!
//! The tests that run a graph need an ONNX Runtime shared library — nothing is linked —
//! named by `OTZARIA_ONNX_RUNTIME`. Without one they skip loudly, as the model-gated
//! tests do; the refusals that happen before a runtime is touched run regardless.

/// Where the target condition lives. All seven must be the same expression: `mod.rs` and
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
    for dependency in ["tokenizers", "libloading"] {
        assert_eq!(
            manifest_condition(dependency),
            condition,
            "`ort` and `{dependency}` are declared for different targets in Cargo.toml"
        );
    }
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
    use otzaria_semantic_search::errors::{ArtifactError, EmbeddingError, SemanticSearchError};
    use otzaria_semantic_search::semantic::backend::{select_backend, Pooling};
    use otzaria_semantic_search::semantic::embedding::{
        EmbeddingConfig, EmbeddingDeployment, EmbeddingRuntime,
    };
    use otzaria_semantic_search::semantic::engine::{SemanticConfig, SemanticEngine};
    use otzaria_semantic_search::semantic::official_index::{
        LocalModel, OfficialIndexConfig, OfficialSemanticIndex,
    };
    use otzaria_semantic_search::semantic::store::VectorStoreConfig;
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

    /// The engine — the on-device development path — for the fixture package.
    fn engine_config(
        dir: &TempDir,
        model_path: PathBuf,
        deployment: EmbeddingDeployment,
    ) -> SemanticConfig {
        let root = dir.0.join("semantic");
        SemanticConfig {
            root_dir: root.clone(),
            model_path,
            deployment,
            embedding_model_id: "onnx-fixture".to_string(),
            embedding_dim: 4,
            pooling: "in-graph".to_string(),
            embedding_max_tokens: 32,
            model_quantization: "fp32".to_string(),
            store: VectorStoreConfig {
                db_path: root.join("vectors"),
                embedding_dim: 4,
                collection_name: "chunks".to_string(),
            },
            ..Default::default()
        }
    }

    /// The application's path for the fixture package, with no artifact installed. The
    /// model is loaded before the artifact is read, so how opening fails says how far the
    /// model got.
    fn official_config(
        dir: &TempDir,
        model_path: PathBuf,
        deployment: EmbeddingDeployment,
    ) -> OfficialIndexConfig {
        OfficialIndexConfig {
            artifact_path: dir.0.join("no-artifact-installed"),
            text:
                otzaria_semantic_search::semantic::versioning::TextIdentity::with_line_text_version(
                    1,
                ),
            model: LocalModel {
                model_path,
                family_id: "onnx-fixture@0".to_string(),
                model_quantization: "fp32".to_string(),
                embedding_dim: 4,
                pooling: "in-graph".to_string(),
                max_tokens: 32,
                embedding_text_version: 2,
                normalization_version: 1,
                chunking_identity: 0,
            },
            deployment,
            published_digest: None,
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

    /// Set by the parent test for its child: the runtime to load after the refused one.
    const GOOD_RUNTIME_ENV: &str = "OTZARIA_TEST_GOOD_ONNX_RUNTIME";

    /// A refused runtime must leave the process able to load a correct one afterwards.
    ///
    /// `ort` 2.0.0-rc.13 alone cannot: after one failed `init_from` its library slot reads
    /// as loaded and empty, the next `init_from` "succeeds" without loading anything, and
    /// the first call after that panics (`dlsym(0x0, OrtGetApiBase)`). The backend checks
    /// a library itself before `ort` ever sees it. Only a process that has loaded nothing
    /// can show the difference, so the check runs in a child of this test binary, alone.
    #[test]
    fn a_refused_runtime_leaves_the_process_able_to_load_a_correct_one() {
        if !runtime_configured() {
            return;
        }
        let good = std::env::var_os(RUNTIME_ENV).expect("checked above");
        // The child inherits this process's environment as it is at the spawn, and
        // `a_bad_tuning_variable_is_refused_through_the_selection_table` writes the tuning
        // variables on another thread: a child spawned in that window refused
        // `OTZARIA_ONNX_THREADS=several` before it reached the runtime. So the spawn holds
        // the lock the writer holds, and the child gets no tuning variable from here at all.
        let _guard = lock_env();
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "with_the_backend::a_correct_runtime_loads_after_a_refused_one_in_a_fresh_process",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .env_remove(ENV_THREADS)
            .env_remove(ENV_SESSIONS)
            .env(RUNTIME_ENV, fixture("expected.json"))
            .env(GOOD_RUNTIME_ENV, good)
            .output()
            .expect("the test binary can run itself");
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success() && stdout.contains("1 passed"),
            "the child process failed:\n{stdout}\n{stderr}"
        );
    }

    /// The child half of `a_refused_runtime_leaves_the_process_able_to_load_a_correct_one`,
    /// meaningless anywhere but in a process of its own.
    #[test]
    #[ignore = "run by a_refused_runtime_leaves_the_process_able_to_load_a_correct_one"]
    fn a_correct_runtime_loads_after_a_refused_one_in_a_fresh_process() {
        let Some(good) = std::env::var_os(GOOD_RUNTIME_ENV) else {
            println!("SKIPPED: only meaningful as the child of the test that spawns it");
            return;
        };
        let dir = TempDir::new("retry");
        let model = package(&dir, "dynamic.onnx");

        // First the environment names a file that is not a runtime.
        match select_backend(&config_for(model.clone())) {
            Err(EmbeddingError::OnnxRuntimeUnavailable { reason }) => {
                assert!(reason.contains("expected.json"), "{reason}");
            }
            Err(other) => panic!("expected OnnxRuntimeUnavailable, got {other}"),
            Ok(backend) => panic!("loaded '{}' from a JSON file", backend.id()),
        }

        // Then a real one, in the same process. A test alone in its process may write
        // the environment.
        std::env::set_var(RUNTIME_ENV, good);
        let backend = select_backend(&config_for(model))
            .expect("a correct runtime loads after a refused one");
        let vectors = backend.embed_batch_raw(&["the fox"]).unwrap();
        assert_eq!(vectors[0].len(), 4);
    }

    /// Set by the parent test for its child: the runtime the child passes as the
    /// application's, with `OTZARIA_ONNX_RUNTIME` removed from its environment.
    const APPLICATION_RUNTIME_ENV: &str = "OTZARIA_TEST_APPLICATION_ONNX_RUNTIME";

    /// A host that ships ONNX Runtime passes its path and needs nothing else — no variable,
    /// no file beside the graph — through `EmbeddingRuntime` and through both public paths
    /// a host opens a session with.
    ///
    /// A process holds one runtime, and this one may already run the variable's, so only a
    /// fresh process can show where its runtime came from: the check runs in a child of
    /// this test binary, alone, with the variable removed.
    #[test]
    fn a_runtime_the_application_passes_needs_neither_the_variable_nor_a_file_beside_the_graph() {
        if !runtime_configured() {
            return;
        }
        let application_runtime = std::env::var_os(RUNTIME_ENV).expect("checked above");
        // Held across the spawn for the reason `a_refused_runtime_leaves_the_process_able_to_
        // load_a_correct_one` gives: the child inherits the environment of that moment.
        let _guard = lock_env();
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "with_the_backend::the_applications_runtime_loads_in_a_fresh_process_without_the_variable",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .env_remove(ENV_THREADS)
            .env_remove(ENV_SESSIONS)
            .env_remove(RUNTIME_ENV)
            .env(APPLICATION_RUNTIME_ENV, application_runtime)
            .output()
            .expect("the test binary can run itself");
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success() && stdout.contains("1 passed"),
            "the child process failed:\n{stdout}\n{stderr}"
        );
    }

    /// The child half of
    /// `a_runtime_the_application_passes_needs_neither_the_variable_nor_a_file_beside_the_graph`,
    /// meaningless anywhere but in a process of its own.
    #[test]
    #[ignore = "run by a_runtime_the_application_passes_needs_neither_the_variable_nor_a_file_beside_the_graph"]
    fn the_applications_runtime_loads_in_a_fresh_process_without_the_variable() {
        let Some(application_runtime) = std::env::var_os(APPLICATION_RUNTIME_ENV) else {
            println!("SKIPPED: only meaningful as the child of the test that spawns it");
            return;
        };
        assert!(
            std::env::var_os(RUNTIME_ENV).is_none(),
            "the parent removes {RUNTIME_ENV}"
        );
        let application_runtime = PathBuf::from(application_runtime);
        let deployment = EmbeddingDeployment {
            onnx_runtime: Some(application_runtime.clone()),
        };
        let dir = TempDir::new("application_runtime");
        let model = package(&dir, "dynamic.onnx");

        // First with nothing anywhere — passed, set, or beside the graph: refused, the three
        // places named in the order they are looked at, and nothing loaded by it.
        match EmbeddingRuntime::new(config_for(model.clone())).load() {
            Err(EmbeddingError::OnnxRuntimeUnavailable { reason }) => {
                let at = |place: &str| {
                    reason
                        .find(place)
                        .unwrap_or_else(|| panic!("{place} is not named: {reason}"))
                };
                assert!(
                    at("EmbeddingDeployment::onnx_runtime") < at(RUNTIME_ENV)
                        && at(RUNTIME_ENV) < at("beside the model"),
                    "{reason}"
                );
            }
            other => panic!("with no runtime anywhere the load must fail, got {other:?}"),
        }

        // The application's path: loaded and run, with nothing else to find it by.
        let mut runtime =
            EmbeddingRuntime::with_deployment(config_for(model.clone()), deployment.clone());
        runtime
            .load()
            .expect("the runtime the application passes loads");
        assert_eq!(runtime.backend_id(), Some("onnxruntime-sentence-v1"));
        assert_eq!(runtime.embed_batch(&["the fox"]).unwrap()[0].len(), 4);

        // Both public paths hand it to the backend. The engine loads its model with it...
        let mut engine =
            SemanticEngine::open(engine_config(&dir, model.clone(), deployment.clone())).unwrap();
        engine
            .load_model()
            .expect("SemanticConfig::deployment reaches the backend");
        assert_eq!(
            engine.status().embedding_backend.as_deref(),
            Some("onnxruntime-sentence-v1")
        );
        // ...and so does the application's path, which loads the model before it reads the
        // artifact: with none installed, what is missing is the artifact, not the runtime.
        match OfficialSemanticIndex::open(official_config(&dir, model.clone(), deployment)) {
            Err(SemanticSearchError::Artifact(ArtifactError::MetadataUnusable {
                path, ..
            })) => {
                assert!(path.ends_with("manifest.json"), "{path}");
            }
            Err(other) => panic!("expected the artifact to be what is missing, got {other}"),
            Ok(_) => panic!("opened an artifact that is not there"),
        }

        // One runtime per process: a different path now is refused, naming both.
        let other = fixture("tokenizer.json");
        let refused = EmbeddingRuntime::with_deployment(
            config_for(model),
            EmbeddingDeployment {
                onnx_runtime: Some(other.clone()),
            },
        )
        .load();
        match refused {
            Err(EmbeddingError::OnnxRuntimeUnavailable { reason }) => {
                for named in [&application_runtime, &other] {
                    let named = named.canonicalize().unwrap().display().to_string();
                    assert!(reason.contains(&named), "{named} is not named: {reason}");
                }
            }
            other => panic!("a second, different runtime must be refused, got {other:?}"),
        }
    }

    /// A path the application passes is not one candidate among three: where it names
    /// nothing, both public paths fail naming it — even where `OTZARIA_ONNX_RUNTIME` names
    /// a working runtime, as it does in CI, that would have loaded. Refused before any
    /// runtime is touched, so it needs no library and changes nothing in this process.
    #[test]
    fn an_application_runtime_that_names_nothing_is_refused_by_both_public_paths() {
        let _guard = lock_env();
        let dir = TempDir::new("application_runtime_absent");
        let model = package(&dir, "dynamic.onnx");
        let absent = dir.0.join("not-installed").join("onnxruntime-library");
        let deployment = EmbeddingDeployment {
            onnx_runtime: Some(absent.clone()),
        };
        let is_the_refusal = |error: &EmbeddingError| {
            matches!(
                error,
                EmbeddingError::OnnxRuntimeUnavailable { reason }
                    if reason.contains("EmbeddingDeployment::onnx_runtime")
                        && reason.contains(&absent.display().to_string())
            )
        };

        let mut engine =
            SemanticEngine::open(engine_config(&dir, model.clone(), deployment.clone())).unwrap();
        match engine.load_model() {
            Err(SemanticSearchError::EmbeddingRuntime(error)) => {
                assert!(is_the_refusal(&error), "{error}");
            }
            other => panic!("the engine must refuse the path, got {other:?}"),
        }
        match OfficialSemanticIndex::open(official_config(&dir, model, deployment)) {
            Err(SemanticSearchError::EmbeddingRuntime(error)) => {
                assert!(is_the_refusal(&error), "{error}");
            }
            Err(other) => panic!("the official index must refuse the path, got {other}"),
            Ok(_) => panic!("opened on a runtime that does not exist"),
        }
    }

    /// End to end through `EmbeddingRuntime::load`: the package is validated and
    /// checksummed as one (design D3/D4), the table builds this backend for it, and the
    /// runtime normalizes what the backend returns raw.
    #[test]
    fn the_runtime_loads_an_onnx_package_and_normalizes_its_vectors() {
        if !runtime_configured() {
            return;
        }
        let _guard = lock_env();
        let dir = TempDir::new("runtime");
        let model = package(&dir, "dynamic.onnx");
        let mut runtime = EmbeddingRuntime::new(config_for(model.clone()));
        runtime
            .load()
            .expect("the fixture package loads end to end");

        assert_eq!(runtime.backend_id(), Some("onnxruntime-sentence-v1"));
        assert!(runtime.backend_is_semantic());
        assert_eq!(runtime.pooling(), Pooling::InGraph);
        assert_eq!(runtime.max_tokens(), 32);

        // The recorded checksum is the package's, computed here from the D4 recipe
        // itself: the graph and the tokenizer, not the graph alone.
        let line = |name: &str| {
            let bytes = std::fs::read(dir.0.join(name)).unwrap();
            format!("{name}\t{}\t{}\n", bytes.len(), sha256_hex(&bytes))
        };
        let manifest = format!(
            "otzaria-onnx-package-v1\n{}{}",
            line("model.onnx"),
            line("tokenizer.json")
        );
        assert_eq!(
            runtime.model_checksum(),
            Some(sha256_hex(manifest.as_bytes()).as_str())
        );

        let vectors = runtime.embed_batch(&["the fox", "תורה"]).unwrap();
        for vector in &vectors {
            let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "the runtime normalizes: {norm}");
        }
        assert_ne!(vectors[0], vectors[1]);
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        use sha2::Digest;
        sha2::Sha256::digest(bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }

    /// The stand-in's stub package — a graph with no nodes, a WordLevel tokenizer —
    /// sent to the real row, as a `mock-embedding,onnx-backend` build does: its
    /// tokenizer loads, its graph is refused as an invalid ONNX model naming the graph,
    /// and nothing panics. Without a runtime the refusal is `OnnxRuntimeUnavailable`,
    /// which `tests/backend_selection.rs` accepts as well.
    #[cfg(feature = "mock-embedding")]
    #[test]
    fn the_stand_ins_stub_package_is_refused_by_the_real_backend() {
        use otzaria_semantic_search::semantic::embedding::mock;

        let _guard = lock_env();
        let dir = TempDir::new("stub");
        let model = mock::write_stub_onnx_package(&dir.0);
        let refused = select_backend(&EmbeddingConfig {
            model_path: model.clone(),
            embedding_dim: 256,
            max_tokens: 256,
            batch_size: 4,
            pooling: Pooling::InGraph,
        })
        .map(|backend| backend.id());
        match refused {
            Err(error @ EmbeddingError::InvalidModelFile { .. }) => {
                let message = error.to_string();
                assert!(
                    message.starts_with("Not a valid ONNX model file"),
                    "{message}"
                );
                assert!(message.contains(&model.display().to_string()), "{message}");
            }
            Err(EmbeddingError::OnnxRuntimeUnavailable { reason })
                if std::env::var_os(RUNTIME_ENV).is_none() =>
            {
                assert!(reason.contains(RUNTIME_ENV), "{reason}");
            }
            other => panic!("the stub package must be refused by the real backend, got {other:?}"),
        }
    }
}
