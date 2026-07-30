//! Asserts that a production build cannot produce embeddings.
//!
//! Both the deterministic stand-in (`mock-embedding`) and real GGUF inference
//! (`llama-backend`) are non-default, precisely so a release binary can never serve
//! fake vectors as if they were semantic and can never silently grow a 400 MB
//! native dependency.
//!
//! [`the_manifest_enables_no_embedding_backend_by_default`] is deliberately not
//! `#[cfg]`-gated. When the whole file was gated on `not(any(..))`, adding either
//! feature to `default` made this target compile to zero tests and pass — a failure
//! mode with no symptom. A `compile_error!` cannot replace it: an inner `#![cfg]`
//! removes the guard along with the crate, and `--features llama-backend` also
//! enables `default`, so only reading the manifest distinguishes "a backend was
//! requested for this build" from "a backend ships by default".
//!
//! [`without_a_backend`] holds the behavioural half, and excludes `llama-backend`
//! for a real reason: it asserts the *absence* of any backend using a stub GGUF, and
//! with real inference compiled in llama.cpp refuses that stub as
//! `InvalidModelFile` instead — correct behaviour, and not what is under test here.

/// `[features] default` must not contain an embedding backend.
///
/// `include_str!` rather than a runtime read, so cargo treats the manifest as an
/// input and rebuilds this test when it changes.
#[test]
fn the_manifest_enables_no_embedding_backend_by_default() {
    const MANIFEST: &str = include_str!("../Cargo.toml");
    const BACKENDS: [&str; 2] = ["mock-embedding", "llama-backend"];

    // Enough of a TOML reader for one key. Comments are stripped first because
    // `# llama-backend` in prose must not count as an entry.
    let mut in_features = false;
    let mut collecting = false;
    let mut default_set = String::new();
    for line in MANIFEST.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.starts_with('[') {
            in_features = line == "[features]";
            continue;
        }
        if !in_features {
            continue;
        }
        if !collecting {
            let Some(rest) = line.strip_prefix("default") else {
                continue;
            };
            let Some(rest) = rest.trim_start().strip_prefix('=') else {
                continue;
            };
            collecting = true;
            default_set.push_str(rest);
        } else {
            default_set.push_str(line);
        }
        if default_set.contains(']') {
            break;
        }
    }

    assert!(
        collecting,
        "no `default = [...]` key was found under [features] in Cargo.toml. Either the \
         manifest changed shape or the default feature set was removed; this test cannot \
         guarantee anything until it can read it again."
    );

    // The features must still exist under these names, or this test passes by
    // looking for something that is no longer there.
    for backend in BACKENDS {
        assert!(
            MANIFEST.contains(&format!("\n{backend} = [")),
            "Cargo.toml no longer declares a `{backend}` feature. If it was renamed, rename \
             it here too — otherwise this gate silently checks for nothing."
        );
        assert!(
            !default_set.contains(backend),
            "`{backend}` has been added to [features] default in Cargo.toml \
             (default = {default_set}).\n\
             A release build must not be able to produce embeddings: `mock-embedding` would \
             let it serve hash vectors as if they were semantic, and `llama-backend` would \
             make every downstream build compile llama.cpp. Both are opt-in by design, and \
             the behavioural half of this file only runs when neither is on — so enabling \
             one by default would leave that half compiling to zero tests. If this change is \
             deliberate, delete this file and the guarantee with it, deliberately."
        );
    }
}

#[cfg(not(any(feature = "mock-embedding", feature = "llama-backend")))]
mod without_a_backend {
    use otzaria_semantic_search::errors::{EmbeddingError, SemanticSearchError};
    use otzaria_semantic_search::semantic::embedding::{EmbeddingConfig, EmbeddingRuntime};
    use otzaria_semantic_search::semantic::engine::{SemanticConfig, SemanticEngine};
    use otzaria_semantic_search::semantic::store::VectorStoreConfig;
    use std::path::{Path, PathBuf};

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "otzaria_production_gate_{name}_{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir_all(&path).unwrap();
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

    /// A minimal structurally valid GGUF with one scalar tensor, so a failure is
    /// about the missing backend rather than about the file.
    fn write_valid_gguf(path: &Path) {
        let mut bytes = Vec::with_capacity(68);
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
        bytes.extend_from_slice(&0u64.to_le_bytes()); // metadata_kv_count
        bytes.extend_from_slice(&1u64.to_le_bytes()); // tensor name length
        bytes.push(b'x');
        bytes.extend_from_slice(&1u32.to_le_bytes()); // dimension count
        bytes.extend_from_slice(&1u64.to_le_bytes()); // one element
        bytes.extend_from_slice(&0u32.to_le_bytes()); // F32
        bytes.extend_from_slice(&0u64.to_le_bytes()); // aligned data offset
        while bytes.len() % 32 != 0 {
            bytes.push(0);
        }
        bytes.extend_from_slice(&0f32.to_le_bytes());
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn loading_a_valid_model_reports_that_no_backend_is_compiled_in() {
        let dir = TempDir::new("runtime");
        let model_path = dir.path().join("model.gguf");
        write_valid_gguf(&model_path);

        let mut runtime = EmbeddingRuntime::new(EmbeddingConfig {
            model_path,
            embedding_dim: 64,
            ..Default::default()
        });

        let error = runtime
            .load()
            .expect_err("a build with no inference backend must not load a model");
        assert!(
            matches!(error, EmbeddingError::BackendUnavailable { .. }),
            "expected BackendUnavailable, got {error}"
        );
        assert!(!runtime.is_loaded());
        assert!(runtime.backend().is_none());
    }

    #[test]
    fn the_engine_refuses_to_embed_and_stays_unavailable() {
        let dir = TempDir::new("engine");
        let model_path = dir.path().join("model.gguf");
        write_valid_gguf(&model_path);

        let root = dir.path().join("semantic");
        let mut engine = SemanticEngine::open(SemanticConfig {
            root_dir: root.clone(),
            model_path,
            embedding_dim: 64,
            store: VectorStoreConfig {
                db_path: root.join("vectors"),
                embedding_dim: 64,
                collection_name: "chunks".to_string(),
            },
            ..Default::default()
        })
        .unwrap();

        let error = engine.load_model().expect_err("no backend is available");
        assert!(
            matches!(
                error,
                SemanticSearchError::EmbeddingRuntime(EmbeddingError::BackendUnavailable { .. })
            ),
            "expected BackendUnavailable, got {error}"
        );

        let status = engine.status();
        assert!(!status.model_loaded);
        assert!(!status.available);
        assert!(status.embedding_backend.is_none());
        assert!(
            status.last_error.is_some(),
            "the reason must be visible to the caller"
        );

        // Only inference is missing, so the app can show a real status, not crash.
        assert_eq!(status.vector_count, 0);
        assert_eq!(status.indexed_book_count, 0);
        assert!(status.needs_full_reindex.is_none());
        assert!(engine.search("בריאת העולם", 5, None).is_err());
    }
}
