//! Asserts that a production build cannot produce embeddings.
//!
//! The deterministic stand-in (`mock-embedding`) and real ONNX inference
//! (`onnx-backend`) are both non-default, precisely so a release binary can never serve
//! fake vectors as if they were semantic and can never silently grow a native dependency.
//!
//! [`the_manifest_enables_no_embedding_backend_by_default`] is deliberately not
//! `#[cfg]`-gated. When the whole file was gated on `not(any(..))`, adding either
//! feature to `default` made this target compile to zero tests and pass — a failure
//! mode with no symptom. A `compile_error!` cannot replace it: an inner `#![cfg]`
//! removes the guard along with the crate, and `--features onnx-backend` also
//! enables `default`, so only reading the manifest distinguishes "a backend was
//! requested for this build" from "a backend ships by default".
//!
//! [`without_an_onnx_backend`] holds the behavioural half, and excludes `onnx-backend`
//! for a real reason: it asserts the *absence* of any backend using a stub ONNX package,
//! and with real inference compiled in the ONNX backend refuses that stub — it has no
//! nodes to run — instead: correct behaviour, and not what is under test here.

/// `[features] default` must not contain an embedding backend.
///
/// `include_str!` rather than a runtime read, so cargo treats the manifest as an
/// input and rebuilds this test when it changes.
#[test]
fn the_manifest_enables_no_embedding_backend_by_default() {
    const MANIFEST: &str = include_str!("../Cargo.toml");
    const BACKENDS: [&str; 2] = ["mock-embedding", "onnx-backend"];

    // Enough of a TOML reader for one key. Comments are stripped first because
    // `# onnx-backend` in prose must not count as an entry.
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
             let it serve hash vectors as if they were semantic, and `onnx-backend` would \
             give every downstream build an ONNX inference backend. Both are opt-in by \
             design, and the behavioural half of this file only runs when they are off — so \
             enabling one by default would leave it compiling to zero tests. If this \
             change is deliberate, delete this file and the guarantee with it, deliberately."
        );
    }
}

#[cfg(not(any(feature = "mock-embedding", feature = "onnx-backend")))]
mod without_an_onnx_backend {
    use otzaria_semantic_search::errors::{EmbeddingError, SemanticSearchError};
    use otzaria_semantic_search::semantic::backend::Pooling;
    use otzaria_semantic_search::semantic::embedding::{EmbeddingConfig, EmbeddingRuntime};
    use otzaria_semantic_search::semantic::engine::{SemanticConfig, SemanticEngine};
    use otzaria_semantic_search::semantic::store::VectorStoreConfig;
    use std::path::{Path, PathBuf};

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(name: &str) -> Self {
            // The clock alone collided: macOS ticks coarser than a test takes to start.
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "otzaria_production_gate_onnx_{name}_{}_{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos(),
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
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

    fn varint(out: &mut Vec<u8>, mut value: u64) {
        while value >= 0x80 {
            out.push((value as u8) | 0x80);
            value >>= 7;
        }
        out.push(value as u8);
    }

    /// A length-delimited field.
    fn field(out: &mut Vec<u8>, number: u64, payload: &[u8]) {
        varint(out, (number << 3) | 2);
        varint(out, payload.len() as u64);
        out.extend_from_slice(payload);
    }

    /// A `ValueInfoProto` naming a tensor of `elem_type`.
    fn tensor_value(name: &str, elem_type: u64) -> Vec<u8> {
        let mut tensor_type = vec![0x08];
        varint(&mut tensor_type, elem_type);
        let mut type_proto = Vec::new();
        field(&mut type_proto, 1, &tensor_type);
        let mut value = Vec::new();
        field(&mut value, 1, name.as_bytes());
        field(&mut value, 2, &type_proto);
        value
    }

    /// A minimal valid ONNX package: IR 8, opset 17, a graph taking `input_ids` and
    /// `attention_mask` and declaring one output, and a `tokenizer.json` beside it —
    /// written by hand, because a build without `mock-embedding` has no fixture writer.
    /// The validator accepts it, so a failure is about the missing backend.
    fn write_valid_onnx_package(dir: &Path) -> PathBuf {
        let mut graph = Vec::new();
        field(&mut graph, 11, &tensor_value("input_ids", 7));
        field(&mut graph, 11, &tensor_value("attention_mask", 7));
        field(&mut graph, 12, &tensor_value("sentence_embedding", 1));
        let mut model = vec![0x08, 0x08]; // ir_version = 8
        field(&mut model, 7, &graph);
        field(&mut model, 8, &[0x10, 0x11]); // opset_import { version: 17 }

        let path = dir.join("model.onnx");
        std::fs::write(&path, model).unwrap();
        std::fs::write(
            dir.join("tokenizer.json"),
            br#"{"model":{"type":"WordLevel"}}"#,
        )
        .unwrap();
        path
    }

    #[test]
    fn loading_a_valid_onnx_package_reports_that_no_onnx_backend_is_compiled_in() {
        let dir = TempDir::new("runtime");
        let model_path = write_valid_onnx_package(dir.path());

        let mut runtime = EmbeddingRuntime::new(EmbeddingConfig {
            model_path,
            embedding_dim: 64,
            pooling: Pooling::InGraph,
            ..Default::default()
        });

        let error = runtime
            .load()
            .expect_err("a build with no ONNX backend must not load an ONNX model");
        match &error {
            EmbeddingError::BackendUnavailable { reason } => assert!(
                reason.contains("onnx-backend"),
                "the fix is a rebuild with the feature, and the message must name it: {reason}"
            ),
            other => panic!("expected BackendUnavailable, got {other}"),
        }
        assert!(!runtime.is_loaded());
        assert!(runtime.backend().is_none());
        assert!(runtime.model_checksum().is_none());
    }

    #[test]
    fn the_engine_refuses_to_embed_with_an_onnx_model_and_stays_unavailable() {
        let dir = TempDir::new("engine");
        let model_path = write_valid_onnx_package(dir.path());

        let root = dir.path().join("semantic");
        let mut engine = SemanticEngine::open(SemanticConfig {
            root_dir: root.clone(),
            model_path,
            pooling: "in-graph".to_string(),
            embedding_dim: 64,
            store: VectorStoreConfig {
                db_path: root.join("vectors"),
                embedding_dim: 64,
                collection_name: "chunks".to_string(),
            },
            ..Default::default()
        })
        .unwrap();

        let error = engine
            .load_model()
            .expect_err("no ONNX backend is available");
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
        assert!(status.last_error.is_some());
        assert_eq!(status.vector_count, 0);
        assert!(engine.search("בריאת העולם", 5, None).is_err());
    }
}
