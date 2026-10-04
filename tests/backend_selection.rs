//! Asserts the backend-selection rule in the one build that can exercise it.
//!
//! `select_backend` walks a table of candidates, and the distinction that matters is
//! between `None` (not compiled into this build — keep walking) and `Some(Err(e))`
//! (compiled in and **failed** — stop and report `e`). With a bare `Option` those two
//! were indistinguishable, so a broken, truncated or non-embedding model was answered
//! with hash vectors from the stand-in, and the manifest recorded `mock-hash-v1` as
//! though that had been the intent.
//!
//! That distinction is unreachable in any single-feature build, so it can only be
//! tested here, with the real ONNX backend and the stand-in both compiled in.
//! `tests/hybrid_integration_test.rs` is the mirror image: it needs the stand-in to be
//! the *selected* backend, so it excludes `onnx-backend`. The last test holds the rule's
//! precondition: a model path that names no ONNX graph reaches neither of them.

#![cfg(all(feature = "mock-embedding", feature = "onnx-backend"))]

use otzaria_semantic_search::errors::{EmbeddingError, SemanticSearchError};
use otzaria_semantic_search::semantic::backend::Pooling;
use otzaria_semantic_search::semantic::embedding::{mock, EmbeddingConfig, EmbeddingRuntime};
use otzaria_semantic_search::semantic::engine::{SemanticConfig, SemanticEngine};
use otzaria_semantic_search::semantic::store::VectorStoreConfig;
use std::path::{Path, PathBuf};

struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> Self {
        // The clock alone collided: macOS ticks coarser than a test takes to start.
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "otzaria_backend_selection_{name}_{}_{}",
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

/// A configuration over `model_path` with the production model's width, pooling and cap
/// — the defaults.
fn config_for(dir: &TempDir, model_path: PathBuf) -> SemanticConfig {
    let root = dir.path().join("semantic");
    SemanticConfig {
        root_dir: root.clone(),
        model_path,
        embedding_dim: 256,
        store: VectorStoreConfig {
            db_path: root.join("vectors"),
            embedding_dim: 256,
            collection_name: "chunks".to_string(),
        },
        ..Default::default()
    }
}

/// The ONNX backend's refusal, whichever it is: the package it is handed holds a graph
/// with no nodes, which no runtime can run — or, where no runtime library is present,
/// the backend's own report that it cannot load one. What must never happen is
/// *success*.
fn is_the_onnx_backends_refusal(error: &EmbeddingError) -> bool {
    matches!(
        error,
        EmbeddingError::InvalidModelFile { .. }
            | EmbeddingError::LoadFailed { .. }
            | EmbeddingError::OnnxRuntimeUnavailable { .. }
    )
}

/// The same regression for ONNX: a package the ONNX backend cannot serve must produce
/// that backend's error, not hash vectors from the stand-in behind it.
#[test]
fn an_onnx_model_the_real_backend_rejects_never_falls_through_to_the_stand_in() {
    let dir = TempDir::new("onnx_stub");
    let model_path = mock::write_stub_onnx_package(dir.path());

    let mut runtime = EmbeddingRuntime::new(EmbeddingConfig {
        model_path: model_path.clone(),
        embedding_dim: 256,
        max_tokens: 256,
        pooling: Pooling::InGraph,
        ..Default::default()
    });

    let error = runtime
        .load()
        .expect_err("the stub package's graph cannot run; loading it must fail");
    assert!(
        is_the_onnx_backends_refusal(&error),
        "expected the ONNX backend's refusal, got {error}"
    );
    if let EmbeddingError::InvalidModelFile { .. } = error {
        assert!(
            error
                .to_string()
                .contains(&model_path.display().to_string()),
            "an invalid model must be named: {error}"
        );
    }
    assert!(!runtime.is_loaded());
    assert!(
        runtime.backend_id().is_none(),
        "no backend id may be reported after a failed load"
    );
    assert!(runtime.model_checksum().is_none());
}

#[test]
fn the_engine_reports_the_onnx_backends_failure_rather_than_becoming_available() {
    let dir = TempDir::new("onnx_engine");
    let model_path = mock::write_stub_onnx_package(dir.path());

    let mut engine = SemanticEngine::open(config_for(&dir, model_path)).unwrap();
    let error = engine
        .load_model()
        .expect_err("the stub package must not yield a working engine");
    assert!(
        matches!(&error, SemanticSearchError::EmbeddingRuntime(inner) if is_the_onnx_backends_refusal(inner)),
        "expected the ONNX backend's refusal, got {error}"
    );

    let status = engine.status();
    assert!(!status.model_loaded);
    assert!(!status.available);
    assert_ne!(status.embedding_backend.as_deref(), Some("mock-hash-v1"));
    assert!(status.last_error.is_some());
    assert_eq!(status.vector_count, 0);
}

#[test]
fn an_absent_onnx_model_is_still_reported_as_absent() {
    let dir = TempDir::new("onnx_absent");
    let mut runtime = EmbeddingRuntime::new(EmbeddingConfig {
        model_path: dir.path().join("never-downloaded.onnx"),
        embedding_dim: 256,
        pooling: Pooling::InGraph,
        ..Default::default()
    });
    assert!(
        matches!(runtime.load(), Err(EmbeddingError::ModelNotFound { .. })),
        "a path that does not exist is not a broken model"
    );
    assert!(runtime.backend_id().is_none());
}

// ── a path that names no graph is asked of no one ───────────────────────────────

/// A model path that names no ONNX graph — a GGUF above all, the format of the llama.cpp
/// backend that is gone — is refused as an invalid model before the walk: neither the real
/// backend nor the stand-in behind it is handed the file, so it can neither half-load nor
/// be answered with hash vectors.
#[test]
fn a_model_that_is_not_an_onnx_graph_reaches_no_backend() {
    let dir = TempDir::new("not_onnx");
    let model_path = dir.path().join("stub.gguf");
    std::fs::write(&model_path, b"GGUF\x03\x00\x00\x00").unwrap();
    let mut runtime = EmbeddingRuntime::new(EmbeddingConfig {
        model_path: model_path.clone(),
        embedding_dim: 32,
        ..Default::default()
    });
    match runtime.load() {
        Err(EmbeddingError::InvalidModelFile { path, reason }) => {
            assert_eq!(path, model_path.display().to_string());
            assert!(reason.contains("does not end in .onnx"), "{reason}");
        }
        other => panic!("a .gguf path must be refused before any backend, got {other:?}"),
    }
    assert!(!runtime.is_loaded());
    assert!(runtime.backend_id().is_none());
    assert!(runtime.model_checksum().is_none());
}
