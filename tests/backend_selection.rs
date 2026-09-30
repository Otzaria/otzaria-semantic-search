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
//! tested here. `tests/hybrid_integration_test.rs` is the mirror image: it needs the
//! stand-in to be the *selected* backend, so it excludes `llama-backend`.
//!
//! The rule holds per format. A model's format picks the candidates walked, so the
//! GGUF tests need `llama-backend`, the ONNX tests `onnx-backend`, and the last two
//! prove the other half of the rule: a real backend of one format is never asked about
//! a model of the other, which the stand-in then serves.

#![cfg(all(
    feature = "mock-embedding",
    any(feature = "llama-backend", feature = "onnx-backend")
))]

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

/// Structurally valid GGUF, one scalar tensor, no model in it: the container
/// validator accepts it and the real backend rejects it.
#[cfg(feature = "llama-backend")]
fn stub_model(dir: &TempDir, name: &str) -> PathBuf {
    let path = dir.path().join(name);
    mock::write_stub_gguf(&path, 3).unwrap();
    path
}

fn config_for(dir: &TempDir, model_path: PathBuf) -> SemanticConfig {
    let root = dir.path().join("semantic");
    SemanticConfig {
        root_dir: root.clone(),
        model_path,
        embedding_dim: 1024,
        store: VectorStoreConfig {
            db_path: root.join("vectors"),
            embedding_dim: 1024,
            collection_name: "chunks".to_string(),
        },
        ..Default::default()
    }
}

/// The regression this file exists for: a model the real backend cannot load
/// must produce that backend's error, not a silent demotion to hash vectors.
#[cfg(feature = "llama-backend")]
#[test]
fn a_model_the_real_backend_rejects_never_falls_through_to_the_stand_in() {
    let dir = TempDir::new("stub");
    let model_path = stub_model(&dir, "stub.gguf");

    let mut runtime = EmbeddingRuntime::new(EmbeddingConfig {
        model_path: model_path.clone(),
        embedding_dim: 1024,
        ..Default::default()
    });

    let error = runtime
        .load()
        .expect_err("a stub GGUF holds no model; loading it must fail");

    // The exact error is llama.cpp's business; what must never happen is *success*.
    assert!(
        matches!(
            error,
            EmbeddingError::InvalidModelFile { .. } | EmbeddingError::LoadFailed { .. }
        ),
        "expected the real backend's load failure, got {error}"
    );
    assert!(
        error
            .to_string()
            .contains(&model_path.display().to_string()),
        "the error must name the file that could not be loaded, so the failure is \
         diagnosable without a debugger: {error}"
    );

    assert!(
        !runtime.is_loaded(),
        "a failed load must install no backend"
    );
    assert!(
        runtime.backend_id().is_none(),
        "no backend id may be reported after a failed load — the id is what the \
         manifest would record as this index's identity"
    );
    assert_ne!(
        runtime.backend_id(),
        Some("mock-hash-v1"),
        "the stand-in must not have been selected: that is the exact silent \
         failure the Option<Result<..>> selection rule exists to prevent"
    );
}

/// The same rule one layer up, where it is the app that would be misled.
#[cfg(feature = "llama-backend")]
#[test]
fn the_engine_reports_the_real_backends_failure_rather_than_becoming_available() {
    let dir = TempDir::new("engine");
    let model_path = stub_model(&dir, "stub.gguf");

    let mut engine = SemanticEngine::open(config_for(&dir, model_path)).unwrap();

    let error = engine
        .load_model()
        .expect_err("a stub GGUF must not yield a working engine");
    assert!(
        matches!(error, SemanticSearchError::EmbeddingRuntime(_)),
        "expected an embedding-runtime error, got {error}"
    );

    let status = engine.status();
    assert!(!status.model_loaded);
    assert!(!status.available);
    assert_ne!(
        status.embedding_backend.as_deref(),
        Some("mock-hash-v1"),
        "status must not advertise the stand-in when the real backend was the one \
         that failed"
    );
    assert!(
        status.last_error.is_some(),
        "the reason must reach the caller — an app that cannot say why semantic \
         search is off will be blamed for the model download instead"
    );

    // Nothing was written on the strength of a backend that never loaded.
    assert_eq!(status.vector_count, 0);
    assert_eq!(status.indexed_book_count, 0);
    assert!(engine.search("בריאת העולם", 5, None).is_err());
}

/// Missing must be diagnosed as missing, not as a load failure: "download the model"
/// and "the model you downloaded is damaged" send a user to different places.
#[cfg(feature = "llama-backend")]
#[test]
fn an_absent_model_is_still_reported_as_absent_with_both_backends_compiled_in() {
    let dir = TempDir::new("absent");
    let mut runtime = EmbeddingRuntime::new(EmbeddingConfig {
        model_path: dir.path().join("never-downloaded.gguf"),
        embedding_dim: 1024,
        ..Default::default()
    });

    assert!(
        matches!(runtime.load(), Err(EmbeddingError::ModelNotFound { .. })),
        "a path that does not exist is not a broken model"
    );
    assert!(runtime.backend_id().is_none());
}

// ── ONNX ─────────────────────────────────────────────────────────────────────────

/// `config_for`, pooling the way an ONNX graph does — the only pooling a backend for
/// that format performs, and so the only one the configuration check lets through.
#[cfg(feature = "onnx-backend")]
fn onnx_config_for(dir: &TempDir, model_path: PathBuf) -> SemanticConfig {
    SemanticConfig {
        pooling: "in-graph".to_string(),
        ..config_for(dir, model_path)
    }
}

/// The ONNX backend's refusal, whichever it is: the package it is handed holds a graph
/// with no nodes, which no runtime can run — or, where no runtime library is present,
/// the backend's own report that it cannot load one. What must never happen is
/// *success*.
#[cfg(feature = "onnx-backend")]
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
#[cfg(feature = "onnx-backend")]
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

#[cfg(feature = "onnx-backend")]
#[test]
fn the_engine_reports_the_onnx_backends_failure_rather_than_becoming_available() {
    let dir = TempDir::new("onnx_engine");
    let model_path = mock::write_stub_onnx_package(dir.path());

    let mut engine = SemanticEngine::open(onnx_config_for(&dir, model_path)).unwrap();
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

#[cfg(feature = "onnx-backend")]
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

// ── the format decides who is asked ──────────────────────────────────────────────

/// With llama.cpp compiled in and no ONNX backend, an ONNX model is the stand-in's:
/// llama.cpp, which would refuse the file and stop the walk, is never handed it.
#[cfg(all(feature = "llama-backend", not(feature = "onnx-backend")))]
#[test]
fn an_onnx_model_never_reaches_llama_cpp() {
    let dir = TempDir::new("onnx_to_stand_in");
    let mut runtime = EmbeddingRuntime::new(EmbeddingConfig {
        model_path: mock::write_stub_onnx_package(dir.path()),
        embedding_dim: 32,
        pooling: Pooling::InGraph,
        ..Default::default()
    });
    runtime
        .load()
        .expect("the stand-in serves an ONNX model when nothing real does");
    assert_eq!(runtime.backend_id(), Some("mock-hash-v1"));
    assert_eq!(runtime.pooling(), Pooling::InGraph);
}

/// The mirror image: a GGUF model never reaches the ONNX backend.
#[cfg(all(feature = "onnx-backend", not(feature = "llama-backend")))]
#[test]
fn a_gguf_model_never_reaches_the_onnx_backend() {
    let dir = TempDir::new("gguf_to_stand_in");
    let model_path = dir.path().join("stub.gguf");
    mock::write_stub_gguf(&model_path, 3).unwrap();
    let mut runtime = EmbeddingRuntime::new(EmbeddingConfig {
        model_path,
        embedding_dim: 32,
        ..Default::default()
    });
    runtime
        .load()
        .expect("the stand-in serves a GGUF model when nothing real does");
    assert_eq!(runtime.backend_id(), Some("mock-hash-v1"));
    assert_eq!(runtime.pooling(), Pooling::LastToken);
}
