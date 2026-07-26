//! Asserts that a production build cannot produce embeddings.
//!
//! The deterministic stand-in embedder lives behind the non-default
//! `mock-embedding` feature precisely so a release binary can never serve fake
//! vectors as if they were semantic. That is a guarantee, not a convention, so it
//! is tested rather than trusted to a `#[cfg]` staying where it is.
//!
//! This file is the mirror image of `hybrid_integration_test.rs`: it compiles
//! only *without* the feature, and CI runs both configurations.

#![cfg(not(feature = "mock-embedding"))]

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

/// A structurally valid, empty GGUF container, so a failure is about the missing
/// backend rather than about the file.
fn write_valid_gguf(path: &Path) {
    let mut bytes = Vec::with_capacity(24);
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
    bytes.extend_from_slice(&0u64.to_le_bytes()); // metadata_kv_count
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

    // The rest of the engine still opens and reports cleanly — only inference is
    // missing, so the app can show a real status instead of crashing.
    assert_eq!(status.vector_count, 0);
    assert_eq!(status.indexed_book_count, 0);
    assert!(status.needs_full_reindex.is_none());
    assert!(engine.search("בריאת העולם", 5, None).is_err());
}
