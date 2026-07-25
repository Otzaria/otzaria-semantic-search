//! Semantic search subsystem.
//!
//! This module contains all components for the semantic (vector-based) search path:
//! - Types and data models
//! - Manifest / versioning
//! - Chunking (text → semantic chunks)
//! - Embedding runtime (GGUF model inference)
//! - Vector store (zvec persistence and retrieval)
//! - Semantic engine (orchestration)

pub mod chunker;
pub mod embedding;
pub mod engine;
pub mod manifest;
pub mod store;
pub mod types;
