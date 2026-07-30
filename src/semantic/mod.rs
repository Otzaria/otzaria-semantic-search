//! Semantic search subsystem.
//!
//! This module contains all components for the semantic (vector-based) search path:
//! - Types and data models
//! - Manifest / versioning
//! - Chunking (text → semantic chunks)
//! - Embedding backend contract (what an inference implementation must provide)
//! - Embedding runtime (model validation, batching, normalization)
//! - Real GGUF inference through llama.cpp (behind the `llama-backend` feature)
//! - Vector store (zvec persistence and retrieval)
//! - Semantic engine (orchestration)

pub mod backend;
pub mod chunker;
pub mod embedding;
pub mod engine;
// Real GGUF inference (roadmap P2 stage 3). Compiled only with
// `--features llama-backend`, which is what keeps a default build from pulling
// llama.cpp and ggml through cmake on every `cargo build`. Which backend a build
// actually gets is decided in `backend`, not here.
//
// A plain comment rather than a doc comment on purpose: an outer `///` here would
// be concatenated with the module's own `//!` header, and rustdoc then resolves
// that whole text's intra-doc links in *this* module's scope instead of the
// module's own — turning every correct link in `llama_backend` into an unresolved
// one under `RUSTDOCFLAGS="-D warnings"`.
#[cfg(feature = "llama-backend")]
pub mod llama_backend;
pub mod manifest;
pub mod store;
pub mod types;
