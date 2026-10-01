//! Semantic search subsystem.
//!
//! This module contains all components for the semantic (vector-based) search path:
//! - Types and data models
//! - Manifest / versioning
//! - Chunking (text → semantic chunks), and the key a chunk's vector is stored under
//! - The recipe versions an artifact declares, as closed sets this build either
//!   implements or refuses
//! - Embedding backend contract (what an inference implementation must provide)
//! - Embedding runtime (the configuration's checks, batching, normalization)
//! - Real ONNX inference through ONNX Runtime (behind the `onnx-backend` feature) — the
//!   only inference this crate has
//! - What a model path names on disk: an ONNX graph, and the package around it
//! - Vector store backend contract, split into a read side the runtime gets and a
//!   write side only a builder gets, plus an in-memory and a snapshot-persisting
//!   implementation. Neither is an ANN index, and neither is the `zvec` library.
//! - Semantic engine (orchestration of the builder path: chunk, embed, write)
//! - The official read-only index (the application path: open a verified artifact
//!   and query it)

pub mod backend;
pub mod chunk_key;
pub mod chunker;
pub mod embedding;
pub mod embedding_cache;
pub mod engine;
pub mod manifest;
pub mod model_package;
pub mod official_index;
// Real ONNX inference through ONNX Runtime. Compiled only with
// `--features onnx-backend`; which backend a build actually gets is decided in
// `backend`, not here.
//
// `backend::onnx_backend` is gated on the same condition, so a target restriction
// added here must be added there too.
//
// A plain comment rather than a doc comment on purpose: an outer `///` here would
// be concatenated with the module's own `//!` header, and rustdoc then resolves
// that whole text's intra-doc links in *this* module's scope instead of the
// module's own — turning every correct link in `onnx_backend` into an unresolved
// one under `RUSTDOCFLAGS="-D warnings"`.
//
// The target half is the desktop set a loadable ONNX Runtime exists for, and mirrors
// the dependency declarations in `Cargo.toml`; `tests/onnx_backend.rs` fails if the
// spellings drift apart.
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
pub mod onnx_backend;
pub mod recipe;
pub mod store;
pub mod store_backend;
pub mod types;
pub mod versioning;
pub mod zevc_store;
