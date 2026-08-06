//! Otzaria Hybrid Semantic Search Engine
//!
//! This crate provides semantic search capabilities for the Otzaria Jewish text
//! library. It operates as a sidecar to the existing Tantivy-based lexical search
//! engine, adding vector similarity search via embeddings.
//!
//! # Architecture
//!
//! ```text
//! Hybrid Coordinator
//!   ├── Lexical path (existing Tantivy/BM25 — not owned by this crate)
//!   ├── Semantic path (embedding + vector store — owned by this crate)
//!   └── Fusion / Ranking / Grouping
//! ```
//!
//! The semantic subsystem is fully independent: its own database, its own
//! manifest, its own lifecycle. A failure in the semantic path never affects
//! the existing lexical search.
//!
//! # Product contract
//!
//! The binding scope definition is `docs/PRODUCT_CONTRACT.md`. Four decisions
//! there shape every module in this crate:
//!
//! 1. **The official vector index is built ahead of time and opened read-only.**
//!    The application installs a prebuilt artifact; it never indexes the library.
//! 2. **No user overlay.** Personal books stay lexical-only; there is no writable
//!    vector layer merged on top of the official index.
//! 3. **No background indexing in the app.** No progress stream, no cancel/resume.
//!    Installing an artifact is a file operation, not inference.
//! 4. **No remote service at query time.** The query never leaves the device; only
//!    its embedding is computed, locally. "Distribution" here means packaging and
//!    installing static files — see [`distribution`].
//!
//! The indexing API still exposed on the engine is prototype scaffolding for tests
//! and for the future builder. It is not the application path.

pub mod api;
pub mod config;
pub mod distribution;
pub mod errors;
pub mod hybrid;
pub mod semantic;
pub mod telemetry;
