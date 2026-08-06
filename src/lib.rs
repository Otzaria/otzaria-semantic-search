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
//!   ├── Semantic path (embedding + zvec — owned by this crate)
//!   └── Fusion / Ranking / Grouping
//! ```
//!
//! The semantic subsystem is fully independent: its own database, its own
//! manifest, its own lifecycle. A failure in the semantic path never affects
//! the existing lexical search.

pub mod api;
pub mod config;
pub mod distribution;
pub mod errors;
pub mod hybrid;
pub mod semantic;
pub mod telemetry;
