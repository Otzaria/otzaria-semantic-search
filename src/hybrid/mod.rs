//! Hybrid search coordination layer.
//!
//! This module merges results from the lexical (Tantivy/BM25) and semantic
//! (embedding/zvec) search paths into a single ranked result set.
//!
//! Components:
//! - Fusion: score normalization and combination
//! - Ranking: dynamic weight computation and bonus/penalty application
//! - Grouping: post-fusion result grouping (SameSection, IdenticalText)
//! - Coordinator: top-level orchestrator

pub mod fusion;
pub mod ranking;
pub mod grouping;
pub mod coordinator;
