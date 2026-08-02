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

pub mod cache;
pub mod coordinator;
pub mod fusion;
pub mod grouping;
pub mod hebrew_normalizer;
pub mod metadata_ranker;
pub mod ranking;
