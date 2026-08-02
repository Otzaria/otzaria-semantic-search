# Otzaria Hybrid Semantic Search Engine

[![CI](https://github.com/Otzaria/otzaria-semantic-search/actions/workflows/ci.yml/badge.svg)](https://github.com/Otzaria/otzaria-semantic-search/actions/workflows/ci.yml)
[![Personal Use License](https://img.shields.io/badge/license-Personal%20Use%201.0-blue.svg?style=flat)](LICENSE)
[![Rust 2021](https://img.shields.io/badge/rust-2021%20edition-orange.svg)](https://www.rust-lang.org)
[![Platform Support](https://img.shields.io/badge/platform-Linux%20%7C%20Windows%20%7C%20macOS-blue.svg)](#ci-pipeline)

**Otzaria Hybrid Semantic Search** is a production-ready hybrid search engine for **Otzaria**, the open rabbinic digital library.

It combines Tantivy/BM25 lexical search with vector similarity search via local GGUF embedding models, multi-stage reranking, persistent vector storage, caching, and telemetry.

---

## 💡 Key Design Principles

1. **Non-Destructive Sidecar Architecture**: Operates as an independent sidecar database (`semantic_db`). It **never** mutates, alters, or replaces Otzaria's existing Tantivy lexical database.
2. **Graceful Fallback & Resilience**: If the semantic path fails (e.g. model missing, disk I/O error), the coordinator automatically falls back to lexical-only mode without crashing the app.
3. **Offline & Private Target**: Runs entirely on-device — inference is local llama.cpp over a GGUF file. The crate performs no model download and no telemetry; obtaining the model is the host application's job.
4. **Multi-Stage Hybrid Reranking**: Advanced signal fusion combining BM25, Cosine similarity, quoted phrase matching, rare term boosting, section coverage, duplicate penalties, and canonical metadata ranking.
5. **Source Retrieval (Not RAG)**: Designed strictly for accurate source and text retrieval within Jewish literature. It returns verifiable textual sources, never hallucinated AI responses.

---

## 🏗️ Architecture Overview

```text
                                ┌─────────────────────────────────────────┐
                                │           Otzaria App (Flutter)         │
                                └────────────────────┬────────────────────┘
                                                     │ FFI (flutter_rust_bridge)
                                                     ▼
                                ┌─────────────────────────────────────────┐
                                │          OtzariaHybridEngine            │
                                └────────────────────┬────────────────────┘
                                                     │
                                                     ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                         HybridCoordinator                                                │
│                                                                                                          │
│   ┌────────────────────────┐         ┌────────────────────────┐         ┌────────────────────────────┐   │
│   │   Hebrew Normalizer    │───────▶ │   Query Cache (LRU)    │───────▶ │      Query Analysis        │   │
│   │  (Nikud/Taamim/Geresh) │         │ (Generation & TTL)     │         │   (Short/Exact/Conceptual) │   │
│   └────────────────────────┘         └────────────────────────┘         └──────────────┬─────────────┘   │
│                                                                                        │                 │
│   ┌────────────────────────┐         ┌────────────────────────┐                        ▼                 │
│   │   Lexical Candidates   │         │  Embedding Cache       │         ┌────────────────────────────┐   │
│   │    (Tantivy BM25)      │───────▶ │ (FNV-1a query vectors) │───────▶ │    Multi-Stage Fusion     │   │
│   └────────────────────────┘         └────────────────────────┘         │ (Adaptive/Weighted/RRF)    │   │
│                                                                         └──────────────┬─────────────┘   │
│                                                                                        │                 │
│   ┌────────────────────────┐         ┌────────────────────────┐                        ▼                 │
│   │   Telemetry Collector  │         │ Metadata Reranker      │         ┌────────────────────────────┐   │
│   │  (Atomic U64 Metrics)  │◀────────│ (Canonical & Era Boost)│◀────────│    HybridSearchResult      │   │
│   └────────────────────────┘         └────────────────────────┘         │  (Paginated & Hydrated)    │   │
│                                                                         └────────────────────────────┘   │
└───────────────────────────────────────────────────┬──────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       SemanticEngine (Sidecar)                                           │
│                                                                                                          │
│   ┌────────────────────────┐         ┌────────────────────────┐         ┌────────────────────────────┐   │
│   │    Anchored Chunker    │         │   LlamaCppBackend      │         │   Zevc Persistent Store    │   │
│   │ (same-section context │───────▶ │ (Qwen3 1024-dim Q4     │───────▶ │ (Vector & metadata binary  │   │
│   │   + SHA256 Anchor IDs) │         │   GGUF inference)      │         │  file persistence)         │   │
│   └────────────────────────┘         └────────────────────────┘         └────────────────────────────┘   │
│                                                                                        ▲                 │
│   ┌────────────────────────────────────────────────────────────────────────────────────┴─────────────┐   │
│   │                                   SemanticManifest & Versioning                             │   │
│   │                IndexVersion tracking, model verification & Tantivy diff tracking                │   │
│   └──────────────────────────────────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🗺️ Code Map & File Index

```text
otzaria-semantic-search/
├── Cargo.toml                          ➜ Crate dependencies & build configuration
├── README.md                            ➜ Master project documentation & guide
├── src/
│   ├── lib.rs                          ➜ Library root module exports
│   ├── errors.rs                       ➜ Strongly-typed error hierarchy (thiserror)
│   ├── api/
│   │   ├── mod.rs                      ➜ API module declaration
│   │   └── hybrid_search.rs            ➜ Flutter / FFI bridge entry point (OtzariaHybridEngine)
│   ├── benchmark/
│   │   └── mod.rs                      ➜ Benchmark suite (latency percentiles, throughput QPS)
│   ├── cloud/
│   │   ├── mod.rs                      ➜ Cloud index builder module declaration
│   │   ├── builder.rs                  ➜ Server-side offline index generator
│   │   ├── importer.rs                 ➜ Client-side package verifier & importer
│   │   └── package.rs                  ➜ .oix package reader/writer with SHA256 checksums
│   ├── config/
│   │   ├── mod.rs                      ➜ Configuration module declaration
│   │   ├── feature_flags.rs            ➜ Runtime feature flags for A/B testing
│   │   └── profiles.rs                 ➜ Fast / Balanced / Best search profiles
│   ├── hybrid/
│   │   ├── mod.rs                      ➜ Hybrid search module declaration
│   │   ├── cache.rs                    ➜ Query result LRU cache & generation tracking
│   │   ├── coordinator.rs              ➜ Hybrid search coordinator & fallback logic
│   │   ├── fusion.rs                   ➜ Multi-stage score normalization & fusion (Weighted/RRF/Adaptive)
│   │   ├── grouping.rs                 ➜ Post-fusion result grouping (Section & IdenticalText)
│   │   ├── hebrew_normalizer.rs        ➜ Nikud, Taamim, and Geresh normalization & language detector
│   │   ├── metadata_ranker.rs          ➜ Primary canon & era affinity ranking boost
│   │   └── ranking.rs                  ➜ Query features, phrase match, rare term & agreement signals
│   ├── semantic/
│   │   ├── mod.rs                      ➜ Semantic subsystem module declaration
│   │   ├── backend.rs                  ➜ EmbeddingBackend contract
│   │   ├── chunker.rs                  ➜ Anchored semantic chunking & SHA256 ID generation
│   │   ├── embedding.rs                ➜ GGUF model runtime interface & L2 normalization
│   │   ├── embedding_cache.rs          ➜ Query vector embedding cache
│   │   ├── engine.rs                   ➜ SemanticEngine sidecar orchestrator
│   │   ├── llama_backend.rs            ➜ llama.cpp GGUF inference backend
│   │   ├── manifest.rs                 ➜ Atomic JSON manifest versioning & Tantivy diff tracker
│   │   ├── store.rs                    ➜ In-memory vector store implementation
│   │   ├── store_backend.rs            ➜ VectorStoreBackend trait abstraction
│   │   ├── types.rs                    ➜ Domain models & DTOs
│   │   ├── versioning.rs               ➜ IndexVersion identity & compatibility checks
│   │   └── zevc_store.rs               ➜ Persistent Zevc vector store implementation
│   └── telemetry/
│       └── mod.rs                      ➜ Lock-free atomic search metrics collector
```

---

## ⚡ Configuration Profiles

The search engine supports predefined profiles tuned for different trade-offs:

| Profile | Strategy | BM25 Saturation | Semantic Threshold | Features Enabled | Candidate Window |
|---|---|---|---|---|---|
| **Fast** | RRF (k=60) | $x / (k+x)$ | 0.5 | Minimal reranking, Query/Embedding Cache | 1.5× limit |
| **Balanced** | Weighted | Adaptive | 0.3 | Agreement, Phrase Match, Dup Penalty | 2.0× limit |
| **Best** | Adaptive | Adaptive | 0.2 | Full Reranking + Metadata Boost + Rare Terms | 3.0× limit |

---

## 🛠️ Building & Testing

### Commands

```bash
# Build release library
cargo build --release

# Run unit tests
cargo test --lib --tests

# Run tests with mock embedding backend
cargo test --lib --tests --features mock-embedding

# Run tests with real llama.cpp backend
cargo test --lib --tests --features llama-backend
```

---

## 📜 License

This repository is distributed under the **Personal Use License 1.0 — Personal Use Only**. See [LICENSE](LICENSE) for details.
