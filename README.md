# Otzaria Hybrid Semantic Search Engine

[![CI](https://github.com/Otzaria/otzaria-semantic-search/actions/workflows/ci.yml/badge.svg)](https://github.com/Otzaria/otzaria-semantic-search/actions/workflows/ci.yml)
[![Personal Use License](https://img.shields.io/badge/license-Personal%20Use%201.0-blue.svg?style=flat)](LICENSE)
[![Rust 2021](https://img.shields.io/badge/rust-2021%20edition-orange.svg)](https://www.rust-lang.org)
[![Platform Support](https://img.shields.io/badge/platform-Linux%20%7C%20Windows%20%7C%20macOS-blue.svg)](#ci-pipeline)

**Otzaria Hybrid Semantic Search** is a correctness-focused Rust prototype for
bringing local semantic search to **Otzaria**, the open rabbinic digital library.
It is not production-ready yet.

The current crate implements chunking, lifecycle contracts, brute-force vector
search, result fusion and a Rust API seam. Real GGUF inference, a persistent ANN
store, generated FFI bindings, Tantivy hydration and application integration are
still roadmap work. Tests can enable a deterministic mock embedder; normal builds
refuse to produce fake semantic vectors.

---

## 💡 Key Design Principles

1. **Non-Destructive Sidecar Architecture**: The semantic engine operates as an independent sidecar database (`semantic_db`). It **never** mutates, alters, or replaces Otzaria's existing Tantivy lexical database.
2. **Graceful Fallback & Resilience**: If the semantic path fails (e.g. model missing, disk I/O error), the coordinator automatically falls back to lexical-only mode without crashing the app.
3. **Offline & Private Target**: The intended production design runs entirely on-device. The present crate performs no model download or telemetry, but the real inference backend is not implemented.
4. **Source Retrieval (Not RAG)**: Designed strictly for accurate source and text retrieval within Jewish literature. It returns verifiable textual sources, never hallucinated AI responses.
5. **Defensive Error Handling**: Known poisoned-lock and input edge cases use error propagation or graceful fallback; this is not an absolute panic-freedom guarantee.

---

## 🏗️ Architecture Overview

```text
                               ┌─────────────────────────────────────────┐
                               │           Otzaria App (Flutter)         │
                               └────────────────────┬────────────────────┘
                                                    │ planned FFI (flutter_rust_bridge)
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
│   │     Query Analysis     │         │    Score Normalization │         │   Grouping & Deduplication │   │
│   │   (Exact/Conceptual)   │───────▶ │     (BM25 + Cosine)    │───────▶ │   (SameSection / Identical)│   │
│   └────────────────────────┘         └────────────────────────┘         └──────────────┬─────────────┘   │
│                                                   ▲                                    │                 │
│                                                   │ Weighted fusion (current)           ▼                 │
│   ┌────────────────────────┐                      │                     ┌────────────────────────────┐   │
│   │   Lexical Candidates   │──────────────────────┴────────────────────▶│    HybridSearchResult      │   │
│   │    (Tantivy BM25)      │                                            │  (Paginated & Fused Items) │   │
│   └────────────────────────┘                                            └────────────────────────────┘   │
└───────────────────────────────────────────────────┬──────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       SemanticEngine (Sidecar)                                           │
│                                                                                                          │
│   ┌────────────────────────┐         ┌────────────────────────┐         ┌────────────────────────────┐   │
│   │    Anchored Chunker    │         │    Runtime Interface   │         │  In-memory Vector Store    │   │
│   │ (same-section context │───────▶ │ (GGUF validation; real │───────▶ │  (Pre-normalized Vectors + │   │
│   │   + SHA256 Anchor IDs) │         │    inference pending)  │         │   BinaryHeap Top-K Search) │   │
│   └────────────────────────┘         └────────────────────────┘         └────────────────────────────┘   │
│                                                                                        ▲                 │
│   ┌────────────────────────────────────────────────────────────────────────────────────┴─────────────┐   │
│   │                                   SemanticManifest (JSON)                                       │   │
│   │                  Atomic versioning, model verification & Tantivy diff tracking                   │   │
│   └──────────────────────────────────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🗺️ Code Map & File Index

For detailed architectural guidelines and invariants, see [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) and [docs/CODE_MAP.md](docs/CODE_MAP.md).

```text
otzaria-semantic-search/
├── Cargo.toml                          ➜ Crate dependencies & build configuration
├── README.md                            ➜ Master project documentation & guide
├── docs/
│   ├── CODE_MAP.md                     ➜ Comprehensive code map & module breakdown
│   └── DEVELOPMENT.md                  ➜ Developer guide, architecture invariants & status
├── .github/workflows/
│   └── ci.yml                          ➜ Multi-platform CI pipeline (Linux, Windows, macOS)
├── tests/
│   └── hybrid_integration_test.rs      ➜ End-to-end integration test suite
└── src/
    ├── lib.rs                          ➜ Library root module exports
    ├── errors.rs                       ➜ Strongly-typed error hierarchy (thiserror)
    ├── api/
    │   ├── mod.rs                      ➜ API module declaration
    │   └── hybrid_search.rs            ➜ Flutter / FFI bridge entry point (OtzariaHybridEngine)
    ├── hybrid/
    │   ├── mod.rs                      ➜ Hybrid search module declaration
    │   ├── coordinator.rs              ➜ Hybrid search coordinator & fallback logic
    │   ├── fusion.rs                   ➜ BM25 saturation & cosine normalization, Weighted & RRF fusion
    │   ├── grouping.rs                 ➜ Post-fusion result grouping (Section & IdenticalText)
    │   └── ranking.rs                  ➜ Query feature analysis & dynamic alpha weight computation
    └── semantic/
        ├── mod.rs                      ➜ Semantic subsystem module declaration
        ├── chunker.rs                  ➜ Anchored semantic chunking & SHA256 ID generation
        ├── embedding.rs                ➜ GGUF model runtime interface & L2 normalization
        ├── engine.rs                   ➜ SemanticEngine sidecar orchestrator
        ├── manifest.rs                 ➜ Atomic JSON manifest versioning & Tantivy diff tracker
        ├── store.rs                    ➜ Pre-normalized vector database & BinaryHeap Top-K search
        └── types.rs                    ➜ Domain models & data transfer objects (DTOs)
```

### Module Breakdown

| Module | File Link | Primary Types / Functions | Purpose |
|--------|-----------|---------------------------|---------|
| **API Boundary** | [`src/api/hybrid_search.rs`](src/api/hybrid_search.rs) | `OtzariaHybridEngine`, `SearchRequest` | High-level thread-safe API wrapper for Flutter / FFI bridge |
| **Error Handling** | [`src/errors.rs`](src/errors.rs) | `SemanticSearchError`, `EmbeddingError`, `VectorStoreError` | Strongly-typed error hierarchy using `thiserror` |
| **Hybrid Coordinator** | [`src/hybrid/coordinator.rs`](src/hybrid/coordinator.rs) | `HybridCoordinator`, `HybridSearchParams` | Main search entry point orchestrating lexical & semantic paths |
| **Score Fusion** | [`src/hybrid/fusion.rs`](src/hybrid/fusion.rs) | `normalize_bm25_scores`, `fuse_weighted`, `fuse_rrf` | BM25 saturation ($x/(k+x)$) & cosine score mapping with clamp bounds |
| **Query Ranking** | [`src/hybrid/ranking.rs`](src/hybrid/ranking.rs) | `analyze_query`, `compute_alpha`, `QueryFeatures` | Dynamic $\alpha$ computation (short/exact $\to 0.7\text{--}0.9$, conceptual $\to 0.2\text{--}0.4$) |
| **Result Grouping** | [`src/hybrid/grouping.rs`](src/hybrid/grouping.rs) | `group_by_section`, `group_by_identical_text` | Section-level grouping and identical text line hash deduplication |
| **Domain Models** | [`src/semantic/types.rs`](src/semantic/types.rs) | `BookLine`, `SemanticChunk`, `FusedCandidate`, `HybridSearchResult` | All data transfer objects, candidate models, and filter definitions |
| **Text Chunker** | [`src/semantic/chunker.rs`](src/semantic/chunker.rs) | `Chunker`, `ChunkerConfig`, `compute_semantic_id` | Anchored chunking with context constrained to the anchor's section |
| **Embedding Runtime** | [`src/semantic/embedding.rs`](src/semantic/embedding.rs) | `EmbeddingRuntime`, `EmbeddingConfig`, `l2_normalize` | GGUF structure/checksum validation and runtime interface; real inference pending |
| **Vector Store** | [`src/semantic/store.rs`](src/semantic/store.rs) | `VectorStore`, `VectorStoreConfig`, `StoredVectorRecord` | Pre-normalized L2 dot-product search with bounded `BinaryHeap` Top-K |
| **Index Manifest** | [`src/semantic/manifest.rs`](src/semantic/manifest.rs) | `SemanticManifest`, `BookManifestEntry`, `validate` | Atomic JSON tracking (`.tmp` write + rename) & Tantivy incremental diffing |
| **Semantic Engine** | [`src/semantic/engine.rs`](src/semantic/engine.rs) | `SemanticEngine`, `SemanticConfig` | Master sidecar engine orchestrating chunking, embedding & storage |
| **Integration Test** | [`tests/hybrid_integration_test.rs`](tests/hybrid_integration_test.rs) | feature-gated integration tests | End-to-end public-API suite using the explicit mock backend |
| **CI Workflow** | [`.github/workflows/ci.yml`](.github/workflows/ci.yml) | `check-and-test` | Multi-platform GitHub Actions CI workflow (Linux, Windows, macOS) |

---

## 🚀 Roadmap & Implementation Status

```text
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         PROJECT IMPLEMENTATION ROADMAP                           │
├──────────────────────────────────────────────────────────────────────────────────┤
│ [✔] Phase 1: Core Architecture, Subsystem Isolation & Error Taxonomy            │
│ [✔] Phase 2: Anchored Chunker, Manifest Version Tracker & In-Memory Vector Store  │
│ [✔] Phase 3: Correct brute-force baseline (Pre-norm Dot Product + Min-Heap)      │
│ [✔] Phase 4: Correctness baseline, lifecycle contracts & complete filters        │
├──────────────────────────────────────────────────────────────────────────────────┤
│ [ ] Phase 5: Native GGUF Model Runtime Integration (Candle / GGUF Tensor)        │
│ [ ] Phase 6: Persistent Disk Vector Store (zvec / HNSW Index on Disk)            │
│ [ ] Phase 7: Flutter Rust Bridge (FRB) Bindings & Otzaria UI Integration         │
│ [ ] Phase 8: Background Streaming Indexer (StreamSink<IndexingProgress>)         │
│ [ ] Phase 9: Search Quality Benchmark & Evaluation Suite (Recall@K, MRR & nDCG) │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Detailed Next Steps

1. **Native GGUF Model Runtime (`candle-core`)**:
   - Replace deterministic feature hashing fallback with `candle-core` & `candle-transformers` GGUF tensor execution.
   - Implement Last-Token Pooling and SIMD/GPU hardware acceleration (AVX2 / Metal / DirectX).
2. **Persistent Disk Vector Store (`zvec-core` / HNSW)**:
   - Connect disk-backed HNSW vector storage in `semantic_db/zvec`.
   - Implement atomic `commit()` and Memory-Mapped File (`mmap`) reads for minimal memory footprint.
3. **Flutter Rust Bridge Integration (`flutter_rust_bridge`)**:
   - Generate Dart FFI bindings for `OtzariaHybridEngine` and `SearchRequest`.
   - Add Otzaria UI controls: Hybrid Search toggle, Search Mode picker, and Indexing Progress indicator.
4. **Background Batch Indexing Stream**:
   - Implement asynchronous background book indexing with real-time progress streaming (`StreamSink<IndexingProgress>`).
5. **Quality Evaluation Suite**:
   - Build a rabbinic test dataset to benchmark Recall@K, MRR, nDCG, and search latency (p50/p95/p99).

---

## ⚡ Performance Optimizations

- **Pre-Normalized Vectors**: Vectors are L2-normalized during batch insertion. Cosine similarity at search time reduces to a single vector dot product ($O(\text{dim})$), eliminating square root calculations during query execution.
- **Bounded BinaryHeap Top-K Selection**: Search queries utilize a bounded Min-Heap to collect candidate matches in $O(N \log k)$ time instead of performing full array clones and $O(N \log N)$ sorting over the entire database. `VectorMetadata` structs are cloned **only** for final selected candidates.
- **Single-Pass UTF-8 Truncation**: Text truncation uses `s.char_indices().nth(max_chars)` to inspect UTF-8 boundaries in a single pass without redundant string allocations.
- **Buffer-Reused Hex Encoding**: SHA256 hex ID generation formats byte digests directly into a pre-allocated `String` (`with_capacity(32)`), avoiding per-byte `format!()` heap allocations.
- **Pre-allocated Fusion HashMaps**: Fusion candidate maps use `HashMap::with_capacity(lexical.len() + semantic.len())` to eliminate dynamic map re-allocations during scoring.

---

## 🛠️ Building & Testing

### Prerequisites
- [Rust Toolchain](https://rustup.rs/) (Stable 2021 Edition)

### Commands

```bash
# Build release library
cargo build --release

# Run unit and integration tests (does not execute the benchmark target)
cargo test --lib --tests
cargo test --lib --tests --features mock-embedding

# Verify formatting
cargo fmt --check

# Run strict Clippy lints
cargo clippy --all-targets -- -D warnings
```

---

## 🤖 CI Pipeline

GitHub Actions runs for pushes to `main` and pull requests targeting `main`
across **three operating systems**:
- `ubuntu-latest`
- `windows-latest`
- `macos-latest`

The matrix checks formatting; default-feature check, Clippy and tests; mock-feature
Clippy and tests; rustdoc links; and a release build of all targets. Tests use
`--lib --tests`: `--all-targets` would execute the large benchmark rather than
merely compile it.

---

## 📖 Developer Documentation

For detailed architectural invariants, subsystem separation rules, and development guidelines, refer to:
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) — Comprehensive developer guide & status (Hebrew)
- [docs/CODE_MAP.md](docs/CODE_MAP.md) — Detailed code map and component descriptions

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Terms (Required Reading)

By submitting code, opening a Pull Request, or editing content in this repository, the contributor agrees to the following terms:

* **Assignment of Rights & Licensing Consent**: The contributor assigns to the project owner (Otzaria Project) full copyright and proprietary rights in their contribution, or grants an exclusive, worldwide, irrevocable, royalty-free, sublicensable license to use, modify, distribute, and license the contribution under any license, including the [Personal Use License](LICENSE).
* **Waiver of Claims**: The contributor waives any demand, royalty, or claim arising from the use of their contribution.
* **Declaration of Ownership**: The contributor declares that the contribution is their own original work and does not infringe third-party rights.
* **Credit Preservation**: Credit to the contributor (in git commit history / contributors list) is preserved.

---

## 📜 License

This repository is distributed under:

**Personal Use License 1.0 — Personal Use Only**

See the [LICENSE](LICENSE) file for complete details.

**Summary of Terms:**
- Personal, private use by an individual natural person only.
- Any public distribution, commercial use, integration into a public service/API/website, or use by an entity (company, nonprofit, institution) is prohibited without prior express written permission.
- Contact for licensing inquiries: **otzaria.1@gmail.com**

> **Note:** This license applies strictly to original project code. Third-party libraries, embedding models, and Otzaria texts remain under their respective original licenses.

---

## ✉️ Contact & Support

- **Email**: otzaria.1@gmail.com
- **Project Repository**: [https://github.com/Otzaria/otzaria-semantic-search](https://github.com/Otzaria/otzaria-semantic-search)
