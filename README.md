# Otzaria Semantic Search

[![CI](https://github.com/Otzaria/otzaria-semantic-search/actions/workflows/ci.yml/badge.svg)](https://github.com/Otzaria/otzaria-semantic-search/actions/workflows/ci.yml)
[![Personal Use License](https://img.shields.io/badge/license-Personal%20Use%201.0-blue.svg?style=flat)](LICENSE)

**Hybrid semantic & lexical search engine for Otzaria** — a local, offline, production-grade Rust library that merges BM25 lexical results with vector similarity search into a single ranked result set.

---

## 🗺️ Code Map & File Index

For a detailed structural guide and module breakdowns, see [docs/CODE_MAP.md](docs/CODE_MAP.md). Below is the quick direct file map:

```text
otzaria-semantic-search/
├── 📄 Cargo.toml                          ➜ Crate dependencies & build config
├── 📄 README.md                            ➜ Primary project overview & documentation
├── 📁 docs/
│   ├── 📄 CODE_MAP.md                      ➜ Comprehensive code map & architecture diagram
│   └── 📄 DEVELOPMENT.md                   ➜ Developer guide, architecture invariants & status
├── 📁 .github/workflows/
│   └── 📄 ci.yml                           ➜ Multi-platform CI pipeline (Linux, Windows, macOS)
├── 📁 tests/
│   └── 📄 hybrid_integration_test.rs       ➜ End-to-end integration test suite
└── 📁 src/
    ├── 📄 lib.rs                           ➜ Library root module exports
    ├── 📄 errors.rs                        ➜ Central strongly-typed error hierarchy (thiserror)
    ├── 📁 api/
    │   ├── 📄 mod.rs                       ➜ API module declaration
    │   └── 📄 hybrid_search.rs             ➜ Flutter / FFI bridge entry point (OtzariaHybridEngine)
    ├── 📁 hybrid/
    │   ├── 📄 mod.rs                       ➜ Hybrid search subsystem module exports
    │   ├── 📄 coordinator.rs               ➜ Main Hybrid Search Coordinator & fallback logic
    │   ├── 📄 fusion.rs                    ➜ Score normalization & fusion (Weighted & RRF)
    │   ├── 📄 grouping.rs                  ➜ Post-fusion result grouping (Section & IdenticalText)
    │   └── 📄 ranking.rs                   ➜ Query analysis & dynamic alpha weight computation
    └── 📁 semantic/
        ├── 📄 mod.rs                       ➜ Semantic subsystem module exports
        ├── 📄 chunker.rs                   ➜ Anchored semantic chunking & SHA256 IDs
        ├── 📄 embedding.rs                 ➜ GGUF Model Runtime Interface & normalization
        ├── 📄 engine.rs                    ➜ SemanticEngine sidecar orchestrator
        ├── 📄 manifest.rs                  ➜ Atomic JSON index versioning & diff tracker
        ├── 📄 store.rs                     ➜ VectorStore DB (Pre-normalized + BinaryHeap Top-K)
        └── 📄 types.rs                     ➜ Core domain types & data transfer objects
```

### Direct Module Links

| Layer | Module / File | Key Types & Functions | Description |
|-------|---------------|-----------------------|-------------|
| **API / FFI** | [`src/api/hybrid_search.rs`](src/api/hybrid_search.rs) | `OtzariaHybridEngine`, `SearchRequest` | High-level API exposed to Flutter via `flutter_rust_bridge` |
| **Errors** | [`src/errors.rs`](src/errors.rs) | `SemanticSearchError`, `EmbeddingError`, `VectorStoreError` | Strongly-typed error definitions for all subsystems |
| **Hybrid Coordinator** | [`src/hybrid/coordinator.rs`](src/hybrid/coordinator.rs) | `HybridCoordinator`, `HybridSearchParams` | Orchestrates BM25 + Semantic search, handles fallbacks |
| **Score Fusion** | [`src/hybrid/fusion.rs`](src/hybrid/fusion.rs) | `normalize_bm25_scores`, `fuse_weighted`, `fuse_rrf` | Normalizes BM25/Cosine scores and combines candidate lists |
| **Query Ranking** | [`src/hybrid/ranking.rs`](src/hybrid/ranking.rs) | `analyze_query`, `compute_alpha`, `QueryFeatures` | Analyzes query type to compute dynamic lexical vs. semantic weight ($\alpha$) |
| **Result Grouping** | [`src/hybrid/grouping.rs`](src/hybrid/grouping.rs) | `group_by_section`, `group_by_identical_text` | Groups results by book section or deduplicates identical line hashes |
| **Domain Types** | [`src/semantic/types.rs`](src/semantic/types.rs) | `BookLine`, `SemanticChunk`, `FusedCandidate`, `HybridSearchResult` | Core domain models, metadata structs, and search result containers |
| **Text Chunker** | [`src/semantic/chunker.rs`](src/semantic/chunker.rs) | `Chunker`, `ChunkerConfig`, `compute_semantic_id` | Anchored chunking with context windows and SHA256 chunk IDs |
| **Embedding Runtime** | [`src/semantic/embedding.rs`](src/semantic/embedding.rs) | `EmbeddingRuntime`, `EmbeddingConfig`, `l2_normalize` | Local quantized GGUF embedding model loader and runner |
| **Vector Database** | [`src/semantic/store.rs`](src/semantic/store.rs) | `VectorStore`, `VectorStoreConfig`, `StoredVectorRecord` | Pre-normalized vector database with $O(N \log k)$ BinaryHeap top-k search |
| **Index Manifest** | [`src/semantic/manifest.rs`](src/semantic/manifest.rs) | `SemanticManifest`, `BookManifestEntry`, `validate` | Atomic JSON index tracking for per-book versioning and Tantivy diffing |
| **Semantic Engine** | [`src/semantic/engine.rs`](src/semantic/engine.rs) | `SemanticEngine`, `SemanticConfig` | Master sidecar engine orchestrating chunking, embedding, and storage |
| **Integration Test** | [`tests/hybrid_integration_test.rs`](tests/hybrid_integration_test.rs) | `test_semantic_engine_and_hybrid_coordinator_end_to_end` | End-to-end integration test verifying full search pipeline |
| **CI Workflow** | [`.github/workflows/ci.yml`](.github/workflows/ci.yml) | `check-and-test` | Automated multi-platform build, lint, and test workflow |

---

## 🏗️ Architecture Overview

```text
                    ┌─────────────────────────────────────────────┐
                    │          HybridCoordinator                  │
                    │  ┌───────────┐    ┌──────────────────────┐  │
  Flutter / FFI ──▶ │  │  Query    │    │  Score Fusion        │  │
                    │  │  Analysis │──▶ │  (Weighted / RRF)    │  │
                    │  └───────────┘    └──────────┬───────────┘  │
                    │                              │              │
                    │  ┌─────────────┐   ┌─────────▼───────────┐  │
                    │  │ Lexical     │   │ Grouping & Ranking  │  │
                    │  │ (Tantivy)   │   │ (Section / Dedup)   │  │
                    │  │ BM25 cands  │   └─────────────────────┘  │
                    │  └─────────────┘                            │
                    └─────────────────────────────────────────────┘
                                         │
                    ┌────────────────────▼────────────────────────┐
                    │          SemanticEngine (Sidecar)           │
                    │  ┌──────────┐ ┌──────────┐ ┌────────────┐  │
                    │  │ Chunker  │ │Embedding │ │VectorStore │  │
                    │  │ (SHA256  │ │ Runtime  │ │(In-Memory  │  │
                    │  │  anchors)│ │ (GGUF)   │ │ + zvec)    │  │
                    │  └──────────┘ └──────────┘ └────────────┘  │
                    │  ┌─────────────────────────────────────┐   │
                    │  │       SemanticManifest (JSON)        │   │
                    │  │  Per-book versioning & diff tracking │   │
                    │  └─────────────────────────────────────┘   │
                    └────────────────────────────────────────────┘
```

## Key Design Principles

- **Non-destructive**: The semantic engine is a separate sidecar. Failures in the semantic path **never** affect the existing Tantivy/BM25 search. Graceful fallback to lexical-only mode.
- **Offline / Local**: No cloud APIs. Runs entirely on-device using a quantized GGUF embedding model.
- **Not RAG**: This is a **retrieval** engine, not a generation engine. It returns relevant source texts, never generated answers.
- **Modular**: Each component (Chunker, Embedding, VectorStore, Fusion, Grouping, Coordinator) is independently testable and replaceable.

## Building

```bash
cargo build
```

## Testing

```bash
cargo test --all-targets
```

## Linting

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
```

## CI Pipeline

GitHub Actions CI runs on every push to `main` and on all PRs, across **3 platforms** (Ubuntu, Windows, macOS):

1. Format check (`cargo fmt --check`)
2. Type check (`cargo check --all-targets`)
3. Lint (`cargo clippy --all-targets -- -D warnings`)
4. Tests (`cargo test --all-targets`)

## Performance Highlights

- **Pre-normalized vectors**: L2 normalization at insert time $\to$ cosine similarity reduces to a single dot product at search time
- **BinaryHeap top-k**: $O(N \log k)$ search instead of $O(N \log N)$ full-sort, with metadata cloned only for final $k$ results
- **Single-pass Unicode truncation**: `char_indices().nth()` instead of double iteration
- **Pre-allocated HashMaps**: `with_capacity()` on fusion maps to avoid rehashing

## Developer Documentation

For a detailed picture of the current state of the project — what is actually implemented, what is still a placeholder, and the architectural rules that must not be broken — see [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) (Hebrew) and [docs/CODE_MAP.md](docs/CODE_MAP.md).

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contribution you make is **greatly appreciated**.

If you have a suggestion that would improve the project, please fork the repository and create a pull request. You can also simply open an issue with the tag "enhancement".

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Terms (required reading)

By opening a Pull Request, submitting code, or editing existing content in this repository, the contributor agrees to the following terms — unless explicitly agreed otherwise in writing at the time the contribution is submitted:

* **Assignment of rights and licensing consent:** The contributor assigns to the project owner (Otzaria Project) the full copyright and proprietary rights in their contribution, or — at the very least — grants an exclusive, worldwide, irrevocable, royalty-free, sublicensable license to use, modify, distribute, and license the contribution under any license, including the [Personal Use License](LICENSE) and future licenses.
* **Waiver of claims:** The contributor waives any demand, royalty, or claim arising from the use of their contribution within the project or in related initiatives.
* **Declaration of ownership:** The contributor declares that the contribution is their own original work, that they are entitled to grant these rights in it, and that it does not infringe the rights of any third party.
* **Credit is preserved:** Credit to the contributor (in the contributors list / git history) is preserved; nothing above derogates from their moral right to be recognized as the author of the contribution.

A contribution that does not meet these terms will not be accepted into the repository.

## License

The original code of this project is distributed under:

**Personal Use License 1.0 — personal use only**

[Full terms in the LICENSE file](LICENSE)

**Summary of the license terms:**
- Personal, private use by an individual natural person only (download, run, study, and modify for one's own use)
- Any public use is prohibited — whether commercial or as part of free / open-source software — without prior express written permission
- Any distribution, publication, or integration into a product / service / website / API is prohibited
- Use by a company, nonprofit, institution, or any other entity is prohibited without prior express written permission
- The obligation to credit 'Otzaria' remains in force even after permission has been granted
- To request permission: otzaria.1@gmail.com

> **Note:** This license applies **only to our own original code**. Third-party components (crates, libraries, embedding models, fonts, and assets) remain under their original licenses, as do the texts and books distributed separately.

## Contact

Support: otzaria.1@gmail.com

Project link: [https://github.com/Otzaria/otzaria-semantic-search](https://github.com/Otzaria/otzaria-semantic-search)
