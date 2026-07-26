# Otzaria Semantic Search

[![CI](https://github.com/Otzaria/otzaria-semantic-search/actions/workflows/ci.yml/badge.svg)](https://github.com/Otzaria/otzaria-semantic-search/actions/workflows/ci.yml)
[![Personal Use License](https://img.shields.io/badge/license-Personal%20Use%201.0-blue.svg?style=flat)](LICENSE)

**Hybrid semantic & lexical search engine for Otzaria** — a local, offline, production-grade Rust library that merges BM25 lexical results with vector similarity search into a single ranked result set.

## Architecture

```
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

## Module Overview

| Module | Path | Description |
|--------|------|-------------|
| **Types** | `src/semantic/types.rs` | All domain types: `BookLine`, `SemanticChunk`, `FusedCandidate`, `HybridSearchResult`, etc. |
| **Chunker** | `src/semantic/chunker.rs` | Anchored semantic chunking with context windows, SHA256 identity hashing |
| **Embedding** | `src/semantic/embedding.rs` | GGUF model runtime interface (deterministic fallback for testing) |
| **VectorStore** | `src/semantic/store.rs` | Pre-normalized vector storage with BinaryHeap top-k search |
| **Manifest** | `src/semantic/manifest.rs` | Atomic JSON manifest for per-book version tracking & diff detection |
| **Engine** | `src/semantic/engine.rs` | Orchestrates chunking → embedding → storage → manifest lifecycle |
| **Fusion** | `src/hybrid/fusion.rs` | BM25 saturation normalization, cosine normalization, weighted & RRF fusion |
| **Ranking** | `src/hybrid/ranking.rs` | Query analysis, dynamic alpha computation (lexical vs. semantic weight) |
| **Grouping** | `src/hybrid/grouping.rs` | Post-fusion grouping by section or identical text (line hash dedup) |
| **Coordinator** | `src/hybrid/coordinator.rs` | Top-level hybrid search orchestrator with fallback |
| **API** | `src/api/hybrid_search.rs` | Clean Flutter/FFI API surface (`SearchRequest`, `OtzariaHybridEngine`) |
| **Errors** | `src/errors.rs` | Strongly-typed `thiserror` error hierarchy per subsystem |

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

## CI

GitHub Actions CI runs on every push to `main` and on all PRs, across **3 platforms** (Ubuntu, Windows, macOS):

1. Format check (`cargo fmt --check`)
2. Type check (`cargo check --all-targets`)
3. Lint (`cargo clippy --all-targets -- -D warnings`)
4. Tests (`cargo test --all-targets`)

## Performance Highlights

- **Pre-normalized vectors**: L2 normalization at insert time → cosine similarity reduces to a single dot product at search time
- **BinaryHeap top-k**: O(N log k) search instead of O(N log N) full-sort, with metadata cloned only for final k results
- **Single-pass Unicode truncation**: `char_indices().nth()` instead of double iteration
- **Pre-allocated HashMaps**: `with_capacity()` on fusion maps to avoid rehashing

## Developer Documentation

For a detailed picture of the current state of the project — what is actually
implemented, what is still a placeholder, and the architectural rules that must
not be broken — see [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) (Hebrew).

## Contributing

Contributions are what make the open-source community such an amazing place to
learn, inspire, and create. Any contribution you make is **greatly appreciated**.

If you have a suggestion that would improve the project, please fork the
repository and create a pull request. You can also simply open an issue with the
tag "enhancement".

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Terms (required reading)

By opening a Pull Request, submitting code, or editing existing content in this
repository, the contributor agrees to the following terms — unless explicitly
agreed otherwise in writing at the time the contribution is submitted:

* **Assignment of rights and licensing consent:** The contributor assigns to the
  project owner (Otzaria Project) the full copyright and proprietary rights in
  their contribution, or — at the very least — grants an exclusive, worldwide,
  irrevocable, royalty-free, sublicensable license to use, modify, distribute,
  and license the contribution under any license, including the
  [Personal Use License](LICENSE) and future licenses.
* **Waiver of claims:** The contributor waives any demand, royalty, or claim
  arising from the use of their contribution within the project or in related
  initiatives.
* **Declaration of ownership:** The contributor declares that the contribution is
  their own original work, that they are entitled to grant these rights in it,
  and that it does not infringe the rights of any third party.
* **Credit is preserved:** Credit to the contributor (in the contributors list /
  git history) is preserved; nothing above derogates from their moral right to be
  recognized as the author of the contribution.

A contribution that does not meet these terms will not be accepted into the
repository.

## License

The original code of this project is distributed under:

**Personal Use License 1.0 — personal use only**

[Full terms in the LICENSE file](LICENSE)

**Summary of the license terms:**
- Personal, private use by an individual natural person only (download, run,
  study, and modify for one's own use)
- Any public use is prohibited — whether commercial or as part of free /
  open-source software — without prior express written permission
- Any distribution, publication, or integration into a product / service /
  website / API is prohibited
- Use by a company, nonprofit, institution, or any other entity is prohibited
  without prior express written permission
- The obligation to credit 'Otzaria' remains in force even after permission has
  been granted
- To request permission: otzaria.1@gmail.com

> **Note:** This license applies **only to our own original code**. Third-party
> components (crates, libraries, embedding models, fonts, and assets) remain
> under their original licenses, as do the texts and books distributed
> separately.

## Contact

Support: otzaria.1@gmail.com

Project link: [https://github.com/Otzaria/otzaria-semantic-search](https://github.com/Otzaria/otzaria-semantic-search)
