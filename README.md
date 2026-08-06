# Otzaria Hybrid Semantic Search Engine

[![CI](https://github.com/Otzaria/otzaria-semantic-search/actions/workflows/ci.yml/badge.svg)](https://github.com/Otzaria/otzaria-semantic-search/actions/workflows/ci.yml)
[![Personal Use License](https://img.shields.io/badge/license-Personal%20Use%201.0-blue.svg?style=flat)](LICENSE)
[![Rust 2021](https://img.shields.io/badge/rust-2021%20edition-orange.svg)](https://www.rust-lang.org)
[![Platform Support](https://img.shields.io/badge/platform-Linux%20%7C%20Windows%20%7C%20macOS-blue.svg)](#ci-pipeline)

**Otzaria Hybrid Semantic Search** is a correctness-focused Rust prototype for
bringing local semantic search to **Otzaria**, the open rabbinic digital library.
It is not production-ready yet.

The current crate implements chunking, lifecycle contracts, brute-force vector
search, result fusion, ranking profiles, caches, telemetry, a Rust API seam,
prototype persistence and packaging, and — behind the non-default `llama-backend`
feature — **real GGUF inference** against the Otzaria Qwen3 embedding model,
verified against committed golden reference vectors.

The artifact identity contract is in place: a package declares which corpus, Tantivy
schema, id scheme, model file, inference backend and store format it was built from, and
it is refused by name before a vector is read. Verification comes at two depths — full
hashing at install, metadata and presence at open — because re-hashing gigabytes at every
launch is not a check anyone keeps. A digest published outside the package is what
separates the official artifact from a self-consistent rebuild, and declining that anchor
has an explicit name ([docs/ARTIFACT_CONTRACT.md](docs/ARTIFACT_CONTRACT.md)).

That verified artifact now gets opened. `OfficialSemanticIndex` takes the token rather
than a path, opens the payload through a store type that has no write on it, checks the
manifest's counts against what the payload actually holds, and answers a query with the
`line_id` the caller hydrates from Tantivy. An installed artifact reopens after a restart
without indexing anything.

Still roadmap work: measuring that path at library scale, the builder that produces the
official artifact from Tantivy, and application integration.

> **Scope, in one line:** the official vector index is built ahead of time on a
> build machine and opened **read-only** on the user's device. The app does not
> index anything, there is no user overlay, and no query ever leaves the device.
> The binding definition is [docs/PRODUCT_CONTRACT.md](docs/PRODUCT_CONTRACT.md);
> the staged plan is [שלבי ויעדי התקדמות.md](שלבי%20ויעדי%20התקדמות.md).

**A default build has no embedding backend at all** and fails loudly
(`EmbeddingError::BackendUnavailable`) rather than producing vectors. That is
deliberate: the two backends are opt-in for different reasons — the deterministic
stand-in (`mock-embedding`) because it is *fake*, and real inference
(`llama-backend`) because it is *expensive*, compiling llama.cpp and ggml through
cmake on every downstream build.

---

## 💡 Key Design Principles

1. **Non-Destructive Sidecar Architecture**: The semantic engine operates as an independent sidecar database (`semantic_db`). It **never** mutates, alters, or replaces Otzaria's existing Tantivy lexical database.
2. **Prebuilt, Read-Only Official Index**: Library vectors are produced on a build machine and shipped as a static artifact. On the user's device the index is opened, verified and read — never rebuilt, and never extended with a writable user overlay.
3. **Graceful Fallback & Resilience**: If the semantic path fails (e.g. model missing, disk I/O error), the coordinator automatically falls back to lexical-only mode without crashing the app. The degradation is reported (`search_mode`, `fallback_reason`), never disguised as a semantic success.
4. **Offline & Private Target**: Runs entirely on-device — inference is local llama.cpp over a GGUF file. The crate performs no model download and no network telemetry; obtaining the model is the host application's job.
5. **Source Retrieval (Not RAG)**: Designed strictly for accurate source and text retrieval within Jewish literature. It returns verifiable textual sources, never hallucinated AI responses.
6. **Defensive Error Handling**: Known poisoned-lock and input edge cases use error propagation or graceful fallback; this is not an absolute panic-freedom guarantee.

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
│                                                   │ Weighted / RRF / adaptive fusion    ▼                 │
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
│   │    Anchored Chunker    │         │   Embedding Runtime    │         │  Vector Store (in-memory)  │   │
│   │ (same-section context  │───────▶ │  (GGUF validation +    │───────▶ │  Pre-normalized vectors +  │   │
│   │   + SHA256 Anchor IDs) │         │   llama.cpp inference) │         │  BinaryHeap Top-K, O(N·D)  │   │
│   └────────────────────────┘         └────────────────────────┘         └────────────────────────────┘   │
│                                                                                        ▲                 │
│   ┌────────────────────────────────────────────────────────────────────────────────────┴─────────────┐   │
│   │                                   SemanticManifest (JSON)                                       │   │
│   │                  Atomic versioning, model verification & Tantivy diff tracking                   │   │
│   └──────────────────────────────────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

Two things the diagram deliberately does not show:

- **The official read path** ([`src/semantic/official_index.rs`](src/semantic/official_index.rs))
  — the diagram is the *builder* path, which chunks, embeds and writes. The
  application's path holds no chunker and no manifest: it opens an installed artifact
  through `ReadOnlyZevcStore` and searches it. That store is **not** an ANN index and
  not the `zvec` library — opening reads every byte, verifies a checksum per record and
  loads every vector into a `HashMap`, and the search scans all of them. Whether a full
  scan and that open can meet the budget at library scale is S2b, and it is unmeasured.
- **The FFI boundary** — this crate stays an `rlib`. The native library, the
  `flutter_rust_bridge` bindings and Tantivy hydration live in
  `otzaria_search_engine`, which depends on this crate. Nothing in Otzaria reaches the
  read path yet; that is S5.

---

## 🗺️ Code Map & File Index

For detailed architectural guidelines and invariants, see [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) and [docs/CODE_MAP.md](docs/CODE_MAP.md).

```text
otzaria-semantic-search/
├── Cargo.toml                          ➜ Crate dependencies & build configuration
├── README.md                            ➜ Master project documentation & guide
├── docs/
│   ├── PRODUCT_CONTRACT.md             ➜ Binding scope definition (read-only index, no overlay)
│   ├── ARTIFACT_CONTRACT.md            ➜ Artifact identity fields & the pre-open verification gate
│   ├── MODEL_DISTRIBUTION.md           ➜ How the embedding model reaches the device
│   ├── CODE_MAP.md                     ➜ Comprehensive code map & module breakdown
│   └── DEVELOPMENT.md                  ➜ Developer guide, architecture invariants & status
├── .github/workflows/
│   └── ci.yml                          ➜ Multi-platform CI pipeline (Linux, Windows, macOS)
├── benches/
│   └── vector_search.rs                ➜ Vector-search latency benchmark (harness = false)
├── tests/
│   ├── artifact_contract.rs            ➜ Artifact identity & install gate, through the public API
│   ├── official_runtime.rs             ➜ Install → open → query an artifact, through the public API
│   ├── hybrid_integration_test.rs      ➜ End-to-end integration test suite
│   └── production_backend_gate.rs      ➜ Proves a default build refuses to embed
└── src/
    ├── lib.rs                          ➜ Library root, module exports & product contract
    ├── main.rs                         ➜ Development CLI (audit / smoke runs)
    ├── errors.rs                       ➜ Strongly-typed error hierarchy (thiserror)
    ├── api/
    │   ├── mod.rs                      ➜ API module declaration
    │   └── hybrid_search.rs            ➜ Flutter / FFI bridge entry point (OtzariaHybridEngine)
    ├── benchmark/
    │   └── mod.rs                      ➜ Query sets, timing & percentile aggregation
    ├── config/
    │   ├── profiles.rs                 ➜ Fast/Balanced/Best profiles & fusion strategy
    │   └── feature_flags.rs            ➜ Per-run overrides onto a RankingProfile
    ├── distribution/
    │   ├── package.rs                  ➜ Index package manifest & SHA-256 payload checksums
    │   └── importer.rs                 ➜ Staged install, with recovery from an interrupted swap
    ├── hybrid/
    │   ├── mod.rs                      ➜ Hybrid search module declaration
    │   ├── coordinator.rs              ➜ Hybrid search coordinator & fallback logic
    │   ├── fusion.rs                   ➜ BM25 saturation & cosine normalization, Weighted & RRF fusion
    │   ├── grouping.rs                 ➜ Post-fusion result grouping (Section & IdenticalText)
    │   ├── ranking.rs                  ➜ Query feature analysis & dynamic alpha weight computation
    │   ├── metadata_ranker.rs          ➜ Facet-derived ranking bonuses
    │   ├── hebrew_normalizer.rs        ➜ Nikud/taamim stripping & query language detection
    │   └── cache.rs                    ➜ Generation-invalidated query result cache
    ├── semantic/
    │   ├── mod.rs                      ➜ Semantic subsystem module declaration
    │   ├── chunker.rs                  ➜ Anchored semantic chunking & SHA256 ID generation
    │   ├── embedding.rs                ➜ GGUF validation, batching & L2 normalization
    │   ├── embedding_cache.rs          ➜ LRU cache of recently embedded texts
    │   ├── backend.rs                  ➜ EmbeddingBackend contract & backend selection
    │   ├── llama_backend.rs            ➜ Real llama.cpp inference (feature `llama-backend`)
    │   ├── engine.rs                   ➜ SemanticEngine: the build-side orchestrator
    │   ├── official_index.rs           ➜ The application's read path over a verified artifact
    │   ├── manifest.rs                 ➜ Atomic JSON manifest versioning & Tantivy diff tracker
    │   ├── store.rs                    ➜ Pre-normalized vector database & BinaryHeap Top-K search
    │   ├── store_backend.rs            ➜ Two contracts: the read side the runtime gets, the write side a builder gets
    │   ├── zevc_store.rs               ➜ The payload format: a writable opener and a read-only one (full scan, not ANN)
    │   ├── versioning.rs               ➜ Artifact identity (corpus/model/store) & typed rejection
    │   └── types.rs                    ➜ Domain models & data transfer objects (DTOs)
    └── telemetry/
        └── mod.rs                      ➜ In-process search metrics aggregation (no network)
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
| **Embedding Runtime** | [`src/semantic/embedding.rs`](src/semantic/embedding.rs) | `EmbeddingRuntime`, `EmbeddingConfig`, `l2_normalize` | GGUF structure/checksum validation; the primary choke point that normalizes and validates every vector |
| **Backend Contract** | [`src/semantic/backend.rs`](src/semantic/backend.rs) | `EmbeddingBackend`, `Pooling`, `select_backend` | `Send + Sync` trait every backend implements; backends return **raw** vectors |
| **Real Inference** | [`src/semantic/llama_backend.rs`](src/semantic/llama_backend.rs) | `LlamaCppBackend`, `ContextPool`, `truncate_with_eos` | llama.cpp GGUF inference behind `--features llama-backend`: Qwen2-BPE tokenizer, EOS appended, last-token pooling, real multi-sequence batching |
| **Vector Store** | [`src/semantic/store.rs`](src/semantic/store.rs) | `VectorStore`, `VectorStoreConfig`, `StoredVectorRecord` | Pre-normalized L2 dot-product search with bounded `BinaryHeap` Top-K. **Volatile**, and what the builder path opens by default |
| **Store Contract** | [`src/semantic/store_backend.rs`](src/semantic/store_backend.rs) | `VectorSearchBackend`, `VectorStoreBackend` | Split in two on purpose: the runtime is handed the read side and so has no `insert` to call; the write side is what a builder gets |
| **Payload Format** | [`src/semantic/zevc_store.rs`](src/semantic/zevc_store.rs) | `ZevcStore`, `ReadOnlyZevcStore` | Checksummed disk snapshots, opened writable by a builder and read-only by the runtime. A checksum **per record** is what catches a same-length edit. **Full scan, not ANN, not `zvec`** |
| **Official Read Path** | [`src/semantic/official_index.rs`](src/semantic/official_index.rs) | `OfficialSemanticIndex`, `LocalModel` | Opens a `VerifiedPackage` — never a path — checks the manifest's counts against the payload's content, and refuses every build-side operation by name |
| **Artifact Identity** | [`src/semantic/versioning.rs`](src/semantic/versioning.rs) | `IndexVersion`, `IdentityField`, `verify_matches` | Corpus, Tantivy schema, id scheme, model file, backend and store format an artifact declares. Every field compared, all mismatches named |
| **Index Manifest** | [`src/semantic/manifest.rs`](src/semantic/manifest.rs) | `SemanticManifest`, `BookManifestEntry`, `validate` | Atomic JSON tracking (`.tmp` write + rename) & Tantivy incremental diffing |
| **Semantic Engine** | [`src/semantic/engine.rs`](src/semantic/engine.rs) | `SemanticEngine`, `SemanticConfig` | Master sidecar engine orchestrating chunking, embedding & storage |
| **Embedding Cache** | [`src/semantic/embedding_cache.rs`](src/semantic/embedding_cache.rs) | `EmbeddingCache` | LRU cache over recently embedded query texts |
| **Search Profiles** | [`src/config/profiles.rs`](src/config/profiles.rs) | `SearchProfile`, `RankingProfile`, `FusionStrategy` | Fast/Balanced/Best presets and the weighted / RRF / adaptive fusion choice |
| **Feature Flags** | [`src/config/feature_flags.rs`](src/config/feature_flags.rs) | `FeatureFlags::apply` | Per-run overrides onto a profile, without a second source of defaults |
| **Query Cache** | [`src/hybrid/cache.rs`](src/hybrid/cache.rs) | `QueryCache`, `QueryCacheStats` | Result cache keyed by query parameters, invalidated by generation |
| **Metadata Ranking** | [`src/hybrid/metadata_ranker.rs`](src/hybrid/metadata_ranker.rs) | `MetadataRanker`, `MetadataSignal` | Small facet-derived bonuses (primary source, era, category) |
| **Hebrew Normalizer** | [`src/hybrid/hebrew_normalizer.rs`](src/hybrid/hebrew_normalizer.rs) | `HebrewNormalizer`, `QueryLanguage` | Nikud/taamim stripping and geresh normalization before embedding |
| **Telemetry** | [`src/telemetry/mod.rs`](src/telemetry/mod.rs) | `TelemetryCollector`, `SearchTelemetry` | In-process counters only — nothing is transmitted anywhere |
| **Index Package** | [`src/distribution/package.rs`](src/distribution/package.rs) | `IndexPackage`, `ArtifactExpectation`, `VerifiedPackage`, `VerificationDepth` | Metadata plus a SHA-256 per payload, and the artifact digest that a published value can be compared against. `verify_for_install` hashes everything; `verify_for_open` does not, and the token records which ran |
| **Package Install** | [`src/distribution/importer.rs`](src/distribution/importer.rs) | `IndexImporter`, `recover_interrupted_install` | Verify the source, copy to staging, verify the copy, swap. The swap is two renames with a window in between, so the intermediate names are deterministic and recovery is a documented step |
| **Benchmark Harness** | [`src/benchmark/mod.rs`](src/benchmark/mod.rs) | `measure`, `aggregate`, `QuerySet` | Timing and percentile helpers. A measurement tool, **not** a relevance dataset |
| **Integration Test** | [`tests/hybrid_integration_test.rs`](tests/hybrid_integration_test.rs) | feature-gated integration tests | End-to-end public-API suite using the explicit mock backend |
| **Official Runtime Test** | [`tests/official_runtime.rs`](tests/official_runtime.rs) | feature-gated integration tests | Builds an artifact the way the packer will, installs it, opens it, and asserts a query returns the `line_id` it was built from — plus that every build-side call is refused and the artifact is never written to |
| **CI Workflow** | [`.github/workflows/ci.yml`](.github/workflows/ci.yml) | `check-and-test` | Multi-platform GitHub Actions CI workflow (Linux, Windows, macOS) |

---

## 🚀 Roadmap & Implementation Status

The stages below are the plan of record from
[שלבי ויעדי התקדמות.md](שלבי%20ויעדי%20התקדמות.md). S4–S8 land in
`otzaria_search_engine` and `otzaria`, not here.

```text
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         PROJECT IMPLEMENTATION ROADMAP                           │
├──────────────────────────────────────────────────────────────────────────────────┤
│ [✔] Core architecture, subsystem isolation & error taxonomy                      │
│ [✔] Anchored chunker, manifest version tracker & in-memory vector store          │
│ [✔] Correct brute-force baseline (pre-norm dot product + min-heap)               │
│ [✔] Correctness baseline, lifecycle contracts & complete filters                 │
│ [✔] Real GGUF inference (llama.cpp) verified against golden vectors              │
│ [✔] Ranking profiles, fusion strategies, caches, telemetry & packaging prototype │
├──────────────────────────────────────────────────────────────────────────────────┤
│ [✔] S0  Product contract alignment (this section, and the docs around it)        │
│ [ ] S1  Representation quality & dimension/precision decision                    │
│ [✔] S2a Read-only runtime path: the artifact's reader, read/write store split    │
│ [ ] S2b Scale: cold-open, latency, RSS and disk at 1M/6M — then the ANN decision │
│ [✔] S3  Artifact contract: identity, two depths, recoverable install, reader     │
│ [ ] S4  Builder that reads the final Tantivy index (otzaria_search_engine)       │
│ [ ] S5  Repin, open/install API, explicit statuses, FFI (otzaria_search_engine)  │
│ [ ] S6  Artifact & model management in the app (otzaria)                         │
│ [ ] S7  RetrievalMode in BLoC and UI (otzaria)                                   │
│ [ ] S8  Release gates: platform matrix, real model, resource budgets             │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Detailed Next Steps

1. **Representation quality & dimensions (S1)**:
   - Measure `line` versus `title + reference + line` versus neighbour context on a labelled rabbinic query set.
   - Choose 1024/512/256/128 dimensions and f32/f16/int8 on measured Recall@K, MRR and nDCG — the size arithmetic (~23.1 GiB at f32/1024 for 6.1M lines) is why this matters.
   - Freeze `embedding_text_version`, dimension, precision, `max_tokens`, pooling and normalization into the index identity.
2. **Scale measurement (S2b)** — the read-only path exists; what it costs is unknown:
   - Run `ZevcStore` as a correctness baseline at 1M and 6M records, and measure cold-open, p50/p95/p99, peak RSS and disk. Opening currently reads every byte, verifies a checksum per record and holds every vector in RAM.
   - Move to a real on-disk ANN only if the measurement says a full scan cannot meet the budget — not because "ANN" sounds faster. `VectorSearchBackend` is the seam either answer slots into.
3. **Official artifact contract (S3) and its reader (S2a)** — identity, two verification
   depths, the published-digest anchor, a recoverable install, and a runtime path that
   opens the verified token landed; see
   [docs/ARTIFACT_CONTRACT.md](docs/ARTIFACT_CONTRACT.md). What is left:
   - Publish the artifact digest (and sign it): the check exists, the anchor does not (S6).
   - Decide whether the distributed artifact is a single archive rather than a directory (packer-side, S4).
   - Measure open and install against a budget on a representative artifact (S2b/S8).
4. **Quality evaluation suite**:
   - Build the rabbinic relevance dataset behind S1 and report against BM25-only and semantic-only baselines.

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
- For `--features llama-backend` only: **cmake** and a C++ toolchain. `llama-cpp-2`
  builds llama.cpp and ggml from source (~1–2 minutes cold).

### Feature matrix

| Build | Backend | `EmbeddingRuntime::load` |
|---|---|---|
| default | none | `Err(BackendUnavailable)` — a release build cannot serve fake vectors |
| `--features mock-embedding` | deterministic hash stand-in | `Ok` — **not a semantic model**, development and testing only |
| `--features llama-backend` | real llama.cpp GGUF inference | `Ok` |
| both | real inference wins | `Ok`, or the real backend's error — never a silent fall-through to the stand-in |

### Commands

```bash
# Build release library
cargo build --release

# Run unit and integration tests (never `--all-targets`: that selects the bench
# target, overriding its `test = false`, and runs a 200k x 1024 workload unoptimized)
cargo test --lib --tests
cargo test --lib --tests --features mock-embedding
cargo test --lib --tests --features llama-backend
cargo test --lib --tests --features mock-embedding,llama-backend

# Verify formatting
cargo fmt --check

# Run strict Clippy lints (run for each feature combination above)
cargo clippy --all-targets -- -D warnings
```

### Testing against the real model

Tests that need the 396 MB GGUF are `#[ignore]`d and **skip loudly** when the model
is absent, so CI stays green without it. To run them, point `OTZARIA_TEST_MODEL` at
the file:

```bash
OTZARIA_TEST_MODEL=/abs/path/Otzaria-Embedding-V1-Flash-0.6B-Q4_K_M.gguf \
  cargo test --lib --features llama-backend -- --ignored --nocapture
```

These assert **exact `token_ids` equality** against the committed golden vectors in
[`tests/data/golden_vectors.json`](tests/data/golden_vectors.json), then cosine and
per-component agreement. The token-id assertion is the primary gate, not the cosine
one — a wrongly prepended BOS scores *higher* than a legitimate independent
reference, so no cosine threshold can separate them. See
[`docs/P2_REFERENCE_VECTORS.md`](docs/P2_REFERENCE_VECTORS.md) for the measurements
and [`tools/README.md`](tools/README.md) for regenerating the goldens.

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

Two further jobs: an **inference backend** job that builds and tests
`llama-backend` on Linux and macOS, and a **golden vectors** job that runs the
real-model parity gate. The golden job needs the `OTZARIA_HF_TOKEN` secret; when
the secret is absent it fails loudly rather than reporting a skip as a pass. That
gate is a reason the model's distribution route matters — see
[docs/MODEL_DISTRIBUTION.md](docs/MODEL_DISTRIBUTION.md).

---

## 📖 Developer Documentation

For detailed architectural invariants, subsystem separation rules, and development guidelines, refer to:
- [docs/PRODUCT_CONTRACT.md](docs/PRODUCT_CONTRACT.md) — **Binding scope definition** (Hebrew); outranks every other document here
- [docs/ARTIFACT_CONTRACT.md](docs/ARTIFACT_CONTRACT.md) — Artifact identity fields, verification order and what is not yet enforced (Hebrew)
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) — Comprehensive developer guide & status (Hebrew)
- [docs/CODE_MAP.md](docs/CODE_MAP.md) — Detailed code map and component descriptions
- [docs/MODEL_DISTRIBUTION.md](docs/MODEL_DISTRIBUTION.md) — How the embedding model reaches the device (Hebrew)
- [שלבי ויעדי התקדמות.md](שלבי%20ויעדי%20התקדמות.md) — Staged plan S0–S8 across the three repositories (Hebrew)

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
