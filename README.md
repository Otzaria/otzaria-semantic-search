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
prototype persistence and packaging, and — behind the non-default `onnx-backend`
feature — **real ONNX inference** through ONNX Runtime against the Meivin Round 2
embedding model, verified against committed golden reference vectors. ONNX is the only
backend: GGUF and llama.cpp support was removed, and the last commit that has it is
`62f0c44`.

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
(`onnx-backend`) because a build able to embed has to be asked for: it brings in `ort`
and the tokenizer, and loads ONNX Runtime, a shared library, when a model loads.

---

## 💡 Key Design Principles

1. **Non-Destructive Sidecar Architecture**: The semantic engine operates as an independent sidecar database (`semantic_db`). It **never** mutates, alters, or replaces Otzaria's existing Tantivy lexical database.
2. **Prebuilt, Read-Only Official Index**: Library vectors are produced on a build machine and shipped as a static artifact. On the user's device the index is opened, verified and read — never rebuilt, and never extended with a writable user overlay.
3. **Graceful Fallback & Resilience**: If the semantic path fails (e.g. model missing, disk I/O error), the coordinator automatically falls back to lexical-only mode without crashing the app. The degradation is reported (`search_mode`, `fallback_reason`), never disguised as a semantic success.
4. **Offline & Private Target**: Runs entirely on-device — inference is local ONNX Runtime over an ONNX graph. The crate performs no model download and no network telemetry; obtaining the model is the host application's job.
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
│   │ (same-section context  │───────▶ │ (ONNX package check +  │───────▶ │  Pre-normalized vectors +  │   │
│   │   + SHA256 Anchor IDs) │         │  ONNX Runtime)         │         │  BinaryHeap Top-K, O(N·D)  │   │
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
  application's path holds no chunker and no manifest: it opens an installed vector set —
  int8 segments addressed by the text each vector was embedded from, mapped rather than
  read — and scans every vector exactly, in integers, the same score on every CPU. A scan
  returns keys and the books and lines they were built at; the host's `CandidateResolver`
  ties them to its live lines. At 6.0M slots the set opens in 3.8 ms and a warm scan takes
  69 ms on one thread, 17 ms on ten — no ANN ([`docs/ARTIFACT_CONTRACT.md`](docs/ARTIFACT_CONTRACT.md)).
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
│   ├── artifact_builder.rs             ➜ S4b's gate: build through the CLI, reproducible, and what it builds installs and answers
│   ├── official_runtime.rs             ➜ Build → install → open → query a vector set through a resolver
│   ├── vector_set_scale.rs             ➜ The set at library scale, measured (#[ignore])
│   ├── hybrid_integration_test.rs      ➜ End-to-end integration test suite
│   └── production_backend_gate.rs      ➜ Proves a default build refuses to embed
└── src/
    ├── lib.rs                          ➜ Library root, module exports & product contract
    ├── main.rs                         ➜ Development CLI (audit / smoke) + the build commands: build / export-plan / embed-shard
    ├── errors.rs                       ➜ Strongly-typed error hierarchy (thiserror)
    ├── cancellation.rs                 ➜ CancellationToken: abandoning a query nobody waits for
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
    │   ├── importer.rs                 ➜ Staged install, with recovery from an interrupted swap
    │   ├── builder.rs                  ➜ Corpus + model → a base segment, its package and release manifest
    │   ├── corpus.rs                   ➜ The port onto the lexical index, and a two-file transcription of one
    │   └── shard.rs                    ➜ The same build cut in two: export a plan, embed shards, check them
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
    │   ├── chunk_key.rs                ➜ ChunkKey: SHA-256 of the embedded text, the address of a vector
    │   ├── chunker.rs                  ➜ Anchored semantic chunking; embedded_text / chunk_keys over a window
    │   ├── embedding.rs                ➜ Configuration checks, batching & L2 normalization
    │   ├── embedding_cache.rs          ➜ LRU cache of recently embedded texts
    │   ├── backend.rs                  ➜ EmbeddingBackend contract & backend selection
    │   ├── model_package.rs            ➜ The ONNX package: what a model path names, its validation & checksum
    │   ├── onnx_backend.rs             ➜ Real ONNX Runtime inference (feature `onnx-backend`), the only backend
    │   ├── engine.rs                   ➜ SemanticEngine: the build-side orchestrator; its indexing API is a prototype scaffold
    │   ├── official_index.rs           ➜ The application's read path: an installed set and a model
    │   ├── resolve.rs                  ➜ VectorHit, and the CandidateResolver port onto live lines
    │   ├── oxv/                        ➜ The segment format, its codecs and the exact int8 scan
    │   ├── segment_set/                ➜ The installed set: install, apply, compact, recover, GC, scrub
    │   ├── manifest.rs                 ➜ Atomic JSON manifest versioning & Tantivy diff tracker
    │   ├── store.rs                    ➜ The development path's in-memory store & BinaryHeap Top-K search
    │   ├── store_backend.rs            ➜ The development path's two contracts: search, and indexing's mutations
    │   ├── versioning.rs               ➜ Identity (text / model family / store) & typed rejection
    │   └── types.rs                    ➜ Domain models & data transfer objects (DTOs)
    └── telemetry/
        └── mod.rs                      ➜ In-process search metrics aggregation (no network)
```

### Module Breakdown

| Module | File Link | Primary Types / Functions | Purpose |
|--------|-----------|---------------------------|---------|
| **API Boundary** | [`src/api/hybrid_search.rs`](src/api/hybrid_search.rs) | `OtzariaHybridEngine`, `SearchRequest` | High-level thread-safe API wrapper for Flutter / FFI bridge |
| **Error Handling** | [`src/errors.rs`](src/errors.rs) | `SemanticSearchError`, `EmbeddingError`, `VectorStoreError` | Strongly-typed error hierarchy using `thiserror` |
| **Query Cancellation** | [`src/cancellation.rs`](src/cancellation.rs) | `CancellationToken`, `SCAN_CHECK_INTERVAL` | A search per keystroke abandons the queries the next keystroke made obsolete: checked before embedding, after it, every 1,024 records of the scan, and around fusion. `SemanticSearchError::Cancelled`, never a lexical fallback, and nothing cached or counted |
| **Hybrid Coordinator** | [`src/hybrid/coordinator.rs`](src/hybrid/coordinator.rs) | `HybridCoordinator`, `HybridSearchParams` | Main search entry point orchestrating lexical & semantic paths |
| **Score Fusion** | [`src/hybrid/fusion.rs`](src/hybrid/fusion.rs) | `normalize_bm25_scores`, `fuse_weighted`, `fuse_rrf` | BM25 saturation ($x/(k+x)$) & cosine score mapping with clamp bounds |
| **Query Ranking** | [`src/hybrid/ranking.rs`](src/hybrid/ranking.rs) | `analyze_query`, `compute_alpha`, `QueryFeatures` | Dynamic $\alpha$ computation (short/exact $\to 0.7\text{--}0.9$, conceptual $\to 0.2\text{--}0.4$) |
| **Result Grouping** | [`src/hybrid/grouping.rs`](src/hybrid/grouping.rs) | `group_by_section`, `group_by_identical_text` | Section-level grouping and identical text line hash deduplication |
| **Domain Models** | [`src/semantic/types.rs`](src/semantic/types.rs) | `BookLine`, `SemanticChunk`, `FusedCandidate`, `HybridSearchResult` | All data transfer objects, candidate models, and filter definitions |
| **Text Chunker** | [`src/semantic/chunker.rs`](src/semantic/chunker.rs) | `Chunker`, `ChunkerConfig`, `compute_semantic_id` | Anchored chunking with context constrained to the anchor's section |
| **Embedding Runtime** | [`src/semantic/embedding.rs`](src/semantic/embedding.rs) | `EmbeddingRuntime`, `EmbeddingConfig`, `l2_normalize` | The configuration's checks — a model path that names no ONNX graph is refused here — and the primary choke point that normalizes and validates every vector |
| **Backend Contract** | [`src/semantic/backend.rs`](src/semantic/backend.rs) | `EmbeddingBackend`, `Pooling`, `select_backend` | `Send + Sync` trait every backend implements; backends return **raw** vectors |
| **Model Package** | [`src/semantic/model_package.rs`](src/semantic/model_package.rs) | `validate_model`, `ensure_onnx_model_path`, `OnnxPackage` | What a model path names — an ONNX graph, or a refusal — and the package around it: the graph, `tokenizer.json` and any external data, each validated, and the one checksum that names them all |
| **Real Inference** | [`src/semantic/onnx_backend.rs`](src/semantic/onnx_backend.rs) | `OnnxBackend`, `OnnxBackendConfig` | ONNX Runtime inference behind `--features onnx-backend`, the only backend: the package's own `tokenizer.json`, one text per run, pooling in the graph; ONNX Runtime is a shared library loaded at run time, never linked |
| **Vector Store** | [`src/semantic/store.rs`](src/semantic/store.rs) | `VectorStore`, `VectorStoreConfig`, `StoredVectorRecord` | Pre-normalized L2 dot-product search with bounded `BinaryHeap` Top-K. **Volatile**: the development path's store |
| **Store Contract** | [`src/semantic/store_backend.rs`](src/semantic/store_backend.rs) | `VectorSearchBackend`, `VectorStoreBackend` | The development path's store, split in two on purpose: a search is handed the read side and so has no `insert` to call |
| **Chunk Keys** | [`src/semantic/chunk_key.rs`](src/semantic/chunk_key.rs) | `ChunkKey`, `KEY_VERSION`, `LineRef` | A vector's address: the first 16 bytes of the SHA-256 of the text it was embedded from. `column_value` is what the host's index stores per line |
| **Segment Format** | [`src/semantic/oxv/`](src/semantic/oxv/mod.rs) | `SegmentBuilder`, `Segment`, `Codec`, `scan_segments` | `.oxv` segments: int8 vectors with a scale each (`i8-sym-vec`, the default; `i8-sym-dim` and `f32` too), keys and hints per slot, records apart from vectors; a streaming writer, a mapped reader checked by CRC, and an exact integer scan — scalar, AVX2 and NEON, bit-identical |
| **Vector Sets** | [`src/semantic/segment_set/`](src/semantic/segment_set/mod.rs) | `SegmentSet`, `install_package`, `compact`, `scrub`, `ReleaseManifest` | The installed set: a base and deltas as generations behind `CURRENT`/`PREVIOUS`, every crash point recoverable, compaction, garbage collection that waits for Windows to unmap |
| **Resolver Port** | [`src/semantic/resolve.rs`](src/semantic/resolve.rs) | `VectorHit`, `CandidateResolver`, `ResolvedLine`, `NoResolver` | What a scan returns — a key and where it was built — and the host's port that ties it to the lines its live index holds now |
| **Official Read Path** | [`src/semantic/official_index.rs`](src/semantic/official_index.rs) | `OfficialSemanticIndex`, `OfficialIndexConfig`, `LocalModel` | Recovers, opens the set, loads the model, holds the set's identity to the installation's; returns hits, reloads a new generation in place, and refuses every build-side operation by name |
| **Identity** | [`src/semantic/versioning.rs`](src/semantic/versioning.rs) | `IndexVersion`, `IdentityField`, `VectorProvenance` | The line recipe and key version, the model family with the query packages it accepts, and the store format. Every field compared — the packages by membership — all mismatches named; provenance recorded, never compared |
| **Index Manifest** | [`src/semantic/manifest.rs`](src/semantic/manifest.rs) | `SemanticManifest`, `BookManifestEntry`, `validate` | Atomic JSON tracking (`.tmp` write + rename) & Tantivy incremental diffing |
| **Semantic Engine** | [`src/semantic/engine.rs`](src/semantic/engine.rs) | `SemanticEngine`, `SemanticConfig` | Chunking, embedding and storage in one engine — a **prototype scaffold**: the library's vectors are built on the build machine only (`build`, `embed-shard`), and the application opens a prebuilt artifact read-only through `OfficialSemanticIndex`, embedding nothing but the query |
| **Embedding Cache** | [`src/semantic/embedding_cache.rs`](src/semantic/embedding_cache.rs) | `EmbeddingCache` | LRU cache over recently embedded query texts |
| **Search Profiles** | [`src/config/profiles.rs`](src/config/profiles.rs) | `SearchProfile`, `RankingProfile`, `FusionStrategy`, `QueryTypeAlphas` | Fast/Balanced/Best presets and the weighted / RRF / adaptive fusion choice. A whole `RankingProfile` — strategy, RRF `k`, alpha per query type, BM25 `k`, threshold, bonuses — can be passed per search and is validated, so the defaults, still unmeasured, can be calibrated from the application without an engine release |
| **Feature Flags** | [`src/config/feature_flags.rs`](src/config/feature_flags.rs) | `FeatureFlags::apply` | Per-run overrides onto a profile, without a second source of defaults |
| **Query Cache** | [`src/hybrid/cache.rs`](src/hybrid/cache.rs) | `QueryCache`, `QueryCacheStats` | Result cache keyed by query parameters, invalidated by generation |
| **Metadata Ranking** | [`src/hybrid/metadata_ranker.rs`](src/hybrid/metadata_ranker.rs) | `MetadataRanker`, `MetadataSignal` | Small facet-derived bonuses (primary source, era, category) |
| **Hebrew Normalizer** | [`src/hybrid/hebrew_normalizer.rs`](src/hybrid/hebrew_normalizer.rs) | `HebrewNormalizer`, `QueryLanguage` | Nikud/taamim stripping and geresh normalization before embedding |
| **Telemetry** | [`src/telemetry/mod.rs`](src/telemetry/mod.rs) | `TelemetryCollector`, `SearchTelemetry` | In-process counters only — nothing is transmitted anywhere |
| **Index Package** | [`src/distribution/package.rs`](src/distribution/package.rs) | `IndexPackage`, `ArtifactExpectation`, `VerifiedPackage`, `VerificationDepth` | Metadata plus a SHA-256 per payload, and the artifact digest that a published value can be compared against. `verify_for_install` hashes everything; `verify_for_open` does not, and the token records which ran |
| **Package Install** | [`src/distribution/importer.rs`](src/distribution/importer.rs) | `IndexImporter`, `recover_interrupted_install` | Verify the source, copy to staging, verify the copy, swap. The swap is two renames with a window in between, so the intermediate names are deterministic and recovery is a documented step |
| **Corpus Port** | [`src/distribution/corpus.rs`](src/distribution/corpus.rs) | `CorpusIndex`, `CorpusBooks`, `CorpusLine`, `JsonlCorpus` | The lexical index a build reads, as a trait — Tantivy is not a dependency of this crate and must not be. The corpus supplies the identity, every stored field, **and the exact set of lines the declared recipe embeds**, so there is no second description of a book to drift from the first, no silently partial artifact, and no vector for a line that should never have been embedded. `CorpusBooks` adds the corpus's *shape* — which lines share a book, and in what order — because a recipe reads a line together with its neighbours; it answers nothing about a line's contents, so the two halves cannot describe a book differently |
| **Artifact Builder** | [`src/distribution/builder.rs`](src/distribution/builder.rs) | `build`, `BuildPlan`, `BuildRequest`, `BuildReport` | A corpus and a model in, a base package out: `segment.oxv`, its metadata-v3 package and `release.json`. The vector, the key of the text it was built from and the model identity come out of one pass over one model; the set of lines to embed is derived from the recipe **before** any inference, and the recipe is pinned to the declared `chunking_identity` |
| **Sharded Build** | [`src/distribution/shard.rs`](src/distribution/shard.rs) | `export_plan`, `embed_shard`, `verify_shards`, `read_vector_inputs` | The same work cut in two: the recipe applied where the corpus is, the inference where the model is, every shard checked against the plan before a byte is read |
| **Benchmark Harness** | [`src/benchmark/mod.rs`](src/benchmark/mod.rs) | `measure`, `aggregate`, `QuerySet` | Timing and percentile helpers. A measurement tool, **not** a relevance dataset |
| **Integration Test** | [`tests/hybrid_integration_test.rs`](tests/hybrid_integration_test.rs) | feature-gated integration tests | End-to-end public-API suite using the explicit mock backend |
| **Official Runtime Test** | [`tests/official_runtime.rs`](tests/official_runtime.rs) | feature-gated integration tests | Builds a package, installs it, opens it, and queries it through the coordinator with a resolver standing in for the live index: the ids are the resolver's, two books sharing an id stay apart, a line whose text changed is not shown, the cache follows the resolver's generation and a reload, and every build-side call is refused with nothing written |
| **Builder Test** | [`tests/artifact_builder.rs`](tests/artifact_builder.rs) | integration tests, most feature-gated | S4b end to end through the binary: a corpus and a model in, a package out that installs against the digest the build printed and finds each line by its own text; two builds produce the same segment, and the line the recipe skips is absent from every result. A mismatched recipe is refused in a default build, before a model is opened — as is a build with no inference backend compiled in |
| **Scale Test** | [`tests/vector_set_scale.rs`](tests/vector_set_scale.rs) | `#[ignore]`d measurement | A synthetic set with the v30 library's shape at N records: write, install, open, scan, a delta and compaction, timed |
| **CI Workflow** | [`.github/workflows/ci.yml`](.github/workflows/ci.yml) | `check-and-test` | Multi-platform GitHub Actions CI workflow (Linux, Windows, macOS) |

---

## 🚀 Roadmap & Implementation Status

The stages below are the plan of record from
[שלבי ויעדי התקדמות.md](שלבי%20ויעדי%20התקדמות.md). S4b is the builder in this repo, and
store v2 — content-addressed int8 segments, installed sets, hits and a resolver — is the
store under all of it; the resolver over a live index, and S5–S8, land in
`otzaria_search_engine` and `otzaria`.

```text
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         PROJECT IMPLEMENTATION ROADMAP                           │
├──────────────────────────────────────────────────────────────────────────────────┤
│ [✔] Core architecture, subsystem isolation & error taxonomy                      │
│ [✔] Anchored chunker, manifest version tracker & in-memory vector store          │
│ [✔] Correct brute-force baseline (pre-norm dot product + min-heap)               │
│ [✔] Correctness baseline, lifecycle contracts & complete filters                 │
│ [✔] Real ONNX inference (ONNX Runtime) verified against golden vectors           │
│ [✔] Ranking profiles, fusion strategies, caches, telemetry & packaging prototype │
├──────────────────────────────────────────────────────────────────────────────────┤
│ [✔] S0  Product contract alignment (this section, and the docs around it)        │
│ [ ] S1  Representation quality & dimension/precision decision                    │
│ [✔] S2a Read-only runtime path: the artifact's reader, read/write store split    │
│ [✔] S2b Scale: 6.0M slots open in 3.8 ms, scan in 17–69 ms — no ANN             │
│ [✔] S3  Artifact contract: identity, two depths, recoverable install, reader     │
│ [✔] S4a Packer (since replaced by store v2's segments, built by key)             │
│ [~] S4b Builder: corpus + model → embeddings → artifact. Live Tantivy remains    │
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
2. **Scale measurement (S2b)** — answered: at 6.4M records (6.0M slots) a set opens in
   3.8 ms and an exact int8 scan takes 69 ms on one thread and 17 ms on ten, so there is no
   ANN. Still to measure: recall against exact f32 on the library's own vectors, and a weak
   laptop.
3. **Official artifact contract (S3) and its reader (S2a)** — identity, two verification
   depths, the published-digest anchor, a recoverable install, and a runtime path that
   opens the verified token landed; see
   [docs/ARTIFACT_CONTRACT.md](docs/ARTIFACT_CONTRACT.md). What is left:
   - Publish the release manifest's digest (and sign it): the check exists, and `build` prints the value, but nobody publishes it yet.
4. **A `CorpusIndex` over a live Tantivy index (S4b, in `otzaria_search_engine`)** — the
   [builder](src/distribution/builder.rs) now produces the vectors itself, from a corpus
   and a model, so what is left of S4b is the one thing Tantivy owns:
   - implement [`CorpusIndex`](src/distribution/corpus.rs) and
     [`CorpusBooks`](src/distribution/corpus.rs) over a live index, which also decides how
     `corpus_id` is derived — the JSONL transcription here is a transcription, not a source
     of truth, and it infers a book's line order from the id scheme rather than knowing it.
   - hand that implementation to `build`, which already applies the recipe, pins it to the
     declared `chunking_identity`, and derives the coverage set before any inference.
5. **Quality evaluation suite**:
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
- For `--features onnx-backend`: nothing at build time. ONNX Runtime is a shared library
  loaded when an ONNX model is, never linked, so running a graph needs one at run time —
  Microsoft's official [1.28.0 release](https://github.com/microsoft/onnxruntime/releases/tag/v1.28.0)
  is the reference. See [`docs/ONNX_BACKEND.md`](docs/ONNX_BACKEND.md).

### Feature matrix

| Build | Backend | `EmbeddingRuntime::load` |
|---|---|---|
| default | none | `Err(BackendUnavailable)` — a release build cannot serve fake vectors |
| `--features mock-embedding` | deterministic hash stand-in | `Ok` — **not a semantic model**, development and testing only |
| `--features onnx-backend` | real ONNX Runtime inference for an ONNX model package (desktop targets) | `Ok` once the runtime library is found — the path the application passes (`EmbeddingDeployment::onnx_runtime`), else `OTZARIA_ONNX_RUNTIME`, else the platform's file name beside the graph — and `Err(OnnxRuntimeUnavailable)` naming each place it looked otherwise |
| `mock-embedding` with `onnx-backend` | real inference wins | `Ok`, or the real backend's error — never a silent fall-through to the stand-in |

A model is an ONNX package: the graph a path ending in `.onnx`, in any case, names, the
`tokenizer.json` beside it and any external data it names. Any other model path — a GGUF
above all — is refused as `EmbeddingError::InvalidModelFile` before a backend is asked. A
build without the backend says which feature to enable.

The application embeds only queries — the library's vectors are built on the build
machine, and the app opens them read-only — so an ONNX model needs **one session** there,
the default; `OTZARIA_ONNX_SESSIONS` above 1 is a build-machine knob, worth it only for
callers that embed concurrently ([docs/ONNX_BACKEND.md](docs/ONNX_BACKEND.md) §6).

An application that ships ONNX Runtime passes its path as
`EmbeddingDeployment::onnx_runtime`, through `OfficialIndexConfig::deployment` or
`SemanticConfig::deployment`. A path passed is the only place looked — one that cannot be
loaded is an error naming it, never a fall-back — and, like everything in
`EmbeddingDeployment`, it is no part of an index's or an artifact's identity. The
application's layout, and where the runtime sits in it:
[docs/ONNX_BACKEND.md](docs/ONNX_BACKEND.md) §3.

The ONNX model, Meivin Round 2, ships as its **int8 graph** by default
([`config/models/meivin-round2-onnx/`](config/models/meivin-round2-onnx/README.md)): 42 MB
on disk against 168 MB for fp32, 169 against 387 MiB of peak memory, vectors within cosine
0.999 of fp32's, and about the same speed on an Apple M4 — decided for users on weak PCs,
where int8's latency on an old x86 CPU without VNNI is still to be measured. The fp32 graph
is the reference, with an identity of its own; see
[docs/MODEL_DISTRIBUTION.md](docs/MODEL_DISTRIBUTION.md) §6.0.

### Commands

```bash
# Build release library
cargo build --release

# Run unit and integration tests (never `--all-targets`: that selects the bench
# target, overriding its `test = false`, and runs a 200k x 1024 workload unoptimized)
cargo test --lib --tests
cargo test --lib --tests --features mock-embedding
# The ONNX tests that run a graph skip without a runtime library to load
OTZARIA_ONNX_RUNTIME=/abs/path/libonnxruntime.dylib cargo test --lib --tests --features onnx-backend
OTZARIA_ONNX_RUNTIME=/abs/path/libonnxruntime.dylib cargo test --lib --tests --features mock-embedding,onnx-backend

# Verify formatting
cargo fmt --check

# Run strict Clippy lints (run for each feature combination above)
cargo clippy --all-targets -- -D warnings
```

### Building an artifact (S4b)

One command: a corpus, a model file and the recipe in, a verified artifact out. It needs an
inference backend compiled in, because a build *is* inference.

```bash
OTZARIA_ONNX_RUNTIME=/abs/path/libonnxruntime.dylib \
cargo run --release --features onnx-backend -- build \
  --corpus-identity corpus-identity.json --corpus-lines corpus-lines.jsonl \
  --model config/models/meivin-round2-onnx/model.json \
  --model-file /abs/path/seforim-embed-round2-int8.onnx \
  --chunking config/models/meivin-round2-onnx/chunking.json \
  --out ./artifact
```

`chunking.json` is a `ChunkerConfig` — the recipe itself, not a description of one. The
production identity's:

```json
{"min_meaningful_chars": 20, "context_window_lines": 2, "max_chunk_chars": 512,
 "min_embeddable_chars": 5, "chunking_version": 1, "embedding_text_version": 2,
 "normalization_version": 1}
```

It must hash to the `chunking_identity` the model declares, and the build refuses it
otherwise. An artifact records the hash, and a hash cannot be turned back into five
numbers — so whoever applies the recipe has to be handed the recipe, and this is the only
thing that establishes they were handed the right one.

What the build establishes that a pack cannot: the vector, the digest of the text it came
from and the model identity all come out of one pass over one model. `model_checksum`,
`embedding_backend`, `embedding_dim`, `pooling` and the effective `max_tokens` are reported
by the loaded runtime and compared against what the artifact declares — the checksum and
the width are facts about the file, the backend id is which implementation was selected,
and pooling is what that implementation performs.

The three recipe versions — `chunking_version`, `embedding_text_version` and
`normalization_version` — are not facts about a model at all; they are versions of code in
this crate. Each is a closed set in [`recipe.rs`](src/semantic/recipe.rs), and the chunker
dispatches on it, so a version nobody has implemented is refused rather than declared. All
three are about the **text**: `normalization_version` is the text preprocessing applied
before the model sees a string — on both sides, so a query reaches the model the same way
the stored vectors did. L2 normalization of the finished vector is an invariant of cosine,
applied by every store unconditionally, and is deliberately not versioned. `model_id` and
`model_quantization` remain declarations — nothing in an ONNX package states either in a
form anything here could check.

Which lines get a vector is **derived** by running the chunker over the corpus, before any
inference. A line too short to carry meaning is skipped, and an artifact that skips it is
complete rather than short.

### Packing an artifact (S4a)

For vectors produced elsewhere. Both commands work in a **default build** — packing never
turns text into a vector, so it needs no inference backend.

```bash
cargo run --release -- pack \
  --vectors vectors.f32 --records vectors.jsonl \
  --corpus-identity corpus-identity.json --corpus-lines corpus-lines.jsonl \
  --model model.json --out ./artifact

cargo run --release -- validate \
  --artifact ./artifact \
  --corpus-identity corpus-identity.json --corpus-lines corpus-lines.jsonl \
  --model model.json --chunking chunking.json
```

| File | Shape |
|------|-------|
| `vectors.f32` | little-endian `f32`, `vector_count × embedding_dim`, no header |
| `vectors.jsonl` | one `{"line_id": N, "source_line_sha256": "...", "embedding_text_sha256": "..."}` per vector, **in the same order** |
| `corpus-identity.json` | the `CorpusIdentity` the lexical index reports |
| `corpus-lines.jsonl` | one document per line: `line_id`, book key, title, reference, section, segment, `is_pdf`, hashes, facets and `text` |
| `chunking.json` | optional here, required by `build`. Its three recipe versions must name behaviour this build implements, and must equal the ones the model identity declares. With it, the lines that must get a vector are the ones the recipe embeds, and the recipe is pinned to the declared `chunking_identity`. Without it the corpus file **is** the coverage contract — the vectors must cover it exactly, with nothing missing and nothing extra, so export exactly the lines that should be embedded |
| `model.json` | a `ModelIdentity` — see [`versioning.rs`](src/semantic/versioning.rs) |

`source_line_sha256` is the SHA-256 of the corpus line's text and is checked against the
corpus: a vector file shifted by one row passes every other check there is.
`embedding_text_sha256` is of the text that was actually embedded — after any title prefix,
neighbour context or truncation — and is recorded as the record's `chunk_hash`, because
that field is defined as a digest of the embedded text and the corpus holds the line.

Both are alignment records, not proof: nothing available to a tool that receives finished
floats can establish that a vector came from that text, by that model. `build` above is
what closes it, by producing all three in one pipeline.

The `Digest:` line the tool prints is what has to be published **outside** the artifact —
without it, a later verification detects damage and the wrong artifact, but not one
deliberately rebuilt to match. Two independent packs of the same vectors produce the same
digest, which is what makes publishing one meaningful.

### Testing against the real model

Tests that need the model are `#[ignore]`d and **skip loudly** when it is absent, so CI
stays green without it. The parity gate has one golden file per graph —
[`tests/data/onnx_golden_vectors_int8.json`](tests/data/onnx_golden_vectors_int8.json) for
the default int8 graph and [`tests/data/onnx_golden_vectors.json`](tests/data/onnx_golden_vectors.json)
for the fp32 reference — each the answers of a Python reference (`tokenizers` and
`onnxruntime`). The gate picks the file by the graph's SHA-256, so it takes either graph,
with its `tokenizer.json` beside it, and a runtime library:

```bash
OTZARIA_TEST_ONNX_MODEL=/abs/path/seforim-embed-round2-int8.onnx \
OTZARIA_ONNX_RUNTIME=/abs/path/libonnxruntime.dylib \
  cargo test --lib --features onnx-backend onnx_backend::golden -- --ignored --nocapture
```

Token ids must match exactly, then every vector within cosine 0.99999 — for int8 on another
CPU family than the goldens', 0.995, because int8 vectors depend on the CPU's int8 kernels.
The token-id assertion is the primary gate, not the cosine one: truncation, the role
tokens and special-token matching are all decided before the graph runs, and the ids are
where an error in any of them shows. The model's package, checksum and licence are in
[`config/models/meivin-round2-onnx/`](config/models/meivin-round2-onnx/README.md).

The build path has one real-weights test of its own: an artifact built from the int8 graph
and verified end to end, where every other builder test runs on the stand-in.

```bash
OTZARIA_TEST_ONNX_MODEL=/abs/path/seforim-embed-round2-int8.onnx \
OTZARIA_ONNX_RUNTIME=/abs/path/libonnxruntime.dylib \
  cargo test --test artifact_builder --features onnx-backend -- --ignored --nocapture
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

Further jobs: an **ONNX backend** job that tests `onnx-backend` on all three, against
Microsoft's ONNX Runtime 1.28.0 fetched per platform and checked against a pinned
SHA-256; a **golden vectors** job that runs the parity gate for both of the model's
graphs, int8 and fp32, fetched from the private mirror
`otzaria/judaic-semantic-round2-onnx-zayit`, checks that the crate and
`tools/onnx_package_checksum.py` compute the same package checksum for each, the one its
identity declares, and builds and verifies one real artifact with the int8 graph; and an
**emulated** golden job that runs the int8 gate under Intel's SDE as three x86 CPUs the
runners do not have. The golden jobs need the `OTZARIA_HF_TOKEN` secret, whose account
must be able to read that mirror; when the secret is absent they fail loudly rather than
reporting a skip as a pass. That gate is a reason the model's distribution route
matters — see [docs/MODEL_DISTRIBUTION.md](docs/MODEL_DISTRIBUTION.md).

---

## 📖 Developer Documentation

For detailed architectural invariants, subsystem separation rules, and development guidelines, refer to:
- [docs/PRODUCT_CONTRACT.md](docs/PRODUCT_CONTRACT.md) — **Binding scope definition** (Hebrew); outranks every other document here
- [docs/ARTIFACT_CONTRACT.md](docs/ARTIFACT_CONTRACT.md) — Artifact identity fields, verification order and what is not yet enforced (Hebrew)
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) — Comprehensive developer guide & status (Hebrew)
- [docs/CODE_MAP.md](docs/CODE_MAP.md) — Detailed code map and component descriptions
- [docs/MODEL_DISTRIBUTION.md](docs/MODEL_DISTRIBUTION.md) — How the embedding model reaches the device, and what an ONNX model package is (Hebrew)
- [docs/ONNX_BACKEND.md](docs/ONNX_BACKEND.md) — The ONNX Runtime backend: the runtime library, tuning and determinism
- [config/README.md](config/README.md) — The model identities: the int8 graph's, which is production, and the fp32 reference's
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
