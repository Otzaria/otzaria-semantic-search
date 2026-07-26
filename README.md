# Otzaria Semantic Search

[![CI](https://github.com/Otzaria/otzaria-semantic-search/actions/workflows/ci.yml/badge.svg)](https://github.com/Otzaria/otzaria-semantic-search/actions/workflows/ci.yml)
[![Personal Use License](https://img.shields.io/badge/license-Personal%20Use%201.0-blue.svg?style=flat)](LICENSE)

**Hybrid semantic & lexical search engine for Otzaria** — a local, offline, production-grade Rust library that merges BM25 lexical results with vector similarity search into a single ranked result set.

---

## 🚀 Roadmap — מה הלאה! (צעדי הפיתוח הבאים)

המערכת בנויה בארכיטקטורה מודולרית מלאה ב-Rust עם 100% טסטים עובדים ו-CI יציב. להלן תוכנית העבודה המדויקת לשלבים הבאים:

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                    ROADMAP & IMPLEMENTATION PHASES                      │
├─────────────────────────────────────────────────────────────────────────┤
│ [✔] Phase 1: Core Architecture, Types & Hybrid Fusion Coordinator       │
│ [✔] Phase 2: Anchored Chunker, Manifest Tracker & Memory Vector Store   │
│ [✔] Phase 3: High-Performance Vector Search (Pre-norm + BinaryHeap Top-K)│
├─────────────────────────────────────────────────────────────────────────┤
│ [ ] Phase 4: Native GGUF Model Runtime Integration (Candle / GGUF)     │
│ [ ] Phase 5: Persistent Disk Vector Store (zvec / HNSW Index on Disk)   │
│ [ ] Phase 6: Flutter Rust Bridge (FRB) & Otzaria App UI Integration     │
│ [ ] Phase 7: Background Streaming Indexer (StreamSink<IndexingProgress>)│
│ [ ] Phase 8: Search Quality Evaluation Suite (Recall@K, MRR & Benchmarks)│
└─────────────────────────────────────────────────────────────────────────┘
```

### 1. חיבור מודל ה-Embedding המקומי (Candle GGUF Inference)
- **המצב כרגע**: הממשק `EmbeddingRuntime` מוכן ומודל. בבדיקות נעשה שימוש ב-SHA256 feature hashing fallback.
- **מה הלאה**:
  - חיבור תלות `candle-core` ו-`candle-transformers` לטעינת קובץ GGUF מקומי (דוגמת `Otzaria-Embedding-V1-Flash-0.6B`).
  - מימוש Last-Token Pooling ואינפרנס SIMD/GPU (AVX2/NEON/Metal/DirectX).

### 2. שמירת הוקטורים בדיסק (Persistent Disk Vector DB)
- **המצב כרגע**: `VectorStore` עובד בזיכרון בסיבוכיות אופטימלית $O(N \log k)$ עם נורמליזציה מראש.
- **מה הלאה**:
  - חיבור `zvec-core` או מנגנון HNSW דיסקי לשמירת הוקטורים בתיקיית `semantic_db/zvec`.
  - מימוש `commit()` אטומי לדיסק ושימוש ב-Memory Mapped Files (`mmap`) לחיסכון בזיכרון RAM.

### 3. חיבור ל-Flutter (`flutter_rust_bridge`) ועדכון ממשק משתמש
- **המצב כרגע**: ה-API החשוף ב-`src/api/hybrid_search.rs` (`OtzariaHybridEngine`, `SearchRequest`) מוכן לייצור קוד Bridge.
- **מה הלאה**:
  - יצירת ה-Dart Bindings בעזרת `flutter_rust_bridge_codegen`.
  - הוספת פקדי UI באפליקציית Otzaria: מתג הפעלה לחיפוש סמנטי, בחירת מצב חיפוש (היברידי / לקסיקלי / סמנטי), ומחוון התקדמות אינדוקס.

### 4. מנגנון אינדוקס ברקע (Background Batch Indexer)
- **המצב כרגע**: אינדוקס ספר בודד מבוצע בצורה סינכרונית.
- **מה הלאה**:
  - מימוש `start_semantic_indexing` שמריץ אינדוקס ברקע לכל הספרים שדורשים עדכון (לפי ה-`IndexDiff`).
  - הזרמת התקדמות חיזותית ל-Flutter בעזרת `StreamSink<IndexingProgress>`.

### 5. ערכת מדידת איכות והערכה (Evaluation Suite)
- **מה הלאה**:
  - יצירת דאטה-סט שאילתות תורניות והלכתיות לבדיקת איכות.
  - חישוב מטריקות איכות: Recall@K, MRR (Mean Reciprocal Rank), ו-nDCG.
  - מדידות ביצועי Latency (p50/p95/p99) וצריכת זיכרון.

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

## Building & Testing

```bash
cargo build
cargo test --all-targets
cargo fmt --check
cargo clippy --all-targets -- -D warnings
```

## CI Pipeline

GitHub Actions CI runs on every push to `main` and on all PRs, across **3 platforms** (Ubuntu, Windows, macOS).

## Performance Highlights

- **Pre-normalized vectors**: L2 normalization at insert time $\to$ cosine similarity reduces to a single dot product at search time.
- **BinaryHeap top-k**: $O(N \log k)$ search instead of $O(N \log N)$ full-sort.
- **Single-pass Unicode truncation**: `char_indices().nth()` instead of double iteration.

## Developer Documentation

For detailed architecture specs and developer guidelines, see [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) and [docs/CODE_MAP.md](docs/CODE_MAP.md).

## License

Personal Use License 1.0 — see [LICENSE](LICENSE) for details.
