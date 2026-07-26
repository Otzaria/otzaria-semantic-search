# Otzaria Hybrid Semantic Search — Code Map (מפת הקוד)

מנגנון החיפוש ההיברידי של אוצריא מורכב משתי תת-מערכות עיקריות:
1. **Semantic Subsystem (Sidecar)** — מנהלת את ה-Embedding, ה-Vector Store, ה-Chunking, וה-Manifest.
2. **Hybrid Coordinator** — מנהלת את ניתוח השאילתא, נורמליזציית הציונים, ה-Fusion (מיזוג ציוני BM25 וסמנטיקה), ה-Ranking, ה-Grouping (קיבוץ לפי קטע או טקסט זהה), וה-Fallback.

---

## 🏛️ מבנה העץ של המאגר

```text
otzaria-semantic-search/
├── Cargo.toml                              # הגדרות תלויות והידור
├── README.md                                # מסמך ראשי ורישיון
├── docs/
│   ├── CODE_MAP.md                         # מפת קוד זו
│   └── DEVELOPMENT.md                      # מדריך ארכיטקטורה ופיתוח מקיף
├── .github/workflows/
│   └── ci.yml                              # CI/CD אוטומטי (Linux, Windows, macOS × 2 קונפיגורציות features)
├── tests/
│   ├── hybrid_integration_test.rs          # בדיקות מקצה לקצה (דורש --features mock-embedding)
│   └── production_backend_gate.rs          # מאמת שבנייה רגילה מסרבת לייצר embeddings
└── src/
    ├── lib.rs                              # נקודת הכניסה לספריה (Module root)
    ├── errors.rs                           # מערכת השגיאות המרכזית (thiserror)
    ├── api/
    │   ├── mod.rs                          # ייצוא רכיבי ה-API
    │   └── hybrid_search.rs                # ממשק API נקי עבור Flutter / FFI
    ├── hybrid/
    │   ├── mod.rs                          # ייצוא רכיבי ה-Hybrid
    │   ├── coordinator.rs                  # מתאם החיפוש ההיברידי הראשי
    │   ├── fusion.rs                       # מיזוג ציונים (Weighted & RRF)
    │   ├── grouping.rs                     # קיבוץ תוצאות (Section & Dedup)
    │   └── ranking.rs                      # ניתוח שאילתא וחישוב משקל אלפא
    └── semantic/
        ├── mod.rs                          # ייצוא רכיבי ה-Semantic
        ├── chunker.rs                      # Anchored Chunking & SHA256 IDs
        ├── embedding.rs                    # GGUF Model Runtime Interface
        ├── engine.rs                       # מתאם תת-המערכת הסמנטית
        ├── manifest.rs                     # מעקב גירסאות קבצים אטומי (JSON)
        ├── store.rs                        # Vector DB (Pre-normalized + Heap)
        └── types.rs                        # הגדרות טיפוסים ומבני נתונים
```

---

## 🧩 מודולים ורכיבים מרכזיים

### 1. נקודת הכניסה ומערכת השגיאות

* [`src/lib.rs`](../src/lib.rs)
  - מייצא את המודולים הראשיים: `api`, `errors`, `hybrid`, `semantic`.
* [`src/errors.rs`](../src/errors.rs)
  - `SemanticSearchError` — השגיאה הראשית המאגדת את כל תת-המערכות.
  - `EmbeddingError` — שגיאות טעינת מודל ואינפרנס.
  - `VectorStoreError` — שגיאות חיפוש, הכנסה ומחיקה ב-Vector DB.
  - `ManifestError` — שגיאות תואמות מודל וגרסאות אינדקס.
  - `ChunkingError` — שגיאות חלוקת ספר לקטעים.

---

### 2. ממשק ה-API עבור Flutter / FFI (`src/api/`)

* [`src/api/hybrid_search.rs`](../src/api/hybrid_search.rs)
  - `OtzariaHybridEngine` — Wrapper ראשי הניתן לחשיפה ל-Flutter באמצעות `flutter_rust_bridge`.
  - `SearchRequest` — Struct המאגד את פרמטרי השאילתא והפילטרים למניעת `too_many_arguments`.
  - `get_semantic_status()` — שאילתת סטטוס זמינות המודל והאינדקס.
  - `get_semantic_index_diff()` — בדיקת פערים בין Tantivy ל-Semantic Store.
  - `index_books()` — אינדוקס ספרים (כתיבת manifest אחת לכל הקבוצה).
  - `reset_semantic_index()` — מסלול ההתאוששות מאינדקס לא תואם; בלעדיו
    `needs_full_reindex` היה מבוי סתום.
  - *לא כאן עדיין (P7):* progress stream, cancel/resume, ניהול הורדת מודל.

---

### 3. תת-המערכת ההיברידית (`src/hybrid/`)

* [`src/hybrid/coordinator.rs`](../src/hybrid/coordinator.rs)
  - `HybridCoordinator` — מתאם החיפוש הראשי. מריץ חיפוש סמנטי לצד מועמדי BM25, מפעיל ניתוח שאילתא, מיזוג ציונים, קיבוץ, ומבצע Fallback ל-BM25 אם ה-Semantic Engine נכשל.
  - `HybridSearchParams` — פרמטרי חיפוש (גבולות, Offset, Grouping, Filters, Force Mode).
  - **שלושת המצבים ממומשים**: `LexicalOnly` אינו נוגע במסלול הסמנטי, `SemanticOnly`
    מזניח את מועמדי BM25 שהועברו, ו-`Hybrid` מתדרדר ל-`LexicalOnly` כשהסמנטי נכשל.
    ה-`alpha` נקבע לפי המצב שרץ בפועל (1.0 / 0.0 / דינמי), כדי שציון ממנוע אחד
    לא יוקטן במשקל של המנוע החסר.
  - כל התדרדרות נראית: `search_mode` הוא המצב שרץ, `fallback_reason` הוא הסיבה.
  - חלון המועמדים הסמנטיים חסום ב-`MAX_SEMANTIC_CANDIDATES` (מדווח ב-log כשנחתך).

* [`src/hybrid/fusion.rs`](../src/hybrid/fusion.rs)
  - `normalize_bm25_scores()` — נורמליזציית רוויה $x / (k + x)$ לציוני BM25 לטווח $[0,1]$.
  - `normalize_semantic_scores()` — נורמליזציה ליניארית $(x + 1) / 2$ לציוני Cosine $[-1,1] \to [0,1]$.
  - `fuse_weighted()` — מיזוג ממושקל לפי אלפא: $\alpha \cdot BM25 + (1-\alpha) \cdot Semantic$.
  - `fuse_rrf()` — מיזוג בשיטת Reciprocal Rank Fusion ($1 / (k + rank)$).

* [`src/hybrid/ranking.rs`](../src/hybrid/ranking.rs)
  - `analyze_query()` — מזהה מאפייני שאילתא (ביטוי במרכאות, שאילתא קצרה, מילות קונספט, מספרים).
  - `compute_alpha()` — מחשב דינמית את משקל האלפא (שאילתות מדויקות/קצרות $\to \alpha \in [0.7, 0.9]$, שאילתות מושגיות ארוכות $\to \alpha \in [0.2, 0.4]$).
  - `BonusConfig` — הגדרת בונוסים וקנסות (בונוס התאמה מדויקת, קנס כפילויות וכו').

* [`src/hybrid/grouping.rs`](../src/hybrid/grouping.rs)
  - `group_by_section()` — מקבץ תוצאות לפי `section_id` וקובץ. הנציג בעל הציון הגבוה ביותר נבחר כ-Representative.
  - `group_by_identical_text()` — מקבץ תוצאות בעלות `line_hash` זהה (מניעת כפילויות של נוסחים זהים).
  - `group_results()` — Dispatcher לפי `GroupingMode`.

---

### 4. תת-המערכת הסמנטית (`src/semantic/`)

* [`src/semantic/types.rs`](../src/semantic/types.rs)
  - `BookLine` & `BookForIndexing` — ייצוג קלט ספר מ-Tantivy.
  - `SemanticChunk` — קטע טקסט מעובד המיועד ל-Embedding עם שדות Anchored context.
  - `VectorMetadata` — מטא-דאטה שנשמר לצד הוקטור ב-Vector Store.
  - `SemanticCandidate` & `LexicalCandidate` — מועמדים מכל נתיב חיפוש.
  - `FusedCandidate` & `GroupedResult` — מועמד מאוחד ותוצאה מקובצת.
  - `HybridSearchResult` & `HybridResultItem` — פלט החיפוש המוחזר ל-UI.

* [`src/semantic/chunker.rs`](../src/semantic/chunker.rs)
  - `Chunker` & `ChunkerConfig` — מנגנון Chunker מעוגן (Anchored Chunking) המוסיף הקשר משורות סמוכות באותו סעיף לשורות קצרות.
  - `compute_semantic_id()` — יצירת מזהה SHA256 hex יציב לפי מפתח ספר, שורה וגרסת חלוקה.
  - `truncate_to_chars()` — חיתוך UTF-8 יעיל במעבר יחיד.

* [`src/semantic/embedding.rs`](../src/semantic/embedding.rs)
  - `EmbeddingRuntime` & `EmbeddingConfig` — ממשק הרצת מודל GGUF מקומי.
  - `validate_and_checksum_gguf()` — אימות magic + version וחישוב SHA-256 **במעבר
    אחד** על הקובץ (מודל של מאות MB נקרא פעם אחת בלבד).
  - `EmbeddingBackendKind` — זהות ה-backend, נשמרת ב-manifest. `is_semantic()` מחזיר
    `false` ל-stand-in, כדי שלא יתחזה למודל.
  - `embed_batch()` — ה-primitive; `embed_one()` עוטף אותו. כל וקטור מאומת בממד
    ובנורמה, כך שווקטור אפס אינו נכנס לאינדקס.
  - `l2_normalize()` — נורמליזציית L2, מחזירה את הנורמה שהייתה לפני כן.
  - `mock` — ה-stand-in הדטרמיניסטי, זמין רק תחת `cfg(test)` או
    `--features mock-embedding`. **אינו מודל סמנטי.**

* [`src/semantic/store.rs`](../src/semantic/store.rs)
  - `VectorStore` & `VectorStoreConfig` — מנגנון האחסון והשליפה הוקטורי.
  - **Pre-normalization**: נורמליזציה בוקטורים בעת ההכנסה המאפשרת חישוב דמיון קוסינוס בעזרת Dot Product בלבד ($O(dim)$).
  - **BinaryHeap Top-K**: שליפת $k$ התוצאות המובילות בסיבוכיות $O(N \log k)$ ללא שכפול מטא-דאטה של כל המאגר. שוויון ציונים נשבר לפי `semantic_id` — בלי זה `HashMap` עם סדר איטרציה מקרי היה מחזיר top-k שונה בכל ריצה.
  - **מנעול אחד** לשתי המפות: קודם היו שני מנעולים ש-insert ו-delete נטלו בסדר הפוך — lock-order inversion שעלול לתקוע את התהליך.
  - `is_persistent()` / `backend_id()` — ה-backend הנוכחי אינו persistent, ומצהיר על כך; ה-engine מסתמך על זה כדי לא להאמין ל-manifest ישן.
  - `dot_product()` — 8 מצברים במקום סכימה סדרתית אחת: ~1.4× מהיר במדידה
    (101ms מול 145ms על 200k וקטורים בממד 1024).

* [`src/semantic/manifest.rs`](../src/semantic/manifest.rs)
  - `SemanticManifest` — ניהול גירסאות אינדקס אטומי: כתיבה ל-`.tmp`, `fsync`, ואז rename. בלי ה-`fsync` הניתן להחלפה אטומית עדיין אפשר לאבד את התוכן בהפסקת חשמל.
  - `validate()` — זיהוי אי-התאמות בכל עשרת הממדים: model id, checksum, embedding backend, ממדים, pooling, quantization, vector precision, vector backend, chunking ו-normalization.
  - `ManifestMismatch::invalidates_vectors()` — האם הווקטורים עצמם פסולים (מודל/ממד) או שרק צריך chunking מחדש.
  - `quarantine()` — קובץ manifest לא קריא מועבר הצידה ולא נמחק, כדי שיהיה מה לחקור.
  - `clear_books()` — מחיקת רשומות הספרים תוך שמירת המטאדאטה של הקונפיגורציה.
  - `book_needs_reindex()` — בדיקת דלתא לפי `content_hash` מול Tantivy.

* [`src/semantic/engine.rs`](../src/semantic/engine.rs)
  - `SemanticEngine` & `SemanticConfig` — המנוע הסמנטי המרכזי המאגד את ה-Chunker, ה-Runtime, ה-VectorStore וה-Manifest.
  - `SemanticConfig::validate()` — פוסל קונפיגורציה שלא תעבוד, ובראשה אי-התאמה בין
    `embedding_dim` ל-`store.embedding_dim` (שקודם התגלתה רק באמצע האינדוקס).
  - `open()` — מפייס את ה-manifest מול הקונפיגורציה: שימוש חוזר, גריעת רשומות
    שהווקטורים שלהן לא שרדו, או quarantine והתחלה מחדש. אינו נכשל בגלל manifest פגום.
  - `index_book()` / `index_books()` — האחרון כותב manifest פעם אחת לכל הקבוצה.
    שניהם **מוחקים** את הווקטורים הקודמים של הספר לפני הכתיבה, ורק אחרי שה-embedding
    הצליח — כך שכשל באמצע לא משאיר ספר בלי וקטורים עם רשומה שמצהירה שהוא מאונדקס.
  - `reset_index()` — מסלול ההתאוששות מ-`IncompatibleIndex`.
  - `diff_against_tantivy()` — דגלי אי-התאימות אמיתיים; פלט בסדר דטרמיניסטי.

---

## 🧪 בדיקות ותשתית

* [`tests/hybrid_integration_test.rs`](../tests/hybrid_integration_test.rs)
  - בדיקות מקצה לקצה דרך ה-API הציבורי: אינדוקס ספרייה, שלושת מצבי החיפוש,
    התדרדרות חיננית, מחזור אינדוקס→הפעלה־מחדש→חיפוש, re-index שמוחק שורות שנעלמו,
    התאוששות מ-manifest פגום, אי-תאימות ו-reset, filters ו-paging.
  - דורש `--features mock-embedding` (אין backend inference בבנייה רגילה).
* [`tests/production_backend_gate.rs`](../tests/production_backend_gate.rs)
  - התמונה ההופכית, מתקמפל **רק בלי** ה-feature: מאמת שבנייה רגילה מסרבת לטעון
    מודל ולייצר וקטורים. זו הערובה שקוד production לא יגיש וקטורים מזויפים —
    ולכן היא נבדקת ולא נסמכת על `#[cfg]` שיישאר במקומו.
* [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)
  - תהליך CI מלא ב-GitHub Actions הרץ על Ubuntu, Windows ו-macOS, **בשתי
    קונפיגורציות features**, כולל `cargo fmt --check`, clippy עם `-D warnings`,
    ואימות קישורי תיעוד (`cargo doc`).

## הרצה מקומית

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings                          # production
cargo clippy --all-targets --features mock-embedding -- -D warnings
cargo test  --all-targets                                          # שער ה-production
cargo test  --all-targets --features mock-embedding                 # החבילה המלאה
```
