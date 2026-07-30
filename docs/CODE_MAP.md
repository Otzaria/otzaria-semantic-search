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
  - `get_semantic_index_diff()` — בדיקת פערים בין Tantivy ל-Semantic Store. הצורה
    המועדפת: הקורא מחליט מה החתימה של ספר, וזו הדרך היחידה שבה PDF יכול להגיע
    ל"מעודכן". `get_semantic_index_diff_from_lexical_hashes()` היא הצורה הידידותית
    ל-FFI (`u64` גולמי), כי enum ש-Dart יכול לבנות הוא enum ש-Dart יכול לבנות שגוי.
  - `index_books()` — אינדוקס ספרים (manifest נשמר פעם אחת בסוף, לא פר-ספר).
  - `remove_semantic_books()` — מיישם בקבוצה את `IndexDiff::removed_books`.
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
  - `index_books()` — נועל את ה-engine **פר-ספר** כדי שחיפושים לא ייחסמו לכל אורך
    האינדוקס, ושומר את ה-manifest פעם אחת בסוף: כל שמירה מסריאלזת את כל הרשומות,
    כך שגם checkpoints של מסמך מלא מוסיפים כתיבה סופר־ליניארית. ה-store הנוכחי
    נדיף, ולכן checkpoint של manifest ממילא אינו יכול לשמר עבודה אחרי קריסה.
    backend persistent יצטרך journal מצטבר או פורמט checkpoint דלתאי.
    `indexing: Mutex` מסדר בתור אינדוקס, reset וגריעת ספרים.

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
  - `BookLine` & `BookForIndexing` — ייצוג קלט ספר מ-Tantivy: `topics` (נתיב
    קטגוריה אחד) ו-`extra_facets` (רשימה — ספר יכול לשאת כמה מחברים).
  - `ContentFingerprint` — `Canonical(NonZeroU64)` / `ContentOnly(NonZeroU64)` /
    `Unverifiable`. שני מלכודות שקטות: `0` הוא מרקר "אין חתימה" ולא hash (Tantivy
    מדווח אותו לכל PDF), ולכן הווריאנטים נושאים `NonZeroU64`; ולא כל חתימה מכסה
    metadata — גודל+mtime של קובץ לא מזיז כשמתקנים מחבר, ולכן רק `Canonical` מגיע
    ל"מעודכן". `ContentFingerprint::canonical()` דורש revision לא־אפס שמכסה את כל
    הקלט ל־`BookForIndexing`: הטקסט המחולץ, מבנה ומזהי השורות/סעיפים, references
    וגרסת החילוץ/OCR. גודל+mtime לבדם הם `ContentOnly`; אפס הוא `Unverifiable`.
  - `BookForIndexing::line_fingerprint()` — חתימה שהמנוע הסמנטי מחשב מהספר עצמו:
    שורות **וגם** כל ה-metadata שנשמר בכל וקטור. זה מה שמכריע ספר שהחתימה החיצונית
    שלו לא הוכיחה דבר.
  - `canonical_facets()` — כל חתימה ממיינת ומסירה כפילויות מ-facets, כמו
    `book_fingerprint` הלקסיקלי: סדר facets אינו מידע, וחתימה שרגישה לו הייתה גורמת
    ל-re-embedding על שינוי סדר בלבד.
  - `SearchFilters` & `CompiledFilters` — רשימת facets שטוחה, מקובצת לממדים לפי
    `FACET_DIMENSION_ROOTS` בדיוק כמו `facet_filter_query` הלקסיקלי. `compile()`
    מקבץ פעם אחת לשאילתה, כי ההתאמה נקראת פעם לכל וקטור באחסון — `VectorStore::search`
    קורא ל-`CompiledFilters::matches` ולא ל-`SearchFilters::matches`.
  - `IndexOutcome` & `IndexingSummary` — `Indexed`/`Skipped`/`Empty` במקום ספירת
    chunks שלא הבדילה בין "נכתב" לבין "כבר היה".
  - `IndexDiff::unverifiable_books` — ספרים שאי אפשר להוכיח שלא השתנו, בנפרד
    מ-`changed_books`.
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
  - `validate_and_checksum_gguf()` — אימות קונטיינר וחישוב SHA-256 **במעבר אחד** על
    הקובץ (מודל של מאות MB נקרא פעם אחת בלבד). ה-header נבדק אחרי 24 בייטים, לפני
    שממשיכים; אחריו נפרסר כל אזור ה-descriptors, ומתוך ה-offsets המוצהרים נגזר חסם
    תחתון על גודל הקובץ — ביט אחד לאיבר, נכון לכל טיפוס ggml. חסם תחתון בכוונה: טבלת
    block sizes שגויה *דוחה מודל תקין*, וזה כשל גרוע יותר. `HashingReader` הוא מה
    שמאפשר לפרסר ולחשב hash בלי לקרוא פעמיים.
    קונטיינר ללא tensors, metadata type לא מוכר בגרסה נתמכת, alignment שאינו כפולה
    של 8 או tensor offset לא מיושר — נדחים; אין fallback לקבלת descriptors שלא
    הצלחנו לפרסר.
  - `embed_batch()` — ה-primitive; `embed_one()` עוטף אותו. זו **נקודת החניקה
    הראשית**: היא מחלקת ל-batches בגודל `batch_size`, בודקת שהוחזר וקטור לכל קלט,
    ומריצה `normalize_validated` על כל אחד. ה-backends מחזירים וקטורים גלמיים ולא
    מנורמלים, כדי שכולם יקבלו את אותו טיפול. ל-`VectorStore` יש guard עצמאי משלו —
    הוא API ציבורי שיכול לקבל וקטורים שלא עברו כאן.
  - `normalize_validated()` — דוחה כל וקטור שלא ניתן להשוות: ממד שגוי, רכיב
    לא-finite, נורמה לא-finite (כולל וקטור finite שגולש), או נורמה `<= MIN_VECTOR_NORM`.
  - `MIN_VECTOR_NORM` — סף אחד ל-crate כולו (`pub(crate)`), עם אותה השוואה בכל שכבה.
    קודם היה `<` בצד אחד ו-`>` בצד השני, כך שהוקטור *בדיוק* על הסף לא נדחה ולא
    נורמל — ונכנס לאינדקס כשהציון שלו הוא הגודל שלו ולא קוסינוס.
  - `l2_normalize()` — נורמליזציית L2, מחזירה את הנורמה שהייתה לפני כן.
  - `mock` — ה-stand-in הדטרמיניסטי, זמין רק תחת `cfg(test)` או
    `--features mock-embedding`. **אינו מודל סמנטי.**

* [`src/semantic/backend.rs`](../src/semantic/backend.rs) — החוזה שכל backend מקיים.
  - `EmbeddingBackend` — trait עם `Send + Sync`, כי הקואורדינטור מחזיק את המנוע
    ב-`RwLock` ו-`search` לוקח `.read()`; חיפושים נכנסים ל-`embed_batch_raw` דרך
    `&self` במקביל. `&mut self` היה מסרייל חיפוש מאחורי אינדוקס.
  - `embed_batch_raw()` — מחזיר וקטורים **גלמיים**. הנרמול אינו תפקיד ה-backend.
  - `tokenize()` — קיים כי בדיקת ה-parity של P2 מחייבת שוויון `token_ids`, ואין דרך
    לאמת אותה בלי לחשוף את הטוקנייזר. ה-stand-in מחזיר `TokenizationUnsupported`
    ולא מימוש מנוון — ids "סבירים" היו הופכים את הבדיקה להשוואה בין שתי המצאות.
  - `Pooling` — `LastToken` / `Mean`, עם התאמת מחרוזות **מדויקת** (לא case-insensitive
    ובלי trim): אותה מחרוזת נשמרת ב-manifest ומושווית תו-בתו בסשן הבא, כך שקבלת
    `"Last-Token"` כאליאס הייתה מייצרת mismatch מדומה ובנייה מחדש של האינדקס.
    `Mean` בר-ייצוג ובלתי-שמיש: הוא מה שמאפשר לבטא "הקונפיג חולק על ה-backend".
  - `CANDIDATES` / `select_backend()` — טבלה אחת שממנה קוראים גם הבחירה וגם בדיקת
    ה-pooling, ולכן הוספת backend היא שורה. הטבלה **אינה** מותנית ב-feature: היא
    מתארת אילו מימושים קיימים ב-crate, אחרת "אין backend" היה מדווח כ"קונפיגורציה
    שגויה". מחזירה `Option<Result<..>>` — `None` = לא מקומפל (המשך לחפש),
    `Some(Err)` = מקומפל ונכשל (עצור ודווח). עם `Option` בלבד, backend אמיתי שנכשל
    היה נראה כחסר, וה-stand-in היה עונה על מודל שבור בווקטורי האש בשקט.

* [`src/semantic/llama_backend.rs`](../src/semantic/llama_backend.rs) — inference אמיתי,
  מאחורי `--features llama-backend` (ראו [`P2_INFERENCE_SPIKE.md`](P2_INFERENCE_SPIKE.md)).
  - `ContextPool` — thread עובד לכל context, שיוצר את ה-context שלו מ-`Arc<LlamaModel>`
    על ה-stack שלו. `LlamaContext<'a>` שואל את המודל, ולכן אחסון שלהם יחד היה
    self-referential; כך ה-borrow לא יוצא ממסגרת ה-stack וה-context ש-`!Sync` לא חוצה
    thread. ה-mutex שומר רק `Vec<usize>` של עובדים פנויים ומוחזק ל-`pop`/`push`,
    **לא** על פני decode. mutex בודד סביב context אחד היה מסרייל את כל ה-inference.
  - `tokenizer::RawVocab` — ה-`unsafe` היחיד ב-crate. `llama-cpp-2` מקדד בקשיחות
    `parse_special = true`, וחוזה הזהב מחייב `false` (מדוד: `<|endoftext|>` בתוך ספר
    שינה טקסט מ-162 ל-158 טוקנים). הפריסה היא **פרט מימוש בלתי מתועד** שהפין המדויק
    לגרסה מקפיא — upstream מכחיש יציבות פריסה במפורש לטיפוסים אחיים.
  - `truncate_with_eos()` — `max_tokens` הוא הסך **כולל** EOS. ה-EOS נדחף *אחרי*
    החיתוך, ולכן שום אורך לא יכול להדיח אותו; בלעדיו pooling של הטוקן האחרון היה
    קורא טוקן תוכן והוקטור היה חסר משמעות. חולץ לפונקציה חופשית כדי שיהיה ניתן
    לבדיקה בלי המודל בן 396MB — הכרחי, כי הסבילות הווקטורית **אינה** רואה באגי
    טרנקציה (off-by-one מקבל cosine 0.99838).
  - `micro_batch_for()` — `n_ubatch = 256` ולא `n_ctx`, מה שחוסך ~162 MiB reserve
    לכל context (טנזור logits ש-backend של embeddings לא קורא). מגודר בקאוזליות
    מוכחת: `GGML_ASSERT((causal_attn || n_ubatch >= n_tokens_all))` — במודל לא-קאוזלי
    התהליך קורס, ולא בטעינה אלא ב-batch האמיתי הראשון. 256 היא ההפחתה הגדולה ביותר
    שמשאירה את כל 65 הוקטורים זהים סיבית.
  - `release_contexts_at_exit()` — נרשם ב-`atexit` מתוך `spawn`, אחרי שה-context
    הראשון קיים. `static` אינו נהרס לעולם, ולכן host שמחזיק את המנוע ב-global היה
    מקבל `GGML_ASSERT` ב-destructor סטטי של ggml **אחרי** עבודה מוצלחת — crash
    reporter מדווח על זה כקריסה. atexit רץ בסדר הפוך לרישום, ולכן ההקדמה מובטחת.

* [`src/semantic/store.rs`](../src/semantic/store.rs)
  - `VectorStore` & `VectorStoreConfig` — מנגנון האחסון והשליפה הוקטורי.
  - **Pre-normalization**: נורמליזציה בוקטורים בעת ההכנסה המאפשרת חישוב דמיון קוסינוס בעזרת Dot Product בלבד ($O(dim)$).
  - **BinaryHeap Top-K**: שליפת $k$ התוצאות המובילות בסיבוכיות $O(N \log k)$ ללא שכפול מטא-דאטה של כל המאגר. שוויון ציונים נשבר לפי `semantic_id` — בלי זה `HashMap` עם סדר איטרציה מקרי היה מחזיר top-k שונה בכל ריצה.
  - **מנעול אחד** לשתי המפות: קודם היו שני מנעולים ש-insert ו-delete נטלו בסדר הפוך — lock-order inversion שעלול לתקוע את התהליך.
  - `is_persistent()` / `backend_id()` — ה-backend הנוכחי אינו persistent, ומצהיר על כך; ה-engine מסתמך על זה כדי לא להאמין ל-manifest ישן.
  - `dot_product()` — 8 מצברים במקום סכימה סדרתית אחת: ~1.4× מהיר במדידה
    (101ms מול 145ms על 200k וקטורים בממד 1024).

* [`src/semantic/manifest.rs`](../src/semantic/manifest.rs)
  - `SemanticManifest` — ניהול גירסאות אינדקס אטומי: כתיבה ל-`.tmp`, `fsync`, ואז
    rename. בלי ה-`fsync` הניתן להחלפה אטומית עדיין אפשר לאבד את התוכן בהפסקת חשמל.
    `load()` משחזר מה שקריסה בתוך `save` יכולה להשאיר — קודם `.previous` (manifest
    שהיה בשירות) ואחריו `.tmp` (מועמד שנשטף), שניהם רק אחרי פרסור מוצלח.
    `sync_directory()` הופך כשל ב-fsync של התיקייה לשגיאת שמירה **ב-Unix**;
    ב-Windows אין מקבילה, וזה מתועד במקום להיות שקוף.
  - `save_count()` — מספר הכתיבות של המופע הזה (`serde(skip)`). לא סטטיסטיקה: כל
    כתיבה מסריאלזת את כל הרשומות, ולכן "כמה פעמים" הוא תכונת נכונות של לופ האינדוקס,
    והדרך היחידה לאמת אותה היא לספור.
  - `failpoints` — הזרקת כשל ל-rename ול-fsync של התיקייה, thread-local. מסלול
    ה-fallback קיים בגלל נעילת קובץ ב-Windows, מצב שאי אפשר להגיע אליו בסידור קבצים;
    בלי הזרקה הוא היה קוד התאוששות שלא נבדק.
  - `validate()` — זיהוי אי-התאמות בכל עשרת הממדים: model id, checksum, embedding backend, ממדים, pooling, quantization, vector precision, vector backend, chunking ו-normalization.
  - `ManifestMismatch::invalidates_vectors()` — האם הווקטורים עצמם פסולים (מודל/ממד) או שרק צריך chunking מחדש.
  - `quarantine()` — קובץ manifest לא קריא מועבר הצידה ולא נמחק, כדי שיהיה מה לחקור.
  - `clear_books()` — מחיקת רשומות הספרים תוך שמירת המטאדאטה של הקונפיגורציה.
  - `book_index_need()` — `Missing` / `Changed` / `Unverifiable` / `UpToDate`.
    ההחלטה הזמינה בזמן diff, לפני שהשורות נטענו.
  - `BookManifestEntry.line_fingerprint` + `chunk_count = 0` כמרקר תקין —
    `clear_books_with_vectors()` מוחק רק רשומות שמצהירות על וקטורים.

* [`src/semantic/engine.rs`](../src/semantic/engine.rs)
  - `SemanticEngine` & `SemanticConfig` — המנוע הסמנטי המרכזי המאגד את ה-Chunker, ה-Runtime, ה-VectorStore וה-Manifest.
  - `SemanticConfig::validate()` — פוסל קונפיגורציה שלא תעבוד, ובראשה אי-התאמה בין
    `embedding_dim` ל-`store.embedding_dim` (שקודם התגלתה רק באמצע האינדוקס).
  - `open()` — מפייס את ה-manifest מול הקונפיגורציה: שימוש חוזר, גריעת רשומות
    שהווקטורים שלהן לא שרדו, או quarantine והתחלה מחדש. אינו נכשל בגלל manifest פגום.
  - `index_book()` / `index_books()` — האחרון כותב manifest פעם אחת לכל הקבוצה.
    שניהם **מוחקים** את הווקטורים הקודמים של הספר לפני הכתיבה, ורק אחרי שה-embedding
    הצליח — כך שכשל באמצע לא משאיר ספר בלי וקטורים עם רשומה שמצהירה שהוא מאונדקס.
  - `index_book_deferred()` + `flush_manifest()` — הפרדת ה-mutation מהשמירה, לקורא
    שמנהל את הלופ בעצמו (`HybridCoordinator::index_books` משחרר נעילה בין ספרים ולכן
    חייב את זה). מי שקורא ל-`index_book_deferred` **חייב** לקרוא ל-`flush_manifest`,
    גם במסלול השגיאה.
  - `manifest_save_count()` — עלות, לא סטטיסטיקה. קיים כדי שמספר הכתיבות של לופ
    האינדוקס ייבדק ולא יונח.
  - `reset_index()` — מסלול ההתאוששות מ-`IncompatibleIndex`.
  - `diff_against_tantivy()` — דגלי אי-התאימות אמיתיים; פלט בסדר דטרמיניסטי.

---

## 🧪 בדיקות ותשתית

* [`tests/hybrid_integration_test.rs`](../tests/hybrid_integration_test.rs)
  - בדיקות מקצה לקצה דרך ה-API הציבורי: אינדוקס ספרייה, שלושת מצבי החיפוש,
    התדרדרות חיננית, מחזור אינדוקס→הפעלה־מחדש→חיפוש, re-index שמוחק שורות שנעלמו,
    התאוששות מ-manifest פגום, אי-תאימות ו-reset, filters ו-paging, וחוזה החתימות של
    PDF: תיקון מחבר בקובץ שלא השתנה מדווח כשינוי, וחתימה שמכסה תוכן בלבד לא מוכרזת
    כמעודכנת.
  - דורש `--features mock-embedding` (אין backend inference בבנייה רגילה).
* [`tests/production_backend_gate.rs`](../tests/production_backend_gate.rs)
  - התמונה ההופכית, מתקמפל **רק בלי** ה-feature: מאמת שבנייה רגילה מסרבת לטעון
    מודל ולייצר וקטורים. זו הערובה שקוד production לא יגיש וקטורים מזויפים —
    ולכן היא נבדקת ולא נסמכת על `#[cfg]` שיישאר במקומו.
* [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)
  - תהליך CI מלא ב-GitHub Actions הרץ על Ubuntu, Windows ו-macOS, **בשתי
    קונפיגורציות features**, כולל `cargo fmt --check`, clippy עם `-D warnings`,
    ואימות קישורי תיעוד (`cargo doc`).

## מדידות

* [`benches/vector_search.rs`](../benches/vector_search.rs) — מודד את
  `VectorStore::search` המלא (לא רק dot-product) ומחלץ מכך את ההערכה לקנה מידה של
  הספרייה. תוכנית רגילה עם `harness = false`: המטרה היא מדידה שניתן לשחזר, ולא כדאי
  לגרור עץ תלויות של framework למאגר שמיועד לבנייה מובילית.

  ```bash
  cargo bench
  cargo bench -- --vectors 1000000 --dim 256
  ```

  מדפיס min/median/max, בכוונה: מספר בודד מזמין ציטוט כ"ה"מספר, ובמכונה שעושה עוד
  משהו הפער בין השלושה גדול מההפרש שמנסים למדוד. בקונפיגורציות קטנות הרעש שולט —
  אל תסיקו מהן. המספרים תלויי-מכונה: השוו ריצות על אותה מכונה בלבד.

  דוגמה למה זה אומר בפועל: תקורת ה-filters נמדדה כ-`-13.2%` ו-`+1.0%` בשתי ריצות של
  100k×512, וכ-`+22.1%`, `-0.9%`, `+15.3%` בשלוש ריצות של 20k×128. כלומר בממד מציאותי
  התקורה בתוך רעש המדידה, ובממד קטן המכונה פשוט לא מבדילה. מה שכן ודאי הוא מבני ולא
  נמדד: `VectorStore::search` מקמפל את המסננים **פעם אחת לשאילתה** ולא פעם לכל וקטור.

  **חשוב:** ל-`[[bench]]` יש `harness = false`, ולכן `cargo test --all-targets`
  *מריץ* אותו במקום רק לקמפל (בחירה מפורשת של target דורסת `test = false`).
  ה-CI מריץ `cargo test --lib --tests`, וקימפול ה-benchmark נעשה ב-release build.

## הרצה מקומית

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings                          # production
cargo clippy --all-targets --features mock-embedding -- -D warnings
cargo test --lib --tests                                           # שער ה-production
cargo test --lib --tests --features mock-embedding                  # החבילה המלאה
```
