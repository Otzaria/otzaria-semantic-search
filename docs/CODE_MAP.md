# Otzaria Hybrid Semantic Search — Code Map (מפת הקוד)

מנגנון החיפוש ההיברידי של אוצריא מורכב משתי תת-מערכות עיקריות:
1. **Semantic Subsystem (Sidecar)** — מנהלת את ה-Embedding, ה-Vector Store, ה-Chunking, וה-Manifest.
2. **Hybrid Coordinator** — מנהלת את ניתוח השאילתא, נורמליזציית הציונים, ה-Fusion (מיזוג ציוני BM25 וסמנטיקה), ה-Ranking, ה-Grouping (קיבוץ לפי קטע או טקסט זהה), וה-Fallback.

סביבן שלוש תת-מערכות תומכות שנוספו ב-PR #3: `config` (פרופילים ודגלים), `telemetry`
(מוני ריצה בתוך התהליך) ו-`distribution` (אריזה והתקנה של אינדקס מוכן).

> היקף המוצר מוגדר ב-[`PRODUCT_CONTRACT.md`](PRODUCT_CONTRACT.md). שלושה דברים שכדאי
> לדעת לפני קריאת המפה: האינדקס הרשמי נבנה מראש ונפתח read-only, ולכן API האינדוקס
> שמתואר כאן הוא **פיגום אב-טיפוס** ולא המסלול של האפליקציה; מסלול האפליקציה הוא
> [`OfficialSemanticIndex`](../src/semantic/official_index.rs), שאין עליו אינדוקס
> לקרוא; ופורמט ה-payload הוא snapshot לדיסק עם סריקה מלאה — לא ANN, לא mmap ולא
> הספרייה `zvec`.

---

## 🏛️ מבנה העץ של המאגר

```text
otzaria-semantic-search/
├── Cargo.toml                              # הגדרות תלויות והידור
├── README.md                                # מסמך ראשי ורישיון
├── docs/
│   ├── PRODUCT_CONTRACT.md                 # חוזה המוצר — גובר על כל מסמך אחר
│   ├── MODEL_DISTRIBUTION.md               # כיצד המודל מגיע למכשיר
│   ├── CODE_MAP.md                         # מפת קוד זו
│   └── DEVELOPMENT.md                      # מדריך ארכיטקטורה ופיתוח מקיף
├── .github/workflows/
│   └── ci.yml                              # CI/CD אוטומטי (מטריצת OS, backend inference, שער golden)
├── benches/
│   └── vector_search.rs                    # מדידת latency של VectorStore::search
├── tests/
│   ├── artifact_contract.rs                # זהות הארטיפקט ושער ההתקנה, דרך ה-API הציבורי בלבד
│   ├── artifact_packer.rs                  # שער הקבלה של S4a: pack/validate ב-CLI, ומה שנארז נפתח ועונה
│   ├── official_runtime.rs                 # התקנה→פתיחה→שאילתה על ארטיפקט (דורש --features mock-embedding)
│   ├── hybrid_integration_test.rs          # בדיקות מקצה לקצה (דורש --features mock-embedding)
│   └── production_backend_gate.rs          # מאמת שבנייה רגילה מסרבת לייצר embeddings
└── src/
    ├── lib.rs                              # נקודת הכניסה לספריה + חוזה המוצר
    ├── main.rs                             # CLI פיתוח (audit / smoke)
    ├── errors.rs                           # מערכת השגיאות המרכזית (thiserror)
    ├── api/
    │   ├── mod.rs                          # ייצוא רכיבי ה-API
    │   └── hybrid_search.rs                # ממשק API נקי עבור Flutter / FFI
    ├── benchmark/
    │   └── mod.rs                          # query sets, תזמון ואגרגציית אחוזונים
    ├── config/
    │   ├── profiles.rs                     # Fast/Balanced/Best + אסטרטגיית fusion
    │   └── feature_flags.rs                # דריסות נקודתיות מעל פרופיל
    ├── distribution/
    │   ├── package.rs                      # manifest של חבילה + SHA-256 לכל payload
    │   ├── importer.rs                     # התקנה בשני renames, עם שחזור מהפרעה
    │   ├── corpus.rs                       # הפורט אל האינדקס הלקסיקלי, ותמלול שלו לשני קבצים
    │   └── packer.rs                       # צד ה-build: וקטורים מוכנים → ארטיפקט מאומת
    ├── hybrid/
    │   ├── mod.rs                          # ייצוא רכיבי ה-Hybrid
    │   ├── coordinator.rs                  # מתאם החיפוש ההיברידי הראשי
    │   ├── fusion.rs                       # מיזוג ציונים (Weighted & RRF)
    │   ├── grouping.rs                     # קיבוץ תוצאות (Section & Dedup)
    │   ├── ranking.rs                      # ניתוח שאילתא וחישוב משקל אלפא
    │   ├── metadata_ranker.rs              # בונוסים מתוך facets
    │   ├── hebrew_normalizer.rs            # הסרת ניקוד/טעמים וזיהוי שפת השאילתה
    │   └── cache.rs                        # cache תוצאות עם פסילה לפי generation
    ├── semantic/
    │   ├── mod.rs                          # ייצוא רכיבי ה-Semantic
    │   ├── chunker.rs                      # Anchored Chunking & SHA256 IDs
    │   ├── embedding.rs                    # אימות GGUF, batching ונרמול
    │   ├── embedding_cache.rs              # cache לווקטורים של טקסטים שהוטמעו
    │   ├── backend.rs                      # חוזה ה-backend ובחירתו
    │   ├── llama_backend.rs                # inference אמיתי (feature `llama-backend`)
    │   ├── engine.rs                       # מתאם צד ה-build: chunk → embed → כתיבה
    │   ├── official_index.rs               # מסלול האפליקציה: פתיחת ארטיפקט מאומת, read-only
    │   ├── manifest.rs                     # מעקב גירסאות קבצים אטומי (JSON)
    │   ├── store.rs                        # Vector DB בזיכרון (Pre-normalized + Heap)
    │   ├── store_backend.rs                # שני חוזים: הקורא שהריצה מקבלת, והכותב של builder
    │   ├── zevc_store.rs                   # פורמט ה-payload: פותח כותב ופותח read-only; סריקה מלאה, לא ANN
    │   ├── versioning.rs                   # זהות הארטיפקט ודחייה מפורשת לפי שדה
    │   └── types.rs                        # הגדרות טיפוסים ומבני נתונים
    └── telemetry/
        └── mod.rs                          # מוני חיפוש בתוך התהליך (ללא רשת)
```

---

## 🧩 מודולים ורכיבים מרכזיים

### 1. נקודת הכניסה ומערכת השגיאות

* [`src/lib.rs`](../src/lib.rs)
  - מייצא את המודולים: `api`, `benchmark`, `config`, `distribution`, `errors`,
    `hybrid`, `semantic`, `telemetry`.
  - נושא את ארבע החלטות ההיקף כ-doc comment ברמת ה-crate, כדי שמי שקורא רק את הקוד
    יראה אותן גם בלי המסמכים.
* [`src/errors.rs`](../src/errors.rs)
  - `SemanticSearchError` — השגיאה הראשית המאגדת את כל תת-המערכות.
  - `EmbeddingError` — שגיאות טעינת מודל ואינפרנס.
  - `VectorStoreError` — שגיאות חיפוש, הכנסה ומחיקה ב-Vector DB.
  - `ManifestError` — שגיאות תואמות מודל וגרסאות אינדקס.
  - `SemanticSearchError::ReadOnlyIndex` — פעולה בונה שנתבקשה מארטיפקט מותקן. **לא**
    „אין אינדקס”: יש, הוא פתוח, והוא read-only. מי שיקרא את הסירוב כ„שום דבר לא
    מוגדר” יציע למשתמש לאנדקס את הספרייה — בדיוק מה שחוזה המוצר שולל.
  - `ArtifactError` — דחיית ארטיפקט רשמי: גרסת metadata זרה, זהות חסרה, אי-התאמת זהות
    (עם רשימת השדות), digest שאינו זה שפורסם, payload חסר/לא-רגיל/פגום, שם payload לא
    פורטבילי (עם הסיבה), manifest שאינו מסכים עם ה-payload, יעד התקנה פסול, והתקנה
    שנקטעה ולא הצליחה להשתחזר. כל וריאנט הוא סירוב, לא התדרדרות — וההבחנה ביניהם קיימת
    כדי שהאפליקציה תוכל להציג „לא מתאים” לעומת „פגום”, שהם שני תיקונים שונים.
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
    מחזיר `Result`: `Ok(None)` הוא „אין אינדקס סמנטי”, ושגיאה היא ארטיפקט מותקן —
    השאלה „אילו ספרים צריך לאנדקס” מניחה שהמכשיר מאנדקס.
  - `get_telemetry_snapshot()` / `reset_telemetry()` / `clear_query_cache()` —
    מוני ריצה ופסילת cache. הכול בתוך התהליך; שום דבר לא נשלח לשום מקום.
  - `index_books()` — אינדוקס ספרים (manifest נשמר פעם אחת בסוף, לא פר-ספר).
  - `remove_semantic_books()` — מיישם בקבוצה את `IndexDiff::removed_books`.
  - `reset_semantic_index()` — מסלול ההתאוששות מאינדקס לא תואם; בלעדיו
    `needs_full_reindex` היה מבוי סתום.

  > **ארבע הפעולות האחרונות הן פיגום אב-טיפוס.** לפי חוזה המוצר האפליקציה מתקינה
  > ארטיפקט מוכן ואינה מאנדקסת, ולכן ב-S5 המסלול הרשמי הוא
  > `open`/`install_official_semantic_index` ולא `semantic_index_books(Vec<...>)`.
  > נכון להיום זהו ה-API שהבדיקות והבנייה משתמשות בו, ולכן הוא מתועד ולא מוסתר —
  > וכשהמנוע נבנה מעל ארטיפקט מותקן, כל אחת מהן **נדחית בשם** ואינה מדווחת הצלחה ריקה.
  > *מה שלא יהיה כאן לעולם:* progress stream ו-cancel/resume של אינדוקס — אין
  > אינדוקס באפליקציה.

---

### 3. תת-המערכת ההיברידית (`src/hybrid/`)

* [`src/hybrid/coordinator.rs`](../src/hybrid/coordinator.rs)
  - `HybridCoordinator` — מתאם החיפוש הראשי. מריץ חיפוש סמנטי לצד מועמדי BM25, מפעיל ניתוח שאילתא, מיזוג ציונים, קיבוץ, ומבצע Fallback ל-BM25 אם ה-Semantic Engine נכשל.
  - `HybridSearchParams` — פרמטרי חיפוש (גבולות, Offset, Grouping, Filters, Force Mode).
  - `SemanticSide` — איזה אינדקס סמנטי מוגש: `Official` (ארטיפקט מותקן, read-only —
    מסלול האפליקציה) או `SelfBuilt` (`SemanticEngine`, צד ה-build והאב-טיפוס). הצד
    הקורא זהה בשניהם, ולכן החיפוש אינו יודע במה הוא מחזיק. כל פעולה בונה עוברת
    ב-accessor שרק `SelfBuilt` מקיים, ולכן ארטיפקט מותקן נדחה בשם
    (`ReadOnlyIndex`) — ולא ב-`None`, שמשמעותו „אין אינדקס סמנטי בכלל”.
  - `with_official_index()` — הבנייה של מסלול האפליקציה; `new()` נשאר מסלול ה-build.
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
    `indexing: Mutex` מסדר בתור אינדוקס, reset וגריעת ספרים. על ארטיפקט מותקן כל
    אלה נדחים לפני שנעשה משהו.

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

* [`src/hybrid/cache.rs`](../src/hybrid/cache.rs)
  - `QueryCache` — cache תוצאות עם מפתח SHA-256 של פרמטרי השאילתה, קיבולת, TTL
    ופסילה לפי `generation`: מוטציה באינדקס מקדמת דור, וכל הרשומות מהדור הקודם
    מפסיקות להיות תקפות בלי לעבור עליהן אחת-אחת.
  - `QueryCacheStats` — `hits`/`misses`/`evictions`/`size`/`generation`. ה-telemetry
    מבדיל בין `cache_lookup` ל-`cache_hit`, כדי ש"לא נבדק" לא ייראה כ"פספוס".

* [`src/hybrid/metadata_ranker.rs`](../src/hybrid/metadata_ranker.rs)
  - `MetadataRanker` — בונוסים קטנים (מקור ראשוני, התאמת דור, התאמת קטגוריה)
    שנגזרים מה-facets של התוצאה. ברירות המחדל בסדר גודל של 0.02–0.03 בכוונה: אלה
    סימני היכר, לא שינוי סדר.
  - `MetadataSignal` — הפירוק לגורמים ולא רק הסך, כדי שאפשר יהיה לדעת למה תוצאה עלתה.

* [`src/hybrid/hebrew_normalizer.rs`](../src/hybrid/hebrew_normalizer.rs)
  - `HebrewNormalizer::normalize_for_embedding()` — הסרת ניקוד וטעמים ואיחוד
    גרש/גרשיים לפני ההטמעה. אותה נורמליזציה חייבת לחול על טקסט האינדוקס ועל
    השאילתה, אחרת שני הצדדים אינם באותו מרחב.
  - `QueryLanguage` — עברית / ארמית / מעורב / אחר.

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
  - `compute_semantic_id()` — יצירת מזהה SHA256 hex יציב לפי מפתח ספר, שורה וטביעת האצבע של ה־ChunkerConfig.
  - `ChunkerConfig::identity()` — טביעת אצבע u64 של כל שדות החלוקה; נשמרת ב־manifest כזהות האינדקס.
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

* [`src/semantic/store.rs`](../src/semantic/store.rs) — **ה-store שברירת המחדל של
  המנוע פותחת** (אב-טיפוס ובדיקות; מסלול הריצה פותח ארטיפקט).
  - `VectorStore` & `VectorStoreConfig` — מנגנון האחסון והשליפה הוקטורי.
  - **Pre-normalization**: נורמליזציה בוקטורים בעת ההכנסה המאפשרת חישוב דמיון קוסינוס בעזרת Dot Product בלבד ($O(dim)$).
  - **BinaryHeap Top-K**: שליפת $k$ התוצאות המובילות בסיבוכיות $O(N \log k)$ ללא שכפול מטא-דאטה של כל המאגר. שוויון ציונים נשבר לפי `semantic_id` — בלי זה `HashMap` עם סדר איטרציה מקרי היה מחזיר top-k שונה בכל ריצה.
  - **מנעול אחד** לשתי המפות: קודם היו שני מנעולים ש-insert ו-delete נטלו בסדר הפוך — lock-order inversion שעלול לתקוע את התהליך.
  - `is_persistent()` / `backend_id()` — ה-backend הנוכחי אינו persistent, ומצהיר על כך; ה-engine מסתמך על זה כדי לא להאמין ל-manifest ישן.
  - `dot_product()` — 8 מצברים במקום סכימה סדרתית אחת: ~1.4× מהיר במדידה
    (101ms מול 145ms על 200k וקטורים בממד 1024).

* [`src/semantic/store_backend.rs`](../src/semantic/store_backend.rs)
  - `VectorSearchBackend` — הצד הקורא, וכל מה שמסלול הריצה מקבל: `backend_id`,
    `is_persistent`, `embedding_dim`, `count`, `search`, `book_keys`,
    `book_vector_count`.
  - `VectorStoreBackend: VectorSearchBackend` — מוסיף `insert_batch`,
    `remove_by_book`, `clear` ו-`commit`. זה מה ש-builder מקבל.
  - **הפיצול הוא החוזה, לא נוחות:** האפליקציה פותחת ארטיפקט שנבנה במכונה אחרת ואסור
    לה לכתוב אליו, ולכן היא מחזיקה טיפוס שאין עליו insert לקרוא. זה מונע כתיבה
    בקומפילציה, ולא בכלל שמישהו צריך לזכור.
  - `SemanticEngine` תלוי בצד הכותב כ-`Box<dyn VectorStoreBackend>`, ולכן בחירת
    ה-backend היא של הקורא (`SemanticEngine::with_store`) ולא קבועה במודול.

* [`src/semantic/zevc_store.rs`](../src/semantic/zevc_store.rs)
  - הפורמט שבו payload של ארטיפקט נכתב: `vectors.bin` (רשומות `f32` little-endian),
    `metadata.jsonl` (אובייקט לרשומה, באותו סדר, עם SHA-256 למטא-דאטה ולווקטור)
    ו-`book_index.json` (כותרת + ספר→ids). השמות חשופים כ-`SNAPSHOT_FILENAMES`, כי
    הם שמות ה-payload שה-packer חייב לכתוב.
  - `ZevcStore` — הפותח הכותב, לצד ה-build. `commit()` הוא נקודת השמירה.
  - `ReadOnlyZevcStore` — התצוגה של מסלול הריצה. ה־constructor שלו מקבל **`VerifiedPackage`
    ולא נתיב**, וגוזר ממנו את התיקייה, את רוחב הרשומה ואת ה־SHA-256 של כל קובץ: חתימה
    שמקבלת נתיב ומפת hashes הייתה מאפשרת גם לקוד פנימי עתידי לפתוח תיקייה שאיש לא אימת, או
    לצרף hashes של חבילה אחרת. מקיים `VectorSearchBackend` בלבד,
    והרשומות אינן משתנות אחרי הפתיחה — ולכן אין גם lock במסלול השאילתה. שם ה-collection
    **מאומץ** מה-payload ולא נדרש: הוא אינו חלק מזהות הארטיפקט, ואין לקורא מול מה
    להשוות אותו.
  - שניהם קוראים דרך פונקציה אחת, כדי שהבדיקות של הקורא לא ייסחפו ביניהן: גרסת
    פורמט, ממד, SHA-256 **לכל רשומה** (זה מה שתופס עריכה באותו אורך), אורך מדויק בלי
    בייטים עודפים, `semantic_id` שאינו כפול, וקטור שיש לו כיוון, ואינדקס הספרים מול
    המטא-דאטה.
  - **מה זה לא:** לא הספרייה `zvec`, לא ANN, לא mmap. הפתיחה קוראת כל בייט, מגבבת כל
    רשומה וטוענת את **כל** הווקטורים ל-`HashMap`; החיפוש סורק את כולם, `O(N·D)`. זו
    העלות של פורמט בלי אינדקס ובלי גישה עצלה — תכונה של ה-backend, לא של חוזה
    הארטיפקט — וזאת המדידה של S2b.

* [`src/semantic/versioning.rs`](../src/semantic/versioning.rs)
  - `IndexVersion` — זהות הארטיפקט בשלוש קבוצות: `CorpusIdentity` (digest של
    הספרייה, גרסת קטלוג, `tantivy_schema_version`, `document_id_scheme_version`),
    `ModelIdentity` (`model_id`, **`model_checksum`**, quantization, backend, ממד,
    pooling, `max_tokens`, `embedding_text_version`, normalization, chunking) ו-
    `StoreIdentity` (`backend_id`, `store_format_version`, `vector_precision`).
    החוזה המלא: [`ARTIFACT_CONTRACT.md`](ARTIFACT_CONTRACT.md).
  - `IdentityField::ALL` — רשימה אחת שההשוואה, ה-digest והודעות הדחייה הולכות לפיה.
    **הכיסוי אינו מובטח על ידי הטיפוס** — `IndexVersion` הוא struct רגיל — אלא על ידי שתי
    בדיקות שנגזרות מה-JSON המסוריאלי: `every_serialized_identity_field_is_comparable`
    (שדה שנשמר ואינו מושווה) ו-`every_serialized_identity_field_is_refused_when_left_unfilled`
    (שדה שוולידציית השלמות שכחה). הוספת שדה בלי וריאנט מפילה את שתיהן.
  - `library_version` **קטלני כמו כל שדה אחר**, לא „אבחון בלבד”: הערך הצפוי בא מאותו
    ארטיפקט Tantivy שממנו בא ה-`corpus_id`, ולכן אי-התאמה היא תקלה בצינור ה-build.
  - `document_id_scheme_version` — גרסה 1 היא `((catalogue_order + 1) << 32) + (ordinal + 1)`,
    בדיוק כמו `otzaria_search_engine`. הנוסחה מדויקת בכוונה: ה-builder ב-S4 צריך לשחזר
    אותה, לא לקרב אותה.
  - `mismatches_against()` / `verify_matches()` — **כל** ההבדלים ברשימה טיפוסית
    (`IdentityMismatch`), לא הראשון ולא `bool`: דחייה שהצטמקה ל-`false` היא מה
    שחוזה המוצר קורא לו ניחוש.
  - `validate_complete()` — מחרוזת ריקה, גרסה 0, checksum שאינו 64 hex קטנות, או תו בקרה
    בתוך ערך זהות — נדחים **לפני** ההשוואה, כי שתי זהויות שלא מולאו משוות שוות זו לזו.
    תו בקרה נדחה גם כדי שהטקסט הקנוני שמאחורי ה-digest יישאר חד-משמעי.
  - כל שדה כאן קטלני: אין „אי-תאימות שדורשת רק chunking מחדש” כמו ב-manifest המקומי,
    כי במכשיר אין מה לבנות מחדש.

* [`src/semantic/embedding_cache.rs`](../src/semantic/embedding_cache.rs)
  - `EmbeddingCache` — cache בגודל חסום לווקטורים של טקסטים שהוטמעו, עם החלפה
    לפי שעון גישה. חוסך inference על שאילתות חוזרות בלבד; אינו נוגע באינדקס.

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
  - `SemanticEngine` & `SemanticConfig` — מנוע **צד ה-build**: מאגד את ה-Chunker,
    ה-Runtime, store כותב וה-Manifest. מסלול האפליקציה הוא `official_index.rs`.
  - `with_store()` — פתיחה מעל backend שהקורא מספק. זה מה שהופך ריצת אינדוקס למשהו
    שאפשר לארוז ממנו ארטיפקט: עם store מתמיד הווקטורים שורדים restart. ה-manifest
    רושם את ה-backend שנפתח **בפועל**, ולכן פתיחה מחדש עם backend אחר היא אי-תאימות
    מדווחת ולא תשובה מ-store ריק.
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

* [`src/semantic/official_index.rs`](../src/semantic/official_index.rs) —
  **מסלול האפליקציה.**
  - `OfficialSemanticIndex::open()` — הצרכן של `VerifiedPackage`. הסדר כפוי ולא נבחר:
    recovery של התקנה שנקטעה → טעינת המודל → אימות → פתיחת ה-payload **מהטוקן**.
    המודל קודם לאימות מפני ש-`model_checksum` ו-`embedding_backend` הם שני שדות זהות
    שאף ארטיפקט אינו יכול לספק; לכן אי-התאמה עולה טעינת מודל אחת, וההשוואה נשארת
    במקום אחד.
  - הזהות מורכבת משלושה מקורות שכל אחד יודע חלק ממנה: **corpus** מ-Tantivy שהקורא
    פתח, **model** מהקובץ הנטען + המתכון שה-build מממש (`LocalModel`), ו-**store**
    ממה ש-build הזה יודע לקרוא (`readable_store_identity`). ה-crate אינו ממציא אף
    אחד מהם.
  - `verify_counts_against_payload()` — הבדיקה שרק קורא יכול לעשות: `vector_count`
    הוא מספר הרשומות, `book_count` מספר ה-`source_book_key` הנפרדים. זו ההגדרה
    שה-packer (S4a) חייב לעמוד בה.
  - `search()` / `status()` — `status` מדווח `vectors_persisted = true` ו-
    `needs_full_reindex = None`, ולא כטענה ריקה: ארטיפקט הוא או הנכון או נדחה בפתיחה,
    ואין במכשיר מה לבנות מחדש.
  - `LocalModel` — מה שההתקנה **מצהירה**: נתיב, `model_id`, quantization, ממד,
    pooling, `max_tokens`, ושלוש גרסאות המתכון. הקורא אינו גוזר את השלוש האחרונות —
    שאילתה אינה עוברת chunking — אבל הן מושוות, כי ארטיפקט שנבנה ממתכון אחר הוא
    ארטיפקט אחר, והתוצאה תהיה סבירה למראה ושגויה.
  - **הכתיבה היחידה שיש כאן** היא recovery של ההתקנה: rename של תיקיות שה-importer
    השאיר, לא נגיעה ב-payload, ובמצב נקי — כלום. `recovery()` מדווח מה נמצא.

---

### 5. תת-מערכות תומכות

* [`src/config/profiles.rs`](../src/config/profiles.rs)
  - `SearchProfile` — `Fast` / `Balanced` / `Best`.
  - `RankingProfile` — כל פרמטרי הכיול במקום אחד (thresholds, בונוסים, קיבולות
    cache, אסטרטגיית fusion). מקור אמת יחיד, כדי שלא יהיו שתי קבוצות ברירות מחדל.
  - `FusionStrategy` — `Weighted` / `RRF { k }` / `Adaptive`.

* [`src/config/feature_flags.rs`](../src/config/feature_flags.rs)
  - `FeatureFlags` — כל שדה הוא `Option`, ולכן „לא צוין” נבדל מ„צוין כברירת המחדל”.
  - `apply()` — דורס פרופיל קיים במקום להחזיק העתק שני שלו. ערכים לא-חוקיים
    (`NaN`, מחוץ לטווח) נבלמים ולא נכנסים לפרופיל.

* [`src/telemetry/mod.rs`](../src/telemetry/mod.rs)
  - `SearchTelemetry` — רשומה לשאילתה: סוג שאילתה, מצב שרץ, אסטרטגיה, alpha,
    ספירות מועמדים, cache, latency (כולל embedding ו-fusion בנפרד) ופרופיל.
  - `TelemetryCollector` / `TelemetrySnapshot` — אגרגציה thread-safe.
  - **אין כאן רשת.** אלה מונים בזיכרון התהליך; המאגר אינו שולח דבר לשום שרת.

* [`src/distribution/package.rs`](../src/distribution/package.rs)
  - `PackageManifest` — `metadata_version` + `IndexVersion` + `created_at` + ספירות
    ספרים/וקטורים + גודל מוצהר. `metadata_version` נקרא ב-probe **לפני** המסמך, כדי
    שפורמט זר ידווח על גרסתו ולא ייפול על שגיאת פרסור של שדה בודד.
  - `verify_for_install()` / `verify_for_open()` — שני עומקים, כי אחד מהם רץ בכל עלייה
    של האפליקציה. שניהם: גרסת metadata, שלמות הזהות, הזהות מול ההתקנה, ה-digest המפורסם,
    ונוכחות+גודל של כל payload. רק הראשון מגבב כל בייט. גיבוב גיגה-בייטים בכל פתיחה אינו
    בתקציב, ובדיקה שאי אפשר להרשות היא בדיקה שמכבים.
  - `VerificationDepth` + `VerifiedPackage::depth()` — הטוקן נושא **מה נבדק בו**. קורא
    שמדווח „מאומת” בלי להסתכל בזה טוען שה-payload גובב כשאולי רק נעשה עליו `stat`.
    מה שהעומק הרזה אינו תופס: עריכה באותו אורך בדיוק. זו בדיקה של קורא ה-store, שמאמת
    SHA-256 לכל רשומה — ושתי בדיקות מתעדות את שני צידי הגבול.
  - `digest()` — SHA-256 מעל טקסט קנוני (גרסת metadata, כל שדות הזהות בסדר `ALL`, ספירות,
    גודל, ו-checksum+גודל לכל payload). זה **עוגן האמון**: `payloads.json` נוסע בתוך החבילה,
    ולכן payload שהוחלף יחד עם ה-checksum שלו עובר כל בדיקה פנימית. רק digest שפורסם
    מחוץ לחבילה מבדיל בין הארטיפקט הרשמי לחבילה עקבית-עם-עצמה. מעל טקסט קנוני ולא מעל
    בייטי JSON, כדי שכתיבה מחדש של ה-metadata בהתקנה לא תשנה אותו; `created_at` מוחרג.
  - `ArtifactExpectation` — `with_published_digest` מול `without_published_digest`. אין
    ברירת מחדל שקטה: מי שמוותר על העוגן קורא לפונקציה ששמה אומר זאת.
  - `VerifiedPackage` — טוקן שאין לו constructor ציבורי אחר, ו-`OfficialSemanticIndex`
    מקבל אותו במקום נתיב. „לאמת לפני שנוגעים בווקטורים” הוא לכן תכונה של הטיפוסים.
  - `verify_integrity()` / `walk_payloads()` — מהלך אחד לשני העומקים, כדי שהבדיקה הזולה
    לא תפסיק בשקט לכסות משהו שהיקרה כן מכסה. ספירות אפס נדחות; ההשוואה מול **תוכן**
    ה-payload נעשית בקורא (`OfficialSemanticIndex`), כי היא דורשת פורמט store.
  - `IndexPackage::write()` — מסרב לכתוב metadata שהקורא היה דוחה (זהות חסרה, payload
    חסר, גודל שאינו מסתכם). חבילה שנכתבה „בהצלחה” בלי לאמת היא בדיוק זו שתיכשל אצל
    המשתמש.
  - `validate_payload_name()` — allowlist על ה**מחרוזת**: `A-Z a-z 0-9 . _ -`, עד 255
    בתים, בלי נקודה בהתחלה/בסוף, לא שמות ה-metadata, ולא שם מכשיר שמור של Windows.
    **לא דרך `Path`** — `Path::components` מפרש `a\b.bin` כשם קובץ אחד ב-Unix וכנתיב
    ב-Windows, כך שחבילה שנכתבה ב-macOS הייתה יכולה להיפרש אחרת ב-Windows. symlink או
    משהו שאינו קובץ רגיל נדחה, לא נעקב.
  - `write_and_sync()` / `sync_dir()` — כתיבת metadata עם `fsync`, ושטיפת רשומת התיקייה
    (Unix; ב-Windows אין מקבילה ווה מתועד). בלי זה הפסקת חשמל מבטלת כתיבה שדווחה כהצלחה.

* [`src/distribution/importer.rs`](../src/distribution/importer.rs)
  - `IndexImporter::import()` — אימות **מלא של המקור** לפני שמועתק משהו, העתקה ל-staging
    עם `fsync` לכל קובץ, אימות **שוב על ה-staging** (הכתיבה מגַבּבת מחדש את מה שהועתק),
    ואז ההחלפה. חבילה שתידחה אינה יוצרת תיקיית יעד.
  - **אין יותר `verify_checksums: bool`.** דגל שמדלג על אימות הוא בדיוק החור שהחוזה
    אוסר; התקנה היא הרגע שבו קריאת כל בייט עוד זולה.
  - `swap_into_place()` — היעד עובר ל-`.<name>.previous`, ה-staging נכנס במקומו, ורשומת
    תיקיית האב נשטפת אחרי כל rename. **יש חלון שבו אין תיקיית יעד** — `rename` מסרב
    להחליף תיקייה לא-ריקה בשתי המערכות, ולכן ההחלפה היא בהכרח שני renames. אי אפשר לבטל
    את החלון; אפשר להפוך אותו למזוהה.
  - `recover_interrupted_install()` — בגלל אותו חלון. שמות דטרמיניסטיים (`.previous`,
    `.staging`) ולא nonce, כדי שיהיה מה למצוא: `previous` בלי target = קריסה בתוך החלון,
    העותק הקודם מוחזר; `previous` **וגם** target = ההחלפה הצליחה והניקוי לא, המיושן נמחק.
    `import` קורא לו לפני שהוא נוגע ביעד, כי כתיבה מעל הפרעה לא-פתורה מוחקת את העותק
    היחיד שיש למכשיר. מי שפותח את היעד חייב לקרוא לו לפני הפתיחה.
  - כשל בהחזרה **אינו נבלע**: `InterruptedInstall` אומר באיזו תיקייה נמצא העותק הטוב.
  - `failpoints` — הזרקת כשל ל-rename, thread-local. „החלפה שנכשלה” ו„גם ההחזרה נכשלה”
    אינם מצבים שאפשר לייצר בסידור קבצים, ומסלול התאוששות שלא נבדק הוא זה שייכשל כשיידרש.
  - מסרב שהיעד יהיה תיקיית החבילה או צאצא שלה — ייבוא כזה היה מוחק את המקור.
  - **מה שמחוץ להיקף ומתועד:** שתי התקנות במקביל לאותו target. אין lock.
  - `OfficialSemanticIndex::open` קורא ל-recovery בעצמו, ולכן מסלול הריצה נכון מעצם
    מבנהו ולא בזכות סדר קריאות שהקורא זוכר.
  - **מה שאינו כאן:** ה-importer אינו חשוף דרך `OtzariaHybridEngine`, ה-FFI או
    אוצריא. ההתקנה עצמה עדיין אינה מופעלת מהאפליקציה — זה S5 ו-S6.

* [`src/distribution/corpus.rs`](../src/distribution/corpus.rs) — **הפורט אל האינדקס
  הלקסיקלי.**
  - `CorpusIndex` — `identity()` ו-`line(line_id)`. ה-packer מקבל את זה ולא נתיב, כי
    Tantivy אינו תלות של ה-crate הזה ואסור שיהיה: האינדקס, הסכמה וסכמת ה-IDs חיים
    ב-`otzaria_search_engine`.
  - **שתי תוצאות, ושתיהן העיקר:** זהות ה-corpus נקראת מהאינדקס ולא מוקלדת ליד
    הווקטורים, וכל שדה ברשומה נגזר מהקורפוס — ולכן אין תיאור שני של ספר שיכול להיפרד
    מהראשון.
  - `CorpusLine` — בדיוק השדות שרשומה נושאת, פחות השלושה שהצד הסמנטי גוזר
    (`semantic_id`, `source_doc_key`, `chunk_hash`), ועוד ה-`text` שממנו הווקטור נבנה.
    הטקסט **אינו** נשמר בארטיפקט; הוא מה שמוכיח שהווקטור שייך לשורה הזאת.
  - `JsonlCorpus` — תמלול לשני קבצים (`identity.json` + `lines.jsonl`), שמאפשר להריץ
    packer בלי Tantivy. **תמלול, לא מקור אמת:** הוא אמין בדיוק כמו מי שכתב אותו, וה-join
    המחייב הוא זה שמימוש מעל אינדקס חי מבצע. הוא גם מחזיק כל שורה בזיכרון — כמו ה-store
    שהוא מזין, וזאת אותה מדידה של S2b.

* [`src/distribution/packer.rs`](../src/distribution/packer.rs) — **צד ה-build (S4a).**
  - `pack()` — וקטורים מוכנים → ארטיפקט. הסדר כפוי: זהות שלמה → יעד פנוי → בדיקת כל
    וקטור מול הקורפוס → commit → metadata → `validate_artifact`. שום דבר אינו נוגע בדיסק
    לפני ה-commit, ולכן דחייה משאירה תיקייה ריקה ואפשר פשוט לחזור על הריצה.
  - **הקלט הוא `line_id`, וקטור ו-SHA-256 של טקסט השורה.** ה-digest הוא מה שהופך את הכלי
    ממחתמת גומי לשער: קובץ וקטורים שנסע בשורה אחת מול רשימת ה-IDs עובר כל בדיקה אחרת
    שקיימת — הספירות מסתדרות, ה-checksums עוברים, וכל תוצאה תהיה שורה שכנה. מה שאי אפשר
    להוכיח: שהיצרן חישב את ה-digest כשהטמיע ולא מהקורפוס בזמן האריזה. זה נאמר, לא מוסתר.
  - `validate_artifact()` — האימות של הריצה (`verify_for_install`, פריסת payload, פתיחת
    `ReadOnlyZevcStore`, `verify_counts_against_payload` — **אותן פונקציות**, לא מימוש שני)
    ועוד ה-join: כל רשומה מושווית שדה־שדה למה שהקורפוס אומר על השורה שלה. חשוף גם לבדו,
    ולכן הוא עונה על „האם התיקייה הזאת שייכת לקורפוס ולמודל האלה” לארטיפקט שלא נבנה כאן.
  - `record_fields()` — טבלת שדות אחת ששני הצדדים נמדדים לפיה, ובדיקה שמשווה את השמות
    שבה ל-JSON של הרשומה: שדה שיתווסף בלי שורה כאן ייכתב לכל ארטיפקט ולא יושווה לדבר.
  - `read_vector_inputs()` — הזרמה של שני קבצי הקלט: `f32` little-endian בלי כותרת,
    ולצידו JSONL באותו סדר. ההצמדה מיקומית, ולכן **שתי** צורות אי-ההתאמה נתפסות: רשומות
    עודפות נגמרות באמצע וקטור, ורשומות חסרות משאירות בייטים — וזה מדווח בסוף ולא נבלע.
  - `store` הוא `readable_store_identity()` ולא בחירה של הקורא: packer שכותב פריסה
    שהריצה שלו אינה יודעת לקרוא מייצר ארטיפקטים לאף אחד.

* [`src/benchmark/mod.rs`](../src/benchmark/mod.rs)
  - `measure()` / `aggregate()` / `QuerySet` / `BenchmarkConfig` — תזמון, אחוזונים
    והערכת תפוקה סדרתית. הקורא מספק את סגירת החיפוש ואת ה-corpus.
  - זה **כלי מדידה**, לא dataset של רלוונטיות תורנית ולא הוכחת סקייל. dataset
    האיכות הוא S1.

---

## 🧪 בדיקות ותשתית

* [`tests/hybrid_integration_test.rs`](../tests/hybrid_integration_test.rs)
  - בדיקות מקצה לקצה דרך ה-API הציבורי: אינדוקס ספרייה, שלושת מצבי החיפוש,
    התדרדרות חיננית, מחזור אינדוקס→הפעלה־מחדש→חיפוש, re-index שמוחק שורות שנעלמו,
    התאוששות מ-manifest פגום, אי-תאימות ו-reset, filters ו-paging, וחוזה החתימות של
    PDF: תיקון מחבר בקובץ שלא השתנה מדווח כשינוי, וחתימה שמכסה תוכן בלבד לא מוכרזת
    כמעודכנת.
  - דורש `--features mock-embedding` (אין backend inference בבנייה רגילה).
* [`tests/official_runtime.rs`](../tests/official_runtime.rs)
  - מסלול הריצה הרשמי דרך ה-API הציבורי בלבד, כמו שה-`otzaria_search_engine` יראה
    אותו: בניית ארטיפקט כמו שה-packer יבנה (ה-store כותב את ה-payload, ואז ה-metadata
    מתאר אותו), התקנה, פתיחה, ושאילתה שמחזירה את ה-`line_id` שממנו נבנה הווקטור —
    ב-`SemanticOnly` וב-`Hybrid`. בנוסף: כל פעולה בונה נדחית בשם והארטיפקט אינו משתנה,
    ופתיחה חוזרת (restart) מחזירה את אותה תשובה בלי לבנות דבר.
  - דורש `--features mock-embedding`.
* [`tests/artifact_packer.rs`](../tests/artifact_packer.rs)
  - שער הקבלה של S4a, דרך הבינארי שצינור build באמת מריץ: `pack` הופך קובץ וקטורים
    לארטיפקט מאומת, `validate` מבסס את אותו הדבר על תיקייה שהוא לא בנה, ושתי הריצות
    מדווחות את אותו digest. ובצד השני — קלט שבו הווקטורים והמזהים נסעו זה מול זה נדחה
    בקוד יציאה ובהודעה, ולא נשארת תיקייה.
  - הבדיקה שמצדיקה את כל השאר: מה שה-packer כתב עובר `IndexImporter`,
    `OfficialSemanticIndex` ושאילתה — בלי fixture שנבנה ביד באמצע. עד עכשיו „ה-packer
    כותב את מה שהקורא קורא” לא הייתה טענה שמשהו היה נכשל אם תפסיק להיות נכונה.
    החלק הזה דורש `--features mock-embedding` (שאילתה צריכה להטמיע); שאר הקובץ רץ
    בבנייה רגילה, כי אריזה אינה מטמיעה דבר.
* [`tests/artifact_contract.rs`](../tests/artifact_contract.rs)
  - חוזה הארטיפקט מבחוץ: התקנה ואימות חוזר, digest מפורסם מול חבילה עקבית-עם-עצמה,
    שחזור מקריסה בין שני ה-renames, דחייה לפי שם שדה, והפרדת „פגום” מ„לא תואם”.
    ה-payload שם הוא בייטים חסרי משמעות בכוונה — מי שקורא אותו הוא `official_runtime`.
* [`tests/production_backend_gate.rs`](../tests/production_backend_gate.rs)
  - התמונה ההופכית, מתקמפל **רק בלי** ה-feature: מאמת שבנייה רגילה מסרבת לטעון
    מודל ולייצר וקטורים. זו הערובה שקוד production לא יגיש וקטורים מזויפים —
    ולכן היא נבדקת ולא נסמכת על `#[cfg]` שיישאר במקומו.
* [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)
  - תהליך CI מלא ב-GitHub Actions הרץ על Ubuntu, Windows ו-macOS, **בשתי
    קונפיגורציות features**, כולל `cargo fmt --check`, clippy עם `-D warnings`,
    ואימות קישורי תיעוד (`cargo doc`).
  - job נפרד ל-`llama-backend` (Linux + macOS), ו-job **Golden Vectors** שמריץ את
    שער ה-parity מול המודל האמיתי. השער דורש את הסוד `OTZARIA_HF_TOKEN`, וכשהוא
    חסר הוא נכשל במפורש ולא מדווח דילוג כהצלחה. ראו
    [`MODEL_DISTRIBUTION.md`](MODEL_DISTRIBUTION.md) §5.

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

## אריזת ארטיפקט (S4a)

```bash
cargo run --release -- pack \
  --vectors vectors.f32 --records vectors.jsonl \
  --corpus-identity corpus-identity.json --corpus-lines corpus-lines.jsonl \
  --model model.json --out ./artifact

cargo run --release -- validate \
  --artifact ./artifact \
  --corpus-identity corpus-identity.json --corpus-lines corpus-lines.jsonl \
  --model model.json
```

* `vectors.f32` — `f32` little-endian, `מספר_וקטורים × embedding_dim`, בלי כותרת.
* `vectors.jsonl` — שורה לכל וקטור, **באותו סדר**:
  `{"line_id": N, "line_sha256": "..."}`. ה-`line_sha256` הוא SHA-256 של טקסט השורה
  בקורפוס, ב-hex קטן; בלעדיו קובץ וקטורים שנסע בשורה אחת היה נארז בלי תלונה.
* `corpus-lines.jsonl` — `{"line_id": N, "source_book_key": ..., "title": ...,
  "reference": ..., "section_id": N, "segment": N, "is_pdf": bool, "line_hash": N,
  "content_hash": N, "facets": [...], "text": "..."}` לכל מסמך.
* `model.json` — `ModelIdentity` (ראו [`versioning.rs`](../src/semantic/versioning.rs)).

שתי הפקודות עובדות ב**בנייה רגילה**: אריזה אינה הופכת טקסט לווקטור, ולכן היא אינה
דורשת backend inference. ה-`Digest` שמודפס הוא מה שצריך להתפרסם **מחוץ** לארטיפקט —
בלעדיו אימות מזהה נזק וארטיפקט לא נכון, ולא ארטיפקט שנבנה מחדש בכוונה.
