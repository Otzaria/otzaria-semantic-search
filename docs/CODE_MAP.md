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
> לקרוא; והווקטורים הרשמיים הם סט של segments בפורמט `.oxv` — int8, ממופים לזיכרון,
> ממוענים לפי הטקסט שהוטמע — שנסרקים במלואם בחשבון שלמים מדויק, בלי ANN
> ([`ARTIFACT_CONTRACT.md`](ARTIFACT_CONTRACT.md)).

---

## 🏛️ מבנה העץ של המאגר

```text
otzaria-semantic-search/
├── Cargo.toml                              # הגדרות תלויות והידור
├── README.md                                # מסמך ראשי ורישיון
├── docs/
│   ├── PRODUCT_CONTRACT.md                 # חוזה המוצר — גובר על כל מסמך אחר
│   ├── ARTIFACT_CONTRACT.md                # שחרור הווקטורים, פורמט ה-segment, הסט על המכשיר, הזהות
│   ├── MODEL_DISTRIBUTION.md               # כיצד המודל מגיע למכשיר
│   ├── CODE_MAP.md                         # מפת קוד זו
│   └── DEVELOPMENT.md                      # מדריך ארכיטקטורה ופיתוח מקיף
├── .github/workflows/
│   └── ci.yml                              # CI/CD אוטומטי (מטריצת OS, backend inference, שער golden)
├── benches/
│   └── vector_search.rs                    # מדידת latency: VectorStore::search, ועם --store oxv סריקת segment
├── tests/
│   ├── artifact_contract.rs                # זהות הארטיפקט ושער ההתקנה, דרך ה-API הציבורי בלבד
│   ├── artifact_builder.rs                 # שער הקבלה של S4b: build ב-CLI, בנייה משוחזרת, ומה שנבנה מותקן ועונה
│   ├── official_runtime.rs                 # בנייה→התקנה→פתיחה→שאילתה דרך resolver (דורש --features mock-embedding)
│   ├── vector_set_scale.rs                 # הסט בקנה מידה של הספרייה (#[ignore], OTZARIA_SCALE_N)
│   ├── hybrid_integration_test.rs          # בדיקות מקצה לקצה (דורש --features mock-embedding)
│   └── production_backend_gate.rs          # מאמת שבנייה רגילה מסרבת לייצר embeddings
└── src/
    ├── lib.rs                              # נקודת הכניסה לספריה + חוזה המוצר
    ├── main.rs                             # CLI פיתוח (audit / smoke)
    ├── errors.rs                           # מערכת השגיאות המרכזית (thiserror)
    ├── cancellation.rs                     # CancellationToken: ביטול שאילתה שאיש אינו ממתין לה
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
    │   ├── builder.rs                      # צד ה-build: קורפוס + מודל → base segment, החבילה ו-release.json
    │   ├── corpus.rs                       # הפורט אל האינדקס הלקסיקלי, ותמלול שלו לשני קבצים
    │   └── shard.rs                        # export_plan / embed_shard / verify_shards: בנייה על מכונות נפרדות
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
    │   ├── chunk_key.rs                    # ChunkKey: SHA-256 של הטקסט המוטמע, 16 בתים — הכתובת של וקטור
    │   ├── chunker.rs                      # Anchored Chunking; embedded_text ו-chunk_keys על חלון שורות
    │   ├── recipe.rs                       # גרסאות המתכון: chunking, טקסט (כולל תחיליות תפקיד), נרמול
    │   ├── embedding.rs                    # בדיקות התצורה, batching ונרמול
    │   ├── model_package.rs                # נתיב המודל: גרף ONNX או דחייה; החבילה: אימות ו-checksum
    │   ├── embedding_cache.rs              # cache לווקטורים של טקסטים שהוטמעו
    │   ├── backend.rs                      # חוזה ה-backend ובחירתו
    │   ├── onnx_backend.rs                 # inference אמיתי ל-ONNX (feature `onnx-backend`) — ה-backend היחיד
    │   ├── engine.rs                       # מתאם צד ה-build: chunk → embed → כתיבה
    │   ├── official_index.rs               # מסלול האפליקציה: סט מותקן + מודל, read-only; מחזיר hits
    │   ├── resolve.rs                      # VectorHit ו-CandidateResolver: הפורט אל השורות החיות של המארח
    │   ├── oxv/                            # פורמט ה-segment: format, codec, writer, reader, kernel, scan
    │   ├── segment_set/                    # הסט על המכשיר: install, compact, recover, GC, scrub
    │   ├── manifest.rs                     # מעקב גירסאות קבצים אטומי (JSON)
    │   ├── store.rs                        # Vector DB בזיכרון למסלול הפיתוח (Pre-normalized + Heap)
    │   ├── store_backend.rs                # שני חוזים למסלול הפיתוח: חיפוש, ומוטציות של אינדוקס
    │   ├── versioning.rs                   # זהות הארטיפקט ודחייה מפורשת לפי שדה
    │   └── types.rs                        # הגדרות טיפוסים ומבני נתונים
    └── telemetry/
        └── mod.rs                          # מוני חיפוש בתוך התהליך (ללא רשת)
```

---

## 🧩 מודולים ורכיבים מרכזיים

### 1. נקודת הכניסה ומערכת השגיאות

* [`src/lib.rs`](../src/lib.rs)
  - מייצא את המודולים: `api`, `benchmark`, `cancellation`, `config`, `distribution`,
    `errors`, `hybrid`, `semantic`, `telemetry`.
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
    שנקטעה ולא הצליחה להשתחזר; ובסט וקטורים — `DeltaDoesNotApply { field, reason }`
    (פער, חפיפה או epoch אחר) ו-`InsufficientSpace { needed, available }`. כל וריאנט הוא
    סירוב, לא התדרדרות — וההבחנה ביניהם קיימת כדי שהאפליקציה תוכל להציג „לא מתאים”
    לעומת „פגום”, שהם שני תיקונים שונים.
  - `SemanticSearchError::Resolution { reason }` — ה-resolver של המארח לא הצליח לקרוא
    את האינדקס שלו; `ResolveError::Cancelled` מומר ל-`Cancelled`.
  - `ChunkingError` — שגיאות חלוקת ספר לקטעים.
  - `SemanticSearchError::InvalidRankingParameter` — פרמטר דירוג שהועבר עם חיפוש ואינו
    בטווח (`NaN`, שלילי, מחוץ לתחום), עם שם השדה (`alpha_by_query_type.short`). נדחה לפני
    שהחיפוש רץ, ולא מקוצץ למשהו שאיש לא ביקש.
  - `SemanticSearchError::Cancelled` — החיפוש בוטל דרך ה-`CancellationToken` שלו. **לא**
    כשל: המארח ביקש זאת, כי שאילתה חדשה החליפה את הישנה. `VectorStoreError::Cancelled`
    של סריקה שנעצרה מומר אליו בכל שכבה (`From` כתוב ידנית, לא `#[from]`), כדי שמי שבודק
    `Cancelled` לא יפספס ביטול שנעטף כשגיאת store.
* [`src/cancellation.rs`](../src/cancellation.rs)
  - `CancellationToken` — `Arc<AtomicBool>`: `Clone` זול, `Send + Sync`, `cancel()` חד-כיווני
    ו-`is_cancelled()`. חיפוש לכל הקשה: כל שאילתה מלבד האחרונה מתיישנת לפני שהיא מסתיימת,
    וסריקה מלאה של סט בגודל הספרייה אורכת עשרות מילישניות.
  - נקודות הבדיקה: לפני הכול (לפני שני ה-caches ולפני ה-embedding), אחרי ה-embedding,
    בתוך כל סריקת וקטורים כל `SCAN_CHECK_INTERVAL` (1,024) רשומות, לפני ה-fusion ואחריו.
    1,024 נבחר במדידה: בדיקה היא טעינה אטומית אחת (~ננו-שנייה), רשומה עולה ~180 ns
    ב-256 ממדים ו-~370 ns ב-1,024, ובבנצ'מרק הסריקות נמדדו זהות עם הבדיקות ובלעדיהן.
    ביטול נקלט תוך 0.2–0.4 ms של סריקה. בסריקת int8 של segment ‏slot עולה ~19 ns, וכל
    חוט קולט ביטול תוך כ-20 µs.
  - חיפוש שבוטל אינו נרשם ב-log, אינו מתדרדר לתוצאות לקסיקליות ואינו משאיר דבר: לא
    תוצאה ב-cache, לא embedding ב-cache (שניהם נכתבים רק אחרי נקודת הבדיקה האחרונה) ולא
    רשומת telemetry.

---

### 2. ממשק ה-API עבור Flutter / FFI (`src/api/`)

* [`src/api/hybrid_search.rs`](../src/api/hybrid_search.rs)
  - `OtzariaHybridEngine` — Wrapper ראשי הניתן לחשיפה ל-Flutter באמצעות `flutter_rust_bridge`.
  - `SearchRequest` — Struct המאגד את פרמטרי השאילתא והפילטרים למניעת `too_many_arguments`.
  - `search_cancellable()` — כמו `search()`, עם `CancellationToken`. מחזיר את
    `SemanticSearchError` עצמו ולא את הודעתו, כי את `Cancelled` צריך להבחין בהתאמה ולא
    בפענוח מחרוזת. `search()` נשאר כשהיה — טוקן שאיש אינו מבטל.
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
  > אינדוקס באפליקציה. ביטול *שאילתה* (`search_cancellable`) הוא עניין אחר: הוא מה שמאפשר
  > חיפוש לכל הקשה.

---

### 3. תת-המערכת ההיברידית (`src/hybrid/`)

* [`src/hybrid/coordinator.rs`](../src/hybrid/coordinator.rs)
  - `HybridCoordinator` — מתאם החיפוש הראשי. מריץ חיפוש סמנטי לצד מועמדי BM25, מפעיל ניתוח שאילתא, מיזוג ציונים, קיבוץ, ומבצע Fallback ל-BM25 אם ה-Semantic Engine נכשל.
  - `HybridSearchParams` — פרמטרי חיפוש (גבולות, Offset, Grouping, Filters, Force Mode), ו-`ranking`:
    `RankingProfile` שלם לחיפוש הזה במקום ה-preset, שנבדק ב-`validate()` לפני שמשהו רץ.
  - `SemanticSide` — איזה אינדקס סמנטי מוגש: `Official` (סט וקטורים מותקן, read-only —
    מסלול האפליקציה) או `SelfBuilt` (`SemanticEngine`, מסלול הפיתוח). `SelfBuilt` מחזיק את
    ה-metadata של השורות ועונה לבדו; `Official` מחזיר hits — מפתח ורשומות — שה-resolver של
    המארח קושר לשורות חיות. כל פעולה בונה עוברת ב-accessor שרק `SelfBuilt` מקיים, ולכן סט
    מותקן נדחה בשם (`ReadOnlyIndex`) — ולא ב-`None`, שמשמעותו „אין אינדקס סמנטי בכלל”.
  - `with_official_index()` — הבנייה של מסלול האפליקציה; `new()` נשאר מסלול ה-build.
  - **שלושת המצבים ממומשים**: `LexicalOnly` אינו נוגע במסלול הסמנטי, `SemanticOnly`
    מזניח את מועמדי BM25 שהועברו, ו-`Hybrid` מתדרדר ל-`LexicalOnly` כשהסמנטי נכשל.
    ה-`alpha` נקבע לפי המצב שרץ בפועל (1.0 / 0.0 / דינמי), כדי שציון ממנוע אחד
    לא יוקטן במשקל של המנוע החסר.
  - כל התדרדרות נראית: `search_mode` הוא המצב שרץ, `fallback_reason` הוא הסיבה.
  - `search_cancellable(query, lexical, params, resolver, cancel)` — מחזיר `Cancelled` מכל
    מצב, ולעולם אינו מתדרדר ל-BM25: לתוצאות הלקסיקליות אין מי שממתין. `search()` הוא אותה
    קריאה עם `NoResolver` וטוקן שאיש אינו מבטל. בצד `Official`: `admissible_books` →
    סריקה בספרים האלה → checkpoint → `resolve` → `SemanticCandidate` שכל id, סעיף,
    `line_hash` ו-facet בו של האינדקס החי. resolver שנכשל הוא כשל סמנטי ככל כשל: החיפוש
    נשאר עם התוצאות הלקסיקליות ו-`fallback_reason`.
  - מפתח מטמון השאילתות כולל את `resolver.generation()` ואת הדור של הסט, כי לחיפוש
    `SemanticOnly` אין קלט לקסיקלי שישתנה עם commit לאינדקס.
  - המיזוג הוא על `(file_path, line_id)` ולא על ה-id לבדו: אינדקס שעודכן ספר אחר ספר
    יכול לתת לשני ספרים אותו טווח ids.
  - `reload_semantic_vectors()` — פותח את הדור ש-`CURRENT` מונה, עם אותו מודל, ומנקה את
    מטמון השאילתות; `vector_set_info()` מדווח על הסט.
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
    `compute_alpha_with()` הוא אותו חישוב מטבלת `QueryTypeAlphas` של הפרופיל — מה שה-coordinator משתמש בו.
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

* [`src/semantic/recipe.rs`](../src/semantic/recipe.rs)
  - `EmbeddingTextRecipe` — אילו טקסט מגיע למודל, משני הצדדים. גרסה 1: השורה או השורה
    בהקשר; גרסה 2 (`RolePrefixedLineOrNeighbourContext`): `"[PASSAGE] "` + הטקסט של
    גרסה 1, בלי רווחים בקצותיו, לכל מסמך, ו-`"[QUERY] "` + השאילתה המנורמלת, בלי רווחים בקצותיה, לכל שאילתה. `passage_text()` מופעל
    ב-chunker **אחרי** הקיטום והנרמול, ולכן התחילית אינה נספרת בתקרת התווים ואינה מנורמלת.
  - `query_input()` — הפונקציה היחידה שדרכה כל מסלול מטמיע שאילתה (המנוע וה-
    `OfficialSemanticIndex`; הקואורדינטור מגיע למודל רק דרכם): נרמול, ואז התחילית. שאילתה
    שאין בה טקסט אחרי הנרמול נדחית כאן, בכל גרסה — אחרת תחילית הייתה מטמיעה את עצמה.
  - `chunk_hash` ו-`embedding_text_sha256` מכסים את התחילית (הם מתארים את מה שהמודל מקבל);
    `source_line_sha256` ו-`line_hash` אינם (הם מתארים את שורת הקורפוס).

* [`src/semantic/model_package.rs`](../src/semantic/model_package.rs) — מה נתיב מודל אומר
  על הדיסק. מקומפל תמיד, בלי תלות inference.
  - `names_an_onnx_graph()` / `ensure_onnx_model_path()` — מודל הוא גרף ONNX: סיומת `onnx`
    (בכל רישיות ASCII). **כל נתיב אחר נדחה** כ-`InvalidModelFile` לפני שנפתח דבר, בין שיש
    שם קובץ ובין שאין — GGUF קודם כול, שהתמיכה בו הוסרה אחרי `62f0c44`.
    `EmbeddingConfig::validate` ו-`validate_model` שואלים אותו שניהם.
  - `validate_model()` — נקודת הכניסה של `EmbeddingRuntime::load()`:
    `ensure_onnx_model_path`, ואחריו `validate_onnx_package`; מחזיר את ה-`OnnxPackage`.
  - `validate_onnx_package()` — החבילה היא הגרף, `tokenizer.json` שלידו (חובה —
    `TokenizerNotFound`, נבדק ראשון) וכל קובץ external-data שהגרף מצביע עליו. הגרף נקרא
    ב-**protobuf walk** חסום וזורם, ומגובב באותו מעבר: אורך שחורג מסוף הקובץ = הורדה שלא
    הושלמה, אורך שחורג מההודעה שמכילה אותו = קובץ פגום, שדה ראשון שאינו של `ModelProto` =
    לא ONNX כלל (Git LFS pointer, דף שגיאה ו-GGUF מזוהים בשמם). מספרי השדות מ-
    `onnx/onnx.proto3`; `raw_data` מדולג ב-64 KiB, לעולם לא מוחזק. טנזורי external-data
    נמצאים בכל מקום שטנזור יכול להיות: initializers, sparse initializers, attributes, תת-גרפים,
    functions ו-training graphs. כל `location` חייב להיות יחסי ובתוך התיקייה (לא מוחלט, לא
    `..`, לא symlink החוצה), והקובץ חייב להגיע לבית הרחוק ביותר שהפניה צריכה — ובלי `length`,
    ביט אחד לאיבר, מתחת לכל טיפוס ONNX.
  - `OnnxPackage` / `onnx_package_manifest()` / `onnx_package_checksum()` — ה-checksum של
    החבילה: SHA-256 של manifest קנוני (`otzaria-onnx-package-v1`, שורה לכל קובץ לפי relpath:
    נתיב, גודל, SHA-256). שום קובץ אחר בתיקייה — README, גרף שני, ספריית ONNX Runtime — אינו
    נכנס. ההגדרה המלאה: [`ARTIFACT_CONTRACT.md`](ARTIFACT_CONTRACT.md) §4.2.1.

* [`src/semantic/embedding.rs`](../src/semantic/embedding.rs)
  - `EmbeddingRuntime` & `EmbeddingConfig` — ממשק הרצת מודל מקומי: חבילת ONNX.
    `validate()` דוחה תצורה שאף backend אינו יכול לשרת, ובכלל זה נתיב שאינו גרף ONNX;
    `load()` מאמת ומחשב את `model_checksum` דרך `model_package::validate_model`. ברירת
    המחדל היא זהות הייצור, גרף ה-int8 של Meivin Round 2.
  - `EmbeddingDeployment` — עובדות פריסה ולא זהות: היכן המכונה הזו מחזיקה את מה שה-backend
    טוען מלבד המודל, היום ספריית ONNX Runtime (`onnx_runtime`). מוחזק **לצד**
    `EmbeddingConfig` ולא בתוכו, כך שקוד שגוזר זהות מ-`EmbeddingConfig` אינו יכול לאסוף
    אותו. האפליקציה מעבירה אותו ב-`OfficialIndexConfig::deployment` או
    `SemanticConfig::deployment`; `EmbeddingRuntime::with_deployment` ו-`select_backend_for`
    מוסרים אותו ל-backend, וה-stand-in מתעלם ממנו.
  - `HashingReader` — קורא קדימה שמגבב כל מה שהוא עובר, כך שגרף של מאות MB נבדק
    ומגובב בקריאה אחת; ה-protobuf walk של `model_package` רץ עליו.
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
    `--features mock-embedding`. **אינו מודל סמנטי.** לצידו ה-fixtures:
    `write_stub_onnx_package`, שכותב חבילת ONNX מינימלית ותקינה (גרף מקודד ביד ו-
    `tokenizer.json`) ומחזיר את נתיב הגרף, ו-`onnx::stub_graph_named`, גרף תקין אחר
    לבדיקה שצריכה מודל שני או קובץ שהוחלף.

* [`src/semantic/backend.rs`](../src/semantic/backend.rs) — החוזה שכל backend מקיים.
  - `EmbeddingBackend` — trait עם `Send + Sync`, כי הקואורדינטור מחזיק את המנוע
    ב-`RwLock` ו-`search` לוקח `.read()`; חיפושים נכנסים ל-`embed_batch_raw` דרך
    `&self` במקביל. `&mut self` היה מסרייל חיפוש מאחורי אינדוקס.
  - `embed_batch_raw()` — מחזיר וקטורים **גלמיים**. הנרמול אינו תפקיד ה-backend.
  - `tokenize()` — קיים כי בדיקת ה-parity של P2 מחייבת שוויון `token_ids`, ואין דרך
    לאמת אותה בלי לחשוף את הטוקנייזר. ה-stand-in מחזיר `TokenizationUnsupported`
    ולא מימוש מנוון — ids "סבירים" היו הופכים את הבדיקה להשוואה בין שתי המצאות.
  - `Pooling` — `Mean` / `InGraph`, עם התאמת מחרוזות **מדויקת** (לא case-insensitive ובלי
    trim): אותה מחרוזת נשמרת ב-manifest ומושווית תו-בתו בסשן הבא, כך שקבלת `"In-Graph"`
    כאליאס הייתה מייצרת mismatch מדומה ובנייה מחדש של האינדקס. `Mean` בר-ייצוג ובלתי-שמיש:
    הוא מה שמאפשר לבטא "הקונפיג חולק על ה-backend". `InGraph` (`"in-graph"`) — הגרף עצמו
    מוציא את וקטור המשפט הגמור; זה מה ש-backend ה-ONNX משרת. `"last-token"`, ה-pooling של
    ה-backend של GGUF, אינו מאוית עוד ונדחה כ-`UnknownPooling`.
  - `ensure_pooling_is_implemented()` — pooling שאיש אינו מבצע נדחה כ-`PoolingNotImplemented`
    בעודו תצורה, כי הוא נכתב ל-manifest לפני שנשאל backend כלשהו.
  - `CANDIDATES` / `select_backend()` — טבלה אחת שממנה קוראים גם הבחירה וגם בדיקת
    ה-pooling, ולכן הוספת backend היא שורה. ה-backend האמיתי קודם ל-stand-in, וה-stand-in
    טוען מה שה-backend האמיתי טוען (`in-graph`). אין backend → `BackendUnavailable` שנוקב
    ב-feature (`onnx-backend`) — או, כשה-feature דלוק על target שאין לו backend, אומר זאת
    במקום לבקש את ה-feature. הטבלה **אינה** מותנית ב-feature: היא
    מתארת אילו מימושים קיימים ב-crate, אחרת "אין backend" היה מדווח כ"קונפיגורציה
    שגויה". מחזירה `Option<Result<..>>` — `None` = לא מקומפל (המשך לחפש),
    `Some(Err)` = מקומפל ונכשל (עצור ודווח). עם `Option` בלבד, backend אמיתי שנכשל
    היה נראה כחסר, וה-stand-in היה עונה על מודל שבור בווקטורי האש בשקט.

* [`src/semantic/onnx_backend.rs`](../src/semantic/onnx_backend.rs) — inference אמיתי
  ל-ONNX דרך ONNX Runtime, מאחורי `--features onnx-backend`. `OnnxBackend::ID` הוא
  `"onnxruntime-sentence-v1"`; `OnnxBackendConfig::from_env_for` קורא את
  `OTZARIA_ONNX_THREADS` ו-`OTZARIA_ONNX_SESSIONS` ודוחה ערך שאינו מספר חיובי. באפליקציה
  session אחד (ברירת המחדל) — היא מטמיעה רק שאילתות; יותר מ-session אחד הוא כפתור של
  מכונת ה-build, ומשתלם רק כשכמה קוראים מטמיעים בו-זמנית. ספריית הריצה: הנתיב שהאפליקציה
  מעבירה (`EmbeddingDeployment::onnx_runtime`), אחריו `OTZARIA_ONNX_RUNTIME`, ואחריו הקובץ
  לצד הגרף — המקום הראשון שהוגדר מכריע, ונתיב שאינו נפתח נדחה ואינו מדולג
  (`resolve_runtime_path`).

* [`src/semantic/store.rs`](../src/semantic/store.rs) — **ה-store שהמנוע פותח** (מסלול
  הפיתוח ובדיקות; מסלול האפליקציה פותח סט מותקן).
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
  - **הפיצול הוא החוזה, לא נוחות:** חיפוש מקבל טיפוס שאין עליו insert לקרוא, ולכן
    read-only הוא תכונה של הטיפוס ולא כלל שמישהו צריך לזכור. מסלול האפליקציה אינו עובר
    באף אחד מהשניים: הוא פותח סט מותקן, שדבר במכשיר אינו כותב אליו.
  - `SemanticEngine` תלוי בצד הכותב כ-`Box<dyn VectorStoreBackend>`, ולכן בחירת
    ה-backend היא של הקורא (`SemanticEngine::with_store`) ולא קבועה במודול.

* [`src/semantic/chunk_key.rs`](../src/semantic/chunk_key.rs)
  - `ChunkKey` — 16 הבתים הראשונים של SHA-256 של הטקסט שהמודל קיבל: הכתובת של וקטור.
    `to_hex()` שווה ל-`compute_chunk_hash`; `column_value()` הוא 8 הבתים הראשונים, big-endian
    — מה שעמודת `chunkKey` של האינדקס מחזיקה (0 = שורה שאינה מוטמעת). `KEY_VERSION` 1.
  - `LineRef` — טקסט השורה וסעיפה; רק *שוויון* סעיפים משנה. `Chunker::embedded_text`
    ו-`Chunker::chunk_keys` עובדים על חלון שורות, עם ליבה אחת המשותפת ל-`chunk_book`, כדי
    שהאפליקציה תחשב מפתח של שורה מחמש שורות ולא מהספר כולו.

* [`src/semantic/oxv/`](../src/semantic/oxv/mod.rs) — **פורמט ה-segment.** הפריסה, שדה
  אחר שדה: [`ARTIFACT_CONTRACT.md`](ARTIFACT_CONTRACT.md) §3.
  - `format` — ה-header של 4,096 בתים, ספריית ה-sections, הרשומות בגודל קבוע. הכותב
    והקורא עוברים דרכו, ולכן אינם יכולים לחלוק על offset.
  - `codec` — `i8-sym-vec`, ברירת המחדל (סקלה לכל וקטור, `s = max|x|/127`, הקודים
    ב-`VECTORS` והסקלות ב-`VECTOR_SCALES`), `i8-sym-dim` (סקלה לכל ממד, `round_half_away`,
    clamp ל-±127; `calibrate_i8_sym_dim` לוקח קוונטיל מדויק בכלל lower) ו-`f32`; `CodecSpec`
    הוא מה שבנייה מבקשת.
  - `writer` — `SegmentBuilder`: הטבלאות קודם, ואז `VectorSink` שמקבל וקטורים בסדר
    ה-slots, ב-buffer, עם CRC לכל block — כך שבנייה בקנה מידה של הספרייה זורמת בזיכרון
    חסום. `segment_id` דטרמיניסטי.
  - `reader` — `Segment::open` ממפה (memmap2), בודק את ה-header, את ה-CRC של כל section
    קטן ואת העקביות המבנית; `verify_blocks` קורא כל block.
  - `kernel` — מכפלה סקלרית בשלמים: scalar, AVX2 ו-NEON, זהות ביט אחר ביט; שאילתה מוכנה
    כך שאף סכום ביניים אינו גולש. ה-`unsafe` היחיד הוא ה-intrinsics, וכל ליבה נבדקת מול
    ה-scalar.
  - `scan` — `scan_segments`: כל slot חי מדורג, על כמה חוטים, תחת מסנן ספרים, `k`
    הטובים נשמרים בסדר מלא; כפילות מפתח נספרת פעם אחת. `default_scan_threads` — חצי
    מהליבות, עד שמונה.

* [`src/semantic/segment_set/`](../src/semantic/segment_set/mod.rs) — **הסט על המכשיר.**
  - `SegmentSet::open` — שחזור כשה-lock פנוי, ואז הדור ש-`CURRENT` מונה, או `PREVIOUS`;
    `scan`, `info`, `generation`, `identity`.
  - `install_package` — segment ו-manifest של שחרור, לפי §5.3 של החוזה: אימות, staging,
    העברה, delta שנפתר במעבר אחד על המפתחות הישנים, דור חדש, החלפת מצביעים, אשפה.
    `ApplyReport` מדווח, ו-`already_applied` הוא delta שהסט כבר בלע.
  - `compact` — `CompactionPolicy` מחליטה, `LiveKeySource` מעגן מחדש; `CompactionReport`.
  - `info(dir)` ו-`scrub(dir)` — בלי מודל: מה מותקן, וקריאת כל block.
  - `ReleaseManifest` — ה-manifest שמתפרסם לצד segment, ו-`package()` — החבילה שהוא מתאר.
  - `space` — כמה מקום נשאר: `statvfs` ו-`GetDiskFreeSpaceExW`, שלוש קריאות `unsafe`
    קטנות במקום תלות.

* [`src/semantic/resolve.rs`](../src/semantic/resolve.rs) — **הפורט אל השורות החיות.**
  - `VectorHit` (ציון, מפתח, רשומות, segment ו-slot), `RecordRef` (ספר ו-hint) ו-`BookSet`.
  - `CandidateResolver` — מה שהמארח מממש מעל האינדקס שפתח: `generation`,
    `admissible_books(filters)` ו-`resolve(hits, filters, cancel)` → `ResolvedLine`.
    `NoResolver` — חיפוש בלי אינדקס חי מאחוריו; סט רשמי דרכו אינו תורם דבר, ואומר למה.
  - `LiveKeySource` — מה שדחיסה שואלת: איזה מפתח מחזיקה כל שורה חיה של ספר.
  - `ResolveError` — `Cancelled`, או `Index { reason }`.

* [`src/semantic/versioning.rs`](../src/semantic/versioning.rs)
  - `IndexVersion` — זהות הסט בשלוש קבוצות, 14 שדות: `TextIdentity`
    (`line_text_version`, `key_version`), `ModelIdentity` — **משפחה**, לא קובץ:
    `family_id`, `tokenizer_checksum`, ממד, pooling, `max_tokens`,
    `embedding_text_version`, normalization, `chunking_identity`, ו-`query_packages` —
    ו-`StoreIdentity` (`backend_id`, `store_format_version`, `vector_precision`).
    החוזה המלא: [`ARTIFACT_CONTRACT.md`](ARTIFACT_CONTRACT.md) §4.
  - `query_packages` מושווה ב**שייכות**: ההתקנה מצהירה על החבילה האחת שטענה, וה-checksum
    שלה חייב להיות ברשימה של הסט. כל השאר בשוויון.
  - `VectorProvenance` — החבילה שהטמיעה את הקטעים וה-worker שהריץ אותה: נרשם ב-manifest,
    נכנס ל-digest של החבילה, ואינו מושווה לדבר.
  - `identity_digest()` — SHA-256 קנוני מעל כל השדות; 32 הבתים שבהם segment, סט
    ו-manifest אומרים „אותה זהות”.
  - `IdentityField::ALL` — רשימה אחת שההשוואה, ה-digest והודעות הדחייה הולכות לפיה.
    **הכיסוי אינו מובטח על ידי הטיפוס** — `IndexVersion` הוא struct רגיל — אלא על ידי שתי
    בדיקות שנגזרות מה-JSON המסוריאלי: `every_serialized_identity_field_is_comparable`
    (שדה שנשמר ואינו מושווה) ו-`every_serialized_identity_field_is_refused_when_left_unfilled`
    (שדה שוולידציית השלמות שכחה). הוספת שדה בלי וריאנט מפילה את שתיהן.
  - `mismatches_against()` / `verify_matches()` — **כל** ההבדלים ברשימה טיפוסית
    (`IdentityMismatch`), לא הראשון ולא `bool`: דחייה שהצטמקה ל-`false` היא מה
    שחוזה המוצר קורא לו ניחוש.
  - `validate_complete()` — מחרוזת ריקה, גרסה 0, checksum שאינו 64 hex קטנות, או תו בקרה
    בתוך ערך זהות — נדחים **לפני** ההשוואה, כי שתי זהויות שלא מולאו משוות שוות זו לזו.
    תו בקרה נדחה גם כדי שהטקסט הקנוני שמאחורי ה-digest יישאר חד-משמעי.
  - כל שדה כאן קטלני: אין „אי-תאימות שדורשת רק chunking מחדש” כמו ב-manifest המקומי,
    כי במכשיר אין מה לבנות מחדש — רק שחרור אחר להוריד.

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
    ה-Runtime, store כותב וה-Manifest. מסלול האפליקציה הוא `official_index.rs`. ה-API
    של האינדוקס שלו (`index_books`) הוא **פיגום אב-טיפוס**: וקטורי הספרייה נבנים רק
    במכונת ה-build (`build`, `embed-shard`), והאפליקציה פותחת סט מותקן read-only דרך
    `OfficialSemanticIndex` ומטמיעה רק את השאילתה.
  - `with_store()` — פתיחה מעל backend שהקורא מספק. ה-manifest רושם את ה-backend שנפתח
    **בפועל**, ולכן פתיחה מחדש עם backend אחר היא אי-תאימות מדווחת ולא תשובה מ-store
    ריק. ה-backend היחיד שיש כיום הוא ה-`VectorStore` שבזיכרון.
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
  - `OfficialSemanticIndex::open(OfficialIndexConfig { vectors_dir, text, model,
    deployment, scan_threads })` — בסדר שבו כל דחייה נוקבת במה שלא הסכים: שחזור → הסט →
    המודל → הזהות. המודל נטען לפני ההשוואה מפני שה-checksum של החבילה ושל ה-tokenizer
    שלה הם עובדות שרק ה-runtime יודע.
  - הזהות הצפויה מורכבת משלושה מקורות: **text** מהאינדקס שהמארח פתח, **model**
    מהמשפחה המוצהרת (`LocalModel::of_family`) עם החבילה שנטענה, ו-**store** ממה ש-build
    הזה יודע לקרוא (`readable_store_identity()`: `otzaria-oxv`, 2, `i8-sym-vec`).
  - `search(query, top_k, books, cancel)` / `search_hits(vector, …)` — hits, לא שורות;
    `reload_vectors()` → `ReloadOutcome`; `status`, `set_info`, `identity`, `generation`,
    `book_count`.
  - `status` מדווח `vectors_persisted = true` ו-`needs_full_reindex = None`, ולא כטענה
    ריקה: סט הוא או הנכון או נדחה בפתיחה, ואין במכשיר מה לבנות מחדש.
  - `LocalModel` — מה שההתקנה **מצהירה**: נתיב, `family_id`, quantization של החבילה,
    ממד, pooling, `max_tokens` ושלוש גרסאות המתכון. הקורא אינו גוזר את השלוש האחרונות —
    שאילתה אינה עוברת chunking — אבל הן מושוות, כי סט שנבנה ממתכון אחר הוא סט אחר.

---

### 5. תת-מערכות תומכות

* [`src/config/profiles.rs`](../src/config/profiles.rs)
  - `SearchProfile` — `Fast` / `Balanced` / `Best`.
  - `RankingProfile` — כל פרמטרי הכיול במקום אחד (thresholds, בונוסים, קיבולות
    cache, אסטרטגיית fusion). מקור אמת יחיד, כדי שלא יהיו שתי קבוצות ברירות מחדל.
    כולל גם את מה שהיה קבוע בקוד: `alpha_by_query_type` (`QueryTypeAlphas` — 1.0 לביטוי
    במרכאות, 0.85 / 0.7 / 0.5 / 0.3 / 0.5) ו-`bm25_saturation_k` (`DEFAULT_BM25_SATURATION_K`
    = 10.0), כך שמארח יכול להעביר פרופיל שלם **לכל חיפוש** (`HybridSearchParams::ranking`,
    `SearchRequest::ranking`) ולכייל מהאפליקציה בלי שחרור של המנוע.
  - `validate()` — דוחה פרמטר שהדירוג אינו מוגדר עבורו (`NaN`, שלילי, מחוץ לטווח)
    ב-`InvalidRankingParameter` שנוקב בשם השדה, במקום לקצץ אותו בשקט. כל ה-presets עוברים.
  - **ברירות המחדל עדיין לא נמדדו.** הכיול צריך את סט הרלוונטיות המתויג של S1 (שאילתות
    עבריות מכל סוג, עם השורות הרלוונטיות לכל אחת), מדד על העמוד (nDCG@10 או recall), וריצות
    שמשנות משפחת פרמטרים אחת בכל פעם. עד אז ברירות המחדל הן הדירוג שהיה תמיד — ובדיקה
    ב-coordinator משווה אותו ביט-לביט מול עותק קפוא של ה-fusion כפי שהיה.
  - `FusionStrategy` — `Weighted` / `RRF { k }` / `Adaptive`.

* [`src/config/feature_flags.rs`](../src/config/feature_flags.rs)
  - `FeatureFlags` — כל שדה הוא `Option`, ולכן „לא צוין” נבדל מ„צוין כברירת המחדל”.
  - `apply()` — דורס פרופיל קיים במקום להחזיק העתק שני שלו. ערכים לא-חוקיים
    (`NaN`, מחוץ לטווח) נבלמים ולא נכנסים לפרופיל.

* [`src/telemetry/mod.rs`](../src/telemetry/mod.rs)
  - `SearchTelemetry` — רשומה לשאילתה: סוג שאילתה, מצב שרץ, אסטרטגיה, alpha,
    ספירות מועמדים, cache, latency (כולל embedding ו-fusion בנפרד) ופרופיל; ולסט רשמי —
    `semantic_hits`, `semantic_unresolved`, `scan_ms` ו-`resolve_ms`.
  - `TelemetryCollector` / `TelemetrySnapshot` — אגרגציה thread-safe; ה-snapshot סוכם את
    `semantic_hits` ואת `semantic_unresolved` — כמה וקטורים לא נמצאה להם שורה חיה.
  - **אין כאן רשת.** אלה מונים בזיכרון התהליך; המאגר אינו שולח דבר לשום שרת.

* [`src/distribution/package.rs`](../src/distribution/package.rs)
  - `PackageManifest` — `metadata_version` (3) + `IndexVersion` + תיאור ה-segment (סוג,
    מהדורות, תג, ספירות) + `VectorProvenance` + `created_at` + גודל מוצהר. זו החבילה
    שה-`packageDigest` של manifest השחרור מתאר. `metadata_version` נקרא ב-probe **לפני** המסמך, כדי
    שפורמט זר ידווח על גרסתו ולא ייפול על שגיאת פרסור של שדה בודד.
  - `verify_for_install()` / `verify_for_open()` — שני עומקים, כי אחד מהם רץ בכל עלייה
    של האפליקציה. שניהם: גרסת metadata, שלמות הזהות, הזהות מול ההתקנה, ה-digest המפורסם,
    ונוכחות+גודל של כל payload. רק הראשון מגבב כל בייט. גיבוב גיגה-בייטים בכל פתיחה אינו
    בתקציב, ובדיקה שאי אפשר להרשות היא בדיקה שמכבים.
  - `VerificationDepth` + `VerifiedPackage::depth()` — הטוקן נושא **מה נבדק בו**. קורא
    שמדווח „מאומת” בלי להסתכל בזה טוען שה-payload גובב כשאולי רק נעשה עליו `stat`.
    מה שהעומק הרזה אינו תופס: עריכה באותו אורך בדיוק. בסט וקטורים זה תפקידם של ה-CRCs
    של ה-segment — בפתיחה ל-sections הקטנים, וב-scrub לכל block.
  - `digest()` — SHA-256 מעל טקסט קנוני (גרסת metadata, כל שדות הזהות בסדר `ALL`, ספירות,
    גודל, ו-checksum+גודל לכל payload). זה **עוגן האמון**: `payloads.json` נוסע בתוך החבילה,
    ולכן payload שהוחלף יחד עם ה-checksum שלו עובר כל בדיקה פנימית. רק digest שפורסם
    מחוץ לחבילה מבדיל בין הארטיפקט הרשמי לחבילה עקבית-עם-עצמה. מעל טקסט קנוני ולא מעל
    בייטי JSON, כדי שכתיבה מחדש של ה-metadata בהתקנה לא תשנה אותו; `created_at` מוחרג.
  - `ArtifactExpectation` — `with_published_digest` מול `without_published_digest`. אין
    ברירת מחדל שקטה: מי שמוותר על העוגן קורא לפונקציה ששמה אומר זאת.
  - `VerifiedPackage` — טוקן שאין לו constructor ציבורי אחר: מי שמקבל אותו יודע שהחבילה
    אומתה, ובאיזה עומק.
  - `verify_integrity()` / `walk_payloads()` — מהלך אחד לשני העומקים, כדי שהבדיקה הזולה
    לא תפסיק בשקט לכסות משהו שהיקרה כן מכסה. ספירות אפס נדחות; ההשוואה מול **תוכן**
    ה-segment נעשית בהתקנה (`install_package`), כי היא דורשת את הפורמט.
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

* [`src/distribution/importer.rs`](../src/distribution/importer.rs) — התקנה של **תיקיית
  חבילה** שלמה בשני renames. סט הווקטורים אינו מותקן כך — הוא נבנה דור אחר דור
  ב-`segment_set::install_package` — והמודול נשאר לחבילות שהן תיקייה.
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
  - **מה שאינו כאן:** ה-importer אינו חשוף דרך `OtzariaHybridEngine`, ה-FFI או
    אוצריא.

* [`src/distribution/corpus.rs`](../src/distribution/corpus.rs) — **הפורט אל האינדקס
  הלקסיקלי, כפי שבנייה קוראת אותו.**
  - `CorpusIndex` — `identity()`, `expected_line_ids(model)` ו-`line(line_id)`. בנייה
    מקבלת את זה ולא נתיב, כי Tantivy אינו תלות של ה-crate הזה ואסור שיהיה: האינדקס, הסכמה
    וסכמת ה-IDs חיים ב-`otzaria_search_engine`.
  - `CorpusBooks` — **צורת** הקורפוס: אילו ספרים יש, ומהו סדר השורות בספר. שורה קצרה
    שואלת הקשר משכנותיה, ולכן החלת מתכון מחייבת את זה. הוא אינו עונה דבר על **תוכן**
    של שורה — זה של `line()` בלבד. מקום השורה ברשימה הוא ה-hint שה-segment שומר.
  - **זהות הקורפוס נקראת מהאינדקס** — מתכון השורות שלו, המהדורה והתג — ולא מוקלדת ליד
    הווקטורים. מה שנשמר על שורה הוא הספר, המקום והמפתח, ולא שום שדה תיאורי: כל שאר
    התיאור הוא מה שהאינדקס החי אומר כשתוצאה נפתרת.
  - **זו אינה „כל השורות באינדקס”:** מתכון ההטמעה מדלג על שורות קצרות מדי, וחבילה
    שדילגה עליהן אינה חסרה. `BuildPlan` גוזר את הקבוצה מהמתכון.
  - `CorpusLine` — מה שהאינדקס יודע על שורה, ועוד ה-`text` שממנו הווקטור נבנה. הטקסט
    **אינו** נשמר בחבילה.
  - `JsonlCorpus` — תמלול לשני קבצים (`identity.json` + `lines.jsonl`), שמאפשר להריץ
    בנייה בלי Tantivy. **תמלול, לא מקור אמת:** הוא אמין בדיוק כמו מי שכתב אותו. הוא
    **מסיק** את סדר השורות בספר מ-`line_id` עולה — נכון תחת `document_id_scheme_version` 1,
    שבו החצי התחתון הוא מיקום השורה — ומחזיק כל שורה בזיכרון.

* [`src/distribution/shard.rs`](../src/distribution/shard.rs) — **בנייה על מכונות שאינן
  נפגשות.** `export_plan` מחיל את המתכון על הקורפוס וכותב את הטקסטים המוגמרים
  (`plan.jsonl`); `embed_shard` מטמיע חלון של רשומות ב-worker שאין לו קורפוס, ובודק את
  ה-digest של כל טקסט לפני שהוא מטמיע אותו; `verify_shards` בודק כל shard מול ה-plan —
  plan, מודל, רוחב, חלון, digests, אורכים — ושהחלונות מכסים אותו בדיוק, לפני שנקרא בית;
  `read_vector_inputs` מזרים זוג קבצים של shard (`f32` little-endian ו-JSONL באותו סדר)
  ותופס את שתי צורות אי-ההתאמה ביניהם. ההרכבה — וקטורים לפי מפתח אל segment — היא
  `assemble.rs`, למטה.

* [`src/distribution/builder.rs`](../src/distribution/builder.rs) — **צד ה-build (S4b).**
  - `build()` — קורפוס ומודל → חבילת base: `segment.oxv`, ה-`manifest.json`
    וה-`payloads.json` שסביבו, ו-`release.json` — ה-manifest שההתקנה מקבלת — אחרון. הסדר
    הוא **עלות ולא טעם**: יעד פנוי → זהות שלמה ותג תקין → מתכון תואם → טעינת המודל
    והשוואתו להצהרה → שער backend לא-סמנטי → תכנית → הטמעה → כיול, כתיבה, חבילה.
  - הקצאת ה-slots היא של §3.4 בחוזה: ספרים בסדר בתי השם, שורות לפי הסדר, הופעה ראשונה של
    מפתח לוקחת slot, הופעה בספר אחר היא extra, ורשומה אחת לכל ספר ומפתח. ה-hint הוא מקום
    השורה בספר.
  - **ה-provenance נקבע כאן.** הטקסט שממנו המפתח מחושב הוא אותו `String` שנמסר ל-backend.
    ה-checksum של החבילה (שחייב להיות ב-`query_packages`), של ה-tokenizer, הרוחב,
    ה-pooling ו-`max_tokens` האפקטיבי **מדווחים על ידי ה-runtime שנטען** ומושווים להצהרה.
    מה שנשאר הצהרה: `family_id` וה-quantization של כל חבילה.
  - **שלוש גרסאות המתכון אינן תכונות של מודל** אלא של הקוד כאן, ולכן הן **מוכרעות** ולא
    מושוות — ראו [`recipe.rs`](../src/semantic/recipe.rs).
  - **חלון TOCTOU שנשאר פתוח ומוצהר:** ה-checksum מחושב, ואז ה-backend פותח את אותו נתיב
    שוב. הסגירה היא staging לעותק content-addressed בצינור שמסביב.
  - `BuildPlan` — קבוצת השורות שהמתכון מטמיע, **לפני שקיים ולו וקטור אחד**. מעבר ההטמעה
    גוזר אותה שוב, ואי-התאמה היא `CoverageMismatch`: הקורפוס השתנה מתחת לבנייה.
  - **המתכון מוצמד ואינו מנוחש.** `chunking_identity` הוא hash חד-כיווני, ולכן הבנייה
    מקבלת את התצורה הממשית ודוחה אחת שאינה מה שהזהות מצהירה עליו.
  - **מה היא מחזיקה:** קטעי ספר אחד, וכל וקטור נבדל כ-`f32` עד הכיול. זה מסלול פיתוח;
    בניית הספרייה זורמת: `assemble.rs`.
  - **שער ה-backend הלא-סמנטי.** וקטורי hash מושלמים מבנית וריקים ממשמעות, ואף בדיקה
    מאוחרת אינה יכולה לדעת — לכן הסירוב חייב להיות כאן. `allow_non_semantic_backend` הוא
    `false` בכל דבר שנשלח.

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
  - מסלול הריצה הרשמי דרך ה-API הציבורי, כמו שה-`otzaria_search_engine` יראה אותו: חבילה
    שנבנית מקורפוס קטן, מותקנת ונפתחת, ונשאלת דרך ה-coordinator עם `FakeResolver` במקום
    האינדקס החי — ב-`SemanticOnly` וב-`Hybrid`. ה-ids הם של ה-resolver ולא של הווקטורים;
    שני ספרים עם אותו id נשארים שתי תוצאות; שורה שהטקסט שלה השתנה אינה מוצגת; המטמון
    עוקב אחרי הדור של ה-resolver ושל הסט; כל פעולה בונה נדחית בשם והסט אינו משתנה.
  - דורש `--features mock-embedding`.
* [`tests/artifact_builder.rs`](../tests/artifact_builder.rs)
  - שער הקבלה של S4b, דרך הבינארי: `build` מקבל קורפוס, מודל ומתכון ומפיק חבילה, שמותקנת
    מול ה-digest שהבנייה הדפיסה, נפתחת, ומוצאת כל שורה בטקסט של עצמה; שתי בניות עם
    `created_at` שונה מפיקות את אותו segment. השורה שהמתכון מדלג עליה נעדרת מכל תוצאה.
  - שתי בדיקות רצות ב**בנייה רגילה**: מתכון שאינו המוצהר נדחה לפני שנפתח מודל, ובנייה
    בלי backend מסרבת במקום להמציא וקטורים. השאר דורש `--features mock-embedding`, ובדיקת
    המודל האמיתי (`--ignored`) דורשת את `onnx-backend`.
* [`tests/vector_set_scale.rs`](../tests/vector_set_scale.rs)
  - `#[ignore]`: סט סינתטי בצורה של ספריית v30 ב-N רשומות (`OTZARIA_SCALE_N`,
    `OTZARIA_SCALE_DIR`) — כתיבה, התקנה, פתיחה, סריקה, delta ודחיסה, נמדדים.
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
  - job נפרד ל-`onnx-backend` (Linux, macOS ו-Windows) מול ONNX Runtime 1.28.0, ו-job
    **Golden Vectors** שמריץ את שער ה-parity לשני הגרפים של המודל האמיתי, ובונה, מתקין
    ושואל חבילה אמיתית אחת מגרף ה-int8. השער דורש את הסוד `OTZARIA_HF_TOKEN`, וכשהוא חסר הוא
    נכשל במפורש ולא מדווח דילוג כהצלחה. ראו [`MODEL_DISTRIBUTION.md`](MODEL_DISTRIBUTION.md) §5.

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

  `cargo bench -- --store oxv` כותב segment, ממפה אותו וסורק בחוט אחד, ב-`--threads`
  ותחת מסנן של 5% מהספרים. על Apple M4, 200,000 וקטורים ב-256: 18.9 ns לווקטור בחוט אחד,
  0.8 ms על שמונה, 0.2 ms במסנן, פתיחה 0.27 ms. בקנה מידה של הספרייה
  ([`tests/vector_set_scale.rs`](../tests/vector_set_scale.rs), 6.0M slots, `i8-sym-vec`):
  פתיחה 3.8 ms, סריקה 69 ms בחוט אחד ו-17 ms בעשרה — ראו
  [`ARTIFACT_CONTRACT.md`](ARTIFACT_CONTRACT.md) §8.

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

## בניית חבילת וקטורים (S4b)

```bash
OTZARIA_ONNX_RUNTIME=/abs/path/libonnxruntime.dylib \
cargo run --release --features onnx-backend -- build \
  --corpus-identity corpus-identity.json --corpus-lines corpus-lines.jsonl \
  --model config/models/meivin-round2-onnx/model.json \
  --model-file /abs/path/seforim-embed-round2-fp32.onnx \
  --chunking config/models/meivin-round2-onnx/chunking.json \
  --out ./package
```

* הפלט: `segment.oxv`, `manifest.json`, `payloads.json` ו-`release.json`. ה-SHA-256 של
  `release.json` שמודפס הוא מה שמתפרסם **מחוץ** לשחרור, ו-`install_package` מקבל את
  ה-segment ואת ה-manifest מולו.
* `chunking.json` — `ChunkerConfig`, כלומר המתכון עצמו ולא תיאור שלו:
  `{"min_meaningful_chars": 20, "context_window_lines": 2, "max_chunk_chars": 512,
  "min_embeddable_chars": 5, "chunking_version": 1, "embedding_text_version": 2,
  "normalization_version": 1}` (של זהות הייצור) — כל השדות חובה. הוא חייב לגבב ל-`chunking_identity`
  שהמשפחה מצהירה, אחרת הבנייה נדחית.
* `--model-file` הוא החבילה שממנה נוצרים הווקטורים בפועל, וה-checksum שלה חייב להיות
  באחת מ-`query_packages` של `model.json`; היא נרשמת כ-`provenance.passage_package`.
  `model-checksum --model-file <path>` מדפיס את ה-checksum, את קובצי החבילה ואת
  ה-manifest המדויק — בבנייה רגילה.
* `--clip-q` — הקוונטיל שבו מכוילת הסקלה של כל ממד; ברירת המחדל 1 אינה קוטמת דבר ב-base.
* מי מקבל וקטור **נגזר**: ה-`Chunker` מוחל על הקורפוס לפני כל inference. שורה קצרה מדי
  מכדי לשאת משמעות מדולגת, וחבילה שדילגה עליה שלמה ולא חסרה.
* `corpus-lines.jsonl` — `{"line_id": N, "source_book_key": ..., "title": ...,
  "reference": ..., "section_id": N, "segment": N, "is_pdf": bool, "line_hash": N,
  "content_hash": N, "facets": [...], "text": "..."}` לכל מסמך.
* דורש backend inference, כי בנייה היא inference.

## בנייה מפוצלת

`plan` מחיל את המתכון במכונה שמחזיקה את הקורפוס, ו-`embed-shard --skip --take`
מטמיע חלון של ה-plan במכונה שמחזיקה את המודל, ובודק את ה-digest של כל טקסט לפני שהוא
מטמיע אותו. `warehouse-add` מקבל shards אל ה-warehouse, ו-`assemble` מרכיב ממנו base או
delta לפי מפתח, עם `--verify` לשערים — ראו [`VECTOR_BUILD.md`](VECTOR_BUILD.md).
