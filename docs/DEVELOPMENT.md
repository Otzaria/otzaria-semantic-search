# otzaria-semantic-search

> המסמך הזה אינו README למשתמשים. מטרתו היא לתת למפתח חדש תמונה מדויקת של מצב הפרויקט, הארכיטקטורה, ההחלטות שכבר התקבלו, מה ממומש בפועל, מה עדיין mock/placeholder, ומה צריך לעשות הלאה.
>
> **חשוב:** אין להסיק ממבנה שמות הקוד או מה־comments שמערכת מסוימת כבר ממומשת. במקומות שבהם הארכיטקטורה קיימת אך backend אמיתי עדיין לא מחובר, הדבר מצוין במפורש.
>
> **קראו קודם:** [`PRODUCT_CONTRACT.md`](PRODUCT_CONTRACT.md) — היקף המוצר. הוא גובר על
> המסמך הזה. בקצרה: האינדקס הרשמי נבנה מראש ונפתח read-only, אין overlay לספרי משתמש,
> אין אינדוקס ברקע באפליקציה, ואין שירות מרוחק בזמן חיפוש. סדר העבודה נמצא ב־
> [`שלבי ויעדי התקדמות.md`](../שלבי%20ויעדי%20התקדמות.md).
>
> חלק מהסעיפים כאן נכתבו לפני PR #2 ו־PR #3 ומתארים כוונות שהתממשו או שהשתנו. הם
> עודכנו; אם נותרה סתירה — החוזה והמסמך המשלים גוברים.

---

# 1. מטרת הפרויקט

הפרויקט הוא תוספת של **Semantic Search מקומי ל־Otzaria**, המתחבר לחיפוש הקיים ולא מחליף אותו.

Otzaria כבר מחזיקה מנוע חיפוש לקסיקלי המבוסס על Tantivy/BM25.

המטרה כאן היא להוסיף מנוע חיפוש שני, המבוסס על:

```text
Text
  ↓
Embedding Model
  ↓
Dense Vector
  ↓
Vector Search
  ↓
Semantic Candidates
```

ולאחר מכן לשלב את התוצאות עם החיפוש הקיים:

```text
                  Query
                    │
          ┌─────────┴─────────┐
          │                   │
          ▼                   ▼
       Tantivy             Embedding
        BM25                  │
          │                   ▼
          │               Vector Search
          │                   │
          └─────────┬─────────┘
                    ▼
                  Fusion
                    ▼
                 Ranking
                    ▼
                 Grouping
                    ▼
              Final Results
```

העיקרון המרכזי:

> **Semantic Search הוא sidecar לחיפוש הקיים, לא החלפה שלו.**

הקוד הראשי מגדיר זאת במפורש: החיפוש הלקסיקלי הקיים נשאר בבעלות Tantivy/Otzaria, ואילו ה־semantic path נמצא בבעלות המערכת החדשה.

---

# 2. כלל ארכיטקטוני שאסור לשבור

## לא לגעת במסד הנתונים הקיים של החיפוש

הפרויקט **לא אמור להעביר את Tantivy למסד הנתונים הסמנטי**.

יש שתי מערכות נפרדות:

```text
┌──────────────────────────────┐
│ Existing Otzaria Search     │
│                              │
│ Tantivy                      │
│ BM25                         │
│ Existing DB / Index         │
└──────────────┬───────────────┘
               │
               │ lexical candidates
               ▼
        Hybrid Coordinator
               ▲
               │ semantic candidates
               │
┌──────────────┴───────────────┐
│ Semantic Search Sidecar      │
│                              │
│ Embedding model              │
│ Chunking                     │
│ Vector DB                    │
│ Semantic manifest            │
│ Semantic retrieval           │
└──────────────────────────────┘
```

לכל צד יש lifecycle עצמאי.

אם ה־semantic engine נופל:

```text
Semantic failure
      ↓
BM25 עדיין עובד
```

זו **דרישת reliability**, לא nice-to-have.

---

# 3. איפה הפרויקט עומד כרגע

## תמונת מצב אמיתית

| רכיב                             | מצב                        |
| -------------------------------- | -------------------------- |
| Rust crate                       | קיים                       |
| חלוקה ל־semantic / hybrid / API  | קיים                       |
| Chunking                         | ממומש                      |
| Semantic IDs                     | ממומש                      |
| Chunk hashes                     | ממומש                      |
| Manifest                         | ממומש                      |
| בדיקת שינויי ספרים               | ממומש                      |
| Embedding abstraction            | קיים                       |
| הגדרת GGUF                       | קיימת                      |
| אימות קונטיינר GGUF + checksum   | ממומש (header לפני hash, פרסור descriptors, חסם תחתון על הגודל) |
| GGUF inference אמיתי             | ממומש מאחורי `--features llama-backend`, מאומת מול golden vectors |
| deterministic embedding fallback | ממומש, **מחוץ ל־production** (feature `mock-embedding`) |
| Batch embedding                  | ממומש (האינדוקס משתמש בו)  |
| VectorStore abstraction          | קיים                       |
| `VectorStoreBackend` trait       | קיים; **ה־engine עדיין אינו תלוי בו** |
| In-memory vector store           | ממומש — וזה מה שה־engine פותח |
| `ZevcStore` (snapshot לדיסק)     | ממומש ונבדק, **לא מחובר**; סריקה מלאה, לא ANN, לא `zvec` |
| Cosine search                    | ממומש                      |
| Metadata filtering               | ממומש (facets שטוחים + חלוקת ממדים כמו במנוע הלקסיקלי) |
| זיהוי שינוי ב-PDF                | ממומש כאשר הקורא מספק source revision סמכותי + metadata; גודל+mtime לבדם אינם קנוניים |
| empty-book marker                | ממומש |
| Hybrid coordinator               | קיים                       |
| Fusion                           | קיים                       |
| Dynamic weighting                | קיים                       |
| RRF                              | ממומש **ובשימוש** — נבחר לפי `FusionStrategy` בפרופיל |
| פרופילי Fast/Balanced/Best        | ממומשים (`config::profiles`) |
| Feature flags                    | ממומשים כדריסות מעל פרופיל |
| Query cache + embedding cache    | ממומשים                    |
| Telemetry                        | ממומש — מוני תהליך, ללא רשת |
| Grouping                         | ממומש                      |
| שלושת מצבי החיפוש                | ממומש (כולל SemanticOnly)  |
| זיהוי אי־תאימות אינדקס           | ממומש (משבית את המסלול הסמנטי) |
| התאוששות מ־manifest פגום         | ממומש (quarantine + reset) |
| עמידות ה-manifest                | אטומי בכל פלטפורמה; durable ב-Unix, best-effort ב-Windows |
| כתיבת manifest באינדוקס מלא      | פעם אחת בסוף (לא פר-ספר) |
| Rust API seam ל-Flutter/FFI      | קיים; bindings אמיתיים נבנים ב־`otzaria_search_engine` |
| חבילת אינדקס + ייבוא אטומי       | ממומשים כאב־טיפוס (`distribution`), **לא חשופים ב־API** |
| `IndexVersion`                   | קיים; **חסרות** זהות corpus, סכמת Tantivy וסכמת ה־ID |
| builder של הארטיפקט הרשמי        | **לא קיים** (S4)           |
| Production persistence במסלול הפעיל | **עדיין חסרה** (S2)     |
| ANN retrieval                    | **חסר** (brute-force בלבד) |
| UI סמנטי באוצריא                 | **לא קיים** (S7)           |

הנקודה החשובה ביותר למפתח חדש:

> **זהו כרגע skeleton ארכיטקטוני עובד חלקית, לא מנוע semantic production-complete.**
>
> מה שכן ממומש באמת: inference אמיתי מול המודל, שלושת מצבי החיפוש, fusion עם פרופילים,
> caches, telemetry, ואב־טיפוס של persistence ואריזה. מה שחסר כדי שיהיה מוצר: backend
> read-only שמחובר למנוע ונמדד בקנה מידה, חוזה זהות שקושר את הארטיפקט ל־corpus מסוים,
> builder שמפיק אותו, והפעלה באפליקציה.

## מה השתנה ב־PR הראשון (Correctness baseline)

ה־PR הראשון במפת הדרכים לא הוסיף יכולות — הוא הפך את השלד לנכון ולכן לניתן־למידה.
מה שחשוב לדעת עליו לפני שנוגעים בקוד:

1. **ה־embedding המזויף אינו זמין ב־production.** בבנייה רגילה
   `EmbeddingRuntime::load()` נכשל ב־`BackendUnavailable`. ה־stand-in נמצא מאחורי
   feature בשם `mock-embedding` (ונדלק אוטומטית ב־`cfg(test)` בתוך ה־crate).
   הטסטים שדורשים backend נמצאים ב־`tests/hybrid_integration_test.rs` ורצים רק עם
   ה־feature; `tests/production_backend_gate.rs` הוא התמונה ההופכית ומאמת שבנייה
   רגילה באמת מסרבת. **לכן ה־CI מריץ את שתי הקונפיגורציות.**
2. **אי־תאימות משביתה את המסלול הסמנטי, לא רק מדפיסה warning.** manifest שאינו
   תואם לקונפיגורציה מחזיר `SemanticSearchError::IncompatibleIndex` גם בחיפוש וגם
   באינדוקס; המסלול הלקסיקלי ממשיך לעבוד, וההתאוששות היא
   `SemanticEngine::reset_index()`.
3. **ה־manifest לא מצהיר על ספרים שהווקטורים שלהם נעלמו.** ה־backend הנוכחי אינו
   persistent (`VectorStore::is_persistent() == false`), ולכן ב־open נמחקות רשומות
   הספרים ו־`diff_against_tantivy` מבקש אינדוקס מחדש. זה מונע בדיוק את המצב שבו
   "מאונדקס" ו"אין וקטורים" מתקיימים יחד.
4. **re-index מוחק לפני שהוא כותב.** שורה שנמחקה מספר לא משאירה וקטור מאחור.
5. **חוזה ה־filters אחיד:** רשימה ריקה אינה מסננת, התאמת topics היררכית כמו facet
   ב־Tantivy, ו־`include_pdf` הוא מפסק *הוצאה* (`Some(false)` מוציא PDF;
   `Some(true)`/`None` לא מסננים).
6. **תוצאה סמנטית מסומנת ב־`needs_hydration`.** ה־vector store אינו משכפל את גוף
   השורה, ולכן טקסט חייב להיטען מ־Tantivy לפי ID. עד ש־P5 יחבר את ה־hydration,
   הדגל הוא החוזה שאומר "הטקסט חסר", במקום כרטיס ריק.
7. **כל degradation נראה לקורא:** `HybridSearchResult::search_mode` הוא המצב שרץ
   בפועל, ו־`fallback_reason` אומר למה המסלול הסמנטי לא השתתף.
8. **מודל ה-metadata תואם לאוצריא, לא דומה לה.** ספר נושא `topics: String` (נתיב
   קטגוריה אחד) ו-`extra_facets: Vec<String>` — בדיוק מה שה-indexer הלקסיקלי מעביר
   ל-Tantivy. זה לא ניסוח יפה יותר: לספר יכולים להיות **כמה מחברים**, וכל אחד הוא
   facet נפרד (`BookFacetMetadataCache.extraFacetsForBook`). `author: Option<String>`
   לא יכול לייצג את זה, וסינון לפי המחבר שלא נכנס היה מחזיר אפס תוצאות.
   הסינון מקבל רשימת facets שטוחה ומקבץ אותה לממדים לפי `FACET_DIMENSION_ROOTS` —
   אותו כלל בדיוק כמו `facet_filter_query`, כדי שלא יהיו שני מימושים שיכולים
   להיפרד.
9. **PDF שהשתנה מזוהה.** Tantivy מדווח `contentHash = 0` לכל PDF (הטקסט המחולץ אינו
   במסד הספרייה), כך שהשוואה רגילה של `0 == 0` הופכת כל PDF ל"לא השתנה" לנצח.
   `ContentFingerprint` מפריד בין hash אמיתי ל"אין חתימה", ה-diff מדווח על ספר כזה
   כדורש בדיקה, ו-`BookForIndexing::line_fingerprint()` — חתימה שהמנוע הסמנטי מחשב
   מהשורות עצמן — היא מה שמכריע. ספר שלא השתנה נחתך אחרי chunking, **בלי inference
   בכלל**, כך שהדיווח הזהיר אינו יקר.
10. **ספר שלא הניב chunks נשאר רשום** (`chunk_count = 0`). בלי זה כל PDF סרוק היה
    מדווח כחדש ומעובד מחדש בכל הפעלה — בדיוק מה שאוצריא פותרת ב-empty-book marker
    שלה. גריעת רשומות ב-backend נדיף מוחקת רק רשומות ש*מצהירות על וקטורים*; מרקר
    ריק לא איבד כלום.
11. **וקטור לא-finite נדחה.** `NaN` מזהם את הנורמה של עצמו, ו-`NaN < MIN` הוא
    `false` — כך שבדיקת נורמה לבדה מכניסה אותו לאינדקס, ואז כל הציונים שלו נזרקים
    בחיפוש והספר נראה מאונדקס אך אינו ניתן לחיפוש. גם וקטור **finite** גדול
    (בסביבות `1e30`) גולש ל-`inf` בנורמה ומתנרמל לאפס. שתי השכבות בודקות זאת.

12. **החתימה שמגיעה ל־diff חייבת לכסות גם metadata.** ה־diff מקבל
    `HashMap<String, ContentFingerprint>`, ולערך יש שלושה מצבים:
    * `Canonical` — מכסה תוכן **וגם** את ה-metadata שנשמר בכל וקטור. רק זה מגיע
      ל"אין מה לעשות". חתימת Tantivy היא כזו: `compute_book_fingerprint` שם כבר
      מערבב כותרת, נתיב קטגוריה, סדר קטלוגי/דורות ו-facets ממוינים.
    * `ContentOnly` — מכסה תוכן בלבד. **גודל+mtime של PDF הוא בדיוק זה**, ולכן
      תיקון מחבר או שינוי קטגוריה לא מזיזים אותו בזמן שהם משנים כל וקטור של הספר.
      התאמה כזו מחזירה `Unverifiable`, לא "מעודכן".
    * `Unverifiable` — אין במה להשוות.

    לספר שאוצריא לא יכולה לתת לו חתימה לקסיקלית יש
    `ContentFingerprint::canonical(source_revision, title, topics, facets, is_pdf)`.
    ה־revision חייב להיות לא־אפס ולכסות טקסט מחולץ, מבנה ומזהי שורות/סעיפים,
    references וגרסת חילוץ/OCR; גודל+mtime בלבד הם `ContentOnly`. הפונקציה מערבבת
    את ה־revision עם ה-metadata; אפס נשאר `Unverifiable`.
    שני המצבים האחרונים נכנסים לרשימה **נפרדת** (`IndexDiff::unverifiable_books`)
    ולא ל־`changed_books` — הם לא *ידועים* כמשתנים, וייצור שלהם עולה לקורא חילוץ
    טקסט מחדש. **אותו ערך בדיוק חייב להיכנס ל־`BookForIndexing::content_fingerprint`
    בזמן האינדוקס ולהיות מועבר ל־diff.** `Hash(0)` אינו ניתן לביטוי בכלל: הווריאנטים
    נושאים `NonZeroU64`, ורשומת manifest שערכה `0` לא מותאמת לשום חתימה.
13. **ערך ההחזרה של אינדוקס מפורש:** `IndexOutcome::{Indexed, Skipped, Empty}`
    ו־`IndexingSummary`. קודם ספר שנחתך החזיר את מספר ה־chunks שכבר היו לו, כאילו
    נכתבו עכשיו.
14. **סדר facets אינו מידע.** כל חתימה ממיינת ומסירה כפילויות מ-`extra_facets`,
    כמו `book_fingerprint` הלקסיקלי, וגם `all_facets()` מחזיר רשימה קנונית. בלי זה
    קורא שמפרט את מחברי הספר בסדר אחר היה גורם ל-re-embedding מלא.
15. **ה־manifest נשמר פעם אחת בסוף, לא פר-ספר ולא ב־checkpoints מלאים.** כל שמירה
    מסריאלזת את כל הרשומות, ולכן שמירה לכל ספר היא `O(B²)` וגם checkpoints חוזרים
    מוסיפים כתיבה סופר־ליניארית. ה־store הנוכחי אינו persistent, ולכן manifest
    ביניים אינו משמר שום עבודת inference אחרי restart. backend persistent יצטרך
    journal append-only או checkpoint דלתאי לפני שיוכל להבטיח resume מקריסה.
16. **פעולות lifecycle כותבות מסודרות בתור** (`indexing: Mutex<()>`). אינדוקס,
    reset וגריעת `removed_books` אינם יכולים להשתלב זה בזה באמצע batch.

מה שמכוון בכוונה **לא** נעשה שם, ונסגר מאז ב־PR #3: threshold לתוצאה סמנטית לא
רלוונטית ובחירת אסטרטגיית fusion (כולל RRF) לפי פרופיל. `total_count` עדיין מדווח את
מספר המועמדים שנכנסו ל־fusion ולא את מספר התוצאות, ומתועד ככזה.

## מה הוסיף PR #2 (Inference אמיתי)

`--features llama-backend` מספק inference אמיתי מול קובץ GGUF, ולא stand-in. הפרטים
המלאים ב־[`P2_INFERENCE_SPIKE.md`](P2_INFERENCE_SPIKE.md) ו־[`P2_REFERENCE_VECTORS.md`](P2_REFERENCE_VECTORS.md);
מה שחשוב לדעת לפני שנוגעים בקוד:

1. **בדיקת ה־parity הראשית היא שוויון `token_ids` מדויק, לא cosine.** נמדד ש־BOS
   מוטעה מקבל cosine *גבוה יותר* (0.9947938) מרפרנס לגיטימי (0.9947909), ולכן אין סף
   cosine שמפריד ביניהם.
2. **בנייה רגילה נשארת בלי backend.** ה־feature אינו ברירת מחדל מפני שהוא בונה
   llama.cpp ו־ggml דרך cmake בכל בנייה של תלוי.
3. **הבחירה בין Candle ל־llama.cpp לא נמדדה.** הוכרעה llama.cpp לפני שנמדד משהו;
   הקריטריונים לפתיחה מחדש רשומים באותו מסמך.

## מה הוסיף PR #3 (Hybrid, פרופילים, אב־טיפוס persistence)

1. **`VectorStoreBackend`** — חוזה משותף לשני ה־stores. ה־engine עדיין תלוי
   ב־`VectorStore` הקונקרטי, ולכן ה־trait הוא הכנה ולא נקודת החלפה בפועל. S2.
2. **`ZevcStore`** — snapshots לדיסק עם checksum לכל payload, ופתיחה מחדש שמאמתת
   אותם. **הוא אינו הספרייה `zvec`, אינו ANN ואינו mmap:** הפתיחה טוענת את כל
   הווקטורים ל־`HashMap` והחיפוש סורק את כולם. השם מטעה, המימוש לא.
3. **`distribution`** (לפני S0: `cloud`) — `IndexPackage` עם manifest ו־SHA-256 לכל
   payload, ו־`IndexImporter` שמעתיק ל־staging, מאמת שוב אחרי ההעתקה, ומחליף תיקייה
   עם גיבוי ו־rollback. אינו חשוף דרך ה־API.
4. **`IndexVersion`** — זהות מודל/chunking/backend/precision. **חסרות** זהות
   corpus, `tantivy_schema_version` ו־`document_id_scheme_version`; בלעדיהן אפשר
   לפתוח חבילה שמצביעה ל־`line_id` של קטלוג אחר. S3.
5. **פרופילים ודגלים** — `Fast`/`Balanced`/`Best`, `Weighted`/`RRF`/`Adaptive`,
   ו־`FeatureFlags` שדורסים פרופיל קיים במקום להחזיק העתק שני של ברירות המחדל.
6. **Caches ו־telemetry** — cache תוצאות עם פסילה לפי `generation`, cache embeddings,
   ומוני ריצה. ה־telemetry מבדיל `cache_lookup` מ־`cache_hit`, כדי ש"לא נבדק" לא
   ייראה כ"פספוס". שום דבר מזה אינו יוצא מהתהליך.
7. **benchmark helpers** — תזמון ואחוזונים. **אינם** dataset של רלוונטיות תורנית
   ואינם הוכחת סקייל על ~6.1 מיליון שורות.

## מגבלה ידועה: אינדוקס חוסם חיפוש

אינדוקס דורש `&mut SemanticEngine` וחיפוש דורש `&`, ולכן הם לא יכולים לרוץ במקביל.
`HybridCoordinator::index_books` נוטל את הנעילה **פר-ספר** ולא לכל הקבוצה, כך
שההמתנה חסומה לספר אחד ואינדוקס ספרייה שלמה נשאר קטיע — אבל זו עדיין המתנה, לא
מקביליות.

פתרון אמיתי דורש או interior mutability עדין יותר בתוך ה־engine, או בנייה לאינדקס
staging והחלפה אטומית. זה שייך לחיבור לאוצריא (P6/P7), שבו מוכרע מודל ה־threading —
אינדוקס מלא ארוך שחוסם את ה־UI הוא חסם שם, וכדאי לפתור אותו פעם אחת כמו שצריך.

## מה בדיקת ה־GGUF כן מוכיחה ומה לא

`validate_and_checksum_gguf` קוראת את הקובץ **פעם אחת** ועושה שלושה דברים:

1. מאמתת את ה-header (magic, גרסה 2–3, מניינים סבירים) **לפני** שהיא קוראת את שאר
   הקובץ — קובץ שאינו GGUF נדחה אחרי 24 בייטים, לא אחרי hash של גיגה-בייטים.
2. מפרסרת את כל אזור ה-descriptors: metadata KV (כולל מערכים ומחרוזות) ותיאורי
   tensor, וכך יודעת איפה מתחיל אזור הנתונים ומה ה-offset שכל tensor מצהיר עליו.
   metadata type לא מוכר בגרסה נתמכת נדחה; גם קונטיינר בלי tensors, alignment שאינו
   כפולה של 8 ו-offset שאינו מכבד אותו נדחים.
3. דורשת שהקובץ יהיה גדול דיו כדי להחזיק את מה שהוא עצמו מתאר, בחסם של **ביט אחד
   לאיבר** — נכון לכל טיפוס ggml, כולל הקוונטיזציות הטרנריות האגרסיביות ביותר.

מה שזה **לא**: החסם הוא חסם תחתון. חישוב הגודל המדויק דורש טבלת block sizes של כל
קוונטיזציה, וטעות באחת מהשורות שם *דוחה מודל תקין* — כשל גרוע יותר מקבלת מודל פגום,
כי הוא הופך את הפיצ'ר לבלתי-זמין במקום להיכשל מאוחר עם שגיאה ברורה. לכן הורדה
שנקטעה **בתוך ה-tensor האחרון** עדיין עוברת. גם אימות הורדה אמיתי אינו כאן: checksum
שהמנוע מחשב מהקובץ אינו יכול להעיד על הקובץ. השוואה מול SHA-256 מפורסם שייכת להפצת
המודל (P2/P9); ה-checksum כאן קיים כדי לזהות שהבייטים מאחורי נתיב המודל **השתנו**
בין הרצות, מה שמבטל בשקט כל וקטור שנשמר.

## עמידות ה־manifest: מה מובטח בכל פלטפורמה

* **אטומיות** מובטחת: כתיבה ל-`.tmp`, `fsync`, ואז `rename` על היעד. `load` יודע
  לשחזר גם את מה שקריסה בתוך `save` יכולה להשאיר — `.previous` (עדיפות ראשונה: זה
  manifest שהיה בשירות) ואחריו `.tmp` (מועמד שנכתב ונשטף). שניהם נפרסרים לפני
  שמקדמים אותם, כך שקובץ חצי-כתוב נפסל ונשאר במקומו כעדות.
* **עמידות (durability)** מובטחת **ב-Unix בלבד**: התיקייה נפתחת ומסונכרנת, וכשל
  מחזיר שגיאה. `Ok` במצב כזה היה שקר — הקורא היה רושם התקדמות שהפסקת חשמל עדיין
  יכולה לבטל. ב-Windows אין `fsync` לתיקייה, ולכן ה-rename נשען על הבטחת מערכת
  הקבצים; `Ok` שם אומר "הנתונים הגיעו לדיסק וה-rename בוצע", לא "ה-rename ישרוד
  הפסקת חשמל".
* מסלולי ה-fallback (rename שנדחה, שחזור, כשל fsync) נבדקים על ידי **הזרקת הכשל**
  שבגללו הם קיימים — `failpoints` ב-`manifest.rs`, thread-local ולכן בטוח למקביליות.
  מסלול התאוששות שלא נבדק הוא המסלול שלא עובד כשבאמת צריך אותו.

---

# 4. מה כבר אמיתי ומה עדיין Fake

זה החלק החשוב ביותר במסמך.

## Embedding

הקוד מציג את המערכת כ־GGUF embedding runtime, עם:

```text
model: EMD123/Otzaria-Embedding-V1-Flash-0.6B
dimension: 1024
quantization: Q4
pooling: last-token
max tokens: 512
batch size: 32
```

`load()` מאמת את קונטיינר ה־GGUF ומחשב SHA-256 של הקובץ במעבר אחד, ואז הבחירה נעשית
ב־`backend::select_backend`:

```text
בנייה רגילה (production)               → Err(BackendUnavailable)
--features mock-embedding               → backend "mock-hash-v1"  (אינו מודל)
--features llama-backend                → inference אמיתי מול ה-GGUF
שני ה-features יחד                      → ה-backend האמיתי מנצח
```

ה־stand-in מבוסס SHA-256 ו־feature hashing (ואחריו L2 normalization). הוא **אינו
זמין ב־production** — ראו "מה השתנה ב־PR הראשון" למעלה. וקטור באורך אפס נדחה
בשגיאה במקום להיכנס לאינדקס.

### שני המסלולים

```text
--features mock-embedding (פיתוח/בדיקות):
GGUF container validated + checksummed
       ↓
fake deterministic embedding
       ↓
vector

--features llama-backend (אמיתי):
GGUF
 ↓
Qwen2-BPE tokenizer (parse_special = false)
 ↓
llama.cpp inference, EOS מצורף אחרי החיתוך
 ↓
last-token pooling
 ↓
L2 normalization (ב-EmbeddingRuntime, לא ב-backend)
 ↓
1024-d vector
```

**אסור להתייחס ל־feature hashing כמודל semantic.**

הוא קיים רק כדי לאפשר לפתח ולבדוק את כל שאר ה־pipeline בלי שה־model runtime יהיה
blocker — ובפרט כדי שבדיקות ה־CI ירוצו גם במכונה שאין בה את קובץ המודל.

---

# 5. מודל ה־Embedding

הקונפיגורציה הנוכחית:

```text
Model:
EMD123/Otzaria-Embedding-V1-Flash-0.6B

GGUF:
models/otzaria-embedding-v1-flash-q4.gguf

Quantization:
Q4

Embedding dimension:
1024

Pooling:
last-token

Vector precision:
f32

Maximum tokens:
512

Batch size:
32
```

ההגדרות נמצאות ב־`SemanticConfig` וב־`EmbeddingConfig`.

הממד, הדיוק, `max_tokens`, pooling וה־normalization אינם „הגדרות” אלא **חלק מזהות
האינדקס**: שינוי של אחד מהם פוסל כל וקטור שנשמר. הבחירה הסופית ביניהם היא S1, ורק
אחריה נכון לקפוא על פורמט ארטיפקט גדול.

כיצד קובץ המודל מגיע למכשיר, ומה כבר הוכרע בעניין: [`MODEL_DISTRIBUTION.md`](MODEL_DISTRIBUTION.md).

---

# 6. למה Q4

המודל מיועד לרוץ מקומית, ולכן quantization הוא חלק מרכזי מהארכיטקטורה.

חשוב להבדיל בין:

```text
Model quantization = Q4
```

לבין:

```text
Vector precision = f32
```

אלה שני דברים שונים.

כרגע הכוונה היא:

```text
Q4 model weights
+
f32 output vectors
```

ולא Q4 vectors.

---

# 7. Chunking

המערכת לא אמורה להפוך כל שורה בודדת ל־embedding באופן עיוור.

ה־Chunker מתחשב במבנה הטקסט.

המטרה היא להגיע ליחידות סמנטיות מספיקות, תוך שמירה על הקשר.

הקונפיגורציה הנוכחית כוללת:

```text
min_meaningful_chars = 20
context_window_lines = 2
max_chunk_chars      = 512
min_embeddable_chars = 5
chunking_version     = 1
```

כל אחד מהערכים האלה משנה את הטקסט שהוטמע, ולכן כולם נכנסים יחד ל־`ChunkerConfig::identity()`
שנשמר ב־manifest: שינוי של אחד מהם מבטל את האינדקס בדיוק כמו העלאת `chunking_version`.

---

# 8. Context Window

כאשר שורה קצרה מדי מכדי לשאת משמעות בפני עצמה, ה־chunk יכול לקבל context מהשורות הסמוכות:

```text
previous line × 2
       +
current line
       +
next line × 2
```

ה־context לא אמור לחצות גבולות section.

זו החלטה חשובה במיוחד עבור טקסטים תורניים, שבהם שורה קצרה יכולה להיות מובנת רק מתוך הפסקה.

---

# 9. Semantic ID

לכל chunk יש ID דטרמיניסטי.

הוא מבוסס על:

```text
source_book_key
+
line_id
+
chunking_identity   ← טביעת אצבע של כל ה־ChunkerConfig, לא רק chunking_version
```

ונוצר באמצעות SHA-256.

המטרה היא שה־ID יישאר יציב בין ריצות indexing כל עוד המקור והאלגוריתם לא השתנו.

---

# 10. Chunk Hash

בנוסף ל־semantic ID יש:

```text
chunk_hash
```

המבוסס על הטקסט שנשלח בפועל ל־embedding.

ההבדל:

```text
semantic_id
= identity of logical source location

chunk_hash
= identity of embedded content
```

זה חשוב ל־incremental indexing עתידי.

---

# 11. מבנה הנתונים של ספר

המערכת מקבלת ספר בצורה דומה ל:

```rust
BookForIndexing {
    source_book_key,
    title,
    content_hash,
    is_pdf,
    topics,
    author,
    era,
    base,
    lines,
}
```

וכל שורה מכילה metadata שמאפשר לחזור למקור:

```text
line_id
section_id
text
line_hash
reference
segment
```

המשמעות היא שה־vector לעולם לא אמור להיות detached מהמקור.

---

# 12. Vector Store

## היעד

היעד הוא אינדקס וקטורי מקומי, נפרד לחלוטין מ־Tantivy, ש**נבנה מראש ונפתח read-only**
אצל המשתמש. פתיחה אינה כותבת, אינה מאנדקסת ואינה בונה מחדש.

הקונפיגורציה כרגע:

```text
db path:
semantic_db/zvec        ← שם היסטורי; הספרייה zvec אינה בשימוש

embedding dimension:
1024

collection:
chunks
```

---

## שני ה־stores שקיימים בקוד

```text
VectorStore (store.rs)          ← זה מה שה-engine פותח היום
  └── RwLock<HashMap<String, StoredVectorRecord>>
      + RwLock<HashMap<String, Vec<String>>>   (מפתחות לפי ספר)
  is_persistent() == false — ומצהיר על כך, כדי שה-manifest לא ישקר

ZevcStore (zevc_store.rs)       ← קיים, נבדק, לא מחובר
  └── snapshots לדיסק: payload לכל ספר, SHA-256 למטא-דאטה ולווקטורים,
      פתיחה מחדש שמאמתת checksums
  is_persistent() == true
```

**מה ש־`ZevcStore` אינו:** אינו הספרייה `zvec`, אינו ANN, אינו mmap. הפתיחה טוענת את
**כל** הווקטורים ל־`HashMap` והחיפוש סורק את כולם ב־`O(N·D)` — אותה סיבוכיות בדיוק כמו
ה־store בזיכרון. מה שהוא כן פותר: הווקטורים שורדים restart, והשלמות שלהם נבדקת.

### לכן המשימה (S2) היא:

```text
עכשיו:
SemanticEngine ──תלוי ישירות──▶ VectorStore (בזיכרון)

היעד:
SemanticEngine ──תלוי ב──▶ VectorStoreBackend
                             ├── in-memory   (בדיקות)
                             └── official read-only (מסלול המוצר)
                                  ללא delete/upsert בזמן ריצת האפליקציה
```

ולאחר מכן **למדוד** לפני שבוחרים: `ZevcStore` כ־baseline נכונות ב־1M וב־6M רשומות,
cold-open, p50/p95/p99, peak RSS וגודל דיסק. ANN אמיתי על הדיסק נכנס רק אם המדידה
מוכיחה שסריקה מלאה אינה עומדת בתקציב — ולא מפני ש"ANN" נשמע מהיר יותר. פתיחה שטוענת
את כל הווקטורים ל־RAM אינה קבילה אלא אם מדידה מראה שהיא עומדת בתקציב בכל היעדים.

ה־API של ה־store צריך להישאר abstraction, כדי שהחלפת backend לא תדרוש לשכתב את
ה־semantic engine.

---

# 13. Vector Search הנוכחי

כרגע החיפוש מתבצע על הווקטורים שב־HashMap.

כלומר למעשה:

```text
for every vector:
    calculate cosine similarity
    keep top K
```

זה brute-force.

המורכבות:

```text
O(N × D)
```

כאשר:

```text
N = מספר הווקטורים
D = 1024
```

זה נכון לשני ה־stores: גם `ZevcStore` סורק את כולם. הוא מוסיף persistence, לא אחזור
תת־ליניארי.

**האם זו הארכיטקטורה הסופית — לא ידוע, ומכוון שלא ידוע.** ההכרעה תלויה במדידה של S2
ובממד/דיוק שייבחרו ב־S1: 6.1 מיליון וקטורים ב־128 ממדים int8 הם ~0.72 GiB, וב־1024
ממדים f32 הם ~23.1 GiB. אלה שני עולמות שונים לחלוטין לשאלה "האם סריקה מלאה קבילה".

---

# 14. SemanticEngine

זהו ה־orchestrator של semantic search.

הוא מחזיק:

```rust
SemanticConfig
SemanticManifest
Chunker
VectorStore
EmbeddingRuntime
last_error
```

כלומר:

```text
SemanticEngine
│
├── configuration
├── lifecycle
├── chunking
├── embeddings
├── vector storage
├── manifest
└── indexing/search
```

ה־engine כבר מספק:

```rust
open()
load_model()
unload_model()
index_book()
remove_book()
search()
diff_against_tantivy()
status()
```

שתי הערות שנוגעות לחוזה המוצר:

1. **`store: VectorStore` הוא תלות קונקרטית.** `open()` פותח את ה־store בזיכרון. זה
   הפער המרכזי של S2, ולא פרט מימוש שאפשר לדחות: כל עוד התלות קונקרטית, `ZevcStore`
   אינו יכול להיכנס למסלול הפעיל בלי שינוי חתימות.
2. **`index_book`/`index_books`/`reset_index` הן פעולות אב־טיפוס.** הן משמשות את
   הבדיקות ואת ה־builder העתידי, לא את האפליקציה. במסלול המוצר האפליקציה מתקינה
   ארטיפקט מוכן ופותחת אותו read-only.

---

# 15. Indexing Flow

> **היכן זה רץ:** במכונת build, או בבדיקות. **לא** באפליקציה. אין באוצריא אינדוקס
> ברקע, אין progress stream ואין cancel/resume — ראו [`PRODUCT_CONTRACT.md`](PRODUCT_CONTRACT.md) §4.
> הזרימה כאן היא מה שה־builder של S4 יפעיל, ספר אחד בכל פעם, מתוך אינדקס Tantivy סופי.

ה־flow הנוכחי:

```text
BookForIndexing
       │
       ▼
Chunker
       │
       ▼
SemanticChunk[]
       │
       ▼
EmbeddingRuntime
       │
       ▼
Vec<f32>
       │
       ▼
VectorMetadata
       │
       ▼
VectorStore.insert_batch()
       │
       ▼
Manifest.mark_book_indexed()
       │
       ▼
Manifest.save()
```

האינדוקס משתמש ב־`embed_batch()`: כל ה־chunks של הספר נשלחים בקבוצות בגודל
`embedding_batch_size` (ברירת מחדל 32).

```text
32 chunks
 ↓
embed_batch()
 ↓
32 vectors
```

בנוסף, `index_books()` כותב את ה־manifest **פעם אחת** בסוף במקום פעם לכל ספר —
serialize של כל המפה אחרי כל ספר הופך אינדוקס ספרייה שלמה ל־I/O ריבועי.

הלופ הציבורי (`HybridCoordinator::index_books`) חייב לשחרר את נעילת ה־engine בין
ספרים כדי שחיפושים יוכלו לרוץ, ולכן הוא לא יכול להשתמש ב־`index_books()` של ה־engine.
במקומו הוא קורא ל־`index_book_deferred()` פר-ספר ול־`flush_manifest()` פעם אחת
בסוף (או במסלול שגיאה). מאחר שה־store כרגע נדיף, checkpoint ביניים של manifest
אינו מאפשר resume אחרי restart; כשה־store יהפוך persistent יידרש journal או
checkpoint דלתאי, לא סריאליזציה חוזרת של כל המפה.

---

# 16. Incremental Indexing

> גם זה **צד ה־build**. אצל המשתמש אין diff ואין upsert פר־ספר: מהדורת ספרייה חדשה
> מקבלת ארטיפקט חדש שמותקן במלואו. השימושיות של ה־diff היא לחסוך inference בבנייה
> חוזרת של הארטיפקט, לא לעדכן אינדקס בזמן ריצה.

אחד הדברים החשובים שכבר בנויים הוא manifest.

לכל ספר נשמר:

```text
source_book_key
content_hash        ← החתימה שהקורא התחייב עליה; 0 = "אין חתימה"
line_fingerprint    ← חתימה שהמנוע מחשב מהספר עצמו: שורות + metadata
chunk_count         ← 0 הוא ערך תקין (empty-book marker)
indexed_at
chunking_identity   ← טביעת אצבע של כל ה־ChunkerConfig
normalization_version
```

כך ניתן לדעת:

```text
book unchanged
    ↓
skip

book changed
    ↓
re-index

new book
    ↓
index

book removed
    ↓
delete vectors
```

---

# 17. Diff מול Tantivy

ה־SemanticEngine יודע להשוות:

```text
Tantivy book hashes
```

מול:

```text
Semantic manifest
```

והוא מפיק:

```text
new_books
changed_books
unverifiable_books   ← אין הוכחה שהם מעודכנים; לא ידועים כמשתנים
removed_books
```

בנוסף המבנה מוכן לדווח על:

```text
model mismatch
chunking mismatch
normalization mismatch
```

שלושת הדגלים האלה מחושבים בפועל מהשוואת ה־manifest לקונפיגורציה. כשאחד מהם דולק
כל הספרים מדווחים כדורשי עבודה — עדכון אינקרמנטלי אינו יכול לתקן שינוי מודל או
chunking. `IndexDiff::needs_full_rebuild()` הוא הבדיקה המרוכזת.

---

# 18. Manifest

ה־semantic index צריך לדעת באיזה configuration הוא נוצר.

לכן manifest שומר metadata כגון:

```text
model ID
embedding dimension
pooling
model quantization
vector precision
chunking version
normalization version
```

כאשר engine נפתח, הוא משווה את ה־manifest הנוכחי לקונפיגורציה.

אם יש mismatch:

```text
warning
+
המסלול הסמנטי מושבת (IncompatibleIndex)
+
BM25 ממשיך לעבוד
```

אין rebuild אוטומטי — זו החלטה של הקורא, דרך `reset_index()`. הסיבה: rebuild של
ספרייה שלמה הוא פעולה ארוכה שדורשת את הספרים מהמאגר, ולא משהו שקורה בשקט בזמן open.

manifest שאינו קריא (JSON פגום, גרסת format אחרת) אינו מפיל את ה־engine: הקובץ עובר
`quarantine` לשם קובץ נפרד, נפתח אינדקס חדש, והסיבה נשמרת ב־`SemanticStatus::last_error`.

---

# 19. Model Checksum

יש מקום ארכיטקטוני ל־model checksum.

המטרה:

```text
GGUF file
   ↓
SHA-256
   ↓
manifest
```

כדי למנוע מצב שבו:

```text
model ID = same
but
actual GGUF = different
```

זה מחובר: `load_model()` מחשב SHA-256 של קובץ ה־GGUF (במעבר אחד על הקובץ, יחד עם
אימות הקונטיינר), משווה אותו למה שנשמר ב־manifest, ומשבית את המסלול הסמנטי
כשהקבצים שונים. אם ה־manifest עדיין לא מכיר checksum — הוא נרשם בטעינה הראשונה.

---

# 20. Hybrid Search

ה־hybrid engine הוא החלק שמחבר את שני העולמות:

```text
Existing BM25 results
+
Semantic results
        ↓
Normalization
        ↓
Fusion
        ↓
Ranking
        ↓
Grouping
        ↓
Final results
```

ה־semantic sidecar לא אמור להפעיל בעצמו את Tantivy.

במקום זאת Otzaria מעבירה ל־hybrid layer את התוצאות הלקסיקליות הקיימות.

זה שומר על separation of concerns.

---

# 21. Query Flow

```text
User query
    │
    ├───────────────────┐
    ▼                   ▼
Existing Tantivy      SemanticEngine
    │                   │
    │                   ├── embedding
    │                   │
    │                   └── vector search
    │
    └──────────┬────────┘
               ▼
          HybridCoordinator
               │
               ▼
             Fusion
               │
               ▼
             Ranking
               │
               ▼
            Grouping
               │
               ▼
          Final results
```

---

# 22. Query Classification

המערכת מסווגת query לצורך בחירת משקל lexical/semantic.

סוגים:

```text
ExactReference
Conceptual
Mixed
Short
Unknown
```

הכוונה היא לא לבנות classifier ML נוסף רק בשביל weighting.

הסיווג מבוסס heuristics.

---

# 23. Dynamic Weighting

העיקרון:

```text
fused =
    α × lexical
    +
    (1 - α) × semantic
```

המשקלים הנוכחיים:

| Query type     | Lexical | Semantic |
| -------------- | ------- | -------- |
| ExactReference | 0.80    | 0.20     |
| Short          | 0.70    | 0.30     |
| Mixed          | 0.50    | 0.50     |
| Conceptual     | 0.30    | 0.70     |
| Unknown        | 0.50    | 0.50     |

המשמעות:

```text
"מסכת ברכות דף כ"
       ↓
BM25 חשוב יותר

"איך חז"ל מסבירים את היחס בין..."
       ↓
Semantic חשוב יותר
```

המשקלים האלה הם **heuristic ראשוני**, לא תוצאה של benchmark איכותי.

זה מקום שצריך ניסוי ומדידה.

---

# 24. Score Normalization

BM25 ו־cosine similarity נמצאים בסולמות שונים.

לכן אסור פשוט לעשות:

```text
BM25 + cosine
```

המערכת מנרמלת.

BM25:

```text
score / (k + score)
```

Semantic:

```text
(cosine + 1) / 2
```

ואז ניתן לבצע fusion.

---

# 25. Fusion Strategies

קיימות שתי אסטרטגיות:

## Weighted Score Fusion

```text
α × lexical
+
(1 - α) × semantic
```

זו כרגע האסטרטגיה המרכזית.

## Reciprocal Rank Fusion

```text
RRF(rank) = 1 / (k + rank)
```

RRF שימושי במיוחד אם calibration של ציוני BM25 ו־semantic יהיה קשה.

שתי השיטות קיימות כ־primitives, ולא צריך להניח שאחת מהן כבר הוכחה כטובה יותר.

---

# 26. Result Provenance

תוצאה יכולה להגיע מ:

```text
Lexical
Semantic
Both
```

זה חשוב.

אם result הופיע בשני המנועים, זה מידע שימושי עבור ranking וגם debugging.

אסור לאבד את המידע הזה במהלך fusion.

---

# 27. Grouping

קיימים שני מנגנוני grouping:

```text
SameSection
IdenticalText
```

## SameSection

תוצאות מקובצות לפי:

```text
(section_id, file_path)
```

והתוצאה הטובה ביותר הופכת ל־representative.

---

## IdenticalText

תוצאות עם אותו:

```text
line_hash
```

מקובצות.

כך ניתן למנוע מצב שבו אותו טקסט מופיע שוב ושוב בתוצאות.

---

# 28. למה Grouping חשוב

Semantic search יכול להחזיר הרבה chunks סמוכים מאותו קטע.

ללא grouping:

```text
Result 1 → same section
Result 2 → same section
Result 3 → same section
Result 4 → same section
Result 5 → same section
```

המשמעות היא שה־top 10 יכולים למעשה לייצג רק passage אחד.

Grouping אמור למנוע את זה.

---

# 29. Search Modes

המערכת מתוכננת לשלושה מצבים:

```text
Hybrid
LexicalOnly
SemanticOnly
```

### Hybrid

שני המנועים.

### LexicalOnly

החיפוש הקיים בלבד.

### SemanticOnly

semantic retrieval בלבד.

---

# 30. Graceful Degradation

זה קריטי.

Semantic search הוא enhancement.

אם:

```text
model loading fails
OR
embedding fails
OR
vector search fails
```

אסור שהחיפוש כולו יקרוס.

ב־Hybrid:

```text
Semantic failure
      ↓
log error
      ↓
Lexical fallback
      ↓
return BM25 results
```

זה חלק מהארכיטקטורה ולא workaround.

**אבל התדרדרות אינה רשות להסתיר.** `search_mode` הוא המצב שרץ בפועל ו־`fallback_reason`
אומר למה. וב־`SemanticOnly` **אין** fallback ל־BM25: מצב שהמשתמש ביקש בו חיפוש סמנטי
מחזיר כשל או תוצאה ריקה מפורשת, ולא תוצאות לקסיקליות שמתחזות לסמנטיות.

---

# 31. FFI / Flutter

ה־Rust crate מיועד להיות משולב בתוך Otzaria באמצעות FFI / `flutter_rust_bridge`.

המטרה היא ש־Flutter **לא יכיר** את:

```text
GGUF
Chunker
VectorStore
Manifest
Fusion implementation
```

Flutter צריך לראות API ברמה של:

```text
search()
status()                     ← missing / installing / ready / incompatible /
                               corrupt / model_missing / model_incompatible /
                               unsupported_platform
install_official_index()     ← התקנת ארטיפקט מוכן, לא אינדוקס
```

ה־Rust layer מחזיק את ה־implementation details.

שימו לב למה שאין ברשימה: `index_books` על מיליוני שורות מ־Dart. העברת הספרייה דרך FFI
היא בדיוק מה שהחוזה שולל — הן מטעמי ביצועים והן מפני שה־metadata כבר שמור ב־Tantivy
ואין לבנות אותו שוב בצד Dart.

---

# 32. Boundary מול Otzaria

הגבול הרצוי:

```text
Flutter
   │
   ▼
Otzaria search layer
   │
   ├── Tantivy
   │
   └── Semantic sidecar
            │
            ▼
       Hybrid result
```

החיפוש הקיים נשאר בעל הבית של lexical retrieval.

ה־semantic sidecar אינו צריך לייבא או לנהל את Tantivy בעצמו.

---

# 33. Data Ownership

## Tantivy / Otzaria

אחראים על:

```text
lexical index
BM25
existing search DB
existing book representation
lexical retrieval
```

## Semantic sidecar

אחראי על:

```text
embedding
semantic chunks
vectors
vector metadata
semantic manifest
semantic retrieval
semantic ranking/fusion
```

---

# 34. Error Handling

יש הפרדה בין:

```text
EmbeddingError
VectorStoreError
ManifestError
ChunkingError
FusionError
ConfigError
```

המטרה היא שה־caller יוכל להבין:

```text
model missing
≠
vector DB failure
≠
manifest incompatibility
≠
bad input
```

וזה חשוב במיוחד עבור fallback.

---

# 35. Performance Problems That Still Need Solving

כרגע קיימים מספר bottlenecks ברורים.

## Vector search

כרגע:

```text
brute-force scan, O(N·D)
```

נמדד: 84–208ms לשאילתה על 200k×1024 (min/max באותה ריצה, אותה מכונה). בהערכה
**ליניארית** — לא במדידה — זה ~2.5–3.2s על 6.1 מיליון שורות באותו ממד ודיוק. זה מספר
שמחייב או ממד/דיוק קטנים יותר (S1) או ANN (S2), ולא אופטימיזציה נקודתית.

---

## Persistence במסלול הפעיל

הווקטורים במסלול שה־engine פותח חיים ב־RAM ואינם שורדים restart. `ZevcStore` פותר את
זה אך אינו מחובר, ופתיחה שלו טוענת בכל מקרה את הכול ל־RAM. S2.

---

## Cold-open ותקציב זיכרון

לא נמדדו בכלל בקנה מידה של הספרייה. `cargo bench` מודד חיפוש, לא פתיחה. אלה מספרים
שחייבים להיות בשער הקבלה של S2, כי הם מה שקובע אם ארטיפקט של מיליוני שורות בכלל
נפתח על מכשיר של משתמש.

---

## Embedding latency ב־inference אמיתי

ה־backend האמיתי קיים, וה־pipeline משתמש ב־`embed_batch()` (ברירת מחדל 32). מה שעדיין
לא נמדד באופן שיטתי: tokens/sec, זיכרון ו־CPU לכל context, וזמן ההטמעה של שאילתה
בודדת על מכשירי יעד. השאילתה היא ה־inference היחיד שרץ אצל המשתמש, ולכן ה־latency שלה
הוא מדד מוצר ולא סקרנות.

---

# 36. מה צריך להיות השלב הבא

הסדר המחייב הוא S0–S8 ב־[`שלבי ויעדי התקדמות.md`](../שלבי%20ויעדי%20התקדמות.md). מה
שנוגע למאגר הזה:

## ✅ נעשה: Real Embedding Runtime (PR #2)

inference אמיתי מול GGUF, tokenizer, EOS, last-token pooling ו־L2, מאומתים מול
golden vectors. מאחורי `--features llama-backend`.

## ✅ נעשה: Batch Embeddings

האינדוקס קורא ל־`embed_batch()`, וה־backend האמיתי מבצע batching מרובה־סדרות אמיתי.

## ✅ נעשה: Manifest Compatibility

עשרת הממדים נבדקים, כולל SHA-256 של קובץ המודל.

## ✅ נעשה: S0 — יישור חוזה המוצר

הגדרת האינדקס הרשמי כ־read-only, ביטול המונח `cloud`, יישור README/CODE_MAP/מפת
דרכים והכרעת הפצת המודל.

---

## S1: איכות הייצוג ובחירת ממד ודיוק

dataset של שאילות תורניות עם רלוונטיות מסומנת; השוואת `line` מול
`title + reference + line` מול שורות הקשר; מדידת 128/256/512/1024 ו־f32/f16/int8;
Recall@K, MRR, nDCG וגודל הארטיפקט.

התוצר הוא **החלטה כתובה** שמקפיאה `embedding_text_version`, ממד, דיוק, `max_tokens`,
pooling ו־normalization כחלק מזהות האינדקס. בלי זה כל פורמט ארטיפקט שנקפא הוא ניחוש.

---

## S2: backend מתמיד, read-only ובר־סקייל

1. להפוך את `SemanticEngine` לתלוי ב־`VectorStoreBackend`.
2. להוסיף מצב `official-read-only` שאינו חושף delete/upsert בזמן ריצת האפליקציה.
3. למדוד `ZevcStore` ב־1M וב־6M: cold-open, p50/p95/p99, peak RSS, דיסק.
4. לבחור ANN אמיתי על הדיסק **רק** אם המדידה מחייבת, ולא לשנות שם בלבד.
5. לאמת reopen, קובץ קטוע, checksum שגוי, הפסקת חשמל בהתקנה ושני תהליכי קריאה.

שער קבלה: פתיחה מחדש מחזירה `vectors_persisted=true`, והחיפוש עומד בתקציבים שנקבעו
**מראש**.

---

## S3: חוזה ארטיפקט רשמי

להרחיב את `IndexVersion` בזהות corpus, `tantivy_schema_version` ו־
`document_id_scheme_version`; להכריע אם `.oix` הוא תיקייה מוגדרת או ארכיון יחיד;
לאמת התאמה בספירות ובגודל ולא רק קיום קבצים; להפריד builder כותב מ־runtime קורא;
ולהגדיר התקנה אטומית עם rollback.

שער קבלה: ארטיפקט שנבנה במכונה אחת נפתח במכונה נקייה בלי כתיבה מחדש, ומסרב להיפתח
אם שונה קובץ אחד או זהות אחת.

---

## ❌ מה **אינו** בתור

- אינדוקס ברקע, progress stream, cancel/resume.
- diff/upsert פר־ספר בזמן ריצת אוצריא, ו־chunk-level incremental indexing אצל המשתמש.
- overlay ניתן לכתיבה לספרי משתמש.

`chunk_hash` וה־semantic IDs נשארים שימושיים לצד ה־build (בנייה חוזרת של ארטיפקט
בלי לחשב מחדש מה שלא השתנה), לא לעדכון בזמן ריצה.

---

# 37. דברים שלא כדאי לעשות

## ❌ לא להחליף את Tantivy

המערכת החדשה היא sidecar.

---

## ❌ לא להכניס את כל ה־semantic metadata ל־DB הקיים

ה־semantic index צריך lifecycle עצמאי.

---

## ❌ לא לבצע inference בתוך Flutter

Inference צריך להישאר Rust-side.

---

## ❌ לא לתת ל־Flutter להכיר את vector backend

Flutter צריך לדבר מול API יציב.

---

## ❌ לא להניח שה־current embedding הוא semantic

הוא test fallback בלבד.

---

## ❌ לא להניח ש־backend מתמיד כבר מחובר

`ZevcStore` קיים ונבדק, אבל `SemanticEngine::open()` פותח את ה־store בזיכרון. ובנוסף:
`ZevcStore` אינו ANN, אינו mmap ואינו הספרייה `zvec` — הוא סורק את כל הווקטורים.

---

## ❌ לא להוסיף overlay לספרי משתמש

ספרים אישיים נשארים לקסיקליים בלבד. overlay דורש מודל פעיל אצל המשתמש, אינדוקס בזמן
ריצה ומיזוג של שני מקורות וקטורים עם זהויות שונות — הכול מחוץ ל־MVP.

---

## ❌ לא להוסיף אינדוקס ברקע באפליקציה

אין progress stream, אין cancel/resume ואין מסך התקדמות אינדוקס. מה שהמשתמש מחכה לו
הוא **התקנת** ארטיפקט: הורדה, checksum, החלפה אטומית.

---

## ❌ לא לקרוא לזה „ענן”

אריזה והתקנה של קבצים סטטיים אינן שירות. המודול נקרא `distribution` מסיבה זו, ושאילתת
המשתמש אינה יוצאת מהמכשיר.

---

## ❌ לא לפתוח ארטיפקט שזהותו אינה תואמת

`line_id` הוא `catalogue_order` ב־32 הסיביות הגבוהות + מספר השורה בנמוכות. ספר אחד
שנוסף באמצע הקטלוג מזיז מזהים של ספרים רבים, ואינדקס מגרסה אחרת יצביע לשורות **לא
נכונות** — לא רק יפספס. אי־התאמה = כשל מפורש.

---

# 38. Current Development Philosophy

הפרויקט בנוי בשכבות כדי שאפשר יהיה להחליף implementation בלי לשבור את כל המערכת.

```text
                Public API
                    │
                    ▼
              Hybrid Layer
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
      Semantic             Lexical
       Engine              (external)
          │
    ┌─────┼─────┐
    ▼     ▼     ▼
 Chunker Embed Store
```

כל שכבה צריכה להישאר כמה שיותר עצמאית.

---

# 39. מה נחשב "Done"

Semantic Search לא ייחשב production-ready רק כאשר הקוד מתקמפל.

ה־Definition of Done צריך לכלול:

### Embedding

* [x] GGUF model נטען באמת (`--features llama-backend`)
* [x] tokenizer עובד — Qwen2-BPE, `parse_special = false`
* [x] inference עובד
* [x] pooling תואם למודל — last-token, EOS מצורף אחרי החיתוך
* [x] normalization תקין — במעבר יחיד ב־`EmbeddingRuntime`
* [x] Q4 inference נבדק מול golden vectors (`token_ids` מדויק, ואז סבילות וקטורית)
* [ ] latency של הטמעת שאילתה על מכשירי היעד

### Vector Store

* [x] persistence — קיים ב־`ZevcStore`
* [ ] persistence **במסלול שה־engine פותח** (S2)
* [ ] מצב official-read-only ללא delete/upsert בזמן ריצה (S2)
* [ ] ANN או הוכחה שאין בו צורך — נמדד: 84–208ms לשאילתה על 200k×1024 (min/max באותה
  ריצה), כלומר ~2.5–3.2s בקנה מידה של הספרייה **בהערכה ליניארית**, לא במדידה
  (`cargo bench`)
* [ ] cold-open, peak RSS וגודל דיסק ב־1M וב־6M רשומות (S2)
* [x] reopen אחרי restart — עקבי (רשומות ספרים לא שורדות backend נדיף)
* [x] insert/update/delete
* [x] filtering
* [x] dimension validation

### Artifact / Distribution

* [x] manifest חבילה + SHA-256 לכל payload
* [x] התקנה אטומית עם staging, גיבוי ו־rollback
* [x] אימות מחדש של ה־payload **אחרי** ההעתקה, לא רק במקור
* [ ] זהות corpus, `tantivy_schema_version`, `document_id_scheme_version` (S3)
* [ ] התאמת ספירות ספרים/וקטורים וגודל מול ה־manifest (S3)
* [ ] חשיפת ה־importer דרך ה־API / FFI (S3–S5)
* [ ] builder שמפיק את הארטיפקט מ־Tantivy הסופי (S4)

### Indexing (צד ה־build בלבד)

* [x] initial full index
* [x] incremental book indexing (ברמת ספר, כולל PDF)
* [ ] chunk-level reuse — כרגע דילוג ברמת ספר שלם לפי fingerprint
* [x] removed-book cleanup
* [x] empty-book marker
* [x] model mismatch detection (כולל SHA-256 של קובץ המודל)
* [x] chunking mismatch detection

### Hybrid

* [x] BM25 + semantic
* [x] score normalization
* [x] dynamic weighting
* [x] RRF בשימוש — נבחר לפי `FusionStrategy` בפרופיל
* [ ] RRF benchmark — איזו אסטרטגיה טובה יותר עדיין לא נמדד
* [x] threshold לתוצאה סמנטית לא רלוונטית
* [x] grouping
* [x] provenance
* [ ] `total_count` אמיתי (כרגע מספר המועמדים שנכנסו ל־fusion)

### Reliability

* [x] semantic failure → BM25 fallback (ומדווח ב־`fallback_reason`)
* [x] corrupted semantic DB does not corrupt Tantivy
* [x] manifest writes atomic (temp → fsync → rename)
* [x] index rebuild recoverable (`reset_index()`)

### Performance

* [ ] embedding benchmark — ה־backend קיים; המדידה השיטתית עדיין לא נעשתה
* [ ] indexing benchmark (צד ה־build)
* [x] vector search benchmark — [`benches/vector_search.rs`](../benches/vector_search.rs)
* [x] תשתית מדידה גנרית — [`src/benchmark/mod.rs`](../src/benchmark/mod.rs)
* [ ] memory benchmark
* [ ] cold-open benchmark
* [ ] end-to-end query latency

### Quality

* [ ] Hebrew benchmark dataset
* [ ] Recall@K
* [ ] MRR
* [ ] NDCG
* [ ] comparison against BM25-only
* [ ] comparison against semantic-only
* [ ] hybrid comparison

---

# 40. Benchmarking Plan

צריך לבנות dataset אמיתי של queries.

לדוגמה:

```text
Query
Expected relevant sources
Expected relevant sections
```

ולבדוק:

```text
BM25
Semantic
Hybrid
```

בנפרד.

מדדים:

```text
Recall@5
Recall@10
Recall@20

MRR@10

NDCG@10

Latency p50
Latency p95

Memory
```

רק אחרי benchmark כזה כדאי לשנות את:

```text
α = 0.8 / 0.7 / 0.5 / 0.3
```

או להחליט ש־RRF עדיף.

כרגע אלה heuristics.

---

# 41. Important Technical Debt

ה־technical debt המרכזי כרגע הוא לא בארכיטקטורה, אלא בפער בין מה שקיים ב־crate לבין
מה שנמצא במסלול הפעיל:

```text
Architecture
     │
     ├── EmbeddingRuntime + backend contract
     │        └── ✅ inference אמיתי קיים (feature)
     │
     ├── VectorStoreBackend contract
     │        ├── ZevcStore קיים ונבדק — אך ה-engine לא תלוי בו   (S2)
     │        └── אין ANN/mmap, וסקייל 6M לא נמדד                  (S2)
     │
     ├── IndexVersion
     │        └── חסרות זהות corpus / Tantivy / ID scheme          (S3)
     │
     └── distribution (package + importer)
              └── קיים ונבדק — אך לא חשוף ב-API, ואין builder      (S3–S4)
```

זה דווקא מצב טוב יחסית: החוזים במקום, וכל פער הוא חיבור או הרחבה ולא שכתוב.

לא צריך לזרוק את הארכיטקטורה. צריך לחבר את מה שקיים ולמדוד אותו.

---

# 42. Recommended Development Order

הסדר המומלץ:

```text
✅ 1. Real GGUF inference
✅ 2. Validate generated embeddings (golden vectors)
✅ 3. Batch inference
✅ 4. S0 — product contract alignment
   ↓
   5. S1 — quality dataset → dimension & precision decision
   ↓
   6. S2 — persistent read-only backend, measured at 1M and 6M
   ↓
   7. S3 — artifact identity contract & atomic install
   ↓
   8. S4 — builder from the final Tantivy index   (otzaria_search_engine)
   ↓
   9. S5 — repin, open/install API, FFI            (otzaria_search_engine)
   ↓
  10. S6–S7 — artifact/model management, BLoC & UI  (otzaria)
   ↓
  11. S8 — release gates on the full platform matrix
```

S1 לפני S2 בכוונה: ממד ודיוק קובעים אם סריקה מלאה בכלל קבילה, ולכן בחירת backend לפני
בחירת ממד היא בחירה בעיניים עצומות. שני השלבים יכולים לרוץ במקביל, אבל אין לקפוא על
פורמט ארטיפקט לפני שה־מדידה של S1 בידיים.

כיול עדין של fusion נשאר אחרון: הוא כיוון ההגה, לא המנוע.

---

# 43. Current Mental Model for New Developers

אם נכנס מפתח חדש לפרויקט, הוא צריך לחשוב עליו כך:

```text
This repository is NOT:
"another search database"

It IS:

A semantic retrieval sidecar for Otzaria.

Existing:
Tantivy/BM25
        │
        │ lexical candidates
        ▼
 ┌─────────────────┐
 │ Hybrid Layer    │
 └────────┬────────┘
          ▲
          │ semantic candidates
          │
 ┌────────┴────────┐
 │ Semantic Engine │
 ├─────────────────┤
 │ Chunking        │
 │ Embedding       │
 │ Vector Store    │
 │ Manifest        │
 │ Incremental     │
 │ Retrieval       │
 └─────────────────┘
```

---

# 44. Current State in One Paragraph

הפרויקט מחזיק semantic sidecar עצמאי עם chunking, stable IDs, manifest, incremental
book diff, חוזה backend ל־embedding עם **inference אמיתי** מאחורי feature ומאומת מול
golden vectors, חוזה backend ל־store, שני stores (בזיכרון ו־snapshot לדיסק), hybrid
fusion עם פרופילים ו־RRF בשימוש, thresholds, grouping, caches, telemetry, אב־טיפוס של
אריזה וייבוא אטומי, ו־Rust API seam ל־FFI. הוא עדיין אינו מוצר: ה־engine פותח את
ה־store שבזיכרון ולכן במסלול הפעיל אין persistence; אין ANN ואין הוכחת סקייל על ~6.1
מיליון שורות; `IndexVersion` אינו קושר את הארטיפקט ל־corpus מסוים; אין builder שמפיק
ארטיפקט מ־Tantivy; ובאוצריא ה־BLoC וה־UI אינם מפעילים את המסלול.

---

# 45. Developer Rule

כאשר מוסיפים feature חדש, יש לשאול:

1. האם הוא שייך ל־semantic subsystem או ל־hybrid layer?
2. האם הוא צריך להיות visible דרך FFI?
3. האם הוא משנה את משמעות ה־vectors?
4. אם כן, האם צריך להעלות version?
5. האם הוא משפיע על ה־existing Tantivy search?
6. האם semantic failure עדיין מאפשר BM25?
7. האם אפשר לבדוק אותו ללא Flutter?
8. האם אפשר להחליף את ה־backend בלי לשנות את ה־API?

אם התשובה ל־5 היא כן, יש לבחון היטב אם feature באמת שייך לריפו הזה.

---

# 46. Repository Map

```text
src/
│
├── lib.rs            → crate architecture, module exports, product contract
├── main.rs           → development CLI
├── errors.rs         → error taxonomy
│
├── api/              → Flutter / FFI boundary
│
├── semantic/
│   ├── chunker.rs        → text → semantic chunks
│   ├── embedding.rs      → validation, batching, normalization (choke point)
│   ├── embedding_cache.rs→ LRU over embedded texts
│   ├── backend.rs        → EmbeddingBackend contract + selection
│   ├── llama_backend.rs  → real inference (feature `llama-backend`)
│   ├── engine.rs         → semantic orchestration
│   ├── manifest.rs       → index compatibility + state
│   ├── store.rs          → in-memory vector store (the active one)
│   ├── store_backend.rs  → VectorStoreBackend contract
│   ├── zevc_store.rs     → disk snapshots, full scan, not wired
│   ├── versioning.rs     → IndexVersion identity
│   └── types.rs          → semantic domain types
│
├── hybrid/
│   ├── coordinator.rs    → complete search orchestration
│   ├── fusion.rs         → score fusion / RRF
│   ├── ranking.rs        → query classification + weighting
│   ├── grouping.rs       → result grouping/deduplication
│   ├── metadata_ranker.rs→ facet-derived bonuses
│   ├── hebrew_normalizer.rs → nikud/taamim, query language
│   └── cache.rs          → query result cache
│
├── config/
│   ├── profiles.rs       → Fast/Balanced/Best + fusion strategy
│   └── feature_flags.rs  → per-run overrides
│
├── distribution/
│   ├── package.rs        → package manifest + payload checksums
│   └── importer.rs       → staged atomic install
│
├── telemetry/            → in-process counters (no network)
└── benchmark/            → timing & percentile helpers
```

---

# 47. The Most Important Next Task

### Decide the representation (S1), then wire a persistent read-only backend (S2).

Replacing the fake embedding — the task this section used to name — is **done**:
`--features llama-backend` runs real GGUF inference, verified against committed
golden vectors. Semantic quality can now actually be measured, which is exactly why
measuring it is the next task rather than an optional one.

The order matters and is not interchangeable:

```text
S1  labelled rabbinic query set
        ↓
    Recall@K / MRR / nDCG per representation and per dimension
        ↓
    frozen: embedding_text_version, dim, precision, max_tokens, pooling, norm
        ↓
S2  SemanticEngine depends on VectorStoreBackend, not VectorStore
        ↓
    official-read-only mode (no delete/upsert at app runtime)
        ↓
    measured at 1M and 6M: cold-open, p50/p95/p99, peak RSS, disk
        ↓
    ANN on disk only if the measurement demands it
```

Doing S2 first means choosing a storage strategy without knowing whether the
vectors are 0.72 GiB or 23.1 GiB — a 32× spread that decides the answer for you.

Only after both does it make sense to freeze an artifact format (S3), build it from
Tantivy (S4), and tune hybrid ranking against real relevance numbers.

---

# 48. Current Project Status

**Architecture:** 🟢 Strong foundation

**Chunking:** 🟢 Implemented

**Manifest:** 🟢 Implemented — real compatibility validation + atomic writes

**Incremental book detection:** 🟢 Implemented

**Index-incompatibility handling:** 🟢 Implemented — disables the semantic path

**Embedding abstraction:** 🟢 Implemented

**Actual embedding inference:** 🟢 Implemented behind `--features llama-backend`,
verified against golden vectors. A default build still has no backend at all, by
design — it fails loudly rather than serving fake vectors.

**Vector abstraction:** 🟢 `VectorStoreBackend` exists

**Persistent vector database:** 🟡 `ZevcStore` persists and reopens with verified
checksums — but the engine still opens the in-memory store, so **the active path has
no persistence**.

**ANN retrieval:** 🔴 Missing — and both stores scan everything. Brute force measures
84–208ms per query over 200k×1024 on one machine (`cargo bench` prints
min/median/max; the spread is the machine, not the code). Extrapolated linearly, that
is ~2.5–3.2s over the 6,058,210-line library — an extrapolation, not a measurement.
Whether ANN is needed at all is an S2 decision that depends on the S1 dimension
choice.

**Artifact identity:** 🟡 `IndexVersion` covers model/chunking/backend; corpus id,
Tantivy schema version and ID-scheme version are missing (S3).

**Packaging & atomic install:** 🟡 Implemented and tested, not exposed through the
API, and no builder produces an official artifact yet.

**Hybrid fusion:** 🟢 Implemented — weighted / RRF / adaptive, chosen by profile

**Search modes (lexical / hybrid / semantic):** 🟢 Implemented

**Graceful degradation:** 🟢 Implemented and observable

**Dynamic weighting:** 🟡 Initial heuristic, unmeasured

**Grouping:** 🟢 Implemented

**Profiles / feature flags / caches / telemetry:** 🟢 Implemented

**FFI boundary:** 🟡 Seam only; bindings are built in `otzaria_search_engine`

**Production integration with Otzaria:** 🔴 The gateway and repository expose the
API; the BLoC and UI do not invoke it.

**Search-quality benchmark:** 🔴 Missing — measurement helpers exist, a labelled
rabbinic relevance dataset does not (S1)

**Production readiness:** 🔴 Not yet

---

# 49. Bottom Line for Contributors

The project should **not** be restarted or architecturally redesigned at this point.

The current architecture already separates the major concerns correctly:

```text
Existing search
      ≠
Semantic search
      ≠
Hybrid ranking
      ≠
Flutter API
```

The main work now is to connect and measure what already exists:

```text
✅ Fake embedding              →  Real GGUF inference
✅ Per-chunk inference         →  Batch inference

Engine bound to VectorStore    →  Engine bound to VectorStoreBackend,
                                  official-read-only, measured at 6M

Model-only index identity      →  Corpus + Tantivy + ID-scheme identity

Package written by tests       →  Artifact built from the final Tantivy index

Heuristic weights              →  Benchmark-driven ranking
```

Two rows deliberately absent, because they are out of scope rather than pending:
chunk-level incremental indexing at user runtime, and a writable overlay for
personal books. See [`PRODUCT_CONTRACT.md`](PRODUCT_CONTRACT.md) §9.

The most important principle remains:

> **The semantic system must enhance Otzaria's existing search without taking ownership of it or becoming a single point of failure.**
