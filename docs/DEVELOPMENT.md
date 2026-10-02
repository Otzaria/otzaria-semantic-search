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
| אימות חבילת ONNX + checksum      | ממומש (נתיב שאינו `.onnx` נדחה; protobuf walk במעבר אחד, external data, `tokenizer.json`) |
| ONNX inference אמיתי             | ממומש מאחורי `--features onnx-backend`, מאומת מול golden vectors — ה־backend היחיד |
| GGUF / llama.cpp                 | **הוסר**; הקומיט האחרון שיש בו תמיכה הוא `62f0c44` |
| deterministic embedding fallback | ממומש, **מחוץ ל־production** (feature `mock-embedding`) |
| Batch embedding                  | ממומש (האינדוקס משתמש בו)  |
| VectorStore abstraction          | קיים                       |
| `VectorStoreBackend` trait       | קיים; **ה־engine עדיין אינו תלוי בו** |
| In-memory vector store           | ממומש — וזה מה שה־engine פותח |
| סט וקטורים רשמי (`.oxv`)          | ממומש — segments של int8 ממופים לזיכרון, ממוענים לפי הטקסט; התקנה, delta, דחיסה, שחזור, scrub ([`ARTIFACT_CONTRACT.md`](ARTIFACT_CONTRACT.md)) |
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
| חבילת אינדקס + התקנה              | ממומשים ומאמתים במלואם (`distribution`), עם שחזור התקנה שנקטעה ו־`fsync`; **לא חשופים ב־API** |
| מסלול ריצה read-only              | ממומש — `OfficialSemanticIndex` פותח סט מותקן ומחזיר hits; ה־`CandidateResolver` של המארח קושר אותם לשורות חיות |
| עוגן אמון לשחרור                 | המכניזם קיים (SHA-256 של ה־manifest, מפורסם בנפרד); **אין מי שמפרסם ואין חתימה** |
| זהות (`IndexVersion`)             | מלאה — מתכון השורות ומפתח / משפחת המודל וחבילות השאילתה / store; נדחית לפי שדה |
| builder שמייצר את הווקטורים עצמם | קיים — `build` מחיל את המתכון, מטמיע ומפיק base segment, חבילה ו־`release.json` |
| צינור הבנייה של הספרייה          | `export-plan` / `embed-shard` קיימים; ה־merge לפי מפתחות, warehouse ו־delta — S7 |
| חיבור ל־Tantivy חי | **לא קיים כאן** — ה־resolver ועמודת `chunkKey` הם של `otzaria_search_engine` (P1–P4) |
| Production persistence במסלול הפעיל | קיימת ונמדדה: סט של 6.0M slots נפתח ב־3.8 ms |
| אחזור תת־ליניארי (ANN)            | **אין, ואין צורך** (S2b): סריקה מלאה מדויקת ב־int8 — 69 ms בחוט אחד, 17 ms בעשרה, על 6.0M slots |
| UI סמנטי באוצריא                 | **לא קיים** (S7)           |

הנקודה החשובה ביותר למפתח חדש:

> **זהו כרגע skeleton ארכיטקטוני עובד חלקית, לא מנוע semantic production-complete.**
>
> מה שכן ממומש באמת: inference אמיתי מול המודל, שלושת מצבי החיפוש, fusion עם פרופילים,
> caches, telemetry, סט וקטורים ממוען־תוכן עם התקנה, delta, דחיסה ושחזור, חוזה זהות
> שקושר אותו למתכון ולמשפחת המודל, ומסלול ריצה read-only שסורק אותו ומחזיר מפתחות
> ורשומות. מה שחסר כדי שיהיה מוצר: ה־resolver מעל Tantivy חי ועמודת `chunkKey`
> (P1–P4), צינור הבנייה והפרסום של הספרייה (S7, L1, U1), והפעלה באפליקציה (A1).

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

> **רשומה בלבד.** ה־backend הזה — GGUF דרך llama.cpp — הוסר יחד עם `P2_INFERENCE_SPIKE.md`,
> `P2_REFERENCE_VECTORS.md` ווקטורי הזהב שלו; כולם בהיסטוריה של git עד `62f0c44`. ה־backend
> היחיד היום הוא ONNX ([`ONNX_BACKEND.md`](ONNX_BACKEND.md)), ושער ה־parity שלו בנוי על אותו
> עיקרון: `token_ids` מדויקים קודם, cosine אחריהם.

`--features llama-backend` סיפק inference אמיתי מול קובץ GGUF, ולא stand-in. מה שנלמד בו:

1. **בדיקת ה־parity הראשית היא שוויון `token_ids` מדויק, לא cosine.** נמדד ש־BOS
   מוטעה מקבל cosine *גבוה יותר* (0.9947938) מרפרנס לגיטימי (0.9947909), ולכן אין סף
   cosine שמפריד ביניהם.
2. **בנייה רגילה נשארת בלי backend.** זה נשאר כך גם עם ONNX: `onnx-backend` אינו ברירת
   מחדל.
3. **הבחירה בין Candle ל־llama.cpp לא נמדדה.** הוכרעה llama.cpp לפני שנמדד משהו; ההחלטה
   הוחלפה כשהמודל עבר ל־ONNX.

## מה הוסיף PR #3 (Hybrid, פרופילים, אב־טיפוס persistence)

1. **`VectorStoreBackend`** — חוזה משותף לשני ה־stores. במצב של PR #3 ה־engine היה
   תלוי ב־`VectorStore` הקונקרטי, ולכן ה־trait היה הכנה ולא נקודת החלפה בפועל.
   **נסגר ב־S2a:** החוזה פוצל לצד קורא וצד כותב, וה־engine תלוי בכותב כ־trait object.
2. **`ZevcStore`** — snapshots לדיסק עם checksum לכל payload, ופתיחה מחדש שמאמתת
   אותם. **הוא אינו הספרייה `zvec`, אינו ANN ואינו mmap:** הפתיחה טוענת את כל
   הווקטורים ל־`HashMap` והחיפוש סורק את כולם. השם מטעה, המימוש לא.
3. **`distribution`** (לפני S0: `cloud`) — `IndexPackage` עם manifest ו־SHA-256 לכל
   payload, ו־`IndexImporter` שמעתיק ל־staging, מאמת שוב אחרי ההעתקה, ומחליף תיקייה
   עם גיבוי ו־rollback. אינו חשוף דרך ה־API. הורחב ב־S3 לשער אימות מלא — ראו
   [`ARTIFACT_CONTRACT.md`](ARTIFACT_CONTRACT.md).
4. **`IndexVersion`** — זהות מודל/chunking/backend/precision. במצב של PR #3 חסרו זהות
   corpus, `tantivy_schema_version` ו־`document_id_scheme_version`, ולכן היה אפשר
   לפתוח חבילה שמצביעה ל־`line_id` של קטלוג אחר. **נסגר ב־S3:** שלוש קבוצות זהות
   (corpus/model/store), 17 שדות, כולם נבדקים ונדחים בשם.
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

## מה בדיקת חבילת ה־ONNX כן מוכיחה ומה לא

`validate_model` דוחה קודם נתיב שאינו מסתיים ב־`.onnx`, ואז `validate_onnx_package` קוראת
כל קובץ בחבילה **פעם אחת**, הזול ביותר קודם:

1. `tokenizer.json` קיים לצד הגרף — חסר נדחה במיקרו־שניות, לפני שהגרף מגובב.
2. הבתים הראשונים של הגרף אינם סוג ידוע של קובץ *אחר* (Git LFS pointer, דף שגיאה, ZIP,
   GGUF) — רק הודעה טובה יותר; ה־walk היה דוחה אותם בכל מקרה.
3. הגרף נקרא כ־protobuf ומגובב באותו מעבר: קובץ שנקטע או פגום נדחה, וכך גם מודל בלי
   `ir_version`, בלי `opset_import`, בלי גרף או עם גרף בלי קלט או פלט.
4. כל קובץ external data שהטנזורים מצביעים עליו נמצא בתוך החבילה, קיים, ולפחות באורך
   שההפניות שלו צריכות; כל אחד מגובב.
5. `tokenizer.json` מגובב והוא אובייקט JSON שלם.

מה שזה **לא**: בלי `length` מפורש, החסם על קובץ external data הוא חסם תחתון — ביט אחד
לאיבר, מתחת לכל טיפוס ONNX — ולכן קובץ כזה שנקטע אחרי החסם עדיין עובר. וגם כאן אין אימות
הורדה אמיתי: checksum שהמנוע מחשב מהקבצים אינו יכול להעיד עליהם. השוואה מול hash מפורסם
שייכת להפצת המודל ([`MODEL_DISTRIBUTION.md`](MODEL_DISTRIBUTION.md)); ה־checksum כאן קיים
כדי לזהות שהבתים מאחורי נתיב המודל **השתנו** בין הרצות, מה שמבטל בשקט כל וקטור שנשמר.

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

הקוד מציג את המערכת כ־ONNX embedding runtime, עם ברירות המחדל של זהות הייצור
(`config/models/meivin-round2-onnx/`):

```text
model: ArieLLL123/judaic-semantic-round2-onnx-zayit (Meivin Round 2, גרף ה-int8)
dimension: 256
quantization: int8
pooling: in-graph
max tokens: 256
batch size: 32
```

`validate()` דוחה נתיב שאינו מסתיים ב־`.onnx` כ־`InvalidModelFile`, לפני שנפתח דבר.
`load()` מאמת את חבילת ה־ONNX — הגרף, `tokenizer.json` שלצדו וכל קובץ external data —
ומחשב את ה־checksum שלה, כל קובץ נקרא פעם אחת, ואז הבחירה נעשית ב־`backend::select_backend`:

```text
בנייה רגילה (production)               → Err(BackendUnavailable)
--features mock-embedding               → backend "mock-hash-v1"  (אינו מודל)
--features onnx-backend                 → inference אמיתי דרך ONNX Runtime
שני ה-features יחד                      → ה-backend האמיתי מנצח
```

ה־backend של GGUF ו־llama.cpp הוסר; הקומיט האחרון שיש בו תמיכה בו הוא `62f0c44`.

ה־stand-in מבוסס SHA-256 ו־feature hashing (ואחריו L2 normalization). הוא **אינו
זמין ב־production** — ראו "מה השתנה ב־PR הראשון" למעלה. וקטור באורך אפס נדחה
בשגיאה במקום להיכנס לאינדקס.

### שני המסלולים

```text
--features mock-embedding (פיתוח/בדיקות):
ONNX package validated + checksummed
       ↓
fake deterministic embedding
       ↓
vector

--features onnx-backend (אמיתי):
ONNX package (graph + tokenizer.json)
 ↓
tokenizer.json של החבילה: תחילית תפקיד, [CLS]/[SEP], חיתוך ל-max_tokens
 ↓
ONNX Runtime, טקסט אחד בכל הרצה
 ↓
pooling ונרמול בתוך הגרף (in-graph)
 ↓
L2 normalization (ב-EmbeddingRuntime, לא ב-backend)
 ↓
256-d vector
```

**אסור להתייחס ל־feature hashing כמודל semantic.**

הוא קיים רק כדי לאפשר לפתח ולבדוק את כל שאר ה־pipeline בלי שה־model runtime יהיה
blocker — ובפרט כדי שבדיקות ה־CI ירוצו גם במכונה שאין בה את קובץ המודל.

---

# 5. מודל ה־Embedding

הקונפיגורציה הנוכחית — זהות הייצור, `config/models/meivin-round2-onnx/model.json`:

```text
Model:
ArieLLL123/judaic-semantic-round2-onnx-zayit (Meivin Round 2)

Graph:
models/meivin-round2-onnx/seforim-embed-round2-int8.onnx  (ו-tokenizer.json לצדו)

Quantization:
int8

Embedding dimension:
256

Pooling:
in-graph

Vector precision:
f32

Maximum tokens:
256

Batch size:
32
```

ההגדרות נמצאות ב־`SemanticConfig` וב־`EmbeddingConfig`, שברירות המחדל שלהן הן הזהות הזו.

הממד, הדיוק, `max_tokens`, pooling וה־normalization אינם „הגדרות” אלא **חלק מזהות
האינדקס**: שינוי של אחד מהם פוסל כל וקטור שנשמר. הבחירה הסופית ביניהם היא S1, ורק
אחריה נכון לקפוא על פורמט ארטיפקט גדול.

כיצד קובץ המודל מגיע למכשיר, ומה כבר הוכרע בעניין: [`MODEL_DISTRIBUTION.md`](MODEL_DISTRIBUTION.md).

---

# 6. למה int8

המודל מיועד לרוץ מקומית, ולכן quantization הוא חלק מרכזי מהארכיטקטורה. גרף ה־int8 הוא
ברירת המחדל; הנימוקים והמדידות ב־[`config/models/meivin-round2-onnx/README.md`](../config/models/meivin-round2-onnx/README.md)
וב־[`ONNX_BACKEND.md`](ONNX_BACKEND.md) §0.

חשוב להבדיל בין:

```text
Model quantization = int8
```

לבין:

```text
Vector precision = f32
```

אלה שני דברים שונים.

כרגע הכוונה היא:

```text
int8 model weights
+
f32 output vectors
```

ולא int8 vectors.

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
256

collection:
chunks
```

---

## שני ה־stores שקיימים בקוד

```text
VectorStore (store.rs)          ← מה שה-engine פותח: מסלול הפיתוח
  └── RwLock<HashMap<String, StoredVectorRecord>>
      + RwLock<HashMap<String, Vec<String>>>   (מפתחות לפי ספר)
  is_persistent() == false — ומצהיר על כך, כדי שה-manifest לא ישקר

SegmentSet (segment_set/)       ← מה שהאפליקציה פותחת: סט וקטורים רשמי
  └── segments של .oxv, ממופים: int8, מפתח של 16 בתים ו-hint לכל slot;
      דורות, deltas, דחיסה, שחזור — docs/ARTIFACT_CONTRACT.md
```

```text
מסלול הפיתוח:     SemanticEngine ──▶ dyn VectorStoreBackend  (insert/remove/clear/commit)
                                       └── VectorStore (בזיכרון)

מסלול האפליקציה:  OfficialSemanticIndex ──▶ SegmentSet ──▶ VectorHit (מפתח + רשומות)
                                                              │
                  HybridCoordinator ──▶ CandidateResolver של המארח ──▶ שורות חיות
```

לסט אין פעולת כתיבה במסלול השאילתה: הוא משתנה רק בהתקנה ובדחיסה, תחת lock, דור אחר דור.

### S2b — נענתה

השאלה הייתה אם סריקה מלאה עומדת בתקציב בקנה מידה של הספרייה, או שנדרש ANN. התשובה,
במדידה (Apple M4, [`tests/vector_set_scale.rs`](../tests/vector_set_scale.rs), 6.4M רשומות,
6.0M slots, 7,765 ספרים, 256 ממדים):

| מדד | ערך |
|---|---|
| גודל | 1.686 GB (`i8-sym-vec`, ממופה — לא נטען ל־RAM) |
| פתיחה | 3.8 ms |
| סריקה חמה | 69 ms בחוט אחד, 17 ms בעשרה |
| סריקה של 5% מהספרים | 2.5 ms |
| התקנת base / החלת delta של 6.5% | 5.7 s / 0.45 s |
| דחיסה | 8.8 s, שיא זיכרון ≈ 0.35 GB |

כלומר **אין צורך ב־ANN**: סריקה מלאה מדויקת של int8 עומדת ביעד (≤ 80 ms) עם מרווח, והיא
דטרמיניסטית — אותו ציון ביט אחר ביט על כל מעבד. אם מחשב חלש לא יעמוד ביעד שלו (≤ 250 ms),
הפורמט שומר מקום לשכבת סינון ראשונה (`TIER0_SIGN`, `TIER0_PQ`) כ־sections מסוג ancillary,
בלי לשבור קוראים קיימים. מה שעוד לא נמדד: recall מול f32 על וקטורי הספרייה האמיתיים,
ומחשב חלש.

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
D = 256 (ממד המודל)
```

זה נכון לשני ה־stores, והוכרע כארכיטקטורה (S2b, למעלה): ב־256 ממדים int8 הסט של
הספרייה הוא ~1.66 GB ממופים, והסריקה המלאה שלו — בשלמים, על כמה חוטים — אורכת עשרות
מילישניות.

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

1. **ה־store הוא `Box<dyn VectorStoreBackend>`.** `open()` פותח את זה שבזיכרון,
   ו־`with_store()` מקבל כל backend אחר; ה־manifest רושם את מי שנפתח בפועל, ולכן
   פתיחה מחדש עם backend אחר מדווחת כאי־תאימות. זה מה שמאפשר לארוז ארטיפקט מריצת
   אינדוקס.
2. **`index_book`/`index_books`/`reset_index` הן פעולות אב־טיפוס.** הן משמשות את
   הבדיקות ואת ה־builder העתידי, לא את האפליקציה. במסלול המוצר האפליקציה מתקינה
   ארטיפקט מוכן ופותחת אותו read-only — דרך `OfficialSemanticIndex`, שאין עליו אף אחת
   מהפעולות האלה.

---

# 15. Indexing Flow

> **היכן זה רץ:** במכונת build, או בבדיקות. **לא** באפליקציה. אין באוצריא אינדוקס
> ברקע, אין progress stream ואין cancel/resume — ראו [`PRODUCT_CONTRACT.md`](PRODUCT_CONTRACT.md) §4.
> הזרימה כאן היא מה ש־[`builder.rs`](../src/distribution/builder.rs) מפעיל, ספר אחד בכל
> פעם, דרך אותו `Chunker` עצמו — ומעל אינדקס Tantivy סופי, כשהפורט ימומש מעליו.
> בנייה מפוצלת ([`shard.rs`](../src/distribution/shard.rs)) מחילה את אותו מתכון ב־`export_plan`,
> ומעבירה ל־worker טקסטים מוגמרים בלבד.

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
ONNX package (graph + tokenizer.json + external data)
   ↓
package checksum
   ↓
manifest
```

כדי למנוע מצב שבו:

```text
model ID = same
but
actual package = different
```

זה מחובר: `load_model()` מחשב את ה־checksum של חבילת ה־ONNX (כל קובץ נקרא פעם אחת, יחד
עם אימות החבילה), משווה אותו למה שנשמר ב־manifest, ומשבית את המסלול הסמנטי כשהחבילות
שונות. אם ה־manifest עדיין לא מכיר checksum — הוא נרשם בטעינה הראשונה.

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
the model
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

ה־store שבזיכרון (מסלול הפיתוח): 79–132ms לשאילתה על 200k×1024 f32. הסט הרשמי: 18.9 ns
לווקטור בחוט אחד ב־256 int8 — 69 ms על 6.0M slots, 17 ms בעשרה חוטים.

---

## Persistence במסלול הפעיל

סט מותקן נפתח מחדש אחרי restart ואינו מאונדקס שוב, ופתיחה ממפה את הווקטורים ואינה
קוראת אותם: 3.8 ms על 6.0M slots. במסלול הפיתוח ה־store הוא בזיכרון, ושם הווקטורים אינם
שורדים restart.

---

## Cold-open ותקציב זיכרון

נמדדו (S2b, למעלה): פתיחה 3.8 ms, ו־private memory קטן — הווקטורים ממופים, והזיכרון
שהסריקה נוגעת בו הוא page cache של מערכת ההפעלה. דחיסה, במכשיר ובזמן סרק, מגיעה לשיא
של ≈ 0.35 GB.

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

## ✅ נעשה: Real Embedding Runtime

inference אמיתי דרך ONNX Runtime: ה־tokenizer של החבילה, pooling בתוך הגרף ו־L2, מאומתים
מול golden vectors, מאחורי `--features onnx-backend`. (ב־PR #2 זה היה GGUF דרך llama.cpp,
שהוסר אחרי `62f0c44`.)

## ✅ נעשה: Batch Embeddings

האינדוקס קורא ל־`embed_batch()`, וה־backend האמיתי מבצע batching מרובה־סדרות אמיתי.

## ✅ נעשה: Manifest Compatibility

עשרת הממדים נבדקים, כולל SHA-256 של קובץ המודל.

## ✅ נעשה: S0 — יישור חוזה המוצר

הגדרת האינדקס הרשמי כ־read-only, ביטול המונח `cloud`, יישור README/CODE_MAP/מפת
דרכים והכרעת הפצת המודל.

---

## הבא בתור

העבודה עצמה ושערי הקבלה מוגדרים במסמך השלבים; לא משוכפלים כאן, כדי שלא יהיו שתי
גרסאות שיכולות להיפרד. מה שכן שייך למסמך הזה הוא נקודת ההתחלה בקוד:

| שלב | מאיפה מתחילים בקוד |
|---|---|
| **S1** — ייצוג, ממד ודיוק | [`chunker.rs`](../src/semantic/chunker.rs) (טקסט ההטמעה) ו־[`benchmark/`](../src/benchmark/) (המדידה). התוצר מקפיא שדות בזהות האינדקס |
| **S2a** — מסלול ריצה read-only | ✅ [`official_index.rs`](../src/semantic/official_index.rs); מאז store v2 — מעל [`SegmentSet`](../src/semantic/segment_set/mod.rs) |
| **S2b** — סקייל ומדידה | ✅ נענתה: סריקה מלאה מדויקת של int8, בלי ANN — §„S2b — נענתה” ו־[`ARTIFACT_CONTRACT.md`](ARTIFACT_CONTRACT.md) §8 |
| **S3** — חוזה ארטיפקט | ✅ הזהות והאימות ב־[`versioning.rs`](../src/semantic/versioning.rs) וב־[`package.rs`](../src/distribution/package.rs); מה שנשאר הוא חשיפת [`IndexImporter`](../src/distribution/importer.rs) ב־API |
| **S4a** — packer לווקטורים מוכנים | הוחלף ב־store v2: וקטורים לפי מפתח אל segment, ב־`assemble` של צינור הבנייה (S7) |
| **S4b** — embeddings | ✅ [`builder.rs`](../src/distribution/builder.rs): המתכון מוחל על הקורפוס, קבוצת הכיסוי נקבעת לפני ה־inference, וזהות המודל נבדקת מול הקובץ שנטען |
| **S4b** — Tantivy חי | לממש `CorpusIndex` ו־`CorpusBooks` מעל האינדקס הסופי ב־`otzaria_search_engine`, ולהעביר אותם ל־`build` |

הסדר בפועל: S3 (חוזה) נעשה לפני S1/S2, מפני שהוא קובע אילו שדות מוצהרים ולא אילו ערכים
נבחרים; אחריו S2a, שנתן לחוזה קורא, ואחריו S4a — הכותב — ו־S4b, שמייצר את מה שהכותב
אורז. הבא בתור היא יתרת S4b: הפורט מעל Tantivy חי. S1 לפני S2b — אך בלי לקפוא על ערכים
לפני שהמדידה בידיים.

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

## ❌ לא להניח שמסלול הפיתוח מתמיד

`SemanticEngine::open()` פותח את ה־store שבזיכרון, והווקטורים שלו אינם שורדים restart.
מה שמתמיד הוא הסט הרשמי, ש־`OfficialSemanticIndex` פותח.

---

## ❌ ארבעה איסורים שנובעים מחוזה המוצר

מוגדרים ומונמקים ב־[`PRODUCT_CONTRACT.md`](PRODUCT_CONTRACT.md) §§3–6 ו־§10; כאן רק
בשורה, כדי שלא יהיו שני ניסוחים שיכולים להיפרד:

1. אין overlay לספרי משתמש (§3).
2. אין אינדוקס ברקע באפליקציה, ולכן גם אין progress/cancel (§4).
3. „ענן” אינו תיאור של המסלול — אריזת קבצים אינה שירות (§5).
4. ארטיפקט שזהותו אינה תואמת אינו נפתח. `line_id` נגזר מסדר הקטלוג, ולכן אינדקס
   מגרסה אחרת מצביע לשורות **לא נכונות** ולא רק מפספס (§6).

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

* [x] מודל ONNX נטען באמת (`--features onnx-backend`)
* [x] tokenizer עובד — `tokenizer.json` של החבילה, תחיליות התפקיד כטוקנים מיוחדים
* [x] inference עובד
* [x] pooling תואם למודל — in-graph: הגרף מוציא את וקטור המשפט הגמור
* [x] normalization תקין — במעבר יחיד ב־`EmbeddingRuntime`
* [x] גרף ה־int8 וגרף ה־fp32 נבדקים מול golden vectors (`token_ids` מדויק, ואז סבילות וקטורית)
* [ ] latency של הטמעת שאילתה על מכשירי היעד

### Vector Store

* [x] persistence — סט וקטורים רשמי, segments ממופים
* [x] persistence **במסלול שהאפליקציה פותחת** — ארטיפקט מותקן, `vectors_persisted=true` (S2a)
* [x] מצב official-read-only ללא delete/upsert בזמן ריצה — טיפוס שאין עליו כתיבה (S2a)
* [x] ה־engine תלוי ב־trait ולא ב־store קונקרטי, וה־manifest רושם את ה־backend שנפתח (S2a)
* [x] ANN או הוכחה שאין בו צורך (S2b) — אין צורך: 69 ms בחוט אחד ו־17 ms בעשרה על 6.0M
  slots, int8 מדויק (`tests/vector_set_scale.rs`)
* [x] פתיחה, זיכרון וגודל דיסק בקנה מידה של הספרייה (S2b) — 3.8 ms, ממופה, 1.686 GB
* [ ] recall מול f32 על וקטורי הספרייה, ומחשב חלש
* [x] reopen אחרי restart — עקבי (רשומות ספרים לא שורדות backend נדיף)
* [x] insert/update/delete
* [x] filtering
* [x] dimension validation

### Artifact / Distribution

* [x] manifest חבילה + SHA-256 לכל payload
* [x] התקנה בשני renames עם staging, גיבוי, שחזור מהפרעה ו־`fsync` — **לא** החלפה אטומית אחת
* [x] אימות מחדש של ה־payload **אחרי** ההעתקה, לא רק במקור
* [x] זהות corpus, `tantivy_schema_version`, `document_id_scheme_version` (S3)
* [x] זהות מודל בתוך החבילה — `model_checksum`, backend, quantization (S3)
* [x] דחייה מפורשת לפי שדה, עם כל אי־ההתאמות ולא הראשונה (S3)
* [x] `metadata_version` עם probe לפני פרסור המסמך (S3)
* [x] התאמת **גודל** מול ה־manifest, ודחיית ספירות אפס (S3)
* [x] שני עומקי אימות — עמוק בהתקנה, metadata+נוכחות בפתיחה, והטוקן מדווח באיזה (S3)
* [x] digest מפורסם כעוגן אמון, ו־`without_published_digest` כוויתור מוצהר (S3)
* [x] שמות payload פורטביליים, נבדקים על המחרוזת ולא דרך `Path` (S3)
* [x] שחזור התקנה שנקטעה, `fsync` לקבצים ולתיקיית האב, ובדיקות הזרקת־כשל (S3)
* [x] התאמת ספירות ספרים/וקטורים מול **תוכן** ה־payload, בפתיחה (S2a)
* [x] קורא שמפעיל את האימות: `OfficialSemanticIndex` מקבל את הטוקן ולא נתיב (S2a)
* [x] זיהוי עריכה באותו אורך בזמן פתיחה — SHA-256 לכל רשומה בקורא ה־store (S2a)
* [ ] זיהוי payload שנערך **יחד עם** ה־checksums שלו — רק digest מפורסם מבדיל (S6)
* [ ] צינור שמפרסם digest, וחתימה (S6)
* [ ] lock לשתי התקנות במקביל לאותו יעד — מתועד כמחוץ להיקף (S6, אם יידרש)
* [x] תקציב זמן נמדד לפתיחה ולהתקנה בגודל ייצוגי — פתיחה 3.8 ms, התקנה 5.7 s, delta 0.45 s
* [ ] חשיפת ה־importer דרך ה־API / FFI (S5)
* [ ] `assemble` לפי מפתחות, warehouse ו־delta לצינור הבנייה של הספרייה (S7)
* [x] builder שמייצר את הווקטורים מקורפוס וממודל, ומחיל את המתכון בעצמו (S4b)
* [ ] מימוש `CorpusIndex`/`CorpusBooks` מעל Tantivy הסופי (יתרת S4b)

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
     ├── store v2 (oxv segments + segment sets)
     │        ├── ✅ int8 ממופה, ממוען לפי טקסט, סריקה מדויקת        (S2–S3 של store v2)
     │        ├── ✅ התקנה, delta, דחיסה, שחזור, scrub               (S4 של store v2)
     │        └── ✅ 6.0M slots: פתיחה 3.8 ms, סריקה 17–69 ms        (S2b)
     │
     ├── IndexVersion
     │        └── ✅ text / משפחת מודל וחבילות שאילתה / store       (S5 של store v2)
     │
     └── distribution
              ├── ✅ builder: מתכון → embeddings → base segment       (S4b)
              ├── ✅ export-plan / embed-shard                       (S4b)
              ├── assemble לפי מפתחות, warehouse, delta             (S7)
              └── resolver מעל Tantivy חי, FFI                      (P1–P4)
```

זה דווקא מצב טוב יחסית: החוזים במקום, וכל פער הוא חיבור או הרחבה ולא שכתוב.

לא צריך לזרוק את הארכיטקטורה. צריך לחבר את מה שקיים ולמדוד אותו.

---

# 42. Recommended Development Order

הסדר המומלץ:

```text
✅ 1. Real inference (GGUF in PR #2; ONNX, the only backend, since)
✅ 2. Validate generated embeddings (golden vectors)
✅ 3. Batch inference
✅ 4. S0 — product contract alignment
   ↓
   5. S1 — quality dataset → dimension & precision decision
   ↓
✅ 6. S2b — scale measured at 6.4M records: an exact int8 full scan, no ANN
   ↓
✅ 7. S3 — artifact identity contract & recoverable install (זהות, אימות בשני עומקים,
      עוגן digest)
   ↓
✅ 7a. S2a — read-only runtime path: the artifact's reader, and read/write split
   ↓
✅ 7b. S4a — packer for ready-made vectors (since replaced by store v2)
   ↓
✅ 7c. S4b — the builder: corpus + model → embeddings → a base segment
   ↓
✅ 7d. store v2 — content-addressed int8 segments, sets, hits and a resolver
   ↓
   8. the resolver and the chunkKey column over the final Tantivy index
             (otzaria_search_engine, P1–P4); the library's build (S7)
   ↓
   9. S5 — repin, open/install API, FFI            (otzaria_search_engine)
   ↓
  10. S6–S7 — artifact/model management, BLoC & UI  (otzaria)
   ↓
  11. S8 — release gates on the full platform matrix
```

S1 לפני S2b בכוונה: ממד ודיוק קובעים אם סריקה מלאה בכלל קבילה, ולכן בחירת backend לפני
בחירת ממד היא בחירה בעיניים עצומות. שני השלבים יכולים לרוץ במקביל, אבל אין לקפוא על
פורמט ארטיפקט לפני שהמדידה של S1 בידיים.

**הסדר בפועל שונה מהמספור, במכוון:** S3 (חוזה הזהות) נעשה לפני S1 ו־S2, מפני שאינו
תלוי בהם — הממד, הדיוק ופורמט ה־store הם **נתונים בתוך** ה־manifest ולא קבועים בקוד,
ולכן הכרעות S1/S2 ממלאות שדות קיימים ואינן משנות את החוזה. אחריו נעשה S2a: מסלול ריצה
read-only שפותח את הארטיפקט המאומת, וזה מה שנתן לחוזה צרכן; ואחריו S4a, ה־packer, שהוא
הצד הכותב שלו, ו־S4b, ה־builder שמייצר את מה שנארז. הבא בתור היא יתרת S4b: הפורט מעל
Tantivy חי. S1 ו־S2b חוזרים לפני יצירת הארטיפקט האמיתי והכרעת backend ה־production.

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
אריזה והתקנה עם שחזור, חוזה זהות ארטיפקט מלא עם אימות שדוחה לפי שדה, ו־Rust API seam
ל־FFI. הוא עדיין אינו מוצר: ה־engine פותח את ה־store שבזיכרון ולכן במסלול הפעיל אין
persistence ואף אחד אינו פותח ארטיפקט מאומת; אין ANN ואין הוכחת סקייל על ~6.1 מיליון
שורות; אין builder שמפיק ארטיפקט מ־Tantivy; ובאוצריא ה־BLoC וה־UI אינם מפעילים את
המסלול.

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
│   ├── model_package.rs  → the ONNX package: path check, validation, checksum
│   ├── onnx_backend.rs   → real inference (feature `onnx-backend`), the only backend
│   ├── engine.rs         → semantic orchestration
│   ├── manifest.rs       → index compatibility + state
│   ├── store.rs          → in-memory vector store (the development path)
│   ├── store_backend.rs  → VectorStoreBackend contract
│   ├── chunk_key.rs      → ChunkKey: the address of a vector, SHA-256 of its text
│   ├── oxv/              → the segment format, its codecs and the exact int8 scan
│   ├── segment_set/      → the installed set: install, apply, compact, recover, GC
│   ├── resolve.rs        → VectorHit and the CandidateResolver port
│   ├── official_index.rs → the application's path: a set and a model, read-only
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
│   ├── importer.rs       → staged install of a package directory + recovery
│   ├── corpus.rs         → the port onto the lexical index (CorpusIndex)
│   ├── builder.rs        → corpus + model → a base segment and its release
│   └── shard.rs          → export_plan / embed_shard / verify_shards
│
├── telemetry/            → in-process counters (no network)
└── benchmark/            → timing & percentile helpers
```

---

# 47. The Most Important Next Task

### Reach the app: the resolver over the live index (P1–P4), then the library's build (S7).

Store v2 is in place: vectors are addressed by the text they were embedded from, stored as
int8 segments that are mapped rather than read, installed base and deltas as generations
that a crash leaves old or new, and scanned exactly — the same score, bit for bit, on every
CPU. A scan returns keys and the books and lines they were built at; the host's
`CandidateResolver` ties them to live lines, so an index commit never shows a vector's line
under the wrong id. S2b is answered by measurement: at 6.0M slots the set opens in 3.8 ms
and a warm scan takes 69 ms on one thread and 17 ms on ten — no ANN.

What that leaves, in an order that is not interchangeable:

```text
P1–P4 the chunkKey column, the book directory and LiveResolver, open/install/compact
      through the FFI                                        (otzaria_search_engine)
        ↓
S7    the library's build: plan by key, the f32 warehouse, assemble a base or a delta
        ↓
L1/U1 publish vectors-<tag> releases; plan the updates on the device
        ↓
      recall@10/@50 against exact f32 on the library's vectors; a weak laptop
```

---

---

# 48. Current Project Status

**Architecture:** 🟢 Strong foundation

**Chunking:** 🟢 Implemented

**Manifest:** 🟢 Implemented — real compatibility validation + atomic writes

**Incremental book detection:** 🟢 Implemented

**Index-incompatibility handling:** 🟢 Implemented — disables the semantic path

**Embedding abstraction:** 🟢 Implemented

**Actual embedding inference:** 🟢 Implemented behind `--features onnx-backend`, the only
backend, verified against golden vectors. A default build still has no backend at all, by
design — it fails loudly rather than serving fake vectors.

**Vector abstraction:** 🟢 The development path's store is split in two —
`VectorSearchBackend` for a search, `VectorStoreBackend` adds the mutations indexing
needs. The application's path holds a `SegmentSet`, which nothing on a device writes to.

**Persistent vector database:** 🟢 The application opens an installed vector set: int8
segments, mapped, opened in 3.8 ms at 6.0M slots, reopened after a restart without
indexing. Installs and deltas are generations; a crash leaves the old one or the new.

**ANN retrieval:** 🟢 Not needed (S2b): an exact int8 full scan takes 69 ms on one
thread and 17 ms on ten at 6.0M slots, 3.0 ms for 5% of the books. A first-pass tier is
reserved in the format, as ancillary sections, if a weak laptop needs one.

**Identity:** 🟢 `IndexVersion` carries the line recipe and the key version, the model
family — tokenizer, dimension, pooling, token cap, text recipe, normalization, chunking —
with the query packages it accepts, and the store format. Every field is compared, the
packages by membership, and every mismatch is named (`docs/ARTIFACT_CONTRACT.md`).

**Authenticity:** 🟡 A published manifest digest can be required and is checked — but
nothing publishes it yet and there is no signature. A release installed without one is
verified against damage and the wrong release, not against forgery.

**Install & recovery:** 🟢 Every byte is checked at install, the cheap structures at
open, and every block by a scrub; crash injection after each install step leaves the old
generation or the new, and a damaged `CURRENT` falls back to `PREVIOUS`. What is missing:
none of it is exposed through the FFI yet (P4).

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
✅ Fake embedding              →  Real ONNX inference
✅ Per-chunk inference         →  Batch inference
✅ Model-only index identity   →  Corpus + Tantivy + ID-scheme identity
✅ Engine bound to VectorStore  →  Read/write split; the application opens a
                                   verified artifact through the read side

Verified artifact, unmeasured  →  cold-open, p50/p95/p99, RSS and disk at 6M

Package written by tests       →  Artifact built from the final Tantivy index

Reader inside the crate        →  Reader reached from Otzaria, through the FFI

Heuristic weights              →  Benchmark-driven ranking
```

Two rows deliberately absent, because they are out of scope rather than pending:
chunk-level incremental indexing at user runtime, and a writable overlay for
personal books. See [`PRODUCT_CONTRACT.md`](PRODUCT_CONTRACT.md) §9.

The most important principle remains:

> **The semantic system must enhance Otzaria's existing search without taking ownership of it or becoming a single point of failure.**
