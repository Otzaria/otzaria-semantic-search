# otzaria-semantic-search

> המסמך הזה אינו README למשתמשים. מטרתו היא לתת למפתח חדש תמונה מדויקת של מצב הפרויקט, הארכיטקטורה, ההחלטות שכבר התקבלו, מה ממומש בפועל, מה עדיין mock/placeholder, ומה צריך לעשות הלאה.
>
> **חשוב:** אין להסיק ממבנה שמות הקוד או מה־comments שמערכת מסוימת כבר ממומשת. במקומות שבהם הארכיטקטורה קיימת אך backend אמיתי עדיין לא מחובר, הדבר מצוין במפורש.

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
| GGUF inference אמיתי             | **לא ממומש עדיין**         |
| deterministic embedding fallback | ממומש, **מחוץ ל־production** (feature `mock-embedding`) |
| Batch embedding                  | ממומש (האינדוקס משתמש בו)  |
| VectorStore abstraction          | קיים                       |
| zvec אמיתי                       | **לא מחובר עדיין**         |
| In-memory vector store           | ממומש                      |
| Cosine search                    | ממומש                      |
| Metadata filtering               | ממומש (facets שטוחים + חלוקת ממדים כמו במנוע הלקסיקלי) |
| זיהוי שינוי ב-PDF                | ממומש כאשר הקורא מספק source revision סמכותי + metadata; גודל+mtime לבדם אינם קנוניים |
| empty-book marker                | ממומש |
| Hybrid coordinator               | קיים                       |
| Fusion                           | קיים                       |
| Dynamic weighting                | קיים                       |
| RRF                              | קיים כ־primitive, **לא בשימוש** |
| Grouping                         | ממומש                      |
| שלושת מצבי החיפוש                | ממומש (כולל SemanticOnly)  |
| זיהוי אי־תאימות אינדקס           | ממומש (משבית את המסלול הסמנטי) |
| התאוששות מ־manifest פגום         | ממומש (quarantine + reset) |
| עמידות ה-manifest                | אטומי בכל פלטפורמה; durable ב-Unix, best-effort ב-Windows |
| כתיבת manifest באינדוקס מלא      | פעם אחת בסוף (לא פר-ספר) |
| Rust API seam ל-Flutter/FFI      | קיים; bindings אמיתיים טרם נוצרו |
| Production indexing pipeline     | **עדיין דורש integration** |
| Production persistence           | **עדיין חסרה**             |
| ANN retrieval                    | **חסר** (brute-force בלבד) |

הנקודה החשובה ביותר למפתח חדש:

> **זהו כרגע skeleton ארכיטקטוני עובד חלקית, לא מנוע semantic production-complete.**

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

מה שמכוון בכוונה **לא** נעשה שם, ומחכה ל־P5: כיול `BM25_SATURATION_K`, threshold
לתוצאה סמנטית לא רלוונטית, מעבר ל־RRF, ו־`total_count` אמיתי (הוא כרגע מספר
המועמדים שנכנסו ל־fusion, ומתועד ככזה).

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

אבל כרגע אין inference אמיתי.

`load()` מאמת את קונטיינר ה־GGUF (magic + version) ומחשב SHA-256 של הקובץ, ואז:

```text
בנייה רגילה (production)      → Err(BackendUnavailable)
--features mock-embedding      → backend "mock-hash-v1"
```

ה־stand-in מבוסס SHA-256 ו־feature hashing (ואחריו L2 normalization). הוא **אינו
זמין ב־production** — ראו "מה השתנה ב־PR הראשון" למעלה. וקטור באורך אפס נדחה
בשגיאה במקום להיכנס לאינדקס.

### לכן:

```text
Current (with the mock feature only):
GGUF container validated + checksummed
       ↓
fake deterministic embedding
       ↓
vector

Target:
GGUF
 ↓
Tokenizer
 ↓
Actual model inference
 ↓
last-token pooling
 ↓
L2 normalization
 ↓
1024-d vector
```

**אסור להתייחס ל־feature hashing כמודל semantic.**

הוא קיים רק כדי לאפשר לפתח ולבדוק את כל שאר ה־pipeline בלי שה־model runtime יהיה blocker.

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

היעד הוא vector database מקומי, נפרד לחלוטין מ־Tantivy.

הקונפיגורציה כרגע:

```text
db path:
semantic_db/zvec

embedding dimension:
1024

collection:
chunks
```

`SemanticConfig` כבר מכוון ל־`semantic_db/zvec`.

---

## אבל בפועל

ה־`VectorStore` הנוכחי **אינו zvec אמיתי**.

בפועל הוא:

```rust
RwLock<HashMap<String, StoredVectorRecord>>
```

ובנוסף:

```rust
RwLock<HashMap<String, Vec<String>>>
```

כלומר:

```text
Vectors
  ↓
RAM HashMap
```

אין כרגע persistence אמיתי של הווקטורים.

`open_or_create()` רק יוצר את התיקייה ומאתחל את ה־HashMaps.

### לכן המשימה היא:

```text
Current:
VectorStore
  ↓
HashMap

Target:
VectorStore
  ↓
zvec
  ↓
persistent local ANN index
```

ה־API של `VectorStore` צריך להישאר abstraction, כדי שהמעבר ל־zvec לא ידרוש לשכתב את ה־semantic engine.

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

זה backend לצורך development/testing.

**זה אינו ה־production retrieval architecture הרצוי.**

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

---

# 15. Indexing Flow

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
status()
index/update lifecycle
```

ה־Rust layer מחזיק את ה־implementation details.

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

## Embedding

כרגע:

```text
one chunk
→ one embed_one()
```

במקום batching מלא.

---

## Vector search

כרגע:

```text
brute-force scan
```

במקום ANN.

---

## Persistence

הווקטורים כרגע חיים ב־RAM.

אין production persistence אמיתי.

---

## Model inference

אין עדיין inference אמיתי.

לכן אין כרגע benchmark אמיתי של:

```text
embedding latency
tokens/sec
memory usage
CPU utilization
```

---

# 36. מה צריך להיות השלב הבא

## Priority 1: Real Embedding Runtime

להחליף:

```text
deterministic feature hashing
```

ב:

```text
real GGUF inference
```

עם:

```text
Tokenizer
Model loading
Q4 inference
last-token pooling
L2 normalization
```

---

## Priority 2: Real Vector Backend

להחליף:

```text
HashMap
```

ב־backend persistent/ANN.

ה־VectorStore API צריך להישאר stable.

---

## Priority 3: Batch Embeddings — ✅ נעשה

האינדוקס קורא ל־`embed_batch()`. מה שנשאר הוא לוודא שה־backend האמיתי (P2) באמת
מנצל את הקבוצה, ולא מפרק אותה בחזרה ל־inference בודד.

---

## Priority 4: Full Incremental Indexing

ליישם:

```text
book-level diff
+
chunk-level diff
```

כדי לא לחשב embeddings מחדש עבור chunks שלא השתנו.

ה־`chunk_hash` וה־semantic IDs שכבר קיימים נותנים בסיס טוב לזה.

---

## Priority 5: Manifest Compatibility — ✅ נעשה

שלושת הדגלים הם בדיקות אמיתיות, ו־`model SHA-256` הוא חלק מה־compatibility check —
יחד עם pooling, quantization, vector precision, embedding backend ו־vector backend.

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

## ❌ לא להניח ש־zvec כבר מחובר

הקוד מוכן לכיוון הזה, אבל ה־implementation הנוכחי הוא in-memory.

---

## ❌ לא לבצע re-index מלא בכל שינוי קטן

המטרה היא incremental indexing.

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

* [ ] GGUF model נטען באמת
* [ ] tokenizer עובד
* [ ] inference עובד
* [ ] pooling תואם למודל
* [ ] normalization תקין
* [ ] Q4 inference נבדק

### Vector Store

* [ ] persistence
* [ ] ANN — נמדד: 84–208ms לשאילתה על 200k×1024 (min/max באותה ריצה), כלומר
  ~2.5–3.2s בקנה מידה של הספרייה **בהערכה ליניארית**, לא במדידה (`cargo bench`)
* [x] reopen אחרי restart — עקבי (רשומות ספרים לא שורדות backend נדיף)
* [x] insert/update/delete
* [x] filtering
* [x] dimension validation

### Indexing

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
* [ ] RRF benchmark
* [x] grouping
* [x] provenance

### Reliability

* [x] semantic failure → BM25 fallback (ומדווח ב־`fallback_reason`)
* [x] corrupted semantic DB does not corrupt Tantivy
* [x] manifest writes atomic (temp → fsync → rename)
* [x] index rebuild recoverable (`reset_index()`)

### Performance

* [ ] embedding benchmark — חסר backend אמיתי
* [ ] indexing benchmark
* [x] vector search benchmark — [`benches/vector_search.rs`](../benches/vector_search.rs)
* [ ] memory benchmark
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

ה־technical debt המרכזי כרגע הוא לא בארכיטקטורה, אלא בפער בין abstraction לבין backend אמיתי:

```text
Architecture
     │
     ├── EmbeddingRuntime abstraction
     │        └── actual inference missing
     │
     └── VectorStore abstraction
              └── production zvec missing
```

זה דווקא מצב טוב יחסית.

לא צריך לזרוק את הארכיטקטורה.

צריך להשלים את ה־implementations מאחוריה.

---

# 42. Recommended Development Order

הסדר המומלץ:

```text
1. Real GGUF inference
        ↓
2. Validate generated embeddings
        ↓
3. Batch inference
        ↓
4. Persistent vector backend
        ↓
5. ANN search
        ↓
6. Incremental indexing
        ↓
7. FFI integration
        ↓
8. Real Otzaria integration
        ↓
9. Benchmarking
        ↓
10. Ranking optimization
```

לא כדאי להשקיע כרגע שעות בכיוון של tuning ל־fusion אם ה־embedding עצמו עדיין feature hashing.

אין הרבה טעם לכוון את ההגה כשעדיין לא התקנו את המנוע.

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

הפרויקט כבר מחזיק שלד ארכיטקטוני רחב של Semantic Search: semantic sidecar עצמאי,
chunking, stable IDs, manifest, incremental book diff, embedding abstraction,
vector-store abstraction, semantic retrieval, hybrid fusion, dynamic weighting,
RRF primitive, grouping ו־Rust API seam ל־FFI. הוא עדיין אינו מנוע production:
בבנייה רגילה אין backend inference בכלל; deterministic feature hashing זמין רק
כ־mock מפורש לבדיקות. ה־`VectorStore` הוא `HashMap` בזיכרון ולא backend
persistent/ANN, וטרם נוצרו bindings או חיבור לאפליקציית אוצריא.

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
├── lib.rs
│   └── crate architecture / module exports
│
├── errors.rs
│   └── error taxonomy
│
├── api/
│   └── Flutter / FFI boundary
│
├── semantic/
│   ├── chunker.rs
│   │   └── text → semantic chunks
│   │
│   ├── embedding.rs
│   │   └── model runtime abstraction
│   │
│   ├── engine.rs
│   │   └── semantic orchestration
│   │
│   ├── manifest.rs
│   │   └── index compatibility + state
│   │
│   ├── store.rs
│   │   └── vector backend abstraction
│   │
│   └── types.rs
│       └── semantic domain types
│
└── hybrid/
    ├── coordinator.rs
    │   └── complete search orchestration
    │
    ├── fusion.rs
    │   └── score fusion / RRF
    │
    ├── ranking.rs
    │   └── query classification + weighting
    │
    └── grouping.rs
        └── result grouping/deduplication
```

---

# 47. The Most Important Next Task

### Replace the fake embedding implementation.

It is already fenced off from production (feature `mock-embedding`), which means a
release build fails loudly instead of serving fake vectors — but it also means
there is **no working semantic path in production at all** until a real backend
lands. That is the next task.

Current (test/dev builds only):

```rust
mock::hash_embedding(text, self.config.embedding_dim)
```

Target:

```text
text
 ↓
tokenizer
 ↓
GGUF model
 ↓
Q4 inference
 ↓
hidden states
 ↓
last-token pooling
 ↓
L2 normalization
 ↓
1024-dimensional embedding
```

Until this exists, **semantic quality cannot actually be evaluated**.

After that, replace the in-memory `HashMap` vector store with the intended persistent ANN backend.

Only then does it make sense to benchmark the complete search system and tune hybrid ranking.

---

# 48. Current Project Status

**Architecture:** 🟢 Strong foundation

**Chunking:** 🟢 Implemented

**Manifest:** 🟢 Implemented — real compatibility validation + atomic writes

**Incremental book detection:** 🟢 Implemented

**Index-incompatibility handling:** 🟢 Implemented — disables the semantic path

**Embedding abstraction:** 🟢 Ready for backend

**Actual embedding inference:** 🔴 Missing (and unavailable in production builds)

**Vector abstraction:** 🟢 Ready for backend

**Persistent vector database:** 🔴 Missing

**ANN retrieval:** 🔴 Missing — brute force measures 84–208ms per query over
200k×1024 on one machine (`cargo bench` prints min/median/max; the spread is the
machine, not the code). Extrapolated linearly, that is ~2.5–3.2s over the
6,058,210-line library — an extrapolation, not a measurement. Load-bearing for P4,
not an optimization.

**Hybrid fusion:** 🟢 Implemented

**Search modes (lexical / hybrid / semantic):** 🟢 Implemented

**Graceful degradation:** 🟢 Implemented and observable

**Dynamic weighting:** 🟡 Initial heuristic, unmeasured

**Grouping:** 🟢 Implemented

**FFI boundary:** 🟡 Skeleton/integration stage

**Production integration with Otzaria:** 🟡 In progress

**Search-quality benchmark:** 🔴 Missing

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

The main work now is to turn the existing abstractions into real production implementations:

```text
Fake embedding
     ↓
Real GGUF inference

HashMap vectors
     ↓
Persistent ANN backend

Per-chunk inference
     ↓
Batch inference

Book-level diff
     ↓
Chunk-level incremental indexing

Heuristic weights
     ↓
Benchmark-driven ranking
```

The most important principle remains:

> **The semantic system must enhance Otzaria's existing search without taking ownership of it or becoming a single point of failure.**
