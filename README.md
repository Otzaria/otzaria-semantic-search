<<<<<<< HEAD
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
| GGUF inference אמיתי             | **לא ממומש עדיין**         |
| deterministic embedding fallback | ממומש                      |
| VectorStore abstraction          | קיים                       |
| zvec אמיתי                       | **לא מחובר עדיין**         |
| In-memory vector store           | ממומש                      |
| Cosine search                    | ממומש                      |
| Metadata filtering               | חלקי                       |
| Hybrid coordinator               | קיים                       |
| Fusion                           | קיים                       |
| Dynamic weighting                | קיים                       |
| RRF                              | קיים                       |
| Grouping                         | קיים                       |
| Flutter/FFI API                  | קיים                       |
| Production indexing pipeline     | **עדיין דורש integration** |
| Production persistence           | **עדיין חסרה**             |

הנקודה החשובה ביותר למפתח חדש:

> **זהו כרגע skeleton ארכיטקטוני עובד חלקית, לא מנוע semantic production-complete.**

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

אבל כרגע:

```rust
load()
```

רק בודק שקובץ המודל קיים.

לא מתבצע inference אמיתי.

`embed_one()` מפעיל:

```text
compute_deterministic_text_embedding()
```

המבוסס על SHA-256 ו־feature hashing. לאחר מכן מתבצע L2 normalization.

### לכן:

```text
Current:
GGUF file exists
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
chunking_version
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

הקוד בפועל עושה embedding אחד לכל chunk כרגע, למרות שקיימת API ל־batch.

### שיפור עתידי ברור:

במקום:

```text
chunk
 ↓
embed_one()
 ↓
chunk
 ↓
embed_one()
```

לעבור ל:

```text
32 chunks
 ↓
embed_batch()
 ↓
32 vectors
```

ולנצל את `batch_size`.

---

# 16. Incremental Indexing

אחד הדברים החשובים שכבר בנויים הוא manifest.

לכל ספר נשמר:

```text
source_book_key
content_hash
chunk_count
indexed_at
chunking_version
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
removed_books
```

בנוסף המבנה מוכן לדווח על:

```text
model mismatch
chunking mismatch
normalization mismatch
```

כרגע שלושת האחרונים עדיין מוחזרים כ־`false` מתוך `diff_against_tantivy()`, ולכן זה **לא feature מלא עדיין**.

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
re-index recommended
```

המערכת כרגע לא בהכרח מבצעת rebuild אוטומטי.

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

כרגע זה עדיין לא מחובר באופן מלא ל־index lifecycle.

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

## Priority 3: Batch Embeddings

להשתמש באמת ב:

```rust
embed_batch()
```

ולא לקרוא ל־`embed_one()` לכל chunk.

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

## Priority 5: Manifest Compatibility

להפוך:

```text
model mismatch
chunking mismatch
normalization mismatch
```

לבדיקות אמיתיות.

בנוסף:

```text
model SHA-256
```

צריך להיות חלק מה־compatibility check.

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
* [ ] ANN
* [ ] reopen אחרי restart
* [ ] insert/update/delete
* [ ] filtering
* [ ] dimension validation

### Indexing

* [ ] initial full index
* [ ] incremental book indexing
* [ ] chunk-level reuse
* [ ] removed-book cleanup
* [ ] model mismatch detection
* [ ] chunking mismatch detection

### Hybrid

* [ ] BM25 + semantic
* [ ] score normalization
* [ ] dynamic weighting
* [ ] RRF benchmark
* [ ] grouping
* [ ] provenance

### Reliability

* [ ] semantic failure → BM25 fallback
* [ ] corrupted semantic DB does not corrupt Tantivy
* [ ] manifest writes atomic
* [ ] index rebuild recoverable

### Performance

* [ ] embedding benchmark
* [ ] indexing benchmark
* [ ] vector search benchmark
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

הפרויקט כבר מחזיק את השלד הארכיטקטוני של Semantic Search מלא: semantic sidecar עצמאי, chunking, stable IDs, manifest, incremental book diff, embedding abstraction, vector-store abstraction, semantic retrieval, hybrid fusion, dynamic weighting, RRF, grouping ו־FFI API. אבל שני החלקים הקריטיים ביותר עדיין אינם production implementations: `EmbeddingRuntime` עדיין משתמש ב־deterministic feature hashing במקום inference אמיתי של מודל GGUF, ו־`VectorStore` עדיין משתמש ב־in-memory `HashMap` במקום backend persistent/ANN אמיתי.

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

Current:

```rust
let mut raw_vec =
    compute_deterministic_text_embedding(
        text,
        self.config.embedding_dim
    );
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

**Manifest:** 🟢 Implemented

**Incremental book detection:** 🟢 Implemented

**Embedding abstraction:** 🟢 Ready for backend

**Actual embedding inference:** 🔴 Missing

**Vector abstraction:** 🟢 Ready for backend

**Persistent vector database:** 🔴 Missing

**ANN retrieval:** 🔴 Missing

**Hybrid fusion:** 🟢 Implemented

**Dynamic weighting:** 🟡 Initial heuristic

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
=======
# Otzaria Semantic Search

[![CI](https://github.com/Otzaria/otzaria-semantic-search/actions/workflows/ci.yml/badge.svg)](https://github.com/Otzaria/otzaria-semantic-search/actions/workflows/ci.yml)

**Hybrid semantic & lexical search engine for Otzaria** — a local, offline, production-grade Rust library that merges BM25 lexical results with vector similarity search into a single ranked result set.

## Architecture

```
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

## Module Overview

| Module | Path | Description |
|--------|------|-------------|
| **Types** | `src/semantic/types.rs` | All domain types: `BookLine`, `SemanticChunk`, `FusedCandidate`, `HybridSearchResult`, etc. |
| **Chunker** | `src/semantic/chunker.rs` | Anchored semantic chunking with context windows, SHA256 identity hashing |
| **Embedding** | `src/semantic/embedding.rs` | GGUF model runtime interface (deterministic fallback for testing) |
| **VectorStore** | `src/semantic/store.rs` | Pre-normalized vector storage with BinaryHeap top-k search |
| **Manifest** | `src/semantic/manifest.rs` | Atomic JSON manifest for per-book version tracking & diff detection |
| **Engine** | `src/semantic/engine.rs` | Orchestrates chunking → embedding → storage → manifest lifecycle |
| **Fusion** | `src/hybrid/fusion.rs` | BM25 saturation normalization, cosine normalization, weighted & RRF fusion |
| **Ranking** | `src/hybrid/ranking.rs` | Query analysis, dynamic alpha computation (lexical vs. semantic weight) |
| **Grouping** | `src/hybrid/grouping.rs` | Post-fusion grouping by section or identical text (line hash dedup) |
| **Coordinator** | `src/hybrid/coordinator.rs` | Top-level hybrid search orchestrator with fallback |
| **API** | `src/api/hybrid_search.rs` | Clean Flutter/FFI API surface (`SearchRequest`, `OtzariaHybridEngine`) |
| **Errors** | `src/errors.rs` | Strongly-typed `thiserror` error hierarchy per subsystem |

## Building

```bash
cargo build
```

## Testing

```bash
cargo test --all-targets
```

## Linting

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
```

## CI

GitHub Actions CI runs on every push to `main` and on all PRs, across **3 platforms** (Ubuntu, Windows, macOS):

1. Format check (`cargo fmt --check`)
2. Type check (`cargo check --all-targets`)
3. Lint (`cargo clippy --all-targets -- -D warnings`)
4. Tests (`cargo test --all-targets`)

## Performance Highlights

- **Pre-normalized vectors**: L2 normalization at insert time → cosine similarity reduces to a single dot product at search time
- **BinaryHeap top-k**: O(N log k) search instead of O(N log N) full-sort, with metadata cloned only for final k results
- **Single-pass Unicode truncation**: `char_indices().nth()` instead of double iteration
- **Pre-allocated HashMaps**: `with_capacity()` on fusion maps to avoid rehashing

## License

See [LICENSE](LICENSE) for details.
>>>>>>> 55910d2 (fix(ci): resolve CI failures, optimize vector search perf & complete architecture)
