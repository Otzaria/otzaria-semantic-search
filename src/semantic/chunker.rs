//! Chunking: book lines → embeddable semantic chunks.
//!
//! One chunk per line, which is the granularity Tantivy indexes and the app
//! displays. A line too short to carry meaning on its own borrows context from
//! its neighbours within the same section.
//!
//! Whether prefixing the book title and reference helps retrieval, and whether
//! neighbour context helps at all, is an open question that stage S1 measures on a
//! labelled query set — the current behaviour is the starting point, not a validated
//! choice.
//!
//! The recipe has one implementation and two entry points. [`Chunker::chunk_book`] is the
//! build machine's: whole chunks, with the metadata a record carried. The application's
//! index needs only what a line embeds as and the [`ChunkKey`] that names its vector —
//! [`Chunker::embedded_text`] and [`Chunker::chunk_keys`], over the lines it stores — and
//! gets them from the same code, so a key computed on a device and a key computed when the
//! vectors were built are one function of one text.

use crate::errors::ArtifactError;
use crate::semantic::chunk_key::{ChunkKey, LineRef};
use crate::semantic::recipe::{ChunkingAlgorithm, EmbeddingTextRecipe, TextNormalizationRecipe};
use crate::semantic::types::{BookForIndexing, BookLine, SemanticChunk};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::borrow::Cow;

/// Serializable because a build declares its recipe in a file: `chunking_identity` in the
/// artifact is a hash, and a hash cannot be turned back into these five numbers. Whoever
/// applies the recipe has to be handed the recipe itself, and
/// [`Self::identity`] is what ties the two together — see
/// [`BuildPlan`](crate::distribution::builder::BuildPlan).
///
/// No `serde(default)`: a field left out of the file would silently become 20 or 512 and
/// change what every vector was built from, which is precisely the class of drift the
/// identity check exists to catch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkerConfig {
    /// Below this length a line borrows context from its neighbours.
    pub min_meaningful_chars: usize,
    /// How many lines on each side may be pulled in as context.
    pub context_window_lines: usize,
    pub max_chunk_chars: usize,
    /// Below this length a line is skipped entirely. Measured after trimming, so
    /// a line of blanks is never embedded.
    pub min_embeddable_chars: usize,
    /// Which algorithm turns a book into chunks — see
    /// [`ChunkingAlgorithm`]. Refused by
    /// [`Chunker::new`] when it names one this build does not implement, rather than
    /// travelling into a hash as an opaque number.
    pub chunking_version: u32,
    /// Which text a chunk carries to the model — see
    /// [`EmbeddingTextRecipe`].
    ///
    /// Here as well as in
    /// [`ModelIdentity`](crate::semantic::versioning::ModelIdentity) because the two need
    /// it for different reasons: the chunker to select a code path, and an installation to
    /// compare an artifact against itself without ever holding a configuration.
    /// [`EmbeddingRecipe::resolve`](crate::semantic::recipe::EmbeddingRecipe::resolve)
    /// compares the copies at the one point both are in hand.
    pub embedding_text_version: u32,
    /// What is done to the text before the model sees it — see
    /// [`TextNormalizationRecipe`]. **Text**, not the finished vector: L2 is an
    /// unconditional invariant of cosine and is deliberately not versioned.
    ///
    /// Here because the digest a record carries has to describe the string that was
    /// actually embedded, and that string is normalized before it is hashed. Also in
    /// `ModelIdentity`, for the same reason as `embedding_text_version`.
    pub normalization_version: u32,
}

impl ChunkerConfig {
    /// Identity of this chunking configuration, recorded in the manifest.
    ///
    /// Every field is folded in, not just `chunking_version`: `max_chunk_chars`,
    /// `context_window_lines` and both minimums all change the text that was embedded,
    /// so a change to any of them has to invalidate the index the same way a version
    /// bump does. `chunking_version` remains the manual lever for a change in the
    /// algorithm rather than in these numbers.
    pub fn identity(&self) -> u64 {
        let mut hasher = Sha256::new();
        // `as u64`, so a 32-bit build derives the same identity as a 64-bit one.
        for field in [
            self.min_meaningful_chars as u64,
            self.context_window_lines as u64,
            self.max_chunk_chars as u64,
            self.min_embeddable_chars as u64,
            u64::from(self.chunking_version),
            u64::from(self.embedding_text_version),
            u64::from(self.normalization_version),
        ] {
            hasher.update(field.to_le_bytes());
        }
        let digest = hasher.finalize();
        u64::from_le_bytes(digest[..8].try_into().expect("SHA-256 yields 32 bytes"))
    }
}

impl Default for ChunkerConfig {
    fn default() -> Self {
        Self {
            min_meaningful_chars: 20,
            context_window_lines: 2,
            max_chunk_chars: 512,
            min_embeddable_chars: 5,
            chunking_version: ChunkingAlgorithm::AnchoredLine.version(),
            embedding_text_version: EmbeddingTextRecipe::LineOrNeighbourContext.version(),
            normalization_version: TextNormalizationRecipe::AsSuppliedByCorpus.version(),
        }
    }
}

pub struct Chunker {
    config: ChunkerConfig,
    /// Resolved once, in [`Self::new`]. Held as the enum rather than as the number it came
    /// from, so [`Self::chunk_book`] cannot run under a version nothing implements.
    algorithm: ChunkingAlgorithm,
    text_recipe: EmbeddingTextRecipe,
    normalization: TextNormalizationRecipe,
}

impl Chunker {
    /// Refuses a configuration naming an algorithm or a text recipe this build does not
    /// have.
    ///
    /// Fallible for one reason: `chunking_version` used to be folded into
    /// [`ChunkerConfig::identity`] and read by nothing else, so a build could declare
    /// version 99 and run version 1's code. The identity would be self-consistent and the
    /// artifact would describe a recipe that exists nowhere.
    pub fn new(config: ChunkerConfig) -> Result<Self, ArtifactError> {
        Ok(Self {
            algorithm: ChunkingAlgorithm::from_version(config.chunking_version)?,
            text_recipe: EmbeddingTextRecipe::from_version(config.embedding_text_version)?,
            normalization: TextNormalizationRecipe::from_version(config.normalization_version)?,
            config,
        })
    }

    /// What this chunker resolved to. Reported so a builder can record what it is about to
    /// run rather than what it was asked for.
    pub fn recipe(
        &self,
    ) -> (
        ChunkingAlgorithm,
        EmbeddingTextRecipe,
        TextNormalizationRecipe,
    ) {
        (self.algorithm, self.text_recipe, self.normalization)
    }

    pub fn chunk_book(&self, book: &BookForIndexing) -> Vec<SemanticChunk> {
        // Exhaustive on purpose: a second algorithm has to be written here before its
        // version number can be accepted anywhere.
        match self.algorithm {
            ChunkingAlgorithm::AnchoredLine => self.chunk_book_anchored(book),
        }
    }

    /// The exact string `lines[index]` is embedded as, or `None` when the recipe does not
    /// embed it — the text [`Self::chunk_book`] puts in that line's chunk, prefix included.
    ///
    /// `lines` is the book's lines in order, as the index stores them, or any window of
    /// them that holds the line and its `context_window_lines` neighbours on each side
    /// (fewer only where the book ends): nothing further out ever reaches a line's text. That
    /// is what lets the application re-check one result from five stored lines rather than
    /// from the whole book.
    ///
    /// # Panics
    ///
    /// When `index` is not inside `lines`, as slice indexing does.
    pub fn embedded_text(&self, lines: &[LineRef<'_>], index: usize) -> Option<String> {
        match self.algorithm {
            ChunkingAlgorithm::AnchoredLine => self.anchored_text(lines, index),
        }
    }

    /// Every line's key, in order: `Some` for a line the recipe embeds, `None` for one it
    /// skips. The keys are exactly the `chunk_hash`es [`Self::chunk_book`] would give the
    /// same lines — one recipe, one implementation, used by the build machine and by the
    /// application alike.
    pub fn chunk_keys(&self, lines: &[LineRef<'_>]) -> Vec<Option<ChunkKey>> {
        (0..lines.len())
            .map(|index| {
                self.embedded_text(lines, index)
                    .map(|text| ChunkKey::of(&text))
            })
            .collect()
    }

    /// One chunk per line, anchored on the line and borrowing context when it is short.
    fn chunk_book_anchored(&self, book: &BookForIndexing) -> Vec<SemanticChunk> {
        let mut chunks = Vec::with_capacity(book.lines.len());
        // Built once per book, not per line: every chunk carries the same set.
        let facets = book.all_facets();
        let chunking_identity = self.config.identity();

        for (i, line) in book.lines.iter().enumerate() {
            let Some(embedded_text) = self.anchored_text(book.lines.as_slice(), i) else {
                continue;
            };
            let chunk_hash = compute_chunk_hash(&embedded_text);
            let semantic_id =
                compute_semantic_id(&book.source_book_key, line.line_id, chunking_identity);

            chunks.push(SemanticChunk {
                semantic_id: semantic_id.clone(),
                source_book_key: book.source_book_key.clone(),
                source_doc_key: format!("{}:{}", book.source_book_key, line.line_id),
                line_id: line.line_id,
                section_id: line.section_id,
                line_hash: line.line_hash,
                anchor_text: line.text.clone(),
                embedding_text: embedded_text,
                chunk_hash,
                content_hash: book.content_fingerprint,
                reference: line.reference.clone(),
                segment: line.segment,
                is_pdf: book.is_pdf,
                title: book.title.clone(),
                facets: facets.clone(),
            });
        }

        chunks
    }

    /// What version 1 of the algorithm embeds for one line, over either shape of lines.
    ///
    /// The one implementation: [`Self::chunk_book`] and [`Self::embedded_text`] both come
    /// here, so the vectors a build produces and the keys an index computes cannot be two
    /// readings of the recipe.
    fn anchored_text<L: Lines + ?Sized>(&self, lines: &L, index: usize) -> Option<String> {
        let text = lines.text(index);
        // Trimmed, so a line made of spaces or a lone newline is skipped
        // rather than embedded: it has no tokens, and a text with no tokens
        // yields a zero vector, which is a direction-less point that matches
        // nothing and pollutes the index.
        let char_count = text.trim().chars().count();

        if char_count < self.config.min_embeddable_chars {
            return None;
        }

        let embedding_text = self.embedding_text_for(lines, index, char_count);

        let truncated_text = truncate_to_chars(embedding_text.trim(), self.config.max_chunk_chars);
        // Normalized *before* the digest, because the digest has to describe the string
        // the model is given. Hashing the pre-normalization text would put a digest of
        // something nothing was built from into every record — the same fault
        // `chunk_hash` describing the corpus line instead of the embedded text would be.
        let normalized = self.normalization.apply(&truncated_text);
        // Judged before any role prefix: a prefix is not content, and a line with none
        // must not embed as the prefix alone.
        if normalized.trim().is_empty() {
            return None;
        }
        // The recipe's last word, after the cap and the normalization, so the content
        // is what version 1 embeds — under version 2 without the space a cap can leave
        // at its end — and is hashed as given, prefix included, because the digest
        // describes the string the model is given. See
        // `EmbeddingTextRecipe::passage_text`.
        Some(self.text_recipe.passage_text(&normalized).into_owned())
    }

    /// The content the model is given for one line, before the cap, the normalization and
    /// the recipe's role prefix.
    ///
    /// The `match` is the point: whether a title prefix or a reference prefix helps is
    /// S1's measurement, and the answer becomes another [`EmbeddingTextRecipe`] with an
    /// arm here. Only a variant with an arm can be accepted as `embedding_text_version` —
    /// which is what stops an artifact declaring a recipe nobody wrote. Version 2 shares
    /// version 1's arm on purpose: its passage is version 1's text, trimmed and prefixed
    /// afterwards.
    fn embedding_text_for<'a, L: Lines + ?Sized>(
        &self,
        lines: &'a L,
        index: usize,
        char_count: usize,
    ) -> Cow<'a, str> {
        match self.text_recipe {
            EmbeddingTextRecipe::LineOrNeighbourContext
            | EmbeddingTextRecipe::RolePrefixedLineOrNeighbourContext => {
                if char_count < self.config.min_meaningful_chars {
                    Cow::Owned(self.build_context_text(lines, index))
                } else {
                    Cow::Borrowed(lines.text(index))
                }
            }
        }
    }

    fn build_context_text<L: Lines + ?Sized>(&self, lines: &L, index: usize) -> String {
        let section_id = lines.section(index);

        let mut start_idx = index;
        for _ in 0..self.config.context_window_lines {
            if start_idx == 0 {
                break;
            }
            if lines.section(start_idx - 1) != section_id {
                break;
            }
            start_idx -= 1;
        }

        let mut end_idx = index;
        for _ in 0..self.config.context_window_lines {
            if end_idx + 1 >= lines.len() {
                break;
            }
            if lines.section(end_idx + 1) != section_id {
                break;
            }
            end_idx += 1;
        }

        let mut context_lines = Vec::new();
        for i in start_idx..=end_idx {
            context_lines.push(lines.text(i));
        }

        context_lines.join(" ")
    }
}

/// The two shapes a book's lines arrive in: a builder's [`BookLine`]s, and the
/// [`LineRef`]s an index hands over from what it stores. The recipe reads a line's text,
/// its section, and how many lines there are — nothing else, which is why one
/// implementation serves both.
trait Lines {
    fn len(&self) -> usize;
    fn text(&self, index: usize) -> &str;
    fn section(&self, index: usize) -> u64;
}

impl Lines for [BookLine] {
    fn len(&self) -> usize {
        <[BookLine]>::len(self)
    }
    fn text(&self, index: usize) -> &str {
        &self[index].text
    }
    fn section(&self, index: usize) -> u64 {
        self[index].section_id
    }
}

impl Lines for [LineRef<'_>] {
    fn len(&self) -> usize {
        <[LineRef<'_>]>::len(self)
    }
    fn text(&self, index: usize) -> &str {
        self[index].text
    }
    fn section(&self, index: usize) -> u64 {
        self[index].section
    }
}

pub fn compute_semantic_id(source_book_key: &str, line_id: u64, chunking_identity: u64) -> String {
    use std::fmt::Write;
    let mut hasher = Sha256::new();
    hasher.update(source_book_key.as_bytes());
    hasher.update(b":");
    hasher.update(line_id.to_string().as_bytes());
    hasher.update(b":");
    hasher.update(chunking_identity.to_string().as_bytes());

    let result = hasher.finalize();
    let mut hex = String::with_capacity(32);
    for b in &result[..16] {
        let _ = write!(hex, "{b:02x}");
    }
    hex
}

/// 32 lowercase hex digits: [`ChunkKey::of`] the embedded text, as the records have always
/// carried it.
pub fn compute_chunk_hash(text: &str) -> String {
    ChunkKey::of(text).to_hex()
}

pub fn truncate_to_chars(s: &str, max_chars: usize) -> String {
    // Single-pass: find byte index of the max_chars-th character
    match s.char_indices().nth(max_chars) {
        Some((byte_idx, _)) => s[..byte_idx].to_string(),
        None => s.to_string(), // string is shorter than max_chars
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every test here drives a configuration this build implements, so the resolution
    /// [`Chunker::new`] performs is not what any of them is about.
    fn chunker(config: ChunkerConfig) -> Chunker {
        Chunker::new(config).expect("the default recipe is implemented")
    }

    fn dummy_book(lines: Vec<(u64, &str)>) -> BookForIndexing {
        BookForIndexing {
            source_book_key: "book1.txt".to_string(),
            title: "Test".to_string(),
            content_fingerprint: 100,
            is_pdf: false,
            topics: String::new(),
            extra_facets: vec![],
            lines: lines
                .into_iter()
                .enumerate()
                .map(|(i, (sec, txt))| BookLine {
                    line_id: i as u64 + 1,
                    section_id: sec,
                    text: txt.to_string(),
                    line_hash: 100,
                    reference: format!("Ref {i}"),
                    segment: i as u64,
                })
                .collect(),
        }
    }

    #[test]
    fn skips_very_short_lines() {
        let chunker = chunker(ChunkerConfig::default());
        let book = dummy_book(vec![(1, "a"), (1, "ab"), (1, "abcde")]);
        let chunks = chunker.chunk_book(&book);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].line_id, 3);
    }

    #[test]
    fn long_lines_stand_alone() {
        let chunker = chunker(ChunkerConfig::default());
        let book = dummy_book(vec![(
            1,
            "This is a very long line that exceeds twenty characters.",
        )]);
        let chunks = chunker.chunk_book(&book);
        assert_eq!(chunks.len(), 1);
        assert_eq!(
            chunks[0].embedding_text,
            "This is a very long line that exceeds twenty characters."
        );
    }

    /// A line of blanks passes a raw character count but has no tokens, so it
    /// would embed to a zero vector — a point with no direction that matches
    /// nothing. It must never reach the model.
    #[test]
    fn skips_lines_that_are_only_whitespace() {
        let chunker = chunker(ChunkerConfig::default());
        let book = dummy_book(vec![
            (1, "          "),
            (1, "\t\t\n  "),
            (1, "\u{00a0}\u{00a0}\u{00a0}\u{00a0}\u{00a0}\u{00a0}"),
            (1, "שורה אמיתית עם תוכן"),
        ]);

        let chunks = chunker.chunk_book(&book);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].line_id, 4);
        assert!(!chunks[0].embedding_text.trim().is_empty());
    }

    /// A short line surrounded only by blank lines must not produce an empty
    /// chunk through the context path either.
    #[test]
    fn a_short_line_whose_only_context_is_blank_still_embeds_its_own_text() {
        let chunker = chunker(ChunkerConfig::default());
        let book = dummy_book(vec![(1, "     "), (1, "אמת ויציב"), (1, "     ")]);

        let chunks = chunker.chunk_book(&book);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].line_id, 2);
        assert!(chunks[0].embedding_text.contains("אמת ויציב"));
        assert!(!chunks[0].embedding_text.trim().is_empty());
    }

    #[test]
    fn short_lines_borrow_context_from_the_same_section_only() {
        let chunker = chunker(ChunkerConfig::default());
        let book = dummy_book(vec![
            (1, "סוף הסעיף הקודם עם מספיק תווים"),
            (2, "פתיחת הסעיף החדש עם מספיק תווים"),
            (2, "אמת ויציב"),
            (2, "המשך הסעיף החדש עם מספיק תווים"),
        ]);

        let chunks = chunker.chunk_book(&book);
        let short = chunks
            .iter()
            .find(|c| c.line_id == 3)
            .expect("the short line is still chunked");

        assert!(short.embedding_text.contains("אמת ויציב"));
        assert!(short.embedding_text.contains("פתיחת הסעיף החדש"));
        assert!(short.embedding_text.contains("המשך הסעיף החדש"));
        assert!(
            !short.embedding_text.contains("סוף הסעיף הקודם"),
            "context must not cross a section boundary"
        );
        assert_eq!(
            short.anchor_text, "אמת ויציב",
            "the anchor stays the line itself, only the embedded text grows"
        );
    }

    #[test]
    fn embedding_text_is_truncated_on_character_boundaries() {
        let chunker = chunker(ChunkerConfig {
            max_chunk_chars: 10,
            ..Default::default()
        });
        // Hebrew is multi-byte: truncating by bytes would split a character.
        let book = dummy_book(vec![(1, "אבגדהוזחטיכלמנסעפצקרשת")]);

        let chunks = chunker.chunk_book(&book);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].embedding_text.chars().count(), 10);
        assert_eq!(chunks[0].embedding_text, "אבגדהוזחטי");
    }

    #[test]
    fn semantic_ids_are_stable_and_unique_per_line_and_version() {
        let a = compute_semantic_id("book.txt", 1, 1);
        assert_eq!(a, compute_semantic_id("book.txt", 1, 1), "must be stable");
        assert_eq!(a.len(), 32);

        assert_ne!(a, compute_semantic_id("book.txt", 2, 1), "line must matter");
        assert_ne!(
            a,
            compute_semantic_id("other.txt", 1, 1),
            "book must matter"
        );
        assert_ne!(
            a,
            compute_semantic_id("book.txt", 1, 2),
            "chunking version must matter"
        );
    }

    /// The id must not be forgeable by shifting the separator: `("a:1", 1)` and
    /// `("a", 11)` would collide under naive concatenation.
    #[test]
    fn semantic_id_components_cannot_bleed_into_each_other() {
        assert_ne!(
            compute_semantic_id("book.txt", 11, 1),
            compute_semantic_id("book.txt", 1, 11)
        );
    }

    #[test]
    fn chunk_hash_tracks_the_embedded_text_not_the_source_line() {
        let chunker = chunker(ChunkerConfig::default());
        let book = dummy_book(vec![(1, "שורה ראשונה עם מספיק תווים כדי לעמוד לבד")]);
        let chunks = chunker.chunk_book(&book);

        assert_eq!(
            chunks[0].chunk_hash,
            compute_chunk_hash(&chunks[0].embedding_text)
        );
        assert_ne!(
            compute_chunk_hash("א"),
            compute_chunk_hash("ב"),
            "different text must hash differently"
        );
    }

    #[test]
    fn every_chunk_carries_its_books_metadata() {
        let chunker = chunker(ChunkerConfig::default());
        let mut book = dummy_book(vec![(7, "שורה ארוכה דיה כדי לעמוד בפני עצמה")]);
        book.title = "ספר הבדיקה".to_string();
        book.topics = "/מקרא/תורה".to_string();
        book.extra_facets = vec![
            "/author/מחבר ראשון".to_string(),
            "/author/מחבר שני".to_string(),
        ];
        book.is_pdf = true;

        let chunks = chunker.chunk_book(&book);
        assert_eq!(chunks.len(), 1);
        let chunk = &chunks[0];
        assert_eq!(chunk.title, "ספר הבדיקה");
        // Sorted: the order of facets carries no meaning, and canonicalizing it
        // is what keeps two descriptions of the same book fingerprinting alike.
        assert_eq!(
            chunk.facets,
            vec![
                "/author/מחבר ראשון".to_string(),
                "/author/מחבר שני".to_string(),
                "/מקרא/תורה".to_string(),
            ],
            "every facet of the book must reach the chunk, including both authors"
        );
        assert!(chunk.is_pdf);
        assert_eq!(chunk.section_id, 7);
        assert_eq!(chunk.content_hash, book.content_fingerprint);
        assert_eq!(chunk.source_book_key, book.source_book_key);
        assert_eq!(chunk.source_doc_key, "book1.txt:1");
    }

    /// A book that exercises every branch of version 1: a line standing alone, a short
    /// line borrowing its neighbours, a line too short to embed, and a line cut by the
    /// character cap.
    fn every_branch_book() -> BookForIndexing {
        dummy_book(vec![
            (1, "a line that stands alone"),
            (1, "short one"),
            (1, "no"),
            (2, "abcdefghijklmnopqrstuvwxyz0123456789"),
        ])
    }

    /// Version 1 is byte-identical to the recipe as it was before a second version
    /// existed. The expected texts are written out, and their digests were computed
    /// outside this crate (Python's hashlib), so neither can drift with the code.
    #[test]
    fn version_one_embeds_exactly_what_it_always_has() {
        let chunker = chunker(ChunkerConfig {
            max_chunk_chars: 30,
            ..ChunkerConfig::default()
        });
        let chunks = chunker.chunk_book(&every_branch_book());
        let produced: Vec<(u64, &str, &str)> = chunks
            .iter()
            .map(|c| (c.line_id, c.embedding_text.as_str(), c.chunk_hash.as_str()))
            .collect();
        assert_eq!(
            produced,
            vec![
                (
                    1,
                    "a line that stands alone",
                    "d5e703cd670772730f9f85bd4e17353c"
                ),
                // Borrowed from both neighbours in its section, then capped at 30.
                (
                    2,
                    "a line that stands alone short",
                    "2b7b0d10688b8fdebefd9b9d5b132a17"
                ),
                (
                    4,
                    "abcdefghijklmnopqrstuvwxyz0123",
                    "0bf245c7abbd87326a228aa4178257fb"
                ),
            ]
        );
    }

    /// Version 2 is version 1's text, trimmed, with `[PASSAGE] ` in front, once, on every
    /// chunk — the same lines, the cap spent on content, the digest over what the model is
    /// given.
    #[test]
    fn version_two_prefixes_every_passage_once_and_changes_nothing_else() {
        let v1 = chunker(ChunkerConfig {
            max_chunk_chars: 30,
            ..ChunkerConfig::default()
        });
        let v2 = chunker(ChunkerConfig {
            max_chunk_chars: 30,
            embedding_text_version: EmbeddingTextRecipe::RolePrefixedLineOrNeighbourContext
                .version(),
            ..ChunkerConfig::default()
        });
        let book = every_branch_book();
        let (before, after) = (v1.chunk_book(&book), v2.chunk_book(&book));

        assert_eq!(
            before.len(),
            after.len(),
            "the prefix decides no line's fate"
        );
        for (old, new) in before.iter().zip(&after) {
            assert_eq!(new.line_id, old.line_id);
            assert_eq!(
                new.embedding_text,
                format!("[PASSAGE] {}", old.embedding_text.trim()),
                "line {}",
                old.line_id
            );
            assert_eq!(new.embedding_text.matches("[PASSAGE]").count(), 1);
            assert_eq!(new.chunk_hash, compute_chunk_hash(&new.embedding_text));
            assert_ne!(new.chunk_hash, old.chunk_hash);
            assert_eq!(
                new.anchor_text, old.anchor_text,
                "the anchor is the line itself"
            );
            // The identity folds the version in, so the two never share an id.
            assert_ne!(new.semantic_id, old.semantic_id);
        }
    }

    /// The character cap can end a line on a space. Version 1 embeds that space, as it
    /// always has; version 2 does not — the space would reach a Metaspace tokenizer as a
    /// lone `▁` — and its digest describes the text without it, since that is what the
    /// model is given.
    #[test]
    fn version_two_drops_the_space_a_truncation_leaves_and_version_one_keeps_it() {
        let config = |embedding_text_version| ChunkerConfig {
            max_chunk_chars: 11,
            embedding_text_version,
            ..ChunkerConfig::default()
        };
        // Long enough to stand alone, and cut by the cap right after a space.
        let book = dummy_book(vec![(1, "abcdefghij klmnopqrstuvwxyz0123")]);

        let v1 = chunker(config(1)).chunk_book(&book);
        assert_eq!(
            v1[0].embedding_text, "abcdefghij ",
            "version 1 is unchanged"
        );

        let v2 = chunker(config(2)).chunk_book(&book);
        assert_eq!(v2[0].embedding_text, "[PASSAGE] abcdefghij");
        assert_eq!(v2[0].chunk_hash, compute_chunk_hash("[PASSAGE] abcdefghij"));
    }

    /// A line whose text already reads like a prefix is content: it is embedded with the
    /// role prefix in front of it, not instead of it.
    #[test]
    fn a_line_that_looks_like_a_prefix_is_still_content() {
        let v2 = chunker(ChunkerConfig {
            embedding_text_version: 2,
            ..ChunkerConfig::default()
        });
        let chunks = v2.chunk_book(&dummy_book(vec![(1, "[PASSAGE] a line about passages")]));
        assert_eq!(
            chunks[0].embedding_text,
            "[PASSAGE] [PASSAGE] a line about passages"
        );
    }

    #[test]
    fn an_empty_book_yields_no_chunks() {
        let chunker = chunker(ChunkerConfig::default());
        assert!(chunker.chunk_book(&dummy_book(vec![])).is_empty());
    }

    #[test]
    fn truncate_to_chars_handles_boundaries() {
        assert_eq!(truncate_to_chars("", 5), "");
        assert_eq!(truncate_to_chars("abc", 5), "abc");
        assert_eq!(truncate_to_chars("abcde", 5), "abcde");
        assert_eq!(truncate_to_chars("abcdef", 5), "abcde");
        assert_eq!(truncate_to_chars("שלום", 0), "");
        assert_eq!(truncate_to_chars("שלום עולם", 4), "שלום");
    }

    // ── one recipe, two entry points ──

    /// The recipe as it was written before [`Chunker::embedded_text`] existed, kept
    /// verbatim as the oracle the shared core is held to. The point of the property tests
    /// below is that the core *is not* this code: a refactor that changed one character of
    /// what a line embeds as fails here, on random books, before any key ships.
    fn reference_chunk_texts(
        config: &ChunkerConfig,
        book: &BookForIndexing,
    ) -> Vec<(usize, String)> {
        let text_recipe = EmbeddingTextRecipe::from_version(config.embedding_text_version).unwrap();
        let normalization =
            TextNormalizationRecipe::from_version(config.normalization_version).unwrap();
        let mut out = Vec::new();
        for (i, line) in book.lines.iter().enumerate() {
            let char_count = line.text.trim().chars().count();
            if char_count < config.min_embeddable_chars {
                continue;
            }
            let embedding_text = if char_count < config.min_meaningful_chars {
                let section_id = line.section_id;
                let mut start_idx = i;
                for _ in 0..config.context_window_lines {
                    if start_idx == 0 || book.lines[start_idx - 1].section_id != section_id {
                        break;
                    }
                    start_idx -= 1;
                }
                let mut end_idx = i;
                for _ in 0..config.context_window_lines {
                    if end_idx + 1 >= book.lines.len()
                        || book.lines[end_idx + 1].section_id != section_id
                    {
                        break;
                    }
                    end_idx += 1;
                }
                (start_idx..=end_idx)
                    .map(|j| book.lines[j].text.as_str())
                    .collect::<Vec<_>>()
                    .join(" ")
            } else {
                line.text.clone()
            };
            let truncated = truncate_to_chars(embedding_text.trim(), config.max_chunk_chars);
            let normalized = normalization.apply(&truncated);
            if normalized.trim().is_empty() {
                continue;
            }
            out.push((i, text_recipe.passage_text(&normalized).into_owned()));
        }
        out
    }

    /// splitmix64: a few lines of deterministic randomness, so a failing book can be
    /// reproduced from its seed.
    struct Random(u64);
    impl Random {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn below(&mut self, bound: u64) -> u64 {
            self.next() % bound.max(1)
        }
    }

    /// A line of the kinds a book holds: blank, whitespace only, a few letters, a sentence,
    /// and now and then one long enough for the 512-character cap.
    fn random_line(random: &mut Random) -> String {
        const PIECES: [&str; 12] = [
            "א",
            "בראשית",
            "ברא",
            " ",
            "  ",
            "\t",
            "\u{00a0}",
            "abc",
            "שלום עולם",
            ".",
            "׃",
            "x",
        ];
        let length = match random.below(10) {
            0 => 0,
            1 => 1,
            2..=5 => random.below(6) as usize,
            6..=8 => 4 + random.below(16) as usize,
            _ => 120 + random.below(200) as usize,
        };
        (0..length)
            .map(|_| PIECES[random.below(PIECES.len() as u64) as usize])
            .collect()
    }

    fn random_book(random: &mut Random) -> BookForIndexing {
        let lines = random.below(40) as usize;
        let mut section = 1u64;
        let mut book = dummy_book(Vec::new());
        for i in 0..lines {
            // Section edges as the index makes them: a run of lines, then a new heading.
            if random.below(5) == 0 {
                section += 1 + random.below(3);
            }
            book.lines.push(BookLine {
                line_id: i as u64 + 1,
                section_id: section,
                text: random_line(random),
                line_hash: 0,
                reference: String::new(),
                segment: i as u64,
            });
        }
        book
    }

    fn random_config(random: &mut Random) -> ChunkerConfig {
        if random.below(3) == 0 {
            return ChunkerConfig {
                embedding_text_version: 1 + random.below(2) as u32,
                ..ChunkerConfig::default()
            };
        }
        ChunkerConfig {
            min_meaningful_chars: random.below(30) as usize,
            context_window_lines: random.below(4) as usize,
            max_chunk_chars: 1 + random.below(600) as usize,
            min_embeddable_chars: random.below(8) as usize,
            embedding_text_version: 1 + random.below(2) as u32,
            ..ChunkerConfig::default()
        }
    }

    fn line_refs(book: &BookForIndexing) -> Vec<LineRef<'_>> {
        book.lines
            .iter()
            .map(|line| LineRef {
                text: &line.text,
                section: line.section_id,
            })
            .collect()
    }

    /// The property S1 promises the application: over random books — blank lines, section
    /// edges, the cap, short lines borrowing context — `chunk_keys` and `embedded_text`
    /// reproduce `chunk_book` exactly, and `chunk_book` reproduces the recipe as it was
    /// written before either existed.
    #[test]
    fn keys_and_texts_reproduce_chunk_book_on_random_books() {
        let mut random = Random(0x0715_2026);
        let mut embedded = 0usize;
        let mut skipped = 0usize;
        for case in 0..3000 {
            let config = random_config(&mut random);
            let book = random_book(&mut random);
            let chunker = chunker(config.clone());
            let refs = line_refs(&book);

            let expected = reference_chunk_texts(&config, &book);
            let chunks = chunker.chunk_book(&book);
            assert_eq!(chunks.len(), expected.len(), "case {case}: {config:?}");
            for (chunk, (index, text)) in chunks.iter().zip(&expected) {
                assert_eq!(chunk.line_id, book.lines[*index].line_id, "case {case}");
                assert_eq!(&chunk.embedding_text, text, "case {case}, line {index}");
                assert_eq!(chunk.chunk_hash, compute_chunk_hash(text), "case {case}");
            }

            let keys = chunker.chunk_keys(&refs);
            assert_eq!(keys.len(), book.lines.len());
            let mut expected = expected.iter().peekable();
            for (index, key) in keys.iter().enumerate() {
                let text = chunker.embedded_text(&refs, index);
                match expected.next_if(|(at, _)| *at == index) {
                    Some((_, want)) => {
                        embedded += 1;
                        assert_eq!(text.as_deref(), Some(want.as_str()), "case {case}");
                        assert_eq!(*key, Some(ChunkKey::of(want)), "case {case}");
                        assert_eq!(key.unwrap().to_hex(), compute_chunk_hash(want));
                    }
                    None => {
                        skipped += 1;
                        assert_eq!(text, None, "case {case}, line {index}");
                        assert_eq!(*key, None, "case {case}, line {index}");
                    }
                }
            }
        }
        // Both branches were exercised in earnest, or the property proved little.
        assert!(
            embedded > 10_000 && skipped > 10_000,
            "{embedded} / {skipped}"
        );
    }

    /// What the application relies on to verify a result: the line and its
    /// `context_window_lines` neighbours on each side are all a line's text depends on.
    #[test]
    fn a_window_of_neighbours_gives_the_same_text_as_the_whole_book() {
        let mut random = Random(0x5EED);
        for case in 0..2000 {
            let config = random_config(&mut random);
            let book = random_book(&mut random);
            let chunker = chunker(config.clone());
            let refs = line_refs(&book);
            let reach = config.context_window_lines;
            for index in 0..refs.len() {
                let low = index.saturating_sub(reach);
                let high = (index + reach).min(refs.len() - 1);
                assert_eq!(
                    chunker.embedded_text(&refs[low..=high], index - low),
                    chunker.embedded_text(&refs, index),
                    "case {case}, line {index}"
                );
            }
        }
    }

    /// The production recipe on a fixed book, against digests Python's hashlib computed: a
    /// line alone, a short line with context, a line cut at the cap, blank lines skipped.
    #[test]
    fn the_production_recipe_keys_a_fixed_book_as_python_does() {
        let chunker = chunker(ChunkerConfig {
            embedding_text_version: 2,
            max_chunk_chars: 30,
            ..ChunkerConfig::default()
        });
        let book = every_branch_book();
        let refs = line_refs(&book);
        let keys: Vec<Option<String>> = chunker
            .chunk_keys(&refs)
            .into_iter()
            .map(|key| key.map(ChunkKey::to_hex))
            .collect();
        assert_eq!(
            keys,
            vec![
                Some("0429557a367f9f5e6681647f940ff66e".to_string()),
                Some(compute_chunk_hash(
                    "[PASSAGE] a line that stands alone short"
                )),
                None,
                Some(compute_chunk_hash(
                    "[PASSAGE] abcdefghijklmnopqrstuvwxyz0123"
                )),
            ]
        );
    }
}
