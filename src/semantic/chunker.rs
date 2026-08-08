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

use crate::errors::ArtifactError;
use crate::semantic::recipe::{ChunkingAlgorithm, EmbeddingTextRecipe, TextNormalizationRecipe};
use crate::semantic::types::{BookForIndexing, SemanticChunk};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

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

    /// One chunk per line, anchored on the line and borrowing context when it is short.
    fn chunk_book_anchored(&self, book: &BookForIndexing) -> Vec<SemanticChunk> {
        let mut chunks = Vec::with_capacity(book.lines.len());
        // Built once per book, not per line: every chunk carries the same set.
        let facets = book.all_facets();
        let chunking_identity = self.config.identity();

        for (i, line) in book.lines.iter().enumerate() {
            // Trimmed, so a line made of spaces or a lone newline is skipped
            // rather than embedded: it has no tokens, and a text with no tokens
            // yields a zero vector, which is a direction-less point that matches
            // nothing and pollutes the index.
            let char_count = line.text.trim().chars().count();

            if char_count < self.config.min_embeddable_chars {
                continue;
            }

            let embedding_text = self.embedding_text_for(book, i, char_count);

            let truncated_text =
                truncate_to_chars(embedding_text.trim(), self.config.max_chunk_chars);
            // Normalized *before* the digest, because the digest has to describe the string
            // the model is given. Hashing the pre-normalization text would put a digest of
            // something nothing was built from into every record — the same fault
            // `chunk_hash` describing the corpus line instead of the embedded text would be.
            let embedded_text = self.normalization.apply(&truncated_text).into_owned();
            if embedded_text.trim().is_empty() {
                continue;
            }
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

    /// What the model is given for one line.
    ///
    /// The `match` is the point: whether a title prefix or a reference prefix helps is
    /// S1's measurement, and the answer becomes a second
    /// [`EmbeddingTextRecipe`] with an arm here. Until one exists, `embedding_text_version`
    /// can only be 1 — which is what stops an artifact declaring a recipe nobody wrote.
    fn embedding_text_for(
        &self,
        book: &BookForIndexing,
        index: usize,
        char_count: usize,
    ) -> String {
        match self.text_recipe {
            EmbeddingTextRecipe::LineOrNeighbourContext => {
                if char_count < self.config.min_meaningful_chars {
                    self.build_context_text(book, index)
                } else {
                    book.lines[index].text.clone()
                }
            }
        }
    }

    fn build_context_text(&self, book: &BookForIndexing, index: usize) -> String {
        let current_line = &book.lines[index];
        let section_id = current_line.section_id;

        let mut start_idx = index;
        for _ in 0..self.config.context_window_lines {
            if start_idx == 0 {
                break;
            }
            if book.lines[start_idx - 1].section_id != section_id {
                break;
            }
            start_idx -= 1;
        }

        let mut end_idx = index;
        for _ in 0..self.config.context_window_lines {
            if end_idx + 1 >= book.lines.len() {
                break;
            }
            if book.lines[end_idx + 1].section_id != section_id {
                break;
            }
            end_idx += 1;
        }

        let mut context_lines = Vec::new();
        for i in start_idx..=end_idx {
            context_lines.push(book.lines[i].text.as_str());
        }

        context_lines.join(" ")
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

pub fn compute_chunk_hash(text: &str) -> String {
    use std::fmt::Write;
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    let result = hasher.finalize();
    let mut hex = String::with_capacity(32);
    for b in &result[..16] {
        let _ = write!(hex, "{b:02x}");
    }
    hex
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
    use crate::semantic::types::BookLine;

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
}
