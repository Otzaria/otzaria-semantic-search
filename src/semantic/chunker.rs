//! Chunking: book lines → embeddable semantic chunks.
//!
//! One chunk per line, which is the granularity Tantivy indexes and the app
//! displays. A line too short to carry meaning on its own borrows context from
//! its neighbours within the same section.
//!
//! Whether prefixing the book title and reference helps retrieval, and whether
//! neighbour context helps at all, is an open question measured in roadmap P3 —
//! the current behaviour is the starting point, not a validated choice.

use crate::semantic::types::{BookForIndexing, SemanticChunk};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone)]
pub struct ChunkerConfig {
    /// Below this length a line borrows context from its neighbours.
    pub min_meaningful_chars: usize,
    /// How many lines on each side may be pulled in as context.
    pub context_window_lines: usize,
    pub max_chunk_chars: usize,
    /// Below this length a line is skipped entirely. Measured after trimming, so
    /// a line of blanks is never embedded.
    pub min_embeddable_chars: usize,
    pub chunking_version: u32,
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
            chunking_version: 1,
        }
    }
}

pub struct Chunker {
    config: ChunkerConfig,
}

impl Chunker {
    pub fn new(config: ChunkerConfig) -> Self {
        Self { config }
    }

    pub fn chunk_book(&self, book: &BookForIndexing) -> Vec<SemanticChunk> {
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

            let embedding_text = if char_count < self.config.min_meaningful_chars {
                self.build_context_text(book, i)
            } else {
                line.text.clone()
            };

            let truncated_text =
                truncate_to_chars(embedding_text.trim(), self.config.max_chunk_chars);
            if truncated_text.trim().is_empty() {
                continue;
            }
            let chunk_hash = compute_chunk_hash(&truncated_text);
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
                embedding_text: truncated_text,
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
        let chunker = Chunker::new(ChunkerConfig::default());
        let book = dummy_book(vec![(1, "a"), (1, "ab"), (1, "abcde")]);
        let chunks = chunker.chunk_book(&book);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].line_id, 3);
    }

    #[test]
    fn long_lines_stand_alone() {
        let chunker = Chunker::new(ChunkerConfig::default());
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
        let chunker = Chunker::new(ChunkerConfig::default());
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
        let chunker = Chunker::new(ChunkerConfig::default());
        let book = dummy_book(vec![(1, "     "), (1, "אמת ויציב"), (1, "     ")]);

        let chunks = chunker.chunk_book(&book);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].line_id, 2);
        assert!(chunks[0].embedding_text.contains("אמת ויציב"));
        assert!(!chunks[0].embedding_text.trim().is_empty());
    }

    #[test]
    fn short_lines_borrow_context_from_the_same_section_only() {
        let chunker = Chunker::new(ChunkerConfig::default());
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
        let chunker = Chunker::new(ChunkerConfig {
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
        let chunker = Chunker::new(ChunkerConfig::default());
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
        let chunker = Chunker::new(ChunkerConfig::default());
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
        let chunker = Chunker::new(ChunkerConfig::default());
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
