use sha2::{Digest, Sha256};
use crate::semantic::types::{BookForIndexing, SemanticChunk};

#[derive(Debug, Clone)]
pub struct ChunkerConfig {
    pub min_meaningful_chars: usize,
    pub context_window_lines: usize,
    pub max_chunk_chars: usize,
    pub min_embeddable_chars: usize,
    pub chunking_version: u32,
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
        let mut chunks = Vec::new();

        for (i, line) in book.lines.iter().enumerate() {
            let char_count = line.text.chars().count();

            if char_count < self.config.min_embeddable_chars {
                continue;
            }

            let embedding_text = if char_count < self.config.min_meaningful_chars {
                self.build_context_text(book, i)
            } else {
                line.text.clone()
            };

            let truncated_text = truncate_to_chars(&embedding_text, self.config.max_chunk_chars);
            let chunk_hash = compute_chunk_hash(&truncated_text);
            let semantic_id = compute_semantic_id(&book.source_book_key, line.line_id, self.config.chunking_version);

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
                content_hash: book.content_hash,
                reference: line.reference.clone(),
                segment: line.segment,
                is_pdf: book.is_pdf,
                title: book.title.clone(),
                topics: book.topics.clone(),
                author: book.author.clone(),
                era: book.era.clone(),
                base: book.base.clone(),
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

pub fn compute_semantic_id(source_book_key: &str, line_id: u64, chunking_version: u32) -> String {
    let mut hasher = Sha256::new();
    hasher.update(source_book_key.as_bytes());
    hasher.update(b":");
    hasher.update(line_id.to_string().as_bytes());
    hasher.update(b":");
    hasher.update(chunking_version.to_string().as_bytes());

    let result = hasher.finalize();
    result[..16].iter().map(|b| format!("{b:02x}")).collect()
}

pub fn compute_chunk_hash(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    let result = hasher.finalize();
    result[..16].iter().map(|b| format!("{b:02x}")).collect()
}

pub fn truncate_to_chars(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    s.chars().take(max_chars).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_book(lines: Vec<(u64, &str)>) -> BookForIndexing {
        BookForIndexing {
            source_book_key: "book1.txt".to_string(),
            title: "Test".to_string(),
            content_hash: 100,
            is_pdf: false,
            topics: vec![],
            author: None,
            era: None,
            base: None,
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
        let book = dummy_book(vec![(1, "This is a very long line that exceeds twenty characters.")]);
        let chunks = chunker.chunk_book(&book);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].embedding_text, "This is a very long line that exceeds twenty characters.");
    }
}
