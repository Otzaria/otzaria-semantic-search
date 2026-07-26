use serde::{Deserialize, Serialize};
use std::fmt;

/// Represents a single input line from a book.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BookLine {
    pub line_id: u64,
    pub section_id: u64,
    pub text: String,
    pub line_hash: u64,
    pub reference: String,
    pub segment: u64,
}

/// Represents a book ready for indexing, containing its lines and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BookForIndexing {
    pub source_book_key: String,
    pub title: String,
    pub content_hash: u64,
    pub is_pdf: bool,
    pub topics: Vec<String>,
    pub author: Option<String>,
    pub era: Option<String>,
    pub base: Option<String>,
    pub lines: Vec<BookLine>,
}

/// A chunk of text derived from a book line, prepared for semantic embedding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticChunk {
    pub semantic_id: String,
    pub source_book_key: String,
    pub source_doc_key: String,
    pub line_id: u64,
    pub section_id: u64,
    pub line_hash: u64,
    pub anchor_text: String,
    pub embedding_text: String,
    pub chunk_hash: String,
    pub content_hash: u64,
    pub reference: String,
    pub segment: u64,
    pub is_pdf: bool,
    pub title: String,
    pub topics: Vec<String>,
    pub author: Option<String>,
    pub era: Option<String>,
    pub base: Option<String>,
}

/// Metadata stored alongside each vector in the vector database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorMetadata {
    pub semantic_id: String,
    pub source_book_key: String,
    pub source_doc_key: String,
    pub line_id: u64,
    pub section_id: u64,
    pub line_hash: u64,
    pub chunk_hash: String,
    pub content_hash: u64,
    pub reference: String,
    pub segment: u64,
    pub is_pdf: bool,
    pub title: String,
    pub topics: Vec<String>,
    pub author: Option<String>,
    pub era: Option<String>,
    pub base: Option<String>,
}

impl From<&SemanticChunk> for VectorMetadata {
    fn from(chunk: &SemanticChunk) -> Self {
        Self {
            semantic_id: chunk.semantic_id.clone(),
            source_book_key: chunk.source_book_key.clone(),
            source_doc_key: chunk.source_doc_key.clone(),
            line_id: chunk.line_id,
            section_id: chunk.section_id,
            line_hash: chunk.line_hash,
            chunk_hash: chunk.chunk_hash.clone(),
            content_hash: chunk.content_hash,
            reference: chunk.reference.clone(),
            segment: chunk.segment,
            is_pdf: chunk.is_pdf,
            title: chunk.title.clone(),
            topics: chunk.topics.clone(),
            author: chunk.author.clone(),
            era: chunk.era.clone(),
            base: chunk.base.clone(),
        }
    }
}

/// A candidate result retrieved from the semantic vector search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticCandidate {
    pub metadata: VectorMetadata,
    pub similarity_score: f32,
}

/// A candidate result retrieved from the lexical (BM25) search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LexicalCandidate {
    pub title: String,
    pub reference: String,
    pub text: String,
    pub line_id: u64,
    pub section_id: u64,
    pub line_hash: u64,
    pub segment: u64,
    pub is_pdf: bool,
    pub file_path: String,
    pub bm25_score: f32,
}

/// Identifies the origin of a search result.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ResultSource {
    Lexical,
    Semantic,
    Both,
}

impl fmt::Display for ResultSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResultSource::Lexical => write!(f, "Lexical"),
            ResultSource::Semantic => write!(f, "Semantic"),
            ResultSource::Both => write!(f, "Both"),
        }
    }
}

/// A fully scored candidate that fuses lexical and semantic scores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedCandidate {
    pub title: String,
    pub reference: String,
    pub text: String,
    pub line_id: u64,
    pub section_id: u64,
    pub line_hash: u64,
    pub segment: u64,
    pub is_pdf: bool,
    pub file_path: String,
    pub source: ResultSource,
    pub raw_bm25_score: Option<f32>,
    pub normalized_bm25: Option<f32>,
    pub raw_semantic_score: Option<f32>,
    pub normalized_semantic: Option<f32>,
    pub fused_score: f32,
    pub lexical_weight: f32,
    pub semantic_weight: f32,
}

/// Represents a merged sibling in the semantic engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedSibling {
    pub title: String,
    pub reference: String,
    pub line_id: u64,
    pub segment: u64,
    pub is_pdf: bool,
    pub file_path: String,
}

/// A grouped set of fused results, collapsed by some criteria.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupedResult {
    pub representative: FusedCandidate,
    pub siblings: Vec<FusedSibling>,
    pub group_count: u32,
}

/// The modes available for executing a search.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SearchMode {
    Hybrid,
    LexicalOnly,
    SemanticOnly,
}

impl fmt::Display for SearchMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SearchMode::Hybrid => write!(f, "Hybrid"),
            SearchMode::LexicalOnly => write!(f, "Lexical Only"),
            SearchMode::SemanticOnly => write!(f, "Semantic Only"),
        }
    }
}

/// A merged sibling that is compatible with the main app's representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridMergedSibling {
    pub title: String,
    pub reference: String,
    pub id: u64,
    pub segment: u64,
    pub is_pdf: bool,
    pub file_path: String,
}

/// An individual hybrid search result item, structured for frontend compatibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridResultItem {
    pub title: String,
    pub reference: String,
    pub text: String,
    pub id: u64,
    pub segment: u64,
    pub is_pdf: bool,
    pub file_path: String,
    pub merged_count: u32,
    pub merged: Vec<HybridMergedSibling>,
    pub lexical_score: Option<f32>,
    pub semantic_score: Option<f32>,
    pub fused_score: f32,
    pub source: ResultSource,
    pub provenance: Option<FusedCandidate>,
}

/// The final payload returned by a hybrid search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridSearchResult {
    pub results: Vec<HybridResultItem>,
    pub total_count: u32,
    pub group_count: Option<u32>,
    pub search_mode: SearchMode,
    pub semantic_available: bool,
    pub latency_ms: u64,
}

/// Filters applied during the search process.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SearchFilters {
    pub book_paths: Option<Vec<String>>,
    pub topics: Option<Vec<String>>,
    pub authors: Option<Vec<String>>,
    pub eras: Option<Vec<String>>,
    pub bases: Option<Vec<String>>,
    pub include_pdf: Option<bool>,
}

impl SearchFilters {
    /// Checks if no filters are actively set.
    pub fn is_empty(&self) -> bool {
        self.book_paths.as_ref().is_none_or(|v| v.is_empty())
            && self.topics.as_ref().is_none_or(|v| v.is_empty())
            && self.authors.as_ref().is_none_or(|v| v.is_empty())
            && self.eras.as_ref().is_none_or(|v| v.is_empty())
            && self.bases.as_ref().is_none_or(|v| v.is_empty())
            && self.include_pdf.is_none()
    }
}

/// Determines how results should be grouped together.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum GroupingMode {
    SameSection,
    IdenticalText,
}

/// Represents the status of the semantic search engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticStatus {
    pub available: bool,
    pub model_loaded: bool,
    pub indexed_book_count: u32,
    pub vector_count: u32,
    pub model_id: String,
    pub embedding_dim: u32,
    pub last_error: Option<String>,
}

/// Represents the computed difference for indexing updates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexDiff {
    pub new_books: Vec<String>,
    pub changed_books: Vec<String>,
    pub removed_books: Vec<String>,
    pub model_mismatch: bool,
    pub chunking_mismatch: bool,
    pub normalization_mismatch: bool,
}

impl IndexDiff {
    /// Checks if the index is completely up to date.
    pub fn is_up_to_date(&self) -> bool {
        self.new_books.is_empty()
            && self.changed_books.is_empty()
            && self.removed_books.is_empty()
            && !self.model_mismatch
            && !self.chunking_mismatch
            && !self.normalization_mismatch
    }

    /// Returns the total number of books that need indexing.
    pub fn books_to_index(&self) -> usize {
        self.new_books.len() + self.changed_books.len()
    }
}

/// Represents the current progress of an indexing operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexingProgress {
    pub phase: String,
    pub current_book: Option<String>,
    pub books_processed: u32,
    pub books_total: u32,
    pub chunks_generated: u32,
    pub vectors_written: u32,
    pub is_complete: bool,
    pub error: Option<String>,
}
