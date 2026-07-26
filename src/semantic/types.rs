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
///
/// # Identity
///
/// `line_id` is the global Tantivy document id of the line and is the key used
/// to merge a lexical and a semantic candidate into one result. Callers must
/// therefore pass ids from the same id space on both paths; a `line_id` that is
/// only unique within a book would merge unrelated lines.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedCandidate {
    pub title: String,
    pub reference: String,
    /// Line body. Empty when the candidate came from the semantic path only —
    /// see [`FusedCandidate::needs_hydration`].
    pub text: String,
    pub line_id: u64,
    pub section_id: u64,
    pub line_hash: u64,
    pub segment: u64,
    pub is_pdf: bool,
    pub file_path: String,
    /// `true` when `text` was not supplied by the lexical path and must be
    /// loaded from Tantivy by `line_id` before display.
    ///
    /// The vector store deliberately does not duplicate line bodies, so a
    /// semantic-only hit carries metadata but no text. Rendering such a result
    /// without hydrating it produces an empty card. Hydration itself is wired up
    /// in roadmap P5; until then this flag is the contract that tells a caller
    /// the text is missing rather than genuinely blank.
    pub needs_hydration: bool,
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
    /// Line body, or empty when [`HybridResultItem::needs_hydration`] is set.
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
    /// `true` when `text` must be loaded from Tantivy by `id` before display.
    /// See [`FusedCandidate::needs_hydration`].
    pub needs_hydration: bool,
    pub source: ResultSource,
    pub provenance: Option<FusedCandidate>,
}

/// The final payload returned by a hybrid search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridSearchResult {
    pub results: Vec<HybridResultItem>,
    /// Number of candidates that took part in fusion — **not** the number of
    /// lines in the library that match the query.
    ///
    /// The coordinator only ever sees the candidate window it was handed
    /// (lexical candidates from Tantivy plus its own top-k semantic hits), so a
    /// corpus-wide total cannot be derived here. A truthful total has to come
    /// from the lexical engine's own count; wiring that through is roadmap P5.
    pub total_count: u32,
    /// Number of groups when grouping was requested, otherwise `None`.
    pub group_count: Option<u32>,
    /// The mode that actually ran. May differ from a requested `force_mode` when
    /// the semantic path was unavailable — compare the two to detect degradation,
    /// and read [`HybridSearchResult::fallback_reason`] for the cause.
    pub search_mode: SearchMode,
    /// Whether the semantic path was actually usable for this query.
    ///
    /// `false` covers every reason the semantic side contributed nothing: no
    /// engine configured, model not loaded, index incompatible, or a failure
    /// mid-query.
    pub semantic_available: bool,
    /// Why the semantic path was skipped or failed, when it was.
    ///
    /// Graceful degradation to BM25 is a design requirement, but a silent
    /// degradation is indistinguishable from "the semantic engine agreed with
    /// BM25". This field makes it observable.
    pub fallback_reason: Option<String>,
    pub latency_ms: u64,
}

/// Filters applied during the search process.
///
/// # Contract
///
/// Each field is one filter *dimension*. Values inside a dimension are OR-ed,
/// and dimensions are AND-ed together — the same semantics the lexical engine
/// applies to its `topics` facet field, so both retrieval paths select the same
/// documents.
///
/// * `None` **and** `Some(vec![])` both mean "this dimension does not filter".
///   An empty list must never select zero documents; that made
///   [`SearchFilters::is_empty`] disagree with the actual matching and silently
///   emptied result sets.
/// * Values match hierarchically, like Tantivy facets: the filter `/מקרא`
///   matches the value `/מקרא/תורה`. An exact value still matches itself.
/// * `include_pdf`: `Some(false)` excludes PDF books; `Some(true)` and `None`
///   include everything. It is an exclusion switch, not a "PDFs only" switch.
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
    /// Checks if no filter actually constrains the result set.
    ///
    /// Stays in lockstep with [`SearchFilters::matches`]: whenever this returns
    /// `true`, `matches` accepts every candidate.
    pub fn is_empty(&self) -> bool {
        active(&self.book_paths).is_none()
            && active(&self.topics).is_none()
            && active(&self.authors).is_none()
            && active(&self.eras).is_none()
            && active(&self.bases).is_none()
            && !self.excludes_pdf()
    }

    /// Whether PDF books are filtered out. See the type-level contract.
    pub fn excludes_pdf(&self) -> bool {
        self.include_pdf == Some(false)
    }

    /// Whether `meta` passes every active filter dimension.
    pub fn matches(&self, meta: &VectorMetadata) -> bool {
        if let Some(paths) = active(&self.book_paths) {
            if !paths.iter().any(|p| p == &meta.source_book_key) {
                return false;
            }
        }

        if self.excludes_pdf() && meta.is_pdf {
            return false;
        }

        if let Some(topics) = active(&self.topics) {
            if !meta
                .topics
                .iter()
                .any(|value| topics.iter().any(|f| facet_matches(value, f)))
            {
                return false;
            }
        }

        if !matches_optional_dimension(active(&self.authors), meta.author.as_deref()) {
            return false;
        }
        if !matches_optional_dimension(active(&self.eras), meta.era.as_deref()) {
            return false;
        }
        if !matches_optional_dimension(active(&self.bases), meta.base.as_deref()) {
            return false;
        }

        true
    }
}

/// Returns the filter values of a dimension, or `None` when it does not filter.
fn active(dimension: &Option<Vec<String>>) -> Option<&[String]> {
    match dimension {
        Some(values) if !values.is_empty() => Some(values),
        _ => None,
    }
}

/// Whether a single-valued metadata dimension satisfies its filter.
///
/// A candidate with no value for a filtered dimension is excluded: "books by
/// רש\"י" must not return a book with no recorded author.
fn matches_optional_dimension(filter: Option<&[String]>, value: Option<&str>) -> bool {
    let Some(filter) = filter else { return true };
    match value {
        Some(value) => filter.iter().any(|f| facet_matches(value, f)),
        None => false,
    }
}

/// Hierarchical facet comparison: `filter` matches `value` when it is `value`
/// itself or one of its ancestors.
///
/// Mirrors Tantivy's facet indexing, which stores every ancestor path as a term
/// so a filter on `/מקרא` also selects `/מקרא/תורה`.
fn facet_matches(value: &str, filter: &str) -> bool {
    if value == filter {
        return true;
    }
    let ancestor = filter.strip_suffix('/').unwrap_or(filter);
    value.len() > ancestor.len()
        && value.starts_with(ancestor)
        && value.as_bytes()[ancestor.len()] == b'/'
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
    /// Whether a semantic query can be served right now — a model is loaded,
    /// the index is compatible, and it holds at least one vector.
    ///
    /// This is not "a semantic engine exists": an engine with an empty or
    /// incompatible index reports `false`, because it cannot contribute results.
    pub available: bool,
    pub model_loaded: bool,
    pub indexed_book_count: u32,
    pub vector_count: u32,
    pub model_id: String,
    pub embedding_dim: u32,
    /// Identifier of the loaded embedding backend, or `None` before a model is
    /// loaded. `Some("mock-hash-v1")` means the vectors are **not** semantic.
    pub embedding_backend: Option<String>,
    /// Identifier of the vector storage backend (e.g. `"in-memory-v1"`).
    pub vector_backend: String,
    /// Whether stored vectors survive a restart. While `false`, every session
    /// starts from an empty index and a full re-index is required.
    pub vectors_persisted: bool,
    /// Set when the whole index must be rebuilt before the semantic path can be
    /// used — either the on-disk index is incompatible with the current
    /// configuration, or vectors were lost. Contains the human-readable reason.
    pub needs_full_reindex: Option<String>,
    pub last_error: Option<String>,
}

/// Represents the computed difference for indexing updates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexDiff {
    pub new_books: Vec<String>,
    pub changed_books: Vec<String>,
    pub removed_books: Vec<String>,
    /// The embedding model, its file checksum, its backend or the vector
    /// precision changed — every existing vector is invalid.
    pub model_mismatch: bool,
    /// The chunking algorithm version changed.
    pub chunking_mismatch: bool,
    /// The text-normalization version changed.
    pub normalization_mismatch: bool,
}

impl IndexDiff {
    /// Checks if the index is completely up to date.
    pub fn is_up_to_date(&self) -> bool {
        self.new_books.is_empty()
            && self.changed_books.is_empty()
            && self.removed_books.is_empty()
            && !self.needs_full_rebuild()
    }

    /// Whether a configuration change invalidated the whole index, so an
    /// incremental update is not enough.
    pub fn needs_full_rebuild(&self) -> bool {
        self.model_mismatch || self.chunking_mismatch || self.normalization_mismatch
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

#[cfg(test)]
mod tests {
    use super::*;

    fn meta() -> VectorMetadata {
        VectorMetadata {
            semantic_id: "id1".to_string(),
            source_book_key: "otzaria/tanach/genesis.txt".to_string(),
            source_doc_key: "otzaria/tanach/genesis.txt:1".to_string(),
            line_id: 1,
            section_id: 10,
            line_hash: 100,
            chunk_hash: "hash".to_string(),
            content_hash: 555,
            reference: "בראשית א:א".to_string(),
            segment: 0,
            is_pdf: false,
            title: "בראשית".to_string(),
            topics: vec!["/מקרא/תורה/בראשית".to_string()],
            author: Some("/author/משה רבנו".to_string()),
            era: Some("/era/תנך".to_string()),
            base: Some("/base/תורה".to_string()),
        }
    }

    #[test]
    fn no_filters_accepts_everything() {
        let filters = SearchFilters::default();
        assert!(filters.is_empty());
        assert!(filters.matches(&meta()));
    }

    /// The bug this pins down: `is_empty()` said an empty list is not a filter
    /// while matching treated it as "must be one of zero values" and dropped
    /// every candidate.
    #[test]
    fn empty_lists_do_not_filter_anything_out() {
        let filters = SearchFilters {
            book_paths: Some(vec![]),
            topics: Some(vec![]),
            authors: Some(vec![]),
            eras: Some(vec![]),
            bases: Some(vec![]),
            include_pdf: None,
        };
        assert!(
            filters.is_empty(),
            "empty lists must not count as active filters"
        );
        assert!(
            filters.matches(&meta()),
            "empty lists must not exclude candidates"
        );
    }

    #[test]
    fn is_empty_agrees_with_matches_for_every_single_dimension() {
        let values = vec!["nothing-matches-this".to_string()];
        let dimensions: Vec<(&str, SearchFilters)> = vec![
            (
                "book_paths",
                SearchFilters {
                    book_paths: Some(values.clone()),
                    ..Default::default()
                },
            ),
            (
                "topics",
                SearchFilters {
                    topics: Some(values.clone()),
                    ..Default::default()
                },
            ),
            (
                "authors",
                SearchFilters {
                    authors: Some(values.clone()),
                    ..Default::default()
                },
            ),
            (
                "eras",
                SearchFilters {
                    eras: Some(values.clone()),
                    ..Default::default()
                },
            ),
            (
                "bases",
                SearchFilters {
                    bases: Some(values),
                    ..Default::default()
                },
            ),
        ];

        for (name, filters) in dimensions {
            assert!(
                !filters.is_empty(),
                "{name} should count as an active filter"
            );
            assert!(
                !filters.matches(&meta()),
                "{name} should exclude the candidate"
            );
        }
    }

    #[test]
    fn book_path_filter_is_an_exact_match_on_the_book_key() {
        let matching = SearchFilters {
            book_paths: Some(vec!["otzaria/tanach/genesis.txt".to_string()]),
            ..Default::default()
        };
        assert!(matching.matches(&meta()));

        // A book key is a path, not a facet: a directory prefix must not match.
        let prefix = SearchFilters {
            book_paths: Some(vec!["otzaria/tanach".to_string()]),
            ..Default::default()
        };
        assert!(!prefix.matches(&meta()));
    }

    #[test]
    fn topic_filters_match_hierarchically_like_tantivy_facets() {
        for filter in ["/מקרא", "/מקרא/תורה", "/מקרא/תורה/בראשית", "/מקרא/"]
        {
            let filters = SearchFilters {
                topics: Some(vec![filter.to_string()]),
                ..Default::default()
            };
            assert!(filters.matches(&meta()), "filter {filter} should match");
        }

        for filter in ["/תלמוד", "/מקרא/נביאים", "/מקר"] {
            let filters = SearchFilters {
                topics: Some(vec![filter.to_string()]),
                ..Default::default()
            };
            assert!(
                !filters.matches(&meta()),
                "filter {filter} should not match"
            );
        }
    }

    #[test]
    fn values_within_a_dimension_are_or_ed() {
        let filters = SearchFilters {
            topics: Some(vec!["/תלמוד".to_string(), "/מקרא".to_string()]),
            ..Default::default()
        };
        assert!(filters.matches(&meta()));
    }

    #[test]
    fn dimensions_are_and_ed() {
        let both_match = SearchFilters {
            topics: Some(vec!["/מקרא".to_string()]),
            eras: Some(vec!["/era/תנך".to_string()]),
            ..Default::default()
        };
        assert!(both_match.matches(&meta()));

        let one_fails = SearchFilters {
            topics: Some(vec!["/מקרא".to_string()]),
            eras: Some(vec!["/era/ראשונים".to_string()]),
            ..Default::default()
        };
        assert!(!one_fails.matches(&meta()));
    }

    #[test]
    fn a_candidate_missing_a_filtered_dimension_is_excluded() {
        let mut without_author = meta();
        without_author.author = None;

        let filters = SearchFilters {
            authors: Some(vec!["/author/משה רבנו".to_string()]),
            ..Default::default()
        };
        assert!(!filters.matches(&without_author));
    }

    #[test]
    fn include_pdf_is_an_exclusion_switch() {
        let mut pdf = meta();
        pdf.is_pdf = true;
        let text = meta();

        let exclude = SearchFilters {
            include_pdf: Some(false),
            ..Default::default()
        };
        assert!(!exclude.is_empty());
        assert!(!exclude.matches(&pdf), "Some(false) must drop PDF books");
        assert!(exclude.matches(&text), "Some(false) must keep text books");

        // Some(true)/None mean "no restriction" — not "PDFs only".
        for include_pdf in [Some(true), None] {
            let filters = SearchFilters {
                include_pdf,
                ..Default::default()
            };
            assert!(
                filters.is_empty(),
                "{include_pdf:?} must not be an active filter"
            );
            assert!(filters.matches(&pdf), "{include_pdf:?} must keep PDF books");
            assert!(
                filters.matches(&text),
                "{include_pdf:?} must keep text books"
            );
        }
    }

    #[test]
    fn index_diff_up_to_date_contract() {
        let clean = IndexDiff {
            new_books: vec![],
            changed_books: vec![],
            removed_books: vec![],
            model_mismatch: false,
            chunking_mismatch: false,
            normalization_mismatch: false,
        };
        assert!(clean.is_up_to_date());
        assert!(!clean.needs_full_rebuild());
        assert_eq!(clean.books_to_index(), 0);

        let model_changed = IndexDiff {
            model_mismatch: true,
            ..clean.clone()
        };
        assert!(!model_changed.is_up_to_date());
        assert!(model_changed.needs_full_rebuild());

        let with_books = IndexDiff {
            new_books: vec!["a".to_string()],
            changed_books: vec!["b".to_string()],
            ..clean
        };
        assert!(!with_books.is_up_to_date());
        assert!(!with_books.needs_full_rebuild());
        assert_eq!(with_books.books_to_index(), 2);
    }
}
