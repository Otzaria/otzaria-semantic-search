use serde::{Deserialize, Serialize};
use std::fmt;
use std::num::NonZeroU64;

/// Reserved first segments of a facet path that denote a filter *dimension*
/// rather than a node of the category tree.
///
/// Mirrors the lexical engine's `FACET_DIMENSION_ROOTS`, deliberately down to the
/// English names: library categories are Hebrew, so these cannot collide with
/// them. Both engines must group facets identically or the same filter would
/// select different documents on each path.
pub const FACET_DIMENSION_ROOTS: [&str; 3] = ["author", "era", "base"];

/// Number of filter dimension groups: the category tree plus one per reserved
/// root.
const FACET_GROUP_COUNT: usize = FACET_DIMENSION_ROOTS.len() + 1;

/// What a book's fingerprint can prove, before its lines have been loaded.
///
/// # Why this is not a bare `u64`
///
/// Two separate traps live here, and both are silent.
///
/// **Zero is a marker, not a hash.** PDF books have no lexical content hash:
/// their extracted text does not live in the library database, so the lexical
/// engine records `contentHash = 0`. Compared as an ordinary hash, `0 == 0`
/// makes every PDF look permanently unchanged and a replaced scan would never be
/// re-indexed. The zero state therefore gets its own variant, and the hash
/// variants carry a [`NonZeroU64`] so a caller *cannot* pass zero as a hash.
///
/// **Not every fingerprint covers metadata.** Title, category path and facets
/// are stored in every vector record and drive filtering and display, but they
/// live outside the book's text. A fingerprint over the content alone — a PDF's
/// file size and mtime, say — does not move when an author is corrected, so
/// matching it would declare an index current while it keeps serving the old
/// author. Which kind of fingerprint the caller has is something only the caller
/// knows, so it is part of the value rather than an assumption.
///
/// The lexical engine's `compute_book_fingerprint` is [`Self::Canonical`]: it
/// hashes the text *and* title, category path, catalogue/generation order and
/// its sorted, deduplicated facets. [`Self::canonical`] builds the equivalent
/// for a book the lexical index cannot fingerprint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContentFingerprint {
    /// Covers content *and* the metadata stored in every vector record. A match
    /// proves the index is current.
    Canonical(NonZeroU64),
    /// Covers content only. A match proves the *text* is unchanged and nothing
    /// more, so the book is still re-examined — cheaply, since
    /// [`BookForIndexing::line_fingerprint`] then settles it without inference.
    ContentOnly(NonZeroU64),
    /// Nothing to compare. The book has to be re-examined from its lines.
    Unverifiable,
}

impl ContentFingerprint {
    /// Interpret a raw `contentHash` from the lexical index.
    ///
    /// Zero is the engine's "no fingerprint" marker, not a hash value. A nonzero
    /// lexical hash is [`Self::Canonical`] because the lexical engine's
    /// fingerprint already folds in the metadata it indexes.
    pub fn from_lexical_hash(content_hash: u64) -> Self {
        match NonZeroU64::new(content_hash) {
            Some(hash) => Self::Canonical(hash),
            None => Self::Unverifiable,
        }
    }

    /// Build a canonical fingerprint for a book the lexical index cannot
    /// fingerprint — a PDF.
    ///
    /// `source_signature` must be the caller's nonzero, authoritative revision
    /// of every source input that determines [`BookForIndexing`]: extracted text,
    /// line/section boundaries and identifiers, references, and the extraction or
    /// OCR version. File size and mtime alone are only a heuristic and belong in
    /// [`Self::content_only`], not here. This method folds the searchable book
    /// metadata in on top, so correcting an author or moving a book between
    /// categories changes the fingerprint even though the file did not.
    ///
    /// Zero means that no authoritative source revision is available and yields
    /// [`Self::Unverifiable`]. Hashing that marker together with metadata would
    /// otherwise turn "unknown" into a seemingly canonical fingerprint.
    ///
    /// The resulting value must be used **twice**: handed to
    /// [`SemanticEngine::diff`](crate::semantic::engine::SemanticEngine::diff)
    /// as this book's fingerprint, and written into
    /// [`BookForIndexing::content_fingerprint`] when the book is indexed. The
    /// two are compared directly, so a mismatch means the book is re-indexed on
    /// every pass.
    pub fn canonical(
        source_signature: u64,
        title: &str,
        topics: &str,
        extra_facets: &[String],
        is_pdf: bool,
    ) -> Self {
        let Some(source_signature) = NonZeroU64::new(source_signature) else {
            return Self::Unverifiable;
        };
        Self::Canonical(canonical_book_fingerprint(
            source_signature,
            title,
            topics,
            extra_facets,
            is_pdf,
        ))
    }

    /// A fingerprint over content alone.
    ///
    /// For a caller that has a file signature and no metadata to fold in.
    /// Prefer [`Self::canonical`]: this cannot reach "nothing to do", because a
    /// metadata-only change is invisible to it. Zero yields
    /// [`Self::Unverifiable`].
    pub fn content_only(source_signature: u64) -> Self {
        match NonZeroU64::new(source_signature) {
            Some(hash) => Self::ContentOnly(hash),
            None => Self::Unverifiable,
        }
    }

    /// The raw value to persist and compare. `0` for [`Self::Unverifiable`].
    pub fn as_raw(&self) -> u64 {
        match self {
            Self::Canonical(hash) | Self::ContentOnly(hash) => hash.get(),
            Self::Unverifiable => 0,
        }
    }

    /// Whether a match proves the *whole* record — content and metadata — is
    /// current. Only [`Self::Canonical`] does.
    pub fn proves_current(&self) -> bool {
        matches!(self, Self::Canonical(_))
    }

    /// Whether there is any hash to compare at all.
    pub fn is_verifiable(&self) -> bool {
        !matches!(self, Self::Unverifiable)
    }
}

/// Canonical book fingerprint: a source signature plus the metadata stored in
/// every vector record.
///
/// Mirrors the shape of the lexical engine's `book_fingerprint` rather than its
/// bytes (the inputs differ — there is no text here): FNV-1a over
/// length-prefixed fields, with the facets sorted and deduplicated because their
/// order carries no meaning in either index. Never returns zero, which is the
/// "no fingerprint" marker.
fn canonical_book_fingerprint(
    source_signature: NonZeroU64,
    title: &str,
    topics: &str,
    extra_facets: &[String],
    is_pdf: bool,
) -> NonZeroU64 {
    let mut fnv = Fnv::new();
    fnv.feed_field(&source_signature.get().to_le_bytes());
    fnv.feed_field(title.as_bytes());
    fnv.feed_field(topics.as_bytes());
    fnv.feed_field(&[u8::from(is_pdf)]);
    for facet in canonical_facets(extra_facets) {
        fnv.feed_field(facet.as_bytes());
    }
    fnv.finish_nonzero()
}

/// Sort and deduplicate facet paths.
///
/// Order and repetition of facets mean nothing to either index — Tantivy indexes
/// each path and its ancestors, and a repeat adds no term. Leaving them raw
/// would make two descriptions of the same book fingerprint differently and
/// trigger a pointless re-index; the lexical engine canonicalizes for the same
/// reason.
fn canonical_facets(facets: &[String]) -> Vec<&str> {
    let mut canonical: Vec<&str> = facets
        .iter()
        .map(|facet| facet.as_str())
        .filter(|facet| !facet.trim().is_empty())
        .collect();
    canonical.sort_unstable();
    canonical.dedup();
    canonical
}

/// Accumulating FNV-1a, the basis of every fingerprint this crate computes.
///
/// `feed_field` length-prefixes, so concatenations of different fields cannot
/// collide (`"ab" + "c"` against `"a" + "bc"`). Stable across runs and
/// platforms, which is the only property required of it — nothing here is
/// adversarial.
struct Fnv(u64);

impl Fnv {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    fn new() -> Self {
        Self(Self::OFFSET)
    }

    fn feed(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 ^= u64::from(*byte);
            self.0 = self.0.wrapping_mul(Self::PRIME);
        }
    }

    fn feed_field(&mut self, bytes: &[u8]) {
        self.feed(&(bytes.len() as u64).to_le_bytes());
        self.feed(bytes);
    }

    fn finish(self) -> u64 {
        self.0
    }

    /// Finish, mapping the one value that would be read as "no fingerprint".
    fn finish_nonzero(self) -> NonZeroU64 {
        NonZeroU64::new(self.0).unwrap_or(NonZeroU64::MIN)
    }
}

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
///
/// The metadata mirrors what the lexical indexer passes to Tantivy, so both
/// engines describe a book the same way.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BookForIndexing {
    pub source_book_key: String,
    pub title: String,
    /// The raw fingerprint the caller vouches for, `0` meaning "none".
    ///
    /// **It must be the raw value of the same [`ContentFingerprint`] handed to
    /// [`SemanticEngine::diff`](crate::semantic::engine::SemanticEngine::diff)
    /// for this book**, since that is what the skip decision compares against.
    /// For a text book that is the lexical engine's `contentHash`; for a PDF,
    /// [`ContentFingerprint::canonical`] builds it and `as_raw()` unwraps it.
    /// Anything else is not wrong so much as useless: the book is re-indexed on
    /// every pass.
    pub content_fingerprint: u64,
    pub is_pdf: bool,
    /// The book's category facet path, e.g. `/מקרא/תורה/בראשית`. One path per
    /// book, as in the lexical engine's `topics` argument.
    pub topics: String,
    /// Additional facet paths under the reserved dimension roots, e.g.
    /// `/author/רש״י`, `/era/ראשונים`, `/base`.
    ///
    /// A list, not a set of single-valued fields: a book can have several
    /// authors, and the lexical indexer adds a facet for each. Collapsing them
    /// to one author would make a filter on any other author of the same book
    /// return nothing.
    pub extra_facets: Vec<String>,
    pub lines: Vec<BookLine>,
}

impl BookForIndexing {
    /// Every facet path describing this book: the category path plus the
    /// dimension facets, which is exactly the set the lexical engine indexes
    /// into its single `topics` field.
    /// Sorted and deduplicated — the list is stored on every vector, and neither
    /// order nor repetition means anything to a filter.
    pub fn all_facets(&self) -> Vec<String> {
        let mut facets = Vec::with_capacity(self.extra_facets.len() + 1);
        facets.extend(self.extra_facets.iter().map(String::as_str));
        if !self.topics.trim().is_empty() {
            facets.push(self.topics.as_str());
        }
        let mut facets = canonical_facets_owned(&facets);
        facets.shrink_to_fit();
        facets
    }

    /// This book's fingerprint as recorded by the caller.
    ///
    /// Read back as [`ContentFingerprint::Canonical`] when nonzero: the caller is
    /// responsible for having put a canonical value in
    /// [`Self::content_fingerprint`], and this crate cannot tell the difference
    /// by looking. What it *can* do is catch the consequence anyway —
    /// [`Self::line_fingerprint`] covers the same metadata, so a book handed over
    /// with stale metadata is still re-embedded.
    pub fn fingerprint(&self) -> ContentFingerprint {
        ContentFingerprint::from_lexical_hash(self.content_fingerprint)
    }

    /// A fingerprint this crate computes from the book itself, for books whose
    /// lexical `contentHash` is [`ContentFingerprint::Unverifiable`].
    ///
    /// # What it must cover
    ///
    /// Everything that ends up inside a [`VectorMetadata`] record, not just the
    /// embedded text. This value gates the "nothing changed, skip it" decision,
    /// so anything it omits can change without the index noticing — a book whose
    /// author, category, title or references were corrected would keep serving
    /// the old values for filtering and display. Renaming a field in
    /// `VectorMetadata` is a reason to revisit this function.
    ///
    /// So: title, category path, dimension facets, the PDF flag, and per line the
    /// id, section, segment, reference, dedup hash and text.
    pub fn line_fingerprint(&self) -> u64 {
        let mut fnv = Fnv::new();

        // ── book-level metadata stored in every record ──
        fnv.feed_field(self.title.as_bytes());
        fnv.feed_field(self.topics.as_bytes());
        fnv.feed_field(&[u8::from(self.is_pdf)]);
        // Sorted and deduplicated, like everything else that hashes facets: two
        // descriptions of the same book must agree, or the book is re-embedded
        // because a caller happened to list its authors in a different order.
        for facet in canonical_facets(&self.extra_facets) {
            fnv.feed_field(facet.as_bytes());
        }

        // ── per-line identity and content ──
        fnv.feed_field(&(self.lines.len() as u64).to_le_bytes());
        for line in &self.lines {
            fnv.feed_field(&line.line_id.to_le_bytes());
            fnv.feed_field(&line.section_id.to_le_bytes());
            fnv.feed_field(&line.segment.to_le_bytes());
            fnv.feed_field(&line.line_hash.to_le_bytes());
            fnv.feed_field(line.reference.as_bytes());
            // `line_hash` is 0 for lines the lexical engine considers too short
            // to deduplicate, so it cannot stand in for the text on its own.
            fnv.feed_field(line.text.as_bytes());
        }
        fnv.finish()
    }
}

/// [`canonical_facets`] for a caller that needs owned strings.
fn canonical_facets_owned(facets: &[&str]) -> Vec<String> {
    let mut canonical: Vec<&str> = facets
        .iter()
        .copied()
        .filter(|facet| !facet.trim().is_empty())
        .collect();
    canonical.sort_unstable();
    canonical.dedup();
    canonical.into_iter().map(str::to_string).collect()
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
    /// All facet paths describing the book — see [`BookForIndexing::all_facets`].
    pub facets: Vec<String>,
}

/// Metadata stored alongside each vector in the vector database.
///
/// Note that the facet list is duplicated per chunk. That is a known cost of the
/// current in-memory backend; how metadata is laid out (shared per book, or
/// hydrated from Tantivy instead of stored at all) is decided with the
/// persistent read-only backend in stage S2.
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
    /// All facet paths describing the book, categories and dimensions together,
    /// exactly as the lexical engine indexes them into its `topics` field.
    pub facets: Vec<String>,
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
            facets: chunk.facets.clone(),
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
    /// without hydrating it produces an empty card. Hydrating by `line_id` is the
    /// host engine's job — `otzaria_search_engine` owns Tantivy — and S5 is where the
    /// official path is wired to it. This flag is the contract that tells a caller
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
    /// from the lexical engine's own count, so wiring it through belongs to the host
    /// engine. Still open.
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
    pub confidence: Option<f32>,
    pub profile: Option<String>,
    pub telemetry: Option<crate::telemetry::SearchTelemetry>,
}

/// Filters applied during the search process.
///
/// # Contract
///
/// `facets` is a flat list of facet paths — categories and dimensions mixed —
/// exactly the shape the lexical engine's facet filter takes. Passing the same
/// list to both engines is what guarantees they select the same documents;
/// re-modelling the dimensions as separate fields here would be a second
/// implementation of the same rule, free to drift from it.
///
/// * Paths sharing a dimension are OR-ed; different dimensions are AND-ed. A
///   path's dimension is its first segment when that segment is one of
///   [`FACET_DIMENSION_ROOTS`], and the category tree otherwise. So
///   "ראשונים AND מסכת ברכות" works while two authors stay "either of them".
/// * Matching is hierarchical, like Tantivy facets, which index every ancestor
///   path as a term: the filter `/מקרא` selects `/מקרא/תורה`.
/// * `None` **and** `Some(vec![])` both mean "does not filter". An empty list
///   must never select zero documents; that made [`SearchFilters::is_empty`]
///   disagree with the actual matching and silently emptied result sets.
/// * `include_pdf`: `Some(false)` excludes PDF books; `Some(true)` and `None`
///   include everything. It is an exclusion switch, not a "PDFs only" switch.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SearchFilters {
    /// Book keys (file paths). Exact matches, not facet paths.
    pub book_paths: Option<Vec<String>>,
    /// Facet paths, e.g. `["/מקרא/תורה", "/author/רש״י"]`.
    pub facets: Option<Vec<String>>,
    pub include_pdf: Option<bool>,
}

impl SearchFilters {
    /// Checks if no filter actually constrains the result set.
    ///
    /// Stays in lockstep with [`SearchFilters::matches`]: whenever this returns
    /// `true`, `matches` accepts every candidate.
    pub fn is_empty(&self) -> bool {
        active(&self.book_paths).is_none() && active(&self.facets).is_none() && !self.excludes_pdf()
    }

    /// Whether PDF books are filtered out. See the type-level contract.
    pub fn excludes_pdf(&self) -> bool {
        self.include_pdf == Some(false)
    }

    /// Group the facet paths by dimension once, so per-candidate matching does no
    /// work that is the same for every candidate.
    ///
    /// Returns `None` when nothing filters. Vector search calls the result once
    /// per stored vector, so grouping inside the match would repeat this for
    /// millions of candidates.
    pub fn compile(&self) -> Option<CompiledFilters<'_>> {
        if self.is_empty() {
            return None;
        }

        let mut facet_groups: [Vec<&str>; FACET_GROUP_COUNT] = Default::default();
        for path in active(&self.facets).unwrap_or_default() {
            facet_groups[facet_dimension(path)].push(path.as_str());
        }

        Some(CompiledFilters {
            book_paths: active(&self.book_paths),
            facet_groups,
            excludes_pdf: self.excludes_pdf(),
        })
    }

    /// Whether `meta` passes every active filter.
    ///
    /// Convenience over [`SearchFilters::compile`]; prefer compiling once when
    /// testing many candidates.
    pub fn matches(&self, meta: &VectorMetadata) -> bool {
        match self.compile() {
            Some(compiled) => compiled.matches(meta),
            None => true,
        }
    }
}

/// [`SearchFilters`] with its facet paths pre-grouped by dimension.
pub struct CompiledFilters<'a> {
    book_paths: Option<&'a [String]>,
    /// Index 0 is the category tree; `i + 1` is `FACET_DIMENSION_ROOTS[i]`.
    facet_groups: [Vec<&'a str>; FACET_GROUP_COUNT],
    excludes_pdf: bool,
}

impl CompiledFilters<'_> {
    /// Whether `meta` passes every active filter.
    pub fn matches(&self, meta: &VectorMetadata) -> bool {
        if let Some(paths) = self.book_paths {
            if !paths.iter().any(|path| path == &meta.source_book_key) {
                return false;
            }
        }

        if self.excludes_pdf && meta.is_pdf {
            return false;
        }

        // Every dimension the caller filtered on must be satisfied by at least
        // one of the book's facets (AND across groups, OR within a group).
        for group in &self.facet_groups {
            if group.is_empty() {
                continue;
            }
            let satisfied = meta
                .facets
                .iter()
                .any(|value| group.iter().any(|filter| facet_matches(value, filter)));
            if !satisfied {
                return false;
            }
        }

        true
    }
}

/// Which filter dimension a facet path belongs to.
///
/// Index 0 is the category tree; `i + 1` is `FACET_DIMENSION_ROOTS[i]`. Same
/// derivation as the lexical engine's facet filter.
fn facet_dimension(path: &str) -> usize {
    let root = path.trim_start_matches('/').split('/').next().unwrap_or("");
    FACET_DIMENSION_ROOTS
        .iter()
        .position(|dimension| *dimension == root)
        .map_or(0, |index| index + 1)
}

/// Returns the values of a filter list, or `None` when it does not filter.
fn active(values: &Option<Vec<String>>) -> Option<&[String]> {
    match values {
        Some(values) if !values.is_empty() => Some(values),
        _ => None,
    }
}

/// Hierarchical facet comparison: `filter` matches `value` when it is `value`
/// itself or one of its ancestors.
///
/// Hierarchical facet comparison: `filter` matches `value` when it is `value`
/// itself or one of its ancestors.
///
/// Mirrors Tantivy's facet indexing, which stores every ancestor path as a term
/// so a filter on `/מקרא` also selects `/מקרא/תורה`.
fn facet_matches(value: &str, filter: &str) -> bool {
    let value_clean = value.trim().trim_end_matches('/');
    let filter_clean = filter.trim().trim_end_matches('/');
    if value_clean == filter_clean {
        return true;
    }
    value_clean.len() > filter_clean.len()
        && value_clean.starts_with(filter_clean)
        && value_clean.as_bytes()[filter_clean.len()] == b'/'
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
    /// Books with no record: never processed.
    pub new_books: Vec<String>,
    /// Books recorded, and provably changed.
    pub changed_books: Vec<String>,
    /// Books recorded, but whose fingerprint cannot prove they are unchanged —
    /// [`ContentFingerprint::Unverifiable`], i.e. PDFs when the caller passes raw
    /// lexical content hashes.
    ///
    /// Kept separate from `changed_books` on purpose. These are not known to have
    /// changed; producing one costs the caller real work (a PDF has to have its
    /// text extracted again just to be compared), so it deserves its own decision
    /// rather than being buried among the genuinely stale. A caller that can
    /// supply an authoritative revision covering the extracted text and its
    /// structure empties this list and gets an honestly up-to-date diff.
    pub unverifiable_books: Vec<String>,
    /// Books recorded here but absent from the library: their vectors should go.
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
    /// An empty diff: nothing to index, nothing stale, nothing to remove.
    pub fn up_to_date() -> Self {
        Self {
            new_books: Vec::new(),
            changed_books: Vec::new(),
            unverifiable_books: Vec::new(),
            removed_books: Vec::new(),
            model_mismatch: false,
            chunking_mismatch: false,
            normalization_mismatch: false,
        }
    }

    /// Checks if the index is completely up to date.
    ///
    /// Books that merely *cannot be proven* current count as work: the index
    /// cannot claim to be current while it does not know.
    pub fn is_up_to_date(&self) -> bool {
        self.new_books.is_empty()
            && self.changed_books.is_empty()
            && self.unverifiable_books.is_empty()
            && self.removed_books.is_empty()
            && !self.needs_full_rebuild()
    }

    /// Whether a configuration change invalidated the whole index, so an
    /// incremental update is not enough.
    pub fn needs_full_rebuild(&self) -> bool {
        self.model_mismatch || self.chunking_mismatch || self.normalization_mismatch
    }

    /// Returns the total number of books to hand back for processing.
    pub fn books_to_index(&self) -> usize {
        self.new_books.len() + self.changed_books.len() + self.unverifiable_books.len()
    }

    /// Every book to hand back, in a stable order: new, then changed, then
    /// unverifiable.
    pub fn books_needing_work(&self) -> impl Iterator<Item = &str> {
        self.new_books
            .iter()
            .chain(self.changed_books.iter())
            .chain(self.unverifiable_books.iter())
            .map(String::as_str)
    }
}

/// What indexing actually did to one book.
///
/// Distinguishes "wrote vectors" from "already current" — a distinction the plain
/// chunk count erased, since a skipped book reported the count it already had as
/// though it had just been written.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexOutcome {
    /// Vectors were written. `chunks` is how many.
    Indexed { chunks: u32 },
    /// Already current, nothing written. `chunks` is what the index holds for it.
    Skipped { chunks: u32 },
    /// Nothing embeddable; an empty-book marker was recorded so the book is not
    /// offered again.
    Empty,
}

impl IndexOutcome {
    /// Chunks embedded and stored by this call. Zero for a skip.
    pub fn chunks_written(&self) -> u32 {
        match self {
            Self::Indexed { chunks } => *chunks,
            Self::Skipped { .. } | Self::Empty => 0,
        }
    }

    /// Chunks the index holds for this book afterwards.
    pub fn chunks_in_index(&self) -> u32 {
        match self {
            Self::Indexed { chunks } | Self::Skipped { chunks } => *chunks,
            Self::Empty => 0,
        }
    }

    /// Whether this call did any embedding work.
    pub fn did_work(&self) -> bool {
        matches!(self, Self::Indexed { .. } | Self::Empty)
    }
}

/// Aggregate result of indexing several books.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexingSummary {
    /// Books whose vectors were written.
    pub books_indexed: u32,
    /// Books already current, skipped without embedding.
    pub books_skipped: u32,
    /// Books that yielded nothing embeddable.
    pub books_empty: u32,
    /// Chunks actually embedded and stored.
    pub chunks_written: u32,
}

impl IndexingSummary {
    /// Fold one book's outcome into the summary.
    pub fn record(&mut self, outcome: IndexOutcome) {
        match outcome {
            IndexOutcome::Indexed { chunks } => {
                self.books_indexed += 1;
                self.chunks_written = self.chunks_written.saturating_add(chunks);
            }
            IndexOutcome::Skipped { .. } => self.books_skipped += 1,
            IndexOutcome::Empty => self.books_empty += 1,
        }
    }

    /// Books processed in total, whatever the outcome.
    pub fn books_processed(&self) -> u32 {
        self.books_indexed + self.books_skipped + self.books_empty
    }
}

// `IndexingProgress` was removed in S0. It was the DTO for streaming indexing
// progress to the application, and the application does not index — it installs a
// prebuilt read-only artifact. Nothing constructed it. A build tool that wants to
// report progress should define its own type next to the tool, rather than keep a
// public DTO here for a cancelled feature.

#[cfg(test)]
mod tests {
    use super::*;

    /// A Genesis line, with the facet set the lexical indexer would build for it.
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
            facets: vec![
                "/מקרא/תורה/בראשית".to_string(),
                "/author/משה רבנו".to_string(),
                "/era/תנך".to_string(),
                "/base".to_string(),
            ],
        }
    }

    fn facet_filter(paths: &[&str]) -> SearchFilters {
        SearchFilters {
            facets: Some(paths.iter().map(|p| p.to_string()).collect()),
            ..Default::default()
        }
    }

    // ── the content fingerprint contract ──

    /// The lexical engine records `contentHash = 0` for PDFs, because their
    /// extracted text is not in the library database. Treating that as a hash
    /// makes every PDF permanently "unchanged".
    #[test]
    fn a_zero_content_hash_is_not_a_hash() {
        assert_eq!(
            ContentFingerprint::from_lexical_hash(0),
            ContentFingerprint::Unverifiable
        );
        assert!(!ContentFingerprint::from_lexical_hash(0).is_verifiable());

        assert_eq!(
            ContentFingerprint::from_lexical_hash(42),
            ContentFingerprint::Canonical(NonZeroU64::new(42).unwrap())
        );
        assert!(ContentFingerprint::from_lexical_hash(42).is_verifiable());
    }

    #[test]
    fn a_zero_source_revision_cannot_become_canonical() {
        assert_eq!(
            ContentFingerprint::canonical(
                0,
                "מסכת ברכות",
                "/תלמוד/בבלי",
                &["/author/חז״ל".to_string()],
                true,
            ),
            ContentFingerprint::Unverifiable
        );
    }

    #[test]
    fn a_fingerprint_round_trips_through_its_raw_value() {
        for raw in [0u64, 1, 42, u64::MAX] {
            let fingerprint = ContentFingerprint::from_lexical_hash(raw);
            assert_eq!(fingerprint.as_raw(), raw);
            assert_eq!(
                ContentFingerprint::from_lexical_hash(fingerprint.as_raw()),
                fingerprint
            );
        }
    }

    /// The whole point of the canonical form: a metadata-only correction moves the
    /// fingerprint even though the file behind it did not change.
    #[test]
    fn metadata_changes_move_a_canonical_fingerprint() {
        let signature = 0xBEEF_CAFE;
        let facets = vec!["/author/רבי אחד".to_string(), "/era/ראשונים".to_string()];
        let baseline =
            ContentFingerprint::canonical(signature, "מסכת ברכות", "/תלמוד", &facets, true);

        assert!(baseline.proves_current());
        assert_ne!(baseline.as_raw(), 0, "zero is the no-fingerprint marker");

        // Each of these is a change a size/mtime signature cannot see, and each is
        // stored in every one of the book's vectors.
        let variants = [
            ContentFingerprint::canonical(signature, "ברכות", "/תלמוד", &facets, true),
            ContentFingerprint::canonical(signature, "מסכת ברכות", "/משנה", &facets, true),
            ContentFingerprint::canonical(
                signature,
                "מסכת ברכות",
                "/תלמוד",
                &["/author/רבי אחר".to_string(), "/era/ראשונים".to_string()],
                true,
            ),
            ContentFingerprint::canonical(signature, "מסכת ברכות", "/תלמוד", &facets, false),
            ContentFingerprint::canonical(signature + 1, "מסכת ברכות", "/תלמוד", &facets, true),
        ];
        for variant in variants {
            assert_ne!(
                variant, baseline,
                "a change stored in the vectors must change the fingerprint"
            );
        }

        // Same inputs, same value — otherwise every diff would report every book.
        assert_eq!(
            baseline,
            ContentFingerprint::canonical(signature, "מסכת ברכות", "/תלמוד", &facets, true)
        );
    }

    /// Facet order and repetition mean nothing to either index, so they must mean
    /// nothing to a fingerprint. Otherwise a caller that happens to list a book's
    /// authors in a different order forces it to be re-embedded.
    #[test]
    fn facet_order_and_repetition_do_not_change_a_fingerprint() {
        let ordered = vec![
            "/author/רבי אחד".to_string(),
            "/author/רבי שני".to_string(),
            "/era/ראשונים".to_string(),
        ];
        let shuffled_with_a_duplicate = vec![
            "/era/ראשונים".to_string(),
            "/author/רבי שני".to_string(),
            "/author/רבי אחד".to_string(),
            "/author/רבי שני".to_string(),
        ];

        assert_eq!(
            ContentFingerprint::canonical(7, "כותרת", "/מקרא", &ordered, false),
            ContentFingerprint::canonical(7, "כותרת", "/מקרא", &shuffled_with_a_duplicate, false)
        );

        let mut book = book_with_lines(&[(1, 1, "שורה ראשונה של הספר", 11)]);
        book.extra_facets = ordered;
        let mut permuted = book.clone();
        permuted.extra_facets = shuffled_with_a_duplicate;

        assert_eq!(
            book.line_fingerprint(),
            permuted.line_fingerprint(),
            "the line fingerprint must be facet-order-independent too"
        );
        assert_eq!(
            book.all_facets(),
            permuted.all_facets(),
            "and the facets stored on every vector are canonical"
        );
    }

    fn book_with_lines(lines: &[(u64, u64, &str, u64)]) -> BookForIndexing {
        BookForIndexing {
            source_book_key: "scan.pdf".to_string(),
            title: "ספר סרוק".to_string(),
            content_fingerprint: 0,
            is_pdf: true,
            topics: "/מקרא".to_string(),
            extra_facets: vec![],
            lines: lines
                .iter()
                .map(|&(line_id, section_id, text, line_hash)| BookLine {
                    line_id,
                    section_id,
                    text: text.to_string(),
                    line_hash,
                    reference: format!("עמוד {line_id}"),
                    segment: line_id,
                })
                .collect(),
        }
    }

    /// The fallback fingerprint for books the lexical index cannot vouch for.
    #[test]
    fn the_line_fingerprint_changes_with_any_line_change() {
        let base = book_with_lines(&[(1, 1, "שורה ראשונה", 111), (2, 1, "שורה שנייה", 222)]);
        let baseline = base.line_fingerprint();

        assert_eq!(
            baseline,
            book_with_lines(&[(1, 1, "שורה ראשונה", 111), (2, 1, "שורה שנייה", 222)])
                .line_fingerprint(),
            "identical content must produce an identical fingerprint"
        );

        let variants: Vec<(&str, BookForIndexing)> = vec![
            (
                "changed text",
                book_with_lines(&[(1, 1, "שורה ראשונה", 111), (2, 1, "שורה אחרת", 222)]),
            ),
            (
                "removed line",
                book_with_lines(&[(1, 1, "שורה ראשונה", 111)]),
            ),
            (
                "added line",
                book_with_lines(&[
                    (1, 1, "שורה ראשונה", 111),
                    (2, 1, "שורה שנייה", 222),
                    (3, 1, "שורה שלישית", 333),
                ]),
            ),
            (
                "reordered lines",
                book_with_lines(&[(2, 1, "שורה שנייה", 222), (1, 1, "שורה ראשונה", 111)]),
            ),
            (
                "changed line id",
                book_with_lines(&[(9, 1, "שורה ראשונה", 111), (2, 1, "שורה שנייה", 222)]),
            ),
            (
                "changed section",
                book_with_lines(&[(1, 7, "שורה ראשונה", 111), (2, 1, "שורה שנייה", 222)]),
            ),
        ];

        for (name, variant) in variants {
            assert_ne!(
                baseline,
                variant.line_fingerprint(),
                "{name} must change the fingerprint"
            );
        }
    }

    /// Lines the lexical engine considers too short to deduplicate carry
    /// `line_hash = 0`, so the text itself has to be part of the fingerprint.
    /// The fingerprint gates the skip, so anything it omits can change behind the
    /// index's back. Every one of these values is persisted in `VectorMetadata`
    /// and drives filtering or display.
    #[test]
    fn the_fingerprint_covers_every_field_stored_in_a_vector_record() {
        let base = book_with_lines(&[(1, 1, "שורה ראשונה", 111)]);
        let baseline = base.line_fingerprint();

        let mut retitled = base.clone();
        retitled.title = "כותרת מתוקנת".to_string();

        let mut recategorized = base.clone();
        recategorized.topics = "/תלמוד".to_string();

        let mut with_author = base.clone();
        with_author.extra_facets = vec!["/author/מחבר".to_string()];

        let mut second_author = base.clone();
        second_author.extra_facets =
            vec!["/author/מחבר".to_string(), "/author/מחבר נוסף".to_string()];

        let mut not_pdf = base.clone();
        not_pdf.is_pdf = false;

        let mut new_reference = base.clone();
        new_reference.lines[0].reference = "הפניה מתוקנת".to_string();

        for (name, variant) in [
            ("title", retitled),
            ("category path", recategorized),
            ("added author facet", with_author.clone()),
            ("reference", new_reference),
            ("is_pdf", not_pdf),
        ] {
            assert_ne!(
                baseline,
                variant.line_fingerprint(),
                "changing the {name} must change the fingerprint — it is stored per vector"
            );
        }

        // And a second author is distinguishable from one.
        assert_ne!(
            with_author.line_fingerprint(),
            second_author.line_fingerprint(),
            "adding an author must change the fingerprint"
        );
    }

    /// Field boundaries must not be shiftable: two different books whose
    /// concatenated bytes coincide have to hash differently.
    #[test]
    fn adjacent_fields_cannot_be_confused_with_each_other() {
        let mut a = book_with_lines(&[(1, 1, "שורה", 1)]);
        a.title = "אב".to_string();
        a.topics = "/ג".to_string();

        let mut b = a.clone();
        b.title = "א".to_string();
        b.topics = "ב/ג".to_string();

        assert_ne!(a.line_fingerprint(), b.line_fingerprint());
    }

    #[test]
    fn the_line_fingerprint_covers_text_even_without_a_line_hash() {
        let a = book_with_lines(&[(1, 1, "אלף", 0)]);
        let b = book_with_lines(&[(1, 1, "בית", 0)]);
        assert_ne!(a.line_fingerprint(), b.line_fingerprint());
    }

    #[test]
    fn an_empty_book_has_a_stable_fingerprint() {
        assert_eq!(
            book_with_lines(&[]).line_fingerprint(),
            book_with_lines(&[]).line_fingerprint()
        );
        assert_ne!(
            book_with_lines(&[]).line_fingerprint(),
            book_with_lines(&[(1, 1, "שורה", 1)]).line_fingerprint()
        );
    }

    // ── the facet model ──

    /// A book can have several authors: the lexical indexer emits one
    /// `/author/...` facet per author. A single-valued field cannot hold that,
    /// and a filter on the author that did not fit would return nothing.
    #[test]
    fn a_book_can_carry_several_authors() {
        let book = BookForIndexing {
            source_book_key: "commentary.txt".to_string(),
            title: "פירוש".to_string(),
            content_fingerprint: 1,
            is_pdf: false,
            topics: "/מפרשים".to_string(),
            extra_facets: vec![
                "/author/רבי אחד".to_string(),
                "/author/רבי שני".to_string(),
                "/era/ראשונים".to_string(),
            ],
            lines: vec![],
        };

        let facets = book.all_facets();
        assert_eq!(facets.len(), 4, "the category path plus three dimensions");
        assert!(facets.contains(&"/מפרשים".to_string()));

        let meta = VectorMetadata { facets, ..meta() };

        // Either author selects the book.
        for author in ["/author/רבי אחד", "/author/רבי שני"] {
            assert!(
                facet_filter(&[author]).matches(&meta),
                "filtering by {author} must match"
            );
        }
        assert!(!facet_filter(&["/author/רבי שלישי"]).matches(&meta));
    }

    #[test]
    fn a_book_without_a_category_path_still_carries_its_dimensions() {
        let book = BookForIndexing {
            source_book_key: "userbook.txt".to_string(),
            title: "ספר משתמש".to_string(),
            content_fingerprint: 1,
            is_pdf: false,
            topics: "   ".to_string(),
            extra_facets: vec!["/author/מחבר".to_string()],
            lines: vec![],
        };
        assert_eq!(book.all_facets(), vec!["/author/מחבר".to_string()]);
    }

    #[test]
    fn facet_dimensions_are_derived_the_way_the_lexical_engine_derives_them() {
        // Index 0 is the category tree.
        assert_eq!(facet_dimension("/מקרא/תורה"), 0);
        assert_eq!(facet_dimension("מקרא/תורה"), 0);
        // The reserved roots each get their own group.
        assert_eq!(facet_dimension("/author/רש״י"), 1);
        assert_eq!(facet_dimension("/era/ראשונים"), 2);
        assert_eq!(facet_dimension("/base"), 3);
        // A Hebrew category that merely resembles one is still a category.
        assert_eq!(facet_dimension("/authors/רש״י"), 0);
    }

    // ── filter semantics ──

    #[test]
    fn no_filters_accepts_everything() {
        let filters = SearchFilters::default();
        assert!(filters.is_empty());
        assert!(filters.compile().is_none());
        assert!(filters.matches(&meta()));
    }

    /// The bug this pins down: `is_empty()` said an empty list is not a filter
    /// while matching treated it as "must be one of zero values" and dropped
    /// every candidate.
    #[test]
    fn empty_lists_do_not_filter_anything_out() {
        let filters = SearchFilters {
            book_paths: Some(vec![]),
            facets: Some(vec![]),
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
    fn is_empty_agrees_with_matches_for_every_dimension() {
        let cases: Vec<(&str, SearchFilters)> = vec![
            (
                "book_paths",
                SearchFilters {
                    book_paths: Some(vec!["no/such/book.txt".to_string()]),
                    ..Default::default()
                },
            ),
            ("categories", facet_filter(&["/תלמוד"])),
            ("authors", facet_filter(&["/author/מישהו אחר"])),
            ("eras", facet_filter(&["/era/אחרונים"])),
        ];

        for (name, filters) in cases {
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
    fn facet_filters_match_hierarchically_like_tantivy_facets() {
        for filter in ["/מקרא", "/מקרא/תורה", "/מקרא/תורה/בראשית", "/מקרא/"]
        {
            assert!(
                facet_filter(&[filter]).matches(&meta()),
                "filter {filter} should match"
            );
        }

        for filter in ["/תלמוד", "/מקרא/נביאים", "/מקר", "/מקרא תורה"] {
            assert!(
                !facet_filter(&[filter]).matches(&meta()),
                "filter {filter} should not match"
            );
        }
    }

    #[test]
    fn paths_in_the_same_dimension_are_or_ed() {
        assert!(facet_filter(&["/תלמוד", "/מקרא"]).matches(&meta()));
        assert!(facet_filter(&["/author/אחר", "/author/משה רבנו"]).matches(&meta()));
    }

    /// "ראשונים AND מסכת ברכות" has to work while two authors stay "either of
    /// them" — the same rule the lexical facet filter applies.
    #[test]
    fn different_dimensions_are_and_ed() {
        assert!(facet_filter(&["/מקרא", "/era/תנך"]).matches(&meta()));
        assert!(!facet_filter(&["/מקרא", "/era/ראשונים"]).matches(&meta()));
        assert!(!facet_filter(&["/תלמוד", "/era/תנך"]).matches(&meta()));

        // Two authors OR-ed, AND-ed against a category: satisfied.
        assert!(facet_filter(&["/author/אחר", "/author/משה רבנו", "/מקרא"]).matches(&meta()));
    }

    #[test]
    fn a_candidate_missing_a_filtered_dimension_is_excluded() {
        let mut without_author = meta();
        without_author.facets.retain(|f| !f.starts_with("/author/"));
        assert!(!facet_filter(&["/author/משה רבנו"]).matches(&without_author));
    }

    /// `/base` is a bare marker in the lexical indexer, not a path with a value.
    #[test]
    fn the_base_marker_filters_as_its_own_dimension() {
        assert!(facet_filter(&["/base"]).matches(&meta()));

        let mut not_base = meta();
        not_base.facets.retain(|f| f != "/base");
        assert!(!facet_filter(&["/base"]).matches(&not_base));
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
    fn compiling_once_gives_the_same_verdict_as_matching_directly() {
        let filters = facet_filter(&["/מקרא", "/author/משה רבנו"]);
        let compiled = filters.compile().expect("filters are active");

        let mut other = meta();
        other.facets = vec!["/תלמוד".to_string()];

        for candidate in [meta(), other] {
            assert_eq!(compiled.matches(&candidate), filters.matches(&candidate));
        }
    }

    #[test]
    fn index_diff_up_to_date_contract() {
        let clean = IndexDiff::up_to_date();
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
