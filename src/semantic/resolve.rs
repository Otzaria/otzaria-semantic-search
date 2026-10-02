//! What a scan of the vector store returns, and what ties it to the lines a search shows.
//!
//! A stored vector knows the text it was embedded from — its key — and where that text
//! was when the vectors were built: a list of records, each a book and the ordinal of a
//! line in it. It does not know the line's id today, its section, its facets or whether it
//! still exists; the application's index does. So a scan returns [`VectorHit`]s, and a
//! [`CandidateResolver`] — the application's, over its live index — turns them into
//! [`ResolvedLine`]s: the lines that hold each hit's key now, whatever their ids, wherever
//! they moved. A hit whose key no live line holds resolves to nothing, and is dropped.
//!
//! The coordinator fuses on what the resolver returns, so every id, section, line hash and
//! facet a result carries is the live index's, never the vectors'.

use crate::cancellation::CancellationToken;
use crate::semantic::chunk_key::ChunkKey;
use crate::semantic::types::SearchFilters;
use std::collections::HashSet;
use std::sync::Arc;

/// One place a hit's key was found when the vectors were built.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RecordRef {
    /// The book's stable key — its `filePath` in the application's index, `id:<book.id>`
    /// for a book of the official library.
    pub book: Arc<str>,
    /// The line's ordinal in that book when the vectors were built: where to look first,
    /// not where the line must be.
    pub hint: u32,
}

/// One vector a scan returned.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorHit {
    /// Cosine similarity to the query, as the store's codec computes it.
    pub score: f32,
    /// The key of the text the vector was embedded from.
    pub key: ChunkKey,
    /// Where that text was: the books admitted by the search's filter first, then the
    /// rest, at most [`MAX_RECORDS_PER_HIT`] in all.
    pub records: Vec<RecordRef>,
    /// Which segment of the set holds the vector, and at which slot. Stable for one
    /// generation of the set; for diagnostics, not for identity.
    pub seg: u16,
    pub slot: u32,
}

/// The most records one hit carries. A boilerplate line can occur thousands of times;
/// lexical search still finds every one, and a semantic result needs a handful.
pub const MAX_RECORDS_PER_HIT: usize = 32;

/// The books a search may return lines from, by stable key.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BookSet(pub HashSet<Box<str>>);

impl BookSet {
    pub fn contains(&self, book: &str) -> bool {
        self.0.contains(book)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl<S: Into<Box<str>>> FromIterator<S> for BookSet {
    fn from_iter<I: IntoIterator<Item = S>>(books: I) -> Self {
        Self(books.into_iter().map(Into::into).collect())
    }
}

/// Why the application's index could not answer a question about live lines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolveError {
    /// The caller's token was cancelled while the index was being read.
    Cancelled,
    /// The index could not be read.
    Index { reason: String },
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cancelled => f.write_str("cancelled"),
            Self::Index { reason } => write!(f, "the index could not be read: {reason}"),
        }
    }
}

impl std::error::Error for ResolveError {}

/// The application's index, as compaction asks it: which key each live line of a book
/// holds today.
pub trait LiveKeySource {
    /// The library version the index holds. Records are re-anchored only on an index of the
    /// set's own version: against another, a missing key is a library change, not a stale
    /// record.
    fn library_version(&self) -> u32;

    /// Fill `out` with `(ordinal, chunk key column value)` for every live line of `book`
    /// — [`ChunkKey::column_value`], `0` for a line that is not embedded — and return
    /// `false` when the index holds no such book, whose records are then kept as they are.
    fn book_keys(&self, book: &str, out: &mut Vec<(u32, u64)>) -> Result<bool, ResolveError>;
}

/// One live line a hit resolved to, described by the live index.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedLine {
    /// Which of the hits handed to [`CandidateResolver::resolve`] it came from.
    pub hit: u32,
    /// The document id the index gives the line now.
    pub line_id: u64,
    /// The book's key, as the index stores it (`filePath`).
    pub file_path: String,
    pub section_id: u64,
    pub line_hash: u64,
    /// The line's ordinal in its book.
    pub segment: u64,
    pub is_pdf: bool,
    /// Every facet path describing the book — see
    /// [`BookForIndexing::all_facets`](crate::semantic::types::BookForIndexing::all_facets).
    pub facets: Arc<[String]>,
    pub title: String,
    /// Empty when the host fills it in when it hydrates the line.
    pub reference: String,
}

/// The application's live index, as a search asks it: which books a filter admits, and
/// which live lines hold the keys a scan returned.
///
/// Implemented by the host — `otzaria_search_engine` — over the index it has open. Every
/// method may look at `cancel` and return [`ResolveError::Cancelled`].
pub trait CandidateResolver: Send + Sync {
    /// The generation of what the resolver reads: it changes whenever the answer to the
    /// same question could. Folded into the query cache's key, so a cached result never
    /// carries ids from before an index commit.
    fn generation(&self) -> u64;

    /// The books `filters` admits, by stable key; `None` when they admit every book.
    fn admissible_books(
        &self,
        filters: Option<&SearchFilters>,
    ) -> Result<Option<BookSet>, ResolveError>;

    /// The live lines that hold each hit's key, in books `filters` admits — at most a few
    /// per hit, the best-placed first. A line two hits resolve to is returned once, for the
    /// better-scored hit; a hit that resolves nowhere contributes nothing.
    fn resolve(
        &self,
        hits: &[VectorHit],
        filters: Option<&SearchFilters>,
        cancel: &CancellationToken,
    ) -> Result<Vec<ResolvedLine>, ResolveError>;
}

/// The resolver of a search with no live index behind it: it admits every book and
/// resolves nothing, so an official vector set searched through it contributes nothing —
/// and says why. The self-built path never asks it anything.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoResolver;

impl CandidateResolver for NoResolver {
    fn generation(&self) -> u64 {
        0
    }

    fn admissible_books(
        &self,
        _filters: Option<&SearchFilters>,
    ) -> Result<Option<BookSet>, ResolveError> {
        Ok(None)
    }

    fn resolve(
        &self,
        _hits: &[VectorHit],
        _filters: Option<&SearchFilters>,
        _cancel: &CancellationToken,
    ) -> Result<Vec<ResolvedLine>, ResolveError> {
        Err(ResolveError::Index {
            reason: "an official vector set needs the host's resolver to tie its hits to live \
                     lines, and this search was given none"
                .to_string(),
        })
    }
}
