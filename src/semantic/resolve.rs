//! What a scan of the vector store returns, and what ties it to the lines a search shows.
//!
//! A stored vector knows the text it was embedded from — its key — and where that text
//! was when the vectors were built: a list of records, each a book and the ordinal of a
//! line in it. It does not know the line's id today, its section, its facets or whether it
//! still exists; the application's index does. So a scan returns [`VectorHit`]s, and the
//! application turns them into live lines.

use crate::semantic::chunk_key::ChunkKey;
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
