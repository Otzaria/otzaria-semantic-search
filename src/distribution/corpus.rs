//! The lexical index a packer joins its vectors against, as a port.
//!
//! A semantic result is a `line_id` and nothing else, so an artifact is only worth
//! anything if every one of its ids names a real document in the corpus the application
//! will hydrate from. That check needs the corpus, and the corpus is Tantivy — which this
//! crate does not depend on and must not: the index lives in `otzaria_search_engine`,
//! together with the schema and the id scheme that produced the ids in the first place.
//!
//! So the packer takes a [`CorpusIndex`] rather than a directory. Two consequences, and
//! both are the point:
//!
//! * **The corpus identity comes from the corpus.** `corpus_id`, `library_version` and
//!   the two scheme versions are read off the index that is actually open, never typed
//!   into a configuration file beside the vectors. An artifact cannot be labelled for a
//!   catalogue it was not built from.
//! * **Every field of a record comes from the corpus too** — except the one only the
//!   producer can know, which is a digest of the text it embedded. The title, reference,
//!   section, facets and the rest are whatever the corpus says today, so there is no
//!   second description of a book to drift from the first.
//! * **The corpus says which vectors there should be.** [`CorpusIndex::expected_line_ids`]
//!   is what makes "complete artifact" a checkable claim rather than a hope: without it a
//!   packer can only vouch for the vectors it was handed, and one good vector would pack
//!   into a valid-looking artifact for a six-million-line library. It takes the model
//!   identity, because the set is a function of the recipe the artifact declares — "which
//!   lines exist" and "which lines get embedded" are different questions, and only the
//!   second one is coverage.
//!
//! [`JsonlCorpus`] is the implementation this crate can offer: a transcription of the
//! index into two files. It is what makes the CLI usable without Tantivy, and what the
//! tests drive. The implementation that reads a live Tantivy index belongs to
//! `otzaria_search_engine` (S4b/S5) — and lands on the same [`crate::distribution::packer`]
//! behind it.

use crate::errors::PackError;
use crate::semantic::versioning::{CorpusIdentity, ModelIdentity};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// What the corpus holds for one line.
///
/// These are exactly the fields a stored vector carries, minus the three the semantic
/// side derives (`semantic_id`, `source_doc_key`, `chunk_hash`), plus the `text` the
/// vector was supposed to have been built from. Nothing here is optional: a field the
/// corpus cannot answer is a field the artifact would have to invent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorpusLine {
    /// The book's key — its file path, as the lexical index stores it.
    pub source_book_key: String,
    pub title: String,
    pub reference: String,
    pub section_id: u64,
    pub segment: u64,
    pub is_pdf: bool,
    /// The lexical engine's dedup hash for the line. `0` for lines it considers too
    /// short to deduplicate, which is why it cannot stand in for the text.
    pub line_hash: u64,
    /// The book's content fingerprint, `0` when the lexical index has none (a PDF).
    pub content_hash: u64,
    /// Every facet path describing the book, categories and dimensions together — the
    /// same flat list the lexical engine indexes.
    pub facets: Vec<String>,
    /// The line's text, exactly as the corpus stores it.
    ///
    /// Not stored in the artifact: it is what the vector was built from, and comparing
    /// it against what the producer claims is what proves a vector belongs to this line
    /// and not to its neighbour.
    pub text: String,
}

/// The corpus a set of vectors claims to describe.
///
/// Implemented by whoever has the lexical index open. Every method may fail, because a
/// real index can: a read error and "there is no such line" are different answers and
/// stay different — the first is a broken build input, the second is a broken pairing.
pub trait CorpusIndex {
    /// Identity of the corpus, as the index reports it. This is what the artifact
    /// declares and what the runtime will compare against; a packer never composes it.
    fn identity(&self) -> Result<CorpusIdentity, PackError>;

    /// Every `line_id` an artifact built under `model` must carry a vector for — no more
    /// and no fewer.
    ///
    /// **This is the completeness contract, and there is no way to opt out of it.**
    /// Without it a packer can only check the vectors it was given, so one good vector
    /// out of six million would produce a perfectly valid "official artifact": the
    /// counts, the checksums and the identity would all agree, and the library would be
    /// missing from itself. The comparison runs in both directions, because an extra
    /// vector is its own fault — a line the recipe skips acquiring one means the artifact
    /// was built by a recipe other than the one it declares.
    ///
    /// **It is not "every document in the index".** The embedding recipe decides what gets
    /// a vector: a line too short to carry meaning is skipped, and an artifact is not
    /// incomplete for skipping it. That is why `model` is a parameter — the set is a
    /// function of `embedding_text_version`, `chunking_identity` and `max_tokens`, which
    /// are exactly what the artifact declares about itself, and an implementation that
    /// answered for some other recipe would be certifying coverage of an artifact nobody
    /// built.
    ///
    /// **Do not derive it from the vectors that were produced.** A batch that died halfway
    /// or a line the backend silently dropped would vanish from both sides at once, and the
    /// check would confirm itself. The set has to be decided before inference, from the
    /// corpus and the recipe.
    ///
    /// **`model.chunking_identity` is an opaque hash**, so an implementation cannot recover
    /// the recipe from it. It has to hold the real
    /// [`ChunkerConfig`](crate::semantic::chunker::ChunkerConfig) and check
    /// `config.identity() == model.chunking_identity` before answering; otherwise the
    /// parameter is in the signature and binds nothing.
    fn expected_line_ids(&self, model: &ModelIdentity) -> Result<BTreeSet<u64>, PackError>;

    /// The line `line_id` names, or `None` when the corpus holds no live document with
    /// that id.
    fn line(&self, line_id: u64) -> Result<Option<CorpusLine>, PackError>;
}

/// One line of the corpus lines file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpusLineRecord {
    pub line_id: u64,
    #[serde(flatten)]
    pub line: CorpusLine,
}

/// A corpus transcribed into two files: an identity document and one JSON object per
/// line.
///
/// ```text
/// corpus-identity.json   the CorpusIdentity the lexical index reports
/// corpus-lines.jsonl     one CorpusLineRecord per line, in any order
/// ```
///
/// **What this is for.** It lets a packer run without linking Tantivy — the CLI in this
/// crate, and every test here. It is a *transcription*, so it is exactly as trustworthy
/// as whatever wrote it; the authoritative join is the one an implementation over the
/// live index performs. That is why the trait exists and why this type is not the only
/// way in.
///
/// **What it cannot express.** One map answers both "what is line N" and "which lines get
/// vectors", so here those two questions can never disagree: a line that exists but is not
/// embedded has no representation. A live index has both — it holds every document while
/// the recipe embeds some of them — so that half of the coverage check is exercised
/// against a corpus that can tell them apart, and enforced for the implementation that
/// will need to.
///
/// **What it costs.** Every line is held in memory, text included. At library scale that
/// is not affordable — but neither is the payload writer this feeds, which holds every
/// vector in RAM until it commits (see [`zevc_store`](crate::semantic::zevc_store)).
/// Both are the same S2b measurement, and neither is hidden behind an interface that
/// implies otherwise.
#[derive(Debug)]
pub struct JsonlCorpus {
    identity: CorpusIdentity,
    lines: HashMap<u64, CorpusLine>,
}

impl JsonlCorpus {
    /// Read both files, refusing a corpus that describes one line twice.
    ///
    /// A duplicate id is refused rather than resolved: the two records would disagree
    /// about which book the line belongs to, and picking one would make the artifact
    /// depend on file order.
    pub fn load(identity_path: &Path, lines_path: &Path) -> Result<Self, PackError> {
        let identity_json =
            std::fs::read_to_string(identity_path).map_err(|source| PackError::Io {
                context: format!("reading {}", identity_path.display()),
                source,
            })?;
        let identity: CorpusIdentity =
            serde_json::from_str(&identity_json).map_err(|error| PackError::Corpus {
                reason: format!(
                    "{} is not a corpus identity: {error}",
                    identity_path.display()
                ),
            })?;

        let file = File::open(lines_path).map_err(|source| PackError::Io {
            context: format!("reading {}", lines_path.display()),
            source,
        })?;

        let mut lines = HashMap::new();
        for (index, text) in BufReader::new(file).lines().enumerate() {
            let text = text.map_err(|source| PackError::Io {
                context: format!("reading {} line {}", lines_path.display(), index + 1),
                source,
            })?;
            if text.trim().is_empty() {
                continue;
            }
            let record: CorpusLineRecord =
                serde_json::from_str(&text).map_err(|error| PackError::Corpus {
                    reason: format!(
                        "{} line {} is not a corpus line: {error}",
                        lines_path.display(),
                        index + 1
                    ),
                })?;
            if lines.insert(record.line_id, record.line).is_some() {
                return Err(PackError::Corpus {
                    reason: format!(
                        "{} describes line_id {} more than once",
                        lines_path.display(),
                        record.line_id
                    ),
                });
            }
        }

        if lines.is_empty() {
            return Err(PackError::Corpus {
                reason: format!("{} holds no lines", lines_path.display()),
            });
        }

        Ok(Self { identity, lines })
    }

    /// How many lines were transcribed. Reported by the CLI so a corpus file that is
    /// obviously the wrong one is visible before anything is written.
    pub fn len(&self) -> usize {
        self.lines.len()
    }

    /// Never true for a loaded corpus — [`Self::load`] refuses an empty one. Present
    /// because `len` without it is a clippy lint, and answering it honestly is cheaper
    /// than an allow.
    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }
}

impl CorpusIndex for JsonlCorpus {
    fn identity(&self) -> Result<CorpusIdentity, PackError> {
        Ok(self.identity.clone())
    }

    /// Every line in the file. The file is therefore the coverage contract as well as the
    /// metadata source: whoever exported it decided which lines get vectors, which is the
    /// same decision the embedding recipe makes.
    ///
    /// `model` is ignored, and that is this type's limit rather than an oversight. A
    /// transcription records a recipe that was **already applied**; nothing in the file can
    /// re-derive the set for a different one, so it cannot notice an export made for
    /// `embedding_text_version` 1 being packed under version 2. An implementation over a
    /// live index re-runs the recipe and can, which is one more reason the trait exists.
    fn expected_line_ids(&self, _model: &ModelIdentity) -> Result<BTreeSet<u64>, PackError> {
        Ok(self.lines.keys().copied().collect())
    }

    fn line(&self, line_id: u64) -> Result<Option<CorpusLine>, PackError> {
        Ok(self.lines.get(&line_id).cloned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    struct TempDir(PathBuf);
    impl TempDir {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "otzaria_test_corpus_{name}_{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn identity() -> CorpusIdentity {
        CorpusIdentity {
            corpus_id: "7a".repeat(32),
            library_version: "otzaria-library-2026-08".to_string(),
            tantivy_schema_version: 3,
            document_id_scheme_version: 1,
        }
    }

    fn line(book: &str, text: &str) -> CorpusLine {
        CorpusLine {
            source_book_key: book.to_string(),
            title: "ספר בדיקה".to_string(),
            reference: "פרק א".to_string(),
            section_id: 1,
            segment: 0,
            is_pdf: false,
            line_hash: 11,
            content_hash: 22,
            facets: vec!["/מקרא/תורה".to_string()],
            text: text.to_string(),
        }
    }

    /// Write both files and return their paths.
    fn write_corpus(dir: &TempDir, records: &[CorpusLineRecord]) -> (PathBuf, PathBuf) {
        let identity_path = dir.path().join("corpus-identity.json");
        let lines_path = dir.path().join("corpus-lines.jsonl");
        std::fs::write(
            &identity_path,
            serde_json::to_vec_pretty(&identity()).unwrap(),
        )
        .unwrap();
        let body: String = records
            .iter()
            .map(|record| format!("{}\n", serde_json::to_string(record).unwrap()))
            .collect();
        std::fs::write(&lines_path, body).unwrap();
        (identity_path, lines_path)
    }

    fn record(line_id: u64, book: &str, text: &str) -> CorpusLineRecord {
        CorpusLineRecord {
            line_id,
            line: line(book, text),
        }
    }

    /// The transcription has to survive a round trip through JSON exactly, or the fields
    /// a record is checked against would not be the fields the corpus declared.
    #[test]
    fn a_transcribed_corpus_answers_with_what_was_written() {
        let dir = TempDir::new("round_trip");
        let (identity_path, lines_path) = write_corpus(
            &dir,
            &[
                record(4_294_967_297, "genesis.txt", "בראשית ברא"),
                record(8_589_934_593, "berachot.txt", "מאימתי קורין"),
            ],
        );

        let corpus = JsonlCorpus::load(&identity_path, &lines_path).unwrap();
        assert_eq!(corpus.len(), 2);
        assert!(!corpus.is_empty());
        assert_eq!(corpus.identity().unwrap(), identity());
        assert_eq!(
            corpus.line(4_294_967_297).unwrap().unwrap(),
            line("genesis.txt", "בראשית ברא")
        );
        assert!(corpus.line(1).unwrap().is_none());
    }

    /// Two records for one id would make the artifact depend on file order.
    #[test]
    fn a_corpus_that_describes_a_line_twice_is_refused() {
        let dir = TempDir::new("duplicate");
        let (identity_path, lines_path) = write_corpus(
            &dir,
            &[
                record(7, "genesis.txt", "בראשית ברא"),
                record(7, "berachot.txt", "מאימתי קורין"),
            ],
        );

        match JsonlCorpus::load(&identity_path, &lines_path) {
            Err(PackError::Corpus { reason }) => {
                assert!(reason.contains("more than once"), "{reason}")
            }
            other => panic!("a duplicated line must be refused, got {other:?}"),
        }
    }

    #[test]
    fn an_empty_or_unreadable_corpus_is_refused_rather_than_treated_as_no_lines() {
        let dir = TempDir::new("empty");
        let (identity_path, lines_path) = write_corpus(&dir, &[]);
        match JsonlCorpus::load(&identity_path, &lines_path) {
            Err(PackError::Corpus { reason }) => assert!(reason.contains("no lines"), "{reason}"),
            other => panic!("an empty corpus must be refused, got {other:?}"),
        }

        std::fs::write(&lines_path, b"{ not json\n").unwrap();
        assert!(matches!(
            JsonlCorpus::load(&identity_path, &lines_path),
            Err(PackError::Corpus { .. })
        ));

        std::fs::write(&identity_path, b"{}").unwrap();
        assert!(matches!(
            JsonlCorpus::load(&identity_path, &lines_path),
            Err(PackError::Corpus { .. })
        ));

        assert!(matches!(
            JsonlCorpus::load(&dir.path().join("absent.json"), &lines_path),
            Err(PackError::Io { .. })
        ));
    }
}
