//! The plan of a vector build: which line of which book holds which key, and the texts
//! that still need a vector.
//!
//! The planner — `export_semantic_plan` in `otzaria_search_engine`, over the release's
//! index, or [`plan_from_corpus`] here, over a transcription — writes one directory:
//!
//! ```text
//! records.bin          b"OXVREC1\n", u64 count, then {book u32, ordinal u32, sha256 [32]}
//!                      per embedded line, strictly ascending by (book, ordinal)
//! books.json           the books' stable keys, a JSON array in strictly ascending byte
//!                      order; a record's book is an index into it
//! embed.jsonl          the texts no warehouse vector exists for, one per key, in order of
//!                      first appearance — see shard
//! embed-manifest.json  what embed.jsonl is: the model, the recipe, the passage package
//! tombstones.bin       keys the previous release held and this one does not — see ledger
//! plan-manifest.json   the identity, the release, the counts and every file's digest
//! ```
//!
//! Every multi-byte integer is little-endian. `sha256` is the SHA-256 of the exact string
//! the model is given — prefix, context and cap included — and its first 16 bytes are the
//! line's [`ChunkKey`]. A line the recipe does not embed has no record.

use crate::distribution::builder::{chunk_book_lines, ensure_recipe_matches};
use crate::distribution::corpus::CorpusBooks;
use crate::distribution::files::{
    hex, io_error, malformed, partial_path, read_json, write_json, FileDigest, KeySet,
};
use crate::distribution::ledger::{split, Ledger, SplitCounts};
use crate::errors::PackError;
use crate::semantic::chunk_key::ChunkKey;
use crate::semantic::chunker::{Chunker, ChunkerConfig};
use crate::semantic::official_index::readable_store_identity;
use crate::semantic::recipe::EmbeddingRecipe;
use crate::semantic::versioning::{IndexVersion, ModelIdentity, ModelPackage};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub const RECORDS_FILE: &str = "records.bin";
pub const BOOKS_FILE: &str = "books.json";
pub const EMBED_PLAN_FILE: &str = "embed.jsonl";
pub const EMBED_MANIFEST_FILE: &str = "embed-manifest.json";
pub const TOMBSTONES_FILE: &str = "tombstones.bin";
pub const PLAN_MANIFEST_FILE: &str = "plan-manifest.json";

/// `format` of a plan manifest.
pub const PLAN_FORMAT: &str = "otzaria-vector-plan";
pub const PLAN_FORMAT_VERSION: u32 = 1;
/// `format` of an embed manifest.
pub const EMBED_PLAN_FORMAT: &str = "otzaria-embed-plan";
pub const EMBED_PLAN_VERSION: u32 = 2;

const RECORDS_MAGIC: &[u8; 8] = b"OXVREC1\n";
const RECORDS_HEADER: usize = 16;
const RECORD_LEN: usize = 40;

/// What a planner asks of the vectors already embedded — the warehouse's: whether one is
/// held for the text whose SHA-256 is `sha256`.
pub trait HeldVectors {
    fn holds(&self, sha256: &[u8; 32]) -> bool;
}

/// One embedded line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanRecord {
    /// Index into `books.json`.
    pub book: u32,
    /// The line's position in its book: the hint a segment records.
    pub ordinal: u32,
    /// SHA-256 of the text the model is given.
    pub sha256: [u8; 32],
}

impl PlanRecord {
    pub fn key(&self) -> ChunkKey {
        ChunkKey(self.key_bytes())
    }

    pub fn key_bytes(&self) -> [u8; 16] {
        self.sha256[..16].try_into().expect("16 of 32")
    }

    fn encode(&self) -> [u8; RECORD_LEN] {
        let mut out = [0u8; RECORD_LEN];
        out[..4].copy_from_slice(&self.book.to_le_bytes());
        out[4..8].copy_from_slice(&self.ordinal.to_le_bytes());
        out[8..].copy_from_slice(&self.sha256);
        out
    }

    fn decode(bytes: &[u8; RECORD_LEN]) -> Self {
        Self {
            book: u32::from_le_bytes(bytes[..4].try_into().expect("4")),
            ordinal: u32::from_le_bytes(bytes[4..8].try_into().expect("4")),
            sha256: bytes[8..].try_into().expect("32"),
        }
    }
}

/// Writes `records.bin`: under a temporary name until [`Self::finish`], so a planner that
/// dies leaves nothing a later step could mistake for a plan.
pub struct RecordsWriter {
    out: BufWriter<File>,
    path: PathBuf,
    partial: PathBuf,
    count: u64,
    last: Option<(u32, u32)>,
}

impl RecordsWriter {
    pub fn create(path: &Path) -> Result<Self, PackError> {
        let partial = partial_path(path);
        let mut out = BufWriter::with_capacity(
            4 << 20,
            File::create(&partial).map_err(io_error(format!("creating {}", partial.display())))?,
        );
        out.write_all(RECORDS_MAGIC)
            .and_then(|()| out.write_all(&0u64.to_le_bytes()))
            .map_err(io_error(format!("writing {}", partial.display())))?;
        Ok(Self {
            out,
            path: path.to_path_buf(),
            partial,
            count: 0,
            last: None,
        })
    }

    /// The next record, strictly after the previous one by (book, ordinal).
    pub fn push(&mut self, record: PlanRecord) -> Result<(), PackError> {
        let at = (record.book, record.ordinal);
        if self.last.is_some_and(|last| at <= last) {
            return Err(malformed(format!(
                "records must ascend strictly by (book, ordinal), and {at:?} follows {:?}",
                self.last.expect("checked")
            )));
        }
        self.last = Some(at);
        self.out
            .write_all(&record.encode())
            .map_err(io_error(format!("writing {}", self.partial.display())))?;
        self.count += 1;
        Ok(())
    }

    pub fn count(&self) -> u64 {
        self.count
    }

    /// Write the count, flush, and rename into place.
    pub fn finish(self) -> Result<u64, PackError> {
        let context = format!("finishing {}", self.partial.display());
        let mut file = self.out.into_inner().map_err(|error| PackError::Io {
            context: context.clone(),
            source: error.into_error(),
        })?;
        (|| {
            file.seek(SeekFrom::Start(8))?;
            file.write_all(&self.count.to_le_bytes())?;
            file.sync_all()?;
            drop(file);
            std::fs::rename(&self.partial, &self.path)
        })()
        .map_err(io_error(context))?;
        Ok(self.count)
    }
}

/// `records.bin`, mapped and checked.
pub struct PlanRecords {
    map: Mmap,
    count: u64,
}

impl PlanRecords {
    /// Map `path` and check it: the magic, a length that is the count's, and strictly
    /// ascending records.
    pub fn open(path: &Path) -> Result<Self, PackError> {
        let file = File::open(path).map_err(io_error(format!("opening {}", path.display())))?;
        // SAFETY: the file is opened read-only and never written while mapped by this
        // process; a plan is written once, renamed into place, and only read after.
        let map =
            unsafe { Mmap::map(&file) }.map_err(io_error(format!("mapping {}", path.display())))?;
        if map.len() < RECORDS_HEADER || &map[..8] != RECORDS_MAGIC {
            return Err(malformed(format!(
                "{} is not a records file",
                path.display()
            )));
        }
        let count = u64::from_le_bytes(map[8..16].try_into().expect("8"));
        let expected = count
            .checked_mul(RECORD_LEN as u64)
            .and_then(|bytes| bytes.checked_add(RECORDS_HEADER as u64));
        if expected != Some(map.len() as u64) {
            return Err(malformed(format!(
                "{} declares {count} records and holds {} bytes",
                path.display(),
                map.len()
            )));
        }
        let records = Self { map, count };
        let mut last = None;
        for record in records.iter() {
            let at = (record.book, record.ordinal);
            if last.is_some_and(|last| at <= last) {
                return Err(malformed(format!(
                    "{}: records do not ascend strictly by (book, ordinal) at {at:?}",
                    path.display()
                )));
            }
            last = Some(at);
        }
        Ok(records)
    }

    pub fn len(&self) -> u64 {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn get(&self, index: u64) -> PlanRecord {
        let at = RECORDS_HEADER + index as usize * RECORD_LEN;
        PlanRecord::decode(self.map[at..at + RECORD_LEN].try_into().expect("in range"))
    }

    pub fn iter(&self) -> impl Iterator<Item = PlanRecord> + '_ {
        self.map[RECORDS_HEADER..]
            .as_chunks::<RECORD_LEN>()
            .0
            .iter()
            .map(PlanRecord::decode)
    }
}

/// `books.json`: the books' stable keys, strictly ascending by bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct BookList(Vec<String>);

impl BookList {
    pub fn new(names: Vec<String>) -> Result<Self, PackError> {
        if let Some(pair) = names
            .windows(2)
            .find(|pair| pair[0].as_bytes() >= pair[1].as_bytes())
        {
            return Err(malformed(format!(
                "books must ascend strictly by bytes, and {:?} follows {:?}",
                pair[1], pair[0]
            )));
        }
        if let Some(name) = names
            .iter()
            .find(|name| name.is_empty() || name.len() > 65_535)
        {
            return Err(malformed(format!(
                "a book key is 1 to 65535 bytes, and {name:?} is not"
            )));
        }
        Ok(Self(names))
    }

    pub fn read(path: &Path) -> Result<Self, PackError> {
        Self::new(read_json::<Vec<String>>(path)?)
    }

    pub fn write(&self, path: &Path) -> Result<(), PackError> {
        write_json(path, &self.0)
    }

    pub fn names(&self) -> &[String] {
        &self.0
    }

    pub fn name(&self, index: u32) -> &str {
        &self.0[index as usize]
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The index of `name`, by binary search.
    pub fn index_of(&self, name: &str) -> Option<u32> {
        self.0
            .binary_search_by(|probe| probe.as_bytes().cmp(name.as_bytes()))
            .ok()
            .map(|index| index as u32)
    }
}

/// One line of `embed.jsonl`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbedRecord {
    /// The first 16 bytes of `embedding_text_sha256`, 32 lowercase hex digits.
    pub key: String,
    /// SHA-256 of `embedding_text`'s UTF-8 bytes, 64 lowercase hex digits.
    pub embedding_text_sha256: String,
    /// The exact string to embed: role prefix, context and cap already applied.
    pub embedding_text: String,
}

impl EmbedRecord {
    pub fn of(text: &str) -> Self {
        let digest = crate::distribution::files::sha256(text.as_bytes());
        Self {
            key: hex(&digest[..16]),
            embedding_text_sha256: hex(&digest),
            embedding_text: text.to_string(),
        }
    }

    /// Refuse a record whose text is not the text its digest names, or whose key is not
    /// its digest's prefix. Returns the digest. `record` is its position, for the error.
    pub fn check(&self, record: u64) -> Result<[u8; 32], PackError> {
        let actual = crate::distribution::files::sha256(self.embedding_text.as_bytes());
        if hex(&actual) != self.embedding_text_sha256 {
            return Err(PackError::PlanTextChanged {
                record,
                declared: self.embedding_text_sha256.clone(),
                actual: hex(&actual),
            });
        }
        if self.key != hex(&actual[..16]) {
            return Err(malformed(format!(
                "record {record}'s key is {} and its text's digest begins {}",
                self.key,
                hex(&actual[..16])
            )));
        }
        Ok(actual)
    }
}

/// `embed-manifest.json`: what `embed.jsonl` is.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbedManifest {
    pub format: String,
    pub version: u32,
    /// Lines of `embed.jsonl`.
    pub records: u64,
    /// SHA-256 of `embed.jsonl`'s bytes: what every shard names.
    pub plan_sha256: String,
    /// The family the texts are embedded for, copied into every shard's manifest.
    pub model: ModelIdentity,
    pub chunking_identity: u64,
    /// The package to embed with: the warehouse's, and one of `model.query_packages`.
    pub passage_package: ModelPackage,
}

impl EmbedManifest {
    pub fn read(dir: &Path) -> Result<Self, PackError> {
        let manifest: Self = read_json(&dir.join(EMBED_MANIFEST_FILE))?;
        if manifest.format != EMBED_PLAN_FORMAT || manifest.version != EMBED_PLAN_VERSION {
            return Err(malformed(format!(
                "{} is {} version {}, and this build reads {EMBED_PLAN_FORMAT} version \
                 {EMBED_PLAN_VERSION}",
                dir.join(EMBED_MANIFEST_FILE).display(),
                manifest.format,
                manifest.version
            )));
        }
        if !manifest
            .model
            .query_packages
            .contains(&manifest.passage_package)
        {
            return Err(malformed(format!(
                "the passage package {} is not one of the family's",
                manifest.passage_package.checksum
            )));
        }
        Ok(manifest)
    }
}

/// Writes `embed.jsonl` and its manifest: each key once, in order of first appearance,
/// and only the keys `warehouse` has no vector for.
///
/// What it holds: the 16-byte keys offered so far, to deduplicate — about 25 bytes a key.
pub struct EmbedPlanWriter<'w> {
    out: BufWriter<File>,
    dir: PathBuf,
    hasher: Sha256,
    seen: KeySet<[u8; 16]>,
    warehouse: Option<&'w dyn HeldVectors>,
    records: u64,
}

impl<'w> EmbedPlanWriter<'w> {
    pub fn create(dir: &Path, warehouse: Option<&'w dyn HeldVectors>) -> Result<Self, PackError> {
        let path = partial_path(&dir.join(EMBED_PLAN_FILE));
        Ok(Self {
            out: BufWriter::with_capacity(
                4 << 20,
                File::create(&path).map_err(io_error(format!("creating {}", path.display())))?,
            ),
            dir: dir.to_path_buf(),
            hasher: Sha256::new(),
            seen: KeySet::default(),
            warehouse,
            records: 0,
        })
    }

    /// Offer a text whose SHA-256 is `sha256`; it is written unless its key was offered
    /// before or the warehouse already holds its vector. Returns whether it was written.
    pub fn offer(&mut self, sha256: &[u8; 32], text: &str) -> Result<bool, PackError> {
        let key: [u8; 16] = sha256[..16].try_into().expect("16 of 32");
        if !self.seen.insert(key) {
            return Ok(false);
        }
        if self
            .warehouse
            .is_some_and(|warehouse| warehouse.holds(sha256))
        {
            return Ok(false);
        }
        let record = EmbedRecord {
            key: hex(&key),
            embedding_text_sha256: hex(sha256),
            embedding_text: text.to_string(),
        };
        let mut line = serde_json::to_vec(&record).expect("a record serializes");
        line.push(b'\n');
        self.hasher.update(&line);
        self.out
            .write_all(&line)
            .map_err(io_error(format!("writing {}", self.dir.display())))?;
        self.records += 1;
        Ok(true)
    }

    pub fn records(&self) -> u64 {
        self.records
    }

    /// Flush, rename into place, and write the manifest beside it.
    pub fn finish(
        self,
        model: &ModelIdentity,
        chunking_identity: u64,
        passage_package: &ModelPackage,
    ) -> Result<EmbedManifest, PackError> {
        let path = self.dir.join(EMBED_PLAN_FILE);
        let context = format!("finishing {}", path.display());
        let file = self.out.into_inner().map_err(|error| PackError::Io {
            context: context.clone(),
            source: error.into_error(),
        })?;
        (|| {
            file.sync_all()?;
            drop(file);
            std::fs::rename(partial_path(&path), &path)
        })()
        .map_err(io_error(context))?;
        let manifest = EmbedManifest {
            format: EMBED_PLAN_FORMAT.to_string(),
            version: EMBED_PLAN_VERSION,
            records: self.records,
            plan_sha256: format!("{:x}", self.hasher.finalize()),
            model: model.clone(),
            chunking_identity,
            passage_package: passage_package.clone(),
        };
        write_json(&self.dir.join(EMBED_MANIFEST_FILE), &manifest)?;
        Ok(manifest)
    }
}

/// The records of `embed.jsonl` from `skip`, at most `take`, each with its position.
pub fn read_embed_plan(
    dir: &Path,
    skip: u64,
    take: u64,
) -> Result<impl Iterator<Item = Result<(u64, EmbedRecord), PackError>>, PackError> {
    let path = dir.join(EMBED_PLAN_FILE);
    let file = File::open(&path).map_err(io_error(format!("reading {}", path.display())))?;
    let lines = BufReader::with_capacity(4 << 20, file).lines();
    Ok(lines
        .enumerate()
        .skip(skip as usize)
        .take(take.min(usize::MAX as u64) as usize)
        .map(move |(index, line)| {
            let line = line.map_err(io_error(format!("reading {}", path.display())))?;
            let record: EmbedRecord = serde_json::from_str(&line).map_err(|error| {
                malformed(format!(
                    "{} line {} is not an embed record: {error}",
                    path.display(),
                    index + 1
                ))
            })?;
            Ok((index as u64, record))
        }))
}

/// `plan-manifest.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanManifest {
    pub format: String,
    pub version: u32,
    /// The identity the vectors will be built under.
    pub identity: IndexVersion,
    pub library_version: u32,
    pub library_release_tag: String,
    /// The release the split was computed against; none for a first base.
    pub previous: Option<PreviousRelease>,
    pub counts: PlanCounts,
    /// Each file of the plan by name.
    pub files: BTreeMap<String, FileDigest>,
    /// The planner's parity gate: documents whose stored key was compared with the one
    /// recomputed from their text, and how many disagreed. `checked` is 0 for a plan made
    /// from a transcription, which has no stored keys to compare.
    pub parity: Parity,
    pub created_at: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreviousRelease {
    pub library_version: u32,
    /// SHA-256 of that release's ledger manifest.
    pub ledger_manifest_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Parity {
    pub checked: u64,
    pub mismatches: u64,
}

/// What a plan holds, and how it splits against the previous release.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct PlanCounts {
    pub records: u64,
    pub books: u64,
    /// Distinct keys.
    pub unique: u64,
    /// Keys the previous release holds already.
    pub reused: u64,
    /// Keys a delta ships as slots: `unique − reused`.
    pub to_ship: u64,
    /// Texts in `embed.jsonl`.
    pub to_embed: u64,
    /// Keys to ship whose vector the warehouse holds already.
    pub revived: u64,
    /// Keys the previous release holds and this one does not.
    pub tombstones: u64,
    /// (book, key) pairs new in this release whose key the previous release holds.
    pub foreign_pairs: u64,
}

impl PlanManifest {
    pub fn read(dir: &Path) -> Result<Self, PackError> {
        let manifest: Self = read_json(&dir.join(PLAN_MANIFEST_FILE))?;
        if manifest.format != PLAN_FORMAT || manifest.version != PLAN_FORMAT_VERSION {
            return Err(malformed(format!(
                "{} is {} version {}, and this build reads {PLAN_FORMAT} version \
                 {PLAN_FORMAT_VERSION}",
                dir.join(PLAN_MANIFEST_FILE).display(),
                manifest.format,
                manifest.version
            )));
        }
        manifest.identity.validate_complete()?;
        if manifest.parity.mismatches != 0 {
            return Err(malformed(format!(
                "the plan's parity gate found {} document(s) whose stored key is not their \
                 text's",
                manifest.parity.mismatches
            )));
        }
        Ok(manifest)
    }

    /// Write the manifest for the files already in `dir`, hashing each.
    #[allow(clippy::too_many_arguments)]
    pub fn write(
        dir: &Path,
        identity: IndexVersion,
        library_version: u32,
        library_release_tag: String,
        previous: Option<PreviousRelease>,
        counts: PlanCounts,
        parity: Parity,
        created_at: String,
    ) -> Result<Self, PackError> {
        identity.validate_complete()?;
        crate::distribution::package::validate_release_tag(&library_release_tag)?;
        let mut files = BTreeMap::new();
        for name in [
            RECORDS_FILE,
            BOOKS_FILE,
            EMBED_PLAN_FILE,
            EMBED_MANIFEST_FILE,
            TOMBSTONES_FILE,
        ] {
            let path = dir.join(name);
            if path.exists() {
                files.insert(name.to_string(), FileDigest::of(&path)?);
            }
        }
        for required in [RECORDS_FILE, BOOKS_FILE] {
            if !files.contains_key(required) {
                return Err(malformed(format!("a plan needs {required}")));
            }
        }
        let manifest = Self {
            format: PLAN_FORMAT.to_string(),
            version: PLAN_FORMAT_VERSION,
            identity,
            library_version,
            library_release_tag,
            previous,
            counts,
            files,
            parity,
            created_at,
        };
        write_json(&dir.join(PLAN_MANIFEST_FILE), &manifest)?;
        Ok(manifest)
    }
}

/// An opened plan: its manifest, books and records, each checked against the manifest.
pub struct Plan {
    pub dir: PathBuf,
    pub manifest: PlanManifest,
    pub books: BookList,
    pub records: PlanRecords,
}

impl Plan {
    pub fn open(dir: &Path) -> Result<Self, PackError> {
        let manifest = PlanManifest::read(dir)?;
        for name in [RECORDS_FILE, BOOKS_FILE] {
            manifest
                .files
                .get(name)
                .ok_or_else(|| malformed(format!("the plan manifest names no {name}")))?
                .verify(&dir.join(name))?;
        }
        let books = BookList::read(&dir.join(BOOKS_FILE))?;
        let records = PlanRecords::open(&dir.join(RECORDS_FILE))?;
        if let Some(record) = records
            .iter()
            .find(|record| record.book as usize >= books.len())
        {
            return Err(malformed(format!(
                "a record names book {} of {}",
                record.book,
                books.len()
            )));
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            manifest,
            books,
            records,
        })
    }
}

/// What [`plan_from_corpus`] needs beyond the corpus.
pub struct PlanRequest<'a> {
    pub out_dir: PathBuf,
    /// The family the vectors are for.
    pub model: ModelIdentity,
    pub chunking: ChunkerConfig,
    /// The package the passages are embedded with; one of `model.query_packages`.
    pub passage_package: ModelPackage,
    pub previous: Option<&'a Ledger>,
    pub warehouse: Option<&'a dyn HeldVectors>,
    pub created_at: String,
}

/// Write a complete plan from a corpus transcription — the planner the plugin implements
/// over the release index, for tests and for a corpus this crate can read.
pub fn plan_from_corpus(
    corpus: &dyn CorpusBooks,
    request: PlanRequest<'_>,
) -> Result<PlanManifest, PackError> {
    let corpus_identity = corpus.identity()?;
    let identity = IndexVersion {
        text: corpus_identity.text.clone(),
        model: request.model.clone(),
        store: readable_store_identity(),
    };
    identity.validate_complete()?;
    EmbeddingRecipe::resolve(&request.chunking, &request.model)?;
    ensure_recipe_matches(&request.chunking, &request.model)?;
    if !request
        .model
        .query_packages
        .contains(&request.passage_package)
    {
        return Err(malformed(format!(
            "the passage package {} is not one of the family's",
            request.passage_package.checksum
        )));
    }
    let dir = &request.out_dir;
    std::fs::create_dir_all(dir).map_err(io_error(format!("creating {}", dir.display())))?;

    let mut names = corpus.book_keys()?;
    names.sort_unstable();
    names.dedup();
    let books = BookList::new(names)?;
    books.write(&dir.join(BOOKS_FILE))?;

    let chunker = Chunker::new(request.chunking.clone())?;
    let mut records = RecordsWriter::create(&dir.join(RECORDS_FILE))?;
    let mut embed = EmbedPlanWriter::create(dir, request.warehouse)?;
    for (index, name) in books.names().iter().enumerate() {
        let (line_ids, chunks) = chunk_book_lines(corpus, &chunker, name)?;
        let ordinals: std::collections::HashMap<u64, u32> = line_ids
            .iter()
            .enumerate()
            .map(|(ordinal, line_id)| (*line_id, ordinal as u32))
            .collect();
        for chunk in chunks {
            let sha256 = crate::distribution::files::sha256(chunk.embedding_text.as_bytes());
            records.push(PlanRecord {
                book: index as u32,
                ordinal: ordinals[&chunk.line_id],
                sha256,
            })?;
            embed.offer(&sha256, &chunk.embedding_text)?;
        }
    }
    let record_count = records.finish()?;
    let to_embed = embed.records();
    embed.finish(
        &request.model,
        request.chunking.identity(),
        &request.passage_package,
    )?;
    if record_count == 0 {
        return Err(PackError::NothingToEmbed { books: books.len() });
    }

    let plan_records = PlanRecords::open(&dir.join(RECORDS_FILE))?;
    let SplitCounts {
        counts: mut plan_counts,
        ..
    } = split(
        &plan_records,
        &books,
        request.previous,
        request.warehouse,
        &dir.join(TOMBSTONES_FILE),
    )?;
    plan_counts.to_embed = to_embed;
    PlanManifest::write(
        dir,
        identity,
        corpus_identity.library_version,
        corpus_identity.library_release_tag,
        request.previous.map(Ledger::as_previous),
        plan_counts,
        Parity::default(),
        request.created_at,
    )
}
