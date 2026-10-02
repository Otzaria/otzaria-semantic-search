//! What a device holds at a library version, and how the next plan splits against it.
//!
//! A release writes the ledger the next release is planned against:
//!
//! ```text
//! ledger-vN.keys           b"OXVKEYS1", u64 count, then the 16-byte keys a device holds
//!                          at vN, strictly ascending
//! pairs-vN.bin             b"OXVPAIR1", u64 count, then {book u32, key [16]} for every
//!                          (book, key) record of vN, strictly ascending; `book` indexes
//!                          the manifest's `books`
//! ledger-vN.manifest.json  the identity digest, the codec epoch and its parameters, the
//!                          two files' digests, the books, and the base this chain grows on
//! ```
//!
//! `tombstones.bin` has the keys file's layout. Every lookup is a binary search over a
//! mapped file, so a split holds neither file in memory.

use crate::distribution::files::{
    hex, io_error, malformed, partial_path, read_json, write_json, KeyMap, KeySet,
};
use crate::distribution::plan::{
    BookList, HeldVectors, PlanCounts, PlanRecord, PlanRecords, PreviousRelease,
};
use crate::errors::PackError;
use crate::semantic::oxv::codec::Codec;
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub const LEDGER_FORMAT: &str = "otzaria-vector-ledger";
pub const LEDGER_FORMAT_VERSION: u32 = 1;

const KEYS_MAGIC: &[u8; 8] = b"OXVKEYS1";
const PAIRS_MAGIC: &[u8; 8] = b"OXVPAIR1";
const HEADER: usize = 16;
const PAIR_LEN: usize = 20;

pub fn keys_file_name(version: u32) -> String {
    format!("ledger-v{version}.keys")
}

pub fn pairs_file_name(version: u32) -> String {
    format!("pairs-v{version}.bin")
}

pub fn manifest_file_name(version: u32) -> String {
    format!("ledger-v{version}.manifest.json")
}

/// `ledger-vN.manifest.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LedgerManifest {
    pub format: String,
    pub version: u32,
    pub library_version: u32,
    pub library_release_tag: String,
    /// The identity every segment of this chain declares.
    pub identity_digest: String,
    /// The codec epoch every segment of this chain is encoded in.
    pub codec: LedgerCodec,
    pub keys: LedgerFile,
    pub pairs: LedgerFile,
    pub books: Vec<String>,
    /// The base the chain grows on.
    pub base: BaseRecord,
    /// Bytes of the deltas published on top of that base, through this version.
    pub deltas_since_base: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LedgerCodec {
    pub id: u16,
    pub dim: u32,
    /// The codec's parameters section, hex.
    pub params: String,
    pub params_sha256: String,
}

impl LedgerCodec {
    pub fn of(codec: &Codec) -> Self {
        Self {
            id: codec.id(),
            dim: codec.dim() as u32,
            params: hex(codec.params()),
            params_sha256: hex(&codec.params_sha256()),
        }
    }

    pub fn codec(&self) -> Result<Codec, PackError> {
        let bytes = (0..self.params.len() / 2)
            .map(|at| u8::from_str_radix(&self.params[at * 2..at * 2 + 2], 16))
            .collect::<Result<Vec<u8>, _>>()
            .map_err(|_| malformed("a ledger's codec parameters are not hex"))?;
        let codec = Codec::from_params(self.id, self.dim as usize, &bytes)
            .map_err(|reason| malformed(format!("a ledger's codec: {reason}")))?;
        if hex(&codec.params_sha256()) != self.params_sha256 {
            return Err(malformed(
                "a ledger's codec parameters do not hash to the digest it declares",
            ));
        }
        Ok(codec)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LedgerFile {
    pub file: String,
    pub count: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BaseRecord {
    pub library_version: u32,
    /// The base segment's size in bytes.
    pub size: u64,
}

impl LedgerManifest {
    /// Refuse a chain this build cannot extend: another identity, or another codec epoch.
    pub fn ensure_matches(
        &self,
        identity_digest: &str,
        codec_params_sha256: Option<&str>,
    ) -> Result<(), PackError> {
        if self.identity_digest != identity_digest {
            return Err(malformed(format!(
                "the ledger of v{} is for identity {} and this plan's is {identity_digest}: \
                 a new identity starts with a base",
                self.library_version, self.identity_digest
            )));
        }
        if let Some(codec) = codec_params_sha256 {
            if self.codec.params_sha256 != codec {
                return Err(malformed(format!(
                    "the ledger of v{} is in codec epoch {} and this build asks for {codec}: \
                     a new epoch starts with a base",
                    self.library_version, self.codec.params_sha256
                )));
            }
        }
        Ok(())
    }
}

/// A ledger, opened: its manifest, and its two files mapped and checked.
pub struct Ledger {
    pub dir: PathBuf,
    pub manifest: LedgerManifest,
    manifest_sha256: String,
    books: BookList,
    keys: Mmap,
    pairs: Mmap,
}

impl Ledger {
    /// Open the ledger of `version` in `dir`, or the only one there when `None`.
    pub fn open(dir: &Path, version: Option<u32>) -> Result<Self, PackError> {
        let name = match version {
            Some(version) => manifest_file_name(version),
            None => {
                let mut found: Vec<String> = std::fs::read_dir(dir)
                    .map_err(io_error(format!("listing {}", dir.display())))?
                    .flatten()
                    .map(|entry| entry.file_name().to_string_lossy().to_string())
                    .filter(|name| name.starts_with("ledger-v") && name.ends_with(".manifest.json"))
                    .collect();
                found.sort();
                match found.as_slice() {
                    [one] => one.clone(),
                    [] => return Err(malformed(format!("{} holds no ledger", dir.display()))),
                    _ => {
                        return Err(malformed(format!(
                            "{} holds {} ledgers; name the version",
                            dir.display(),
                            found.len()
                        )))
                    }
                }
            }
        };
        let path = dir.join(&name);
        let bytes =
            std::fs::read(&path).map_err(io_error(format!("reading {}", path.display())))?;
        let manifest: LedgerManifest = read_json(&path)?;
        if manifest.format != LEDGER_FORMAT || manifest.version != LEDGER_FORMAT_VERSION {
            return Err(malformed(format!(
                "{} is {} version {}, and this build reads {LEDGER_FORMAT} version \
                 {LEDGER_FORMAT_VERSION}",
                path.display(),
                manifest.format,
                manifest.version
            )));
        }
        let keys = map_checked(
            &dir.join(&manifest.keys.file),
            KEYS_MAGIC,
            16,
            &manifest.keys,
        )?;
        let pairs = map_checked(
            &dir.join(&manifest.pairs.file),
            PAIRS_MAGIC,
            PAIR_LEN,
            &manifest.pairs,
        )?;
        let books = BookList::new(manifest.books.clone())?;
        manifest.codec.codec()?;
        Ok(Self {
            dir: dir.to_path_buf(),
            manifest_sha256: hex(&Sha256::digest(&bytes)),
            manifest,
            books,
            keys,
            pairs,
        })
    }

    pub fn library_version(&self) -> u32 {
        self.manifest.library_version
    }

    pub fn key_count(&self) -> u64 {
        self.manifest.keys.count
    }

    pub fn key(&self, index: u64) -> [u8; 16] {
        let at = HEADER + index as usize * 16;
        self.keys[at..at + 16].try_into().expect("in range")
    }

    /// The position of `key` among the ledger's keys.
    pub fn find_key(&self, key: &[u8; 16]) -> Option<u64> {
        let keys = self.keys[HEADER..].as_chunks::<16>().0;
        keys.binary_search(key).ok().map(|index| index as u64)
    }

    pub fn pair_count(&self) -> u64 {
        self.manifest.pairs.count
    }

    /// Whether the ledger's version records `key` in its book `book`.
    pub fn contains_pair(&self, book: u32, key: &[u8; 16]) -> bool {
        let pairs = self.pairs[HEADER..].as_chunks::<PAIR_LEN>().0;
        let mut probe = [0u8; PAIR_LEN];
        probe[..4].copy_from_slice(&book.to_be_bytes());
        probe[4..].copy_from_slice(key);
        pairs
            .binary_search_by(|pair| pair_order(pair).cmp(&probe))
            .is_ok()
    }

    pub fn books(&self) -> &BookList {
        &self.books
    }

    pub fn codec(&self) -> Result<Codec, PackError> {
        self.manifest.codec.codec()
    }

    pub fn as_previous(&self) -> PreviousRelease {
        PreviousRelease {
            library_version: self.manifest.library_version,
            ledger_manifest_sha256: self.manifest_sha256.clone(),
        }
    }
}

/// A pair as it sorts: the book big-endian, then the key — the order the file is in.
fn pair_order(pair: &[u8; PAIR_LEN]) -> [u8; PAIR_LEN] {
    let mut order = *pair;
    let book = u32::from_le_bytes(pair[..4].try_into().expect("4"));
    order[..4].copy_from_slice(&book.to_be_bytes());
    order
}

fn map_checked(
    path: &Path,
    magic: &[u8; 8],
    element: usize,
    declared: &LedgerFile,
) -> Result<Mmap, PackError> {
    let actual = crate::distribution::files::FileDigest::of(path)?;
    if actual.sha256 != declared.sha256 {
        return Err(malformed(format!(
            "{} hashes to {} and its ledger declares {}",
            path.display(),
            actual.sha256,
            declared.sha256
        )));
    }
    let file = File::open(path).map_err(io_error(format!("opening {}", path.display())))?;
    // SAFETY: a ledger is written once and renamed into place; nothing writes it after.
    let map =
        unsafe { Mmap::map(&file) }.map_err(io_error(format!("mapping {}", path.display())))?;
    if map.len() < HEADER || &map[..8] != magic {
        return Err(malformed(format!(
            "{} is not the file its name says",
            path.display()
        )));
    }
    let count = u64::from_le_bytes(map[8..16].try_into().expect("8"));
    if count != declared.count || (map.len() - HEADER) as u64 != count * element as u64 {
        return Err(malformed(format!(
            "{} holds {count} entries in {} bytes, and its ledger declares {}",
            path.display(),
            map.len(),
            declared.count
        )));
    }
    Ok(map)
}

/// Streams a sorted file of fixed-size entries behind a magic and a count.
pub(crate) struct SortedFileWriter {
    out: BufWriter<File>,
    path: PathBuf,
    partial: PathBuf,
    count: u64,
    last: Vec<u8>,
}

impl SortedFileWriter {
    fn create(path: &Path, magic: &[u8; 8]) -> Result<Self, PackError> {
        let partial = partial_path(path);
        let mut out = BufWriter::with_capacity(
            4 << 20,
            File::create(&partial).map_err(io_error(format!("creating {}", partial.display())))?,
        );
        out.write_all(magic)
            .and_then(|()| out.write_all(&0u64.to_le_bytes()))
            .map_err(io_error(format!("writing {}", partial.display())))?;
        Ok(Self {
            out,
            path: path.to_path_buf(),
            partial,
            count: 0,
            last: Vec::new(),
        })
    }

    pub(crate) fn keys(path: &Path) -> Result<Self, PackError> {
        Self::create(path, KEYS_MAGIC)
    }

    pub(crate) fn pairs(path: &Path) -> Result<Self, PackError> {
        Self::create(path, PAIRS_MAGIC)
    }

    pub(crate) fn push_key(&mut self, key: &[u8; 16]) -> Result<(), PackError> {
        self.push(key, key)
    }

    pub(crate) fn push_pair(&mut self, book: u32, key: &[u8; 16]) -> Result<(), PackError> {
        let mut entry = [0u8; PAIR_LEN];
        entry[..4].copy_from_slice(&book.to_le_bytes());
        entry[4..].copy_from_slice(key);
        self.push(&entry, &pair_order(&entry))
    }

    fn push(&mut self, entry: &[u8], order: &[u8]) -> Result<(), PackError> {
        if self.count > 0 && order <= self.last.as_slice() {
            return Err(malformed(format!(
                "{} must ascend strictly",
                self.path.display()
            )));
        }
        self.last.clear();
        self.last.extend_from_slice(order);
        self.out
            .write_all(entry)
            .map_err(io_error(format!("writing {}", self.partial.display())))?;
        self.count += 1;
        Ok(())
    }

    /// Write the count, flush, rename, and return what a manifest records of the file.
    pub(crate) fn finish(self) -> Result<LedgerFile, PackError> {
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
        let digest = crate::distribution::files::FileDigest::of(&self.path)?;
        Ok(LedgerFile {
            file: self
                .path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
            count: self.count,
            sha256: digest.sha256,
        })
    }
}

/// Read a keys file — a ledger's or `tombstones.bin` — into memory.
pub fn read_keys_file(path: &Path) -> Result<Vec<[u8; 16]>, PackError> {
    let bytes = std::fs::read(path).map_err(io_error(format!("reading {}", path.display())))?;
    if bytes.len() < HEADER || &bytes[..8] != KEYS_MAGIC {
        return Err(malformed(format!("{} is not a keys file", path.display())));
    }
    let count = u64::from_le_bytes(bytes[8..16].try_into().expect("8"));
    if (bytes.len() - HEADER) as u64 != count * 16 {
        return Err(malformed(format!(
            "{} declares {count} keys and holds {} bytes",
            path.display(),
            bytes.len()
        )));
    }
    Ok(bytes[HEADER..].as_chunks::<16>().0.to_vec())
}

/// What a record of a plan is, against the previous release.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Disposition {
    /// The first record of a key the previous release does not hold: the key's slot.
    Slot { slot: u32 },
    /// A later record, in another book, of a key this release ships.
    Extra { slot: u32 },
    /// A record, new in its book, of a key the previous release holds.
    Foreign,
    /// A (book, key) record the previous release holds already.
    Held,
    /// A second record of a key in one book: a record is a book and a key, and its hint
    /// is the first line that holds it.
    Repeat,
}

/// What [`classify`] found.
pub(crate) struct Classified {
    /// Distinct keys of the plan.
    pub unique: u64,
    /// Of those, the ones the previous release holds.
    pub reused: u64,
    /// Keys the plan ships, by slot.
    pub slots: u32,
    pub foreign: u64,
    /// One bit per previous ledger key: set when the plan still holds it.
    pub kept: Vec<u64>,
    /// The keys the plan ships, sorted.
    pub new_keys: Vec<[u8; 16]>,
}

/// Decide every record of `records` against `previous`, in record order: books in byte
/// order, each book's lines in order, so slots are assigned as a segment stores them.
///
/// What it holds: the keys it ships, about 25 bytes each, one book's keys, and a bit per
/// previous key — never the previous release's keys or pairs, which it searches mapped.
pub(crate) fn classify(
    records: &PlanRecords,
    books: &BookList,
    previous: Option<&Ledger>,
    mut visit: impl FnMut(u64, &PlanRecord, Disposition) -> Result<(), PackError>,
) -> Result<Classified, PackError> {
    let mut slots: KeyMap<[u8; 16], u32> = KeyMap::default();
    let ledger_keys = previous.map_or(0, Ledger::key_count);
    let mut kept = vec![0u64; ledger_keys.div_ceil(64) as usize];
    let (mut reused, mut foreign) = (0u64, 0u64);
    let mut in_book: KeySet<[u8; 16]> = KeySet::default();
    let mut current = None;
    let mut previous_book = None;
    for (index, record) in records.iter().enumerate() {
        if current != Some(record.book) {
            current = Some(record.book);
            in_book.clear();
            previous_book =
                previous.and_then(|ledger| ledger.books().index_of(books.name(record.book)));
        }
        let key = record.key_bytes();
        let disposition = if !in_book.insert(key) {
            Disposition::Repeat
        } else if let Some(at) = previous.and_then(|ledger| ledger.find_key(&key)) {
            let (word, bit) = ((at / 64) as usize, at % 64);
            if kept[word] & (1 << bit) == 0 {
                kept[word] |= 1 << bit;
                reused += 1;
            }
            let ledger = previous.expect("found in it");
            if previous_book.is_some_and(|book| ledger.contains_pair(book, &key)) {
                Disposition::Held
            } else {
                foreign += 1;
                Disposition::Foreign
            }
        } else if let Some(&slot) = slots.get(&key) {
            Disposition::Extra { slot }
        } else {
            let slot = u32::try_from(slots.len())
                .map_err(|_| malformed("more keys than a segment can hold"))?;
            slots.insert(key, slot);
            Disposition::Slot { slot }
        };
        visit(index as u64, &record, disposition)?;
    }
    let mut new_keys: Vec<[u8; 16]> = slots.keys().copied().collect();
    new_keys.sort_unstable();
    Ok(Classified {
        unique: reused + slots.len() as u64,
        reused,
        slots: slots.len() as u32,
        foreign,
        kept,
        new_keys,
    })
}

impl Classified {
    /// Whether the previous ledger's key at `index` is still held.
    pub(crate) fn is_kept(&self, index: u64) -> bool {
        self.kept
            .get((index / 64) as usize)
            .is_some_and(|word| word & (1 << (index % 64)) != 0)
    }
}

/// What a split counted.
pub struct SplitCounts {
    /// Every count but `to_embed`, which the plan's embed writer knows.
    pub counts: PlanCounts,
}

/// Split a plan against the release before it: count what it reuses, ships, revives
/// from the warehouse and tombstones, and write the tombstones to `tombstones`.
pub fn split(
    records: &PlanRecords,
    books: &BookList,
    previous: Option<&Ledger>,
    warehouse: Option<&dyn HeldVectors>,
    tombstones: &Path,
) -> Result<SplitCounts, PackError> {
    let mut revived = 0u64;
    let classified = classify(records, books, previous, |_, record, disposition| {
        if matches!(disposition, Disposition::Slot { .. })
            && warehouse.is_some_and(|warehouse| warehouse.holds(&record.sha256))
        {
            revived += 1;
        }
        Ok(())
    })?;
    let mut writer = SortedFileWriter::keys(tombstones)?;
    let mut dead = 0u64;
    if let Some(ledger) = previous {
        for index in 0..ledger.key_count() {
            if !classified.is_kept(index) {
                writer.push_key(&ledger.key(index))?;
                dead += 1;
            }
        }
    }
    writer.finish()?;
    Ok(SplitCounts {
        counts: PlanCounts {
            records: records.len(),
            books: books.len() as u64,
            unique: classified.unique,
            reused: classified.reused,
            to_ship: u64::from(classified.slots),
            to_embed: 0,
            revived,
            tombstones: dead,
            foreign_pairs: classified.foreign,
        },
    })
}

/// Write the ledger of the release `plan` describes, on top of `previous` — how a build
/// machine that lost its state rebuilds it from the plan of the release last published.
pub fn write_plan_ledger(
    plan: &crate::distribution::plan::Plan,
    previous: Option<&Ledger>,
    codec: &Codec,
    base: BaseRecord,
    deltas_since_base: u64,
    dir: &Path,
) -> Result<LedgerManifest, PackError> {
    let classified = classify(&plan.records, &plan.books, previous, |_, _, _| Ok(()))?;
    write_ledger(
        dir,
        plan.manifest.library_version,
        &plan.manifest.library_release_tag,
        &plan.manifest.identity.identity_digest_hex(),
        codec,
        base,
        deltas_since_base,
        &plan.records,
        &plan.books,
        previous,
        &classified,
    )
}

/// Write the ledger of the release a plan describes: its keys, the previous release's
/// kept keys and the new ones merged; its pairs, every non-repeat record's; its manifest.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_ledger(
    dir: &Path,
    library_version: u32,
    library_release_tag: &str,
    identity_digest: &str,
    codec: &Codec,
    base: BaseRecord,
    deltas_since_base: u64,
    records: &PlanRecords,
    books: &BookList,
    previous: Option<&Ledger>,
    classified: &Classified,
) -> Result<LedgerManifest, PackError> {
    let mut keys = SortedFileWriter::keys(&dir.join(keys_file_name(library_version)))?;
    let mut new = classified.new_keys.iter().peekable();
    if let Some(ledger) = previous {
        for index in 0..ledger.key_count() {
            if !classified.is_kept(index) {
                continue;
            }
            let old = ledger.key(index);
            while let Some(fresh) = new.next_if(|fresh| **fresh < old) {
                keys.push_key(fresh)?;
            }
            keys.push_key(&old)?;
        }
    }
    for fresh in new {
        keys.push_key(fresh)?;
    }
    let keys = keys.finish()?;

    let mut pairs = SortedFileWriter::pairs(&dir.join(pairs_file_name(library_version)))?;
    let mut book_keys: Vec<[u8; 16]> = Vec::new();
    let mut current = None;
    let mut flush = |book: u32, book_keys: &mut Vec<[u8; 16]>| -> Result<(), PackError> {
        book_keys.sort_unstable();
        book_keys.dedup();
        for key in book_keys.iter() {
            pairs.push_pair(book, key)?;
        }
        book_keys.clear();
        Ok(())
    };
    for record in records.iter() {
        if current != Some(record.book) {
            if let Some(book) = current {
                flush(book, &mut book_keys)?;
            }
            current = Some(record.book);
        }
        book_keys.push(record.key_bytes());
    }
    if let Some(book) = current {
        flush(book, &mut book_keys)?;
    }
    let pairs = pairs.finish()?;

    let manifest = LedgerManifest {
        format: LEDGER_FORMAT.to_string(),
        version: LEDGER_FORMAT_VERSION,
        library_version,
        library_release_tag: library_release_tag.to_string(),
        identity_digest: identity_digest.to_string(),
        codec: LedgerCodec::of(codec),
        keys,
        pairs,
        books: books.names().to_vec(),
        base,
        deltas_since_base,
    };
    write_json(&dir.join(manifest_file_name(library_version)), &manifest)?;
    Ok(manifest)
}
