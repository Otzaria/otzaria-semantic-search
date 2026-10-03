//! The files of a vector set directory, other than the segments themselves.
//!
//! ```text
//! <vectors>/
//!   .lock                        held for install, compaction and recovery
//!   CURRENT                      {"generation":7,"set":"gen-000007/set.json","set_sha256":"…"}
//!   PREVIOUS                     the generation CURRENT pointed to before the last flip
//!   segments/<sid>.oxv           immutable segments: a name never takes other bytes
//!   segments/<sid>.package.json  the release manifest a segment was installed from
//!   segments/<sid>.corrupt       a scrub's verdict: the SHA-256 of the bytes that failed
//!   gen-000007/set.json          one generation: which segments, in which order
//!   gen-000007/<sid>.del         which slots of a segment are dead in that generation
//!   gen-000007/<sid>.links       where a delta's foreign records resolve, in that generation
//!   staging/                     work in progress, removed by recovery
//!   incoming/                    where a caller leaves a download for install to take
//! ```
//!
//! Every file here is written once, under a temporary name, flushed, and renamed into
//! place, and its directory is flushed before a pointer names it — a segment taken from
//! `incoming/`, which the set did not write, is flushed too; the two pointer files are the
//! only ones ever replaced, and `std::fs::rename` replaces atomically on every platform the
//! crate builds for (`MoveFileExW` with `MOVEFILE_REPLACE_EXISTING` on Windows).

use crate::cancellation::CancellationToken;
use crate::distribution::package::{sync_dir, PackageKind};
use crate::errors::{ArtifactError, SemanticSearchError, VectorStoreError};
use crate::semantic::oxv::scan::{Link, LINK_UNRESOLVED};
use crate::semantic::versioning::{IndexVersion, VectorProvenance};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

pub(crate) const CURRENT: &str = "CURRENT";
pub(crate) const PREVIOUS: &str = "PREVIOUS";
pub(crate) const SET_FILE: &str = "set.json";
pub(crate) const LOCK_FILE: &str = ".lock";
pub(crate) const SEGMENTS_DIR: &str = "segments";
pub(crate) const STAGING_DIR: &str = "staging";
pub(crate) const INCOMING_DIR: &str = "incoming";

/// `set.json`'s `format`.
pub(crate) const SET_FORMAT: &str = "otzaria-vector-set";
/// `set.json`'s `format_version`.
pub(crate) const SET_FORMAT_VERSION: u32 = 1;

const DEL_MAGIC: &[u8; 8] = b"OXVDEL1\n";
const LINKS_MAGIC: &[u8; 8] = b"OXVLNK1\n";

pub(crate) fn generation_dir(generation: u64) -> String {
    format!("gen-{generation:06}")
}

/// The generation an entry of the set directory is, when its name is a generation's.
pub(crate) fn generation_of(name: &str) -> Option<u64> {
    let digits = name.strip_prefix("gen-")?;
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

/// The number the next generation takes: past every generation a pointer names and every
/// `gen-` entry in the directory. A new generation never lands on one that exists — a
/// pointer's, one that only an unreadable pointer could name, or one a crash left — so
/// writing it removes and overwrites nothing.
pub(crate) fn next_generation(dir: &Path) -> Result<u64, ArtifactError> {
    let mut highest = 0u64;
    for name in [CURRENT, PREVIOUS] {
        if let Ok(Some(pointer)) = read_pointer(dir, name) {
            highest = highest.max(pointer.generation);
        }
    }
    let listing = io_error(format!("listing {}", dir.display()));
    for entry in fs::read_dir(dir).map_err(listing)? {
        let entry = entry.map_err(io_error(format!("listing {}", dir.display())))?;
        if let Some(generation) = generation_of(&entry.file_name().to_string_lossy()) {
            highest = highest.max(generation);
        }
    }
    highest.checked_add(1).ok_or_else(|| ArtifactError::Io {
        context: format!("{}: no generation number is left", dir.display()),
        source: io::Error::from(io::ErrorKind::InvalidData),
    })
}

/// Point `name` — `CURRENT` or `PREVIOUS` — at a generation, `bytes` being the serialized
/// [`Pointer`]. A pointer is a file: a directory in its place, which nothing here makes, is
/// no pointer, and nothing can be renamed over one, so an empty one is removed first.
pub(crate) fn write_pointer(dir: &Path, name: &str, bytes: &[u8]) -> io::Result<()> {
    let path = dir.join(name);
    if fs::symlink_metadata(&path).is_ok_and(|metadata| metadata.is_dir()) {
        fs::remove_dir(&path)?;
    }
    write_atomically(&path, bytes)
}

pub(crate) fn segment_file(id: &str) -> String {
    format!("{SEGMENTS_DIR}/{id}.oxv")
}

/// Whether `id` is a segment id — 32 lowercase hex digits, as [`hex`] writes one. An id names
/// files — `segments/<id>.oxv`, a generation's `<id>.del` — so nothing else may stand for
/// one.
///
/// [`hex`]: crate::semantic::versioning::hex
pub(crate) fn is_segment_id(id: &str) -> bool {
    id.len() == 32
        && id
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// What `CURRENT` and `PREVIOUS` hold: a generation, and the digest of its `set.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Pointer {
    pub generation: u64,
    /// Relative to the vectors directory.
    pub set: String,
    pub set_sha256: String,
}

/// One generation of a set: `gen-NNNNNN/set.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct SetDocument {
    pub format: String,
    pub format_version: u32,
    pub generation: u64,
    pub identity: IndexVersion,
    pub identity_digest: String,
    pub codec_params_sha256: String,
    /// The `to` version of the newest segment.
    pub library_version: u32,
    pub library_release_tag: String,
    /// Oldest first: a base or compacted segment, then the deltas in the order applied.
    pub segments: Vec<SetSegment>,
    pub stats: SetStats,
    pub created_at: String,
}

/// One segment of a generation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct SetSegment {
    /// 32 hex digits — the segment id.
    pub id: String,
    /// Relative to the vectors directory.
    pub file: String,
    pub kind: PackageKind,
    pub from: u32,
    pub to: u32,
    pub sha256: String,
    pub size: u64,
    pub slots: u64,
    /// The digest of the package it was installed from; none for one made here.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub package_digest: Option<String>,
    pub provenance: VectorProvenance,
    pub del: DerivedFile,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub links: Option<DerivedFile>,
}

/// A file a generation derives for one segment, and what it must say.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub(crate) struct DerivedFile {
    /// Relative to the generation's directory.
    pub file: String,
    /// The file's CRC-32 trailer.
    pub crc32: u32,
    /// Dead slots for a `.del`; unresolved foreign records for a `.links`.
    pub count: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub(crate) struct SetStats {
    pub slots: u64,
    pub slots_dead: u64,
    pub foreign: u64,
    pub foreign_unresolved: u64,
    pub bytes: u64,
}

impl SetDocument {
    /// Refuse a document this build does not read, or one that contradicts itself.
    pub(crate) fn check(&self, generation: u64) -> Result<(), String> {
        if self.format != SET_FORMAT || self.format_version != SET_FORMAT_VERSION {
            return Err(format!(
                "it is {} version {}, and this build reads {SET_FORMAT} version \
                 {SET_FORMAT_VERSION}",
                self.format, self.format_version
            ));
        }
        if self.generation != generation {
            return Err(format!(
                "it is generation {}, and its pointer names {generation}",
                self.generation
            ));
        }
        if self.identity_digest != self.identity.identity_digest_hex() {
            return Err("its identity digest is not its identity's".to_string());
        }
        if self.segments.is_empty() {
            return Err("it names no segment".to_string());
        }
        let mut ids = std::collections::BTreeSet::new();
        for (index, segment) in self.segments.iter().enumerate() {
            // Its id names its files, in segments/ and in the generation's directory.
            if !is_segment_id(&segment.id) {
                return Err(format!("segment {index}'s id {:?} is not one", segment.id));
            }
            if !ids.insert(&segment.id) {
                return Err(format!("segment {} is in it twice", segment.id));
            }
            let named = |file: &DerivedFile, extension: &str| {
                file.file == format!("{}.{extension}", segment.id)
            };
            if !named(&segment.del, "del")
                || segment
                    .links
                    .as_ref()
                    .is_some_and(|links| !named(links, "links"))
            {
                return Err(format!(
                    "segment {index}'s derived files are not named by its id"
                ));
            }
            let first = index == 0;
            if first == (segment.kind == PackageKind::Delta) {
                return Err(format!(
                    "segment {index} is a {}, and a set is one base or compacted segment \
                     followed by deltas",
                    segment.kind
                ));
            }
            if segment.links.is_some() != (segment.kind == PackageKind::Delta) {
                return Err(format!("segment {index}'s links do not match its kind"));
            }
            if segment.file != segment_file(&segment.id) {
                return Err(format!(
                    "segment {index} is filed as {}, not as its id",
                    segment.file
                ));
            }
        }
        if self.library_version != self.segments[self.segments.len() - 1].to {
            return Err("its library version is not its newest segment's".to_string());
        }
        Ok(())
    }
}

/// One bit per slot, set for a slot no search may return.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Deleted {
    pub slots: u64,
    pub words: Vec<u64>,
}

impl Deleted {
    pub(crate) fn none(slots: u64) -> Self {
        Self {
            slots,
            words: vec![0; slots.div_ceil(64) as usize],
        }
    }

    pub(crate) fn is_set(&self, slot: u32) -> bool {
        self.words
            .get(slot as usize / 64)
            .is_some_and(|word| word & (1 << (slot % 64)) != 0)
    }

    /// Set the bit; whether it was clear before.
    pub(crate) fn set(&mut self, slot: u32) -> bool {
        let word = &mut self.words[slot as usize / 64];
        let bit = 1u64 << (slot % 64);
        let was_clear = *word & bit == 0;
        *word |= bit;
        was_clear
    }

    pub(crate) fn count(&self) -> u64 {
        self.words
            .iter()
            .map(|word| u64::from(word.count_ones()))
            .sum()
    }

    /// `OXVDEL1\n`, the slot count, the dead count, the words, a CRC-32 trailer.
    pub(crate) fn encode(&self) -> (Vec<u8>, u32) {
        let mut bytes = Vec::with_capacity(24 + self.words.len() * 8 + 4);
        bytes.extend_from_slice(DEL_MAGIC);
        bytes.extend_from_slice(&self.slots.to_le_bytes());
        bytes.extend_from_slice(&self.count().to_le_bytes());
        for word in &self.words {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        let crc = crc32fast::hash(&bytes);
        bytes.extend_from_slice(&crc.to_le_bytes());
        (bytes, crc)
    }

    pub(crate) fn decode(bytes: &[u8], expected: &DerivedFile, slots: u64) -> Result<Self, String> {
        let body = trailer(bytes, DEL_MAGIC, expected.crc32)?;
        let declared_slots = u64::from_le_bytes(body[8..16].try_into().expect("8 bytes"));
        let declared_dead = u64::from_le_bytes(body[16..24].try_into().expect("8 bytes"));
        if declared_slots != slots || body.len() != 24 + slots.div_ceil(64) as usize * 8 {
            return Err(format!(
                "it describes {declared_slots} slot(s) in {} bytes, and the segment has {slots}",
                body.len()
            ));
        }
        let words: Vec<u64> = body[24..]
            .as_chunks::<8>()
            .0
            .iter()
            .map(|word| u64::from_le_bytes(*word))
            .collect();
        let deleted = Self { slots, words };
        // Bits past the last slot would delete slots that do not exist.
        if !slots.is_multiple_of(64) {
            if let Some(last) = deleted.words.last() {
                if last >> (slots % 64) != 0 {
                    return Err("it marks slots past the segment's end".to_string());
                }
            }
        }
        let dead = deleted.count();
        if dead != declared_dead || dead != expected.count {
            return Err(format!(
                "it marks {dead} slot(s) dead, and declares {declared_dead} — the set {}",
                expected.count
            ));
        }
        Ok(deleted)
    }
}

/// `OXVLNK1\n`, the count, `{seg u16, flags u16, slot u32}` per foreign record, a CRC-32
/// trailer.
pub(crate) fn encode_links(links: &[Link]) -> (Vec<u8>, u32) {
    let mut bytes = Vec::with_capacity(16 + links.len() * 8 + 4);
    bytes.extend_from_slice(LINKS_MAGIC);
    bytes.extend_from_slice(&(links.len() as u64).to_le_bytes());
    for link in links {
        bytes.extend_from_slice(&link.seg.to_le_bytes());
        bytes.extend_from_slice(&link.flags.to_le_bytes());
        bytes.extend_from_slice(&link.slot.to_le_bytes());
    }
    let crc = crc32fast::hash(&bytes);
    bytes.extend_from_slice(&crc.to_le_bytes());
    (bytes, crc)
}

pub(crate) fn decode_links(
    bytes: &[u8],
    expected: &DerivedFile,
    foreign: u64,
) -> Result<Vec<Link>, String> {
    let body = trailer(bytes, LINKS_MAGIC, expected.crc32)?;
    let count = u64::from_le_bytes(body[8..16].try_into().expect("8 bytes"));
    if count != foreign || body.len() as u64 != 16 + count * 8 {
        return Err(format!(
            "it holds {count} link(s), and the segment has {foreign} foreign record(s)"
        ));
    }
    let links: Vec<Link> = body[16..]
        .as_chunks::<8>()
        .0
        .iter()
        .map(|entry| Link {
            seg: u16::from_le_bytes(entry[0..2].try_into().expect("2 bytes")),
            flags: u16::from_le_bytes(entry[2..4].try_into().expect("2 bytes")),
            slot: u32::from_le_bytes(entry[4..8].try_into().expect("4 bytes")),
        })
        .collect();
    let unresolved = links
        .iter()
        .filter(|link| link.flags & LINK_UNRESOLVED != 0)
        .count() as u64;
    if unresolved != expected.count {
        return Err(format!(
            "{unresolved} link(s) are unresolved, and the set declares {}",
            expected.count
        ));
    }
    Ok(links)
}

/// The body of a derived file, once its magic and its CRC trailer check out.
fn trailer<'a>(bytes: &'a [u8], magic: &[u8; 8], expected_crc: u32) -> Result<&'a [u8], String> {
    if bytes.len() < 8 + 8 + 4 || &bytes[..8] != magic {
        return Err("it is not the file its name says".to_string());
    }
    let (body, crc) = bytes.split_at(bytes.len() - 4);
    let crc = u32::from_le_bytes(crc.try_into().expect("4 bytes"));
    if crc != crc32fast::hash(body) {
        return Err("its CRC does not match its contents".to_string());
    }
    if crc != expected_crc {
        return Err(format!(
            "its CRC is {crc:08x}, and the set declares {expected_crc:08x}"
        ));
    }
    Ok(body)
}

/// What the set's code did to make a write durable, in order. Every publish — a segment, a
/// generation, a pointer — is written, its file flushed, renamed into place and its
/// directory flushed, and only then does a pointer name it; the tests read the order back
/// from [`JOURNAL`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) enum Durable {
    /// A file's bytes, flushed.
    File(PathBuf),
    /// A file renamed into place: from, to.
    Renamed(PathBuf, PathBuf),
    /// A directory's entries, flushed. Only where that happens — Unix: Windows cannot flush a
    /// directory, and keeps a rename's order itself.
    Dir(PathBuf),
}

#[cfg(test)]
thread_local! {
    pub(crate) static JOURNAL: std::cell::RefCell<Vec<Durable>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Record a step in [`JOURNAL`]; nothing outside a test build.
#[inline]
pub(crate) fn note(step: impl FnOnce() -> Durable) {
    #[cfg(test)]
    JOURNAL.with(|journal| journal.borrow_mut().push(step()));
    #[cfg(not(test))]
    let _ = step;
}

/// Flush a file the set is about to publish.
pub(crate) fn sync_file(file: &File, path: &Path) -> io::Result<()> {
    file.sync_all()?;
    note(|| Durable::File(path.to_path_buf()));
    Ok(())
}

/// Flush a directory of the set, so what was renamed or created in it survives a power
/// loss before anything names it.
pub(crate) fn sync_set_dir(path: &Path) -> io::Result<()> {
    sync_dir(path)?;
    // `sync_dir` flushes on Unix and does nothing elsewhere; only a flush is noted.
    #[cfg(unix)]
    note(|| Durable::Dir(path.to_path_buf()));
    Ok(())
}

/// Rename a flushed file into place.
pub(crate) fn rename_into_place(from: &Path, to: &Path) -> io::Result<()> {
    fs::rename(from, to)?;
    note(|| Durable::Renamed(from.to_path_buf(), to.to_path_buf()));
    Ok(())
}

/// Write `bytes` to `path` under a temporary name, flush, rename into place, and flush the
/// directory.
pub(crate) fn write_atomically(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let temporary = path.with_extension(match path.extension() {
        Some(extension) => format!("{}.tmp", extension.to_string_lossy()),
        None => "tmp".to_string(),
    });
    {
        let mut file = File::create(&temporary)?;
        file.write_all(bytes)?;
        sync_file(&file, &temporary)?;
    }
    rename_into_place(&temporary, path)?;
    if let Some(parent) = path.parent() {
        sync_set_dir(parent)?;
    }
    Ok(())
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// SHA-256 of a file, read in blocks, cancellably.
pub(crate) fn hash_file(
    path: &Path,
    cancel: &CancellationToken,
) -> Result<String, SemanticSearchError> {
    let mut file = File::open(path).map_err(io_error(format!("reading {}", path.display())))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        if cancel.is_cancelled() {
            return Err(SemanticSearchError::Cancelled);
        }
        let read = file
            .read(&mut buffer)
            .map_err(io_error(format!("reading {}", path.display())))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// `segments/<id>.corrupt`: what the segment's file holds, as the last to judge it found it —
/// a scrub, bytes that failed their CRCs; an install, the bytes it put there in place of
/// others a generation still names. A generation that names other bytes for the segment does
/// not open: they are not there.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Verdict {
    /// SHA-256 of what the file holds. Empty for a marker that names nothing — one written
    /// before verdicts named bytes, or one that does not read as a verdict — which condemns
    /// whatever any generation names.
    pub sha256: String,
    /// Why.
    pub reason: String,
}

pub(crate) fn verdict_path(dir: &Path, id: &str) -> PathBuf {
    dir.join(SEGMENTS_DIR).join(format!("{id}.corrupt"))
}

/// The verdict on segment `id`, if there is one. Opening cannot hash a segment, so a verdict
/// stands until what replaces or verifies the segment's bytes settles it: an install, or
/// the scrub that wrote it ([`condemn`]).
pub(crate) fn read_verdict(dir: &Path, id: &str) -> Option<Verdict> {
    let path = verdict_path(dir, id);
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return None,
        Err(error) => {
            return Some(Verdict {
                sha256: String::new(),
                reason: format!("{} could not be read: {error}", path.display()),
            })
        }
    };
    Some(serde_json::from_slice(&bytes).unwrap_or_else(|_| Verdict {
        sha256: String::new(),
        reason: String::from_utf8_lossy(&bytes).trim().to_string(),
    }))
}

/// The verdict that keeps a generation naming `sha256` for segment `id` shut, if one does: a
/// verdict on other bytes than those — or on none it names.
pub(crate) fn verdict_against(dir: &Path, id: &str, sha256: &str) -> Option<Verdict> {
    read_verdict(dir, id).filter(|verdict| verdict.sha256.is_empty() || verdict.sha256 != sha256)
}

/// Condemn segment `id`: a verdict naming `sha256` — the bytes a scrub read and found
/// damaged — and `reason`. The file is then hashed again, and a verdict on bytes it no longer
/// holds is withdrawn at once — an install replaced the file while the scrub read the old one,
/// and its bytes were never judged — leaving what [`settle_verdict`] leaves for the bytes it
/// holds. Whether the verdict stands.
pub(crate) fn condemn(
    dir: &Path,
    id: &str,
    sha256: &str,
    reason: &str,
) -> Result<bool, SemanticSearchError> {
    let path = verdict_path(dir, id);
    let verdict = Verdict {
        sha256: sha256.to_string(),
        reason: reason.to_string(),
    };
    let bytes = serde_json::to_vec_pretty(&verdict).expect("a verdict serializes");
    write_atomically(&path, &bytes).map_err(io_error(format!("marking {}", path.display())))?;
    // A file that cannot be hashed keeps the verdict: it condemns, which is the safe side.
    match hash_file(&dir.join(segment_file(id)), &CancellationToken::new()) {
        Ok(held) if held != sha256 => {
            settle_verdict(dir, id, &held)?;
            Ok(false)
        }
        _ => Ok(true),
    }
}

/// The generations the set opens now: those `CURRENT` and `PREVIOUS` name that open — every
/// segment where its generation says, as large, as many slots and of the kind it says, no
/// verdict against the bytes it names, every derived file sound. Only these stand on a
/// segment's bytes; opening maps the segments and reads none of their vectors, and every
/// mapping is gone when this returns.
fn live_generations(dir: &Path) -> Vec<SetDocument> {
    let mut documents: Vec<SetDocument> = Vec::new();
    for name in [CURRENT, PREVIOUS] {
        let Ok(Some(pointer)) = read_pointer(dir, name) else {
            continue;
        };
        if documents
            .iter()
            .any(|document| document.generation == pointer.generation)
        {
            continue;
        }
        if let Ok(set) = super::SegmentSet::open_generation(dir, &pointer) {
            documents.push(set.document().clone());
        }
    }
    documents
}

/// Every generation on disk whose `set.json` reads, named by a pointer or not: what may name a
/// segment's bytes, now or once a pointer is mended.
fn generations_on_disk(dir: &Path) -> Vec<SetDocument> {
    let mut documents = Vec::new();
    for entry in fs::read_dir(dir).into_iter().flatten().flatten() {
        let Some(generation) = generation_of(&entry.file_name().to_string_lossy()) else {
            continue;
        };
        let Ok(bytes) = fs::read(entry.path().join(SET_FILE)) else {
            continue;
        };
        if let Ok(document) = serde_json::from_slice::<SetDocument>(&bytes) {
            if document.check(generation).is_ok() {
                documents.push(document);
            }
        }
    }
    documents
}

/// Leave the verdict segment `id` needs once its file holds the bytes `sha256`, just put there
/// or verified there: none, when no generation on disk names other bytes for it; otherwise a
/// verdict naming `sha256`, so a generation that names bytes no longer there never opens on
/// these.
pub(crate) fn settle_verdict(dir: &Path, id: &str, sha256: &str) -> Result<(), ArtifactError> {
    let other = generations_on_disk(dir).into_iter().find_map(|document| {
        document
            .segments
            .into_iter()
            .find(|entry| entry.id == id && entry.sha256 != sha256)
            .map(|entry| (document.generation, entry.sha256))
    });
    let Some((generation, named)) = other else {
        if clear_verdict(dir, id)? {
            log::info!("The verdict on segment {id} is withdrawn: it holds verified bytes now");
        }
        return Ok(());
    };
    let path = verdict_path(dir, id);
    let verdict = Verdict {
        sha256: sha256.to_string(),
        reason: format!(
            "segment {id} holds {sha256} now, and generation {generation} names {named}, which \
             is gone"
        ),
    };
    let bytes = serde_json::to_vec_pretty(&verdict).expect("a verdict serializes");
    write_atomically(&path, &bytes).map_err(io_error(format!("marking {}", path.display())))
}

/// What [`place_segment`] did with a verified segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Placed {
    /// A file of the same bytes was in place already, and stays; the staged file is left
    /// where it was, for the caller to remove.
    Kept,
    /// The staged file is the segment's file now; there was none.
    New,
    /// The staged file replaced one of other bytes that no generation that opens stood on:
    /// damaged bytes, or bytes only shut or unreadable generations named.
    Replaced,
}

/// Put the verified segment at `staged`, whose SHA-256 is `sha256`, in place as
/// `segments/<id>.oxv` — the one name its id gives it — flush the directory, and settle the
/// segment's verdict for the bytes it holds ([`settle_verdict`]).
///
/// A file of the same bytes already there is kept — mapped by a reader, perhaps. A file of
/// other bytes is replaced unless a generation that opens names exactly the bytes it holds:
/// those are served, and are never rewritten, so `refuse` makes the error from what the file
/// holds, and nothing changes. A file whose bytes no generation that opens names is damage,
/// or what only shut generations stand on, and replacing it is a repair. Anything that fails
/// after the rename moves the file back to `staged`.
pub(crate) fn place_segment(
    dir: &Path,
    id: &str,
    staged: &Path,
    sha256: &str,
    cancel: &CancellationToken,
    refuse: impl FnOnce(String) -> SemanticSearchError,
) -> Result<Placed, SemanticSearchError> {
    let segments = dir.join(SEGMENTS_DIR);
    fs::create_dir_all(&segments).map_err(io_error(format!("creating {}", segments.display())))?;
    let target = dir.join(segment_file(id));
    let held = match target.exists() {
        true => Some(hash_file(&target, cancel)?),
        false => None,
    };
    let placed = match held {
        Some(held) if held == sha256 => Placed::Kept,
        held => {
            if let Some(held) = held.as_deref() {
                let served = live_generations(dir).iter().any(|document| {
                    document
                        .segments
                        .iter()
                        .any(|entry| entry.id == id && entry.sha256 == held)
                });
                if served {
                    return Err(refuse(held.to_string()));
                }
            }
            rename_into_place(staged, &target)
                .map_err(io_error(format!("moving {} into place", staged.display())))?;
            if held.is_some() {
                Placed::Replaced
            } else {
                Placed::New
            }
        }
    };
    let settled = sync_set_dir(&segments)
        .map_err(io_error(format!("flushing {}", segments.display())))
        .and_then(|()| settle_verdict(dir, id, sha256));
    if let Err(error) = settled {
        if placed != Placed::Kept {
            let _ = fs::rename(&target, staged);
        }
        return Err(error.into());
    }
    Ok(placed)
}

/// Withdraw any verdict on segment `id`, once its file holds bytes that were just verified
/// whole. Whether there was one.
pub(crate) fn clear_verdict(dir: &Path, id: &str) -> Result<bool, ArtifactError> {
    let path = verdict_path(dir, id);
    match fs::remove_file(&path) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(io_error(format!("removing {}", path.display()))(error)),
    }
    let segments = dir.join(SEGMENTS_DIR);
    sync_set_dir(&segments).map_err(io_error(format!("flushing {}", segments.display())))?;
    Ok(true)
}

/// Read a pointer file: `None` when there is none.
pub(crate) fn read_pointer(dir: &Path, name: &str) -> Result<Option<Pointer>, String> {
    let path = dir.join(name);
    match fs::read(&path) {
        Ok(bytes) => serde_json::from_slice(&bytes)
            .map(Some)
            .map_err(|error| format!("{name} is not a pointer: {error}")),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(format!("{name} could not be read: {error}")),
    }
}

/// Read the generation a pointer names, checking the digest the pointer carries.
pub(crate) fn read_generation(dir: &Path, pointer: &Pointer) -> Result<SetDocument, String> {
    if pointer.set != format!("{}/{SET_FILE}", generation_dir(pointer.generation)) {
        return Err(format!(
            "its pointer names {}, which is not generation {}'s set",
            pointer.set, pointer.generation
        ));
    }
    let bytes = fs::read(dir.join(&pointer.set))
        .map_err(|error| format!("{} could not be read: {error}", pointer.set))?;
    if sha256_hex(&bytes) != pointer.set_sha256 {
        return Err(format!("{} is not the file its pointer names", pointer.set));
    }
    let document: SetDocument = serde_json::from_slice(&bytes)
        .map_err(|error| format!("{} is not a set this build reads: {error}", pointer.set))?;
    document.check(pointer.generation)?;
    Ok(document)
}

/// The set directory's lock, held for as long as the value lives.
pub(crate) struct SetLock {
    _file: File,
}

impl SetLock {
    /// Take the lock, or `None` when another install, compaction or recovery holds it.
    pub(crate) fn try_take(dir: &Path) -> Result<Option<Self>, ArtifactError> {
        fs::create_dir_all(dir).map_err(io_error(format!("creating {}", dir.display())))?;
        let path = dir.join(LOCK_FILE);
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&path)
            .map_err(io_error(format!("opening {}", path.display())))?;
        match file.try_lock() {
            Ok(()) => Ok(Some(Self { _file: file })),
            Err(fs::TryLockError::WouldBlock) => Ok(None),
            Err(fs::TryLockError::Error(error)) => {
                Err(io_error(format!("locking {}", path.display()))(error))
            }
        }
    }

    /// Take the lock, or refuse: an install or a compaction never waits behind another.
    pub(crate) fn take(dir: &Path) -> Result<Self, ArtifactError> {
        Self::try_take(dir)?.ok_or_else(|| ArtifactError::Io {
            context: format!(
                "locking {}: another install or compaction of this vector set is running",
                dir.join(LOCK_FILE).display()
            ),
            source: io::Error::from(io::ErrorKind::WouldBlock),
        })
    }
}

pub(crate) fn io_error(context: String) -> impl FnOnce(io::Error) -> ArtifactError {
    move |source| ArtifactError::Io { context, source }
}

pub(crate) fn corrupted(reason: String) -> VectorStoreError {
    VectorStoreError::Corrupted { reason }
}

/// Where a generation's derived files go.
pub(crate) fn generation_path(dir: &Path, generation: u64) -> PathBuf {
    dir.join(generation_dir(generation))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_deleted_bitmap_round_trips_and_refuses_damage() {
        let mut deleted = Deleted::none(130);
        assert!(deleted.set(0));
        assert!(deleted.set(129));
        assert!(!deleted.set(129));
        let (bytes, crc) = deleted.encode();
        let expected = DerivedFile {
            file: "x.del".to_string(),
            crc32: crc,
            count: 2,
        };
        assert_eq!(Deleted::decode(&bytes, &expected, 130).unwrap(), deleted);
        assert!(Deleted::decode(&bytes, &expected, 131).is_err());
        let mut damaged = bytes.clone();
        damaged[30] ^= 1;
        assert!(Deleted::decode(&damaged, &expected, 130).is_err());
        let wrong_count = DerivedFile {
            count: 3,
            ..expected.clone()
        };
        assert!(Deleted::decode(&bytes, &wrong_count, 130).is_err());
    }

    #[test]
    fn links_round_trip_and_count_what_is_unresolved() {
        let links = vec![
            Link {
                seg: 0,
                flags: 0,
                slot: 7,
            },
            Link {
                seg: 0,
                flags: LINK_UNRESOLVED,
                slot: 0,
            },
        ];
        let (bytes, crc) = encode_links(&links);
        let expected = DerivedFile {
            file: "x.links".to_string(),
            crc32: crc,
            count: 1,
        };
        assert_eq!(decode_links(&bytes, &expected, 2).unwrap(), links);
        assert!(decode_links(&bytes, &expected, 3).is_err());
        let none_unresolved = DerivedFile {
            count: 0,
            ..expected
        };
        assert!(decode_links(&bytes, &none_unresolved, 2).is_err());
    }
}
