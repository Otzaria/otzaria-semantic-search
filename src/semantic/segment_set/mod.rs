//! A vector set on a device: segments, generations, and the one file that says which
//! generation is live.
//!
//! A set is a base segment — or one compacted on the device — and the deltas applied on top
//! of it, oldest first. What the deltas change is not written into the segments, which never
//! change: each **generation** has its own derived files, a `.del` bitmap per segment (the
//! slots a later delta tombstoned or shipped again) and a `.links` table per delta (the older
//! slot each of its foreign records resolves to). `CURRENT` names the live generation and
//! `PREVIOUS` the one before it; an install or a compaction writes a new generation beside
//! the old ones — numbered past every one on disk, so it never lands on one — and flips
//! `CURRENT` in one rename, so a crash at any point leaves one or the other
//! (`docs/ARTIFACT_CONTRACT.md` has the crash matrix).
//!
//! | operation | what it needs | what it changes |
//! |---|---|---|
//! | [`SegmentSet::open`] | the directory | nothing, unless it can clean up after a crash |
//! | [`install_package`] | a segment and its release manifest | a new generation |
//! | [`compact()`] | the set, optionally the live index's keys | a new generation of one segment |
//! | [`info()`] | the directory | nothing |
//! | [`scrub`] | the directory | marks a damaged segment |
//!
//! None of them loads a model: an install checks the identity a release declares against
//! the one an installation declares, and the vectors' bytes against their checksums.
//!
//! **Garbage** — generations neither pointer names, and segments only they used — is
//! removed after every flip and at every open that can take the lock. On Windows a file that
//! is mapped cannot be deleted, so a segment a running reader still holds stays until a
//! later open or install finds it free; nothing depends on it being gone.

mod compact;
mod files;
mod install;
mod space;
#[cfg(test)]
mod tests;

pub use compact::{compact, CompactionPolicy, CompactionReport};
pub use install::{
    install_package, ApplyReport, InstallExpectation, InstallSource, ReleaseFile, ReleaseManifest,
    ReleaseSegment,
};

use crate::cancellation::CancellationToken;
use crate::distribution::package::PackageKind;
use crate::errors::{ArtifactError, SemanticSearchError, VectorStoreError};
use crate::semantic::oxv::codec::Codec;
use crate::semantic::oxv::kernel::PreparedQuery;
use crate::semantic::oxv::reader::Segment;
use crate::semantic::oxv::scan::{scan, Link, ReverseLink, ScanRequest, ScanSegment, ScanSet};
use crate::semantic::resolve::VectorHit;
use crate::semantic::versioning::{
    hex, IdentityField, IdentityMismatch, IndexVersion, VectorProvenance,
};
use files::{
    corrupted, decode_links, generation_dir, io_error, read_generation, read_pointer, Deleted,
    Pointer, SetDocument, SetLock, CURRENT, INCOMING_DIR, PREVIOUS, SEGMENTS_DIR, STAGING_DIR,
};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

/// The backend id a vector set declares in its store identity.
pub const STORE_BACKEND_ID: &str = "otzaria-oxv";

/// Where a caller leaves a downloaded segment for [`install_package`] to take: inside the
/// set's directory, so taking it is a rename, and outside anything recovery cleans up.
pub fn incoming_dir(dir: &Path) -> PathBuf {
    dir.join(INCOMING_DIR)
}

/// What a set is: its generation, identity and segments, and how much of it is dead.
#[derive(Debug, Clone, PartialEq)]
pub struct SetInfo {
    pub generation: u64,
    pub identity: IndexVersion,
    pub identity_digest: String,
    /// The codec epoch every segment shares.
    pub codec_params_sha256: String,
    /// The library version the newest segment brings the set to.
    pub library_version: u32,
    pub library_release_tag: String,
    pub segments: Vec<SegmentInfo>,
    pub slots_live: u64,
    pub slots_dead: u64,
    /// The segments' sizes, which is what the set occupies beyond a few small files.
    pub bytes_on_disk: u64,
    /// Whether [`CompactionPolicy::default`] would compact it now.
    pub needs_compaction: bool,
    /// `CURRENT`'s generation could not be opened, and this is `PREVIOUS`'s.
    pub recovered_from_previous: bool,
}

/// One segment of a set.
#[derive(Debug, Clone, PartialEq)]
pub struct SegmentInfo {
    /// The segment id, as 32 hex digits.
    pub id: String,
    pub kind: PackageKind,
    pub from_library_version: u32,
    pub to_library_version: u32,
    pub slots: u64,
    pub slots_dead: u64,
    pub foreign_unresolved: u64,
    pub size: u64,
    pub sha256: String,
    pub provenance: VectorProvenance,
}

/// What a scrub read, and what it found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScrubReport {
    pub generation: u64,
    pub segments: u32,
    pub bytes_checked: u64,
    pub elapsed_ms: u64,
}

/// One generation of a set, open for scans: its segments mapped, their derived files in
/// memory.
pub struct SegmentSet {
    dir: PathBuf,
    /// The pointer it was opened through — `CURRENT`'s, or `PREVIOUS`'s on a fallback.
    pointer: Pointer,
    document: SetDocument,
    segments: Vec<Segment>,
    deleted: Vec<Deleted>,
    links: Vec<Vec<Link>>,
    reverse: Vec<ReverseLink>,
    info: SetInfo,
}

impl std::fmt::Debug for SegmentSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SegmentSet")
            .field("dir", &self.dir)
            .field("generation", &self.document.generation)
            .field("segments", &self.segments.len())
            .finish_non_exhaustive()
    }
}

impl SegmentSet {
    /// Open the live generation of the set at `dir`: `CURRENT`'s, or `PREVIOUS`'s when
    /// `CURRENT`'s does not open — which [`SetInfo::recovered_from_previous`] then says.
    ///
    /// Recovery runs first when nothing else holds the set's lock: what a crashed install
    /// left in `staging/` is removed, and so is garbage. Opening reads every small section
    /// of every segment and every derived file, and checks each segment against the
    /// generation that names it; the vectors are mapped, not read.
    ///
    /// # Errors
    ///
    /// [`ArtifactError::MetadataUnusable`] when no set is installed;
    /// [`ArtifactError::IdentityMismatch`] naming `store.backend_id` for a directory that
    /// holds a v1 artifact; [`VectorStoreError::Corrupted`] when neither generation opens.
    pub fn open(dir: &Path) -> Result<Self, SemanticSearchError> {
        if let Some(_lock) = SetLock::try_take(dir)? {
            recover(dir)?;
        }
        Self::open_unlocked(dir)
    }

    /// [`Self::open`] without recovery, for a caller already holding the lock.
    pub(crate) fn open_unlocked(dir: &Path) -> Result<Self, SemanticSearchError> {
        let current = read_pointer(dir, CURRENT);
        let previous = read_pointer(dir, PREVIOUS);
        if matches!((&current, &previous), (Ok(None), Ok(None))) {
            refuse_a_v1_artifact(dir)?;
            return Err(ArtifactError::MetadataUnusable {
                path: dir.join(CURRENT).display().to_string(),
                reason: "no vector set is installed here".to_string(),
            }
            .into());
        }
        let first = match &current {
            Ok(Some(pointer)) => match Self::open_generation(dir, pointer) {
                Ok(set) => return Ok(set),
                Err(error) => error,
            },
            Ok(None) => corrupted(format!(
                "{}: there is a PREVIOUS and no CURRENT",
                dir.display()
            ))
            .into(),
            Err(reason) => corrupted(format!("{}: {reason}", dir.display())).into(),
        };
        let current_pointer = match &current {
            Ok(Some(pointer)) => Some(pointer),
            _ => None,
        };
        match &previous {
            Ok(Some(pointer)) if current_pointer != Some(pointer) => {
                match Self::open_generation(dir, pointer) {
                    Ok(mut set) => {
                        log::warn!(
                            "The vector set at {} fell back to generation {}: CURRENT does \
                             not open ({first})",
                            dir.display(),
                            pointer.generation
                        );
                        set.info.recovered_from_previous = true;
                        Ok(set)
                    }
                    // CURRENT's failure is the one to report: it is the generation that
                    // should have opened.
                    Err(_) => Err(first),
                }
            }
            _ => Err(first),
        }
    }

    fn open_generation(dir: &Path, pointer: &Pointer) -> Result<Self, SemanticSearchError> {
        let document = read_generation(dir, pointer)
            .map_err(|reason| corrupted(format!("{}: {reason}", dir.display())))?;
        let generation = dir.join(generation_dir(pointer.generation));
        let mut segments = Vec::with_capacity(document.segments.len());
        let mut deleted = Vec::with_capacity(document.segments.len());
        let mut links = Vec::with_capacity(document.segments.len());
        if let Some(reason) = condemnation(dir, &document) {
            return Err(corrupted(reason).into());
        }
        for entry in &document.segments {
            let segment = Segment::open(&dir.join(&entry.file))?;
            check_segment(&document, entry, &segment)?;
            let del_path = generation.join(&entry.del.file);
            let bytes = fs::read(&del_path)
                .map_err(|error| corrupted(format!("{}: {error}", del_path.display())))?;
            deleted.push(
                Deleted::decode(&bytes, &entry.del, entry.slots)
                    .map_err(|reason| corrupted(format!("{}: {reason}", del_path.display())))?,
            );
            links.push(match &entry.links {
                Some(file) => {
                    let path = generation.join(&file.file);
                    let bytes = fs::read(&path)
                        .map_err(|error| corrupted(format!("{}: {error}", path.display())))?;
                    decode_links(&bytes, file, u64::from(segment.foreign_count()))
                        .map_err(|reason| corrupted(format!("{}: {reason}", path.display())))?
                }
                None => Vec::new(),
            });
            segments.push(segment);
        }
        let count = segments.len();
        let mut reverse = Vec::new();
        for (from, links) in links.iter().enumerate() {
            for (foreign, link) in links.iter().enumerate() {
                if let Some((seg, slot)) = link.target() {
                    if seg as usize >= from || slot >= segments[seg as usize].slot_count() {
                        return Err(corrupted(format!(
                            "{}: a link of segment {from} points to segment {seg} slot {slot}",
                            dir.display()
                        ))
                        .into());
                    }
                    reverse.push(ReverseLink {
                        target_seg: seg,
                        target_slot: slot,
                        from_seg: from as u16,
                        foreign: foreign as u32,
                    });
                }
            }
        }
        debug_assert_eq!(count, deleted.len());
        reverse.sort_unstable();
        let dead: Vec<u64> = deleted.iter().map(Deleted::count).collect();
        let unresolved: Vec<u64> = links
            .iter()
            .map(|links| links.iter().filter(|link| link.target().is_none()).count() as u64)
            .collect();
        let info = info_of(&document, &dead, &unresolved, false);
        Ok(Self {
            dir: dir.to_path_buf(),
            pointer: pointer.clone(),
            document,
            segments,
            deleted,
            links,
            reverse,
            info,
        })
    }

    /// Scan every live vector of the set: the best `request.top_k` for `query`, a unit
    /// vector of the set's width.
    pub fn scan(
        &self,
        query: &[f32],
        request: &ScanRequest<'_>,
        cancel: &CancellationToken,
    ) -> Result<Vec<VectorHit>, VectorStoreError> {
        let prepared = PreparedQuery::new(query, self.codec())?;
        let views: Vec<ScanSegment<'_>> = self
            .segments
            .iter()
            .zip(&self.deleted)
            .zip(&self.links)
            .map(|((segment, deleted), links)| ScanSegment {
                segment,
                deleted: &deleted.words,
                links,
            })
            .collect();
        let set = ScanSet {
            segments: &views,
            reverse_links: &self.reverse,
        };
        scan(&set, &prepared, request, cancel)
    }

    pub fn info(&self) -> &SetInfo {
        &self.info
    }

    pub fn generation(&self) -> u64 {
        self.document.generation
    }

    pub fn identity(&self) -> &IndexVersion {
        &self.document.identity
    }

    /// The codec every segment of the set shares.
    pub fn codec(&self) -> &Codec {
        self.segments[0].codec()
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub(crate) fn document(&self) -> &SetDocument {
        &self.document
    }

    /// The pointer that names this generation: what `PREVIOUS` takes when a generation is
    /// built on it.
    pub(crate) fn pointer(&self) -> &Pointer {
        &self.pointer
    }

    pub(crate) fn segments(&self) -> &[Segment] {
        &self.segments
    }

    pub(crate) fn deleted(&self) -> &[Deleted] {
        &self.deleted
    }

    pub(crate) fn links(&self) -> &[Vec<Link>] {
        &self.links
    }
}

/// Why a scrub's verdict condemns `document`, if one does: a verdict on any of its segments.
/// `open` and `info` both ask, before reading a segment, so they refuse — and fall back —
/// alike.
fn condemnation(dir: &Path, document: &SetDocument) -> Option<String> {
    document.segments.iter().find_map(|entry| {
        files::read_verdict(dir, &entry.id).map(|verdict| {
            format!(
                "segment {} failed a scrub ({}); the set has to be installed again",
                entry.id, verdict.reason
            )
        })
    })
}

/// The segment is the one its generation names: id, identity, codec epoch, size, slots,
/// kind and versions.
fn check_segment(
    document: &SetDocument,
    entry: &files::SetSegment,
    segment: &Segment,
) -> Result<(), SemanticSearchError> {
    let disagree = |what: &str| {
        Err(corrupted(format!(
            "segment {} is not the one its set names: its {what} differs",
            entry.id
        ))
        .into())
    };
    if hex(&segment.segment_id()) != entry.id {
        return disagree("id");
    }
    if hex(&segment.identity_digest()) != document.identity_digest {
        return disagree("identity digest");
    }
    if hex(&segment.codec_params_sha256()) != document.codec_params_sha256 {
        return disagree("codec epoch");
    }
    if segment.size() != entry.size {
        return disagree("size");
    }
    if u64::from(segment.slot_count()) != entry.slots {
        return disagree("slot count");
    }
    if segment.kind() != entry.kind
        || segment.from_library_version() != entry.from
        || segment.to_library_version() != entry.to
    {
        return disagree("kind or library versions");
    }
    Ok(())
}

fn info_of(
    document: &SetDocument,
    dead: &[u64],
    unresolved: &[u64],
    recovered_from_previous: bool,
) -> SetInfo {
    let segments: Vec<SegmentInfo> = document
        .segments
        .iter()
        .zip(dead.iter().zip(unresolved))
        .map(|(entry, (dead, unresolved))| SegmentInfo {
            id: entry.id.clone(),
            kind: entry.kind,
            from_library_version: entry.from,
            to_library_version: entry.to,
            slots: entry.slots,
            slots_dead: *dead,
            foreign_unresolved: *unresolved,
            size: entry.size,
            sha256: entry.sha256.clone(),
            provenance: entry.provenance.clone(),
        })
        .collect();
    let slots: u64 = segments.iter().map(|segment| segment.slots).sum();
    let dead: u64 = segments.iter().map(|segment| segment.slots_dead).sum();
    let mut info = SetInfo {
        generation: document.generation,
        identity: document.identity.clone(),
        identity_digest: document.identity_digest.clone(),
        codec_params_sha256: document.codec_params_sha256.clone(),
        library_version: document.library_version,
        library_release_tag: document.library_release_tag.clone(),
        bytes_on_disk: segments.iter().map(|segment| segment.size).sum(),
        segments,
        slots_live: slots.saturating_sub(dead),
        slots_dead: dead,
        needs_compaction: false,
        recovered_from_previous,
    };
    info.needs_compaction = CompactionPolicy::default().wants(&info).is_some();
    info
}

/// What is installed at `dir`, without opening a segment: `None` when nothing is.
///
/// Reads the pointers and the generation they name, and the derived files' counts as the
/// generation declares them; nothing is cleaned up, and nothing is mapped. A generation a
/// scrub's verdict condemns is passed over for `PREVIOUS`'s, as [`SegmentSet::open`] passes
/// it over.
pub fn info(dir: &Path) -> Result<Option<SetInfo>, SemanticSearchError> {
    let current = read_pointer(dir, CURRENT);
    let previous = read_pointer(dir, PREVIOUS);
    if matches!((&current, &previous), (Ok(None), Ok(None))) {
        refuse_a_v1_artifact(dir)?;
        return Ok(None);
    }
    let declared = |document: &SetDocument, recovered| {
        let dead: Vec<u64> = document
            .segments
            .iter()
            .map(|entry| entry.del.count)
            .collect();
        let unresolved: Vec<u64> = document
            .segments
            .iter()
            .map(|entry| entry.links.as_ref().map_or(0, |links| links.count))
            .collect();
        info_of(document, &dead, &unresolved, recovered)
    };
    let usable = |pointer: &Result<Option<Pointer>, String>| -> Result<SetDocument, String> {
        let pointer = match pointer {
            Ok(Some(pointer)) => pointer,
            Ok(None) => return Err("there is none".to_string()),
            Err(reason) => return Err(reason.clone()),
        };
        let document = read_generation(dir, pointer)?;
        match condemnation(dir, &document) {
            Some(reason) => Err(reason),
            None => Ok(document),
        }
    };
    let first = match usable(&current) {
        Ok(document) => return Ok(Some(declared(&document, false))),
        Err(reason) => reason,
    };
    if let Ok(document) = usable(&previous) {
        return Ok(Some(declared(&document, true)));
    }
    Err(corrupted(format!(
        "{}: neither CURRENT nor PREVIOUS names a generation that can be read: {first}",
        dir.display()
    ))
    .into())
}

/// Read every block of every segment of the live generation and check its CRC — the check
/// opening leaves out, run on demand. A segment that fails is condemned by a verdict
/// (`segments/<id>.corrupt`) naming the SHA-256 of the bytes that failed, so every later open
/// and [`info()`] pass its generation over rather than serve it, until an install writes or
/// verifies the segment again; and the scrub returns [`VectorStoreError::Corrupted`].
pub fn scrub(dir: &Path, cancel: &CancellationToken) -> Result<ScrubReport, SemanticSearchError> {
    let started = std::time::Instant::now();
    let set = SegmentSet::open(dir)?;
    let mut bytes = 0u64;
    for (segment, entry) in set.segments.iter().zip(&set.document.segments) {
        if let Err(error) = segment.verify_blocks(cancel, |block| bytes += block) {
            if let VectorStoreError::Corrupted { reason } = &error {
                // The bytes that failed are the ones mapped, whatever the path holds now.
                let failed = files::sha256_hex(segment.file_bytes());
                if files::condemn(dir, &entry.id, &failed, reason)? {
                    log::error!("Scrub of {}: {reason}", dir.display());
                } else {
                    log::warn!(
                        "Scrub of {}: {reason} — in bytes segment {} no longer holds, so its \
                         verdict is withdrawn",
                        dir.display(),
                        entry.id
                    );
                }
            }
            return Err(error.into());
        }
    }
    Ok(ScrubReport {
        generation: set.generation(),
        segments: set.segments.len() as u32,
        bytes_checked: bytes,
        elapsed_ms: started.elapsed().as_millis() as u64,
    })
}

/// A directory holding a v1 artifact is a store this build does not read, and says so by
/// the identity field that names the store.
fn refuse_a_v1_artifact(dir: &Path) -> Result<(), ArtifactError> {
    let manifest = dir.join(crate::distribution::package::MANIFEST_FILENAME);
    let Ok(bytes) = fs::read(&manifest) else {
        return Ok(());
    };
    let backend = serde_json::from_slice::<serde_json::Value>(&bytes)
        .ok()
        .and_then(|value| {
            value["identity"]["store"]["backend_id"]
                .as_str()
                .map(str::to_string)
        })
        .unwrap_or_else(|| "unknown".to_string());
    Err(ArtifactError::IdentityMismatch {
        mismatches: vec![IdentityMismatch {
            field: IdentityField::StoreBackendId,
            artifact: backend,
            expected: STORE_BACKEND_ID.to_string(),
        }],
    })
}

/// Clean up after a crash: `staging/`, half-written pointers, and garbage. Called with the
/// lock held.
pub(crate) fn recover(dir: &Path) -> Result<(), ArtifactError> {
    let staging = dir.join(STAGING_DIR);
    if staging.exists() {
        fs::remove_dir_all(&staging)
            .map_err(io_error(format!("removing {}", staging.display())))?;
    }
    for name in [format!("{CURRENT}.tmp"), format!("{PREVIOUS}.tmp")] {
        let path = dir.join(name);
        if path.exists() {
            fs::remove_file(&path).map_err(io_error(format!("removing {}", path.display())))?;
        }
    }
    collect_garbage(dir);
    Ok(())
}

/// Remove generations neither pointer names and segments only they used. Best effort: a
/// file that cannot be removed — mapped by a reader, on Windows — is left for the next
/// call, and a pointer that cannot be read stops the collection altogether, since what it
/// names cannot be known.
pub(crate) fn collect_garbage(dir: &Path) {
    let mut live_generations = BTreeSet::new();
    let mut live_segments = BTreeSet::new();
    for name in [CURRENT, PREVIOUS] {
        match read_pointer(dir, name) {
            Ok(Some(pointer)) => match read_generation(dir, &pointer) {
                Ok(document) => {
                    live_generations.insert(generation_dir(pointer.generation));
                    live_segments.extend(document.segments.iter().map(|entry| entry.id.clone()));
                }
                // A generation that does not read names nothing that can be kept or
                // removed with certainty.
                Err(_) => return,
            },
            Ok(None) => {}
            Err(_) => return,
        }
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with("gen-") && !live_generations.contains(&name) {
            if let Err(error) = fs::remove_dir_all(entry.path()) {
                log::debug!("Garbage {} stays for now: {error}", entry.path().display());
            }
        }
    }
    let Ok(entries) = fs::read_dir(dir.join(SEGMENTS_DIR)) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        let id = name.split('.').next().unwrap_or_default();
        if !live_segments.contains(id) {
            if let Err(error) = fs::remove_file(entry.path()) {
                // Expected on Windows while a reader maps the segment.
                log::debug!("Garbage {} stays for now: {error}", entry.path().display());
            }
        }
    }
}
