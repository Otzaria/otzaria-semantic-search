//! Installing a published segment: a base that replaces the set, or a delta applied to it.
//!
//! The order is the spec's §1.6, and every step before the flip leaves the live generation
//! untouched, so a crash anywhere before it is recovered by removing what was written:
//!
//! 1. **lock** the set;
//! 2. **check that the release applies**: its manifest is the published one, its identity
//!    is this installation's, its package digest is its own, and a delta starts where the
//!    set stands and shares its codec epoch;
//! 3. **stage and verify** the segment: its SHA-256 and size, its structure, its header
//!    against the manifest, every block CRC;
//! 4. **move** it into `segments/`;
//! 5. **resolve keys**, for a delta: one sequential pass over the older segments' keys marks
//!    the slots its tombstones and its own slots supersede, and finds where its foreign
//!    records resolve;
//! 6. **write a new generation**, numbered past every one on disk — its `.del` and `.links`
//!    files and `set.json`;
//! 7. **flip** — `PREVIOUS` ← the generation the install was built on, the one the set
//!    opened at, then `CURRENT`;
//! 8. **collect garbage**.

use super::files::{
    encode_links, generation_dir, generation_path, hash_file, io_error, next_generation,
    place_segment, read_pointer, segment_file, sha256_hex, sync_file, sync_set_dir,
    write_atomically, write_pointer, Deleted, DerivedFile, Pointer, SetDocument, SetLock,
    SetSegment, SetStats, CURRENT, INCOMING_DIR, PREVIOUS, SEGMENTS_DIR, SET_FILE, SET_FORMAT,
    SET_FORMAT_VERSION, STAGING_DIR,
};
use super::{collect_garbage, recover, space, CompactionPolicy, SegmentSet};
use crate::cancellation::CancellationToken;
use crate::distribution::package::{
    utc_timestamp, IndexPackage, PackageCounts, PackageDescription, PackageKind, PackageManifest,
    PayloadDescriptor,
};
use crate::errors::{ArtifactError, SemanticSearchError, VectorStoreError};
use crate::semantic::chunk_key::ChunkKey;
use crate::semantic::oxv::reader::Segment;
use crate::semantic::oxv::scan::{Link, LINK_UNRESOLVED};
use crate::semantic::oxv::writer::WrittenSegment;
use crate::semantic::versioning::{hex, IndexVersion, VectorProvenance};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::Path;

/// `format` of a release manifest.
pub const RELEASE_FORMAT: &str = "otzaria-vectors-release";
/// `formatVersion` of a release manifest.
pub const RELEASE_FORMAT_VERSION: u32 = 1;

/// The name the segment carries in the package a release manifest describes.
const PACKAGE_PAYLOAD: &str = "segment.oxv";

/// The manifest published beside a segment: what it is, what it holds, and the digests
/// that tie the two together.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReleaseManifest {
    pub format: String,
    pub format_version: u32,
    pub kind: PackageKind,
    pub identity: IndexVersion,
    pub identity_digest: String,
    pub codec_params_sha256: String,
    pub from_library_version: u32,
    pub to_library_version: u32,
    pub library_release_tag: String,
    pub segment_id: String,
    pub counts: PackageCounts,
    /// The uncompressed segment.
    pub segment: ReleaseSegment,
    /// [`IndexPackage::digest`] of the package this manifest describes.
    pub package_digest: String,
    pub provenance: VectorProvenance,
    /// The files a client downloads, compressed or split; the client's business.
    #[serde(default)]
    pub files: Vec<ReleaseFile>,
    /// What the release requires of an installation, for the client; not read here.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requires: Option<serde_json::Value>,
    /// Which build produced it, for people.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub built_by: Option<serde_json::Value>,
    pub created_at: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReleaseSegment {
    pub sha256: String,
    pub size: u64,
}

/// One downloadable file, in the shape the updater's patch entries have.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReleaseFile {
    pub file: String,
    pub compression: String,
    pub sha256: String,
    pub size: u64,
    pub uncompressed_sha256: String,
    pub uncompressed_size: u64,
}

impl ReleaseManifest {
    /// The manifest of a segment a build just wrote: everything but the download files,
    /// which depend on how it is compressed and split for publication.
    pub fn for_segment(
        written: &WrittenSegment,
        identity: &IndexVersion,
        codec_params_sha256: [u8; 32],
        provenance: VectorProvenance,
        created_at: String,
    ) -> Self {
        let mut manifest = Self {
            format: RELEASE_FORMAT.to_string(),
            format_version: RELEASE_FORMAT_VERSION,
            kind: written.spec.kind,
            identity: identity.clone(),
            identity_digest: identity.identity_digest_hex(),
            codec_params_sha256: hex(&codec_params_sha256),
            from_library_version: written.spec.from_library_version,
            to_library_version: written.spec.to_library_version,
            library_release_tag: written.spec.library_release_tag.clone(),
            segment_id: hex(&written.segment_id),
            counts: written.counts,
            segment: ReleaseSegment {
                sha256: hex(&written.sha256),
                size: written.size,
            },
            package_digest: String::new(),
            provenance,
            files: Vec::new(),
            requires: None,
            built_by: None,
            created_at,
        };
        manifest.package_digest = manifest.package().digest();
        manifest
    }

    /// The package this manifest describes — `manifest.json` and `payloads.json` around
    /// `segment.oxv`, metadata version 3 — whose digest it carries.
    pub fn package(&self) -> IndexPackage {
        IndexPackage {
            manifest: PackageManifest::new(
                self.identity.clone(),
                PackageDescription {
                    kind: self.kind,
                    from_library_version: self.from_library_version,
                    to_library_version: self.to_library_version,
                    library_release_tag: self.library_release_tag.clone(),
                    counts: self.counts,
                },
                self.provenance.clone(),
                self.created_at.clone(),
                self.segment.size,
            ),
            payloads: BTreeMap::from([(
                PACKAGE_PAYLOAD.to_string(),
                PayloadDescriptor {
                    sha256: self.segment.sha256.clone(),
                    size_bytes: self.segment.size,
                },
            )]),
        }
    }

    /// As published: pretty JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("a manifest serializes")
    }
}

/// What to install: the uncompressed segment, and the release manifest published beside
/// it, as published — its bytes are what a published digest names.
#[derive(Debug, Clone, Copy)]
pub struct InstallSource<'a> {
    /// Inside [`incoming_dir`](super::incoming_dir) it is moved into the set; anywhere else
    /// it is copied and left where it is.
    pub segment: &'a Path,
    pub manifest_json: &'a str,
}

/// What this installation requires of a release.
#[derive(Debug, Clone)]
pub struct InstallExpectation {
    /// What this installation is: the line recipe of its index, the model family it runs —
    /// with, as `query_packages`, the packages it will embed queries with, each of which
    /// the release must accept — and the store this build reads.
    pub identity: IndexVersion,
    /// The SHA-256 of the release manifest, published outside it. Without it an install
    /// detects damage and the wrong release, not a deliberately rebuilt one.
    pub published_manifest_sha256: Option<String>,
}

/// What an install did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApplyReport {
    pub kind: PackageKind,
    /// The set's library version afterwards.
    pub library_version: u32,
    pub generation: u64,
    pub segments: u32,
    pub slots_added: u64,
    /// Older slots the release's tombstones deleted.
    pub tombstones_applied: u64,
    /// Older slots of keys the release shipped again, deleted in favour of its own.
    pub duplicates_removed: u64,
    /// Foreign records whose key no older segment holds live.
    pub foreign_unresolved: u64,
    pub bytes_on_disk: u64,
    pub needs_compaction: bool,
    /// The set already stood at or past the release's version; nothing changed.
    pub already_applied: bool,
}

/// The points an install can be cut off at, for the tests that prove each one recoverable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Step {
    Staged,
    Moved,
    GenerationWritten,
    PreviousWritten,
    CurrentFlipped,
}

#[cfg(test)]
thread_local! {
    pub(crate) static CRASH_AT: std::cell::Cell<Option<Step>> = const { std::cell::Cell::new(None) };
}

/// Stop here, as a crash would, when a test asked for it; nothing outside a test build.
fn reached(step: Step) -> Result<(), ArtifactError> {
    #[cfg(test)]
    if CRASH_AT.with(|crash| crash.get()) == Some(step) {
        return Err(ArtifactError::InterruptedInstall {
            reason: format!("a crash injected after {step:?}"),
        });
    }
    let _ = step;
    Ok(())
}

/// Install a published segment into the set at `dir`: a base replaces the set; a delta is
/// applied to it. See the module documentation for the steps.
///
/// # Errors
///
/// [`ArtifactError::UnexpectedArtifactDigest`] for a manifest that is not the published
/// one; [`ArtifactError::IdentityMismatch`] for a release of another identity;
/// [`ArtifactError::DeltaDoesNotApply`] for a delta that does not start where the set
/// stands or is of another codec epoch; [`ArtifactError::PayloadChecksumFailed`] and
/// [`ArtifactError::ManifestDisagreesWithPayload`] for a segment that is not the one the
/// manifest describes; [`VectorStoreError::Corrupted`] for one whose bytes are damaged;
/// [`ArtifactError::InsufficientSpace`]; [`SemanticSearchError::Cancelled`].
pub fn install_package(
    dir: &Path,
    source: &InstallSource<'_>,
    expect: &InstallExpectation,
    cancel: &CancellationToken,
) -> Result<ApplyReport, SemanticSearchError> {
    // 1. The lock, and whatever a crash left behind.
    let _lock = SetLock::take(dir)?;
    recover(dir)?;

    // 2. The release applies.
    if let Some(published) = &expect.published_manifest_sha256 {
        let actual = sha256_hex(source.manifest_json.as_bytes());
        if &actual != published {
            return Err(ArtifactError::UnexpectedArtifactDigest {
                expected: published.clone(),
                actual,
            }
            .into());
        }
    }
    let manifest: ReleaseManifest =
        serde_json::from_str(source.manifest_json).map_err(|error| {
            ArtifactError::MetadataUnusable {
                path: source.segment.display().to_string(),
                reason: format!("its release manifest is not one this build reads: {error}"),
            }
        })?;
    check_manifest(&manifest, expect)?;

    // The set as `open` sees it — `CURRENT`'s generation, or `PREVIOUS`'s when that one does
    // not open — whenever either pointer is there to read, or to fail to.
    let installed = !matches!(
        (read_pointer(dir, CURRENT), read_pointer(dir, PREVIOUS)),
        (Ok(None), Ok(None))
    );
    let current = installed.then(|| SegmentSet::open_unlocked(dir));
    let current = match (manifest.kind, current) {
        (PackageKind::Delta, None) => {
            return Err(ArtifactError::DeltaDoesNotApply {
                field: "delta.from_library_version",
                reason: "no base is installed for it to apply to".to_string(),
            }
            .into())
        }
        (PackageKind::Delta, Some(set)) => Some(set?),
        // A base replaces whatever is there, readable or not.
        (_, set) => set.and_then(Result::ok),
    };
    if let (PackageKind::Delta, Some(set)) = (manifest.kind, &current) {
        if let Some(report) = check_delta_applies(&manifest, set)? {
            return Ok(report);
        }
    }

    // 3. Stage and verify.
    let staging = dir.join(STAGING_DIR);
    fs::create_dir_all(&staging).map_err(io_error(format!("creating {}", staging.display())))?;
    let staged = staging.join(format!("{}.oxv", manifest.segment_id));
    let incoming = dir.join(INCOMING_DIR);
    let moving = source
        .segment
        .parent()
        .and_then(|parent| parent.canonicalize().ok())
        .zip(incoming.canonicalize().ok())
        .is_some_and(|(parent, incoming)| parent == incoming);
    if !moving {
        let needed = manifest.segment.size;
        if let Some(available) = space::available(dir) {
            if available < needed {
                return Err(ArtifactError::InsufficientSpace { needed, available }.into());
            }
        }
    }
    let sha256 = if moving {
        fs::rename(source.segment, &staged).map_err(io_error(format!(
            "moving {} into the set",
            source.segment.display()
        )))?;
        // Written by the caller, who need not have flushed it; it is, before anything names
        // it. Opened for writing, which Windows needs to flush a file.
        OpenOptions::new()
            .write(true)
            .open(&staged)
            .and_then(|file| sync_file(&file, &staged))
            .map_err(io_error(format!("flushing {}", staged.display())))?;
        hash_file(&staged, cancel)?
    } else {
        copy_hashing(source.segment, &staged, manifest.segment.size, cancel)?
    };
    let size = fs::metadata(&staged)
        .map_err(io_error(format!("inspecting {}", staged.display())))?
        .len();
    if size != manifest.segment.size {
        return Err(ArtifactError::ManifestDisagreesWithPayload {
            reason: format!(
                "the segment is {size} bytes, and its manifest declares {}",
                manifest.segment.size
            ),
        }
        .into());
    }
    if sha256 != manifest.segment.sha256 {
        return Err(ArtifactError::PayloadChecksumFailed {
            payload: PACKAGE_PAYLOAD.to_string(),
            expected: manifest.segment.sha256.clone(),
            actual: sha256,
        }
        .into());
    }
    {
        let segment = Segment::open(&staged)?;
        check_header(&manifest, &segment)?;
        segment.verify_blocks(cancel, |_| {})?;
    }
    reached(Step::Staged)?;

    // 4. Move it into the set — beside every segment there is, and never over the bytes a
    // generation names.
    let file = segment_file(&manifest.segment_id);
    let target = dir.join(&file);
    place_segment(
        dir,
        &manifest.segment_id,
        &staged,
        &manifest.segment.sha256,
        cancel,
        |reason| ArtifactError::ManifestDisagreesWithPayload { reason }.into(),
    )?;
    let segments_dir = dir.join(SEGMENTS_DIR);
    let provenance_path = segments_dir.join(format!("{}.package.json", manifest.segment_id));
    write_atomically(&provenance_path, source.manifest_json.as_bytes())
        .map_err(space_or_io(&provenance_path, 0))?;
    reached(Step::Moved)?;

    // 5. Resolve keys, and 6–8: the new generation, the flip, the garbage.
    let segment = Segment::open(&target)?;
    let entry = SetSegment {
        id: manifest.segment_id.clone(),
        file,
        kind: manifest.kind,
        from: manifest.from_library_version,
        to: manifest.to_library_version,
        sha256: manifest.segment.sha256.clone(),
        size: manifest.segment.size,
        slots: manifest.counts.slots,
        package_digest: Some(manifest.package_digest.clone()),
        provenance: manifest.provenance.clone(),
        del: DerivedFile::default(),
        links: None,
    };
    let mut generation = match (manifest.kind, &current) {
        (PackageKind::Delta, Some(set)) => NewGeneration::from_set(set),
        _ => NewGeneration::empty(&manifest.identity, &manifest.codec_params_sha256),
    };
    let older: &[Segment] = match (manifest.kind, &current) {
        (PackageKind::Delta, Some(set)) => set.segments(),
        _ => &[],
    };
    let resolution = generation.push(entry, &segment, older, cancel)?;
    generation.library_release_tag = manifest.library_release_tag.clone();
    let base = current.as_ref().map(|set| set.pointer().clone());
    let document = generation.commit(dir, base.as_ref())?;

    let bytes_on_disk = document.stats.bytes;
    let info = super::info_of(
        &document,
        &document
            .segments
            .iter()
            .map(|entry| entry.del.count)
            .collect::<Vec<_>>(),
        &document
            .segments
            .iter()
            .map(|entry| entry.links.as_ref().map_or(0, |links| links.count))
            .collect::<Vec<_>>(),
        false,
    );
    log::info!(
        "Installed {} segment {} into {}: generation {}, library version {}, {} slot(s) added, \
         {} tombstone(s) applied, {} duplicate(s) removed, {} foreign record(s) unresolved",
        manifest.kind,
        manifest.segment_id,
        dir.display(),
        document.generation,
        document.library_version,
        manifest.counts.slots,
        resolution.tombstones_applied,
        resolution.duplicates_removed,
        resolution.foreign_unresolved
    );
    Ok(ApplyReport {
        kind: manifest.kind,
        library_version: document.library_version,
        generation: document.generation,
        segments: document.segments.len() as u32,
        slots_added: manifest.counts.slots,
        tombstones_applied: resolution.tombstones_applied,
        duplicates_removed: resolution.duplicates_removed,
        foreign_unresolved: resolution.foreign_unresolved,
        bytes_on_disk,
        needs_compaction: CompactionPolicy::default().wants(&info).is_some(),
        already_applied: false,
    })
}

/// The manifest is well formed, describes itself consistently, and names an identity
/// this installation accepts.
fn check_manifest(
    manifest: &ReleaseManifest,
    expect: &InstallExpectation,
) -> Result<(), ArtifactError> {
    let disagree = |reason: String| Err(ArtifactError::ManifestDisagreesWithPayload { reason });
    if manifest.format != RELEASE_FORMAT || manifest.format_version != RELEASE_FORMAT_VERSION {
        return Err(ArtifactError::MetadataUnusable {
            path: "release manifest".to_string(),
            reason: format!(
                "it is {} version {}, and this build reads {RELEASE_FORMAT} version \
                 {RELEASE_FORMAT_VERSION}",
                manifest.format, manifest.format_version
            ),
        });
    }
    manifest.identity.validate_complete()?;
    manifest.identity.verify_matches(&expect.identity)?;
    if manifest.identity_digest != manifest.identity.identity_digest_hex() {
        return disagree("its identity digest is not its identity's".to_string());
    }
    if manifest.kind == PackageKind::Compacted {
        return disagree("a compacted segment is made on a device and never published".into());
    }
    let package = manifest.package();
    // The floor every package is held to: a chain position that describes a segment, a
    // release tag a header can carry, a provenance that says something.
    package.manifest.validate()?;
    if package.digest() != manifest.package_digest {
        return disagree(
            "its package digest is not the digest of the package it describes".to_string(),
        );
    }
    Ok(())
}

/// A delta applies when it starts where the set stands, shares its identity and its
/// codec epoch. One the set has already absorbed is reported as such, and changes nothing.
fn check_delta_applies(
    manifest: &ReleaseManifest,
    set: &SegmentSet,
) -> Result<Option<ApplyReport>, ArtifactError> {
    let document = set.document();
    if manifest.identity_digest != document.identity_digest {
        let mismatches = manifest.identity.mismatches_against(&document.identity);
        if !mismatches.is_empty() {
            return Err(ArtifactError::IdentityMismatch { mismatches });
        }
        return Err(ArtifactError::DeltaDoesNotApply {
            field: "delta.identity",
            reason: format!(
                "its identity digest is {}, and the set's is {}",
                manifest.identity_digest, document.identity_digest
            ),
        });
    }
    if manifest.codec_params_sha256 != document.codec_params_sha256 {
        return Err(ArtifactError::DeltaDoesNotApply {
            field: "delta.codec_params",
            reason: format!(
                "it is of codec epoch {}, and the set of {}",
                manifest.codec_params_sha256, document.codec_params_sha256
            ),
        });
    }
    let stands = document.library_version;
    if manifest.to_library_version <= stands {
        let info = set.info();
        return Ok(Some(ApplyReport {
            kind: PackageKind::Delta,
            library_version: stands,
            generation: document.generation,
            segments: document.segments.len() as u32,
            slots_added: 0,
            tombstones_applied: 0,
            duplicates_removed: 0,
            foreign_unresolved: 0,
            bytes_on_disk: info.bytes_on_disk,
            needs_compaction: info.needs_compaction,
            already_applied: true,
        }));
    }
    if manifest.from_library_version != stands {
        return Err(ArtifactError::DeltaDoesNotApply {
            field: "delta.from_library_version",
            reason: format!(
                "it goes from version {} to {}, and the set stands at {stands}",
                manifest.from_library_version, manifest.to_library_version
            ),
        });
    }
    Ok(None)
}

/// The staged segment is the one its manifest describes.
fn check_header(manifest: &ReleaseManifest, segment: &Segment) -> Result<(), ArtifactError> {
    let disagree = |what: &str, header: String, declared: String| {
        Err(ArtifactError::ManifestDisagreesWithPayload {
            reason: format!(
                "the segment's {what} is {header}, and its manifest declares {declared}"
            ),
        })
    };
    if hex(&segment.segment_id()) != manifest.segment_id {
        return disagree(
            "id",
            hex(&segment.segment_id()),
            manifest.segment_id.clone(),
        );
    }
    if hex(&segment.identity_digest()) != manifest.identity_digest {
        return disagree(
            "identity digest",
            hex(&segment.identity_digest()),
            manifest.identity_digest.clone(),
        );
    }
    if hex(&segment.codec_params_sha256()) != manifest.codec_params_sha256 {
        return disagree(
            "codec epoch",
            hex(&segment.codec_params_sha256()),
            manifest.codec_params_sha256.clone(),
        );
    }
    if segment.codec().name() != manifest.identity.store.vector_precision {
        return disagree(
            "codec",
            segment.codec().name().to_string(),
            manifest.identity.store.vector_precision.clone(),
        );
    }
    if segment.kind() != manifest.kind
        || segment.from_library_version() != manifest.from_library_version
        || segment.to_library_version() != manifest.to_library_version
    {
        return disagree(
            "place in the chain",
            format!(
                "{} {}→{}",
                segment.kind(),
                segment.from_library_version(),
                segment.to_library_version()
            ),
            format!(
                "{} {}→{}",
                manifest.kind, manifest.from_library_version, manifest.to_library_version
            ),
        );
    }
    if segment.counts() != manifest.counts {
        return disagree(
            "counts",
            format!("{:?}", segment.counts()),
            format!("{:?}", manifest.counts),
        );
    }
    if segment.library_release_tag() != manifest.library_release_tag {
        return disagree(
            "release tag",
            segment.library_release_tag().to_string(),
            manifest.library_release_tag.clone(),
        );
    }
    Ok(())
}

/// What resolving a new segment's keys against the set did.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Resolution {
    pub tombstones_applied: u64,
    pub duplicates_removed: u64,
    pub foreign_unresolved: u64,
}

/// A generation being assembled: the segments, oldest first, with the derived files each
/// will have.
pub(crate) struct NewGeneration {
    pub identity: IndexVersion,
    pub identity_digest: String,
    pub codec_params_sha256: String,
    pub library_release_tag: String,
    pub entries: Vec<SetSegment>,
    pub deleted: Vec<Deleted>,
    pub links: Vec<Option<Vec<Link>>>,
}

impl NewGeneration {
    pub(crate) fn empty(identity: &IndexVersion, codec_params_sha256: &str) -> Self {
        Self {
            identity: identity.clone(),
            identity_digest: identity.identity_digest_hex(),
            codec_params_sha256: codec_params_sha256.to_string(),
            library_release_tag: String::new(),
            entries: Vec::new(),
            deleted: Vec::new(),
            links: Vec::new(),
        }
    }

    /// The live generation of `set`, carried over to be extended.
    pub(crate) fn from_set(set: &SegmentSet) -> Self {
        let document = set.document();
        Self {
            identity: document.identity.clone(),
            identity_digest: document.identity_digest.clone(),
            codec_params_sha256: document.codec_params_sha256.clone(),
            library_release_tag: document.library_release_tag.clone(),
            entries: document.segments.clone(),
            deleted: set.deleted().to_vec(),
            links: set
                .links()
                .iter()
                .zip(&document.segments)
                .map(|(links, entry)| entry.links.as_ref().map(|_| links.clone()))
                .collect(),
        }
    }

    /// Add `segment` as the newest: for a delta, resolve its tombstones, its own keys and
    /// its foreign records against `older` — the generation's segments so far, oldest
    /// first — in one sequential pass over their keys.
    pub(crate) fn push(
        &mut self,
        entry: SetSegment,
        segment: &Segment,
        older: &[Segment],
        cancel: &CancellationToken,
    ) -> Result<Resolution, SemanticSearchError> {
        debug_assert_eq!(older.len(), self.entries.len());
        let mut resolution = Resolution::default();
        let foreign = segment.foreign_count();
        let mut links = vec![
            Link {
                seg: 0,
                flags: LINK_UNRESOLVED,
                slot: 0,
            };
            foreign as usize
        ];
        if segment.kind() == PackageKind::Delta {
            // What the new segment wants found, by key.
            #[derive(Default)]
            struct Want {
                tombstone: bool,
                shipped: bool,
                foreign: Vec<u32>,
            }
            let mut wants: HashMap<ChunkKey, Want> = HashMap::new();
            for index in 0..segment.tombstone_count() {
                wants.entry(segment.tombstone(index)).or_default().tombstone = true;
            }
            for slot in 0..segment.slot_count() {
                let want = wants.entry(segment.key(slot)).or_default();
                if want.shipped {
                    return Err(VectorStoreError::Corrupted {
                        reason: format!(
                            "segment {} ships key {} twice",
                            entry.id,
                            segment.key(slot)
                        ),
                    }
                    .into());
                }
                want.shipped = true;
            }
            for index in 0..foreign {
                let (key, _, _) = segment.foreign_record(index);
                wants.entry(key).or_default().foreign.push(index);
            }
            for (seg, older) in older.iter().enumerate() {
                let keys = older.key_bytes();
                for (slot, key) in keys.as_chunks::<16>().0.iter().enumerate() {
                    if slot % (1 << 20) == 0 && cancel.is_cancelled() {
                        return Err(SemanticSearchError::Cancelled);
                    }
                    let slot = slot as u32;
                    if self.deleted[seg].is_set(slot) {
                        continue;
                    }
                    let Some(want) = wants.get(&ChunkKey(*key)) else {
                        continue;
                    };
                    if want.tombstone && self.deleted[seg].set(slot) {
                        resolution.tombstones_applied += 1;
                    } else if want.shipped && self.deleted[seg].set(slot) {
                        resolution.duplicates_removed += 1;
                    }
                    if !self.deleted[seg].is_set(slot) {
                        for index in &want.foreign {
                            links[*index as usize] = Link {
                                seg: seg as u16,
                                flags: 0,
                                slot,
                            };
                        }
                    }
                }
            }
            resolution.foreign_unresolved = links
                .iter()
                .filter(|link| link.flags & LINK_UNRESOLVED != 0)
                .count() as u64;
        }
        self.entries.push(entry);
        self.deleted
            .push(Deleted::none(u64::from(segment.slot_count())));
        self.links
            .push((segment.kind() == PackageKind::Delta).then_some(links));
        Ok(resolution)
    }

    /// Write a new generation — derived files, then `set.json` — flip the pointers to it, and
    /// collect garbage. `base` is the generation it was built on: the one the set opened at,
    /// `CURRENT`'s or, on a fallback, `PREVIOUS`'s; `None` when none opened. Returns what was
    /// written.
    ///
    /// The new generation is numbered past every one on disk, so it is written beside them
    /// and never over one; nothing that exists is touched before the flip, and a failure at
    /// any point leaves the pointers naming what they named.
    pub(crate) fn commit(
        mut self,
        dir: &Path,
        base: Option<&Pointer>,
    ) -> Result<SetDocument, SemanticSearchError> {
        let generation = next_generation(dir)?;
        let path = generation_path(dir, generation);
        fs::create_dir(&path).map_err(io_error(format!("creating {}", path.display())))?;
        let mut stats = SetStats::default();
        for ((entry, deleted), links) in self.entries.iter_mut().zip(&self.deleted).zip(&self.links)
        {
            let (bytes, crc32) = deleted.encode();
            let file = format!("{}.del", entry.id);
            let del_path = path.join(&file);
            write_atomically(&del_path, &bytes).map_err(space_or_io(&del_path, bytes.len()))?;
            entry.del = DerivedFile {
                file,
                crc32,
                count: deleted.count(),
            };
            entry.links = match links {
                Some(links) => {
                    let (bytes, crc32) = encode_links(links);
                    let file = format!("{}.links", entry.id);
                    let links_path = path.join(&file);
                    write_atomically(&links_path, &bytes)
                        .map_err(space_or_io(&links_path, bytes.len()))?;
                    let unresolved = links
                        .iter()
                        .filter(|link| link.flags & LINK_UNRESOLVED != 0)
                        .count() as u64;
                    stats.foreign += links.len() as u64;
                    stats.foreign_unresolved += unresolved;
                    Some(DerivedFile {
                        file,
                        crc32,
                        count: unresolved,
                    })
                }
                None => None,
            };
            stats.slots += entry.slots;
            stats.slots_dead += entry.del.count;
            stats.bytes += entry.size;
        }
        let newest = self.entries.last().expect("a generation has a segment");
        let document = SetDocument {
            format: SET_FORMAT.to_string(),
            format_version: SET_FORMAT_VERSION,
            generation,
            identity: self.identity,
            identity_digest: self.identity_digest,
            codec_params_sha256: self.codec_params_sha256,
            library_version: newest.to,
            library_release_tag: self.library_release_tag,
            segments: self.entries,
            stats,
            created_at: utc_timestamp(std::time::SystemTime::now()),
        };
        let bytes = serde_json::to_vec_pretty(&document).expect("a set serializes");
        let set_path = path.join(SET_FILE);
        write_atomically(&set_path, &bytes).map_err(space_or_io(&set_path, bytes.len()))?;
        // The generation's own entry, and `segments/`'s if this created it: durable before a
        // pointer names them.
        sync_set_dir(dir).map_err(io_error(format!("flushing {}", dir.display())))?;
        reached(Step::GenerationWritten)?;

        let pointer = Pointer {
            generation,
            set: format!("{}/{SET_FILE}", generation_dir(generation)),
            set_sha256: sha256_hex(&bytes),
        };
        // PREVIOUS: the generation this one was built on, or — when none opened — what
        // CURRENT names, if it reads. Never CURRENT's bytes as they are: an unreadable
        // CURRENT would overwrite the one pointer that still opens, and on a fallback
        // CURRENT names the generation that did not.
        let previous = match base {
            Some(base) => Some(base.clone()),
            None => read_pointer(dir, CURRENT).ok().flatten(),
        };
        if let Some(previous) = previous {
            if read_pointer(dir, PREVIOUS).ok().flatten().as_ref() != Some(&previous) {
                let bytes = serde_json::to_vec(&previous).expect("a pointer serializes");
                let path = dir.join(PREVIOUS);
                write_pointer(dir, PREVIOUS, &bytes).map_err(space_or_io(&path, bytes.len()))?;
            }
        }
        reached(Step::PreviousWritten)?;
        let bytes = serde_json::to_vec(&pointer).expect("a pointer serializes");
        let current = dir.join(CURRENT);
        write_pointer(dir, CURRENT, &bytes).map_err(space_or_io(&current, bytes.len()))?;
        reached(Step::CurrentFlipped)?;

        collect_garbage(dir);
        Ok(document)
    }
}

/// Copy `from` to `to` in blocks, hashing as it goes; a full filesystem is
/// [`ArtifactError::InsufficientSpace`], and the partial copy is removed either way.
fn copy_hashing(
    from: &Path,
    to: &Path,
    size: u64,
    cancel: &CancellationToken,
) -> Result<String, SemanticSearchError> {
    let result = (|| {
        let mut source =
            File::open(from).map_err(io_error(format!("reading {}", from.display())))?;
        let mut target =
            File::create(to).map_err(io_error(format!("creating {}", to.display())))?;
        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; 1 << 20];
        let mut written = 0u64;
        loop {
            if cancel.is_cancelled() {
                return Err(SemanticSearchError::Cancelled);
            }
            let read = source
                .read(&mut buffer)
                .map_err(io_error(format!("reading {}", from.display())))?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
            target
                .write_all(&buffer[..read])
                .map_err(|error| full_or_io(error, to, size, written))?;
            written += read as u64;
        }
        sync_file(&target, to).map_err(|error| full_or_io(error, to, size, written))?;
        Ok(format!("{:x}", hasher.finalize()))
    })();
    if result.is_err() {
        let _ = fs::remove_file(to);
    }
    result
}

/// A write that ran out of room, as the error that says so.
pub(crate) fn full_or_io(
    error: io::Error,
    path: &Path,
    needed: u64,
    written: u64,
) -> SemanticSearchError {
    if is_storage_full(&error) {
        ArtifactError::InsufficientSpace {
            needed,
            available: written,
        }
        .into()
    } else {
        io_error(format!("writing {}", path.display()))(error).into()
    }
}

pub(crate) fn space_or_io(
    path: &Path,
    needed: usize,
) -> impl FnOnce(io::Error) -> SemanticSearchError + '_ {
    move |error| full_or_io(error, path, needed as u64, 0)
}

pub(crate) fn is_storage_full(error: &io::Error) -> bool {
    matches!(
        error.kind(),
        io::ErrorKind::StorageFull | io::ErrorKind::QuotaExceeded
    )
}
