//! The on-disk shape of an official artifact, and the checks that must pass before
//! anything inside it is read.
//!
//! An artifact is a directory holding the vector payload files of the chosen store
//! backend plus two metadata files: `manifest.json` (identity, counts, total size) and
//! `payloads.json` (a SHA-256 **and a size** per payload). Nothing here decides *what* the payload
//! looks like — that belongs to the store backend, and its format version is part of
//! the identity so a reader refuses a layout it cannot read.
//!
//! [`IndexPackage::verify_for_install`] and [`IndexPackage::verify_for_open`] are the only
//! ways to obtain a [`VerifiedPackage`], and
//! [`OfficialSemanticIndex`](crate::semantic::official_index::OfficialSemanticIndex) is what
//! consumes one: it opens the payload from the token — never from a path a caller supplied —
//! which is what makes "verify before reading a vector" a property of the types rather than
//! a call order someone has to remember.
//!
//! # Two depths, because one of them runs at every startup
//!
//! Hashing the payload is affordable once, at install. It is *not* affordable on every
//! open: at library scale the payload is measured in gigabytes, and re-hashing it would
//! turn every launch into a full read of the index. So the two entry points prove
//! different things, and [`VerifiedPackage::depth`] records which one ran — a reader
//! must never report more than was actually checked.
//!
//! # What verification does not prove
//!
//! `payloads.json` travels *inside* the package it describes. Recomputing it therefore
//! proves the package is self-consistent and undamaged — not that it is the artifact we
//! published. A payload replaced together with its checksum passes. The only thing that
//! separates the official artifact from a self-consistent impostor is a digest published
//! **outside** the package, which is why [`ArtifactExpectation`] carries one and why
//! omitting it has an explicit name.
//!
//! That anchor only reaches the payload because the reader completes the chain: this layer
//! pins the *declared* hashes at open without recomputing them, and the store reader
//! compares each file it loads against them while reading it. Neither half is sufficient
//! alone — a digest over declarations nobody checks proves nothing about the bytes, and
//! checks over bytes with no external anchor prove only self-consistency.
//!
//! See `docs/ARTIFACT_CONTRACT.md`.

use crate::errors::ArtifactError;
use crate::semantic::versioning::{IdentityField, IndexVersion};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};

/// Version of the artifact *metadata* documents — this module's two JSON files.
///
/// Distinct from the identity fields it carries: `store.store_format_version` versions
/// the payload, `corpus.tantivy_schema_version` versions the lexical index, and this
/// versions the envelope that declares them. Bump it when the documents change shape.
/// Version 2 replaced the bare `name → sha256` map with [`PayloadDescriptor`], because a
/// checksum alone left the cheap open path unable to check anything per file: it could
/// only compare the *sum* of the sizes, and two payloads whose lengths changed in
/// opposite directions cancelled out.
pub const ARTIFACT_METADATA_VERSION: u32 = 2;

pub const MANIFEST_FILENAME: &str = "manifest.json";
/// Renamed from `checksums.json` in metadata version 2: the file carries sizes as well.
pub const PAYLOADS_FILENAME: &str = "payloads.json";

/// What the manifest declares about one payload file.
///
/// The size is not decoration and not redundant with the checksum. It is what
/// [`IndexPackage::verify_for_open`] can check without reading the file, and it has to be
/// **per payload**: a package holding three files — which `ZevcStore` already does — can
/// otherwise lose bytes from one and gain them in another while the total stays right.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PayloadDescriptor {
    /// SHA-256 of the file, as 64 hex digits.
    pub sha256: String,
    pub size_bytes: u64,
}

impl PayloadDescriptor {
    /// Describe a file that is already on disk.
    pub fn of_file(path: &Path) -> Result<Self, ArtifactError> {
        let metadata =
            fs::metadata(path).map_err(io_error(format!("inspecting {}", path.display())))?;
        Ok(Self {
            sha256: sha256_file(path)?,
            size_bytes: metadata.len(),
        })
    }

    /// Describe bytes that have not been written yet — for a builder assembling a package
    /// in memory, and for tests.
    pub fn of_bytes(bytes: &[u8]) -> Self {
        Self {
            sha256: format!("{:x}", Sha256::digest(bytes)),
            size_bytes: bytes.len() as u64,
        }
    }
}

/// What the artifact declares about itself.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageManifest {
    pub metadata_version: u32,
    /// The corpus, model and store this artifact belongs to. Verified in full before
    /// any payload is read — see [`IndexVersion`].
    pub identity: IndexVersion,
    pub created_at: String,
    pub book_count: u32,
    pub vector_count: u32,
    /// Sum of the payload file sizes. Checked against the files, so a manifest that
    /// describes a different package than the one it ships with is a rejection.
    pub total_size_bytes: u64,
}

impl PackageManifest {
    pub fn new(
        identity: IndexVersion,
        created_at: String,
        book_count: u32,
        vector_count: u32,
        total_size_bytes: u64,
    ) -> Self {
        Self {
            metadata_version: ARTIFACT_METADATA_VERSION,
            identity,
            created_at,
            book_count,
            vector_count,
            total_size_bytes,
        }
    }

    /// Refuse a manifest that declares nothing.
    ///
    /// What this cannot check is the payload's *content*: that it holds exactly
    /// `vector_count` vectors across `book_count` books is a claim only a reader of the
    /// store format can settle, and that check lands with the read-only open path and
    /// the packer. So this is the floor, not the whole agreement.
    fn validate_counts(&self) -> Result<(), ArtifactError> {
        for (label, count) in [
            ("book_count", self.book_count),
            ("vector_count", self.vector_count),
        ] {
            if count == 0 {
                return Err(ArtifactError::ManifestDisagreesWithPayload {
                    reason: format!("{label} is zero, so there is nothing to open"),
                });
            }
        }
        Ok(())
    }
}

/// What the installation requires of an artifact.
///
/// `identity` is what this installation *is*: the corpus identity of the Tantivy index
/// that is actually open, the model identity of the model file that is actually loaded.
/// It is never this crate's constant.
///
/// `published_digest` is the trust anchor — see [`IndexPackage::digest`]. Without it,
/// verification detects damage and the wrong artifact but cannot detect a deliberately
/// rebuilt one, so the constructor that omits it says so in its name.
#[derive(Debug, Clone)]
pub struct ArtifactExpectation {
    pub identity: IndexVersion,
    pub published_digest: Option<String>,
}

impl ArtifactExpectation {
    /// Verify against a digest published outside the package. The only form that
    /// distinguishes the official artifact from a self-consistent impostor.
    pub fn with_published_digest(identity: IndexVersion, published_digest: String) -> Self {
        Self {
            identity,
            published_digest: Some(published_digest),
        }
    }

    /// Identity and integrity only.
    ///
    /// Named for what it gives up: this detects a damaged package and a package built
    /// for another corpus or model, and it does **not** establish that the package is
    /// the one we published. Correct for a locally built artifact and for a development
    /// fixture; not sufficient for one that arrived over a network.
    pub fn without_published_digest(identity: IndexVersion) -> Self {
        Self {
            identity,
            published_digest: None,
        }
    }
}

/// How much of an artifact was actually checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationDepth {
    /// Metadata, identity, digest, and that every payload is present, is a regular
    /// file, and has the declared size. No payload byte was read.
    MetadataAndPresence,
    /// The above plus a SHA-256 over every payload byte.
    FullPayload,
}

/// An artifact's metadata, read but not yet verified against an installation.
pub struct IndexPackage {
    pub manifest: PackageManifest,
    pub payloads: BTreeMap<String, PayloadDescriptor>,
}

impl IndexPackage {
    /// Write the metadata for a payload that is already in `root`.
    ///
    /// Refuses to write metadata the runtime would reject: a foreign metadata version, an
    /// incomplete identity, a payload that is missing, or one whose size or checksum does
    /// not match what the manifest declares. A package that writes "successfully" without
    /// being checked is exactly the package that fails on the user's machine.
    ///
    /// A manifest carrying another `metadata_version` is refused rather than restamped.
    /// Restamping looked harmless and was not: the document on disk got the current
    /// version while the caller's object kept the old one, so a publisher that wrote a
    /// package and *then* asked it for [`Self::digest`] would announce a digest belonging
    /// to no file on disk.
    pub fn write(root: &Path, package: &IndexPackage) -> Result<(), ArtifactError> {
        if package.manifest.metadata_version != ARTIFACT_METADATA_VERSION {
            return Err(ArtifactError::UnsupportedMetadataVersion {
                found: package.manifest.metadata_version,
                supported: ARTIFACT_METADATA_VERSION,
            });
        }

        fs::create_dir_all(root).map_err(io_error(format!("creating {}", root.display())))?;

        package.manifest.identity.validate_complete()?;
        package.verify_integrity(root)?;

        write_json(&root.join(MANIFEST_FILENAME), &package.manifest)?;
        write_json(&root.join(PAYLOADS_FILENAME), &package.payloads)?;

        // The metadata is what the next open reads first; unflushed, a power loss can
        // leave a directory of good payloads that no longer describes itself.
        sync_dir(root).map_err(io_error(format!("flushing {}", root.display())))
    }

    /// Digest over everything the artifact claims about itself: the metadata version,
    /// every identity field in [`IdentityField::ALL`] order, the counts, the declared
    /// total size, and every payload's name, checksum and size.
    ///
    /// This is the value that can be published *outside* the package and compared
    /// against it, which is the only way to tell the official artifact from one rebuilt
    /// with matching internal checksums. Computed over a canonical text rather than over
    /// the JSON bytes, so re-serializing the same metadata cannot change it.
    ///
    /// `created_at` is excluded on purpose: it is a timestamp, not part of what the
    /// artifact *is*, and excluding it lets the same build produce the same digest
    /// twice. It is therefore also the one field a published digest does not pin.
    pub fn digest(&self) -> String {
        let mut canonical = String::from("otzaria-artifact-digest-v1\n");
        canonical.push_str(&format!(
            "metadata_version={}\n",
            self.manifest.metadata_version
        ));
        for field in IdentityField::ALL {
            canonical.push_str(&format!(
                "{}={}\n",
                field.path(),
                self.manifest.identity.value(field)
            ));
        }
        canonical.push_str(&format!("book_count={}\n", self.manifest.book_count));
        canonical.push_str(&format!("vector_count={}\n", self.manifest.vector_count));
        canonical.push_str(&format!(
            "total_size_bytes={}\n",
            self.manifest.total_size_bytes
        ));
        for (payload, descriptor) in &self.payloads {
            canonical.push_str(&format!(
                "payload {payload}={} {}\n",
                descriptor.sha256, descriptor.size_bytes
            ));
        }
        format!("{:x}", Sha256::digest(canonical.as_bytes()))
    }

    /// Read the two metadata files. Verifies nothing about the payload and nothing
    /// about the installation — use [`Self::verify_for_install`] or
    /// [`Self::verify_for_open`] for that.
    pub fn read(root: &Path) -> Result<Self, ArtifactError> {
        let manifest_path = root.join(MANIFEST_FILENAME);
        let manifest_json = read_to_string(&manifest_path)?;

        // The version before the document, so a foreign metadata format reports its
        // version instead of a confusing field-level parse error.
        let probe: MetadataVersionProbe =
            serde_json::from_str(&manifest_json).map_err(parse_error(&manifest_path))?;
        if probe.metadata_version != ARTIFACT_METADATA_VERSION {
            return Err(ArtifactError::UnsupportedMetadataVersion {
                found: probe.metadata_version,
                supported: ARTIFACT_METADATA_VERSION,
            });
        }

        let manifest: PackageManifest =
            serde_json::from_str(&manifest_json).map_err(parse_error(&manifest_path))?;

        let payloads_path = root.join(PAYLOADS_FILENAME);
        let payloads_json = read_to_string(&payloads_path)?;
        let payloads: BTreeMap<String, PayloadDescriptor> =
            serde_json::from_str(&payloads_json).map_err(parse_error(&payloads_path))?;
        validate_payload_table(&payloads)?;

        Ok(Self { manifest, payloads })
    }

    /// Full verification, including a SHA-256 over every payload byte.
    ///
    /// For installing a package and for publishing one — the moments when reading the
    /// whole artifact once is the cheapest thing about the operation. **Not** for
    /// opening an installed artifact; see [`Self::verify_for_open`].
    ///
    /// The order is deliberate — cheapest and most actionable first:
    ///
    /// 1. metadata version, so a foreign document is not parsed leniently;
    /// 2. identity completeness, because two unfilled identities agree;
    /// 3. identity against the expectation, which is the difference between "wrong
    ///    artifact" and "damaged artifact" for the user;
    /// 4. the published digest, if there is one;
    /// 5. payload integrity, which reads every byte in the package.
    pub fn verify_for_install(
        root: &Path,
        expected: &ArtifactExpectation,
    ) -> Result<VerifiedPackage, ArtifactError> {
        let package = Self::verify_metadata(root, expected)?;
        package.verify_integrity(root)?;
        Ok(package.into_verified(root, VerificationDepth::FullPayload))
    }

    /// Verification that does not read a payload byte: metadata, identity, published
    /// digest, and that every payload is present, is a regular file and has the
    /// declared size.
    ///
    /// This is the check an already-installed artifact gets on every open. It is
    /// deliberately weaker than [`Self::verify_for_install`], because the alternative is
    /// re-hashing gigabytes at every launch — and a check nobody can afford to run is a
    /// check that gets skipped. What it still catches: the wrong artifact, a truncated
    /// or replaced-with-different-length payload, a payload that vanished, and metadata
    /// that no longer describes its files.
    ///
    /// What it does not catch is a same-length payload edit. Detecting that at open time
    /// is the store backend's job, on the structures it actually reads.
    pub fn verify_for_open(
        root: &Path,
        expected: &ArtifactExpectation,
    ) -> Result<VerifiedPackage, ArtifactError> {
        let package = Self::verify_metadata(root, expected)?;
        package.verify_payload_presence(root)?;
        Ok(package.into_verified(root, VerificationDepth::MetadataAndPresence))
    }

    /// Everything that can be decided from the metadata alone. Shared by both depths so
    /// that neither can drift into checking less than the other.
    fn verify_metadata(root: &Path, expected: &ArtifactExpectation) -> Result<Self, ArtifactError> {
        let package = Self::read(root)?;
        package.manifest.identity.validate_complete()?;
        package
            .manifest
            .identity
            .verify_matches(&expected.identity)?;

        if let Some(published) = &expected.published_digest {
            let actual = package.digest();
            if &actual != published {
                return Err(ArtifactError::UnexpectedArtifactDigest {
                    expected: published.clone(),
                    actual,
                });
            }
        }

        Ok(package)
    }

    fn into_verified(self, root: &Path, depth: VerificationDepth) -> VerifiedPackage {
        VerifiedPackage {
            root: root.to_path_buf(),
            digest: self.digest(),
            manifest: self.manifest,
            payloads: self.payloads,
            depth,
        }
    }

    /// Payload names, per-file SHA-256, and the manifest's agreement with the files.
    ///
    /// Separate from [`Self::verify_for_install`] because it is also what the writer runs
    /// before publishing a package and what the importer runs again on the staged copy.
    pub fn verify_integrity(&self, root: &Path) -> Result<(), ArtifactError> {
        self.walk_payloads(root, VerificationDepth::FullPayload)
    }

    /// Presence, type and declared size — no payload byte is read. See
    /// [`Self::verify_for_open`] for why this depth exists.
    fn verify_payload_presence(&self, root: &Path) -> Result<(), ArtifactError> {
        self.walk_payloads(root, VerificationDepth::MetadataAndPresence)
    }

    /// The one walk both depths share, so the cheap check cannot quietly stop covering
    /// something the expensive one covers. The depth decides one thing only: whether the
    /// bytes are read.
    fn walk_payloads(&self, root: &Path, depth: VerificationDepth) -> Result<(), ArtifactError> {
        validate_payload_table(&self.payloads)?;
        self.manifest.validate_counts()?;

        let mut payload_bytes = 0u64;
        for (payload, declared) in &self.payloads {
            let path = root.join(payload);
            let metadata = fs::symlink_metadata(&path).map_err(|error| {
                if error.kind() == io::ErrorKind::NotFound {
                    ArtifactError::PayloadMissing {
                        payload: payload.clone(),
                    }
                } else {
                    io_error(format!("inspecting {}", path.display()))(error)
                }
            })?;
            let file_type = metadata.file_type();
            if !file_type.is_file() || file_type.is_symlink() {
                return Err(ArtifactError::PayloadNotRegularFile {
                    payload: payload.clone(),
                });
            }
            // Per payload, not only in the total: with more than one file — and
            // `ZevcStore` already writes three — bytes lost from one and gained by another
            // leave the sum untouched.
            if metadata.len() != declared.size_bytes {
                return Err(ArtifactError::ManifestDisagreesWithPayload {
                    reason: format!(
                        "{payload:?} is declared as {} bytes but holds {}",
                        declared.size_bytes,
                        metadata.len()
                    ),
                });
            }
            payload_bytes = payload_bytes.saturating_add(metadata.len());

            if depth == VerificationDepth::FullPayload {
                let actual = sha256_file(&path)?;
                if actual != declared.sha256 {
                    return Err(ArtifactError::PayloadChecksumFailed {
                        payload: payload.clone(),
                        expected: declared.sha256.clone(),
                        actual,
                    });
                }
            }
        }

        if payload_bytes != self.manifest.total_size_bytes {
            return Err(ArtifactError::ManifestDisagreesWithPayload {
                reason: format!(
                    "total_size_bytes is {}, but the {} payload file(s) hold {payload_bytes}",
                    self.manifest.total_size_bytes,
                    self.payloads.len()
                ),
            });
        }

        Ok(())
    }
}

/// Proof that an artifact matched this installation.
///
/// Only [`IndexPackage::verify_for_install`] and [`IndexPackage::verify_for_open`]
/// construct one, and there is no public constructor, so a reader that asks for a
/// `VerifiedPackage` cannot be handed an unverified directory path by mistake.
///
/// [`Self::depth`] is part of the token because the two entry points prove different
/// things. A caller that reported "verified" without consulting it would be claiming a
/// payload was hashed when it may only have been stat'ed.
#[derive(Debug, Clone)]
pub struct VerifiedPackage {
    root: PathBuf,
    manifest: PackageManifest,
    payloads: BTreeMap<String, PayloadDescriptor>,
    digest: String,
    depth: VerificationDepth,
}

impl VerifiedPackage {
    /// Directory the verification was performed against. A payload replaced after this
    /// point is outside what the token claims.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// How much was actually checked. See [`VerificationDepth`].
    pub fn depth(&self) -> VerificationDepth {
        self.depth
    }

    /// The artifact's own digest — [`IndexPackage::digest`]. Equal to the published
    /// digest when the expectation carried one; otherwise only what this package
    /// computes about itself.
    pub fn artifact_digest(&self) -> &str {
        &self.digest
    }

    pub fn manifest(&self) -> &PackageManifest {
        &self.manifest
    }

    pub fn identity(&self) -> &IndexVersion {
        &self.manifest.identity
    }

    /// Payload name → what the manifest declares about it, in name order. Whether the
    /// hashes were recomputed depends on [`Self::depth`]; the sizes were checked either
    /// way.
    pub fn payloads(&self) -> &BTreeMap<String, PayloadDescriptor> {
        &self.payloads
    }

    pub fn payload_names(&self) -> Vec<&str> {
        self.payloads.keys().map(String::as_str).collect()
    }

    pub fn book_count(&self) -> u32 {
        self.manifest.book_count
    }

    pub fn vector_count(&self) -> u32 {
        self.manifest.vector_count
    }
}

/// Longest payload name accepted. Below every filesystem limit this crate can land on,
/// and long enough for any name a builder needs.
const PAYLOAD_NAME_MAX_BYTES: usize = 255;

/// Names Windows treats as devices whatever the extension. A package carrying `nul.bin`
/// installs on Unix and cannot be created on Windows.
const WINDOWS_RESERVED_STEMS: [&str; 22] = [
    "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8",
    "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
];

/// Characters a payload name may contain. Everything else is refused.
///
/// An allowlist rather than a blocklist, because the dangerous case is not a known-bad
/// character but platform-dependent parsing: `Path::components` reads `a\b.bin` as one
/// file name on Unix and as a two-segment path on Windows, so a package written on macOS
/// could be refused — or resolved as a path — on Windows. A name restricted to this set
/// means the same thing everywhere.
fn is_portable_name_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-')
}

/// A payload name must be a portable single file name, and must not be one of the
/// metadata files.
///
/// This blocks `../` from escaping the package directory, blocks a payload from
/// overwriting the manifest that describes it, and keeps a package written on one
/// platform readable on the others. Checked on the *string*, not through `Path`,
/// precisely because `Path` answers differently per platform.
pub(crate) fn validate_payload_name(filename: &str) -> Result<(), ArtifactError> {
    let refuse = |reason: &str| {
        Err(ArtifactError::UnsafePayloadName {
            name: filename.to_string(),
            reason: reason.to_string(),
        })
    };

    if filename.is_empty() {
        return refuse("is empty");
    }
    if filename.len() > PAYLOAD_NAME_MAX_BYTES {
        return refuse("is longer than 255 bytes");
    }
    if let Some(byte) = filename.bytes().find(|byte| !is_portable_name_byte(*byte)) {
        // `/`, `\`, `:`, control characters, spaces and non-ASCII all land here.
        return refuse(&format!(
            "contains {:?}, which is not one of A-Z a-z 0-9 . _ -",
            byte as char
        ));
    }
    if filename.starts_with('.') {
        // Covers `.` and `..` as well as hidden names.
        return refuse("starts with a dot");
    }
    if filename.ends_with('.') {
        return refuse("ends with a dot, which Windows drops");
    }
    // Compared case-insensitively because that is how Windows and a default macOS volume
    // compare them: `MANIFEST.JSON` passed this check and then overwrote `manifest.json`.
    if filename.eq_ignore_ascii_case(MANIFEST_FILENAME)
        || filename.eq_ignore_ascii_case(PAYLOADS_FILENAME)
    {
        return refuse("would overwrite the artifact metadata");
    }

    let stem = filename
        .split('.')
        .next()
        .unwrap_or(filename)
        .to_uppercase();
    if WINDOWS_RESERVED_STEMS.contains(&stem.as_str()) {
        return refuse("is a reserved device name on Windows");
    }

    Ok(())
}

fn validate_payload_table(
    payloads: &BTreeMap<String, PayloadDescriptor>,
) -> Result<(), ArtifactError> {
    if payloads.is_empty() {
        return Err(ArtifactError::NoPayload);
    }

    let mut seen_lowercase: BTreeMap<String, &String> = BTreeMap::new();
    for (filename, descriptor) in payloads {
        validate_payload_name(filename)?;

        // Two entries differing only in case are two entries here and one file on Windows
        // and on a default macOS volume: the second copy overwrites the first, and
        // whichever checksum lost the race fails for no visible reason.
        if let Some(other) = seen_lowercase.insert(filename.to_ascii_lowercase(), filename) {
            return Err(ArtifactError::UnsafePayloadName {
                name: filename.clone(),
                reason: format!(
                    "differs from {other:?} only in case, and they are the same file on a \
                     case-insensitive filesystem"
                ),
            });
        }

        if descriptor.sha256.len() != 64
            || !descriptor
                .sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(ArtifactError::MalformedPayloadChecksum {
                payload: filename.clone(),
            });
        }
    }
    Ok(())
}

/// Streamed, because a payload is sized for a library and not for memory.
fn sha256_file(path: &Path) -> Result<String, ArtifactError> {
    let mut file = File::open(path).map_err(io_error(format!("reading {}", path.display())))?;
    let mut hasher = Sha256::new();
    io::copy(&mut file, &mut hasher).map_err(io_error(format!("hashing {}", path.display())))?;
    Ok(format!("{:x}", hasher.finalize()))
}

fn read_to_string(path: &Path) -> Result<String, ArtifactError> {
    fs::read_to_string(path).map_err(|error| ArtifactError::MetadataUnusable {
        path: path.display().to_string(),
        reason: error.to_string(),
    })
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), ArtifactError> {
    let json =
        serde_json::to_vec_pretty(value).map_err(|error| ArtifactError::MetadataUnusable {
            path: path.display().to_string(),
            reason: format!("could not be serialized: {error}"),
        })?;
    write_and_sync(path, &json).map_err(io_error(format!("writing {}", path.display())))
}

/// Write a file and flush it, so a power loss cannot leave a metadata document that the
/// directory entry claims exists.
pub(crate) fn write_and_sync(path: &Path, bytes: &[u8]) -> io::Result<()> {
    use std::io::Write;
    let mut file = File::create(path)?;
    file.write_all(bytes)?;
    file.sync_all()
}

/// Flush a directory entry so a rename or a create inside it survives a power loss.
///
/// Unix only — Windows cannot open a directory as a file, so there the filesystem's own
/// ordering is what we have. Documented rather than silently skipped, the same way
/// [`SemanticManifest`](crate::semantic::manifest::SemanticManifest) treats it.
#[cfg(unix)]
pub(crate) fn sync_dir(path: &Path) -> io::Result<()> {
    File::open(path)?.sync_all()
}

/// See the Unix implementation.
#[cfg(not(unix))]
pub(crate) fn sync_dir(_path: &Path) -> io::Result<()> {
    Ok(())
}

fn parse_error(path: &Path) -> impl FnOnce(serde_json::Error) -> ArtifactError + '_ {
    move |error| ArtifactError::MetadataUnusable {
        path: path.display().to_string(),
        reason: format!("is not the JSON this build expects: {error}"),
    }
}

fn io_error(context: String) -> impl FnOnce(io::Error) -> ArtifactError {
    move |source| ArtifactError::Io { context, source }
}

/// Minimal view used to read `metadata_version` before the full document.
#[derive(Deserialize)]
struct MetadataVersionProbe {
    metadata_version: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic::versioning::{test_identity, IdentityField, IdentityMismatch};

    struct TempDir(PathBuf);
    impl TempDir {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "otzaria_test_pkg_{name}_{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            let _ = std::fs::create_dir_all(&path);
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

    /// A complete, self-consistent package with one payload, ready to be verified.
    fn write_sample_package(root: &Path, payload: &[u8]) -> IndexPackage {
        fs::create_dir_all(root).unwrap();
        fs::write(root.join("vectors.bin"), payload).unwrap();

        let payloads = BTreeMap::from([(
            "vectors.bin".to_string(),
            PayloadDescriptor::of_bytes(payload),
        )]);
        let package = IndexPackage {
            manifest: PackageManifest::new(
                test_identity(),
                "2026-08-06T00:00:00Z".to_string(),
                10,
                100,
                payload.len() as u64,
            ),
            payloads,
        };
        IndexPackage::write(root, &package).unwrap();
        package
    }

    fn expected_identity() -> IndexVersion {
        test_identity()
    }

    fn expectation() -> ArtifactExpectation {
        ArtifactExpectation::without_published_digest(test_identity())
    }

    #[test]
    fn a_written_package_reads_back_and_verifies() {
        let dir = TempDir::new("rw");
        write_sample_package(dir.path(), b"vectors");

        let package = IndexPackage::read(dir.path()).unwrap();
        assert_eq!(package.manifest.metadata_version, ARTIFACT_METADATA_VERSION);
        assert_eq!(package.manifest.book_count, 10);
        assert_eq!(package.manifest.vector_count, 100);

        let verified = IndexPackage::verify_for_install(dir.path(), &expectation()).unwrap();
        assert_eq!(verified.payload_names(), ["vectors.bin"]);
        assert_eq!(verified.payloads().len(), 1);
        assert_eq!(verified.vector_count(), 100);
        assert_eq!(verified.book_count(), 10);
        assert_eq!(verified.root(), dir.path());
        assert!(verified.identity().is_compatible(&expected_identity()));
    }

    /// The gate: a package built for another corpus is refused, and the rejection says
    /// which field disagreed rather than "incompatible".
    #[test]
    fn a_package_from_another_corpus_is_refused_by_name() {
        let dir = TempDir::new("wrong_corpus");
        write_sample_package(dir.path(), b"vectors");

        let mut expected = expected_identity();
        expected.corpus.corpus_id = "d".repeat(64);

        match IndexPackage::verify_for_install(
            dir.path(),
            &ArtifactExpectation::without_published_digest(expected),
        ) {
            Err(ArtifactError::IdentityMismatch { mismatches }) => assert_eq!(
                mismatches,
                vec![IdentityMismatch {
                    field: IdentityField::CorpusId,
                    artifact: "c".repeat(64),
                    expected: "d".repeat(64),
                }]
            ),
            other => panic!("expected an identity rejection, got {other:?}"),
        }
    }

    /// Identity is checked before the payload is read: the user's next action differs
    /// ("get the right artifact" versus "download this one again"), and hashing a
    /// library-sized payload to reach a verdict already available is wasted work.
    #[test]
    fn identity_is_refused_before_the_payload_is_hashed() {
        let dir = TempDir::new("order");
        write_sample_package(dir.path(), b"vectors");
        // Corrupt the payload *and* mismatch the identity.
        fs::write(dir.path().join("vectors.bin"), b"tampered").unwrap();

        let mut expected = expected_identity();
        expected.model.model_checksum = "b".repeat(64);

        assert!(matches!(
            IndexPackage::verify_for_install(
                dir.path(),
                &ArtifactExpectation::without_published_digest(expected)
            ),
            Err(ArtifactError::IdentityMismatch { .. })
        ));
    }

    /// A builder describes files on disk; a test describes bytes. They must agree, or a
    /// package assembled one way would not verify when read the other.
    #[test]
    fn a_descriptor_of_a_file_matches_a_descriptor_of_its_bytes() {
        let dir = TempDir::new("descriptor");
        let bytes = b"some payload bytes";
        let path = dir.path().join("vectors.bin");
        fs::write(&path, bytes).unwrap();

        assert_eq!(
            PayloadDescriptor::of_file(&path).unwrap(),
            PayloadDescriptor::of_bytes(bytes)
        );
        assert_eq!(
            PayloadDescriptor::of_bytes(bytes).size_bytes,
            bytes.len() as u64
        );
    }

    /// A same-length edit: the declared size still matches, so only the checksum can
    /// catch it.
    #[test]
    fn a_tampered_payload_is_refused_by_checksum() {
        let dir = TempDir::new("tampered");
        write_sample_package(dir.path(), b"vectors");
        fs::write(dir.path().join("vectors.bin"), b"vectorZ").unwrap();

        match IndexPackage::verify_for_install(dir.path(), &expectation()) {
            Err(ArtifactError::PayloadChecksumFailed { payload, .. }) => {
                assert_eq!(payload, "vectors.bin")
            }
            other => panic!("expected a checksum rejection, got {other:?}"),
        }
    }

    /// The hole a per-payload size closes: with more than one file, bytes moved from one
    /// to another leave `total_size_bytes` correct. Only the per-file size sees it, and it
    /// sees it without reading a byte.
    #[test]
    fn payload_sizes_that_cancel_out_in_the_total_are_still_refused() {
        let dir = TempDir::new("compensated");
        let first = b"aaaaaaaa";
        let second = b"bb";

        fs::create_dir_all(dir.path()).unwrap();
        fs::write(dir.path().join("one.bin"), first).unwrap();
        fs::write(dir.path().join("two.bin"), second).unwrap();
        let package = IndexPackage {
            manifest: PackageManifest::new(
                test_identity(),
                "2026-08-06T00:00:00Z".to_string(),
                2,
                20,
                (first.len() + second.len()) as u64,
            ),
            payloads: BTreeMap::from([
                ("one.bin".to_string(), PayloadDescriptor::of_bytes(first)),
                ("two.bin".to_string(), PayloadDescriptor::of_bytes(second)),
            ]),
        };
        IndexPackage::write(dir.path(), &package).unwrap();
        assert!(IndexPackage::verify_for_open(dir.path(), &expectation()).is_ok());

        // One loses two bytes, the other gains two. The total is untouched.
        fs::write(dir.path().join("one.bin"), b"aaaaaa").unwrap();
        fs::write(dir.path().join("two.bin"), b"bbbb").unwrap();

        for verdict in [
            IndexPackage::verify_for_open(dir.path(), &expectation()),
            IndexPackage::verify_for_install(dir.path(), &expectation()),
        ] {
            match verdict {
                Err(ArtifactError::ManifestDisagreesWithPayload { reason }) => {
                    assert!(reason.contains("one.bin"), "{reason}")
                }
                other => panic!("compensated sizes must be refused, got {other:?}"),
            }
        }
    }

    /// On Windows and on a default macOS volume these are one file, so a package that
    /// declares both is a package that cannot be installed as declared.
    #[test]
    fn payload_names_that_differ_only_in_case_are_refused() {
        let dir = TempDir::new("case_collision");
        fs::create_dir_all(dir.path()).unwrap();
        fs::write(dir.path().join("vectors.bin"), b"vectors").unwrap();

        let package = IndexPackage {
            manifest: PackageManifest::new(
                test_identity(),
                "2026-08-06T00:00:00Z".to_string(),
                1,
                1,
                14,
            ),
            payloads: BTreeMap::from([
                (
                    "vectors.bin".to_string(),
                    PayloadDescriptor::of_bytes(b"vectors"),
                ),
                (
                    "VECTORS.BIN".to_string(),
                    PayloadDescriptor::of_bytes(b"vectors"),
                ),
            ]),
        };

        match IndexPackage::write(dir.path(), &package) {
            Err(ArtifactError::UnsafePayloadName { reason, .. }) => {
                assert!(reason.contains("only in case"), "{reason}")
            }
            other => panic!("a case collision must be refused, got {other:?}"),
        }

        // And a payload that would overwrite the metadata under a different case.
        for name in ["MANIFEST.JSON", "Payloads.Json"] {
            match validate_payload_name(name) {
                Err(ArtifactError::UnsafePayloadName { reason, .. }) => assert!(
                    reason.contains("overwrite the artifact metadata"),
                    "{name}: {reason}"
                ),
                other => panic!("{name} must be refused, got {other:?}"),
            }
        }
    }

    /// A publisher writes a package and then announces its digest. If `write` quietly
    /// restamped the metadata version, that digest would describe no file on disk.
    #[test]
    fn write_refuses_a_foreign_metadata_version_rather_than_restamping_it() {
        let dir = TempDir::new("foreign_version");
        let package = write_sample_package(dir.path(), b"vectors");

        let stale = IndexPackage {
            manifest: PackageManifest {
                metadata_version: ARTIFACT_METADATA_VERSION - 1,
                ..package.manifest.clone()
            },
            payloads: package.payloads.clone(),
        };
        match IndexPackage::write(dir.path(), &stale) {
            Err(ArtifactError::UnsupportedMetadataVersion { found, supported }) => {
                assert_eq!(found, ARTIFACT_METADATA_VERSION - 1);
                assert_eq!(supported, ARTIFACT_METADATA_VERSION);
            }
            other => panic!("a foreign metadata version must be refused, got {other:?}"),
        }

        // What the publisher announces is what the files say.
        let written = IndexPackage::read(dir.path()).unwrap();
        assert_eq!(written.digest(), package.digest());
    }

    #[test]
    fn a_missing_payload_is_refused_as_missing_not_as_an_io_error() {
        let dir = TempDir::new("missing_payload");
        write_sample_package(dir.path(), b"vectors");
        fs::remove_file(dir.path().join("vectors.bin")).unwrap();

        match IndexPackage::verify_for_install(dir.path(), &expectation()) {
            Err(ArtifactError::PayloadMissing { payload }) => assert_eq!(payload, "vectors.bin"),
            other => panic!("expected a missing-payload rejection, got {other:?}"),
        }
    }

    /// A manifest is a claim about the payload. A package whose declared size does not
    /// match its files describes a different package, even if every checksum passes.
    #[test]
    fn a_manifest_that_understates_the_payload_is_refused() {
        let dir = TempDir::new("size_lie");
        let package = write_sample_package(dir.path(), b"vectors");

        let lying = IndexPackage {
            manifest: PackageManifest {
                total_size_bytes: 1,
                ..package.manifest.clone()
            },
            payloads: package.payloads.clone(),
        };
        write_json(&dir.path().join(MANIFEST_FILENAME), &lying.manifest).unwrap();

        match IndexPackage::verify_for_install(dir.path(), &expectation()) {
            Err(ArtifactError::ManifestDisagreesWithPayload { reason }) => {
                assert!(reason.contains("total_size_bytes"), "{reason}")
            }
            other => panic!("expected a manifest/payload rejection, got {other:?}"),
        }

        // And the writer refuses to publish it in the first place.
        assert!(matches!(
            IndexPackage::write(dir.path(), &lying),
            Err(ArtifactError::ManifestDisagreesWithPayload { .. })
        ));
    }

    #[test]
    fn a_manifest_declaring_nothing_is_refused() {
        let dir = TempDir::new("zero_counts");
        let package = write_sample_package(dir.path(), b"vectors");

        for (label, manifest) in [
            (
                "books",
                PackageManifest {
                    book_count: 0,
                    ..package.manifest.clone()
                },
            ),
            (
                "vectors",
                PackageManifest {
                    vector_count: 0,
                    ..package.manifest.clone()
                },
            ),
        ] {
            write_json(&dir.path().join(MANIFEST_FILENAME), &manifest).unwrap();
            assert!(
                matches!(
                    IndexPackage::verify_for_install(dir.path(), &expectation()),
                    Err(ArtifactError::ManifestDisagreesWithPayload { .. })
                ),
                "a manifest with no {label} must be refused"
            );
        }
    }

    #[test]
    fn an_incomplete_identity_is_refused_even_against_an_equally_incomplete_installation() {
        let dir = TempDir::new("blank_identity");
        let package = write_sample_package(dir.path(), b"vectors");

        let mut blank = package.manifest.clone();
        blank.identity.corpus.corpus_id = String::new();
        write_json(&dir.path().join(MANIFEST_FILENAME), &blank).unwrap();

        let mut equally_blank = expected_identity();
        equally_blank.corpus.corpus_id = String::new();

        match IndexPackage::verify_for_install(
            dir.path(),
            &ArtifactExpectation::without_published_digest(equally_blank),
        ) {
            Err(ArtifactError::IncompleteIdentity { field, .. }) => {
                assert_eq!(field, IdentityField::CorpusId)
            }
            other => panic!("a blank corpus id must not open anything, got {other:?}"),
        }
    }

    #[test]
    fn metadata_from_another_format_version_reports_its_version() {
        let dir = TempDir::new("metadata_version");
        write_sample_package(dir.path(), b"vectors");

        for version in [0, ARTIFACT_METADATA_VERSION + 1] {
            let mut document: serde_json::Value = serde_json::from_str(
                &fs::read_to_string(dir.path().join(MANIFEST_FILENAME)).unwrap(),
            )
            .unwrap();
            document["metadata_version"] = version.into();
            fs::write(
                dir.path().join(MANIFEST_FILENAME),
                serde_json::to_vec_pretty(&document).unwrap(),
            )
            .unwrap();

            match IndexPackage::verify_for_install(dir.path(), &expectation()) {
                Err(ArtifactError::UnsupportedMetadataVersion { found, supported }) => {
                    assert_eq!(found, version);
                    assert_eq!(supported, ARTIFACT_METADATA_VERSION);
                }
                other => panic!("expected a metadata-version rejection, got {other:?}"),
            }
        }
    }

    #[test]
    fn unreadable_or_truncated_metadata_is_refused() {
        let dir = TempDir::new("unusable");

        assert!(matches!(
            IndexPackage::read(dir.path()),
            Err(ArtifactError::MetadataUnusable { .. })
        ));

        fs::write(dir.path().join(MANIFEST_FILENAME), b"{ not json").unwrap();
        assert!(matches!(
            IndexPackage::read(dir.path()),
            Err(ArtifactError::MetadataUnusable { .. })
        ));

        // Right version, nothing else — must not deserialize into defaults.
        fs::write(
            dir.path().join(MANIFEST_FILENAME),
            format!("{{\"metadata_version\": {ARTIFACT_METADATA_VERSION}}}"),
        )
        .unwrap();
        assert!(matches!(
            IndexPackage::read(dir.path()),
            Err(ArtifactError::MetadataUnusable { .. })
        ));
    }

    #[test]
    fn rejects_path_traversal_and_a_package_with_no_payload() {
        let dir = TempDir::new("unsafe_names");
        let manifest = PackageManifest::new(
            expected_identity(),
            "2026-08-06T00:00:00Z".to_string(),
            1,
            1,
            0,
        );

        let empty = IndexPackage {
            manifest: manifest.clone(),
            payloads: BTreeMap::new(),
        };
        assert!(matches!(
            IndexPackage::write(dir.path(), &empty),
            Err(ArtifactError::NoPayload)
        ));

        for name in ["../outside.bin", "nested/vectors.bin", MANIFEST_FILENAME] {
            let package = IndexPackage {
                manifest: manifest.clone(),
                payloads: BTreeMap::from([(
                    name.to_string(),
                    PayloadDescriptor {
                        sha256: "0".repeat(64),
                        size_bytes: 0,
                    },
                )]),
            };
            assert!(
                matches!(
                    IndexPackage::write(dir.path(), &package),
                    Err(ArtifactError::UnsafePayloadName { .. })
                ),
                "{name} must be refused as a payload name"
            );
        }

        let malformed = IndexPackage {
            manifest,
            payloads: BTreeMap::from([(
                "vectors.bin".to_string(),
                PayloadDescriptor {
                    sha256: "not-a-hash".to_string(),
                    size_bytes: 7,
                },
            )]),
        };
        assert!(matches!(
            IndexPackage::write(dir.path(), &malformed),
            Err(ArtifactError::MalformedPayloadChecksum { .. })
        ));
    }

    /// The hole the published digest exists to close: a package whose payload was
    /// replaced *together with* its checksum is perfectly self-consistent, so every
    /// internal check passes. Only a digest that travelled outside the package can tell
    /// the official artifact from that one.
    #[test]
    fn only_a_published_digest_separates_the_official_artifact_from_a_rebuilt_one() {
        let dir = TempDir::new("published_digest");

        let official = dir.path().join("official");
        let package = write_sample_package(&official, b"vectors");
        let published = package.digest();

        let verified = IndexPackage::verify_for_install(
            &official,
            &ArtifactExpectation::with_published_digest(test_identity(), published.clone()),
        )
        .unwrap();
        assert_eq!(verified.artifact_digest(), published);

        // A different artifact, rebuilt to be internally consistent: same identity, same
        // length, its own matching checksum.
        let rebuilt = dir.path().join("rebuilt");
        write_sample_package(&rebuilt, b"vectorZ");

        assert!(
            IndexPackage::verify_for_install(&rebuilt, &expectation()).is_ok(),
            "self-consistency is all the in-package checks can prove — this is the gap"
        );

        match IndexPackage::verify_for_install(
            &rebuilt,
            &ArtifactExpectation::with_published_digest(test_identity(), published.clone()),
        ) {
            Err(ArtifactError::UnexpectedArtifactDigest { expected, actual }) => {
                assert_eq!(expected, published);
                assert_ne!(actual, published);
            }
            other => panic!("a rebuilt artifact must fail the published digest, got {other:?}"),
        }
    }

    /// The digest has to survive a round trip through JSON, or a published value would
    /// stop matching the moment the metadata is rewritten — which the installer does on
    /// every install.
    #[test]
    fn the_digest_is_stable_across_re_serialization_and_moves_with_the_identity() {
        let dir = TempDir::new("digest_stability");
        let package = write_sample_package(dir.path(), b"vectors");
        let digest = package.digest();

        assert_eq!(IndexPackage::read(dir.path()).unwrap().digest(), digest);

        // `created_at` is excluded on purpose — see `IndexPackage::digest`.
        let restamped = IndexPackage {
            manifest: PackageManifest {
                created_at: "2030-01-01T00:00:00Z".to_string(),
                ..package.manifest.clone()
            },
            payloads: package.payloads.clone(),
        };
        assert_eq!(restamped.digest(), digest);

        // Everything that is identity does move it.
        let mut other_corpus = test_identity();
        other_corpus.corpus.corpus_id = "d".repeat(64);
        let elsewhere = IndexPackage {
            manifest: PackageManifest {
                identity: other_corpus,
                ..package.manifest.clone()
            },
            payloads: package.payloads.clone(),
        };
        assert_ne!(elsewhere.digest(), digest);
    }

    /// Opening an installed artifact must not re-read it. At library scale the payload is
    /// gigabytes, and a check that costs a full read on every launch is a check that gets
    /// turned off.
    #[test]
    fn opening_an_installed_artifact_checks_metadata_and_presence_but_reads_no_payload() {
        let dir = TempDir::new("open_depth");
        write_sample_package(dir.path(), b"vectors");

        let opened = IndexPackage::verify_for_open(dir.path(), &expectation()).unwrap();
        assert_eq!(opened.depth(), VerificationDepth::MetadataAndPresence);
        assert_eq!(
            IndexPackage::verify_for_install(dir.path(), &expectation())
                .unwrap()
                .depth(),
            VerificationDepth::FullPayload
        );

        // A same-length edit is exactly what this depth cannot see. Stated as a test so
        // the limit is a documented property and not a surprise.
        fs::write(dir.path().join("vectors.bin"), b"vectorZ").unwrap();
        assert!(IndexPackage::verify_for_open(dir.path(), &expectation()).is_ok());
        assert!(matches!(
            IndexPackage::verify_for_install(dir.path(), &expectation()),
            Err(ArtifactError::PayloadChecksumFailed { .. })
        ));

        // What it does see: a length that no longer matches the manifest, and a payload
        // that is gone.
        fs::write(dir.path().join("vectors.bin"), b"longer than before").unwrap();
        assert!(matches!(
            IndexPackage::verify_for_open(dir.path(), &expectation()),
            Err(ArtifactError::ManifestDisagreesWithPayload { .. })
        ));
        fs::remove_file(dir.path().join("vectors.bin")).unwrap();
        assert!(matches!(
            IndexPackage::verify_for_open(dir.path(), &expectation()),
            Err(ArtifactError::PayloadMissing { .. })
        ));
    }

    /// A payload name has to mean the same thing on every platform. `Path` does not
    /// guarantee that — `a\\b.bin` is one file name on Unix and a path on Windows — so the
    /// rule is checked on the string.
    #[test]
    fn payload_names_are_refused_identically_on_every_platform() {
        for (name, expected_reason) in [
            ("", "is empty"),
            ("../outside.bin", "not one of"),
            ("nested/vectors.bin", "not one of"),
            ("nested\\vectors.bin", "not one of"),
            ("C:vectors.bin", "not one of"),
            ("vec tors.bin", "not one of"),
            ("וקטורים.bin", "not one of"),
            ("vectors\n.bin", "not one of"),
            (".hidden.bin", "starts with a dot"),
            ("..", "starts with a dot"),
            ("vectors.", "ends with a dot"),
            (MANIFEST_FILENAME, "overwrite the artifact metadata"),
            (PAYLOADS_FILENAME, "overwrite the artifact metadata"),
            ("nul.bin", "reserved device name"),
            ("COM1", "reserved device name"),
            ("lpt9.payload", "reserved device name"),
        ] {
            match validate_payload_name(name) {
                Err(ArtifactError::UnsafePayloadName { reason, .. }) => assert!(
                    reason.contains(expected_reason),
                    "{name:?}: expected {expected_reason:?}, got {reason:?}"
                ),
                other => panic!("{name:?} must be refused, got {other:?}"),
            }
        }

        assert!(validate_payload_name(&"v".repeat(256)).is_err());
        for name in [
            "vectors.bin",
            "v",
            "book-1_of-2.zevc",
            "v".repeat(255).as_str(),
        ] {
            assert!(
                validate_payload_name(name).is_ok(),
                "{name:?} must be accepted"
            );
        }
    }

    /// Following a symlink would let a package's checksum describe a file outside it.
    #[cfg(unix)]
    #[test]
    fn a_symlinked_payload_is_refused_rather_than_followed() {
        let dir = TempDir::new("symlink");
        let root = dir.path().join("package");
        write_sample_package(&root, b"vectors");

        let outside = dir.path().join("outside.bin");
        fs::write(&outside, b"vectors").unwrap();
        fs::remove_file(root.join("vectors.bin")).unwrap();
        std::os::unix::fs::symlink(&outside, root.join("vectors.bin")).unwrap();

        match IndexPackage::verify_for_install(&root, &expectation()) {
            Err(ArtifactError::PayloadNotRegularFile { payload }) => {
                assert_eq!(payload, "vectors.bin")
            }
            other => panic!("expected a payload-type rejection, got {other:?}"),
        }
    }
}
