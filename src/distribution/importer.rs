//! Installing a verified artifact into the directory the runtime opens.
//!
//! The install is a file operation, not inference: verify the source, copy it to a
//! staging directory beside the target, verify the copy, then swap it in.
//!
//! # Why there is a recovery step
//!
//! Replacing a directory is not one atomic operation. `rename` refuses to replace a
//! non-empty directory on both Unix and Windows, so the swap is necessarily two renames:
//!
//! ```text
//! 1. target  → <name>.previous
//! 2. staging → target
//! ```
//!
//! Between them the target does not exist. A process killed in that window leaves the
//! only good copy under `<name>.previous`, and a reader that just looks at the target
//! sees nothing at all. That is why the intermediate names are **deterministic** and why
//! [`recover_interrupted_install`] exists: the state left by a crash is designed to be
//! readable, and resolving it is a documented step rather than a hope.
//!
//! Callers must run [`recover_interrupted_install`] before opening the target.
//! [`IndexImporter::import`] runs it for them before it installs, because writing over
//! an unresolved interruption would destroy the previous artifact — the one copy the
//! device still has.
//!
//! Durability: payload files are flushed before the swap, and the parent directory entry
//! is flushed after every rename (Unix only — Windows cannot open a directory as a file).
//! Without that, a power loss can undo a rename this code already reported as done.
//!
//! # What is out of scope
//!
//! Two installs into the same target at the same time. There is no lock; the second one
//! would recover away the first one's staging directory. The product installs from one
//! place, sequentially, and the alternative — a lock file with its own stale-lock
//! recovery — is not warranted before S6 says otherwise.
//!
//! Verification, on the other hand, is never optional here. An install is the one moment
//! when reading every byte is affordable, and the alternative is discovering a truncated
//! download at query time, where the only available answer is a wrong one.

use crate::distribution::package::{
    sync_dir, validate_payload_name, ArtifactExpectation, IndexPackage,
};
use crate::errors::ArtifactError;
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};

/// Suffix of the directory a half-finished install leaves the previous artifact under.
const PREVIOUS_SUFFIX: &str = "previous";

/// Suffix of the directory an install builds the new artifact in.
const STAGING_SUFFIX: &str = "staging";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportConfig {
    /// Directory holding the artifact to install.
    pub source_path: PathBuf,
    /// Directory the runtime opens. Replaced by a swap, never written into directly.
    pub target_store_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportResult {
    pub books_imported: u32,
    pub vectors_imported: u32,
    pub bytes_imported: u64,
    pub import_duration_ms: u128,
}

/// What [`recover_interrupted_install`] found and resolved. Every field false means the
/// target was in a clean state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct InstallRecovery {
    /// A crash between the two renames left no target; the previous artifact was moved
    /// back into place.
    pub restored_previous: bool,
    /// The swap had completed and only the cleanup had not; the superseded copy was
    /// removed.
    pub discarded_previous: bool,
    /// A staging directory from an install that never finished was removed.
    pub discarded_staging: bool,
}

impl InstallRecovery {
    pub fn recovered_anything(&self) -> bool {
        self.restored_previous || self.discarded_previous || self.discarded_staging
    }
}

/// The four paths an install works with, resolved and validated together.
///
/// One resolver for both [`IndexImporter::import`] and [`recover_interrupted_install`],
/// because recovery calls `remove_dir_all` and `rename` on the derived names: a target
/// whose file name cannot be determined — `.`, `foo/..`, a bare `/` — must be refused
/// *before* anything is derived from it, not after.
#[derive(Debug, Clone)]
struct InstallPaths {
    target: PathBuf,
    parent: PathBuf,
    previous: PathBuf,
    staging: PathBuf,
}

impl InstallPaths {
    fn resolve(target: &Path) -> Result<Self, ArtifactError> {
        let refuse = |reason: String| ArtifactError::InvalidInstallTarget { reason };

        // `file_name()` is None for `/`, `.` and anything ending in `..`, and those are
        // exactly the paths whose siblings would be somewhere unintended.
        let name = target
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| {
                refuse(format!(
                    "{} has no directory name of its own",
                    target.display()
                ))
            })?
            .to_string();
        if name == "." || name == ".." {
            return Err(refuse(format!(
                "{} names a relative directory, not an artifact",
                target.display()
            )));
        }

        // A single-component relative target ("semantic_db") has an *empty* parent, not
        // none. Canonicalizing or flushing "" fails, so it is normalized here, once, for
        // both callers.
        let parent = match target.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => parent.to_path_buf(),
            Some(_) => PathBuf::from("."),
            None => {
                return Err(refuse(format!(
                    "{} has no parent directory",
                    target.display()
                )))
            }
        };

        Ok(Self {
            target: parent.join(&name),
            previous: parent.join(format!(".{name}.{PREVIOUS_SUFFIX}")),
            staging: parent.join(format!(".{name}.{STAGING_SUFFIX}")),
            parent,
        })
    }

    /// Re-resolve against a canonical parent, so the swap and the guards work on absolute
    /// paths regardless of what the caller passed.
    fn canonicalized(&self) -> Result<Self, ArtifactError> {
        let mut resolved = Self::resolve(
            &canonicalize(&self.parent)?.join(
                self.target
                    .file_name()
                    .expect("resolve() guaranteed a file name"),
            ),
        )?;
        resolved.parent = canonicalize(&resolved.parent)?;
        Ok(resolved)
    }
}

/// Where a half-finished install parks the previous artifact.
///
/// Errors for a target that has no name of its own (`.`, `foo/..`, `/`): those are exactly
/// the paths whose derived siblings would land somewhere unintended, and recovery deletes
/// directories by these names.
pub fn previous_path(target: &Path) -> Result<PathBuf, ArtifactError> {
    Ok(InstallPaths::resolve(target)?.previous)
}

/// Where an install assembles the new artifact.
pub fn staging_path(target: &Path) -> Result<PathBuf, ArtifactError> {
    Ok(InstallPaths::resolve(target)?.staging)
}

/// Resolve whatever a crashed install left behind, and report what that was.
///
/// Must be called before the target is opened and before it is replaced. The two states
/// a crash can leave are distinguishable, which is the whole reason the names are fixed:
///
/// * `previous` exists and the target does not — the crash landed between the renames.
///   The previous artifact is restored, and the interrupted install has to be repeated.
///   Restoring rather than promoting the staged copy is deliberate: without an
///   expectation to verify against, this function cannot judge whether staging is
///   complete, and the previous artifact is known-good.
/// * `previous` and the target both exist — the swap finished and only the cleanup did
///   not. The superseded copy is removed.
///
/// Idempotent, and safe to call when nothing happened.
pub fn recover_interrupted_install(target: &Path) -> Result<InstallRecovery, ArtifactError> {
    let paths = InstallPaths::resolve(target)?;
    recover(&paths)
}

fn recover(paths: &InstallPaths) -> Result<InstallRecovery, ArtifactError> {
    let InstallPaths {
        target,
        parent,
        previous,
        staging,
    } = paths;

    let mut recovery = InstallRecovery::default();

    // Each change is flushed before the next step, not once at the end. Deferring the
    // flush made the restore's durability depend on the cleanup that follows it
    // succeeding: a failed `remove_dir` returned early, and the rename this process had
    // already observed was not necessarily on disk.
    if previous.exists() {
        if target.exists() {
            log::warn!(
                "Removing {}: the install it belonged to completed",
                previous.display()
            );
            remove_dir(previous)?;
            flush(parent)?;
            recovery.discarded_previous = true;
        } else {
            log::warn!(
                "No artifact at {}, but a previous copy is parked at {}; an install was \
                 interrupted. Restoring it.",
                target.display(),
                previous.display()
            );
            rename(previous, target).map_err(|error| ArtifactError::InterruptedInstall {
                reason: format!(
                    "could not restore {} to {}: {error}. The previous artifact is intact \
                     where it is; do not install over it until it is moved back",
                    previous.display(),
                    target.display()
                ),
            })?;
            flush(parent).map_err(|error| durability_unknown(previous, target, &error))?;
            recovery.restored_previous = true;
        }
    }

    if staging.exists() {
        log::warn!("Removing {}: an install left it behind", staging.display());
        remove_dir(staging)?;
        flush(parent)?;
        recovery.discarded_staging = true;
    }

    Ok(recovery)
}

/// A rename this process performed, whose durability could not be established.
///
/// Both outcomes a restart can show are recoverable — the artifact is either under
/// `target` (nothing to do) or still under `previous` (recovery moves it back) — so this
/// says exactly that instead of pretending the state is known. What it must not do is
/// report success, and what it must not be is a discarded error.
fn durability_unknown(previous: &Path, target: &Path, cause: &ArtifactError) -> ArtifactError {
    ArtifactError::InterruptedInstall {
        reason: format!(
            "{} was moved back to {} in this process, but the change could not be flushed to \
             disk ({cause}). After a restart the artifact may be at either path; calling \
             recovery again resolves both",
            previous.display(),
            target.display()
        ),
    }
}

pub struct IndexImporter {
    config: ImportConfig,
}

impl IndexImporter {
    pub fn new(config: ImportConfig) -> Self {
        Self { config }
    }

    /// Verify and install.
    ///
    /// `expected` is what this installation requires — see [`ArtifactExpectation`]. Both
    /// verifications matter and neither replaces the other: the source is verified before
    /// anything is copied, so a package that will be refused never touches the target's
    /// parent directory, and the staged copy is verified again, so a copy that lost bytes
    /// never becomes the live artifact.
    pub fn import(&self, expected: &ArtifactExpectation) -> Result<ImportResult, ArtifactError> {
        let start_time = Instant::now();

        let verified = IndexPackage::verify_for_install(&self.config.source_path, expected)?;
        let source = canonicalize(&self.config.source_path)?;

        // Validate the target before deriving anything from it, then re-resolve against a
        // canonical parent so the guards below compare absolute paths.
        let requested = InstallPaths::resolve(&self.config.target_store_path)?;
        fs::create_dir_all(&requested.parent)
            .map_err(io_error(format!("creating {}", requested.parent.display())))?;
        let paths = requested.canonicalized()?;
        let InstallPaths {
            target,
            parent,
            previous,
            staging,
        } = paths.clone();

        if target == source || target.starts_with(&source) {
            // Such an install would delete its own source during the swap.
            return Err(ArtifactError::InvalidInstallTarget {
                reason: "the target must not be the package directory or one of its children"
                    .to_string(),
            });
        }
        if target.exists() && !target.is_dir() {
            return Err(ArtifactError::InvalidInstallTarget {
                reason: format!("{} exists but is not a directory", target.display()),
            });
        }

        // Before anything is written: an unresolved interruption means the previous
        // artifact is parked under a recovery name, and installing over it would leave
        // the device with neither copy.
        recover(&paths)?;

        fs::create_dir(&staging).map_err(|error| {
            if error.kind() == io::ErrorKind::AlreadyExists {
                ArtifactError::InterruptedInstall {
                    reason: format!(
                        "{} appeared after recovery removed it; another install is running, \
                         which is not supported",
                        staging.display()
                    ),
                }
            } else {
                io_error(format!("creating {}", staging.display()))(error)
            }
        })?;

        let staged = (|| {
            for payload in verified.payload_names() {
                validate_payload_name(payload)?;
                let source_file = source.join(payload);
                let metadata = fs::symlink_metadata(&source_file)
                    .map_err(io_error(format!("inspecting {}", source_file.display())))?;
                let file_type = metadata.file_type();
                if !file_type.is_file() || file_type.is_symlink() {
                    return Err(ArtifactError::PayloadNotRegularFile {
                        payload: payload.to_string(),
                    });
                }
                copy_and_sync(&source_file, &staging.join(payload))
                    .map_err(io_error(format!("copying {payload} into staging")))?;
            }

            // Writing the metadata re-hashes the staged payloads and re-checks the
            // manifest against them, so this is the verification of the copy and not
            // only its metadata. It flushes the staging directory too.
            IndexPackage::write(
                &staging,
                &IndexPackage {
                    manifest: verified.manifest().clone(),
                    payloads: verified.payloads().clone(),
                },
            )?;

            swap_into_place(&staging, &target, &previous, &parent)
        })();

        if staged.is_err() && staging.exists() {
            let _ = fs::remove_dir_all(&staging);
            let _ = sync_dir(&parent);
        }
        staged?;

        Ok(ImportResult {
            books_imported: verified.book_count(),
            vectors_imported: verified.vector_count(),
            bytes_imported: verified.manifest().total_size_bytes,
            import_duration_ms: start_time.elapsed().as_millis(),
        })
    }
}

/// Swap `staging` in for `target`, flushing the parent's directory entry after each
/// rename so a reported success survives a power loss.
///
/// A failed swap puts the previous artifact back. If even that fails, the error says so
/// and names the directory the previous artifact is sitting in — it is not swallowed,
/// because at that point the device's only good copy is under a name nothing opens, and
/// [`recover_interrupted_install`] is what puts it back.
fn swap_into_place(
    staging: &Path,
    target: &Path,
    previous: &Path,
    parent: &Path,
) -> Result<(), ArtifactError> {
    if !target.exists() {
        rename(staging, target).map_err(io_error(format!(
            "moving staging into {}",
            target.display()
        )))?;
        return flush(parent);
    }

    if previous.exists() {
        return Err(ArtifactError::InterruptedInstall {
            reason: format!(
                "{} already exists; recovery should have resolved it",
                previous.display()
            ),
        });
    }

    rename(target, previous).map_err(io_error(format!(
        "moving {} aside to {}",
        target.display(),
        previous.display()
    )))?;

    // Everything from here runs with the previous artifact parked under a name nothing
    // opens. Each step can fail — including the flushes, which is not theoretical on a
    // full or dying disk — so they share one exit that either puts the previous artifact
    // back or says where it is. A bare `?` on the flush used to return a plain IO error
    // with no target directory on disk and no hint that the only good copy was next to it.
    let swapped = (|| -> Result<(), ArtifactError> {
        flush(parent)?;
        // The window the module documentation describes: no target exists here.
        rename(staging, target).map_err(io_error(format!(
            "moving staging into {}",
            target.display()
        )))?;
        flush(parent)
    })();

    if let Err(cause) = swapped {
        return Err(restore_parked(previous, target, parent, cause));
    }

    // From here the install is durable. A failure to clean up is not a failed install —
    // recovery removes the superseded copy on the next call.
    match fs::remove_dir_all(previous) {
        Ok(()) => flush(parent),
        Err(error) => {
            log::warn!(
                "Installed the artifact, but could not remove the superseded copy {}: {error}",
                previous.display()
            );
            Ok(())
        }
    }
}

/// Undo a swap that could not be completed, and return the error the caller should see.
///
/// If the target is already in place, the swap itself succeeded and only a flush did not:
/// the state on disk is consistent — `previous` beside a good `target`, which recovery
/// discards — so the original cause is returned unchanged. Rolling back there would be
/// wrong, and would fail anyway, since `rename` cannot replace a non-empty directory.
///
/// If the target is absent, the previous artifact is moved back. Only if *that* fails does
/// this become an [`ArtifactError::InterruptedInstall`], naming the directory that holds
/// the device's only good copy.
fn restore_parked(
    previous: &Path,
    target: &Path,
    parent: &Path,
    cause: ArtifactError,
) -> ArtifactError {
    if target.exists() {
        return cause;
    }
    match rename(previous, target) {
        Ok(()) => match flush(parent) {
            // The rollback is durable, so the caller's problem is the original failure.
            Ok(()) => cause,
            // It is not, and discarding that used to make "fsync after every rename" a
            // claim this code did not keep.
            Err(error) => durability_unknown(previous, target, &error),
        },
        Err(restore) => ArtifactError::InterruptedInstall {
            reason: format!(
                "the install could not be completed ({cause}), and the previous artifact could \
                 not be moved back from {} to {} ({restore}). The previous artifact is intact \
                 where it is; recovery restores it",
                previous.display(),
                target.display()
            ),
        },
    }
}

/// Copy a payload and flush it, so the swap cannot make a file durable that has no
/// contents yet.
///
/// The destination is written through a handle this function owns, rather than by
/// `fs::copy` followed by reopening the result. Two reasons, both Windows:
///
/// * `sync_all` is `FlushFileBuffers` there, which refuses a handle opened for reading —
///   `ERROR_ACCESS_DENIED`, on a file this process just created.
/// * `fs::copy` carries the source's permission bits across, so a package delivered
///   read-only would install read-only, and `remove_dir_all` cannot delete a read-only
///   file on Windows. The superseded artifact would then be undeletable.
fn copy_and_sync(source: &Path, destination: &Path) -> io::Result<()> {
    let mut source = File::open(source)?;
    let mut destination = File::create(destination)?;
    io::copy(&mut source, &mut destination)?;
    destination.sync_all()
}

/// Remove a directory tree, with a test-only failure injection point.
///
/// A cleanup that fails is what a permission problem or a file still open produces, and it
/// is the step that must not be able to return before an earlier rename is durable.
fn remove_dir(path: &Path) -> Result<(), ArtifactError> {
    #[cfg(test)]
    if failpoints::next_remove_dir_fails() {
        return Err(ArtifactError::Io {
            context: format!("removing {} (import failpoint)", path.display()),
            source: io::Error::other("injected directory removal failure"),
        });
    }
    fs::remove_dir_all(path).map_err(io_error(format!("removing {}", path.display())))
}

/// Flush a directory entry, with a test-only failure injection point.
///
/// A failing `fsync` is what a full or dying disk produces, and it lands in the one place
/// where the target directory does not exist. Without injection that path is unreachable
/// from a test.
fn flush(dir: &Path) -> Result<(), ArtifactError> {
    #[cfg(test)]
    if failpoints::next_dir_sync_fails() {
        return Err(ArtifactError::Io {
            context: format!("flushing {} (import failpoint)", dir.display()),
            source: io::Error::other("injected directory sync failure"),
        });
    }
    sync_dir(dir).map_err(io_error(format!("flushing {}", dir.display())))
}

fn canonicalize(path: &Path) -> Result<PathBuf, ArtifactError> {
    fs::canonicalize(path).map_err(io_error(format!("resolving {}", path.display())))
}

fn io_error(context: String) -> impl FnOnce(io::Error) -> ArtifactError {
    move |source| ArtifactError::Io { context, source }
}

/// `std::fs::rename`, with a test-only failure injection point.
///
/// The interesting failures here — a swap that cannot complete, a restore that cannot
/// either — cannot be produced by arranging files on disk, and an untested recovery path
/// is the one that fails when it is finally needed.
fn rename(from: &Path, to: &Path) -> io::Result<()> {
    #[cfg(test)]
    if failpoints::next_rename_fails() {
        return Err(io::Error::other(
            "injected rename failure (import failpoint)",
        ));
    }
    fs::rename(from, to)
}

/// Failure injection for the swap paths. A schedule rather than a count, because the
/// interesting case is "*this* rename fails".
#[cfg(test)]
mod failpoints {
    use std::cell::RefCell;
    use std::collections::VecDeque;

    thread_local! {
        static RENAME_SCHEDULE: RefCell<VecDeque<bool>> = const { RefCell::new(VecDeque::new()) };
        static DIR_SYNC_SCHEDULE: RefCell<VecDeque<bool>> = const { RefCell::new(VecDeque::new()) };
        static REMOVE_DIR_SCHEDULE: RefCell<VecDeque<bool>> =
            const { RefCell::new(VecDeque::new()) };
    }

    /// `[false, true]` passes the first rename and fails the second.
    pub fn schedule_rename_failures(schedule: &[bool]) {
        RENAME_SCHEDULE.with(|queue| *queue.borrow_mut() = schedule.iter().copied().collect());
    }

    /// Which of the next directory flushes fail.
    pub fn schedule_dir_sync_failures(schedule: &[bool]) {
        DIR_SYNC_SCHEDULE.with(|queue| *queue.borrow_mut() = schedule.iter().copied().collect());
    }

    pub fn next_rename_fails() -> bool {
        RENAME_SCHEDULE.with(|queue| queue.borrow_mut().pop_front().unwrap_or(false))
    }

    /// Which of the next directory removals fail.
    pub fn schedule_remove_dir_failures(schedule: &[bool]) {
        REMOVE_DIR_SCHEDULE.with(|queue| *queue.borrow_mut() = schedule.iter().copied().collect());
    }

    pub fn next_dir_sync_fails() -> bool {
        DIR_SYNC_SCHEDULE.with(|queue| queue.borrow_mut().pop_front().unwrap_or(false))
    }

    pub fn next_remove_dir_fails() -> bool {
        REMOVE_DIR_SCHEDULE.with(|queue| queue.borrow_mut().pop_front().unwrap_or(false))
    }

    pub fn reset() {
        schedule_rename_failures(&[]);
        schedule_dir_sync_failures(&[]);
        schedule_remove_dir_failures(&[]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::package::{
        PackageManifest, PayloadDescriptor, ARTIFACT_METADATA_VERSION,
    };
    use crate::semantic::versioning::{test_identity, IdentityField, IndexVersion};
    use std::collections::BTreeMap;

    struct TempDir(PathBuf);
    impl TempDir {
        /// Canonicalized at creation, because the install reports the paths it derived from
        /// a *canonical* parent, and the tests derive the same names from this root. The
        /// two must be the same string: `%TEMP%` on a Windows runner is an 8.3 short path
        /// (`C:\Users\RUNNER~1\...`), and `/var` on macOS is a symlink, so an unresolved
        /// fixture root names the same directory differently than the code does.
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "otzaria_test_imp_{name}_{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir_all(&path).unwrap();
            Self(std::fs::canonicalize(&path).unwrap())
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            failpoints::reset();
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn expectation() -> ArtifactExpectation {
        ArtifactExpectation::without_published_digest(test_identity())
    }

    /// A complete package at `root`, built from `identity`.
    fn write_package(root: &Path, identity: IndexVersion, payload: &[u8]) {
        fs::create_dir_all(root).unwrap();
        fs::write(root.join("vectors.bin"), payload).unwrap();
        let package = IndexPackage {
            manifest: PackageManifest::new(
                identity,
                "2026-08-06T00:00:00Z".to_string(),
                5,
                50,
                payload.len() as u64,
            ),
            payloads: BTreeMap::from([(
                "vectors.bin".to_string(),
                PayloadDescriptor::of_bytes(payload),
            )]),
        };
        IndexPackage::write(root, &package).unwrap();
    }

    fn install(source: &Path, target: &Path) -> Result<ImportResult, ArtifactError> {
        IndexImporter::new(ImportConfig {
            source_path: source.to_path_buf(),
            target_store_path: target.to_path_buf(),
        })
        .import(&expectation())
    }

    fn payload_of(target: &Path) -> Vec<u8> {
        fs::read(target.join("vectors.bin")).unwrap()
    }

    #[test]
    fn a_matching_artifact_is_installed_and_reopens_from_the_target() {
        let dir = TempDir::new("install");
        let source = dir.path().join("source");
        let target = dir.path().join("target");
        write_package(&source, test_identity(), b"vectors");

        let result = install(&source, &target).unwrap();
        assert_eq!(result.books_imported, 5);
        assert_eq!(result.vectors_imported, 50);
        assert_eq!(result.bytes_imported, 7);

        // The installed copy stands on its own: it verifies from the target directory,
        // with no reference to where it came from.
        let installed = IndexPackage::verify_for_install(&target, &expectation()).unwrap();
        assert_eq!(installed.payload_names(), ["vectors.bin"]);
        assert_eq!(
            installed.manifest().metadata_version,
            ARTIFACT_METADATA_VERSION
        );
        // And nothing was left beside it.
        assert!(!staging_path(&target).unwrap().exists());
        assert!(!previous_path(&target).unwrap().exists());
    }

    #[test]
    fn a_second_install_replaces_the_first_and_leaves_no_leftovers() {
        let dir = TempDir::new("replace");
        let first = dir.path().join("first");
        let second = dir.path().join("second");
        let target = dir.path().join("target");
        write_package(&first, test_identity(), b"first");
        write_package(&second, test_identity(), b"second!");

        install(&first, &target).unwrap();
        install(&second, &target).unwrap();

        assert_eq!(payload_of(&target), b"second!");
        assert!(IndexPackage::verify_for_install(&target, &expectation()).is_ok());
        assert!(!previous_path(&target).unwrap().exists());
        assert!(!staging_path(&target).unwrap().exists());
    }

    /// The gate: a package for another corpus is refused, and the previous artifact is
    /// still the one on disk afterwards.
    #[test]
    fn an_artifact_from_another_corpus_leaves_the_installed_one_untouched() {
        let dir = TempDir::new("wrong_corpus");
        let source = dir.path().join("source");
        let target = dir.path().join("target");

        write_package(&source, test_identity(), b"first");
        install(&source, &target).unwrap();

        let mut other_corpus = test_identity();
        other_corpus.corpus.corpus_id = "d".repeat(64);
        let replacement = dir.path().join("replacement");
        write_package(&replacement, other_corpus, b"second");

        match install(&replacement, &target) {
            Err(ArtifactError::IdentityMismatch { mismatches }) => {
                assert_eq!(mismatches[0].field, IdentityField::CorpusId)
            }
            other => panic!("expected an identity rejection, got {other:?}"),
        }

        assert_eq!(payload_of(&target), b"first");
        assert!(IndexPackage::verify_for_install(&target, &expectation()).is_ok());
        assert!(!staging_path(&target).unwrap().exists());
        assert!(!previous_path(&target).unwrap().exists());
    }

    /// A damaged download, in both shapes: same length (only the checksum sees it) and a
    /// truncation (the declared size sees it too).
    #[test]
    fn a_corrupt_payload_is_refused_before_anything_is_copied() {
        let dir = TempDir::new("corrupt");
        let source = dir.path().join("source");
        let target = dir.path().join("target");
        write_package(&source, test_identity(), b"vectors");

        fs::write(source.join("vectors.bin"), b"vectorZ").unwrap();
        assert!(matches!(
            install(&source, &target),
            Err(ArtifactError::PayloadChecksumFailed { .. })
        ));

        fs::write(source.join("vectors.bin"), b"trunc").unwrap();
        assert!(matches!(
            install(&source, &target),
            Err(ArtifactError::ManifestDisagreesWithPayload { .. })
        ));

        assert!(!target.exists(), "a refused install must create nothing");
        assert!(!staging_path(&target).unwrap().exists());
    }

    #[test]
    fn the_target_may_not_be_the_package_directory_or_a_child_of_it() {
        let dir = TempDir::new("self_target");
        let source = dir.path().join("source");
        write_package(&source, test_identity(), b"vectors");

        for target in [source.clone(), source.join("nested")] {
            assert!(
                matches!(
                    install(&source, &target),
                    Err(ArtifactError::InvalidInstallTarget { .. })
                ),
                "{} must be refused as a target",
                target.display()
            );
        }

        assert!(IndexPackage::verify_for_install(&source, &expectation()).is_ok());
    }

    /// The crash window the module documents: killed between the two renames, the target
    /// is gone and the only good copy is parked. Recovery has to put it back — otherwise
    /// the device looks like it never had an artifact.
    #[test]
    fn a_crash_between_the_two_renames_is_recovered_to_the_previous_artifact() {
        let dir = TempDir::new("crash_midswap");
        let source = dir.path().join("source");
        let target = dir.path().join("target");
        write_package(&source, test_identity(), b"installed");
        install(&source, &target).unwrap();

        // Reproduce the on-disk state of that crash, using the same path helpers the
        // installer uses — a hand-built state that did not come from these functions
        // would prove nothing about them.
        let staging = staging_path(&target).unwrap();
        fs::create_dir(&staging).unwrap();
        fs::write(staging.join("half-copied.bin"), b"partial").unwrap();
        fs::rename(&target, previous_path(&target).unwrap()).unwrap();
        assert!(!target.exists(), "the crash window has no target");

        let recovery = recover_interrupted_install(&target).unwrap();
        assert!(recovery.restored_previous);
        assert!(recovery.discarded_staging);
        assert!(!recovery.discarded_previous);

        assert_eq!(payload_of(&target), b"installed");
        assert!(IndexPackage::verify_for_install(&target, &expectation()).is_ok());
        assert!(!staging.exists());
        assert!(!previous_path(&target).unwrap().exists());
    }

    /// The other window: the swap completed and only the cleanup did not. The new
    /// artifact is live, so recovery must remove the superseded copy — not restore it
    /// over the artifact that just replaced it.
    #[test]
    fn a_crash_after_the_swap_discards_the_superseded_copy() {
        let dir = TempDir::new("crash_postswap");
        let source = dir.path().join("source");
        let target = dir.path().join("target");
        write_package(&source, test_identity(), b"new-one");
        install(&source, &target).unwrap();

        let previous = previous_path(&target).unwrap();
        fs::create_dir(&previous).unwrap();
        fs::write(previous.join("vectors.bin"), b"old-one").unwrap();

        let recovery = recover_interrupted_install(&target).unwrap();
        assert!(recovery.discarded_previous);
        assert!(!recovery.restored_previous);

        assert_eq!(payload_of(&target), b"new-one");
        assert!(!previous.exists());
    }

    #[test]
    fn recovery_on_a_clean_target_does_nothing_and_is_idempotent() {
        let dir = TempDir::new("recover_clean");
        let source = dir.path().join("source");
        let target = dir.path().join("target");
        write_package(&source, test_identity(), b"vectors");
        install(&source, &target).unwrap();

        for _ in 0..2 {
            let recovery = recover_interrupted_install(&target).unwrap();
            assert!(!recovery.recovered_anything());
        }
        assert!(IndexPackage::verify_for_install(&target, &expectation()).is_ok());
    }

    /// A swap that cannot complete must put the previous artifact back in the same call.
    #[test]
    fn a_failed_swap_restores_the_previous_artifact() {
        let dir = TempDir::new("failed_swap");
        let first = dir.path().join("first");
        let second = dir.path().join("second");
        let target = dir.path().join("target");
        write_package(&first, test_identity(), b"first");
        write_package(&second, test_identity(), b"second!");
        install(&first, &target).unwrap();

        // First rename (target → previous) succeeds, second (staging → target) fails.
        failpoints::schedule_rename_failures(&[false, true]);
        let attempt = install(&second, &target);
        assert!(attempt.is_err(), "the swap must not report success");

        assert_eq!(payload_of(&target), b"first");
        assert!(IndexPackage::verify_for_install(&target, &expectation()).is_ok());
        assert!(!previous_path(&target).unwrap().exists());
        assert!(!staging_path(&target).unwrap().exists());
    }

    /// And if the restore fails too, the failure is reported with the directory the only
    /// good copy is in — never swallowed — and recovery puts it back.
    #[test]
    fn a_failed_restore_is_reported_and_then_recoverable() {
        let dir = TempDir::new("failed_restore");
        let first = dir.path().join("first");
        let second = dir.path().join("second");
        let target = dir.path().join("target");
        write_package(&first, test_identity(), b"first");
        write_package(&second, test_identity(), b"second!");
        install(&first, &target).unwrap();

        // Park succeeds; swap fails; restore fails.
        failpoints::schedule_rename_failures(&[false, true, true]);
        let attempt = install(&second, &target);
        match attempt {
            Err(ArtifactError::InterruptedInstall { reason }) => {
                assert!(
                    reason.contains(&previous_path(&target).unwrap().display().to_string()),
                    "the error must name where the good copy is: {reason}"
                );
            }
            other => panic!("expected an interrupted-install error, got {other:?}"),
        }
        assert!(!target.exists(), "this is the state a crash would leave");

        failpoints::reset();
        let recovery = recover_interrupted_install(&target).unwrap();
        assert!(recovery.restored_previous);
        assert_eq!(payload_of(&target), b"first");
        assert!(IndexPackage::verify_for_install(&target, &expectation()).is_ok());
    }

    /// An install must not write over an unresolved interruption: the previous artifact
    /// is parked under a recovery name, and losing it would leave the device with none.
    #[test]
    fn an_install_recovers_an_interrupted_one_before_replacing_anything() {
        let dir = TempDir::new("install_recovers");
        let first = dir.path().join("first");
        let second = dir.path().join("second");
        let target = dir.path().join("target");
        write_package(&first, test_identity(), b"first");
        write_package(&second, test_identity(), b"second!");
        install(&first, &target).unwrap();

        // The mid-swap crash state again.
        fs::rename(&target, previous_path(&target).unwrap()).unwrap();

        install(&second, &target).unwrap();
        assert_eq!(payload_of(&target), b"second!");
        assert!(!previous_path(&target).unwrap().exists());
        assert!(!staging_path(&target).unwrap().exists());
    }

    /// A flush that fails right after the previous artifact was parked is the one moment
    /// where no target directory exists. It has to end with the previous artifact back in
    /// place — not with an IO error and an empty target.
    #[test]
    fn a_failed_flush_after_parking_restores_the_previous_artifact() {
        let dir = TempDir::new("flush_after_park");
        let first = dir.path().join("first");
        let second = dir.path().join("second");
        let target = dir.path().join("target");
        write_package(&first, test_identity(), b"first");
        write_package(&second, test_identity(), b"second!");
        install(&first, &target).unwrap();

        // The next directory flush fails; in an install over an existing target that is
        // the one immediately after `target → previous`.
        failpoints::schedule_dir_sync_failures(&[true]);
        assert!(install(&second, &target).is_err());
        failpoints::reset();

        assert_eq!(payload_of(&target), b"first");
        assert!(IndexPackage::verify_for_install(&target, &expectation()).is_ok());
        assert!(!previous_path(&target).unwrap().exists());
        assert!(!staging_path(&target).unwrap().exists());
    }

    /// And if the rollback cannot happen either, the error must say where the good copy is
    /// instead of reporting a bare IO failure over an empty target.
    #[test]
    fn a_failed_flush_that_cannot_be_rolled_back_names_the_parked_copy() {
        let dir = TempDir::new("flush_no_rollback");
        let first = dir.path().join("first");
        let second = dir.path().join("second");
        let target = dir.path().join("target");
        write_package(&first, test_identity(), b"first");
        write_package(&second, test_identity(), b"second!");
        install(&first, &target).unwrap();

        // Park succeeds, the flush after it fails, and the restore rename fails too.
        failpoints::schedule_dir_sync_failures(&[true]);
        failpoints::schedule_rename_failures(&[false, true]);
        match install(&second, &target) {
            Err(ArtifactError::InterruptedInstall { reason }) => assert!(
                reason.contains(&previous_path(&target).unwrap().display().to_string()),
                "the error must name where the good copy is: {reason}"
            ),
            other => panic!("expected an interrupted-install error, got {other:?}"),
        }
        failpoints::reset();

        assert!(!target.exists(), "this is the state a crash would leave");
        let recovery = recover_interrupted_install(&target).unwrap();
        assert!(recovery.restored_previous);
        assert_eq!(payload_of(&target), b"first");
    }

    /// A rollback whose flush also fails must not be reported as a plain IO failure: the
    /// rename happened in this process, but nothing knows whether it survived. Both
    /// outcomes are recoverable, and the error has to say so.
    #[test]
    fn a_rollback_that_cannot_be_flushed_reports_unknown_durability() {
        let dir = TempDir::new("rollback_flush");
        let first = dir.path().join("first");
        let second = dir.path().join("second");
        let target = dir.path().join("target");
        write_package(&first, test_identity(), b"first");
        write_package(&second, test_identity(), b"second!");
        install(&first, &target).unwrap();

        // The flush after parking fails, and so does the flush after the rollback.
        failpoints::schedule_dir_sync_failures(&[true, true]);
        match install(&second, &target) {
            Err(ArtifactError::InterruptedInstall { reason }) => {
                assert!(reason.contains("could not be flushed"), "{reason}");
                assert!(reason.contains("either path"), "{reason}");
            }
            other => panic!("expected an unknown-durability error, got {other:?}"),
        }
        failpoints::reset();

        // The rollback did take effect in this process; what was unknown is only whether
        // it reached the disk.
        assert_eq!(payload_of(&target), b"first");
        assert!(IndexPackage::verify_for_install(&target, &expectation()).is_ok());
    }

    /// The ordering the deferred flush got wrong: a cleanup failure must not be able to
    /// return before the restore itself is on disk. Only the ordering is observable from a
    /// test — an `fsync` leaves no trace — so this asserts that the restore had already
    /// happened when the cleanup failed.
    #[test]
    fn a_restore_precedes_the_cleanup_that_can_fail_after_it() {
        let dir = TempDir::new("restore_then_cleanup");
        let source = dir.path().join("source");
        let target = dir.path().join("target");
        write_package(&source, test_identity(), b"installed");
        install(&source, &target).unwrap();

        // The mid-swap crash state, with a staging directory left behind.
        let staging = staging_path(&target).unwrap();
        fs::create_dir(&staging).unwrap();
        fs::write(staging.join("half-copied.bin"), b"partial").unwrap();
        fs::rename(&target, previous_path(&target).unwrap()).unwrap();

        // Removing the staging directory fails; the restore before it must stand.
        failpoints::schedule_remove_dir_failures(&[true]);
        assert!(recover_interrupted_install(&target).is_err());
        failpoints::reset();

        assert_eq!(payload_of(&target), b"installed");
        assert!(!previous_path(&target).unwrap().exists());
        assert!(staging.exists(), "the cleanup is what failed");

        // And a second call finishes the job.
        let recovery = recover_interrupted_install(&target).unwrap();
        assert!(recovery.discarded_staging);
        assert!(!recovery.restored_previous, "already restored");
        assert!(!staging.exists());
    }

    /// Recovery deletes directories, so a target it cannot name must be refused before any
    /// sibling path is derived from it.
    #[test]
    fn a_target_without_a_name_of_its_own_is_refused_by_both_paths() {
        let dir = TempDir::new("nameless");
        for target in [
            PathBuf::from("."),
            PathBuf::from(".."),
            dir.path().join(".."),
            PathBuf::from("/"),
        ] {
            assert!(
                matches!(
                    recover_interrupted_install(&target),
                    Err(ArtifactError::InvalidInstallTarget { .. })
                ),
                "recovery must refuse {}",
                target.display()
            );
            assert!(
                previous_path(&target).is_err() && staging_path(&target).is_err(),
                "no recovery names may be derived from {}",
                target.display()
            );
        }
    }

    /// A single-component relative target has an *empty* parent, not none. Both paths have
    /// to normalize it the same way, or one of them fails on `canonicalize("")`.
    ///
    /// Tested through the resolver rather than by changing the process's working directory,
    /// which is global state shared with every other test in this binary.
    #[test]
    fn a_single_component_relative_target_resolves_against_the_current_directory() {
        let paths = InstallPaths::resolve(Path::new("semantic_db")).unwrap();
        assert_eq!(paths.parent, Path::new("."));
        assert_eq!(paths.target, Path::new("./semantic_db"));
        assert_eq!(paths.previous, Path::new("./.semantic_db.previous"));
        assert_eq!(paths.staging, Path::new("./.semantic_db.staging"));

        // And the same names for the equivalent explicit form.
        let explicit = InstallPaths::resolve(Path::new("./semantic_db")).unwrap();
        assert_eq!(explicit.previous, paths.previous);
        assert_eq!(explicit.staging, paths.staging);
    }
}
