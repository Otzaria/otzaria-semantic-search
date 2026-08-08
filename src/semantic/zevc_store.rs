//! The snapshot format the official artifact's payload is written in, and the two ways
//! it is opened.
//!
//! A snapshot is three files in one directory: `vectors.bin` (dense `f32` little-endian
//! records), `metadata.jsonl` (one JSON object per record, in the same order, carrying a
//! SHA-256 of the metadata and of the vector) and `book_index.json` (the header, plus
//! book key → its `semantic_id`s).
//!
//! Two openers, one reader:
//!
//! * [`ZevcStore`] — writable, for a builder. It scans every vector and holds all of
//!   them in memory.
//! * [`ReadOnlyZevcStore`] — the runtime's view of an installed artifact. It implements
//!   [`VectorSearchBackend`] only, so there is no mutation to call on it.
//!
//! Both read through one function, so the checks a reader performs cannot drift between
//! them.
//!
//! # This is a reference backend, not a scale answer
//!
//! Opening reads every byte of the payload, verifies a SHA-256 per record, and keeps
//! every vector in RAM; searching scans all of them, `O(N·D)`. That is the honest cost of
//! a format with no index and no lazy access, and it is what makes
//! [`VerificationDepth::MetadataAndPresence`](crate::distribution::package::VerificationDepth)
//! cheap and this open expensive. Both facts are true at once, and the second one is
//! this backend's, not the artifact contract's. S2b measures it and decides what
//! replaces it — see `docs/DEVELOPMENT.md`. Nothing here is an ANN index, and this
//! module is not the `zvec` library.

use crate::distribution::package::VerifiedPackage;
use crate::errors::VectorStoreError;
// One threshold for the whole crate, so no two layers can disagree about which vector
// has "no direction".
use crate::semantic::embedding::MIN_VECTOR_NORM;
use crate::semantic::store_backend::{VectorSearchBackend, VectorStoreBackend};
use crate::semantic::types::{SearchFilters, SemanticCandidate, VectorMetadata};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap, HashMap};
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, RwLock};

/// Identifier of this backend, recorded as `store.backend_id` in an artifact's identity.
pub const BACKEND_ID: &str = "zevc-persistent-v1";

/// Version of the on-disk layout, recorded as `store.store_format_version` and written
/// into the snapshot header. Separate from [`BACKEND_ID`] so a format change inside one
/// backend is a rejection rather than a misread payload.
pub const STORE_FORMAT_VERSION: u32 = 1;

/// Precision this format stores vectors at, recorded as `store.vector_precision`. The
/// records are dense `f32`; there is no other layout this reader can decode, which is
/// why the value is a constant here and data in the manifest.
pub const VECTOR_PRECISION: &str = "f32";

/// The vectors, laid out as `embedding_dim` little-endian `f32` per record.
pub const VECTORS_FILENAME: &str = "vectors.bin";

/// One JSON object per record, in the same order as [`VECTORS_FILENAME`].
pub const METADATA_FILENAME: &str = "metadata.jsonl";

/// Header and book → ids map.
pub const BOOK_INDEX_FILENAME: &str = "book_index.json";

/// Every file a snapshot is made of — the payload names an artifact of this backend
/// declares, and exactly the names a packer has to write.
pub const SNAPSHOT_FILENAMES: [&str; 3] =
    [VECTORS_FILENAME, METADATA_FILENAME, BOOK_INDEX_FILENAME];

#[derive(Debug, Clone)]
pub struct ZevcStoreConfig {
    pub db_path: PathBuf,
    pub embedding_dim: u32,
    pub collection_name: String,
    /// Whether to persist after every batch or defer to explicit flush.
    pub auto_persist: bool,
}

#[derive(Debug, Clone)]
pub struct StoredVectorRecord {
    pub metadata: VectorMetadata,
    pub vector: Vec<f32>,
}

#[derive(Serialize, Deserialize)]
struct PersistedMetadata {
    metadata: VectorMetadata,
    metadata_sha256: String,
    vector_sha256: String,
}

#[derive(Serialize, Deserialize)]
struct PersistedBookIndex {
    format_version: u32,
    embedding_dim: u32,
    collection_name: String,
    books: HashMap<String, Vec<String>>,
}

#[derive(Default)]
struct StoreState {
    records: HashMap<String, StoredVectorRecord>,
    book_index: HashMap<String, Vec<String>>,
}

/// Borrows its id from the record it scores: the scan visits every vector, and cloning a
/// `String` per visit would add millions of allocations per query to work that is supposed
/// to be arithmetic. The records outlive the scan, so there is nothing to own.
struct ScoredEntry<'a> {
    score: f32,
    semantic_id: &'a str,
}

impl PartialEq for ScoredEntry<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for ScoredEntry<'_> {}
impl PartialOrd for ScoredEntry<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ScoredEntry<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .total_cmp(&self.score)
            .then_with(|| self.semantic_id.cmp(other.semantic_id))
    }
}

pub struct ZevcStore {
    config: ZevcStoreConfig,
    state: RwLock<StoreState>,
    persist_lock: Mutex<()>,
}

impl ZevcStore {
    pub fn open_or_create(config: ZevcStoreConfig) -> Result<Self, VectorStoreError> {
        if config.embedding_dim == 0 {
            return Err(VectorStoreError::OpenFailed {
                reason: "embedding_dim must be greater than zero".to_string(),
            });
        }
        if config.collection_name.trim().is_empty() {
            return Err(VectorStoreError::OpenFailed {
                reason: "collection_name must not be empty".to_string(),
            });
        }
        if let Err(e) = std::fs::create_dir_all(&config.db_path) {
            return Err(VectorStoreError::OpenFailed {
                reason: format!("Failed to create DB directory: {e}"),
            });
        }

        let store = Self {
            config,
            state: RwLock::new(StoreState::default()),
            persist_lock: Mutex::new(()),
        };

        store.load_from_disk()?;

        log::info!(
            "ZevcStore (persistent) initialized at: {}",
            store.config.db_path.display()
        );
        Ok(store)
    }

    fn vectors_path(&self) -> PathBuf {
        self.config.db_path.join(VECTORS_FILENAME)
    }

    fn metadata_path(&self) -> PathBuf {
        self.config.db_path.join(METADATA_FILENAME)
    }

    fn book_index_path(&self) -> PathBuf {
        self.config.db_path.join(BOOK_INDEX_FILENAME)
    }

    pub fn save_to_disk(&self) -> Result<(), VectorStoreError> {
        let _persist = self
            .persist_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let state = self
            .state
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let v_path = self.vectors_path();
        let m_path = self.metadata_path();
        let b_path = self.book_index_path();

        let v_tmp = v_path.with_extension("bin.tmp");
        let m_tmp = m_path.with_extension("jsonl.tmp");
        let b_tmp = b_path.with_extension("json.tmp");

        let mut v_file =
            BufWriter::new(
                File::create(&v_tmp).map_err(|e| VectorStoreError::CommitFailed {
                    reason: e.to_string(),
                })?,
            );
        let mut m_file =
            BufWriter::new(
                File::create(&m_tmp).map_err(|e| VectorStoreError::CommitFailed {
                    reason: e.to_string(),
                })?,
            );

        for record in state.records.values() {
            let mut vector_bytes = Vec::with_capacity(record.vector.len() * 4);
            for value in &record.vector {
                vector_bytes.extend_from_slice(&value.to_le_bytes());
            }
            let persisted = PersistedMetadata {
                metadata: record.metadata.clone(),
                metadata_sha256: format!(
                    "{:x}",
                    Sha256::digest(serde_json::to_vec(&record.metadata).map_err(|e| {
                        VectorStoreError::CommitFailed {
                            reason: e.to_string(),
                        }
                    })?)
                ),
                vector_sha256: format!("{:x}", Sha256::digest(&vector_bytes)),
            };
            let meta_json =
                serde_json::to_string(&persisted).map_err(|e| VectorStoreError::CommitFailed {
                    reason: e.to_string(),
                })?;
            writeln!(m_file, "{meta_json}").map_err(|e| VectorStoreError::CommitFailed {
                reason: e.to_string(),
            })?;

            v_file
                .write_all(&vector_bytes)
                .map_err(|e| VectorStoreError::CommitFailed {
                    reason: e.to_string(),
                })?;
        }

        let b_file = File::create(&b_tmp).map_err(|e| VectorStoreError::CommitFailed {
            reason: e.to_string(),
        })?;
        let persisted_book_index = PersistedBookIndex {
            format_version: STORE_FORMAT_VERSION,
            embedding_dim: self.config.embedding_dim,
            collection_name: self.config.collection_name.clone(),
            books: state.book_index.clone(),
        };
        serde_json::to_writer(&b_file, &persisted_book_index).map_err(|e| {
            VectorStoreError::CommitFailed {
                reason: e.to_string(),
            }
        })?;

        v_file.flush().map_err(|e| VectorStoreError::CommitFailed {
            reason: e.to_string(),
        })?;
        m_file.flush().map_err(|e| VectorStoreError::CommitFailed {
            reason: e.to_string(),
        })?;
        v_file
            .get_ref()
            .sync_all()
            .and_then(|_| m_file.get_ref().sync_all())
            .and_then(|_| b_file.sync_all())
            .map_err(|e| VectorStoreError::CommitFailed {
                reason: e.to_string(),
            })?;

        // Windows refuses to rename an open file. Flushing is not enough: all
        // handles must be closed before the snapshot swap begins.
        drop(v_file);
        drop(m_file);
        drop(b_file);

        replace_snapshot(&[(&v_tmp, &v_path), (&m_tmp, &m_path), (&b_tmp, &b_path)])?;

        Ok(())
    }

    fn load_from_disk(&self) -> Result<(), VectorStoreError> {
        let Some(snapshot) = read_snapshot(
            &self.config.db_path,
            &SnapshotExpectation {
                embedding_dim: self.config.embedding_dim,
                format_version: STORE_FORMAT_VERSION,
                collection_name: Some(self.config.collection_name.clone()),
                // A store reopening its own directory has nothing external to compare
                // against; the per-record checksums are all it has.
                declared_sha256: None,
            },
        )?
        else {
            return Ok(());
        };

        let mut state = self
            .state
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.records = snapshot.records;
        state.book_index = snapshot.book_index;
        Ok(())
    }
}

/// A snapshot opened read-only: the runtime's view of an installed artifact's payload.
///
/// Implements [`VectorSearchBackend`] and not [`VectorStoreBackend`], which is the whole
/// point — the type a query holds has no way to modify what a build machine produced.
/// The records are immutable once opened, so there is no lock to take on the query path
/// either.
pub struct ReadOnlyZevcStore {
    records: HashMap<String, StoredVectorRecord>,
    book_index: HashMap<String, Vec<String>>,
    embedding_dim: u32,
    collection_name: String,
}

impl ReadOnlyZevcStore {
    /// Open the payload of a verified artifact.
    ///
    /// **The token is the only argument, and that is the point.** Everything this reader
    /// needs is derived from it here rather than by the caller: the directory, the record
    /// width, and the SHA-256 each payload file must have. A signature taking a path and a
    /// hash table would let in-crate code hand over a directory nobody verified, or a
    /// dimension and a set of hashes belonging to some other package — so "no reading
    /// without a verified token" is enforced by the type rather than by whoever calls it.
    ///
    /// It is also `pub(crate)`: outside this crate the way in is
    /// [`OfficialSemanticIndex`](crate::semantic::official_index::OfficialSemanticIndex),
    /// which is what obtains the token in the first place.
    ///
    /// What the derived values do:
    ///
    /// * the record width decides how the payload is decoded, so a payload holding a
    ///   different one is caught as a truncated or trailing read rather than misparsed;
    /// * the declared hashes are checked against the bytes while they are read — see
    ///   [`SnapshotExpectation::declared_sha256`] for why a reader that skips them leaves a
    ///   published digest unable to protect the payload.
    ///
    /// The collection name is **adopted from the payload**, not required: it is not part
    /// of an artifact's identity, so a reader has nothing to compare it against and must
    /// not invent a value to demand. What is compared is everything that *is* identity —
    /// by [`IndexPackage::verify_for_open`](crate::distribution::package::IndexPackage::verify_for_open),
    /// which is what produced this token.
    ///
    /// An empty directory is a rejection here, unlike for [`ZevcStore`]: a builder may
    /// legitimately start from nothing, an artifact may not be nothing. Reaching that branch
    /// means the files disappeared between verification and this read — the token proves they
    /// were there, not that they still are.
    pub(crate) fn open(verified: &VerifiedPackage) -> Result<Self, VectorStoreError> {
        let dir = verified.root();
        let embedding_dim = verified.identity().model.embedding_dim;
        let snapshot = read_snapshot(
            dir,
            &SnapshotExpectation {
                embedding_dim,
                format_version: STORE_FORMAT_VERSION,
                collection_name: None,
                declared_sha256: Some(
                    verified
                        .payloads()
                        .iter()
                        .map(|(name, descriptor)| (name.clone(), descriptor.sha256.clone()))
                        .collect(),
                ),
            },
        )?
        .ok_or_else(|| VectorStoreError::Corrupted {
            reason: format!(
                "{} holds no {BACKEND_ID} snapshot: none of {} is there",
                dir.display(),
                SNAPSHOT_FILENAMES.join(", ")
            ),
        })?;

        log::info!(
            "Opened {} vector(s) read-only from {} (collection '{}')",
            snapshot.records.len(),
            dir.display(),
            snapshot.collection_name
        );

        Ok(Self {
            records: snapshot.records,
            book_index: snapshot.book_index,
            embedding_dim,
            collection_name: snapshot.collection_name,
        })
    }

    /// Collection name the payload declares. Reported, never enforced: it is not part of an
    /// artifact's identity, so a reader has nothing to compare it against.
    pub fn collection_name(&self) -> &str {
        &self.collection_name
    }
}

impl VectorSearchBackend for ReadOnlyZevcStore {
    fn backend_id(&self) -> &'static str {
        BACKEND_ID
    }

    fn is_persistent(&self) -> bool {
        true
    }

    fn embedding_dim(&self) -> u32 {
        self.embedding_dim
    }

    fn count(&self) -> u32 {
        self.records.len().min(u32::MAX as usize) as u32
    }

    fn search(
        &self,
        query_vector: &[f32],
        top_k: usize,
        filters: Option<&SearchFilters>,
    ) -> Result<Vec<SemanticCandidate>, VectorStoreError> {
        search_records(
            &self.records,
            self.embedding_dim,
            query_vector,
            top_k,
            filters,
        )
    }

    fn book_keys(&self) -> Vec<String> {
        let mut keys: Vec<String> = self.book_index.keys().cloned().collect();
        keys.sort();
        keys
    }

    fn book_vector_count(&self, source_book_key: &str) -> usize {
        self.book_index
            .get(source_book_key)
            .map_or(0, |ids| ids.len())
    }
}

/// What a snapshot must agree with to be readable.
pub(crate) struct SnapshotExpectation {
    /// Record width, in `f32`s.
    pub(crate) embedding_dim: u32,
    pub(crate) format_version: u32,
    /// `Some` for a store reopening its own directory, where a different collection means
    /// the caller is pointing at someone else's data. `None` adopts what the payload
    /// declares — see [`ReadOnlyZevcStore::open`].
    pub(crate) collection_name: Option<String>,
    /// SHA-256 the *artifact* declares for each snapshot file, checked against the bytes
    /// while they are read.
    ///
    /// This is what makes a published digest reach the payload. The digest pins the hashes
    /// in `payloads.json`; without comparing those hashes to the files, an attacker who
    /// edits a vector and the matching per-record checksum — both fixed-length, so no size
    /// changes — passes every check the cheap open path and this reader would otherwise
    /// make. See [`read_snapshot`].
    ///
    /// `None` for [`ZevcStore`] reopening a directory it wrote itself: there is no external
    /// declaration to compare against, and the per-record checksums are all it has.
    pub(crate) declared_sha256: Option<BTreeMap<String, String>>,
}

impl SnapshotExpectation {
    /// Compare one file's computed hash against what the artifact declared, when there is a
    /// declaration to compare against.
    ///
    /// A file the artifact does not declare is a fault rather than a pass: the reader is
    /// about to load it, so "not covered by the token" means the token proves nothing about
    /// what will be in memory.
    fn verify_file(&self, filename: &str, computed: &str) -> Result<(), VectorStoreError> {
        let Some(declared) = &self.declared_sha256 else {
            return Ok(());
        };
        let Some(expected) = declared.get(filename) else {
            return Err(VectorStoreError::Corrupted {
                reason: format!("{filename} is not one of the payloads this artifact declares"),
            });
        };
        if computed != expected {
            return Err(VectorStoreError::Corrupted {
                reason: format!(
                    "{filename} does not match the SHA-256 the artifact declares for it \
                     (declared {expected}, found {computed})"
                ),
            });
        }
        Ok(())
    }
}

/// Wraps a reader and hashes every byte that passes through it, so a file can be
/// authenticated by the same pass that parses it.
struct HashingReader<R> {
    inner: R,
    hasher: Sha256,
}

impl<R: Read> HashingReader<R> {
    fn new(inner: R) -> Self {
        Self {
            inner,
            hasher: Sha256::new(),
        }
    }

    /// The hash of everything read so far, as 64 hex digits. Meaningful only once the caller
    /// has read to the end of the file — see [`read_snapshot`].
    fn finalize(self) -> String {
        format!("{:x}", self.hasher.finalize())
    }
}

impl<R: Read> Read for HashingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let read = self.inner.read(buf)?;
        self.hasher.update(&buf[..read]);
        Ok(read)
    }
}

/// A snapshot's contents, checked but not yet owned by a store.
struct Snapshot {
    records: HashMap<String, StoredVectorRecord>,
    book_index: HashMap<String, Vec<String>>,
    collection_name: String,
}

/// Read and check the snapshot in `dir`, or report that there is none.
///
/// `Ok(None)` means the directory holds no snapshot at all, which is a fresh store rather
/// than a fault. A *partial* one is corruption: the three files are written as one commit,
/// so two out of three means an interrupted write, not an empty index.
///
/// What is checked:
///
/// * the header's format version, dimension and (optionally) collection name;
/// * **the SHA-256 of each file against what the artifact declared**, when
///   [`SnapshotExpectation::declared_sha256`] is given. Computed from the same bytes this
///   pass already reads, so it costs no extra I/O — and it is the only check that makes a
///   digest published outside the package protect the payload at *open*: everything else
///   here is a checksum that travels inside the bytes it guards;
/// * a SHA-256 per record over the metadata and over the vector's stored bytes, which
///   localizes damage to one record rather than one file;
/// * exactly `embedding_dim` values per record and no trailing bytes, so the two files
///   describe the same number of records;
/// * no duplicate `semantic_id`;
/// * every vector finite and with a usable direction, because an unscorable vector is a
///   record that exists and can never be returned;
/// * the stored book index against the one rebuilt from the metadata.
///
/// The per-record and per-file hashes cover the same bytes twice, which is a real cost and
/// kept anyway: the first tells the user *which* record broke, and the second is the only
/// one an attacker cannot update from inside the payload.
///
/// What no check here can catch: a payload edited together with its per-record checksums
/// **and** `payloads.json` re-stamped to match, in a package that was verified without a
/// published digest. That is self-consistent at every layer, and only an external digest
/// separates it from the real artifact — see
/// [`ArtifactExpectation`](crate::distribution::package::ArtifactExpectation).
fn read_snapshot(
    dir: &Path,
    expected: &SnapshotExpectation,
) -> Result<Option<Snapshot>, VectorStoreError> {
    let v_path = dir.join(VECTORS_FILENAME);
    let m_path = dir.join(METADATA_FILENAME);
    let b_path = dir.join(BOOK_INDEX_FILENAME);

    let present = [v_path.exists(), m_path.exists(), b_path.exists()];
    if present.iter().all(|exists| !exists) {
        return Ok(None);
    }
    if !present.iter().all(|exists| *exists) {
        return Err(VectorStoreError::Corrupted {
            reason: format!(
                "{} holds an incomplete snapshot: {}",
                dir.display(),
                SNAPSHOT_FILENAMES
                    .iter()
                    .zip(present)
                    .map(|(name, exists)| format!(
                        "{name} is {}",
                        if exists { "present" } else { "missing" }
                    ))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        });
    }

    // The header first: a foreign format version must not be parsed as if it were this
    // one, and a dimension disagreement decides how every record is decoded. Read as bytes
    // rather than streamed into `serde`, so the declared hash covers the file even when
    // parsing fails.
    let book_index_bytes = fs::read(&b_path).map_err(|e| VectorStoreError::OpenFailed {
        reason: e.to_string(),
    })?;
    expected.verify_file(
        BOOK_INDEX_FILENAME,
        &format!("{:x}", Sha256::digest(&book_index_bytes)),
    )?;
    let persisted_book_index: PersistedBookIndex = serde_json::from_slice(&book_index_bytes)
        .map_err(|e| {
            log::warn!("Corrupted book index: {e}");
            VectorStoreError::Corrupted {
                reason: e.to_string(),
            }
        })?;

    if persisted_book_index.format_version != expected.format_version {
        return Err(VectorStoreError::Corrupted {
            reason: format!(
                "snapshot is format version {}, and this build reads {}",
                persisted_book_index.format_version, expected.format_version
            ),
        });
    }
    if persisted_book_index.embedding_dim != expected.embedding_dim {
        return Err(VectorStoreError::Corrupted {
            reason: format!(
                "snapshot holds {}-dimensional vectors, and {} were expected",
                persisted_book_index.embedding_dim, expected.embedding_dim
            ),
        });
    }
    if let Some(required) = &expected.collection_name {
        if &persisted_book_index.collection_name != required {
            return Err(VectorStoreError::Corrupted {
                reason: format!(
                    "snapshot holds collection '{}', and '{required}' was expected",
                    persisted_book_index.collection_name
                ),
            });
        }
    }

    // Both payloads are hashed through the reader, so the declared hash covers exactly the
    // bytes the parser consumed — in order, whatever the line endings, without a second
    // pass over the file.
    let m_file = File::open(&m_path).map_err(|e| VectorStoreError::OpenFailed {
        reason: e.to_string(),
    })?;
    let mut v_file =
        HashingReader::new(
            File::open(&v_path).map_err(|e| VectorStoreError::OpenFailed {
                reason: e.to_string(),
            })?,
        );
    let mut m_reader = BufReader::new(HashingReader::new(m_file));

    let dim = expected.embedding_dim as usize;
    let vec_byte_size = dim * 4;
    let mut records: HashMap<String, StoredVectorRecord> = HashMap::new();

    for line in m_reader.by_ref().lines() {
        let line = line.map_err(|e| VectorStoreError::OpenFailed {
            reason: e.to_string(),
        })?;
        if line.is_empty() {
            continue;
        }
        let persisted: PersistedMetadata = serde_json::from_str(&line).map_err(|e| {
            log::warn!("Corrupted metadata: {e}");
            VectorStoreError::Corrupted {
                reason: e.to_string(),
            }
        })?;

        let metadata_hash = format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&persisted.metadata).map_err(|e| {
                VectorStoreError::Corrupted {
                    reason: e.to_string(),
                }
            })?)
        );
        if metadata_hash != persisted.metadata_sha256 {
            return Err(VectorStoreError::Corrupted {
                reason: format!(
                    "metadata checksum mismatch for {}",
                    persisted.metadata.semantic_id
                ),
            });
        }

        let mut buf = vec![0u8; vec_byte_size];
        if let Err(e) = v_file.read_exact(&mut buf) {
            log::warn!("Corrupted vectors file: {e}");
            return Err(VectorStoreError::Corrupted {
                reason: e.to_string(),
            });
        }

        // Over the bytes as stored, before they are decoded: this names the record that
        // broke, where the per-file hash names only the file.
        let actual_hash = format!("{:x}", Sha256::digest(&buf));
        if actual_hash != persisted.vector_sha256 {
            return Err(VectorStoreError::Corrupted {
                reason: format!(
                    "vector checksum mismatch for {}",
                    persisted.metadata.semantic_id
                ),
            });
        }

        let mut vector = Vec::with_capacity(dim);
        for i in 0..dim {
            let bytes: [u8; 4] = buf[i * 4..i * 4 + 4].try_into().unwrap();
            vector.push(f32::from_le_bytes(bytes));
        }
        normalize_for_scoring(&mut vector).map_err(|reason| VectorStoreError::Corrupted {
            reason: format!(
                "vector for {} {reason}; no search could return it",
                persisted.metadata.semantic_id
            ),
        })?;

        let semantic_id = persisted.metadata.semantic_id.clone();
        if records.contains_key(&semantic_id) {
            return Err(VectorStoreError::Corrupted {
                reason: format!("duplicate semantic id in metadata: {semantic_id}"),
            });
        }
        records.insert(
            semantic_id,
            StoredVectorRecord {
                metadata: persisted.metadata,
                vector,
            },
        );
    }

    let mut trailing = [0u8; 1];
    if v_file
        .read(&mut trailing)
        .map_err(|e| VectorStoreError::OpenFailed {
            reason: e.to_string(),
        })?
        != 0
    {
        return Err(VectorStoreError::Corrupted {
            reason: format!(
                "{VECTORS_FILENAME} holds more bytes than the {} record(s) in \
                 {METADATA_FILENAME} account for",
                records.len()
            ),
        });
    }

    // Both files have now been read to their end, so the hashes cover them whole. Checked
    // after the structural pass rather than before it, because that pass is what proves the
    // reader consumed every byte — a hash over a file nobody finished reading proves
    // nothing about what was loaded.
    expected.verify_file(METADATA_FILENAME, &m_reader.into_inner().finalize())?;
    expected.verify_file(VECTORS_FILENAME, &v_file.finalize())?;

    let mut rebuilt_book_index: HashMap<String, Vec<String>> = HashMap::new();
    for record in records.values() {
        rebuilt_book_index
            .entry(record.metadata.source_book_key.clone())
            .or_default()
            .push(record.metadata.semantic_id.clone());
    }
    for ids in rebuilt_book_index.values_mut() {
        ids.sort();
    }
    let mut stored_book_index = persisted_book_index.books;
    for ids in stored_book_index.values_mut() {
        ids.sort();
    }
    if stored_book_index != rebuilt_book_index {
        return Err(VectorStoreError::Corrupted {
            reason: "book index disagrees with vector metadata".to_string(),
        });
    }

    Ok(Some(Snapshot {
        records,
        book_index: rebuilt_book_index,
        collection_name: persisted_book_index.collection_name,
    }))
}

/// Normalize a vector for scoring, and refuse one no search could ever return.
///
/// Returns the reason it is unusable, for the caller to wrap in the error its own layer
/// reports — an insert rejects a caller's vector, a read rejects a payload's.
///
/// Rejected: a non-finite component (scores `NaN`), a norm that overflowed to `inf`
/// (whose reciprocal silently zeroes the vector), and a norm at or below
/// [`MIN_VECTOR_NORM`] (no direction to compare). Either way the record would exist and
/// be unreachable, which is worse than a rejection — the counts agree and the result is
/// simply missing.
///
/// Normalizing on the way *out* of the file as well as on the way in is deliberate: the
/// query is normalized too, so scoring is one dot product, and an artifact whose builder
/// did not normalize still scores as a cosine rather than by magnitude. It is an
/// in-memory change; nothing is written back.
fn normalize_for_scoring(vector: &mut [f32]) -> Result<(), String> {
    if let Some(position) = vector.iter().position(|value| !value.is_finite()) {
        return Err(format!(
            "has a non-finite component at {position} ({})",
            vector[position]
        ));
    }
    let norm = crate::semantic::embedding::l2_normalize(vector);
    if !norm.is_finite() {
        return Err(format!(
            "has a non-finite norm ({norm}); its magnitudes overflowed f32"
        ));
    }
    // `<=`: a vector exactly on the threshold is rejected, as in every layer.
    if norm <= MIN_VECTOR_NORM {
        return Err(format!(
            "has a norm of {norm}, at or below the minimum {MIN_VECTOR_NORM}"
        ));
    }
    Ok(())
}

fn replace_snapshot(files: &[(&Path, &Path)]) -> Result<(), VectorStoreError> {
    for (_, destination) in files {
        let backup = destination.with_extension("previous");
        if backup.exists() {
            return Err(VectorStoreError::CommitFailed {
                reason: format!("refusing to overwrite recovery file {}", backup.display()),
            });
        }
    }

    let mut parked = Vec::new();
    for (_, destination) in files {
        if destination.exists() {
            let backup = destination.with_extension("previous");
            if let Err(error) = fs::rename(destination, &backup) {
                for (parked_destination, parked_backup) in parked.iter().rev() {
                    let _ = fs::rename(parked_backup, parked_destination);
                }
                return Err(VectorStoreError::CommitFailed {
                    reason: error.to_string(),
                });
            }
            parked.push((destination.to_path_buf(), backup));
        }
    }

    let mut installed = Vec::new();
    for (temp, destination) in files {
        if let Err(error) = fs::rename(temp, destination) {
            for path in installed.iter().rev() {
                let _ = fs::remove_file(path);
            }
            for (parked_destination, parked_backup) in parked.iter().rev() {
                let _ = fs::rename(parked_backup, parked_destination);
            }
            return Err(VectorStoreError::CommitFailed {
                reason: error.to_string(),
            });
        }
        installed.push(destination.to_path_buf());
    }

    for (_, backup) in parked {
        if let Err(error) = fs::remove_file(&backup) {
            log::warn!(
                "Store committed, but could not remove recovery file {}: {error}",
                backup.display()
            );
        }
    }
    Ok(())
}

impl VectorSearchBackend for ZevcStore {
    fn backend_id(&self) -> &'static str {
        BACKEND_ID
    }

    fn is_persistent(&self) -> bool {
        true
    }

    fn embedding_dim(&self) -> u32 {
        self.config.embedding_dim
    }

    fn count(&self) -> u32 {
        self.state
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .records
            .len()
            .min(u32::MAX as usize) as u32
    }

    fn search(
        &self,
        query_vector: &[f32],
        top_k: usize,
        filters: Option<&SearchFilters>,
    ) -> Result<Vec<SemanticCandidate>, VectorStoreError> {
        let state = self
            .state
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        search_records(
            &state.records,
            self.config.embedding_dim,
            query_vector,
            top_k,
            filters,
        )
    }

    fn book_keys(&self) -> Vec<String> {
        let mut keys: Vec<String> = self
            .state
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .book_index
            .keys()
            .cloned()
            .collect();
        keys.sort();
        keys
    }

    fn book_vector_count(&self, source_book_key: &str) -> usize {
        self.state
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .book_index
            .get(source_book_key)
            .map_or(0, |ids| ids.len())
    }
}

impl VectorStoreBackend for ZevcStore {
    fn insert_batch(
        &self,
        mut batch: Vec<(VectorMetadata, Vec<f32>)>,
    ) -> Result<u32, VectorStoreError> {
        for (meta, vector) in &mut batch {
            if vector.len() as u32 != self.config.embedding_dim {
                return Err(VectorStoreError::DimensionMismatch {
                    store_dim: self.config.embedding_dim,
                    vector_dim: vector.len() as u32,
                });
            }
            normalize_for_scoring(vector).map_err(|reason| VectorStoreError::InsertFailed {
                reason: format!("vector for {} {reason}", meta.semantic_id),
            })?;
        }

        let inserted = batch.len().min(u32::MAX as usize) as u32;
        let mut state = self.state.write().unwrap_or_else(|poisoned| {
            log::warn!("ZevcStore lock was poisoned; recovering state");
            poisoned.into_inner()
        });

        for (meta, vector) in batch {
            let id = meta.semantic_id.clone();
            let record = StoredVectorRecord {
                metadata: meta.clone(),
                vector,
            };

            match state.records.insert(id.clone(), record) {
                None => {
                    state
                        .book_index
                        .entry(meta.source_book_key.clone())
                        .or_default()
                        .push(id);
                }
                Some(previous) => {
                    let previous_book = previous.metadata.source_book_key;
                    if previous_book != meta.source_book_key {
                        if let Some(ids) = state.book_index.get_mut(&previous_book) {
                            ids.retain(|existing| existing != &id);
                            if ids.is_empty() {
                                state.book_index.remove(&previous_book);
                            }
                        }
                        state
                            .book_index
                            .entry(meta.source_book_key.clone())
                            .or_default()
                            .push(id);
                    }
                }
            }
        }

        drop(state); // Drop lock before saving

        if self.config.auto_persist {
            self.save_to_disk()?;
        }

        Ok(inserted)
    }

    fn remove_by_book(&self, source_book_key: &str) -> Result<u32, VectorStoreError> {
        let mut state = self
            .state
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut deleted = 0u32;
        if let Some(ids) = state.book_index.remove(source_book_key) {
            for id in ids {
                if state.records.remove(&id).is_some() {
                    deleted += 1;
                }
            }
        }

        drop(state);
        if self.config.auto_persist && deleted > 0 {
            self.save_to_disk()?;
        }

        Ok(deleted)
    }

    fn clear(&self) -> Result<u32, VectorStoreError> {
        let _persist = self
            .persist_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut state = self
            .state
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let removed = state.records.len() as u32;
        state.records.clear();
        state.book_index.clear();

        drop(state);

        for path in [
            self.vectors_path(),
            self.metadata_path(),
            self.book_index_path(),
        ] {
            if let Err(error) = fs::remove_file(path) {
                if error.kind() != std::io::ErrorKind::NotFound {
                    return Err(VectorStoreError::CommitFailed {
                        reason: error.to_string(),
                    });
                }
            }
        }

        Ok(removed)
    }

    /// Write the snapshot. Unlike the volatile backend's no-op, this one is the commit
    /// point: everything inserted since the last call is only in memory until it returns.
    fn commit(&self) -> Result<(), VectorStoreError> {
        self.save_to_disk()
    }
}

/// Scan every record and keep the best `top_k`, shared by the writable store and the
/// read-only one so the two cannot answer the same query differently.
///
/// Both sides are normalized, so the dot product *is* the cosine. An incomparable query
/// yields nothing rather than `NaN`-scoring the whole store, and ties break on
/// `semantic_id` because `HashMap` order is randomized per run.
fn search_records(
    records: &HashMap<String, StoredVectorRecord>,
    embedding_dim: u32,
    query_vector: &[f32],
    top_k: usize,
    filters: Option<&SearchFilters>,
) -> Result<Vec<SemanticCandidate>, VectorStoreError> {
    if query_vector.len() as u32 != embedding_dim {
        return Err(VectorStoreError::DimensionMismatch {
            store_dim: embedding_dim,
            vector_dim: query_vector.len() as u32,
        });
    }
    if top_k == 0 {
        return Ok(Vec::new());
    }
    let mut query = query_vector.to_vec();
    if normalize_for_scoring(&mut query).is_err() {
        return Ok(Vec::new());
    }

    // Group facet paths by dimension once per scan rather than per record.
    let filters = filters.and_then(SearchFilters::compile);
    let mut heap: BinaryHeap<ScoredEntry> = BinaryHeap::with_capacity(top_k + 1);

    for record in records.values() {
        if let Some(compiled) = filters.as_ref() {
            if !compiled.matches(&record.metadata) {
                continue;
            }
        }

        let score = dot_product(&query, &record.vector);
        if score.is_nan() {
            continue;
        }

        let entry = ScoredEntry {
            score,
            semantic_id: &record.metadata.semantic_id,
        };
        if heap.len() < top_k {
            heap.push(entry);
        } else if let Some(weakest) = heap.peek() {
            // Worst-first ordering, so `<` reads as "better than the weakest kept".
            if entry < *weakest {
                heap.pop();
                heap.push(entry);
            }
        }
    }

    let mut candidates: Vec<SemanticCandidate> = heap
        .into_iter()
        .filter_map(|entry| {
            records
                .get(entry.semantic_id)
                .map(|record| SemanticCandidate {
                    metadata: record.metadata.clone(),
                    similarity_score: entry.score,
                })
        })
        .collect();

    candidates.sort_by(|a, b| {
        b.similarity_score
            .total_cmp(&a.similarity_score)
            .then_with(|| a.metadata.semantic_id.cmp(&b.metadata.semantic_id))
    });

    Ok(candidates)
}

#[inline]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    const LANES: usize = 8;
    let mut acc = [0.0f32; LANES];
    let mut a_chunks = a.chunks_exact(LANES);
    let mut b_chunks = b.chunks_exact(LANES);

    for (x, y) in a_chunks.by_ref().zip(b_chunks.by_ref()) {
        for lane in 0..LANES {
            acc[lane] += x[lane] * y[lane];
        }
    }

    let mut sum = acc.iter().sum::<f32>();
    for (x, y) in a_chunks.remainder().iter().zip(b_chunks.remainder().iter()) {
        sum += x * y;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempDir(PathBuf);
    impl TempDir {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "otzaria_test_zevc_{name}_{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            let _ = std::fs::create_dir_all(&path);
            Self(path)
        }
        fn path(&self) -> &std::path::Path {
            &self.0
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn sample_metadata(id: &str, book: &str) -> VectorMetadata {
        VectorMetadata {
            semantic_id: id.to_string(),
            source_book_key: book.to_string(),
            source_doc_key: format!("{book}:{id}"),
            line_id: 1,
            section_id: 10,
            line_hash: 100,
            chunk_hash: "hash".to_string(),
            content_hash: 555,
            reference: "Ref 1".to_string(),
            segment: 0,
            is_pdf: false,
            title: "Test Book".to_string(),
            facets: vec![],
        }
    }

    #[test]
    fn test_zevc_store_crud_and_persistence() {
        let dir = TempDir::new("crud");
        let config = ZevcStoreConfig {
            db_path: dir.path().to_path_buf(),
            embedding_dim: 4,
            collection_name: "test".to_string(),
            auto_persist: true,
        };

        let store = ZevcStore::open_or_create(config.clone()).unwrap();

        let batch = vec![
            (
                sample_metadata("id1", "book1.txt"),
                vec![1.0, 0.0, 0.0, 0.0],
            ),
            (
                sample_metadata("id2", "book1.txt"),
                vec![0.0, 1.0, 0.0, 0.0],
            ),
        ];
        store.insert_batch(batch).unwrap();
        assert_eq!(store.count(), 2);

        let hits = store.search(&[1.0, 0.0, 0.0, 0.0], 10, None).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].metadata.semantic_id, "id1");

        // Test persistence by opening a new instance
        let store2 = ZevcStore::open_or_create(config.clone()).unwrap();
        assert_eq!(store2.count(), 2);
        let hits2 = store2.search(&[1.0, 0.0, 0.0, 0.0], 10, None).unwrap();
        assert_eq!(hits2.len(), 2);

        // Test remove by book
        store2.remove_by_book("book1.txt").unwrap();
        assert_eq!(store2.count(), 0);

        // Test clear
        store2
            .insert_batch(vec![(
                sample_metadata("id3", "b2"),
                vec![0.0, 0.0, 1.0, 0.0],
            )])
            .unwrap();
        assert_eq!(store2.count(), 1);
        store2.clear().unwrap();
        assert_eq!(store2.count(), 0);
    }

    #[test]
    fn test_corrupted_files() {
        let dir = TempDir::new("corrupt");
        let config = ZevcStoreConfig {
            db_path: dir.path().to_path_buf(),
            embedding_dim: 4,
            collection_name: "test".to_string(),
            auto_persist: true,
        };

        let store = ZevcStore::open_or_create(config.clone()).unwrap();
        store
            .insert_batch(vec![(
                sample_metadata("id1", "book1.txt"),
                vec![1.0, 0.0, 0.0, 0.0],
            )])
            .unwrap();

        // Corrupt vectors
        let mut v_file = fs::OpenOptions::new()
            .write(true)
            .open(store.vectors_path())
            .unwrap();
        v_file.write_all(b"bad").unwrap();

        let res = ZevcStore::open_or_create(config.clone());
        assert!(matches!(res, Err(VectorStoreError::Corrupted { .. })));
    }

    #[test]
    fn invalid_batch_does_not_partially_mutate_the_store() {
        let dir = TempDir::new("atomic_batch");
        let store = ZevcStore::open_or_create(ZevcStoreConfig {
            db_path: dir.path().to_path_buf(),
            embedding_dim: 4,
            collection_name: "test".to_string(),
            auto_persist: false,
        })
        .unwrap();

        let result = store.insert_batch(vec![
            (sample_metadata("valid", "book"), vec![1.0, 0.0, 0.0, 0.0]),
            (sample_metadata("invalid", "book"), vec![1.0, 0.0]),
        ]);

        assert!(result.is_err());
        assert_eq!(store.count(), 0);
    }

    #[test]
    fn snapshot_replacement_rolls_back_every_file_on_error() {
        let dir = TempDir::new("snapshot_rollback");
        let old_a = dir.path().join("a.bin");
        let old_b = dir.path().join("b.bin");
        let new_a = dir.path().join("a.tmp");
        let missing_new_b = dir.path().join("missing.tmp");
        fs::write(&old_a, b"old-a").unwrap();
        fs::write(&old_b, b"old-b").unwrap();
        fs::write(&new_a, b"new-a").unwrap();

        let result = replace_snapshot(&[
            (new_a.as_path(), old_a.as_path()),
            (missing_new_b.as_path(), old_b.as_path()),
        ]);

        assert!(result.is_err());
        assert_eq!(fs::read(old_a).unwrap(), b"old-a");
        assert_eq!(fs::read(old_b).unwrap(), b"old-b");
    }

    #[test]
    fn concurrent_saves_produce_one_readable_snapshot() {
        let dir = TempDir::new("concurrent_saves");
        let config = ZevcStoreConfig {
            db_path: dir.path().to_path_buf(),
            embedding_dim: 4,
            collection_name: "test".to_string(),
            auto_persist: false,
        };
        let store = std::sync::Arc::new(ZevcStore::open_or_create(config.clone()).unwrap());
        store
            .insert_batch(vec![(
                sample_metadata("id", "book"),
                vec![1.0, 0.0, 0.0, 0.0],
            )])
            .unwrap();

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let store = std::sync::Arc::clone(&store);
                std::thread::spawn(move || store.save_to_disk())
            })
            .collect();
        for handle in handles {
            handle.join().unwrap().unwrap();
        }

        assert_eq!(ZevcStore::open_or_create(config).unwrap().count(), 1);
    }

    // ── the read-only opener ──

    /// Write a two-book snapshot into `dir` and return its dimension.
    fn write_snapshot(dir: &TempDir) -> u32 {
        let store = ZevcStore::open_or_create(ZevcStoreConfig {
            db_path: dir.path().to_path_buf(),
            embedding_dim: 4,
            collection_name: "chunks".to_string(),
            auto_persist: false,
        })
        .unwrap();
        store
            .insert_batch(vec![
                (sample_metadata("a1", "book_a"), vec![1.0, 0.0, 0.0, 0.0]),
                (sample_metadata("a2", "book_a"), vec![0.0, 1.0, 0.0, 0.0]),
                (sample_metadata("b1", "book_b"), vec![0.0, 0.0, 1.0, 0.0]),
            ])
            .unwrap();
        store.commit().unwrap();
        4
    }

    /// Read a snapshot the way [`ReadOnlyZevcStore::open`] does, without a
    /// [`VerifiedPackage`].
    ///
    /// The reader's own constructor takes nothing but the token, on purpose, and a token can
    /// only come from verifying a whole artifact — model identity included. These tests are
    /// about the *format*, so they drive the function both openers share and pass the
    /// declaration directly; the artifact-level path is covered in
    /// [`official_index`](crate::semantic::official_index).
    fn read_as_artifact(
        dir: &TempDir,
        embedding_dim: u32,
        declared_sha256: BTreeMap<String, String>,
    ) -> Result<usize, VectorStoreError> {
        let snapshot = read_snapshot(
            dir.path(),
            &SnapshotExpectation {
                embedding_dim,
                format_version: STORE_FORMAT_VERSION,
                collection_name: None,
                declared_sha256: Some(declared_sha256),
            },
        )?;
        Ok(snapshot.map_or(0, |snapshot| snapshot.records.len()))
    }

    /// What an artifact declares about its payload files, as they are on disk *right now*.
    ///
    /// Which moment a test calls this at is the whole point: captured before a tamper, it
    /// stands for `payloads.json` as the packer wrote it; captured after, it stands for an
    /// attacker who re-stamped it — and then only the checks inside the payload are left.
    fn declared(dir: &TempDir) -> BTreeMap<String, String> {
        SNAPSHOT_FILENAMES
            .iter()
            .filter(|name| dir.path().join(name).exists())
            .map(|name| {
                (
                    (*name).to_string(),
                    format!(
                        "{:x}",
                        Sha256::digest(fs::read(dir.path().join(name)).unwrap())
                    ),
                )
            })
            .collect()
    }

    #[test]
    fn a_read_only_store_answers_the_same_query_as_the_writable_one() {
        let dir = TempDir::new("read_only_open");
        let dim = write_snapshot(&dir);

        let snapshot = read_snapshot(
            dir.path(),
            &SnapshotExpectation {
                embedding_dim: dim,
                format_version: STORE_FORMAT_VERSION,
                collection_name: None,
                declared_sha256: Some(declared(&dir)),
            },
        )
        .unwrap()
        .expect("the snapshot is there");
        let reader = ReadOnlyZevcStore {
            records: snapshot.records,
            book_index: snapshot.book_index,
            embedding_dim: dim,
            collection_name: snapshot.collection_name,
        };

        assert_eq!(reader.count(), 3);
        assert_eq!(reader.book_keys(), ["book_a", "book_b"]);
        assert_eq!(reader.book_vector_count("book_a"), 2);
        assert_eq!(reader.book_vector_count("absent"), 0);
        assert!(reader.is_persistent());
        assert_eq!(reader.backend_id(), BACKEND_ID);
        assert_eq!(reader.collection_name(), "chunks");

        let hits = reader.search(&[0.0, 0.0, 1.0, 0.0], 2, None).unwrap();
        assert_eq!(hits[0].metadata.semantic_id, "b1");
        assert!((hits[0].similarity_score - 1.0).abs() < 1e-6);
    }

    /// Rewrite the first record's vector, and the checksum that record carries for it, so
    /// the payload is internally consistent about a vector nobody published. Both edits keep
    /// their length: the vector is fixed-width, and a SHA-256 is 64 hex digits either way.
    fn forge_first_record(dir: &TempDir, dim: u32) {
        let vectors_path = dir.path().join(VECTORS_FILENAME);
        let mut bytes = fs::read(&vectors_path).unwrap();
        let length_before = bytes.len();
        bytes[0] ^= 0xff;
        fs::write(&vectors_path, &bytes).unwrap();
        assert_eq!(
            fs::metadata(&vectors_path).unwrap().len() as usize,
            length_before
        );

        let metadata_path = dir.path().join(METADATA_FILENAME);
        let text = fs::read_to_string(&metadata_path).unwrap();
        let length_before = text.len();
        let mut lines: Vec<String> = text.lines().map(str::to_string).collect();
        let mut first: PersistedMetadata = serde_json::from_str(&lines[0]).unwrap();
        first.vector_sha256 = format!("{:x}", Sha256::digest(&bytes[..dim as usize * 4]));
        lines[0] = serde_json::to_string(&first).unwrap();
        fs::write(&metadata_path, format!("{}\n", lines.join("\n"))).unwrap();
        assert_eq!(
            fs::read_to_string(&metadata_path).unwrap().len(),
            length_before,
            "the forgery must not change the file's length, or a size check would catch it"
        );
    }

    /// The attack the per-record checksums cannot see, because they travel inside the bytes
    /// they guard: a vector edited *together with* its own checksum. What catches it is the
    /// hash the artifact declared for the whole file — the one an attacker can only change
    /// in `payloads.json`, which a published digest pins.
    #[test]
    fn a_vector_forged_together_with_its_own_checksum_is_refused_by_the_declared_hash() {
        let dir = TempDir::new("read_only_forgery");
        let dim = write_snapshot(&dir);
        let as_published = declared(&dir);

        forge_first_record(&dir, dim);

        // Named for `metadata.jsonl`, and that is not incidental: a record's checksum lives
        // in a *different file* from the bytes it covers, so forging a vector always
        // disturbs two files, and the metadata file is the one read to its end first.
        match read_as_artifact(&dir, dim, as_published) {
            Err(VectorStoreError::Corrupted { reason }) => assert!(
                reason.contains(METADATA_FILENAME) && reason.contains("SHA-256"),
                "{reason}"
            ),
            other => panic!("a forged record must be refused, got {other:?}"),
        }
    }

    /// The honest limit of the same attack: re-stamp the declaration too, and every layer
    /// agrees. Nothing inside the artifact can tell — only a digest published outside it,
    /// which is checked before the reader ever runs.
    #[test]
    fn a_forgery_that_also_restamps_the_declaration_is_beyond_what_the_reader_can_see() {
        let dir = TempDir::new("read_only_restamped");
        let dim = write_snapshot(&dir);

        forge_first_record(&dir, dim);

        let records = read_as_artifact(&dir, dim, declared(&dir))
            .expect("a self-consistent forgery is indistinguishable from here");
        assert_eq!(records, 3);
    }

    /// And the inner layer still earns its place: an edit that does *not* fix the record's
    /// checksum is named per record, which is what tells the user where the damage is.
    #[test]
    fn a_same_length_edit_of_one_vector_is_named_by_its_record() {
        let dir = TempDir::new("read_only_edit");
        let dim = write_snapshot(&dir);

        let vectors_path = dir.path().join(VECTORS_FILENAME);
        let mut bytes = fs::read(&vectors_path).unwrap();
        let length_before = bytes.len();
        bytes[0] ^= 0xff;
        fs::write(&vectors_path, &bytes).unwrap();
        assert_eq!(
            fs::metadata(&vectors_path).unwrap().len() as usize,
            length_before
        );

        match read_as_artifact(&dir, dim, declared(&dir)) {
            Err(VectorStoreError::Corrupted { reason }) => {
                assert!(reason.contains("checksum mismatch"), "{reason}")
            }
            other => panic!("expected a per-record checksum rejection, got {other:?}"),
        }
    }

    #[test]
    fn a_payload_the_reader_cannot_decode_is_refused_rather_than_misread() {
        let dir = TempDir::new("read_only_shape");
        let dim = write_snapshot(&dir);

        // A width this build does not read: the header says otherwise, so nothing is
        // decoded at the wrong offset.
        match read_as_artifact(&dir, dim + 1, declared(&dir)) {
            Err(VectorStoreError::Corrupted { reason }) => {
                assert!(reason.contains("dimensional"), "{reason}")
            }
            other => panic!("expected a dimension rejection, got {other:?}"),
        }

        // Two of three files: the snapshot is written as one commit, so this is an
        // interrupted write and not an empty index.
        fs::remove_file(dir.path().join(BOOK_INDEX_FILENAME)).unwrap();
        match read_as_artifact(&dir, dim, declared(&dir)) {
            Err(VectorStoreError::Corrupted { reason }) => {
                assert!(reason.contains(BOOK_INDEX_FILENAME), "{reason}")
            }
            other => panic!("expected an incompleteness rejection, got {other:?}"),
        }
    }

    /// A builder may start from nothing, so "no snapshot" is not a fault at this level. It
    /// is one for an artifact, and [`ReadOnlyZevcStore::open`] is where that is decided.
    #[test]
    fn a_directory_with_no_snapshot_is_reported_as_absent_rather_than_corrupt() {
        let dir = TempDir::new("read_only_empty");

        assert_eq!(
            ZevcStore::open_or_create(ZevcStoreConfig {
                db_path: dir.path().to_path_buf(),
                embedding_dim: 4,
                collection_name: "chunks".to_string(),
                auto_persist: false,
            })
            .unwrap()
            .count(),
            0
        );

        assert_eq!(
            read_as_artifact(&dir, 4, BTreeMap::new()).unwrap(),
            0,
            "no snapshot at all is not corruption at this level; refusing it is the \
             artifact reader's job"
        );
    }

    /// A vector no search could ever return is a record that exists and is unreachable,
    /// which is worse than a rejection: the counts agree and the result is missing.
    #[test]
    fn a_stored_vector_with_no_usable_direction_is_refused_at_open() {
        let dir = TempDir::new("read_only_direction");
        let dim = 4;
        let store = ZevcStore::open_or_create(ZevcStoreConfig {
            db_path: dir.path().to_path_buf(),
            embedding_dim: dim,
            collection_name: "chunks".to_string(),
            auto_persist: false,
        })
        .unwrap();
        store
            .insert_batch(vec![(
                sample_metadata("a1", "book_a"),
                vec![1.0, 0.0, 0.0, 0.0],
            )])
            .unwrap();
        store.commit().unwrap();

        // Rewrite the one record as zeros, with the checksum the file's own metadata
        // would carry, so the only thing left to catch it is the direction check.
        let zeros = vec![0u8; dim as usize * 4];
        fs::write(dir.path().join(VECTORS_FILENAME), &zeros).unwrap();
        let metadata_path = dir.path().join(METADATA_FILENAME);
        let line = fs::read_to_string(&metadata_path).unwrap();
        let mut persisted: PersistedMetadata = serde_json::from_str(line.trim()).unwrap();
        persisted.vector_sha256 = format!("{:x}", Sha256::digest(&zeros));
        fs::write(
            &metadata_path,
            format!("{}\n", serde_json::to_string(&persisted).unwrap()),
        )
        .unwrap();

        match read_as_artifact(&dir, dim, declared(&dir)) {
            Err(VectorStoreError::Corrupted { reason }) => {
                assert!(reason.contains("no search could return it"), "{reason}")
            }
            other => panic!("expected a direction rejection, got {other:?}"),
        }
    }
}
