//! The build machine's vectors: every one ever embedded, as `f32`, by the full SHA-256 of
//! its text.
//!
//! ```text
//! <warehouse>/
//!   warehouse.json   the identity — the family's width, tokenizer, pooling and token cap,
//!                    and the one passage package every vector here was embedded with —
//!                    the record count, and one entry per batch added
//!   vectors.f32      records × dim little-endian f32s, append-only
//!   keys.bin         records × 32 bytes: each record's SHA-256, in the same order
//!   index.bin        b"OXVWIDX1", u64 count, then {sha256 [32], record u64} by sha256
//!   .lock            held while a batch is added
//! ```
//!
//! **One package per warehouse.** Vectors of two packages of a family are close, not equal;
//! a warehouse that mixed them would assemble segments whose vectors came from both. A shard
//! of another package is refused.
//!
//! **`warehouse.json` is the commit point.** A batch appends to both data files, writes a
//! new index and renames it into place, and only then records the new count. Opening for an
//! append truncates whatever a crash left past the count; opening to read ignores it.
//!
//! **Checked before reuse.** [`Warehouse::verify`] re-hashes the batches and checks the index;
//! assembly, the gates and an append run it themselves, and every lookup confirms its key.

use crate::distribution::files::{hex, io_error, malformed, partial_path, read_json, write_json};
use crate::distribution::plan::HeldVectors;
use crate::distribution::shard::{
    verify_shards, CheckedShard, ParityCertificate, ShardPolicy, WorkerInfo, KEYS_FILE,
    VECTORS_FILE,
};
use crate::errors::PackError;
use crate::semantic::versioning::{ModelIdentity, ModelPackage};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Reverse;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

pub const WAREHOUSE_FORMAT: &str = "otzaria-vector-warehouse";
pub const WAREHOUSE_FORMAT_VERSION: u32 = 1;
const MANIFEST_FILE: &str = "warehouse.json";
const VECTORS: &str = "vectors.f32";
const KEYS: &str = "keys.bin";
const INDEX: &str = "index.bin";
const LOCK: &str = ".lock";
const INDEX_MAGIC: &[u8; 8] = b"OXVWIDX1";
const HEADER: usize = 16;
const ENTRY: usize = 40;

/// What every vector of a warehouse was embedded under.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WarehouseIdentity {
    pub family_id: String,
    pub tokenizer_checksum: String,
    pub embedding_dim: u32,
    pub pooling: String,
    pub max_tokens: usize,
    pub passage_package: ModelPackage,
}

impl WarehouseIdentity {
    /// The part of a family a vector depends on, given its exact text: the weights, the
    /// tokenizer, the width, the pooling and the token cap — not the text recipes, which
    /// the key already covers.
    pub fn of(model: &ModelIdentity, package: &ModelPackage) -> Self {
        Self {
            family_id: model.family_id.clone(),
            tokenizer_checksum: model.tokenizer_checksum.clone(),
            embedding_dim: model.embedding_dim,
            pooling: model.pooling.clone(),
            max_tokens: model.max_tokens,
            passage_package: package.clone(),
        }
    }

    /// Check against an independently declared family, including both fields of the
    /// passage package. Reconstructing an identity with an unchecked package from this
    /// warehouse would only compare that package with itself.
    pub fn ensure_model(&self, model: &ModelIdentity) -> Result<(), PackError> {
        if !model.query_packages.contains(&self.passage_package) {
            return Err(malformed(format!(
                "the warehouse's passage package {} ({}) is not one of the family's",
                self.passage_package.checksum, self.passage_package.quantization
            )));
        }
        if Self::of(model, &self.passage_package) != *self {
            return Err(malformed(format!(
                "the warehouse's vector identity does not match family {}",
                model.family_id
            )));
        }
        Ok(())
    }
}

/// One batch a warehouse took in.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Batch {
    /// The first record it added.
    pub first: u64,
    /// Records it added; keys the warehouse held already are not added again.
    pub records: u64,
    pub vectors_sha256: String,
    pub keys_sha256: String,
    /// The embed plan its shards were checked against; none for an import.
    pub plan_sha256: Option<String>,
    pub shards: u64,
    pub worker: WorkerInfo,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parity: Option<ParityCertificate>,
    pub added_at: String,
}

/// `warehouse.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WarehouseManifest {
    pub format: String,
    pub version: u32,
    pub identity: WarehouseIdentity,
    pub records: u64,
    pub batches: Vec<Batch>,
}

/// What [`Warehouse::add_shards`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AddReport {
    pub shards: usize,
    /// Records the shards held.
    pub records: u64,
    /// Of those, the ones added.
    pub added: u64,
    /// Of those, the ones the warehouse held already.
    pub held: u64,
    /// Records in the warehouse now.
    pub total: u64,
}

/// What [`Warehouse::verify`] checked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Verified {
    pub records: u64,
    pub batches: usize,
    /// Bytes re-hashed: both data files, up to the count.
    pub bytes: u64,
    /// Why an append rebuilt the index from the verified keys, if it did.
    pub index_rebuilt: Option<String>,
}

/// What a check found wrong: data, which nothing here repairs, and the index, which the
/// verified keys can rebuild.
struct Faults {
    data: Vec<String>,
    /// Every data fault is a digest recorded for a batch of no records.
    manifest_only: bool,
    index: Option<String>,
}

/// A warehouse, open.
pub struct Warehouse {
    dir: PathBuf,
    manifest: WarehouseManifest,
    vectors: Option<Mmap>,
    keys: Option<Mmap>,
    index: Option<Mmap>,
    /// Set once the mapped files have passed [`Warehouse::verify`].
    verified: OnceLock<Verified>,
    _lock: Option<AppendLock>,
}

/// The lock an append holds on its warehouse, for as long as the value lives.
struct AppendLock(File);

impl AppendLock {
    fn take(dir: &Path) -> Result<Self, PackError> {
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(dir.join(LOCK))
            .map_err(io_error(format!("opening {}", dir.join(LOCK).display())))?;
        match lock.try_lock() {
            Ok(()) => Ok(Self(lock)),
            Err(std::fs::TryLockError::WouldBlock) => Err(malformed(format!(
                "{} is being added to by another process",
                dir.display()
            ))),
            Err(std::fs::TryLockError::Error(source)) => Err(PackError::Io {
                context: format!("locking {}", dir.display()),
                source,
            }),
        }
    }
}

impl Drop for AppendLock {
    /// Unlocked, not only closed: a process spawned meanwhile shares the descriptor until it
    /// execs, and closing ours would leave the lock held by its copy.
    fn drop(&mut self) {
        let _ = self.0.unlock();
    }
}

impl Warehouse {
    /// Create an empty warehouse for `identity` at `dir`, which must not hold one.
    pub fn create(dir: &Path, identity: WarehouseIdentity) -> Result<Self, PackError> {
        std::fs::create_dir_all(dir).map_err(io_error(format!("creating {}", dir.display())))?;
        if dir.join(MANIFEST_FILE).exists() {
            return Err(PackError::UnusableOutput {
                path: dir.display().to_string(),
                reason: "it holds a warehouse already".to_string(),
            });
        }
        for name in [VECTORS, KEYS] {
            File::create(dir.join(name))
                .map_err(io_error(format!("creating {}", dir.join(name).display())))?;
        }
        write_index(&dir.join(INDEX), std::iter::empty())?;
        let manifest = WarehouseManifest {
            format: WAREHOUSE_FORMAT.to_string(),
            version: WAREHOUSE_FORMAT_VERSION,
            identity,
            records: 0,
            batches: Vec::new(),
        };
        write_json(&dir.join(MANIFEST_FILE), &manifest)?;
        Self::open(dir)
    }

    /// Open to read: what `warehouse.json` counts, whatever a crash left past it. Checks
    /// sizes and headers only; [`Self::verify`] checks the contents.
    pub fn open(dir: &Path) -> Result<Self, PackError> {
        let deadline = Instant::now() + Duration::from_secs(1);
        loop {
            let manifest = read_manifest(dir)?;
            let records = manifest.records;
            // An add renames its index into place before it writes the count: wait up to 1 s
            // for the count, holding no map.
            let wait = || Instant::now() < deadline && index_ahead(dir, records);
            if !wait() {
                let mut warehouse = Self::unmapped(dir, manifest);
                match warehouse.map() {
                    Err(_) if wait() => {}
                    result => return result.map(|()| warehouse),
                }
            }
            let left = deadline.saturating_duration_since(Instant::now());
            std::thread::sleep(left.min(Duration::from_millis(20)));
        }
    }

    fn unmapped(dir: &Path, manifest: WarehouseManifest) -> Self {
        Self {
            dir: dir.to_path_buf(),
            manifest,
            vectors: None,
            keys: None,
            index: None,
            verified: OnceLock::new(),
            _lock: None,
        }
    }

    /// Open to add a batch: hold the lock, cut what a crash left past the count, verify the
    /// data, and rebuild an index that is not the verified keys'.
    pub fn open_for_append(dir: &Path) -> Result<Self, PackError> {
        let lock = AppendLock::take(dir)?;
        let manifest = read_manifest(dir)?;
        let width = manifest.identity.embedding_dim as u64 * 4;
        for (name, bytes) in [
            (VECTORS, manifest.records * width),
            (KEYS, manifest.records * 32),
        ] {
            let path = dir.join(name);
            let file = OpenOptions::new()
                .write(true)
                .open(&path)
                .map_err(io_error(format!("opening {}", path.display())))?;
            let length = file
                .metadata()
                .map_err(io_error(format!("inspecting {}", path.display())))?
                .len();
            if length < bytes {
                return Err(malformed(format!(
                    "{} is {length} bytes and the warehouse counts {} records",
                    path.display(),
                    manifest.records
                )));
            }
            if length > bytes {
                file.set_len(bytes)
                    .map_err(io_error(format!("truncating {}", path.display())))?;
                file.sync_all()
                    .map_err(io_error(format!("flushing {}", path.display())))?;
            }
        }
        let mut warehouse = Self::unmapped(dir, manifest);
        warehouse._lock = Some(lock);
        warehouse.map_data()?;
        let index = map_file(&dir.join(INDEX));
        let mut faults = warehouse.faults(index.as_deref().ok())?;
        if let Err(error) = &index {
            faults.index = Some(error.to_string());
        }
        // Unmapped before it is replaced: Windows refuses to rename over a mapped file.
        drop(index);
        if !faults.data.is_empty() {
            return Err(warehouse.refusal(faults));
        }
        let records = warehouse.manifest.records as usize;
        if faults.index.is_some() {
            let keys = &warehouse.keys.as_deref().expect("mapped")[..records * 32];
            rebuild_index(&dir.join(INDEX), keys)?;
        }
        warehouse.map_index()?;
        if faults.index.is_some() {
            let keys = warehouse.keys.as_deref().expect("mapped");
            let index = warehouse.index.as_deref().expect("mapped");
            if let Some(fault) = index_fault(index, keys, records as u64) {
                return Err(malformed(format!(
                    "{} was rebuilt and is still not the index of {KEYS}: {fault}",
                    dir.join(INDEX).display()
                )));
            }
        }
        let verified = warehouse.report(faults.index);
        let _ = warehouse.verified.set(verified);
        Ok(warehouse)
    }

    fn map(&mut self) -> Result<(), PackError> {
        self.map_data()?;
        self.map_index()
    }

    /// Map both data files, each at least the count's length.
    fn map_data(&mut self) -> Result<(), PackError> {
        let records = self.manifest.records;
        let map = |name: &str, width: u64| -> Result<Mmap, PackError> {
            let path = self.dir.join(name);
            let map = map_file(&path)?;
            if (map.len() as u64) < records * width {
                return Err(malformed(format!(
                    "{} is short of the {records} records the warehouse counts",
                    path.display()
                )));
            }
            Ok(map)
        };
        let vectors = map(VECTORS, self.width() as u64)?;
        let keys = map(KEYS, 32)?;
        self.vectors = Some(vectors);
        self.keys = Some(keys);
        Ok(())
    }

    /// Map the index, refusing one whose header is not the count's.
    fn map_index(&mut self) -> Result<(), PackError> {
        let path = self.dir.join(INDEX);
        let index = map_file(&path)?;
        let records = self.manifest.records;
        if let Some(fault) = header_fault(&index, records) {
            let ahead = header_count(&index).is_some_and(|count| count > records);
            let added_to = if ahead {
                "the warehouse may be being added to; if not, "
            } else {
                ""
            };
            return Err(malformed(format!(
                "{} is not the index of the warehouse's {records} records ({fault}): {added_to}\
                 adding to the warehouse, or warehouse-verify --repair, rebuilds it",
                path.display()
            )));
        }
        self.index = Some(index);
        Ok(())
    }

    /// Re-hash every batch against `warehouse.json` and check the index against `keys.bin`;
    /// once per open, on up to 8 threads reading 1 MiB at a time.
    pub fn verify(&self) -> Result<Verified, PackError> {
        if let Some(verified) = self.verified.get() {
            return Ok(verified.clone());
        }
        let (Some(_), Some(_), Some(index)) = (&self.vectors, &self.keys, &self.index) else {
            return Err(malformed(format!(
                "{} is not open: an add to it failed",
                self.dir.display()
            )));
        };
        let faults = self.faults(Some(index))?;
        if !faults.data.is_empty() || faults.index.is_some() {
            return Err(self.refusal(faults));
        }
        Ok(self.verified.get_or_init(|| self.report(None)).clone())
    }

    fn report(&self, index_rebuilt: Option<String>) -> Verified {
        Verified {
            records: self.manifest.records,
            batches: self.manifest.batches.len(),
            bytes: self.manifest.records * (self.width() as u64 + 32),
            index_rebuilt,
        }
    }

    /// Hash every batch, largest first, and check `index` if given, all in parallel.
    fn faults(&self, index: Option<&[u8]>) -> Result<Faults, PackError> {
        let keys = self.keys.as_deref().expect("mapped");
        let width = self.width() as u64;
        let mut jobs: Vec<(usize, &str, u64, u64, &str)> = Vec::new();
        for (at, batch) in self.manifest.batches.iter().enumerate() {
            let (first, records) = (batch.first, batch.records);
            let vectors = &batch.vectors_sha256;
            jobs.push((at, VECTORS, first * width, records * width, vectors));
            jobs.push((at, KEYS, first * 32, records * 32, &batch.keys_sha256));
        }
        jobs.sort_by_key(|job| Reverse(job.3));
        let next = AtomicUsize::new(0);
        let threads = std::thread::available_parallelism()
            .map_or(1, usize::from)
            .min(8)
            .min(jobs.len())
            .max(1);
        let (data, index) = std::thread::scope(|scope| {
            let index = scope
                .spawn(|| index.and_then(|index| index_fault(index, keys, self.manifest.records)));
            let hashers: Vec<_> = (0..threads)
                .map(|_| {
                    scope.spawn(|| -> Result<Vec<_>, PackError> {
                        let mut wrong = Vec::new();
                        while let Some(&(at, name, offset, len, recorded)) =
                            jobs.get(next.fetch_add(1, Ordering::Relaxed))
                        {
                            let actual = hash_range(&self.dir.join(name), offset, len)?;
                            if actual != recorded {
                                wrong.push((at, name, actual, recorded));
                            }
                        }
                        Ok(wrong)
                    })
                })
                .collect();
            let mut data = Vec::new();
            for hasher in hashers {
                data.push(hasher.join().expect("a hashing thread"));
            }
            (data, index.join().expect("the index check"))
        });
        let mut wrong = Vec::new();
        for found in data {
            wrong.extend(found?);
        }
        wrong.sort_unstable();
        let manifest_only = wrong
            .iter()
            .all(|(at, ..)| self.manifest.batches[*at].records == 0);
        let data = wrong
            .into_iter()
            .map(|(at, name, actual, recorded)| {
                let batch = &self.manifest.batches[at];
                let found = if batch.records == 0 {
                    format!(
                        "it added no bytes, so warehouse.json's {recorded} for {name} is corrupt"
                    )
                } else {
                    format!("{name} hashes to {actual}, and warehouse.json records {recorded}")
                };
                format!(
                    "batch {at} (records {}..{}, added {}): {found}",
                    batch.first,
                    batch.first + batch.records,
                    batch.added_at
                )
            })
            .collect();
        Ok(Faults {
            data,
            manifest_only,
            index,
        })
    }

    fn refusal(&self, faults: Faults) -> PackError {
        let mut found = faults.data.clone();
        if let Some(index) = faults.index {
            found.push(format!("{INDEX} is not the index of {KEYS}: {index}"));
        }
        let remedy = if faults.data.is_empty() {
            "the keys are sound, so adding to the warehouse, or warehouse-verify --repair, \
             rebuilds the index from them"
        } else if faults.manifest_only {
            "the data is sound: restore warehouse.json, or set that digest to SHA-256 of nothing, \
             e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        } else {
            "data that fails its digest is not repaired: restore the warehouse from a copy, or \
             move it aside and the next build embeds every text again"
        };
        malformed(format!(
            "warehouse {}: {}; {remedy}",
            self.dir.display(),
            found.join("; ")
        ))
    }

    pub fn identity(&self) -> &WarehouseIdentity {
        &self.manifest.identity
    }

    pub fn manifest(&self) -> &WarehouseManifest {
        &self.manifest
    }

    pub fn dim(&self) -> usize {
        self.manifest.identity.embedding_dim as usize
    }

    fn width(&self) -> usize {
        self.dim() * 4
    }

    pub fn len(&self) -> u64 {
        self.manifest.records
    }

    pub fn is_empty(&self) -> bool {
        self.manifest.records == 0
    }

    fn entries(&self) -> Option<&[[u8; ENTRY]]> {
        Some(self.index.as_ref()?[HEADER..].as_chunks::<ENTRY>().0)
    }

    /// The record holding the vector of the text whose SHA-256 is `sha256`. None as well
    /// when `keys.bin` holds another key there: a corrupt index finds nothing.
    pub fn find(&self, sha256: &[u8; 32]) -> Option<u64> {
        self.confirmed(self.indexed(sha256)?)
    }

    /// The index's entry for `sha256`, unconfirmed: for an append, which maps no data file
    /// and runs on an index it verified.
    fn indexed(&self, sha256: &[u8; 32]) -> Option<&[u8; ENTRY]> {
        let entries = self.entries()?;
        let at = entries
            .binary_search_by(|entry| entry[..32].cmp(sha256))
            .ok()?;
        Some(&entries[at])
    }

    /// The record of a text whose SHA-256 starts with `key`, a segment's `ChunkKey`
    /// (prefixes of a digest-ordered index are in order too), confirmed as [`Self::find`] is.
    pub fn find_key(&self, key: &[u8; 16]) -> Option<u64> {
        let entries = self.entries()?;
        let at = entries.partition_point(|entry| entry[..16] < key[..]);
        entries
            .get(at)
            .filter(|entry| entry[..16] == key[..])
            .and_then(|entry| self.confirmed(entry))
    }

    /// The entry's record, if it is counted and `keys.bin` holds the entry's key there.
    fn confirmed(&self, entry: &[u8; ENTRY]) -> Option<u64> {
        let record = u64::from_le_bytes(entry[32..].try_into().expect("8"));
        let keys = self.keys.as_ref()?;
        (record < self.manifest.records && keys[record as usize * 32..][..32] == entry[..32])
            .then_some(record)
    }

    /// Record `record`'s vector, into `out`.
    pub fn vector(&self, record: u64, out: &mut [f32]) {
        let width = self.width();
        let bytes = &self.vectors.as_ref().expect("mapped")[record as usize * width..][..width];
        for (value, bytes) in out.iter_mut().zip(bytes.as_chunks::<4>().0) {
            *value = f32::from_le_bytes(*bytes);
        }
    }

    /// Check `dirs` — against the plan in `plan_dir`, or without it for an import — and
    /// add every vector of a key the warehouse does not hold.
    ///
    /// What it holds: 40 bytes for every record it adds, to sort into the index.
    pub fn add_shards(
        &mut self,
        plan_dir: Option<&Path>,
        dirs: &[PathBuf],
        policy: &ShardPolicy,
        added_at: String,
    ) -> Result<AddReport, PackError> {
        if self._lock.is_none() {
            return Err(malformed("a warehouse is added to through open_for_append"));
        }
        let shards = verify_shards(plan_dir, dirs, policy)?;
        for shard in &shards {
            let identity =
                WarehouseIdentity::of(&shard.manifest.model, &shard.manifest.passage_package);
            if identity != self.manifest.identity {
                return Err(malformed(format!(
                    "{} was embedded with package {} of family {}, and this warehouse holds \
                     package {} of family {}: a warehouse holds one package",
                    shard.dir.display(),
                    identity.passage_package.checksum,
                    identity.family_id,
                    self.manifest.identity.passage_package.checksum,
                    self.manifest.identity.family_id
                )));
            }
        }
        let first = self.manifest.records;
        let width = self.width();
        // Nothing maps a file this appends to: Windows refuses to resize a mapped one.
        self.vectors = None;
        self.keys = None;
        self.verified = OnceLock::new();
        let result = self.append(&shards, width);
        let (entries, vectors_sha256, keys_sha256, records) = match result {
            Ok(appended) => appended,
            Err(error) => {
                self.truncate(first)?;
                self.map_data()?;
                return Err(error);
            }
        };
        let added = entries.len() as u64;
        let total = first + added;

        // The new index: the old one and the new entries, merged as they stream; the old
        // one is unmapped only once the new one is written, to rename over it.
        let partial = {
            let old = self.index.as_ref().expect("mapped")[HEADER..]
                .as_chunks::<ENTRY>()
                .0
                .iter()
                .copied();
            write_index_partial(&self.dir.join(INDEX), Merge::new(old, entries.into_iter()))?
        };
        self.index = None;
        commit_index(&partial, &self.dir.join(INDEX))?;

        let reference = &shards[0].manifest;
        self.manifest.records = total;
        self.manifest.batches.push(Batch {
            first,
            records: added,
            vectors_sha256,
            keys_sha256,
            plan_sha256: plan_dir.map(|_| reference.plan_sha256.clone()),
            shards: shards.len() as u64,
            worker: reference.worker.clone(),
            parity: reference.parity.clone(),
            added_at,
        });
        write_json(&self.dir.join(MANIFEST_FILE), &self.manifest)?;
        self.map()?;
        Ok(AddReport {
            shards: shards.len(),
            records,
            added,
            held: records - added,
            total,
        })
    }

    /// Append the shards' new records; return their sorted index entries, the digests of
    /// what was appended, and the records the shards held.
    #[allow(clippy::type_complexity)]
    fn append(
        &self,
        shards: &[CheckedShard],
        width: usize,
    ) -> Result<(Vec<[u8; ENTRY]>, String, String, u64), PackError> {
        let open = |name: &str| -> Result<BufWriter<File>, PackError> {
            let path = self.dir.join(name);
            Ok(BufWriter::with_capacity(
                4 << 20,
                OpenOptions::new()
                    .append(true)
                    .open(&path)
                    .map_err(io_error(format!("opening {}", path.display())))?,
            ))
        };
        let (mut vectors_out, mut keys_out) = (open(VECTORS)?, open(KEYS)?);
        let (mut vectors_hash, mut keys_hash) = (Sha256::new(), Sha256::new());
        let mut entries: Vec<[u8; ENTRY]> = Vec::new();
        let mut next = self.manifest.records;
        let mut records = 0u64;
        let (mut vector, mut key) = (vec![0u8; width], [0u8; 32]);
        for shard in shards {
            let reader = |name: &str| -> Result<BufReader<File>, PackError> {
                let path = shard.dir.join(name);
                Ok(BufReader::with_capacity(
                    4 << 20,
                    File::open(&path).map_err(io_error(format!("reading {}", path.display())))?,
                ))
            };
            let (mut vectors_in, mut keys_in) = (reader(VECTORS_FILE)?, reader(KEYS_FILE)?);
            for _ in 0..shard.manifest.records {
                vectors_in
                    .read_exact(&mut vector)
                    .and_then(|()| keys_in.read_exact(&mut key))
                    .map_err(io_error(format!("reading {}", shard.dir.display())))?;
                records += 1;
                if self.indexed(&key).is_some() {
                    continue;
                }
                let context = || format!("appending to {}", self.dir.display());
                vectors_out
                    .write_all(&vector)
                    .map_err(io_error(context()))?;
                keys_out.write_all(&key).map_err(io_error(context()))?;
                vectors_hash.update(&vector);
                keys_hash.update(key);
                let mut entry = [0u8; ENTRY];
                entry[..32].copy_from_slice(&key);
                entry[32..].copy_from_slice(&next.to_le_bytes());
                entries.push(entry);
                next += 1;
            }
        }
        for mut out in [vectors_out, keys_out] {
            out.flush()
                .and_then(|()| out.get_ref().sync_all())
                .map_err(io_error(format!("flushing {}", self.dir.display())))?;
        }
        entries.sort_unstable_by(|a, b| a[..32].cmp(&b[..32]));
        if let Some(pair) = entries
            .windows(2)
            .find(|pair| pair[0][..32] == pair[1][..32])
        {
            return Err(malformed(format!(
                "the shards hold key {} twice",
                hex(&pair[0][..32])
            )));
        }
        Ok((
            entries,
            hex(&vectors_hash.finalize()),
            hex(&keys_hash.finalize()),
            records,
        ))
    }

    /// Cut both data files back to `records`, undoing an append that failed.
    fn truncate(&self, records: u64) -> Result<(), PackError> {
        for (name, bytes) in [
            (VECTORS, records * self.width() as u64),
            (KEYS, records * 32),
        ] {
            let path = self.dir.join(name);
            OpenOptions::new()
                .write(true)
                .open(&path)
                .and_then(|file| file.set_len(bytes))
                .map_err(io_error(format!("truncating {}", path.display())))?;
        }
        Ok(())
    }
}

impl HeldVectors for Warehouse {
    fn holds(&self, sha256: &[u8; 32]) -> bool {
        self.find(sha256).is_some()
    }
}

fn map_file(path: &Path) -> Result<Mmap, PackError> {
    let file = File::open(path).map_err(io_error(format!("opening {}", path.display())))?;
    // SAFETY: a warehouse's files are only appended to, under its lock, past what the
    // manifest counts — never written where a reader has mapped — and the index is replaced
    // by a rename, which leaves an open mapping of the old one valid.
    unsafe { Mmap::map(&file) }.map_err(io_error(format!("mapping {}", path.display())))
}

/// SHA-256 of `len` bytes of `path` from `offset`.
fn hash_range(path: &Path, offset: u64, len: u64) -> Result<String, PackError> {
    use std::io::{Seek, SeekFrom};
    let context = || format!("reading {}", path.display());
    let mut file = File::open(path).map_err(io_error(context()))?;
    file.seek(SeekFrom::Start(offset))
        .map_err(io_error(context()))?;
    let mut reader = file.take(len);
    let (mut buffer, mut hasher, mut read) = (vec![0u8; 1 << 20], Sha256::new(), 0u64);
    loop {
        let n = reader.read(&mut buffer).map_err(io_error(context()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
        read += n as u64;
    }
    if read != len {
        return Err(malformed(format!(
            "{} ends {} bytes short of the warehouse's count",
            path.display(),
            len - read
        )));
    }
    Ok(hex(&hasher.finalize()))
}

/// `warehouse.json`, of this format, its batches tiling `0..records` in order.
fn read_manifest(dir: &Path) -> Result<WarehouseManifest, PackError> {
    let manifest: WarehouseManifest = read_json(&dir.join(MANIFEST_FILE))?;
    if manifest.format != WAREHOUSE_FORMAT || manifest.version != WAREHOUSE_FORMAT_VERSION {
        return Err(malformed(format!(
            "{} is {} version {}, and this build reads {WAREHOUSE_FORMAT} version \
             {WAREHOUSE_FORMAT_VERSION}",
            dir.display(),
            manifest.format,
            manifest.version
        )));
    }
    let mut end = 0u64;
    for (at, batch) in manifest.batches.iter().enumerate() {
        if batch.first != end {
            return Err(malformed(format!(
                "{}: batch {at} starts at record {}, and the batches before it end at {end}",
                dir.join(MANIFEST_FILE).display(),
                batch.first
            )));
        }
        end = end.checked_add(batch.records).ok_or_else(|| {
            malformed(format!(
                "{}: batch {at} overflows",
                dir.join(MANIFEST_FILE).display()
            ))
        })?;
    }
    if end != manifest.records {
        return Err(malformed(format!(
            "{}: the batches cover records 0..{end}, and it counts {}",
            dir.join(MANIFEST_FILE).display(),
            manifest.records
        )));
    }
    Ok(manifest)
}

/// Whether `index.bin` counts more than `records`, as between an add's index and its count.
fn index_ahead(dir: &Path, records: u64) -> bool {
    let mut header = [0u8; HEADER];
    File::open(dir.join(INDEX))
        .and_then(|mut file| file.read_exact(&mut header))
        .is_ok()
        && header_count(&header).is_some_and(|count| count > records)
}

/// The count an index header holds, if `index` starts with one.
fn header_count(index: &[u8]) -> Option<u64> {
    (index.len() >= HEADER && index[..8] == INDEX_MAGIC[..])
        .then(|| u64::from_le_bytes(index[8..HEADER].try_into().expect("8")))
}

/// Why `index`'s header is not that of `records` entries, if it is not.
fn header_fault(index: &[u8], records: u64) -> Option<String> {
    let Some(count) = header_count(index) else {
        return Some("no index header".to_string());
    };
    let entries = (index.len() - HEADER) as u64;
    (count != records || entries != records * ENTRY as u64)
        .then(|| format!("it counts {count} entries in {} bytes", index.len()))
}

/// Why `index` is not exactly the index of `keys`' first `records` keys, if it is not.
/// Ascending keys that each match their record point at distinct records: a bijection.
fn index_fault(index: &[u8], keys: &[u8], records: u64) -> Option<String> {
    if let Some(fault) = header_fault(index, records) {
        return Some(fault);
    }
    let entries = index[HEADER..].as_chunks::<ENTRY>().0;
    for (at, entry) in entries.iter().enumerate() {
        if at > 0 && entries[at - 1][..32] >= entry[..32] {
            return Some(format!("entry {at} is not above the entry before it"));
        }
        let record = u64::from_le_bytes(entry[32..].try_into().expect("8"));
        if record >= records {
            return Some(format!(
                "entry {at} points at record {record}, past the count"
            ));
        }
        let key = &keys[record as usize * 32..][..32];
        if *key != entry[..32] {
            return Some(format!(
                "entry {at} is key {} at record {record}, which holds key {}",
                hex(&entry[..32]),
                hex(key)
            ));
        }
    }
    None
}

/// Two sorted runs of index entries, as one.
struct Merge<A: Iterator<Item = [u8; ENTRY]>, B: Iterator<Item = [u8; ENTRY]>> {
    a: std::iter::Peekable<A>,
    b: std::iter::Peekable<B>,
}

impl<A: Iterator<Item = [u8; ENTRY]>, B: Iterator<Item = [u8; ENTRY]>> Merge<A, B> {
    fn new(a: A, b: B) -> Self {
        Self {
            a: a.peekable(),
            b: b.peekable(),
        }
    }
}

impl<A: Iterator<Item = [u8; ENTRY]>, B: Iterator<Item = [u8; ENTRY]>> Iterator for Merge<A, B> {
    type Item = [u8; ENTRY];

    fn next(&mut self) -> Option<Self::Item> {
        match (self.a.peek(), self.b.peek()) {
            (Some(a), Some(b)) if b[..32] < a[..32] => self.b.next(),
            (Some(_), _) => self.a.next(),
            (None, _) => self.b.next(),
        }
    }
}

/// Write `entries`, sorted by key, as `path`, under a temporary name renamed into place.
fn write_index(path: &Path, entries: impl Iterator<Item = [u8; ENTRY]>) -> Result<(), PackError> {
    let partial = write_index_partial(path, entries)?;
    commit_index(&partial, path)
}

fn commit_index(partial: &Path, path: &Path) -> Result<(), PackError> {
    std::fs::rename(partial, path).map_err(io_error(format!("replacing {}", path.display())))
}

/// [`write_index`]'s first half: the entries under `path`'s temporary name, flushed.
fn write_index_partial(
    path: &Path,
    entries: impl Iterator<Item = [u8; ENTRY]>,
) -> Result<PathBuf, PackError> {
    let partial = partial_path(path);
    let context = format!("writing {}", partial.display());
    let mut out = BufWriter::with_capacity(
        4 << 20,
        File::create(&partial).map_err(io_error(context.clone()))?,
    );
    out.write_all(INDEX_MAGIC)
        .and_then(|()| out.write_all(&0u64.to_le_bytes()))
        .map_err(io_error(context.clone()))?;
    let mut count = 0u64;
    for entry in entries {
        out.write_all(&entry).map_err(io_error(context.clone()))?;
        count += 1;
    }
    let mut file = out.into_inner().map_err(|error| PackError::Io {
        context: context.clone(),
        source: error.into_error(),
    })?;
    (|| {
        use std::io::{Seek, SeekFrom};
        file.seek(SeekFrom::Start(8))?;
        file.write_all(&count.to_le_bytes())?;
        file.sync_all()
    })()
    .map_err(io_error(context))?;
    Ok(partial)
}

/// Write `path`, the index of `keys`: every record's key, a record each.
fn rebuild_index(path: &Path, keys: &[u8]) -> Result<(), PackError> {
    let mut entries: Vec<[u8; ENTRY]> = keys
        .as_chunks::<32>()
        .0
        .iter()
        .enumerate()
        .map(|(record, key)| {
            let mut entry = [0u8; ENTRY];
            entry[..32].copy_from_slice(key);
            entry[32..].copy_from_slice(&(record as u64).to_le_bytes());
            entry
        })
        .collect();
    entries.sort_unstable_by(|a, b| a[..32].cmp(&b[..32]));
    if let Some(pair) = entries
        .windows(2)
        .find(|pair| pair[0][..32] == pair[1][..32])
    {
        return Err(malformed(format!(
            "{KEYS} holds key {} twice: no index can be built",
            hex(&pair[0][..32])
        )));
    }
    write_index(path, entries.into_iter())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::testing::TempDir;

    #[test]
    fn a_dropped_lock_is_free_whoever_shares_its_descriptor() {
        let work = TempDir::new("warehouse_lock_shared");
        let held = AppendLock::take(work.path()).unwrap();
        // What a process spawned while the lock is held keeps until it execs.
        let inherited = held.0.try_clone().unwrap();
        drop(held);
        assert!(AppendLock::take(work.path()).is_ok());
        drop(inherited);
    }
}
