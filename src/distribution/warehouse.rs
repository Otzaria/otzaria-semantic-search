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
//! append truncates whatever a crash left past the count and rebuilds an index that is not
//! the count's; opening to read ignores both.

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
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

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

/// A warehouse, open.
pub struct Warehouse {
    dir: PathBuf,
    manifest: WarehouseManifest,
    vectors: Option<Mmap>,
    index: Option<Mmap>,
    _lock: Option<File>,
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

    /// Open to read: what `warehouse.json` counts, whatever a crash left past it.
    pub fn open(dir: &Path) -> Result<Self, PackError> {
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
        let mut warehouse = Self {
            dir: dir.to_path_buf(),
            manifest,
            vectors: None,
            index: None,
            _lock: None,
        };
        warehouse.map()?;
        Ok(warehouse)
    }

    /// Open to add a batch: hold the lock, and repair what a crash left.
    pub fn open_for_append(dir: &Path) -> Result<Self, PackError> {
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(dir.join(LOCK))
            .map_err(io_error(format!("opening {}", dir.join(LOCK).display())))?;
        match lock.try_lock() {
            Ok(()) => {}
            Err(std::fs::TryLockError::WouldBlock) => {
                return Err(malformed(format!(
                    "{} is being added to by another process",
                    dir.display()
                )))
            }
            Err(std::fs::TryLockError::Error(source)) => {
                return Err(PackError::Io {
                    context: format!("locking {}", dir.display()),
                    source,
                })
            }
        }
        let manifest: WarehouseManifest = read_json(&dir.join(MANIFEST_FILE))?;
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
        if index_count(&dir.join(INDEX))? != Some(manifest.records) {
            rebuild_index(dir, manifest.records)?;
        }
        let mut warehouse = Self::open(dir)?;
        warehouse._lock = Some(lock);
        Ok(warehouse)
    }

    fn map(&mut self) -> Result<(), PackError> {
        let records = self.manifest.records;
        let width = self.width() as u64;
        let vectors = map_file(&self.dir.join(VECTORS))?;
        if (vectors.len() as u64) < records * width {
            return Err(malformed(format!(
                "{} is short of the {records} records the warehouse counts",
                self.dir.join(VECTORS).display()
            )));
        }
        let index = map_file(&self.dir.join(INDEX))?;
        if index.len() < HEADER
            || &index[..8] != INDEX_MAGIC
            || u64::from_le_bytes(index[8..16].try_into().expect("8")) != records
            || (index.len() - HEADER) as u64 != records * ENTRY as u64
        {
            return Err(malformed(format!(
                "{} is not the index of the warehouse's {records} records; adding to the \
                 warehouse rebuilds it",
                self.dir.join(INDEX).display()
            )));
        }
        self.vectors = Some(vectors);
        self.index = Some(index);
        Ok(())
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

    /// The record holding the vector of the text whose SHA-256 is `sha256`.
    pub fn find(&self, sha256: &[u8; 32]) -> Option<u64> {
        let index = self.index.as_ref()?;
        let entries = index[HEADER..].as_chunks::<ENTRY>().0;
        entries
            .binary_search_by(|entry| entry[..32].cmp(sha256))
            .ok()
            .map(|at| u64::from_le_bytes(entries[at][32..].try_into().expect("8")))
    }

    /// The record holding a vector of a text whose SHA-256 starts with `key` — a segment's
    /// [`ChunkKey`](crate::semantic::chunk_key::ChunkKey): the index is in digest order,
    /// so its prefixes are in order too.
    pub fn find_key(&self, key: &[u8; 16]) -> Option<u64> {
        let index = self.index.as_ref()?;
        let entries = index[HEADER..].as_chunks::<ENTRY>().0;
        let at = entries.partition_point(|entry| entry[..16] < key[..]);
        entries
            .get(at)
            .filter(|entry| entry[..16] == key[..])
            .map(|entry| u64::from_le_bytes(entry[32..].try_into().expect("8")))
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
        let result = self.append(&shards, width);
        let (entries, vectors_sha256, keys_sha256, records) = match result {
            Ok(appended) => appended,
            Err(error) => {
                self.truncate(first)?;
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
                if self.find(&key).is_some() {
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

fn index_count(path: &Path) -> Result<Option<u64>, PackError> {
    let Ok(mut file) = File::open(path) else {
        return Ok(None);
    };
    let mut header = [0u8; HEADER];
    if file.read_exact(&mut header).is_err() || &header[..8] != INDEX_MAGIC {
        return Ok(None);
    }
    let count = u64::from_le_bytes(header[8..].try_into().expect("8"));
    let length = file
        .metadata()
        .map_err(io_error(format!("inspecting {}", path.display())))?
        .len();
    Ok((length == HEADER as u64 + count * ENTRY as u64).then_some(count))
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

/// Rebuild the index of the first `records` keys from `keys.bin`.
fn rebuild_index(dir: &Path, records: u64) -> Result<(), PackError> {
    let path = dir.join(KEYS);
    let mut reader = BufReader::with_capacity(
        4 << 20,
        File::open(&path).map_err(io_error(format!("reading {}", path.display())))?,
    );
    let mut entries = Vec::with_capacity(records as usize);
    let mut key = [0u8; 32];
    for record in 0..records {
        reader
            .read_exact(&mut key)
            .map_err(io_error(format!("reading {}", path.display())))?;
        let mut entry = [0u8; ENTRY];
        entry[..32].copy_from_slice(&key);
        entry[32..].copy_from_slice(&record.to_le_bytes());
        entries.push(entry);
    }
    entries.sort_unstable_by(|a, b| a[..32].cmp(&b[..32]));
    write_index(&dir.join(INDEX), entries.into_iter())
}
