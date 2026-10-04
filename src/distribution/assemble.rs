//! A release's segment, assembled by key from a plan and the warehouse.
//!
//! **A base** holds every (book, key) record of the plan's release: books in byte order,
//! each book's lines in order, a key's first record its slot. **A delta** from the release
//! the plan was split against holds the keys that release did not have as slots, their
//! further records as extras, the new records of keys it had as foreign records, and the
//! keys it had and this release does not as tombstones — see
//! [`ledger::classify`](super::ledger).
//!
//! Every slot's vector comes from the warehouse, by the full SHA-256 of its text, and is
//! encoded in the codec epoch: a new one for a base (`i8-sym-vec` by default; `i8-sym-dim`
//! calibrated here on the slots' own vectors, exactly), the previous release's for a delta.
//! The codec is part of the release's identity — [`release_identity`]. Assembly writes the
//! package — `segment.oxv`, `manifest.json`, `payloads.json` — the release manifest
//! `release.json`, and the ledger the next release is planned against.
//!
//! **Bounded memory.** The plan's records, the previous ledger and the warehouse are
//! mapped, not read. What is held, per slot the segment ships: its key and hint (20 bytes),
//! its warehouse record (8), and the classification's entry for a new key (about 40) —
//! some 70 bytes, 450 MB for the library's base, a few percent of that for a delta; plus a
//! window of 4096 vectors, read in warehouse order, and for `i8-sym-dim` 64 MB of
//! histograms per 256 dimensions.
//!
//! **Deterministic:** the same plan, warehouse, previous ledger and `created_at` give the
//! same bytes — quantization rounds half away from zero in `f32`, calibration takes an exact
//! quantile, and nothing depends on a hash map's order.

use crate::distribution::files::{hex, io_error, malformed, sha256, write_atomically, FileDigest};
use crate::distribution::ledger::{
    classify, write_ledger, BaseRecord, Disposition, Ledger, LedgerManifest,
};
use crate::distribution::package::{IndexPackage, PackageKind};
use crate::distribution::plan::{EmbedManifest, Plan, EMBED_MANIFEST_FILE};
use crate::distribution::warehouse::Warehouse;
use crate::errors::PackError;
use crate::semantic::chunk_key::ChunkKey;
use crate::semantic::oxv::codec::{Codec, CodecSpec};
use crate::semantic::oxv::reader::Segment;
use crate::semantic::oxv::writer::{SegmentBuilder, SegmentSpec, VectorSink, WrittenSegment};
use crate::semantic::segment_set::{ReleaseFile, ReleaseManifest};
use crate::semantic::versioning::{EmbeddingWorker, IndexVersion, VectorProvenance};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// The segment's name in an assembled package.
pub const SEGMENT_FILE: &str = "segment.oxv";
/// The release manifest an assembly writes beside its package.
pub const RELEASE_FILE: &str = "release.json";
/// The index schema whose `chunkKey` column a release is resolved through: a manifest's
/// `requires.indexSchemaVersion`, the application's business.
pub const INDEX_SCHEMA_VERSION: u32 = 5;
/// Vectors read from the warehouse at a time, in its order, before they are written in the
/// segment's.
const WINDOW: usize = 4096;

/// Which codec epoch a segment is encoded in.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EpochChoice {
    /// A new epoch, for a base.
    New(CodecSpec),
    /// The previous release's: always a delta's, and a base's that keeps the chain's.
    Previous,
}

/// What to assemble.
pub struct AssembleRequest<'a> {
    pub plan: &'a Plan,
    pub warehouse: &'a Warehouse,
    pub kind: PackageKind,
    /// The release the plan was split against: required for a delta, and for a base that
    /// keeps its epoch.
    pub previous: Option<&'a Ledger>,
    pub epoch: EpochChoice,
    pub out_dir: PathBuf,
    pub created_at: String,
    /// Which build made it, for people: the manifest's `builtBy`.
    pub built_by: Option<serde_json::Value>,
}

/// What an assembly wrote.
#[derive(Debug, Clone)]
pub struct AssembleReport {
    pub out_dir: PathBuf,
    pub manifest: ReleaseManifest,
    /// SHA-256 of `release.json`, as written: without `files`.
    pub manifest_sha256: String,
    /// The ledger of this release, in `out_dir`.
    pub ledger: LedgerManifest,
    /// Components the codec clipped: always 0 but for `i8-sym-dim`.
    pub clipped_components: u64,
}

/// The identity a release built from a plan declares: the plan's, with the codec's name as
/// its `store.vector_precision`.
pub fn release_identity(plan_identity: &IndexVersion, codec: &Codec) -> IndexVersion {
    let mut identity = plan_identity.clone();
    identity.store.vector_precision = codec.name().to_string();
    identity
}

/// Check the warehouse against the plan's independent identity and, when present, its
/// digest-bound embed manifest. Plans without an embed manifest still require a package
/// declared by the family.
pub(crate) fn check_warehouse(plan: &Plan, warehouse: &Warehouse) -> Result<(), PackError> {
    warehouse
        .identity()
        .ensure_model(&plan.manifest.identity.model)?;
    if let Some(digest) = plan.manifest.files.get(EMBED_MANIFEST_FILE) {
        digest.verify(&plan.dir.join(EMBED_MANIFEST_FILE))?;
        let embed = EmbedManifest::read(&plan.dir)?;
        if embed.model != plan.manifest.identity.model {
            return Err(malformed(
                "the embed manifest's model does not match the plan's",
            ));
        }
        if embed.passage_package != warehouse.identity().passage_package {
            return Err(malformed(
                "the warehouse's passage package does not match the plan's embed manifest",
            ));
        }
    }
    Ok(())
}

/// Assemble the release `request` describes. See the module documentation.
pub fn assemble(request: &AssembleRequest<'_>) -> Result<AssembleReport, PackError> {
    let plan = request.plan;
    let warehouse = request.warehouse;
    check_warehouse(plan, warehouse)?;
    warehouse.verify()?;
    let package = &warehouse.identity().passage_package;
    let delta = match request.kind {
        PackageKind::Delta => Some(
            request
                .previous
                .ok_or_else(|| malformed("a delta is assembled against the previous ledger"))?,
        ),
        PackageKind::Base => None,
        PackageKind::Compacted => {
            return Err(malformed(
                "a compacted segment is made on a device, not assembled",
            ))
        }
    };
    let kept_epoch = match (request.epoch, delta) {
        (EpochChoice::Previous, _) | (_, Some(_)) => Some(
            request
                .previous
                .ok_or_else(|| malformed("keeping the epoch needs the previous ledger"))?
                .codec()?,
        ),
        (EpochChoice::New(_), None) => None,
    };
    if let Some(previous) = delta {
        let codec = kept_epoch.as_ref().expect("a delta keeps the epoch");
        let identity = release_identity(&plan.manifest.identity, codec);
        previous.manifest.ensure_matches(
            &identity.identity_digest_hex(),
            Some(&hex(&codec.params_sha256())),
        )?;
        let split_against = plan.manifest.previous.as_ref().map(|p| p.library_version);
        if split_against != Some(previous.library_version()) {
            return Err(malformed(format!(
                "the plan was split against {}, and the ledger is of v{}",
                split_against.map_or("nothing".to_string(), |v| format!("v{v}")),
                previous.library_version()
            )));
        }
        if let Some(chain) = &previous.manifest.passage_package {
            if chain != package {
                return Err(malformed(format!(
                    "the chain's passages are embedded with package {}, and the warehouse \
                     holds package {}'s: a new package starts with a base",
                    chain.checksum, package.checksum
                )));
            }
        }
    }

    // The tables, book by book as the records are classified; every slot's warehouse
    // record.
    let dim = warehouse.dim();
    let placeholder = match &kept_epoch {
        Some(codec) => codec.clone(),
        None => Codec::f32(dim).map_err(malformed)?,
    };
    let mut builder = SegmentBuilder::new(
        SegmentSpec {
            kind: request.kind,
            identity_digest: [0; 32],
            from_library_version: delta.map_or(0, Ledger::library_version),
            to_library_version: plan.manifest.library_version,
            library_release_tag: plan.manifest.library_release_tag.clone(),
        },
        placeholder,
    );
    let mut book = BookRecords::new(u32::MAX);
    let mut slot_records: Vec<u64> = Vec::new();
    let (mut missing, mut first_missing) = (0u64, None);
    let classified = classify(
        &plan.records,
        &plan.books,
        delta,
        |_, record, disposition| {
            if record.book != book.book {
                book.add_to(&mut builder, plan)?;
                book.book = record.book;
            }
            match disposition {
                Disposition::Slot { slot } => {
                    debug_assert_eq!(slot as usize, slot_records.len());
                    book.primary.push((record.key(), record.ordinal));
                    slot_records.push(warehouse.find(&record.sha256).unwrap_or_else(|| {
                        missing += 1;
                        first_missing.get_or_insert(hex(&record.sha256));
                        u64::MAX
                    }));
                }
                Disposition::Extra { slot } => book.extras.push((record.ordinal, slot)),
                Disposition::Foreign => book.foreign.push((record.key(), record.ordinal)),
                Disposition::Held | Disposition::Repeat => {}
            }
            Ok(())
        },
    )?;
    book.add_to(&mut builder, plan)?;
    if missing > 0 {
        return Err(malformed(format!(
            "the warehouse holds no vector for {missing} key(s) the release ships, the first \
             of text SHA-256 {}: embed the plan's embed.jsonl and add the shards",
            first_missing.expect("one is missing")
        )));
    }
    if let Some(previous) = delta {
        builder.set_tombstones(
            (0..previous.key_count())
                .filter(|index| !classified.is_kept(*index))
                .map(|index| ChunkKey(previous.key(index)))
                .collect(),
        );
    }

    let codec = match (kept_epoch, request.epoch) {
        (Some(codec), _) => codec,
        (None, EpochChoice::New(CodecSpec::I8SymDim { clip_q })) => {
            calibrate_i8_sym_dim(warehouse, &slot_records, clip_q)?
        }
        (None, EpochChoice::New(spec)) => spec.build(dim, &[]).map_err(malformed)?,
        (None, EpochChoice::Previous) => unreachable!("kept above"),
    };
    let identity = release_identity(&plan.manifest.identity, &codec);
    identity.validate_complete()?;
    builder
        .set_codec(codec.clone(), identity.identity_digest())
        .map_err(io_error("the codec".to_string()))?;
    let out = &request.out_dir;
    std::fs::create_dir_all(out).map_err(io_error(format!("creating {}", out.display())))?;
    let path = out.join(SEGMENT_FILE);
    let written = builder
        .write(&path)
        .and_then(|sink| write_vectors(sink, warehouse, &slot_records))
        .map_err(io_error(format!("writing {}", path.display())))?;

    let mut manifest = ReleaseManifest::for_segment(
        &written,
        &identity,
        codec.params_sha256(),
        VectorProvenance {
            passage_package: package.clone(),
            worker: workers_of(warehouse, &slot_records),
        },
        request.created_at.clone(),
    );
    manifest.requires = Some(serde_json::json!({
        "indexSchemaVersion": INDEX_SCHEMA_VERSION,
        "lineTextVersion": identity.text.line_text_version,
        "keyVersion": identity.text.key_version,
    }));
    manifest.built_by = request.built_by.clone();
    IndexPackage::write(out, &manifest.package())?;
    Segment::open(&path)?;
    let json = manifest.to_json();
    write_atomically(&out.join(RELEASE_FILE), json.as_bytes())?;

    let (base, deltas_since_base) = match delta {
        Some(previous) => (
            previous.manifest.base,
            previous.manifest.deltas_since_base + written.size,
        ),
        None => (
            BaseRecord {
                library_version: plan.manifest.library_version,
                size: written.size,
            },
            0,
        ),
    };
    let ledger = write_ledger(
        out,
        plan.manifest.library_version,
        &plan.manifest.library_release_tag,
        &identity.identity_digest_hex(),
        &codec,
        base,
        deltas_since_base,
        &plan.records,
        &plan.books,
        delta,
        &classified,
        Some(package),
    )?;
    Ok(AssembleReport {
        out_dir: out.clone(),
        manifest_sha256: hex(&sha256(json.as_bytes())),
        manifest,
        ledger,
        clipped_components: written.clipped_components,
    })
}

/// One book's records, as a segment stores them.
struct BookRecords {
    book: u32,
    primary: Vec<(ChunkKey, u32)>,
    extras: Vec<(u32, u32)>,
    foreign: Vec<(ChunkKey, u32)>,
}

impl BookRecords {
    fn new(book: u32) -> Self {
        Self {
            book,
            primary: Vec::new(),
            extras: Vec::new(),
            foreign: Vec::new(),
        }
    }

    /// Add the book to `builder`, if it is one, and empty it.
    fn add_to(&mut self, builder: &mut SegmentBuilder, plan: &Plan) -> Result<(), PackError> {
        if self.book != u32::MAX {
            let name = plan.books.name(self.book);
            builder
                .add_book(name, &self.primary, &self.extras, &self.foreign)
                .map_err(io_error(format!("adding {name}")))?;
        }
        self.primary.clear();
        self.extras.clear();
        self.foreign.clear();
        Ok(())
    }
}

/// Every slot's vector, in slot order, read a window at a time in warehouse order.
fn write_vectors(
    mut sink: VectorSink,
    warehouse: &Warehouse,
    records: &[u64],
) -> std::io::Result<WrittenSegment> {
    let dim = warehouse.dim();
    let mut window = vec![0f32; WINDOW * dim];
    let mut order: Vec<(u64, usize)> = Vec::with_capacity(WINDOW);
    for chunk in records.chunks(WINDOW) {
        order.clear();
        order.extend(chunk.iter().enumerate().map(|(at, record)| (*record, at)));
        order.sort_unstable();
        for (record, at) in &order {
            warehouse.vector(*record, &mut window[at * dim..][..dim]);
        }
        for vector in window[..chunk.len() * dim].chunks_exact(dim) {
            sink.push_f32(vector)?;
        }
    }
    sink.finish()
}

/// The workers of the batches the slots' vectors came from, as provenance: their names and
/// devices, each set in order, so the same warehouse gives the same digest.
fn workers_of(warehouse: &Warehouse, records: &[u64]) -> EmbeddingWorker {
    let batches = &warehouse.manifest().batches;
    let mut used = vec![false; batches.len()];
    for record in records {
        let at = batches.partition_point(|batch| batch.first + batch.records <= *record);
        if let Some(flag) = used.get_mut(at) {
            *flag = true;
        }
    }
    let (mut names, mut devices) = (BTreeSet::new(), BTreeSet::new());
    for (batch, _) in batches.iter().zip(&used).filter(|(_, used)| **used) {
        let worker = &batch.worker;
        names.insert(format!("{} {}", worker.name, worker.version));
        devices.insert(format!("{} ({} {})", worker.device, worker.ep, worker.mode));
    }
    let join = |set: BTreeSet<String>| set.into_iter().collect::<Vec<_>>().join(" + ");
    EmbeddingWorker {
        backend: join(names),
        device: join(devices),
    }
}

/// `i8-sym-dim`'s scales on the slots' own vectors, exactly as
/// [`Codec::calibrate_i8_sym_dim`] takes them — each dimension's ⌊clip_q · (n − 1)⌋-th
/// smallest absolute value — in two passes over the warehouse and 256 KiB of histogram per
/// dimension, not every value in memory. A non-negative `f32` orders as its bits: the
/// first pass counts the upper 16 bits of each `|x|` and finds the bucket holding the rank,
/// the second counts the lower 16 bits inside that bucket.
pub(crate) fn calibrate_i8_sym_dim(
    warehouse: &Warehouse,
    records: &[u64],
    clip_q: f32,
) -> Result<Codec, PackError> {
    let dim = warehouse.dim();
    let n = records.len();
    if n == 0 {
        return Err(malformed("there are no vectors to calibrate on"));
    }
    if !(clip_q.is_finite() && clip_q > 0.0 && clip_q <= 1.0) {
        return Err(malformed(format!(
            "clip_q is {clip_q}, and it must be in (0, 1]"
        )));
    }
    let rank = ((f64::from(clip_q) * (n - 1) as f64).floor() as usize).min(n - 1) as u64;
    let mut vector = vec![0f32; dim];
    let mut counts = vec![0u32; dim << 16];
    let mut count = |bucket_of: &dyn Fn(usize, u32) -> Option<u32>, counts: &mut [u32]| {
        counts.fill(0);
        for record in records {
            warehouse.vector(*record, &mut vector);
            for (d, value) in vector.iter().enumerate() {
                if let Some(bucket) = bucket_of(d, value.abs().to_bits()) {
                    counts[(d << 16) | bucket as usize] += 1;
                }
            }
        }
    };
    // The bucket of dimension `d` holding rank `rank`, and the rank inside it.
    let find = |counts: &[u32], d: usize, rank: u64| -> (u32, u64) {
        let mut below = 0u64;
        for (bucket, count) in counts[d << 16..(d + 1) << 16].iter().enumerate() {
            if below + u64::from(*count) > rank {
                return (bucket as u32, rank - below);
            }
            below += u64::from(*count);
        }
        unreachable!("the rank is below the count")
    };
    count(&|_, bits| Some(bits >> 16), &mut counts);
    let high: Vec<(u32, u64)> = (0..dim).map(|d| find(&counts, d, rank)).collect();
    count(
        &|d, bits| (bits >> 16 == high[d].0).then_some(bits & 0xFFFF),
        &mut counts,
    );
    let scales = (0..dim)
        .map(|d| {
            let scale = f32::from_bits(high[d].0 << 16 | find(&counts, d, high[d].1).0);
            if scale > 0.0 && scale.is_finite() {
                scale
            } else {
                1.0
            }
        })
        .collect();
    Codec::i8_sym_dim(scales, clip_q).map_err(malformed)
}

/// The release manifest at `release` with `files` — what a client downloads, in order:
/// the compressed segment, or the parts it is split into — listed in the shape of the
/// updater's patch entries, each with the uncompressed segment's digest. Written to `out`;
/// returns it and the SHA-256 of what was written, the value published outside it.
pub fn release_with_files(
    release: &Path,
    files: &[PathBuf],
    compression: &str,
    out: &Path,
) -> Result<(ReleaseManifest, String), PackError> {
    let mut manifest: ReleaseManifest = crate::distribution::files::read_json(release)?;
    manifest.files = files
        .iter()
        .map(|path| {
            let digest = FileDigest::of(path)?;
            let file = path
                .file_name()
                .map(|name| name.to_string_lossy().to_string())
                .ok_or_else(|| malformed(format!("{} names no file", path.display())))?;
            Ok(ReleaseFile {
                file,
                compression: compression.to_string(),
                sha256: digest.sha256,
                size: digest.size,
                uncompressed_sha256: manifest.segment.sha256.clone(),
                uncompressed_size: manifest.segment.size,
            })
        })
        .collect::<Result<_, PackError>>()?;
    let json = manifest.to_json();
    write_atomically(out, json.as_bytes())?;
    Ok((manifest, hex(&sha256(json.as_bytes()))))
}

/// The name a release's segment is published under, before `.oxv.zst`:
/// `otzaria-vectors-<id8>-v<from>-v<to>` for a delta, `otzaria-vectors-<id8>-v<to>-base`
/// for a base.
pub fn asset_stem(manifest: &ReleaseManifest) -> String {
    let id8 = &manifest.identity_digest[..8];
    match manifest.kind {
        PackageKind::Delta => format!(
            "otzaria-vectors-{id8}-v{}-v{}",
            manifest.from_library_version, manifest.to_library_version
        ),
        _ => format!(
            "otzaria-vectors-{id8}-v{}-base",
            manifest.to_library_version
        ),
    }
}
