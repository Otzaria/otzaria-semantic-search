//! Embedding a plan on machines that never see the corpus — the external embedding
//! interface.
//!
//! A worker is handed a plan's `embed.jsonl` and `embed-manifest.json` (see
//! [`plan`](super::plan)) and a window of it, and writes one directory:
//!
//! ```text
//! vectors.f32          records × dim little-endian f32s, in plan order: finite, unit norm
//! keys.bin             records × 32 bytes: each record's embedding_text_sha256, raw
//! shard-manifest.json  ShardManifest: the window, both files' SHA-256, the model and the
//!                      passage package it was embedded with, the worker, its parity
//! ```
//!
//! **What a worker checks before it embeds a record:** that the text hashes to the digest
//! beside it ([`PackError::PlanTextChanged`]), and — for a worker that runs the shipped
//! graph — that the package it loaded is the plan's passage package, of the plan's family,
//! field by field ([`check_runtime`]).
//!
//! **Where a shard boundary falls changes no vector.** The ONNX backend runs one text per
//! session run, so a vector depends on its text alone; a window may start anywhere and hold
//! any number of records.
//!
//! **What acceptance checks** ([`verify_shards`]): the shards tile the plan exactly — no
//! hole, no overlap; each names the plan's digest, model and passage package; each file has
//! the length its count implies and the digest its manifest declares; every vector is finite
//! and of unit norm; every key is the plan's at its position; and a worker that is not ONNX
//! Runtime on a CPU carries a parity certificate. Without the plan — a whole-library import —
//! everything but the plan's own keys is checked.

use crate::distribution::files::{hex, io_error, malformed, partial_path, read_json, write_json};
use crate::distribution::plan::{read_embed_plan, EmbedManifest};
use crate::errors::PackError;
use crate::semantic::embedding::EmbeddingRuntime;
use crate::semantic::versioning::{ModelIdentity, ModelPackage};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

pub const VECTORS_FILE: &str = "vectors.f32";
pub const KEYS_FILE: &str = "keys.bin";
pub const SHARD_MANIFEST_FILE: &str = "shard-manifest.json";
/// `format` of a shard manifest.
pub const SHARD_FORMAT: &str = "otzaria-embed-shard";
pub const SHARD_FORMAT_VERSION: u32 = 2;

/// The lowest cosine to ONNX Runtime on a CPU a parity certificate may report.
pub const PARITY_MIN_COSINE: f64 = 0.999;
/// The fewest texts a parity certificate may be measured on.
pub const PARITY_MIN_SAMPLES: u64 = 1_000;
/// How far from 1 a stored vector's norm may be.
pub const UNIT_NORM_TOLERANCE: f64 = 1e-3;

/// `worker.mode` of ONNX Runtime running the shipped graph.
pub const MODE_ONNXRUNTIME: &str = "onnxruntime";
/// `worker.mode` of the deterministic stand-in, for tests.
pub const MODE_MOCK: &str = "mock";

/// What produced a shard's vectors. Provenance: recorded, compared by nothing but the
/// parity rule.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerInfo {
    /// The program, e.g. `otzaria-semantic-search`.
    pub name: String,
    pub version: String,
    /// The hardware, e.g. `cpu`, or a GPU's name.
    pub device: String,
    /// The execution provider: `cpu`, `cuda`, `dml`, `rocm`, …
    pub ep: String,
    /// What ran the graph: [`MODE_ONNXRUNTIME`] for ONNX Runtime with the shipped graph;
    /// anything else (e.g. `torch`) names a re-implementation.
    pub mode: String,
}

impl WorkerInfo {
    /// Whether this worker is the reference itself — ONNX Runtime on a CPU — and so needs
    /// no parity certificate.
    pub fn is_reference(&self) -> bool {
        self.ep == "cpu" && self.mode == MODE_ONNXRUNTIME
    }
}

/// A worker's measured agreement with ONNX Runtime on a CPU.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParityCertificate {
    /// What the worker was compared against, e.g. `onnxruntime 1.28.0 cpu fp32`.
    pub reference: String,
    /// Texts compared: the int8 goldens and plan texts sampled across the plan.
    pub samples: u64,
    /// The lowest cosine between the worker's vector and the reference's.
    pub min_cosine: f64,
    pub mean_cosine: f64,
    /// SHA-256 of the full certificate document, kept beside the build's logs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub document_sha256: Option<String>,
}

/// `shard-manifest.json`, version 2.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShardManifest {
    pub format: String,
    pub version: u32,
    /// The embed plan's `plan_sha256`.
    pub plan_sha256: String,
    /// The window: records from `skip`, at most `take`.
    pub skip: u64,
    pub take: u64,
    /// Records written: `take`, or what remained of the plan after `skip`.
    pub records: u64,
    pub dim: u32,
    pub vectors_sha256: String,
    pub keys_sha256: String,
    /// The embed manifest's `model`, verbatim.
    pub model: ModelIdentity,
    /// The package the vectors were embedded with: the plan's passage package.
    pub passage_package: ModelPackage,
    pub worker: WorkerInfo,
    /// Required unless the worker [is the reference](WorkerInfo::is_reference).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parity: Option<ParityCertificate>,
}

/// What acceptance permits beyond the rules.
#[derive(Debug, Clone, Copy, Default)]
pub struct ShardPolicy {
    /// Accept the deterministic stand-in's vectors, which mean nothing. Tests only.
    pub allow_non_semantic: bool,
}

impl ShardManifest {
    pub fn read(dir: &Path) -> Result<Self, PackError> {
        let manifest: Self = read_json(&dir.join(SHARD_MANIFEST_FILE))?;
        if manifest.format != SHARD_FORMAT || manifest.version != SHARD_FORMAT_VERSION {
            return Err(malformed(format!(
                "{} is {} version {}, and this build reads {SHARD_FORMAT} version \
                 {SHARD_FORMAT_VERSION}",
                dir.display(),
                manifest.format,
                manifest.version
            )));
        }
        Ok(manifest)
    }

    /// The rules a shard is held to without its plan: its family and package agree, its
    /// width is the family's, and its worker is the reference or certified.
    pub fn check(&self, policy: &ShardPolicy) -> Result<(), PackError> {
        let refuse = |reason: String| {
            malformed(format!(
                "the shard of plan {} at {}: {reason}",
                self.plan_sha256, self.skip
            ))
        };
        if self.dim != self.model.embedding_dim {
            return Err(refuse(format!(
                "its vectors are {} wide and the family's are {}",
                self.dim, self.model.embedding_dim
            )));
        }
        if !self.model.query_packages.contains(&self.passage_package) {
            return Err(refuse(format!(
                "the package {} is not one of the family's",
                self.passage_package.checksum
            )));
        }
        if self.records > self.take {
            return Err(refuse(format!(
                "it holds {} records of a {}-record window",
                self.records, self.take
            )));
        }
        if self.worker.mode == MODE_MOCK {
            return if policy.allow_non_semantic {
                Ok(())
            } else {
                Err(refuse(
                    "its vectors come from the deterministic stand-in, which means nothing"
                        .to_string(),
                ))
            };
        }
        if self.worker.is_reference() {
            return Ok(());
        }
        match &self.parity {
            None => Err(refuse(format!(
                "worker {} ({} on {}, {}) is not ONNX Runtime on a CPU and carries no parity \
                 certificate",
                self.worker.name, self.worker.mode, self.worker.ep, self.worker.device
            ))),
            Some(parity) if parity.samples < PARITY_MIN_SAMPLES => Err(refuse(format!(
                "its parity certificate covers {} texts, fewer than {PARITY_MIN_SAMPLES}",
                parity.samples
            ))),
            Some(parity)
                if !(parity.min_cosine >= PARITY_MIN_COSINE && parity.min_cosine <= 1.0 + 1e-9) =>
            {
                Err(refuse(format!(
                    "its parity certificate's lowest cosine is {}, below {PARITY_MIN_COSINE}",
                    parity.min_cosine
                )))
            }
            Some(_) => Ok(()),
        }
    }
}

/// Hold a loaded runtime to the family and package a plan was made for: the package
/// checksum is the plan's passage package, and the tokenizer, width, pooling and token cap
/// are the family's.
pub fn check_runtime(
    runtime: &EmbeddingRuntime,
    model: &ModelIdentity,
    package: &ModelPackage,
) -> Result<(), PackError> {
    let checksum = runtime.model_checksum().unwrap_or_default();
    if checksum != package.checksum {
        return Err(PackError::ModelDisagreesWithFile {
            field: "passage_package",
            declared: format!("{} {}", package.quantization, package.checksum),
            loaded: checksum.to_string(),
        });
    }
    for (field, declared, loaded) in [
        (
            "tokenizer_checksum",
            model.tokenizer_checksum.clone(),
            runtime.tokenizer_checksum().unwrap_or_default().to_string(),
        ),
        (
            "embedding_dim",
            model.embedding_dim.to_string(),
            runtime.dim().to_string(),
        ),
        (
            "pooling",
            model.pooling.clone(),
            runtime.pooling().to_string(),
        ),
        (
            "max_tokens",
            model.max_tokens.to_string(),
            runtime.max_tokens().to_string(),
        ),
    ] {
        if declared != loaded {
            return Err(PackError::ModelDisagreesWithFile {
                field,
                declared,
                loaded,
            });
        }
    }
    Ok(())
}

/// Embed records `skip..skip+take` of the plan in `plan_dir` into a shard at `out`.
///
/// `out` may hold leftovers of a session that died — retrying a window is normal — but not
/// a finished shard: all three files present is refused rather than overwritten. The two
/// data files are written as `.partial`, flushed, renamed, and the manifest written last.
#[allow(clippy::too_many_arguments)]
pub fn embed_shard(
    plan_dir: &Path,
    plan: &EmbedManifest,
    skip: u64,
    take: u64,
    runtime: &EmbeddingRuntime,
    batch_size: usize,
    worker: WorkerInfo,
    out: &Path,
) -> Result<ShardManifest, PackError> {
    check_runtime(runtime, &plan.model, &plan.passage_package)?;
    if !runtime.backend_is_semantic() && worker.mode != MODE_MOCK {
        return Err(PackError::NonSemanticBackend {
            backend: runtime.backend_id().unwrap_or("none").to_string(),
        });
    }
    std::fs::create_dir_all(out).map_err(io_error(format!("creating {}", out.display())))?;
    let finished = [VECTORS_FILE, KEYS_FILE, SHARD_MANIFEST_FILE]
        .iter()
        .all(|name| out.join(name).symlink_metadata().is_ok());
    if finished {
        return Err(PackError::UnusableOutput {
            path: out.display().to_string(),
            reason: "it holds a finished shard; embed into another directory, or remove it to \
                     re-run this window"
                .to_string(),
        });
    }

    let dim = plan.model.embedding_dim as usize;
    let mut vectors = HashingWriter::create(&out.join(VECTORS_FILE))?;
    let mut keys = HashingWriter::create(&out.join(KEYS_FILE))?;
    let mut batch: Vec<(u64, [u8; 32], String)> = Vec::with_capacity(batch_size.max(1));
    let mut records = 0u64;
    let flush = |batch: &mut Vec<(u64, [u8; 32], String)>,
                 vectors: &mut HashingWriter,
                 keys: &mut HashingWriter|
     -> Result<(), PackError> {
        if batch.is_empty() {
            return Ok(());
        }
        let texts: Vec<&str> = batch.iter().map(|(_, _, text)| text.as_str()).collect();
        let embedded = runtime.embed_batch(&texts)?;
        if embedded.len() != batch.len() || embedded.iter().any(|vector| vector.len() != dim) {
            return Err(malformed(format!(
                "{} vector(s) came back for {} text(s), and each must be {dim} wide",
                embedded.len(),
                batch.len()
            )));
        }
        for ((_, digest, _), vector) in batch.drain(..).zip(embedded) {
            let bytes: Vec<u8> = vector
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            vectors.write(&bytes)?;
            keys.write(&digest)?;
        }
        Ok(())
    };
    for record in read_embed_plan(plan_dir, skip, take)? {
        let (position, record) = record?;
        let digest = record.check(position)?;
        batch.push((position, digest, record.embedding_text));
        records += 1;
        if batch.len() >= batch_size.max(1) {
            flush(&mut batch, &mut vectors, &mut keys)?;
        }
    }
    flush(&mut batch, &mut vectors, &mut keys)?;
    let owed = take.min(plan.records.saturating_sub(skip));
    if records != owed {
        return Err(malformed(format!(
            "the plan held {records} record(s) from {skip}, and its manifest promises {owed}"
        )));
    }

    let manifest = ShardManifest {
        format: SHARD_FORMAT.to_string(),
        version: SHARD_FORMAT_VERSION,
        plan_sha256: plan.plan_sha256.clone(),
        skip,
        take,
        records,
        dim: dim as u32,
        vectors_sha256: vectors.finish()?,
        keys_sha256: keys.finish()?,
        model: plan.model.clone(),
        passage_package: plan.passage_package.clone(),
        worker,
        parity: None,
    };
    write_json(&out.join(SHARD_MANIFEST_FILE), &manifest)?;
    Ok(manifest)
}

/// A file written under `.partial`, hashed as it goes, renamed when finished.
struct HashingWriter {
    out: BufWriter<File>,
    path: PathBuf,
    hasher: Sha256,
}

impl HashingWriter {
    fn create(path: &Path) -> Result<Self, PackError> {
        let partial = partial_path(path);
        Ok(Self {
            out: BufWriter::with_capacity(
                4 << 20,
                File::create(&partial)
                    .map_err(io_error(format!("creating {}", partial.display())))?,
            ),
            path: path.to_path_buf(),
            hasher: Sha256::new(),
        })
    }

    fn write(&mut self, bytes: &[u8]) -> Result<(), PackError> {
        self.hasher.update(bytes);
        self.out
            .write_all(bytes)
            .map_err(io_error(format!("writing {}", self.path.display())))
    }

    fn finish(self) -> Result<String, PackError> {
        let context = format!("finishing {}", self.path.display());
        let file = self.out.into_inner().map_err(|error| PackError::Io {
            context: context.clone(),
            source: error.into_error(),
        })?;
        (|| {
            file.sync_all()?;
            drop(file);
            std::fs::rename(partial_path(&self.path), &self.path)
        })()
        .map_err(io_error(context))?;
        Ok(format!("{:x}", self.hasher.finalize()))
    }
}

/// A shard that passed [`verify_shards`].
#[derive(Debug, Clone)]
pub struct CheckedShard {
    pub dir: PathBuf,
    pub manifest: ShardManifest,
}

/// Check a set of shards, with the plan they were cut from or — for a whole-library
/// import — without it, and return them in plan order. See the module documentation.
pub fn verify_shards(
    plan_dir: Option<&Path>,
    dirs: &[PathBuf],
    policy: &ShardPolicy,
) -> Result<Vec<CheckedShard>, PackError> {
    if dirs.is_empty() {
        return Err(PackError::NoVectors);
    }
    let plan = plan_dir.map(EmbedManifest::read).transpose()?;
    let manifests = dirs
        .iter()
        .map(|dir| ShardManifest::read(dir))
        .collect::<Result<Vec<_>, _>>()?;
    let mut shards = Vec::with_capacity(dirs.len());
    for (dir, manifest) in dirs.iter().zip(manifests.iter().cloned()) {
        manifest.check(policy)?;
        let named = dir.display();
        let first = &manifests[0];
        if manifest.plan_sha256 != first.plan_sha256
            || manifest.model != first.model
            || manifest.passage_package != first.passage_package
        {
            return Err(malformed(format!(
                "{named} was embedded from another plan, family or package than the other \
                 shards"
            )));
        }
        if let Some(plan) = &plan {
            if manifest.plan_sha256 != plan.plan_sha256 {
                return Err(malformed(format!(
                    "{named} was embedded from plan {} and this plan is {}",
                    manifest.plan_sha256, plan.plan_sha256
                )));
            }
            if manifest.model != plan.model || manifest.passage_package != plan.passage_package {
                return Err(malformed(format!(
                    "{named} was embedded for another family or package than the plan's"
                )));
            }
            let owed = manifest
                .take
                .min(plan.records.saturating_sub(manifest.skip));
            if manifest.records != owed {
                return Err(malformed(format!(
                    "{named} was given {owed} record(s) from {} and wrote {}",
                    manifest.skip, manifest.records
                )));
            }
        }
        check_files(dir, &manifest)?;
        shards.push(CheckedShard {
            dir: dir.clone(),
            manifest,
        });
    }

    shards.sort_by_key(|shard| (shard.manifest.skip, shard.manifest.records));
    let mut covered = 0u64;
    for shard in &shards {
        if shard.manifest.skip != covered {
            return Err(malformed(if shard.manifest.skip > covered {
                format!(
                    "records {covered}..{} are covered by no shard",
                    shard.manifest.skip
                )
            } else {
                format!(
                    "{} starts at {} and records up to {covered} are covered already",
                    shard.dir.display(),
                    shard.manifest.skip
                )
            }));
        }
        covered += shard.manifest.records;
    }
    if let Some(plan) = &plan {
        if covered != plan.records {
            return Err(malformed(format!(
                "the shards cover {covered} record(s) and the plan holds {}",
                plan.records
            )));
        }
        check_keys_against_plan(plan_dir.expect("a plan"), &shards)?;
    }
    Ok(shards)
}

/// Both files: their lengths, their digests, and every vector finite and of unit norm.
fn check_files(dir: &Path, manifest: &ShardManifest) -> Result<(), PackError> {
    let dim = manifest.dim as usize;
    let keys = digest_of(&dir.join(KEYS_FILE), manifest.records, 32, |_| Ok(()))?;
    if keys != manifest.keys_sha256 {
        return Err(malformed(format!(
            "{}/{KEYS_FILE} hashes to {keys} and its manifest declares {}",
            dir.display(),
            manifest.keys_sha256
        )));
    }
    let mut index = 0u64;
    let vectors = digest_of(
        &dir.join(VECTORS_FILE),
        manifest.records,
        dim * 4,
        |chunk| {
            for vector in chunk.chunks_exact(dim * 4) {
                check_vector(vector, index, dir)?;
                index += 1;
            }
            Ok(())
        },
    )?;
    if vectors != manifest.vectors_sha256 {
        return Err(malformed(format!(
            "{}/{VECTORS_FILE} hashes to {vectors} and its manifest declares {}",
            dir.display(),
            manifest.vectors_sha256
        )));
    }
    Ok(())
}

/// A stored vector: little-endian f32s, finite, of norm 1 within the tolerance.
pub(crate) fn check_vector(bytes: &[u8], index: u64, dir: &Path) -> Result<(), PackError> {
    let mut norm = 0f64;
    for value in bytes.as_chunks::<4>().0 {
        let value = f32::from_le_bytes(*value);
        if !value.is_finite() {
            return Err(malformed(format!(
                "{}: vector {index} holds a value that is not finite",
                dir.display()
            )));
        }
        norm += f64::from(value) * f64::from(value);
    }
    if (norm.sqrt() - 1.0).abs() > UNIT_NORM_TOLERANCE {
        return Err(malformed(format!(
            "{}: vector {index} has norm {}, and a stored vector is a unit vector",
            dir.display(),
            norm.sqrt()
        )));
    }
    Ok(())
}

/// SHA-256 of `path`, which must hold exactly `records` records of `width` bytes, handing
/// `inspect` whole records as it reads them.
pub(crate) fn digest_of(
    path: &Path,
    records: u64,
    width: usize,
    mut inspect: impl FnMut(&[u8]) -> Result<(), PackError>,
) -> Result<String, PackError> {
    let file = File::open(path).map_err(io_error(format!("reading {}", path.display())))?;
    let length = records * width as u64;
    let actual = file
        .metadata()
        .map_err(io_error(format!("inspecting {}", path.display())))?
        .len();
    if actual != length {
        return Err(malformed(format!(
            "{} is {actual} bytes and its manifest's count makes it {length}",
            path.display()
        )));
    }
    let mut reader = BufReader::with_capacity(1 << 20, file);
    let mut hasher = Sha256::new();
    let per_read = ((4 << 20) / width).max(1) * width;
    let mut buffer = vec![0u8; per_read];
    let mut remaining = length;
    while remaining > 0 {
        let take = (per_read as u64).min(remaining) as usize;
        reader
            .read_exact(&mut buffer[..take])
            .map_err(io_error(format!("reading {}", path.display())))?;
        hasher.update(&buffer[..take]);
        inspect(&buffer[..take])?;
        remaining -= take as u64;
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Every shard's keys, in plan order, against the plan's — one pass over the plan.
fn check_keys_against_plan(plan_dir: &Path, shards: &[CheckedShard]) -> Result<(), PackError> {
    let mut plan = read_embed_plan(plan_dir, 0, u64::MAX)?;
    for shard in shards {
        let path = shard.dir.join(KEYS_FILE);
        let mut keys = BufReader::with_capacity(
            1 << 20,
            File::open(&path).map_err(io_error(format!("reading {}", path.display())))?,
        );
        for _ in 0..shard.manifest.records {
            let (position, record) = plan
                .next()
                .ok_or_else(|| malformed("the plan ended before the shards did"))??;
            let mut key = [0u8; 32];
            keys.read_exact(&mut key)
                .map_err(io_error(format!("reading {}", path.display())))?;
            if hex(&key) != record.embedding_text_sha256 {
                return Err(malformed(format!(
                    "{}: record {position} is keyed {} and the plan's text there is {}",
                    shard.dir.display(),
                    hex(&key),
                    record.embedding_text_sha256
                )));
            }
        }
    }
    Ok(())
}
