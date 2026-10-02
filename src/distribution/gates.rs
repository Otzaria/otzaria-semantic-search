//! The gates a release passes before it is published (spec §5.6).
//!
//! [`verify_release`] — `assemble --verify` — checks the ones the build's own inputs
//! decide: **G1** identity and codec, **G5** fidelity, **G7** size, **G8** determinism,
//! **G9** the package — and reports **G10**, growth, as a number. The others need the
//! release index and belong to the plugin's validator, which takes from here what it
//! needs:
//!
//! * [`simulate_device`] — the set a device holds after installing the previous published
//!   state and the new release, opened with the runtime reader (G3, G4, G6);
//! * [`coverage`] — every record of a plan reachable in that set (G3);
//! * [`book_records`] — a book's records in the set, with their hints, to resolve (G4);
//! * [`ExactReference`] and [`recall`] — the exact `f32` scan of the same keys a retrieval
//!   check compares the set's scan with (G6).

use crate::cancellation::CancellationToken;
use crate::distribution::assemble::{
    assemble, AssembleRequest, EpochChoice, RELEASE_FILE, SEGMENT_FILE,
};
use crate::distribution::files::{hex, malformed, read_json, sha256, FileDigest};
use crate::distribution::ledger::{manifest_file_name, Ledger};
use crate::distribution::package::{ArtifactExpectation, IndexPackage, PackageKind};
use crate::distribution::plan::Plan;
use crate::distribution::warehouse::Warehouse;
use crate::errors::{ArtifactError, PackError, SemanticSearchError};
use crate::semantic::chunk_key::ChunkKey;
use crate::semantic::oxv::codec::CodecSpec;
use crate::semantic::oxv::reader::Segment;
use crate::semantic::oxv::scan::LINK_UNRESOLVED;
use crate::semantic::segment_set::{
    install_package, InstallExpectation, InstallSource, ReleaseManifest, SegmentSet,
};
use serde::Serialize;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::path::{Path, PathBuf};

/// G1: the most of a segment's components `i8-sym-dim` may clip.
pub const G1_MAX_CLIP_RATE: f64 = 1e-4;
/// G5: vectors compared with their `f32` originals.
pub const G5_SAMPLES: usize = 20_000;
/// G5: the least mean cosine of a decoded vector and its original.
pub const G5_MIN_MEAN_COSINE: f64 = 0.9995;
/// G5: the least 0.1st-percentile cosine.
pub const G5_MIN_P001_COSINE: f64 = 0.998;
/// G7: the largest base, in bytes.
pub const G7_MAX_BASE_BYTES: u64 = 2_000_000_000;
/// G7: the largest delta, as a fraction of its base; a larger one is published as a base.
pub const G7_MAX_DELTA_RATIO: f64 = 0.15;
/// G10: the most a base and its deltas should be, as a multiple of the base; past it the
/// next release should be a base. Reported, not enforced.
pub const G10_MAX_GROWTH: f64 = 1.3;

/// One gate's verdict.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Gate {
    pub gate: &'static str,
    pub passed: bool,
    pub detail: String,
}

/// Every gate [`verify_release`] checks, in order.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GateReport {
    pub segment_id: String,
    pub gates: Vec<Gate>,
}

impl GateReport {
    pub fn passed(&self) -> bool {
        self.gates.iter().all(|gate| gate.passed)
    }
}

/// What [`verify_release`] checks a release against: what it was assembled from.
pub struct VerifyRequest<'a> {
    /// The assembled release: its package and `release.json`.
    pub release_dir: &'a Path,
    pub plan: &'a Plan,
    pub warehouse: &'a Warehouse,
    /// The ledger it was assembled against: a delta's, or a base's that kept the epoch.
    pub previous: Option<&'a Ledger>,
    /// Where G8 assembles the release again; removed after.
    pub scratch_dir: PathBuf,
    /// G5's sample; [`G5_SAMPLES`] for a release.
    pub samples: usize,
}

/// Check G1, G5, G7, G8 and G9, and report G10. An `Err` is a release that could not be checked at
/// all; a gate that fails is in the report.
pub fn verify_release(request: &VerifyRequest<'_>) -> Result<GateReport, PackError> {
    let dir = request.release_dir;
    let manifest: ReleaseManifest = read_json(&dir.join(RELEASE_FILE))?;
    let segment = Segment::open(&dir.join(SEGMENT_FILE))?;
    let content = content_pass(&segment, request.warehouse, request.samples);
    let gates = vec![
        g1(&manifest, &segment, &content),
        g5(&content),
        g7(&manifest, request.previous),
        g8(request, &manifest)?,
        g9(dir, &manifest),
        g10(&manifest, request.previous),
    ];
    Ok(GateReport {
        segment_id: manifest.segment_id,
        gates,
    })
}

/// What one pass over every slot found: each slot's codes against its warehouse vector's
/// encoding, the clipped components, and the cosines of the sample.
struct Content {
    slots: u64,
    without_vector: u64,
    mismatched: u64,
    bad_scales: u64,
    clipped: u64,
    cosines: Vec<f64>,
}

fn content_pass(segment: &Segment, warehouse: &Warehouse, samples: usize) -> Content {
    let codec = segment.codec();
    let dim = codec.dim();
    let slots = segment.slot_count();
    let step = (slots as usize).div_ceil(samples.max(1)).max(1);
    let (mut original, mut decoded) = (vec![0f32; dim], vec![0f32; dim]);
    let mut encoded = vec![0u8; codec.bytes_per_vector()];
    let mut content = Content {
        slots: u64::from(slots),
        without_vector: 0,
        mismatched: 0,
        bad_scales: 0,
        clipped: 0,
        cosines: Vec::new(),
    };
    for slot in 0..slots {
        let scale = segment.vector_scale(slot);
        if scale.is_some_and(|scale| !(scale.is_finite() && scale > 0.0)) {
            content.bad_scales += 1;
        }
        let Some(record) = warehouse.find_key(&segment.key(slot).0) else {
            content.without_vector += 1;
            continue;
        };
        warehouse.vector(record, &mut original);
        let again = codec.encode(&original, &mut encoded);
        content.clipped += again.clipped as u64;
        if encoded != segment.vector(slot)
            || again.scale.map(f32::to_bits) != scale.map(f32::to_bits)
        {
            content.mismatched += 1;
        }
        if (slot as usize).is_multiple_of(step) {
            codec.decode(segment.vector(slot), scale, &mut decoded);
            content.cosines.push(cosine(&original, &decoded));
        }
    }
    content
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (f64::from(*x), f64::from(*y));
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    if aa == 0.0 || bb == 0.0 {
        return 0.0;
    }
    ab / (aa.sqrt() * bb.sqrt())
}

fn gate(gate: &'static str, passed: bool, detail: String) -> Gate {
    Gate {
        gate,
        passed,
        detail,
    }
}

/// G1: the identity is complete and the segment's own; every scale is finite and
/// positive; `i8-sym-dim` clips at most [`G1_MAX_CLIP_RATE`] of the components.
fn g1(manifest: &ReleaseManifest, segment: &Segment, content: &Content) -> Gate {
    let codec = segment.codec();
    let mut faults = Vec::new();
    if let Err(error) = manifest.identity.validate_complete() {
        faults.push(format!("the identity is incomplete: {error}"));
    }
    if manifest.identity.identity_digest() != segment.identity_digest()
        || manifest.identity_digest != manifest.identity.identity_digest_hex()
    {
        faults.push("the identity digest is not the segment's".to_string());
    }
    if manifest.identity.store.vector_precision != codec.name()
        || manifest.codec_params_sha256 != hex(&segment.codec_params_sha256())
    {
        faults.push("the codec is not the one the manifest declares".to_string());
    }
    let dim_scales = codec.scales().unwrap_or_default();
    let bad = content.bad_scales
        + dim_scales
            .iter()
            .filter(|scale| !(scale.is_finite() && **scale > 0.0))
            .count() as u64;
    if bad > 0 {
        faults.push(format!("{bad} scale(s) are not finite and positive"));
    }
    let components = content.slots * codec.dim() as u64;
    let clip_rate = content.clipped as f64 / components.max(1) as f64;
    let clip = if codec.clip_q().is_some() {
        if clip_rate > G1_MAX_CLIP_RATE {
            faults.push(format!(
                "{clip_rate:.2e} of the components are clipped, more than {G1_MAX_CLIP_RATE:e}"
            ));
        }
        format!("clip rate {clip_rate:.2e}")
    } else {
        "no clipping in this codec".to_string()
    };
    gate(
        "G1",
        faults.is_empty(),
        if faults.is_empty() {
            format!("identity complete; {} scales sound; {clip}", codec.name())
        } else {
            faults.join("; ")
        },
    )
}

/// G5: every slot holds its key's warehouse vector, encoded; on the sample, the decoded
/// vectors' cosine with their originals has a mean of at least [`G5_MIN_MEAN_COSINE`] and
/// a 0.1st percentile of at least [`G5_MIN_P001_COSINE`].
fn g5(content: &Content) -> Gate {
    let mut cosines = content.cosines.clone();
    cosines.sort_by(f64::total_cmp);
    let mean = cosines.iter().sum::<f64>() / cosines.len().max(1) as f64;
    let p001 = cosines
        .get(((cosines.len().max(1) - 1) as f64 * 0.001) as usize)
        .copied()
        .unwrap_or(0.0);
    let mut faults = Vec::new();
    if content.without_vector > 0 {
        faults.push(format!(
            "{} slot(s) have no vector in the warehouse",
            content.without_vector
        ));
    }
    if content.mismatched > 0 {
        faults.push(format!(
            "{} slot(s) are not their warehouse vector, encoded",
            content.mismatched
        ));
    }
    if mean < G5_MIN_MEAN_COSINE || p001 < G5_MIN_P001_COSINE {
        faults.push(format!(
            "cosine mean {mean:.6} and p0.1 {p001:.6}, under {G5_MIN_MEAN_COSINE} and \
             {G5_MIN_P001_COSINE}"
        ));
    }
    gate(
        "G5",
        faults.is_empty(),
        if faults.is_empty() {
            format!(
                "{} slot(s) encode their vectors; on {} sampled, cosine mean {mean:.6}, \
                 p0.1 {p001:.6}",
                content.slots,
                cosines.len()
            )
        } else {
            faults.join("; ")
        },
    )
}

/// G7: a base is at most [`G7_MAX_BASE_BYTES`]; a delta at most [`G7_MAX_DELTA_RATIO`] of
/// its base, or it is published as a base.
fn g7(manifest: &ReleaseManifest, previous: Option<&Ledger>) -> Gate {
    let size = manifest.segment.size;
    match manifest.kind {
        PackageKind::Delta => match previous {
            Some(ledger) => {
                let base = ledger.manifest.base.size;
                let ratio = size as f64 / base.max(1) as f64;
                gate(
                    "G7",
                    ratio <= G7_MAX_DELTA_RATIO,
                    format!(
                        "the delta is {size} bytes, {ratio:.3} of the v{} base's {base}{}",
                        ledger.manifest.base.library_version,
                        if ratio <= G7_MAX_DELTA_RATIO {
                            ""
                        } else {
                            ": publish a base"
                        }
                    ),
                )
            }
            None => gate(
                "G7",
                false,
                "a delta's size is checked against its base: give the previous ledger".to_string(),
            ),
        },
        _ => gate(
            "G7",
            size <= G7_MAX_BASE_BYTES,
            format!("the base is {size} bytes, of at most {G7_MAX_BASE_BYTES}"),
        ),
    }
}

/// G8: assembling again gives the same segment, package, manifest and ledger.
fn g8(request: &VerifyRequest<'_>, manifest: &ReleaseManifest) -> Result<Gate, PackError> {
    let segment = Segment::open(&request.release_dir.join(SEGMENT_FILE))?;
    let codec = segment.codec();
    let keeps =
        |ledger: &Ledger| ledger.manifest.codec.params_sha256 == manifest.codec_params_sha256;
    let epoch = match (manifest.kind, request.previous) {
        (PackageKind::Base, Some(ledger)) if keeps(ledger) => EpochChoice::Previous,
        (PackageKind::Base, _) => EpochChoice::New(match codec.clip_q() {
            Some(clip_q) => CodecSpec::I8SymDim { clip_q },
            None => CodecSpec::parse(codec.name(), 1.0).map_err(malformed)?,
        }),
        _ => EpochChoice::Previous,
    };
    let scratch = &request.scratch_dir;
    if scratch.exists() {
        std::fs::remove_dir_all(scratch).map_err(crate::distribution::files::io_error(format!(
            "removing {}",
            scratch.display()
        )))?;
    }
    let again = assemble(&AssembleRequest {
        plan: request.plan,
        warehouse: request.warehouse,
        kind: manifest.kind,
        previous: request.previous,
        epoch,
        out_dir: scratch.clone(),
        created_at: manifest.created_at.clone(),
        built_by: manifest.built_by.clone(),
    });
    let same =
        |name: &str| -> Result<bool, PackError> {
            Ok(FileDigest::of(&request.release_dir.join(name))?
                == FileDigest::of(&scratch.join(name))?)
        };
    let verdict = match again {
        Err(error) => gate("G8", false, format!("assembling again failed: {error}")),
        Ok(report) => {
            let ledger = manifest_file_name(manifest.to_library_version);
            let mut differ = Vec::new();
            for name in [
                SEGMENT_FILE,
                RELEASE_FILE,
                "manifest.json",
                "payloads.json",
                &ledger,
            ] {
                if !same(name)? {
                    differ.push(name.to_string());
                }
            }
            if differ.is_empty() {
                gate(
                    "G8",
                    true,
                    format!(
                        "assembled again: segment SHA-256 {} and package digest {} alike",
                        report.manifest.segment.sha256, report.manifest.package_digest
                    ),
                )
            } else {
                gate(
                    "G8",
                    false,
                    format!("assembled again, {} differ", differ.join(", ")),
                )
            }
        }
    };
    let _ = std::fs::remove_dir_all(scratch);
    Ok(verdict)
}

/// G9: the package verifies for install, every payload byte read, under the digest the
/// manifest carries — which is its own.
fn g9(dir: &Path, manifest: &ReleaseManifest) -> Gate {
    let expectation = ArtifactExpectation::with_published_digest(
        manifest.identity.clone(),
        manifest.package_digest.clone(),
    );
    let verdict = IndexPackage::verify_for_install(dir, &expectation)
        .map_err(|error| error.to_string())
        .and_then(|_| {
            let segment = FileDigest::of(&dir.join(SEGMENT_FILE)).map_err(|e| e.to_string())?;
            if segment.sha256 != manifest.segment.sha256 || segment.size != manifest.segment.size {
                return Err("the segment is not the one the manifest names".to_string());
            }
            Ok(())
        });
    match verdict {
        Ok(()) => gate(
            "G9",
            true,
            format!(
                "verify_for_install passes; package digest {}",
                manifest.package_digest
            ),
        ),
        Err(reason) => gate("G9", false, reason),
    }
}

/// G10, reported rather than enforced: the base and every delta on it, this one included,
/// as a multiple of the base — past [`G10_MAX_GROWTH`] the next release should be a base.
fn g10(manifest: &ReleaseManifest, previous: Option<&Ledger>) -> Gate {
    let detail = match (manifest.kind, previous) {
        (PackageKind::Delta, Some(ledger)) => {
            let base = ledger.manifest.base.size;
            let chain = base + ledger.manifest.deltas_since_base + manifest.segment.size;
            let growth = chain as f64 / base.max(1) as f64;
            format!(
                "base and deltas are {chain} bytes, {growth:.3} × the base{}",
                if growth <= G10_MAX_GROWTH {
                    ""
                } else {
                    ": the next release should be a base"
                }
            )
        }
        (PackageKind::Delta, None) => "no previous ledger: growth unknown".to_string(),
        _ => "a base starts a chain".to_string(),
    };
    gate("G10", true, detail)
}

/// The set a device holds after installing `chain` — assembled release directories, the
/// published state first and the new release last — into a new set at `dir`, which must
/// not hold one; opened with the runtime reader.
pub fn simulate_device(dir: &Path, chain: &[&Path]) -> Result<SegmentSet, SemanticSearchError> {
    for release in chain {
        let path = release.join(RELEASE_FILE);
        let unusable = |reason: String| ArtifactError::MetadataUnusable {
            path: path.display().to_string(),
            reason,
        };
        let json = std::fs::read_to_string(&path).map_err(|error| unusable(error.to_string()))?;
        let manifest: ReleaseManifest =
            serde_json::from_str(&json).map_err(|error| unusable(error.to_string()))?;
        install_package(
            dir,
            &InstallSource {
                segment: &release.join(SEGMENT_FILE),
                manifest_json: &json,
            },
            &InstallExpectation {
                identity: manifest.identity,
                published_manifest_sha256: Some(hex(&sha256(json.as_bytes()))),
            },
            &CancellationToken::new(),
        )?;
    }
    SegmentSet::open(dir)
}

/// Every record of `book` a scan of `set` reaches, `(key, hint)`, sorted and deduplicated:
/// the primary and extra records of live slots, and the foreign records that resolve to
/// one.
pub fn book_records(set: &SegmentSet, book: &str, out: &mut Vec<(ChunkKey, u32)>) {
    out.clear();
    let segments = set.segments();
    let live = |seg: usize, slot: u32| !set.deleted()[seg].is_set(slot);
    for (seg, segment) in segments.iter().enumerate() {
        let books = segment.books();
        let Ok(at) = books.binary_search_by(|entry| entry.name.as_bytes().cmp(book.as_bytes()))
        else {
            continue;
        };
        let entry = &books[at];
        for slot in entry.slots.clone() {
            if live(seg, slot) {
                out.push((segment.key(slot), segment.hint(slot)));
            }
        }
        for index in entry.extras.clone() {
            let (slot, hint) = segment.extra(index);
            if live(seg, slot) {
                out.push((segment.key(slot), hint));
            }
        }
        for index in entry.foreign.clone() {
            let link = set.links()[seg][index as usize];
            if link.flags & LINK_UNRESOLVED == 0 && live(link.seg as usize, link.slot) {
                let (key, _, hint) = segment.foreign_record(index);
                out.push((key, hint));
            }
        }
    }
    out.sort_unstable();
    out.dedup();
}

/// How much of a plan a set reaches.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Coverage {
    pub records: u64,
    pub reachable: u64,
    /// The first record a scan does not reach, `(book, ordinal)`.
    pub first_unreachable: Option<(String, u32)>,
}

impl Coverage {
    pub fn complete(&self) -> bool {
        self.records == self.reachable
    }
}

/// G3's count: the plan's records whose (book, key) a scan of `set` reaches. Holds one
/// book's records at a time.
pub fn coverage(set: &SegmentSet, plan: &Plan) -> Coverage {
    let mut coverage = Coverage {
        records: 0,
        reachable: 0,
        first_unreachable: None,
    };
    let (mut current, mut records) = (u32::MAX, Vec::new());
    for record in plan.records.iter() {
        if record.book != current {
            current = record.book;
            book_records(set, plan.books.name(current), &mut records);
        }
        coverage.records += 1;
        let key = record.key();
        let at = records.partition_point(|(other, _)| *other < key);
        if records.get(at).is_some_and(|(other, _)| *other == key) {
            coverage.reachable += 1;
        } else if coverage.first_unreachable.is_none() {
            coverage.first_unreachable =
                Some((plan.books.name(record.book).to_string(), record.ordinal));
        }
    }
    coverage
}

/// The `f32` vectors of every key a set holds live, from the warehouse they were
/// assembled from: the exact scan a retrieval check (G6) compares the set's with.
pub struct ExactReference<'a> {
    warehouse: &'a Warehouse,
    /// `(warehouse record, key)`, in record order.
    entries: Vec<(u64, ChunkKey)>,
}

impl<'a> ExactReference<'a> {
    /// Every live key of `set`. Refused if the warehouse lacks one's vector.
    pub fn new(set: &SegmentSet, warehouse: &'a Warehouse) -> Result<Self, PackError> {
        let mut entries = Vec::new();
        for (seg, segment) in set.segments().iter().enumerate() {
            for slot in 0..segment.slot_count() {
                if set.deleted()[seg].is_set(slot) {
                    continue;
                }
                let key = segment.key(slot);
                let record = warehouse.find_key(&key.0).ok_or_else(|| {
                    malformed(format!(
                        "the warehouse holds no vector of key {}",
                        hex(&key.0)
                    ))
                })?;
                entries.push((record, key));
            }
        }
        entries.sort_unstable();
        Ok(Self { warehouse, entries })
    }

    /// Keys it scans.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The `top_k` keys whose vectors' dot product with `query` is highest, best first,
    /// ties by key; on `threads` threads.
    pub fn top_k(&self, query: &[f32], top_k: usize, threads: usize) -> Vec<(ChunkKey, f32)> {
        let dim = self.warehouse.dim();
        let chunk = self.entries.len().div_ceil(threads.max(1)).max(1);
        let best = |entries: &[(u64, ChunkKey)]| {
            let mut heap: BinaryHeap<Reverse<Scored>> = BinaryHeap::with_capacity(top_k + 1);
            let mut vector = vec![0f32; dim];
            for (record, key) in entries {
                self.warehouse.vector(*record, &mut vector);
                let score = dot(&vector, query);
                heap.push(Reverse(Scored(score, *key)));
                if heap.len() > top_k {
                    heap.pop();
                }
            }
            heap.into_iter()
                .map(|Reverse(scored)| scored)
                .collect::<Vec<_>>()
        };
        let mut all: Vec<Scored> = std::thread::scope(|scope| {
            let workers: Vec<_> = self
                .entries
                .chunks(chunk)
                .map(|entries| scope.spawn(move || best(entries)))
                .collect();
            workers
                .into_iter()
                .flat_map(|worker| worker.join().expect("a scan thread"))
                .collect()
        });
        all.sort_unstable_by(|a, b| b.cmp(a));
        all.truncate(top_k);
        all.into_iter()
            .map(|Scored(score, key)| (key, score))
            .collect()
    }
}

/// A score and its key, ordered by score and then by key, reversed — so the better of two
/// equal scores is the lower key.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Scored(f32, ChunkKey);

impl Eq for Scored {}

impl PartialOrd for Scored {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Scored {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .total_cmp(&other.0)
            .then_with(|| other.1.cmp(&self.1))
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut lanes = [0f32; 8];
    let (a8, a_rest) = a.as_chunks::<8>();
    let (b8, b_rest) = b.as_chunks::<8>();
    for (x, y) in a8.iter().zip(b8) {
        for lane in 0..8 {
            lanes[lane] += x[lane] * y[lane];
        }
    }
    lanes.iter().sum::<f32>() + a_rest.iter().zip(b_rest).map(|(x, y)| x * y).sum::<f32>()
}

/// recall@k: the share of `exact`'s keys that `found` has too.
pub fn recall(found: &[ChunkKey], exact: &[ChunkKey]) -> f64 {
    if exact.is_empty() {
        return 1.0;
    }
    let found: std::collections::HashSet<&ChunkKey> = found.iter().collect();
    exact.iter().filter(|key| found.contains(key)).count() as f64 / exact.len() as f64
}
