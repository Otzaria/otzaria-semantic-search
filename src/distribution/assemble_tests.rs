use super::assemble::*;
use super::gates::*;
use super::ledger::Ledger;
use super::package::PackageKind;
use super::plan::{plan_from_corpus, Plan, PlanRequest};
use super::shard::ShardPolicy;
use super::testing::{corpus, family, heap, passage_package, raw_shard, stub_model, TempDir};
use super::warehouse::{Warehouse, WarehouseIdentity};
use crate::cancellation::CancellationToken;
use crate::semantic::chunk_key::ChunkKey;
use crate::semantic::chunker::ChunkerConfig;
use crate::semantic::oxv::codec::{Codec, CodecSpec};
use crate::semantic::oxv::reader::Segment;
use crate::semantic::resolve::{LiveKeySource, ResolveError};
use crate::semantic::segment_set::{compact, CompactionPolicy, SegmentSet};
use crate::semantic::versioning::{ModelIdentity, ModelPackage};
use std::collections::BTreeMap;
use std::path::PathBuf;

const A: &str = "otzaria/a.txt";
const B: &str = "otzaria/b.txt";
const C: &str = "otzaria/c.txt";
const D: &str = "otzaria/d.txt";
const CREATED: &str = "2026-10-02T00:00:00Z";

/// Line `i`'s text: eight Hebrew words, the first spelling `i`, long enough to be embedded
/// as itself.
fn text(i: usize) -> String {
    let letter = |n: usize| char::from_u32(0x05D0 + (n % 22) as u32).unwrap();
    let mut words = vec![(0..4)
        .map(|d| letter(i / 22usize.pow(d) + d as usize))
        .collect()];
    words.extend((1..8).map(|w| (0..5).map(|k| letter(w * 5 + k)).collect::<String>()));
    words.join(" ")
}

/// A build machine's state: the family, its warehouse, and the plans and releases made.
struct Machine {
    dir: TempDir,
    model: ModelIdentity,
    package: ModelPackage,
    warehouse: PathBuf,
}

impl Machine {
    fn new(name: &str) -> Self {
        let dir = TempDir::new(name);
        let model_path = stub_model(dir.path());
        let (model, package) = (family(&model_path), passage_package(&model_path));
        let warehouse = dir.join("warehouse");
        Warehouse::create(&warehouse, WarehouseIdentity::of(&model, &package)).unwrap();
        Self {
            dir,
            model,
            package,
            warehouse,
        }
    }

    /// Plan library `version`, `books` of `(book, [text index])`, against `previous`, and
    /// embed what the warehouse lacks.
    fn plan(
        &self,
        name: &str,
        version: u32,
        books: &[(&str, &[usize])],
        previous: Option<&Ledger>,
    ) -> Plan {
        let lines: Vec<(u64, String, String)> = books
            .iter()
            .flat_map(|(book, texts)| texts.iter().map(move |i| (book.to_string(), text(*i))))
            .enumerate()
            .map(|(at, (book, text))| (at as u64 + 1, book, text))
            .collect();
        let lines: Vec<(u64, &str, &str)> = lines
            .iter()
            .map(|(id, book, text)| (*id, book.as_str(), text.as_str()))
            .collect();
        let out = self.dir.join(name);
        {
            let warehouse = Warehouse::open(&self.warehouse).unwrap();
            plan_from_corpus(
                &corpus(self.dir.path(), name, version, &lines),
                PlanRequest {
                    out_dir: out.clone(),
                    model: self.model.clone(),
                    chunking: ChunkerConfig::default(),
                    passage_package: self.package.clone(),
                    previous,
                    warehouse: Some(&warehouse),
                    created_at: CREATED.to_string(),
                },
            )
            .unwrap();
        }
        let shard = raw_shard(&out, &self.dir.join(&format!("{name}-shard")));
        Warehouse::open_for_append(&self.warehouse)
            .unwrap()
            .add_shards(
                Some(&out),
                &[shard],
                &ShardPolicy {
                    allow_non_semantic: true,
                },
                CREATED.to_string(),
            )
            .unwrap();
        Plan::open(&out).unwrap()
    }

    fn assemble(
        &self,
        plan: &Plan,
        kind: PackageKind,
        previous: Option<&Ledger>,
        epoch: EpochChoice,
        out: &str,
    ) -> AssembleReport {
        let warehouse = Warehouse::open(&self.warehouse).unwrap();
        assemble(&AssembleRequest {
            plan,
            warehouse: &warehouse,
            kind,
            previous,
            epoch,
            out_dir: self.dir.join(out),
            created_at: CREATED.to_string(),
            built_by: Some(serde_json::json!({ "runId": "test" })),
        })
        .unwrap()
    }

    fn verify(&self, plan: &Plan, release: &str, previous: Option<&Ledger>) -> GateReport {
        let warehouse = Warehouse::open(&self.warehouse).unwrap();
        verify_release(&VerifyRequest {
            release_dir: &self.dir.join(release),
            plan,
            warehouse: &warehouse,
            previous,
            scratch_dir: self.dir.join(&format!("{release}-again")),
            samples: G5_SAMPLES,
        })
        .unwrap()
    }
}

const NEW: EpochChoice = EpochChoice::New(CodecSpec::I8SymVec);

fn v1() -> Vec<(&'static str, &'static [usize])> {
    // Text 2 twice in one book; text 1 in two books.
    vec![(A, &[1, 2, 3, 2, 4]), (B, &[5, 1, 6, 7]), (C, &[8, 11])]
}

fn v2() -> Vec<(&'static str, &'static [usize])> {
    // A: 3 moved, 9 new, 6 in from B, 4 out to the new D, with 10 new. C is gone, and
    // with it texts 8 and 11.
    vec![(A, &[1, 3, 9, 2, 6]), (B, &[5, 1, 7]), (D, &[10, 4])]
}

/// A segment's slots — key, hint, codes, scale bits — its extras and its books.
type Contents = (
    Vec<(ChunkKey, u32, Vec<u8>, Option<u32>)>,
    Vec<(u32, u32)>,
    Vec<String>,
);

fn contents(segment: &Segment) -> Contents {
    let slots = (0..segment.slot_count())
        .map(|slot| {
            (
                segment.key(slot),
                segment.hint(slot),
                segment.vector(slot).to_vec(),
                segment.vector_scale(slot).map(f32::to_bits),
            )
        })
        .collect();
    let extras = (0..segment.extra_count())
        .map(|e| segment.extra(e))
        .collect();
    let books = segment.books().iter().map(|b| b.name.to_string()).collect();
    (slots, extras, books)
}

/// The live index of a plan's library, for compaction to re-anchor on.
struct Live(u32, BTreeMap<String, Vec<(u32, u64)>>);

impl Live {
    fn of(plan: &Plan) -> Self {
        let mut books: BTreeMap<String, Vec<(u32, u64)>> = BTreeMap::new();
        for record in plan.records.iter() {
            books
                .entry(plan.books.name(record.book).to_string())
                .or_default()
                .push((record.ordinal, record.key().column_value()));
        }
        Self(plan.manifest.library_version, books)
    }
}

impl LiveKeySource for Live {
    fn library_version(&self) -> u32 {
        self.0
    }

    fn book_keys(&self, book: &str, out: &mut Vec<(u32, u64)>) -> Result<bool, ResolveError> {
        Ok(self.1.get(book).is_some_and(|lines| {
            out.extend_from_slice(lines);
            true
        }))
    }
}

/// A base of v1, a delta to v2 installed on it and compacted on the live v2 index, is the
/// base of v2: slot for slot, the same keys, hints, codes and scales, the same extras and
/// books. And each passes the sidecar's gates.
#[test]
fn delta_equals_rebuild() {
    let machine = Machine::new("delta_equals_rebuild");
    let p1 = machine.plan("p1", 1, &v1(), None);
    let base1 = machine.assemble(&p1, PackageKind::Base, None, NEW, "base1");
    assert_eq!(base1.manifest.counts.slots, 9);
    let ledger1 = Ledger::open(&base1.out_dir, Some(1)).unwrap();

    let p2 = machine.plan("p2", 2, &v2(), Some(&ledger1));
    assert_eq!(
        p2.manifest.counts.to_embed, 2,
        "only texts 9 and 10 are new"
    );
    let delta = machine.assemble(&p2, PackageKind::Delta, Some(&ledger1), NEW, "delta2");
    let counts = delta.manifest.counts;
    assert_eq!((counts.slots, counts.foreign, counts.tombstones), (2, 2, 2));
    assert_eq!(delta.ledger.deltas_since_base, delta.manifest.segment.size);
    let base2 = machine.assemble(&p2, PackageKind::Base, None, NEW, "base2");

    let work = &machine.dir;
    let chain = simulate_device(&work.join("device"), &[&base1.out_dir, &delta.out_dir]).unwrap();
    assert!(coverage(&chain, &p2).complete());
    let only_base = simulate_device(&work.join("old-device"), &[&base1.out_dir]).unwrap();
    let partial = coverage(&only_base, &p2);
    assert!(!partial.complete());
    assert_eq!(partial.records, 10);
    drop(chain);
    compact(
        &work.join("device"),
        &CompactionPolicy {
            force: true,
            ..CompactionPolicy::default()
        },
        Some(&Live::of(&p2)),
        &CancellationToken::new(),
    )
    .unwrap();
    let compacted = SegmentSet::open(&work.join("device")).unwrap();
    let direct = simulate_device(&work.join("direct"), &[&base2.out_dir]).unwrap();
    assert_eq!(compacted.segments().len(), 1);
    assert_eq!(
        contents(&compacted.segments()[0]),
        contents(&direct.segments()[0])
    );
    assert_eq!(
        compacted.segments()[0].counts(),
        direct.segments()[0].counts()
    );

    for (release, previous) in [("base1", None), ("base2", None), ("delta2", Some(&ledger1))] {
        let plan = if release == "base1" { &p1 } else { &p2 };
        let report = machine.verify(plan, release, previous);
        for gate in &report.gates {
            let sized = release == "delta2" && gate.gate == "G7";
            assert!(
                gate.passed() || sized,
                "{release} {}: {}",
                gate.gate,
                gate.detail
            );
        }
        assert_eq!(report.gates.len(), 6);
    }
}

/// Assembling twice gives the same bytes: the segment, the package, the manifest, the
/// ledger.
#[test]
fn assembly_is_deterministic() {
    let machine = Machine::new("assemble_twice");
    let p1 = machine.plan("p1", 1, &v1(), None);
    let one = machine.assemble(&p1, PackageKind::Base, None, NEW, "one");
    let two = machine.assemble(&p1, PackageKind::Base, None, NEW, "two");
    assert_eq!(one.manifest, two.manifest);
    assert_eq!(one.manifest_sha256, two.manifest_sha256);
    for name in std::fs::read_dir(&one.out_dir).unwrap() {
        let name = name.unwrap().file_name();
        assert_eq!(
            std::fs::read(one.out_dir.join(&name)).unwrap(),
            std::fs::read(two.out_dir.join(&name)).unwrap(),
            "{name:?}"
        );
    }
}

/// A release's segment id, and every file assembly writes for it with its SHA-256.
type Assembled = (&'static str, [(&'static str, &'static str); 7]);

/// What bc4c854 assembled from [`v1`] as a base, and from [`v2`] as a delta on it.
const BC4C854_BASE: Assembled = (
    "7b771ad67ebad7eb17cc7b56b4caa32d",
    [
        (
            "ledger-v1.keys",
            "b027fcfc70d4bc88593ed5ba42c4c94fee4c87eb907ff19b73c4b21900a322cb",
        ),
        (
            "ledger-v1.manifest.json",
            "f0a2eca82de1fadb5daba2425b323af70f1edce71b5ded9794b6d8a962645071",
        ),
        (
            "manifest.json",
            "ed8e7cec23717a78ac7b89fd26f8197f89a97f589ce60750e8600f3a5a89bc2f",
        ),
        (
            "pairs-v1.bin",
            "059dfa5b51b0d2d89aaf2d0e3d0f0be41a4fbddf549f779f8e4d4b2f34e3c199",
        ),
        (
            "payloads.json",
            "8270a9cc329904ac3f6a294bbc9cdfed35eb07bc481581931679c02ee69573b5",
        ),
        (
            "release.json",
            "6c6ee9007b4cc1a60bcdd313ac72d8b834c59676e1fec1c6b73ce39ff1b04a22",
        ),
        (
            "segment.oxv",
            "52cc9ed4cbbd2604c7c363e2917a97d8846561b35d93694c9ff20abfdb59999c",
        ),
    ],
);
const BC4C854_DELTA: Assembled = (
    "d6f92436ce0d73e54cd11882f857072b",
    [
        (
            "ledger-v2.keys",
            "30f923e6e514e09d0867bd59968ec877624607a289ada06978c63a057b01da2d",
        ),
        (
            "ledger-v2.manifest.json",
            "4d90e5a4a5d201bd67ee5a2a9d521179e2f17b35ebfc5de0762789f826fe55ae",
        ),
        (
            "manifest.json",
            "493e77609c85cebd9e529d29e7db9db585d94b2bcb3b055f7567d5cf1ec3b5b1",
        ),
        (
            "pairs-v2.bin",
            "885c640eb222b3e2de481c09cf8caa3c7921df9e8960caf29798809557aac8f7",
        ),
        (
            "payloads.json",
            "0217cefa78a539abbb819d5daf41e92b92b3c50aed7286b54cf51b747cd6dc9a",
        ),
        (
            "release.json",
            "30a6709b6bfabd69578396c3c7bb41f929fb3e72afda8518f8504090e53935d5",
        ),
        (
            "segment.oxv",
            "1a6eae0b0b557526a35d9afafa7c03be3983f4c5a7646650ee380a38810feea2",
        ),
    ],
);

/// What assembly writes is what it wrote at bc4c854, which assembled the published v30 and
/// which the library's build still runs: every file of a base and of a delta, byte for byte,
/// and the segment ids they install under. And this revision's installer takes them and
/// opens the set they make.
#[test]
fn a_release_is_assembled_into_the_bytes_bc4c854_assembled() {
    use sha2::{Digest, Sha256};
    let machine = Machine::new("assemble_bc4c854");
    let p1 = machine.plan("p1", 1, &v1(), None);
    let base = machine.assemble(&p1, PackageKind::Base, None, NEW, "base1");
    let ledger1 = Ledger::open(&base.out_dir, Some(1)).unwrap();
    let p2 = machine.plan("p2", 2, &v2(), Some(&ledger1));
    let delta = machine.assemble(&p2, PackageKind::Delta, Some(&ledger1), NEW, "delta2");

    // File by file, the bytes bc4c854 wrote.
    for (report, (id, files)) in [(&base, BC4C854_BASE), (&delta, BC4C854_DELTA)] {
        let kind = report.manifest.kind;
        assert_eq!(report.manifest.segment_id, id, "{kind}");
        let mut written: Vec<String> = std::fs::read_dir(&report.out_dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().to_string())
            .collect();
        written.sort();
        assert_eq!(written, files.map(|(name, _)| name), "{kind}");
        for (name, sha256) in files {
            let bytes = std::fs::read(report.out_dir.join(name)).unwrap();
            assert_eq!(
                format!("{:x}", Sha256::digest(&bytes)),
                sha256,
                "{kind} {name}"
            );
        }
    }

    let set = simulate_device(
        &machine.dir.join("device"),
        &[&base.out_dir, &delta.out_dir],
    )
    .unwrap();
    assert_eq!(set.generation(), 2);
    let ids: Vec<&str> = set
        .info()
        .segments
        .iter()
        .map(|segment| segment.id.as_str())
        .collect();
    assert_eq!(ids, [BC4C854_BASE.0, BC4C854_DELTA.0]);
    assert!(coverage(&set, &p2).complete());
}

/// A delta that ships no vector — a version that only dropped texts, or only copied texts it
/// held into another book — names no worker, since none embedded anything, and installs like
/// any delta: the set opens on it at its version, it reaches every record of the version, and
/// a dropped text is gone. A segment that ships vectors and names no worker is still
/// refused.
#[test]
fn a_delta_that_ships_no_vector_installs() {
    use crate::semantic::oxv::scan::ScanRequest;
    use crate::semantic::segment_set::{install_package, InstallExpectation, InstallSource};
    use crate::semantic::versioning::EmbeddingWorker;
    use std::collections::BTreeSet;

    // Book C dropped, so texts 8 and 11 leave the library; or texts 5 and 3 copied into D.
    let dropped: Vec<(&str, &[usize])> = vec![(A, &[1, 2, 3, 2, 4]), (B, &[5, 1, 6, 7])];
    let copied: Vec<(&str, &[usize])> = vec![
        (A, &[1, 2, 3, 2, 4]),
        (B, &[5, 1, 6, 7]),
        (C, &[8, 11]),
        (D, &[5, 3]),
    ];
    for (what, next, tombstones, foreign) in [("dropped", dropped, 2, 0), ("copied", copied, 0, 2)]
    {
        let machine = Machine::new(&format!("no_vector_{what}"));
        let p1 = machine.plan("p1", 1, &v1(), None);
        let base = machine.assemble(&p1, PackageKind::Base, None, NEW, "base1");
        let ledger1 = Ledger::open(&base.out_dir, Some(1)).unwrap();
        let p2 = machine.plan("p2", 2, &next, Some(&ledger1));
        assert_eq!(p2.manifest.counts.to_embed, 0, "{what}");
        let delta = machine.assemble(&p2, PackageKind::Delta, Some(&ledger1), NEW, "delta2");
        let counts = delta.manifest.counts;
        assert_eq!(
            (counts.slots, counts.tombstones, counts.foreign),
            (0, tombstones, foreign),
            "{what}"
        );
        let no_worker = EmbeddingWorker {
            backend: String::new(),
            device: String::new(),
        };
        assert_eq!(delta.manifest.provenance.worker, no_worker, "{what}");

        let device = machine.dir.join("device");
        let set = simulate_device(&device, &[&base.out_dir, &delta.out_dir])
            .unwrap_or_else(|error| panic!("{what}: a delta with no vector must install: {error}"));
        assert_eq!(
            (set.generation(), set.info().library_version),
            (2, 2),
            "{what}"
        );
        assert!(coverage(&set, &p2).complete(), "{what}");
        let mut query = vec![0f32; set.codec().dim()];
        query[0] = 1.0;
        let reached: BTreeSet<ChunkKey> = set
            .scan(
                &query,
                &ScanRequest {
                    top_k: 1000,
                    books: None,
                    threads: 1,
                },
                &CancellationToken::new(),
            )
            .unwrap()
            .into_iter()
            .map(|hit| hit.key)
            .collect();
        let held: BTreeSet<ChunkKey> = p2.records.iter().map(|record| record.key()).collect();
        assert_eq!(
            reached, held,
            "{what}: the version's texts, and no dropped one"
        );
        drop(set);

        // The base ships vectors: without a worker, it is refused.
        let json = std::fs::read_to_string(base.out_dir.join(RELEASE_FILE)).unwrap();
        let mut manifest: crate::semantic::segment_set::ReleaseManifest =
            serde_json::from_str(&json).unwrap();
        manifest.provenance.worker = no_worker.clone();
        manifest.package_digest = manifest.package().digest();
        let refused = install_package(
            &machine.dir.join("refused"),
            &InstallSource {
                segment: &base.out_dir.join(SEGMENT_FILE),
                manifest_json: &manifest.to_json(),
            },
            &InstallExpectation {
                identity: manifest.identity.clone(),
                published_manifest_sha256: None,
            },
            &CancellationToken::new(),
        );
        match refused {
            Err(crate::errors::SemanticSearchError::Artifact(
                crate::errors::ArtifactError::ManifestDisagreesWithPayload { reason },
            )) => assert!(reason.contains("worker"), "{what}: {reason}"),
            other => panic!("{what}: a segment with vectors and no worker, got {other:?}"),
        }
    }
}

/// `books` books of `lines` consecutive texts from text 0, then `more`.
fn library(books: usize, lines: usize, more: &[(&str, &[usize])]) -> Vec<(String, Vec<usize>)> {
    (0..books)
        .map(|book| {
            (
                format!("otzaria/{book:02}.txt"),
                (book * lines..(book + 1) * lines).collect(),
            )
        })
        .chain(
            more.iter()
                .map(|(book, texts)| (book.to_string(), texts.to_vec())),
        )
        .collect()
}

fn borrowed(library: &[(String, Vec<usize>)]) -> Vec<(&str, &[usize])> {
    library
        .iter()
        .map(|(book, texts)| (book.as_str(), texts.as_slice()))
        .collect()
}

fn g5_of(report: &GateReport) -> &Gate {
    report.gates.iter().find(|gate| gate.gate == "G5").unwrap()
}

/// A delta of deletions and/or reuse passes every gate, G5 as not applicable; a delta with
/// a new vector and the base are measured as before.
#[test]
fn a_delta_that_ships_no_vector_passes_the_gates() {
    // The base is large enough for G7 to pass a small delta.
    let c: &[usize] = &[600, 601, 602, 603];
    let d: &[usize] = &[5, 305];
    let v1 = library(2, 300, &[(C, c)]);
    let deletions = library(2, 300, &[]);
    let reuse = library(2, 300, &[(C, c), (D, d)]);
    let both = library(2, 300, &[(D, d)]);
    let new = library(2, 300, &[(D, &[5, 305, 700])]);
    let dim = EpochChoice::New(CodecSpec::I8SymDim { clip_q: 1.0 });
    for (name, epoch) in [("vec", NEW), ("dim", dim)] {
        let machine = Machine::new(&format!("no_vector_gates_{name}"));
        let p1 = machine.plan("p1", 1, &borrowed(&v1), None);
        let base = machine.assemble(&p1, PackageKind::Base, None, epoch, "base1");
        let ledger1 = Ledger::open(&base.out_dir, Some(1)).unwrap();

        let report = machine.verify(&p1, "base1", None);
        assert!(report.passed(), "{name} base: {:?}", report.gates);
        assert!(
            report
                .gates
                .iter()
                .all(|gate| gate.status == GateStatus::Passed),
            "{name} base: {:?}",
            report.gates
        );
        let g5 = g5_of(&report);
        assert!(
            g5.detail
                .starts_with("604 slot(s) encode their vectors; on 604 sampled, cosine mean 0.99"),
            "{name} base: {}",
            g5.detail
        );
        let json = serde_json::to_value(&report).unwrap();
        for gate in json["gates"].as_array().unwrap() {
            let mut fields: Vec<&str> = gate
                .as_object()
                .unwrap()
                .keys()
                .map(String::as_str)
                .collect();
            fields.sort_unstable();
            assert_eq!(
                fields,
                ["detail", "gate", "passed", "status"],
                "{name} base"
            );
            assert_eq!(
                (&gate["passed"], &gate["status"]),
                (&true.into(), &"passed".into())
            );
        }

        for (what, next, slots, tombstones, foreign) in [
            ("deletions", &deletions, 0, 4, 0),
            ("reuse", &reuse, 0, 0, 2),
            ("deletions and reuse", &both, 0, 4, 2),
            ("a new text", &new, 1, 4, 2),
        ] {
            let out = format!("delta-{}", what.replace(' ', "-"));
            let p2 = machine.plan(&format!("p2-{out}"), 2, &borrowed(next), Some(&ledger1));
            assert_eq!(p2.manifest.counts.to_embed, slots, "{name} {what}");
            let delta = machine.assemble(&p2, PackageKind::Delta, Some(&ledger1), NEW, &out);
            let counts = delta.manifest.counts;
            assert_eq!(
                (counts.slots, counts.tombstones, counts.foreign),
                (slots, tombstones, foreign),
                "{name} {what}"
            );
            let report = machine.verify(&p2, &out, Some(&ledger1));
            assert_eq!(report.gates.len(), 6);
            for gate in &report.gates {
                assert!(
                    gate.passed(),
                    "{name} {what}: {} failed: {}",
                    gate.gate,
                    gate.detail
                );
            }
            assert!(report.passed(), "{name} {what}");
            let g5 = g5_of(&report);
            if slots == 0 {
                assert_eq!(g5.status, GateStatus::NotApplicable, "{name} {what}");
                assert_eq!(
                    g5.detail,
                    format!(
                        "not applicable: the delta ships no vector, only {tombstones} \
                         tombstone(s) and {foreign} foreign record(s), so there is no new \
                         vector to compare with its original"
                    ),
                    "{name} {what}"
                );
                let json = serde_json::to_value(g5).unwrap();
                assert_eq!(
                    (&json["passed"], &json["status"]),
                    (&true.into(), &"notApplicable".into())
                );
                for gate in report.gates.iter().filter(|gate| gate.gate != "G5") {
                    assert_eq!(gate.status, GateStatus::Passed, "{name} {what}");
                }
                if name == "dim" {
                    let g1 = &report.gates[0];
                    assert!(
                        g1.detail
                            .ends_with("no component to clip: the segment ships no vector"),
                        "{}",
                        g1.detail
                    );
                }
            } else {
                assert_eq!(g5.status, GateStatus::Passed, "{name} {what}");
                assert!(
                    g5.detail.starts_with(
                        "1 slot(s) encode their vectors; on 1 sampled, cosine mean 0.99"
                    ),
                    "{name} {what}: {}",
                    g5.detail
                );
            }
        }
    }
}

/// Nothing to measure where something should be fails, with no number made up.
#[test]
fn a_gate_with_nothing_to_measure_fails_unless_it_does_not_apply() {
    let machine = Machine::new("nothing_to_measure");
    let p1 = machine.plan("p1", 1, &v1(), None);
    let base = machine.assemble(&p1, PackageKind::Base, None, NEW, "base1");
    let ledger1 = Ledger::open(&base.out_dir, Some(1)).unwrap();
    let p2 = machine.plan("p2", 2, &v2(), Some(&ledger1));
    machine.assemble(&p2, PackageKind::Delta, Some(&ledger1), NEW, "delta2");

    // A warehouse without the delta's two new vectors: no cosine is sampled.
    let empty = machine.dir.join("empty-warehouse");
    Warehouse::create(
        &empty,
        WarehouseIdentity::of(&machine.model, &machine.package),
    )
    .unwrap();
    let report = verify_release(&VerifyRequest {
        release_dir: &machine.dir.join("delta2"),
        plan: &p2,
        warehouse: &Warehouse::open(&empty).unwrap(),
        previous: Some(&ledger1),
        scratch_dir: machine.dir.join("delta2-empty"),
        samples: G5_SAMPLES,
    })
    .unwrap();
    let g5 = g5_of(&report);
    assert_eq!((g5.passed(), g5.status), (false, GateStatus::Failed));
    assert_eq!(
        g5.detail,
        "2 slot(s) have no vector in the warehouse; no vector of the 2 slot(s) was sampled: \
         the cosines could not be measured"
    );
    assert!(!report.passed());

    let zero = machine.dir.join("zero-base");
    std::fs::create_dir(&zero).unwrap();
    for name in ["ledger-v1.keys", "pairs-v1.bin"] {
        std::fs::copy(base.out_dir.join(name), zero.join(name)).unwrap();
    }
    let manifest = base.out_dir.join("ledger-v1.manifest.json");
    let mut json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(manifest).unwrap()).unwrap();
    json["base"]["size"] = 0.into();
    std::fs::write(
        zero.join("ledger-v1.manifest.json"),
        serde_json::to_vec(&json).unwrap(),
    )
    .unwrap();
    let zero = Ledger::open(&zero, Some(1)).unwrap();
    assert_eq!(zero.manifest.base.size, 0);
    let report = machine.verify(&p2, "delta2", Some(&zero));
    let (g7, g10) = (&report.gates[2], &report.gates[5]);
    assert_eq!((g7.gate, g7.status), ("G7", GateStatus::Failed));
    assert_eq!(
        g7.detail,
        "the ledger records the v1 base as 0 bytes: the delta's size could not be checked \
         against it"
    );
    assert_eq!((g10.gate, g10.status), ("G10", GateStatus::Passed));
    assert_eq!(
        g10.detail,
        "the ledger records a base of 0 bytes: growth unknown"
    );
    assert!(!report.passed());
}

/// `i8-sym-dim` is calibrated on the slots' vectors exactly as the in-memory calibration
/// takes it, and G1 holds its clip rate to 1e-4.
#[test]
fn i8_sym_dim_calibration_is_exact_and_its_clip_rate_is_gated() {
    let machine = Machine::new("assemble_calibrate");
    let p1 = machine.plan("p1", 1, &v1(), None);
    let warehouse = Warehouse::open(&machine.warehouse).unwrap();
    let records: Vec<u64> = (0..warehouse.len()).collect();
    let mut vectors = Vec::new();
    for record in &records {
        let mut vector = vec![0f32; warehouse.dim()];
        warehouse.vector(*record, &mut vector);
        vectors.push(vector);
    }
    let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();
    for clip_q in [0.5, 0.9, 1.0] {
        assert_eq!(
            super::assemble::calibrate_i8_sym_dim(&warehouse, &records, clip_q).unwrap(),
            Codec::calibrate_i8_sym_dim(&refs, clip_q).unwrap()
        );
    }

    for (clip_q, passes) in [(1.0, true), (0.5, false)] {
        let out = format!("dim-{clip_q}");
        let epoch = EpochChoice::New(CodecSpec::I8SymDim { clip_q });
        let report = machine.assemble(&p1, PackageKind::Base, None, epoch, &out);
        assert_eq!(
            report.manifest.identity.store.vector_precision,
            "i8-sym-dim"
        );
        assert_eq!(report.clipped_components == 0, passes);
        let g1 = machine.verify(&p1, &out, None).gates.remove(0);
        assert_eq!((g1.gate, g1.passed()), ("G1", passes), "{}", g1.detail);
    }
}

/// A delta needs the ledger it was planned against, in its epoch, with every vector it
/// ships in the warehouse.
#[test]
fn a_delta_is_refused_off_its_chain() {
    let machine = Machine::new("assemble_refused");
    let p1 = machine.plan("p1", 1, &v1(), None);
    let base1 = machine.assemble(&p1, PackageKind::Base, None, NEW, "base1");
    let ledger1 = Ledger::open(&base1.out_dir, Some(1)).unwrap();
    let p2 = machine.plan("p2", 2, &v2(), Some(&ledger1));
    let warehouse = Warehouse::open(&machine.warehouse).unwrap();
    let request = |plan, kind, previous| AssembleRequest {
        plan,
        warehouse: &warehouse,
        kind,
        previous,
        epoch: NEW,
        out_dir: machine.dir.join("refused"),
        created_at: CREATED.to_string(),
        built_by: None,
    };
    // No ledger; a ledger the plan was not split against; a compacted segment.
    assert!(assemble(&request(&p2, PackageKind::Delta, None)).is_err());
    assert!(assemble(&request(&p1, PackageKind::Delta, Some(&ledger1))).is_err());
    assert!(assemble(&request(&p2, PackageKind::Compacted, None)).is_err());
    // A delta keeps its chain's epoch, whatever it is asked for.
    let f32_base = machine.assemble(
        &p1,
        PackageKind::Base,
        None,
        EpochChoice::New(CodecSpec::F32),
        "f32",
    );
    let f32_ledger = Ledger::open(&f32_base.out_dir, Some(1)).unwrap();
    let f32_delta = assemble(&request(&p2, PackageKind::Delta, Some(&f32_ledger))).unwrap();
    assert_eq!(f32_delta.manifest.identity.store.vector_precision, "f32");
    assert_eq!(
        f32_delta.manifest.codec_params_sha256,
        f32_base.manifest.codec_params_sha256
    );
    // A vector the warehouse lacks.
    let empty = machine.dir.join("empty-warehouse");
    Warehouse::create(
        &empty,
        WarehouseIdentity::of(&machine.model, &machine.package),
    )
    .unwrap();
    let empty = Warehouse::open(&empty).unwrap();
    let error = assemble(&AssembleRequest {
        warehouse: &empty,
        ..request(&p1, PackageKind::Base, None)
    })
    .unwrap_err()
    .to_string();
    assert!(error.contains("no vector for 9 key(s)"), "{error}");
}

/// A release whose segment changed after assembly fails G9 and G8 — and the files a
/// client downloads are listed in the patch-entry shape.
#[test]
fn a_changed_release_fails_its_gates_and_files_are_listed() {
    let machine = Machine::new("assemble_tampered");
    let p1 = machine.plan("p1", 1, &v1(), None);
    let base = machine.assemble(&p1, PackageKind::Base, None, NEW, "base1");
    assert!(machine.verify(&p1, "base1", None).passed());

    let part = machine.dir.join("x.oxv.zst");
    std::fs::write(&part, b"compressed").unwrap();
    let (manifest, digest) = release_with_files(
        &base.out_dir.join(RELEASE_FILE),
        &[part],
        "zstd",
        &machine.dir.join("x.manifest.json"),
    )
    .unwrap();
    assert_eq!(manifest.files.len(), 1);
    assert_eq!(
        manifest.files[0].uncompressed_sha256,
        manifest.segment.sha256
    );
    assert_eq!(manifest.files[0].size, 10);
    assert_eq!(manifest.package_digest, base.manifest.package_digest);
    assert_ne!(digest, base.manifest_sha256);
    assert!(asset_stem(&manifest).ends_with("-v1-base"));

    let segment = base.out_dir.join(SEGMENT_FILE);
    let codes = Segment::open(&segment).unwrap().vector(3).to_vec();
    let mut bytes = std::fs::read(&segment).unwrap();
    let at = bytes.windows(codes.len()).position(|w| w == codes).unwrap();
    bytes[at] ^= 1;
    std::fs::write(&segment, bytes).unwrap();
    let report = machine.verify(&p1, "base1", None);
    let failed: Vec<&str> = report
        .gates
        .iter()
        .filter(|gate| !gate.passed())
        .map(|gate| gate.gate)
        .collect();
    assert_eq!(failed, ["G5", "G8", "G9"]);
}

/// The exact reference ranks a key's own vector first.
#[test]
fn the_exact_reference_finds_a_vector_itself() {
    let machine = Machine::new("assemble_exact");
    let p1 = machine.plan("p1", 1, &v1(), None);
    let base = machine.assemble(&p1, PackageKind::Base, None, NEW, "base1");
    let set = simulate_device(&machine.dir.join("device"), &[&base.out_dir]).unwrap();
    let warehouse = Warehouse::open(&machine.warehouse).unwrap();
    let exact = ExactReference::new(&set, &warehouse).unwrap();
    assert_eq!(exact.len(), 9);
    let sha = super::files::sha256(text(6).as_bytes());
    let mut query = vec![0f32; warehouse.dim()];
    warehouse.vector(warehouse.find(&sha).unwrap(), &mut query);
    let top = exact.top_k(&query, 3, 2);
    assert_eq!(top.len(), 3);
    assert_eq!(top[0].0 .0[..], sha[..16]);
    assert!((top[0].1 - 1.0).abs() < 1e-5 && top[1].1 <= top[0].1);
    let keys: Vec<ChunkKey> = top.iter().map(|(key, _)| *key).collect();
    assert_eq!(recall(&keys[..1], &keys), 1.0 / 3.0);
}

/// The audit's case: a vector bit flipped in the warehouse after a base was assembled from
/// it. Assembling again, the gates and G6's exact reference all refuse the warehouse,
/// naming the batch, rather than check the release against its corrupt vectors.
#[test]
fn a_corrupt_warehouse_fails_assembly_the_gates_and_the_exact_reference() {
    let machine = Machine::new("assemble_corrupt_warehouse");
    let p1 = machine.plan("p1", 1, &v1(), None);
    let base = machine.assemble(&p1, PackageKind::Base, None, NEW, "base1");
    let set = simulate_device(&machine.dir.join("device"), &[&base.out_dir]).unwrap();
    let vectors = machine.warehouse.join("vectors.f32");
    let mut bytes = std::fs::read(&vectors).unwrap();
    let middle = bytes.len() / 2;
    bytes[middle] ^= 0x40;
    std::fs::write(&vectors, bytes).unwrap();

    let warehouse = Warehouse::open(&machine.warehouse).unwrap();
    let assembled = assemble(&AssembleRequest {
        plan: &p1,
        warehouse: &warehouse,
        kind: PackageKind::Base,
        previous: None,
        epoch: NEW,
        out_dir: machine.dir.join("again"),
        created_at: CREATED.to_string(),
        built_by: None,
    })
    .err()
    .unwrap();
    let verified = verify_release(&VerifyRequest {
        release_dir: &base.out_dir,
        plan: &p1,
        warehouse: &warehouse,
        previous: None,
        scratch_dir: machine.dir.join("base1-again"),
        samples: G5_SAMPLES,
    })
    .err()
    .unwrap();
    let exact = ExactReference::new(&set, &warehouse).err().unwrap();
    for error in [assembled, verified, exact] {
        let error = error.to_string();
        assert!(
            error.contains("batch 0 (records 0..9") && error.contains("vectors.f32 hashes to"),
            "{error}"
        );
    }
    assert!(!machine.dir.join("again").exists());
}

/// Assembly's heap grows with the slots it ships, by much less than a vector each: the
/// vectors stream from the mapped warehouse.
#[test]
fn assembly_memory_is_bounded() {
    let machine = Machine::new("assemble_memory");
    let library = |lines: usize| -> Vec<(String, Vec<usize>)> {
        (0..lines / 500)
            .map(|book| {
                (
                    format!("otzaria/{book:04}.txt"),
                    (book * 500..(book + 1) * 500).collect(),
                )
            })
            .collect()
    };
    let mut peaks = Vec::new();
    for (version, lines) in [(1, 10_000), (2, 30_000)] {
        let books = library(lines);
        let books: Vec<(&str, &[usize])> = books
            .iter()
            .map(|(name, texts)| (name.as_str(), texts.as_slice()))
            .collect();
        let plan = machine.plan(&format!("p{version}"), version, &books, None);
        let held = heap::start();
        let report = machine.assemble(&plan, PackageKind::Base, None, NEW, &format!("b{version}"));
        peaks.push((heap::peak() - held) as f64);
        assert_eq!(report.manifest.counts.slots as usize, lines);
    }
    let per_slot = (peaks[1] - peaks[0]) / 20_000.0;
    let vector = (Warehouse::open(&machine.warehouse).unwrap().dim() * 4) as f64;
    assert!(
        per_slot < 200.0 && per_slot < vector,
        "{per_slot:.0} bytes a slot, peaks {peaks:?}"
    );
}
