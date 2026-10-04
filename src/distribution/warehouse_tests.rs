use super::files::sha256;
use super::plan::{plan_from_corpus, EmbedManifest, HeldVectors, PlanRequest};
use super::shard::*;
use super::testing::{corpus, family, passage_package, raw_shard, stub_model, TempDir, DIM};
use super::warehouse::{Warehouse, WarehouseIdentity};
use crate::semantic::backend::Pooling;
use crate::semantic::chunker::ChunkerConfig;
use crate::semantic::embedding::{EmbeddingConfig, EmbeddingRuntime};
use crate::semantic::versioning::ModelIdentity;
use std::path::{Path, PathBuf};

const TEXTS: [&str; 4] = [
    "בראשית ברא אלהים את השמים ואת הארץ",
    "והארץ היתה תהו ובהו וחשך על פני תהום רבה",
    "ויאמר אלהים יהי אור ויהי אור וירא אלהים",
    "מאימתי קורין את שמע בערבית משעה שהכהנים",
];

const POLICY: ShardPolicy = ShardPolicy {
    allow_non_semantic: true,
};

struct Fixture {
    dir: TempDir,
    plan: PathBuf,
    model: ModelIdentity,
    runtime: EmbeddingRuntime,
}

fn fixture(name: &str, texts: &[&str]) -> Fixture {
    let dir = TempDir::new(name);
    let model_path = stub_model(dir.path());
    let lines: Vec<(u64, &str, &str)> = texts
        .iter()
        .enumerate()
        .map(|(index, text)| (index as u64 + 1, "otzaria/a.txt", *text))
        .collect();
    let plan = dir.join("plan");
    let model = family(&model_path);
    plan_from_corpus(
        &corpus(dir.path(), "v1", 1, &lines),
        PlanRequest {
            out_dir: plan.clone(),
            model: model.clone(),
            chunking: ChunkerConfig::default(),
            passage_package: passage_package(&model_path),
            previous: None,
            warehouse: None,
            created_at: "2026-10-02T00:00:00Z".to_string(),
        },
    )
    .unwrap();
    let mut runtime = EmbeddingRuntime::new(EmbeddingConfig {
        model_path,
        embedding_dim: DIM,
        max_tokens: 512,
        batch_size: 2,
        pooling: Pooling::InGraph,
    });
    runtime.load().unwrap();
    Fixture {
        dir,
        plan,
        model,
        runtime,
    }
}

impl Fixture {
    fn shard(&self, name: &str, skip: u64, take: u64) -> PathBuf {
        let out = self.dir.join(name);
        let worker = WorkerInfo {
            name: "test".into(),
            version: "0".into(),
            device: "cpu".into(),
            ep: "cpu".into(),
            mode: MODE_MOCK.into(),
        };
        let manifest = EmbedManifest::read(&self.plan).unwrap();
        embed_shard(
            &self.plan,
            &manifest,
            skip,
            take,
            &self.runtime,
            2,
            worker,
            &out,
        )
        .unwrap();
        out
    }

    fn warehouse(&self) -> PathBuf {
        let at = self.dir.join("warehouse");
        let manifest = EmbedManifest::read(&self.plan).unwrap();
        Warehouse::create(
            &at,
            WarehouseIdentity::of(&self.model, &manifest.passage_package),
        )
        .unwrap();
        at
    }
}

fn vector_of(shard: &Path, record: usize) -> Vec<f32> {
    let bytes = std::fs::read(shard.join(VECTORS_FILE)).unwrap();
    bytes[record * DIM as usize * 4..][..DIM as usize * 4]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|value| f32::from_le_bytes(*value))
        .collect()
}

/// A warehouse takes a plan's shards once: every vector is found by its text's full
/// SHA-256, a second add of the same shards adds nothing, and a plan made against the
/// warehouse asks for nothing it holds.
#[test]
fn a_warehouse_takes_each_vector_once_and_finds_it_by_its_digest() {
    let fixture = fixture("warehouse_add", &TEXTS);
    let shards = [fixture.shard("s0", 0, 2), fixture.shard("s1", 2, 9)];
    let at = fixture.warehouse();
    let mut warehouse = Warehouse::open_for_append(&at).unwrap();
    let report = warehouse
        .add_shards(Some(&fixture.plan), &shards, &POLICY, "now".into())
        .unwrap();
    assert_eq!(
        (report.records, report.added, report.held, report.total),
        (4, 4, 0, 4)
    );

    let mut vector = vec![0f32; DIM as usize];
    for (index, text) in TEXTS.iter().enumerate() {
        let record = warehouse.find(&sha256(text.as_bytes())).unwrap();
        warehouse.vector(record, &mut vector);
        let (shard, at) = if index < 2 {
            (&shards[0], index)
        } else {
            (&shards[1], index - 2)
        };
        assert_eq!(vector, vector_of(shard, at), "{index}");
    }
    assert!(warehouse.find(&sha256(b"not embedded")).is_none());

    let again = warehouse
        .add_shards(Some(&fixture.plan), &shards, &POLICY, "later".into())
        .unwrap();
    assert_eq!((again.added, again.held, again.total), (0, 4, 4));
    drop(warehouse);

    let reopened = Warehouse::open(&at).unwrap();
    assert_eq!(reopened.len(), 4);
    assert_eq!(reopened.manifest().batches.len(), 2);
    let model_path = fixture.dir.join("model").join("model.onnx");
    let warm = fixture.dir.join("warm");
    let counts = plan_from_corpus(
        &corpus(
            fixture.dir.path(),
            "warm",
            1,
            &[
                (1, "otzaria/a.txt", TEXTS[0]),
                (2, "otzaria/a.txt", "שורה חדשה שעוד לא הוטמעה בשום מקום"),
            ],
        ),
        PlanRequest {
            out_dir: warm,
            model: fixture.model.clone(),
            chunking: ChunkerConfig::default(),
            passage_package: passage_package(&model_path),
            previous: None,
            warehouse: Some(&reopened),
            created_at: "2026-10-02T00:00:00Z".to_string(),
        },
    )
    .unwrap()
    .counts;
    assert_eq!((counts.to_embed, counts.revived), (1, 1));
}

/// What a crash leaves — data past the count, an index that is not the count's — is
/// repaired by the next add; a shard of another package is refused; and an import
/// without the plan is checked for everything but the plan's own keys.
#[test]
fn a_warehouse_repairs_a_crash_refuses_another_package_and_imports_without_a_plan() {
    let fixture = fixture("warehouse_crash", &TEXTS);
    let shards = [fixture.shard("all", 0, 9)];
    let at = fixture.warehouse();
    Warehouse::open_for_append(&at)
        .unwrap()
        .add_shards(Some(&fixture.plan), &shards[..], &POLICY, "now".into())
        .unwrap();

    // A crash after the data was appended and the index written, before the count.
    for name in ["vectors.f32", "keys.bin"] {
        let mut bytes = std::fs::read(at.join(name)).unwrap();
        bytes.extend_from_slice(&[7u8; 96]);
        std::fs::write(at.join(name), bytes).unwrap();
    }
    std::fs::write(at.join("index.bin"), b"OXVWIDX1\x09\0\0\0\0\0\0\0").unwrap();
    assert!(
        Warehouse::open(&at).is_err(),
        "a reader names the index it cannot use"
    );
    let mut repaired = Warehouse::open_for_append(&at).unwrap();
    assert_eq!(repaired.len(), 4);
    assert!(repaired.find(&sha256(TEXTS[3].as_bytes())).is_some());

    // Another package of the same family.
    let other = fixture.dir.join("other");
    std::fs::create_dir_all(&other).unwrap();
    for name in [VECTORS_FILE, KEYS_FILE, SHARD_MANIFEST_FILE] {
        std::fs::copy(shards[0].join(name), other.join(name)).unwrap();
    }
    let mut manifest = ShardManifest::read(&other).unwrap();
    manifest.passage_package = fixture.model.query_packages[0].clone();
    std::fs::write(
        other.join(SHARD_MANIFEST_FILE),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    let refused = repaired
        .add_shards(None, &[other], &POLICY, "now".into())
        .unwrap_err()
        .to_string();
    assert!(refused.contains("one package"), "{refused}");
    assert_eq!(repaired.len(), 4, "a refused batch adds nothing");

    // An import: a whole worker's output in one shard, certified, without its plan.
    let more = fixture_texts_shard(&fixture);
    let imported = repaired
        .add_shards(None, &[more], &ShardPolicy::default(), "now".into())
        .unwrap();
    assert_eq!((imported.added, imported.total), (1, 5));
    assert_eq!(
        repaired.manifest().batches.last().unwrap().plan_sha256,
        None
    );
}

/// One more text, embedded and relabelled as a certified GPU worker's output.
fn fixture_texts_shard(fixture: &Fixture) -> PathBuf {
    let extra = fixture_with(&fixture.dir, "אשרי יושבי ביתך עוד יהללוך סלה אשרי העם");
    let mut manifest = ShardManifest::read(&extra).unwrap();
    manifest.worker = WorkerInfo {
        name: "torch_bert".into(),
        version: "1".into(),
        device: "AMD Radeon RX 9060 XT".into(),
        ep: "rocm".into(),
        mode: "torch".into(),
    };
    manifest.parity = Some(ParityCertificate {
        reference: "onnxruntime 1.28.0 cpu fp32".into(),
        samples: 20_480,
        min_cosine: 0.99999969,
        mean_cosine: 0.99999986,
        document_sha256: None,
    });
    std::fs::write(
        extra.join(SHARD_MANIFEST_FILE),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    extra
}

/// A shard of `text` alone, from its own plan beside `dir`.
fn fixture_with(dir: &TempDir, text: &str) -> PathBuf {
    let model_path = dir.join("model").join("model.onnx");
    let plan = dir.join("plan-extra");
    plan_from_corpus(
        &corpus(dir.path(), "extra", 1, &[(1, "otzaria/z.txt", text)]),
        PlanRequest {
            out_dir: plan.clone(),
            model: family(&model_path),
            chunking: ChunkerConfig::default(),
            passage_package: passage_package(&model_path),
            previous: None,
            warehouse: None,
            created_at: "2026-10-02T00:00:00Z".to_string(),
        },
    )
    .unwrap();
    let mut runtime = EmbeddingRuntime::new(EmbeddingConfig {
        model_path,
        embedding_dim: DIM,
        max_tokens: 512,
        batch_size: 2,
        pooling: Pooling::InGraph,
    });
    runtime.load().unwrap();
    let out = dir.join("extra-shard");
    let worker = WorkerInfo {
        name: "x".into(),
        version: "0".into(),
        device: "cpu".into(),
        ep: "cpu".into(),
        mode: MODE_MOCK.into(),
    };
    embed_shard(
        &plan,
        &EmbedManifest::read(&plan).unwrap(),
        0,
        9,
        &runtime,
        2,
        worker,
        &out,
    )
    .unwrap();
    out
}

/// A warehouse of `TEXTS` in two batches, records 0..2 and 2..4, verified sound.
fn two_batches(name: &str) -> (TempDir, PathBuf) {
    let dir = TempDir::new(name);
    let model_path = stub_model(dir.path());
    let (model, package) = (family(&model_path), passage_package(&model_path));
    let at = dir.join("warehouse");
    Warehouse::create(&at, WarehouseIdentity::of(&model, &package)).unwrap();
    for (n, texts) in TEXTS.chunks(2).enumerate() {
        let lines: Vec<(u64, &str, &str)> = texts
            .iter()
            .enumerate()
            .map(|(line, text)| (line as u64 + 1, "otzaria/a.txt", *text))
            .collect();
        let plan = dir.join(&format!("plan{n}"));
        plan_from_corpus(
            &corpus(dir.path(), &format!("c{n}"), 1, &lines),
            PlanRequest {
                out_dir: plan.clone(),
                model: model.clone(),
                chunking: ChunkerConfig::default(),
                passage_package: package.clone(),
                previous: None,
                warehouse: None,
                created_at: "2026-10-02T00:00:00Z".to_string(),
            },
        )
        .unwrap();
        let shard = raw_shard(&plan, &dir.join(&format!("shard{n}")));
        Warehouse::open_for_append(&at)
            .unwrap()
            .add_shards(Some(&plan), &[shard], &POLICY, "now".into())
            .unwrap();
    }
    let warehouse = Warehouse::open(&at).unwrap();
    let verified = warehouse.verify().unwrap();
    assert_eq!(
        (verified.records, verified.batches, verified.bytes),
        (4, 2, 4 * (DIM as u64 * 4 + 32))
    );
    for (record, text) in TEXTS.iter().enumerate() {
        assert_eq!(
            warehouse.find(&sha256(text.as_bytes())),
            Some(record as u64)
        );
    }
    (dir, at)
}

fn flip(path: &Path, at: usize, mask: u8) {
    let mut bytes = std::fs::read(path).unwrap();
    bytes[at] ^= mask;
    std::fs::write(path, bytes).unwrap();
}

/// Every file of the warehouse, to show that a refusal changed none.
fn files(at: &Path) -> Vec<Vec<u8>> {
    ["warehouse.json", "vectors.f32", "keys.bin", "index.bin"]
        .iter()
        .map(|name| std::fs::read(at.join(name)).unwrap())
        .collect()
}

/// Where `text`'s entry starts in `index.bin`.
fn entry_of(at: &Path, text: &str) -> usize {
    let index = std::fs::read(at.join("index.bin")).unwrap();
    let key = sha256(text.as_bytes());
    (16..index.len())
        .step_by(40)
        .find(|&entry| index[entry..entry + 32] == key)
        .unwrap()
}

fn prefix(text: &str) -> [u8; 16] {
    sha256(text.as_bytes())[..16].try_into().unwrap()
}

/// The audit's first case, a flipped vector bit: open reads it, but verify and an append
/// refuse it, naming the batch, and change nothing.
#[test]
fn a_flipped_vector_bit_is_refused_naming_its_batch() {
    let (_dir, at) = two_batches("warehouse_vector_bit");
    flip(&at.join("vectors.f32"), 3 * DIM as usize * 4 + 5, 0x40);
    let before = files(&at);

    let warehouse = Warehouse::open(&at).unwrap();
    let error = warehouse.verify().unwrap_err().to_string();
    assert!(
        error.contains("batch 1 (records 2..4, added now): vectors.f32 hashes to"),
        "{error}"
    );
    assert!(
        !error.contains("batch 0") && !error.contains("keys.bin hashes"),
        "{error}"
    );
    assert!(
        error.contains("not repaired") && error.contains("move it aside"),
        "{error}"
    );
    drop(warehouse);

    let error = Warehouse::open_for_append(&at).err().unwrap().to_string();
    assert!(error.contains("batch 1 (records 2..4"), "{error}");
    assert_eq!(files(&at), before, "a refused warehouse is left as it was");
}

/// The audit's second case, a pointer to another text's record: lookups find nothing,
/// verify names the entry, and an append rebuilds the index from the verified keys.
#[test]
fn an_index_pointing_at_another_record_finds_nothing_until_an_append_rebuilds_it() {
    let (_dir, at) = two_batches("warehouse_index_pointer");
    let entry = entry_of(&at, TEXTS[1]);
    flip(&at.join("index.bin"), entry + 32, 1);
    let before = files(&at);

    let warehouse = Warehouse::open(&at).unwrap();
    let key = sha256(TEXTS[1].as_bytes());
    assert_eq!(warehouse.find(&key), None);
    assert_eq!(warehouse.find_key(&prefix(TEXTS[1])), None);
    assert!(!warehouse.holds(&key));
    assert_eq!(warehouse.find(&sha256(TEXTS[0].as_bytes())), Some(0));
    let error = warehouse.verify().unwrap_err().to_string();
    assert!(
        error.contains("index.bin is not the index of keys.bin")
            && error.contains("at record 0, which holds key"),
        "{error}"
    );
    assert!(error.contains("rebuilds the index"), "{error}");
    drop(warehouse);
    assert_eq!(files(&at), before, "reading repairs nothing");

    let repaired = Warehouse::open_for_append(&at).unwrap();
    let rebuilt = repaired.verify().unwrap().index_rebuilt.unwrap();
    assert!(rebuilt.contains("at record 0"), "{rebuilt}");
    assert_eq!(repaired.find(&key), Some(1));
    drop(repaired);
    assert_eq!(files(&at)[..3], before[..3], "only the index is rewritten");

    let reopened = Warehouse::open(&at).unwrap();
    assert_eq!(reopened.verify().unwrap().index_rebuilt, None);
    assert_eq!(reopened.find_key(&prefix(TEXTS[1])), Some(1));
}

/// A flipped byte of an index entry's key: the text is no longer found, the entry finds
/// nothing either, and an append rebuilds the index.
#[test]
fn a_flipped_index_key_byte_is_caught_and_rebuilt() {
    let (_dir, at) = two_batches("warehouse_index_key");
    let entry = entry_of(&at, TEXTS[2]);
    flip(&at.join("index.bin"), entry + 31, 0x01);
    let mut flipped = sha256(TEXTS[2].as_bytes());
    flipped[31] ^= 0x01;

    let warehouse = Warehouse::open(&at).unwrap();
    assert_eq!(warehouse.find(&sha256(TEXTS[2].as_bytes())), None);
    assert_eq!(warehouse.find(&flipped), None);
    let error = warehouse.verify().unwrap_err().to_string();
    assert!(
        error.contains("index.bin is not the index of keys.bin"),
        "{error}"
    );
    drop(warehouse);

    let repaired = Warehouse::open_for_append(&at).unwrap();
    assert!(repaired.verify().unwrap().index_rebuilt.is_some());
    assert_eq!(repaired.find(&sha256(TEXTS[2].as_bytes())), Some(2));
}

/// A flipped byte of keys.bin fails its batch's digest. The index cannot be rebuilt from
/// keys that are not sound, so an append refuses and changes nothing.
#[test]
fn a_flipped_key_is_refused_and_no_index_is_rebuilt_from_it() {
    let (_dir, at) = two_batches("warehouse_key_byte");
    flip(&at.join("keys.bin"), 7, 0x10);
    let before = files(&at);

    let warehouse = Warehouse::open(&at).unwrap();
    assert_eq!(warehouse.find(&sha256(TEXTS[0].as_bytes())), None);
    let error = warehouse.verify().unwrap_err().to_string();
    assert!(
        error.contains("batch 0 (records 0..2, added now): keys.bin hashes to"),
        "{error}"
    );
    assert!(error.contains("not repaired"), "{error}");
    drop(warehouse);

    let error = Warehouse::open_for_append(&at).err().unwrap().to_string();
    assert!(error.contains("keys.bin hashes to"), "{error}");
    assert_eq!(files(&at), before);
}

/// A batch that added nothing has the digests of nothing: any other is warehouse.json's fault.
#[test]
fn an_empty_batch_with_another_digest_names_warehouse_json() {
    let (_dir, at) = two_batches("warehouse_empty_batch");
    let path = at.join("warehouse.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let mut empty = manifest["batches"][1].clone();
    empty["first"] = 4.into();
    empty["records"] = 0.into();
    empty["vectors_sha256"] = "0".repeat(64).into();
    empty["keys_sha256"] = super::files::hex(&sha256(b"")).into();
    manifest["batches"].as_array_mut().unwrap().push(empty);
    std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();

    let error = Warehouse::open(&at)
        .unwrap()
        .verify()
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("batch 2 (records 4..4, added now): it added no bytes, so warehouse.json's")
            && error.contains("for vectors.f32 is corrupt"),
        "{error}"
    );
    assert!(!error.contains("keys.bin"), "{error}");
    assert!(
        error.contains("restore warehouse.json")
            && error.contains("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
            && !error.contains("move it aside"),
        "{error}"
    );
}

/// Between an add's index and its count, a reader is told the warehouse may be being added
/// to; once the count is written, it opens.
#[test]
fn a_reader_between_an_adds_index_and_its_count_is_told_so() {
    let (_dir, at) = two_batches("warehouse_mid_add");
    let path = at.join("warehouse.json");
    let committed = std::fs::read(&path).unwrap();
    let mut before: serde_json::Value = serde_json::from_slice(&committed).unwrap();
    before["records"] = 2.into();
    before["batches"].as_array_mut().unwrap().pop();
    std::fs::write(&path, serde_json::to_vec(&before).unwrap()).unwrap();

    let started = std::time::Instant::now();
    let error = Warehouse::open(&at).err().unwrap().to_string();
    let waited = started.elapsed().as_secs_f64();
    assert!(error.contains("may be being added to"), "{error}");
    assert!((1.0..2.0).contains(&waited), "waited {waited} s");
    std::fs::write(&path, committed).unwrap();
    assert_eq!(Warehouse::open(&at).unwrap().len(), 4);
}

/// An index header no add leaves, behind the count or without its magic, is refused at
/// once, with the repair alone.
#[test]
fn an_index_header_no_add_leaves_is_refused_at_once() {
    let (_dir, at) = two_batches("warehouse_index_behind");
    let original = std::fs::read(at.join("index.bin")).unwrap();
    for (case, at_byte, value) in [("behind", 8, 3u8), ("no magic", 0, b'X')] {
        let mut index = original.clone();
        index[at_byte] = value;
        std::fs::write(at.join("index.bin"), index).unwrap();
        let started = std::time::Instant::now();
        let error = Warehouse::open(&at).err().unwrap().to_string();
        assert!(started.elapsed().as_secs_f64() < 0.5, "{case}: it waited");
        assert!(
            error.contains("warehouse-verify --repair, rebuilds it")
                && !error.contains("being added to"),
            "{case}: {error}"
        );
    }
}

/// Batches that do not tile the count, in order and without a gap, are refused on open.
#[test]
fn batches_that_do_not_tile_the_count_are_refused() {
    let (_dir, at) = two_batches("warehouse_tiling");
    let path = at.join("warehouse.json");
    let original: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let refused = |case: &str, edit: &dyn Fn(&mut serde_json::Value), expected: &str| {
        let mut manifest = original.clone();
        edit(&mut manifest);
        std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        let before = files(&at);
        for error in [
            Warehouse::open(&at).err(),
            Warehouse::open_for_append(&at).err(),
        ] {
            let error = error.unwrap().to_string();
            assert!(error.contains(expected), "{case}: {error}");
        }
        assert_eq!(files(&at), before, "{case}: nothing is truncated");
    };
    refused(
        "a gap",
        &|manifest| manifest["batches"][1]["first"] = 3.into(),
        "batch 1 starts at record 3, and the batches before it end at 2",
    );
    refused(
        "an overlap",
        &|manifest| manifest["batches"][0]["records"] = 3.into(),
        "batch 1 starts at record 2, and the batches before it end at 3",
    );
    refused(
        "a count past the batches",
        &|manifest| manifest["records"] = 5.into(),
        "the batches cover records 0..4, and it counts 5",
    );
}
