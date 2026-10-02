use super::files::sha256;
use super::plan::{plan_from_corpus, EmbedManifest, PlanRequest};
use super::shard::*;
use super::testing::{corpus, family, passage_package, stub_model, TempDir, DIM};
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
