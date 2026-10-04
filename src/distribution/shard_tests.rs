use super::files::{hex, sha256};
use super::plan::{plan_from_corpus, EmbedManifest, PlanRequest, EMBED_PLAN_FILE};
use super::shard::*;
use super::testing::{corpus, family, passage_package, stub_model, TempDir, DIM};
use crate::errors::PackError;
use crate::semantic::backend::Pooling;
use crate::semantic::chunker::ChunkerConfig;
use crate::semantic::embedding::{EmbeddingConfig, EmbeddingRuntime};
use std::path::{Path, PathBuf};

const TEXTS: [&str; 5] = [
    "בראשית ברא אלהים את השמים ואת הארץ",
    "והארץ היתה תהו ובהו וחשך על פני תהום רבה",
    "ויאמר אלהים יהי אור ויהי אור וירא אלהים",
    "מאימתי קורין את שמע בערבית משעה שהכהנים",
    "אשרי יושבי ביתך עוד יהללוך סלה אשרי העם",
];

/// A plan of the five texts, the stand-in's runtime, and the directory they live in.
fn fixture(name: &str) -> (TempDir, PathBuf, EmbedManifest, EmbeddingRuntime) {
    let dir = TempDir::new(name);
    let model_path = stub_model(dir.path());
    let lines: Vec<(u64, &str, &str)> = TEXTS
        .iter()
        .enumerate()
        .map(|(index, text)| (index as u64 + 1, "otzaria/a.txt", *text))
        .collect();
    let plan_dir = dir.join("plan");
    plan_from_corpus(
        &corpus(dir.path(), "v1", 1, &lines),
        PlanRequest {
            out_dir: plan_dir.clone(),
            model: family(&model_path),
            chunking: ChunkerConfig::default(),
            passage_package: passage_package(&model_path),
            previous: None,
            warehouse: None,
            created_at: "2026-10-02T00:00:00Z".to_string(),
        },
    )
    .unwrap();
    let manifest = EmbedManifest::read(&plan_dir).unwrap();
    let mut runtime = EmbeddingRuntime::new(EmbeddingConfig {
        model_path,
        embedding_dim: DIM,
        max_tokens: 512,
        batch_size: 2,
        pooling: Pooling::InGraph,
    });
    runtime.load().unwrap();
    (dir, plan_dir, manifest, runtime)
}

fn worker() -> WorkerInfo {
    WorkerInfo {
        name: "test".to_string(),
        version: "0".to_string(),
        device: "cpu".to_string(),
        ep: "cpu".to_string(),
        mode: MODE_MOCK.to_string(),
    }
}

const POLICY: ShardPolicy = ShardPolicy {
    allow_non_semantic: true,
};

fn embed(
    plan_dir: &Path,
    plan: &EmbedManifest,
    runtime: &EmbeddingRuntime,
    out: &Path,
    windows: &[(u64, u64)],
) -> Vec<PathBuf> {
    windows
        .iter()
        .enumerate()
        .map(|(index, (skip, take))| {
            let dir = out.join(format!("shard-{index}"));
            embed_shard(plan_dir, plan, *skip, *take, runtime, 2, worker(), &dir).unwrap();
            dir
        })
        .collect()
}

fn rewrite_manifest(dir: &Path, edit: impl FnOnce(&mut ShardManifest)) {
    let mut manifest = ShardManifest::read(dir).unwrap();
    edit(&mut manifest);
    std::fs::write(
        dir.join(SHARD_MANIFEST_FILE),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
}

fn refusal(result: Result<Vec<CheckedShard>, PackError>) -> String {
    match result {
        Err(error) => error.to_string(),
        Ok(_) => panic!("expected a refusal"),
    }
}

/// Where a window boundary falls changes no byte: three windows and one hold the same
/// vectors and keys, and both verify against the plan.
#[test]
fn a_plan_embedded_in_windows_verifies_and_matches_one_window() {
    let (dir, plan_dir, plan, runtime) = fixture("shard_windows");
    let three = embed(
        &plan_dir,
        &plan,
        &runtime,
        &dir.join("three"),
        &[(0, 2), (2, 2), (4, 9)],
    );
    let one = embed(
        &plan_dir,
        &plan,
        &runtime,
        &dir.join("one"),
        &[(0, u64::MAX)],
    );
    let checked = verify_shards(Some(&plan_dir), &three, &POLICY).unwrap();
    assert_eq!(
        checked
            .iter()
            .map(|shard| shard.manifest.records)
            .collect::<Vec<_>>(),
        [2, 2, 1]
    );
    verify_shards(Some(&plan_dir), &one, &POLICY).unwrap();
    // Out of order is the same set.
    verify_shards(
        Some(&plan_dir),
        &[three[2].clone(), three[0].clone(), three[1].clone()],
        &POLICY,
    )
    .unwrap();

    let concatenated = |dirs: &[PathBuf], file: &str| -> Vec<u8> {
        dirs.iter()
            .flat_map(|dir| std::fs::read(dir.join(file)).unwrap())
            .collect()
    };
    assert_eq!(
        concatenated(&three, VECTORS_FILE),
        concatenated(&one, VECTORS_FILE)
    );
    let keys = concatenated(&one, KEYS_FILE);
    assert_eq!(keys.len(), 5 * 32);
    assert_eq!(
        &keys[..32],
        sha256(TEXTS[0].as_bytes()),
        "keys are full digests, in plan order"
    );

    // Without the plan — an import — the same shards verify.
    verify_shards(None, &three, &POLICY).unwrap();

    // A finished shard is not written over.
    match embed_shard(&plan_dir, &plan, 0, 2, &runtime, 2, worker(), &three[0]) {
        Err(PackError::UnusableOutput { .. }) => {}
        other => panic!("a finished shard must be refused, got {other:?}"),
    }
}

#[test]
fn acceptance_refuses_holes_overlaps_other_plans_and_damage() {
    let (dir, plan_dir, plan, runtime) = fixture("shard_refusals");
    let shards = embed(
        &plan_dir,
        &plan,
        &runtime,
        &dir.join("s"),
        &[(0, 2), (2, 3)],
    );

    assert!(refusal(verify_shards(Some(&plan_dir), &shards[..1], &POLICY)).contains("cover 2"));
    assert!(refusal(verify_shards(None, &shards[1..], &POLICY)).contains("covered by no shard"));
    let again = embed(&plan_dir, &plan, &runtime, &dir.join("again"), &[(0, 2)]);
    assert!(refusal(verify_shards(
        Some(&plan_dir),
        &[shards[0].clone(), again[0].clone(), shards[1].clone()],
        &POLICY
    ))
    .contains("covered already"));

    // A shard of another plan, with a window of exactly the right shape.
    let other = dir.join("other");
    copy_dir(&shards[1], &other);
    rewrite_manifest(&other, |manifest| manifest.plan_sha256 = "0".repeat(64));
    assert!(refusal(verify_shards(
        Some(&plan_dir),
        &[shards[0].clone(), other],
        &POLICY
    ))
    .contains("another plan"));

    // A flipped bit in a vector: the digest says so.
    let flipped = dir.join("flipped");
    copy_dir(&shards[1], &flipped);
    let mut bytes = std::fs::read(flipped.join(VECTORS_FILE)).unwrap();
    bytes[5] ^= 1;
    std::fs::write(flipped.join(VECTORS_FILE), &bytes).unwrap();
    assert!(refusal(verify_shards(
        Some(&plan_dir),
        &[shards[0].clone(), flipped.clone()],
        &POLICY
    ))
    .contains("hashes to"));

    // The same vector scaled, its digest restamped: not a unit vector.
    let scaled = dir.join("scaled");
    copy_dir(&shards[1], &scaled);
    let mut bytes = std::fs::read(scaled.join(VECTORS_FILE)).unwrap();
    for value in bytes[..DIM as usize * 4].as_chunks_mut::<4>().0 {
        *value = (f32::from_le_bytes(*value) * 2.0).to_le_bytes();
    }
    std::fs::write(scaled.join(VECTORS_FILE), &bytes).unwrap();
    rewrite_manifest(&scaled, |manifest| {
        manifest.vectors_sha256 = hex(&sha256(&bytes))
    });
    assert!(refusal(verify_shards(None, &[shards[0].clone(), scaled], &POLICY)).contains("norm"));

    // Two keys swapped, the digest restamped: only the plan can tell.
    let swapped = dir.join("swapped");
    copy_dir(&shards[1], &swapped);
    let mut keys = std::fs::read(swapped.join(KEYS_FILE)).unwrap();
    let (first, second) = keys.split_at_mut(32);
    first.swap_with_slice(&mut second[..32]);
    std::fs::write(swapped.join(KEYS_FILE), &keys).unwrap();
    rewrite_manifest(&swapped, |manifest| {
        manifest.keys_sha256 = hex(&sha256(&keys))
    });
    verify_shards(None, &[shards[0].clone(), swapped.clone()], &POLICY).unwrap();
    assert!(refusal(verify_shards(
        Some(&plan_dir),
        &[shards[0].clone(), swapped],
        &POLICY
    ))
    .contains("keyed"));

    // A plan whose text changed after it was hashed is refused before it is embedded.
    let text = std::fs::read_to_string(plan_dir.join(EMBED_PLAN_FILE)).unwrap();
    std::fs::write(
        plan_dir.join(EMBED_PLAN_FILE),
        text.replacen("בראשית", "בראשיה", 1),
    )
    .unwrap();
    match embed_shard(
        &plan_dir,
        &plan,
        0,
        1,
        &runtime,
        2,
        worker(),
        &dir.join("edited"),
    ) {
        Err(PackError::PlanTextChanged { record: 0, .. }) => {}
        other => panic!("an edited text must be refused, got {other:?}"),
    }
}

/// A worker that is not ONNX Runtime on a CPU is accepted on its parity certificate, and
/// only on a good one; the stand-in only where a test says so.
#[test]
fn a_worker_that_is_not_the_reference_needs_a_parity_certificate() {
    let (dir, plan_dir, plan, runtime) = fixture("shard_parity");
    let shards = embed(&plan_dir, &plan, &runtime, &dir.join("s"), &[(0, u64::MAX)]);
    let check = |edit: &dyn Fn(&mut ShardManifest)| {
        let at = dir.join(&format!("edit-{}", rand_suffix()));
        copy_dir(&shards[0], &at);
        rewrite_manifest(&at, edit);
        verify_shards(Some(&plan_dir), &[at], &ShardPolicy::default()).map(|_| ())
    };
    let gpu = |manifest: &mut ShardManifest| {
        manifest.worker = WorkerInfo {
            name: "seforim-gpu-worker".to_string(),
            version: "1".to_string(),
            device: "NVIDIA RTX".to_string(),
            ep: "cuda".to_string(),
            mode: "torch".to_string(),
        };
    };
    let certificate = |min_cosine: f64, samples: u64| ParityCertificate {
        reference: "onnxruntime 1.28.0 cpu fp32".to_string(),
        samples,
        min_cosine,
        mean_cosine: 0.99999995,
        document_sha256: None,
    };

    assert!(
        check(&|_| {}).is_err(),
        "the stand-in is refused by default"
    );
    assert!(check(&|manifest| gpu(manifest)).is_err(), "no certificate");
    check(&|manifest| {
        gpu(manifest);
        manifest.parity = Some(certificate(0.9999997, 1_000));
    })
    .unwrap();
    assert!(check(&|manifest| {
        gpu(manifest);
        manifest.parity = Some(certificate(0.99, 1_000));
    })
    .is_err());
    assert!(check(&|manifest| {
        gpu(manifest);
        manifest.parity = Some(certificate(0.9999997, 10));
    })
    .is_err());
    check(&|manifest| {
        manifest.worker.mode = MODE_ONNXRUNTIME.to_string();
    })
    .unwrap();
}

fn rand_suffix() -> u64 {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), to.join(entry.file_name())).unwrap();
    }
}
