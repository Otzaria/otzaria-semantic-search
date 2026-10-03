//! The vector build, end to end through the CLI: plan, an external worker's shard adopted,
//! the warehouse, a base and a delta assembled and verified, and the release's files.

use otzaria_semantic_search::distribution::corpus::{CorpusIdentity, CorpusLine, CorpusLineRecord};
use otzaria_semantic_search::semantic::chunker::ChunkerConfig;
use otzaria_semantic_search::semantic::versioning::{ModelIdentity, ModelPackage, TextIdentity};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::process::Command;

const BIN: &str = env!("CARGO_BIN_EXE_otzaria-semantic-search");
const DIM: usize = 64;
const AT: &str = "2026-10-02T00:00:00Z";

fn run(args: &[&str]) -> (i32, String) {
    let output = Command::new(BIN).args(args).output().unwrap();
    let text = String::from_utf8_lossy(&output.stdout).to_string()
        + &String::from_utf8_lossy(&output.stderr);
    (output.status.code().unwrap_or(-1), text)
}

fn ok(args: &[&str]) -> String {
    let (code, text) = run(args);
    assert_eq!(code, 0, "{args:?}:\n{text}");
    text
}

/// Text `i`: eight Hebrew words, the first spelling `i`.
fn text(i: usize) -> String {
    let letter = |n: usize| char::from_u32(0x05D0 + (n % 22) as u32).unwrap();
    let mut words = vec![(0..4u32)
        .map(|d| letter(i / 22usize.pow(d) + d as usize))
        .collect()];
    words.extend((1..8).map(|w| (0..5).map(|k| letter(w * 5 + k)).collect::<String>()));
    words.join(" ")
}

fn unit_vector(seed: &[u8]) -> Vec<f32> {
    let mut state = u64::from_le_bytes(seed[..8].try_into().unwrap()) | 1;
    let mut vector: Vec<f32> = (0..DIM)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        })
        .collect();
    let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    vector.iter_mut().for_each(|x| *x /= norm);
    vector
}

struct Work(PathBuf);

impl Work {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }

    fn path(&self, name: &str) -> String {
        self.0.join(name).to_string_lossy().to_string()
    }
}

impl Drop for Work {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Library `version`'s corpus: `(book, [text])`.
fn corpus(work: &Work, version: u32, books: &[(&str, &[usize])]) -> (String, String) {
    let identity = work.path(&format!("v{version}-identity.json"));
    std::fs::write(
        &identity,
        serde_json::to_vec(&CorpusIdentity {
            text: TextIdentity::with_line_text_version(1),
            library_version: version,
            library_release_tag: format!("v{version}-20260930120000"),
            document_id_scheme_version: 1,
        })
        .unwrap(),
    )
    .unwrap();
    let mut lines = String::new();
    let mut line_id = 0;
    for (book, texts) in books {
        for i in *texts {
            line_id += 1;
            let line = CorpusLine {
                source_book_key: book.to_string(),
                title: String::new(),
                reference: String::new(),
                section_id: 1,
                segment: 0,
                is_pdf: false,
                line_hash: 0,
                content_hash: 1,
                facets: Vec::new(),
                text: text(*i),
            };
            lines += &serde_json::to_string(&CorpusLineRecord { line_id, line }).unwrap();
            lines.push('\n');
        }
    }
    let path = work.path(&format!("v{version}-lines.jsonl"));
    std::fs::write(&path, lines).unwrap();
    (identity, path)
}

/// What an external worker writes for a plan's embed.jsonl: vectors.f32 and keys.bin.
fn worker_output(plan: &str, out: &Path) -> String {
    std::fs::create_dir_all(out).unwrap();
    let (mut vectors, mut keys) = (Vec::new(), Vec::new());
    for line in std::fs::read_to_string(Path::new(plan).join("embed.jsonl"))
        .unwrap()
        .lines()
    {
        let record: serde_json::Value = serde_json::from_str(line).unwrap();
        let sha = Sha256::digest(record["embedding_text"].as_str().unwrap().as_bytes());
        keys.extend_from_slice(&sha);
        for value in unit_vector(&sha) {
            vectors.extend_from_slice(&value.to_le_bytes());
        }
    }
    std::fs::write(out.join("vectors.f32"), vectors).unwrap();
    std::fs::write(out.join("keys.bin"), keys).unwrap();
    let manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(Path::new(plan).join("embed-manifest.json")).unwrap(),
    )
    .unwrap();
    manifest["plan_sha256"].as_str().unwrap().to_string()
}

fn gates(release: &str) -> Vec<(String, bool)> {
    let report: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(Path::new(release).join("gates.json")).unwrap(),
    )
    .unwrap();
    report["gates"]
        .as_array()
        .unwrap()
        .iter()
        .map(|gate| {
            (
                gate["gate"].as_str().unwrap().to_string(),
                gate["passed"].as_bool().unwrap(),
            )
        })
        .collect()
}

struct Build {
    work: Work,
    model: String,
    chunking: String,
    warehouse: String,
}

impl Build {
    fn new(name: &str) -> Self {
        let work = Work::new(name);
        let chunking = work.path("chunking.json");
        std::fs::write(
            &chunking,
            serde_json::to_vec(&ChunkerConfig::default()).unwrap(),
        )
        .unwrap();
        let model = work.path("model.json");
        let identity = ModelIdentity {
            family_id: "otzaria-cli-family".to_string(),
            tokenizer_checksum: "b".repeat(64),
            embedding_dim: DIM as u32,
            pooling: "in-graph".to_string(),
            max_tokens: 512,
            embedding_text_version: 1,
            normalization_version: 1,
            chunking_identity: ChunkerConfig::default().identity(),
            query_packages: vec![
                ModelPackage {
                    checksum: "a".repeat(64),
                    quantization: "int8".to_string(),
                },
                ModelPackage {
                    checksum: "c".repeat(64),
                    quantization: "fp32".to_string(),
                },
            ],
        };
        std::fs::write(&model, serde_json::to_vec(&identity).unwrap()).unwrap();
        let warehouse = work.path("warehouse");
        Self {
            work,
            model,
            chunking,
            warehouse,
        }
    }

    /// Plan `version`, then embed and add to the warehouse whatever the plan lists.
    fn step(&self, version: u32, books: &[(&str, &[usize])], previous: Option<&str>) -> String {
        let work = &self.work;
        let (corpus_identity, corpus_lines) = corpus(work, version, books);
        let plan = work.path(&format!("plan-v{version}"));
        let mut args: Vec<&str> = vec![
            "plan",
            "--corpus-identity",
            &corpus_identity,
            "--corpus-lines",
            &corpus_lines,
            "--model",
            &self.model,
            "--chunking",
            &self.chunking,
            "--out",
            &plan,
            "--created-at",
            AT,
        ];
        if let Some(previous) = previous {
            args.extend([
                "--previous-ledger",
                previous,
                "--warehouse",
                self.warehouse.as_str(),
            ]);
        }
        ok(&args);
        if std::fs::metadata(Path::new(&plan).join("embed.jsonl"))
            .unwrap()
            .len()
            == 0
        {
            return plan;
        }
        let shard = work.0.join(format!("shard-v{version}"));
        let plan_sha256 = worker_output(&plan, &shard);
        let shard = shard.to_string_lossy().to_string();
        ok(&[
            "adopt-shard",
            "--dir",
            &shard,
            "--model",
            &self.model,
            "--plan-sha256",
            &plan_sha256,
            "--worker-name",
            "cli-test",
            "--worker-version",
            "1",
            "--device",
            "cpu",
            "--ep",
            "cpu",
            "--mode",
            "mock",
        ]);
        ok(&[
            "warehouse-add",
            "--warehouse",
            &self.warehouse,
            "--create",
            "--model",
            &self.model,
            "--plan",
            &plan,
            "--shards",
            &shard,
            "--allow-non-semantic",
        ]);
        plan
    }
}

#[test]
fn a_base_and_a_delta_round_trip_through_the_cli() {
    let build = Build::new("vector-build-cli");
    let (work, warehouse) = (&build.work, &build.warehouse);
    let step = |version, books: &[(&str, &[usize])], previous| build.step(version, books, previous);

    let plan1 = step(
        1,
        &[("otzaria/a.txt", &[1, 2, 3]), ("otzaria/b.txt", &[4, 1])],
        None,
    );
    let base = work.path("base-v1");
    let text = ok(&[
        "assemble",
        "--kind",
        "base",
        "--plan",
        &plan1,
        "--warehouse",
        warehouse,
        "--out",
        &base,
        "--created-at",
        AT,
        "--built-by",
        r#"{"runId":"cli"}"#,
        "--verify",
    ]);
    assert!(text.contains("4 slot(s), 1 extra(s)"), "{text}");
    assert!(gates(&base).iter().all(|(_, passed)| *passed));
    // The release already in --out, verified again.
    ok(&[
        "assemble",
        "--verify",
        "--plan",
        &plan1,
        "--warehouse",
        warehouse,
        "--out",
        &base,
    ]);

    let plan2 = step(
        2,
        &[("otzaria/a.txt", &[1, 3, 5]), ("otzaria/b.txt", &[4, 1, 2])],
        Some(&base),
    );
    let delta = work.path("delta-v2");
    let (code, text) = run(&[
        "assemble",
        "--kind",
        "delta",
        "--plan",
        &plan2,
        "--warehouse",
        warehouse,
        "--previous",
        &base,
        "--out",
        &delta,
        "--created-at",
        AT,
        "--verify",
    ]);
    assert!(
        text.contains("1 slot(s), 0 extra(s), 1 foreign, 0 tombstone(s)"),
        "{text}"
    );
    // A delta this small is large next to its base: only the size gate may say so.
    for (gate, passed) in gates(&delta) {
        assert!(passed || gate == "G7", "{gate}: {text}");
    }
    assert!(code == 0 || code == 2, "{text}");

    let part = work.path("delta.oxv.zst");
    std::fs::write(&part, b"compressed").unwrap();
    let published = work.path("delta.manifest.json");
    let text = ok(&[
        "release-files",
        "--release",
        &format!("{delta}/release.json"),
        "--files",
        &part,
        "--out",
        &published,
    ]);
    assert!(text.contains("-v1-v2"), "{text}");
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&published).unwrap()).unwrap();
    assert_eq!(manifest["files"][0]["compression"], "zstd");
    assert_eq!(
        manifest["files"][0]["uncompressedSha256"],
        manifest["segment"]["sha256"]
    );

    // A delta without its ledger is refused.
    let (code, _) = run(&[
        "assemble",
        "--kind",
        "delta",
        "--plan",
        &plan2,
        "--warehouse",
        warehouse,
        "--out",
        &work.path("refused"),
    ]);
    assert_eq!(code, 1);
}

/// A delta of deletions and reuse verifies with exit 0, G5 `notApplicable`.
#[test]
fn a_delta_that_ships_no_vector_verifies_through_the_cli() {
    let build = Build::new("vector-build-cli-no-vector");
    let (work, warehouse) = (&build.work, build.warehouse.as_str());
    let a: Vec<usize> = (0..300).collect();
    let b: Vec<usize> = (300..600).collect();
    let (c, d): (&[usize], &[usize]) = (&[600, 601, 602, 603], &[5, 305]);
    let plan1 = build.step(
        1,
        &[
            ("otzaria/a.txt", &a),
            ("otzaria/b.txt", &b),
            ("otzaria/c.txt", c),
        ],
        None,
    );
    let base = work.path("base-v1");
    ok(&[
        "assemble",
        "--kind",
        "base",
        "--plan",
        &plan1,
        "--warehouse",
        warehouse,
        "--out",
        &base,
        "--created-at",
        AT,
        "--verify",
    ]);
    let report: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(Path::new(&base).join("gates.json")).unwrap(),
    )
    .unwrap();
    for gate in report["gates"].as_array().unwrap() {
        let mut fields: Vec<&str> = gate
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        fields.sort_unstable();
        assert_eq!(fields, ["detail", "gate", "passed", "status"], "{gate}");
        assert_eq!(gate["passed"], true, "{gate}");
        assert_eq!(gate["status"], "passed", "{gate}");
    }

    let plan2 = build.step(
        2,
        &[
            ("otzaria/a.txt", &a),
            ("otzaria/b.txt", &b),
            ("otzaria/d.txt", d),
        ],
        Some(&base),
    );
    let delta = work.path("delta-v2");
    let (code, text) = run(&[
        "assemble",
        "--kind",
        "delta",
        "--plan",
        &plan2,
        "--warehouse",
        warehouse,
        "--previous",
        &base,
        "--out",
        &delta,
        "--created-at",
        AT,
        "--verify",
    ]);
    assert_eq!(code, 0, "{text}");
    assert!(
        text.contains("0 slot(s), 0 extra(s), 2 foreign, 4 tombstone(s)"),
        "{text}"
    );
    assert!(
        text.contains("G5   n/a   not applicable: the delta ships no vector"),
        "{text}"
    );
    assert_eq!(
        gates(&delta),
        ["G1", "G5", "G7", "G8", "G9", "G10"].map(|gate| (gate.to_string(), true))
    );
    let report: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(Path::new(&delta).join("gates.json")).unwrap(),
    )
    .unwrap();
    let statuses: Vec<&str> = report["gates"]
        .as_array()
        .unwrap()
        .iter()
        .map(|gate| gate["status"].as_str().unwrap())
        .collect();
    assert_eq!(
        statuses,
        [
            "passed",
            "notApplicable",
            "passed",
            "passed",
            "passed",
            "passed"
        ]
    );
}
