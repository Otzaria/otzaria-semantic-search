//! The update path as the CI actually runs it: five subcommands and a shell.
//!
//! `two_generation_reuse` proves the *library* produces the right artifact. It calls
//! `ledger_from_artifact` directly and builds a `LedgerManifest` by hand, so two fixes
//! that live only in `main.rs` are invisible to it:
//!
//! * the ledger's identity is read from the verified package, never from `--model` — the
//!   bug was a manifest that could declare model B over vectors built by model A;
//! * a release where no text changed reuses everything, with no shard directory at all,
//!   which was the one path `verify_shards` refused outright.
//!
//! Both can regress with every Rust test still green, so they are pinned here through the
//! binary. The corpus is four lines; what is being tested is the wiring, not the volume.

#![cfg(all(feature = "mock-embedding", not(feature = "llama-backend")))]

use otzaria_semantic_search::distribution::corpus::{CorpusLine, CorpusLineRecord};
use otzaria_semantic_search::semantic::chunker::ChunkerConfig;
use otzaria_semantic_search::semantic::embedding::{mock, validate_and_checksum_gguf};
use otzaria_semantic_search::semantic::versioning::{CorpusIdentity, ModelIdentity};
use std::path::{Path, PathBuf};
use std::process::Command;

const BIN: &str = env!("CARGO_BIN_EXE_otzaria-semantic-search");
const DIM: u32 = 64;
const CREATED_AT: &str = "2026-08-10T00:00:00Z";
const BOOK: &str = "otzaria/tanach/genesis.txt";

struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> Self {
        // The clock alone collided: macOS ticks coarser than a test takes to start.
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "otzaria_cli_update_{name}_{}_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn at(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Run the CLI and require it to succeed, returning stdout.
fn must(args: &[&str]) -> String {
    let output = Command::new(BIN)
        .args(args)
        .output()
        .expect("the CLI is built by cargo for this test");
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    assert!(
        output.status.success(),
        "`{}` failed\n--- stdout\n{stdout}\n--- stderr\n{}",
        args.join(" "),
        String::from_utf8_lossy(&output.stderr)
    );
    stdout
}

/// Run the CLI and require it to refuse, for the stated reason.
fn refused(args: &[&str], expected: &str) {
    let output = Command::new(BIN)
        .args(args)
        .output()
        .expect("the CLI is built by cargo for this test");
    let said = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        !output.status.success(),
        "`{}` was accepted; it should have refused with {expected:?}\n{said}",
        args.join(" ")
    );
    assert!(
        said.contains(expected),
        "expected {expected:?}, got:\n{said}"
    );
}

/// The value printed after a `Label:` in the CLI's own report.
fn field(stdout: &str, label: &str) -> String {
    stdout
        .lines()
        .find_map(|line| line.strip_prefix(label))
        .unwrap_or_else(|| panic!("no {label:?} in:\n{stdout}"))
        .trim()
        .to_string()
}

fn sha256_of(path: &Path) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(std::fs::read(path).unwrap()))
}

fn write_corpus(dir: &TempDir, name: &str, lines: &[(u64, &str)]) -> (PathBuf, PathBuf) {
    let identity = dir.at(&format!("{name}-identity.json"));
    let body = dir.at(&format!("{name}-lines.jsonl"));
    std::fs::write(
        &identity,
        serde_json::to_vec_pretty(&CorpusIdentity {
            corpus_id: "7f".repeat(32),
            library_version: "otzaria-library-test".to_string(),
            tantivy_schema_version: 3,
            document_id_scheme_version: 1,
        })
        .unwrap(),
    )
    .unwrap();
    let text: String = lines
        .iter()
        .map(|(line_id, line)| {
            format!(
                "{}\n",
                serde_json::to_string(&CorpusLineRecord {
                    line_id: *line_id,
                    line: CorpusLine {
                        source_book_key: BOOK.to_string(),
                        title: "ספר".to_string(),
                        reference: format!("{BOOK} :: {line_id}"),
                        section_id: 1,
                        segment: 0,
                        is_pdf: false,
                        line_hash: 0,
                        content_hash: 7,
                        facets: vec!["/מקרא".to_string()],
                        text: (*line).to_string(),
                    },
                })
                .unwrap()
            )
        })
        .collect();
    std::fs::write(&body, text).unwrap();
    (identity, body)
}

#[test]
fn the_ledger_takes_its_identity_from_the_artifact_and_an_unchanged_corpus_needs_no_gpu() {
    let dir = TempDir::new("chain");
    let chunking = ChunkerConfig::default();
    let chunking_path = dir.at("chunking.json");
    std::fs::write(
        &chunking_path,
        serde_json::to_vec_pretty(&chunking).unwrap(),
    )
    .unwrap();

    let model_file = dir.at("model.gguf");
    mock::write_stub_gguf(&model_file, 3).unwrap();
    let checksum = validate_and_checksum_gguf(&model_file).unwrap();
    let model = ModelIdentity {
        model_id: "otzaria-embedding-v1".to_string(),
        model_checksum: checksum,
        model_quantization: "Q4_K_M".to_string(),
        embedding_backend: "mock-hash-v1".to_string(),
        embedding_dim: DIM,
        pooling: "last-token".to_string(),
        max_tokens: 512,
        embedding_text_version: 1,
        normalization_version: 1,
        chunking_identity: chunking.identity(),
    };
    let model_path = dir.at("model.json");
    std::fs::write(&model_path, serde_json::to_vec_pretty(&model).unwrap()).unwrap();

    // Same width, same recipe, another file. A manifest that copied `--model` would take
    // this and `VerifiedBase` would then verify the ledger perfectly against it.
    let foreign_path = dir.at("foreign-model.json");
    std::fs::write(
        &foreign_path,
        serde_json::to_vec_pretty(&ModelIdentity {
            model_checksum: "cd".repeat(32),
            ..model.clone()
        })
        .unwrap(),
    )
    .unwrap();

    let (identity, lines) = write_corpus(
        &dir,
        "gen1",
        &[
            (10, "בראשית ברא אלהים את השמים ואת הארץ"),
            (11, "והארץ היתה תהו ובהו וחשך על פני תהום רבה"),
            (12, "ויאמר אלהים יהי אור ויהי אור וירא אלהים כי טוב"),
            (13, "ויבדל אלהים בין האור ובין החשך ויקרא לאור יום"),
        ],
    );
    let strings = |paths: &[&Path]| -> Vec<String> {
        paths
            .iter()
            .map(|path| path.display().to_string())
            .collect()
    };
    let corpus = strings(&[&identity, &lines]);
    let common = [
        "--corpus-identity",
        &corpus[0],
        "--corpus-lines",
        &corpus[1],
        "--chunking",
        &chunking_path.display().to_string(),
        "--model",
        &model_path.display().to_string(),
    ]
    .map(String::from);
    let common: Vec<&str> = common.iter().map(String::as_str).collect();

    // 1. The plan.
    let plan_dir = dir.at("plan");
    let exported = must(
        &[
            &["export-plan", "--out", &plan_dir.display().to_string()],
            common.as_slice(),
        ]
        .concat(),
    );
    let records: usize = field(&exported, "Records:").parse().unwrap();
    assert!(records >= 4, "four lines get at least four vectors");
    let plan = plan_dir.join("plan.jsonl");

    // 2. One shard over the whole plan.
    let shard = dir.at("shards/0");
    must(&[
        "embed-shard",
        "--plan",
        &plan.display().to_string(),
        "--model",
        &model_path.display().to_string(),
        "--model-file",
        &model_file.display().to_string(),
        "--out",
        &shard.display().to_string(),
        "--allow-non-semantic",
    ]);

    // 3. Assemble, with no base: the first generation reuses nothing.
    let merged = dir.at("merged1");
    must(&[
        "assemble",
        "--shards",
        &dir.at("shards").display().to_string(),
        "--plan-sha256",
        &sha256_of(&plan),
        "--embed-records",
        &records.to_string(),
        "--model",
        &model_path.display().to_string(),
        "--out",
        &merged.display().to_string(),
    ]);

    // 4. Pack it.
    let artifact = dir.at("artifact1");
    let packed = must(
        &[
            &[
                "pack",
                "--vectors",
                &merged.join("vectors.f32").display().to_string(),
                "--records",
                &merged.join("records.jsonl").display().to_string(),
                "--out",
                &artifact.display().to_string(),
                "--created-at",
                CREATED_AT,
            ],
            common.as_slice(),
        ]
        .concat(),
    );
    let digest = field(&packed, "Digest:");
    assert_eq!(digest.len(), 64, "a digest is 64 hex digits: {digest:?}");

    // 5. The ledger — with `--model` naming the *foreign* identity, which must have no
    //    effect at all. It is the artifact that says what these vectors are.
    let ledger = dir.at("ledger.jsonl");
    let ledger_manifest = dir.at("ledger.manifest.json");
    must(&[
        "ledger",
        "--artifact",
        &artifact.display().to_string(),
        "--records",
        &merged.join("records.jsonl").display().to_string(),
        "--out",
        &ledger.display().to_string(),
        "--model",
        &foreign_path.display().to_string(),
        "--artifact-digest",
        &digest,
    ]);
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&ledger_manifest).unwrap()).unwrap();
    assert_eq!(
        manifest["model"]["model_checksum"], model.model_checksum,
        "the model comes from the package, not from --model"
    );
    assert_eq!(
        manifest["artifact_digest"], digest,
        "the published digest is the artifact's own"
    );
    assert_eq!(
        manifest["ledger_sha256"],
        sha256_of(&ledger),
        "the manifest describes the ledger that was written"
    );

    // 6. A digest that names another artifact is refused — and leaves what was there.
    let before = std::fs::read(&ledger).unwrap();
    refused(
        &[
            "ledger",
            "--artifact",
            &artifact.display().to_string(),
            "--records",
            &merged.join("records.jsonl").display().to_string(),
            "--out",
            &ledger.display().to_string(),
            "--artifact-digest",
            &"ab".repeat(32),
        ],
        "This is not the artifact that digest was published for",
    );
    assert_eq!(
        std::fs::read(&ledger).unwrap(),
        before,
        "a refusal must not replace the ledger it refused to rebuild"
    );
    assert!(
        !dir.at("ledger.partial").exists(),
        "and must not leave a half-written one either"
    );

    // 6b. A records file that does not describe this artifact fails *after* the entries
    //     have been written, which is what the `.partial` is for: the previous ledger has
    //     to survive it, because a new ledger beside the previous manifest is the one
    //     pairing nothing downstream can detect — each file is internally valid.
    let short = dir.at("short-records.jsonl");
    let all = std::fs::read_to_string(merged.join("records.jsonl")).unwrap();
    let kept: Vec<&str> = all.lines().take(records - 1).collect();
    std::fs::write(&short, format!("{}\n", kept.join("\n"))).unwrap();
    refused(
        &[
            "ledger",
            "--artifact",
            &artifact.display().to_string(),
            "--records",
            &short.display().to_string(),
            "--out",
            &ledger.display().to_string(),
        ],
        "The ledger could not be built",
    );
    assert_eq!(
        std::fs::read(&ledger).unwrap(),
        before,
        "the ledger that was already good must still be there"
    );

    // 7. The same corpus again. Every digest is known, so nothing is embedded.
    let split = dir.at("split2");
    let base = [
        "--ledger",
        &ledger.display().to_string(),
        "--ledger-manifest",
        &ledger_manifest.display().to_string(),
        "--base-vectors",
        &artifact.join("vectors.bin").display().to_string(),
    ]
    .map(String::from);
    let base: Vec<&str> = base.iter().map(String::as_str).collect();
    let report = must(
        &[
            &[
                "plan-split",
                "--plan",
                &plan.display().to_string(),
                "--out",
                &split.display().to_string(),
                "--model",
                &model_path.display().to_string(),
            ],
            base.as_slice(),
        ]
        .concat(),
    );
    assert_eq!(field(&report, "Reused:"), records.to_string());
    assert_eq!(field(&report, "To embed:"), "0");
    assert_eq!(
        std::fs::read_to_string(split.join("embed.jsonl")).unwrap(),
        "",
        "nothing goes to a GPU"
    );

    // 8. Assemble from the base alone: no shard directory exists, and none should.
    let merged2 = dir.at("merged2");
    let empty = dir.at("no-shards");
    std::fs::create_dir_all(&empty).unwrap();
    must(
        &[
            &[
                "assemble",
                "--shards",
                &empty.display().to_string(),
                "--plan-sha256",
                &sha256_of(&split.join("embed.jsonl")),
                "--embed-records",
                "0",
                "--reuse",
                &split.join("reuse.jsonl").display().to_string(),
                "--model",
                &model_path.display().to_string(),
                "--out",
                &merged2.display().to_string(),
            ],
            base.as_slice(),
        ]
        .concat(),
    );

    // 9. Repacked, it is the same release — every identity, every count, every payload.
    let artifact2 = dir.at("artifact2");
    let repacked = must(
        &[
            &[
                "pack",
                "--vectors",
                &merged2.join("vectors.f32").display().to_string(),
                "--records",
                &merged2.join("records.jsonl").display().to_string(),
                "--out",
                &artifact2.display().to_string(),
                "--created-at",
                CREATED_AT,
            ],
            common.as_slice(),
        ]
        .concat(),
    );
    assert_eq!(
        field(&repacked, "Digest:"),
        digest,
        "a release rebuilt entirely from reuse is the release it was rebuilt from"
    );
    for payload in ["vectors.bin", "metadata.jsonl", "book_index.json"] {
        assert_eq!(
            std::fs::read(artifact.join(payload)).unwrap(),
            std::fs::read(artifact2.join(payload)).unwrap(),
            "{payload} differs, and the digest did not say so"
        );
    }

    // 10. Half a base is refused rather than quietly ignored. `--ledgr` is a typo that
    //     used to read as "there is no previous release" and buy a full rebuild.
    refused(
        &[
            "plan-split",
            "--plan",
            &plan.display().to_string(),
            "--out",
            &dir.at("split3").display().to_string(),
            "--model",
            &model_path.display().to_string(),
            "--ledger-manifest",
            &ledger_manifest.display().to_string(),
            "--base-vectors",
            &artifact.join("vectors.bin").display().to_string(),
        ],
        "Reuse needs --ledger, --ledger-manifest, --base-vectors and --model together",
    );

    // 11. A width the model does not declare is not a width.
    refused(
        &[
            "assemble",
            "--shards",
            &dir.at("shards").display().to_string(),
            "--plan-sha256",
            &sha256_of(&plan),
            "--embed-records",
            &records.to_string(),
            "--model",
            &model_path.display().to_string(),
            "--dim",
            "1024",
            "--out",
            &dir.at("merged3").display().to_string(),
        ],
        "--dim disagrees with the model",
    );
}
