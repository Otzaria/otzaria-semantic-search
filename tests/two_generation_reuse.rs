//! What the whole update path is for, as one comparison.
//!
//! Every other test here checks one link: that a shard boundary reproduces, that a ledger
//! follows the artifact's order, that a foreign base is refused. None of them checks that
//! the chain *produces the right artifact*, and that is the only claim a release rests on:
//!
//! > A release assembled from recycled vectors plus newly embedded ones is the release a
//! > full rebuild would have produced.
//!
//! So this builds generation one the long way, derives its ledger, changes the corpus in
//! the four ways a library actually changes, builds generation two twice — once reusing
//! and once from scratch — and requires the same artifact digest. The digest covers every
//! identity field, both counts and every payload's checksum, so "the same" is not a
//! summary.
//!
//! The four changes are not decoration. A line whose own text is untouched but whose
//! *neighbour* changed is the case an id-keyed diff gets silently wrong, and it is the
//! reason the ledger is keyed on the embedding text instead — see
//! `distribution::reuse`. A corpus of independent lines would pass either design.
//!
//! Two generations have to be embedded here, so a backend is required, and the stub GGUF
//! this uses is one real inference rightly refuses — hence `mock-embedding` without
//! `llama-backend`, the same gate the other end-to-end tests carry.

#![cfg(all(feature = "mock-embedding", not(feature = "llama-backend")))]

use otzaria_semantic_search::distribution::builder::PlannedCorpus;
use otzaria_semantic_search::distribution::corpus::{CorpusLine, CorpusLineRecord, JsonlCorpus};
use otzaria_semantic_search::distribution::package::IndexPackage;
use otzaria_semantic_search::distribution::packer::{pack, read_vector_inputs, PackRequest};
use otzaria_semantic_search::distribution::reuse::{
    assemble, ledger_from_artifact, plan_split, verify_shards, LedgerManifest, ReuseEntry,
    VerifiedBase,
};
use otzaria_semantic_search::distribution::shard::{embed_shard, export_plan, read_plan};
use otzaria_semantic_search::semantic::backend::Pooling;
use otzaria_semantic_search::semantic::chunker::ChunkerConfig;
use otzaria_semantic_search::semantic::embedding::{
    mock, validate_and_checksum_gguf, EmbeddingConfig, EmbeddingRuntime,
};
use otzaria_semantic_search::semantic::versioning::{CorpusIdentity, ModelIdentity};
use std::path::{Path, PathBuf};

const DIM: u32 = 64;
const BOOK: &str = "otzaria/tanach/genesis.txt";
const OTHER: &str = "otzaria/mishna/berachot.txt";

/// Long enough to stand alone under the default recipe; short enough to read.
const LONG_A: &str = "בראשית ברא אלהים את השמים ואת הארץ";
const LONG_B: &str = "והארץ היתה תהו ובהו וחשך על פני תהום רבה";
const LONG_B_EDITED: &str = "והארץ היתה תהו ובהו ורוח אלהים מרחפת על פני המים";
const LONG_C: &str = "ויאמר אלהים יהי אור ויהי אור וירא אלהים כי טוב";
const LONG_D: &str = "מאימתי קורין את שמע בערבית משעה שהכהנים נכנסין";
const LONG_E: &str = "ויבדל אלהים בין האור ובין החשך ויקרא לאור יום";
/// Under `min_meaningful_chars`: this one borrows text from its neighbours, so editing
/// `LONG_B` changes *its* embedding text without changing its own line.
const BORROWS: &str = "ויהי ערב";

struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "otzaria_two_gen_{name}_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
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

fn corpus_line(book: &str, text: &str) -> CorpusLine {
    CorpusLine {
        source_book_key: book.to_string(),
        title: "ספר".to_string(),
        reference: format!("{book} :: {}", text.chars().take(6).collect::<String>()),
        section_id: 1,
        segment: 0,
        is_pdf: false,
        line_hash: 0,
        content_hash: 7,
        facets: vec!["/מקרא".to_string()],
        text: text.to_string(),
    }
}

/// `(line_id, book, text)` into the two files a corpus transcription is.
///
/// `corpus_id` is derived from the lines, so two generations of a changed corpus get
/// different identities — which is what makes the two artifacts comparable only through
/// their vectors, not through a shared identity string.
fn write_corpus(dir: &TempDir, name: &str, lines: &[(u64, &str, &str)]) -> JsonlCorpus {
    let identity = dir.at(&format!("{name}-identity.json"));
    let lines_path = dir.at(&format!("{name}-lines.jsonl"));
    std::fs::write(
        &identity,
        serde_json::to_vec_pretty(&CorpusIdentity {
            // One identity for both generations: the artifacts must differ because their
            // vectors differ, not because their corpus ids do.
            corpus_id: "7f".repeat(32),
            library_version: "otzaria-library-test".to_string(),
            tantivy_schema_version: 3,
            document_id_scheme_version: 1,
        })
        .unwrap(),
    )
    .unwrap();
    let body: String = lines
        .iter()
        .map(|(line_id, book, text)| {
            format!(
                "{}\n",
                serde_json::to_string(&CorpusLineRecord {
                    line_id: *line_id,
                    line: corpus_line(book, text),
                })
                .unwrap()
            )
        })
        .collect();
    std::fs::write(&lines_path, body).unwrap();
    JsonlCorpus::load(&identity, &lines_path).unwrap()
}

fn model_for(checksum: &str, chunking: &ChunkerConfig) -> ModelIdentity {
    ModelIdentity {
        model_id: "otzaria-embedding-v1".to_string(),
        model_checksum: checksum.to_string(),
        model_quantization: "Q4_K_M".to_string(),
        embedding_backend: "mock-hash-v1".to_string(),
        embedding_dim: DIM,
        pooling: "last-token".to_string(),
        max_tokens: 512,
        embedding_text_version: 1,
        normalization_version: 1,
        chunking_identity: chunking.identity(),
    }
}

fn runtime(model_path: &Path) -> EmbeddingRuntime {
    let mut runtime = EmbeddingRuntime::new(EmbeddingConfig {
        model_path: model_path.to_path_buf(),
        embedding_dim: DIM,
        max_tokens: 512,
        batch_size: 2,
        pooling: Pooling::LastToken,
    });
    runtime.load().unwrap();
    runtime
}

/// Export the plan, embed all of it in two windows, assemble and pack. The long way, which
/// is the only way that also produces the `records.jsonl` a ledger needs.
fn build_the_long_way(
    dir: &TempDir,
    name: &str,
    corpus: &JsonlCorpus,
    chunking: &ChunkerConfig,
    model: &ModelIdentity,
    model_path: &Path,
    base: Option<(&VerifiedBase, PathBuf)>,
) -> (PathBuf, PathBuf, usize) {
    let plan_dir = dir.at(&format!("{name}-plan"));
    std::fs::create_dir_all(&plan_dir).unwrap();
    let plan_path = plan_dir.join("plan.jsonl");
    let mut sink = std::fs::File::create(&plan_path).unwrap();
    let plan = export_plan(corpus, chunking, model, &mut sink).unwrap();
    drop(sink);

    // Split against the base, if there is one. Without a base every record is new, which
    // is the first generation.
    let (to_embed, reuse): (PathBuf, Vec<ReuseEntry>) = match &base {
        Some((verified, _)) => {
            let split = dir.at(&format!("{name}-split"));
            std::fs::create_dir_all(&split).unwrap();
            let mut reuse_sink = std::fs::File::create(split.join("reuse.jsonl")).unwrap();
            let mut embed_sink = std::fs::File::create(split.join("embed.jsonl")).unwrap();
            let report = plan_split(
                std::io::BufReader::new(std::fs::File::open(&plan_path).unwrap()),
                Some(verified),
                &mut reuse_sink,
                &mut embed_sink,
            )
            .unwrap();
            drop((reuse_sink, embed_sink));
            assert!(
                report.reused > 0 && report.to_embed > 0,
                "the second generation must exercise both paths, got {report:?}"
            );
            let entries = std::fs::read_to_string(split.join("reuse.jsonl"))
                .unwrap()
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect();
            (split.join("embed.jsonl"), entries)
        }
        None => (plan_path.clone(), Vec::new()),
    };

    // Two windows, so the shard-tiling check has something to tile.
    let records_to_embed = std::fs::read_to_string(&to_embed).unwrap().lines().count();
    let embed_sha = sha256_of(&to_embed);
    let first = records_to_embed / 2;
    let runtime = runtime(model_path);
    let mut shard_dirs = Vec::new();
    for (index, (skip, take)) in [(0, first), (first, records_to_embed - first)]
        .into_iter()
        .enumerate()
    {
        if take == 0 {
            continue;
        }
        let shard = dir.at(&format!("{name}-shard-{index}"));
        std::fs::create_dir_all(&shard).unwrap();
        let mut vectors = std::fs::File::create(shard.join("vectors.f32")).unwrap();
        let mut records = std::fs::File::create(shard.join("records.jsonl")).unwrap();
        let report = embed_shard(
            read_plan(
                std::io::BufReader::new(std::fs::File::open(&to_embed).unwrap()),
                skip,
                take,
            ),
            (embed_sha.clone(), skip, take),
            model,
            &runtime,
            2,
            &mut vectors,
            &mut records,
        )
        .unwrap();
        drop((vectors, records));
        std::fs::write(
            shard.join("shard-manifest.json"),
            serde_json::to_vec_pretty(&report).unwrap(),
        )
        .unwrap();
        shard_dirs.push((shard, report));
    }

    // Through the real gate, so the test covers the check and not a bypass of it.
    let streams = verify_shards(&shard_dirs, &embed_sha, model, records_to_embed).unwrap();

    let merged = dir.at(&format!("{name}-merged"));
    std::fs::create_dir_all(&merged).unwrap();
    let mut vectors = std::fs::File::create(merged.join("vectors.f32")).unwrap();
    let mut records = std::fs::File::create(merged.join("records.jsonl")).unwrap();
    let report = assemble(
        reuse,
        base.as_ref().map(|(verified, _)| *verified),
        streams,
        DIM as usize,
        &mut vectors,
        &mut records,
    )
    .unwrap();
    drop((vectors, records));
    assert_eq!(
        report.vectors, plan.records,
        "every planned line got a vector"
    );

    let artifact = dir.at(&format!("{name}-artifact"));
    let planned = PlannedCorpus::new(corpus, chunking, model).unwrap();
    pack(
        PackRequest {
            output_path: artifact.clone(),
            model: model.clone(),
            created_at: "2026-08-10T00:00:00Z".to_string(),
            collection_name: "chunks".to_string(),
        },
        read_vector_inputs(
            &merged.join("vectors.f32"),
            &merged.join("records.jsonl"),
            DIM,
        )
        .unwrap(),
        &planned,
    )
    .unwrap();

    (artifact, merged.join("records.jsonl"), plan.records)
}

fn sha256_of(path: &Path) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(std::fs::read(path).unwrap()))
}

fn digest_of(artifact: &Path) -> String {
    IndexPackage::read(artifact).unwrap().digest()
}

#[test]
fn a_release_assembled_from_reuse_is_the_release_a_full_rebuild_would_produce() {
    let dir = TempDir::new("chain");
    let chunking = ChunkerConfig::default();
    let model_path = dir.at("model.gguf");
    mock::write_stub_gguf(&model_path, 3).unwrap();
    let model = model_for(&validate_and_checksum_gguf(&model_path).unwrap(), &chunking);

    // Generation one.
    let v1 = write_corpus(
        &dir,
        "v1",
        &[
            (4_294_967_297, BOOK, LONG_A),
            (4_294_967_298, BOOK, LONG_B),
            (4_294_967_299, BOOK, BORROWS),
            (4_294_967_300, BOOK, LONG_C),
            (8_589_934_593, OTHER, LONG_D),
        ],
    );
    let (artifact_v1, records_v1, count_v1) =
        build_the_long_way(&dir, "v1", &v1, &chunking, &model, &model_path, None);
    assert_eq!(count_v1, 5);

    // Its ledger, built from the artifact — the order `pack` published, not the order
    // `assemble` wrote.
    let ledger_path = dir.at("v1-ledger.jsonl");
    let mut sink = std::fs::File::create(&ledger_path).unwrap();
    let entries = ledger_from_artifact(
        std::io::BufReader::new(std::fs::File::open(artifact_v1.join("metadata.jsonl")).unwrap()),
        std::io::BufReader::new(std::fs::File::open(&records_v1).unwrap()),
        &mut sink,
    )
    .unwrap();
    drop(sink);
    assert_eq!(entries, count_v1);

    let base = VerifiedBase::open(
        LedgerManifest {
            artifact_digest: digest_of(&artifact_v1),
            vectors_sha256: sha256_of(&artifact_v1.join("vectors.bin")),
            ledger_sha256: sha256_of(&ledger_path),
            vector_count: entries,
            embedding_dim: DIM,
            model: model.clone(),
        },
        &ledger_path,
        &artifact_v1.join("vectors.bin"),
        &model,
    )
    .unwrap();

    // Generation two: the four ways a library changes. `LONG_A` and `LONG_D` are
    // untouched and must be reused; `LONG_B` is edited, which also changes what `BORROWS`
    // embeds because it borrows from its neighbours; `LONG_C` is gone; `LONG_E` is new.
    let v2 = write_corpus(
        &dir,
        "v2",
        &[
            (4_294_967_297, BOOK, LONG_A),
            (4_294_967_298, BOOK, LONG_B_EDITED),
            (4_294_967_299, BOOK, BORROWS),
            (4_294_967_301, BOOK, LONG_E),
            (8_589_934_593, OTHER, LONG_D),
        ],
    );

    let (reused_artifact, _, count_v2) = build_the_long_way(
        &dir,
        "v2-reused",
        &v2,
        &chunking,
        &model,
        &model_path,
        Some((&base, artifact_v1.join("vectors.bin"))),
    );
    let (fresh_artifact, _, count_fresh) =
        build_the_long_way(&dir, "v2-fresh", &v2, &chunking, &model, &model_path, None);

    assert_eq!(count_v2, count_fresh);
    assert_eq!(
        digest_of(&reused_artifact),
        digest_of(&fresh_artifact),
        "a release built from recycled vectors must be the release a full rebuild produces"
    );

    // And the payloads themselves, not only the digest over their checksums — a digest
    // that agreed while the bytes differed would mean the digest was the wrong summary.
    for payload in ["vectors.bin", "metadata.jsonl", "book_index.json"] {
        assert_eq!(
            std::fs::read(reused_artifact.join(payload)).unwrap(),
            std::fs::read(fresh_artifact.join(payload)).unwrap(),
            "{payload} differs between the reusing build and the full one"
        );
    }
}
