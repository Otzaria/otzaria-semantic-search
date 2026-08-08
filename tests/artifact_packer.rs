//! S4a's acceptance gate, from outside the crate.
//!
//! Two claims, and the stage is only done if both hold:
//!
//! 1. **One command** takes ready-made vectors, a corpus and a model, writes an artifact
//!    in the official format, and verifies it against that same corpus — and fails loudly
//!    when the vectors do not belong to the lines they claim.
//! 2. **What it writes is what the runtime opens.** The artifact goes through
//!    [`IndexImporter`] and [`OfficialSemanticIndex`] untouched, and a semantic query
//!    comes back with the `line_id` the vector was built from.
//!
//! The second one is the reason the first is worth anything, and it is asserted here
//! rather than assumed: everything before this stage built its fixtures by hand, so
//! "the packer writes what the reader reads" had no test that would fail if it stopped
//! being true.
//!
//! Claim 2 needs an embedding backend to turn a query into a vector, so it is compiled
//! only under `mock-embedding`. Claim 1 needs none — packing never embeds anything — and
//! runs in a default build, which is the build a release pipeline has.

use otzaria_semantic_search::distribution::corpus::{CorpusLine, CorpusLineRecord, JsonlCorpus};
use otzaria_semantic_search::distribution::packer::{
    pack, read_vector_inputs, validate_artifact, PackRequest, VectorInputRecord,
};
use otzaria_semantic_search::semantic::versioning::{CorpusIdentity, ModelIdentity};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::process::Command;

const DIM: u32 = 64;
const GENESIS: &str = "otzaria/tanach/genesis.txt";
const BERACHOT: &str = "otzaria/mishna/berachot.txt";

/// `(line_id, book, text)`. The ids are formed the way `document_id_scheme_version` 1
/// forms them — `((catalogue_order + 1) << 32) + (ordinal + 1)` — because a semantic
/// result *is* one of these numbers and nothing else.
const LINES: [(u64, &str, &str); 4] = [
    (4_294_967_297, GENESIS, "בראשית ברא אלהים את השמים ואת הארץ"),
    (
        4_294_967_298,
        GENESIS,
        "והארץ היתה תהו ובהו וחשך על פני תהום",
    ),
    (4_294_967_299, GENESIS, "ויאמר אלהים יהי אור ויהי אור"),
    (
        8_589_934_593,
        BERACHOT,
        "מאימתי קורין את שמע בערבית משעה שהכהנים נכנסין לאכול בתרומתן",
    ),
];

struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "otzaria_packer_{name}_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// The corpus identity a build takes off the Tantivy index it opened.
fn corpus_identity() -> CorpusIdentity {
    CorpusIdentity {
        corpus_id: "3f".repeat(32),
        library_version: "otzaria-library-2026-08".to_string(),
        tantivy_schema_version: 3,
        document_id_scheme_version: 1,
    }
}

fn corpus_line(book: &str, text: &str) -> CorpusLine {
    CorpusLine {
        source_book_key: book.to_string(),
        title: if book == GENESIS {
            "בראשית"
        } else {
            "משנה ברכות"
        }
        .to_string(),
        reference: format!("{book} :: {}", text.chars().take(8).collect::<String>()),
        section_id: 1,
        segment: 0,
        is_pdf: false,
        line_hash: 0,
        content_hash: 4242,
        facets: vec!["/מקרא/תורה".to_string(), "/era/תנך".to_string()],
        text: text.to_string(),
    }
}

/// Write the corpus transcription a build machine would export from Tantivy.
fn write_corpus(dir: &Path) -> (PathBuf, PathBuf) {
    let identity_path = dir.join("corpus-identity.json");
    let lines_path = dir.join("corpus-lines.jsonl");

    std::fs::write(
        &identity_path,
        serde_json::to_vec_pretty(&corpus_identity()).unwrap(),
    )
    .unwrap();
    let body: String = LINES
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

    (identity_path, lines_path)
}

/// Write the two files a vector producer emits: the floats, and one record per vector in
/// the same order. `vector_for` decides which vector each line gets, so a test can hand
/// over a set that is subtly misaligned.
fn write_vectors(
    dir: &Path,
    name: &str,
    vector_for: impl Fn(usize) -> (u64, Vec<f32>, String, String),
) -> (PathBuf, PathBuf) {
    let vectors_path = dir.join(format!("{name}.f32"));
    let records_path = dir.join(format!("{name}.jsonl"));

    let mut bytes = Vec::new();
    let mut records = String::new();
    for index in 0..LINES.len() {
        let (line_id, vector, source_line_sha256, embedding_text_sha256) = vector_for(index);
        for value in &vector {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        records.push_str(&format!(
            "{}\n",
            serde_json::to_string(&VectorInputRecord {
                line_id,
                source_line_sha256,
                embedding_text_sha256,
            })
            .unwrap()
        ));
    }
    std::fs::write(&vectors_path, bytes).unwrap();
    std::fs::write(&records_path, records).unwrap();

    (vectors_path, records_path)
}

fn write_model(dir: &Path, model: &ModelIdentity) -> PathBuf {
    let path = dir.join("model.json");
    std::fs::write(&path, serde_json::to_vec_pretty(model).unwrap()).unwrap();
    path
}

/// A deterministic stand-in for real vectors, for the checks that never embed anything.
/// Distinct per text, so a vector paired with the wrong line is a different vector.
fn stand_in_vector(text: &str) -> Vec<f32> {
    let digest = Sha256::digest(text.as_bytes());
    (0..DIM)
        .map(|i| f32::from(digest[i as usize % 32]) + 1.0)
        .collect()
}

/// A line whose recipe embedded it unchanged: both digests are of the same text.
fn embedded_unchanged(line_id: u64, text: &str) -> (u64, Vec<f32>, String, String) {
    (
        line_id,
        stand_in_vector(text),
        sha256_hex(text.as_bytes()),
        sha256_hex(text.as_bytes()),
    )
}

fn stand_in_model() -> ModelIdentity {
    ModelIdentity {
        model_id: "EMD123/Otzaria-Embedding-V1-Flash-0.6B".to_string(),
        model_checksum: "c".repeat(64),
        model_quantization: "Q4_K_M".to_string(),
        embedding_backend: "llama-cpp-2-0.1.153".to_string(),
        embedding_dim: DIM,
        pooling: "last-token".to_string(),
        max_tokens: 512,
        embedding_text_version: 1,
        normalization_version: 1,
        chunking_identity: 0x0BAD_C0DE,
    }
}

/// The stage's first claim, through the binary a build pipeline actually runs: one
/// command turns a vector file into a verified artifact, and `validate` re-establishes
/// the same thing on a directory it did not build.
#[test]
fn the_cli_packs_ready_made_vectors_into_a_verified_artifact() {
    let dir = TempDir::new("cli");
    let (corpus_identity_path, corpus_lines_path) = write_corpus(dir.path());
    let model_path = write_model(dir.path(), &stand_in_model());
    let (vectors_path, records_path) = write_vectors(dir.path(), "good", |index| {
        let (line_id, _, text) = LINES[index];
        embedded_unchanged(line_id, text)
    });
    let out = dir.path().join("artifact");

    let packed = Command::new(env!("CARGO_BIN_EXE_otzaria-semantic-search"))
        .args([
            "pack",
            "--vectors",
            vectors_path.to_str().unwrap(),
            "--records",
            records_path.to_str().unwrap(),
            "--corpus-identity",
            corpus_identity_path.to_str().unwrap(),
            "--corpus-lines",
            corpus_lines_path.to_str().unwrap(),
            "--model",
            model_path.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
        ])
        .output()
        .expect("the CLI binary runs");

    let stdout = String::from_utf8_lossy(&packed.stdout).into_owned();
    assert!(
        packed.status.success(),
        "pack failed: {}{stdout}",
        String::from_utf8_lossy(&packed.stderr)
    );
    assert!(stdout.contains("Packed an official artifact"), "{stdout}");
    assert!(
        stdout.contains(&format!("Vectors:         {}", LINES.len())),
        "{stdout}"
    );
    assert!(stdout.contains("Books:           2"), "{stdout}");

    // The artifact is a directory of this backend's payload plus the two metadata
    // documents — nothing else, and nothing missing.
    let mut written: Vec<String> = std::fs::read_dir(&out)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    written.sort();
    assert_eq!(
        written,
        [
            "book_index.json",
            "manifest.json",
            "metadata.jsonl",
            "payloads.json",
            "vectors.bin"
        ]
    );

    let validated = Command::new(env!("CARGO_BIN_EXE_otzaria-semantic-search"))
        .args([
            "validate",
            "--artifact",
            out.to_str().unwrap(),
            "--corpus-identity",
            corpus_identity_path.to_str().unwrap(),
            "--corpus-lines",
            corpus_lines_path.to_str().unwrap(),
            "--model",
            model_path.to_str().unwrap(),
        ])
        .output()
        .expect("the CLI binary runs");
    let revalidated = String::from_utf8_lossy(&validated.stdout).into_owned();
    assert!(validated.status.success(), "{revalidated}");
    assert!(revalidated.contains("Artifact verified"), "{revalidated}");

    // Both runs report the same digest — the value a publisher announces, and the only
    // thing that later separates this artifact from a self-consistent rebuild.
    let digest_of = |output: &str| {
        output
            .lines()
            .find_map(|line| line.strip_prefix("Digest:"))
            .map(|digest| digest.trim().to_string())
            .expect("a report carries a digest")
    };
    assert_eq!(digest_of(&stdout), digest_of(&revalidated));
    assert_eq!(digest_of(&stdout).len(), 64);
}

/// The failure the join exists for, driven through the same command: the vectors were
/// written in one order and the ids in another. Every count still adds up, every checksum
/// still passes, and the artifact would return the wrong line for every query.
#[test]
fn the_cli_refuses_vectors_that_do_not_belong_to_the_lines_they_name() {
    let dir = TempDir::new("cli_shifted");
    let (corpus_identity_path, corpus_lines_path) = write_corpus(dir.path());
    let model_path = write_model(dir.path(), &stand_in_model());
    let (vectors_path, records_path) = write_vectors(dir.path(), "shifted", |index| {
        let (line_id, _, _) = LINES[index];
        let (_, _, neighbour) = LINES[(index + 1) % LINES.len()];
        // Every id is still present exactly once, so coverage and the counts are
        // untouched: only the source digest can see this.
        let (_, vector, source, embedded) = embedded_unchanged(line_id, neighbour);
        (line_id, vector, source, embedded)
    });
    let out = dir.path().join("artifact");

    let packed = Command::new(env!("CARGO_BIN_EXE_otzaria-semantic-search"))
        .args([
            "pack",
            "--vectors",
            vectors_path.to_str().unwrap(),
            "--records",
            records_path.to_str().unwrap(),
            "--corpus-identity",
            corpus_identity_path.to_str().unwrap(),
            "--corpus-lines",
            corpus_lines_path.to_str().unwrap(),
            "--model",
            model_path.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
        ])
        .output()
        .expect("the CLI binary runs");

    assert!(
        !packed.status.success(),
        "a shifted pairing must not pack successfully"
    );
    let stderr = String::from_utf8_lossy(&packed.stderr);
    assert!(
        stderr.contains("built from text the corpus does not hold"),
        "the refusal must say what is wrong: {stderr}"
    );
    // Nothing to publish and nothing half-built: the payload writer buffers until it
    // commits, so a rejection leaves the output directory empty.
    assert!(
        !out.exists() || std::fs::read_dir(&out).unwrap().count() == 0,
        "a refused pack must not leave an artifact behind"
    );
}

/// The library entry point behind that command, over the same fixture files, so a caller
/// that has Tantivy open in process — `otzaria_search_engine`, in S4b/S5 — reaches the
/// identical checks without going through a CLI or a transcription.
#[test]
fn the_library_entry_point_packs_the_same_artifact_the_cli_does() {
    let dir = TempDir::new("library");
    let (corpus_identity_path, corpus_lines_path) = write_corpus(dir.path());
    let (vectors_path, records_path) = write_vectors(dir.path(), "good", |index| {
        let (line_id, _, text) = LINES[index];
        embedded_unchanged(line_id, text)
    });
    let corpus = JsonlCorpus::load(&corpus_identity_path, &corpus_lines_path).unwrap();
    let out = dir.path().join("artifact");

    let report = pack(
        PackRequest {
            output_path: out.clone(),
            model: stand_in_model(),
            created_at: "2026-08-08T00:00:00Z".to_string(),
            collection_name: "chunks".to_string(),
        },
        read_vector_inputs(&vectors_path, &records_path, DIM).unwrap(),
        &corpus,
    )
    .unwrap();

    assert_eq!(report.vector_count, LINES.len() as u32);
    assert_eq!(report.book_count, 2);
    assert_eq!(report.identity.corpus, corpus_identity());
    assert_eq!(report.identity.model, stand_in_model());
    assert_eq!(
        validate_artifact(&out, &stand_in_model(), &corpus)
            .unwrap()
            .digest,
        report.digest
    );
}

/// Two independent packs of the same vectors produce the same artifact, byte for byte.
///
/// This is what a published digest rests on. The digest covers the payload checksums, so
/// unless the payload is written deterministically, "the same build produces the same
/// digest twice" is false and a rebuilt artifact can never be checked against what was
/// announced. Two separate directories, because comparing a pack against a `validate` of
/// the same directory would pass no matter what the writer did.
#[test]
fn packing_the_same_vectors_twice_produces_the_same_artifact() {
    let dir = TempDir::new("reproducible");
    let (corpus_identity_path, corpus_lines_path) = write_corpus(dir.path());
    let (vectors_path, records_path) = write_vectors(dir.path(), "good", |index| {
        let (line_id, _, text) = LINES[index];
        embedded_unchanged(line_id, text)
    });
    let corpus = JsonlCorpus::load(&corpus_identity_path, &corpus_lines_path).unwrap();

    let pack_into = |name: &str| {
        pack(
            PackRequest {
                output_path: dir.path().join(name),
                model: stand_in_model(),
                // Deliberately different, because `created_at` is excluded from the digest
                // and this is the test that would notice if it stopped being.
                created_at: format!("2026-08-0{}T00:00:00Z", name.len()),
                collection_name: "chunks".to_string(),
            },
            read_vector_inputs(&vectors_path, &records_path, DIM).unwrap(),
            &corpus,
        )
        .unwrap()
    };

    let first = pack_into("one");
    let second = pack_into("two2");
    assert_eq!(first.digest, second.digest);

    for payload in ["vectors.bin", "metadata.jsonl", "book_index.json"] {
        assert_eq!(
            std::fs::read(dir.path().join("one").join(payload)).unwrap(),
            std::fs::read(dir.path().join("two2").join(payload)).unwrap(),
            "{payload} must not depend on anything but the vectors and the corpus"
        );
    }
}

/// An artifact that holds part of the library passes every count, checksum and identity
/// field there is. Only the corpus knows how many vectors there should have been.
#[test]
fn the_cli_refuses_an_artifact_that_covers_part_of_the_corpus() {
    let dir = TempDir::new("cli_coverage");
    let (corpus_identity_path, corpus_lines_path) = write_corpus(dir.path());
    let model_path = write_model(dir.path(), &stand_in_model());
    let (vectors_path, records_path) = write_vectors(dir.path(), "good", |index| {
        let (line_id, _, text) = LINES[index];
        embedded_unchanged(line_id, text)
    });

    // Keep the first record and its vector; drop the rest, as a truncated export would.
    let partial_records = dir.path().join("partial.jsonl");
    let partial_vectors = dir.path().join("partial.f32");
    let records = std::fs::read_to_string(&records_path).unwrap();
    std::fs::write(
        &partial_records,
        format!("{}\n", records.lines().next().unwrap()),
    )
    .unwrap();
    let vectors = std::fs::read(&vectors_path).unwrap();
    std::fs::write(&partial_vectors, &vectors[..DIM as usize * 4]).unwrap();

    let out = dir.path().join("artifact");
    let packed = Command::new(env!("CARGO_BIN_EXE_otzaria-semantic-search"))
        .args([
            "pack",
            "--vectors",
            partial_vectors.to_str().unwrap(),
            "--records",
            partial_records.to_str().unwrap(),
            "--corpus-identity",
            corpus_identity_path.to_str().unwrap(),
            "--corpus-lines",
            corpus_lines_path.to_str().unwrap(),
            "--model",
            model_path.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
        ])
        .output()
        .expect("the CLI binary runs");

    assert!(
        !packed.status.success(),
        "a partial artifact must not pack successfully"
    );
    let stderr = String::from_utf8_lossy(&packed.stderr);
    assert!(
        stderr.contains("no vector") && stderr.contains(&format!("{}", LINES.len())),
        "the refusal must say how much is missing: {stderr}"
    );
    assert!(
        !out.exists() || std::fs::read_dir(&out).unwrap().count() == 0,
        "a refused pack must not leave an artifact behind"
    );
}

/// The stage's second claim: the packer's output is the runtime's input.
///
/// Packed here, installed through the importer, opened through the official read path,
/// and queried — with no fixture assembled by hand anywhere in between. If the packer and
/// the reader ever stop agreeing about the payload, the identity or the counts, this is
/// what fails.
#[cfg(all(feature = "mock-embedding", not(feature = "llama-backend")))]
#[test]
fn an_artifact_this_packer_wrote_installs_opens_and_answers_a_query() {
    use otzaria_semantic_search::distribution::importer::{ImportConfig, IndexImporter};
    use otzaria_semantic_search::distribution::package::ArtifactExpectation;
    use otzaria_semantic_search::semantic::backend::MockHashBackend;
    use otzaria_semantic_search::semantic::embedding::{mock, validate_and_checksum_gguf};
    use otzaria_semantic_search::semantic::official_index::{
        LocalModel, OfficialIndexConfig, OfficialSemanticIndex,
    };

    let dir = TempDir::new("runtime");
    let model_path = dir.path().join("model.gguf");
    mock::write_stub_gguf(&model_path, 3).unwrap();

    // What the installation declares about itself, and what the build records: the same
    // values, plus the two facts only the loaded model can supply.
    let local = LocalModel {
        model_path: model_path.clone(),
        model_id: "EMD123/Otzaria-Embedding-V1-Flash-0.6B".to_string(),
        model_quantization: "Q4_K_M".to_string(),
        embedding_dim: DIM,
        pooling: "last-token".to_string(),
        max_tokens: 512,
        embedding_text_version: 1,
        normalization_version: 1,
        chunking_identity: 0x0BAD_C0DE,
    };
    let model = ModelIdentity {
        model_id: local.model_id.clone(),
        model_checksum: validate_and_checksum_gguf(&model_path).unwrap(),
        model_quantization: local.model_quantization.clone(),
        embedding_backend: MockHashBackend::ID.to_string(),
        embedding_dim: local.embedding_dim,
        pooling: local.pooling.clone(),
        max_tokens: local.max_tokens,
        embedding_text_version: local.embedding_text_version,
        normalization_version: local.normalization_version,
        chunking_identity: local.chunking_identity,
    };

    let (corpus_identity_path, corpus_lines_path) = write_corpus(dir.path());
    // Real vectors this time: whatever the runtime's embedder produces for the line, so a
    // query for that line's text has to come back with that line's id.
    let (vectors_path, records_path) = write_vectors(dir.path(), "embedded", |index| {
        let (line_id, _, text) = LINES[index];
        (
            line_id,
            mock::hash_embedding(text, DIM),
            sha256_hex(text.as_bytes()),
            sha256_hex(text.as_bytes()),
        )
    });
    let corpus = JsonlCorpus::load(&corpus_identity_path, &corpus_lines_path).unwrap();

    let source = dir.path().join("build-output");
    let report = pack(
        PackRequest {
            output_path: source.clone(),
            model,
            created_at: "2026-08-08T00:00:00Z".to_string(),
            collection_name: "chunks".to_string(),
        },
        read_vector_inputs(&vectors_path, &records_path, DIM).unwrap(),
        &corpus,
    )
    .unwrap();

    // Installed against the digest the packer published, which is the whole point of
    // reporting one.
    let target = dir.path().join("semantic_index");
    let installed = IndexImporter::new(ImportConfig {
        source_path: source,
        target_store_path: target.clone(),
    })
    .import(&ArtifactExpectation::with_published_digest(
        report.identity.clone(),
        report.digest.clone(),
    ))
    .unwrap();
    assert_eq!(installed.vectors_imported, LINES.len() as u32);

    let index = OfficialSemanticIndex::open(OfficialIndexConfig {
        artifact_path: target,
        corpus: corpus_identity(),
        model: local,
        published_digest: Some(report.digest.clone()),
    })
    .unwrap();

    assert_eq!(index.identity(), &report.identity);
    assert_eq!(index.artifact_digest(), report.digest);
    assert_eq!(index.vector_count(), LINES.len() as u32);
    assert_eq!(index.book_count(), 2);
    assert_eq!(index.book_keys(), [BERACHOT, GENESIS]);

    // Every line resolves to itself, and the metadata the runtime hands back is the
    // corpus's — never something the vector producer supplied.
    for (line_id, book, text) in LINES {
        let hit = &index.search(text, 1, None).unwrap()[0];
        assert_eq!(hit.metadata.line_id, line_id);
        assert_eq!(hit.metadata.source_book_key, book);
        assert_eq!(hit.metadata.reference, corpus_line(book, text).reference);
        assert_eq!(hit.metadata.title, corpus_line(book, text).title);
        assert!((hit.similarity_score - 1.0).abs() < 1e-5);
    }
}
