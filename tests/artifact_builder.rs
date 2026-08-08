//! S4b's acceptance gate, from outside the crate.
//!
//! The stage's claim is one command: a corpus and a model in, a full semantic artifact
//! out, verified against that same corpus. S4a proved a *packer* — it took finished floats
//! and could only check that they lined up. What is asserted here is the step before that
//! one, and the three things only a producer can be held to:
//!
//! 1. **The recipe is applied, not described.** Which lines get a vector is derived by
//!    running the chunker over the corpus, and the chunker configuration is pinned to the
//!    `chunking_identity` the artifact declares.
//! 2. **The vectors come from the model the artifact names.** The build loads it and
//!    compares what the file reports against what was declared.
//! 3. **What it writes is what the runtime opens** — through the importer and the official
//!    read path, with no fixture assembled by hand anywhere in between.
//!
//! Everything but the first needs an embedding backend, because a build is inference. The
//! first does not, and runs in a default build: a misconfigured recipe is refused before a
//! model is opened, which is the difference between a build that fails in a second and one
//! that fails after loading half a gigabyte of weights.

use otzaria_semantic_search::distribution::corpus::{CorpusLine, CorpusLineRecord};
use otzaria_semantic_search::semantic::chunker::ChunkerConfig;
use otzaria_semantic_search::semantic::versioning::{CorpusIdentity, ModelIdentity};
use std::path::{Path, PathBuf};
use std::process::Command;

const DIM: u32 = 64;
const GENESIS: &str = "otzaria/tanach/genesis.txt";
const BERACHOT: &str = "otzaria/mishna/berachot.txt";

/// `(line_id, book, text, section_id)`, with ids formed the way
/// `document_id_scheme_version` 1 forms them: `((catalogue_order + 1) << 32) + (ordinal +
/// 1)`.
///
/// Two of these are not ordinary lines. `ויהי אור` is under the recipe's
/// `min_meaningful_chars`, so it is embedded together with its neighbours; `או` is under
/// `min_embeddable_chars`, so it is not embedded at all — and an artifact that skips it is
/// complete rather than short. A corpus of uniformly long lines would let a build that
/// ignored the recipe entirely pass every assertion below.
const LINES: [(u64, &str, &str, u64); 5] = [
    (
        4_294_967_297,
        GENESIS,
        "בראשית ברא אלהים את השמים ואת הארץ",
        1,
    ),
    (
        4_294_967_298,
        GENESIS,
        "והארץ היתה תהו ובהו וחשך על פני תהום רבה",
        1,
    ),
    (4_294_967_299, GENESIS, "ויהי אור", 1),
    (4_294_967_300, GENESIS, "או", 1),
    (
        8_589_934_593,
        BERACHOT,
        "מאימתי קורין את שמע בערבית משעה שהכהנים נכנסין לאכול בתרומתן",
        1,
    ),
];

struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "otzaria_builder_gate_{name}_{}",
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

fn corpus_identity() -> CorpusIdentity {
    CorpusIdentity {
        corpus_id: "9e".repeat(32),
        library_version: "otzaria-library-2026-08".to_string(),
        tantivy_schema_version: 3,
        document_id_scheme_version: 1,
    }
}

fn corpus_line(book: &str, text: &str, section_id: u64) -> CorpusLine {
    CorpusLine {
        source_book_key: book.to_string(),
        title: if book == GENESIS {
            "בראשית"
        } else {
            "משנה ברכות"
        }
        .to_string(),
        reference: format!("{book} :: {}", text.chars().take(8).collect::<String>()),
        section_id,
        segment: 0,
        is_pdf: false,
        line_hash: 0,
        content_hash: 4242,
        facets: vec!["/מקרא/תורה".to_string(), "/era/תנך".to_string()],
        text: text.to_string(),
    }
}

/// The corpus transcription a build machine would export from Tantivy, and the recipe it
/// declares. Every path a `build` invocation needs, except the model.
struct Fixture {
    corpus_identity: PathBuf,
    corpus_lines: PathBuf,
    chunking: PathBuf,
    model: PathBuf,
}

fn write_fixture(dir: &Path, model: &ModelIdentity, chunking: &ChunkerConfig) -> Fixture {
    let fixture = Fixture {
        corpus_identity: dir.join("corpus-identity.json"),
        corpus_lines: dir.join("corpus-lines.jsonl"),
        chunking: dir.join("chunking.json"),
        model: dir.join("model.json"),
    };

    std::fs::write(
        &fixture.corpus_identity,
        serde_json::to_vec_pretty(&corpus_identity()).unwrap(),
    )
    .unwrap();
    let body: String = LINES
        .iter()
        .map(|(line_id, book, text, section_id)| {
            format!(
                "{}\n",
                serde_json::to_string(&CorpusLineRecord {
                    line_id: *line_id,
                    line: corpus_line(book, text, *section_id),
                })
                .unwrap()
            )
        })
        .collect();
    std::fs::write(&fixture.corpus_lines, body).unwrap();
    std::fs::write(
        &fixture.chunking,
        serde_json::to_vec_pretty(chunking).unwrap(),
    )
    .unwrap();
    std::fs::write(&fixture.model, serde_json::to_vec_pretty(model).unwrap()).unwrap();

    fixture
}

/// A model identity that is complete and honest about everything except the checksum,
/// which only a real file can supply.
fn model_identity(checksum: &str, chunking: &ChunkerConfig) -> ModelIdentity {
    ModelIdentity {
        model_id: "EMD123/Otzaria-Embedding-V1-Flash-0.6B".to_string(),
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

fn cli() -> Command {
    Command::new(env!("CARGO_BIN_EXE_otzaria-semantic-search"))
}

/// `chunking_identity` is a hash of the whole configuration, so an artifact cannot be read
/// back into a recipe — the build has to be handed one, and this is what establishes it was
/// handed the right one.
///
/// Asserted through the binary in a **default build**, and deliberately: the check comes
/// before the model is opened, so it is available to a release pipeline that has no
/// inference backend compiled in, and it fails in the time it takes to hash five integers.
#[test]
fn a_recipe_that_is_not_the_one_the_model_declares_is_refused() {
    let dir = TempDir::new("recipe");
    let declared = ChunkerConfig::default();
    let model = model_identity(&"ab".repeat(32), &declared);

    // What the build will actually apply: the same recipe with one number moved, which
    // changes the text every borrowed chunk is built from.
    let applied = ChunkerConfig {
        context_window_lines: 5,
        ..ChunkerConfig::default()
    };
    assert_ne!(applied.identity(), declared.identity());
    let fixture = write_fixture(dir.path(), &model, &applied);

    let out = dir.path().join("artifact");
    let built = cli()
        .args([
            "build",
            "--corpus-identity",
            fixture.corpus_identity.to_str().unwrap(),
            "--corpus-lines",
            fixture.corpus_lines.to_str().unwrap(),
            "--model",
            fixture.model.to_str().unwrap(),
            "--chunking",
            fixture.chunking.to_str().unwrap(),
            // Never opened: the recipe is checked first, so this path does not have to
            // exist for the rejection to be the right one.
            "--model-file",
            dir.path().join("absent.gguf").to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
        ])
        .output()
        .expect("the CLI binary runs");

    assert!(
        !built.status.success(),
        "a mismatched recipe must not build"
    );
    let stderr = String::from_utf8_lossy(&built.stderr);
    assert!(
        stderr.contains("chunking_identity"),
        "the refusal must name the field that disagrees: {stderr}"
    );
    assert!(!out.exists(), "a refused build writes nothing");
}

/// A release binary must not be able to produce vectors at all — the same guarantee
/// `tests/production_backend_gate.rs` makes for the engine, at the one command whose entire
/// job is inference. A default build gets as far as opening the model and stops there.
#[cfg(not(any(feature = "mock-embedding", feature = "llama-backend")))]
#[test]
fn a_build_without_an_inference_backend_refuses_rather_than_inventing_vectors() {
    let dir = TempDir::new("no_backend");
    let chunking = ChunkerConfig::default();
    let model = model_identity(&"ab".repeat(32), &chunking);
    let fixture = write_fixture(dir.path(), &model, &chunking);

    let model_file = dir.path().join("model.gguf");
    std::fs::write(&model_file, b"not a model").unwrap();

    let out = dir.path().join("artifact");
    let built = cli()
        .args([
            "build",
            "--corpus-identity",
            fixture.corpus_identity.to_str().unwrap(),
            "--corpus-lines",
            fixture.corpus_lines.to_str().unwrap(),
            "--model",
            fixture.model.to_str().unwrap(),
            "--chunking",
            fixture.chunking.to_str().unwrap(),
            "--model-file",
            model_file.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
        ])
        .output()
        .expect("the CLI binary runs");

    assert!(
        !built.status.success(),
        "a build with no backend must not produce an artifact"
    );
    assert!(!out.exists(), "a refused build writes nothing");
}

/// Everything below is a build, and a build is inference.
#[cfg(all(feature = "mock-embedding", not(feature = "llama-backend")))]
mod with_a_backend {
    use super::*;
    use otzaria_semantic_search::distribution::importer::{ImportConfig, IndexImporter};
    use otzaria_semantic_search::distribution::package::ArtifactExpectation;
    use otzaria_semantic_search::semantic::embedding::{mock, validate_and_checksum_gguf};
    use otzaria_semantic_search::semantic::official_index::{
        LocalModel, OfficialIndexConfig, OfficialSemanticIndex,
    };

    /// The lines the recipe embeds: everything in `LINES` but the one below
    /// `min_embeddable_chars`.
    const EMBEDDED: usize = 4;

    /// Pull `Digest: <value>` out of what the CLI printed.
    fn reported_digest(stdout: &str) -> String {
        stdout
            .lines()
            .find_map(|line| line.strip_prefix("Digest:"))
            .unwrap_or_else(|| panic!("the CLI reports a digest:\n{stdout}"))
            .trim()
            .to_string()
    }

    /// A stub GGUF and the checksum an artifact must declare for it.
    fn write_model_file(dir: &Path) -> (PathBuf, String) {
        let path = dir.join("model.gguf");
        mock::write_stub_gguf(&path, 3).unwrap();
        let checksum = validate_and_checksum_gguf(&path).unwrap();
        (path, checksum)
    }

    /// Run `build`, returning what it printed.
    fn run_build(fixture: &Fixture, model_file: &Path, out: &Path, created_at: &str) -> String {
        let built = cli()
            .args([
                "build",
                "--corpus-identity",
                fixture.corpus_identity.to_str().unwrap(),
                "--corpus-lines",
                fixture.corpus_lines.to_str().unwrap(),
                "--model",
                fixture.model.to_str().unwrap(),
                "--chunking",
                fixture.chunking.to_str().unwrap(),
                "--model-file",
                model_file.to_str().unwrap(),
                "--out",
                out.to_str().unwrap(),
                "--created-at",
                created_at,
                "--batch",
                "2",
                // The stand-in's vectors mean nothing, and the builder refuses them unless
                // told in as many words. Saying so here is what keeps that refusal the
                // default everywhere else.
                "--allow-non-semantic",
            ])
            .output()
            .expect("the CLI binary runs");

        assert!(
            built.status.success(),
            "build failed:\n{}\n{}",
            String::from_utf8_lossy(&built.stdout),
            String::from_utf8_lossy(&built.stderr)
        );
        String::from_utf8_lossy(&built.stdout).to_string()
    }

    /// The stage's claim, through the binary: one command from a corpus and a model to a
    /// verified artifact, and a second command that verifies it again from nothing but the
    /// same three inputs.
    #[test]
    fn one_command_turns_a_corpus_and_a_model_into_a_verified_artifact() {
        let dir = TempDir::new("gate");
        let chunking = ChunkerConfig::default();
        let (model_file, checksum) = write_model_file(dir.path());
        let fixture = write_fixture(dir.path(), &model_identity(&checksum, &chunking), &chunking);

        let out = dir.path().join("artifact");
        let stdout = run_build(&fixture, &model_file, &out, "2026-08-08T00:00:00Z");

        assert!(
            stdout.contains(&format!("Vectors:         {EMBEDDED}")),
            "one of the {} corpus lines is below min_embeddable_chars and must not have a \
             vector:\n{stdout}",
            LINES.len()
        );

        // Verified again from the outside, with the recipe supplied rather than assumed —
        // which is the only way "complete" is a checkable claim about a corpus whose lines
        // are not all embedded.
        let validated = cli()
            .args([
                "validate",
                "--artifact",
                out.to_str().unwrap(),
                "--corpus-identity",
                fixture.corpus_identity.to_str().unwrap(),
                "--corpus-lines",
                fixture.corpus_lines.to_str().unwrap(),
                "--model",
                fixture.model.to_str().unwrap(),
                "--chunking",
                fixture.chunking.to_str().unwrap(),
            ])
            .output()
            .expect("the CLI binary runs");

        assert!(
            validated.status.success(),
            "validation failed:\n{}",
            String::from_utf8_lossy(&validated.stderr)
        );
        assert_eq!(
            reported_digest(&String::from_utf8_lossy(&validated.stdout)),
            reported_digest(&stdout),
            "a validation that re-derives the identity must reach the same digest"
        );

        // And without the recipe the same artifact is *incomplete*, because the plain
        // transcription reports every line it holds — including the one nothing embeds.
        let unplanned = cli()
            .args([
                "validate",
                "--artifact",
                out.to_str().unwrap(),
                "--corpus-identity",
                fixture.corpus_identity.to_str().unwrap(),
                "--corpus-lines",
                fixture.corpus_lines.to_str().unwrap(),
                "--model",
                fixture.model.to_str().unwrap(),
            ])
            .output()
            .expect("the CLI binary runs");

        assert!(!unplanned.status.success());
        let stderr = String::from_utf8_lossy(&unplanned.stderr);
        assert!(
            stderr.contains("no vector") && stderr.contains("4294967300"),
            "the difference must be the line the recipe skips: {stderr}"
        );
    }

    /// A published digest is a promise that the bytes are reproducible. Two builds of the
    /// same corpus, by the same model, under the same recipe, have to agree on it — and on
    /// every payload byte behind it — or announcing one means nothing.
    #[test]
    fn two_builds_of_one_corpus_produce_the_same_artifact() {
        let dir = TempDir::new("reproducible");
        let chunking = ChunkerConfig::default();
        let (model_file, checksum) = write_model_file(dir.path());
        let fixture = write_fixture(dir.path(), &model_identity(&checksum, &chunking), &chunking);

        let first = dir.path().join("first");
        let second = dir.path().join("second");
        // Different timestamps on purpose: `created_at` is excluded from the digest, so a
        // build that let it leak into the payload would fail here.
        let one = run_build(&fixture, &model_file, &first, "2026-08-08T00:00:00Z");
        let two = run_build(&fixture, &model_file, &second, "2027-01-01T12:34:56Z");

        assert_eq!(reported_digest(&one), reported_digest(&two));
        for payload in ["vectors.bin", "metadata.jsonl", "book_index.json"] {
            assert_eq!(
                std::fs::read(first.join(payload)).unwrap(),
                std::fs::read(second.join(payload)).unwrap(),
                "{payload} differs between two builds of the same corpus"
            );
        }
    }

    /// The other half of the stage: what a build writes is what the application opens.
    ///
    /// Built, installed through the importer, opened through the official read path, and
    /// queried — and the query is the *line's own text*, so a build that paired a vector
    /// with the wrong line returns the wrong id here with a perfect score.
    #[test]
    fn a_built_artifact_installs_opens_and_answers_a_query() {
        let dir = TempDir::new("runtime");
        let chunking = ChunkerConfig::default();
        let (model_file, checksum) = write_model_file(dir.path());
        let model = model_identity(&checksum, &chunking);
        let fixture = write_fixture(dir.path(), &model, &chunking);

        let source = dir.path().join("build-output");
        let stdout = run_build(&fixture, &model_file, &source, "2026-08-08T00:00:00Z");
        let digest = reported_digest(&stdout);

        let local = LocalModel {
            model_path: model_file.clone(),
            model_id: model.model_id.clone(),
            model_quantization: model.model_quantization.clone(),
            embedding_dim: model.embedding_dim,
            pooling: model.pooling.clone(),
            max_tokens: model.max_tokens,
            embedding_text_version: model.embedding_text_version,
            normalization_version: model.normalization_version,
            chunking_identity: model.chunking_identity,
        };
        let identity = otzaria_semantic_search::semantic::versioning::IndexVersion {
            corpus: corpus_identity(),
            model: model.clone(),
            store: otzaria_semantic_search::semantic::official_index::readable_store_identity(),
        };

        let target = dir.path().join("semantic_index");
        let installed = IndexImporter::new(ImportConfig {
            source_path: source,
            target_store_path: target.clone(),
        })
        .import(&ArtifactExpectation::with_published_digest(
            identity.clone(),
            digest.clone(),
        ))
        .unwrap();
        assert_eq!(installed.vectors_imported, EMBEDDED as u32);

        let index = OfficialSemanticIndex::open(OfficialIndexConfig {
            artifact_path: target,
            corpus: corpus_identity(),
            model: local,
            published_digest: Some(digest),
        })
        .unwrap();

        assert_eq!(index.identity(), &identity);
        assert_eq!(index.vector_count(), EMBEDDED as u32);
        assert_eq!(index.book_keys(), [BERACHOT, GENESIS]);

        // The lines that stand alone are embedded as themselves, so querying one is asking
        // the index for the exact vector it stored for it.
        for (line_id, book, text, _) in LINES {
            if text.chars().count() < ChunkerConfig::default().min_meaningful_chars {
                continue;
            }
            let hit = &index.search(text, 1, None).unwrap()[0];
            assert_eq!(hit.metadata.line_id, line_id);
            assert_eq!(hit.metadata.source_book_key, book);
            // Read out of the corpus at pack time, never supplied by whatever produced the
            // vector — the build has no second description of a book to offer.
            assert_eq!(hit.metadata.title, corpus_line(book, text, 1).title);
            assert_eq!(hit.metadata.reference, corpus_line(book, text, 1).reference);
            assert!((hit.similarity_score - 1.0).abs() < 1e-5);
        }

        // The skipped line has no vector, and therefore no way back into a result.
        assert!(index
            .search("או", 5, None)
            .unwrap()
            .iter()
            .all(|hit| hit.metadata.line_id != 4_294_967_300));
    }
}
