//! S4b's acceptance gate, from outside the crate.
//!
//! The stage's claim is one command: a corpus and a model in, a base vector package out,
//! which the application installs and opens. What is asserted here are the three things
//! only a producer can be held to:
//!
//! 1. **The recipe is applied, not described.** Which lines get a vector is derived by
//!    running the chunker over the corpus, and the chunker configuration is pinned to the
//!    `chunking_identity` the package declares.
//! 2. **The vectors come from the model the package names.** The build loads it and
//!    compares what the file reports against what was declared.
//! 3. **What it writes is what the runtime opens** — through the install and the official
//!    read path, with no fixture assembled by hand anywhere in between.
//!
//! Everything but the first needs an embedding backend, because a build is inference. The
//! first does not, and runs in a default build: a misconfigured recipe is refused before a
//! model is opened, which is the difference between a build that fails in a second and one
//! that fails after loading half a gigabyte of weights.

use otzaria_semantic_search::distribution::corpus::CorpusIdentity;
use otzaria_semantic_search::distribution::corpus::{CorpusLine, CorpusLineRecord};
use otzaria_semantic_search::semantic::chunker::ChunkerConfig;
use otzaria_semantic_search::semantic::versioning::{ModelIdentity, ModelPackage};
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
/// `min_embeddable_chars`, so it is not embedded at all — and a package that skips it is
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
        // The clock alone collided: macOS ticks coarser than a test takes to start.
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "otzaria_builder_gate_{name}_{}_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
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
        text: otzaria_semantic_search::semantic::versioning::TextIdentity::with_line_text_version(
            1,
        ),
        library_version: 30,
        library_release_tag: "v30-20260930120000".to_string(),
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
        family_id: "ArieLLL123/judaic-semantic-round2-onnx-zayit@1ec8dc6".to_string(),
        tokenizer_checksum: "7".repeat(64),
        query_packages: vec![ModelPackage {
            checksum: checksum.to_string(),
            quantization: "int8".to_string(),
        }],
        embedding_dim: DIM,
        pooling: "in-graph".to_string(),
        max_tokens: 512,
        embedding_text_version: 1,
        normalization_version: 1,
        chunking_identity: chunking.identity(),
    }
}

fn cli() -> Command {
    Command::new(env!("CARGO_BIN_EXE_otzaria-semantic-search"))
}

/// `chunking_identity` is a hash of the whole configuration, so a package cannot be read
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

    let out = dir.path().join("package");
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
            dir.path().join("absent.onnx").to_str().unwrap(),
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
#[cfg(not(any(feature = "mock-embedding", feature = "onnx-backend")))]
#[test]
fn a_build_without_an_inference_backend_refuses_rather_than_inventing_vectors() {
    let dir = TempDir::new("no_backend");
    let chunking = ChunkerConfig::default();
    let model = model_identity(&"ab".repeat(32), &chunking);
    let fixture = write_fixture(dir.path(), &model, &chunking);

    let model_file = dir.path().join("model.onnx");
    std::fs::write(&model_file, b"not a model").unwrap();

    let out = dir.path().join("package");
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
        "a build with no backend must not produce a package"
    );
    assert!(!out.exists(), "a refused build writes nothing");
}

/// Everything below is a build, and a build is inference — on the deterministic
/// stand-in, which has to be the selected backend: the model is a weightless stub ONNX
/// package that real inference rightly refuses, hence no `onnx-backend`.
#[cfg(all(feature = "mock-embedding", not(feature = "onnx-backend")))]
mod with_a_backend {
    use super::*;
    use otzaria_semantic_search::cancellation::CancellationToken;
    use otzaria_semantic_search::semantic::chunk_key::ChunkKey;
    use otzaria_semantic_search::semantic::embedding::{mock, EmbeddingDeployment};
    use otzaria_semantic_search::semantic::model_package::validate_model;
    use otzaria_semantic_search::semantic::official_index::{
        LocalModel, OfficialIndexConfig, OfficialSemanticIndex,
    };
    use otzaria_semantic_search::semantic::segment_set::{
        install_package, InstallExpectation, InstallSource,
    };

    /// [`model_identity`] for the stub package: its tokenizer is the stub's.
    fn stub_identity(checksum: &str, chunking: &ChunkerConfig) -> ModelIdentity {
        ModelIdentity {
            tokenizer_checksum: mock::stub_tokenizer_checksum(),
            ..model_identity(checksum, chunking)
        }
    }

    /// The lines the recipe embeds: everything in `LINES` but the one below
    /// `min_embeddable_chars`. Their texts are all different, so each is its own vector.
    const EMBEDDED: usize = 4;

    /// The value the CLI printed after `label`.
    fn reported(stdout: &str, label: &str) -> String {
        stdout
            .lines()
            .find_map(|line| line.strip_prefix(label))
            .unwrap_or_else(|| panic!("the CLI reports {label:?}:\n{stdout}"))
            .trim()
            .to_string()
    }

    /// The SHA-256 the CLI printed for the release manifest.
    fn reported_manifest_sha256(stdout: &str) -> String {
        let line = reported(stdout, "Manifest:");
        line.rsplit_once("SHA-256 ")
            .and_then(|(_, rest)| rest.strip_suffix(')'))
            .unwrap_or_else(|| panic!("a manifest digest in {line:?}"))
            .to_string()
    }

    /// A stub ONNX package and the checksum an identity must declare for it.
    fn write_model_file(dir: &Path) -> (PathBuf, String) {
        let path = mock::write_stub_onnx_package(&dir.join("model"));
        let checksum = validate_model(&path).unwrap().checksum().to_string();
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

    /// The stage's claim, through the binary and then through the application's own path:
    /// one command from a corpus and a model to a package, which installs against the
    /// digest the build announced, opens, and answers each line's own text with that line.
    ///
    /// The query is the line's text, so a build that paired a vector with the wrong line
    /// returns the wrong position here with a perfect score.
    #[test]
    fn one_command_builds_a_package_the_application_installs_and_queries() {
        let dir = TempDir::new("gate");
        let chunking = ChunkerConfig::default();
        let (model_file, checksum) = write_model_file(dir.path());
        let model = stub_identity(&checksum, &chunking);
        let fixture = write_fixture(dir.path(), &model, &chunking);

        let out = dir.path().join("package");
        let stdout = run_build(&fixture, &model_file, &out, "2026-08-08T00:00:00Z");
        assert_eq!(
            reported(&stdout, "Lines embedded:"),
            EMBEDDED.to_string(),
            "one of the {} corpus lines is below min_embeddable_chars and must not have a \
             vector:\n{stdout}",
            LINES.len()
        );
        assert_eq!(reported(&stdout, "Vectors:"), EMBEDDED.to_string());

        let identity = otzaria_semantic_search::semantic::versioning::IndexVersion {
            text: corpus_identity().text,
            model: model.clone(),
            store: otzaria_semantic_search::semantic::official_index::readable_store_identity(),
        };
        let vectors_dir = dir.path().join("vectors");
        let manifest_json = std::fs::read_to_string(out.join("release.json")).unwrap();
        let applied = install_package(
            &vectors_dir,
            &InstallSource {
                segment: &out.join("segment.oxv"),
                manifest_json: &manifest_json,
            },
            &InstallExpectation {
                identity: identity.clone(),
                published_manifest_sha256: Some(reported_manifest_sha256(&stdout)),
            },
            &CancellationToken::new(),
        )
        .unwrap();
        assert_eq!(applied.slots_added, EMBEDDED as u64);
        assert_eq!(applied.library_version, 30);

        let index = OfficialSemanticIndex::open(OfficialIndexConfig {
            vectors_dir,
            text: corpus_identity().text,
            model: LocalModel::of_family(model_file, &model, "int8"),
            deployment: EmbeddingDeployment::default(),
            scan_threads: None,
        })
        .unwrap();
        assert_eq!(index.identity(), &identity);
        assert_eq!(index.set_info().slots_live, EMBEDDED as u64);
        assert_eq!(index.book_count(), 2);

        // The lines that stand alone are embedded as themselves, so querying one is asking
        // the index for the exact vector it stored for it — at the line's position in its
        // book, which is all a segment records about where it came from.
        let cancel = CancellationToken::new();
        for (ordinal, (book, text)) in [(0, (GENESIS, LINES[0].2)), (1, (GENESIS, LINES[1].2))]
            .into_iter()
            .chain([(0, (BERACHOT, LINES[4].2))])
        {
            let hit = &index.search(text, 1, None, &cancel).unwrap()[0];
            assert_eq!(hit.key, ChunkKey::of(text));
            assert_eq!(hit.records.len(), 1);
            assert_eq!(
                (&*hit.records[0].book, hit.records[0].hint),
                (book, ordinal)
            );
            assert!(hit.score > 0.99, "{}", hit.score);
        }

        // The skipped line has no vector, and therefore no way back into a result.
        assert!(index
            .search("או", 5, None, &cancel)
            .unwrap()
            .iter()
            .flat_map(|hit| &hit.records)
            .all(|record| (&*record.book, record.hint) != (GENESIS, 3)));
    }

    /// A published digest is a promise that the bytes are reproducible. Two builds of the
    /// same corpus, by the same model, under the same recipe, have to agree on the segment
    /// and on the package digest — or announcing one means nothing.
    #[test]
    fn two_builds_of_one_corpus_produce_the_same_package() {
        let dir = TempDir::new("reproducible");
        let chunking = ChunkerConfig::default();
        let (model_file, checksum) = write_model_file(dir.path());
        let fixture = write_fixture(dir.path(), &stub_identity(&checksum, &chunking), &chunking);

        let first = dir.path().join("first");
        let second = dir.path().join("second");
        // Different timestamps on purpose: `created_at` is excluded from the package
        // digest, so a build that let it leak into the segment would fail here.
        let one = run_build(&fixture, &model_file, &first, "2026-08-08T00:00:00Z");
        let two = run_build(&fixture, &model_file, &second, "2027-01-01T12:34:56Z");

        assert_eq!(
            reported(&one, "Package digest:"),
            reported(&two, "Package digest:")
        );
        assert_eq!(
            std::fs::read(first.join("segment.oxv")).unwrap(),
            std::fs::read(second.join("segment.oxv")).unwrap(),
            "the segment differs between two builds of the same corpus"
        );
    }
}

/// The whole build, once, against the real model — not the deterministic stand-in.
///
/// Every other test here runs on a backend that echoes its own configuration back, so the
/// half of the model check that only real weights can drive is never exercised: a declared
/// width, pooling or token cap that the model does not have. This is also the only place
/// the recipe is applied to text a real tokenizer sees — text recipe 2, the role prefixes
/// the production identity declares.
///
/// `#[ignore]`d and **skips loudly**, matching the rest of the crate's real-model tests:
/// the ordinary suite stays green without the gated download, and CI's `golden-onnx` job
/// runs it with `--ignored` after fetching the package and the runtime.
#[cfg(all(feature = "onnx-backend", not(feature = "mock-embedding")))]
#[test]
#[ignore = "needs a Meivin graph and ONNX Runtime; set OTZARIA_TEST_ONNX_MODEL and OTZARIA_ONNX_RUNTIME"]
fn the_real_model_builds_a_package_that_installs_and_answers() {
    use otzaria_semantic_search::cancellation::CancellationToken;
    use otzaria_semantic_search::distribution::builder::{
        build, BuildRequest, RELEASE_MANIFEST_FILENAME, SEGMENT_FILENAME,
    };
    use otzaria_semantic_search::distribution::corpus::JsonlCorpus;
    use otzaria_semantic_search::semantic::embedding::EmbeddingDeployment;
    use otzaria_semantic_search::semantic::model_package::validate_model;
    use otzaria_semantic_search::semantic::official_index::{
        LocalModel, OfficialIndexConfig, OfficialSemanticIndex,
    };
    use otzaria_semantic_search::semantic::segment_set::{
        install_package, InstallExpectation, InstallSource,
    };

    let Ok(model_file) = std::env::var("OTZARIA_TEST_ONNX_MODEL") else {
        println!(
            "SKIPPED: OTZARIA_TEST_ONNX_MODEL is not set. This test needs one of the Meivin \
             Round 2 graphs, with its tokenizer.json beside it."
        );
        return;
    };
    let model_file = PathBuf::from(model_file);
    if !model_file.exists() {
        println!("SKIPPED: OTZARIA_TEST_ONNX_MODEL points at {model_file:?}, which does not exist");
        return;
    }
    if std::env::var_os("OTZARIA_ONNX_RUNTIME").is_none() {
        println!(
            "SKIPPED: OTZARIA_ONNX_RUNTIME is not set; running the graph needs an ONNX Runtime \
             shared library"
        );
        return;
    }

    let dir = TempDir::new("real_model");
    let chunking = ChunkerConfig {
        embedding_text_version: 2,
        ..ChunkerConfig::default()
    };
    // The dimension, the pooling and the token cap are the model's, not this test's:
    // declaring anything else is exactly what the build is supposed to refuse, and
    // asserting that here would be asserting the check rather than the build.
    let package = validate_model(&model_file).unwrap();
    let tokenizer = package
        .files()
        .iter()
        .find(|file| file.relpath == "tokenizer.json")
        .unwrap()
        .sha256
        .clone();
    let model = ModelIdentity {
        tokenizer_checksum: tokenizer,
        embedding_dim: 256,
        max_tokens: 256,
        embedding_text_version: 2,
        ..model_identity(package.checksum(), &chunking)
    };
    let fixture = write_fixture(dir.path(), &model, &chunking);

    let out = dir.path().join("package");
    let report = build(
        BuildRequest {
            output_path: out.clone(),
            model_path: model_file.clone(),
            model: model.clone(),
            chunking: chunking.clone(),
            created_at: "2026-08-09T00:00:00Z".to_string(),
            batch_size: 4,
            clip_q: 1.0,
            // Real inference: the gate this flag exists for must stay shut.
            allow_non_semantic_backend: false,
        },
        &JsonlCorpus::load(&fixture.corpus_identity, &fixture.corpus_lines).unwrap(),
    )
    .expect("the real model builds a package");

    assert_eq!(
        report.planned_lines, 4,
        "one line is below min_embeddable_chars"
    );
    assert_eq!(report.manifest.counts.slots, 4);
    assert_eq!(report.manifest.identity.model.embedding_dim, 256);

    // Installed against the digest the build announced, and opened by the application's
    // path with the same package for queries.
    let vectors_dir = dir.path().join("vectors");
    let manifest_json = std::fs::read_to_string(out.join(RELEASE_MANIFEST_FILENAME)).unwrap();
    install_package(
        &vectors_dir,
        &InstallSource {
            segment: &out.join(SEGMENT_FILENAME),
            manifest_json: &manifest_json,
        },
        &InstallExpectation {
            identity: report.manifest.identity.clone(),
            published_manifest_sha256: Some(report.manifest_sha256.clone()),
        },
        &CancellationToken::new(),
    )
    .unwrap();
    let index = OfficialSemanticIndex::open(OfficialIndexConfig {
        vectors_dir,
        text: corpus_identity().text,
        model: LocalModel::of_family(model_file, &model, "int8"),
        deployment: EmbeddingDeployment::default(),
        scan_threads: None,
    })
    .unwrap();
    let hits = index
        .search(LINES[4].2, 10, None, &CancellationToken::new())
        .unwrap();
    assert_eq!(hits.len(), 4, "every stored vector is a candidate");
    assert_eq!(
        &*hits[0].records[0].book, BERACHOT,
        "a passage's own text, as a query, finds the passage"
    );
}
