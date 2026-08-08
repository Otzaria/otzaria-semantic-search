//! The official read-only runtime as the application sees it.
//!
//! `otzaria_search_engine` is the consumer: it opens the Tantivy index, opens the
//! installed artifact beside it, and serves queries from both. These tests run that path
//! through the public API only — the artifact is built the way the packer will build it
//! (the store backend writes the payload, then the metadata describes it), installed
//! through [`IndexImporter`], opened through [`OfficialSemanticIndex`], and queried
//! through [`OtzariaHybridEngine`].
//!
//! What is asserted here and not in the crate's own tests is the *product* surface: that
//! a semantic query over an installed artifact reaches fusion and comes back with the
//! `line_id` the caller has to hydrate, and that every build-side operation the seam still
//! exposes refuses by name instead of quietly doing nothing.
//!
//! Driving a query end to end needs an embedding backend, and needs the deterministic
//! stand-in to be the one actually selected — the fixture's model is a weightless stub
//! that real inference rightly refuses — hence `mock-embedding` without `llama-backend`,
//! as in `tests/hybrid_integration_test.rs`.

#![cfg(all(feature = "mock-embedding", not(feature = "llama-backend")))]

use otzaria_semantic_search::api::hybrid_search::{OtzariaHybridEngine, SearchRequest};
use otzaria_semantic_search::distribution::importer::{ImportConfig, IndexImporter};
use otzaria_semantic_search::distribution::package::{
    ArtifactExpectation, IndexPackage, PackageManifest, PayloadDescriptor,
};
use otzaria_semantic_search::hybrid::coordinator::HybridCoordinator;
use otzaria_semantic_search::semantic::backend::MockHashBackend;
use otzaria_semantic_search::semantic::embedding::{mock, validate_and_checksum_gguf};
use otzaria_semantic_search::semantic::official_index::{
    readable_store_identity, LocalModel, OfficialIndexConfig, OfficialSemanticIndex,
};
use otzaria_semantic_search::semantic::store_backend::VectorStoreBackend;
use otzaria_semantic_search::semantic::types::{
    ContentFingerprint, LexicalCandidate, SearchMode, VectorMetadata,
};
use otzaria_semantic_search::semantic::versioning::{CorpusIdentity, IndexVersion, ModelIdentity};
use otzaria_semantic_search::semantic::zevc_store::{
    ZevcStore, ZevcStoreConfig, SNAPSHOT_FILENAMES,
};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

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
            "otzaria_official_runtime_{name}_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// The corpus identity a caller takes from the Tantivy index it has open.
fn corpus() -> CorpusIdentity {
    CorpusIdentity {
        corpus_id: "4d".repeat(32),
        library_version: "otzaria-library-2026-08".to_string(),
        tantivy_schema_version: 3,
        document_id_scheme_version: 1,
    }
}

fn local_model(model_path: &Path) -> LocalModel {
    LocalModel {
        model_path: model_path.to_path_buf(),
        model_id: "EMD123/Otzaria-Embedding-V1-Flash-0.6B".to_string(),
        model_quantization: "Q4_K_M".to_string(),
        embedding_dim: DIM,
        pooling: "last-token".to_string(),
        max_tokens: 512,
        embedding_text_version: 1,
        normalization_version: 1,
        chunking_identity: 0x0BAD_C0DE,
    }
}

fn identity(model_path: &Path) -> IndexVersion {
    let model = local_model(model_path);
    IndexVersion {
        corpus: corpus(),
        model: ModelIdentity {
            model_id: model.model_id,
            // What the builder had, computed here from the file the runtime will load.
            model_checksum: validate_and_checksum_gguf(model_path).unwrap(),
            model_quantization: model.model_quantization,
            embedding_backend: MockHashBackend::ID.to_string(),
            embedding_dim: model.embedding_dim,
            pooling: model.pooling,
            max_tokens: model.max_tokens,
            embedding_text_version: model.embedding_text_version,
            normalization_version: model.normalization_version,
            chunking_identity: model.chunking_identity,
        },
        store: readable_store_identity(),
    }
}

fn metadata(line_id: u64, book: &str) -> VectorMetadata {
    VectorMetadata {
        semantic_id: format!("{book}#{line_id}"),
        source_book_key: book.to_string(),
        source_doc_key: format!("{book}#{line_id}"),
        line_id,
        section_id: line_id,
        line_hash: line_id.wrapping_mul(31),
        chunk_hash: format!("chunk-{line_id}"),
        content_hash: 0,
        reference: format!("הפניה {line_id}"),
        segment: 0,
        is_pdf: false,
        title: "ספר בדיקה".to_string(),
        facets: vec!["/מקרא/תורה".to_string()],
    }
}

/// Install an artifact into `dir`, and return the model it was built with and the
/// directory the runtime opens.
fn install(dir: &TempDir) -> (PathBuf, PathBuf) {
    let model_path = dir.path().join("model.gguf");
    mock::write_stub_gguf(&model_path, 3).unwrap();

    let source = dir.path().join("build-output");
    let store = ZevcStore::open_or_create(ZevcStoreConfig {
        db_path: source.clone(),
        embedding_dim: DIM,
        collection_name: "chunks".to_string(),
        auto_persist: false,
    })
    .unwrap();
    store
        .insert_batch(
            LINES
                .iter()
                .map(|(line_id, book, text)| {
                    (metadata(*line_id, book), mock::hash_embedding(text, DIM))
                })
                .collect(),
        )
        .unwrap();
    store.commit().unwrap();

    let payloads: BTreeMap<String, PayloadDescriptor> = SNAPSHOT_FILENAMES
        .iter()
        .map(|name| {
            (
                (*name).to_string(),
                PayloadDescriptor::of_file(&source.join(name)).unwrap(),
            )
        })
        .collect();
    let package = IndexPackage {
        manifest: PackageManifest::new(
            identity(&model_path),
            "2026-08-06T00:00:00Z".to_string(),
            2,
            LINES.len() as u32,
            payloads.values().map(|payload| payload.size_bytes).sum(),
        ),
        payloads,
    };
    IndexPackage::write(&source, &package).unwrap();

    let target = dir.path().join("semantic_index");
    let result = IndexImporter::new(ImportConfig {
        source_path: source,
        target_store_path: target.clone(),
    })
    .import(&ArtifactExpectation::with_published_digest(
        identity(&model_path),
        package.digest(),
    ))
    .unwrap();
    assert_eq!(result.vectors_imported, LINES.len() as u32);

    (model_path, target)
}

fn open_official(target: &Path, model_path: &Path) -> OfficialSemanticIndex {
    OfficialSemanticIndex::open(OfficialIndexConfig {
        artifact_path: target.to_path_buf(),
        corpus: corpus(),
        model: local_model(model_path),
        published_digest: None,
    })
    .unwrap()
}

fn lexical(line_id: u64, text: &str, score: f32) -> LexicalCandidate {
    LexicalCandidate {
        line_id,
        section_id: line_id,
        text: text.to_string(),
        title: "ספר בדיקה".to_string(),
        reference: format!("הפניה {line_id}"),
        bm25_score: score,
        segment: 0,
        is_pdf: false,
        file_path: "books/test.txt".to_string(),
        line_hash: line_id,
    }
}

/// Name, length **and SHA-256** of every file in `dir`.
///
/// The hash is the point. Without it, "opening changed nothing" would still hold after a
/// rewrite that kept every length — which is precisely the class of change this stage is
/// about, so a length-only fingerprint would assert the weakest version of the claim.
fn fingerprint(dir: &Path) -> Vec<(String, u64, String)> {
    let mut entries: Vec<(String, u64, String)> = fs::read_dir(dir)
        .unwrap()
        .map(|entry| {
            let entry = entry.unwrap();
            let bytes = fs::read(entry.path()).unwrap();
            (
                entry.file_name().to_string_lossy().into_owned(),
                bytes.len() as u64,
                format!("{:x}", Sha256::digest(&bytes)),
            )
        })
        .collect();
    entries.sort();
    entries
}

/// The stage's headline claim, through the seam Otzaria links against: a query over an
/// installed artifact produces the `line_id` the caller hydrates, in both modes that
/// consult the semantic path.
#[test]
fn a_query_over_an_installed_artifact_returns_the_line_id_it_was_built_from() {
    let dir = TempDir::new("query");
    let (model_path, target) = install(&dir);
    let api = OtzariaHybridEngine::new(HybridCoordinator::with_official_index(open_official(
        &target,
        &model_path,
    )));

    let status = api.get_semantic_status();
    assert!(status.available);
    assert!(status.vectors_persisted);
    assert_eq!(status.vector_count, LINES.len() as u32);
    assert_eq!(status.indexed_book_count, 2);
    assert!(status.needs_full_reindex.is_none());

    let (line_id, _, text) = LINES[3];
    let semantic = api
        .search(SearchRequest {
            query: text.to_string(),
            // Deliberately supplied and deliberately discarded: a semantic-only request
            // must not be answered with BM25 wearing a semantic label.
            lexical_candidates: vec![lexical(999, "שורה לקסיקלית בלבד", 99.0)],
            force_mode: Some(SearchMode::SemanticOnly),
            ..Default::default()
        })
        .unwrap();

    assert_eq!(semantic.search_mode, SearchMode::SemanticOnly);
    assert!(semantic.semantic_available);
    assert_eq!(semantic.results[0].id, line_id);
    assert!(semantic.results.iter().all(|item| item.id != 999));

    // Hybrid: both sources reach fusion over the same id space.
    let hybrid = api
        .search(SearchRequest {
            query: text.to_string(),
            lexical_candidates: vec![lexical(line_id, text, 18.0)],
            force_mode: Some(SearchMode::Hybrid),
            ..Default::default()
        })
        .unwrap();

    assert_eq!(hybrid.search_mode, SearchMode::Hybrid);
    assert!(hybrid.fallback_reason.is_none());
    let fused = &hybrid.results[0];
    assert_eq!(fused.id, line_id);
    assert!(fused.lexical_score.is_some());
    assert!(fused.semantic_score.is_some());
}

/// An artifact is not something this device may write to, and the seam has to say so
/// rather than report a no-op: a caller that read the refusal as "no semantic index"
/// would offer indexing the library as the fix, which is what the product contract rules
/// out.
#[test]
fn every_build_side_operation_is_refused_on_an_installed_artifact() {
    let dir = TempDir::new("refusals");
    let (model_path, target) = install(&dir);
    let api = OtzariaHybridEngine::new(HybridCoordinator::with_official_index(open_official(
        &target,
        &model_path,
    )));

    let before = fingerprint(&target);
    let refusals: Vec<(&str, String)> = vec![
        (
            "index_books",
            api.index_books(&[]).expect_err("indexing must be refused"),
        ),
        (
            "remove_semantic_books",
            api.remove_semantic_books(&[GENESIS.to_string()])
                .expect_err("removal must be refused"),
        ),
        (
            "reset_semantic_index",
            api.reset_semantic_index()
                .expect_err("a reset must be refused"),
        ),
        (
            "semantic_index_diff",
            api.get_semantic_index_diff(&HashMap::from([(
                GENESIS.to_string(),
                ContentFingerprint::from_lexical_hash(7),
            )]))
            .expect_err("a diff must be refused"),
        ),
        (
            "semantic_index_diff",
            api.get_semantic_index_diff_from_lexical_hashes(&HashMap::from([(
                GENESIS.to_string(),
                7u64,
            )]))
            .expect_err("a diff must be refused"),
        ),
    ];

    for (operation, message) in refusals {
        assert!(
            message.contains(operation) && message.contains("read-only"),
            "{operation}: unhelpful refusal {message:?}"
        );
    }

    // Nothing was half-done, and the index still answers: a refusal is not a failure
    // state the caller has to recover from.
    assert_eq!(fingerprint(&target), before);
    assert_eq!(
        api.search(SearchRequest {
            query: LINES[0].2.to_string(),
            force_mode: Some(SearchMode::SemanticOnly),
            ..Default::default()
        })
        .unwrap()
        .results[0]
            .id,
        LINES[0].0
    );
}

/// Restart: the same directory opens again and answers the same query, with nothing
/// rebuilt. This is what `vectors_persisted` is a claim about.
#[test]
fn a_restart_opens_the_same_artifact_without_rebuilding_anything() {
    let dir = TempDir::new("restart");
    let (model_path, target) = install(&dir);

    let before = fingerprint(&target);
    let (line_id, _, text) = LINES[1];

    for _ in 0..2 {
        let index = open_official(&target, &model_path);
        assert_eq!(index.vector_count(), LINES.len() as u32);
        // Sorted, not insertion-ordered: `HashMap` order changes between runs.
        assert_eq!(index.book_keys(), [BERACHOT, GENESIS]);
        assert_eq!(
            index.search(text, 2, None).unwrap()[0].metadata.line_id,
            line_id
        );
        assert_eq!(
            fingerprint(&target),
            before,
            "opening an artifact must not write to it"
        );
    }
}
