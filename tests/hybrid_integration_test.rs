//! End-to-end tests for the Otzaria hybrid semantic search engine: a full index →
//! restart → search cycle, mode selection through [`SearchRequest`], and recovery
//! from a damaged index directory.
//!
//! Driving the engine end to end requires an embedding backend, and requires the
//! deterministic stand-in to be the one actually *selected* — every fixture builds
//! its model with [`mock::write_stub_gguf`], a weightless stub that real inference
//! rightly refuses — hence `mock-embedding` without `llama-backend`.
//!
//! Consequently the hybrid pipeline is not exercised in a build holding both
//! backends. That is acceptable: fusion, grouping, paging, filters and the manifest
//! lifecycle are backend-agnostic. The selection rule that such a build does need
//! asserted lives in `tests/backend_selection.rs`, and the opposite side of the gate
//! — a default build refusing to embed at all — in
//! `tests/production_backend_gate.rs`.

#![cfg(all(feature = "mock-embedding", not(feature = "llama-backend")))]

use otzaria_semantic_search::api::hybrid_search::{OtzariaHybridEngine, SearchRequest};
use otzaria_semantic_search::hybrid::coordinator::HybridCoordinator;
use otzaria_semantic_search::semantic::embedding::mock;
use otzaria_semantic_search::semantic::engine::{SemanticConfig, SemanticEngine};
use otzaria_semantic_search::semantic::manifest::SemanticManifest;
use otzaria_semantic_search::semantic::store::VectorStoreConfig;
use otzaria_semantic_search::semantic::types::{
    BookForIndexing, BookLine, ContentFingerprint, GroupingMode, LexicalCandidate, ResultSource,
    SearchFilters, SearchMode,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const GENESIS: &str = "otzaria/tanach/genesis.txt";
const BERACHOT: &str = "otzaria/mishna/berachot.txt";

const LINE_ONE: &str = "בראשית ברא אלהים את השמים ואת הארץ";
const LINE_TWO: &str = "והארץ היתה תהו ובהו וחשך על פני תהום";
const LINE_THREE: &str = "ויאמר אלהים יהי אור ויהי אור מאיר";
const BERACHOT_LINE: &str = "מאימתי קורין את שמע בערבית משעה שהכהנים נכנסין לאכול בתרומתן";

struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "otzaria_integration_test_{name}_{}",
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

/// A configuration rooted in `dir`, with a small embedding dimension for speed.
fn config_at(dir: &TempDir) -> SemanticConfig {
    let model_path = dir.path().join("model.gguf");
    mock::write_stub_gguf(&model_path, 3).unwrap();

    let root = dir.path().join("semantic");
    SemanticConfig {
        root_dir: root.clone(),
        model_path,
        embedding_dim: 64,
        store: VectorStoreConfig {
            db_path: root.join("vectors"),
            embedding_dim: 64,
            collection_name: "chunks".to_string(),
        },
        ..Default::default()
    }
}

fn genesis_book() -> BookForIndexing {
    BookForIndexing {
        source_book_key: GENESIS.to_string(),
        title: "בראשית".to_string(),
        content_fingerprint: 987_654,
        is_pdf: false,
        topics: "/מקרא/תורה/בראשית".to_string(),
        extra_facets: vec![
            "/author/משה רבנו".to_string(),
            "/author/עזרא הסופר".to_string(),
            "/era/תנך".to_string(),
            "/base".to_string(),
        ],
        lines: vec![
            BookLine {
                line_id: 1,
                section_id: 100,
                text: LINE_ONE.to_string(),
                line_hash: 11_111,
                reference: "בראשית א:א".to_string(),
                segment: 1,
            },
            BookLine {
                line_id: 2,
                section_id: 100,
                text: LINE_TWO.to_string(),
                line_hash: 22_222,
                reference: "בראשית א:ב".to_string(),
                segment: 2,
            },
            BookLine {
                line_id: 3,
                section_id: 101,
                text: LINE_THREE.to_string(),
                line_hash: 33_333,
                reference: "בראשית א:ג".to_string(),
                segment: 3,
            },
        ],
    }
}

/// A PDF book carrying the caller's own canonical fingerprint rather than Tantivy's
/// `0`. Leaving it `0` is legal but means "I cannot vouch for this", and such a book
/// is re-examined on every diff.
fn berachot_book() -> BookForIndexing {
    let mut book = berachot_without_fingerprint();
    book.content_fingerprint = caller_fingerprint(&book, PDF_SIGNATURE).as_raw();
    book
}

/// The file's own signature folded together with the metadata that ends up inside
/// every vector — a signature over the bytes alone cannot notice a renamed book or a
/// corrected author.
fn caller_fingerprint(book: &BookForIndexing, source_signature: u64) -> ContentFingerprint {
    ContentFingerprint::canonical(
        source_signature,
        &book.title,
        &book.topics,
        &book.extra_facets,
        book.is_pdf,
    )
}

fn berachot_without_fingerprint() -> BookForIndexing {
    BookForIndexing {
        source_book_key: BERACHOT.to_string(),
        title: "מסכת ברכות".to_string(),
        content_fingerprint: 0,
        is_pdf: true,
        topics: "/תלמוד/משנה/ברכות".to_string(),
        extra_facets: vec!["/era/תנאים".to_string()],
        lines: vec![BookLine {
            line_id: 501,
            section_id: 900,
            text: BERACHOT_LINE.to_string(),
            line_hash: 55_555,
            reference: "ברכות א:א".to_string(),
            segment: 1,
        }],
    }
}

fn facet_filter(paths: &[&str]) -> SearchFilters {
    SearchFilters {
        facets: Some(paths.iter().map(|p| p.to_string()).collect()),
        ..Default::default()
    }
}

fn lexical_hit(line_id: u64, text: &str, bm25_score: f32) -> LexicalCandidate {
    LexicalCandidate {
        title: "בראשית".to_string(),
        reference: format!("בראשית א:{line_id}"),
        text: text.to_string(),
        line_id,
        section_id: 100,
        line_hash: line_id * 11_111,
        segment: line_id,
        is_pdf: false,
        file_path: GENESIS.to_string(),
        bm25_score,
    }
}

/// Stands in for a caller-computed PDF extraction revision, which the lexical index
/// cannot supply: it records `contentHash = 0` for PDFs.
const PDF_SIGNATURE: u64 = 0xBEEF_CAFE;

/// The fingerprint map a caller with real signatures would supply.
fn library_fingerprints() -> HashMap<String, ContentFingerprint> {
    HashMap::from([
        (
            GENESIS.to_string(),
            ContentFingerprint::from_lexical_hash(genesis_book().content_fingerprint),
        ),
        (
            BERACHOT.to_string(),
            caller_fingerprint(&berachot_book(), PDF_SIGNATURE),
        ),
    ])
}

/// An API handle over a freshly indexed two-book library.
fn indexed_api(config: SemanticConfig) -> OtzariaHybridEngine {
    let mut engine = SemanticEngine::open(config).unwrap();
    engine
        .index_books(&[genesis_book(), berachot_book()])
        .unwrap();
    OtzariaHybridEngine::new(HybridCoordinator::new(Some(engine)))
}

// ───────────────────────── end-to-end ─────────────────────────

#[test]
fn indexes_a_library_and_serves_a_hybrid_search_through_the_api() {
    let dir = TempDir::new("end_to_end");
    let api = indexed_api(config_at(&dir));

    let status = api.get_semantic_status();
    assert!(status.available);
    assert_eq!(status.indexed_book_count, 2);
    assert_eq!(status.vector_count, 4);
    assert_eq!(status.embedding_backend.as_deref(), Some("mock-hash-v1"));
    assert!(status.needs_full_reindex.is_none());

    let result = api
        .search(SearchRequest {
            query: LINE_ONE.to_string(),
            lexical_candidates: vec![lexical_hit(1, LINE_ONE, 15.5)],
            limit: Some(10),
            grouping: Some(GroupingMode::SameSection),
            ..Default::default()
        })
        .unwrap();

    assert_eq!(result.search_mode, SearchMode::Hybrid);
    assert!(result.semantic_available);
    assert!(result.fallback_reason.is_none());
    assert!(!result.results.is_empty());
    assert!(result.total_count > 0);

    let top = &result.results[0];
    assert_eq!(top.id, 1);
    assert_eq!(top.source, ResultSource::Both);
    assert!(top.lexical_score.is_some());
    assert!(top.semantic_score.is_some());
    assert!(!top.needs_hydration);
    assert_eq!(top.text, LINE_ONE);
}

#[test]
fn a_semantic_only_search_returns_semantic_hits_and_no_lexical_ones() {
    let dir = TempDir::new("semantic_only");
    let api = indexed_api(config_at(&dir));

    let result = api
        .search(SearchRequest {
            query: LINE_TWO.to_string(),
            // A line only the lexical engine knows about.
            lexical_candidates: vec![lexical_hit(4242, "שורה שאינה מאונדקסת סמנטית", 99.0)],
            force_mode: Some(SearchMode::SemanticOnly),
            ..Default::default()
        })
        .unwrap();

    assert_eq!(result.search_mode, SearchMode::SemanticOnly);
    assert!(result.semantic_available);
    assert!(!result.results.is_empty());
    assert!(
        result.results.iter().all(|r| r.id != 4242),
        "a lexical-only candidate must never appear in semantic-only mode"
    );
    assert!(result
        .results
        .iter()
        .all(|r| r.source == ResultSource::Semantic));

    // Semantic hits carry metadata but no line body; the flag is what says so.
    for item in &result.results {
        assert!(item.needs_hydration);
        assert!(item.text.is_empty());
        assert!(!item.reference.is_empty(), "the reference is still usable");
    }

    assert_eq!(result.results[0].id, 2, "the exact line ranks first");
}

#[test]
fn a_lexical_only_search_never_touches_the_semantic_index() {
    let dir = TempDir::new("lexical_only");
    let api = indexed_api(config_at(&dir));

    let result = api
        .search(SearchRequest {
            query: LINE_ONE.to_string(),
            lexical_candidates: vec![lexical_hit(1, LINE_ONE, 15.5)],
            force_mode: Some(SearchMode::LexicalOnly),
            ..Default::default()
        })
        .unwrap();

    assert_eq!(result.search_mode, SearchMode::LexicalOnly);
    assert_eq!(result.results.len(), 1);
    assert_eq!(result.results[0].source, ResultSource::Lexical);
    assert!(result.results[0].semantic_score.is_none());
    assert!(result.fallback_reason.is_none());
}

#[test]
fn lexical_search_keeps_working_when_the_semantic_model_is_missing() {
    let dir = TempDir::new("degraded");
    let mut config = config_at(&dir);
    config.model_path = dir.path().join("never-downloaded.gguf");

    let engine = SemanticEngine::open(config).unwrap();
    let api = OtzariaHybridEngine::new(HybridCoordinator::new(Some(engine)));

    let result = api
        .search(SearchRequest {
            query: LINE_ONE.to_string(),
            lexical_candidates: vec![lexical_hit(1, LINE_ONE, 15.5)],
            ..Default::default()
        })
        .unwrap();

    assert_eq!(
        result.search_mode,
        SearchMode::LexicalOnly,
        "a hybrid request must degrade, not fail"
    );
    assert!(!result.semantic_available);
    assert!(
        result.fallback_reason.is_some(),
        "the degradation must be visible to the caller"
    );
    assert_eq!(result.results.len(), 1);
    assert_eq!(result.results[0].text, LINE_ONE);
}

// ───────────────────────── reopen ─────────────────────────

/// The dangerous state this pins down: a manifest that survives a restart while
/// the vectors it describes do not. The index would report "up to date" while
/// every semantic query came back empty.
#[test]
fn reopening_the_index_never_claims_books_whose_vectors_are_gone() {
    let dir = TempDir::new("reopen");
    let config = config_at(&dir);

    {
        let api = indexed_api(config.clone());
        assert_eq!(api.get_semantic_status().vector_count, 4);
    }

    let engine = SemanticEngine::open(config.clone()).unwrap();
    let api = OtzariaHybridEngine::new(HybridCoordinator::new(Some(engine)));

    let status = api.get_semantic_status();
    assert_eq!(status.vector_count, 0);
    assert_eq!(
        status.indexed_book_count, 0,
        "book records must not outlive their vectors"
    );
    assert!(!status.available);
    assert!(!status.vectors_persisted);

    let fingerprints = library_fingerprints();
    let diff = api.get_semantic_index_diff(&fingerprints).unwrap().unwrap();
    assert!(!diff.is_up_to_date());
    assert_eq!(diff.books_to_index(), 2);
    assert!(!diff.needs_full_rebuild(), "the configuration is unchanged");

    let summary = api
        .index_books(&[genesis_book(), berachot_book()])
        .unwrap()
        .expect("the semantic path is enabled");
    assert_eq!(summary.books_indexed, 2);
    assert_eq!(summary.chunks_written, 4);
    assert!(api.get_semantic_status().available);
    assert!(api
        .get_semantic_index_diff(&fingerprints)
        .unwrap()
        .unwrap()
        .is_up_to_date());
}

#[test]
fn reopening_preserves_the_index_configuration() {
    let dir = TempDir::new("reopen_config");
    let config = config_at(&dir);

    {
        let _ = indexed_api(config.clone());
    }

    let manifest = SemanticManifest::load(&config.root_dir).unwrap();
    assert_eq!(manifest.embedding_model_id, config.embedding_model_id);
    assert_eq!(manifest.embedding_dim, 64);
    assert_eq!(manifest.embedding_backend.as_deref(), Some("mock-hash-v1"));
    assert_eq!(
        manifest.model_checksum.as_ref().map(String::len),
        Some(64),
        "the model checksum must be recorded so a swapped file is detectable"
    );

    let engine = SemanticEngine::open(config).unwrap();
    assert!(
        engine.incompatibilities().is_empty(),
        "reopening with an unchanged configuration must not look incompatible"
    );
    assert!(engine.status().last_error.is_none());
}

// ───────────────────────── re-index ─────────────────────────

/// The bug: a re-index inserted new chunks without deleting the old ones, so a
/// line removed from a book kept its vector and kept being returned.
#[test]
fn reindexing_a_book_removes_the_vectors_of_lines_that_no_longer_exist() {
    let dir = TempDir::new("reindex");
    let config = config_at(&dir);
    let api = indexed_api(config);

    assert_eq!(api.get_semantic_status().vector_count, 4);

    // Genesis loses its third line and its content hash changes.
    let mut shrunk = genesis_book();
    shrunk.lines.pop();
    shrunk.content_fingerprint = 111_222;
    let summary = api.index_books(&[shrunk.clone()]).unwrap().unwrap();
    assert_eq!(summary.books_indexed, 1);
    assert_eq!(summary.chunks_written, 2);

    let status = api.get_semantic_status();
    assert_eq!(
        status.vector_count, 3,
        "2 remaining Genesis lines + 1 Berachot line"
    );
    assert_eq!(status.indexed_book_count, 2, "Berachot is untouched");

    let result = api
        .search(SearchRequest {
            query: LINE_THREE.to_string(),
            force_mode: Some(SearchMode::SemanticOnly),
            limit: Some(20),
            ..Default::default()
        })
        .unwrap();
    assert!(
        result.results.iter().all(|r| r.id != 3),
        "the removed line must not survive a re-index"
    );

    let fingerprints = HashMap::from([
        (
            GENESIS.to_string(),
            ContentFingerprint::from_lexical_hash(111_222),
        ),
        (
            BERACHOT.to_string(),
            caller_fingerprint(&berachot_book(), PDF_SIGNATURE),
        ),
    ]);
    assert!(api
        .get_semantic_index_diff(&fingerprints)
        .unwrap()
        .unwrap()
        .is_up_to_date());
}

#[test]
fn reindexing_unchanged_content_does_not_duplicate_vectors() {
    let dir = TempDir::new("reindex_idempotent");
    let api = indexed_api(config_at(&dir));

    for _ in 0..3 {
        api.index_books(&[genesis_book(), berachot_book()]).unwrap();
    }

    assert_eq!(api.get_semantic_status().vector_count, 4);
    assert_eq!(api.get_semantic_status().indexed_book_count, 2);
}

/// The failure a file signature alone cannot catch: the PDF is byte-identical and the
/// library corrected its author, which every vector carries and filters on.
#[test]
fn correcting_a_pdfs_author_is_visible_at_diff_time() {
    let dir = TempDir::new("pdf_metadata_diff");
    let api = indexed_api(config_at(&dir));

    let mut corrected = berachot_without_fingerprint();
    corrected.extra_facets = vec![
        "/era/תנאים".to_string(),
        "/author/רבי יהודה הנשיא".to_string(),
    ];
    corrected.content_fingerprint = caller_fingerprint(&corrected, PDF_SIGNATURE).as_raw();
    assert_ne!(
        corrected.content_fingerprint,
        berachot_book().content_fingerprint,
        "the metadata is part of the fingerprint, so this must differ"
    );

    let fingerprints = HashMap::from([
        (
            GENESIS.to_string(),
            ContentFingerprint::from_lexical_hash(genesis_book().content_fingerprint),
        ),
        (
            BERACHOT.to_string(),
            caller_fingerprint(&corrected, PDF_SIGNATURE),
        ),
    ]);

    let diff = api.get_semantic_index_diff(&fingerprints).unwrap().unwrap();
    assert_eq!(
        diff.changed_books,
        vec![BERACHOT.to_string()],
        "a metadata-only correction must be reported before the lines are loaded"
    );
    assert!(diff.unverifiable_books.is_empty());
    assert!(!diff.is_up_to_date());

    api.index_books(&[corrected]).unwrap().unwrap();
    let count_with = |facet: &str| {
        api.search(SearchRequest {
            query: BERACHOT_LINE.to_string(),
            force_mode: Some(SearchMode::SemanticOnly),
            limit: Some(20),
            filters: Some(facet_filter(&[facet])),
            ..Default::default()
        })
        .unwrap()
        .results
        .len()
    };
    assert_eq!(count_with("/author/רבי יהודה הנשיא"), 1);
    assert_eq!(
        count_with("/era/תנאים"),
        1,
        "the untouched facet must survive the re-index"
    );
}

/// A file-only signature leaves the metadata unproven, so the book comes back as
/// unverifiable and its lines decide.
#[test]
fn a_file_only_signature_cannot_declare_a_pdf_current() {
    let dir = TempDir::new("pdf_content_only");
    let api = indexed_api(config_at(&dir));

    // Indexed with a signature over the file and nothing else.
    let mut file_only = berachot_without_fingerprint();
    file_only.content_fingerprint = PDF_SIGNATURE;
    api.index_books(&[file_only.clone()]).unwrap().unwrap();

    let fingerprints = HashMap::from([
        (
            GENESIS.to_string(),
            ContentFingerprint::from_lexical_hash(genesis_book().content_fingerprint),
        ),
        (
            BERACHOT.to_string(),
            ContentFingerprint::content_only(PDF_SIGNATURE),
        ),
    ]);

    let diff = api.get_semantic_index_diff(&fingerprints).unwrap().unwrap();
    assert_eq!(
        diff.unverifiable_books,
        vec![BERACHOT.to_string()],
        "content-only proof leaves the metadata unproven"
    );
    assert!(diff.changed_books.is_empty());
    assert!(!diff.is_up_to_date());

    let summary = api.index_books(&[file_only]).unwrap().unwrap();
    assert_eq!(summary.books_skipped, 1);
    assert_eq!(summary.chunks_written, 0);
}

/// A caller with nothing but Tantivy's hashes gets every PDF back on every diff.
#[test]
fn a_pdf_without_a_caller_supplied_signature_is_always_offered() {
    let dir = TempDir::new("pdf_no_signature");
    let api = indexed_api(config_at(&dir));

    // Raw lexical hashes: 0 for the PDF.
    let mut tantivy = HashMap::new();
    tantivy.insert(GENESIS.to_string(), genesis_book().content_fingerprint);
    tantivy.insert(BERACHOT.to_string(), 0u64);

    let diff = api
        .get_semantic_index_diff_from_lexical_hashes(&tantivy)
        .unwrap()
        .unwrap();

    assert_eq!(
        diff.unverifiable_books,
        vec![BERACHOT.to_string()],
        "a PDF with no usable fingerprint must be re-examined"
    );
    assert!(
        diff.changed_books.is_empty(),
        "it is not known to have changed, only unproven"
    );
    assert!(!diff.is_up_to_date());
    assert!(!diff.needs_full_rebuild());

    let summary = api.index_books(&[berachot_book()]).unwrap().unwrap();
    assert_eq!(
        summary.books_skipped, 1,
        "an unchanged book must be skipped even when the diff could not prove it"
    );
    assert_eq!(summary.chunks_written, 0);
}

#[test]
fn a_book_deleted_from_the_library_is_reported_as_removed() {
    let dir = TempDir::new("removed_book");
    let api = indexed_api(config_at(&dir));

    let only_genesis = HashMap::from([(
        GENESIS.to_string(),
        ContentFingerprint::from_lexical_hash(genesis_book().content_fingerprint),
    )]);

    let diff = api.get_semantic_index_diff(&only_genesis).unwrap().unwrap();
    assert_eq!(diff.removed_books, vec![BERACHOT.to_string()]);
    assert!(diff.new_books.is_empty());
    assert!(diff.changed_books.is_empty());
    assert!(!diff.is_up_to_date());
}

#[test]
fn removed_books_can_be_applied_through_the_public_api() {
    let dir = TempDir::new("apply_removed_book");
    let api = indexed_api(config_at(&dir));
    let only_genesis = HashMap::from([(
        GENESIS.to_string(),
        ContentFingerprint::from_lexical_hash(genesis_book().content_fingerprint),
    )]);
    let diff = api.get_semantic_index_diff(&only_genesis).unwrap().unwrap();

    assert_eq!(
        api.remove_semantic_books(&diff.removed_books).unwrap(),
        Some(1)
    );
    let status = api.get_semantic_status();
    assert_eq!(status.indexed_book_count, 1);
    assert_eq!(status.vector_count, 3);
    assert!(
        api.get_semantic_index_diff(&only_genesis)
            .unwrap()
            .unwrap()
            .is_up_to_date(),
        "applying removed_books must make the diff converge"
    );

    let result = api
        .search(SearchRequest {
            query: BERACHOT_LINE.to_string(),
            force_mode: Some(SearchMode::SemanticOnly),
            ..Default::default()
        })
        .unwrap();
    assert!(
        result.results.iter().all(|item| item.id != 501),
        "deleted vectors must not remain searchable"
    );
}

// ───────────────────────── manifest ─────────────────────────

#[test]
fn a_corrupt_manifest_is_quarantined_and_the_engine_recovers() {
    let dir = TempDir::new("corrupt_manifest");
    let config = config_at(&dir);

    {
        let _ = indexed_api(config.clone());
    }

    // A half-written manifest, the failure mode an unflushed rename can produce.
    std::fs::write(
        SemanticManifest::file_path(&config.root_dir),
        b"{\"format_version\": 2, \"embedding_model_i",
    )
    .unwrap();

    let engine = SemanticEngine::open(config.clone()).unwrap();
    let api = OtzariaHybridEngine::new(HybridCoordinator::new(Some(engine)));

    let status = api.get_semantic_status();
    assert_eq!(status.indexed_book_count, 0);
    assert!(
        status.last_error.is_some(),
        "a silent reset would be untraceable"
    );
    assert!(
        status.needs_full_reindex.is_none(),
        "a corrupt manifest is a fresh start, not an incompatibility"
    );

    let kept = std::fs::read_dir(&config.root_dir)
        .unwrap()
        .filter_map(Result::ok)
        .any(|e| e.file_name().to_string_lossy().contains("corrupt"));
    assert!(kept, "the unusable manifest must be preserved, not deleted");

    let summary = api.index_books(&[genesis_book()]).unwrap().unwrap();
    assert_eq!(summary.chunks_written, 3);
    assert!(api.get_semantic_status().available);
}

/// Changing the embedding dimension invalidates every stored vector. Continuing
/// to search would compare vectors from two different spaces.
#[test]
fn an_incompatible_configuration_disables_semantic_search_until_it_is_reset() {
    let dir = TempDir::new("incompatible");
    let config = config_at(&dir);

    {
        let _ = indexed_api(config.clone());
    }

    let mut changed = config.clone();
    changed.embedding_dim = 32;
    changed.store.embedding_dim = 32;

    let engine = SemanticEngine::open(changed).unwrap();
    let api = OtzariaHybridEngine::new(HybridCoordinator::new(Some(engine)));

    let status = api.get_semantic_status();
    assert!(!status.available);
    let reason = status
        .needs_full_reindex
        .expect("the caller must be told a rebuild is required");
    assert!(reason.contains("Dimensions"), "unhelpful reason: {reason}");

    let result = api
        .search(SearchRequest {
            query: LINE_ONE.to_string(),
            lexical_candidates: vec![lexical_hit(1, LINE_ONE, 15.5)],
            ..Default::default()
        })
        .unwrap();
    assert_eq!(result.search_mode, SearchMode::LexicalOnly);
    assert!(!result.semantic_available);
    assert!(result.fallback_reason.is_some());
    assert_eq!(result.results.len(), 1);

    // Indexing is refused rather than mixing vector spaces.
    assert!(api.index_books(&[genesis_book()]).is_err());

    api.reset_semantic_index().unwrap();
    assert!(api.get_semantic_status().needs_full_reindex.is_none());
    let summary = api.index_books(&[genesis_book()]).unwrap().unwrap();
    assert_eq!(summary.chunks_written, 3);
    assert!(api.get_semantic_status().available);
}

#[test]
fn a_swapped_model_file_behind_the_same_id_is_detected() {
    let dir = TempDir::new("swapped_model");
    let config = config_at(&dir);

    {
        let _ = indexed_api(config.clone());
    }

    // Same path, same model id, different bytes.
    let mut bytes = std::fs::read(&config.model_path).unwrap();
    bytes.extend_from_slice(b"a completely different set of weights");
    std::fs::write(&config.model_path, bytes).unwrap();

    let mut engine = SemanticEngine::open(config).unwrap();
    engine.load_model().unwrap();
    assert!(
        !engine.incompatibilities().is_empty(),
        "only the file checksum can catch this"
    );
    assert!(engine.status().needs_full_reindex.is_some());
}

// ───────────────────────── filters ─────────────────────────

#[test]
fn filters_select_the_same_documents_the_lexical_facets_would() {
    let dir = TempDir::new("filters");
    let api = indexed_api(config_at(&dir));

    let semantic_search = |filters: Option<SearchFilters>| {
        api.search(SearchRequest {
            query: LINE_ONE.to_string(),
            force_mode: Some(SearchMode::SemanticOnly),
            limit: Some(50),
            filters,
            ..Default::default()
        })
        .unwrap()
        .results
        .len()
    };

    assert_eq!(semantic_search(None), 4, "all four indexed lines");

    // A category filter matches hierarchically, like a Tantivy facet.
    assert_eq!(
        semantic_search(Some(facet_filter(&["/מקרא"]))),
        3,
        "the three Genesis lines"
    );
    assert_eq!(semantic_search(Some(facet_filter(&["/תלמוד/משנה"]))), 1);

    // Paths in one dimension are OR-ed; different dimensions are AND-ed.
    assert_eq!(semantic_search(Some(facet_filter(&["/מקרא", "/תלמוד"]))), 4);
    assert_eq!(
        semantic_search(Some(facet_filter(&["/מקרא", "/era/תנאים"]))),
        0,
        "no line is both Torah and Tannaitic"
    );

    // Both of Genesis's authors select it: a single-valued field could not.
    for author in ["/author/משה רבנו", "/author/עזרא הסופר"] {
        assert_eq!(
            semantic_search(Some(facet_filter(&[author]))),
            3,
            "filtering by {author} must select Genesis"
        );
    }
    assert_eq!(
        semantic_search(Some(facet_filter(&["/author/מחבר שאינו קיים"]))),
        0
    );

    // `/base` is a bare marker, not a path with a value.
    assert_eq!(
        semantic_search(Some(facet_filter(&["/base"]))),
        3,
        "only Genesis is marked as a foundational book"
    );

    assert_eq!(
        semantic_search(Some(SearchFilters {
            book_paths: Some(vec![BERACHOT.to_string()]),
            ..Default::default()
        })),
        1
    );

    // include_pdf excludes; it does not mean "PDFs only".
    assert_eq!(
        semantic_search(Some(SearchFilters {
            include_pdf: Some(false),
            ..Default::default()
        })),
        3,
        "Berachot is the PDF book"
    );
    assert_eq!(
        semantic_search(Some(SearchFilters {
            include_pdf: Some(true),
            ..Default::default()
        })),
        4
    );
}

/// The bug: `SearchFilters::is_empty()` treated an empty list as "not a filter"
/// while matching treated it as "must be one of zero values", silently emptying
/// the result set.
#[test]
fn empty_filter_lists_do_not_empty_the_result_set() {
    let dir = TempDir::new("empty_filters");
    let api = indexed_api(config_at(&dir));

    let result = api
        .search(SearchRequest {
            query: LINE_ONE.to_string(),
            force_mode: Some(SearchMode::SemanticOnly),
            limit: Some(50),
            filters: Some(SearchFilters {
                book_paths: Some(vec![]),
                facets: Some(vec![]),
                include_pdf: None,
            }),
            ..Default::default()
        })
        .unwrap();

    assert_eq!(result.results.len(), 4);
}

#[test]
fn a_filter_that_matches_nothing_returns_an_empty_result_not_an_error() {
    let dir = TempDir::new("filter_no_match");
    let api = indexed_api(config_at(&dir));

    let result = api
        .search(SearchRequest {
            query: LINE_ONE.to_string(),
            lexical_candidates: vec![],
            filters: Some(SearchFilters {
                book_paths: Some(vec!["otzaria/does/not/exist.txt".to_string()]),
                ..Default::default()
            }),
            ..Default::default()
        })
        .unwrap();

    assert!(result.results.is_empty());
    assert_eq!(result.total_count, 0);
    assert!(
        result.semantic_available,
        "an empty result is not a failure"
    );
}

// ───────────────────────── grouping and paging ─────────────────────────

#[test]
fn grouping_collapses_a_section_and_still_reports_the_true_size() {
    let dir = TempDir::new("grouping");
    let api = indexed_api(config_at(&dir));

    let result = api
        .search(SearchRequest {
            query: LINE_ONE.to_string(),
            lexical_candidates: vec![
                lexical_hit(1, LINE_ONE, 15.0),
                lexical_hit(2, LINE_TWO, 12.0),
            ],
            grouping: Some(GroupingMode::SameSection),
            limit: Some(10),
            ..Default::default()
        })
        .unwrap();

    // Lines 1-2 share section 100; line 3 is section 101; Berachot is its own.
    assert_eq!(result.group_count, Some(3));
    let collapsed = result
        .results
        .iter()
        .find(|r| r.merged_count > 1)
        .expect("section 100 should collapse");
    assert_eq!(collapsed.merged_count, 2);
    assert_eq!(collapsed.merged.len(), 1);
}

#[test]
fn paging_covers_every_result_exactly_once() {
    let dir = TempDir::new("paging");
    let api = indexed_api(config_at(&dir));

    let page = |offset: u32| {
        api.search(SearchRequest {
            query: LINE_ONE.to_string(),
            force_mode: Some(SearchMode::SemanticOnly),
            limit: Some(2),
            offset: Some(offset),
            ..Default::default()
        })
        .unwrap()
    };

    let first = page(0);
    let second = page(2);
    let third = page(4);

    assert_eq!(first.total_count, 4);
    assert_eq!(first.results.len(), 2);
    assert_eq!(second.results.len(), 2);
    assert!(third.results.is_empty());

    let mut ids: Vec<u64> = first.results.iter().map(|r| r.id).collect();
    ids.extend(second.results.iter().map(|r| r.id));
    ids.sort_unstable();
    assert_eq!(ids, vec![1, 2, 3, 501]);

    let first_ids: Vec<u64> = first.results.iter().map(|r| r.id).collect();
    let again_ids: Vec<u64> = page(0).results.iter().map(|r| r.id).collect();
    assert_eq!(first_ids, again_ids);
}
