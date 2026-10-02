//! The official read-only runtime as the application sees it.
//!
//! `otzaria_search_engine` is the consumer: it opens the Tantivy index, opens the vector
//! set installed beside it, and serves queries from both. These tests run that path
//! through the public API only — a base package built from a tiny corpus by the
//! deterministic stand-in, installed with [`install_package`], opened through
//! [`OfficialSemanticIndex`], and queried through [`HybridCoordinator`] with a
//! [`FakeResolver`] standing in for the application's live index.
//!
//! What is asserted here and not in the crate's own tests is the *product* surface: that
//! a semantic query over an installed set reaches fusion with the ids and books the live
//! index gives its lines — not the ones the vectors were built with — that a line whose
//! text changed since is never shown for a vector of the old text, that the query cache
//! follows the live index, and that every build-side operation the seam still exposes
//! refuses by name instead of quietly doing nothing.
//!
//! Driving a query end to end needs an embedding backend, and needs the deterministic
//! stand-in to be the one actually selected — the fixture's model is a weightless stub
//! that real inference rightly refuses — hence `mock-embedding` without `onnx-backend`,
//! as in `tests/hybrid_integration_test.rs`.

#![cfg(all(feature = "mock-embedding", not(feature = "onnx-backend")))]

use otzaria_semantic_search::api::hybrid_search::OtzariaHybridEngine;
use otzaria_semantic_search::cancellation::CancellationToken;
use otzaria_semantic_search::distribution::builder::{
    build, BuildRequest, RELEASE_MANIFEST_FILENAME, SEGMENT_FILENAME,
};
use otzaria_semantic_search::distribution::corpus::{
    CorpusIdentity, CorpusLine, CorpusLineRecord, JsonlCorpus,
};
use otzaria_semantic_search::errors::SemanticSearchError;
use otzaria_semantic_search::hybrid::coordinator::{HybridCoordinator, HybridSearchParams};
use otzaria_semantic_search::semantic::chunk_key::ChunkKey;
use otzaria_semantic_search::semantic::chunker::ChunkerConfig;
use otzaria_semantic_search::semantic::embedding::{mock, EmbeddingDeployment};
use otzaria_semantic_search::semantic::model_package::validate_model;
use otzaria_semantic_search::semantic::official_index::{
    LocalModel, OfficialIndexConfig, OfficialSemanticIndex, ReloadOutcome,
};
use otzaria_semantic_search::semantic::resolve::{
    BookSet, CandidateResolver, ResolveError, ResolvedLine, VectorHit,
};
use otzaria_semantic_search::semantic::segment_set::{
    install_package, InstallExpectation, InstallSource,
};
use otzaria_semantic_search::semantic::types::{
    ContentFingerprint, LexicalCandidate, SearchFilters, SearchMode,
};
use otzaria_semantic_search::semantic::versioning::{ModelIdentity, ModelPackage};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

const DIM: u32 = 64;
const GENESIS: &str = "otzaria/tanach/genesis.txt";
const BERACHOT: &str = "otzaria/mishna/berachot.txt";

/// `(line_id, book, text)`, every one long enough to be embedded as itself.
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
        // The clock alone collided: macOS ticks coarser than a test takes to start.
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "otzaria_official_runtime_{name}_{}_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, Ordering::Relaxed)
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

fn facets_of(book: &str) -> Vec<String> {
    if book == GENESIS {
        vec!["/מקרא/תורה".to_string()]
    } else {
        vec!["/משנה".to_string()]
    }
}

fn title_of(book: &str) -> String {
    if book == GENESIS {
        "בראשית"
    } else {
        "ברכות"
    }
    .to_string()
}

/// The application's index, as far as a search asks it: each book's lines by position,
/// with the ids and the text the index holds for them now.
struct FakeResolver {
    generation: AtomicU64,
    lines: Mutex<HashMap<(String, u32), (u64, String)>>,
}

impl FakeResolver {
    /// The index the vectors were built from: every line where it was, with its id.
    fn of_the_corpus() -> Self {
        let mut ordinals: HashMap<&str, u32> = HashMap::new();
        let lines = LINES
            .iter()
            .map(|(line_id, book, text)| {
                let ordinal = ordinals.entry(book).or_default();
                let at = (book.to_string(), *ordinal);
                *ordinal += 1;
                (at, (*line_id, text.to_string()))
            })
            .collect();
        Self {
            generation: AtomicU64::new(1),
            lines: Mutex::new(lines),
        }
    }

    /// An index commit: the line at `(book, ordinal)` now has `line_id` and `text`.
    fn commit(&self, book: &str, ordinal: u32, line_id: u64, text: &str) {
        self.lines
            .lock()
            .unwrap()
            .insert((book.to_string(), ordinal), (line_id, text.to_string()));
        self.generation.fetch_add(1, Ordering::SeqCst);
    }
}

impl CandidateResolver for FakeResolver {
    fn generation(&self) -> u64 {
        self.generation.load(Ordering::SeqCst)
    }

    fn admissible_books(
        &self,
        filters: Option<&SearchFilters>,
    ) -> Result<Option<BookSet>, ResolveError> {
        let Some(compiled) = filters.and_then(SearchFilters::compile) else {
            return Ok(None);
        };
        Ok(Some(
            [GENESIS, BERACHOT]
                .into_iter()
                .filter(|book| compiled.matches_book(book, &facets_of(book), false))
                .collect(),
        ))
    }

    fn resolve(
        &self,
        hits: &[VectorHit],
        filters: Option<&SearchFilters>,
        cancel: &CancellationToken,
    ) -> Result<Vec<ResolvedLine>, ResolveError> {
        if cancel.is_cancelled() {
            return Err(ResolveError::Cancelled);
        }
        let compiled = filters.and_then(SearchFilters::compile);
        let lines = self.lines.lock().unwrap();
        let mut resolved = Vec::new();
        for (index, hit) in hits.iter().enumerate() {
            for record in &hit.records {
                let Some((line_id, text)) = lines.get(&(record.book.to_string(), record.hint))
                else {
                    continue;
                };
                // The live line has to hold the text the vector was built from: a line
                // edited since is not what the vector describes.
                if ChunkKey::of(text) != hit.key {
                    continue;
                }
                let facets = facets_of(&record.book);
                if compiled
                    .as_ref()
                    .is_some_and(|compiled| !compiled.matches_book(&record.book, &facets, false))
                {
                    continue;
                }
                resolved.push(ResolvedLine {
                    hit: index as u32,
                    line_id: *line_id,
                    file_path: record.book.to_string(),
                    section_id: 1,
                    line_hash: line_id ^ 0xABCD,
                    segment: u64::from(record.hint),
                    is_pdf: false,
                    facets: facets.into(),
                    title: title_of(&record.book),
                    reference: String::new(),
                });
            }
        }
        Ok(resolved)
    }
}

fn corpus_identity(library_version: u32) -> CorpusIdentity {
    CorpusIdentity {
        text: otzaria_semantic_search::semantic::versioning::TextIdentity::with_line_text_version(
            1,
        ),
        library_version,
        library_release_tag: format!("v{library_version}-20260930120000"),
        document_id_scheme_version: 1,
    }
}

fn model_identity(model_path: &Path) -> ModelIdentity {
    ModelIdentity {
        family_id: "otzaria-test-family".to_string(),
        tokenizer_checksum: mock::stub_tokenizer_checksum(),
        query_packages: vec![ModelPackage {
            checksum: validate_model(model_path).unwrap().checksum().to_string(),
            quantization: "int8".to_string(),
        }],
        embedding_dim: DIM,
        pooling: "in-graph".to_string(),
        max_tokens: 512,
        embedding_text_version: 1,
        normalization_version: 1,
        chunking_identity: ChunkerConfig::default().identity(),
    }
}

/// Build a base package of `lines` for `library_version` and install it into `vectors`.
fn build_and_install(
    dir: &TempDir,
    name: &str,
    model_path: &Path,
    library_version: u32,
    lines: &[(u64, &str, &str)],
    vectors: &Path,
) {
    let identity_path = dir.path().join(format!("{name}-corpus-identity.json"));
    let lines_path = dir.path().join(format!("{name}-corpus-lines.jsonl"));
    std::fs::write(
        &identity_path,
        serde_json::to_vec(&corpus_identity(library_version)).unwrap(),
    )
    .unwrap();
    let body: String = lines
        .iter()
        .map(|(line_id, book, text)| {
            serde_json::to_string(&CorpusLineRecord {
                line_id: *line_id,
                line: CorpusLine {
                    source_book_key: book.to_string(),
                    title: title_of(book),
                    reference: String::new(),
                    section_id: 1,
                    segment: 0,
                    is_pdf: false,
                    line_hash: 0,
                    content_hash: 7,
                    facets: facets_of(book),
                    text: text.to_string(),
                },
            })
            .unwrap()
                + "\n"
        })
        .collect();
    std::fs::write(&lines_path, body).unwrap();

    let out = dir.path().join(name);
    let report = build(
        BuildRequest {
            output_path: out.clone(),
            model_path: model_path.to_path_buf(),
            model: model_identity(model_path),
            chunking: ChunkerConfig::default(),
            created_at: "2026-10-01T00:00:00Z".to_string(),
            batch_size: 2,
            codec: otzaria_semantic_search::semantic::oxv::codec::CodecSpec::default(),
            allow_non_semantic_backend: true,
        },
        &JsonlCorpus::load(&identity_path, &lines_path).unwrap(),
    )
    .unwrap();
    install_package(
        vectors,
        &InstallSource {
            segment: &out.join(SEGMENT_FILENAME),
            manifest_json: &std::fs::read_to_string(out.join(RELEASE_MANIFEST_FILENAME)).unwrap(),
        },
        &InstallExpectation {
            identity: report.manifest.identity.clone(),
            published_manifest_sha256: Some(report.manifest_sha256),
        },
        &CancellationToken::new(),
    )
    .unwrap();
}

/// The stub model and an installed set of [`LINES`].
fn install(dir: &TempDir) -> (PathBuf, PathBuf) {
    let model_path = mock::write_stub_onnx_package(&dir.path().join("model"));
    let vectors = dir.path().join("vectors");
    build_and_install(dir, "v30", &model_path, 30, &LINES, &vectors);
    (model_path, vectors)
}

fn open_official(vectors: &Path, model_path: &Path) -> OfficialSemanticIndex {
    OfficialSemanticIndex::open(OfficialIndexConfig {
        vectors_dir: vectors.to_path_buf(),
        text: corpus_identity(30).text,
        model: LocalModel::of_family(
            model_path.to_path_buf(),
            &model_identity(model_path),
            "int8",
        ),
        deployment: EmbeddingDeployment::default(),
        scan_threads: None,
    })
    .unwrap()
}

fn lexical(line_id: u64, book: &str, text: &str, score: f32) -> LexicalCandidate {
    LexicalCandidate {
        title: title_of(book),
        reference: format!("{book} :: {line_id}"),
        text: text.to_string(),
        line_id,
        section_id: 1,
        line_hash: line_id ^ 0xABCD,
        segment: 0,
        is_pdf: false,
        file_path: book.to_string(),
        bm25_score: score,
    }
}

fn mode(mode: SearchMode) -> HybridSearchParams {
    HybridSearchParams {
        force_mode: Some(mode),
        ..Default::default()
    }
}

/// Every file of the set with its size and content digest: what "opening it did not write
/// to it" is checked against.
fn fingerprint(dir: &Path) -> Vec<(String, u64, String)> {
    fn walk(dir: &Path, root: &Path, out: &mut Vec<(String, u64, String)>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                walk(&path, root, out);
            } else {
                let bytes = std::fs::read(&path).unwrap();
                out.push((
                    path.strip_prefix(root).unwrap().display().to_string(),
                    bytes.len() as u64,
                    format!("{:x}", <sha2::Sha256 as sha2::Digest>::digest(&bytes)),
                ));
            }
        }
    }
    let mut out = Vec::new();
    walk(dir, dir, &mut out);
    out.sort();
    out
}

/// The shortest complete path: a semantic query over an installed set comes back with the
/// live line the resolver tied its hit to, in both modes that consult the semantic path.
#[test]
fn a_query_over_an_installed_set_returns_the_live_line_it_resolves_to() {
    let dir = TempDir::new("query");
    let (model_path, vectors) = install(&dir);
    let coordinator = HybridCoordinator::with_official_index(open_official(&vectors, &model_path));
    let resolver = FakeResolver::of_the_corpus();
    let cancel = CancellationToken::new();

    let status = coordinator.status();
    assert!(status.available);
    assert!(status.vectors_persisted);
    assert_eq!(status.vector_count, LINES.len() as u32);
    assert_eq!(status.indexed_book_count, 2);
    assert_eq!(status.vector_backend, "otzaria-oxv");

    let (line_id, book, text) = LINES[3];
    let semantic = coordinator
        .search_cancellable(
            text,
            // Deliberately supplied and deliberately discarded: a semantic-only request
            // must not be answered with BM25 wearing a semantic label.
            vec![lexical(999, GENESIS, "שורה לקסיקלית בלבד", 99.0)],
            &mode(SearchMode::SemanticOnly),
            &resolver,
            &cancel,
        )
        .unwrap();
    assert_eq!(semantic.search_mode, SearchMode::SemanticOnly);
    assert!(semantic.semantic_available);
    assert!(semantic.fallback_reason.is_none());
    assert_eq!(
        (
            semantic.results[0].id,
            semantic.results[0].file_path.as_str()
        ),
        (line_id, book)
    );
    assert!(semantic.results.iter().all(|item| item.id != 999));

    // Hybrid: both sources reach fusion over the same lines.
    let hybrid = coordinator
        .search_cancellable(
            text,
            vec![lexical(line_id, book, text, 18.0)],
            &mode(SearchMode::Hybrid),
            &resolver,
            &cancel,
        )
        .unwrap();
    assert_eq!(hybrid.search_mode, SearchMode::Hybrid);
    assert!(hybrid.fallback_reason.is_none());
    let fused = &hybrid.results[0];
    assert_eq!((fused.id, fused.file_path.as_str()), (line_id, book));
    assert!(fused.lexical_score.is_some());
    assert!(fused.semantic_score.is_some());

    // A filter reaches the scan as the books it admits.
    let filtered = coordinator
        .search_cancellable(
            LINES[0].2,
            vec![],
            &HybridSearchParams {
                force_mode: Some(SearchMode::SemanticOnly),
                filters: Some(SearchFilters {
                    facets: Some(vec!["/משנה".to_string()]),
                    ..Default::default()
                }),
                ..Default::default()
            },
            &resolver,
            &cancel,
        )
        .unwrap();
    assert!(!filtered.results.is_empty());
    assert!(filtered
        .results
        .iter()
        .all(|item| item.file_path == BERACHOT));
}

/// The ids a result carries are the live index's. A commit that renumbered a book — or
/// gave two books one id range — changes nothing about the vectors, and the results follow
/// the index without a rebuild.
#[test]
fn the_ids_are_the_live_index_s_and_two_books_sharing_one_stay_apart() {
    let dir = TempDir::new("live_ids");
    let (model_path, vectors) = install(&dir);
    let coordinator = HybridCoordinator::with_official_index(open_official(&vectors, &model_path));
    let resolver = FakeResolver::of_the_corpus();
    let cancel = CancellationToken::new();

    // Berachot's line now has the id genesis's first line has: two books, one id.
    let (shared, genesis_text) = (LINES[0].0, LINES[0].2);
    resolver.commit(BERACHOT, 0, shared, LINES[3].2);

    let result = coordinator
        .search_cancellable(
            genesis_text,
            vec![lexical(shared, GENESIS, genesis_text, 18.0)],
            &HybridSearchParams {
                force_mode: Some(SearchMode::Hybrid),
                limit: 10,
                ..Default::default()
            },
            &resolver,
            &cancel,
        )
        .unwrap();
    let of = |book: &str| {
        result
            .results
            .iter()
            .find(|item| item.id == shared && item.file_path == book)
            .unwrap_or_else(|| panic!("line {shared} of {book} is a result of its own"))
    };
    assert!(of(GENESIS).lexical_score.is_some() && of(GENESIS).semantic_score.is_some());
    assert!(
        of(BERACHOT).lexical_score.is_none() && of(BERACHOT).semantic_score.is_some(),
        "the lexical hit is genesis's, and must not be fused into berachot's line"
    );
}

/// A line whose text changed since the vectors were built is not shown for them: its hit
/// resolves to nothing, and the telemetry counts it.
#[test]
fn a_line_whose_text_changed_is_not_shown_for_the_old_vector() {
    let dir = TempDir::new("changed_text");
    let (model_path, vectors) = install(&dir);
    let coordinator = HybridCoordinator::with_official_index(open_official(&vectors, &model_path));
    let resolver = FakeResolver::of_the_corpus();

    let (line_id, book, text) = LINES[3];
    resolver.commit(book, 0, line_id, "שורה שנערכה מאז שנבנו הווקטורים");
    let result = coordinator
        .search_cancellable(
            text,
            vec![],
            &mode(SearchMode::SemanticOnly),
            &resolver,
            &CancellationToken::new(),
        )
        .unwrap();
    assert!(result.results.iter().all(|item| item.id != line_id));
    let telemetry = coordinator.get_telemetry_snapshot();
    assert_eq!(telemetry.semantic_hits, LINES.len() as u64);
    assert_eq!(telemetry.semantic_unresolved, 1);
}

/// A semantic-only query has no lexical input to change when the index commits, so the
/// query cache has to key on the resolver's generation: without it, a cached result would
/// carry the ids from before the commit. And a reloaded vector set is a new generation of
/// its own.
#[test]
fn the_query_cache_follows_the_index_and_the_vector_set() {
    let dir = TempDir::new("cache");
    let (model_path, vectors) = install(&dir);
    let coordinator = HybridCoordinator::with_official_index(open_official(&vectors, &model_path));
    let resolver = FakeResolver::of_the_corpus();
    let cancel = CancellationToken::new();
    let (line_id, book, text) = LINES[3];
    let search = || {
        coordinator
            .search_cancellable(
                text,
                vec![],
                &mode(SearchMode::SemanticOnly),
                &resolver,
                &cancel,
            )
            .unwrap()
    };

    assert_eq!(search().results[0].id, line_id);
    assert_eq!(search().results[0].id, line_id);
    assert_eq!(coordinator.get_telemetry_snapshot().cache_hits, 1);

    // A commit renumbers the line; the next search is answered from the index, not the cache.
    resolver.commit(book, 0, line_id + 1000, text);
    assert_eq!(search().results[0].id, line_id + 1000);
    assert_eq!(coordinator.get_telemetry_snapshot().cache_hits, 1);

    // A new library version installed beside the open set: nothing changes until the
    // coordinator reloads, and then the new generation answers.
    let mut changed = LINES;
    changed[3].2 = "תנו רבנן מאימתי מתחילין לקרות את שמע בשחרית משיכיר בין תכלת ללבן";
    build_and_install(&dir, "v31", &model_path, 31, &changed, &vectors);
    let before = coordinator.vector_set_info().unwrap();
    assert_eq!(before.library_version, 30);
    match coordinator.reload_semantic_vectors().unwrap() {
        Some(ReloadOutcome::Reloaded {
            from_generation,
            to_generation,
        }) => assert!(to_generation > from_generation),
        other => panic!("the new generation must be opened, got {other:?}"),
    }
    assert_eq!(coordinator.vector_set_info().unwrap().library_version, 31);
    assert_eq!(
        coordinator.reload_semantic_vectors().unwrap(),
        Some(ReloadOutcome::Unchanged {
            generation: coordinator.vector_set_info().unwrap().generation
        })
    );
    // The old text has no vector any more, so its line is found by its neighbours' at best.
    assert!(search()
        .results
        .iter()
        .all(|item| item.id != line_id + 1000 || item.semantic_score.unwrap_or(0.0) < 0.99));
}

/// The path the application takes on every keystroke: a search over the installed set,
/// abandoned because the next keystroke superseded it. It must come back as `Cancelled` in
/// both modes that scan the set — not as a lexical fallback the host would display — and
/// the set must serve the next query as if nothing had happened.
#[test]
fn a_cancelled_query_over_an_installed_set_is_dropped_and_the_next_one_served() {
    let dir = TempDir::new("cancelled");
    let (model_path, vectors) = install(&dir);
    let coordinator = HybridCoordinator::with_official_index(open_official(&vectors, &model_path));
    let resolver = FakeResolver::of_the_corpus();
    let (line_id, book, text) = LINES[0];

    for search_mode in [SearchMode::SemanticOnly, SearchMode::Hybrid] {
        let superseded = CancellationToken::new();
        let next_keystroke = superseded.clone();
        std::thread::spawn(move || next_keystroke.cancel())
            .join()
            .unwrap();
        match coordinator.search_cancellable(
            text,
            vec![lexical(line_id, book, text, 18.0)],
            &mode(search_mode),
            &resolver,
            &superseded,
        ) {
            Err(SemanticSearchError::Cancelled) => {}
            other => panic!("{search_mode}: a superseded search must be cancelled, got {other:?}"),
        }

        let served = coordinator
            .search_cancellable(
                text,
                vec![lexical(line_id, book, text, 18.0)],
                &mode(search_mode),
                &resolver,
                &CancellationToken::new(),
            )
            .unwrap();
        assert_eq!(served.search_mode, search_mode);
        assert!(served.fallback_reason.is_none());
        assert_eq!(served.results[0].id, line_id);
    }
    assert_eq!(coordinator.get_telemetry_snapshot().total_searches, 2);
}

/// A vector set is not something this device indexes into, and the seam has to say so
/// rather than report a no-op: a caller that read the refusal as "no semantic index" would
/// offer indexing the library as the fix, which is what the product contract rules out.
#[test]
fn every_build_side_operation_is_refused_on_an_installed_set() {
    let dir = TempDir::new("refusals");
    let (model_path, vectors) = install(&dir);
    let api = OtzariaHybridEngine::new(HybridCoordinator::with_official_index(open_official(
        &vectors,
        &model_path,
    )));

    let before = fingerprint(&vectors);
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
    // Nothing was half-done: a refusal is not a failure state the caller has to recover
    // from.
    assert_eq!(fingerprint(&vectors), before);
}

/// Restart: the same directory opens again and answers the same query, with nothing
/// rebuilt and nothing written. This is what `vectors_persisted` is a claim about.
#[test]
fn a_restart_opens_the_same_set_without_writing_to_it() {
    let dir = TempDir::new("restart");
    let (model_path, vectors) = install(&dir);

    let before = fingerprint(&vectors);
    let (_, book, text) = LINES[1];
    let mut generations = Vec::new();
    for _ in 0..2 {
        let index = open_official(&vectors, &model_path);
        assert_eq!(index.set_info().slots_live, LINES.len() as u64);
        assert_eq!(index.book_count(), 2);
        generations.push(index.generation());
        let hit = &index
            .search(text, 2, None, &CancellationToken::new())
            .unwrap()[0];
        assert_eq!(hit.key, ChunkKey::of(text));
        assert_eq!((&*hit.records[0].book, hit.records[0].hint), (book, 1));
        assert_eq!(
            fingerprint(&vectors),
            before,
            "opening a set must not write to it"
        );
    }
    assert_eq!(generations[0], generations[1]);
}
