//! The application's path: open an installed vector set and query it, read-only.
//!
//! [`OfficialSemanticIndex`] is a [`SegmentSet`] and the model a query is embedded with,
//! opened in the order that lets every refusal name what disagreed:
//!
//! 1. **recover** — what an interrupted install left in the set's directory is removed, and
//!    a `CURRENT` that does not open falls back to `PREVIOUS` (see [`segment_set`]);
//! 2. **the set** — every segment mapped and checked, every derived file read;
//! 3. **the model** — loaded, so its package checksum and its tokenizer's are known;
//! 4. **the identity** — the set's against this installation's: the line recipe of the
//!    index the caller has open, the model family it runs with the package it loaded, and
//!    the store this build reads.
//!
//! A search embeds the query, scans the set and returns [`VectorHit`]s — keys and the
//! records where they were built, not lines. Turning them into live lines is the caller's
//! [`CandidateResolver`](crate::semantic::resolve::CandidateResolver), which the
//! [`HybridCoordinator`](crate::hybrid::coordinator::HybridCoordinator) asks.
//!
//! # Which failure is which
//!
//! | Error | Means |
//! |---|---|
//! | [`ArtifactError::MetadataUnusable`] | no set is installed |
//! | [`ArtifactError::IdentityMismatch`] | the wrong set — the field names what disagreed; a v1 artifact names `store.backend_id` |
//! | [`VectorStoreError::Corrupted`](crate::errors::VectorStoreError::Corrupted) | neither generation opens: a segment or a derived file is damaged |
//! | [`EmbeddingError`](crate::errors::EmbeddingError) | the model is missing or does not fit this configuration |

use crate::cancellation::CancellationToken;
use crate::errors::{ArtifactError, SemanticSearchError};
use crate::semantic::backend::Pooling;
use crate::semantic::embedding::{EmbeddingConfig, EmbeddingDeployment, EmbeddingRuntime};
use crate::semantic::oxv::format::SEGMENT_FORMAT_VERSION;
use crate::semantic::oxv::scan::{default_scan_threads, ScanRequest};
use crate::semantic::recipe::{EmbeddingTextRecipe, TextNormalizationRecipe};
use crate::semantic::resolve::{BookSet, VectorHit};
use crate::semantic::segment_set::{self, SegmentSet, SetInfo, STORE_BACKEND_ID};
use crate::semantic::types::SemanticStatus;
use crate::semantic::versioning::{
    IndexVersion, ModelIdentity, ModelPackage, StoreIdentity, TextIdentity,
};
use std::collections::BTreeSet;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

/// The store this build reads: `otzaria-oxv` segments, format 2, in the `i8-sym-vec` codec.
///
/// What the installation *requires*, not something read out of the set: a set in another
/// codec or format is a rejection naming `store.vector_precision` or
/// `store.store_format_version`.
pub fn readable_store_identity() -> StoreIdentity {
    StoreIdentity {
        backend_id: STORE_BACKEND_ID.to_string(),
        store_format_version: SEGMENT_FORMAT_VERSION,
        vector_precision: "i8-sym-vec".to_string(),
    }
}

/// The model this installation will embed queries with, and the text recipe it
/// implements.
///
/// Some of these are facts about the file (the dimension and pooling a real backend reads
/// out of the model and this struct must agree with) and some are declarations about how
/// the artifact's vectors were built (`embedding_text_version`, `normalization_version`,
/// `chunking_identity`). The read path derives none of the second group: a query is not
/// chunked, so nothing here could infer them. They are compared anyway, because an
/// artifact built from differently-chunked or differently-normalized text is a different
/// artifact — the results would be plausible and subtly wrong.
///
/// The package's checksum and its tokenizer's are not declared at all: the loaded runtime
/// computes both, and the package must be one the artifact accepts for queries.
#[derive(Debug, Clone)]
pub struct LocalModel {
    pub model_path: PathBuf,
    /// The family the package at `model_path` belongs to — see
    /// [`ModelIdentity::family_id`].
    pub family_id: String,
    /// Quantization of the package at `model_path`, e.g. `"int8"`. Redundant against its
    /// checksum by design: it is what makes a rejection readable.
    pub model_quantization: String,
    pub embedding_dim: u32,
    pub pooling: String,
    pub max_tokens: usize,
    pub embedding_text_version: u32,
    pub normalization_version: u32,
    pub chunking_identity: u64,
}

impl LocalModel {
    /// The package at `model_path`, of precision `quantization`, as a member of `family` —
    /// the family a deployment declares (`config/models/meivin-round2-onnx/model.json`).
    pub fn of_family(model_path: PathBuf, family: &ModelIdentity, quantization: &str) -> Self {
        Self {
            model_path,
            family_id: family.family_id.clone(),
            model_quantization: quantization.to_string(),
            embedding_dim: family.embedding_dim,
            pooling: family.pooling.clone(),
            max_tokens: family.max_tokens,
            embedding_text_version: family.embedding_text_version,
            normalization_version: family.normalization_version,
            chunking_identity: family.chunking_identity,
        }
    }

    /// Compose the model half of the identity from what was declared here and what the
    /// loaded runtime knows: the one package it loaded, and that package's tokenizer.
    fn identity(&self, runtime: &EmbeddingRuntime) -> Result<ModelIdentity, SemanticSearchError> {
        let unknown = |what: &str| {
            SemanticSearchError::Config(format!(
                "the embedding runtime reports no {what} after loading {}, so the \
                 artifact's model identity cannot be compared",
                self.model_path.display()
            ))
        };

        Ok(ModelIdentity {
            family_id: self.family_id.clone(),
            tokenizer_checksum: runtime
                .tokenizer_checksum()
                .ok_or_else(|| unknown("tokenizer checksum"))?
                .to_string(),
            embedding_dim: self.embedding_dim,
            pooling: self.pooling.clone(),
            max_tokens: self.max_tokens,
            embedding_text_version: self.embedding_text_version,
            normalization_version: self.normalization_version,
            chunking_identity: self.chunking_identity,
            query_packages: vec![ModelPackage {
                checksum: runtime
                    .model_checksum()
                    .ok_or_else(|| unknown("model checksum"))?
                    .to_string(),
                quantization: self.model_quantization.clone(),
            }],
        })
    }

    /// The typed pooling strategy, refusing a spelling [`Pooling`] cannot parse and one
    /// no backend implements — the caller's configuration error either way.
    fn pooling_strategy(&self) -> Result<Pooling, SemanticSearchError> {
        let pooling = Pooling::parse(&self.pooling)
            .map_err(|e| SemanticSearchError::Config(e.to_string()))?;
        crate::semantic::backend::ensure_pooling_is_implemented(pooling)
            .map_err(|e| SemanticSearchError::Config(e.to_string()))?;
        Ok(pooling)
    }
}

/// What to open, and what it has to agree with.
#[derive(Debug, Clone)]
pub struct OfficialIndexConfig {
    /// The vector set's directory — `<root>/vectors`, where
    /// [`install_package`](crate::semantic::segment_set::install_package) installs.
    pub vectors_dir: PathBuf,
    /// The line recipe of the index this installation actually has open, and the key
    /// version it computes keys with. Never this crate's constant.
    pub text: TextIdentity,
    pub model: LocalModel,
    /// Where this machine keeps what the model runs on: the ONNX Runtime library the
    /// application ships. Not part of what the set has to agree with — see
    /// [`EmbeddingDeployment`].
    pub deployment: EmbeddingDeployment,
    /// Threads a scan uses; [`default_scan_threads`] when `None`.
    pub scan_threads: Option<NonZeroUsize>,
}

/// What [`OfficialSemanticIndex::reload_vectors`] found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReloadOutcome {
    /// `CURRENT` still names the generation that is open.
    Unchanged { generation: u64 },
    /// A newer generation was opened, with the same model.
    Reloaded {
        from_generation: u64,
        to_generation: u64,
    },
}

/// An installed vector set, verified and open for queries.
pub struct OfficialSemanticIndex {
    set: SegmentSet,
    runtime: EmbeddingRuntime,
    /// The text normalization the vectors were built under, applied to every query.
    normalization: TextNormalizationRecipe,
    /// The text recipe likewise: under version 2 it marks every query as one, as the
    /// stored passages were marked as passages.
    text_recipe: EmbeddingTextRecipe,
    /// What this installation is, which every generation it opens must accept.
    expected: IndexVersion,
    scan_threads: usize,
    /// Distinct books the set holds records for, counted at open.
    book_count: u32,
}

impl OfficialSemanticIndex {
    /// Recover, open the set, load the model, and hold the set's identity to this
    /// installation's — see the module documentation for why in that order.
    pub fn open(config: OfficialIndexConfig) -> Result<Self, SemanticSearchError> {
        let OfficialIndexConfig {
            vectors_dir,
            text,
            model,
            deployment,
            scan_threads,
        } = config;

        // Versions of this crate's code, settled before anything is opened: an
        // installation declaring a recipe nothing implements is refused here rather than
        // agreed with by a set that declares the same number.
        let text_recipe = EmbeddingTextRecipe::from_version(model.embedding_text_version)?;
        let normalization = TextNormalizationRecipe::from_version(model.normalization_version)?;

        let set = SegmentSet::open(&vectors_dir)?;

        let mut runtime = EmbeddingRuntime::with_deployment(
            EmbeddingConfig {
                model_path: model.model_path.clone(),
                embedding_dim: model.embedding_dim,
                pooling: model.pooling_strategy()?,
                max_tokens: model.max_tokens,
                // One query at a time is all this path ever embeds.
                batch_size: 1,
            },
            deployment,
        );
        runtime.load()?;

        let expected = IndexVersion {
            text,
            model: model.identity(&runtime)?,
            store: readable_store_identity(),
        };
        set.identity().verify_matches(&expected)?;
        let book_count = count_books(&set);
        let index = Self {
            set,
            runtime,
            normalization,
            text_recipe,
            expected,
            scan_threads: scan_threads.map_or_else(default_scan_threads, NonZeroUsize::get),
            book_count,
        };
        log::info!(
            "Opened the vector set at {}: generation {}, library version {}, {} live vector(s) \
             across {} book(s) in {} segment(s){}",
            vectors_dir.display(),
            index.set.generation(),
            index.set_info().library_version,
            index.set_info().slots_live,
            book_count,
            index.set_info().segments.len(),
            if index.set_info().recovered_from_previous {
                ", recovered from PREVIOUS"
            } else {
                ""
            }
        );
        Ok(index)
    }

    /// Embed `query` and return the `top_k` closest stored vectors, in books `books`
    /// admits when given.
    pub fn search(
        &self,
        query: &str,
        top_k: usize,
        books: Option<&BookSet>,
        cancel: &CancellationToken,
    ) -> Result<Vec<VectorHit>, SemanticSearchError> {
        cancel.checkpoint()?;
        let query_vector = self.embed_query(query)?;
        cancel.checkpoint()?;
        self.search_hits(&query_vector, top_k, books, cancel)
    }

    /// Scan with a vector this index's runtime already produced.
    pub fn search_hits(
        &self,
        query_vector: &[f32],
        top_k: usize,
        books: Option<&BookSet>,
        cancel: &CancellationToken,
    ) -> Result<Vec<VectorHit>, SemanticSearchError> {
        Ok(self.set.scan(
            query_vector,
            &ScanRequest {
                top_k,
                books,
                threads: self.scan_threads,
            },
            cancel,
        )?)
    }

    /// Embed a query separately, so the coordinator can cache the vector.
    pub(crate) fn embed_query(&self, query: &str) -> Result<Vec<f32>, SemanticSearchError> {
        // The same recipe the stored vectors were built under. A query embedded from raw
        // text against vectors built from normalized text — or without the role prefix
        // its passages were built with — scores nonsense with full confidence.
        let text =
            crate::semantic::recipe::query_input(self.text_recipe, self.normalization, query)?;
        Ok(self.runtime.embed_one(&text)?)
    }

    /// Open the generation `CURRENT` names now, if it is not the open one, keeping the
    /// model. The new generation is held to the same identity; until it opens, the old
    /// one stays in service.
    pub fn reload_vectors(&mut self) -> Result<ReloadOutcome, SemanticSearchError> {
        let from_generation = self.set.generation();
        let dir = self.set.dir().to_path_buf();
        let Some(info) = segment_set::info(&dir)? else {
            return Err(ArtifactError::MetadataUnusable {
                path: dir.display().to_string(),
                reason: "the vector set is gone".to_string(),
            }
            .into());
        };
        if info.generation == from_generation && !info.recovered_from_previous {
            return Ok(ReloadOutcome::Unchanged {
                generation: from_generation,
            });
        }
        let set = SegmentSet::open(&dir)?;
        set.identity().verify_matches(&self.expected)?;
        if set.generation() == from_generation {
            return Ok(ReloadOutcome::Unchanged {
                generation: from_generation,
            });
        }
        self.book_count = count_books(&set);
        self.set = set;
        log::info!(
            "Reloaded the vector set at {}: generation {from_generation} → {}",
            dir.display(),
            self.set.generation()
        );
        Ok(ReloadOutcome::Reloaded {
            from_generation,
            to_generation: self.set.generation(),
        })
    }

    /// Operational status, in the same shape the self-built path reports.
    pub fn status(&self) -> SemanticStatus {
        let info = self.set.info();
        let vector_count = info.slots_live.min(u64::from(u32::MAX)) as u32;
        SemanticStatus {
            available: vector_count > 0 && self.runtime.is_loaded(),
            model_loaded: self.runtime.is_loaded(),
            indexed_book_count: self.book_count,
            vector_count,
            model_id: self.expected.model.family_id.clone(),
            embedding_dim: self.expected.model.embedding_dim,
            embedding_backend: self.runtime.backend_id().map(str::to_string),
            vector_backend: STORE_BACKEND_ID.to_string(),
            vectors_persisted: true,
            needs_full_reindex: None,
            last_error: None,
        }
    }

    /// The open generation: its identity, segments and counts.
    pub fn set_info(&self) -> &SetInfo {
        self.set.info()
    }

    /// The identity the set declares, which this installation accepted.
    pub fn identity(&self) -> &IndexVersion {
        self.set.identity()
    }

    pub fn vectors_dir(&self) -> &Path {
        self.set.dir()
    }

    /// The open generation's number: what a query cache keys on.
    pub fn generation(&self) -> u64 {
        self.set.generation()
    }

    /// Books the set holds records for, as counted when the generation was opened.
    pub fn book_count(&self) -> u32 {
        self.book_count
    }
}

fn count_books(set: &SegmentSet) -> u32 {
    let books: BTreeSet<&str> = set
        .segments()
        .iter()
        .flat_map(|segment| segment.books().iter().map(|book| &*book.name))
        .collect();
    books.len().min(u32::MAX as usize) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::builder::{
        build, BuildRequest, RELEASE_MANIFEST_FILENAME, SEGMENT_FILENAME,
    };
    use crate::distribution::corpus::{CorpusIdentity, CorpusLine, CorpusLineRecord, JsonlCorpus};
    use crate::errors::EmbeddingError;
    use crate::semantic::chunk_key::ChunkKey;
    use crate::semantic::chunker::ChunkerConfig;
    use crate::semantic::embedding::mock;
    use crate::semantic::model_package::validate_model;
    use crate::semantic::oxv::testing::TempDir;
    use crate::semantic::segment_set::{install_package, InstallExpectation, InstallSource};
    use crate::semantic::versioning::IdentityField;
    use std::fs;

    const DIM: u32 = 64;
    const GENESIS: &str = "otzaria/tanach/genesis.txt";
    const BERACHOT: &str = "otzaria/mishna/berachot.txt";

    /// `(book, text)`, every one long enough to be embedded as itself.
    const LINES: [(&str, &str); 3] = [
        (GENESIS, "בראשית ברא אלהים את השמים ואת הארץ"),
        (GENESIS, "והארץ היתה תהו ובהו וחשך על פני תהום"),
        (
            BERACHOT,
            "מאימתי קורין את שמע בערבית משעה שהכהנים נכנסין לאכול בתרומתן",
        ),
    ];

    fn package(path: &Path, quantization: &str) -> ModelPackage {
        ModelPackage {
            checksum: validate_model(path).unwrap().checksum().to_string(),
            quantization: quantization.to_string(),
        }
    }

    /// The family a test set is built for: the packages it accepts for queries, every other
    /// field the stub's.
    fn family(query_packages: Vec<ModelPackage>, embedding_text_version: u32) -> ModelIdentity {
        let chunking = ChunkerConfig {
            embedding_text_version,
            ..ChunkerConfig::default()
        };
        ModelIdentity {
            family_id: "otzaria-test-family".to_string(),
            tokenizer_checksum: mock::stub_tokenizer_checksum(),
            embedding_dim: DIM,
            pooling: "in-graph".to_string(),
            max_tokens: 512,
            embedding_text_version,
            normalization_version: 1,
            chunking_identity: chunking.identity(),
            query_packages,
        }
    }

    /// Build a base of [`LINES`] for library `version` with the package at `model_path`,
    /// and install it into `vectors`.
    fn install(
        dir: &TempDir,
        name: &str,
        vectors: &Path,
        model_path: &Path,
        model: &ModelIdentity,
        version: u32,
    ) {
        let identity_path = dir.join(&format!("{name}-identity.json"));
        let lines_path = dir.join(&format!("{name}-lines.jsonl"));
        let corpus = CorpusIdentity {
            text: TextIdentity::with_line_text_version(1),
            library_version: version,
            library_release_tag: format!("v{version}-20260930120000"),
            document_id_scheme_version: 1,
        };
        fs::write(&identity_path, serde_json::to_vec(&corpus).unwrap()).unwrap();
        let mut ordinals = std::collections::HashMap::new();
        let body: String = LINES
            .iter()
            .map(|(book, text)| {
                let ordinal: &mut u64 = ordinals.entry(*book).or_default();
                *ordinal += 1;
                let catalogue = if *book == BERACHOT { 2 } else { 1 };
                serde_json::to_string(&CorpusLineRecord {
                    line_id: (catalogue << 32) + *ordinal,
                    line: CorpusLine {
                        source_book_key: book.to_string(),
                        title: String::new(),
                        reference: String::new(),
                        section_id: 1,
                        segment: 0,
                        is_pdf: false,
                        line_hash: 0,
                        content_hash: 1,
                        facets: Vec::new(),
                        text: text.to_string(),
                    },
                })
                .unwrap()
                    + "\n"
            })
            .collect();
        fs::write(&lines_path, body).unwrap();

        let out = dir.join(name);
        let report = build(
            BuildRequest {
                output_path: out.clone(),
                model_path: model_path.to_path_buf(),
                model: model.clone(),
                chunking: ChunkerConfig {
                    embedding_text_version: model.embedding_text_version,
                    ..ChunkerConfig::default()
                },
                created_at: "2026-10-01T00:00:00Z".to_string(),
                batch_size: 2,
                codec: crate::semantic::oxv::codec::CodecSpec::default(),
                allow_non_semantic_backend: true,
            },
            &JsonlCorpus::load(&identity_path, &lines_path).unwrap(),
        )
        .unwrap();
        install_package(
            vectors,
            &InstallSource {
                segment: &out.join(SEGMENT_FILENAME),
                manifest_json: &fs::read_to_string(out.join(RELEASE_MANIFEST_FILENAME)).unwrap(),
            },
            &InstallExpectation {
                identity: report.manifest.identity.clone(),
                published_manifest_sha256: Some(report.manifest_sha256),
            },
            &CancellationToken::new(),
        )
        .unwrap();
    }

    fn config(
        vectors: &Path,
        model_path: &Path,
        model: &ModelIdentity,
        quantization: &str,
    ) -> OfficialIndexConfig {
        OfficialIndexConfig {
            vectors_dir: vectors.to_path_buf(),
            text: TextIdentity::with_line_text_version(1),
            model: LocalModel::of_family(model_path.to_path_buf(), model, quantization),
            deployment: EmbeddingDeployment::default(),
            scan_threads: NonZeroUsize::new(1),
        }
    }

    /// The stub package, and a set of [`LINES`] built with it and installed.
    fn installed(dir: &TempDir) -> (PathBuf, PathBuf, ModelIdentity) {
        let model_path = mock::write_stub_onnx_package(&dir.join("model"));
        let model = family(vec![package(&model_path, "int8")], 1);
        let vectors = dir.join("vectors");
        install(dir, "v30", &vectors, &model_path, &model, 30);
        (model_path, vectors, model)
    }

    fn refused_field(
        result: Result<OfficialSemanticIndex, SemanticSearchError>,
    ) -> Vec<IdentityField> {
        match result.map(|index| index.generation()) {
            Err(SemanticSearchError::Artifact(ArtifactError::IdentityMismatch { mismatches })) => {
                mismatches.iter().map(|mismatch| mismatch.field).collect()
            }
            other => panic!("expected an identity mismatch, got {other:?}"),
        }
    }

    /// The shortest complete path: an installed set opens, reports itself, and a query for
    /// a line's own text finds the key of that text, at the line's position in its book.
    #[test]
    fn an_installed_set_opens_and_a_query_finds_the_line_it_was_built_from() {
        let dir = TempDir::new("official_open");
        let (model_path, vectors, model) = installed(&dir);
        let index =
            OfficialSemanticIndex::open(config(&vectors, &model_path, &model, "int8")).unwrap();

        let status = index.status();
        assert!(status.available && status.model_loaded && status.vectors_persisted);
        assert_eq!(status.vector_count, LINES.len() as u32);
        assert_eq!(status.indexed_book_count, 2);
        assert_eq!(status.vector_backend, STORE_BACKEND_ID);
        assert_eq!(index.set_info().library_version, 30);
        assert_eq!(index.identity().store, readable_store_identity());

        let cancel = CancellationToken::new();
        for (hint, (book, text)) in [(0, LINES[0]), (1, LINES[1]), (0, LINES[2])] {
            let hits = index.search(text, 2, None, &cancel).unwrap();
            assert_eq!(hits.len(), 2);
            assert_eq!(hits[0].key, ChunkKey::of(text));
            assert_eq!(
                (&*hits[0].records[0].book, hits[0].records[0].hint),
                (book, hint)
            );
            assert!(hits[0].score > 0.99, "{}", hits[0].score);
        }

        // A filter is books, and only books it admits are scanned.
        let berachot: BookSet = [BERACHOT].into_iter().collect();
        let hits = index
            .search(LINES[0].1, 5, Some(&berachot), &cancel)
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(&*hits[0].records[0].book, BERACHOT);
    }

    /// A set's passages come from one package of a family, and a query may come from any
    /// package the set accepts. An installation running the int8 package and one running
    /// the fp32 package both open a set built from fp32 that accepts the two; one running a
    /// package outside the list, a package with another tokenizer, another family or
    /// another recipe is refused, by the field that disagreed.
    #[test]
    fn an_installation_opens_a_set_with_any_package_the_set_accepts() {
        let dir = TempDir::new("official_packages");
        let int8 = mock::write_stub_onnx_package(&dir.join("int8"));
        let fp32 = mock::write_stub_onnx_package(&dir.join("fp32"));
        fs::write(&fp32, mock::onnx::stub_graph_named("fp32 weights")).unwrap();
        let model = family(vec![package(&int8, "int8"), package(&fp32, "fp32")], 1);
        let vectors = dir.join("vectors");
        // Built with fp32: the passages' package is on record, and compared by nothing.
        install(&dir, "fp32-built", &vectors, &fp32, &model, 30);

        for (model_path, quantization) in [(&int8, "int8"), (&fp32, "fp32")] {
            let index =
                OfficialSemanticIndex::open(config(&vectors, model_path, &model, quantization))
                    .unwrap_or_else(|error| panic!("{quantization} must open the set: {error}"));
            assert_eq!(index.set_info().slots_live, LINES.len() as u64);
            let provenance = &index.set_info().segments[0].provenance;
            assert_eq!(provenance.passage_package, package(&fp32, "fp32"));
        }

        let int4 = mock::write_stub_onnx_package(&dir.join("int4"));
        fs::write(&int4, mock::onnx::stub_graph_named("int4 weights")).unwrap();
        let retokenized = mock::write_stub_onnx_package(&dir.join("retokenized"));
        fs::write(
            retokenized.with_file_name("tokenizer.json"),
            mock::STUB_TOKENIZER_JSON.replace("[UNK]", "[unk]"),
        )
        .unwrap();
        let mut another_family = config(&vectors, &int8, &model, "int8");
        another_family.model.family_id.push_str("-round3");
        let mut another_recipe = config(&vectors, &int8, &model, "int8");
        another_recipe.model.chunking_identity += 1;
        let mut another_line_recipe = config(&vectors, &int8, &model, "int8");
        another_line_recipe.text.line_text_version = 2;

        for (config, field) in [
            (
                config(&vectors, &int4, &model, "int4"),
                IdentityField::QueryPackages,
            ),
            (
                config(&vectors, &retokenized, &model, "int8"),
                IdentityField::TokenizerChecksum,
            ),
            (another_family, IdentityField::FamilyId),
            (another_recipe, IdentityField::ChunkingIdentity),
            (another_line_recipe, IdentityField::LineTextVersion),
        ] {
            let fields = refused_field(OfficialSemanticIndex::open(config));
            assert!(fields.contains(&field), "{field} must be among {fields:?}");
        }
    }

    /// No set, and a v1 artifact where a set should be, are two different refusals: one is
    /// fixed by installing, the other is a store this build does not read.
    #[test]
    fn no_set_and_a_version_one_artifact_are_refused_by_name() {
        let dir = TempDir::new("official_absent");
        let model_path = mock::write_stub_onnx_package(&dir.join("model"));
        let model = family(vec![package(&model_path, "int8")], 1);
        let vectors = dir.join("vectors");

        match OfficialSemanticIndex::open(config(&vectors, &model_path, &model, "int8"))
            .map(|i| i.generation())
        {
            Err(SemanticSearchError::Artifact(ArtifactError::MetadataUnusable { .. })) => {}
            other => panic!("an empty directory holds no set, got {other:?}"),
        }

        fs::create_dir_all(&vectors).unwrap();
        fs::write(
            vectors.join(crate::distribution::package::MANIFEST_FILENAME),
            r#"{"metadata_version":2,"identity":{"store":{"backend_id":"zevc-persistent-v1"}}}"#,
        )
        .unwrap();
        let fields = refused_field(OfficialSemanticIndex::open(config(
            &vectors,
            &model_path,
            &model,
            "int8",
        )));
        assert_eq!(fields, [IdentityField::StoreBackendId]);
    }

    /// A set built under text recipe 2 is queried the way its passages were built:
    /// normalized, then marked as a query — once.
    #[test]
    fn a_version_two_set_embeds_every_query_with_its_role_prefix() {
        let dir = TempDir::new("official_prefix");
        let model_path = mock::write_stub_onnx_package(&dir.join("model"));
        let model = family(vec![package(&model_path, "int8")], 2);
        let vectors = dir.join("vectors");
        install(&dir, "v2", &vectors, &model_path, &model, 30);
        let index =
            OfficialSemanticIndex::open(config(&vectors, &model_path, &model, "int8")).unwrap();

        let query = LINES[1].1;
        let embedded = |text: &str| {
            let mut vector = mock::hash_embedding(text, DIM);
            crate::semantic::embedding::normalize_validated(&mut vector, DIM).unwrap();
            vector
        };
        let produced = index.embed_query(query).unwrap();
        assert_eq!(produced, embedded(&format!("[QUERY] {query}")));
        assert_ne!(produced, embedded(query));
        assert_ne!(produced, embedded(&format!("[QUERY] [QUERY] {query}")));
        assert!(
            index.embed_query("   ").is_err(),
            "an empty query has nothing to embed"
        );

        // And the passages were stored as passages.
        let stored = index
            .search(query, 1, None, &CancellationToken::new())
            .unwrap();
        assert_eq!(stored[0].key, ChunkKey::of(&format!("[PASSAGE] {query}")));
    }

    /// Where this machine keeps the runtime is no part of what a set must agree with: the
    /// same set opens, under the same identity, whatever the deployment says.
    #[test]
    fn the_deployment_is_not_compared_with_the_set() {
        let dir = TempDir::new("official_deployment");
        let (model_path, vectors, model) = installed(&dir);
        let plain =
            OfficialSemanticIndex::open(config(&vectors, &model_path, &model, "int8")).unwrap();
        let index = OfficialSemanticIndex::open(OfficialIndexConfig {
            deployment: EmbeddingDeployment {
                onnx_runtime: Some(dir.join("bundled").join("onnxruntime.dll")),
            },
            ..config(&vectors, &model_path, &model, "int8")
        })
        .unwrap();
        assert_eq!(index.identity(), plain.identity());
    }

    /// The application's path, cancelled: before the query is embedded the set is never
    /// scanned, and once the scan has begun it stops there — the same error either way, and
    /// the index answers the next query as if nothing had happened.
    #[test]
    fn a_cancelled_query_stops_before_the_scan_or_inside_it() {
        use crate::cancellation::probe;

        let dir = TempDir::new("official_cancel");
        let (model_path, vectors, model) = installed(&dir);
        let index =
            OfficialSemanticIndex::open(config(&vectors, &model_path, &model, "int8")).unwrap();
        let text = LINES[1].1;

        let cancelled = CancellationToken::new();
        cancelled.cancel();
        let (result, checkpoints) =
            probe::checkpoints_of(|| index.search(text, 3, None, &cancelled));
        assert!(
            matches!(result, Err(SemanticSearchError::Cancelled)),
            "{result:?}"
        );
        assert!(checkpoints.is_empty(), "the scan must never have started");

        let cancel = CancellationToken::new();
        let (result, checkpoints) =
            probe::cancelling_at(&cancel, 0, || index.search(text, 3, None, &cancel));
        assert!(
            matches!(result, Err(SemanticSearchError::Cancelled)),
            "{result:?}"
        );
        assert_eq!(checkpoints, [0]);

        let hits = index
            .search(text, 3, None, &CancellationToken::new())
            .unwrap();
        assert_eq!(hits[0].key, ChunkKey::of(text));
    }

    /// A missing model is not a broken set, and the host has to be able to tell them apart
    /// — one is fixed by fetching the model, the other by fetching the vectors.
    #[test]
    fn a_missing_model_is_reported_as_an_embedding_error() {
        let dir = TempDir::new("official_no_model");
        let (_, vectors, model) = installed(&dir);
        let absent = dir.join("not-installed.onnx");
        match OfficialSemanticIndex::open(config(&vectors, &absent, &model, "int8"))
            .map(|i| i.generation())
        {
            Err(SemanticSearchError::EmbeddingRuntime(EmbeddingError::ModelNotFound { path })) => {
                assert!(path.contains("not-installed.onnx"), "{path}")
            }
            other => panic!("expected a model error, got {other:?}"),
        }
    }

    /// An install beside an open index changes nothing until it reloads; a reload opens the
    /// new generation with the same model; and a generation this installation does not
    /// accept is refused while the open one stays in service.
    #[test]
    fn a_reload_opens_the_new_generation_and_keeps_the_old_one_until_it_does() {
        let dir = TempDir::new("official_reload");
        let (model_path, vectors, model) = installed(&dir);
        let mut index =
            OfficialSemanticIndex::open(config(&vectors, &model_path, &model, "int8")).unwrap();
        let first = index.generation();
        assert_eq!(
            index.reload_vectors().unwrap(),
            ReloadOutcome::Unchanged { generation: first }
        );

        install(&dir, "v31", &vectors, &model_path, &model, 31);
        assert_eq!(
            index.set_info().library_version,
            30,
            "nothing changes until a reload"
        );
        let ReloadOutcome::Reloaded {
            from_generation,
            to_generation,
        } = index.reload_vectors().unwrap()
        else {
            panic!("the new generation must be opened")
        };
        assert_eq!(from_generation, first);
        assert!(to_generation > first);
        assert_eq!(index.generation(), to_generation);
        assert_eq!(index.set_info().library_version, 31);

        // A base of another family replaces the set on disk; this installation does not
        // accept it, and goes on serving the generation it has open.
        let other = ModelIdentity {
            family_id: "otzaria-other-family".to_string(),
            ..model.clone()
        };
        install(&dir, "other", &vectors, &model_path, &other, 32);
        let fields = match index.reload_vectors() {
            Err(SemanticSearchError::Artifact(ArtifactError::IdentityMismatch { mismatches })) => {
                mismatches
                    .iter()
                    .map(|mismatch| mismatch.field)
                    .collect::<Vec<_>>()
            }
            other => panic!("another family must be refused, got {other:?}"),
        };
        assert_eq!(fields, [IdentityField::FamilyId]);
        assert_eq!(index.generation(), to_generation);
        let hits = index
            .search(LINES[0].1, 1, None, &CancellationToken::new())
            .unwrap();
        assert_eq!(hits[0].key, ChunkKey::of(LINES[0].1));
    }
}
