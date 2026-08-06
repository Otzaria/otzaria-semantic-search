//! The application's path: open an installed official artifact and query it, read-only.
//!
//! This is the consumer the artifact contract was written for. Everything in
//! [`distribution`](crate::distribution) up to now produced a
//! [`VerifiedPackage`] that nothing read; [`OfficialSemanticIndex`] is what reads it,
//! which is what turns "verify before touching a vector" from a call order someone has to
//! remember into a property of the types: the payload is opened from the token, and the
//! store behind it is a [`VectorSearchBackend`] with no mutation on it at all.
//!
//! # What the caller supplies, and why it is not this crate's constant
//!
//! Opening needs the full [`IndexVersion`] the artifact must match, assembled from three
//! sources that each know a different part of it:
//!
//! * **corpus** — from the Tantivy index that is actually open. It is the half no vector
//!   can reveal: an artifact from another catalogue points confidently at the wrong
//!   lines. The caller passes it; this crate never invents it.
//! * **model** — from the model file that is actually loaded, plus the text recipe this
//!   build implements. [`LocalModel`] declares the recipe; the file's SHA-256 and the
//!   backend that will run it come from the loaded runtime, because they are facts about
//!   this machine rather than claims anyone can make up.
//! * **store** — from what this build can *read*: [`readable_store_identity`]. A payload
//!   in another backend's format, or another version of this one, is then refused by the
//!   ordinary identity comparison rather than by a special case buried in a reader.
//!
//! # Read-only, with one exception that is not a write to the artifact
//!
//! Opening runs [`recover_interrupted_install`] first. An install killed between its two
//! renames leaves the device's only good copy parked beside the target, and a reader that
//! looked only at the target would report no artifact at all. That resolution renames
//! directories the installer left behind; it never writes into a payload, and when there
//! is nothing to resolve it touches nothing. What was found is reported by
//! [`OfficialSemanticIndex::recovery`].
//!
//! # Which failure is which
//!
//! The host has to tell a damaged artifact from a foreign one — one is fixed by fetching
//! this artifact again, the other by fetching the right one — so the errors stay distinct:
//!
//! | Error | Means |
//! |---|---|
//! | [`ArtifactError::IdentityMismatch`] | wrong artifact: the field names what disagreed |
//! | [`ArtifactError::ManifestDisagreesWithPayload`] | the artifact does not describe itself: a payload's size, or a count the payload does not hold |
//! | [`ArtifactError::UnexpectedArtifactDigest`] | self-consistent, but not the artifact that was published |
//! | [`VectorStoreError::Corrupted`](crate::errors::VectorStoreError::Corrupted) | the payload's own structure or per-record checksums are broken |
//! | [`EmbeddingError`](crate::errors::EmbeddingError) | the model is missing or does not fit this configuration |
//!
//! Mapping those onto user-facing states (`ready`, `corrupt`, `incompatible`,
//! `model_missing`) is the host application's job — S5 and S6.

use crate::distribution::importer::{recover_interrupted_install, InstallRecovery};
use crate::distribution::package::{
    ArtifactExpectation, IndexPackage, VerificationDepth, VerifiedPackage,
};
use crate::errors::{ArtifactError, SemanticSearchError};
use crate::semantic::backend::Pooling;
use crate::semantic::embedding::{EmbeddingConfig, EmbeddingRuntime};
use crate::semantic::store_backend::VectorSearchBackend;
use crate::semantic::types::{SearchFilters, SemanticCandidate, SemanticStatus};
use crate::semantic::versioning::{CorpusIdentity, IndexVersion, ModelIdentity, StoreIdentity};
use crate::semantic::zevc_store::{self, ReadOnlyZevcStore, SNAPSHOT_FILENAMES};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// The payload layout this build can read.
///
/// Deliberately a constant and not something read out of the artifact: it is what the
/// installation *requires*. One reader exists today — the snapshot format in
/// [`zevc_store`] — so an artifact written by any other backend, or by a future version
/// of this one, is a rejection naming `store.backend_id` or
/// `store.store_format_version`.
pub fn readable_store_identity() -> StoreIdentity {
    StoreIdentity {
        backend_id: zevc_store::BACKEND_ID.to_string(),
        store_format_version: zevc_store::STORE_FORMAT_VERSION,
        vector_precision: zevc_store::VECTOR_PRECISION.to_string(),
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
/// artifact — the results would be plausible and subtly wrong, and the granularity a
/// `line_id` refers to would no longer be the one the caller hydrates.
#[derive(Debug, Clone)]
pub struct LocalModel {
    pub model_path: PathBuf,
    pub model_id: String,
    /// Quantization of the weights, e.g. `"Q4_K_M"`. Redundant against the checksum by
    /// design: it is what makes a rejection readable.
    pub model_quantization: String,
    pub embedding_dim: u32,
    pub pooling: String,
    pub max_tokens: usize,
    pub embedding_text_version: u32,
    pub normalization_version: u32,
    pub chunking_identity: u64,
}

impl LocalModel {
    /// Compose the model half of the identity from what was declared here and what the
    /// loaded runtime knows.
    fn identity(&self, runtime: &EmbeddingRuntime) -> Result<ModelIdentity, SemanticSearchError> {
        let unknown = |what: &str| {
            SemanticSearchError::Config(format!(
                "the embedding runtime reports no {what} after loading {}, so the \
                 artifact's model identity cannot be compared",
                self.model_path.display()
            ))
        };

        Ok(ModelIdentity {
            model_id: self.model_id.clone(),
            model_checksum: runtime
                .model_checksum()
                .ok_or_else(|| unknown("model checksum"))?
                .to_string(),
            model_quantization: self.model_quantization.clone(),
            embedding_backend: runtime
                .backend_id()
                .ok_or_else(|| unknown("backend id"))?
                .to_string(),
            embedding_dim: self.embedding_dim,
            pooling: self.pooling.clone(),
            max_tokens: self.max_tokens,
            embedding_text_version: self.embedding_text_version,
            normalization_version: self.normalization_version,
            chunking_identity: self.chunking_identity,
        })
    }

    /// The typed pooling strategy, refusing both a spelling [`Pooling`] cannot parse and
    /// one no backend implements — the caller's configuration error either way.
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
    /// Directory the artifact was installed into — the target
    /// [`IndexImporter`](crate::distribution::importer::IndexImporter) swaps into place.
    pub artifact_path: PathBuf,
    /// Identity of the corpus this installation actually has open. Never this crate's
    /// constant; see the module documentation.
    pub corpus: CorpusIdentity,
    pub model: LocalModel,
    /// A digest that arrived from outside the artifact, when there is one. Without it,
    /// opening detects damage and the wrong artifact but not a deliberately rebuilt one —
    /// see [`ArtifactExpectation`].
    pub published_digest: Option<String>,
}

/// An installed artifact, verified and open for queries.
///
/// Holds no chunker, no manifest and no diff: nothing about this index is repairable on
/// the device, so there is nothing to compare a library against and nothing to re-index.
/// A mismatch means this is the wrong artifact, and the answer is to install the right
/// one.
pub struct OfficialSemanticIndex {
    verified: VerifiedPackage,
    store: Box<dyn VectorSearchBackend>,
    runtime: EmbeddingRuntime,
    recovery: InstallRecovery,
    /// Counted once, at open, because the payload cannot change under a read-only store —
    /// and because counting means listing every book key, which is not something
    /// [`Self::status`] should allocate on every call.
    book_count: u32,
}

impl OfficialSemanticIndex {
    /// Resolve any interrupted install, load the model, verify the artifact against this
    /// installation, and open its payload.
    ///
    /// The order is forced rather than chosen. The model is loaded before the artifact is
    /// verified because two identity fields — the file's checksum and the backend that
    /// will run it — are only knowable once it is; a mismatch therefore costs one model
    /// load, and the alternative (comparing the cheap half of the identity first, from a
    /// package nothing has verified yet) would put identity comparison in two places.
    ///
    /// The payload is opened only from the verified token, and then checked against the
    /// counts the manifest declares — the one check the contract layer cannot make,
    /// because counting vectors means reading the store's format.
    pub fn open(config: OfficialIndexConfig) -> Result<Self, SemanticSearchError> {
        let OfficialIndexConfig {
            artifact_path,
            corpus,
            model,
            published_digest,
        } = config;

        let recovery = recover_interrupted_install(&artifact_path)?;
        if recovery.recovered_anything() {
            log::warn!(
                "Resolved an interrupted install at {} before opening it: {recovery:?}",
                artifact_path.display()
            );
        }

        let mut runtime = EmbeddingRuntime::new(EmbeddingConfig {
            model_path: model.model_path.clone(),
            embedding_dim: model.embedding_dim,
            pooling: model.pooling_strategy()?,
            max_tokens: model.max_tokens,
            // One query at a time is all this path ever embeds; batching belongs to the
            // builder, which has a library to get through.
            batch_size: 1,
        });
        runtime.load()?;

        let identity = IndexVersion {
            corpus,
            model: model.identity(&runtime)?,
            store: readable_store_identity(),
        };
        let expected = match published_digest {
            Some(digest) => ArtifactExpectation::with_published_digest(identity, digest),
            None => ArtifactExpectation::without_published_digest(identity),
        };

        let verified = IndexPackage::verify_for_open(&artifact_path, &expected)?;
        // Everything the reader gets comes off the token: the payload set it is allowed to
        // read, the hash each of those files must have, and the record width — which
        // verification has just proved equal to the model's. Nothing here is a path or a
        // number the caller could have supplied.
        let store = ReadOnlyZevcStore::open(
            verified.root(),
            verified.identity().model.embedding_dim,
            snapshot_payloads(&verified)?,
        )?;
        let book_count = verify_counts_against_payload(&verified, &store)?;
        let index = Self {
            verified,
            store: Box::new(store),
            runtime,
            recovery,
            book_count,
        };

        log::info!(
            "Opened official semantic index at {}: {} vector(s) across {} book(s), \
             backend '{}', digest {}",
            index.root().display(),
            index.vector_count(),
            index.book_count(),
            index.store.backend_id(),
            index.artifact_digest()
        );
        Ok(index)
    }

    /// Embed a query and return the closest stored vectors.
    ///
    /// A result carries the `line_id` the artifact was built with; resolving it to a
    /// book, a reference and its text is the lexical index's job, in the caller.
    pub fn search(
        &self,
        query: &str,
        top_k: usize,
        filters: Option<&SearchFilters>,
    ) -> Result<Vec<SemanticCandidate>, SemanticSearchError> {
        let query_vector = self.embed_query(query)?;
        self.search_vector(&query_vector, top_k, filters)
    }

    /// Embed a query separately, so the coordinator can cache the vector.
    pub(crate) fn embed_query(&self, query: &str) -> Result<Vec<f32>, SemanticSearchError> {
        Ok(self.runtime.embed_one(query)?)
    }

    /// Search with a vector this index's runtime already produced.
    pub(crate) fn search_vector(
        &self,
        query_vector: &[f32],
        top_k: usize,
        filters: Option<&SearchFilters>,
    ) -> Result<Vec<SemanticCandidate>, SemanticSearchError> {
        Ok(self.store.search(query_vector, top_k, filters)?)
    }

    /// Operational status, in the same shape the self-built path reports.
    ///
    /// `needs_full_reindex` is always `None`, and not because nothing was checked: an
    /// artifact is either the right one or refused at [`Self::open`]. There is no state
    /// here that re-indexing on the device could repair.
    pub fn status(&self) -> SemanticStatus {
        let vector_count = self.store.count();
        SemanticStatus {
            available: vector_count > 0 && self.runtime.is_loaded(),
            model_loaded: self.runtime.is_loaded(),
            indexed_book_count: self.book_count(),
            vector_count,
            model_id: self.identity().model.model_id.clone(),
            embedding_dim: self.store.embedding_dim(),
            embedding_backend: self.runtime.backend_id().map(str::to_string),
            vector_backend: self.store.backend_id().to_string(),
            vectors_persisted: self.store.is_persistent(),
            needs_full_reindex: None,
            last_error: None,
        }
    }

    /// The artifact directory this index reads.
    pub fn root(&self) -> &Path {
        self.verified.root()
    }

    /// Identity the artifact declared and this installation agreed with.
    pub fn identity(&self) -> &IndexVersion {
        self.verified.identity()
    }

    /// The artifact's own digest — equal to the published one when the configuration
    /// carried it.
    pub fn artifact_digest(&self) -> &str {
        self.verified.artifact_digest()
    }

    /// How much the contract layer checked at open.
    ///
    /// Always [`VerificationDepth::MetadataAndPresence`]: re-hashing the payload at every
    /// launch is not in the budget. It is not the whole story either — the store reader
    /// verifies a checksum per record while it loads, which is what catches a same-length
    /// edit. See [`zevc_store`].
    pub fn verification_depth(&self) -> VerificationDepth {
        self.verified.depth()
    }

    /// What an interrupted install left behind, resolved before this index was opened.
    pub fn recovery(&self) -> InstallRecovery {
        self.recovery
    }

    /// Vectors the payload holds, which the manifest agreed with at open.
    pub fn vector_count(&self) -> u32 {
        self.store.count()
    }

    /// Books the payload's vectors belong to, as counted at open.
    pub fn book_count(&self) -> u32 {
        self.book_count
    }

    /// Book keys the artifact holds vectors for, in a deterministic order.
    pub fn book_keys(&self) -> Vec<String> {
        self.store.book_keys()
    }
}

/// The payload table this backend's reader is allowed to read, taken from the token.
///
/// Refuses anything that is not exactly this backend's layout. Two different faults would
/// otherwise slip through: a package that declares payloads under other names while
/// shipping snapshot files beside them — the reader would then load files the token covers
/// nothing about — and a package that omits one of the three, which is an incomplete
/// snapshot rather than a smaller one.
///
/// The names are compared as a set: [`SNAPSHOT_FILENAMES`] is in read order, and the token's
/// table is sorted.
fn snapshot_payloads(
    verified: &VerifiedPackage,
) -> Result<BTreeMap<String, String>, ArtifactError> {
    let declared: BTreeSet<&str> = verified.payload_names().into_iter().collect();
    let required: BTreeSet<&str> = SNAPSHOT_FILENAMES.into_iter().collect();
    if declared != required {
        return Err(ArtifactError::ManifestDisagreesWithPayload {
            reason: format!(
                "an artifact of store backend '{}' must declare exactly {:?}, and this one \
                 declares {:?}",
                zevc_store::BACKEND_ID,
                required,
                declared
            ),
        });
    }

    Ok(verified
        .payloads()
        .iter()
        .map(|(name, descriptor)| (name.clone(), descriptor.sha256.clone()))
        .collect())
}

/// Check the manifest's counts against what the payload actually holds, and return the
/// book count so nothing has to list every key again.
///
/// [`PackageManifest`](crate::distribution::package::PackageManifest) can only refuse a
/// count of zero on its own — deciding that a payload holds exactly `vector_count` vectors
/// across `book_count` books needs a reader of the store format. The definition the packer
/// has to match: `vector_count` is the number of records, and `book_count` is the number of
/// **distinct `source_book_key`s** among them.
fn verify_counts_against_payload(
    verified: &VerifiedPackage,
    store: &dyn VectorSearchBackend,
) -> Result<u32, ArtifactError> {
    let disagrees = |reason: String| ArtifactError::ManifestDisagreesWithPayload { reason };

    let vectors = store.count();
    if vectors != verified.vector_count() {
        return Err(disagrees(format!(
            "the manifest declares {} vector(s) and the payload holds {vectors}",
            verified.vector_count()
        )));
    }

    let books = store.book_keys().len().min(u32::MAX as usize) as u32;
    if books != verified.book_count() {
        return Err(disagrees(format!(
            "the manifest declares {} book(s) and the payload's vectors belong to {books}",
            verified.book_count()
        )));
    }
    Ok(books)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::importer::{previous_path, ImportConfig, IndexImporter};
    use crate::distribution::package::{IndexPackage, PackageManifest, PayloadDescriptor};
    use crate::errors::{EmbeddingError, VectorStoreError};
    use crate::semantic::embedding::{mock, validate_and_checksum_gguf};
    use crate::semantic::store_backend::VectorStoreBackend;
    use crate::semantic::types::VectorMetadata;
    use crate::semantic::versioning::IdentityField;
    use crate::semantic::zevc_store::{
        ZevcStore, ZevcStoreConfig, METADATA_FILENAME, SNAPSHOT_FILENAMES, VECTORS_FILENAME,
    };
    use sha2::{Digest, Sha256};
    use std::collections::BTreeMap;
    use std::fs;

    struct TempDir(PathBuf);
    impl TempDir {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "otzaria_official_{name}_{}",
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

    const DIM: u32 = 64;

    /// `(line_id, book, text)`, with ids formed the way `document_id_scheme_version` 1
    /// forms them: `((catalogue_order + 1) << 32) + (ordinal + 1)`.
    const LINES: [(u64, &str, &str); 4] = [
        (
            4_294_967_297,
            "otzaria/tanach/genesis.txt",
            "בראשית ברא אלהים את השמים ואת הארץ",
        ),
        (
            4_294_967_298,
            "otzaria/tanach/genesis.txt",
            "והארץ היתה תהו ובהו וחשך על פני תהום",
        ),
        (
            4_294_967_299,
            "otzaria/tanach/genesis.txt",
            "ויאמר אלהים יהי אור ויהי אור",
        ),
        (
            8_589_934_593,
            "otzaria/talmud/berachot.txt",
            "מאימתי קורין את שמע בערבית",
        ),
    ];

    fn corpus() -> CorpusIdentity {
        CorpusIdentity {
            corpus_id: "1f".repeat(32),
            library_version: "otzaria-library-2026-08".to_string(),
            tantivy_schema_version: 3,
            document_id_scheme_version: 1,
        }
    }

    /// What this installation implements — the values a host passes in.
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

    /// What a builder records: the same declarations, plus the two facts only the model
    /// file and the loaded backend can supply.
    fn built_identity(model_path: &Path) -> IndexVersion {
        let model = local_model(model_path);
        IndexVersion {
            corpus: corpus(),
            model: ModelIdentity {
                model_id: model.model_id,
                model_checksum: validate_and_checksum_gguf(model_path).unwrap(),
                model_quantization: model.model_quantization,
                embedding_backend: crate::semantic::backend::MockHashBackend::ID.to_string(),
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

    /// Build an artifact the way the packer will: write the payload with the backend that
    /// owns the format, then the metadata describing it. `adjust` is where a test makes
    /// the package disagree with this installation. Returns the digest a publisher would
    /// announce.
    fn build_artifact(
        root: &Path,
        model_path: &Path,
        adjust: impl FnOnce(&mut PackageManifest),
    ) -> String {
        let store = ZevcStore::open_or_create(ZevcStoreConfig {
            db_path: root.to_path_buf(),
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
                    PayloadDescriptor::of_file(&root.join(name)).unwrap(),
                )
            })
            .collect();
        let mut manifest = PackageManifest::new(
            built_identity(model_path),
            "2026-08-06T00:00:00Z".to_string(),
            2,
            LINES.len() as u32,
            payloads.values().map(|payload| payload.size_bytes).sum(),
        );
        adjust(&mut manifest);

        let package = IndexPackage { manifest, payloads };
        IndexPackage::write(root, &package).unwrap();
        package.digest()
    }

    fn config_for(artifact_path: &Path, model_path: &Path) -> OfficialIndexConfig {
        OfficialIndexConfig {
            artifact_path: artifact_path.to_path_buf(),
            corpus: corpus(),
            model: local_model(model_path),
            published_digest: None,
        }
    }

    /// A model, an artifact built for it, and that artifact installed into `target`.
    fn installed(dir: &TempDir) -> (PathBuf, PathBuf, String) {
        let model_path = dir.path().join("model.gguf");
        mock::write_stub_gguf(&model_path, 3).unwrap();

        let source = dir.path().join("build-output");
        let target = dir.path().join("semantic_index");
        let digest = build_artifact(&source, &model_path, |_| {});

        IndexImporter::new(ImportConfig {
            source_path: source,
            target_store_path: target.clone(),
        })
        .import(&ArtifactExpectation::without_published_digest(
            built_identity(&model_path),
        ))
        .unwrap();

        (model_path, target, digest)
    }

    /// Name, length and SHA-256 of everything in `dir`, so "opening changed nothing" can
    /// be asserted rather than assumed.
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

    #[test]
    fn an_installed_artifact_opens_and_a_query_returns_the_line_it_was_built_from() {
        let dir = TempDir::new("open_and_query");
        let (model_path, target, digest) = installed(&dir);

        let before = fingerprint(&target);
        let index = OfficialSemanticIndex::open(OfficialIndexConfig {
            published_digest: Some(digest.clone()),
            ..config_for(&target, &model_path)
        })
        .unwrap();

        assert_eq!(index.identity(), &built_identity(&model_path));
        assert_eq!(index.artifact_digest(), digest);
        assert_eq!(
            index.verification_depth(),
            VerificationDepth::MetadataAndPresence
        );
        assert_eq!(index.vector_count(), LINES.len() as u32);
        assert_eq!(index.book_count(), 2);
        assert!(!index.recovery().recovered_anything());

        let status = index.status();
        assert!(status.available);
        assert!(
            status.vectors_persisted,
            "an installed artifact is on disk; reporting otherwise would license a re-index"
        );
        assert_eq!(status.vector_count, LINES.len() as u32);
        assert_eq!(status.indexed_book_count, 2);
        assert_eq!(
            status.vector_backend,
            crate::semantic::zevc_store::BACKEND_ID
        );
        assert!(status.needs_full_reindex.is_none());
        assert!(status.last_error.is_none());

        let (line_id, book, text) = LINES[2];
        let hits = index.search(text, 3, None).unwrap();
        assert_eq!(hits[0].metadata.line_id, line_id);
        assert_eq!(hits[0].metadata.source_book_key, book);
        assert!((hits[0].similarity_score - 1.0).abs() < 1e-5);

        // Opening and querying an artifact is a read: no manifest of its own, no
        // re-embedding, not one byte rewritten.
        assert_eq!(fingerprint(&target), before);

        // And a restart opens the same artifact again, without building anything.
        drop(index);
        let reopened = OfficialSemanticIndex::open(config_for(&target, &model_path)).unwrap();
        assert_eq!(reopened.vector_count(), LINES.len() as u32);
        assert_eq!(
            reopened.search(text, 1, None).unwrap()[0].metadata.line_id,
            line_id
        );
        assert_eq!(fingerprint(&target), before);
    }

    /// The gap the contract layer cannot close on its own: a count is a claim about the
    /// payload's *content*, and settling it means reading the store's format.
    #[test]
    fn counts_the_payload_does_not_hold_are_refused_at_open() {
        let dir = TempDir::new("counts");
        let model_path = dir.path().join("model.gguf");
        mock::write_stub_gguf(&model_path, 3).unwrap();

        for (label, adjust) in [
            (
                "vector",
                (|m: &mut PackageManifest| m.vector_count += 1) as fn(&mut PackageManifest),
            ),
            ("book", |m: &mut PackageManifest| m.book_count = 7),
        ] {
            let source = dir.path().join(format!("artifact-{label}"));
            build_artifact(&source, &model_path, adjust);

            match OfficialSemanticIndex::open(config_for(&source, &model_path))
                .map(|index| index.vector_count())
            {
                Err(SemanticSearchError::Artifact(
                    ArtifactError::ManifestDisagreesWithPayload { reason },
                )) => assert!(reason.contains(label), "{reason}"),
                other => panic!("a wrong {label} count must be refused, got {other:?}"),
            }
        }
    }

    /// The pairing that makes the two verification depths honest: the cheap one cannot see
    /// this edit, and the reader can — so the claim "an artifact that is tampered with
    /// stops opening" holds for the *runtime* path even though it does not hold for
    /// `verify_for_open` alone.
    ///
    /// Both forms are exercised, because they are caught by different things. A raw edit is
    /// caught by the record's own checksum. An edit that also repairs that checksum — the
    /// one a payload's internal checks are structurally unable to see — is caught only
    /// because the reader compares each file against the hash `payloads.json` declares, and
    /// that declaration is what a published digest pins.
    #[test]
    fn a_same_length_payload_edit_passes_verification_and_is_caught_by_the_reader() {
        for forge_the_checksum_too in [false, true] {
            let dir = TempDir::new("tamper");
            let (model_path, target, _) = installed(&dir);

            let vectors_path = target.join(VECTORS_FILENAME);
            let mut bytes = fs::read(&vectors_path).unwrap();
            let before = bytes.len();
            bytes[0] ^= 0xff;
            fs::write(&vectors_path, &bytes).unwrap();
            assert_eq!(fs::metadata(&vectors_path).unwrap().len() as usize, before);

            if forge_the_checksum_too {
                let metadata_path = target.join(METADATA_FILENAME);
                let text = fs::read_to_string(&metadata_path).unwrap();
                let before = text.len();
                let mut lines: Vec<String> = text.lines().map(str::to_string).collect();
                let mut first: serde_json::Value = serde_json::from_str(&lines[0]).unwrap();
                first["vector_sha256"] = serde_json::Value::String(format!(
                    "{:x}",
                    Sha256::digest(&bytes[..DIM as usize * 4])
                ));
                lines[0] = serde_json::to_string(&first).unwrap();
                fs::write(&metadata_path, format!("{}\n", lines.join("\n"))).unwrap();
                assert_eq!(
                    fs::read_to_string(&metadata_path).unwrap().len(),
                    before,
                    "a forgery that changes a length would be caught by the cheap depth"
                );
            }

            assert!(
                IndexPackage::verify_for_open(
                    &target,
                    &ArtifactExpectation::without_published_digest(built_identity(&model_path))
                )
                .is_ok(),
                "a same-length edit is invisible without hashing the payload"
            );

            match OfficialSemanticIndex::open(config_for(&target, &model_path))
                .map(|index| index.vector_count())
            {
                Err(SemanticSearchError::VectorStore(VectorStoreError::Corrupted { .. })) => {}
                other => panic!(
                    "the reader must refuse an edited payload \
                     (checksum forged: {forge_the_checksum_too}), got {other:?}"
                ),
            }
        }
    }

    /// An artifact of this backend is exactly three payload files. A package that declares
    /// anything else — while shipping snapshot files beside them — would have the reader
    /// loading bytes the token covers nothing about.
    #[test]
    fn a_package_that_does_not_declare_this_backends_payloads_is_refused() {
        let dir = TempDir::new("payload_set");
        let model_path = dir.path().join("model.gguf");
        mock::write_stub_gguf(&model_path, 3).unwrap();

        let source = dir.path().join("build-output");
        build_artifact(&source, &model_path, |_| {});

        // Re-declare the package over a decoy payload, leaving the real snapshot in place.
        let decoy = "decoy.bin";
        fs::write(source.join(decoy), b"not a snapshot").unwrap();
        let payloads = BTreeMap::from([(
            decoy.to_string(),
            PayloadDescriptor::of_file(&source.join(decoy)).unwrap(),
        )]);
        let manifest = PackageManifest::new(
            built_identity(&model_path),
            "2026-08-06T00:00:00Z".to_string(),
            2,
            LINES.len() as u32,
            payloads[decoy].size_bytes,
        );
        IndexPackage::write(&source, &IndexPackage { manifest, payloads }).unwrap();

        match OfficialSemanticIndex::open(config_for(&source, &model_path))
            .map(|index| index.vector_count())
        {
            Err(SemanticSearchError::Artifact(ArtifactError::ManifestDisagreesWithPayload {
                reason,
            })) => assert!(
                reason.contains(decoy) || reason.contains(VECTORS_FILENAME),
                "{reason}"
            ),
            other => panic!("a foreign payload set must be refused, got {other:?}"),
        }
    }

    #[test]
    fn an_artifact_built_for_another_corpus_or_another_model_is_refused_by_field_name() {
        let dir = TempDir::new("mismatch");
        let model_path = dir.path().join("model.gguf");
        mock::write_stub_gguf(&model_path, 3).unwrap();

        // Same library name, one book inserted in the middle: every `line_id` after it
        // now names a different line, and nothing in the vectors says so.
        let foreign_corpus = dir.path().join("foreign-corpus");
        build_artifact(&foreign_corpus, &model_path, |manifest| {
            manifest.identity.corpus.corpus_id = "2e".repeat(32)
        });

        // Same `model_id`, different weights behind it.
        let other_model = dir.path().join("other-model.gguf");
        mock::write_stub_gguf(&other_model, 2).unwrap();
        let foreign_model = dir.path().join("foreign-model");
        build_artifact(&foreign_model, &model_path, |manifest| {
            manifest.identity.model.model_checksum =
                validate_and_checksum_gguf(&other_model).unwrap()
        });

        for (source, field) in [
            (foreign_corpus, IdentityField::CorpusId),
            (foreign_model, IdentityField::ModelChecksum),
        ] {
            match OfficialSemanticIndex::open(config_for(&source, &model_path))
                .map(|index| index.vector_count())
            {
                Err(SemanticSearchError::Artifact(ArtifactError::IdentityMismatch {
                    mismatches,
                })) => {
                    assert_eq!(mismatches.len(), 1, "{field}");
                    assert_eq!(mismatches[0].field, field);
                }
                other => panic!("{field} must refuse the artifact, got {other:?}"),
            }
        }
    }

    /// The reason opening runs recovery at all: killed between the installer's two
    /// renames, the target is gone and the device's only good copy is parked beside it.
    #[test]
    fn opening_resolves_an_install_that_was_interrupted_between_the_two_renames() {
        let dir = TempDir::new("interrupted");
        let (model_path, target, _) = installed(&dir);

        fs::rename(&target, previous_path(&target).unwrap()).unwrap();
        assert!(!target.exists());

        let index = OfficialSemanticIndex::open(config_for(&target, &model_path)).unwrap();
        assert!(index.recovery().restored_previous);
        assert_eq!(index.vector_count(), LINES.len() as u32);
        assert_eq!(
            index.search(LINES[0].2, 1, None).unwrap()[0]
                .metadata
                .line_id,
            LINES[0].0
        );
    }

    /// A missing model is not a broken artifact, and the host has to be able to tell them
    /// apart — one is fixed by fetching the model, the other by fetching the index.
    #[test]
    fn a_missing_model_is_reported_as_an_embedding_error() {
        let dir = TempDir::new("no_model");
        let (_, target, _) = installed(&dir);

        let absent = dir.path().join("not-installed.gguf");
        match OfficialSemanticIndex::open(config_for(&target, &absent))
            .map(|index| index.vector_count())
        {
            Err(SemanticSearchError::EmbeddingRuntime(EmbeddingError::ModelNotFound { path })) => {
                assert!(path.contains("not-installed.gguf"), "{path}")
            }
            other => panic!("expected a model error, got {other:?}"),
        }
    }
}
