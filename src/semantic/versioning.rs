//! Identity of an official semantic artifact — what a package must declare, and what
//! the installation must agree with before a single vector is read.
//!
//! A semantic result is a global `line_id` that Tantivy resolves back to a book, a
//! reference and a text. One book added in the middle of the catalogue shifts the ids
//! of every book after it, so an artifact built from another catalogue does not merely
//! miss results — it points at the *wrong lines*, with no symptom a caller can see.
//! The model side behaves the same way: the same `model_id` over different weights
//! produces vectors in a different space, and the vectors themselves never say so.
//!
//! [`IndexVersion`] is therefore the whole set of facts that has to agree: which
//! corpus, which Tantivy schema and id scheme, which model file and inference backend,
//! and which store format. Every value is **data carried by the artifact**, not a
//! constant in this crate — S1 has not chosen the dimension or the precision yet, and
//! this contract does not need it to.
//!
//! Two things separate this from
//! [`SemanticManifest`](crate::semantic::manifest::SemanticManifest), which tracks an
//! index this installation built itself:
//!
//! * Nothing here is repairable on the device. There is no re-chunking and no partial
//!   re-index — a mismatch means this is the wrong artifact, so every field is fatal
//!   and none is a "carry on with a warning".
//! * Nothing here may be left unknown. A field the builder did not fill is refused by
//!   [`IndexVersion::validate_complete`] rather than compared as an empty string,
//!   because an unfilled identity would otherwise match another unfilled identity.
//!
//! See `docs/ARTIFACT_CONTRACT.md` for the field-by-field contract and for who fills
//! each value.

use crate::errors::ArtifactError;
use serde::{Deserialize, Serialize};

/// Full identity of a built artifact, in the three groups that must agree
/// independently.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IndexVersion {
    pub corpus: CorpusIdentity,
    pub model: ModelIdentity,
    pub store: StoreIdentity,
}

/// Which library edition the vectors describe, and how its line ids were formed.
///
/// This is the group that cannot be recovered from the vectors: everything else at
/// worst produces bad scores, while a corpus mismatch produces confident results for
/// the wrong lines.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CorpusIdentity {
    /// Deterministic digest of the official library documents the artifact was built
    /// from — the same value the Tantivy artifact reports, so the two can be paired
    /// without trusting a version string.
    pub corpus_id: String,
    /// Library/catalogue release the digest belongs to.
    ///
    /// Compared like everything else here, and therefore fatal — it is not a label along
    /// for the ride. The expected value comes from the same Tantivy artifact that
    /// supplies `corpus_id`, so a disagreement means two releases claim the same
    /// documents under different names, which is a build-pipeline fault and not
    /// something a reader may paper over. The cost of refusing is one relabelled
    /// rebuild; the cost of accepting is a version string nobody can trust.
    pub library_version: String,
    /// Version of the Tantivy schema whose stored fields hydration reads.
    pub tantivy_schema_version: u32,
    /// Version of the scheme that composes a `line_id`.
    ///
    /// Version 1 is what `otzaria_search_engine` builds today, matching the app's
    /// `buildCatalogueDocumentId`: `((catalogue_order + 1) << 32) + (ordinal + 1)`. Both
    /// halves are 1-based, so no live document has id 0. The exact arithmetic matters
    /// because the builder in S4 has to reproduce it from Tantivy, not approximate it.
    pub document_id_scheme_version: u32,
}

/// What turned text into vectors. A change anywhere here means the query embedder and
/// the stored vectors live in different spaces.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelIdentity {
    pub model_id: String,
    /// SHA-256 of the model file the vectors were produced with, as 64 lowercase hex
    /// digits. Guards what `model_id` cannot: the same id over different weights.
    pub model_checksum: String,
    /// Quantization the vectors were produced under (e.g. `"Q4_K_M"`). Redundant
    /// against `model_checksum` by design — it is what makes a rejection readable.
    pub model_quantization: String,
    /// Inference backend that produced the vectors (e.g. `"llama-cpp-2-0.1.153"`).
    /// Two backends over the same weights agree to about cosine 0.995, not exactly.
    pub embedding_backend: String,
    pub embedding_dim: u32,
    pub pooling: String,
    /// Token cap the embedded texts were built under: it decides how much of a long
    /// line reached the model at all.
    pub max_tokens: usize,
    /// Which text a vector was built from — line alone, title + reference + line,
    /// neighbour context. The recipe is S1's decision; the field exists so that the
    /// decision is recorded in the artifact rather than compiled into a reader.
    pub embedding_text_version: u32,
    /// Text normalization version applied before embedding.
    pub normalization_version: u32,
    /// Identity of the whole chunker configuration — see
    /// [`ChunkerConfig::identity`](crate::semantic::chunker::ChunkerConfig::identity).
    pub chunking_identity: u64,
}

/// How the vectors are laid out on disk. Decides whether this build can read the
/// payload at all.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StoreIdentity {
    /// Backend that wrote the payload, matching
    /// [`VectorSearchBackend::backend_id`](crate::semantic::store_backend::VectorSearchBackend::backend_id).
    pub backend_id: String,
    /// Version of that backend's on-disk format. Separate from `backend_id` so a
    /// format change inside one backend is a rejection and not a misread payload.
    pub store_format_version: u32,
    /// Precision the vectors are stored at (`"f32"`, `"f16"`, `"int8"`). The chosen
    /// value is S1's measurement; carrying it is not.
    pub vector_precision: String,
}

/// One comparable identity field, named as it appears in the artifact metadata.
///
/// The enum exists so that comparison, the artifact digest and rejection messages all
/// walk the *same* list, rather than three hand-written traversals that can disagree.
///
/// It is not a compile-time guarantee of coverage: `IndexVersion` is a plain struct, and
/// nothing in the type system forces a new field to appear here. Two tests do that job
/// instead, both driven by the *serialized* identity, so they see exactly the fields an
/// artifact carries:
///
/// * `every_serialized_identity_field_is_comparable` — a field that is stored but absent
///   from [`Self::ALL`] fails, because it would be shipped and never compared.
/// * `every_serialized_identity_field_is_refused_when_left_unfilled` — a field
///   [`IndexVersion::validate_complete`] forgot fails, because a blank value would be
///   compared against another blank value and agree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IdentityField {
    CorpusId,
    LibraryVersion,
    TantivySchemaVersion,
    DocumentIdSchemeVersion,
    ModelId,
    ModelChecksum,
    ModelQuantization,
    EmbeddingBackend,
    EmbeddingDim,
    Pooling,
    MaxTokens,
    EmbeddingTextVersion,
    NormalizationVersion,
    ChunkingIdentity,
    StoreBackendId,
    StoreFormatVersion,
    VectorPrecision,
}

/// The three things that must agree independently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IdentityGroup {
    Corpus,
    Model,
    Store,
}

impl IdentityField {
    /// Every comparable field, in the order rejections report them.
    pub const ALL: [IdentityField; 17] = [
        Self::CorpusId,
        Self::LibraryVersion,
        Self::TantivySchemaVersion,
        Self::DocumentIdSchemeVersion,
        Self::ModelId,
        Self::ModelChecksum,
        Self::ModelQuantization,
        Self::EmbeddingBackend,
        Self::EmbeddingDim,
        Self::Pooling,
        Self::MaxTokens,
        Self::EmbeddingTextVersion,
        Self::NormalizationVersion,
        Self::ChunkingIdentity,
        Self::StoreBackendId,
        Self::StoreFormatVersion,
        Self::VectorPrecision,
    ];

    pub fn group(self) -> IdentityGroup {
        match self {
            Self::CorpusId
            | Self::LibraryVersion
            | Self::TantivySchemaVersion
            | Self::DocumentIdSchemeVersion => IdentityGroup::Corpus,
            Self::ModelId
            | Self::ModelChecksum
            | Self::ModelQuantization
            | Self::EmbeddingBackend
            | Self::EmbeddingDim
            | Self::Pooling
            | Self::MaxTokens
            | Self::EmbeddingTextVersion
            | Self::NormalizationVersion
            | Self::ChunkingIdentity => IdentityGroup::Model,
            Self::StoreBackendId | Self::StoreFormatVersion | Self::VectorPrecision => {
                IdentityGroup::Store
            }
        }
    }

    /// Path of the field in the serialized metadata, so a rejection names something
    /// the reader can find in `manifest.json`.
    pub fn path(self) -> &'static str {
        match self {
            Self::CorpusId => "corpus.corpus_id",
            Self::LibraryVersion => "corpus.library_version",
            Self::TantivySchemaVersion => "corpus.tantivy_schema_version",
            Self::DocumentIdSchemeVersion => "corpus.document_id_scheme_version",
            Self::ModelId => "model.model_id",
            Self::ModelChecksum => "model.model_checksum",
            Self::ModelQuantization => "model.model_quantization",
            Self::EmbeddingBackend => "model.embedding_backend",
            Self::EmbeddingDim => "model.embedding_dim",
            Self::Pooling => "model.pooling",
            Self::MaxTokens => "model.max_tokens",
            Self::EmbeddingTextVersion => "model.embedding_text_version",
            Self::NormalizationVersion => "model.normalization_version",
            Self::ChunkingIdentity => "model.chunking_identity",
            Self::StoreBackendId => "store.backend_id",
            Self::StoreFormatVersion => "store.store_format_version",
            Self::VectorPrecision => "store.vector_precision",
        }
    }
}

impl std::fmt::Display for IdentityField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.path())
    }
}

impl std::fmt::Display for IdentityGroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Corpus => "corpus",
            Self::Model => "model",
            Self::Store => "store",
        })
    }
}

/// One field on which an artifact and the installation disagree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IdentityMismatch {
    pub field: IdentityField,
    /// What the artifact declares.
    pub artifact: String,
    /// What this installation requires.
    pub expected: String,
}

impl std::fmt::Display for IdentityMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: artifact='{}', expected='{}'",
            self.field, self.artifact, self.expected
        )
    }
}

/// Render a rejection as one line, listing every disagreement rather than the first.
pub fn describe_identity_mismatches(mismatches: &[IdentityMismatch]) -> String {
    mismatches
        .iter()
        .map(|mismatch| mismatch.to_string())
        .collect::<Vec<_>>()
        .join("; ")
}

impl IndexVersion {
    /// The value of one field, formatted the same way for both sides of a comparison.
    pub fn value(&self, field: IdentityField) -> String {
        use IdentityField as F;
        match field {
            F::CorpusId => self.corpus.corpus_id.clone(),
            F::LibraryVersion => self.corpus.library_version.clone(),
            F::TantivySchemaVersion => self.corpus.tantivy_schema_version.to_string(),
            F::DocumentIdSchemeVersion => self.corpus.document_id_scheme_version.to_string(),
            F::ModelId => self.model.model_id.clone(),
            F::ModelChecksum => self.model.model_checksum.clone(),
            F::ModelQuantization => self.model.model_quantization.clone(),
            F::EmbeddingBackend => self.model.embedding_backend.clone(),
            F::EmbeddingDim => self.model.embedding_dim.to_string(),
            F::Pooling => self.model.pooling.clone(),
            F::MaxTokens => self.model.max_tokens.to_string(),
            F::EmbeddingTextVersion => self.model.embedding_text_version.to_string(),
            F::NormalizationVersion => self.model.normalization_version.to_string(),
            F::ChunkingIdentity => self.model.chunking_identity.to_string(),
            F::StoreBackendId => self.store.backend_id.clone(),
            F::StoreFormatVersion => self.store.store_format_version.to_string(),
            F::VectorPrecision => self.store.vector_precision.clone(),
        }
    }

    /// Every field on which this artifact disagrees with what the installation
    /// requires, in [`IdentityField::ALL`] order. All of them, not the first: a
    /// rejection the user has to fix one field per attempt is a rejection nobody
    /// finishes reading.
    pub fn mismatches_against(&self, expected: &IndexVersion) -> Vec<IdentityMismatch> {
        IdentityField::ALL
            .iter()
            .filter_map(|&field| {
                let artifact = self.value(field);
                let required = expected.value(field);
                (artifact != required).then_some(IdentityMismatch {
                    field,
                    artifact,
                    expected: required,
                })
            })
            .collect()
    }

    pub fn is_compatible(&self, expected: &IndexVersion) -> bool {
        self.mismatches_against(expected).is_empty()
    }

    /// Refuse the artifact unless it matches the installation exactly.
    ///
    /// Deliberately not a bool: the caller has to be able to report *what* disagreed,
    /// and a rejection reduced to `false` is what the product contract calls a guess.
    pub fn verify_matches(&self, expected: &IndexVersion) -> Result<(), ArtifactError> {
        let mismatches = self.mismatches_against(expected);
        if mismatches.is_empty() {
            Ok(())
        } else {
            Err(ArtifactError::IdentityMismatch { mismatches })
        }
    }

    /// Refuse an identity whose fields exist but say nothing: a blank string, a zero
    /// version, a checksum that is not one.
    ///
    /// This runs before [`Self::verify_matches`] because two unfilled identities
    /// compare equal. A builder that forgot to record `corpus_id` would otherwise
    /// produce an artifact that opens against any corpus at all.
    pub fn validate_complete(&self) -> Result<(), ArtifactError> {
        use IdentityField as F;

        require_text(F::CorpusId, &self.corpus.corpus_id)?;
        require_text(F::LibraryVersion, &self.corpus.library_version)?;
        require_positive(
            F::TantivySchemaVersion,
            self.corpus.tantivy_schema_version.into(),
        )?;
        require_positive(
            F::DocumentIdSchemeVersion,
            self.corpus.document_id_scheme_version.into(),
        )?;

        require_text(F::ModelId, &self.model.model_id)?;
        require_sha256(F::ModelChecksum, &self.model.model_checksum)?;
        require_text(F::ModelQuantization, &self.model.model_quantization)?;
        require_text(F::EmbeddingBackend, &self.model.embedding_backend)?;
        require_positive(F::EmbeddingDim, self.model.embedding_dim.into())?;
        require_text(F::Pooling, &self.model.pooling)?;
        require_positive(F::MaxTokens, self.model.max_tokens as u64)?;
        require_positive(
            F::EmbeddingTextVersion,
            self.model.embedding_text_version.into(),
        )?;
        require_positive(
            F::NormalizationVersion,
            self.model.normalization_version.into(),
        )?;
        require_positive(F::ChunkingIdentity, self.model.chunking_identity)?;

        require_text(F::StoreBackendId, &self.store.backend_id)?;
        require_positive(
            F::StoreFormatVersion,
            self.store.store_format_version.into(),
        )?;
        require_text(F::VectorPrecision, &self.store.vector_precision)?;

        Ok(())
    }
}

fn require_text(field: IdentityField, value: &str) -> Result<(), ArtifactError> {
    if value.trim().is_empty() {
        return Err(ArtifactError::IncompleteIdentity {
            field,
            reason: "is blank".to_string(),
        });
    }
    // A newline inside an identity value would make the canonical text behind
    // `IndexPackage::digest` ambiguous — two different identities could serialize to the
    // same bytes — and a control character makes a rejection message unreadable.
    if value.chars().any(char::is_control) {
        return Err(ArtifactError::IncompleteIdentity {
            field,
            reason: "contains a control character".to_string(),
        });
    }
    Ok(())
}

/// Zero is what an unfilled numeric field deserializes to, so the contract numbers
/// every version and identity from 1.
fn require_positive(field: IdentityField, value: u64) -> Result<(), ArtifactError> {
    if value == 0 {
        return Err(ArtifactError::IncompleteIdentity {
            field,
            reason: "is zero".to_string(),
        });
    }
    Ok(())
}

/// Lowercase is required, not normalized: comparison is a string equality, and
/// accepting both cases would make the same checksum mismatch itself.
fn require_sha256(field: IdentityField, value: &str) -> Result<(), ArtifactError> {
    let valid = value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase());
    if !valid {
        return Err(ArtifactError::IncompleteIdentity {
            field,
            reason: "is not a SHA-256 of 64 lowercase hex digits".to_string(),
        });
    }
    Ok(())
}

impl std::fmt::Display for IndexVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let checksum: String = self.model.model_checksum.chars().take(12).collect();
        write!(
            f,
            "corpus {} ({}, tantivy-schema v{}, id-scheme v{}); \
             model {} {} via {} [{}…] (dim={}, {}, {} tok, text v{}, norm v{}, chunk {}); \
             store {} v{} {}",
            self.corpus.corpus_id,
            self.corpus.library_version,
            self.corpus.tantivy_schema_version,
            self.corpus.document_id_scheme_version,
            self.model.model_id,
            self.model.model_quantization,
            self.model.embedding_backend,
            checksum,
            self.model.embedding_dim,
            self.model.pooling,
            self.model.max_tokens,
            self.model.embedding_text_version,
            self.model.normalization_version,
            self.model.chunking_identity,
            self.store.backend_id,
            self.store.store_format_version,
            self.store.vector_precision,
        )
    }
}

/// A complete identity for tests across this crate: every field filled with something
/// that passes [`IndexVersion::validate_complete`], so a test that wants to exercise a
/// *specific* rejection changes one field and nothing else.
#[cfg(test)]
pub(crate) fn test_identity() -> IndexVersion {
    IndexVersion {
        corpus: CorpusIdentity {
            corpus_id: "c".repeat(64),
            library_version: "otzaria-library-2026-08".to_string(),
            tantivy_schema_version: 3,
            document_id_scheme_version: 1,
        },
        model: ModelIdentity {
            model_id: "EMD123/Otzaria-Embedding-V1-Flash-0.6B".to_string(),
            model_checksum: "a".repeat(64),
            model_quantization: "Q4_K_M".to_string(),
            embedding_backend: "llama-cpp-2-0.1.153".to_string(),
            embedding_dim: 1024,
            pooling: "last-token".to_string(),
            max_tokens: 512,
            embedding_text_version: 1,
            normalization_version: 1,
            chunking_identity: 0x51A1_1E55,
        },
        store: StoreIdentity {
            backend_id: "zevc-persistent-v1".to_string(),
            store_format_version: 1,
            vector_precision: "f32".to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn sample_identity() -> IndexVersion {
        test_identity()
    }

    /// A single-field edit to an otherwise complete identity.
    type Mutation = (IdentityField, fn(&mut IndexVersion));

    /// One mutation per comparable field. The test below asserts the table covers
    /// [`IdentityField::ALL`], so adding a field to the identity without adding it
    /// here fails rather than shipping a field nobody compares.
    fn field_mutations() -> Vec<Mutation> {
        use IdentityField as F;
        vec![
            (F::CorpusId, |v| v.corpus.corpus_id = "d".repeat(64)),
            (F::LibraryVersion, |v| {
                v.corpus.library_version = "otzaria-library-2026-09".to_string()
            }),
            (F::TantivySchemaVersion, |v| {
                v.corpus.tantivy_schema_version = 4
            }),
            (F::DocumentIdSchemeVersion, |v| {
                v.corpus.document_id_scheme_version = 2
            }),
            (F::ModelId, |v| v.model.model_id = "other/model".to_string()),
            (F::ModelChecksum, |v| {
                v.model.model_checksum = "b".repeat(64)
            }),
            (F::ModelQuantization, |v| {
                v.model.model_quantization = "Q8_0".to_string()
            }),
            (F::EmbeddingBackend, |v| {
                v.model.embedding_backend = "candle-gguf-v1".to_string()
            }),
            (F::EmbeddingDim, |v| v.model.embedding_dim = 256),
            (F::Pooling, |v| v.model.pooling = "mean".to_string()),
            (F::MaxTokens, |v| v.model.max_tokens = 8192),
            (F::EmbeddingTextVersion, |v| {
                v.model.embedding_text_version = 2
            }),
            (F::NormalizationVersion, |v| {
                v.model.normalization_version = 2
            }),
            (F::ChunkingIdentity, |v| v.model.chunking_identity = 999),
            (F::StoreBackendId, |v| {
                v.store.backend_id = "mmap-flat-v1".to_string()
            }),
            (F::StoreFormatVersion, |v| v.store.store_format_version = 2),
            (F::VectorPrecision, |v| {
                v.store.vector_precision = "int8".to_string()
            }),
        ]
    }

    #[test]
    fn an_identical_identity_matches() {
        let identity = sample_identity();
        assert!(identity.is_compatible(&sample_identity()));
        assert!(identity.verify_matches(&sample_identity()).is_ok());
        assert!(identity.validate_complete().is_ok());
    }

    /// The acceptance gate for the artifact contract: a change in *any* identity is a
    /// named rejection, not a warning and not a partial open.
    #[test]
    fn changing_any_single_identity_field_is_refused_and_named() {
        let expected = sample_identity();
        let mutations = field_mutations();

        let covered: HashSet<IdentityField> = mutations.iter().map(|(field, _)| *field).collect();
        let all: HashSet<IdentityField> = IdentityField::ALL.into_iter().collect();
        assert_eq!(
            covered, all,
            "every identity field needs a mutation here, or it is carried but never compared"
        );

        for (field, mutate) in mutations {
            let mut artifact = sample_identity();
            mutate(&mut artifact);

            assert!(
                !artifact.is_compatible(&expected),
                "{field} must make the artifact incompatible"
            );

            let mismatches = artifact.mismatches_against(&expected);
            assert_eq!(mismatches.len(), 1, "{field} must report exactly one field");
            assert_eq!(mismatches[0].field, field);
            assert_eq!(mismatches[0].expected, expected.value(field));
            assert_eq!(mismatches[0].artifact, artifact.value(field));

            match artifact.verify_matches(&expected) {
                Err(ArtifactError::IdentityMismatch { mismatches }) => {
                    assert!(describe_identity_mismatches(&mismatches).contains(field.path()));
                }
                other => panic!("{field} must be rejected, got {other:?}"),
            }
        }
    }

    /// The scenario the corpus group exists for: same library, one book inserted, so
    /// the ids the vectors carry point at the wrong lines.
    #[test]
    fn an_artifact_from_another_catalogue_is_refused() {
        let expected = sample_identity();
        let mut artifact = sample_identity();
        artifact.corpus.corpus_id = "e".repeat(64);

        let mismatches = artifact.mismatches_against(&expected);
        assert_eq!(mismatches[0].field, IdentityField::CorpusId);
        assert_eq!(mismatches[0].field.group(), IdentityGroup::Corpus);
    }

    /// `model_id` cannot see this, which is why the checksum is part of the identity
    /// and not only of the local manifest.
    #[test]
    fn a_model_file_swapped_behind_the_same_id_is_refused() {
        let expected = sample_identity();
        let mut artifact = sample_identity();
        artifact.model.model_checksum = "f".repeat(64);

        assert_eq!(artifact.model.model_id, expected.model.model_id);
        let mismatches = artifact.mismatches_against(&expected);
        assert_eq!(mismatches.len(), 1);
        assert_eq!(mismatches[0].field, IdentityField::ModelChecksum);
    }

    #[test]
    fn every_disagreement_is_reported_at_once_in_field_order() {
        let expected = sample_identity();
        let mut artifact = sample_identity();
        artifact.store.vector_precision = "f16".to_string();
        artifact.corpus.corpus_id = "d".repeat(64);
        artifact.model.embedding_dim = 256;

        let mismatches = artifact.mismatches_against(&expected);
        let fields: Vec<IdentityField> = mismatches.iter().map(|m| m.field).collect();
        assert_eq!(
            fields,
            vec![
                IdentityField::CorpusId,
                IdentityField::EmbeddingDim,
                IdentityField::VectorPrecision
            ]
        );

        let rendered = describe_identity_mismatches(&mismatches);
        for field in fields {
            assert!(rendered.contains(field.path()), "{field} must be listed");
        }
    }

    /// Every field path in the *serialized* identity, as `group.field`. Reading it off
    /// the JSON rather than off a hand-written list is the point: this is exactly the set
    /// of fields an artifact carries on disk.
    fn serialized_field_paths() -> Vec<String> {
        let value = serde_json::to_value(sample_identity()).unwrap();
        let mut paths = Vec::new();
        for (group, fields) in value.as_object().expect("the identity is a JSON object") {
            for field in fields
                .as_object()
                .unwrap_or_else(|| panic!("group {group} is a JSON object"))
                .keys()
            {
                paths.push(format!("{group}.{field}"));
            }
        }
        paths.sort();
        paths
    }

    /// A field that is stored in the artifact but missing from `IdentityField::ALL` would
    /// be shipped, trusted, and never compared. This is what makes that impossible to add
    /// by accident.
    #[test]
    fn every_serialized_identity_field_is_comparable() {
        let carried: HashSet<String> = serialized_field_paths().into_iter().collect();
        let compared: HashSet<String> = IdentityField::ALL
            .iter()
            .map(|field| field.path().to_string())
            .collect();

        assert_eq!(
            carried, compared,
            "the artifact carries fields that are not compared, or names fields it does not carry"
        );
    }

    /// The other half: a field `validate_complete` forgot could be left blank on both
    /// sides, and two blanks agree. Driven off the serialized shape for the same reason.
    #[test]
    fn every_serialized_identity_field_is_refused_when_left_unfilled() {
        for path in serialized_field_paths() {
            let (group, field) = path.split_once('.').expect("group.field");

            let mut document = serde_json::to_value(sample_identity()).unwrap();
            let slot = &mut document[group][field];
            *slot = match slot {
                serde_json::Value::String(_) => serde_json::Value::String(String::new()),
                serde_json::Value::Number(_) => serde_json::json!(0),
                other => panic!("unexpected identity value at {path}: {other}"),
            };
            let unfilled: IndexVersion = serde_json::from_value(document).unwrap();

            match unfilled.validate_complete() {
                Err(ArtifactError::IncompleteIdentity {
                    field: reported, ..
                }) => assert_eq!(
                    reported.path(),
                    path,
                    "an unfilled {path} must be reported as {path}"
                ),
                other => panic!("an unfilled {path} must be refused, got {other:?}"),
            }

            // And it must be refused before comparison, because it would otherwise agree
            // with another artifact that skipped the same field.
            assert!(
                unfilled.is_compatible(&unfilled),
                "two identical unfilled identities do compare equal — which is why \
                 completeness is checked first, not instead"
            );
        }
    }

    /// The canonical text behind the artifact digest is line-oriented, so a value
    /// carrying its own newline could let two identities digest identically.
    #[test]
    fn an_identity_value_may_not_carry_a_control_character() {
        let mut identity = sample_identity();
        identity.corpus.library_version = "otzaria-library-2026-08\ncorpus_id=x".to_string();

        match identity.validate_complete() {
            Err(ArtifactError::IncompleteIdentity { field, reason }) => {
                assert_eq!(field, IdentityField::LibraryVersion);
                assert!(reason.contains("control character"), "{reason}");
            }
            other => panic!("a value with a newline must be refused, got {other:?}"),
        }
    }

    #[test]
    fn a_checksum_that_is_not_a_lowercase_sha256_is_refused() {
        for bad in [
            String::new(),
            "deadbeef".to_string(),
            "A".repeat(64),
            "z".repeat(64),
            format!("{}g", "a".repeat(63)),
            format!(" {}", "a".repeat(64)),
        ] {
            let mut identity = sample_identity();
            identity.model.model_checksum = bad.clone();
            match identity.validate_complete() {
                Err(ArtifactError::IncompleteIdentity { field, .. }) => {
                    assert_eq!(field, IdentityField::ModelChecksum, "for {bad:?}")
                }
                other => panic!("{bad:?} must be refused as a checksum, got {other:?}"),
            }
        }
    }

    #[test]
    fn every_field_reports_its_group_and_its_metadata_path() {
        for field in IdentityField::ALL {
            let path = field.path();
            assert!(
                path.starts_with(&format!("{}.", field.group())),
                "{path} must sit under its group"
            );
            assert_eq!(field.to_string(), path);
        }

        let identity = sample_identity();
        let rendered = identity.to_string();
        for fragment in ["corpus", "model", "store", "1024", "last-token", "f32"] {
            assert!(
                rendered.contains(fragment),
                "{fragment} missing from Display"
            );
        }
        assert!(
            !rendered.contains(&identity.model.model_checksum),
            "Display abbreviates the checksum; the full value belongs in a mismatch"
        );
    }
}
