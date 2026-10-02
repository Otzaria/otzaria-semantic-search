//! Identity of an official vector set — what a package must declare, and what the
//! installation must agree with before a single vector is read.
//!
//! A stored vector is addressed by the text it was embedded from: its key is the SHA-256 of
//! that string ([`ChunkKey`](crate::semantic::chunk_key::ChunkKey)), and the application
//! ties it to a live line by recomputing the key from what its index stores. Nothing
//! positional is part of the bargain — not a line id, not a catalogue order, not a corpus
//! digest — so a library update leaves every unchanged line's vector valid.
//!
//! What still has to agree is everything that decides *which string* a line becomes and
//! *what vector* a string becomes. [`IndexVersion`] is that whole set: how the application
//! turned the library into lines and how a line's key is computed (the text group), which
//! model, backend and recipe produced the vectors (the model group), and how they are laid
//! out on disk (the store group). Every value is **data carried by the artifact**, not a
//! constant in this crate — the store reads what it was handed and compares.
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
//! The library edition the vectors were built from is **not** identity. It travels in the
//! package manifest, where it orders a chain of deltas, and a set built from v29 serves an
//! index at v30 — every line whose text did not change still resolves.
//!
//! See `docs/ARTIFACT_CONTRACT.md` for the field-by-field contract and for who fills
//! each value.

use crate::errors::ArtifactError;
use crate::semantic::chunk_key::KEY_VERSION;
use crate::semantic::recipe::{EmbeddingTextRecipe, TextNormalizationRecipe};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Full identity of a built artifact, in the three groups that must agree
/// independently.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IndexVersion {
    pub text: TextIdentity,
    pub model: ModelIdentity,
    pub store: StoreIdentity,
}

/// How a line of the library becomes the text a key is computed from, and how the key is
/// computed.
///
/// The group no vector can reveal: a set keyed under another line-text recipe resolves
/// nothing — every key is computed from a different string — and says so only by never
/// matching. Both fields are versions of code, so a change to that code is a change of
/// identity and a new base, never a silent loss of coverage.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TextIdentity {
    /// Version of the application's line recipe: how a book's text is split into lines,
    /// how a line is normalized before the index stores it, and which lines open a section.
    /// The application owns it and declares it; this crate cannot know it.
    pub line_text_version: u32,
    /// Version of the key function, [`KEY_VERSION`]. This crate's code, so an unimplemented
    /// one is refused outright rather than compared.
    pub key_version: u32,
}

impl TextIdentity {
    /// The text identity of an index whose lines follow `line_text_version`, keyed by the
    /// key function this build implements.
    pub fn with_line_text_version(line_text_version: u32) -> Self {
        Self {
            line_text_version,
            key_version: KEY_VERSION,
        }
    }
}

/// What turns text into vectors in this space: one model family, the recipe its texts
/// were built under, and the packages a query may be embedded with.
///
/// **A family is one model, exported more than once.** Meivin Round 2 ships an fp32 graph
/// and an int8 graph quantized from it; the library is embedded with one and a query with
/// either, and both land in the same space — what they must share exactly is everything
/// here but the packages: the weights' source and revision, the tokenizer, the width, the
/// pooling, the token cap and the text recipe.
///
/// **The packages are compared by membership.** [`Self::query_packages`] lists every package
/// of the family a query may come from; an installation names the one it has loaded, and
/// it must be in the list. Which package embedded the *passages*, and on what worker, is
/// provenance ([`VectorProvenance`]): recorded with the vectors, never compared — a worker's
/// parity is certified when the vectors are built, not checked when they are read.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelIdentity {
    /// The model family: the source of its weights and the revision every package of it
    /// was exported from, as `<repository>@<revision>`.
    pub family_id: String,
    /// SHA-256 of the `tokenizer.json` every package of the family carries, as 64
    /// lowercase hex digits. It decides the token ids a graph sees, so two packages over
    /// one family's weights but different tokenizers are two spaces.
    pub tokenizer_checksum: String,
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
    /// The family's packages a query may be embedded with. An installation's identity
    /// lists the one package it loaded, and that package must be among a vector set's.
    pub query_packages: Vec<ModelPackage>,
}

/// One export of a model family: what a checksum names.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelPackage {
    /// The package checksum, as 64 lowercase hex digits: the graph, its external data and
    /// its `tokenizer.json`, as [`model_package`](crate::semantic::model_package) defines
    /// it.
    pub checksum: String,
    /// The package's precision (`"int8"`, `"fp32"`). Redundant against the checksum by
    /// design: it is what makes a rejection readable.
    pub quantization: String,
}

/// What produced a set's vectors: recorded beside them, compared by nothing.
///
/// The space is the family's ([`ModelIdentity`]); which of its packages embedded the
/// passages, and which implementation ran it on which device, changes the arithmetic only
/// within the parity a build certifies when it publishes the vectors.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VectorProvenance {
    /// The package the passages were embedded with.
    pub passage_package: ModelPackage,
    /// What ran it.
    pub worker: EmbeddingWorker,
}

/// The implementation that embedded the passages: an embedding backend of this crate, or
/// an external worker certified against one.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EmbeddingWorker {
    /// The implementation and its version, e.g. `onnxruntime-sentence-v1`.
    pub backend: String,
    /// Where it ran: an execution provider or a device, e.g. `cpu`, `cuda:0`.
    pub device: String,
}

impl VectorProvenance {
    /// Refuse a record that says nothing, or that could not be written into a
    /// line-oriented digest.
    pub fn validate(&self) -> Result<(), ArtifactError> {
        let refuse = |what: &str, reason: &str| {
            Err(ArtifactError::ManifestDisagreesWithPayload {
                reason: format!("provenance: {what} {reason}"),
            })
        };
        if !is_sha256(&self.passage_package.checksum) {
            return refuse(
                "passage_package.checksum",
                "is not a SHA-256 of 64 lowercase hex digits",
            );
        }
        for (what, value) in [
            (
                "passage_package.quantization",
                &self.passage_package.quantization,
            ),
            ("worker.backend", &self.worker.backend),
            ("worker.device", &self.worker.device),
        ] {
            if value.trim().is_empty() {
                return refuse(what, "is blank");
            }
            if value.chars().any(char::is_control) {
                return refuse(what, "contains a control character");
            }
        }
        Ok(())
    }
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
    LineTextVersion,
    KeyVersion,
    FamilyId,
    TokenizerChecksum,
    EmbeddingDim,
    Pooling,
    MaxTokens,
    EmbeddingTextVersion,
    NormalizationVersion,
    ChunkingIdentity,
    /// Compared by membership: every package the installation names must be one the
    /// artifact accepts.
    QueryPackages,
    StoreBackendId,
    StoreFormatVersion,
    VectorPrecision,
}

/// The three things that must agree independently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IdentityGroup {
    Text,
    Model,
    Store,
}

impl IdentityField {
    /// Every comparable field, in the order rejections report them.
    pub const ALL: [IdentityField; 14] = [
        Self::LineTextVersion,
        Self::KeyVersion,
        Self::FamilyId,
        Self::TokenizerChecksum,
        Self::EmbeddingDim,
        Self::Pooling,
        Self::MaxTokens,
        Self::EmbeddingTextVersion,
        Self::NormalizationVersion,
        Self::ChunkingIdentity,
        Self::QueryPackages,
        Self::StoreBackendId,
        Self::StoreFormatVersion,
        Self::VectorPrecision,
    ];

    pub fn group(self) -> IdentityGroup {
        match self {
            Self::LineTextVersion | Self::KeyVersion => IdentityGroup::Text,
            Self::FamilyId
            | Self::TokenizerChecksum
            | Self::EmbeddingDim
            | Self::Pooling
            | Self::MaxTokens
            | Self::EmbeddingTextVersion
            | Self::NormalizationVersion
            | Self::ChunkingIdentity
            | Self::QueryPackages => IdentityGroup::Model,
            Self::StoreBackendId | Self::StoreFormatVersion | Self::VectorPrecision => {
                IdentityGroup::Store
            }
        }
    }

    /// Path of the field in the serialized metadata, so a rejection names something
    /// the reader can find in `manifest.json` or `set.json`.
    pub fn path(self) -> &'static str {
        match self {
            Self::LineTextVersion => "text.line_text_version",
            Self::KeyVersion => "text.key_version",
            Self::FamilyId => "model.family_id",
            Self::TokenizerChecksum => "model.tokenizer_checksum",
            Self::EmbeddingDim => "model.embedding_dim",
            Self::Pooling => "model.pooling",
            Self::MaxTokens => "model.max_tokens",
            Self::EmbeddingTextVersion => "model.embedding_text_version",
            Self::NormalizationVersion => "model.normalization_version",
            Self::ChunkingIdentity => "model.chunking_identity",
            Self::QueryPackages => "model.query_packages",
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
            Self::Text => "text",
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
            F::LineTextVersion => self.text.line_text_version.to_string(),
            F::KeyVersion => self.text.key_version.to_string(),
            F::FamilyId => self.model.family_id.clone(),
            F::TokenizerChecksum => self.model.tokenizer_checksum.clone(),
            F::EmbeddingDim => self.model.embedding_dim.to_string(),
            F::Pooling => self.model.pooling.clone(),
            F::MaxTokens => self.model.max_tokens.to_string(),
            F::EmbeddingTextVersion => self.model.embedding_text_version.to_string(),
            F::NormalizationVersion => self.model.normalization_version.to_string(),
            F::ChunkingIdentity => self.model.chunking_identity.to_string(),
            F::QueryPackages => {
                let mut packages: Vec<String> = self
                    .model
                    .query_packages
                    .iter()
                    .map(|package| format!("{} {}", package.checksum, package.quantization))
                    .collect();
                packages.sort();
                packages.join(",")
            }
            F::StoreBackendId => self.store.backend_id.clone(),
            F::StoreFormatVersion => self.store.store_format_version.to_string(),
            F::VectorPrecision => self.store.vector_precision.clone(),
        }
    }

    /// Every field on which this artifact disagrees with what the installation
    /// requires, in [`IdentityField::ALL`] order. All of them, not the first: a
    /// rejection the user has to fix one field per attempt is a rejection nobody
    /// finishes reading.
    ///
    /// Every field is compared for equality but one: [`IdentityField::QueryPackages`] is
    /// compared by membership — every package `expected` names must be one this artifact
    /// accepts — so an installation that loaded the family's int8 package and one that
    /// loaded its fp32 package both open a set that accepts the two.
    pub fn mismatches_against(&self, expected: &IndexVersion) -> Vec<IdentityMismatch> {
        IdentityField::ALL
            .iter()
            .filter_map(|&field| {
                let artifact = self.value(field);
                let required = expected.value(field);
                let agrees = match field {
                    IdentityField::QueryPackages => {
                        expected.model.query_packages.iter().all(|wanted| {
                            self.model
                                .query_packages
                                .iter()
                                .any(|accepted| accepted.checksum == wanted.checksum)
                        })
                    }
                    _ => artifact == required,
                };
                (!agrees).then_some(IdentityMismatch {
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

    /// The digest a vector set and every segment in it carry for this identity: SHA-256
    /// over `"otzaria-vector-identity-v1\n"` and one `path=value\n` line per field, in
    /// [`IdentityField::ALL`] order.
    ///
    /// What lets a segment header, a set and a published manifest say "the same identity"
    /// in 32 bytes. Equal digests mean equal identities as compared here — the same
    /// canonical values, so it agrees with [`Self::verify_matches`] by construction — and
    /// [`Self::validate_complete`] refusing newlines is what keeps the text unambiguous.
    pub fn identity_digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"otzaria-vector-identity-v1\n");
        for field in IdentityField::ALL {
            hasher.update(field.path().as_bytes());
            hasher.update(b"=");
            hasher.update(self.value(field).as_bytes());
            hasher.update(b"\n");
        }
        hasher.finalize().into()
    }

    /// [`Self::identity_digest`] as 64 lowercase hex digits.
    pub fn identity_digest_hex(&self) -> String {
        hex(&self.identity_digest())
    }

    /// Refuse an identity whose fields exist but say nothing: a blank string, a zero
    /// version, a checksum that is not one.
    ///
    /// This runs before [`Self::verify_matches`] because two unfilled identities
    /// compare equal. A builder that forgot to record `line_text_version` would otherwise
    /// produce an artifact that opens against an index of any line recipe at all.
    pub fn validate_complete(&self) -> Result<(), ArtifactError> {
        use IdentityField as F;

        require_positive(F::LineTextVersion, self.text.line_text_version.into())?;
        require_positive(F::KeyVersion, self.text.key_version.into())?;
        // The key function is this crate's code, like the recipe versions below: a version
        // nobody wrote is refused here rather than agreed with. The line-text version is
        // the application's, so it can only be compared.
        if self.text.key_version != KEY_VERSION {
            return Err(ArtifactError::UnsupportedRecipeVersion {
                field: "key_version",
                found: self.text.key_version,
                supported: KEY_VERSION.to_string(),
            });
        }

        require_text(F::FamilyId, &self.model.family_id)?;
        require_sha256(F::TokenizerChecksum, &self.model.tokenizer_checksum)?;
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
        if self.model.query_packages.is_empty() {
            return Err(ArtifactError::IncompleteIdentity {
                field: F::QueryPackages,
                reason: "is empty, so no query could be embedded into this space".to_string(),
            });
        }
        for package in &self.model.query_packages {
            require_sha256(F::QueryPackages, &package.checksum)?;
            require_text(F::QueryPackages, &package.quantization)?;
        }
        let mut checksums: Vec<&str> = self
            .model
            .query_packages
            .iter()
            .map(|package| package.checksum.as_str())
            .collect();
        checksums.sort_unstable();
        if checksums.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(ArtifactError::IncompleteIdentity {
                field: F::QueryPackages,
                reason: "names one package twice".to_string(),
            });
        }

        // Filled in is not the same as implemented. These two are versions of *this
        // crate's code*, so unlike every other field here they can be settled outright
        // rather than compared against another copy of themselves — and a number nobody
        // wrote code for has to be refused at both ends: a build must not produce it, and
        // an installation must not open it just because its own configuration repeats it.
        // `chunking_version` is absent because an artifact carries the hash of the chunker
        // configuration and not the configuration; see
        // [`recipe`](crate::semantic::recipe).
        EmbeddingTextRecipe::from_version(self.model.embedding_text_version)?;
        TextNormalizationRecipe::from_version(self.model.normalization_version)?;

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
    if !is_sha256(value) {
        return Err(ArtifactError::IncompleteIdentity {
            field,
            reason: "is not a SHA-256 of 64 lowercase hex digits".to_string(),
        });
    }
    Ok(())
}

/// 64 lowercase hex digits.
pub(crate) fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

impl std::fmt::Display for IndexVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let tokenizer: String = self.model.tokenizer_checksum.chars().take(12).collect();
        let packages: Vec<String> = self
            .model
            .query_packages
            .iter()
            .map(|package| {
                let checksum: String = package.checksum.chars().take(12).collect();
                format!("{} [{checksum}…]", package.quantization)
            })
            .collect();
        write!(
            f,
            "text line v{} key v{}; \
             model {} tokenizer [{}…] (dim={}, {}, {} tok, text v{}, norm v{}, chunk {}), \
             queries {}; store {} v{} {}",
            self.text.line_text_version,
            self.text.key_version,
            self.model.family_id,
            tokenizer,
            self.model.embedding_dim,
            self.model.pooling,
            self.model.max_tokens,
            self.model.embedding_text_version,
            self.model.normalization_version,
            self.model.chunking_identity,
            packages.join(", "),
            self.store.backend_id,
            self.store.store_format_version,
            self.store.vector_precision,
        )
    }
}

/// Lowercase hex, for the digests this module and the store write into JSON.
pub(crate) fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut text = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(text, "{byte:02x}");
    }
    text
}

/// A complete identity for tests across this crate: every field filled with something
/// that passes [`IndexVersion::validate_complete`], so a test that wants to exercise a
/// *specific* rejection changes one field and nothing else.
#[cfg(test)]
pub(crate) fn test_identity() -> IndexVersion {
    IndexVersion {
        text: TextIdentity {
            line_text_version: 1,
            key_version: KEY_VERSION,
        },
        model: test_model_identity(),
        store: StoreIdentity {
            backend_id: "otzaria-oxv".to_string(),
            store_format_version: 2,
            vector_precision: "i8-sym-vec".to_string(),
        },
    }
}

/// The model half of [`test_identity`]: a family of two packages, int8 and fp32, as the
/// production family is.
#[cfg(test)]
pub(crate) fn test_model_identity() -> ModelIdentity {
    ModelIdentity {
        family_id: "ArieLLL123/judaic-semantic-round2-onnx-zayit@1ec8dc6".to_string(),
        tokenizer_checksum: "7".repeat(64),
        embedding_dim: 1024,
        pooling: "in-graph".to_string(),
        max_tokens: 512,
        embedding_text_version: 1,
        normalization_version: 1,
        chunking_identity: 0x51A1_1E55,
        query_packages: vec![
            ModelPackage {
                checksum: "a".repeat(64),
                quantization: "int8".to_string(),
            },
            ModelPackage {
                checksum: "f".repeat(64),
                quantization: "fp32".to_string(),
            },
        ],
    }
}

/// Provenance for tests: the fp32 package of [`test_model_identity`], run on a CPU.
#[cfg(test)]
pub(crate) fn test_provenance() -> VectorProvenance {
    VectorProvenance {
        passage_package: ModelPackage {
            checksum: "f".repeat(64),
            quantization: "fp32".to_string(),
        },
        worker: EmbeddingWorker {
            backend: "onnxruntime-sentence-v1".to_string(),
            device: "cpu".to_string(),
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
            (F::LineTextVersion, |v| v.text.line_text_version = 2),
            (F::KeyVersion, |v| v.text.key_version = 2),
            (F::FamilyId, |v| {
                v.model.family_id = "ArieLLL123/judaic-semantic-round3@0000000".to_string()
            }),
            (F::TokenizerChecksum, |v| {
                v.model.tokenizer_checksum = "8".repeat(64)
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
            (F::QueryPackages, |v| {
                v.model.query_packages = vec![ModelPackage {
                    checksum: "b".repeat(64),
                    quantization: "int4".to_string(),
                }]
            }),
            (F::StoreBackendId, |v| {
                v.store.backend_id = "mmap-flat-v1".to_string()
            }),
            (F::StoreFormatVersion, |v| v.store.store_format_version = 3),
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

    /// The scenario the text group exists for: an index whose lines are produced by another
    /// recipe computes every key from another string, so nothing would resolve — and
    /// nothing about the vectors says so.
    #[test]
    fn a_set_keyed_under_another_line_recipe_is_refused() {
        let expected = sample_identity();
        let mut artifact = sample_identity();
        artifact.text.line_text_version = 2;

        let mismatches = artifact.mismatches_against(&expected);
        assert_eq!(mismatches[0].field, IdentityField::LineTextVersion);
        assert_eq!(mismatches[0].field.group(), IdentityGroup::Text);
    }

    /// The key function is this crate's own code, so a version it does not implement is
    /// refused even when both sides declare it.
    #[test]
    fn a_key_version_this_build_does_not_implement_is_refused() {
        let mut identity = sample_identity();
        identity.text.key_version = KEY_VERSION + 1;
        match identity.validate_complete() {
            Err(ArtifactError::UnsupportedRecipeVersion { field, found, .. }) => {
                assert_eq!(field, "key_version");
                assert_eq!(found, KEY_VERSION + 1);
            }
            other => panic!("an unimplemented key version must be refused, got {other:?}"),
        }
    }

    /// One digest per identity, moving with every field and with nothing else.
    #[test]
    fn the_identity_digest_moves_with_every_field() {
        let base = sample_identity();
        assert_eq!(base.identity_digest(), sample_identity().identity_digest());
        assert_eq!(base.identity_digest_hex().len(), 64);
        let mut seen = HashSet::from([base.identity_digest()]);
        for (field, mutate) in field_mutations() {
            let mut changed = sample_identity();
            mutate(&mut changed);
            assert!(
                seen.insert(changed.identity_digest()),
                "{field} must move the digest"
            );
        }
    }

    /// The identity an installation presents: the family it declares, and the one package
    /// it loaded.
    fn installation(package: &str) -> IndexVersion {
        let mut identity = sample_identity();
        identity
            .model
            .query_packages
            .retain(|p| p.quantization == package);
        assert_eq!(identity.model.query_packages.len(), 1);
        identity
    }

    /// The vectors of a set were embedded by one package of a family; a query may come from
    /// any package the set accepts. An installation running the int8 package and one
    /// running the fp32 package both open a set built with fp32 that accepts both.
    #[test]
    fn any_accepted_package_of_the_family_may_embed_the_query() {
        let set = sample_identity();
        for package in ["int8", "fp32"] {
            assert!(
                set.verify_matches(&installation(package)).is_ok(),
                "{package} must open the set"
            );
        }
        // And a set that accepts only fp32 refuses an int8 installation, naming the field.
        let mut fp32_only = sample_identity();
        fp32_only
            .model
            .query_packages
            .retain(|p| p.quantization == "fp32");
        let mismatches = fp32_only.mismatches_against(&installation("int8"));
        assert_eq!(mismatches.len(), 1);
        assert_eq!(mismatches[0].field, IdentityField::QueryPackages);
        assert_eq!(mismatches[0].field.path(), "model.query_packages");
        assert!(mismatches[0].expected.contains(&"a".repeat(64)));
    }

    /// Everything but the package must match exactly: another family, another tokenizer,
    /// another recipe — or a package of the right quantization that is not one the set
    /// accepts.
    #[test]
    fn another_family_tokenizer_recipe_or_package_is_refused() {
        let set = sample_identity();
        let changes: [Mutation; 6] = [
            (IdentityField::FamilyId, |v| {
                v.model.family_id.push_str("-other")
            }),
            (IdentityField::TokenizerChecksum, |v| {
                v.model.tokenizer_checksum = "9".repeat(64)
            }),
            (IdentityField::EmbeddingTextVersion, |v| {
                v.model.embedding_text_version = 2
            }),
            (IdentityField::NormalizationVersion, |v| {
                v.model.normalization_version = 2
            }),
            (IdentityField::ChunkingIdentity, |v| {
                v.model.chunking_identity += 1
            }),
            (IdentityField::QueryPackages, |v| {
                v.model.query_packages[0].checksum = "c".repeat(64)
            }),
        ];
        for (field, change) in changes {
            let mut runtime = installation("int8");
            change(&mut runtime);
            match set.verify_matches(&runtime) {
                Err(ArtifactError::IdentityMismatch { mismatches }) => {
                    assert_eq!(mismatches.len(), 1, "{field}");
                    assert_eq!(mismatches[0].field, field);
                }
                other => panic!("{field} must be refused, got {other:?}"),
            }
        }
    }

    /// Provenance is recorded, and compared by nothing: the identity has no field for the
    /// package that embedded the passages or the worker that ran it.
    #[test]
    fn provenance_is_validated_and_never_compared() {
        let provenance = test_provenance();
        provenance.validate().unwrap();
        let serialized = serde_json::to_string(&sample_identity()).unwrap();
        assert!(!serialized.contains("worker") && !serialized.contains("passage"));

        let mut blank = test_provenance();
        blank.worker.device = " ".to_string();
        assert!(blank.validate().is_err());
        let mut bad = test_provenance();
        bad.passage_package.checksum = "F".repeat(64);
        assert!(bad.validate().is_err());
    }

    #[test]
    fn every_disagreement_is_reported_at_once_in_field_order() {
        let expected = sample_identity();
        let mut artifact = sample_identity();
        artifact.store.vector_precision = "f16".to_string();
        artifact.text.line_text_version = 9;
        artifact.model.embedding_dim = 256;

        let mismatches = artifact.mismatches_against(&expected);
        let fields: Vec<IdentityField> = mismatches.iter().map(|m| m.field).collect();
        assert_eq!(
            fields,
            vec![
                IdentityField::LineTextVersion,
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
                serde_json::Value::Array(_) => serde_json::json!([]),
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
        identity.model.family_id = "EMD123/model\nmodel.pooling=x".to_string();

        match identity.validate_complete() {
            Err(ArtifactError::IncompleteIdentity { field, reason }) => {
                assert_eq!(field, IdentityField::FamilyId);
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
            for (field, set) in [
                (
                    IdentityField::TokenizerChecksum,
                    (|v, bad| v.model.tokenizer_checksum = bad) as fn(&mut IndexVersion, String),
                ),
                (IdentityField::QueryPackages, |v, bad| {
                    v.model.query_packages[1].checksum = bad
                }),
            ] {
                let mut identity = sample_identity();
                set(&mut identity, bad.clone());
                match identity.validate_complete() {
                    Err(ArtifactError::IncompleteIdentity {
                        field: reported, ..
                    }) => {
                        assert_eq!(reported, field, "for {bad:?}")
                    }
                    other => panic!("{bad:?} must be refused as a checksum, got {other:?}"),
                }
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
        for fragment in ["text", "model", "store", "1024", "in-graph", "i8-sym-vec"] {
            assert!(
                rendered.contains(fragment),
                "{fragment} missing from Display"
            );
        }
        assert!(rendered.contains("int8") && rendered.contains("fp32"));
        assert!(
            !rendered.contains(&identity.model.query_packages[0].checksum),
            "Display abbreviates the checksums; the full value belongs in a mismatch"
        );

        // A package named twice is a list that says less than it seems to.
        let mut twice = sample_identity();
        twice.model.query_packages[1].checksum = twice.model.query_packages[0].checksum.clone();
        assert!(matches!(
            twice.validate_complete(),
            Err(ArtifactError::IncompleteIdentity {
                field: IdentityField::QueryPackages,
                ..
            })
        ));
    }
}
