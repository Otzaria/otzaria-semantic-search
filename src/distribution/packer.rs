//! The build-side tool: ready-made vectors in, an official artifact out.
//!
//! This is S4a. It assumes the vectors already exist — producing them from text is a
//! separate stage — and its job is everything between "a file of floats" and "a directory
//! the runtime will open": pair each vector with the corpus line it claims, compose the
//! identity, write the payload, describe it, and then prove the result verifies.
//!
//! # What the input is, and what it deliberately is not
//!
//! An input record is a `line_id`, a vector, and **two** digests: of the corpus line, and
//! of the text that was actually embedded. That is all. Every other field a stored record
//! carries — book key, title, reference, section, segment, facets — is read out of the
//! [`CorpusIndex`] at pack time, so there is never a second description of a book that can
//! drift from the lexical index the application hydrates from.
//!
//! **`source_line_sha256`** is checked against the corpus. It catches one specific and
//! very likely failure: the vector file and the id list drifted apart — an off-by-one, a
//! different sort order, a subset. That failure is otherwise invisible, because the counts
//! still add up, the checksums still pass and the identity still matches, so every result
//! is a neighbouring line returned with full confidence.
//!
//! **`embedding_text_sha256`** is not checked against anything; it is *recorded*, as the
//! record's `chunk_hash`. The embedded text is not the corpus line whenever the recipe
//! prefixes a title, borrows context from neighbours or truncates — so a `chunk_hash`
//! computed from the corpus line would describe a text nothing was ever built from, and
//! [`Chunker`](crate::semantic::chunker::Chunker) defines that field as a digest of the
//! embedded text. The producer is the only party that has it.
//!
//! ## What the digests do not prove
//!
//! They are an alignment check, not provenance. Neither one establishes that the vector
//! was produced from that text, by the declared model, under the declared normalization —
//! a producer that hashed the corpus at pack time rather than at embedding time satisfies
//! both. Nothing available to a tool that receives finished floats can establish more than
//! that, and no wording here should suggest otherwise. Closing it means producing the
//! vector, its digest and the model identity in one pipeline, from the model and the index
//! that are actually open; that is S4b's, and it reaches this code through the same input.
//!
//! # Two entry points over one set of checks
//!
//! [`pack`] writes an artifact and then calls [`validate_artifact`] on it;
//! [`validate_artifact`] alone answers "does this directory hold an artifact that belongs
//! to this corpus and this model" for one nobody here built. So a freshly packed artifact
//! is held to exactly the checks an arbitrary one is, and the acceptance gate is a
//! property of the code rather than of the order a build script calls things in.
//!
//! The post-write checks are the runtime's own — [`IndexPackage::verify_for_install`],
//! the payload layout, the reader, and the counts
//! [`OfficialSemanticIndex`](crate::semantic::official_index::OfficialSemanticIndex)
//! checks — plus two only a build machine can make. Every record read back out of the
//! payload is compared, field by field, against what the corpus says about its line; and
//! the set of ids is compared against [`CorpusIndex::expected_line_ids`] in both
//! directions, so an artifact that is internally perfect and covers a tenth of the library
//! is a rejection — and so is one carrying a vector for a line the recipe does not embed.
//! Over every record, not a sample: the corpus is already in memory, and a
//! sampled gate makes a failing build depend on which records the sample happened to
//! include.
//!
//! # What it costs, stated rather than hidden
//!
//! A pack reads the committed payload four times: once to compute each declared hash,
//! once inside [`IndexPackage::write`], which refuses to publish metadata the runtime
//! would reject, once for the full verification afterwards, and once more when the reader
//! opens it. Three of those are hashes of gigabytes at library scale. They are kept
//! because each answers a different question and this runs on a build machine, not on a
//! device — but the arithmetic is here so nobody has to rediscover it from a slow build.
//! Everything before the commit is in memory, so the same payload is also held whole in
//! RAM: that is the payload writer's cost, and the S2b measurement's problem.
//!
//! # What this does not decide
//!
//! Nothing about the payload's shape. The store identity is
//! [`readable_store_identity`], i.e. what this build can *read* — a packer that wrote a
//! format its own runtime cannot open would be producing artifacts for nobody. The
//! dimension, the precision and the text recipe are all data in the model identity the
//! caller supplies, exactly as `docs/ARTIFACT_CONTRACT.md` requires; S1 will fill them in
//! without changing anything here.

use crate::distribution::corpus::{CorpusIndex, CorpusLine};
use crate::distribution::package::{
    ArtifactExpectation, IndexPackage, PackageManifest, PayloadDescriptor, VerifiedPackage,
};
use crate::errors::{EmbeddingError, PackError};
use crate::semantic::chunker::compute_semantic_id;
use crate::semantic::embedding::normalize_validated;
use crate::semantic::official_index::{
    ensure_snapshot_layout, readable_store_identity, verify_counts_against_payload,
};
use crate::semantic::store_backend::{VectorSearchBackend, VectorStoreBackend};
use crate::semantic::types::VectorMetadata;
use crate::semantic::versioning::{IndexVersion, ModelIdentity};
use crate::semantic::zevc_store::{
    ReadOnlyZevcStore, ZevcStore, ZevcStoreConfig, SNAPSHOT_FILENAMES,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

/// Vectors handed to the payload writer per call. Large enough that the per-batch
/// overhead disappears, small enough that a rejected vector is reported before the whole
/// input has been read.
const INSERT_BATCH: usize = 1024;

/// One ready-made vector and the line it belongs to.
#[derive(Debug, Clone)]
pub struct VectorInput {
    /// The global document id the vector describes, in the corpus's own id scheme.
    pub line_id: u64,
    /// SHA-256 of the **corpus line's** text as the producer read it, in 64 lowercase hex
    /// digits. Checked against the corpus; see the module documentation.
    pub source_line_sha256: String,
    /// SHA-256 of the text that was actually **embedded** — after whatever prefixing,
    /// neighbour context and truncation the recipe applies. Recorded as the record's
    /// `chunk_hash`, and equal to the corpus line's digest only when the recipe embedded
    /// the line unchanged.
    pub embedding_text_sha256: String,
    pub vector: Vec<f32>,
}

/// One line of the records file that accompanies a raw vector file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorInputRecord {
    pub line_id: u64,
    pub source_line_sha256: String,
    pub embedding_text_sha256: String,
}

/// What to build, beyond the vectors and the corpus.
#[derive(Debug, Clone)]
pub struct PackRequest {
    /// Directory the artifact is written into. Must not exist, or be an empty directory.
    pub output_path: PathBuf,
    /// Everything about how the vectors were produced. The one half of the identity
    /// neither the corpus nor this crate can know — see [`ModelIdentity`].
    pub model: ModelIdentity,
    /// Timestamp recorded in the manifest. Excluded from the artifact digest, so it does
    /// not have to be reproducible for the digest to be.
    pub created_at: String,
    /// Collection name written into the payload header. Not part of the identity, and
    /// therefore not something the runtime compares — see
    /// [`ReadOnlyZevcStore::open`](crate::semantic::zevc_store::ReadOnlyZevcStore).
    pub collection_name: String,
}

/// What a pack or a validation established about an artifact.
#[derive(Debug, Clone)]
pub struct PackReport {
    pub artifact_path: PathBuf,
    pub identity: IndexVersion,
    /// The value a publisher announces outside the package — see
    /// [`IndexPackage::digest`]. Without it, a later verification detects damage and the
    /// wrong artifact but not a deliberately rebuilt one.
    pub digest: String,
    /// Records in the payload, as counted from the payload.
    pub vector_count: u32,
    /// Distinct `source_book_key`s among them.
    pub book_count: u32,
    pub total_size_bytes: u64,
}

/// Build an artifact from ready-made vectors, and prove the result opens.
///
/// The order is forced rather than chosen:
///
/// 1. Compose the identity from the corpus, the caller's model and what this build can
///    read, and refuse it if any field is unfilled — before a single vector is read,
///    because an artifact with a blank identity opens against anything.
/// 2. Refuse an output path that already holds something.
/// 3. Per input: the declared width, a vector a search could return, an id seen once, a
///    line the corpus holds, and the text that line actually carries.
/// 4. Coverage: the ids are exactly the ones the corpus expects for this recipe.
/// 5. Commit the payload, describe it, and write the metadata.
/// 6. [`validate_artifact`], which re-reads everything just written.
///
/// Nothing reaches the disk before step 5: the payload writer buffers until it commits,
/// so a rejection in steps 3–4 leaves the output directory empty and the run can simply be
/// repeated. A failure *after* the commit leaves a partial artifact in place, and the
/// next attempt refuses it rather than writing over it — that directory is evidence.
pub fn pack(
    request: PackRequest,
    inputs: impl IntoIterator<Item = Result<VectorInput, PackError>>,
    corpus: &dyn CorpusIndex,
) -> Result<PackReport, PackError> {
    let identity = compose_identity(corpus, &request.model)?;
    ensure_output_is_free(&request.output_path)?;

    let embedding_dim = identity.model.embedding_dim;
    let chunking_identity = identity.model.chunking_identity;
    let store = ZevcStore::open_or_create(ZevcStoreConfig {
        db_path: request.output_path.clone(),
        embedding_dim,
        collection_name: request.collection_name.clone(),
        auto_persist: false,
    })?;

    let mut seen: BTreeSet<u64> = BTreeSet::new();
    let mut batch: Vec<(VectorMetadata, Vec<f32>)> = Vec::with_capacity(INSERT_BATCH);
    let mut accepted: u32 = 0;

    for input in inputs {
        let mut input = input?;
        // Cheapest first, and it also spares the corpus a lookup for an id already seen.
        if !seen.insert(input.line_id) {
            return Err(PackError::DuplicateLineId {
                line_id: input.line_id,
            });
        }
        let metadata = join_to_corpus(&mut input, corpus, embedding_dim, chunking_identity)?;

        batch.push((metadata, input.vector));
        accepted = accepted.saturating_add(1);
        if batch.len() == INSERT_BATCH {
            store.insert_batch(std::mem::take(&mut batch))?;
            batch.reserve(INSERT_BATCH);
        }
    }
    if !batch.is_empty() {
        store.insert_batch(batch)?;
    }
    if accepted == 0 {
        return Err(PackError::NoVectors);
    }
    verify_coverage(&seen, corpus, &identity.model)?;

    // The payload writer replaces on a duplicate `semantic_id` rather than failing, so
    // "everything accepted is in the payload" is counted and not assumed.
    let stored = store.count();
    if stored != accepted {
        return Err(PackError::VectorCountChanged { accepted, stored });
    }
    let book_count = store.book_keys().len().min(u32::MAX as usize) as u32;
    store.commit()?;
    drop(store);

    write_metadata(
        &request.output_path,
        &identity,
        &request.created_at,
        book_count,
        stored,
    )?;

    log::info!(
        "Packed {stored} vector(s) across {book_count} book(s) into {}",
        request.output_path.display()
    );
    validate_artifact(&request.output_path, &request.model, corpus)
}

/// Verify an artifact against the corpus and model it claims to belong to.
///
/// Everything the runtime checks at open, minus the model file it cannot load here, plus
/// the join only a build machine can make. In order:
///
/// 1. the identity is complete and matches this corpus and model;
/// 2. every payload byte hashes to what `payloads.json` declares
///    ([`VerificationDepth::FullPayload`](crate::distribution::package::VerificationDepth));
/// 3. the payload set is exactly this backend's;
/// 4. the payload opens through the verified token, which checks a SHA-256 per record;
/// 5. the manifest's counts are what the payload holds;
/// 6. every record's metadata is what the corpus says about its line;
/// 7. the ids are exactly the lines the corpus expects for this recipe.
///
/// Steps 6 and 7 are what make this more than a re-run of the runtime's checks. An
/// artifact can be internally perfect and still describe books by a title the catalogue no
/// longer uses — and it can be internally perfect while holding a hundredth of the
/// library, which every count, checksum and identity field would agree with.
pub fn validate_artifact(
    artifact_path: &Path,
    model: &ModelIdentity,
    corpus: &dyn CorpusIndex,
) -> Result<PackReport, PackError> {
    let identity = compose_identity(corpus, model)?;
    let verified = IndexPackage::verify_for_install(
        artifact_path,
        &ArtifactExpectation::without_published_digest(identity.clone()),
    )?;
    ensure_snapshot_layout(&verified)?;

    let store = ReadOnlyZevcStore::open(&verified)?;
    let book_count = verify_counts_against_payload(&verified, &store)?;
    verify_records_against_corpus(&store, corpus, identity.model.chunking_identity)?;
    verify_coverage(
        &store
            .stored_metadata()
            .map(|record| record.line_id)
            .collect(),
        corpus,
        &identity.model,
    )?;

    Ok(report(&verified, identity, book_count))
}

/// Refuse an artifact whose vectors are not exactly the lines the recipe embeds.
///
/// **Both directions, and neither is the other's mirror.**
///
/// A line with no vector is invisible: every count in the manifest, every payload checksum
/// and every identity field agrees with an artifact holding one line out of six million.
/// That is the half [`PackError::LineNotInCorpus`] cannot reach, because nothing is there
/// to be rejected.
///
/// A vector for a line the recipe does **not** embed is a different fault, and it is not
/// caught by the per-input join either: that check asks the corpus whether the line
/// exists, and a line skipped for being too short to carry meaning exists perfectly well.
/// It should simply never have been embedded — so its presence says the vectors came from
/// a recipe other than the one the artifact declares.
///
/// Checked as a whole set rather than per input, so a build log gets both totals at once
/// instead of stopping on whichever id came first. The example ids are the smallest of
/// each kind, so two runs over the same fault name the same lines.
///
/// The agreeing case is one comparison and no allocation, and the disagreeing case counts
/// rather than collects. At library scale either side is millions of ids, so materializing
/// the differences would allocate tens of MiB on the path where something has already gone
/// wrong — which is the worst moment to ask a build machine for memory.
fn verify_coverage(
    covered: &BTreeSet<u64>,
    corpus: &dyn CorpusIndex,
    model: &ModelIdentity,
) -> Result<(), PackError> {
    let expected = corpus.expected_line_ids(model)?;
    if &expected == covered {
        return Ok(());
    }

    // Two sets are equal exactly when both differences are empty, so past that check at
    // least one of these is non-zero and there is a real disagreement to report.
    let (missing, first_missing) = difference_summary(&expected, covered);
    let (unexpected, first_unexpected) = difference_summary(covered, &expected);

    Err(PackError::CoverageMismatch {
        expected: expected.len(),
        covered: covered.len(),
        missing,
        unexpected,
        first_missing,
        first_unexpected,
    })
}

/// How many ids are in `from` and not in `other`, and the smallest of them.
///
/// One pass and constant extra memory. `BTreeSet::difference` yields in ascending order,
/// so the first id it produces is the smallest.
fn difference_summary(from: &BTreeSet<u64>, other: &BTreeSet<u64>) -> (usize, Option<u64>) {
    from.difference(other)
        .fold((0, None), |(count, first), line_id| {
            (count + 1, first.or(Some(*line_id)))
        })
}

/// The three sources that each know a part of the identity, and the completeness check
/// that has to come before any of them is compared to anything.
fn compose_identity(
    corpus: &dyn CorpusIndex,
    model: &ModelIdentity,
) -> Result<IndexVersion, PackError> {
    let identity = IndexVersion {
        corpus: corpus.identity()?,
        model: model.clone(),
        // Not the caller's: an artifact written in a layout this build cannot read would
        // be an artifact for nobody.
        store: readable_store_identity(),
    };
    identity.validate_complete()?;
    Ok(identity)
}

/// Refuse an output path that is not an empty place to write a whole artifact.
fn ensure_output_is_free(path: &Path) -> Result<(), PackError> {
    let unusable = |reason: String| PackError::UnusableOutput {
        path: path.display().to_string(),
        reason,
    };

    let metadata = match std::fs::metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(source) => {
            return Err(PackError::Io {
                context: format!("inspecting {}", path.display()),
                source,
            })
        }
    };
    if !metadata.is_dir() {
        return Err(unusable("it exists and is not a directory".to_string()));
    }

    let entries = std::fs::read_dir(path)
        .map_err(|source| PackError::Io {
            context: format!("listing {}", path.display()),
            source,
        })?
        .count();
    if entries > 0 {
        return Err(unusable(format!(
            "it already holds {entries} entr{}, and a pack writes a whole artifact",
            if entries == 1 { "y" } else { "ies" }
        )));
    }
    Ok(())
}

/// Check one input against the corpus and turn it into the record that will be stored.
///
/// The vector is normalized here rather than left to the payload writer, because
/// [`normalize_validated`] is the crate's one choke point for "can this vector be
/// compared at all" and its rejection is reported against the `line_id` a build log can
/// act on. The writer normalizes again; doing it twice is idempotent and costs one pass.
fn join_to_corpus(
    input: &mut VectorInput,
    corpus: &dyn CorpusIndex,
    embedding_dim: u32,
    chunking_identity: u64,
) -> Result<VectorMetadata, PackError> {
    if input.vector.len() as u32 != embedding_dim {
        return Err(PackError::VectorDimensionMismatch {
            line_id: input.line_id,
            expected: embedding_dim,
            found: input.vector.len(),
        });
    }
    normalize_validated(&mut input.vector, embedding_dim).map_err(|error| {
        PackError::UnusableVector {
            line_id: input.line_id,
            // Unwrapped from the label the runtime puts on these, which does not apply
            // here: nothing in a packer inferred anything. The reason itself does.
            reason: match error {
                EmbeddingError::InferenceFailed { reason } => reason,
                other => other.to_string(),
            },
        }
    })?;

    for (field, declared) in [
        ("source_line_sha256", &input.source_line_sha256),
        ("embedding_text_sha256", &input.embedding_text_sha256),
    ] {
        if !is_lowercase_sha256(declared) {
            return Err(PackError::MalformedInput {
                reason: format!(
                    "line {} declares {declared:?} as its {field}, which is not 64 \
                     lowercase hex digits",
                    input.line_id
                ),
            });
        }
    }

    let line = corpus
        .line(input.line_id)?
        .ok_or(PackError::LineNotInCorpus {
            line_id: input.line_id,
        })?;
    let actual = sha256_hex(line.text.as_bytes());
    if actual != input.source_line_sha256 {
        return Err(PackError::LineTextMismatch {
            line_id: input.line_id,
            declared: input.source_line_sha256.clone(),
            actual,
        });
    }

    Ok(record_for(
        input.line_id,
        &line,
        chunking_identity,
        // Safe to slice: the digest was checked to be 64 ASCII hex digits above.
        &input.embedding_text_sha256[..CHUNK_HASH_HEX_LEN],
    ))
}

/// The record a line gets: the corpus for everything it knows, and the producer for the
/// one thing it cannot.
///
/// `semantic_id` and `source_doc_key` are composed the way the chunker composes them, so
/// an artifact built here and one built by the prototype indexing path key their vectors
/// identically.
///
/// `chunk_hash` is [`compute_chunk_hash`](crate::semantic::chunker::compute_chunk_hash) of
/// the **embedded** text, which is the
/// definition the chunker set and the reason the producer has to declare a second digest.
/// Whenever the recipe prefixes a title, borrows context from a neighbour or truncates, the
/// embedded text is not the corpus line — so deriving this field from the corpus would
/// have made every record describe a text nothing was ever built from. The caller passes
/// the first 128 bits of the declared SHA-256, which is exactly what that function
/// produces; `a_chunk_hash_is_the_documented_prefix_of_a_sha256` pins that so the two
/// cannot drift apart.
fn record_for(
    line_id: u64,
    line: &CorpusLine,
    chunking_identity: u64,
    chunk_hash: &str,
) -> VectorMetadata {
    VectorMetadata {
        semantic_id: compute_semantic_id(&line.source_book_key, line_id, chunking_identity),
        source_book_key: line.source_book_key.clone(),
        source_doc_key: format!("{}:{}", line.source_book_key, line_id),
        line_id,
        section_id: line.section_id,
        line_hash: line.line_hash,
        chunk_hash: chunk_hash.to_string(),
        content_hash: line.content_hash,
        reference: line.reference.clone(),
        segment: line.segment,
        is_pdf: line.is_pdf,
        title: line.title.clone(),
        facets: line.facets.clone(),
    }
}

/// Length of a `chunk_hash` — the chunker emits the first 16 bytes of a SHA-256 as hex.
const CHUNK_HASH_HEX_LEN: usize = 32;

/// Every stored field the corpus decides, paired with the name a rejection reports it
/// under.
///
/// A table rather than a hand-written comparison, so the two sides cannot be walked
/// differently. Values are rendered for the message *and* compared, so anything with
/// internal structure is rendered unambiguously: `facets` goes through JSON, because
/// joining it on `", "` made `["/a, /b"]` and `["/a", "/b"]` compare equal.
fn corpus_fields(record: &VectorMetadata) -> [(&'static str, String); 12] {
    [
        ("semantic_id", record.semantic_id.clone()),
        ("source_book_key", record.source_book_key.clone()),
        ("source_doc_key", record.source_doc_key.clone()),
        ("line_id", record.line_id.to_string()),
        ("section_id", record.section_id.to_string()),
        ("line_hash", record.line_hash.to_string()),
        ("content_hash", record.content_hash.to_string()),
        ("reference", record.reference.clone()),
        ("segment", record.segment.to_string()),
        ("is_pdf", record.is_pdf.to_string()),
        ("title", record.title.clone()),
        (
            "facets",
            serde_json::to_string(&record.facets)
                .unwrap_or_else(|_| format!("{:?}", record.facets)),
        ),
    ]
}

/// The one stored field the corpus cannot answer.
///
/// `chunk_hash` describes the text the *producer* embedded, and the corpus holds the line
/// rather than the recipe's output. Named as a constant so
/// `every_stored_field_is_compared_or_deliberately_not` can require every serialized field
/// to be either in [`corpus_fields`] or here — a field in neither would be written into
/// every artifact and checked by nothing.
#[cfg(test)]
const UNCOMPARABLE_FIELD: &str = "chunk_hash";

/// Compare every record in an opened payload against the corpus it names.
///
/// Sorted by `line_id` first, because the payload is a `HashMap` and an unsorted walk
/// would report a different one of several disagreements on every run — which is the
/// difference between a build failure someone can fix and one they re-run until it names
/// something else.
///
/// The producer's own field is not compared, because there is nothing to compare it to:
/// what is checked instead is that it has the shape the chunker produces, so a record
/// carrying a full SHA-256, an empty string or a sentence is still a rejection.
fn verify_records_against_corpus(
    store: &ReadOnlyZevcStore,
    corpus: &dyn CorpusIndex,
    chunking_identity: u64,
) -> Result<(), PackError> {
    let mut records: Vec<&VectorMetadata> = store.stored_metadata().collect();
    records.sort_by_key(|record| record.line_id);

    for stored in records {
        let line = corpus
            .line(stored.line_id)?
            .ok_or(PackError::LineNotInCorpus {
                line_id: stored.line_id,
            })?;
        // Before the record is rebuilt around it: this reads an artifact nobody here
        // necessarily wrote, and a `chunk_hash` that is not one has to be a rejection
        // rather than something every later step has to be careful with.
        if !is_chunk_hash(&stored.chunk_hash) {
            return Err(PackError::MalformedInput {
                reason: format!(
                    "the record for line {} carries {:?} as its chunk_hash, which is not \
                     {CHUNK_HASH_HEX_LEN} lowercase hex digits",
                    stored.line_id, stored.chunk_hash
                ),
            });
        }

        let expected = record_for(stored.line_id, &line, chunking_identity, &stored.chunk_hash);

        for ((field, artifact), (_, corpus_value)) in corpus_fields(stored)
            .into_iter()
            .zip(corpus_fields(&expected))
        {
            if artifact != corpus_value {
                return Err(PackError::RecordDisagreesWithCorpus {
                    line_id: stored.line_id,
                    field,
                    artifact,
                    corpus: corpus_value,
                });
            }
        }
    }
    Ok(())
}

/// Describe the payload the store just committed, and write the two metadata documents.
fn write_metadata(
    root: &Path,
    identity: &IndexVersion,
    created_at: &str,
    book_count: u32,
    vector_count: u32,
) -> Result<(), PackError> {
    let mut payloads: BTreeMap<String, PayloadDescriptor> = BTreeMap::new();
    for name in SNAPSHOT_FILENAMES {
        payloads.insert(
            name.to_string(),
            PayloadDescriptor::of_file(&root.join(name))?,
        );
    }
    let total_size_bytes = payloads.values().map(|payload| payload.size_bytes).sum();

    let package = IndexPackage {
        manifest: PackageManifest::new(
            identity.clone(),
            created_at.to_string(),
            book_count,
            vector_count,
            total_size_bytes,
        ),
        payloads,
    };
    Ok(IndexPackage::write(root, &package)?)
}

fn report(verified: &VerifiedPackage, identity: IndexVersion, book_count: u32) -> PackReport {
    PackReport {
        artifact_path: verified.root().to_path_buf(),
        identity,
        digest: verified.artifact_digest().to_string(),
        vector_count: verified.vector_count(),
        book_count,
        total_size_bytes: verified.manifest().total_size_bytes,
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Lowercase is required rather than normalized, for the reason
/// [`IndexVersion::validate_complete`](crate::semantic::versioning::IndexVersion::validate_complete)
/// requires it of a model checksum: the comparison is a string equality, and accepting
/// both cases would let the same digest fail to match itself.
fn is_lowercase_sha256(value: &str) -> bool {
    is_lowercase_hex(value, 64)
}

/// The shape of a stored `chunk_hash`. Checked rather than assumed, because
/// [`validate_artifact`] runs over artifacts this process did not write.
fn is_chunk_hash(value: &str) -> bool {
    is_lowercase_hex(value, CHUNK_HASH_HEX_LEN)
}

fn is_lowercase_hex(value: &str, digits: usize) -> bool {
    value.len() == digits
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

/// Stream the two input files as [`VectorInput`]s.
///
/// The vectors file is `vector_count × embedding_dim` little-endian `f32`s with no header
/// — what a producer gets from dumping an array — and the records file is one
/// [`VectorInputRecord`] per line, in the same order. Two files rather than one document
/// because the floats are the bulk and JSON is the wrong container for them.
///
/// The pairing is positional, so the two ways it can be wrong are both caught: a records
/// file longer than the vectors file runs out of bytes mid-record, and a shorter one
/// leaves bytes over, which is reported when the iterator ends rather than ignored.
pub fn read_vector_inputs(
    vectors_path: &Path,
    records_path: &Path,
    embedding_dim: u32,
) -> Result<impl Iterator<Item = Result<VectorInput, PackError>>, PackError> {
    let open = |path: &Path| -> Result<File, PackError> {
        File::open(path).map_err(|source| PackError::Io {
            context: format!("reading {}", path.display()),
            source,
        })
    };

    // Before the arithmetic below, which divides by it. A model identity that declares no
    // dimension is refused by `validate_complete` — but a caller reads the dimension out
    // of that identity to call this, and reaching a division by zero on the way to a good
    // error message is not a way to report anything.
    if embedding_dim == 0 {
        return Err(PackError::MalformedInput {
            reason: "the model identity declares an embedding_dim of 0, so there is no \
                     record width to read the vectors at"
                .to_string(),
        });
    }

    let vectors = open(vectors_path)?;
    let record_bytes = embedding_dim as u64 * 4;
    let length = vectors
        .metadata()
        .map_err(|source| PackError::Io {
            context: format!("inspecting {}", vectors_path.display()),
            source,
        })?
        .len();
    if length % record_bytes != 0 {
        return Err(PackError::MalformedInput {
            reason: format!(
                "{} holds {length} bytes, which is not a whole number of {embedding_dim}-\
                 dimensional f32 vectors ({record_bytes} bytes each)",
                vectors_path.display()
            ),
        });
    }

    Ok(VectorInputReader {
        records: BufReader::new(open(records_path)?).lines(),
        vectors: BufReader::new(vectors),
        vectors_path: vectors_path.to_path_buf(),
        records_path: records_path.to_path_buf(),
        embedding_dim: embedding_dim as usize,
        line_number: 0,
        done: false,
    })
}

struct VectorInputReader {
    records: io::Lines<BufReader<File>>,
    vectors: BufReader<File>,
    vectors_path: PathBuf,
    records_path: PathBuf,
    embedding_dim: usize,
    /// Lines of the records file consumed so far, across calls. A per-call counter looked
    /// right and named every fault "line 1", which is worse than no line number at all.
    line_number: usize,
    done: bool,
}

impl VectorInputReader {
    fn read_one(&mut self, line: &str, number: usize) -> Result<VectorInput, PackError> {
        let record: VectorInputRecord =
            serde_json::from_str(line).map_err(|error| PackError::MalformedInput {
                reason: format!(
                    "{} line {number} is not a vector record: {error}",
                    self.records_path.display()
                ),
            })?;

        let mut bytes = vec![0u8; self.embedding_dim * 4];
        self.vectors
            .read_exact(&mut bytes)
            .map_err(|error| PackError::MalformedInput {
                reason: format!(
                    "{} has no vector for record {number} (line_id {}): {error}",
                    self.vectors_path.display(),
                    record.line_id
                ),
            })?;

        let vector = bytes
            .chunks_exact(4)
            .map(|value| f32::from_le_bytes(value.try_into().expect("chunks of four bytes")))
            .collect();

        Ok(VectorInput {
            line_id: record.line_id,
            source_line_sha256: record.source_line_sha256,
            embedding_text_sha256: record.embedding_text_sha256,
            vector,
        })
    }

    /// Vectors left over once the records are exhausted, which means the two files
    /// describe different numbers of records.
    fn refuse_trailing_vectors(&mut self) -> Option<Result<VectorInput, PackError>> {
        let mut trailing = [0u8; 1];
        match self.vectors.read(&mut trailing) {
            Ok(0) => None,
            Ok(_) => Some(Err(PackError::MalformedInput {
                reason: format!(
                    "{} holds more vectors than {} has records",
                    self.vectors_path.display(),
                    self.records_path.display()
                ),
            })),
            Err(source) => Some(Err(PackError::Io {
                context: format!("reading {}", self.vectors_path.display()),
                source,
            })),
        }
    }
}

impl Iterator for VectorInputReader {
    type Item = Result<VectorInput, PackError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        loop {
            self.line_number += 1;
            let item = match self.records.next() {
                None => self.refuse_trailing_vectors(),
                Some(Err(source)) => Some(Err(PackError::Io {
                    context: format!("reading {}", self.records_path.display()),
                    source,
                })),
                Some(Ok(line)) if line.trim().is_empty() => continue,
                Some(Ok(line)) => {
                    let number = self.line_number;
                    Some(self.read_one(&line, number))
                }
            };
            // Nothing after a fault is meaningful: the two files are read in lockstep, so
            // one bad record leaves every later pairing off by one.
            if !matches!(item, Some(Ok(_))) {
                self.done = true;
            }
            return item;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::ArtifactError;
    use crate::semantic::chunker::compute_chunk_hash;
    use crate::semantic::versioning::{CorpusIdentity, IdentityField};
    use crate::semantic::zevc_store::{METADATA_FILENAME, VECTORS_FILENAME};
    use std::collections::HashMap;

    struct TempDir(PathBuf);
    impl TempDir {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "otzaria_test_packer_{name}_{}",
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

    const DIM: u32 = 8;
    const GENESIS: &str = "otzaria/tanach/genesis.txt";
    const BERACHOT: &str = "otzaria/mishna/berachot.txt";

    /// `(line_id, book, text)`, with ids formed the way `document_id_scheme_version` 1
    /// forms them: `((catalogue_order + 1) << 32) + (ordinal + 1)`.
    const LINES: [(u64, &str, &str); 3] = [
        (4_294_967_297, GENESIS, "בראשית ברא אלהים את השמים ואת הארץ"),
        (4_294_967_298, GENESIS, "ויאמר אלהים יהי אור ויהי אור"),
        (8_589_934_593, BERACHOT, "מאימתי קורין את שמע בערבית"),
    ];

    /// A corpus that answers from a table, and can be made to disagree with an artifact
    /// after one was built from it.
    ///
    /// The two sets are held **separately** on purpose. A real index answers `line()` for
    /// every document it holds while the recipe embeds only some of them, and a fake that
    /// derived one from the other could not express the case where a vector exists for a
    /// line that should never have been embedded — which is exactly the direction of the
    /// coverage check that is easy to leave out.
    struct FakeCorpus {
        identity: CorpusIdentity,
        lines: HashMap<u64, CorpusLine>,
        expected: BTreeSet<u64>,
    }

    impl FakeCorpus {
        fn new() -> Self {
            Self {
                identity: CorpusIdentity {
                    corpus_id: "9c".repeat(32),
                    library_version: "otzaria-library-2026-08".to_string(),
                    tantivy_schema_version: 3,
                    document_id_scheme_version: 1,
                },
                lines: LINES
                    .iter()
                    .map(|(line_id, book, text)| (*line_id, corpus_line(book, text)))
                    .collect(),
                expected: LINES.iter().map(|(line_id, _, _)| *line_id).collect(),
            }
        }

        /// The same corpus, with `line_id` still answerable but no longer embedded — a
        /// line the recipe skips for being too short to carry meaning.
        fn not_embedding(mut self, line_id: u64) -> Self {
            self.expected.remove(&line_id);
            self
        }
    }

    impl CorpusIndex for FakeCorpus {
        fn identity(&self) -> Result<CorpusIdentity, PackError> {
            Ok(self.identity.clone())
        }
        fn expected_line_ids(&self, _model: &ModelIdentity) -> Result<BTreeSet<u64>, PackError> {
            Ok(self.expected.clone())
        }
        fn line(&self, line_id: u64) -> Result<Option<CorpusLine>, PackError> {
            Ok(self.lines.get(&line_id).cloned())
        }
    }

    fn corpus_line(book: &str, text: &str) -> CorpusLine {
        CorpusLine {
            source_book_key: book.to_string(),
            title: if book == GENESIS {
                "בראשית"
            } else {
                "ברכות"
            }
            .to_string(),
            reference: format!("{book} — {}", text.chars().take(6).collect::<String>()),
            section_id: 1,
            segment: 0,
            is_pdf: false,
            line_hash: 0,
            content_hash: 42,
            facets: vec!["/מקרא/תורה".to_string()],
            text: text.to_string(),
        }
    }

    fn model() -> ModelIdentity {
        ModelIdentity {
            model_id: "EMD123/Otzaria-Embedding-V1-Flash-0.6B".to_string(),
            model_checksum: "a".repeat(64),
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

    fn request(output: &Path) -> PackRequest {
        PackRequest {
            output_path: output.to_path_buf(),
            model: model(),
            created_at: "2026-08-08T00:00:00Z".to_string(),
            collection_name: "chunks".to_string(),
        }
    }

    /// A deterministic vector that differs per text, so a misplaced one is visible.
    fn vector_for(text: &str) -> Vec<f32> {
        let digest = Sha256::digest(text.as_bytes());
        (0..DIM)
            .map(|i| f32::from(digest[i as usize]) + 1.0)
            .collect()
    }

    /// What a correct producer emits for a line whose recipe embedded it unchanged.
    fn input(line_id: u64, text: &str) -> VectorInput {
        embedded_as(line_id, text, text)
    }

    /// A line whose recipe embedded something else — a title prefix, neighbour context, a
    /// truncation. The source digest still names the corpus line; the embedding digest
    /// does not.
    fn embedded_as(line_id: u64, source: &str, embedded: &str) -> VectorInput {
        VectorInput {
            line_id,
            source_line_sha256: sha256_hex(source.as_bytes()),
            embedding_text_sha256: sha256_hex(embedded.as_bytes()),
            vector: vector_for(embedded),
        }
    }

    /// The inputs a correct producer emits for [`LINES`].
    fn good_inputs() -> Vec<Result<VectorInput, PackError>> {
        LINES
            .iter()
            .map(|(line_id, _, text)| Ok(input(*line_id, text)))
            .collect()
    }

    #[test]
    fn a_packed_artifact_verifies_and_reports_what_it_holds() {
        let dir = TempDir::new("happy");
        let output = dir.path().join("artifact");
        let corpus = FakeCorpus::new();

        let report = pack(request(&output), good_inputs(), &corpus).unwrap();

        assert_eq!(report.artifact_path, output);
        assert_eq!(report.vector_count, LINES.len() as u32);
        assert_eq!(report.book_count, 2);
        assert_eq!(report.identity.corpus, corpus.identity().unwrap());
        assert_eq!(report.identity.model, model());
        assert_eq!(report.identity.store, readable_store_identity());
        assert!(report.total_size_bytes > 0);

        // The digest a publisher announces is the artifact's own, and validating the
        // written directory reproduces every number.
        let revalidated = validate_artifact(&output, &model(), &corpus).unwrap();
        assert_eq!(revalidated.digest, report.digest);
        assert_eq!(revalidated.vector_count, report.vector_count);
        assert_eq!(revalidated.book_count, report.book_count);

        // And it is an artifact of this backend: exactly the snapshot files, plus the two
        // metadata documents.
        let mut written: Vec<String> = std::fs::read_dir(&output)
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
    }

    /// The failure the join exists for: the vectors and the ids drifted apart, so every
    /// vector describes its neighbour's line. Nothing in the vectors says so.
    #[test]
    fn vectors_paired_with_the_wrong_lines_are_refused() {
        let dir = TempDir::new("shifted");
        let corpus = FakeCorpus::new();

        // Each record keeps its id and takes the *next* line's vector and digests, which
        // is what a vector file sorted differently from its id list looks like. Coverage
        // is untouched — every id is still present exactly once — so only the text digest
        // can see it.
        let shifted: Vec<Result<VectorInput, PackError>> = LINES
            .iter()
            .enumerate()
            .map(|(index, (line_id, _, _))| {
                let (_, _, text) = LINES[(index + 1) % LINES.len()];
                Ok(input(*line_id, text))
            })
            .collect();

        match pack(request(&dir.path().join("artifact")), shifted, &corpus) {
            Err(PackError::LineTextMismatch { line_id, .. }) => {
                assert_eq!(line_id, LINES[0].0)
            }
            other => panic!("a shifted pairing must be refused, got {other:?}"),
        }
    }

    #[test]
    fn a_vector_for_a_line_the_corpus_does_not_hold_is_refused() {
        let dir = TempDir::new("absent_line");
        let mut inputs = good_inputs();
        inputs.push(Ok(input(999, "שורה שאינה בקטלוג")));

        match pack(
            request(&dir.path().join("artifact")),
            inputs,
            &FakeCorpus::new(),
        ) {
            Err(PackError::LineNotInCorpus { line_id }) => assert_eq!(line_id, 999),
            other => panic!("an unknown line must be refused, got {other:?}"),
        }
    }

    #[test]
    fn a_duplicated_line_is_refused_rather_than_silently_replacing_its_twin() {
        let dir = TempDir::new("duplicate");
        let mut inputs = good_inputs();
        inputs.push(Ok(input(LINES[0].0, LINES[0].2)));

        match pack(
            request(&dir.path().join("artifact")),
            inputs,
            &FakeCorpus::new(),
        ) {
            Err(PackError::DuplicateLineId { line_id }) => assert_eq!(line_id, LINES[0].0),
            other => panic!("a duplicated line must be refused, got {other:?}"),
        }
    }

    /// Uniform dimension and a usable direction, per vector rather than sampled: a record
    /// that cannot be scored is one that exists, counts, and can never be returned.
    #[test]
    fn a_vector_that_could_never_be_returned_is_refused_by_line() {
        let dir = TempDir::new("unusable");
        let corpus = FakeCorpus::new();

        let mut narrow = input(LINES[1].0, LINES[1].2);
        narrow.vector.pop();
        match pack(
            request(&dir.path().join("narrow")),
            vec![Ok(input(LINES[0].0, LINES[0].2)), Ok(narrow)],
            &corpus,
        ) {
            Err(PackError::VectorDimensionMismatch {
                line_id,
                expected,
                found,
            }) => {
                assert_eq!(line_id, LINES[1].0);
                assert_eq!(expected, DIM);
                assert_eq!(found, DIM as usize - 1);
            }
            other => panic!("a short vector must be refused, got {other:?}"),
        }

        for (label, values) in [
            ("non-finite", vec![f32::NAN; DIM as usize]),
            ("directionless", vec![0.0; DIM as usize]),
        ] {
            let mut broken = input(LINES[0].0, LINES[0].2);
            broken.vector = values;
            match pack(request(&dir.path().join(label)), vec![Ok(broken)], &corpus) {
                Err(PackError::UnusableVector { line_id, .. }) => {
                    assert_eq!(line_id, LINES[0].0, "{label}")
                }
                other => panic!("a {label} vector must be refused, got {other:?}"),
            }
        }
    }

    /// An incomplete identity is caught before a vector is read, because an artifact with
    /// a blank field opens against anything that left the same field blank.
    #[test]
    fn an_incomplete_identity_is_refused_before_anything_is_written() {
        let dir = TempDir::new("blank_identity");
        let output = dir.path().join("artifact");

        let mut blank = model();
        blank.model_checksum = String::new();
        match pack(
            PackRequest {
                model: blank,
                ..request(&output)
            },
            good_inputs(),
            &FakeCorpus::new(),
        ) {
            Err(PackError::Artifact(ArtifactError::IncompleteIdentity { field, .. })) => {
                assert_eq!(field, IdentityField::ModelChecksum)
            }
            other => panic!("a blank checksum must be refused, got {other:?}"),
        }
        assert!(
            !output.exists(),
            "a rejected pack must not leave a directory behind"
        );
    }

    /// Packing into a directory that already holds an artifact would load it and append
    /// to it, shipping vectors this run never joined to the corpus.
    #[test]
    fn packing_over_an_existing_artifact_is_refused() {
        let dir = TempDir::new("occupied");
        let output = dir.path().join("artifact");
        let corpus = FakeCorpus::new();
        pack(request(&output), good_inputs(), &corpus).unwrap();

        match pack(request(&output), good_inputs(), &corpus) {
            Err(PackError::UnusableOutput { reason, .. }) => {
                assert!(reason.contains("already holds"), "{reason}")
            }
            other => panic!("a second pack must be refused, got {other:?}"),
        }

        // An empty directory is a fine place to write one; a file is not.
        let empty = dir.path().join("empty");
        std::fs::create_dir_all(&empty).unwrap();
        assert!(pack(request(&empty), good_inputs(), &corpus).is_ok());

        let file = dir.path().join("a-file");
        std::fs::write(&file, b"not a directory").unwrap();
        match pack(request(&file), good_inputs(), &corpus) {
            Err(PackError::UnusableOutput { reason, .. }) => {
                assert!(reason.contains("not a directory"), "{reason}")
            }
            other => panic!("a file target must be refused, got {other:?}"),
        }
    }

    #[test]
    fn an_input_with_no_vectors_is_refused() {
        let dir = TempDir::new("empty_input");
        assert!(matches!(
            pack(
                request(&dir.path().join("artifact")),
                Vec::new(),
                &FakeCorpus::new()
            ),
            Err(PackError::NoVectors)
        ));
    }

    /// The catalogue moved under a finished artifact: the payload is intact, its
    /// checksums pass, and its records now describe a book by a title the corpus no
    /// longer uses. Only the join sees it.
    #[test]
    fn an_artifact_whose_corpus_has_moved_on_fails_validation_by_field() {
        let dir = TempDir::new("drift");
        let output = dir.path().join("artifact");
        let mut corpus = FakeCorpus::new();
        pack(request(&output), good_inputs(), &corpus).unwrap();

        corpus.lines.get_mut(&LINES[0].0).unwrap().reference = "מראה מקום מתוקן".to_string();

        match validate_artifact(&output, &model(), &corpus) {
            Err(PackError::RecordDisagreesWithCorpus {
                line_id,
                field,
                corpus: corpus_value,
                ..
            }) => {
                assert_eq!(line_id, LINES[0].0);
                assert_eq!(field, "reference");
                assert_eq!(corpus_value, "מראה מקום מתוקן");
            }
            other => panic!("a drifted corpus must fail validation, got {other:?}"),
        }

        // A line that disappeared from the catalogue is the other half of the same claim.
        let mut without = FakeCorpus::new();
        without.lines.remove(&LINES[2].0);
        match validate_artifact(&output, &model(), &without) {
            Err(PackError::LineNotInCorpus { line_id }) => assert_eq!(line_id, LINES[2].0),
            other => panic!("a removed line must fail validation, got {other:?}"),
        }
    }

    /// An artifact validates against the corpus and model it was built for, and against
    /// no other. This is the same comparison the runtime makes, minus the model file.
    #[test]
    fn validation_refuses_an_artifact_built_for_another_corpus_or_model() {
        let dir = TempDir::new("foreign");
        let output = dir.path().join("artifact");
        let corpus = FakeCorpus::new();
        pack(request(&output), good_inputs(), &corpus).unwrap();

        let mut elsewhere = FakeCorpus::new();
        elsewhere.identity.corpus_id = "1b".repeat(32);
        let mut other_weights = model();
        other_weights.model_checksum = "b".repeat(64);

        for (field, verdict) in [
            (
                IdentityField::CorpusId,
                validate_artifact(&output, &model(), &elsewhere),
            ),
            (
                IdentityField::ModelChecksum,
                validate_artifact(&output, &other_weights, &corpus),
            ),
        ] {
            match verdict {
                Err(PackError::Artifact(ArtifactError::IdentityMismatch { mismatches })) => {
                    assert_eq!(mismatches.len(), 1, "{field}");
                    assert_eq!(mismatches[0].field, field);
                }
                other => panic!("{field} must refuse the artifact, got {other:?}"),
            }
        }
    }

    /// Validation reads the payload, so damage the metadata alone cannot see is damage it
    /// reports — the same two layers the runtime relies on.
    #[test]
    fn validation_refuses_a_tampered_payload() {
        let dir = TempDir::new("tampered");
        let output = dir.path().join("artifact");
        let corpus = FakeCorpus::new();
        pack(request(&output), good_inputs(), &corpus).unwrap();

        let vectors = output.join(VECTORS_FILENAME);
        let mut bytes = std::fs::read(&vectors).unwrap();
        let before = bytes.len();
        bytes[0] ^= 0xff;
        std::fs::write(&vectors, &bytes).unwrap();
        assert_eq!(std::fs::metadata(&vectors).unwrap().len() as usize, before);

        match validate_artifact(&output, &model(), &corpus) {
            Err(PackError::Artifact(ArtifactError::PayloadChecksumFailed { payload, .. })) => {
                assert_eq!(payload, VECTORS_FILENAME)
            }
            other => panic!("a tampered payload must be refused, got {other:?}"),
        }

        // And a record removed from the payload is a count the manifest no longer holds.
        std::fs::write(&vectors, &bytes[..bytes.len() - DIM as usize * 4]).unwrap();
        let metadata_path = output.join(METADATA_FILENAME);
        let text = std::fs::read_to_string(&metadata_path).unwrap();
        let kept: Vec<&str> = text.lines().take(LINES.len() - 1).collect();
        std::fs::write(&metadata_path, format!("{}\n", kept.join("\n"))).unwrap();

        assert!(matches!(
            validate_artifact(&output, &model(), &corpus),
            Err(PackError::Artifact(_))
        ));
    }

    /// A field added to a stored record that is in neither list would be written into
    /// every artifact and checked by nothing. Driven off the serialized record, so it sees
    /// exactly what the payload carries.
    #[test]
    fn every_stored_field_is_compared_or_deliberately_not() {
        let record = record_for(
            LINES[0].0,
            &corpus_line(GENESIS, LINES[0].2),
            7,
            &sha256_hex(b"embedded"),
        );
        let mut carried: Vec<String> = serde_json::to_value(&record)
            .unwrap()
            .as_object()
            .expect("a record is a JSON object")
            .keys()
            .cloned()
            .collect();

        let mut accounted: Vec<String> = corpus_fields(&record)
            .iter()
            .map(|(name, _)| (*name).to_string())
            .chain([UNCOMPARABLE_FIELD.to_string()])
            .collect();
        accounted.sort();
        carried.sort();
        assert_eq!(
            accounted, carried,
            "a stored field is in neither list, or a list names a field the record does \
             not carry"
        );
    }

    /// `chunk_hash` is stored as the first 128 bits of the producer's SHA-256, and the
    /// chunker computes it as the first 128 bits of its own SHA-256. If those ever stop
    /// being the same operation, an artifact from this packer and one from the prototype
    /// indexing path would describe the same text differently.
    #[test]
    fn a_chunk_hash_is_the_documented_prefix_of_a_sha256() {
        for text in ["בראשית ברא", "", "a longer line with some words in it"] {
            assert_eq!(
                compute_chunk_hash(text),
                sha256_hex(text.as_bytes())[..CHUNK_HASH_HEX_LEN],
                "for {text:?}"
            );
        }
    }

    /// The gap that made "official artifact" mean nothing: one good vector out of a whole
    /// library passes every count, every checksum and every identity field.
    #[test]
    fn an_artifact_that_covers_part_of_the_corpus_is_refused() {
        let dir = TempDir::new("coverage");
        let corpus = FakeCorpus::new();

        match pack(
            request(&dir.path().join("artifact")),
            vec![Ok(input(LINES[0].0, LINES[0].2))],
            &corpus,
        ) {
            Err(PackError::CoverageMismatch {
                expected,
                covered,
                missing,
                unexpected,
                first_missing,
                first_unexpected,
            }) => {
                assert_eq!(expected, LINES.len());
                assert_eq!(covered, 1);
                assert_eq!(missing, LINES.len() - 1);
                assert_eq!(unexpected, 0);
                // The smallest missing id, so two runs over the same fault agree.
                assert_eq!(first_missing, Some(LINES[1].0));
                assert_eq!(first_unexpected, None);
            }
            other => panic!("a partial artifact must be refused, got {other:?}"),
        }

        // And a corpus that grows after an artifact was built leaves that artifact
        // incomplete, which validation on its own has to see.
        let output = dir.path().join("complete");
        pack(request(&output), good_inputs(), &corpus).unwrap();

        let mut grown = FakeCorpus::new();
        grown
            .lines
            .insert(12_884_901_889, corpus_line("otzaria/new.txt", "שורה חדשה"));
        grown.expected.insert(12_884_901_889);
        match validate_artifact(&output, &model(), &grown) {
            Err(PackError::CoverageMismatch { first_missing, .. }) => {
                assert_eq!(first_missing, Some(12_884_901_889))
            }
            other => panic!("a corpus that grew must fail validation, got {other:?}"),
        }
    }

    /// The example id in a coverage rejection is the smallest, not the first one a walk
    /// happens to reach. Pinned directly, because the counts are summarized in one pass
    /// with no list to sort afterwards, and "two runs name the same line" rests on it.
    #[test]
    fn a_difference_is_summarized_by_its_size_and_its_smallest_member() {
        let expected: BTreeSet<u64> = [90, 10, 50, 30].into_iter().collect();
        let covered: BTreeSet<u64> = [50, 7].into_iter().collect();

        assert_eq!(difference_summary(&expected, &covered), (3, Some(10)));
        assert_eq!(difference_summary(&covered, &expected), (1, Some(7)));
        assert_eq!(difference_summary(&expected, &expected), (0, None));
    }

    /// The other direction, and the one that is easy to leave out: a vector for a line the
    /// recipe does **not** embed.
    ///
    /// Nothing else here can see it. The per-input join asks whether the corpus holds the
    /// line and it does — a line skipped for being too short to carry meaning exists
    /// perfectly well — so only "is this one of the lines that should have been embedded"
    /// catches it. And it matters: a vector that should not exist means the artifact was
    /// built by a recipe other than the one it declares.
    #[test]
    fn a_vector_for_a_line_the_recipe_does_not_embed_is_refused() {
        let dir = TempDir::new("unexpected_coverage");
        let skipped = LINES[1].0;
        let corpus = FakeCorpus::new().not_embedding(skipped);

        // Every remaining line is covered, so the missing side is clean.
        match pack(
            request(&dir.path().join("artifact")),
            good_inputs(),
            &corpus,
        ) {
            Err(PackError::CoverageMismatch {
                expected,
                covered,
                missing,
                unexpected,
                first_missing,
                first_unexpected,
            }) => {
                assert_eq!(expected, LINES.len() - 1);
                assert_eq!(covered, LINES.len());
                assert_eq!(missing, 0);
                assert_eq!(unexpected, 1);
                assert_eq!(first_missing, None);
                assert_eq!(first_unexpected, Some(skipped));
            }
            other => panic!("an extra vector must be refused, got {other:?}"),
        }

        // The corpus still answers for that line, which is why nothing before the coverage
        // check could have refused it.
        assert!(corpus.line(skipped).unwrap().is_some());

        // And an artifact already holding it fails validation on its own, which is what a
        // recipe change after a build looks like.
        let output = dir.path().join("built-under-the-old-recipe");
        pack(request(&output), good_inputs(), &FakeCorpus::new()).unwrap();
        match validate_artifact(&output, &model(), &corpus) {
            Err(PackError::CoverageMismatch {
                unexpected,
                first_unexpected,
                ..
            }) => {
                assert_eq!(unexpected, 1);
                assert_eq!(first_unexpected, Some(skipped));
            }
            other => panic!("an extra vector must fail validation, got {other:?}"),
        }
    }

    /// The recipe's text is not the corpus line whenever it prefixes, borrows context or
    /// truncates. The record then has to describe what was embedded — deriving it from the
    /// corpus would put a digest of a text nothing was built from into every record.
    #[test]
    fn a_chunk_hash_describes_the_embedded_text_and_not_the_corpus_line() {
        let dir = TempDir::new("embedded_text");
        let output = dir.path().join("artifact");
        let corpus = FakeCorpus::new();

        // The first line is embedded with its title prefixed; the rest are embedded raw.
        let contextualized = format!("בראשית — {}", LINES[0].2);
        let mut inputs = vec![Ok(embedded_as(LINES[0].0, LINES[0].2, &contextualized))];
        inputs.extend(
            LINES[1..]
                .iter()
                .map(|(line_id, _, text)| Ok(input(*line_id, text))),
        );

        pack(request(&output), inputs, &corpus).unwrap();

        let stored = std::fs::read_to_string(output.join(METADATA_FILENAME)).unwrap();
        assert!(
            stored.contains(&compute_chunk_hash(&contextualized)),
            "the record must carry a digest of the text that was embedded"
        );
        assert!(
            !stored.contains(&compute_chunk_hash(LINES[0].2)),
            "and not one of the corpus line, which nothing was built from"
        );

        // It survives validation, because the corpus is not asked about it.
        assert!(validate_artifact(&output, &model(), &corpus).is_ok());
    }

    /// The field the corpus cannot answer still has a shape, and an artifact carrying
    /// something else in it is not one this packer would have written.
    #[test]
    fn a_chunk_hash_that_is_not_one_is_refused_by_validation() {
        let dir = TempDir::new("bad_chunk_hash");
        let output = dir.path().join("artifact");
        let corpus = FakeCorpus::new();
        pack(request(&output), good_inputs(), &corpus).unwrap();

        // Rewrite one record's chunk_hash, and the per-record checksum that covers it, so
        // the payload stays internally consistent and only the shape check is left.
        let metadata_path = output.join(METADATA_FILENAME);
        let text = std::fs::read_to_string(&metadata_path).unwrap();
        let mut lines: Vec<String> = text.lines().map(str::to_string).collect();
        let mut first: serde_json::Value = serde_json::from_str(&lines[0]).unwrap();
        // Through the struct, because that is what the reader hashes: it deserializes the
        // record and re-serializes it, so the checksum covers the canonical field order
        // and not the order the bytes happen to be in.
        let mut metadata: VectorMetadata =
            serde_json::from_value(first["metadata"].take()).unwrap();
        metadata.chunk_hash = "not a hash".to_string();
        let canonical = serde_json::to_vec(&metadata).unwrap();
        first["metadata"] = serde_json::from_slice(&canonical).unwrap();
        first["metadata_sha256"] = serde_json::Value::String(sha256_hex(&canonical));
        lines[0] = serde_json::to_string(&first).unwrap();
        std::fs::write(&metadata_path, format!("{}\n", lines.join("\n"))).unwrap();
        // The payload declarations have to follow, or this is caught as damage instead.
        redeclare(&output);

        match validate_artifact(&output, &model(), &corpus) {
            Err(PackError::MalformedInput { reason }) => {
                assert!(reason.contains("chunk_hash"), "{reason}")
            }
            other => panic!("a malformed chunk_hash must be refused, got {other:?}"),
        }
    }

    /// Re-describe a directory's payload after a test edited it, so what is under test is
    /// the check being aimed at rather than the checksum that would fire first.
    fn redeclare(root: &Path) {
        let existing = IndexPackage::read(root).unwrap();
        let payloads: BTreeMap<String, PayloadDescriptor> = SNAPSHOT_FILENAMES
            .iter()
            .map(|name| {
                (
                    (*name).to_string(),
                    PayloadDescriptor::of_file(&root.join(name)).unwrap(),
                )
            })
            .collect();
        let manifest = PackageManifest {
            total_size_bytes: payloads.values().map(|payload| payload.size_bytes).sum(),
            ..existing.manifest
        };
        IndexPackage::write(root, &IndexPackage { manifest, payloads }).unwrap();
    }

    // ── the input files ──

    /// Write the two files a producer emits, and return their paths.
    fn write_inputs(dir: &TempDir, name: &str, inputs: &[VectorInput]) -> (PathBuf, PathBuf) {
        let vectors_path = dir.path().join(format!("{name}.f32"));
        let records_path = dir.path().join(format!("{name}.jsonl"));

        let mut bytes = Vec::new();
        let mut records = String::new();
        for input in inputs {
            for value in &input.vector {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            records.push_str(&format!(
                "{}\n",
                serde_json::to_string(&VectorInputRecord {
                    line_id: input.line_id,
                    source_line_sha256: input.source_line_sha256.clone(),
                    embedding_text_sha256: input.embedding_text_sha256.clone(),
                })
                .unwrap()
            ));
        }
        std::fs::write(&vectors_path, bytes).unwrap();
        std::fs::write(&records_path, records).unwrap();
        (vectors_path, records_path)
    }

    #[test]
    fn the_input_files_stream_back_the_records_that_were_written() {
        let dir = TempDir::new("input_round_trip");
        let written: Vec<VectorInput> = LINES
            .iter()
            .map(|(line_id, _, text)| input(*line_id, text))
            .collect();
        let (vectors_path, records_path) = write_inputs(&dir, "good", &written);

        let read: Vec<VectorInput> = read_vector_inputs(&vectors_path, &records_path, DIM)
            .unwrap()
            .map(Result::unwrap)
            .collect();

        assert_eq!(read.len(), written.len());
        for (read, written) in read.iter().zip(&written) {
            assert_eq!(read.line_id, written.line_id);
            assert_eq!(read.source_line_sha256, written.source_line_sha256);
            assert_eq!(read.embedding_text_sha256, written.embedding_text_sha256);
            assert_eq!(read.vector, written.vector);
        }
    }

    /// The pairing is positional, so both ways of losing alignment have to be caught —
    /// and neither is visible from one file alone.
    #[test]
    fn input_files_that_describe_different_numbers_of_records_are_refused() {
        let dir = TempDir::new("input_lengths");
        let written: Vec<VectorInput> = LINES
            .iter()
            .map(|(line_id, _, text)| input(*line_id, text))
            .collect();
        let (vectors_path, records_path) = write_inputs(&dir, "base", &written);

        // One record too few: bytes are left over when the records run out.
        let short_records = dir.path().join("short.jsonl");
        let text = std::fs::read_to_string(&records_path).unwrap();
        std::fs::write(
            &short_records,
            format!("{}\n", text.lines().take(2).collect::<Vec<_>>().join("\n")),
        )
        .unwrap();
        let verdict: Vec<Result<VectorInput, PackError>> =
            read_vector_inputs(&vectors_path, &short_records, DIM)
                .unwrap()
                .collect();
        match verdict.last() {
            Some(Err(PackError::MalformedInput { reason })) => {
                assert!(reason.contains("more vectors"), "{reason}")
            }
            other => panic!("leftover vectors must be refused, got {other:?}"),
        }

        // One record too many: the vector file runs out mid-record.
        let long_records = dir.path().join("long.jsonl");
        std::fs::write(
            &long_records,
            format!("{text}{}", text.lines().next().unwrap()),
        )
        .unwrap();
        let verdict: Vec<Result<VectorInput, PackError>> =
            read_vector_inputs(&vectors_path, &long_records, DIM)
                .unwrap()
                .collect();
        match verdict.last() {
            Some(Err(PackError::MalformedInput { reason })) => {
                assert!(reason.contains("no vector for record"), "{reason}")
            }
            other => panic!("a missing vector must be refused, got {other:?}"),
        }

        // A vector file that is not a whole number of records is refused before a byte of
        // it is paired with anything.
        let ragged = dir.path().join("ragged.f32");
        let mut bytes = std::fs::read(&vectors_path).unwrap();
        bytes.push(0);
        std::fs::write(&ragged, bytes).unwrap();
        match read_vector_inputs(&ragged, &records_path, DIM).map(|_| ()) {
            Err(PackError::MalformedInput { reason }) => {
                assert!(reason.contains("whole number"), "{reason}")
            }
            other => panic!("a ragged vector file must be refused, got {other:?}"),
        }
    }

    /// A build log names the line to fix. The counter therefore has to survive between
    /// calls to `next` — a per-call one reported every fault as line 1.
    #[test]
    fn a_malformed_record_is_reported_against_the_line_it_is_on() {
        let dir = TempDir::new("input_line_number");
        let written: Vec<VectorInput> = LINES
            .iter()
            .map(|(line_id, _, text)| input(*line_id, text))
            .collect();
        let (vectors_path, records_path) = write_inputs(&dir, "base", &written);

        // Break the third record, leaving the first two well formed.
        let text = std::fs::read_to_string(&records_path).unwrap();
        let mut lines: Vec<String> = text.lines().map(str::to_string).collect();
        lines[2] = "{ not a record".to_string();
        std::fs::write(&records_path, format!("{}\n", lines.join("\n"))).unwrap();

        let verdict: Vec<Result<VectorInput, PackError>> =
            read_vector_inputs(&vectors_path, &records_path, DIM)
                .unwrap()
                .collect();
        assert_eq!(verdict.len(), 3, "the reader stops at the first fault");
        match verdict.last() {
            Some(Err(PackError::MalformedInput { reason })) => {
                assert!(reason.contains("line 3"), "{reason}")
            }
            other => panic!("a malformed record must be refused, got {other:?}"),
        }
    }

    /// A digest in the wrong shape is a producer bug, and saying so beats reporting it as
    /// a text mismatch against a corpus that is perfectly fine.
    #[test]
    fn a_text_digest_that_is_not_a_lowercase_sha256_is_named_as_malformed_input() {
        let dir = TempDir::new("bad_digest");
        for field in ["source_line_sha256", "embedding_text_sha256"] {
            let mut wrong_case = input(LINES[0].0, LINES[0].2);
            match field {
                "source_line_sha256" => {
                    wrong_case.source_line_sha256 = wrong_case.source_line_sha256.to_uppercase()
                }
                _ => {
                    wrong_case.embedding_text_sha256 =
                        wrong_case.embedding_text_sha256.to_uppercase()
                }
            }

            match pack(
                request(&dir.path().join(field)),
                vec![Ok(wrong_case)],
                &FakeCorpus::new(),
            ) {
                Err(PackError::MalformedInput { reason }) => {
                    assert!(
                        reason.contains("lowercase hex") && reason.contains(field),
                        "{reason}"
                    )
                }
                other => panic!("an uppercase {field} must be refused, got {other:?}"),
            }
        }
    }

    /// A corpus that cannot answer is a broken build input, and it must not be reported
    /// as a vector whose line does not exist — the fixes are different.
    #[test]
    fn a_corpus_that_fails_is_reported_as_a_corpus_fault() {
        struct Failing;
        impl CorpusIndex for Failing {
            fn identity(&self) -> Result<CorpusIdentity, PackError> {
                Err(PackError::Corpus {
                    reason: "the index could not be opened".to_string(),
                })
            }
            fn expected_line_ids(
                &self,
                _model: &ModelIdentity,
            ) -> Result<BTreeSet<u64>, PackError> {
                unreachable!("the identity is read first")
            }
            fn line(&self, _line_id: u64) -> Result<Option<CorpusLine>, PackError> {
                unreachable!("the identity is read first")
            }
        }

        let dir = TempDir::new("corpus_failure");
        match pack(
            request(&dir.path().join("artifact")),
            good_inputs(),
            &Failing,
        ) {
            Err(PackError::Corpus { reason }) => assert!(reason.contains("could not be opened")),
            other => panic!("a corpus failure must be reported as one, got {other:?}"),
        }
    }
}
