//! Carrying vectors forward from one library release to the next.
//!
//! A vector is a function of the text that was embedded and nothing else — same model,
//! same backend, same recipe, same float. So the question "has this line already been
//! embedded?" has a complete answer that costs no inference: **is its
//! `embedding_text_sha256` in the previous release's ledger?**
//!
//! That key is deliberately not a line identity. Two things go wrong with one:
//!
//! * The runtime id (`document_id_scheme_version` 1) is positional, so inserting a book
//!   moves the ids of every book after it and a diff sees millions of changed lines where
//!   nothing changed.
//! * A stable id would be worse than useless here, because it would be *quietly* wrong:
//!   a line whose own text is untouched still gets a new embedding text when the
//!   neighbour it borrows context from changes. An id-keyed diff calls that unchanged and
//!   reuses a vector built from text the library no longer contains.
//!
//! The digest cannot make either mistake. It describes the string that was actually
//! embedded, after context and truncation, which is exactly what the vector depends on.
//!
//! ```text
//! plan_split           plan + ledger + its manifest -> reuse.jsonl + embed.jsonl
//! embed_shard          embed.jsonl                  -> vectors + records     (GPU)
//! assemble             reuse + base + shards        -> vectors + records
//! pack                 those two                    -> artifact
//! ledger_from_artifact artifact + records           -> the next ledger
//! ```
//!
//! `assemble` writes in whatever order is cheapest, because
//! [`pack`](super::packer::pack) compares the *set* of ids it was handed against the
//! recipe's expected set and sorts internally. What it may not do is pair a vector with
//! the wrong record, which is why the two files are written in lockstep here as well.
//!
//! **And that sort is why the ledger is built last.** `pack` orders the payload by
//! `semantic_id`, so the offsets `assemble` could report are not offsets into anything
//! downloadable. The first published ledger was built from the assembler and every entry
//! was wrong. [`ledger_from_artifact`] reads the artifact's own order instead, and joins
//! it to `records.jsonl` for the full digest — the payload keeps a 32-hex `chunk_hash`,
//! which is enough to compare at runtime and not enough to decide what to skip.

use crate::distribution::packer::VectorInputRecord;
use crate::distribution::shard::{PlannedChunk, ShardReport};
use crate::errors::PackError;
use crate::semantic::versioning::ModelIdentity;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::{BufRead, Read, Seek, SeekFrom, Write};

/// One line of a ledger: a digest, and where its vector sits in the **published**
/// `vectors.bin`.
///
/// Published beside an artifact and never shipped to a user — it exists so the *next*
/// build can decide what to skip, and a user's installation has nothing to decide.
///
/// **The offset is into the artifact, not into whatever `assemble` wrote.** `pack` sorts
/// the payload by `semantic_id` before writing it (`ZevcStore::save_to_disk`), so the
/// order `assemble` produced is not the order anybody can download. A ledger built from
/// the assembler's order was published once and is wrong in every entry: 20 000 of 20 000
/// sampled offsets named a different line. It is therefore built by
/// [`ledger_from_artifact`], after packing, from the artifact itself.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerEntry {
    pub embedding_text_sha256: String,
    /// Index of the vector, not a byte offset: the width is the artifact's
    /// `embedding_dim`, and multiplying here would bake it into the file.
    pub offset: u64,
}

/// A planned line whose vector already exists, and where to find it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReuseEntry {
    pub line_id: u64,
    pub source_line_sha256: String,
    pub embedding_text_sha256: String,
    pub base_offset: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitReport {
    pub planned: usize,
    pub reused: usize,
    pub to_embed: usize,
}

/// One shard's vectors and the records that pair with them, already opened.
pub type ShardStreams = (Box<dyn Read>, Box<dyn BufRead>);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssembleReport {
    pub vectors: usize,
    pub reused: usize,
    pub embedded: usize,
    pub embedding_dim: usize,
}

/// Everything a ledger's offsets are only meaningful against.
///
/// Published beside the ledger, and checked before a single vector is copied. Without it
/// `plan_split` sees digests and integers: an artifact built by a different model, or a
/// `vectors.bin` that is not the file the ledger was written from, both reuse cleanly and
/// silently, and the result is a library of vectors from two incompatible spaces.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LedgerManifest {
    /// Digest of the artifact this ledger describes, as published outside it.
    pub artifact_digest: String,
    /// SHA-256 of the `vectors.bin` the offsets index into.
    pub vectors_sha256: String,
    /// SHA-256 of the ledger itself. Without it the manifest binds the file the offsets
    /// point *into* and not the offsets, so a single edited line reuses cleanly.
    pub ledger_sha256: String,
    pub vector_count: usize,
    pub embedding_dim: u32,
    /// The whole vector space, not a summary of it. Compared field by field.
    pub model: ModelIdentity,
}

impl LedgerManifest {
    /// Refuse a ledger that describes a different vector space from the one being built.
    ///
    /// Every field of [`ModelIdentity`] is compared, and deliberately: the ones that look
    /// cosmetic are not. A different `pooling` reads a different token, a different
    /// `max_tokens` truncates elsewhere, and a different `chunking_identity` means the text
    /// that produced the digest was assembled by another recipe. Two vectors can agree on
    /// their embedding text and still be incomparable.
    ///
    /// # Errors
    ///
    /// [`PackError::LedgerDisagreesWithBuild`], naming the first field that differs.
    pub fn ensure_matches(&self, model: &ModelIdentity, dim: u32) -> Result<(), PackError> {
        for (field, base, target) in [
            ("model_id", &self.model.model_id, &model.model_id),
            (
                "model_checksum",
                &self.model.model_checksum,
                &model.model_checksum,
            ),
            (
                "model_quantization",
                &self.model.model_quantization,
                &model.model_quantization,
            ),
            (
                "embedding_backend",
                &self.model.embedding_backend,
                &model.embedding_backend,
            ),
            ("pooling", &self.model.pooling, &model.pooling),
        ] {
            if base != target {
                return Err(PackError::LedgerDisagreesWithBuild {
                    field,
                    ledger: base.clone(),
                    build: target.clone(),
                });
            }
        }
        for (field, base, target) in [
            (
                "embedding_dim",
                u64::from(self.model.embedding_dim),
                u64::from(model.embedding_dim),
            ),
            (
                "max_tokens",
                self.model.max_tokens as u64,
                model.max_tokens as u64,
            ),
            (
                "embedding_text_version",
                u64::from(self.model.embedding_text_version),
                u64::from(model.embedding_text_version),
            ),
            (
                "normalization_version",
                u64::from(self.model.normalization_version),
                u64::from(model.normalization_version),
            ),
            (
                "chunking_identity",
                self.model.chunking_identity,
                model.chunking_identity,
            ),
            (
                "declared embedding_dim",
                u64::from(self.embedding_dim),
                u64::from(dim),
            ),
        ] {
            if base != target {
                return Err(PackError::LedgerDisagreesWithBuild {
                    field,
                    ledger: base.to_string(),
                    build: target.to_string(),
                });
            }
        }
        Ok(())
    }
}

/// Build the ledger from the artifact that was actually published.
///
/// `metadata` is the artifact's own `metadata.jsonl`, whose order **is** the order of
/// `vectors.bin`; `records` is the pairing that travelled with the vectors, and the only
/// place the full digest exists — the artifact stores a 32-hex `chunk_hash`, which is
/// enough to compare at runtime and not enough to decide whether inference can be skipped.
/// The join is on `line_id`, so no text is re-read and no vector is re-embedded.
///
/// # Errors
///
/// [`PackError::Corpus`] for malformed input, and
/// [`PackError::LineNotInCorpus`] for an artifact record whose `line_id` the records file
/// does not describe — which would otherwise become a ledger entry pointing at a vector
/// nobody can identify.
pub fn ledger_from_artifact(
    metadata: impl BufRead,
    records: impl BufRead,
    sink: &mut dyn Write,
) -> Result<usize, PackError> {
    let mut digests: HashMap<u64, String> = HashMap::new();
    for line in records.lines() {
        let line = line.map_err(read_error)?;
        if line.trim().is_empty() {
            continue;
        }
        let record: VectorInputRecord =
            serde_json::from_str(&line).map_err(|error| PackError::Corpus {
                reason: format!("a record is malformed: {error}"),
            })?;
        if !is_sha256_hex(&record.embedding_text_sha256) {
            return Err(PackError::MalformedInput {
                reason: format!(
                    "line {}'s digest is {:?}, which is not 64 lowercase hex digits",
                    record.line_id, record.embedding_text_sha256
                ),
            });
        }
        // A `HashMap` would have taken the last quietly. Two records for one line is a
        // records file assembled from more than one run, and which digest wins decides
        // which vector a future build reuses.
        if digests
            .insert(record.line_id, record.embedding_text_sha256)
            .is_some()
        {
            return Err(PackError::DuplicateLineId {
                line_id: record.line_id,
            });
        }
    }

    let mut offset = 0u64;
    for line in metadata.lines() {
        let line = line.map_err(read_error)?;
        if line.trim().is_empty() {
            continue;
        }
        // Only what is needed: the id, and the truncated digest the payload already holds.
        #[derive(Deserialize)]
        struct Stored {
            metadata: StoredMetadata,
        }
        #[derive(Deserialize)]
        struct StoredMetadata {
            line_id: u64,
            chunk_hash: String,
        }
        let stored: Stored = serde_json::from_str(&line).map_err(|error| PackError::Corpus {
            reason: format!("an artifact metadata record is malformed: {error}"),
        })?;
        let line_id = stored.metadata.line_id;
        let embedding_text_sha256 = digests
            .remove(&line_id)
            .ok_or(PackError::LineNotInCorpus { line_id })?;

        // The join is on `line_id`, which a records file from another build shares. This
        // is what makes it a join and not a coincidence: the artifact's own `chunk_hash`
        // is the first 32 characters of the full digest, so a records file describing
        // different text for the same id cannot pass.
        // Exactly 32, compared directly. `min(a.len(), b.len())` accepted an empty
        // `chunk_hash` — the empty prefix equals the empty prefix — and any short one that
        // happened to match, which is the prefix check quietly not happening.
        if !is_chunk_hash(&stored.metadata.chunk_hash) {
            return Err(PackError::MalformedInput {
                reason: format!(
                    "line {line_id}'s stored chunk_hash is {:?}, which is not 32 lowercase \
                     hex digits",
                    stored.metadata.chunk_hash
                ),
            });
        }
        if stored.metadata.chunk_hash != embedding_text_sha256[..CHUNK_HASH_LEN] {
            return Err(PackError::LineTextMismatch {
                line_id,
                declared: embedding_text_sha256,
                actual: stored.metadata.chunk_hash,
            });
        }

        write_json(
            sink,
            &LedgerEntry {
                embedding_text_sha256,
                offset,
            },
        )?;
        offset += 1;
    }

    // Everything left over described a vector the artifact does not hold. A records file
    // longer than the payload is one that belongs to a different build, and a ledger built
    // from it would be complete, well-formed and about something else.
    if let Some(line_id) = digests.keys().min().copied() {
        return Err(PackError::MalformedInput {
            reason: format!(
                "{} record(s) describe lines the artifact does not hold, the first being {line_id}",
                digests.len()
            ),
        });
    }
    sink.flush().map_err(read_error)?;
    Ok(offset as usize)
}

/// How much of the digest the payload keeps. Enough to compare at runtime, and not
/// enough to decide whether inference can be skipped — which is why the ledger needs the
/// records file as well as the artifact.
const CHUNK_HASH_LEN: usize = 32;

/// Exactly [`CHUNK_HASH_LEN`] lowercase hex digits.
fn is_chunk_hash(value: &str) -> bool {
    value.len() == CHUNK_HASH_LEN && value.bytes().all(is_lower_hex)
}

/// 64 lowercase hex digits, and nothing else.
///
/// A digest is compared as a string everywhere it is used, so `"ABC…"` and `"abc…"` are
/// two different keys for one vector — and a truncated one silently matches a prefix.
fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(is_lower_hex)
}

fn is_lower_hex(byte: u8) -> bool {
    byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)
}

/// The lookup a first build gets: nothing to reuse from.
static EMPTY_OFFSETS: std::sync::LazyLock<HashMap<String, u64>> =
    std::sync::LazyLock::new(HashMap::new);

/// A base artifact that has been checked, and the only thing reuse can be given.
///
/// The point is that it cannot be constructed without the check. Before this type
/// existed, `LedgerManifest` declared `vectors_sha256`, `ledger_sha256` and
/// `vector_count` and nothing compared any of them: a foreign `vectors.bin` of the same
/// length, or one offset moved to another in-range value, was copied and then accepted —
/// the packer re-normalises what it is handed and the record still names the right line,
/// so nothing downstream could see it.
///
/// So the fields are private and [`Self::open`] is the only way in.
pub struct VerifiedBase {
    manifest: LedgerManifest,
    /// Digest to offset, already bounds-checked against `vector_count`.
    offsets: HashMap<String, u64>,
    vectors: std::fs::File,
}

impl VerifiedBase {
    /// Hash both files, bound every offset, and hold the model to the manifest.
    ///
    /// Hashing 23 GB of vectors is minutes, once per release, and it is the only thing
    /// that distinguishes the base artifact from a file of the same size. Skipping it
    /// would leave the manifest as documentation.
    ///
    /// # Errors
    ///
    /// [`PackError::LedgerDisagreesWithBuild`] for an identity mismatch, and
    /// [`PackError::MalformedInput`] for a digest that does not match, a vector file whose
    /// length is not `vector_count * dim * 4`, or an offset outside it.
    pub fn open(
        manifest: LedgerManifest,
        ledger_path: &std::path::Path,
        vectors_path: &std::path::Path,
        model: &ModelIdentity,
    ) -> Result<Self, PackError> {
        manifest.ensure_matches(model, model.embedding_dim)?;

        let ledger_bytes = std::fs::read(ledger_path).map_err(read_error)?;
        let actual = sha256_hex(&ledger_bytes);
        if actual != manifest.ledger_sha256 {
            return Err(PackError::MalformedInput {
                reason: format!(
                    "the ledger hashes to {actual} and its manifest declares {}",
                    manifest.ledger_sha256
                ),
            });
        }

        let width = u64::from(manifest.embedding_dim) * 4;
        let expected_bytes = (manifest.vector_count as u64)
            .checked_mul(width)
            .ok_or_else(|| PackError::MalformedInput {
                reason: "the declared vector count and width overflow a file length".to_string(),
            })?;
        let mut vectors = std::fs::File::open(vectors_path).map_err(read_error)?;
        let length = vectors.metadata().map_err(read_error)?.len();
        if length != expected_bytes {
            return Err(PackError::MalformedInput {
                reason: format!(
                    "the base holds {length} byte(s) and {} vector(s) of width {width} need \
                     {expected_bytes}",
                    manifest.vector_count
                ),
            });
        }
        // Hashed through the handle that will later be read from, and rewound — not
        // reopened by path. Reopening leaves a window in which the file that was checked
        // and the file that is used are two different files.
        let actual = sha256_reader(&mut vectors)?;
        vectors.seek(SeekFrom::Start(0)).map_err(read_error)?;
        if actual != manifest.vectors_sha256 {
            return Err(PackError::MalformedInput {
                reason: format!(
                    "the base vectors hash to {actual} and the manifest declares {}",
                    manifest.vectors_sha256
                ),
            });
        }

        let mut offsets = HashMap::new();
        for line in ledger_bytes.split(|byte| *byte == b'\n') {
            if line.iter().all(u8::is_ascii_whitespace) {
                continue;
            }
            let entry: LedgerEntry =
                serde_json::from_slice(line).map_err(|error| PackError::Corpus {
                    reason: format!("a ledger entry is malformed: {error}"),
                })?;
            if entry.offset >= manifest.vector_count as u64 {
                return Err(PackError::MalformedInput {
                    reason: format!(
                        "ledger offset {} is outside the {} vector(s) the base holds",
                        entry.offset, manifest.vector_count
                    ),
                });
            }
            // Last writer wins, and it does not matter: one digest is one text and
            // therefore one vector, wherever a healthy ledger records it.
            offsets.insert(entry.embedding_text_sha256, entry.offset);
        }

        Ok(Self {
            manifest,
            offsets,
            vectors,
        })
    }

    /// The artifact digest this base was published under, for the run manifest.
    pub fn artifact_digest(&self) -> &str {
        &self.manifest.artifact_digest
    }

    /// Digest to offset, every one of them already inside `vector_count`.
    pub(crate) fn offsets(&self) -> &HashMap<String, u64> {
        &self.offsets
    }
}

fn sha256_reader(file: &mut impl Read) -> Result<String, PackError> {
    Ok(sha256_and_lines(file)?.0)
}

/// The digest and the number of non-empty lines, from one pass over the bytes.
///
/// Both facts come from the same read because the second one is not optional: the digest
/// says the file is the file its manifest describes, and the count says how many vectors
/// [`assemble`] will pair with it. Hashing without counting left that to be discovered
/// mid-merge, with output already written.
///
/// "Non-empty" is the same test `assemble` applies when it walks the records — a line of
/// nothing but whitespace is skipped there and not counted here.
fn sha256_and_lines(file: &mut impl Read) -> Result<(String, usize), PackError> {
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    let mut lines = 0usize;
    let mut has_content = false;
    loop {
        let read = file.read(&mut buffer).map_err(read_error)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        for byte in &buffer[..read] {
            if *byte == b'\n' {
                lines += usize::from(has_content);
                has_content = false;
            } else if !byte.is_ascii_whitespace() {
                has_content = true;
            }
        }
    }
    // A last line with no newline after it is still a line.
    lines += usize::from(has_content);
    Ok((format!("{:x}", hasher.finalize()), lines))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Split a plan against a ledger: what can be copied, and what has to be embedded.
///
/// Streams the plan, so the memory cost is the ledger alone — 32 bytes of digest and 8 of
/// offset per previous vector, which for a six-million-line library is a few hundred
/// megabytes on a build machine and nothing on the GPU worker, which never sees it.
///
/// An empty ledger is not an error. It is the first build, and every line goes to the GPU.
///
/// # Errors
///
/// [`PackError::Corpus`] for a malformed ledger or plan line, or a sink that will not
/// accept bytes.
pub fn plan_split(
    plan: impl BufRead,
    base: Option<&VerifiedBase>,
    reuse_sink: &mut dyn Write,
    embed_sink: &mut dyn Write,
) -> Result<SplitReport, PackError> {
    // A [`VerifiedBase`] or nothing. There is no way to hand this function a ledger whose
    // digests, length and offsets have not already been checked against the artifact they
    // claim to describe, because that type cannot be built without checking them.
    let known: &HashMap<String, u64> = match base {
        Some(base) => base.offsets(),
        None => &EMPTY_OFFSETS,
    };

    let (mut planned, mut reused) = (0usize, 0usize);
    for line in plan.lines() {
        let line = line.map_err(read_error)?;
        if line.trim().is_empty() {
            continue;
        }
        let chunk: PlannedChunk =
            serde_json::from_str(&line).map_err(|error| PackError::Corpus {
                reason: format!("a plan record is malformed: {error}"),
            })?;
        planned += 1;

        match known.get(&chunk.embedding_text_sha256) {
            Some(&base_offset) => {
                reused += 1;
                write_json(
                    reuse_sink,
                    &ReuseEntry {
                        line_id: chunk.line_id,
                        source_line_sha256: chunk.source_line_sha256,
                        embedding_text_sha256: chunk.embedding_text_sha256,
                        base_offset,
                    },
                )?;
            }
            // The plan record travels whole: the worker embeds `embedding_text` and can
            // carry no corpus, so anything dropped here cannot be recovered there.
            None => write_json(embed_sink, &chunk)?,
        }
    }

    reuse_sink.flush().map_err(read_error)?;
    embed_sink.flush().map_err(read_error)?;
    Ok(SplitReport {
        planned,
        reused,
        to_embed: planned - reused,
    })
}

/// Every shard's manifest, checked against the plan they claim to cover.
///
/// Before this, `assemble` opened `vectors.f32` and `records.jsonl` directly and never
/// read `shard-manifest.json` at all: its counts and digests were written and then
/// believed by nobody. A shard from another export, a corrupted vector that stayed
/// finite, or two shards covering one window and none covering another all merged
/// cleanly — the packer re-normalises the floats and the id set can still be complete.
///
/// What is checked, per shard: the plan it was cut from, its model *and* the width that
/// model declares, that it wrote exactly the records its window asked for, that
/// `vectors.f32` holds that many vectors and `records.jsonl` that many records, and that
/// both files hash to what its own manifest recorded. Then, across shards: the windows
/// tile `[0, total)` exactly — no hole, no overlap.
///
/// Nothing in [`ShardReport`] is read and then ignored. `take` and `embedding_dim` were,
/// for a while, and a manifest field nobody compares is a field that can say anything.
///
/// Every one of those is checked before a byte is copied, which is the point of doing it
/// here rather than leaving it to [`assemble`]: the merge writes as it reads, so a shard it
/// refuses halfway has already put output on disk.
///
/// # Errors
///
/// [`PackError::MalformedInput`], naming the shard and what disagreed.
pub fn verify_shards(
    shards: &[(std::path::PathBuf, ShardReport)],
    plan_sha256: &str,
    model: &ModelIdentity,
    total: usize,
) -> Result<Vec<ShardStreams>, PackError> {
    // A release where no embedding text changed needs no GPU at all: every vector comes
    // from the base and there are no shard directories. That is the cheapest path there
    // is, and refusing an empty set unconditionally made it the one path that could not
    // run.
    if total == 0 {
        return if shards.is_empty() {
            Ok(Vec::new())
        } else {
            Err(PackError::MalformedInput {
                reason: format!(
                    "{} shard(s) were produced for a plan that needs no embedding",
                    shards.len()
                ),
            })
        };
    }
    if shards.is_empty() {
        return Err(PackError::NoVectors);
    }
    // Path, window, and the two handles that were hashed. Ordering happens after every
    // shard has been checked, so a refusal never depends on which one came first.
    let mut checked: Vec<(usize, usize, &std::path::Path, std::fs::File, std::fs::File)> =
        Vec::with_capacity(shards.len());

    for (dir, manifest) in shards {
        let named = dir.display();
        if manifest.plan_sha256 != plan_sha256 {
            return Err(PackError::MalformedInput {
                reason: format!(
                    "{named} was embedded from plan {} and this build's plan is {plan_sha256}",
                    manifest.plan_sha256
                ),
            });
        }
        if manifest.model != *model {
            return Err(PackError::MalformedInput {
                reason: format!("{named} was embedded by a different model identity"),
            });
        }
        // The width the worker actually got back from its backend, against the width the
        // model promises. `assemble` strides through both files by this number, so a shard
        // that disagrees would be read at the wrong offset from its first vector on.
        if manifest.embedding_dim != model.embedding_dim as usize {
            return Err(PackError::MalformedInput {
                reason: format!(
                    "{named} holds {}-wide vectors and the model declares {}",
                    manifest.embedding_dim, model.embedding_dim
                ),
            });
        }
        // The window, held to the plan. `read_plan` skips and takes over *records*, so a
        // shard covers exactly what remains of the plan after its skip, capped by its take
        // — and a shard that stopped early is a truncated GPU session, not a short window.
        let remaining =
            total
                .checked_sub(manifest.skip)
                .ok_or_else(|| PackError::MalformedInput {
                    reason: format!(
                        "{named} starts at {} and the plan holds {total} record(s)",
                        manifest.skip
                    ),
                })?;
        let owed = manifest.take.min(remaining);
        if manifest.records != owed {
            return Err(PackError::MalformedInput {
                reason: format!(
                    "{named} was given {} record(s) from {} and wrote {}",
                    owed, manifest.skip, manifest.records
                ),
            });
        }
        // Opened once, hashed through the handle, rewound, and carried out of here. The
        // caller cannot reopen by path, so the file that was checked is the file that is
        // read.
        let open = |file: &str, declared: &str| -> Result<(std::fs::File, usize), PackError> {
            let mut handle = std::fs::File::open(dir.join(file)).map_err(read_error)?;
            let (actual, lines) = sha256_and_lines(&mut handle)?;
            if actual != declared {
                return Err(PackError::MalformedInput {
                    reason: format!(
                        "{named}/{file} hashes to {actual} and its manifest declares {declared}"
                    ),
                });
            }
            handle.seek(SeekFrom::Start(0)).map_err(read_error)?;
            Ok((handle, lines))
        };
        let (vectors, _) = open("vectors.f32", &manifest.vectors_sha256)?;
        let (records, lines) = open("records.jsonl", &manifest.records_sha256)?;
        // The count `assemble` will actually pair with vectors, from the pass that hashed
        // the file. A shard with three vectors, a manifest saying three, and two records in
        // it used to reach the merge — which copied part of its output before noticing.
        if lines != manifest.records {
            return Err(PackError::MalformedInput {
                reason: format!(
                    "{named}/records.jsonl holds {lines} record(s) and its manifest declares {}",
                    manifest.records
                ),
            });
        }
        // Through the handle that was hashed, before a byte is copied: a file of the wrong
        // length would otherwise surface as `assemble` running out of vectors halfway.
        let owed_bytes = (manifest.records as u64)
            .checked_mul(manifest.embedding_dim as u64)
            .and_then(|values| values.checked_mul(4))
            .ok_or_else(|| PackError::MalformedInput {
                reason: format!(
                    "{named} declares {} vectors of {} floats, which is no file",
                    manifest.records, manifest.embedding_dim
                ),
            })?;
        let length = vectors.metadata().map_err(read_error)?.len();
        if length != owed_bytes {
            return Err(PackError::MalformedInput {
                reason: format!(
                    "{named}/vectors.f32 is {length} bytes and {} vector(s) of {} floats \
                     are {owed_bytes}",
                    manifest.records, manifest.embedding_dim
                ),
            });
        }
        checked.push((
            manifest.skip,
            manifest.records,
            dir.as_path(),
            vectors,
            records,
        ));
    }

    checked.sort_by_key(|(skip, records, dir, _, _)| (*skip, *records, *dir));
    let mut covered = 0usize;
    for (skip, records, dir, _, _) in &checked {
        if *skip != covered {
            return Err(PackError::MalformedInput {
                reason: if *skip > covered {
                    format!(
                        "records {covered}..{skip} are covered by no shard; {} starts at {skip}",
                        dir.display()
                    )
                } else {
                    format!(
                        "{} starts at {skip} and records up to {covered} are already covered",
                        dir.display()
                    )
                },
            });
        }
        covered = covered
            .checked_add(*records)
            .ok_or_else(|| PackError::MalformedInput {
                reason: format!("the shards claim more records than a count can hold: {covered}"),
            })?;
    }
    if covered != total {
        return Err(PackError::MalformedInput {
            reason: format!("the shards cover {covered} record(s) and the plan holds {total}"),
        });
    }

    Ok(checked
        .into_iter()
        .map(|(_, _, _, vectors, records)| {
            (
                Box::new(std::io::BufReader::new(vectors)) as Box<dyn Read>,
                Box::new(std::io::BufReader::new(records)) as Box<dyn BufRead>,
            )
        })
        .collect())
}

/// Copy the reusable vectors out of the base artifact, append this run's shards, and
/// write the ledger the *next* release will split against.
///
/// Reads of the base file are issued in ascending offset order rather than in plan order.
/// The base is tens of gigabytes and the reuse set is most of it; in plan order those
/// reads are a random walk over the whole file, and sorting first turns them into a
/// forward scan. It costs one sort of 8-byte keys and changes nothing about the result,
/// because the pairing travels in the record written beside each vector.
///
/// # Errors
///
/// [`PackError::MalformedInput`] for a base file too short for an offset it was asked
/// for, and [`PackError::Corpus`] for unreadable input or an unwritable sink.
pub fn assemble(
    mut reuse: Vec<ReuseEntry>,
    base: Option<&VerifiedBase>,
    shards: Vec<ShardStreams>,
    embedding_dim: usize,
    vectors_sink: &mut dyn Write,
    records_sink: &mut dyn Write,
) -> Result<AssembleReport, PackError> {
    let width = embedding_dim
        .checked_mul(4)
        .ok_or_else(|| PackError::MalformedInput {
            reason: format!("an embedding dimension of {embedding_dim} has no byte width"),
        })?;
    let mut buffer = vec![0u8; width];

    let reused = reuse.len();
    if reused > 0 {
        let base = base.ok_or_else(|| PackError::MalformedInput {
            reason: format!("{reused} vector(s) are to be reused and no verified base was given"),
        })?;
        let mut base = &base.vectors;
        reuse.sort_unstable_by_key(|entry| entry.base_offset);
        for entry in reuse {
            let at = entry.base_offset.checked_mul(width as u64).ok_or_else(|| {
                PackError::MalformedInput {
                    reason: format!("offset {} is beyond any file", entry.base_offset),
                }
            })?;
            base.seek(SeekFrom::Start(at)).map_err(read_error)?;
            base.read_exact(&mut buffer)
                .map_err(|error| PackError::MalformedInput {
                    reason: format!(
                        "the base artifact has no vector at offset {}: {error}",
                        entry.base_offset
                    ),
                })?;
            vectors_sink.write_all(&buffer).map_err(read_error)?;
            write_json(
                records_sink,
                &VectorInputRecord {
                    line_id: entry.line_id,
                    source_line_sha256: entry.source_line_sha256,
                    embedding_text_sha256: entry.embedding_text_sha256,
                },
            )?;
        }
    }

    let mut embedded = 0usize;
    for (mut shard_vectors, shard_records) in shards {
        for line in shard_records.lines() {
            let line = line.map_err(read_error)?;
            if line.trim().is_empty() {
                continue;
            }
            let record: VectorInputRecord =
                serde_json::from_str(&line).map_err(|error| PackError::Corpus {
                    reason: format!("a shard record is malformed: {error}"),
                })?;
            // One vector per record, read in lockstep. A shard whose two files disagree
            // on length fails here rather than silently shifting every pairing after it.
            shard_vectors
                .read_exact(&mut buffer)
                .map_err(|error| PackError::MalformedInput {
                    reason: format!(
                        "a shard has {} record(s) and ran out of vectors: {error}",
                        embedded + 1
                    ),
                })?;
            vectors_sink.write_all(&buffer).map_err(read_error)?;
            write_json(records_sink, &record)?;
            embedded += 1;
        }
        let mut tail = [0u8; 1];
        if shard_vectors.read(&mut tail).map_err(read_error)? != 0 {
            return Err(PackError::MalformedInput {
                reason: "a shard has more vector bytes than records".to_string(),
            });
        }
    }

    vectors_sink.flush().map_err(read_error)?;
    records_sink.flush().map_err(read_error)?;

    Ok(AssembleReport {
        vectors: reused + embedded,
        reused,
        embedded,
        embedding_dim,
    })
}

fn write_json<T: Serialize>(sink: &mut dyn Write, value: &T) -> Result<(), PackError> {
    let mut line = serde_json::to_vec(value).map_err(|error| PackError::Corpus {
        reason: format!("a record could not be written: {error}"),
    })?;
    line.push(b'\n');
    sink.write_all(&line).map_err(read_error)
}

fn read_error(error: std::io::Error) -> PackError {
    PackError::Corpus {
        reason: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic::chunker::ChunkerConfig;
    use std::io::Cursor;

    const DIM: usize = 2;
    const DIM_U32: u32 = 2;

    struct TempDir(std::path::PathBuf);

    impl TempDir {
        fn new(name: &str) -> Self {
            // The clock alone collided: macOS ticks coarser than a test takes to start.
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "otzaria_reuse_{name}_{}_{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos(),
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn model_for(checksum: &str, chunking: &ChunkerConfig) -> ModelIdentity {
        ModelIdentity {
            model_id: "otzaria-embedding-v1".to_string(),
            model_checksum: checksum.to_string(),
            model_quantization: "Q4_K_M".to_string(),
            embedding_backend: "mock-hash-v1".to_string(),
            embedding_dim: DIM_U32,
            pooling: "last-token".to_string(),
            max_tokens: 512,
            embedding_text_version: 1,
            normalization_version: 1,
            chunking_identity: chunking.identity(),
        }
    }

    fn planned(line_id: u64, digest: &str) -> String {
        serde_json::to_string(&PlannedChunk {
            line_id,
            source_line_sha256: format!("{line_id:064}"),
            embedding_text_sha256: digest.to_string(),
            embedding_text: format!("text of {line_id}"),
        })
        .unwrap()
            + "\n"
    }

    fn vectors_of(values: &[[f32; DIM]]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| v.iter().flat_map(|f| f.to_le_bytes()))
            .collect()
    }

    /// A base on disk, with an honest manifest. Tests then damage one thing at a time.
    struct Base {
        dir: TempDir,
        manifest: LedgerManifest,
        model: ModelIdentity,
    }

    fn base_with(entries: &[(&str, u64)], values: &[[f32; DIM]]) -> Base {
        let dir = TempDir::new("base");
        let ledger: String = entries
            .iter()
            .map(|(digest, offset)| {
                serde_json::to_string(&LedgerEntry {
                    embedding_text_sha256: (*digest).to_string(),
                    offset: *offset,
                })
                .unwrap()
                    + "\n"
            })
            .collect();
        let ledger_path = dir.0.join("ledger.jsonl");
        std::fs::write(&ledger_path, &ledger).unwrap();
        std::fs::write(dir.0.join("vectors.bin"), vectors_of(values)).unwrap();
        // From the bytes on disk, which is what `open` reads. Hashing the in-memory
        // string instead would make a difference between them invisible to the test.
        let ledger_on_disk = std::fs::read(&ledger_path).unwrap();
        let model = model_for(&"ab".repeat(32), &ChunkerConfig::default());
        Base {
            manifest: LedgerManifest {
                artifact_digest: "d".repeat(64),
                vectors_sha256: sha256_hex(&vectors_of(values)),
                ledger_sha256: sha256_hex(&ledger_on_disk),
                vector_count: values.len(),
                embedding_dim: DIM_U32,
                model: model.clone(),
            },
            model,
            dir,
        }
    }

    impl Base {
        fn open(&self) -> Result<VerifiedBase, PackError> {
            VerifiedBase::open(
                self.manifest.clone(),
                &self.dir.0.join("ledger.jsonl"),
                &self.dir.0.join("vectors.bin"),
                &self.model,
            )
        }
    }

    /// The whole point, as one assertion: a line whose embedding text is unchanged does
    /// not reach the GPU, and one whose text is new does.
    #[test]
    fn only_the_digests_the_ledger_does_not_know_are_sent_to_be_embedded() {
        let base = base_with(
            &[("aa", 7), ("cc", 0)],
            &[
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
                [4.0, 4.0],
                [5.0, 5.0],
                [6.0, 6.0],
                [7.0, 7.0],
            ],
        );
        let verified = base.open().unwrap();
        let plan = planned(1, "aa") + &planned(2, "bb") + &planned(3, "cc");
        let (mut reuse, mut embed) = (Vec::new(), Vec::new());

        let report =
            plan_split(Cursor::new(plan), Some(&verified), &mut reuse, &mut embed).unwrap();

        assert_eq!((report.planned, report.reused, report.to_embed), (3, 2, 1));
        let reused: Vec<ReuseEntry> = String::from_utf8(reuse)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(
            reused
                .iter()
                .map(|e| (e.line_id, e.base_offset))
                .collect::<Vec<_>>(),
            vec![(1, 7), (3, 0)],
            "each reused line must name the offset its own digest had"
        );
        let embedded: Vec<PlannedChunk> = String::from_utf8(embed)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(embedded.len(), 1);
        assert_eq!(embedded[0].line_id, 2);
        assert_eq!(
            embedded[0].embedding_text, "text of 2",
            "the worker has no corpus, so the text has to travel with the record"
        );
    }

    /// A reused vector must be the base vector its offset names.
    #[test]
    fn a_reused_vector_is_the_one_its_offset_points_at() {
        let base = base_with(
            &[("d1", 1), ("d3", 3)],
            &[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        );
        let verified = base.open().unwrap();
        let reuse = vec![
            ReuseEntry {
                line_id: 10,
                source_line_sha256: "s3".to_string(),
                embedding_text_sha256: "d3".to_string(),
                base_offset: 3,
            },
            ReuseEntry {
                line_id: 11,
                source_line_sha256: "s1".to_string(),
                embedding_text_sha256: "d1".to_string(),
                base_offset: 1,
            },
        ];
        let (mut vectors, mut records) = (Vec::new(), Vec::new());

        let report = assemble(
            reuse,
            Some(&verified),
            Vec::new(),
            DIM,
            &mut vectors,
            &mut records,
        )
        .unwrap();

        assert_eq!((report.vectors, report.reused, report.embedded), (2, 2, 0));
        // Ascending base offset, not the order the reuse list was given in.
        assert_eq!(vectors, vectors_of(&[[1.0, 1.0], [3.0, 3.0]]));
        let ids: Vec<u64> = String::from_utf8(records)
            .unwrap()
            .lines()
            .map(|l| {
                serde_json::from_str::<VectorInputRecord>(l)
                    .unwrap()
                    .line_id
            })
            .collect();
        assert_eq!(ids, vec![11, 10], "the record must follow its own vector");
    }

    /// Each of these was reusable before `VerifiedBase` existed, and none of them is
    /// visible afterwards: the packer re-normalises whatever it is handed and the record
    /// still names the right line, so a wrong vector is accepted in silence.
    #[test]
    fn a_base_that_is_not_the_one_the_manifest_describes_is_refused() {
        let values = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
        let entries = [("d1", 1), ("d3", 3)];

        // A different file of exactly the same length.
        let swapped = base_with(&entries, &values);
        std::fs::write(
            swapped.dir.0.join("vectors.bin"),
            vectors_of(&[[9.0, 9.0], [8.0, 8.0], [7.0, 7.0], [6.0, 6.0]]),
        )
        .unwrap();
        expect_refusal(swapped.open(), "hash to");

        // One offset moved to another value that is still in range.
        let mut moved = base_with(&[("d1", 2), ("d3", 3)], &values);
        moved.manifest.ledger_sha256 = sha256_hex(
            &std::fs::read(base_with(&entries, &values).dir.0.join("ledger.jsonl")).unwrap(),
        );
        expect_refusal(moved.open(), "ledger hashes to");

        // An offset outside the vectors the base holds.
        let outside = base_with(&[("d1", 99)], &values);
        expect_refusal(outside.open(), "outside");

        // A count that does not match the file.
        let mut short = base_with(&entries, &values);
        short.manifest.vector_count = 3;
        expect_refusal(short.open(), "byte(s)");

        // And the honest one still opens, so the tests above are not all failing for some
        // shared reason.
        assert!(base_with(&entries, &values).open().is_ok());
    }

    fn expect_refusal(outcome: Result<VerifiedBase, PackError>, expected: &str) {
        match outcome {
            Err(error) => {
                let text = error.to_string();
                assert!(
                    text.contains(expected),
                    "expected a refusal mentioning {expected:?}, got {text:?}"
                );
            }
            Ok(_) => panic!("a base that is not the one described must not open"),
        }
    }

    /// The bug that was published once, as a test.
    ///
    /// `assemble` writes in one order and `pack` sorts the payload by `semantic_id`, so a
    /// ledger built from the assembler is wrong in every entry. This builds one from the
    /// artifact's own order and asserts the offsets follow *that* — with the two orders
    /// deliberately different, because identical orders would pass either way.
    #[test]
    fn the_ledger_follows_the_artifact_order_and_not_the_assembler_s() {
        let a = "a".repeat(64);
        let b = "b".repeat(64);
        let records = format!(
            "{{\"line_id\":10,\"source_line_sha256\":\"s\",\"embedding_text_sha256\":\"{a}\"}}\n\
             {{\"line_id\":11,\"source_line_sha256\":\"s\",\"embedding_text_sha256\":\"{b}\"}}\n"
        );
        // What `pack` published: 11 first, because `semantic_id` sorted it there.
        let metadata = format!(
            "{{\"metadata\":{{\"line_id\":11,\"chunk_hash\":\"{}\"}}}}\n\
             {{\"metadata\":{{\"line_id\":10,\"chunk_hash\":\"{}\"}}}}\n",
            &b[..32],
            &a[..32]
        );

        let mut ledger = Vec::new();
        let written =
            ledger_from_artifact(Cursor::new(metadata), Cursor::new(records), &mut ledger).unwrap();

        assert_eq!(written, 2);
        let entries: Vec<LedgerEntry> = String::from_utf8(ledger)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(
            entries
                .iter()
                .map(|e| (e.embedding_text_sha256.clone(), e.offset))
                .collect::<Vec<_>>(),
            vec![(b, 0), (a, 1)],
            "offset 0 must name the line the artifact holds at offset 0"
        );
    }

    /// Every one of these merged cleanly while the manifests were decoration: the packer
    /// re-normalises the floats and the id set can still come out complete.
    #[test]
    fn shards_that_do_not_tile_this_plan_are_refused() {
        let dir = TempDir::new("shards");
        let model = model_for(&"ab".repeat(32), &ChunkerConfig::default());
        let write = |name: &str, skip: usize, records: usize, plan: &str, model: &ModelIdentity| {
            let at = dir.0.join(name);
            std::fs::create_dir_all(&at).unwrap();
            let vectors = vectors_of(&vec![[1.0, 1.0]; records]);
            let body: String = (0..records)
                .map(|i| {
                    format!("{{\"line_id\":{i},\"source_line_sha256\":\"s\",\"embedding_text_sha256\":\"d\"}}\n")
                })
                .collect();
            std::fs::write(at.join("vectors.f32"), &vectors).unwrap();
            std::fs::write(at.join("records.jsonl"), &body).unwrap();
            (
                at,
                ShardReport {
                    plan_sha256: plan.to_string(),
                    skip,
                    take: records,
                    records,
                    embedding_dim: DIM,
                    vectors_sha256: sha256_hex(&vectors),
                    records_sha256: sha256_hex(body.as_bytes()),
                    model: model.clone(),
                },
            )
        };

        let a = write("a", 0, 2, "plan", &model);
        let b = write("b", 2, 2, "plan", &model);
        assert!(verify_shards(&[a.clone(), b.clone()], "plan", &model, 4).is_ok());

        // A hole: nothing covers records 2..4.
        let far = write("far", 4, 2, "plan", &model);
        expect_reason(
            verify_shards(&[a.clone(), far], "plan", &model, 6),
            "covered by no shard",
        );

        // An overlap: two shards claim the same window.
        let again = write("again", 0, 2, "plan", &model);
        expect_reason(
            verify_shards(&[a.clone(), again], "plan", &model, 4),
            "already covered",
        );

        // A shard of another export, with a window of exactly the right shape.
        let foreign = write("foreign", 2, 2, "other-plan", &model);
        expect_reason(
            verify_shards(&[a.clone(), foreign], "plan", &model, 4),
            "and this build's plan is",
        );

        // A shard embedded by another model.
        let other_model = ModelIdentity {
            model_checksum: "cd".repeat(32),
            ..model.clone()
        };
        let mixed = write("mixed", 2, 2, "plan", &other_model);
        expect_reason(
            verify_shards(&[a.clone(), mixed], "plan", &model, 4),
            "different model identity",
        );

        // A bit flipped in a vector, leaving it finite — which is what the packer would
        // have normalised and accepted.
        let flipped = write("flipped", 2, 2, "plan", &model);
        std::fs::write(
            flipped.0.join("vectors.f32"),
            vectors_of(&[[1.0, 1.0], [2.0, 2.0]]),
        )
        .unwrap();
        expect_reason(
            verify_shards(&[a.clone(), flipped], "plan", &model, 4),
            "hashes to",
        );

        // Short of the plan.
        expect_reason(
            verify_shards(&[a, b], "plan", &model, 6),
            "cover 4 record(s)",
        );
    }

    /// A manifest field nobody compares is a field that can say anything. Every shard
    /// below is internally consistent — both digests match its own two files — and every
    /// one of them is refused, because the *plan* says something else.
    #[test]
    fn a_shard_manifest_that_disagrees_with_the_plan_is_refused() {
        let dir = TempDir::new("fields");
        let model = model_for(&"ab".repeat(32), &ChunkerConfig::default());
        // `vectors` vectors and `lines` records on disk, hashed honestly, with a manifest
        // the caller then bends one field of.
        let write = |name: &str, vectors: usize, lines: usize, bend: &dyn Fn(&mut ShardReport)| {
            let at = dir.0.join(name);
            std::fs::create_dir_all(&at).unwrap();
            let floats = vectors_of(&vec![[1.0, 1.0]; vectors]);
            let body: String = (0..lines)
                .map(|i| format!("{{\"line_id\":{i},\"source_line_sha256\":\"s\",\"embedding_text_sha256\":\"d\"}}\n"))
                .collect();
            std::fs::write(at.join("vectors.f32"), &floats).unwrap();
            std::fs::write(at.join("records.jsonl"), &body).unwrap();
            let mut manifest = ShardReport {
                plan_sha256: "plan".to_string(),
                skip: 0,
                take: lines,
                records: lines,
                embedding_dim: DIM,
                vectors_sha256: sha256_hex(&floats),
                records_sha256: sha256_hex(body.as_bytes()),
                model: model.clone(),
            };
            bend(&mut manifest);
            vec![(at, manifest)]
        };

        // A width the model does not declare. `assemble` strides by this number, so every
        // vector after the first would be read from the middle of its neighbour.
        expect_reason(
            verify_shards(
                &write("wide", 2, 2, &|m| m.embedding_dim = 4),
                "plan",
                &model,
                2,
            ),
            "holds 4-wide vectors and the model declares 2",
        );

        // A session that stopped early: the window asked for four records and two came
        // back. Nothing else in the set would notice, because this shard is the whole set.
        expect_reason(
            verify_shards(&write("short", 2, 2, &|m| m.take = 4), "plan", &model, 4),
            "was given 4 record(s) from 0 and wrote 2",
        );

        // A window that begins past the end of the plan.
        expect_reason(
            verify_shards(&write("beyond", 2, 2, &|m| m.skip = 8), "plan", &model, 4),
            "starts at 8 and the plan holds 4",
        );

        // A record lost from the file its own manifest counted: three vectors, three
        // declared, two written. Both digests honest.
        expect_reason(
            verify_shards(
                &write("missing", 3, 2, &|m| {
                    m.records = 3;
                    m.take = 3;
                }),
                "plan",
                &model,
                3,
            ),
            "records.jsonl holds 2 record(s) and its manifest declares 3",
        );

        // And one too many, which is the same defect from the other side: the merge would
        // have paired the third record with the first vector of whatever came next.
        expect_reason(
            verify_shards(&write("extra", 2, 3, &|m| m.records = 2), "plan", &model, 2),
            "records.jsonl holds 3 record(s) and its manifest declares 2",
        );

        // Three records, three lines, and a vectors file holding two — every digest
        // honest. Before the length check this reached `assemble` and ran out of floats
        // halfway through the merge.
        expect_reason(
            verify_shards(
                &write("truncated", 2, 3, &|m| m.records = 3),
                "plan",
                &model,
                3,
            ),
            "is 16 bytes and 3 vector(s) of 2 floats are 24",
        );

        // The same shard, honest about all of it, is accepted.
        assert!(verify_shards(&write("whole", 2, 2, &|_| {}), "plan", &model, 2).is_ok());
    }

    fn expect_reason<T>(outcome: Result<T, PackError>, expected: &str) {
        match outcome {
            Err(error) => {
                let text = error.to_string();
                assert!(
                    text.contains(expected),
                    "expected {expected:?}, got {text:?}"
                );
            }
            Ok(_) => panic!("expected a refusal"),
        }
    }

    /// A records file from another build shares the line ids and describes other text.
    /// The artifact's own truncated `chunk_hash` is what makes the join checkable, and
    /// each of these was accepted before it was compared.
    #[test]
    fn a_records_file_that_does_not_describe_this_artifact_is_refused() {
        let a = "a".repeat(64);
        let b = "b".repeat(64);
        let record = |id: u64, digest: &str| {
            format!("{{\"line_id\":{id},\"source_line_sha256\":\"s\",\"embedding_text_sha256\":\"{digest}\"}}\n")
        };
        let stored = |id: u64, hash: &str| {
            format!("{{\"metadata\":{{\"line_id\":{id},\"chunk_hash\":\"{hash}\"}}}}\n")
        };
        let build = |metadata: String, records: String| {
            let mut out = Vec::new();
            ledger_from_artifact(Cursor::new(metadata), Cursor::new(records), &mut out)
        };

        // Right ids, wrong text: the digest does not begin with the stored chunk_hash.
        match build(stored(10, &b[..32]), record(10, &a)) {
            Err(PackError::LineTextMismatch { line_id, .. }) => assert_eq!(line_id, 10),
            other => panic!("expected a text mismatch, got {other:?}"),
        }

        // Two records for one line — a `HashMap` took the last one silently.
        match build(stored(10, &a[..32]), record(10, &a) + &record(10, &b)) {
            Err(PackError::DuplicateLineId { line_id }) => assert_eq!(line_id, 10),
            other => panic!("expected a duplicate, got {other:?}"),
        }

        // A records file longer than the payload belongs to a different build.
        match build(stored(10, &a[..32]), record(10, &a) + &record(11, &b)) {
            Err(PackError::MalformedInput { reason }) => {
                assert!(reason.contains("11"), "{reason}");
            }
            other => panic!("expected leftover records to be refused, got {other:?}"),
        }

        // An artifact line no record describes.
        match build(stored(10, &a[..32]) + &stored(11, &b[..32]), record(10, &a)) {
            Err(PackError::LineNotInCorpus { line_id }) => assert_eq!(line_id, 11),
            other => panic!("expected a missing record, got {other:?}"),
        }

        // A digest that is not 64 lowercase hex is not a digest.
        match build(stored(10, &a[..32]), record(10, &"A".repeat(64))) {
            Err(PackError::MalformedInput { reason }) => {
                assert!(reason.contains("lowercase hex"), "{reason}");
            }
            other => panic!("expected a malformed digest to be refused, got {other:?}"),
        }

        // And the honest pair still builds.
        assert_eq!(build(stored(10, &a[..32]), record(10, &a)).unwrap(), 1);
    }

    /// An empty or short `chunk_hash` compared equal under `min(a.len(), b.len())`, which
    /// is the whole prefix check quietly not happening.
    #[test]
    fn a_chunk_hash_that_is_not_32_lowercase_hex_is_refused() {
        let a = "a".repeat(64);
        let record = format!(
            "{{\"line_id\":10,\"source_line_sha256\":\"s\",\"embedding_text_sha256\":\"{a}\"}}\n"
        );
        for stored in ["", "a", &a[..31], &a[..33], &"A".repeat(32)] {
            let metadata =
                format!("{{\"metadata\":{{\"line_id\":10,\"chunk_hash\":\"{stored}\"}}}}\n");
            let mut out = Vec::new();
            match ledger_from_artifact(Cursor::new(metadata), Cursor::new(record.clone()), &mut out)
            {
                Err(PackError::MalformedInput { reason }) => {
                    assert!(reason.contains("32 lowercase hex"), "{stored:?}: {reason}");
                }
                other => panic!("{stored:?} must be refused, got {other:?}"),
            }
        }
        // And exactly 32 lowercase hex, matching, still passes.
        let metadata = format!(
            "{{\"metadata\":{{\"line_id\":10,\"chunk_hash\":\"{}\"}}}}\n",
            &a[..32]
        );
        let mut out = Vec::new();
        assert_eq!(
            ledger_from_artifact(Cursor::new(metadata), Cursor::new(record), &mut out).unwrap(),
            1
        );
    }

    /// A ledger from another model reuses just as cleanly as one from this model, and
    /// nothing downstream can tell: the digests match and the counts match.
    #[test]
    fn a_ledger_from_a_different_vector_space_is_refused() {
        let chunking = ChunkerConfig::default();
        let mine = model_for(&"ab".repeat(32), &chunking);
        let manifest = LedgerManifest {
            artifact_digest: "d".repeat(64),
            vectors_sha256: "e".repeat(64),
            ledger_sha256: "f".repeat(64),
            vector_count: 2,
            embedding_dim: DIM_U32,
            model: ModelIdentity {
                // Same texts, same digests, different weights.
                model_checksum: "cd".repeat(32),
                ..mine.clone()
            },
        };
        match manifest.ensure_matches(&mine, DIM_U32) {
            Err(PackError::LedgerDisagreesWithBuild { field, .. }) => {
                assert_eq!(field, "model_checksum");
            }
            other => panic!("expected a refusal, got {other:?}"),
        }

        let pooling_differs = LedgerManifest {
            model: ModelIdentity {
                pooling: "mean".to_string(),
                ..mine.clone()
            },
            ..manifest.clone()
        };
        match pooling_differs.ensure_matches(&mine, DIM_U32) {
            Err(PackError::LedgerDisagreesWithBuild { field, .. }) => {
                assert_eq!(
                    field, "pooling",
                    "a different token pooled is a different vector"
                );
            }
            other => panic!("expected a refusal, got {other:?}"),
        }

        let same = LedgerManifest {
            model: mine.clone(),
            ..manifest
        };
        assert!(same.ensure_matches(&mine, DIM_U32).is_ok());
    }

    /// A shard whose two files disagree on length would shift every pairing after it, and
    /// nothing downstream could see it: the ids would all be present and the count would
    /// be right.
    #[test]
    fn a_shard_with_more_vectors_than_records_is_refused() {
        let records =
            "{\"line_id\":1,\"source_line_sha256\":\"s\",\"embedding_text_sha256\":\"d\"}\n";
        let shards: Vec<ShardStreams> = vec![(
            Box::new(Cursor::new(vectors_of(&[[1.0, 1.0], [2.0, 2.0]]))),
            Box::new(Cursor::new(records)),
        )];
        let (mut vectors, mut out_records) = (Vec::new(), Vec::new());

        let outcome = assemble(
            Vec::new(),
            None,
            shards,
            DIM,
            &mut vectors,
            &mut out_records,
        );
        match outcome {
            Err(PackError::MalformedInput { reason }) => {
                assert!(
                    reason.contains("more vector bytes than records"),
                    "{reason}"
                );
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
    }
}
