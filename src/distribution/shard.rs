//! Splitting one build across machines that never meet.
//!
//! [`build`](super::builder::build) does the whole job in one process: read the corpus,
//! apply the recipe, embed, write the segment. That is the right shape when one machine has both the
//! corpus and the arithmetic, and the wrong one for the library, where the corpus lives on
//! a build machine and the inference is spread over rented machines that see it for a few
//! hours and are then destroyed.
//!
//! So the same work is cut in two, at the one seam that does not weaken anything:
//!
//! ```text
//! export_plan    corpus + recipe -> plan.jsonl          build machine, no model
//! embed_shard    a slice of plan  -> vectors + records  worker, no corpus
//! verify_shards  every shard      -> checked streams    build machine
//! ```
//!
//! What the build machine makes of the checked streams — vectors keyed by the text they
//! were built from, written as a segment — is the assembler's business, and
//! [`read_vector_inputs`] is how it reads them.
//!
//! **The recipe is applied exactly once, here.** A worker is handed finished strings, not
//! a corpus and a configuration, which is what makes it impossible for two workers on
//! different hardware to disagree about neighbour context, truncation or normalization —
//! and it is why a shard boundary cannot cut a context window: the windows were already
//! resolved when the plan was written.
//!
//! **A shard is a range of records, not a range of ids.** A vector is addressed by the
//! text it was built from, never by where it was written, so shards merge by
//! concatenation in any order. Sharding by id would demand
//! the plan be written in ascending id order, which would demand it be sorted, which would
//! demand the whole thing in memory — 5.9 million passages of text — to buy nothing.
//!
//! **Where a shard boundary falls changes no vector.** The ONNX backend runs one text per
//! session run, so a vector depends on its text alone — measured bit-identical batched and
//! one at a time, at 1 to 8 threads and 1 or 2 sessions (`docs/ONNX_BACKEND.md` §5, §7). A
//! window may start at any record and hold any number of them, and the vectors of a plan
//! embedded in shards are the ones a single window, or [`build`](super::builder::build),
//! would have produced from the same corpus — provided every shard runs the same graph on
//! the same kind of CPU, since an int8 vector depends on the CPU's int8 kernels
//! (`docs/ONNX_BACKEND.md` §0).
//!
//! **What each side can check, it checks.** The worker cannot recompute
//! `source_line_sha256`; it has no corpus, and that digest is the build machine's business
//! at merge time. It *can* recompute `embedding_text_sha256`, so it does, on every record —
//! see [`PackError::PlanTextChanged`]. What neither side can check alone is checked at the
//! merge, against the corpus.

use crate::distribution::builder::{chunks_for_book, ensure_recipe_matches};
use crate::distribution::corpus::CorpusBooks;
use crate::errors::PackError;
use crate::semantic::chunker::{Chunker, ChunkerConfig};
use crate::semantic::embedding::EmbeddingRuntime;
use crate::semantic::versioning::ModelIdentity;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// One ready-made vector and the line it belongs to.
#[derive(Debug, Clone)]
pub struct VectorInput {
    /// The global document id the vector describes, in the corpus's own id scheme.
    pub line_id: u64,
    /// SHA-256 of the **corpus line's** text as the producer read it, in 64 lowercase hex
    /// digits; the build machine's to check against the corpus at the merge.
    pub source_line_sha256: String,
    /// SHA-256 of the text that was actually **embedded** — after whatever prefixing,
    /// neighbour context and truncation the recipe applies. Its first 16 bytes are the
    /// vector's [`ChunkKey`](crate::semantic::chunk_key::ChunkKey), and it equals the
    /// corpus line's digest only when the recipe embedded the line unchanged.
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

/// One line's work, as the build machine hands it to the machine that embeds it.
///
/// `embedding_text` is the string the backend will be given, complete: prefixed,
/// context-borrowed and truncated already. Nothing downstream may re-derive it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannedChunk {
    pub line_id: u64,
    /// SHA-256 of the corpus line the vector will be attributed to. Carried rather than
    /// computed because the worker has no corpus, and checked by `pack` against a fresh
    /// read of one.
    pub source_line_sha256: String,
    /// SHA-256 of `embedding_text`. Redundant on a healthy file and the whole point on a
    /// damaged one — the worker recomputes it before embedding.
    pub embedding_text_sha256: String,
    pub embedding_text: String,
}

/// What an export produced, for the manifest that travels with it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanReport {
    pub records: usize,
    pub min_line_id: u64,
    pub max_line_id: u64,
    /// SHA-256 of the bytes written, so a worker can prove it received the file the build
    /// machine sent before it spends an hour on it.
    pub plan_sha256: String,
    pub chunking_identity: u64,
}

/// What one worker produced, for the merge that has to prove nothing was lost.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardReport {
    /// The `plan_sha256` of the file this window was taken from, so the merge can refuse a
    /// shard that belongs to another export. Two exports of a changed corpus produce
    /// windows of the same shape over different text.
    pub plan_sha256: String,
    /// The window, as it was asked for. `records` is what was actually written, and the
    /// two differ legitimately only for the last shard of a plan.
    pub skip: usize,
    pub take: usize,
    pub records: usize,
    pub embedding_dim: usize,
    pub vectors_sha256: String,
    pub records_sha256: String,
    /// What produced the floats. A shard embedded by another model is arithmetically
    /// unrelated to its neighbours, and nothing about the vectors says so.
    pub model: ModelIdentity,
}

/// Apply the recipe to the whole corpus and write one JSON object per line that gets a
/// vector.
///
/// The output is the same set [`BuildPlan`](super::builder::BuildPlan) computes and the
/// same text [`build`](super::builder::build) would have embedded, so an artifact merged
/// from shards is the artifact a single-process build would have produced — modulo the
/// arithmetic of whichever backend each worker ran, which the manifest already governs.
///
/// Streams: one book is resolved at a time and written out before the next is read, so
/// the peak cost is the largest book rather than the library.
///
/// # Errors
///
/// [`PackError::RecipeMismatch`] before anything is read, [`PackError::DuplicateLineId`]
/// for a corpus where two books claim one line, [`PackError::NothingToEmbed`] for a corpus
/// the recipe empties, and [`PackError::Corpus`] for a sink that will not accept bytes.
pub fn export_plan(
    corpus: &(dyn CorpusBooks + Sync),
    chunking: &ChunkerConfig,
    model: &ModelIdentity,
    sink: &mut dyn Write,
) -> Result<PlanReport, PackError> {
    let chunking_identity = ensure_recipe_matches(chunking, model)?;
    let books = corpus.book_keys()?;

    // Hashing the bytes as they are written, rather than reading the file back: the digest
    // then describes what was actually sent, including a partial write that a later read
    // would not have distinguished from a short plan.
    let mut hasher = Sha256::new();
    let mut records = 0usize;
    let (mut min_line_id, mut max_line_id) = (u64::MAX, 0u64);
    // Ids only. Holding the text as well would be a second copy of the corpus in memory,
    // which is the cost this whole module exists to avoid.
    let mut seen = std::collections::BTreeSet::new();

    // Books are resolved in parallel and written in order. Applying the recipe to a book
    // touches nothing outside it, so the only sequential parts left are the ones whose
    // *value* depends on order — the digest of the file, and the file itself. A window at
    // a time rather than the whole library at once, so the peak cost is a few dozen books
    // of serialized text rather than six million lines of it.
    let width = std::thread::available_parallelism().map_or(4, |n| n.get());
    let window = width.saturating_mul(4).max(1);

    for batch in books.chunks(window) {
        let resolved = resolve_books(corpus, chunking, batch, width)?;
        for (line_ids, body) in resolved {
            for line_id in line_ids {
                // Two books claiming one line, or one book listing it twice. Embedded, it
                // would be two records of one line; saying so here names the recipe's input
                // instead of the vector stream, which is where the fault is.
                if !seen.insert(line_id) {
                    return Err(PackError::DuplicateLineId { line_id });
                }
                records += 1;
                min_line_id = min_line_id.min(line_id);
                max_line_id = max_line_id.max(line_id);
            }
            hasher.update(&body);
            write(sink, &body)?;
        }
    }

    if records == 0 {
        return Err(PackError::NothingToEmbed { books: books.len() });
    }
    flush(sink)?;

    Ok(PlanReport {
        records,
        min_line_id,
        max_line_id,
        plan_sha256: format!("{:x}", hasher.finalize()),
        chunking_identity,
    })
}

/// One book's contribution to the plan: the ids it produced, and its records already
/// serialized. Held as bytes rather than as values because the only thing left to do
/// with them is hash them and write them, both in order.
type ResolvedBook = (Vec<u64>, Vec<u8>);

/// Apply the recipe to one window of books, `width` at a time, and return each book's ids
/// and its serialized records **in the order the window gave them**.
///
/// `std::thread::scope` rather than a work-stealing pool: this crate is compiled into the
/// application, and a build-side pass is not a reason to put a thread pool in a phone.
fn resolve_books(
    corpus: &(dyn CorpusBooks + Sync),
    chunking: &ChunkerConfig,
    batch: &[String],
    width: usize,
) -> Result<Vec<ResolvedBook>, PackError> {
    let mut slots: Vec<Option<Result<ResolvedBook, PackError>>> =
        (0..batch.len()).map(|_| None).collect();

    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for (chunk_of_books, chunk_of_slots) in batch
            .chunks(batch.len().div_ceil(width.max(1)).max(1))
            .zip(slots.chunks_mut(batch.len().div_ceil(width.max(1)).max(1)))
        {
            handles.push(scope.spawn(move || {
                for (book_key, slot) in chunk_of_books.iter().zip(chunk_of_slots.iter_mut()) {
                    *slot = Some(resolve_book(corpus, chunking, book_key));
                }
            }));
        }
        for handle in handles {
            // A panicking worker is not a build error; it is a bug, and it must not be
            // reported as a corpus that could not be read.
            handle.join().expect("a book resolver must not panic");
        }
    });

    slots
        .into_iter()
        .map(|slot| slot.expect("every book was assigned to a worker"))
        .collect()
}

fn resolve_book(
    corpus: &dyn CorpusBooks,
    chunking: &ChunkerConfig,
    book_key: &str,
) -> Result<ResolvedBook, PackError> {
    // One per worker: resolving a recipe is a few enum lookups, and sharing one would
    // mean sharing it across threads for no gain.
    let chunker = Chunker::new(chunking.clone())?;
    let mut line_ids = Vec::new();
    let mut body = Vec::new();
    for chunk in chunks_for_book(corpus, &chunker, book_key)? {
        let planned = PlannedChunk {
            line_id: chunk.line_id,
            source_line_sha256: sha256_hex(chunk.anchor_text.as_bytes()),
            embedding_text_sha256: sha256_hex(chunk.embedding_text.as_bytes()),
            embedding_text: chunk.embedding_text,
        };
        serde_json::to_writer(&mut body, &planned).map_err(|error| PackError::Corpus {
            reason: format!(
                "line {} could not be written to the plan: {error}",
                planned.line_id
            ),
        })?;
        body.push(b'\n');
        line_ids.push(planned.line_id);
    }
    Ok((line_ids, body))
}

/// Read a plan back, one record per line, skipping `skip` and yielding at most `take`.
///
/// The window is applied to *records*, and a worker is told only its own window — which is
/// what makes two workers' outputs disjoint without either knowing the other exists.
pub fn read_plan(
    reader: impl BufRead,
    skip: usize,
    take: usize,
) -> impl Iterator<Item = Result<PlannedChunk, PackError>> {
    reader
        .lines()
        .skip(skip)
        .take(take)
        .filter(|line| !matches!(line, Ok(text) if text.trim().is_empty()))
        .map(|line| {
            let line = line.map_err(|error| PackError::Corpus {
                reason: format!("the plan could not be read: {error}"),
            })?;
            serde_json::from_str::<PlannedChunk>(&line).map_err(|error| PackError::Corpus {
                reason: format!("a plan record is malformed: {error}"),
            })
        })
}

/// Embed a slice of the plan, writing raw vectors and the records that pair with them.
///
/// The two files are written in lockstep, one record for one vector, in the order the plan
/// gave them. That pairing is the only thing holding a vector to an id here, which is why
/// the digest check below is not optional and why the merge still verifies both files
/// against the corpus.
///
/// # Errors
///
/// [`PackError::PlanTextChanged`] for a record whose text is not what its digest names,
/// [`PackError::Embedding`] for a backend failure, [`PackError::NoVectors`] for an empty
/// window, and [`PackError::Corpus`] for a sink that will not accept bytes.
pub fn embed_shard(
    plan: impl Iterator<Item = Result<PlannedChunk, PackError>>,
    window: (String, usize, usize),
    model: &ModelIdentity,
    runtime: &EmbeddingRuntime,
    batch_size: usize,
    vectors: &mut dyn Write,
    records: &mut dyn Write,
) -> Result<ShardReport, PackError> {
    let (plan_sha256, skip, take) = window;
    let mut vector_hasher = Sha256::new();
    let mut record_hasher = Sha256::new();
    let mut written = 0usize;
    let mut dim = 0usize;
    // A batch of zero would embed nothing forever.
    let batch_size = batch_size.max(1);
    let mut batch: Vec<PlannedChunk> = Vec::with_capacity(batch_size);

    for chunk in plan.chain(std::iter::once_with(|| Err(PackError::NoVectors))) {
        // The sentinel above is the flush: it turns "the plan ended" into one more trip
        // through the loop body, so a partial final batch cannot be dropped by an early
        // return the way a `for` loop followed by a tail flush invites.
        let last = match chunk {
            Ok(chunk) => {
                verify_text(&chunk)?;
                batch.push(chunk);
                batch.len() < batch_size
            }
            Err(PackError::NoVectors) => false,
            Err(error) => return Err(error),
        };
        if last || batch.is_empty() {
            continue;
        }

        let texts: Vec<&str> = batch
            .iter()
            .map(|chunk| chunk.embedding_text.as_str())
            .collect();
        let embedded = runtime.embed_batch(&texts)?;
        if embedded.len() != batch.len() {
            return Err(PackError::MalformedInput {
                reason: format!(
                    "{} vector(s) came back for {} text(s)",
                    embedded.len(),
                    batch.len()
                ),
            });
        }

        for (chunk, vector) in batch.drain(..).zip(embedded) {
            dim = vector.len();
            let mut bytes = Vec::with_capacity(vector.len() * 4);
            for value in &vector {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            vector_hasher.update(&bytes);
            write(vectors, &bytes)?;

            let mut record = serde_json::to_vec(&VectorInputRecord {
                line_id: chunk.line_id,
                source_line_sha256: chunk.source_line_sha256,
                embedding_text_sha256: chunk.embedding_text_sha256,
            })
            .map_err(|error| PackError::Corpus {
                reason: format!("a shard record could not be written: {error}"),
            })?;
            record.push(b'\n');
            record_hasher.update(&record);
            write(records, &record)?;
            written += 1;
        }
    }

    if written == 0 {
        return Err(PackError::NoVectors);
    }
    flush(vectors)?;
    flush(records)?;

    Ok(ShardReport {
        plan_sha256,
        skip,
        take,
        records: written,
        embedding_dim: dim,
        vectors_sha256: format!("{:x}", vector_hasher.finalize()),
        records_sha256: format!("{:x}", record_hasher.finalize()),
        model: model.clone(),
    })
}

/// Pair a plan record with a vector, for a caller that embeds without writing files.
///
/// Exists so the in-process path and the sharded path build a [`VectorInput`] the same
/// way rather than twice.
pub fn vector_input_for(chunk: PlannedChunk, vector: Vec<f32>) -> VectorInput {
    VectorInput {
        line_id: chunk.line_id,
        source_line_sha256: chunk.source_line_sha256,
        embedding_text_sha256: chunk.embedding_text_sha256,
        vector,
    }
}

fn verify_text(chunk: &PlannedChunk) -> Result<(), PackError> {
    let actual = sha256_hex(chunk.embedding_text.as_bytes());
    if actual != chunk.embedding_text_sha256 {
        return Err(PackError::PlanTextChanged {
            line_id: chunk.line_id,
            declared: chunk.embedding_text_sha256.clone(),
            actual,
        });
    }
    Ok(())
}

fn flush(sink: &mut dyn Write) -> Result<(), PackError> {
    sink.flush().map_err(|error| PackError::Corpus {
        reason: format!("a shard file could not be flushed: {error}"),
    })
}

fn write(sink: &mut dyn Write, bytes: &[u8]) -> Result<(), PackError> {
    sink.write_all(bytes).map_err(|error| PackError::Corpus {
        reason: format!("the output could not be written: {error}"),
    })
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
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
            .as_chunks::<4>()
            .0
            .iter()
            .map(|value| f32::from_le_bytes(*value))
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

/// One shard's vectors and the records that pair with them, already opened.
pub type ShardStreams = (Box<dyn Read>, Box<dyn BufRead>);

/// Every shard's manifest, checked against the plan they claim to cover.
///
/// A merge that opened `vectors.f32` and `records.jsonl` directly, without reading
/// `shard-manifest.json`, once believed counts and digests nobody checked: a shard from
/// another export, a corrupted vector that stayed finite, or two shards covering one
/// window and none covering another all merged cleanly.
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
/// here rather than leaving it to the merge: a merge writes as it reads, so a shard it
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
    // A release where no embedding text changed needs no inference at all: every vector is
    // already held and there are no shard directories. That is the cheapest path there
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
        // model promises. A merge strides through both files by this number, so a shard
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
        // — and a shard that stopped early is a truncated session, not a short window.
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
        // The count a merge will actually pair with vectors, from the pass that hashed
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
        // length would otherwise surface as a merge running out of vectors halfway.
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

/// The digest and the number of non-empty lines, from one pass over the bytes.
///
/// Both facts come from the same read because the second one is not optional: the digest
/// says the file is the file its manifest describes, and the count says how many vectors
/// a merge will pair with it. Hashing without counting left that to be discovered
/// mid-merge, with output already written.
///
/// "Non-empty" is the same test [`read_vector_inputs`] applies when it walks the records —
/// a line of nothing but whitespace is skipped there and not counted here.
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

fn read_error(error: std::io::Error) -> PackError {
    PackError::Corpus {
        reason: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::builder::{build, BuildRequest, SEGMENT_FILENAME};
    use crate::distribution::corpus::CorpusIdentity;
    use crate::distribution::corpus::{CorpusLine, CorpusLineRecord, JsonlCorpus};
    use crate::semantic::backend::Pooling;
    use crate::semantic::chunk_key::ChunkKey;
    use crate::semantic::embedding::{mock, EmbeddingConfig};
    use crate::semantic::model_package::validate_model;
    use crate::semantic::oxv::reader::Segment;
    use crate::semantic::versioning::ModelPackage;
    use std::path::PathBuf;

    const DIM: u32 = 64;
    const GENESIS: &str = "otzaria/tanach/genesis.txt";
    const BERACHOT: &str = "otzaria/mishna/berachot.txt";
    /// Long enough to stand alone under the default recipe.
    const LONG: &str = "בראשית ברא אלהים את השמים ואת הארץ";
    const LONG_TWO: &str = "והארץ היתה תהו ובהו וחשך על פני תהום רבה";
    const LONG_THREE: &str = "ויאמר אלהים יהי אור ויהי אור וירא אלהים";
    /// Under `min_meaningful_chars`: embedded with its neighbours' text folded in, which
    /// is what makes a shard boundary interesting.
    const BORROWS: &str = "ויהי ערב";
    /// Under `min_embeddable_chars`: never embedded, in either path.
    const SKIPPED: &str = "או";

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(name: &str) -> Self {
            // The clock alone collided: macOS ticks coarser than a test takes to start.
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "otzaria_shard_{name}_{}_{}",
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

    fn corpus_identity() -> CorpusIdentity {
        CorpusIdentity {
            text: crate::semantic::versioning::TextIdentity::with_line_text_version(1),
            library_version: 30,
            library_release_tag: "v30-20260930120000".to_string(),
            document_id_scheme_version: 1,
        }
    }

    fn corpus_line(book: &str, text: &str) -> CorpusLine {
        CorpusLine {
            source_book_key: book.to_string(),
            title: "ספר".to_string(),
            reference: format!("{book} :: {}", text.chars().take(6).collect::<String>()),
            section_id: 1,
            segment: 0,
            is_pdf: false,
            line_hash: 0,
            content_hash: 7,
            facets: vec!["/מקרא".to_string()],
            text: text.to_string(),
        }
    }

    /// Two books, six lines, one of them skipped by the recipe and one of them borrowing
    /// from its neighbours — so a plan that ignored the recipe would not match.
    fn corpus(dir: &TempDir) -> JsonlCorpus {
        let lines: [(u64, &str, &str); 6] = [
            (4_294_967_297, GENESIS, LONG),
            (4_294_967_298, GENESIS, BORROWS),
            (4_294_967_299, GENESIS, LONG_TWO),
            (4_294_967_300, GENESIS, SKIPPED),
            (8_589_934_593, BERACHOT, LONG_THREE),
            (8_589_934_594, BERACHOT, LONG),
        ];
        let identity_path = dir.0.join("corpus-identity.json");
        let lines_path = dir.0.join("corpus-lines.jsonl");
        std::fs::write(
            &identity_path,
            serde_json::to_vec_pretty(&corpus_identity()).unwrap(),
        )
        .unwrap();
        let body: String = lines
            .iter()
            .map(|(line_id, book, text)| {
                format!(
                    "{}\n",
                    serde_json::to_string(&CorpusLineRecord {
                        line_id: *line_id,
                        line: corpus_line(book, text),
                    })
                    .unwrap()
                )
            })
            .collect();
        std::fs::write(&lines_path, body).unwrap();
        JsonlCorpus::load(&identity_path, &lines_path).unwrap()
    }

    fn model_for(checksum: &str, chunking: &ChunkerConfig) -> ModelIdentity {
        ModelIdentity {
            family_id: "otzaria-embedding-v1".to_string(),
            tokenizer_checksum: crate::semantic::embedding::mock::stub_tokenizer_checksum(),
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

    fn runtime_for(model_path: &std::path::Path) -> EmbeddingRuntime {
        let mut runtime = EmbeddingRuntime::new(EmbeddingConfig {
            model_path: model_path.to_path_buf(),
            embedding_dim: DIM,
            max_tokens: 512,
            batch_size: 2,
            pooling: Pooling::InGraph,
        });
        runtime.load().unwrap();
        runtime
    }

    /// The whole claim of this module, as one comparison.
    ///
    /// If a vector embedded in a shard is not the vector one process would have stored for
    /// the same text, then the split changed the product — and every argument about which
    /// machine runs which step is worthless. Each sharded vector, encoded with the codec
    /// the one-process build calibrated, is compared byte for byte with the slot the build
    /// stored for its key.
    #[test]
    fn a_sharded_embedding_produces_the_vectors_one_process_stores() {
        let dir = TempDir::new("equivalence");
        let corpus = corpus(&dir);
        let chunking = ChunkerConfig::default();
        let model_path = mock::write_stub_onnx_package(&dir.0.join("model"));
        let model = model_for(validate_model(&model_path).unwrap().checksum(), &chunking);

        let whole = build(
            BuildRequest {
                output_path: dir.0.join("whole"),
                model_path: model_path.clone(),
                model: model.clone(),
                chunking: chunking.clone(),
                created_at: "2026-08-09T00:00:00Z".to_string(),
                batch_size: 2,
                clip_q: 1.0,
                allow_non_semantic_backend: true,
            },
            &corpus,
        )
        .unwrap();

        let plan_path = dir.0.join("plan.jsonl");
        let mut sink = std::fs::File::create(&plan_path).unwrap();
        let plan = export_plan(&corpus, &chunking, &model, &mut sink).unwrap();
        assert_eq!(
            plan.records, whole.planned_lines,
            "the plan and the build must agree on which lines get a vector"
        );

        // Three windows over five records, the last one short: a shard count that does not
        // divide the plan is the ordinary case, not the exceptional one.
        let runtime = runtime_for(&model_path);
        let vectors_path = dir.0.join("vectors.f32");
        let records_path = dir.0.join("records.jsonl");
        let mut vectors = std::fs::File::create(&vectors_path).unwrap();
        let mut records = std::fs::File::create(&records_path).unwrap();
        let mut total = 0;
        for skip in (0..plan.records).step_by(2) {
            let file = std::io::BufReader::new(std::fs::File::open(&plan_path).unwrap());
            let report = embed_shard(
                read_plan(file, skip, 2),
                ("plan".to_string(), skip, 2),
                &model,
                &runtime,
                2,
                &mut vectors,
                &mut records,
            )
            .unwrap();
            total += report.records;
        }
        assert_eq!(
            total, plan.records,
            "every planned line belongs to one shard"
        );
        drop((vectors, records));

        let segment = Segment::open(&whole.output_path.join(SEGMENT_FILENAME)).unwrap();
        let slots: std::collections::HashMap<ChunkKey, u32> = (0..segment.slot_count())
            .map(|slot| (segment.key(slot), slot))
            .collect();
        let mut encoded = vec![0u8; segment.codec().bytes_per_vector()];
        let mut compared = 0;
        for input in read_vector_inputs(&vectors_path, &records_path, DIM).unwrap() {
            let input = input.unwrap();
            let key = ChunkKey::from_hex(&input.embedding_text_sha256[..32]).unwrap();
            let slot = slots[&key];
            segment.codec().encode(&input.vector, &mut encoded);
            assert_eq!(
                encoded,
                segment.vector(slot),
                "line {}: a sharded vector must be the one a single-process build stores",
                input.line_id
            );
            compared += 1;
        }
        assert_eq!(compared, plan.records);
        assert_eq!(
            slots.len(),
            plan.records - 1,
            "the text two books share is one slot"
        );
    }

    /// Text recipe 2 is applied where every recipe is, in the plan: the worker is handed
    /// the passage already marked, and its digest covers the mark. The corpus line's
    /// digest does not — it describes the line, which the prefix does not change — and
    /// the build packs and validates exactly as under version 1.
    #[test]
    fn a_version_two_plan_carries_the_passage_prefixed_text_and_its_digest() {
        let dir = TempDir::new("plan_v2");
        let corpus = corpus(&dir);
        let model_path = mock::write_stub_onnx_package(&dir.0.join("model"));
        let checksum = validate_model(&model_path).unwrap().checksum().to_string();

        let plan_under = |chunking: &ChunkerConfig, model: &ModelIdentity| {
            let mut sink = Vec::new();
            export_plan(&corpus, chunking, model, &mut sink).unwrap();
            read_plan(std::io::Cursor::new(sink), 0, usize::MAX)
                .collect::<Result<Vec<_>, _>>()
                .unwrap()
        };
        let v1_chunking = ChunkerConfig::default();
        let v2_chunking = ChunkerConfig {
            embedding_text_version: 2,
            ..ChunkerConfig::default()
        };
        let v2_model = ModelIdentity {
            embedding_text_version: 2,
            ..model_for(&checksum, &v2_chunking)
        };
        let v1 = plan_under(&v1_chunking, &model_for(&checksum, &v1_chunking));
        let v2 = plan_under(&v2_chunking, &v2_model);

        assert_eq!(v1.len(), v2.len(), "the prefix decides no line's fate");
        for (old, new) in v1.iter().zip(&v2) {
            assert_eq!(new.line_id, old.line_id);
            assert_eq!(
                new.embedding_text,
                format!("[PASSAGE] {}", old.embedding_text.trim())
            );
            assert_eq!(new.embedding_text.matches("[PASSAGE]").count(), 1);
            assert_eq!(
                new.embedding_text_sha256,
                sha256_hex(new.embedding_text.as_bytes())
            );
            assert_eq!(
                new.source_line_sha256, old.source_line_sha256,
                "the corpus line is the same line"
            );
        }

        // A worker embeds it unchanged — its digest check passes on the prefixed text —
        // and one process builds the same plan.
        let runtime = runtime_for(&model_path);
        let (mut vectors, mut records) = (Vec::new(), Vec::new());
        let report = embed_shard(
            v2.clone().into_iter().map(Ok),
            ("plan".to_string(), 0, v2.len()),
            &v2_model,
            &runtime,
            2,
            &mut vectors,
            &mut records,
        )
        .unwrap();
        assert_eq!(report.records, v2.len());

        let built = build(
            BuildRequest {
                output_path: dir.0.join("v2-artifact"),
                model_path: model_path.clone(),
                model: v2_model.clone(),
                chunking: v2_chunking.clone(),
                created_at: "2026-08-09T00:00:00Z".to_string(),
                batch_size: 2,
                clip_q: 1.0,
                allow_non_semantic_backend: true,
            },
            &corpus,
        )
        .unwrap();
        assert_eq!(built.planned_lines, v2.len());
    }

    /// A cap that ends a passage on a space: the plan hands the worker version 2's text
    /// without it — the text the model is given — and the digest the worker checks is
    /// that text's. Version 1's plan keeps the space, byte for byte as it always has.
    #[test]
    fn a_version_two_plan_carries_a_capped_passage_without_its_trailing_space() {
        let dir = TempDir::new("plan_v2_trim");
        let corpus = corpus(&dir);
        let model_path = mock::write_stub_onnx_package(&dir.0.join("model"));
        let checksum = validate_model(&model_path).unwrap().checksum().to_string();
        // "בראשית ברא …" capped at 7 characters is "בראשית ", space included.
        let chunking = |embedding_text_version| ChunkerConfig {
            max_chunk_chars: 7,
            embedding_text_version,
            ..ChunkerConfig::default()
        };
        let plan_under = |version: u32| {
            let chunking = chunking(version);
            let model = ModelIdentity {
                embedding_text_version: version,
                ..model_for(&checksum, &chunking)
            };
            let mut sink = Vec::new();
            export_plan(&corpus, &chunking, &model, &mut sink).unwrap();
            read_plan(std::io::Cursor::new(sink), 0, usize::MAX)
                .collect::<Result<Vec<_>, _>>()
                .unwrap()
        };
        let (v1, v2) = (plan_under(1), plan_under(2));

        let capped: Vec<_> = v1
            .iter()
            .zip(&v2)
            .filter(|(old, _)| old.embedding_text.ends_with(' '))
            .collect();
        assert!(
            !capped.is_empty(),
            "the cap must end some passage on a space"
        );
        for (old, new) in capped {
            assert_eq!(
                new.embedding_text,
                format!("[PASSAGE] {}", old.embedding_text.trim_end())
            );
            assert_eq!(
                new.embedding_text_sha256,
                sha256_hex(new.embedding_text.as_bytes())
            );
            assert_eq!(
                old.embedding_text_sha256,
                sha256_hex(old.embedding_text.as_bytes())
            );
        }
    }

    /// A shard that never ran is a hole, and the merge's check of the shards is what has
    /// to see it — a set of shards short of the plan is internally perfect.
    #[test]
    fn a_missing_shard_is_refused_at_the_merge() {
        let dir = TempDir::new("hole");
        let corpus = corpus(&dir);
        let chunking = ChunkerConfig::default();
        let model_path = mock::write_stub_onnx_package(&dir.0.join("model"));
        let model = model_for(validate_model(&model_path).unwrap().checksum(), &chunking);

        let plan_path = dir.0.join("plan.jsonl");
        let mut sink = std::fs::File::create(&plan_path).unwrap();
        let plan = export_plan(&corpus, &chunking, &model, &mut sink).unwrap();
        drop(sink);

        // Every record but the last one, in one shard directory.
        let runtime = runtime_for(&model_path);
        let shard = dir.0.join("shard-0");
        std::fs::create_dir_all(&shard).unwrap();
        let mut vectors = std::fs::File::create(shard.join("vectors.f32")).unwrap();
        let mut records = std::fs::File::create(shard.join("records.jsonl")).unwrap();
        let file = std::io::BufReader::new(std::fs::File::open(&plan_path).unwrap());
        let report = embed_shard(
            read_plan(file, 0, plan.records - 1),
            ("plan".to_string(), 0, plan.records - 1),
            &model,
            &runtime,
            2,
            &mut vectors,
            &mut records,
        )
        .unwrap();
        drop((vectors, records));

        let shards = [(shard, report)];
        assert!(verify_shards(&shards, "plan", &model, plan.records - 1).is_ok());
        match verify_shards(&shards, "plan", &model, plan.records) {
            Err(PackError::MalformedInput { reason }) => {
                assert!(reason.contains("cover 4 record(s)"), "{reason}")
            }
            Ok(_) => panic!("a merge missing a shard must be refused"),
            Err(other) => panic!("expected a refusal naming the coverage, got {other:?}"),
        }
    }

    /// The one check a worker can make on its own, and the reason the digest travels.
    #[test]
    fn a_plan_whose_text_no_longer_matches_its_digest_is_refused_before_it_is_embedded() {
        let dir = TempDir::new("tampered");
        let model_path = mock::write_stub_onnx_package(&dir.0.join("model"));
        let runtime = runtime_for(&model_path);

        let honest = PlannedChunk {
            line_id: 4_294_967_297,
            source_line_sha256: sha256_hex(LONG.as_bytes()),
            embedding_text_sha256: sha256_hex(LONG.as_bytes()),
            embedding_text: LONG.to_string(),
        };
        let tampered = PlannedChunk {
            embedding_text: LONG_TWO.to_string(),
            ..honest.clone()
        };

        let mut vectors = Vec::new();
        let mut records = Vec::new();
        let outcome = embed_shard(
            [Ok(honest), Ok(tampered)].into_iter(),
            ("plan".to_string(), 0, 2),
            &model_for(&"ab".repeat(32), &ChunkerConfig::default()),
            &runtime,
            2,
            &mut vectors,
            &mut records,
        );
        match outcome {
            Err(PackError::PlanTextChanged { line_id, .. }) => {
                assert_eq!(line_id, 4_294_967_297);
            }
            Ok(_) => panic!("text that is not what its digest names must not be embedded"),
            Err(other) => panic!("expected PlanTextChanged, got {other:?}"),
        }
    }

    /// The two files a shard is, read back — and the checks a merge makes on a set of them.
    mod streams {
        use super::super::*;
        use super::TempDir;
        use crate::semantic::chunker::ChunkerConfig;
        use crate::semantic::versioning::ModelPackage;

        const DIM: u32 = 8;
        const GENESIS: &str = "otzaria/tanach/genesis.txt";
        const BERACHOT: &str = "otzaria/mishna/berachot.txt";

        const LINES: [(u64, &str, &str); 3] = [
            (4_294_967_297, GENESIS, "בראשית ברא אלהים את השמים ואת הארץ"),
            (4_294_967_298, GENESIS, "ויאמר אלהים יהי אור ויהי אור"),
            (8_589_934_593, BERACHOT, "מאימתי קורין את שמע בערבית"),
        ];

        /// A deterministic vector that differs per text, so a misplaced one is visible.
        fn input(line_id: u64, text: &str) -> VectorInput {
            let digest = Sha256::digest(text.as_bytes());
            VectorInput {
                line_id,
                source_line_sha256: sha256_hex(text.as_bytes()),
                embedding_text_sha256: sha256_hex(text.as_bytes()),
                vector: (0..DIM)
                    .map(|i| f32::from(digest[i as usize]) + 1.0)
                    .collect(),
            }
        }

        fn model_for(checksum: &str, chunking: &ChunkerConfig) -> ModelIdentity {
            ModelIdentity {
                family_id: "otzaria-embedding-v1".to_string(),
                tokenizer_checksum: crate::semantic::embedding::mock::stub_tokenizer_checksum(),
                query_packages: vec![ModelPackage {
                    checksum: checksum.to_string(),
                    quantization: "int8".to_string(),
                }],
                embedding_dim: 2,
                pooling: "in-graph".to_string(),
                max_tokens: 512,
                embedding_text_version: 1,
                normalization_version: 1,
                chunking_identity: chunking.identity(),
            }
        }

        fn vectors_of(values: &[[f32; 2]]) -> Vec<u8> {
            values
                .iter()
                .flat_map(|v| v.iter().flat_map(|f| f.to_le_bytes()))
                .collect()
        }

        /// Write the two files a producer emits, and return their paths.
        fn write_inputs(dir: &TempDir, name: &str, inputs: &[VectorInput]) -> (PathBuf, PathBuf) {
            let vectors_path = dir.0.join(format!("{name}.f32"));
            let records_path = dir.0.join(format!("{name}.jsonl"));

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
            let short_records = dir.0.join("short.jsonl");
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
            let long_records = dir.0.join("long.jsonl");
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
            let ragged = dir.0.join("ragged.f32");
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

        /// Every one of these merged cleanly while the manifests were decoration: the floats
        /// are re-normalised at the merge, and the id set can still come out complete.
        #[test]
        fn shards_that_do_not_tile_this_plan_are_refused() {
            let dir = TempDir::new("shards");
            let model = model_for(&"ab".repeat(32), &ChunkerConfig::default());
            let write = |name: &str,
                         skip: usize,
                         records: usize,
                         plan: &str,
                         model: &ModelIdentity| {
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
                        embedding_dim: 2,
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
                family_id: "another/model@0000000".to_string(),
                ..model.clone()
            };
            let mixed = write("mixed", 2, 2, "plan", &other_model);
            expect_reason(
                verify_shards(&[a.clone(), mixed], "plan", &model, 4),
                "different model identity",
            );

            // A bit flipped in a vector, leaving it finite — which a merge would have
            // normalised and accepted.
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
            let write = |name: &str,
                         vectors: usize,
                         lines: usize,
                         bend: &dyn Fn(&mut ShardReport)| {
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
                    embedding_dim: 2,
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
    }
}
