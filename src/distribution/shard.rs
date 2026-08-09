//! Splitting one build across machines that never meet.
//!
//! [`build`](super::builder::build) does the whole job in one process: read the corpus,
//! apply the recipe, embed, pack. That is the right shape when one machine has both the
//! corpus and the arithmetic, and the wrong one for the library, where the corpus lives on
//! a build machine and the only affordable inference is on rented GPUs that see it for a
//! few hours and are then destroyed.
//!
//! So the same work is cut in two, at the one seam that does not weaken anything:
//!
//! ```text
//! export_plan    corpus + recipe -> plan.jsonl          build machine, no model
//! embed_shard    a slice of plan  -> vectors + records  GPU worker, no corpus
//! pack           all the vectors  -> artifact           build machine, full verification
//! ```
//!
//! **The recipe is applied exactly once, here.** A worker is handed finished strings, not
//! a corpus and a configuration, which is what makes it impossible for two workers on
//! different hardware to disagree about neighbour context, truncation or normalization —
//! and it is why a shard boundary cannot cut a context window: the windows were already
//! resolved when the plan was written.
//!
//! **A shard is a range of records, not a range of ids.** [`pack`](super::packer::pack)
//! compares the *set* of ids it was given against the recipe's expected set and sorts
//! internally, so shards merge by concatenation in any order. Sharding by id would demand
//! the plan be written in ascending id order, which would demand it be sorted, which would
//! demand the whole thing in memory — 5.9 million passages of text — to buy nothing.
//!
//! **A shard boundary should fall on a multiple of the batch size**, and the reason is
//! arithmetic rather than tidiness. llama.cpp's output depends on how a batch is composed,
//! so two runs that group the same texts differently produce vectors that differ in the
//! last bits. Measured on 2 396 real lines with the real GGUF: three shards of 800/800/796
//! at batch 32 gave a `vectors.f32` **byte-identical** to embedding the whole plan in one
//! window (`6530649db051…`), because 800 is a multiple of 32 and every batch was therefore
//! the same batch.
//!
//! The same measurement is why an artifact merged from shards is *not* byte-identical to
//! one [`build`](super::builder::build) produced from the same corpus: `PlannedEmbeddings`
//! refills from one book at a time, so its batches never span books and every book ends in
//! a short one. Neither grouping is more correct — the divergence is the same order as the
//! CPU-versus-Metal disagreement `docs/P2_REFERENCE_VECTORS.md` §5 measures and the
//! manifest already governs — but only one of them can be the artifact, and for the library
//! it is this one. It is also the faster one: 7 285 books is 7 285 partial batches.
//!
//! **What each side can check, it checks.** The worker cannot recompute
//! `source_line_sha256`; it has no corpus, and that digest is the packer's business at
//! merge time. It *can* recompute `embedding_text_sha256`, so it does, on every record —
//! see [`PackError::PlanTextChanged`]. What neither side can check alone, `pack` checks
//! afterwards against the corpus, unchanged from S4a.

use crate::distribution::builder::{chunks_for_book, ensure_recipe_matches};
use crate::distribution::corpus::CorpusBooks;
use crate::distribution::packer::{VectorInput, VectorInputRecord};
use crate::errors::PackError;
use crate::semantic::chunker::{Chunker, ChunkerConfig};
use crate::semantic::embedding::EmbeddingRuntime;
use crate::semantic::versioning::ModelIdentity;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::{BufRead, Write};

/// One line's work, as the build machine hands it to a machine with a GPU.
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardReport {
    pub records: usize,
    pub embedding_dim: usize,
    pub vectors_sha256: String,
    pub records_sha256: String,
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
    corpus: &dyn CorpusBooks,
    chunking: &ChunkerConfig,
    model: &ModelIdentity,
    sink: &mut dyn Write,
) -> Result<PlanReport, PackError> {
    let chunking_identity = ensure_recipe_matches(chunking, model)?;
    let chunker = Chunker::new(chunking.clone())?;

    // Hashing the bytes as they are written, rather than reading the file back: the digest
    // then describes what was actually sent, including a partial write that a later read
    // would not have distinguished from a short plan.
    let mut hasher = Sha256::new();
    let mut records = 0usize;
    let mut books = 0usize;
    let (mut min_line_id, mut max_line_id) = (u64::MAX, 0u64);
    // Ids only. Holding the text as well would be a second copy of the corpus in memory,
    // which is the cost this whole module exists to avoid.
    let mut seen = std::collections::BTreeSet::new();

    for book_key in corpus.book_keys()? {
        books += 1;
        for chunk in chunks_for_book(corpus, &chunker, &book_key)? {
            if !seen.insert(chunk.line_id) {
                return Err(PackError::DuplicateLineId {
                    line_id: chunk.line_id,
                });
            }
            let planned = PlannedChunk {
                line_id: chunk.line_id,
                source_line_sha256: sha256_hex(chunk.anchor_text.as_bytes()),
                embedding_text_sha256: sha256_hex(chunk.embedding_text.as_bytes()),
                embedding_text: chunk.embedding_text,
            };
            let mut line = serde_json::to_vec(&planned).map_err(|error| PackError::Corpus {
                reason: format!(
                    "line {} could not be written to the plan: {error}",
                    planned.line_id
                ),
            })?;
            line.push(b'\n');
            hasher.update(&line);
            write(sink, &line)?;

            records += 1;
            min_line_id = min_line_id.min(planned.line_id);
            max_line_id = max_line_id.max(planned.line_id);
        }
    }

    if records == 0 {
        return Err(PackError::NothingToEmbed { books });
    }
    sink.flush().map_err(|error| PackError::Corpus {
        reason: format!("the plan could not be flushed: {error}"),
    })?;

    Ok(PlanReport {
        records,
        min_line_id,
        max_line_id,
        plan_sha256: format!("{:x}", hasher.finalize()),
        chunking_identity,
    })
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
    runtime: &EmbeddingRuntime,
    batch_size: usize,
    vectors: &mut dyn Write,
    records: &mut dyn Write,
) -> Result<ShardReport, PackError> {
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
        records: written,
        embedding_dim: dim,
        vectors_sha256: format!("{:x}", vector_hasher.finalize()),
        records_sha256: format!("{:x}", record_hasher.finalize()),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::builder::{build, BuildRequest, PlannedCorpus};
    use crate::distribution::corpus::{CorpusLine, CorpusLineRecord, JsonlCorpus};
    use crate::distribution::packer::{pack, read_vector_inputs, PackRequest};
    use crate::semantic::backend::Pooling;
    use crate::semantic::embedding::{mock, validate_and_checksum_gguf, EmbeddingConfig};
    use crate::semantic::versioning::CorpusIdentity;
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
            let path = std::env::temp_dir().join(format!(
                "otzaria_shard_{name}_{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
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
            corpus_id: "3a".repeat(32),
            library_version: "otzaria-library-2026-08".to_string(),
            tantivy_schema_version: 3,
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
            model_id: "otzaria-embedding-v1".to_string(),
            model_checksum: checksum.to_string(),
            model_quantization: "Q4_K_M".to_string(),
            embedding_backend: "mock-hash-v1".to_string(),
            embedding_dim: DIM,
            pooling: "last-token".to_string(),
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
            pooling: Pooling::LastToken,
        });
        runtime.load().unwrap();
        runtime
    }

    /// The whole claim of this module, as one comparison.
    ///
    /// If a build cut into shards and reassembled is not byte-for-byte the artifact one
    /// process would have produced, then the split changed the product — and every
    /// argument about which machine runs which step is worthless. The digest covers the
    /// vectors and every stored record, and `created_at` is fixed on both sides because
    /// it is deliberately excluded from it.
    #[test]
    fn a_sharded_build_reassembles_into_the_artifact_one_process_would_have_written() {
        let dir = TempDir::new("equivalence");
        let corpus = corpus(&dir);
        let chunking = ChunkerConfig::default();
        let model_path = dir.0.join("model.gguf");
        mock::write_stub_gguf(&model_path, 3).unwrap();
        let model = model_for(&validate_and_checksum_gguf(&model_path).unwrap(), &chunking);

        let whole = build(
            BuildRequest {
                output_path: dir.0.join("whole"),
                model_path: model_path.clone(),
                model: model.clone(),
                chunking: chunking.clone(),
                created_at: "2026-08-09T00:00:00Z".to_string(),
                collection_name: "chunks".to_string(),
                batch_size: 2,
                allow_non_semantic_backend: true,
            },
            &corpus,
        )
        .unwrap();

        let plan_path = dir.0.join("plan.jsonl");
        let mut sink = std::fs::File::create(&plan_path).unwrap();
        let plan = export_plan(&corpus, &chunking, &model, &mut sink).unwrap();
        assert_eq!(
            plan.records, whole.vector_count as usize,
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

        let planned = PlannedCorpus::new(&corpus, &chunking, &model).unwrap();
        let merged = pack(
            PackRequest {
                output_path: dir.0.join("merged"),
                model: model.clone(),
                created_at: "2026-08-09T00:00:00Z".to_string(),
                collection_name: "chunks".to_string(),
            },
            read_vector_inputs(&vectors_path, &records_path, DIM).unwrap(),
            &planned,
        )
        .unwrap();

        assert_eq!(
            merged.digest, whole.digest,
            "a sharded build must produce the artifact a single-process build produces"
        );
    }

    /// A shard that never ran is a hole, and `pack` is what has to see it.
    ///
    /// Deliberately checked here rather than trusted from S4a: the whole reason a shard
    /// may be re-run in isolation is that a lost one is detectable, and this is the
    /// statement of that.
    #[test]
    fn a_missing_shard_is_refused_at_the_merge() {
        let dir = TempDir::new("hole");
        let corpus = corpus(&dir);
        let chunking = ChunkerConfig::default();
        let model_path = dir.0.join("model.gguf");
        mock::write_stub_gguf(&model_path, 3).unwrap();
        let model = model_for(&validate_and_checksum_gguf(&model_path).unwrap(), &chunking);

        let plan_path = dir.0.join("plan.jsonl");
        let mut sink = std::fs::File::create(&plan_path).unwrap();
        let plan = export_plan(&corpus, &chunking, &model, &mut sink).unwrap();
        drop(sink);

        let runtime = runtime_for(&model_path);
        let vectors_path = dir.0.join("vectors.f32");
        let records_path = dir.0.join("records.jsonl");
        let mut vectors = std::fs::File::create(&vectors_path).unwrap();
        let mut records = std::fs::File::create(&records_path).unwrap();
        // Every record but the last one.
        let file = std::io::BufReader::new(std::fs::File::open(&plan_path).unwrap());
        embed_shard(
            read_plan(file, 0, plan.records - 1),
            &runtime,
            2,
            &mut vectors,
            &mut records,
        )
        .unwrap();
        drop((vectors, records));

        let planned = PlannedCorpus::new(&corpus, &chunking, &model).unwrap();
        let outcome = pack(
            PackRequest {
                output_path: dir.0.join("merged"),
                model: model.clone(),
                created_at: "2026-08-09T00:00:00Z".to_string(),
                collection_name: "chunks".to_string(),
            },
            read_vector_inputs(&vectors_path, &records_path, DIM).unwrap(),
            &planned,
        );
        match outcome {
            Err(PackError::CoverageMismatch { .. }) => {}
            Ok(_) => panic!("a merge missing a shard must not produce an artifact"),
            Err(other) => panic!("expected a coverage mismatch, got {other:?}"),
        }
    }

    /// The one check a worker can make on its own, and the reason the digest travels.
    #[test]
    fn a_plan_whose_text_no_longer_matches_its_digest_is_refused_before_it_is_embedded() {
        let dir = TempDir::new("tampered");
        let model_path = dir.0.join("model.gguf");
        mock::write_stub_gguf(&model_path, 3).unwrap();
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
}
