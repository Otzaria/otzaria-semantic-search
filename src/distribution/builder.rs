//! S4b in this crate: a corpus and a model in, a base vector package out.
//!
//! The development one-shot of a build. It opens the model, applies the recipe to the
//! corpus, embeds every distinct text the recipe derives, and writes what an installation
//! takes: a base [segment](crate::semantic::oxv) holding every (book, key) pair of the
//! corpus, the metadata-v3 package around it, and the release manifest that describes both
//! (see [`install_package`](crate::semantic::segment_set::install_package)). The library is
//! built on machines that never see the corpus (see [`shard`](super::shard)); this is the
//! same recipe, the same model checks and the same segment, in one process and in memory.
//!
//! # What that closes
//!
//! A tool handed finished floats can only check that they line up with the corpus. Here
//! the vector, its key and the model identity come out of one pass over one model:
//!
//! * the text that is hashed into the key is the same `String` that is handed to the
//!   backend — there is no path by which one can describe the other;
//! * `tokenizer_checksum`, `embedding_dim`, `pooling`, the effective `max_tokens` and the
//!   package checksum are **reported by the loaded runtime** and compared against what the
//!   identity declares, so the declaration is a checked claim rather than a copied string.
//!
//! "Reported by the runtime" rather than "read from the file", because they are not all the
//! same kind of fact: the checksums are of the bytes on disk; the width and the token cap
//! are what the weights actually carry; and pooling is what the selected implementation
//! performs. All of them are settled by the thing that is about to produce the vectors,
//! which is what makes comparing them worth anything.
//!
//! The three **recipe versions** — `chunking_version`, `embedding_text_version` and
//! `normalization_version` — are not properties of a model at all. They are versions of
//! code in this crate, so they are settled outright rather than compared:
//! [`EmbeddingRecipe::resolve`] refuses any that names behaviour nobody has written, and
//! the [`Chunker`] dispatches on the resolved values. All three describe what happens to
//! the **text**; L2 normalization of the finished vector is an unconditional invariant of
//! cosine and is deliberately not versioned. See [`recipe`](crate::semantic::recipe).
//!
//! What is left declared and unverifiable: `family_id` and each package's quantization.
//! Nothing in an ONNX package states either in a form anything here could check, and
//! inventing a check that reads them from the same place that wrote them would prove
//! nothing.
//!
//! **One window stays open, and is not closed here.** The checksum is computed, and then
//! the backend opens the same path again; a model file swapped between those two reads
//! would be hashed as one file and executed as another. Closing it means giving the build a
//! content-addressed copy it owns — a staging step in the pipeline around this, not a check
//! inside it.
//!
//! # The plan comes before the inference
//!
//! [`BuildPlan`] is the set of lines the recipe embeds, derived from the corpus and the
//! chunker configuration **before a single vector exists**. The embedding pass derives the
//! same set again, line by line, and a build whose two passes disagree is refused as
//! [`PackError::CoverageMismatch`]: the corpus changed underneath it.
//!
//! Deriving that set from the vectors instead would make the check confirm itself: a book
//! the corpus stopped answering for would disappear from both sides at once and the
//! package would be certified complete for the part that happened to survive.
//!
//! # The recipe cannot be guessed, so it is supplied and pinned
//!
//! `chunking_identity` is a one-way hash of a whole [`ChunkerConfig`]. A build is
//! therefore handed the configuration itself, and [`BuildPlan::compute`] refuses one whose
//! identity is not the value the package will declare. Without that, the model identity
//! would be a label on a recipe nobody applied.
//!
//! # What it costs
//!
//! Each line is chunked twice — once to plan, once to embed — and read from the corpus
//! twice with it; the second read is what proves the corpus still says what the plan
//! assumed. Every distinct vector is held as floats until the codec is calibrated on all
//! of them, which is what makes this a development path: the library's build streams.

use crate::distribution::corpus::{CorpusBooks, CorpusLine};
use crate::distribution::package::{IndexPackage, PackageKind};
use crate::errors::PackError;
use crate::semantic::backend::Pooling;
use crate::semantic::chunk_key::ChunkKey;
use crate::semantic::chunker::{Chunker, ChunkerConfig};
use crate::semantic::embedding::{EmbeddingConfig, EmbeddingRuntime};
use crate::semantic::official_index::readable_store_identity;
use crate::semantic::oxv::codec::Codec;
use crate::semantic::oxv::reader::Segment;
use crate::semantic::oxv::writer::{SegmentBuilder, SegmentSpec};
use crate::semantic::recipe::EmbeddingRecipe;
use crate::semantic::segment_set::ReleaseManifest;
use crate::semantic::types::{BookForIndexing, BookLine, SemanticChunk};
use crate::semantic::versioning::{EmbeddingWorker, IndexVersion, ModelIdentity, VectorProvenance};
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::io;
use std::path::{Path, PathBuf};

/// The segment's name in a built package — the one payload of the package a release
/// manifest describes.
pub const SEGMENT_FILENAME: &str = "segment.oxv";

/// The release manifest a build writes beside its package: with the segment, what an
/// installation is handed.
pub const RELEASE_MANIFEST_FILENAME: &str = "release.json";

/// What to build, beyond the corpus.
#[derive(Debug, Clone)]
pub struct BuildRequest {
    /// Directory the package is written into. Must not exist, or be an empty directory.
    pub output_path: PathBuf,
    /// The ONNX graph the vectors are produced with, its package beside it. Its checksum
    /// has to be one of `model.query_packages`.
    pub model_path: PathBuf,
    /// The model family the package declares. The half a model file can answer for is
    /// checked against it; see the module documentation for the half that cannot be.
    pub model: ModelIdentity,
    /// The recipe itself. `model.chunking_identity` is its hash, and a disagreement is
    /// [`PackError::RecipeMismatch`].
    pub chunking: ChunkerConfig,
    pub created_at: String,
    /// Texts per inference call.
    pub batch_size: usize,
    /// The quantile each dimension's int8 scale is calibrated at — see
    /// [`Codec::calibrate_i8_sym_dim`]. `1.0` clips nothing the base holds.
    pub clip_q: f32,
    /// Permit a backend whose vectors carry no meaning.
    ///
    /// `false` in anything that ships. The deterministic stand-in produces vectors that
    /// are structurally perfect and semantically empty, and a package built from them
    /// passes every check in this crate — so the refusal has to be here, where the backend
    /// is still identifiable, rather than downstream where it is not.
    pub allow_non_semantic_backend: bool,
}

/// What a build wrote.
#[derive(Debug, Clone)]
pub struct BuildReport {
    pub output_path: PathBuf,
    /// As written to [`RELEASE_MANIFEST_FILENAME`]: the identity, the counts, the segment's
    /// digest and the package digest.
    pub manifest: ReleaseManifest,
    /// SHA-256 of the release manifest's bytes: the value a publisher announces outside it,
    /// and an installation expects as
    /// [`published_manifest_sha256`](crate::semantic::segment_set::InstallExpectation).
    pub manifest_sha256: String,
    /// Lines the recipe embeds.
    pub planned_lines: usize,
    /// Components the codec clipped.
    pub clipped_components: u64,
}

/// The lines a recipe embeds, decided before anything is embedded.
///
/// Deliberately just the ids. Holding each line's embedded text would turn a library-scale
/// plan into a second copy of the corpus in RAM, and the text is cheap to derive again from
/// the same corpus and the same configuration — which is also a check, since a corpus that
/// changed underneath the build no longer produces the planned set and the coverage
/// comparison says so.
#[derive(Debug, Clone)]
pub struct BuildPlan {
    line_ids: BTreeSet<u64>,
}

impl BuildPlan {
    /// Apply `chunking` to every book of `corpus`, and refuse a configuration that is not
    /// the one `model` declares.
    ///
    /// The identity check is the whole reason this takes both: `model.chunking_identity`
    /// is a hash, so it can be compared and never inverted. A build handed the wrong
    /// configuration would otherwise embed one recipe's text, declare another recipe's
    /// identity, and certify its own coverage against the recipe it applied rather than
    /// the one it announced.
    pub fn compute(
        corpus: &dyn CorpusBooks,
        chunking: &ChunkerConfig,
        model: &ModelIdentity,
    ) -> Result<Self, PackError> {
        ensure_recipe_matches(chunking, model)?;
        let chunker = Chunker::new(chunking.clone())?;
        let mut line_ids = BTreeSet::new();
        let mut books = 0usize;
        for book_key in corpus.book_keys()? {
            books += 1;
            for chunk in chunks_for_book(corpus, &chunker, &book_key)? {
                // Two books claiming one line, or one book listing it twice. Built, it
                // would be recorded twice, under two positions; refusing here names the
                // corpus, which is where the fault is.
                if !line_ids.insert(chunk.line_id) {
                    return Err(PackError::DuplicateLineId {
                        line_id: chunk.line_id,
                    });
                }
            }
        }

        if line_ids.is_empty() {
            return Err(PackError::NothingToEmbed { books });
        }
        Ok(Self { line_ids })
    }

    /// How many lines will be embedded. Reported before the model is asked for anything,
    /// so a corpus that is obviously the wrong one is visible before a long build starts.
    pub fn len(&self) -> usize {
        self.line_ids.len()
    }

    /// Never true for a computed plan — [`Self::compute`] refuses an empty one.
    pub fn is_empty(&self) -> bool {
        self.line_ids.is_empty()
    }

    /// The planned ids, in ascending order.
    pub fn line_ids(&self) -> &BTreeSet<u64> {
        &self.line_ids
    }
}

/// The declared `chunking_identity` and the recipe in hand are the same recipe.
///
/// Hoisted out of [`BuildPlan::compute`] so [`build`] can ask it before opening a model or
/// reading a corpus: it costs one hash of five integers, and it is the check most likely to
/// fail on a misconfigured build.
pub(crate) fn ensure_recipe_matches(
    chunking: &ChunkerConfig,
    model: &ModelIdentity,
) -> Result<u64, PackError> {
    let actual = chunking.identity();
    if actual != model.chunking_identity {
        return Err(PackError::RecipeMismatch {
            declared: model.chunking_identity,
            actual,
        });
    }
    Ok(actual)
}

/// Build a base vector package from a corpus and a model.
///
/// The order is cost, not taste — each step is the cheapest way to fail that is still
/// available:
///
/// 1. Refuse an output path that already holds something. Nothing else is worth doing if
///    the result cannot be written.
/// 2. Refuse an identity with a field left unfilled, or a release tag a segment cannot
///    carry — one read of the corpus identity, and fatal at the end of a build just as
///    surely as at the start.
/// 3. Refuse a recipe that is not the declared one: five integers hashed, no model and no
///    corpus.
/// 4. Load the model, and hold the declared identity to what the file reports — before six
///    million lines are chunked rather than after.
/// 5. Refuse a backend whose vectors are not semantic, unless the caller has said otherwise
///    in as many words.
/// 6. Plan: apply the recipe to the whole corpus.
/// 7. Embed, book by book in byte order of their names, every text whose key has no slot
///    yet; a key's first appearance takes the next slot, a later one in another book is an
///    extra record of it, and the plan is derived again and compared.
/// 8. Calibrate the codec on those vectors; write the segment, the package around it —
///    which hashes the segment again before describing it — and the release manifest.
///
/// Steps 1–7 write nothing, so any rejection up to there leaves the output directory empty
/// and the run can simply be repeated. The release manifest is written last, under a
/// temporary name renamed into place: a directory without one holds nothing an
/// installation takes, and the next attempt refuses it rather than writing over it.
pub fn build(request: BuildRequest, corpus: &dyn CorpusBooks) -> Result<BuildReport, PackError> {
    ensure_output_is_free(&request.output_path)?;
    let corpus_identity = corpus.identity()?;
    let identity = IndexVersion {
        text: corpus_identity.text.clone(),
        model: request.model.clone(),
        // Not the caller's: a package in a layout this build cannot read would be a package
        // for nobody.
        store: readable_store_identity(),
    };
    identity.validate_complete()?;
    crate::distribution::package::validate_release_tag(&corpus_identity.library_release_tag)?;
    // Every version the build is about to act under, settled before anything is opened:
    // the chunker will resolve them again from the same configuration, and the text
    // normalization is applied there, where the key of the embedded string is computed.
    EmbeddingRecipe::resolve(&request.chunking, &request.model)?;
    ensure_recipe_matches(&request.chunking, &request.model)?;
    let (runtime, provenance) = load_model(&request)?;

    let plan = BuildPlan::compute(corpus, &request.chunking, &request.model)?;
    log::info!(
        "The recipe embeds {} line(s) of {}",
        plan.len(),
        request.output_path.display()
    );
    let assembled = embed_corpus(corpus, &runtime, &request, &plan)?;

    let dim = request.model.embedding_dim as usize;
    let vectors: Vec<&[f32]> = assembled.vectors.chunks_exact(dim).collect();
    let codec = Codec::calibrate_i8_sym_dim(&vectors, request.clip_q).map_err(|reason| {
        PackError::MalformedInput {
            reason: format!("the int8 codec cannot be calibrated: {reason}"),
        }
    })?;

    let root = &request.output_path;
    std::fs::create_dir_all(root).map_err(io_error(format!("creating {}", root.display())))?;
    let mut segment = SegmentBuilder::new(
        SegmentSpec {
            kind: PackageKind::Base,
            identity_digest: identity.identity_digest(),
            from_library_version: 0,
            to_library_version: corpus_identity.library_version,
            library_release_tag: corpus_identity.library_release_tag.clone(),
        },
        codec,
    );
    for book in &assembled.books {
        segment
            .add_book(&book.name, &book.primary, &book.extras, &[])
            .map_err(io_error(format!("assembling book {:?}", book.name)))?;
    }
    let codec_params_sha256 = segment.codec().params_sha256();
    let segment_path = root.join(SEGMENT_FILENAME);
    let written = (|| {
        let mut sink = segment.write(&segment_path)?;
        for vector in &vectors {
            sink.push_f32(vector)?;
        }
        sink.finish()
    })()
    .map_err(io_error(format!("writing {}", segment_path.display())))?;

    let manifest = ReleaseManifest::for_segment(
        &written,
        &identity,
        codec_params_sha256,
        provenance,
        request.created_at.clone(),
    );
    IndexPackage::write(root, &manifest.package())?;
    // What the package's own check cannot see: the tables inside the segment.
    Segment::open(&segment_path)?;

    let json = manifest.to_json();
    let manifest_path = root.join(RELEASE_MANIFEST_FILENAME);
    let partial = root.join(format!("{RELEASE_MANIFEST_FILENAME}.partial"));
    (|| {
        // Flushed through the handle that wrote it: Windows refuses to flush a handle
        // opened only for reading.
        let mut file = std::fs::File::create(&partial)?;
        io::Write::write_all(&mut file, json.as_bytes())?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&partial, &manifest_path)?;
        crate::distribution::package::sync_dir(root)
    })()
    .map_err(io_error(format!("writing {}", manifest_path.display())))?;

    log::info!(
        "Built {}: {} vector(s) and {} further record(s) across {} book(s), {} bytes",
        root.display(),
        written.counts.slots,
        written.counts.extras,
        written.counts.books,
        written.size
    );
    Ok(BuildReport {
        output_path: request.output_path,
        manifest_sha256: format!("{:x}", Sha256::digest(json.as_bytes())),
        manifest,
        planned_lines: plan.len(),
        clipped_components: written.clipped_components,
    })
}

/// Open the model and prove it is a package of the family the artifact will name.
///
/// The runtime refuses a width or a pooling that disagrees with the loaded backend before
/// this returns, so two of the comparisons below can only fail through it. They are listed
/// anyway: this table is the statement of what the loaded runtime can be held to, and
/// leaving a field out of it because something else happens to cover it today is how such
/// a check quietly stops covering it.
///
/// The package itself has to be one of the family's: its checksum among
/// `query_packages`, which is also where the precision it is recorded under comes from.
/// What it is, and the backend that ran it, become the artifact's provenance.
fn load_model(request: &BuildRequest) -> Result<(EmbeddingRuntime, VectorProvenance), PackError> {
    let declared = &request.model;
    let mut runtime = EmbeddingRuntime::new(EmbeddingConfig {
        model_path: request.model_path.clone(),
        embedding_dim: declared.embedding_dim,
        max_tokens: declared.max_tokens,
        batch_size: request.batch_size,
        pooling: Pooling::parse(&declared.pooling)?,
    });
    runtime.load()?;

    for (field, declared, loaded) in [
        (
            "tokenizer_checksum",
            declared.tokenizer_checksum.clone(),
            runtime.tokenizer_checksum().unwrap_or_default().to_string(),
        ),
        (
            "embedding_dim",
            declared.embedding_dim.to_string(),
            runtime.dim().to_string(),
        ),
        (
            "pooling",
            declared.pooling.clone(),
            runtime.pooling().to_string(),
        ),
        (
            // The backend clamps the request to its model's context length, so a build
            // that asks for more than the weights allow embeds truncated text while the
            // artifact declares the cap nobody honoured.
            "max_tokens",
            declared.max_tokens.to_string(),
            runtime.max_tokens().to_string(),
        ),
    ] {
        if declared != loaded {
            return Err(PackError::ModelDisagreesWithFile {
                field,
                declared,
                loaded,
            });
        }
    }

    let checksum = runtime.model_checksum().unwrap_or_default();
    let package = declared
        .query_packages
        .iter()
        .find(|package| package.checksum == checksum)
        .ok_or_else(|| PackError::ModelDisagreesWithFile {
            field: "query_packages",
            declared: declared
                .query_packages
                .iter()
                .map(|package| format!("{} {}", package.quantization, package.checksum))
                .collect::<Vec<_>>()
                .join(", "),
            loaded: checksum.to_string(),
        })?
        .clone();

    if !runtime.backend_is_semantic() && !request.allow_non_semantic_backend {
        return Err(PackError::NonSemanticBackend {
            backend: runtime.backend_id().unwrap_or("none").to_string(),
        });
    }

    let provenance = VectorProvenance {
        passage_package: package,
        worker: EmbeddingWorker {
            backend: runtime.backend_id().unwrap_or("none").to_string(),
            device: "cpu".to_string(),
        },
    };
    Ok((runtime, provenance))
}

/// The recipe applied to one book: the corpus's lines in corpus order, chunked.
///
/// Assembling a [`BookForIndexing`] and calling the real [`Chunker`] is the point — the
/// recipe has exactly one implementation, and a build applies *that* one rather than a
/// second reading of it.
///
/// The book-level metadata below reaches nothing that is stored: a segment stores a book's
/// name, and each line's key and position. What the chunker actually consumes is each
/// line's text, its `section_id` and its position in the list.
pub(crate) fn chunks_for_book(
    corpus: &dyn CorpusBooks,
    chunker: &Chunker,
    book_key: &str,
) -> Result<Vec<SemanticChunk>, PackError> {
    Ok(chunk_book_lines(corpus, chunker, book_key)?.1)
}

/// [`chunks_for_book`], with the book's line ids in the order they were read.
fn chunk_book_lines(
    corpus: &dyn CorpusBooks,
    chunker: &Chunker,
    book_key: &str,
) -> Result<(Vec<u64>, Vec<SemanticChunk>), PackError> {
    let line_ids = corpus.book_line_ids(book_key)?;
    let mut lines = Vec::with_capacity(line_ids.len());
    /// The book-level fields, taken from whichever line comes first. None of them reaches
    /// the segment, so a book whose lines disagreed about its title builds all the same.
    struct BookFields {
        title: String,
        content_hash: u64,
        is_pdf: bool,
        facets: Vec<String>,
    }
    let mut book_fields: Option<BookFields> = None;

    for &line_id in &line_ids {
        let line = corpus
            .line(line_id)?
            .ok_or(PackError::LineNotInCorpus { line_id })?;
        // Destructured rather than read field by field, so nothing is copied for the
        // millions of lines that only contribute their text — and so a field added to
        // `CorpusLine` has to be placed here rather than silently ignored.
        let CorpusLine {
            source_book_key,
            title,
            reference,
            section_id,
            segment,
            is_pdf,
            line_hash,
            content_hash,
            facets,
            text,
        } = line;

        // The two halves of the port describing one line differently is a corpus that
        // contradicts itself, and this is the only place both are visible at once. Left
        // unchecked it would group a line into the wrong book's context window and silently
        // embed it against text from elsewhere.
        if source_book_key != book_key {
            return Err(PackError::Corpus {
                reason: format!(
                    "line {line_id} is listed under book {book_key:?} and says it belongs \
                     to {source_book_key:?}"
                ),
            });
        }

        book_fields.get_or_insert(BookFields {
            title,
            content_hash,
            is_pdf,
            facets,
        });
        lines.push(BookLine {
            line_id,
            section_id,
            text,
            line_hash,
            reference,
            segment,
        });
    }

    let Some(book) = book_fields else {
        return Err(PackError::Corpus {
            reason: format!("book {book_key:?} holds no lines"),
        });
    };

    let chunks = chunker.chunk_book(&BookForIndexing {
        source_book_key: book_key.to_string(),
        title: book.title,
        content_fingerprint: book.content_hash,
        is_pdf: book.is_pdf,
        topics: String::new(),
        extra_facets: book.facets,
        lines,
    });
    Ok((line_ids, chunks))
}

/// One book's records, as a segment stores them.
struct AssembledBook {
    name: String,
    /// `(key, hint)` of each record whose key got its slot in this book, in hint order.
    primary: Vec<(ChunkKey, u32)>,
    /// `(hint, slot)` of each record whose key already had a slot.
    extras: Vec<(u32, u32)>,
}

/// Every book's records, and every distinct vector in slot order.
struct Assembled {
    books: Vec<AssembledBook>,
    /// `slots × embedding_dim` floats.
    vectors: Vec<f32>,
}

/// Apply the recipe to every book and embed what it derives, assigning slots the way a
/// segment stores them: books in byte order of their names, each book's lines in order, and
/// a key takes the next slot the first time it appears. A later appearance in another book
/// is an extra record of that slot; a later appearance in the same book is not recorded at
/// all, because a record is a book and a key, and its hint is the first line that has it.
///
/// A line's hint is its position in the book as [`CorpusBooks::book_line_ids`] lists it —
/// the line's ordinal, in the application's terms.
///
/// What it holds: one book's chunks, and every distinct vector so far.
fn embed_corpus(
    corpus: &dyn CorpusBooks,
    runtime: &EmbeddingRuntime,
    request: &BuildRequest,
    plan: &BuildPlan,
) -> Result<Assembled, PackError> {
    let chunker = Chunker::new(request.chunking.clone())?;
    // A batch of zero would embed nothing forever.
    let batch_size = request.batch_size.max(1);
    let dim = request.model.embedding_dim as usize;

    let mut book_keys = corpus.book_keys()?;
    book_keys.sort_unstable();
    if let Some(pair) = book_keys.windows(2).find(|pair| pair[0] == pair[1]) {
        return Err(PackError::Corpus {
            reason: format!("book {:?} is listed twice", pair[0]),
        });
    }

    let mut slots: HashMap<ChunkKey, u32> = HashMap::new();
    let mut vectors = Vec::new();
    let mut books = Vec::with_capacity(book_keys.len());
    let mut chunked = BTreeSet::new();
    for name in book_keys {
        let (line_ids, chunks) = chunk_book_lines(corpus, &chunker, &name)?;
        if line_ids.len() > crate::semantic::oxv::format::HINT_MASK as usize {
            return Err(PackError::Corpus {
                reason: format!(
                    "book {name:?} holds {} lines, more than a hint can name",
                    line_ids.len()
                ),
            });
        }
        let ordinals: HashMap<u64, u32> = line_ids
            .iter()
            .enumerate()
            .map(|(ordinal, line_id)| (*line_id, ordinal as u32))
            .collect();

        let mut book = AssembledBook {
            name,
            primary: Vec::new(),
            extras: Vec::new(),
        };
        let mut recorded = HashSet::new();
        let mut texts = Vec::new();
        for chunk in chunks {
            chunked.insert(chunk.line_id);
            let hint = ordinals[&chunk.line_id];
            // The string the backend is handed below, and nothing derived from it.
            let key = ChunkKey::of(&chunk.embedding_text);
            if !recorded.insert(key) {
                continue;
            }
            match slots.get(&key) {
                Some(&slot) => book.extras.push((hint, slot)),
                None => {
                    slots.insert(key, slots.len() as u32);
                    book.primary.push((key, hint));
                    texts.push(chunk.embedding_text);
                }
            }
        }

        for batch in texts.chunks(batch_size) {
            let batch: Vec<&str> = batch.iter().map(String::as_str).collect();
            let embedded = runtime.embed_batch(&batch)?;
            // The runtime already refuses a short batch from a backend; this is the same
            // invariant one layer up, where the slots would silently shift instead.
            if embedded.len() != batch.len() || embedded.iter().any(|v| v.len() != dim) {
                return Err(PackError::MalformedInput {
                    reason: format!(
                        "{} vector(s) came back for {} text(s) of book {:?}, and each must \
                         be {dim} wide",
                        embedded.len(),
                        batch.len(),
                        book.name
                    ),
                });
            }
            for vector in embedded {
                vectors.extend_from_slice(&vector);
            }
        }
        books.push(book);
    }

    if chunked != *plan.line_ids() {
        let missing: Vec<u64> = plan.line_ids().difference(&chunked).copied().collect();
        let unexpected: Vec<u64> = chunked.difference(plan.line_ids()).copied().collect();
        return Err(PackError::CoverageMismatch {
            expected: plan.len(),
            covered: chunked.len(),
            missing: missing.len(),
            unexpected: unexpected.len(),
            first_missing: missing.first().copied(),
            first_unexpected: unexpected.first().copied(),
        });
    }
    Ok(Assembled { books, vectors })
}

/// Refuse an output path that is not an empty place to write a whole package.
pub(crate) fn ensure_output_is_free(path: &Path) -> Result<(), PackError> {
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
            "it already holds {entries} entr{}, and a build writes a whole package",
            if entries == 1 { "y" } else { "ies" }
        )));
    }
    Ok(())
}

fn io_error(context: String) -> impl FnOnce(io::Error) -> PackError {
    move |source| PackError::Io { context, source }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cancellation::CancellationToken;
    use crate::distribution::corpus::{CorpusIdentity, CorpusIndex, CorpusLineRecord, JsonlCorpus};
    use crate::semantic::chunker::compute_chunk_hash;
    use crate::semantic::embedding::mock;
    use crate::semantic::model_package::validate_model;
    use crate::semantic::oxv::scan::ScanRequest;
    use crate::semantic::segment_set::{
        install_package, InstallExpectation, InstallSource, SegmentSet,
    };
    use crate::semantic::versioning::ModelPackage;
    use std::collections::BTreeMap;

    const DIM: u32 = 64;
    const GENESIS: &str = "otzaria/tanach/genesis.txt";
    const BERACHOT: &str = "otzaria/mishna/berachot.txt";

    /// Long enough to stand alone under the default recipe (20 characters).
    const LONG: &str = "בראשית ברא אלהים את השמים ואת הארץ";
    const LONG_TWO: &str = "והארץ היתה תהו ובהו וחשך על פני תהום רבה";
    /// Below `min_meaningful_chars` and above `min_embeddable_chars`: embedded, but with
    /// its neighbours' text folded in.
    const BORROWS: &str = "ויהי אור";
    /// Below `min_embeddable_chars`: not embedded at all.
    const SKIPPED: &str = "או";

    struct TempDir(PathBuf);
    impl TempDir {
        fn new(name: &str) -> Self {
            // The clock alone collided: macOS ticks coarser than a test takes to start.
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "otzaria_builder_{name}_{}_{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos(),
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
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

    fn corpus_identity() -> CorpusIdentity {
        CorpusIdentity {
            text: crate::semantic::versioning::TextIdentity::with_line_text_version(1),
            library_version: 30,
            library_release_tag: "v30-20260930120000".to_string(),
            document_id_scheme_version: 1,
        }
    }

    fn corpus_line(book: &str, text: &str, section_id: u64) -> CorpusLine {
        CorpusLine {
            source_book_key: book.to_string(),
            title: if book == GENESIS {
                "בראשית"
            } else {
                "ברכות"
            }
            .to_string(),
            reference: format!("{book} :: {}", text.chars().take(6).collect::<String>()),
            section_id,
            segment: 0,
            is_pdf: false,
            line_hash: 0,
            content_hash: 7,
            facets: vec!["/מקרא/תורה".to_string()],
            text: text.to_string(),
        }
    }

    /// `(line_id, book, text, section_id)` into the two files a transcription is.
    fn write_corpus(dir: &TempDir, lines: &[(u64, &str, &str, u64)]) -> JsonlCorpus {
        let identity_path = dir.path().join("corpus-identity.json");
        let lines_path = dir.path().join("corpus-lines.jsonl");
        std::fs::write(
            &identity_path,
            serde_json::to_vec_pretty(&corpus_identity()).unwrap(),
        )
        .unwrap();
        let body: String = lines
            .iter()
            .map(|(line_id, book, text, section_id)| {
                format!(
                    "{}\n",
                    serde_json::to_string(&CorpusLineRecord {
                        line_id: *line_id,
                        line: corpus_line(book, text, *section_id),
                    })
                    .unwrap()
                )
            })
            .collect();
        std::fs::write(&lines_path, body).unwrap();
        JsonlCorpus::load(&identity_path, &lines_path).unwrap()
    }

    /// The corpus most tests use: three embeddable lines and one the recipe skips.
    fn standard_corpus(dir: &TempDir) -> JsonlCorpus {
        write_corpus(
            dir,
            &[
                (4_294_967_297, GENESIS, LONG, 1),
                (4_294_967_298, GENESIS, BORROWS, 1),
                (4_294_967_299, GENESIS, SKIPPED, 1),
                (8_589_934_593, BERACHOT, LONG_TWO, 1),
            ],
        )
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

    /// Where [`write_model`] puts the stub graph.
    fn model_path_in(dir: &TempDir) -> PathBuf {
        dir.path().join("model").join("model.onnx")
    }

    /// A stub ONNX package and the checksum a build must declare for it.
    fn write_model(dir: &TempDir) -> (PathBuf, String) {
        let path = mock::write_stub_onnx_package(&dir.path().join("model"));
        assert_eq!(path, model_path_in(dir));
        let checksum = validate_model(&path).unwrap().checksum().to_string();
        (path, checksum)
    }

    fn build_request(dir: &TempDir, model: ModelIdentity, chunking: ChunkerConfig) -> BuildRequest {
        let model_path = model_path_in(dir);
        BuildRequest {
            output_path: dir.path().join("artifact"),
            model_path,
            model,
            chunking,
            created_at: "2026-08-08T00:00:00Z".to_string(),
            batch_size: 2,
            clip_q: 1.0,
            allow_non_semantic_backend: true,
        }
    }

    /// Every record of a built segment: `(book, hint)` to the key's hex and whether the
    /// record holds the key's slot.
    fn stored_records(report: &BuildReport) -> BTreeMap<(String, u32), (String, bool)> {
        let segment = Segment::open(&report.output_path.join(SEGMENT_FILENAME)).unwrap();
        let mut records = BTreeMap::new();
        for book in segment.books() {
            for slot in book.slots.clone() {
                let entry = (segment.key(slot).to_hex(), true);
                records.insert((book.name.to_string(), segment.hint(slot)), entry);
            }
            for extra in book.extras.clone() {
                let (slot, hint) = segment.extra(extra);
                records.insert(
                    (book.name.to_string(), hint),
                    (segment.key(slot).to_hex(), false),
                );
            }
        }
        records
    }

    /// The recipe decides what gets a vector, and it is not "every line the corpus holds".
    /// A line too short to carry meaning exists perfectly well and must stay out of the
    /// plan — otherwise every build would be short of the coverage it promised.
    #[test]
    fn the_plan_is_the_recipe_applied_rather_than_every_line() {
        let dir = TempDir::new("plan");
        let corpus = standard_corpus(&dir);
        let chunking = ChunkerConfig::default();
        let model = model_for(&"ab".repeat(32), &chunking);

        let plan = BuildPlan::compute(&corpus, &chunking, &model).unwrap();

        assert_eq!(plan.len(), 3);
        assert!(!plan.is_empty());
        assert_eq!(
            plan.line_ids().iter().copied().collect::<Vec<_>>(),
            vec![4_294_967_297, 4_294_967_298, 8_589_934_593],
            "the line below min_embeddable_chars is not part of the coverage contract"
        );
    }

    /// `chunking_identity` is a hash, so a build cannot recover the recipe from the model
    /// identity — it has to be handed one, and this is the only thing that establishes it
    /// was handed the right one.
    #[test]
    fn a_recipe_that_is_not_the_declared_one_is_refused() {
        let dir = TempDir::new("recipe");
        let corpus = standard_corpus(&dir);
        let declared = ChunkerConfig::default();
        let model = model_for(&"ab".repeat(32), &declared);

        let applied = ChunkerConfig {
            max_chunk_chars: 64,
            ..ChunkerConfig::default()
        };
        assert_ne!(applied.identity(), declared.identity());

        match BuildPlan::compute(&corpus, &applied, &model) {
            Err(PackError::RecipeMismatch { declared, actual }) => {
                assert_eq!(declared, model.chunking_identity);
                assert_eq!(actual, applied.identity());
            }
            other => panic!("a recipe that is not the declared one must be refused, got {other:?}"),
        }
    }

    /// An empty plan would produce an artifact that verifies and holds nothing. Refused
    /// before the model is opened, not after the payload is written.
    #[test]
    fn a_corpus_the_recipe_embeds_nothing_from_is_refused() {
        let dir = TempDir::new("nothing");
        let corpus = write_corpus(&dir, &[(4_294_967_297, GENESIS, SKIPPED, 1)]);
        let chunking = ChunkerConfig::default();
        let model = model_for(&"ab".repeat(32), &chunking);

        match BuildPlan::compute(&corpus, &chunking, &model) {
            Err(PackError::NothingToEmbed { books }) => assert_eq!(books, 1),
            other => panic!("a corpus with nothing to embed must be refused, got {other:?}"),
        }
    }

    /// The two halves of the port must describe the same library. A line grouped under a
    /// book it says it does not belong to would take its neighbour context from another
    /// book entirely, and the artifact would record a digest for a passage nothing holds.
    #[test]
    fn a_corpus_that_files_a_line_under_the_wrong_book_is_refused() {
        struct Contradictory(JsonlCorpus);
        impl CorpusIndex for Contradictory {
            fn identity(&self) -> Result<CorpusIdentity, PackError> {
                self.0.identity()
            }
            fn expected_line_ids(&self, model: &ModelIdentity) -> Result<BTreeSet<u64>, PackError> {
                self.0.expected_line_ids(model)
            }
            fn line(&self, line_id: u64) -> Result<Option<CorpusLine>, PackError> {
                self.0.line(line_id)
            }
        }
        impl CorpusBooks for Contradictory {
            fn book_keys(&self) -> Result<Vec<String>, PackError> {
                Ok(vec![GENESIS.to_string()])
            }
            /// Claims a line that says it belongs to the other book.
            fn book_line_ids(&self, _book_key: &str) -> Result<Vec<u64>, PackError> {
                Ok(vec![4_294_967_297, 8_589_934_593])
            }
        }

        let dir = TempDir::new("wrong_book");
        let corpus = Contradictory(standard_corpus(&dir));
        let chunking = ChunkerConfig::default();
        let model = model_for(&"ab".repeat(32), &chunking);

        match BuildPlan::compute(&corpus, &chunking, &model) {
            Err(PackError::Corpus { reason }) => {
                assert!(reason.contains("says it belongs to"), "{reason}")
            }
            other => panic!("a line filed under the wrong book must be refused, got {other:?}"),
        }
    }

    /// A book listing an id the corpus has no document for is a corpus contradicting
    /// itself, and is refused rather than skipped: skipping would silently shrink the
    /// coverage contract to whatever happened to resolve.
    #[test]
    fn a_book_listing_a_line_the_corpus_does_not_hold_is_refused() {
        struct Phantom(JsonlCorpus);
        impl CorpusIndex for Phantom {
            fn identity(&self) -> Result<CorpusIdentity, PackError> {
                self.0.identity()
            }
            fn expected_line_ids(&self, model: &ModelIdentity) -> Result<BTreeSet<u64>, PackError> {
                self.0.expected_line_ids(model)
            }
            fn line(&self, line_id: u64) -> Result<Option<CorpusLine>, PackError> {
                self.0.line(line_id)
            }
        }
        impl CorpusBooks for Phantom {
            fn book_keys(&self) -> Result<Vec<String>, PackError> {
                Ok(vec![GENESIS.to_string()])
            }
            fn book_line_ids(&self, _book_key: &str) -> Result<Vec<u64>, PackError> {
                Ok(vec![4_294_967_297, 77])
            }
        }

        let dir = TempDir::new("phantom");
        let corpus = Phantom(standard_corpus(&dir));
        let chunking = ChunkerConfig::default();
        let model = model_for(&"ab".repeat(32), &chunking);

        assert!(matches!(
            BuildPlan::compute(&corpus, &chunking, &model),
            Err(PackError::LineNotInCorpus { line_id: 77 })
        ));
    }

    /// Two books claiming one line. Built, the line would be recorded in both; refusing
    /// here names the corpus, which is where the fault actually is.
    #[test]
    fn two_books_claiming_one_line_are_refused() {
        struct Shared(JsonlCorpus);
        impl CorpusIndex for Shared {
            fn identity(&self) -> Result<CorpusIdentity, PackError> {
                self.0.identity()
            }
            fn expected_line_ids(&self, model: &ModelIdentity) -> Result<BTreeSet<u64>, PackError> {
                self.0.expected_line_ids(model)
            }
            fn line(&self, line_id: u64) -> Result<Option<CorpusLine>, PackError> {
                self.0.line(line_id)
            }
        }
        impl CorpusBooks for Shared {
            fn book_keys(&self) -> Result<Vec<String>, PackError> {
                Ok(vec![GENESIS.to_string(), GENESIS.to_string()])
            }
            fn book_line_ids(&self, _book_key: &str) -> Result<Vec<u64>, PackError> {
                Ok(vec![4_294_967_297])
            }
        }

        let dir = TempDir::new("shared");
        let corpus = Shared(standard_corpus(&dir));
        let chunking = ChunkerConfig::default();
        let model = model_for(&"ab".repeat(32), &chunking);

        assert!(matches!(
            BuildPlan::compute(&corpus, &chunking, &model),
            Err(PackError::DuplicateLineId {
                line_id: 4_294_967_297
            })
        ));
    }

    /// The declaration is held to the file. This is the half of provenance a build machine
    /// can actually establish: a tool handed finished floats can be told anything about the
    /// model, and a tool that loads the model cannot.
    ///
    /// **Two of the five fields are driven here, and the other three cannot be with this
    /// backend** — which is a fact about the stand-in rather than a gap in the check. The
    /// deterministic backend is constructed *from* the configuration and echoes its width
    /// and token cap straight back, so declaring the wrong one configures the backend to
    /// agree; and it always reports `in-graph`, so the only other pooling this crate can
    /// spell is refused earlier, for having no implementation at all. Real inference reads
    /// all three out of the weights, which is where those comparisons acquire teeth. The
    /// checksum and the backend id are not echoed by anything, and are exercised.
    #[test]
    fn the_declared_model_identity_is_held_to_the_model_file() {
        let dir = TempDir::new("model_identity");
        let corpus = standard_corpus(&dir);
        let chunking = ChunkerConfig::default();
        let (_, checksum) = write_model(&dir);
        let truthful = model_for(&checksum, &chunking);

        // The honest declaration builds.
        let report = build(
            build_request(&dir, truthful.clone(), chunking.clone()),
            &corpus,
        )
        .unwrap();
        assert_eq!(report.manifest.counts.slots, 3);

        for (field, wrong) in [
            (
                "query_packages",
                ModelIdentity {
                    query_packages: vec![ModelPackage {
                        checksum: "cd".repeat(32),
                        quantization: "int8".to_string(),
                    }],
                    ..truthful.clone()
                },
            ),
            (
                "tokenizer_checksum",
                ModelIdentity {
                    tokenizer_checksum: "cd".repeat(32),
                    ..truthful.clone()
                },
            ),
        ] {
            let mut request = build_request(&dir, wrong, chunking.clone());
            request.output_path = dir.path().join(format!("artifact_{field}"));
            match build(request, &corpus) {
                Err(PackError::ModelDisagreesWithFile { field: named, .. }) => {
                    assert_eq!(named, field)
                }
                other => panic!("a wrong {field} must be refused, got {other:?}"),
            }
            assert!(
                !dir.path().join(format!("artifact_{field}")).exists(),
                "a rejected build writes nothing"
            );
        }
    }

    /// The cheap refusals come first, and the test for that is which error arrives.
    ///
    /// Both requests below name a model file that does not exist, so a build that reached
    /// the model would say so. Neither does: an unfilled identity field and a recipe that is
    /// not the declared one are both settled from data already in hand, and a long build
    /// must not get as far as loading weights before discovering either.
    #[test]
    fn the_checks_that_need_nothing_happen_before_the_model_is_opened() {
        let dir = TempDir::new("ordering");
        let corpus = standard_corpus(&dir);
        let chunking = ChunkerConfig::default();
        let truthful = model_for(&"ab".repeat(32), &chunking);

        let mut incomplete = build_request(
            &dir,
            ModelIdentity {
                family_id: String::new(),
                ..truthful.clone()
            },
            chunking.clone(),
        );
        incomplete.output_path = dir.path().join("incomplete");
        assert!(
            matches!(
                build(incomplete, &corpus),
                Err(PackError::Artifact(
                    crate::errors::ArtifactError::IncompleteIdentity { .. }
                ))
            ),
            "a blank identity field must be refused before the model is opened"
        );

        let mut wrong_recipe = build_request(&dir, truthful, chunking);
        wrong_recipe.output_path = dir.path().join("wrong_recipe");
        wrong_recipe.chunking = ChunkerConfig {
            min_embeddable_chars: 9,
            ..ChunkerConfig::default()
        };
        assert!(matches!(
            build(wrong_recipe, &corpus),
            Err(PackError::RecipeMismatch { .. })
        ));
    }

    /// **The three recipe versions are claims about this crate's code, and each is refused
    /// on its own.**
    ///
    /// Without this, all three were free labels: a build could declare
    /// `embedding_text_version = 27`, run the only text recipe that exists, and produce an
    /// artifact where every count, checksum and identity field agreed — including with an
    /// installation whose configuration repeated the same 27. The artifact would describe a
    /// recipe nobody had written, and nothing anywhere would notice.
    ///
    /// Each case below names a model file that does not exist, so a build that got as far
    /// as inference would fail differently. None does.
    #[test]
    fn a_recipe_version_this_build_does_not_implement_is_refused_before_the_model_opens() {
        let dir = TempDir::new("recipe_versions");
        let corpus = standard_corpus(&dir);
        let default = ChunkerConfig::default();
        let truthful = model_for(&"ab".repeat(32), &default);

        // `embedding_text_version` lives in both the configuration and the identity, so
        // moving it alone means moving both — the disagreement between them is its own
        // rejection, tested separately in `recipe`.
        // Version 2 of the text recipe exists — the role prefixes — so 3 is the first one
        // nothing implements.
        let text_chunking = ChunkerConfig {
            embedding_text_version: 3,
            ..ChunkerConfig::default()
        };
        let cases: [(&str, ChunkerConfig, ModelIdentity, u32, &str); 3] = [
            (
                "embedding_text_version",
                text_chunking.clone(),
                ModelIdentity {
                    embedding_text_version: 3,
                    chunking_identity: text_chunking.identity(),
                    ..truthful.clone()
                },
                3,
                "1, 2",
            ),
            (
                // Carried in both places, like the text version above, so both move.
                "normalization_version",
                ChunkerConfig {
                    normalization_version: 2,
                    ..ChunkerConfig::default()
                },
                ModelIdentity {
                    normalization_version: 2,
                    chunking_identity: ChunkerConfig {
                        normalization_version: 2,
                        ..ChunkerConfig::default()
                    }
                    .identity(),
                    ..truthful.clone()
                },
                2,
                "1",
            ),
            (
                // Only the configuration carries this one — the artifact carries the hash
                // of the configuration — so the declared identity moves with it.
                "chunking_version",
                ChunkerConfig {
                    chunking_version: 2,
                    ..ChunkerConfig::default()
                },
                ModelIdentity {
                    chunking_identity: ChunkerConfig {
                        chunking_version: 2,
                        ..ChunkerConfig::default()
                    }
                    .identity(),
                    ..truthful.clone()
                },
                2,
                "1",
            ),
        ];

        for (field, chunking, model, unimplemented, implemented) in cases {
            let mut request = build_request(&dir, model, chunking);
            request.output_path = dir.path().join(format!("artifact_{field}"));
            match build(request, &corpus) {
                Err(PackError::Artifact(
                    crate::errors::ArtifactError::UnsupportedRecipeVersion {
                        field: named,
                        found,
                        supported,
                    },
                )) => {
                    assert_eq!(named, field);
                    assert_eq!(found, unimplemented);
                    assert_eq!(supported, implemented);
                }
                other => panic!("{field} {unimplemented} must be refused, got {other:?}"),
            }
            assert!(
                !dir.path().join(format!("artifact_{field}")).exists(),
                "a build refused for {field} writes nothing"
            );
        }
    }

    /// A version carried in two places is a fact that can drift, so the two are compared.
    ///
    /// The chunker needs `embedding_text_version` and `normalization_version` to choose a
    /// code path; an installation compares identities and never holds a configuration. Left
    /// unchecked, a build would apply one recipe and declare another — the same fault
    /// `chunking_identity` guards against, on the two fields that guard is not made of.
    ///
    /// The configuration is the side that moves here, because that is the only reachable
    /// shape: a *declared* version nothing implements is refused earlier, by the identity's
    /// own completeness check.
    #[test]
    fn a_configuration_that_contradicts_the_declared_versions_is_refused() {
        let dir = TempDir::new("recipe_disagreement");
        let corpus = standard_corpus(&dir);

        type Move = (&'static str, fn(&mut ChunkerConfig));
        let moves: [Move; 2] = [
            ("embedding_text_version", |c| c.embedding_text_version = 2),
            ("normalization_version", |c| c.normalization_version = 2),
        ];

        for (field, apply) in moves {
            let mut chunking = ChunkerConfig::default();
            apply(&mut chunking);
            // The declared identity still says 1 for both — and its `chunking_identity` is
            // the moved configuration's, so this is not a hash mismatch either.
            let model = ModelIdentity {
                chunking_identity: chunking.identity(),
                ..model_for(&"ab".repeat(32), &ChunkerConfig::default())
            };

            let mut request = build_request(&dir, model, chunking);
            request.output_path = dir.path().join(format!("artifact_{field}"));
            match build(request, &corpus) {
                Err(PackError::Artifact(
                    crate::errors::ArtifactError::RecipeDisagreesWithIdentity {
                        field: named,
                        configured,
                        declared,
                    },
                )) => {
                    assert_eq!(named, field);
                    assert_eq!((configured, declared), (2, 1));
                }
                other => panic!("a contradicted {field} must be refused, got {other:?}"),
            }
            assert!(!dir.path().join(format!("artifact_{field}")).exists());
        }
    }

    /// Hash vectors are structurally perfect and mean nothing, and no later check can tell.
    /// The refusal has to be here, while the backend is still identifiable.
    #[test]
    fn a_backend_whose_vectors_mean_nothing_is_refused_unless_allowed() {
        let dir = TempDir::new("non_semantic");
        let corpus = standard_corpus(&dir);
        let chunking = ChunkerConfig::default();
        let (_, checksum) = write_model(&dir);
        let model = model_for(&checksum, &chunking);

        let mut request = build_request(&dir, model, chunking);
        request.allow_non_semantic_backend = false;
        match build(request, &corpus) {
            Err(PackError::NonSemanticBackend { backend }) => assert_eq!(backend, "mock-hash-v1"),
            other => panic!("a non-semantic backend must be refused by default, got {other:?}"),
        }
    }

    /// A record's key describes the text the model was actually given — which for a short
    /// line is the line *plus its neighbours*, and for a long one is the line itself.
    /// Deriving it from the corpus line would have keyed every borrowed chunk by a text
    /// nothing was built from.
    #[test]
    fn a_key_describes_the_text_the_model_was_given() {
        let dir = TempDir::new("chunk_hash");
        let corpus = standard_corpus(&dir);
        let chunking = ChunkerConfig::default();
        let (_, checksum) = write_model(&dir);
        let model = model_for(&checksum, &chunking);

        let report = build(build_request(&dir, model, chunking), &corpus).unwrap();
        let records = stored_records(&report);

        assert_eq!(
            records[&(GENESIS.to_string(), 0)].0,
            compute_chunk_hash(LONG),
            "a line that stands alone is embedded as itself"
        );
        let borrowed = &records[&(GENESIS.to_string(), 1)].0;
        assert_ne!(
            *borrowed,
            compute_chunk_hash(BORROWS),
            "a short line is embedded with the context around it, not on its own"
        );
        assert_eq!(
            *borrowed,
            compute_chunk_hash(&format!("{LONG} {BORROWS} {SKIPPED}")),
            "and the context is its section's neighbours, in order"
        );
        assert!(
            !records.contains_key(&(GENESIS.to_string(), 2)),
            "the line the recipe skips has no record"
        );
    }

    /// A text is embedded once, wherever it occurs: its first book in byte order holds the
    /// slot, a later book an extra record of it, and a second line of the same book with
    /// the same text no record at all — a record is a book and a key.
    #[test]
    fn a_shared_text_is_embedded_once_and_recorded_once_per_book() {
        let dir = TempDir::new("shared_text");
        let corpus = write_corpus(
            &dir,
            &[
                (4_294_967_297, GENESIS, LONG, 1),
                (4_294_967_298, GENESIS, LONG_TWO, 2),
                (4_294_967_299, GENESIS, LONG, 3),
                (8_589_934_593, BERACHOT, LONG, 1),
            ],
        );
        let chunking = ChunkerConfig::default();
        let (_, checksum) = write_model(&dir);
        let model = model_for(&checksum, &chunking);

        let report = build(build_request(&dir, model, chunking), &corpus).unwrap();
        let counts = report.manifest.counts;
        assert_eq!((counts.books, counts.slots, counts.extras), (2, 2, 1));
        assert_eq!(report.planned_lines, 4);

        let records = stored_records(&report);
        let long = compute_chunk_hash(LONG);
        assert_eq!(
            records.into_iter().collect::<Vec<_>>(),
            vec![
                ((BERACHOT.to_string(), 0), (long.clone(), true)),
                ((GENESIS.to_string(), 0), (long, false)),
                (
                    (GENESIS.to_string(), 1),
                    (compute_chunk_hash(LONG_TWO), true)
                ),
            ],
            "berachot sorts first and takes the slot; genesis records it once, at its first line"
        );
    }

    /// The stage's own claim, end to end and without a hand-built fixture anywhere: a
    /// corpus and a model in, a package out that installs, opens and finds what it holds.
    #[test]
    fn a_built_package_installs_and_answers_with_its_own_vectors() {
        let dir = TempDir::new("end_to_end");
        let corpus = standard_corpus(&dir);
        let chunking = ChunkerConfig::default();
        let (model_path, checksum) = write_model(&dir);
        let model = model_for(&checksum, &chunking);

        let report = build(build_request(&dir, model.clone(), chunking), &corpus).unwrap();
        let counts = report.manifest.counts;
        assert_eq!((counts.books, counts.slots, counts.extras), (2, 3, 0));
        assert_eq!(report.manifest.to_library_version, 30);
        assert_eq!(report.clipped_components, 0, "clip_q 1 clips nothing");
        assert_eq!(
            report.manifest.provenance.passage_package.checksum, checksum,
            "the package that embedded the passages is on record"
        );

        // The package around the segment is the one the manifest names.
        let package = IndexPackage::read(&report.output_path).unwrap();
        package.verify_integrity(&report.output_path).unwrap();
        assert_eq!(package.digest(), report.manifest.package_digest);

        // Installed, against the digest published for it.
        let json =
            std::fs::read_to_string(report.output_path.join(RELEASE_MANIFEST_FILENAME)).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(json.as_bytes())),
            report.manifest_sha256
        );
        let vectors = dir.path().join("vectors");
        let applied = install_package(
            &vectors,
            &InstallSource {
                segment: &report.output_path.join(SEGMENT_FILENAME),
                manifest_json: &json,
            },
            &InstallExpectation {
                identity: report.manifest.identity.clone(),
                published_manifest_sha256: Some(report.manifest_sha256.clone()),
            },
            &CancellationToken::new(),
        )
        .unwrap();
        assert_eq!(applied.slots_added, 3);

        // Each stored text, embedded again as a query would be, finds its own record first.
        let set = SegmentSet::open(&vectors).unwrap();
        let mut runtime = EmbeddingRuntime::new(EmbeddingConfig {
            model_path,
            embedding_dim: DIM,
            max_tokens: 512,
            batch_size: 1,
            pooling: Pooling::InGraph,
        });
        runtime.load().unwrap();
        for (book, hint, text) in [(GENESIS, 0, LONG), (BERACHOT, 0, LONG_TWO)] {
            let hits = set
                .scan(
                    &runtime.embed_one(text).unwrap(),
                    &ScanRequest {
                        top_k: 3,
                        books: None,
                        threads: 1,
                    },
                    &CancellationToken::new(),
                )
                .unwrap();
            assert_eq!(hits.len(), 3);
            assert_eq!(hits[0].key, ChunkKey::of(text));
            assert_eq!(
                (&*hits[0].records[0].book, hits[0].records[0].hint),
                (book, hint)
            );
            assert!(hits[0].score > 0.99, "{}", hits[0].score);
        }

        // A second build into a used directory is refused rather than writing over it.
        let again = build(
            build_request(&dir, model, ChunkerConfig::default()),
            &corpus,
        );
        assert!(
            matches!(again, Err(PackError::UnusableOutput { .. })),
            "{again:?}"
        );
    }

    /// Two builds of one corpus are the same bytes: the slot order, the calibration and the
    /// rounding are all deterministic.
    #[test]
    fn two_builds_of_one_corpus_write_the_same_segment() {
        let dir = TempDir::new("deterministic");
        let corpus = standard_corpus(&dir);
        let chunking = ChunkerConfig::default();
        let (_, checksum) = write_model(&dir);
        let model = model_for(&checksum, &chunking);

        let first = build(
            build_request(&dir, model.clone(), chunking.clone()),
            &corpus,
        )
        .unwrap();
        let mut request = build_request(&dir, model, chunking);
        request.output_path = dir.path().join("again");
        let second = build(request, &corpus).unwrap();
        assert_eq!(first.manifest.segment, second.manifest.segment);
        assert_eq!(
            first.manifest.package_digest,
            second.manifest.package_digest
        );
        assert_eq!(first.manifest_sha256, second.manifest_sha256);
    }
}
