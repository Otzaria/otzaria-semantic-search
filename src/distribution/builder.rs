//! S4b in this crate: a corpus and a model in, a verified artifact out.
//!
//! [`packer`](crate::distribution::packer) takes finished floats and can only check that
//! they line up with the corpus. This module produces them — it opens the model, applies
//! the recipe to the corpus, embeds the text it derived, and hands the results to the same
//! packer. Everything the packer already enforces still runs; what changes is who the
//! producer is.
//!
//! # What that closes
//!
//! S4a's two digests are an alignment check and nothing more: a producer that hashed the
//! corpus at pack time rather than at embedding time satisfies both, and no tool receiving
//! finished vectors can tell. Here the vector, `embedding_text_sha256` and the model
//! identity come out of one pass over one model:
//!
//! * the text that is hashed is the same `String` that is handed to the backend, in the
//!   same expression — there is no path by which one can describe the other;
//! * `model_checksum`, `embedding_backend`, `embedding_dim`, `pooling` and the effective
//!   `max_tokens` are **reported by the loaded runtime** and compared against what the
//!   artifact declares, so the declaration is a checked claim rather than a copied string.
//!
//! "Reported by the runtime" rather than "read from the file", because they are not all the
//! same kind of fact: the checksum is of the bytes on disk; the width and the token cap are
//! what the weights actually carry; the backend id is which implementation was selected;
//! and pooling is what that implementation performs. All five are settled by the thing that
//! is about to produce the vectors, which is what makes comparing them worth anything —
//! but only the first three are properties of the file.
//!
//! The three **recipe versions** — `embedding_text_version`, `normalization_version` and
//! `chunking_version` — are not properties of a model at all. They are versions of code in
//! this crate, so they are settled outright rather than compared:
//! [`EmbeddingRecipe::resolve`] refuses any of them that names behaviour nobody has
//! written, and the code that does the work dispatches on the resolved value. See
//! [`recipe`](crate::semantic::recipe).
//!
//! What is left declared and unverifiable: `model_id` and `model_quantization`. Nothing in
//! a GGUF file states either, and inventing a check that reads them from the same place
//! that wrote them would prove nothing.
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
//! chunker configuration **before a single vector exists**. It is then what
//! [`PlannedCorpus`] answers `expected_line_ids` with, so the packer's coverage check
//! compares what was produced against what was intended.
//!
//! Deriving that set from the vectors instead would make the check confirm itself: a batch
//! that died halfway, or a book the corpus stopped answering for, would disappear from both
//! sides at once and the artifact would be certified complete for the subset that happened
//! to survive.
//!
//! # The recipe cannot be guessed, so it is supplied and pinned
//!
//! `chunking_identity` in an artifact is a one-way hash of a whole [`ChunkerConfig`]. A
//! build is therefore handed the configuration itself, and [`BuildPlan::compute`] refuses
//! one whose identity is not the value the artifact will declare. Without that, the model
//! identity would be a label on a recipe nobody applied.
//!
//! # What it costs
//!
//! Each line is chunked twice — once to plan, once to embed — and read from the corpus
//! twice with it. Holding every embedded text between the two passes would cost more than
//! recomputing it, and the second read is not waste: it is what proves the corpus still
//! says what the plan assumed. The packer then reads each line a third time to join the
//! finished vector back. All three are build-machine costs, and none is on a device.

use crate::distribution::corpus::{CorpusBooks, CorpusIndex, CorpusLine};
use crate::distribution::packer::{
    compose_identity, ensure_output_is_free, pack, PackReport, PackRequest, VectorInput,
};
use crate::errors::PackError;
use crate::semantic::backend::Pooling;
use crate::semantic::chunker::{Chunker, ChunkerConfig};
use crate::semantic::embedding::{EmbeddingConfig, EmbeddingRuntime};
use crate::semantic::recipe::EmbeddingRecipe;
use crate::semantic::types::{BookForIndexing, BookLine, SemanticChunk};
use crate::semantic::versioning::{CorpusIdentity, ModelIdentity};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::path::PathBuf;

/// What to build, beyond the corpus.
#[derive(Debug, Clone)]
pub struct BuildRequest {
    /// Directory the artifact is written into. Must not exist, or be an empty directory.
    pub output_path: PathBuf,
    /// The GGUF file the vectors are produced with. Its checksum has to be the one
    /// `model.model_checksum` declares.
    pub model_path: PathBuf,
    /// What the artifact will declare about how its vectors were made. The half a model
    /// file can answer for is checked against it; see the module documentation for the
    /// half that cannot be.
    pub model: ModelIdentity,
    /// The recipe itself. `model.chunking_identity` is its hash, and a disagreement is
    /// [`PackError::RecipeMismatch`].
    pub chunking: ChunkerConfig,
    pub created_at: String,
    pub collection_name: String,
    /// Texts per inference call. Also the granularity at which vectors reach the packer,
    /// so it bounds what a build holds beyond the payload itself.
    pub batch_size: usize,
    /// Permit a backend whose vectors carry no meaning.
    ///
    /// `false` in anything that ships. The deterministic stand-in produces vectors that
    /// are structurally perfect and semantically empty, and an artifact built from them
    /// passes every check in this crate — so the refusal has to be here, where the backend
    /// is still identifiable, rather than downstream where it is not.
    pub allow_non_semantic_backend: bool,
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
    chunking_identity: u64,
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
        let chunking_identity = ensure_recipe_matches(chunking, model)?;
        let chunker = Chunker::new(chunking.clone())?;
        let mut line_ids = BTreeSet::new();
        let mut books = 0usize;
        for book_key in corpus.book_keys()? {
            books += 1;
            for chunk in chunks_for_book(corpus, &chunker, &book_key)? {
                // Two books claiming one line, or one book listing it twice. The packer
                // would reject it later as a duplicate; saying so here names the recipe's
                // input instead of the vector stream, which is where the fault is.
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
        Ok(Self {
            chunking_identity,
            line_ids,
        })
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

/// A corpus seen through the recipe that will be applied to it.
///
/// The only thing it changes is the answer to [`CorpusIndex::expected_line_ids`]: the
/// underlying corpus reports which lines it *holds*, and this reports which lines the
/// recipe *embeds*. Those are different questions, and only the second one is coverage —
/// a line too short to carry meaning exists perfectly well and must not get a vector.
///
/// It is also what lets a corpus with no opinion on the recipe — the JSONL transcription,
/// which records a recipe that was already applied — be packed against one anyway. Wrap it,
/// and the expected set is derived rather than assumed.
pub struct PlannedCorpus<'a> {
    corpus: &'a dyn CorpusBooks,
    plan: BuildPlan,
}

impl<'a> PlannedCorpus<'a> {
    pub fn new(
        corpus: &'a dyn CorpusBooks,
        chunking: &ChunkerConfig,
        model: &ModelIdentity,
    ) -> Result<Self, PackError> {
        let plan = BuildPlan::compute(corpus, chunking, model)?;
        Ok(Self { corpus, plan })
    }

    pub fn plan(&self) -> &BuildPlan {
        &self.plan
    }
}

impl CorpusIndex for PlannedCorpus<'_> {
    fn identity(&self) -> Result<CorpusIdentity, PackError> {
        self.corpus.identity()
    }

    /// The plan's ids — but only for the recipe the plan was computed for.
    ///
    /// The model reaches this method a second time, from inside the packer, and it need not
    /// be the one the plan was built with: a caller can wrap a corpus once and pack twice.
    /// Answering anyway would be certifying coverage for a recipe that was never applied,
    /// which is exactly what the parameter exists to prevent.
    fn expected_line_ids(&self, model: &ModelIdentity) -> Result<BTreeSet<u64>, PackError> {
        if model.chunking_identity != self.plan.chunking_identity {
            return Err(PackError::RecipeMismatch {
                declared: model.chunking_identity,
                actual: self.plan.chunking_identity,
            });
        }
        Ok(self.plan.line_ids.clone())
    }

    fn line(&self, line_id: u64) -> Result<Option<CorpusLine>, PackError> {
        self.corpus.line(line_id)
    }
}

impl CorpusBooks for PlannedCorpus<'_> {
    fn book_keys(&self) -> Result<Vec<String>, PackError> {
        self.corpus.book_keys()
    }

    fn book_line_ids(&self, book_key: &str) -> Result<Vec<u64>, PackError> {
        self.corpus.book_line_ids(book_key)
    }
}

/// The declared `chunking_identity` and the recipe in hand are the same recipe.
///
/// Hoisted out of [`BuildPlan::compute`] so [`build`] can ask it before opening a model or
/// reading a corpus: it costs one hash of five integers, and it is the check most likely to
/// fail on a misconfigured build.
fn ensure_recipe_matches(
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

/// Build an official artifact from a corpus and a model.
///
/// The order is cost, not taste — each step is the cheapest way to fail that is still
/// available:
///
/// 1. Refuse an output path that already holds something. Nothing else is worth doing if
///    the result cannot be written.
/// 2. Refuse an identity with a field left unfilled — one read of the corpus identity, and
///    it is fatal at the end of a build just as surely as at the start.
/// 3. Refuse a recipe that is not the declared one: five integers hashed, no model and no
///    corpus.
/// 4. Load the model, and hold the declared identity to what the file reports — before six
///    million lines are chunked rather than after.
/// 5. Refuse a backend whose vectors are not semantic, unless the caller has said otherwise
///    in as many words.
/// 6. Plan: apply the recipe to the whole corpus.
/// 7. Embed, book by book, and hand the results to [`pack`], which performs every check S4a
///    performs — including comparing the ids produced against the plan.
///
/// Steps 1–6 write nothing. Step 7 writes nothing until the payload commits, so any
/// rejection up to that point leaves the output directory empty and the run can simply be
/// repeated.
pub fn build(request: BuildRequest, corpus: &dyn CorpusBooks) -> Result<PackReport, PackError> {
    ensure_output_is_free(&request.output_path)?;
    compose_identity(corpus, &request.model)?;
    let recipe = EmbeddingRecipe::resolve(&request.chunking, &request.model)?;
    ensure_recipe_matches(&request.chunking, &request.model)?;
    let runtime = load_model(&request, recipe)?;

    let planned = PlannedCorpus::new(corpus, &request.chunking, &request.model)?;
    log::info!(
        "The recipe embeds {} line(s) of {}",
        planned.plan().len(),
        request.output_path.display()
    );

    let inputs = PlannedEmbeddings::new(&planned, &runtime, &request.chunking, request.batch_size)?;

    pack(
        PackRequest {
            output_path: request.output_path,
            model: request.model,
            created_at: request.created_at,
            collection_name: request.collection_name,
        },
        inputs,
        &planned,
    )
}

/// Open the model and prove it is the one the artifact will name.
///
/// The runtime refuses a width or a pooling that disagrees with the loaded backend before
/// this returns, so two of the five comparisons below can only fail through it. They are
/// listed anyway: this table is the statement of what the loaded runtime can be held to,
/// and leaving a field out of it because something else happens to cover it today is how
/// such a check quietly stops covering it.
///
/// Not all five are facts about the *file* — see the module header. The checksum is; the
/// backend id is which implementation was selected for it.
fn load_model(
    request: &BuildRequest,
    recipe: EmbeddingRecipe,
) -> Result<EmbeddingRuntime, PackError> {
    let declared = &request.model;
    let mut runtime = EmbeddingRuntime::new(EmbeddingConfig {
        model_path: request.model_path.clone(),
        embedding_dim: declared.embedding_dim,
        max_tokens: declared.max_tokens,
        batch_size: request.batch_size,
        pooling: Pooling::parse(&declared.pooling)?,
        // Resolved from `normalization_version`, so the runtime performs the strategy the
        // artifact will declare rather than the only one that happens to be written.
        normalization: recipe.normalization,
    });
    runtime.load()?;

    for (field, declared, loaded) in [
        (
            "model_checksum",
            declared.model_checksum.clone(),
            runtime.model_checksum().unwrap_or_default().to_string(),
        ),
        (
            "embedding_backend",
            declared.embedding_backend.clone(),
            runtime.backend_id().unwrap_or_default().to_string(),
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

    if !runtime.backend_is_semantic() && !request.allow_non_semantic_backend {
        return Err(PackError::NonSemanticBackend {
            backend: runtime.backend_id().unwrap_or("none").to_string(),
        });
    }

    Ok(runtime)
}

/// The recipe applied to one book: the corpus's lines in corpus order, chunked.
///
/// Assembling a [`BookForIndexing`] and calling the real [`Chunker`] is the point — the
/// recipe has exactly one implementation, and a build applies *that* one rather than a
/// second reading of it.
///
/// The book-level metadata below reaches nothing that is stored: a chunk's title, facets
/// and content hash are discarded here, and the packer reads every stored field from
/// [`CorpusIndex::line`] instead. What the chunker actually consumes is each line's text,
/// its `section_id` and its position in the list.
fn chunks_for_book(
    corpus: &dyn CorpusBooks,
    chunker: &Chunker,
    book_key: &str,
) -> Result<Vec<SemanticChunk>, PackError> {
    let line_ids = corpus.book_line_ids(book_key)?;
    let mut lines = Vec::with_capacity(line_ids.len());
    /// The book-level fields, taken from whichever line comes first. None of them reaches
    /// the artifact, so a book whose lines disagreed about its title would still be
    /// described by [`CorpusIndex::line`] alone when the packer builds the records.
    struct BookFields {
        title: String,
        content_hash: u64,
        is_pdf: bool,
        facets: Vec<String>,
    }
    let mut book_fields: Option<BookFields> = None;

    for line_id in line_ids {
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

    Ok(chunker.chunk_book(&BookForIndexing {
        source_book_key: book_key.to_string(),
        title: book.title,
        content_fingerprint: book.content_hash,
        is_pdf: book.is_pdf,
        topics: String::new(),
        extra_facets: book.facets,
        lines,
    }))
}

/// The corpus, chunked and embedded, one batch at a time.
///
/// An iterator rather than a `Vec` because the packer consumes one at a time, so a build
/// never holds a second complete copy of the payload the writer is accumulating.
///
/// What it does hold, stated rather than implied: **one book's chunks and one batch of
/// vectors.** A book is chunked whole because neighbour context needs the lines around a
/// line, so `pending` is as large as the longest book in the corpus — that is the number to
/// measure, not the batch size. Both are inside the same S2b measurement as the payload
/// writer, and neither is on a device.
///
/// The first error ends the stream. Continuing after one would mean the packer sees a
/// truncated set and reports a coverage mismatch — a second, louder symptom of a fault
/// already named precisely.
struct PlannedEmbeddings<'a> {
    corpus: &'a dyn CorpusBooks,
    runtime: &'a EmbeddingRuntime,
    chunker: Chunker,
    batch_size: usize,
    books: std::vec::IntoIter<String>,
    /// Chunks of the book being worked through, still to embed.
    pending: std::vec::IntoIter<SemanticChunk>,
    /// Embedded and ready to hand over.
    ready: std::vec::IntoIter<VectorInput>,
    failed: bool,
}

impl<'a> PlannedEmbeddings<'a> {
    fn new(
        corpus: &'a dyn CorpusBooks,
        runtime: &'a EmbeddingRuntime,
        chunking: &ChunkerConfig,
        batch_size: usize,
    ) -> Result<Self, PackError> {
        Ok(Self {
            corpus,
            runtime,
            chunker: Chunker::new(chunking.clone())?,
            // A batch of zero would embed nothing forever.
            batch_size: batch_size.max(1),
            books: corpus.book_keys()?.into_iter(),
            pending: Vec::new().into_iter(),
            ready: Vec::new().into_iter(),
            failed: false,
        })
    }

    /// Fill [`Self::ready`] from the next batch that has one, or report the corpus is
    /// exhausted.
    fn refill(&mut self) -> Result<bool, PackError> {
        loop {
            let batch: Vec<SemanticChunk> = self.pending.by_ref().take(self.batch_size).collect();
            if !batch.is_empty() {
                self.ready = self.embed(batch)?.into_iter();
                return Ok(true);
            }
            let Some(book_key) = self.books.next() else {
                return Ok(false);
            };
            self.pending = chunks_for_book(self.corpus, &self.chunker, &book_key)?.into_iter();
        }
    }

    /// Embed one batch, and pair each vector with the digests of the two texts it came
    /// from.
    ///
    /// `embedding_text` is hashed and handed to the backend in the same breath, which is
    /// what makes the recorded `chunk_hash` describe the text that was actually embedded
    /// rather than a text that was merely available. `anchor_text` is the corpus line
    /// verbatim, so its digest is what the packer will compare against a fresh read of the
    /// corpus.
    fn embed(&self, batch: Vec<SemanticChunk>) -> Result<Vec<VectorInput>, PackError> {
        let texts: Vec<&str> = batch
            .iter()
            .map(|chunk| chunk.embedding_text.as_str())
            .collect();
        let vectors = self.runtime.embed_batch(&texts)?;

        // The runtime already refuses a short batch from a backend; this is the same
        // invariant one layer up, where `zip` would silently drop the tail instead.
        if vectors.len() != batch.len() {
            return Err(PackError::MalformedInput {
                reason: format!(
                    "{} vector(s) came back for {} text(s)",
                    vectors.len(),
                    batch.len()
                ),
            });
        }

        Ok(batch
            .into_iter()
            .zip(vectors)
            .map(|(chunk, vector)| VectorInput {
                line_id: chunk.line_id,
                source_line_sha256: sha256_hex(chunk.anchor_text.as_bytes()),
                embedding_text_sha256: sha256_hex(chunk.embedding_text.as_bytes()),
                vector,
            })
            .collect())
    }
}

impl Iterator for PlannedEmbeddings<'_> {
    type Item = Result<VectorInput, PackError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.failed {
                return None;
            }
            if let Some(input) = self.ready.next() {
                return Some(Ok(input));
            }
            match self.refill() {
                Ok(true) => continue,
                Ok(false) => return None,
                Err(error) => {
                    self.failed = true;
                    return Some(Err(error));
                }
            }
        }
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::corpus::{CorpusLineRecord, JsonlCorpus};
    use crate::distribution::package::{ArtifactExpectation, IndexPackage};
    use crate::semantic::chunker::compute_chunk_hash;
    use crate::semantic::embedding::{mock, validate_and_checksum_gguf};
    use crate::semantic::versioning::IndexVersion;
    use crate::semantic::zevc_store::ReadOnlyZevcStore;
    use std::collections::HashMap;
    use std::path::Path;

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
            let path = std::env::temp_dir().join(format!(
                "otzaria_builder_{name}_{}",
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

    fn corpus_identity() -> CorpusIdentity {
        CorpusIdentity {
            corpus_id: "5c".repeat(32),
            library_version: "otzaria-library-2026-08".to_string(),
            tantivy_schema_version: 3,
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

    /// A stub GGUF and the checksum a build must declare for it.
    fn write_model(dir: &TempDir) -> (PathBuf, String) {
        let path = dir.path().join("model.gguf");
        mock::write_stub_gguf(&path, 3).unwrap();
        let checksum = validate_and_checksum_gguf(&path).unwrap();
        (path, checksum)
    }

    fn build_request(dir: &TempDir, model: ModelIdentity, chunking: ChunkerConfig) -> BuildRequest {
        let (model_path, _) = (dir.path().join("model.gguf"), ());
        BuildRequest {
            output_path: dir.path().join("artifact"),
            model_path,
            model,
            chunking,
            created_at: "2026-08-08T00:00:00Z".to_string(),
            collection_name: "chunks".to_string(),
            batch_size: 2,
            allow_non_semantic_backend: true,
        }
    }

    /// Every stored record of an artifact, keyed by `line_id`.
    fn stored_records(
        path: &Path,
        model: &ModelIdentity,
    ) -> HashMap<u64, crate::semantic::types::VectorMetadata> {
        let identity = IndexVersion {
            corpus: corpus_identity(),
            model: model.clone(),
            store: crate::semantic::official_index::readable_store_identity(),
        };
        let verified = IndexPackage::verify_for_install(
            path,
            &ArtifactExpectation::without_published_digest(identity),
        )
        .unwrap();
        let store = ReadOnlyZevcStore::open(&verified).unwrap();
        store
            .stored_metadata()
            .map(|record| (record.line_id, record.clone()))
            .collect()
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

    /// A wrapped corpus is not a corpus that answers for any recipe. The packer asks it a
    /// second time, with whatever model the caller passed there, and answering that with a
    /// set derived from a different one would certify coverage nobody built.
    #[test]
    fn a_planned_corpus_answers_only_for_the_recipe_it_planned() {
        let dir = TempDir::new("planned_model");
        let corpus = standard_corpus(&dir);
        let chunking = ChunkerConfig::default();
        let model = model_for(&"ab".repeat(32), &chunking);
        let planned = PlannedCorpus::new(&corpus, &chunking, &model).unwrap();

        assert_eq!(planned.expected_line_ids(&model).unwrap().len(), 3);
        assert_eq!(planned.identity().unwrap(), corpus_identity());
        assert_eq!(
            planned.line(4_294_967_297).unwrap().unwrap().text,
            LONG,
            "the wrapper changes the expected set and nothing else"
        );

        let other = ModelIdentity {
            chunking_identity: model.chunking_identity ^ 1,
            ..model.clone()
        };
        assert!(matches!(
            planned.expected_line_ids(&other),
            Err(PackError::RecipeMismatch { .. })
        ));
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

    /// Two books claiming one line. The packer would refuse the duplicate vector later;
    /// refusing here names the corpus, which is where the fault actually is.
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
    /// agree; and it always reports `last-token`, so the only other pooling this crate can
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
        assert_eq!(report.vector_count, 3);

        for (field, wrong) in [
            (
                "model_checksum",
                ModelIdentity {
                    model_checksum: "cd".repeat(32),
                    ..truthful.clone()
                },
            ),
            (
                "embedding_backend",
                ModelIdentity {
                    embedding_backend: "llama-cpp-2-0.1.153".to_string(),
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
                model_id: String::new(),
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
        let text_chunking = ChunkerConfig {
            embedding_text_version: 2,
            ..ChunkerConfig::default()
        };
        let cases: [(&str, ChunkerConfig, ModelIdentity); 3] = [
            (
                "embedding_text_version",
                text_chunking.clone(),
                ModelIdentity {
                    embedding_text_version: 2,
                    chunking_identity: text_chunking.identity(),
                    ..truthful.clone()
                },
            ),
            (
                "normalization_version",
                default.clone(),
                ModelIdentity {
                    normalization_version: 2,
                    ..truthful.clone()
                },
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
            ),
        ];

        for (field, chunking, model) in cases {
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
                    assert_eq!(found, 2);
                    assert_eq!(supported, "1");
                }
                other => panic!("{field} 2 must be refused, got {other:?}"),
            }
            assert!(
                !dir.path().join(format!("artifact_{field}")).exists(),
                "a build refused for {field} writes nothing"
            );
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

    /// The record's `chunk_hash` describes the text the model was actually given — which
    /// for a short line is the line *plus its neighbours*, and for a long one is the line
    /// itself. Deriving it from the corpus line would have made every borrowed chunk record
    /// a digest of a text nothing was built from.
    #[test]
    fn a_chunk_hash_describes_the_text_the_model_was_given() {
        let dir = TempDir::new("chunk_hash");
        let corpus = standard_corpus(&dir);
        let chunking = ChunkerConfig::default();
        let (_, checksum) = write_model(&dir);
        let model = model_for(&checksum, &chunking);

        let report = build(build_request(&dir, model.clone(), chunking), &corpus).unwrap();
        let records = stored_records(&report.artifact_path, &model);

        assert_eq!(
            records[&4_294_967_297].chunk_hash,
            compute_chunk_hash(LONG),
            "a line that stands alone is embedded as itself"
        );
        let borrowed = &records[&4_294_967_298].chunk_hash;
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
    }

    /// The stage's own claim, end to end and without a hand-built fixture anywhere: a
    /// corpus and a model in, an artifact out that verifies against the same corpus.
    #[test]
    fn a_built_artifact_verifies_against_the_corpus_it_was_built_from() {
        let dir = TempDir::new("end_to_end");
        let corpus = standard_corpus(&dir);
        let chunking = ChunkerConfig::default();
        let (_, checksum) = write_model(&dir);
        let model = model_for(&checksum, &chunking);

        let report = build(
            build_request(&dir, model.clone(), chunking.clone()),
            &corpus,
        )
        .unwrap();
        assert_eq!(report.vector_count, 3);
        assert_eq!(report.book_count, 2);

        // Independently, through the packer's own entry point and the same wrapper the
        // build used: the artifact covers the lines the recipe embeds, and no others.
        let planned = PlannedCorpus::new(&corpus, &chunking, &model).unwrap();
        let revalidated =
            crate::distribution::packer::validate_artifact(&report.artifact_path, &model, &planned)
                .unwrap();
        assert_eq!(revalidated.digest, report.digest);

        // Against the unwrapped transcription it is *incomplete*, because that corpus
        // reports every line it holds — including the one the recipe skips.
        match crate::distribution::packer::validate_artifact(&report.artifact_path, &model, &corpus)
        {
            Err(PackError::CoverageMismatch {
                missing,
                unexpected,
                first_missing,
                ..
            }) => {
                assert_eq!((missing, unexpected), (1, 0));
                assert_eq!(first_missing, Some(4_294_967_299));
            }
            other => panic!("the skipped line must show up as missing, got {other:?}"),
        }
    }
}
