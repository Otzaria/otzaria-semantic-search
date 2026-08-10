//! Binary CLI interface for `otzaria-semantic-search`.
//!
//! Provides a standalone command-line application for querying, indexing, and
//! inspecting the Otzaria semantic search engine, and the build-side `build`, `pack` and
//! `validate` commands that produce an official artifact.
//!
//! `pack` and `validate` need no embedding backend — they never turn text into a vector —
//! so they work in a default build, which is the one a release pipeline has. `build` does
//! turn text into vectors, and so needs one compiled in.

use otzaria_semantic_search::api::hybrid_search::{OtzariaHybridEngine, SearchRequest};
use otzaria_semantic_search::distribution::builder::{build, BuildRequest, PlannedCorpus};
use otzaria_semantic_search::distribution::corpus::{CorpusIndex, JsonlCorpus};
use otzaria_semantic_search::distribution::package::IndexPackage;
use otzaria_semantic_search::distribution::packer::{
    pack, read_vector_inputs, validate_artifact, PackReport, PackRequest,
};
use otzaria_semantic_search::distribution::reuse::{
    assemble, ledger_from_artifact, plan_split, verify_shards, LedgerManifest, ReuseEntry,
    VerifiedBase,
};
use otzaria_semantic_search::distribution::shard::{
    embed_shard, export_plan, read_plan, ShardReport,
};
use otzaria_semantic_search::hybrid::coordinator::HybridCoordinator;
use otzaria_semantic_search::semantic::backend::Pooling;
use otzaria_semantic_search::semantic::chunker::ChunkerConfig;
use otzaria_semantic_search::semantic::embedding::{EmbeddingConfig, EmbeddingRuntime};
use otzaria_semantic_search::semantic::engine::{SemanticConfig, SemanticEngine};
use otzaria_semantic_search::semantic::types::{BookForIndexing, BookLine, SearchMode};
use otzaria_semantic_search::semantic::versioning::ModelIdentity;
use otzaria_semantic_search::semantic::zevc_store::VECTORS_FILENAME;
use std::env;
use std::path::{Path, PathBuf};
use std::process;
use std::time::{SystemTime, UNIX_EPOCH};

fn print_usage() {
    println!(
        r#"otzaria-semantic-search CLI v{}

Usage:
  otzaria-semantic-search <command> [options]

Commands:
  version                             Display version and format information.
  status [--dir <path>]               Display search engine index and model status.
  search <query> [options]            Execute a search query against the engine.
  index-text <key> <title> <text>     Index a plain-text book into the database.
  build [options]                     Embed a corpus and write an official artifact.
  export-plan [options]               Apply the recipe and write the work out, for a
                                      machine that will embed it elsewhere.
  embed-shard [options]               Embed one window of an exported plan.
  plan-split [options]                Split a plan against a previous release's ledger.
  assemble [options]                  Gather reused and freshly embedded vectors into one
                                      pair of files.
  ledger [options]                    Build the next release's reuse ledger from a packed
                                      artifact.
  pack [options]                      Build an official artifact from ready-made vectors.
  validate [options]                  Verify an artifact against a corpus and a model.

Options for 'search':
  --dir <path>       Directory holding semantic database (default: "./semantic_db")
  --mode <mode>      Retrieval mode: hybrid (default), semantic, lexical
  --limit <N>        Maximum results to return (default: 10)

Options for 'status':
  --dir <path>       Directory holding semantic database (default: "./semantic_db")

Options for 'build':
  --corpus-identity <path>   JSON CorpusIdentity, as the lexical index reports it
  --corpus-lines <path>      JSONL, one corpus line per document
  --model <path>             JSON ModelIdentity describing how the vectors are produced
  --model-file <path>        The GGUF the vectors are produced with
  --chunking <path>          JSON ChunkerConfig — the recipe itself (see below)
  --out <dir>                Output directory; must not exist, or be empty
  --batch <N>                Texts per inference call (default: 32)
  --collection <name>        Collection name in the payload header (default: "chunks")
  --created-at <timestamp>   Manifest timestamp (default: now, UTC)
  --allow-non-semantic       Write an artifact from a backend whose vectors mean nothing.
                             For tests only: such an artifact passes every check here and
                             answers nonsense.

Options for 'export-plan':
  --corpus-identity <path>   As for 'build'
  --corpus-lines <path>      As for 'build'
  --model <path>             As for 'build'; no model file is opened, and none is needed
  --chunking <path>          The recipe to apply
  --out <dir>                Receives plan.jsonl and export-manifest.json

Options for 'embed-shard':
  --plan <path>              plan.jsonl, as 'export-plan' wrote it
  --model <path>             The identity the plan was exported under
  --model-file <path>        The GGUF; held to every field the identity declares
  --skip <N>                 Records to skip (default: 0)
  --take <N>                 Records to embed (default: all that remain)
  --batch <N>                Texts per inference call (default: 32)
  --out <dir>                Receives vectors.f32, records.jsonl, shard-manifest.json
  --allow-non-semantic       As for 'build'

Options for 'plan-split':
  --plan <path>              plan.jsonl for the release being built
  --ledger <path>            ledger.jsonl of the release to reuse from. Omit it for a
                             full baseline: every line then goes to the GPU.
  --out <dir>                Receives reuse.jsonl and embed.jsonl

Options for 'assemble':
  --shards <dir>             Root holding every shard's output, at any depth
  --plan-sha256 <hex>        The digest every shard's manifest must name
  --embed-records <N>        How many records the shards must cover, with no hole
  --model <path>             The identity every shard must have been embedded under, and
                             the width every file is strided by. There is no --dim: a
                             width the model does not declare is not a width.
  --reuse <path>             reuse.jsonl from 'plan-split'; omit for a full baseline
  --base-vectors <path>      vectors.f32 of the release being reused from
  --out <dir>                Receives vectors.f32 and records.jsonl

Options for 'ledger':
  --artifact <dir>           A packed artifact; its metadata.jsonl order is the payload's
  --records <path>           records.jsonl from 'assemble' — the only place the full
                             64-hex digest exists, since the payload stores 32
  --artifact-digest <hex>    Optional. The digest published outside the artifact; it is
                             compared against the computed one, never copied into the
                             manifest. The model identity comes from the package.
  --out <path>               ledger.jsonl; the manifest is written beside it

Build the ledger from the artifact and never from 'assemble': packing sorts the payload by
semantic_id, so the assembler's order is not the published one.

Options for 'plan-split' (continued):
  --ledger-manifest <path>   Required with --ledger. Names the artifact digest, the
                             vectors.bin digest and the model identity the offsets mean
                             something under.
  --model <path>             Required with --ledger-manifest, to compare against.

The reuse key is embedding_text_sha256, not a line id: a vector is a function of the text
that was embedded and nothing else, so a digest the previous ledger knows names a vector
that is already correct — whatever id it now carries. It also catches what an id-keyed
diff misses silently, a line whose own text is unchanged but whose neighbour moved.

A shard is a window of *records*, not of line_ids. Merge by concatenating every shard's
vectors.f32 and records.jsonl in any order, then 'pack' the pair: it compares the ids it
was given against the recipe's expected set, so a lost shard is a refusal and not a
smaller artifact.

Options for 'pack':
  --vectors <path>           Raw little-endian f32 vectors, count x embedding_dim, no header
  --records <path>           JSONL, one record per vector, in the same order (see below)
  --corpus-identity <path>   As above
  --corpus-lines <path>      As above
  --model <path>             As above
  --chunking <path>          Optional; see 'The coverage contract' below
  --out <dir>                Output directory; must not exist, or be empty
  --collection <name>        Collection name in the payload header (default: "chunks")
  --created-at <timestamp>   Manifest timestamp (default: now, UTC)

Options for 'validate':
  --artifact <dir>           The artifact directory to verify
  --corpus-identity <path>   As above
  --corpus-lines <path>      As above
  --model <path>             As above
  --chunking <path>          Optional; as for 'pack'

A record is {{"line_id":N,"source_line_sha256":"...","embedding_text_sha256":"..."}}.
Both digests are lowercase hex SHA-256.

  source_line_sha256     of the corpus line's text. Checked against the corpus: this is
                         what catches a vector file that drifted out of step with its id
                         list, which nothing else here would notice.
  embedding_text_sha256  of the text that was actually embedded, after any title prefix,
                         neighbour context or truncation. Recorded as the record's
                         chunk_hash; not checked against anything, because the corpus
                         holds the line and not the recipe's output.

A chunker configuration is
{{"min_meaningful_chars":20,"context_window_lines":2,"max_chunk_chars":512,
  "min_embeddable_chars":5,"chunking_version":1}}, and its hash must be the
chunking_identity the model declares — an artifact records the hash, and a hash cannot
be turned back into the recipe.

The coverage contract: with --chunking, the lines that must get a vector are the ones the
recipe embeds, derived from the corpus. Without it, they are every line in the corpus
file, so it has to hold exactly the lines that should be embedded. 'build' always derives
them; there is nothing to declare that it does not apply.

Examples:
  otzaria-semantic-search version
  otzaria-semantic-search status --dir ./semantic_db
  otzaria-semantic-search search "מצות תפילין" --mode semantic --limit 5
  otzaria-semantic-search index-text "otzaria/demo.txt" "ספר הדגמה" "כל העוסק בתורה בלילה שכינה כנגדו"
  otzaria-semantic-search build --corpus-identity corpus.json --corpus-lines corpus.jsonl \
      --model model.json --model-file model.gguf --chunking chunking.json --out ./artifact
  otzaria-semantic-search pack --vectors v.f32 --records v.jsonl \
      --corpus-identity corpus.json --corpus-lines corpus.jsonl \
      --model model.json --out ./artifact
"#,
        env!("CARGO_PKG_VERSION")
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage();
        process::exit(0);
    }

    let command = args[1].to_lowercase();
    match command.as_str() {
        "version" | "-v" | "--version" => {
            println!("otzaria-semantic-search CLI {}", env!("CARGO_PKG_VERSION"));
            println!("Engine: Hybrid Tantivy + Local Vector Engine (GGUF MRL support)");
            println!("Crate Targets: rlib, cdylib, staticlib, binary CLI");
        }
        "status" => {
            let db_dir = parse_arg(&args, "--dir").unwrap_or_else(|| "./semantic_db".to_string());
            let config = SemanticConfig {
                root_dir: PathBuf::from(&db_dir),
                ..Default::default()
            };
            let engine_res = SemanticEngine::open(config);
            let coordinator = match engine_res {
                Ok(engine) => HybridCoordinator::new(Some(engine)),
                Err(_) => HybridCoordinator::new(None),
            };
            let hybrid = OtzariaHybridEngine::new(coordinator);
            let status = hybrid.get_semantic_status();

            println!("=== Otzaria Semantic Engine Status ===");
            println!("Available:          {}", status.available);
            println!("Model Loaded:       {}", status.model_loaded);
            println!("Model ID:           {}", status.model_id);
            println!("Embedding Dim:      {}", status.embedding_dim);
            println!("Indexed Books:      {}", status.indexed_book_count);
            println!("Stored Vectors:     {}", status.vector_count);
        }
        "search" => {
            if args.len() < 3 {
                eprintln!("Error: 'search' requires a query string.");
                eprintln!("Example: otzaria-semantic-search search \"שאילתה\"");
                process::exit(1);
            }
            let query = &args[2];
            let db_dir = parse_arg(&args, "--dir").unwrap_or_else(|| "./semantic_db".to_string());
            let mode_str = parse_arg(&args, "--mode").unwrap_or_else(|| "hybrid".to_string());
            let limit: u32 = parse_arg(&args, "--limit")
                .and_then(|s| s.parse().ok())
                .unwrap_or(10);

            let force_mode = match mode_str.to_lowercase().as_str() {
                "semantic" | "sem" => Some(SearchMode::SemanticOnly),
                "lexical" | "lex" => Some(SearchMode::LexicalOnly),
                _ => Some(SearchMode::Hybrid),
            };

            let config = SemanticConfig {
                root_dir: PathBuf::from(&db_dir),
                ..Default::default()
            };
            let engine_res = SemanticEngine::open(config);
            let coordinator = match engine_res {
                Ok(engine) => HybridCoordinator::new(Some(engine)),
                Err(_) => HybridCoordinator::new(None),
            };
            let hybrid = OtzariaHybridEngine::new(coordinator);

            let req = SearchRequest {
                query: query.clone(),
                lexical_candidates: vec![],
                limit: Some(limit),
                offset: Some(0),
                grouping: None,
                filters: None,
                force_mode,
                profile: None,
                feature_flags: None,
            };

            match hybrid.search(req) {
                Ok(res) => {
                    println!(
                        "Results for '{}' (mode: {:?}, available: {}, total: {}, latency: {}ms):",
                        query,
                        res.search_mode,
                        res.semantic_available,
                        res.total_count,
                        res.latency_ms
                    );
                    if let Some(reason) = &res.fallback_reason {
                        println!("Note: {reason}");
                    }
                    if res.results.is_empty() {
                        println!("No matching items found.");
                    } else {
                        for (idx, item) in res.results.iter().enumerate() {
                            println!(
                                " [{}] {} - {} (Score: {:.4})",
                                idx + 1,
                                item.title,
                                item.reference,
                                item.fused_score
                            );
                            if !item.text.is_empty() {
                                println!("     {}", item.text);
                            }
                        }
                    }
                }
                Err(err) => {
                    eprintln!("Search error: {err}");
                    process::exit(1);
                }
            }
        }
        "index-text" => {
            if args.len() < 5 {
                eprintln!("Error: 'index-text' requires <key> <title> <text>");
                eprintln!("Example: otzaria-semantic-search index-text \"otzaria/book1.txt\" \"כותרת\" \"תוכן השורה\"");
                process::exit(1);
            }
            let key = args[2].clone();
            let title = args[3].clone();
            let text = args[4].clone();
            let db_dir = parse_arg(&args, "--dir").unwrap_or_else(|| "./semantic_db".to_string());

            let config = SemanticConfig {
                root_dir: PathBuf::from(&db_dir),
                ..Default::default()
            };
            let engine = match SemanticEngine::open(config) {
                Ok(engine) => engine,
                Err(e) => {
                    eprintln!("Engine open error: {e}");
                    process::exit(1);
                }
            };
            let coordinator = HybridCoordinator::new(Some(engine));
            let hybrid = OtzariaHybridEngine::new(coordinator);

            let book = BookForIndexing {
                source_book_key: key.clone(),
                title: title.clone(),
                content_fingerprint: 1,
                is_pdf: false,
                topics: "/מקרא".to_string(),
                extra_facets: vec![],
                lines: vec![BookLine {
                    line_id: 1,
                    section_id: 1,
                    segment: 1,
                    reference: format!("{title} א, א"),
                    line_hash: 1001,
                    text,
                }],
            };

            match hybrid.index_books(&[book]) {
                Ok(Some(summary)) => {
                    println!(
                        "Successfully indexed book '{}': {} chunks written.",
                        key, summary.chunks_written
                    );
                }
                Ok(None) => {
                    eprintln!("Error: Semantic engine path disabled.");
                    process::exit(1);
                }
                Err(e) => {
                    eprintln!("Indexing error: {e}");
                    process::exit(1);
                }
            }
        }
        "build" => run_build(&args),
        "export-plan" => run_export_plan(&args),
        "embed-shard" => run_embed_shard(&args),
        "plan-split" => run_plan_split(&args),
        "assemble" => run_assemble(&args),
        "ledger" => run_ledger(&args),
        "pack" => run_pack(&args),
        "validate" => run_validate(&args),
        "help" | "-h" | "--help" => {
            print_usage();
        }
        other => {
            eprintln!("Unknown command: '{other}'");
            print_usage();
            process::exit(1);
        }
    }
}

fn parse_arg(args: &[String], flag: &str) -> Option<String> {
    for i in 0..args.len().saturating_sub(1) {
        if args[i] == flag {
            return Some(args[i + 1].clone());
        }
    }
    None
}

/// A flag with no default. Missing means the command cannot run, so it exits rather than
/// substituting a path nobody asked for.
fn require_arg(args: &[String], flag: &str) -> String {
    parse_arg(args, flag).unwrap_or_else(|| {
        eprintln!("Error: {flag} is required.");
        eprintln!("Run 'otzaria-semantic-search help' for the full option list.");
        process::exit(1);
    })
}

fn exit_with<E: std::fmt::Display>(context: &str, error: E) -> ! {
    eprintln!("{context}: {error}");
    process::exit(1);
}

/// Read the `ModelIdentity` a build declares for its vectors.
///
/// A file rather than a dozen flags: it is half of the artifact's identity, it is written
/// once per model release, and it belongs under version control beside the model rather
/// than in a shell history.
fn read_model(path: &str) -> ModelIdentity {
    let json = std::fs::read_to_string(path)
        .unwrap_or_else(|error| exit_with(&format!("Could not read {path}"), error));
    serde_json::from_str(&json)
        .unwrap_or_else(|error| exit_with(&format!("{path} is not a model identity"), error))
}

fn load_corpus(args: &[String]) -> JsonlCorpus {
    let identity_path = require_arg(args, "--corpus-identity");
    let lines_path = require_arg(args, "--corpus-lines");
    let corpus = JsonlCorpus::load(Path::new(&identity_path), Path::new(&lines_path))
        .unwrap_or_else(|error| exit_with("Could not read the corpus", error));
    println!("Corpus: {} line(s) from {lines_path}", corpus.len());
    corpus
}

/// Read the recipe a build applies.
///
/// A separate file from the model identity because it is a different kind of fact: the
/// identity says what the artifact *declares*, and this says what will actually be done to
/// the text. The build refuses to proceed unless one hashes to the other.
fn read_chunking(path: &str) -> ChunkerConfig {
    let json = std::fs::read_to_string(path)
        .unwrap_or_else(|error| exit_with(&format!("Could not read {path}"), error));
    serde_json::from_str(&json)
        .unwrap_or_else(|error| exit_with(&format!("{path} is not a chunker configuration"), error))
}

/// Wrap the corpus in the recipe, when one was given.
///
/// Without `--chunking` the corpus file is the coverage contract, which is what it has
/// always been and is exactly as trustworthy as whoever exported it. With one, the lines
/// that must get a vector are derived by applying the recipe — and the recipe is pinned to
/// the `chunking_identity` the artifact declares, so an export made under one recipe cannot
/// be packed under another.
fn plan_corpus<'a>(
    args: &[String],
    corpus: &'a JsonlCorpus,
    model: &ModelIdentity,
) -> Option<PlannedCorpus<'a>> {
    let chunking = read_chunking(&parse_arg(args, "--chunking")?);
    let planned = PlannedCorpus::new(corpus, &chunking, model)
        .unwrap_or_else(|error| exit_with("Could not apply the recipe to the corpus", error));
    println!("Recipe: {} line(s) get a vector", planned.plan().len());
    Some(planned)
}

/// Apply the recipe on the machine that holds the corpus, and write the work out.
///
/// No model is opened and none is needed: this is the half of a build that is arithmetic
/// on strings. What it writes is what a worker with a GPU and no corpus can act on.
fn run_export_plan(args: &[String]) {
    let out = PathBuf::from(require_arg(args, "--out"));
    let model = read_model(&require_arg(args, "--model"));
    let chunking = read_chunking(&require_arg(args, "--chunking"));
    let corpus = load_corpus(args);

    std::fs::create_dir_all(&out)
        .unwrap_or_else(|error| exit_with("Could not create the output directory", error));
    let plan_path = out.join("plan.jsonl");
    let file = std::fs::File::create(&plan_path)
        .unwrap_or_else(|error| exit_with("Could not write the plan", error));
    let mut sink = std::io::BufWriter::new(file);

    let report = export_plan(&corpus, &chunking, &model, &mut sink)
        .unwrap_or_else(|error| exit_with("Export failed", error));
    let manifest = out.join("export-manifest.json");
    std::fs::write(&manifest, serde_json::to_vec_pretty(&report).unwrap())
        .unwrap_or_else(|error| exit_with("Could not write the export manifest", error));

    println!("\n=== Exported a build plan ===");
    println!("Plan:            {}", plan_path.display());
    println!("Records:         {}", report.records);
    println!(
        "line_id range:   {}..={}",
        report.min_line_id, report.max_line_id
    );
    println!("Plan SHA-256:    {}", report.plan_sha256);
    println!("Chunking:        {}", report.chunking_identity);
    println!(
        "\nSplit it by record: --skip and --take name a window, and every record must fall in\n\
         exactly one. The merge refuses a hole rather than packing around it."
    );
}

/// Embed one window of a plan. The half of a build that needs a model and no corpus.
fn run_embed_shard(args: &[String]) {
    let out = PathBuf::from(require_arg(args, "--out"));
    let plan_path = require_arg(args, "--plan");
    let model = read_model(&require_arg(args, "--model"));
    let model_file = require_arg(args, "--model-file");
    let skip: usize = parse_arg(args, "--skip").map_or(0, |value| {
        value
            .parse()
            .unwrap_or_else(|_| exit_with("--skip", "not a number"))
    });
    let take: usize = parse_arg(args, "--take").map_or(usize::MAX, |value| {
        value
            .parse()
            .unwrap_or_else(|_| exit_with("--take", "not a number"))
    });

    let mut runtime = EmbeddingRuntime::new(EmbeddingConfig {
        model_path: PathBuf::from(&model_file),
        embedding_dim: model.embedding_dim,
        max_tokens: model.max_tokens,
        batch_size: parse_arg(args, "--batch")
            .and_then(|value| value.parse().ok())
            .unwrap_or(32),
        pooling: Pooling::parse(&model.pooling).unwrap_or_else(|error| {
            exit_with("The model declares a pooling nothing performs", error)
        }),
    });
    runtime
        .load()
        .unwrap_or_else(|error| exit_with("Could not load the model", error));

    // The same five comparisons `build` makes, for the same reason: a worker that embeds
    // with a different file or a different width produces vectors the merge cannot use,
    // and it should learn that in the second it takes rather than at the end of the shard.
    for (field, declared, loaded) in [
        (
            "model_checksum",
            model.model_checksum.clone(),
            runtime.model_checksum().unwrap_or_default().to_string(),
        ),
        (
            "embedding_backend",
            model.embedding_backend.clone(),
            runtime.backend_id().unwrap_or_default().to_string(),
        ),
        (
            "embedding_dim",
            model.embedding_dim.to_string(),
            runtime.dim().to_string(),
        ),
        (
            "pooling",
            model.pooling.clone(),
            runtime.pooling().to_string(),
        ),
        (
            "max_tokens",
            model.max_tokens.to_string(),
            runtime.max_tokens().to_string(),
        ),
    ] {
        if declared != loaded {
            exit_with(
                "The model file is not the one the plan was made for",
                format!("{field}: declared {declared}, loaded {loaded}"),
            );
        }
    }
    if !runtime.backend_is_semantic() && !args.iter().any(|arg| arg == "--allow-non-semantic") {
        exit_with(
            "This backend's vectors mean nothing",
            runtime.backend_id().unwrap_or("none").to_string(),
        );
    }

    std::fs::create_dir_all(&out)
        .unwrap_or_else(|error| exit_with("Could not create the output directory", error));
    let plan = std::io::BufReader::new(
        std::fs::File::open(&plan_path)
            .unwrap_or_else(|error| exit_with("Could not read the plan", error)),
    );
    // `.partial` until the counts and digests are known: a shard killed by a session
    // timeout must not leave a file the merge could mistake for a finished one.
    let vectors_partial = out.join("vectors.f32.partial");
    let records_partial = out.join("records.jsonl.partial");
    let mut vectors = std::io::BufWriter::new(
        std::fs::File::create(&vectors_partial)
            .unwrap_or_else(|error| exit_with("Could not write the vectors", error)),
    );
    let mut records = std::io::BufWriter::new(
        std::fs::File::create(&records_partial)
            .unwrap_or_else(|error| exit_with("Could not write the records", error)),
    );

    // The plan's own digest travels into the manifest, so the merge can tell a shard of
    // this export from a shard of another export with the same window.
    let plan_sha256 = sha256_of(Path::new(&plan_path));
    let report = embed_shard(
        read_plan(plan, skip, take),
        (plan_sha256, skip, take),
        &model,
        &runtime,
        runtime.batch_size(),
        &mut vectors,
        &mut records,
    )
    .unwrap_or_else(|error| exit_with("The shard failed", error));
    drop((vectors, records));

    for (partial, final_name) in [
        (&vectors_partial, "vectors.f32"),
        (&records_partial, "records.jsonl"),
    ] {
        std::fs::rename(partial, out.join(final_name))
            .unwrap_or_else(|error| exit_with("Could not publish the shard", error));
    }
    std::fs::write(
        out.join("shard-manifest.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap_or_else(|error| exit_with("Could not write the shard manifest", error));

    println!("\n=== Embedded a shard ===");
    println!("Path:            {}", out.display());
    println!("Records:         {} (skip {skip})", report.records);
    println!("Dimension:       {}", report.embedding_dim);
    println!("vectors SHA-256: {}", report.vectors_sha256);
    println!("records SHA-256: {}", report.records_sha256);
}

/// Split a plan against the previous release's ledger.
///
/// The half of an update that decides what does not have to be embedded again. Reads no
/// model and touches no GPU: it is a digest lookup per line.
fn run_plan_split(args: &[String]) {
    let out = PathBuf::from(require_arg(args, "--out"));
    let plan = require_arg(args, "--plan");

    std::fs::create_dir_all(&out)
        .unwrap_or_else(|error| exit_with("Could not create the output directory", error));
    let plan = std::io::BufReader::new(
        std::fs::File::open(&plan)
            .unwrap_or_else(|error| exit_with("Could not read the plan", error)),
    );
    // A ledger without its manifest, its vectors and the model is a set of integers
    // nothing can check, so reuse requires all of them or none. No ledger at all is the
    // first build rather than a mistake.
    let base = verified_base(args);
    let mut reuse = std::io::BufWriter::new(
        std::fs::File::create(out.join("reuse.jsonl"))
            .unwrap_or_else(|error| exit_with("Could not write reuse.jsonl", error)),
    );
    let mut embed = std::io::BufWriter::new(
        std::fs::File::create(out.join("embed.jsonl"))
            .unwrap_or_else(|error| exit_with("Could not write embed.jsonl", error)),
    );

    let report = plan_split(plan, base.as_ref(), &mut reuse, &mut embed)
        .unwrap_or_else(|error| exit_with("The split failed", error));

    println!("\n=== Split a plan against a ledger ===");
    println!("Planned:   {}", report.planned);
    println!("Reused:    {}", report.reused);
    println!("To embed:  {}", report.to_embed);
}

/// Assemble one release's vectors from what was reused and what was embedded.
///
/// Shard directories are taken in sorted order, and each is expected to hold a
/// `vectors.f32` and a `records.jsonl` of matching length — a pair that disagrees is
/// refused rather than shifting every pairing after it.
fn run_assemble(args: &[String]) {
    let out = PathBuf::from(require_arg(args, "--out"));
    let shard_root = PathBuf::from(require_arg(args, "--shards"));
    // The width comes from the identity the shards were verified against, not from the
    // command line. It was a `--dim` argument, and a `--dim` that disagreed with the model
    // made every stride through the base and the shards the wrong length — while the
    // identity the artifact publishes still said the model's number.
    let shard_model = read_model(&require_arg(args, "--model"));
    let embedding_dim = shard_model.embedding_dim as usize;
    if let Some(given) = parse_arg(args, "--dim") {
        if given.parse::<usize>() != Ok(embedding_dim) {
            exit_with(
                "--dim disagrees with the model and is no longer needed",
                format!("{given} against the model's {embedding_dim}"),
            );
        }
    }

    let reuse: Vec<ReuseEntry> = match parse_arg(args, "--reuse") {
        Some(path) => std::fs::read_to_string(&path)
            .unwrap_or_else(|error| exit_with("Could not read the reuse list", error))
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| {
                serde_json::from_str(line)
                    .unwrap_or_else(|error| exit_with("A reuse entry is malformed", error))
            })
            .collect(),
        None => Vec::new(),
    };
    // The same verified bundle `plan-split` used. Reuse cannot be handed a bare file.
    let base = verified_base(args);

    // Every `vectors.f32` under the root, at any depth: a shard produced by a
    // multi-GPU session is a directory of directories, and flattening it here means the
    // caller does not have to.
    let mut shards = Vec::new();
    collect_shards(&shard_root, &mut shards);
    shards.sort();
    println!(
        "Shards: {} half-shard(s) under {}",
        shards.len(),
        shard_root.display()
    );
    // Read every manifest and hold the set to the plan before a byte is copied. Opening
    // the two files directly — which is what this did — left the manifests as decoration:
    // a shard from another export, a corrupted vector that stayed finite, or two shards
    // covering one window and none covering another all merged without a word.
    let manifests: Vec<(PathBuf, ShardReport)> = shards
        .iter()
        .map(|dir| {
            let path = dir.join("shard-manifest.json");
            let manifest: ShardReport =
                serde_json::from_str(&std::fs::read_to_string(&path).unwrap_or_else(|error| {
                    exit_with(&format!("Could not read {}", path.display()), error)
                }))
                .unwrap_or_else(|error| {
                    exit_with(
                        &format!("{} is not a shard manifest", path.display()),
                        error,
                    )
                });
            (dir.clone(), manifest)
        })
        .collect();
    let plan_sha256 = require_arg(args, "--plan-sha256");
    let expected: usize = require_arg(args, "--embed-records")
        .parse()
        .unwrap_or_else(|_| exit_with("--embed-records", "not a number"));
    let opened = verify_shards(&manifests, &plan_sha256, &shard_model, expected)
        .unwrap_or_else(|error| exit_with("The shards do not cover this plan", error));
    println!("Shards verified: {expected} record(s), one plan, one model, no hole");

    std::fs::create_dir_all(&out)
        .unwrap_or_else(|error| exit_with("Could not create the output directory", error));
    let mut vectors = std::io::BufWriter::new(
        std::fs::File::create(out.join("vectors.f32"))
            .unwrap_or_else(|error| exit_with("Could not write vectors.f32", error)),
    );
    let mut records = std::io::BufWriter::new(
        std::fs::File::create(out.join("records.jsonl"))
            .unwrap_or_else(|error| exit_with("Could not write records.jsonl", error)),
    );
    let report = assemble(
        reuse,
        base.as_ref(),
        opened,
        embedding_dim,
        &mut vectors,
        &mut records,
    )
    .unwrap_or_else(|error| exit_with("Assembly failed", error));

    println!("\n=== Assembled a release's vectors ===");
    println!("Vectors:   {}", report.vectors);
    println!("Reused:    {}", report.reused);
    println!("Embedded:  {}", report.embedded);
    println!("Dimension: {}", report.embedding_dim);
    println!(
        "\nPack them against the corpus next; nothing here checked them against it — and \
         build the\nledger from the packed artifact, not from this, because packing \
         reorders the payload."
    );
}

/// Build the ledger the next release will reuse from, out of the artifact that was
/// published.
///
/// A separate command from `assemble`, and that is the whole point: `pack` sorts the
/// payload by `semantic_id`, so the order `assemble` wrote is not the order anybody can
/// download. A ledger built from the assembler's order was published once and every one
/// of its offsets named a different line.
fn run_ledger(args: &[String]) {
    let artifact = PathBuf::from(require_arg(args, "--artifact"));
    let records = require_arg(args, "--records");
    let out = PathBuf::from(require_arg(args, "--out"));

    // The artifact first, and every byte of it. Nothing about the ledger is worth deriving
    // from an artifact that does not verify, and the identity below has to come from the
    // package rather than from the caller: `--model` and `--artifact-digest` used to be
    // copied into the manifest unchecked, which let a manifest declare model B over
    // vectors built by model A of the same width — and `VerifiedBase` would then verify
    // the ledger perfectly against that lie.
    let package = IndexPackage::read(&artifact)
        .unwrap_or_else(|error| exit_with("The artifact could not be read", error));
    println!("Verifying the artifact: reading every payload…");
    package
        .verify_integrity(&artifact)
        .unwrap_or_else(|error| exit_with("The artifact does not verify", error));
    let identity = package.manifest.identity.clone();
    let digest = package.digest();
    // An external anchor is compared, not copied. If the caller has the published digest,
    // a mismatch means this is not the artifact they think it is — and it is worth learning
    // that before the ledger is derived rather than after.
    if let Some(published) = parse_arg(args, "--artifact-digest") {
        if published != digest {
            exit_with(
                "This is not the artifact that digest was published for",
                format!("published {published}, computed {digest}"),
            );
        }
    }
    // Already read and checked by `verify_integrity`, one line above. Hashing the file a
    // second time cost another pass over 23.5 GB and could only agree.
    let vectors_sha256 = package
        .payloads
        .get(VECTORS_FILENAME)
        .map(|payload| payload.sha256.clone())
        .unwrap_or_else(|| exit_with("The artifact has no vectors payload", VECTORS_FILENAME));

    let metadata = std::io::BufReader::new(
        std::fs::File::open(artifact.join("metadata.jsonl")).unwrap_or_else(|error| {
            exit_with("Could not read the artifact's metadata.jsonl", error)
        }),
    );
    let records = std::io::BufReader::new(
        std::fs::File::open(&records)
            .unwrap_or_else(|error| exit_with("Could not read the records", error)),
    );
    // Both files land by rename, and only once both are known good. Writing the ledger
    // first meant a failure here — a records file from another build, a digest that did not
    // match — left a new ledger beside the previous manifest, which is the one pairing
    // `VerifiedBase` cannot detect: each file is internally valid.
    let partial = out.with_extension("partial");
    let mut sink = std::io::BufWriter::new(
        std::fs::File::create(&partial)
            .unwrap_or_else(|error| exit_with("Could not write the ledger", error)),
    );

    let entries = ledger_from_artifact(metadata, records, &mut sink)
        .unwrap_or_else(|error| exit_with("The ledger could not be built", error));
    std::io::Write::flush(&mut sink)
        .unwrap_or_else(|error| exit_with("Could not finish writing the ledger", error));
    sink.into_inner()
        .unwrap_or_else(|error| exit_with("Could not finish writing the ledger", error))
        .sync_all()
        .unwrap_or_else(|error| exit_with("Could not flush the ledger to disk", error));

    if entries != package.manifest.vector_count as usize {
        exit_with(
            "The ledger does not describe this artifact",
            format!(
                "{entries} entry(ies) against {} vector(s)",
                package.manifest.vector_count
            ),
        );
    }
    let manifest = LedgerManifest {
        artifact_digest: digest,
        vectors_sha256,
        // The bytes that are about to be renamed into place, which are the bytes
        // `VerifiedBase` will hash when it opens them.
        ledger_sha256: sha256_of(&partial),
        vector_count: entries,
        embedding_dim: identity.model.embedding_dim,
        model: identity.model,
    };
    let manifest_path = out.with_extension("manifest.json");
    let manifest_partial = out.with_extension("manifest.partial");
    write_and_sync(
        &manifest_partial,
        &serde_json::to_vec_pretty(&manifest).unwrap(),
    );
    // The ledger before the manifest: a manifest with no ledger beside it is a missing
    // file, and a ledger with no manifest is one command away from being described. A
    // manifest describing the *previous* ledger is neither.
    std::fs::rename(&partial, &out)
        .unwrap_or_else(|error| exit_with("Could not put the ledger in place", error));
    std::fs::rename(&manifest_partial, &manifest_path)
        .unwrap_or_else(|error| exit_with("Could not put the ledger manifest in place", error));

    println!("\n=== Built a ledger from the artifact ===");
    println!("Ledger:   {}", out.display());
    println!("Manifest: {}", manifest_path.display());
    println!("Entries:  {entries}");
}

/// Write a small file and get it onto the disk before anything renames it into place.
fn write_and_sync(path: &Path, bytes: &[u8]) {
    use std::io::Write;
    let mut file = std::fs::File::create(path)
        .unwrap_or_else(|error| exit_with(&format!("Could not write {}", path.display()), error));
    file.write_all(bytes)
        .unwrap_or_else(|error| exit_with(&format!("Could not write {}", path.display()), error));
    file.sync_all()
        .unwrap_or_else(|error| exit_with(&format!("Could not flush {}", path.display()), error));
}

/// SHA-256 of a file, streamed.
fn sha256_of(path: &Path) -> String {
    use sha2::{Digest, Sha256};
    let mut file = std::fs::File::open(path)
        .unwrap_or_else(|error| exit_with(&format!("Could not read {}", path.display()), error));
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let read = std::io::Read::read(&mut file, &mut buffer)
            .unwrap_or_else(|error| exit_with("Could not read the file to hash it", error));
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    format!("{:x}", hasher.finalize())
}

/// Open the base artifact both `plan-split` and `assemble` reuse from, or nothing.
///
/// Every argument or none: a ledger names offsets, the manifest says what they mean, the
/// vectors are what they point into, and the model is what makes them comparable. Three
/// out of four is a check that cannot be performed, so it is refused rather than skipped.
fn verified_base(
    args: &[String],
) -> Option<otzaria_semantic_search::distribution::reuse::VerifiedBase> {
    let ledger = parse_arg(args, "--ledger");
    let manifest = parse_arg(args, "--ledger-manifest");
    let vectors = parse_arg(args, "--base-vectors");
    let model = parse_arg(args, "--model");
    // Asking for reuse at all means asking for all of it. Returning early on a missing
    // `--ledger` ignored the other two in silence, so `--ledgr ledger.jsonl` — with the
    // manifest and the vectors both correctly named — read as "there is no base" and spent
    // a full rebuild on a typo.
    let named: Vec<&str> = [
        ("--ledger", ledger.is_some()),
        ("--ledger-manifest", manifest.is_some()),
        ("--base-vectors", vectors.is_some()),
    ]
    .iter()
    .filter_map(|(name, given)| given.then_some(*name))
    .collect();
    if !named.is_empty() && named.len() < 3 {
        exit_with(
            "Reuse needs --ledger, --ledger-manifest, --base-vectors and --model together",
            format!(
                "only {} was given, and offsets cannot be checked without the file they \
                 index and the identity they were built under",
                named.join(", ")
            ),
        )
    }
    if named.is_empty() {
        return None;
    }
    let (Some(manifest_path), Some(vectors_path), Some(model_path)) = (manifest, vectors, model)
    else {
        exit_with(
            "Reuse also needs --model",
            "the identity the base was built under is what makes its vectors comparable \
             with the ones about to be embedded",
        )
    };
    let manifest: LedgerManifest = serde_json::from_str(
        &std::fs::read_to_string(&manifest_path)
            .unwrap_or_else(|error| exit_with("Could not read the ledger manifest", error)),
    )
    .unwrap_or_else(|error| exit_with("That is not a ledger manifest", error));
    let model = read_model(&model_path);

    println!("Verifying the base artifact: hashing its ledger and its vectors…");
    let base = VerifiedBase::open(
        manifest,
        Path::new(ledger.as_deref().expect("named holds all three")),
        Path::new(&vectors_path),
        &model,
    )
    .unwrap_or_else(|error| exit_with("The base artifact cannot be reused", error));
    println!("Base verified: artifact {}", base.artifact_digest());
    Some(base)
}

/// Every directory holding a `vectors.f32`, depth-first.
fn collect_shards(root: &Path, found: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    if root.join("vectors.f32").is_file() {
        found.push(root.to_path_buf());
        return;
    }
    for entry in entries.flatten() {
        if entry.path().is_dir() {
            collect_shards(&entry.path(), found);
        }
    }
}

fn run_build(args: &[String]) {
    let out = require_arg(args, "--out");
    let model = read_model(&require_arg(args, "--model"));
    let model_file = require_arg(args, "--model-file");
    let chunking = read_chunking(&require_arg(args, "--chunking"));
    let corpus = load_corpus(args);

    let report = build(
        BuildRequest {
            output_path: PathBuf::from(&out),
            model_path: PathBuf::from(&model_file),
            model,
            chunking,
            created_at: parse_arg(args, "--created-at")
                .unwrap_or_else(|| utc_timestamp(SystemTime::now())),
            collection_name: parse_arg(args, "--collection")
                .unwrap_or_else(|| "chunks".to_string()),
            batch_size: parse_arg(args, "--batch")
                .and_then(|value| value.parse().ok())
                .unwrap_or(32),
            allow_non_semantic_backend: args.iter().any(|arg| arg == "--allow-non-semantic"),
        },
        &corpus,
    )
    .unwrap_or_else(|error| exit_with("Build failed", error));

    println!("\n=== Built an official artifact ===");
    print_report(&report);
}

fn run_pack(args: &[String]) {
    let vectors = require_arg(args, "--vectors");
    let records = require_arg(args, "--records");
    let out = require_arg(args, "--out");
    let model = read_model(&require_arg(args, "--model"));
    let corpus = load_corpus(args);
    let planned = plan_corpus(args, &corpus, &model);
    let target: &dyn CorpusIndex = planned.as_ref().map_or(&corpus, |planned| planned);

    let inputs = read_vector_inputs(
        Path::new(&vectors),
        Path::new(&records),
        model.embedding_dim,
    )
    .unwrap_or_else(|error| exit_with("Could not read the vectors", error));

    let report = pack(
        PackRequest {
            output_path: PathBuf::from(&out),
            model,
            created_at: parse_arg(args, "--created-at")
                .unwrap_or_else(|| utc_timestamp(SystemTime::now())),
            collection_name: parse_arg(args, "--collection")
                .unwrap_or_else(|| "chunks".to_string()),
        },
        inputs,
        target,
    )
    .unwrap_or_else(|error| exit_with("Packing failed", error));

    println!("\n=== Packed an official artifact ===");
    print_report(&report);
}

fn run_validate(args: &[String]) {
    let artifact = require_arg(args, "--artifact");
    let model = read_model(&require_arg(args, "--model"));
    let corpus = load_corpus(args);
    let planned = plan_corpus(args, &corpus, &model);
    let target: &dyn CorpusIndex = planned.as_ref().map_or(&corpus, |planned| planned);

    let report = validate_artifact(Path::new(&artifact), &model, target)
        .unwrap_or_else(|error| exit_with("Validation failed", error));

    println!("\n=== Artifact verified ===");
    print_report(&report);
}

fn print_report(report: &PackReport) {
    println!("Path:            {}", report.artifact_path.display());
    println!("Vectors:         {}", report.vector_count);
    println!("Books:           {}", report.book_count);
    println!("Payload bytes:   {}", report.total_size_bytes);
    println!("Identity:        {}", report.identity);
    println!("Digest:          {}", report.digest);
    // The digest is only a trust anchor once it travels outside the package: recomputing
    // it from the package proves the package agrees with itself and nothing more.
    println!(
        "\nPublish that digest outside the artifact. Verified without it, an install \
         detects damage\nand the wrong artifact, but not one deliberately rebuilt to \
         match."
    );
}

/// `YYYY-MM-DDTHH:MM:SSZ`, for the manifest's `created_at`.
///
/// Hand-rolled because this crate carries no date dependency and needs one string in one
/// place. The value is excluded from the artifact digest, so it cannot make a build
/// irreproducible — but it is what a human reads off a manifest, and seconds since an
/// epoch is not that.
fn utc_timestamp(time: SystemTime) -> String {
    let seconds = time
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_secs() as i64);
    let (year, month, day) = civil_from_days(seconds.div_euclid(86_400));
    let second_of_day = seconds.rem_euclid(86_400);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        second_of_day / 3600,
        (second_of_day % 3600) / 60,
        second_of_day % 60
    )
}

/// Days since 1970-01-01 → civil date. Howard Hinnant's `civil_from_days`, which is
/// exact for the whole proleptic Gregorian calendar and needs no lookup tables.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    // Shift the epoch to 0000-03-01, so leap days land at the end of the cycle.
    let shifted = days + 719_468;
    let era = shifted.div_euclid(146_097);
    let day_of_era = shifted.rem_euclid(146_097); // [0, 146096]
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365; // [0, 399]
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100); // [0, 365]
    let month_position = (5 * day_of_year + 2) / 153; // [0, 11], March = 0
    let day = (day_of_year - (153 * month_position + 2) / 5 + 1) as u32;
    let month = if month_position < 10 {
        month_position + 3
    } else {
        month_position - 9
    } as u32;
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Anchored on dates that are checkable by hand, including the boundary the
    /// March-based arithmetic exists to get right.
    #[test]
    fn the_timestamp_matches_known_instants() {
        for (seconds, expected) in [
            (0, "1970-01-01T00:00:00Z"),
            (1_000_000_000, "2001-09-09T01:46:40Z"),
            (1_582_934_400, "2020-02-29T00:00:00Z"),
            (1_583_020_800, "2020-03-01T00:00:00Z"),
            (1_609_459_199, "2020-12-31T23:59:59Z"),
            (1_609_459_200, "2021-01-01T00:00:00Z"),
        ] {
            assert_eq!(
                utc_timestamp(UNIX_EPOCH + std::time::Duration::from_secs(seconds)),
                expected
            );
        }
    }
}
