//! Binary CLI interface for `otzaria-semantic-search`.
//!
//! Provides a standalone command-line application for querying, indexing, and
//! inspecting the Otzaria semantic search engine, and the build-side commands: `build`,
//! which embeds a corpus into a base vector package, and `export-plan` and `embed-shard`,
//! which cut that work in two for machines that never meet.
//!
//! `export-plan` needs no embedding backend — it never turns text into a vector — so it
//! works in a default build, which is the one a release pipeline has. So does
//! `model-checksum`, which reads a model without running it. `build` and `embed-shard` do
//! turn text into vectors, and so need one compiled in.

use otzaria_semantic_search::api::hybrid_search::{OtzariaHybridEngine, SearchRequest};
use otzaria_semantic_search::distribution::builder::{
    build, BuildRequest, RELEASE_MANIFEST_FILENAME, SEGMENT_FILENAME,
};
use otzaria_semantic_search::distribution::corpus::JsonlCorpus;
use otzaria_semantic_search::distribution::package::utc_timestamp;
use otzaria_semantic_search::distribution::shard::{embed_shard, export_plan, read_plan};
use otzaria_semantic_search::hybrid::coordinator::HybridCoordinator;
use otzaria_semantic_search::semantic::backend::Pooling;
use otzaria_semantic_search::semantic::chunker::ChunkerConfig;
use otzaria_semantic_search::semantic::embedding::{EmbeddingConfig, EmbeddingRuntime};
use otzaria_semantic_search::semantic::engine::{SemanticConfig, SemanticEngine};
use otzaria_semantic_search::semantic::model_package::validate_model;
use otzaria_semantic_search::semantic::types::{BookForIndexing, BookLine, SearchMode};
use otzaria_semantic_search::semantic::versioning::ModelIdentity;
use std::env;
use std::path::{Path, PathBuf};
use std::process;
use std::time::SystemTime;

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
  build [options]                     Embed a corpus and write a base vector package.
  export-plan [options]               Apply the recipe and write the work out, for a
                                      machine that will embed it elsewhere.
  embed-shard [options]               Embed one window of an exported plan.
  model-checksum --model-file <path>  Validate a model and print the model_checksum an
                                      identity has to declare for it.

Options for 'search':
  --dir <path>       Directory holding semantic database (default: "./semantic_db")
  --mode <mode>      Retrieval mode: hybrid (default), semantic, lexical
  --limit <N>        Maximum results to return (default: 10)

Options for 'status':
  --dir <path>       Directory holding semantic database (default: "./semantic_db")

Options for 'build':
  --corpus-identity <path>   JSON CorpusIdentity, as the lexical index reports it: its
                             text identity, library version and release tag, and id scheme
  --corpus-lines <path>      JSONL, one corpus line per document
  --model <path>             JSON ModelIdentity describing how the vectors are produced
  --model-file <path>        The model the vectors are produced with: an ONNX graph, with
                             its package beside it (see 'model-checksum')
  --chunking <path>          JSON ChunkerConfig — the recipe itself (see below)
  --out <dir>                Output directory; must not exist, or be empty. Receives
                             segment.oxv, the package's manifest.json and payloads.json,
                             and release.json — the manifest an installation is handed
                             with the segment
  --batch <N>                Texts per inference call (default: 32)
  --clip-q <q>               The quantile each dimension's int8 scale is calibrated at,
                             in (0, 1] (default: 1, which clips nothing the base holds)
  --created-at <timestamp>   Manifest timestamp (default: now, UTC)
  --allow-non-semantic       Write a package from a backend whose vectors mean nothing.
                             For tests only: such a package passes every check here and
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
  --model-file <path>        The ONNX graph; held to every field the identity declares
  --skip <N>                 Records to skip (default: 0)
  --take <N>                 Records to embed (default: all that remain)
  --batch <N>                Texts per inference call (default: 32)
  --out <dir>                Receives vectors.f32, records.jsonl, shard-manifest.json.
                             Leftovers from a session that died are overwritten — retrying a
                             window is normal — but a directory holding all three is not.
  --allow-non-semantic       As for 'build'

Options for 'model-checksum':
  --model-file <path>        An .onnx graph, its tokenizer.json beside it. No other kind of
                             model is read: a path that does not end in .onnx is refused.

An ONNX model is a package: the graph, the tokenizer.json beside it and every external-data
file the graph names, and the checksum is the SHA-256 of a manifest listing each of them
with its size and SHA-256 — printed here exactly as it is hashed. Nothing else in the
directory is part of it: not a README, not a second graph, not an ONNX Runtime library.

A shard's records.jsonl holds one record per vector, in the order of vectors.f32:
{{"line_id":N,"source_line_sha256":"...","embedding_text_sha256":"..."}}. Both digests
are lowercase hex SHA-256.

  source_line_sha256     of the corpus line's text, for the merge to check against the
                         corpus: what catches a vector file that drifted out of step with
                         its records, which nothing else would notice.
  embedding_text_sha256  of the text that was actually embedded, after any role prefix,
                         neighbour context or truncation. Its first 16 bytes are the
                         vector's chunk key: what it is stored under, and found by.

A chunker configuration is
{{"min_meaningful_chars":20,"context_window_lines":2,"max_chunk_chars":512,
  "min_embeddable_chars":5,"chunking_version":1,"embedding_text_version":1,
  "normalization_version":1}}, and its hash must be the chunking_identity the model
declares — a package records the hash, and a hash cannot be turned back into the recipe.
Every field is required. embedding_text_version 2 puts "[PASSAGE] " before every passage
and "[QUERY] " before every query, for a model trained with those role prefixes.

Examples:
  otzaria-semantic-search version
  otzaria-semantic-search status --dir ./semantic_db
  otzaria-semantic-search search "מצות תפילין" --mode semantic --limit 5
  otzaria-semantic-search index-text "otzaria/demo.txt" "ספר הדגמה" "כל העוסק בתורה בלילה שכינה כנגדו"
  otzaria-semantic-search build --corpus-identity corpus.json --corpus-lines corpus.jsonl \
      --model model.json --model-file model.onnx --chunking chunking.json --out ./package
  otzaria-semantic-search model-checksum --model-file models/meivin/model-fp32.onnx
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
            println!("Engine: Hybrid Tantivy + Local Vector Engine (ONNX models)");
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
                ranking: None,
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
        "model-checksum" => run_model_checksum(&args),
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

/// Validate a model the way `build` and the runtime will, and print what an identity has
/// to declare for it.
///
/// Opens no backend, so it works in a default build: the checksum is a fact about the
/// files, and writing `model.json` for a new model should not need the model to run.
fn run_model_checksum(args: &[String]) {
    let model_file = PathBuf::from(require_arg(args, "--model-file"));
    let package = validate_model(&model_file)
        .unwrap_or_else(|error| exit_with("The model cannot be used", error));

    println!("=== Model checksum ===");
    println!("Model:           {}", model_file.display());
    println!("Format:          ONNX");
    println!("model_checksum:  {}", package.checksum());

    let facts = package.graph_facts();
    println!("Package root:    {}", package.root().display());
    println!(
        "Graph:           IR {}, {} opset import(s), {} input(s), {} output(s), {} \
         external tensor reference(s)",
        facts.ir_version,
        facts.opset_imports,
        facts.graph_inputs,
        facts.graph_outputs,
        facts.external_tensors
    );
    println!("\nPackage files ({}):", package.files().len());
    let width = package
        .files()
        .iter()
        .map(|file| file.relpath.chars().count())
        .max()
        .unwrap_or(0);
    for file in package.files() {
        println!(
            "  {:<width$}  {:>12} bytes  sha256 {}",
            file.relpath, file.size, file.sha256
        );
    }
    println!("\nThe checksum is the SHA-256 of exactly this text:");
    print!("{}", package.manifest_text());
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
/// A file rather than a dozen flags: it is half of the package's identity, it is written
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
/// identity says what the package *declares*, and this says what will actually be done to
/// the text. The build refuses to proceed unless one hashes to the other.
fn read_chunking(path: &str) -> ChunkerConfig {
    let json = std::fs::read_to_string(path)
        .unwrap_or_else(|error| exit_with(&format!("Could not read {path}"), error));
    serde_json::from_str(&json)
        .unwrap_or_else(|error| exit_with(&format!("{path} is not a chunker configuration"), error))
}

/// Apply the recipe on the machine that holds the corpus, and write the work out.
///
/// No model is opened and none is needed: this is the half of a build that is arithmetic
/// on strings. What it writes is what a worker with the model and no corpus can act on.
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

    // The comparisons `build` makes, for the same reason: a worker that embeds with a
    // package of another family or a different width produces vectors the merge cannot
    // use, and it should learn that in the second it takes rather than at the end of the
    // shard.
    let checksum = runtime.model_checksum().unwrap_or_default().to_string();
    if !model
        .query_packages
        .iter()
        .any(|package| package.checksum == checksum)
    {
        exit_with(
            "The model file is not a package of the family the plan was made for",
            format!("its package checksum is {checksum}"),
        );
    }
    for (field, declared, loaded) in [
        (
            "tokenizer_checksum",
            model.tokenizer_checksum.clone(),
            runtime.tokenizer_checksum().unwrap_or_default().to_string(),
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
    // A finished shard is three files, and this refuses to write over one. Unlike the merge,
    // *re-running* is normal here — a session that timed out gets retried on another account
    // — so a directory holding leftovers is fair game and only a complete shard is protected.
    // Overwriting one silently discarded an hour of embedding and reported success.
    let manifest_path = out.join("shard-manifest.json");
    if [
        &out.join("vectors.f32"),
        &out.join("records.jsonl"),
        &manifest_path,
    ]
    .iter()
    .all(|path| path.symlink_metadata().is_ok())
    {
        exit_with(
            &format!("{} already holds a finished shard", out.display()),
            "its vectors, records and manifest are all there; embed into another directory, \
             or remove them to re-run this window",
        );
    }
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
    // Onto the disk before either name is published, and the manifest last: it is the digest
    // witness for both files, so a crash between the renames leaves a pair with the previous
    // manifest — which `verify_shards` refuses, loudly, because the digests will not match.
    for writer in [vectors, records] {
        writer
            .into_inner()
            .unwrap_or_else(|error| exit_with("Could not finish writing the shard", error))
            .sync_all()
            .unwrap_or_else(|error| exit_with("Could not flush the shard to disk", error));
    }
    for (partial, final_name) in [
        (&vectors_partial, "vectors.f32"),
        (&records_partial, "records.jsonl"),
    ] {
        std::fs::rename(partial, out.join(final_name))
            .unwrap_or_else(|error| exit_with("Could not publish the shard", error));
    }
    write_and_sync(&manifest_path, &serde_json::to_vec_pretty(&report).unwrap());
    sync_directory(&out);

    println!("\n=== Embedded a shard ===");
    println!("Path:            {}", out.display());
    println!("Records:         {} (skip {skip})", report.records);
    println!("Dimension:       {}", report.embedding_dim);
    println!("vectors SHA-256: {}", report.vectors_sha256);
    println!("records SHA-256: {}", report.records_sha256);
}

/// Flush a directory entry, so a rename this command has already reported survives a power
/// loss.
///
/// Unix only, and fatal there rather than best-effort — the same rule
/// [`crate::semantic::manifest`] follows: Windows cannot open a directory as a file, so the
/// rename is left as the filesystem's own guarantee, and where the call *is* available a
/// failure is not quietly downgraded to "probably durable".
#[cfg(unix)]
fn sync_directory(dir: &Path) {
    let handle = std::fs::File::open(dir).unwrap_or_else(|error| {
        exit_with(
            &format!("Could not open {} to flush its entries", dir.display()),
            error,
        )
    });
    handle.sync_all().unwrap_or_else(|error| {
        exit_with(
            &format!("Could not flush the directory entries of {}", dir.display()),
            error,
        )
    });
}

/// See the Unix implementation. Nothing to do here; documented, not silent.
#[cfg(not(unix))]
fn sync_directory(_dir: &Path) {}

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
            batch_size: parse_arg(args, "--batch")
                .and_then(|value| value.parse().ok())
                .unwrap_or(32),
            clip_q: parse_arg(args, "--clip-q").map_or(1.0, |value| {
                value
                    .parse()
                    .unwrap_or_else(|_| exit_with("--clip-q", "not a number"))
            }),
            allow_non_semantic_backend: args.iter().any(|arg| arg == "--allow-non-semantic"),
        },
        &corpus,
    )
    .unwrap_or_else(|error| exit_with("Build failed", error));

    let manifest = &report.manifest;
    println!("\n=== Built a base vector package ===");
    println!("Path:            {}", report.output_path.display());
    println!(
        "Segment:         {} ({} bytes, SHA-256 {})",
        SEGMENT_FILENAME, manifest.segment.size, manifest.segment.sha256
    );
    println!("Lines embedded:  {}", report.planned_lines);
    println!("Vectors:         {}", manifest.counts.slots);
    println!("Further records: {}", manifest.counts.extras);
    println!("Books:           {}", manifest.counts.books);
    println!(
        "Clipped:         {} component(s)",
        report.clipped_components
    );
    println!("Identity:        {}", manifest.identity);
    println!("Identity digest: {}", manifest.identity_digest);
    println!("Package digest:  {}", manifest.package_digest);
    println!(
        "Manifest:        {} (SHA-256 {})",
        RELEASE_MANIFEST_FILENAME, report.manifest_sha256
    );
    // The digest is only a trust anchor once it travels outside the release: recomputing
    // it from the manifest proves the manifest agrees with itself and nothing more.
    println!(
        "\nPublish the manifest's SHA-256 outside it. Installed without it, a release is \
         checked for damage\nand for the wrong identity, but not for one deliberately \
         rebuilt to match."
    );
}
