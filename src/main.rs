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
  plan [options]                      Apply the recipe to a corpus and write a vector
                                      build's plan, split against the release before.
  embed-shard [options]               Embed a window of a plan's embed.jsonl on this CPU,
                                      in one process or several.
  adopt-shard [options]               Write the v2 manifest of an external worker's output,
                                      so a warehouse can import it.
  warehouse-add [options]             Check shards and add their vectors to a warehouse.
  warehouse-verify [options]          Re-hash a warehouse against its digests and check its
                                      index; with --repair, rebuild a bad index.
  assemble [options]                  Assemble a base or a delta from a plan and the
                                      warehouse; with --verify, check gates G1, G5, G7-G10.
  release-files [options]             List the files a release downloads as, in the shape
                                      of the updater's patch entries, in its manifest.
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
  --codec <name>             i8-sym-vec (default: an int8 scale per vector), i8-sym-dim
                             (a scale per dimension, calibrated) or f32. The application
                             reads the default only
  --clip-q <q>               For i8-sym-dim: the quantile each dimension's scale is
                             calibrated at, in (0, 1] (default: 1, which clips nothing)
  --created-at <timestamp>   Manifest timestamp (default: now, UTC)
  --allow-non-semantic       Write a package from a backend whose vectors mean nothing.
                             For tests only: such a package passes every check here and
                             answers nonsense.

Options for 'plan':
  --corpus-identity <path>   As for 'build'
  --corpus-lines <path>      As for 'build'
  --model <path>             The family the vectors are for (JSON ModelIdentity)
  --chunking <path>          The recipe to apply
  --passage-quantization <q> Which of the family's packages embeds the passages
                             (default: fp32)
  --previous-ledger <dir>    The ledger of the release before; omit for a first base
  --warehouse <dir>          Leave out of embed.jsonl the texts it holds vectors for
  --out <dir>                Receives records.bin, books.json, embed.jsonl,
                             embed-manifest.json, tombstones.bin and plan-manifest.json
  --created-at <timestamp>   Manifest timestamp (default: now, UTC)

Options for 'embed-shard':
  --plan <dir>               The plan: embed.jsonl and embed-manifest.json. The manifest
                             names the family and the passage package; nothing else is read
  --model-file <path>        The ONNX graph of the passage package; held to every field
                             the family declares, and to the package checksum
  --skip <N>                 Records to skip (default: 0)
  --take <N>                 Records to embed (default: all that remain)
  --batch <N>                Texts per inference call (default: 32)
  --processes <P>            Split the window among P processes, each writing its own
                             shard into <out>/shard-NNN (default: 1, a shard in <out>)
  --threads <T>              ONNX Runtime threads per process (OTZARIA_ONNX_THREADS)
  --out <dir>                Receives vectors.f32, keys.bin, shard-manifest.json.
                             Leftovers from a session that died are overwritten — retrying a
                             window is normal — but a finished shard is not.
  --allow-non-semantic       As for 'build'

Options for 'adopt-shard':
  --dir <dir>                Holds vectors.f32 and keys.bin (32 raw bytes a record: rename
                             a worker's keys.sha256); receives shard-manifest.json
  --model <path>             The family (JSON ModelIdentity)
  --passage-quantization <q> The package the vectors were embedded with (default: fp32)
  --plan-sha256 <hex>        The digest of the plan the worker embedded, for the record
  --worker-name, --worker-version, --device, --ep, --mode
                             What ran it: mode `onnxruntime` for ONNX Runtime with the
                             shipped graph, anything else for a re-implementation
  --parity-reference <text>, --parity-samples <N>, --parity-min-cosine <x>,
  --parity-mean-cosine <x>, --parity-document <path>
                             The parity certificate, required unless ep is cpu and mode is
                             onnxruntime

Options for 'warehouse-add':
  --warehouse <dir>          The warehouse
  --create                   Create it if missing, for --model's --passage-quantization
  --model <path>             With --create: the family
  --passage-quantization <q> With --create: the package (default: fp32)
  --plan <dir>               The plan the shards were cut from; without it, an import:
                             everything but the plan's own keys is checked
  --shards <dir>             A shard, or a directory holding shards at any depth; repeat
  --allow-non-semantic       Accept the stand-in's vectors (tests only)

  The warehouse is verified first, as 'warehouse-verify --repair' does, and nothing is
  added to one that fails.

Options for 'warehouse-verify':
  --warehouse <dir>          The warehouse; assemble and warehouse-add verify it themselves
  --repair                   Under the append lock, cut what a crash left past the count
                             and rebuild an index that is not the verified keys'. Data
                             that fails its digests is never repaired

Options for 'assemble':
  --plan <dir>               The plan
  --warehouse <dir>          The warehouse, holding every vector the release ships
  --out <dir>                The release: segment.oxv, manifest.json, payloads.json,
                             release.json, and this version's ledger
  --kind <base|delta>        What to assemble; without it, --verify checks the release
                             already in --out
  --previous <dir>           The ledger of the release the plan was split against:
                             required for a delta and for --keep-epoch
  --previous-version <N>     Its version, when the directory holds several
  --codec <name>             A base's codec epoch: i8-sym-vec (default), i8-sym-dim, f32
  --clip-q <q>               For i8-sym-dim: the calibration quantile (default: 1)
  --keep-epoch               A base in the previous release's codec epoch
  --created-at <time>        The manifest's createdAt (default: now); the same inputs and
                             time give the same bytes
  --built-by <json>          The manifest's builtBy, e.g. {{"runId":"..."}}
  --verify                   Check the gates and write <out>/gates.json; a failed gate
                             exits with status 2
  --samples <N>              G5's sample (default: 20000)

Options for 'release-files':
  --release <path>           An assembled release.json
  --files <path>             A file the release downloads as, in order; repeat
  --compression <name>       How they are compressed (default: zstd)
  --out <path>               The manifest to write, with its files

Options for 'model-checksum':
  --model-file <path>        An .onnx graph, its tokenizer.json beside it. No other kind of
                             model is read: a path that does not end in .onnx is refused.

An ONNX model is a package: the graph, the tokenizer.json beside it and every external-data
file the graph names, and the checksum is the SHA-256 of a manifest listing each of them
with its size and SHA-256 — printed here exactly as it is hashed. Nothing else in the
directory is part of it: not a README, not a second graph, not an ONNX Runtime library.

A shard is the external embedding interface: docs/VECTOR_BUILD.md describes embed.jsonl,
vectors.f32, keys.bin and shard-manifest.json exactly, for a worker that is not this CLI.

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
        "plan" => run_plan(&args),
        "embed-shard" => run_embed_shard(&args),
        "adopt-shard" => run_adopt_shard(&args),
        "warehouse-add" => run_warehouse_add(&args),
        "warehouse-verify" => run_warehouse_verify(&args),
        "assemble" => run_assemble(&args),
        "release-files" => run_release_files(&args),
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

/// Write a vector build's plan from a corpus transcription.
fn run_plan(args: &[String]) {
    use otzaria_semantic_search::distribution::ledger::Ledger;
    use otzaria_semantic_search::distribution::plan::{plan_from_corpus, PlanRequest};

    let model = read_model(&require_arg(args, "--model"));
    let quantization = parse_arg(args, "--passage-quantization").unwrap_or_else(|| "fp32".into());
    let passage_package = model
        .query_packages
        .iter()
        .find(|package| package.quantization == quantization)
        .cloned()
        .unwrap_or_else(|| {
            exit_with(
                "The family has no such package",
                format!("no {quantization} package among its query packages"),
            )
        });
    let previous = parse_arg(args, "--previous-ledger").map(|dir| {
        Ledger::open(Path::new(&dir), None)
            .unwrap_or_else(|error| exit_with("Could not open the previous ledger", error))
    });
    let warehouse = parse_arg(args, "--warehouse").map(|dir| {
        otzaria_semantic_search::distribution::warehouse::Warehouse::open(Path::new(&dir))
            .unwrap_or_else(|error| exit_with("Could not open the warehouse", error))
    });
    let corpus = load_corpus(args);
    let manifest = plan_from_corpus(
        &corpus,
        PlanRequest {
            out_dir: PathBuf::from(require_arg(args, "--out")),
            model,
            chunking: read_chunking(&require_arg(args, "--chunking")),
            passage_package,
            previous: previous.as_ref(),
            warehouse: warehouse.as_ref().map(|warehouse| {
                warehouse as &dyn otzaria_semantic_search::distribution::plan::HeldVectors
            }),
            created_at: parse_arg(args, "--created-at")
                .unwrap_or_else(|| utc_timestamp(SystemTime::now())),
        },
    )
    .unwrap_or_else(|error| exit_with("Planning failed", error));

    let counts = manifest.counts;
    println!("\n=== Planned v{} ===", manifest.library_version);
    println!("Records:         {}", counts.records);
    println!("Books:           {}", counts.books);
    println!("Distinct keys:   {}", counts.unique);
    println!("Reused:          {}", counts.reused);
    println!("To ship:         {}", counts.to_ship);
    println!("To embed:        {}", counts.to_embed);
    println!("Revived:         {}", counts.revived);
    println!("Tombstones:      {}", counts.tombstones);
    println!("Foreign pairs:   {}", counts.foreign_pairs);
}

/// Embed a window of a plan on this machine's CPU: one shard, or — with `--processes` — one
/// child process per sub-window, each writing its own.
fn run_embed_shard(args: &[String]) {
    use otzaria_semantic_search::distribution::plan::EmbedManifest;
    use otzaria_semantic_search::distribution::shard::{embed_shard, WorkerInfo, MODE_MOCK};

    let out = PathBuf::from(require_arg(args, "--out"));
    let plan_dir = PathBuf::from(require_arg(args, "--plan"));
    let plan = EmbedManifest::read(&plan_dir)
        .unwrap_or_else(|error| exit_with("Could not read the plan", error));
    let number = |flag: &str, default: u64| {
        parse_arg(args, flag).map_or(default, |value| {
            value
                .parse()
                .unwrap_or_else(|_| exit_with(flag, "not a number"))
        })
    };
    let skip = number("--skip", 0);
    let take = number("--take", u64::MAX).min(plan.records.saturating_sub(skip));
    let processes = number("--processes", 1).max(1);

    if processes > 1 {
        // Sub-windows as even as integers allow; each child is this command with its own.
        let per = take.div_ceil(processes);
        let mut children = Vec::new();
        for index in 0..processes {
            let from = skip + index * per;
            let count = per.min((skip + take).saturating_sub(from));
            if count == 0 {
                break;
            }
            let dir = out.join(format!("shard-{index:03}"));
            let mut command = process::Command::new(
                env::current_exe().unwrap_or_else(|error| exit_with("Cannot find myself", error)),
            );
            command.args(["embed-shard", "--plan"]).arg(&plan_dir);
            for flag in ["--model-file", "--batch"] {
                if let Some(value) = parse_arg(args, flag) {
                    command.args([flag, &value]);
                }
            }
            command
                .args(["--skip", &from.to_string(), "--take", &count.to_string()])
                .arg("--out")
                .arg(&dir);
            if args.iter().any(|arg| arg == "--allow-non-semantic") {
                command.arg("--allow-non-semantic");
            }
            if let Some(threads) = parse_arg(args, "--threads") {
                command.env("OTZARIA_ONNX_THREADS", threads);
            }
            let child = command
                .spawn()
                .unwrap_or_else(|error| exit_with("Could not start a worker process", error));
            children.push((dir, child));
        }
        let mut failed = 0;
        for (dir, mut child) in children {
            let status = child
                .wait()
                .unwrap_or_else(|error| exit_with("A worker process was lost", error));
            if !status.success() {
                eprintln!("{} failed: {status}", dir.display());
                failed += 1;
            }
        }
        if failed > 0 {
            exit_with(
                "Embedding failed",
                format!("{failed} worker process(es) failed"),
            );
        }
        println!("\n=== Embedded {take} record(s) from {skip} in {processes} processes ===");
        return;
    }

    if let Some(threads) = parse_arg(args, "--threads") {
        env::set_var("OTZARIA_ONNX_THREADS", threads);
    }
    let mut runtime = EmbeddingRuntime::new(EmbeddingConfig {
        model_path: PathBuf::from(require_arg(args, "--model-file")),
        embedding_dim: plan.model.embedding_dim,
        max_tokens: plan.model.max_tokens,
        batch_size: parse_arg(args, "--batch")
            .and_then(|value| value.parse().ok())
            .unwrap_or(32),
        pooling: Pooling::parse(&plan.model.pooling).unwrap_or_else(|error| {
            exit_with("The family declares a pooling nothing performs", error)
        }),
    });
    runtime
        .load()
        .unwrap_or_else(|error| exit_with("Could not load the model", error));
    let semantic = runtime.backend_is_semantic();
    if !semantic && !args.iter().any(|arg| arg == "--allow-non-semantic") {
        exit_with(
            "This backend's vectors mean nothing",
            runtime.backend_id().unwrap_or("none").to_string(),
        );
    }
    let worker = WorkerInfo {
        name: env!("CARGO_PKG_NAME").to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        device: otzaria_semantic_search::semantic::embedding::cpu_description(),
        ep: "cpu".to_string(),
        mode: if semantic {
            otzaria_semantic_search::distribution::shard::MODE_ONNXRUNTIME.to_string()
        } else {
            MODE_MOCK.to_string()
        },
    };
    let manifest = embed_shard(
        &plan_dir,
        &plan,
        skip,
        take,
        &runtime,
        runtime.batch_size(),
        worker,
        &out,
    )
    .unwrap_or_else(|error| exit_with("The shard failed", error));

    println!("\n=== Embedded a shard ===");
    println!("Path:            {}", out.display());
    println!("Records:         {} (skip {skip})", manifest.records);
    println!("Dimension:       {}", manifest.dim);
    println!("vectors SHA-256: {}", manifest.vectors_sha256);
    println!("keys SHA-256:    {}", manifest.keys_sha256);
}

/// Every value of a repeated flag.
fn parse_args_all(args: &[String], flag: &str) -> Vec<String> {
    args.windows(2)
        .filter(|pair| pair[0] == flag)
        .map(|pair| pair[1].clone())
        .collect()
}

/// The package of `model` whose quantization is `--passage-quantization` (default fp32).
fn passage_package_of(
    args: &[String],
    model: &ModelIdentity,
) -> otzaria_semantic_search::semantic::versioning::ModelPackage {
    let quantization = parse_arg(args, "--passage-quantization").unwrap_or_else(|| "fp32".into());
    model
        .query_packages
        .iter()
        .find(|package| package.quantization == quantization)
        .cloned()
        .unwrap_or_else(|| {
            exit_with(
                "The family has no such package",
                format!("no {quantization} package among its query packages"),
            )
        })
}

/// Write the v2 shard manifest of an external worker's output.
fn run_adopt_shard(args: &[String]) {
    use otzaria_semantic_search::distribution::shard::{
        ParityCertificate, ShardManifest, WorkerInfo, KEYS_FILE, SHARD_FORMAT,
        SHARD_FORMAT_VERSION, SHARD_MANIFEST_FILE, VECTORS_FILE,
    };
    let dir = PathBuf::from(require_arg(args, "--dir"));
    let model = read_model(&require_arg(args, "--model"));
    let passage_package = passage_package_of(args, &model);
    let digest = |name: &str| {
        let path = dir.join(name);
        let length = std::fs::metadata(&path)
            .unwrap_or_else(|error| exit_with(&format!("Could not read {}", path.display()), error))
            .len();
        (length, sha256_file(&path))
    };
    let (keys_length, keys_sha256) = digest(KEYS_FILE);
    let (vectors_length, vectors_sha256) = digest(VECTORS_FILE);
    let records = keys_length / 32;
    if keys_length % 32 != 0 || vectors_length != records * u64::from(model.embedding_dim) * 4 {
        exit_with(
            "The files do not describe one set of records",
            format!(
                "{KEYS_FILE} is {keys_length} bytes and {VECTORS_FILE} {vectors_length}, \
                 for {}-wide vectors",
                model.embedding_dim
            ),
        );
    }
    let float = |flag: &str| {
        parse_arg(args, flag).map(|value| {
            value
                .parse::<f64>()
                .unwrap_or_else(|_| exit_with(flag, "not a number"))
        })
    };
    let parity = parse_arg(args, "--parity-reference").map(|reference| ParityCertificate {
        reference,
        samples: parse_arg(args, "--parity-samples")
            .and_then(|value| value.parse().ok())
            .unwrap_or_else(|| exit_with("--parity-samples", "required with a certificate")),
        min_cosine: float("--parity-min-cosine")
            .unwrap_or_else(|| exit_with("--parity-min-cosine", "required with a certificate")),
        mean_cosine: float("--parity-mean-cosine")
            .unwrap_or_else(|| exit_with("--parity-mean-cosine", "required with a certificate")),
        document_sha256: parse_arg(args, "--parity-document")
            .map(|path| sha256_file(Path::new(&path))),
    });
    let manifest = ShardManifest {
        format: SHARD_FORMAT.to_string(),
        version: SHARD_FORMAT_VERSION,
        plan_sha256: require_arg(args, "--plan-sha256"),
        skip: 0,
        take: records,
        records,
        dim: model.embedding_dim,
        vectors_sha256,
        keys_sha256,
        model,
        passage_package,
        worker: WorkerInfo {
            name: require_arg(args, "--worker-name"),
            version: require_arg(args, "--worker-version"),
            device: require_arg(args, "--device"),
            ep: require_arg(args, "--ep"),
            mode: require_arg(args, "--mode"),
        },
        parity,
    };
    std::fs::write(
        dir.join(SHARD_MANIFEST_FILE),
        serde_json::to_vec_pretty(&manifest).expect("a manifest serializes"),
    )
    .unwrap_or_else(|error| exit_with("Could not write the shard manifest", error));
    println!(
        "\n=== Adopted {} record(s) in {} ===",
        records,
        dir.display()
    );
}

/// Add shards to a warehouse, creating it when asked.
fn run_warehouse_add(args: &[String]) {
    use otzaria_semantic_search::distribution::shard::{ShardPolicy, SHARD_MANIFEST_FILE};
    use otzaria_semantic_search::distribution::warehouse::{Warehouse, WarehouseIdentity};

    let dir = PathBuf::from(require_arg(args, "--warehouse"));
    if args.iter().any(|arg| arg == "--create") && !dir.join("warehouse.json").exists() {
        let model = read_model(&require_arg(args, "--model"));
        let package = passage_package_of(args, &model);
        Warehouse::create(&dir, WarehouseIdentity::of(&model, &package))
            .unwrap_or_else(|error| exit_with("Could not create the warehouse", error));
    }
    fn collect(root: &Path, found: &mut Vec<PathBuf>) {
        if root.join(SHARD_MANIFEST_FILE).exists() {
            found.push(root.to_path_buf());
            return;
        }
        let mut entries: Vec<PathBuf> = std::fs::read_dir(root)
            .map(|entries| entries.flatten().map(|entry| entry.path()).collect())
            .unwrap_or_default();
        entries.sort();
        for entry in entries.into_iter().filter(|path| path.is_dir()) {
            collect(&entry, found);
        }
    }
    let mut shards = Vec::new();
    for root in parse_args_all(args, "--shards") {
        collect(Path::new(&root), &mut shards);
    }
    if shards.is_empty() {
        exit_with("Nothing to add", "no shard-manifest.json under --shards");
    }
    let mut warehouse = Warehouse::open_for_append(&dir)
        .unwrap_or_else(|error| exit_with("Could not open the warehouse", error));
    print_verified(&warehouse);
    let plan = parse_arg(args, "--plan").map(PathBuf::from);
    let report = warehouse
        .add_shards(
            plan.as_deref(),
            &shards,
            &ShardPolicy {
                allow_non_semantic: args.iter().any(|arg| arg == "--allow-non-semantic"),
            },
            utc_timestamp(SystemTime::now()),
        )
        .unwrap_or_else(|error| exit_with("The shards were refused", error));
    println!("\n=== Added to {} ===", dir.display());
    println!("Shards:          {}", report.shards);
    println!("Records:         {}", report.records);
    println!("Added:           {}", report.added);
    println!("Held already:    {}", report.held);
    println!("Warehouse:       {} record(s)", report.total);
}

/// Verify a warehouse, or repair what `warehouse-add` would before adding.
fn run_warehouse_verify(args: &[String]) {
    use otzaria_semantic_search::distribution::warehouse::Warehouse;

    let dir = PathBuf::from(require_arg(args, "--warehouse"));
    let started = std::time::Instant::now();
    let warehouse = if args.iter().any(|arg| arg == "--repair") {
        Warehouse::open_for_append(&dir)
    } else {
        Warehouse::open(&dir)
    }
    .unwrap_or_else(|error| exit_with("Could not open the warehouse", error));
    print_verified(&warehouse);
    println!("Took:            {:.1} s", started.elapsed().as_secs_f64());
}

/// Verify `warehouse`, once per process, or exit naming what failed.
fn print_verified(warehouse: &otzaria_semantic_search::distribution::warehouse::Warehouse) {
    let verified = warehouse
        .verify()
        .unwrap_or_else(|error| exit_with("The warehouse failed its check", error));
    println!(
        "Verified:        {} record(s) in {} batch(es), {} bytes re-hashed",
        verified.records, verified.batches, verified.bytes
    );
    if let Some(reason) = &verified.index_rebuilt {
        println!("Index rebuilt:   {reason}");
    }
}

fn run_assemble(args: &[String]) {
    use otzaria_semantic_search::distribution::assemble::{
        assemble, AssembleRequest, EpochChoice, RELEASE_FILE,
    };
    use otzaria_semantic_search::distribution::gates::{
        verify_release, GateStatus, VerifyRequest, G5_SAMPLES,
    };
    use otzaria_semantic_search::distribution::ledger::Ledger;
    use otzaria_semantic_search::distribution::package::PackageKind;
    use otzaria_semantic_search::distribution::plan::Plan;
    use otzaria_semantic_search::distribution::warehouse::Warehouse;
    use otzaria_semantic_search::semantic::oxv::codec::CodecSpec;

    let flag = |name: &str| args.iter().any(|arg| arg == name);
    let out = PathBuf::from(require_arg(args, "--out"));
    let plan = Plan::open(Path::new(&require_arg(args, "--plan")))
        .unwrap_or_else(|error| exit_with("Could not open the plan", error));
    let warehouse = Warehouse::open(Path::new(&require_arg(args, "--warehouse")))
        .unwrap_or_else(|error| exit_with("Could not open the warehouse", error));
    print_verified(&warehouse);
    let previous = parse_arg(args, "--previous").map(|dir| {
        let version = parse_arg(args, "--previous-version").map(|value| {
            value
                .parse::<u32>()
                .unwrap_or_else(|_| exit_with("--previous-version", "not a number"))
        });
        Ledger::open(Path::new(&dir), version)
            .unwrap_or_else(|error| exit_with("Could not open the previous ledger", error))
    });
    if let Some(kind) = parse_arg(args, "--kind") {
        let kind = match kind.as_str() {
            "base" => PackageKind::Base,
            "delta" => PackageKind::Delta,
            other => exit_with("--kind", format!("{other:?} is neither base nor delta")),
        };
        let epoch = if flag("--keep-epoch") {
            EpochChoice::Previous
        } else {
            let clip_q = parse_arg(args, "--clip-q").map_or(1.0, |value| {
                value
                    .parse::<f32>()
                    .unwrap_or_else(|_| exit_with("--clip-q", "not a number"))
            });
            let name = parse_arg(args, "--codec").unwrap_or_else(|| "i8-sym-vec".to_string());
            EpochChoice::New(
                CodecSpec::parse(&name, clip_q).unwrap_or_else(|error| exit_with("--codec", error)),
            )
        };
        let built_by = parse_arg(args, "--built-by").map(|json| {
            serde_json::from_str(&json).unwrap_or_else(|error| exit_with("--built-by", error))
        });
        let report = assemble(&AssembleRequest {
            plan: &plan,
            warehouse: &warehouse,
            kind,
            previous: previous.as_ref(),
            epoch,
            out_dir: out.clone(),
            created_at: parse_arg(args, "--created-at")
                .unwrap_or_else(|| utc_timestamp(SystemTime::now())),
            built_by,
        })
        .unwrap_or_else(|error| exit_with("Assembly failed", error));
        let manifest = &report.manifest;
        println!("\n=== Assembled {} ===", out.display());
        println!("Kind:            {:?}", manifest.kind);
        println!(
            "Library:         v{} -> v{} ({})",
            manifest.from_library_version,
            manifest.to_library_version,
            manifest.library_release_tag
        );
        println!(
            "Codec:           {}",
            manifest.identity.store.vector_precision
        );
        println!(
            "Counts:          {} slot(s), {} extra(s), {} foreign, {} tombstone(s), {} book(s)",
            manifest.counts.slots,
            manifest.counts.extras,
            manifest.counts.foreign,
            manifest.counts.tombstones,
            manifest.counts.books
        );
        println!(
            "Segment:         {} bytes, SHA-256 {}",
            manifest.segment.size, manifest.segment.sha256
        );
        println!("Package digest:  {}", manifest.package_digest);
        println!("Manifest SHA-256 {}", report.manifest_sha256);
        println!(
            "Clipped:         {} component(s)",
            report.clipped_components
        );
        println!(
            "Ledger:          v{}, {} key(s), {} pair(s)",
            report.ledger.library_version, report.ledger.keys.count, report.ledger.pairs.count
        );
    } else if !flag("--verify") {
        exit_with(
            "Nothing to do",
            "give --kind to assemble, --verify to check, or both",
        );
    } else if !out.join(RELEASE_FILE).exists() {
        exit_with(
            "Nothing to verify",
            format!("{} holds no {RELEASE_FILE}", out.display()),
        );
    }
    if flag("--verify") {
        let samples = parse_arg(args, "--samples").map_or(G5_SAMPLES, |value| {
            value
                .parse()
                .unwrap_or_else(|_| exit_with("--samples", "not a number"))
        });
        let report = verify_release(&VerifyRequest {
            release_dir: &out,
            plan: &plan,
            warehouse: &warehouse,
            previous: previous.as_ref(),
            scratch_dir: out.with_extension("g8"),
            samples,
        })
        .unwrap_or_else(|error| exit_with("The release could not be verified", error));
        std::fs::write(
            out.join("gates.json"),
            serde_json::to_vec_pretty(&report).expect("a report serializes"),
        )
        .unwrap_or_else(|error| exit_with("Could not write gates.json", error));
        println!("\n=== Gates ===");
        for gate in &report.gates {
            println!(
                "{:<4} {}  {}",
                gate.gate,
                match gate.status {
                    GateStatus::Passed => "pass",
                    GateStatus::Failed => "FAIL",
                    GateStatus::NotApplicable => "n/a ",
                },
                gate.detail
            );
        }
        if !report.passed() {
            process::exit(2);
        }
    }
}

fn run_release_files(args: &[String]) {
    use otzaria_semantic_search::distribution::assemble::{asset_stem, release_with_files};
    let files: Vec<PathBuf> = parse_args_all(args, "--files")
        .into_iter()
        .map(PathBuf::from)
        .collect();
    if files.is_empty() {
        exit_with("Nothing to list", "give each file with --files");
    }
    let (manifest, sha256) = release_with_files(
        Path::new(&require_arg(args, "--release")),
        &files,
        &parse_arg(args, "--compression").unwrap_or_else(|| "zstd".to_string()),
        Path::new(&require_arg(args, "--out")),
    )
    .unwrap_or_else(|error| exit_with("Could not list the files", error));
    println!("\n=== {} ===", asset_stem(&manifest));
    for file in &manifest.files {
        println!("{}  {} bytes  {}", file.file, file.size, file.sha256);
    }
    println!("Manifest SHA-256 {sha256}");
}

/// SHA-256 of a file, streamed; exits naming the file when it cannot be read.
fn sha256_file(path: &Path) -> String {
    use std::io::Read;
    let mut file = std::fs::File::open(path)
        .unwrap_or_else(|error| exit_with(&format!("Could not read {}", path.display()), error));
    let mut hasher = <sha2::Sha256 as sha2::Digest>::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let read = file.read(&mut buffer).unwrap_or_else(|error| {
            exit_with(&format!("Could not read {}", path.display()), error)
        });
        if read == 0 {
            break;
        }
        sha2::Digest::update(&mut hasher, &buffer[..read]);
    }
    format!("{:x}", sha2::Digest::finalize(hasher))
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
            codec: otzaria_semantic_search::semantic::oxv::codec::CodecSpec::parse(
                &parse_arg(args, "--codec").unwrap_or_else(|| "i8-sym-vec".to_string()),
                parse_arg(args, "--clip-q").map_or(1.0, |value| {
                    value
                        .parse()
                        .unwrap_or_else(|_| exit_with("--clip-q", "not a number"))
                }),
            )
            .unwrap_or_else(|error| exit_with("--codec", error)),
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
