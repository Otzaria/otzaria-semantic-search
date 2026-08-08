//! Binary CLI interface for `otzaria-semantic-search`.
//!
//! Provides a standalone command-line application for querying, indexing, and
//! inspecting the Otzaria semantic search engine, and the build-side `pack` and
//! `validate` commands that produce an official artifact from ready-made vectors.
//!
//! `pack` and `validate` need no embedding backend — they never turn text into a vector —
//! so they work in a default build, which is the one a release pipeline has.

use otzaria_semantic_search::api::hybrid_search::{OtzariaHybridEngine, SearchRequest};
use otzaria_semantic_search::distribution::corpus::JsonlCorpus;
use otzaria_semantic_search::distribution::packer::{
    pack, read_vector_inputs, validate_artifact, PackReport, PackRequest,
};
use otzaria_semantic_search::hybrid::coordinator::HybridCoordinator;
use otzaria_semantic_search::semantic::engine::{SemanticConfig, SemanticEngine};
use otzaria_semantic_search::semantic::types::{BookForIndexing, BookLine, SearchMode};
use otzaria_semantic_search::semantic::versioning::ModelIdentity;
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
  pack [options]                      Build an official artifact from ready-made vectors.
  validate [options]                  Verify an artifact against a corpus and a model.

Options for 'search':
  --dir <path>       Directory holding semantic database (default: "./semantic_db")
  --mode <mode>      Retrieval mode: hybrid (default), semantic, lexical
  --limit <N>        Maximum results to return (default: 10)

Options for 'status':
  --dir <path>       Directory holding semantic database (default: "./semantic_db")

Options for 'pack':
  --vectors <path>           Raw little-endian f32 vectors, count x embedding_dim, no header
  --records <path>           JSONL, one {{"line_id":N,"line_sha256":"..."}} per vector, same order
  --corpus-identity <path>   JSON CorpusIdentity, as the lexical index reports it
  --corpus-lines <path>      JSONL, one corpus line per document
  --model <path>             JSON ModelIdentity describing how the vectors were produced
  --out <dir>                Output directory; must not exist, or be empty
  --collection <name>        Collection name in the payload header (default: "chunks")
  --created-at <timestamp>   Manifest timestamp (default: now, UTC)

Options for 'validate':
  --artifact <dir>           The artifact directory to verify
  --corpus-identity <path>   As above
  --corpus-lines <path>      As above
  --model <path>             As above

`line_sha256` is the SHA-256 of the corpus line's text, in lowercase hex. It is what
proves each vector was built from the line its id names; without it a shifted vector
file would pack without complaint.

Examples:
  otzaria-semantic-search version
  otzaria-semantic-search status --dir ./semantic_db
  otzaria-semantic-search search "מצות תפילין" --mode semantic --limit 5
  otzaria-semantic-search index-text "otzaria/demo.txt" "ספר הדגמה" "כל העוסק בתורה בלילה שכינה כנגדו"
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

fn run_pack(args: &[String]) {
    let vectors = require_arg(args, "--vectors");
    let records = require_arg(args, "--records");
    let out = require_arg(args, "--out");
    let model = read_model(&require_arg(args, "--model"));
    let corpus = load_corpus(args);

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
        &corpus,
    )
    .unwrap_or_else(|error| exit_with("Packing failed", error));

    println!("\n=== Packed an official artifact ===");
    print_report(&report);
}

fn run_validate(args: &[String]) {
    let artifact = require_arg(args, "--artifact");
    let model = read_model(&require_arg(args, "--model"));
    let corpus = load_corpus(args);

    let report = validate_artifact(Path::new(&artifact), &model, &corpus)
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
