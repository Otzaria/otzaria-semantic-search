//! Binary CLI interface for `otzaria-semantic-search`.
//!
//! Provides a standalone command-line application for querying, indexing, and
//! inspecting the Otzaria semantic search engine.

use otzaria_semantic_search::api::hybrid_search::{OtzariaHybridEngine, SearchRequest};
use otzaria_semantic_search::hybrid::coordinator::HybridCoordinator;
use otzaria_semantic_search::semantic::engine::{SemanticConfig, SemanticEngine};
use otzaria_semantic_search::semantic::types::{BookForIndexing, BookLine, SearchMode};
use std::env;
use std::path::PathBuf;
use std::process;

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

Options for 'search':
  --dir <path>       Directory holding semantic database (default: "./semantic_db")
  --mode <mode>      Retrieval mode: hybrid (default), semantic, lexical
  --limit <N>        Maximum results to return (default: 10)

Options for 'status':
  --dir <path>       Directory holding semantic database (default: "./semantic_db")

Examples:
  otzaria-semantic-search version
  otzaria-semantic-search status --dir ./semantic_db
  otzaria-semantic-search search "מצות תפילין" --mode semantic --limit 5
  otzaria-semantic-search index-text "otzaria/demo.txt" "ספר הדגמה" "כל העוסק בתורה בלילה שכינה כנגדו"
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
