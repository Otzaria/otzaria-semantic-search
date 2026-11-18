//! Reproduce the prepared-text conformance fixture without a model or database.
//! Usage: cargo run --example recipe_fixtures -- [--check FILE | --write FILE]
use otzaria_semantic_search::semantic::chunk_key::{ChunkKey, LineRef};
use otzaria_semantic_search::semantic::chunker::{Chunker, ChunkerConfig};
use otzaria_semantic_search::semantic::types::{BookForIndexing, BookLine};
use serde_json::{json, Value};

fn line(id: u64, section: u64, text: impl Into<String>) -> BookLine {
    let text = text.into();
    BookLine {
        line_id: id,
        section_id: section,
        semantic_text: Some(text.clone()),
        text,
        semantic_notes: Vec::new(),
        line_hash: 0,
        reference: format!("fixture {id}"),
        segment: id - 1,
    }
}

/// Fixed inputs deliberately exercise the recipe's externally visible branches.
pub fn fixtures() -> Value {
    let config = ChunkerConfig {
        min_meaningful_chars: 20,
        context_window_lines: 2,
        max_chunk_chars: 64,
        min_embeddable_chars: 5,
        chunking_version: 3,
        embedding_text_version: 4,
        normalization_version: 1,
        window_chars: 64,
        window_overlap_chars: 16,
        max_windows_per_line: 255,
    };
    let mut noted = line(7, 3, "גוף הטקסט 12 עם הערת שוליים בנושא אחר וחזרה אל הטענה");
    noted.semantic_text = Some("גוף הטקסט עם חזרה אל הטענה העיקרית".to_string());
    noted.semantic_notes = vec![
        "הערת שוליים בנושא עצמאי שצריכה לקבל חלון משלה".to_string(),
        "רמבם פסק אחרת".to_string(),
        "א".to_string(),
    ];
    let mut note_only = line(8, 4, "12 הערה ללא טקסט ראשי");
    note_only.semantic_text = Some(String::new());
    note_only.semantic_notes = vec!["הערה עצמאית משמשת חלון ראשון כאשר גוף השורה ריק".to_string()];
    let book = BookForIndexing {
        source_book_key: "semantic-recipe-fixture.txt".to_string(),
        title: "דוגמאות התאמה למתכון".to_string(),
        content_fingerprint: 1,
        is_pdf: false,
        topics: String::new(),
        extra_facets: Vec::new(),
        lines: vec![
            line(1, 1, "טקסט מהסעיף הקודם אינו משמש הקשר לסעיף הבא"),
            line(2, 2, format!("{} סמוך לפני", "הקדמה ".repeat(15))),
            line(3, 2, "שורת עוגן"),
            line(4, 2, format!("סמוך אחרי {}", "המשך ".repeat(15))),
            line(5, 2, "  \t\u{00a0}  "),
            line(
                6,
                3,
                (0..40)
                    .map(|i| format!("מילה{i:02}"))
                    .collect::<Vec<_>>()
                    .join("\t"),
            ),
            noted,
            note_only,
            line(9, 5, "א".repeat(150)),
            line(10, 6, "אב"),
        ],
    };
    let chunker = Chunker::new(config.clone()).unwrap();
    chunker.validate_book_coverage(&book).unwrap();
    let refs: Vec<_> = book
        .lines
        .iter()
        .map(|line| LineRef {
            text: line.semantic_text.as_deref().unwrap_or(&line.text),
            section: line.section_id,
        })
        .collect();
    let lines: Vec<_> = book
        .lines
        .iter()
        .enumerate()
        .map(|(index, line)| {
            let windows = chunker.embedded_windows_with_notes(&refs, index, &line.semantic_notes);
            json!({ "line_id": line.line_id, "windows": windows.iter().map(|window| json!({
            "ordinal": window.ordinal, "chars": [window.chars.start, window.chars.end],
            "embedding_text": window.text, "key": ChunkKey::of(&window.text).to_hex(),
        })).collect::<Vec<_>>() })
        })
        .collect();
    json!({
        "format": "otzaria-semantic-recipe-conformance", "format_version": 1,
        "scope": "Prepared corpus text to semantic windows and keys; HTML/line normalization is upstream.",
        "character_offsets": "Unicode scalar values within trimmed main text or the individual note; context windows retain the short line's anchor range.",
        "chunking": config, "chunking_identity": config.identity(), "book": book, "expected": lines,
    })
}

fn main() {
    let result = fixtures();
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [] => println!("{}", serde_json::to_string_pretty(&result).unwrap()),
        [flag, path] if flag == "--write" => {
            std::fs::write(
                path,
                format!("{}\n", serde_json::to_string_pretty(&result).unwrap()),
            )
            .unwrap();
        }
        [flag, path] if flag == "--check" => {
            let expected: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
            assert_eq!(
                result, expected,
                "recipe fixture changed; review its version before regenerating"
            );
        }
        _ => {
            eprintln!("Usage: recipe_fixtures [--check FILE | --write FILE]");
            std::process::exit(1);
        }
    }
}
