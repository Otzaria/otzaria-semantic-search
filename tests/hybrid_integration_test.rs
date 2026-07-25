//! Integration tests for Otzaria Hybrid Semantic Search Engine.

use otzaria_semantic_search::api::hybrid_search::OtzariaHybridEngine;
use otzaria_semantic_search::hybrid::coordinator::{HybridCoordinator, HybridSearchParams};
use otzaria_semantic_search::semantic::engine::{SemanticEngine, SemanticConfig};
use otzaria_semantic_search::semantic::types::{
    BookForIndexing, BookLine, GroupingMode, LexicalCandidate, SearchMode,
};

fn create_mock_book() -> BookForIndexing {
    BookForIndexing {
        source_book_key: "otzaria/tanach/genesis.txt".to_string(),
        title: "בראשית".to_string(),
        content_hash: 987654,
        is_pdf: false,
        topics: vec!["/מקרא/תורה".to_string()],
        author: Some("משה רבנו".to_string()),
        era: Some("תנך".to_string()),
        base: None,
        lines: vec![
            BookLine {
                line_id: 1,
                section_id: 100,
                text: "בראשית ברא אלהים את השמים ואת הארץ".to_string(),
                line_hash: 11111,
                reference: "בראשית א:א".to_string(),
                segment: 1,
            },
            BookLine {
                line_id: 2,
                section_id: 100,
                text: "והארץ היתה תהו ובהו וחשך על פני תהום".to_string(),
                line_hash: 22222,
                reference: "בראשית א:ב".to_string(),
                segment: 2,
            },
        ],
    }
}

#[test]
fn test_semantic_engine_and_hybrid_coordinator_end_to_end() {
    let tmp_dir = tempfile::tempdir().unwrap();
    let config = SemanticConfig {
        root_dir: tmp_dir.path().to_path_buf(),
        ..Default::default()
    };

    let mut semantic_engine = SemanticEngine::open(config).unwrap();
    let book = create_mock_book();

    // Index mock book
    let chunks_indexed = semantic_engine.index_book(&book).unwrap();
    assert_eq!(chunks_indexed, 2);

    let status = semantic_engine.status();
    assert_eq!(status.indexed_book_count, 1);
    assert_eq!(status.vector_count, 2);

    // Build Coordinator
    let coordinator = HybridCoordinator::new(Some(semantic_engine));
    let engine_api = OtzariaHybridEngine::new(coordinator);

    // Mock Lexical Candidates
    let lexical_cands = vec![LexicalCandidate {
        title: "בראשית".to_string(),
        reference: "בראשית א:א".to_string(),
        text: "בראשית ברא אלהים את השמים ואת הארץ".to_string(),
        line_id: 1,
        section_id: 100,
        line_hash: 11111,
        segment: 1,
        is_pdf: false,
        file_path: "otzaria/tanach/genesis.txt".to_string(),
        bm25_score: 15.5,
    }];

    // Perform Search
    let result = engine_api
        .search(
            "בריאת העולם".to_string(),
            lexical_cands,
            Some(10),
            Some(0),
            Some(GroupingMode::SameSection),
            None,
            None,
        )
        .unwrap();

    assert!(result.total_count > 0);
    assert_eq!(result.search_mode, SearchMode::Hybrid);
    assert!(result.semantic_available);
    assert!(!result.results.is_empty());
}
