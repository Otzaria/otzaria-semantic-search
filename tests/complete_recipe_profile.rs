use otzaria_semantic_search::semantic::chunker::{Chunker, ChunkerConfig};
use otzaria_semantic_search::semantic::recipe::EmbeddingRecipe;
use otzaria_semantic_search::semantic::versioning::ModelIdentity;

#[test]
fn complete_profile_names_implemented_recipe_and_preserves_old_identity() {
    let config: ChunkerConfig = serde_json::from_str(include_str!(
        "../config/models/meivin-qa491-complete-notes-onnx/chunking.json"
    ))
    .unwrap();
    let model: ModelIdentity = serde_json::from_str(include_str!(
        "../config/models/meivin-qa491-complete-notes-onnx/model.json"
    ))
    .unwrap();
    assert_eq!(
        (
            config.chunking_version,
            config.embedding_text_version,
            config.max_windows_per_line
        ),
        (3, 4, 255)
    );
    assert_eq!(config.identity(), 12_470_596_585_207_933_847);
    assert_eq!(config.identity(), model.chunking_identity);
    EmbeddingRecipe::resolve(&config, &model).unwrap();
    Chunker::new(config).unwrap();
    let old: ChunkerConfig = serde_json::from_str(include_str!(
        "../config/models/meivin-qa491-windows-onnx/chunking.json"
    ))
    .unwrap();
    assert_eq!(old.identity(), 320_260_894_818_558_297);
}
