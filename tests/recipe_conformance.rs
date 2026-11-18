//! External implementations consume this JSON to verify their prepared-text recipe.
#[allow(dead_code)]
#[path = "../examples/recipe_fixtures.rs"]
mod generator;

#[test]
fn versioned_recipe_matches_published_conformance_fixture() {
    let fixture =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/semantic_recipe_v3.json");
    let expected: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture).unwrap()).unwrap();
    assert_eq!(generator::fixtures(), expected);
}
