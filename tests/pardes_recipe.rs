use otzaria_semantic_search::semantic::recipe::{
    query_input, EmbeddingTextRecipe, TextNormalizationRecipe,
};
#[test]
fn pardes_roles_clean_then_prefix_without_meivin_markers() {
    let text = EmbeddingTextRecipe::from_version(3).unwrap();
    let norm = TextNormalizationRecipe::from_version(2).unwrap();
    let raw = "  <b>ש״ס</b>&nbsp;\u{feff}ר׳  ";
    let cleaned = norm.apply(raw);
    assert_eq!(cleaned, "ש\"ס ר'");
    assert_eq!(text.passage_text(&cleaned), "ש\"ס ר'");
    assert_eq!(query_input(text, norm, raw).unwrap(), "query: ש\"ס ר'");
    assert!(query_input(text, norm, "<br>&nbsp; ").is_err());
    assert_eq!(text.version(), 3);
    assert_eq!(norm.version(), 2);
}
