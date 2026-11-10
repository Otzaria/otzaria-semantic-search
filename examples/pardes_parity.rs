//! Bounded actual Pardes recipe, Rust tokenizer and ONNX parity probe.
use otzaria_semantic_search::semantic::{
    backend::{EmbeddingBackend, Pooling},
    onnx_backend::{OnnxBackend, OnnxBackendConfig},
    recipe::{query_input, EmbeddingTextRecipe, TextNormalizationRecipe},
};
use std::path::PathBuf;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 && args.len() != 4 {
        return Err("usage: pardes_parity PACKAGE_DIR RUNTIME_LIBRARY [FIXTURE_JSON]".into());
    }
    let root = PathBuf::from(&args[1]);
    let runtime = PathBuf::from(&args[2]);
    if let Some(path) = args.get(3) {
        let fixtures: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
        let rows = fixtures["rows"].as_array().ok_or("missing fixture rows")?;
        let mut out = Vec::new();
        for (role, cap) in [("query", 128), ("passage", 512)] {
            let backend = OnnxBackend::open_with_kernel_policy(
                &root.join("model.onnx"),
                &root.join("tokenizer.json"),
                cap,
                Pooling::InGraph,
                &OnnxBackendConfig {
                    intra_threads: 4,
                    sessions: 1,
                },
                Some(&runtime),
                true,
            )?;
            for row in rows.iter().filter(|row| row["role"].as_str() == Some(role)) {
                let input = row["input"].as_str().ok_or("missing fixture input")?;
                let ids = backend.tokenize(input)?;
                assert_eq!(serde_json::to_value(&ids)?, row["ids"]);
                let vectors = backend.embed_batch_raw(&[input])?;
                out.push(serde_json::json!({"role":role,"cap":cap,"input":input,"ids":ids,"vector":vectors[0]}));
            }
        }
        println!("{}", serde_json::to_string(&out)?);
        return Ok(());
    }
    let mut texts = vec![
        "שלום עולם".to_string(),
        "  <b>ש״ס</b>&nbsp;ר׳  ".to_string(),
        "בְּרֵאשִׁית בָּרָא אֱלֹהִים".to_string(),
        "&amp;lt; &#x05D0; &#0;".to_string(),
        "query: &#x41; <abc".to_string(),
    ];
    texts.push("אמר רב אמר רבא ".repeat(600));
    let norm = TextNormalizationRecipe::from_version(2)?;
    let recipe = EmbeddingTextRecipe::from_version(3)?;
    let mut out = Vec::new();
    for (role, cap) in [("query", 128), ("passage", 512)] {
        let backend = OnnxBackend::open_with_kernel_policy(
            &root.join("model.onnx"),
            &root.join("tokenizer.json"),
            cap,
            Pooling::InGraph,
            &OnnxBackendConfig {
                intra_threads: 4,
                sessions: 1,
            },
            Some(&runtime),
            true,
        )?;
        for raw in &texts {
            let input = if role == "query" {
                query_input(recipe, norm, raw)?
            } else {
                recipe.passage_text(&norm.apply(raw)).into_owned()
            };
            let ids = backend.tokenize(&input)?;
            let vectors = backend.embed_batch_raw(&[&input])?;
            out.push(serde_json::json!({"role":role,"cap":cap,"raw":raw,"input":input,"ids":ids,"vector":vectors[0]}));
        }
    }
    println!("{}", serde_json::to_string(&out)?);
    Ok(())
}
