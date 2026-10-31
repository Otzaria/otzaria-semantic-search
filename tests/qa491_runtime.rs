#![cfg(feature = "onnx-backend")]

use otzaria_semantic_search::semantic::backend::{EmbeddingBackend, Pooling};
use otzaria_semantic_search::semantic::onnx_backend::{OnnxBackend, OnnxBackendConfig};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

#[test]
#[ignore = "requires the released QA491 graphs, reference vectors and production ONNX Runtime"]
fn qa491_production_backend_matches_role_capped_reference() {
    let root = PathBuf::from(std::env::var("OTZARIA_QA491_PACKAGE").expect("package path"));
    let golden: serde_json::Value = serde_json::from_slice(
        &std::fs::read(std::env::var("OTZARIA_QA491_GOLDENS").expect("goldens path")).unwrap(),
    )
    .unwrap();
    let library = PathBuf::from(std::env::var("OTZARIA_ONNX_RUNTIME").expect("runtime path"));
    let mut lowest = 1.0_f64;
    let mut largest = 0.0_f64;
    let mut count = 0;
    for variant in golden["variants"].as_array().unwrap() {
        let cap = variant["max_tokens"].as_u64().unwrap() as usize;
        let graph = root.join(variant["graph"].as_str().unwrap());
        let graph_digest = format!("{:x}", Sha256::digest(std::fs::read(&graph).unwrap()));
        assert_eq!(graph_digest, variant["graph_sha256"].as_str().unwrap());
        let tokenizer_digest = format!(
            "{:x}",
            Sha256::digest(std::fs::read(root.join("tokenizer.json")).unwrap())
        );
        assert_eq!(
            tokenizer_digest,
            golden["tokenizer_sha256"].as_str().unwrap()
        );
        let backend = OnnxBackend::open(
            &graph,
            &root.join("tokenizer.json"),
            cap,
            Pooling::InGraph,
            &OnnxBackendConfig::default(),
            Some(&library),
        )
        .expect("actual QA491 graph must load");
        assert_eq!(backend.max_tokens(), cap);
        let mut capped = false;
        for case in variant["cases"].as_array().unwrap() {
            let input = case["input"].as_str().unwrap();
            let expected_ids: Vec<u32> = case["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as u32)
                .collect();
            let ids = backend.tokenize(input).unwrap();
            assert_eq!(ids, expected_ids);
            assert!(ids.len() <= cap);
            capped |= ids.len() == cap;
            let v = backend.embed_batch_raw(&[input]).unwrap().remove(0);
            assert_eq!(v.len(), 256);
            let expected: Vec<f64> = case["vector"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect();
            let dot: f64 = v
                .iter()
                .zip(&expected)
                .map(|(a, b)| f64::from(*a) * b)
                .sum();
            let norm: f64 = v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
            let other: f64 = expected.iter().map(|x| x * x).sum::<f64>().sqrt();
            let cosine = dot / (norm * other);
            let error = v
                .iter()
                .zip(expected)
                .map(|(a, b)| (f64::from(*a) - b).abs())
                .fold(0.0, f64::max);
            assert!(cosine >= 0.99999, "cosine {cosine}");
            assert!(error <= 1e-5, "max abs {error}");
            assert!((norm - 1.0).abs() <= 1e-5, "norm {norm}");
            lowest = lowest.min(cosine);
            largest = largest.max(error);
            count += 1;
        }
        assert!(capped, "long text must exercise role cap {cap}");
    }
    println!("QA491 actual production backend: cases={count}, min_cosine={lowest:.12}, max_abs={largest:.12}");
}
