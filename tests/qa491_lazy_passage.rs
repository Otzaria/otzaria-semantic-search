#![cfg(feature = "onnx-backend")]
use otzaria_semantic_search::{
    cancellation::CancellationToken,
    distribution::package::PackageKind,
    semantic::{
        chunk_key::ChunkKey,
        embedding::EmbeddingDeployment,
        official_index::{
            readable_store_identity, LocalModel, OfficialIndexConfig, OfficialSemanticIndex,
        },
        oxv::{
            codec::Codec,
            writer::{SegmentBuilder, SegmentSpec},
        },
        segment_set::{install_package, InstallExpectation, InstallSource, ReleaseManifest},
        versioning::{
            EmbeddingWorker, IndexVersion, ModelIdentity, TextIdentity, VectorProvenance,
        },
    },
};
use std::path::{Path, PathBuf};

fn fixture_set(dir: &Path, model: &ModelIdentity) {
    let identity = IndexVersion {
        text: TextIdentity::with_line_text_version(1),
        model: model.clone(),
        store: readable_store_identity(),
    };
    let stage = dir.with_extension("stage");
    std::fs::create_dir(&stage).unwrap();
    let mut b = SegmentBuilder::new(
        SegmentSpec {
            kind: PackageKind::Base,
            identity_digest: identity.identity_digest(),
            from_library_version: 0,
            to_library_version: 1,
            library_release_tag: "v1-qa491-test".into(),
        },
        Codec::i8_sym_vec(256).unwrap(),
    );
    b.add_book(
        "fixture",
        &[(
            ChunkKey::of("synthetic unit vector; not a semantic evaluation"),
            0,
        )],
        &[],
        &[],
    )
    .unwrap();
    let params = b.codec().params_sha256();
    let mut sink = b.write(&stage.join("segment.oxv")).unwrap();
    let mut v = vec![0.; 256];
    v[0] = 1.;
    sink.push_f32(&v).unwrap();
    let written = sink.finish().unwrap();
    let manifest = ReleaseManifest::for_segment(
        &written,
        &identity,
        params,
        VectorProvenance {
            passage_package: model
                .query_packages
                .iter()
                .find(|p| p.quantization == "fp32")
                .unwrap()
                .clone(),
            worker: EmbeddingWorker {
                backend: "synthetic-test-fixture-not-semantic".into(),
                device: "test".into(),
            },
        },
        "2026-10-31T00:00:00Z".into(),
    );
    install_package(
        dir,
        &InstallSource {
            segment: &written.path,
            manifest_json: &manifest.to_json(),
        },
        &InstallExpectation {
            identity,
            published_manifest_sha256: None,
        },
        &CancellationToken::new(),
    )
    .unwrap();
}

#[test]
#[ignore = "requires actual QA491 release package and production ONNX Runtime"]
fn qa491_lazy_fp32_passages_use_the_production_official_index() {
    let source = PathBuf::from(std::env::var("OTZARIA_QA491_PACKAGE").unwrap());
    let temp = std::env::temp_dir().join(format!("qa491-official-lazy-{}", std::process::id()));
    std::fs::create_dir(&temp).unwrap();
    for name in ["query.int8.onnx", "tokenizer.json"] {
        std::fs::copy(source.join(name), temp.join(name)).unwrap();
    }
    let model: ModelIdentity =
        serde_json::from_slice(&std::fs::read(source.join("model.json")).unwrap()).unwrap();
    let vectors = temp.join("vectors");
    fixture_set(&vectors, &model);
    let index = OfficialSemanticIndex::open(OfficialIndexConfig {
        vectors_dir: vectors,
        text: TextIdentity::with_line_text_version(1),
        model: LocalModel::of_family(temp.join("query.int8.onnx"), &model, "int8"),
        deployment: EmbeddingDeployment {
            onnx_runtime: Some(PathBuf::from(
                std::env::var("OTZARIA_ONNX_RUNTIME").unwrap(),
            )),
        },
        scan_threads: None,
    })
    .unwrap();
    let cancel = CancellationToken::new();
    let query = "תפילה קודם הנץ החמה";
    let text = "תפילה קודם הנץ החמה";
    let before = index.search(query, 1, None, &cancel).unwrap()[0].score;
    assert!(
        !temp.join("document.fp32.onnx").exists(),
        "query cold open must not need FP32"
    );
    assert!(
        index.embed_passages(&[text], &cancel).is_err(),
        "missing FP32 must fail; no INT8 fallback"
    );
    assert_eq!(
        index.search(query, 1, None, &cancel).unwrap()[0].score,
        before
    );
    std::fs::copy(
        source.join("query.int8.onnx"),
        temp.join("document.fp32.onnx"),
    )
    .unwrap();
    assert!(
        index.embed_passages(&[text], &cancel).is_err(),
        "a valid graph from the wrong package must be refused"
    );
    std::fs::copy(
        source.join("document.fp32.onnx"),
        temp.join("document.fp32.onnx"),
    )
    .unwrap();
    use std::io::Write;
    std::fs::OpenOptions::new()
        .append(true)
        .open(temp.join("document.fp32.onnx"))
        .unwrap()
        .write_all(&[0])
        .unwrap();
    assert!(
        index.embed_passages(&[text], &cancel).is_err(),
        "tampered passage bytes must fail without caching"
    );
    std::fs::copy(
        source.join("document.fp32.onnx"),
        temp.join("document.fp32.onnx"),
    )
    .unwrap();
    std::fs::OpenOptions::new()
        .append(true)
        .open(temp.join("tokenizer.json"))
        .unwrap()
        .write_all(b"\n")
        .unwrap();
    assert!(
        index.embed_passages(&[text], &cancel).is_err(),
        "noncanonical tokenizer must fail"
    );
    std::fs::copy(source.join("tokenizer.json"), temp.join("tokenizer.json")).unwrap();
    let encoded = index.embed_passages(&[text], &cancel).unwrap();
    assert_eq!(encoded[0].len(), 256);
    let golden: serde_json::Value = serde_json::from_slice(
        &std::fs::read(std::env::var("OTZARIA_QA491_GOLDENS").unwrap()).unwrap(),
    )
    .unwrap();
    let expected = golden["variants"][1]["cases"][0]["vector"]
        .as_array()
        .unwrap();
    let error = encoded[0]
        .iter()
        .zip(expected)
        .map(|(a, b)| (f64::from(*a) - b.as_f64().unwrap()).abs())
        .fold(0.0, f64::max);
    assert!(error <= 1e-5, "FP32 passage parity {error}");
    std::fs::remove_file(temp.join("document.fp32.onnx")).unwrap();
    assert_eq!(
        index.embed_passages(&[text], &cancel).unwrap(),
        encoded,
        "verified FP32 backend must be cached"
    );
    assert_eq!(
        index.search(query, 1, None, &cancel).unwrap()[0].score,
        before
    );
    drop(index);
    std::fs::remove_dir_all(temp).unwrap();
    println!("QA491 OfficialSemanticIndex lazy FP32 PASS: cold-query works without FP32; missing/tamper/tokenizer mismatch fail; retry succeeds; FP32 native256 parity max_abs={error}; cached passages/query stable");
}
