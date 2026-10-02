//! Fixtures for the build pipeline's tests: corpora, families and a stub model.

use crate::distribution::corpus::{CorpusIdentity, CorpusLine, CorpusLineRecord, JsonlCorpus};
use crate::distribution::plan::HeldVectors;
use crate::semantic::chunker::ChunkerConfig;
use crate::semantic::embedding::mock;
use crate::semantic::model_package::validate_model;
use crate::semantic::versioning::{ModelIdentity, ModelPackage, TextIdentity};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

pub(crate) use crate::semantic::oxv::testing::TempDir;

/// The dimension of the stub model's vectors.
pub(crate) const DIM: u32 = 64;

/// A corpus of `(line_id, book, text)` lines, every line in section 1, at `version`.
pub(crate) fn corpus(
    dir: &Path,
    name: &str,
    version: u32,
    lines: &[(u64, &str, &str)],
) -> JsonlCorpus {
    let identity = dir.join(format!("{name}-identity.json"));
    let records = dir.join(format!("{name}-lines.jsonl"));
    std::fs::write(
        &identity,
        serde_json::to_vec(&CorpusIdentity {
            text: TextIdentity::with_line_text_version(1),
            library_version: version,
            library_release_tag: format!("v{version}-20260930120000"),
            document_id_scheme_version: 1,
        })
        .unwrap(),
    )
    .unwrap();
    let body: String = lines
        .iter()
        .map(|(line_id, book, text)| {
            serde_json::to_string(&CorpusLineRecord {
                line_id: *line_id,
                line: CorpusLine {
                    source_book_key: book.to_string(),
                    title: String::new(),
                    reference: String::new(),
                    section_id: 1,
                    segment: 0,
                    is_pdf: false,
                    line_hash: 0,
                    content_hash: 1,
                    facets: Vec::new(),
                    text: text.to_string(),
                },
            })
            .unwrap()
                + "\n"
        })
        .collect();
    std::fs::write(&records, body).unwrap();
    JsonlCorpus::load(&identity, &records).unwrap()
}

/// The stub ONNX package, written once into `dir/model`.
pub(crate) fn stub_model(dir: &Path) -> PathBuf {
    mock::write_stub_onnx_package(&dir.join("model"))
}

/// A family whose fp32 package is the stub at `model_path` and whose int8 package is
/// another — so the passage package and a query package can differ.
pub(crate) fn family(model_path: &Path) -> ModelIdentity {
    ModelIdentity {
        family_id: "otzaria-test-family".to_string(),
        tokenizer_checksum: mock::stub_tokenizer_checksum(),
        embedding_dim: DIM,
        pooling: "in-graph".to_string(),
        max_tokens: 512,
        embedding_text_version: 1,
        normalization_version: 1,
        chunking_identity: ChunkerConfig::default().identity(),
        query_packages: vec![
            ModelPackage {
                checksum: "a".repeat(64),
                quantization: "int8".to_string(),
            },
            passage_package(model_path),
        ],
    }
}

/// The stub's package, as the family lists it.
pub(crate) fn passage_package(model_path: &Path) -> ModelPackage {
    ModelPackage {
        checksum: validate_model(model_path).unwrap().checksum().to_string(),
        quantization: "fp32".to_string(),
    }
}

/// Vectors "held" for a fixed set of texts.
pub(crate) struct Holding(pub HashSet<[u8; 32]>);

impl Holding {
    pub(crate) fn of(texts: &[&str]) -> Self {
        Self(
            texts
                .iter()
                .map(|text| crate::distribution::files::sha256(text.as_bytes()))
                .collect(),
        )
    }
}

impl HeldVectors for Holding {
    fn holds(&self, sha256: &[u8; 32]) -> bool {
        self.0.contains(sha256)
    }
}
