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

/// A unit vector that is a function of `seed` alone: the same text, the same vector.
pub(crate) fn unit_vector(seed: &[u8; 32], dim: usize) -> Vec<f32> {
    let mut state = u64::from_le_bytes(seed[..8].try_into().unwrap()) | 1;
    let mut vector: Vec<f32> = (0..dim)
        .map(|_| {
            // Four uniforms, summed: near enough to normal for a spread of magnitudes.
            (0..4)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
                })
                .sum()
        })
        .collect();
    let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    vector.iter_mut().for_each(|x| *x /= norm);
    vector
}

/// The shard an external worker would write for the whole of `plan_dir`'s embed plan, in
/// `out`: each text's [`unit_vector`], under the deterministic stand-in's mode.
pub(crate) fn raw_shard(plan_dir: &Path, out: &Path) -> PathBuf {
    use crate::distribution::files::hex;
    use crate::distribution::plan::{read_embed_plan, EmbedManifest};
    use crate::distribution::shard::*;
    let manifest = EmbedManifest::read(plan_dir).unwrap();
    let dim = manifest.model.embedding_dim as usize;
    let (mut vectors, mut keys) = (Vec::new(), Vec::new());
    for record in read_embed_plan(plan_dir, 0, u64::MAX).unwrap() {
        let (_, record) = record.unwrap();
        let sha = crate::distribution::files::sha256(record.embedding_text.as_bytes());
        keys.extend_from_slice(&sha);
        for value in unit_vector(&sha, dim) {
            vectors.extend_from_slice(&value.to_le_bytes());
        }
    }
    std::fs::create_dir_all(out).unwrap();
    std::fs::write(out.join(VECTORS_FILE), &vectors).unwrap();
    std::fs::write(out.join(KEYS_FILE), &keys).unwrap();
    let records = (keys.len() / 32) as u64;
    let shard = ShardManifest {
        format: SHARD_FORMAT.to_string(),
        version: SHARD_FORMAT_VERSION,
        plan_sha256: manifest.plan_sha256,
        skip: 0,
        take: records,
        records,
        dim: dim as u32,
        vectors_sha256: hex(&crate::distribution::files::sha256(&vectors)),
        keys_sha256: hex(&crate::distribution::files::sha256(&keys)),
        model: manifest.model,
        passage_package: manifest.passage_package,
        worker: WorkerInfo {
            name: "test-worker".into(),
            version: "1".into(),
            device: "cpu".into(),
            ep: "cpu".into(),
            mode: MODE_MOCK.into(),
        },
        parity: None,
    };
    std::fs::write(
        out.join(SHARD_MANIFEST_FILE),
        serde_json::to_vec_pretty(&shard).unwrap(),
    )
    .unwrap();
    out.to_path_buf()
}

/// Heap bytes this thread holds, and the most it has held since [`heap::start`]: a check
/// that a pipeline's memory is bounded, which other tests' threads do not disturb.
pub(crate) mod heap {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    thread_local! {
        static HELD: Cell<isize> = const { Cell::new(0) };
        static PEAK: Cell<isize> = const { Cell::new(0) };
    }

    struct Counting;

    fn count(by: isize) {
        let _ = HELD.try_with(|held| {
            let now = held.get() + by;
            held.set(now);
            let _ = PEAK.try_with(|peak| peak.set(peak.get().max(now)));
        });
    }

    // SAFETY: every call is `System`'s, and the counting allocates nothing.
    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let ptr = unsafe { System.alloc(layout) };
            if !ptr.is_null() {
                count(layout.size() as isize);
            }
            ptr
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            let ptr = unsafe { System.alloc_zeroed(layout) };
            if !ptr.is_null() {
                count(layout.size() as isize);
            }
            ptr
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) };
            count(-(layout.size() as isize));
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            let new = unsafe { System.realloc(ptr, layout, new_size) };
            if !new.is_null() {
                count(new_size as isize - layout.size() as isize);
            }
            new
        }
    }

    #[global_allocator]
    static COUNTING: Counting = Counting;

    /// Measure from here: the peak becomes what the thread holds now, which is returned.
    pub(crate) fn start() -> isize {
        let now = HELD.with(Cell::get);
        PEAK.with(|peak| peak.set(now));
        now
    }

    /// The most this thread has held since [`start`].
    pub(crate) fn peak() -> isize {
        PEAK.with(Cell::get)
    }
}
