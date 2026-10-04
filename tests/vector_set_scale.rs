//! The vector store at library scale, measured: a synthetic set of N unit vectors with the
//! library's shape, installed, opened, scanned, extended by a delta and compacted.
//!
//! ```text
//! OTZARIA_SCALE_N=6400000 OTZARIA_SCALE_DIR=/somewhere/with/2GB \
//!     cargo test --release --test vector_set_scale -- --ignored --nocapture
//! ```
//!
//! `#[ignore]`d: at the library's size it writes about 1.8 GB twice and takes minutes. The
//! shape is the v30 library's, scaled to N: 7,765 books, 6% of records an extra occurrence
//! of a key another record already has, and a delta of about 6% new keys — new books and new
//! lines in old ones — with 2% of its records foreign and 1.5% of the old keys tombstoned.
//! Vectors are random, so the scores mean nothing; the times and sizes are the point.
//! Everything it writes is removed at the end.

use otzaria_semantic_search::cancellation::CancellationToken;
use otzaria_semantic_search::distribution::package::PackageKind;
use otzaria_semantic_search::semantic::chunk_key::{ChunkKey, KEY_VERSION};
use otzaria_semantic_search::semantic::oxv::codec::{Codec, CodecSpec};
use otzaria_semantic_search::semantic::oxv::scan::ScanRequest;
use otzaria_semantic_search::semantic::oxv::writer::{SegmentBuilder, SegmentSpec};
use otzaria_semantic_search::semantic::resolve::BookSet;
use otzaria_semantic_search::semantic::segment_set::{
    compact, install_package, CompactionPolicy, InstallExpectation, InstallSource, ReleaseManifest,
    SegmentSet, STORE_BACKEND_ID,
};
use otzaria_semantic_search::semantic::versioning::{
    EmbeddingWorker, IndexVersion, ModelIdentity, ModelPackage, StoreIdentity, TextIdentity,
    VectorProvenance,
};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const DIM: usize = 256;
const BOOKS: u64 = 7_765;

fn env(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The vector a key embeds as: random, unit, and the same wherever the key appears.
fn vector(key: u64, out: &mut [f32]) {
    let mut state = mix(key);
    let mut norm = 0f32;
    for value in out.iter_mut() {
        state = mix(state);
        *value = (state >> 40) as f32 / (1u64 << 23) as f32 - 1.0;
        norm += *value * *value;
    }
    let norm = norm.sqrt();
    for value in out.iter_mut() {
        *value /= norm;
    }
}

fn chunk_key(key: u64) -> ChunkKey {
    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&mix(key).to_be_bytes());
    bytes[8..].copy_from_slice(&key.to_be_bytes());
    ChunkKey(bytes)
}

/// `i8-sym-vec`, or what `OTZARIA_SCALE_CODEC` names (`i8-sym-dim` calibrates at 0.9999).
fn codec_spec() -> CodecSpec {
    CodecSpec::parse(
        &std::env::var("OTZARIA_SCALE_CODEC").unwrap_or_else(|_| "i8-sym-vec".to_string()),
        0.9999,
    )
    .unwrap()
}

fn identity() -> IndexVersion {
    IndexVersion {
        text: TextIdentity {
            line_text_version: 1,
            key_version: KEY_VERSION,
        },
        model: ModelIdentity {
            family_id: "synthetic@0".to_string(),
            tokenizer_checksum: "7".repeat(64),
            embedding_dim: DIM as u32,
            pooling: "in-graph".to_string(),
            max_tokens: 256,
            embedding_text_version: 2,
            normalization_version: 1,
            chunking_identity: 1,
            query_packages: vec![ModelPackage {
                checksum: "a".repeat(64),
                quantization: "int8".to_string(),
            }],
        },
        store: StoreIdentity {
            backend_id: STORE_BACKEND_ID.to_string(),
            store_format_version: 2,
            vector_precision: codec_spec().name().to_string(),
        },
    }
}

fn provenance() -> VectorProvenance {
    VectorProvenance {
        passage_package: ModelPackage {
            checksum: "a".repeat(64),
            quantization: "int8".to_string(),
        },
        worker: EmbeddingWorker {
            backend: "synthetic".to_string(),
            device: "cpu".to_string(),
        },
    }
}

/// One book's records: `(hint, key, is this the key's first appearance)`.
type Book = Vec<(u32, u64, bool)>;

/// The base library: record `i` of book `b` has key `b · 2³² + i`, except one record in 16
/// — at `i ≡ 7 (mod 16)` in a book after the first — which repeats the key of record `i − 1`
/// of the book before: an extra.
fn base_book(book: u64, per_book: u64) -> Book {
    (0..per_book)
        .map(|i| {
            if book > 0 && i % 16 == 7 {
                (i as u32, ((book - 1) << 32) | (i - 1), false)
            } else {
                (i as u32, (book << 32) | i, true)
            }
        })
        .collect()
}

/// Write one segment. `books` come in name order; a record whose key is not new in this
/// segment and not `shipped` becomes a foreign record.
fn write(
    path: &Path,
    kind: PackageKind,
    from: u32,
    to: u32,
    books: &[(String, Book)],
    tombstones: Vec<ChunkKey>,
    codec: &Codec,
) -> ReleaseManifest {
    let mut builder = SegmentBuilder::new(
        SegmentSpec {
            kind,
            identity_digest: identity().identity_digest(),
            from_library_version: from,
            to_library_version: to,
            library_release_tag: format!("v{to}-synthetic"),
        },
        codec.clone(),
    );
    let mut slot_of = std::collections::HashMap::new();
    let mut order = Vec::new();
    for (name, records) in books {
        let mut primary = Vec::new();
        let mut extras = Vec::new();
        let mut foreign = Vec::new();
        for &(hint, key, shipped) in records {
            if !shipped {
                match slot_of.get(&key) {
                    Some(slot) => extras.push((hint, *slot)),
                    None => foreign.push((chunk_key(key), hint)),
                }
                continue;
            }
            if let Some(slot) = slot_of.get(&key) {
                extras.push((hint, *slot));
                continue;
            }
            let slot = order.len() as u32;
            slot_of.insert(key, slot);
            order.push(key);
            primary.push((chunk_key(key), hint));
        }
        builder.add_book(name, &primary, &extras, &foreign).unwrap();
    }
    builder.set_tombstones(tombstones);
    let mut sink = builder.write(path).unwrap();
    let mut buffer = vec![0f32; DIM];
    for key in order {
        vector(key, &mut buffer);
        sink.push_f32(&buffer).unwrap();
    }
    let written = sink.finish().unwrap();
    ReleaseManifest::for_segment(
        &written,
        &identity(),
        codec.params_sha256(),
        provenance(),
        "2026-10-02T00:00:00Z".to_string(),
    )
}

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn install(dir: &Path, segment: &Path, manifest: &ReleaseManifest) -> Duration {
    let started = Instant::now();
    install_package(
        dir,
        &InstallSource {
            segment,
            manifest_json: &manifest.to_json(),
        },
        &InstallExpectation {
            identity: identity(),
            published_manifest_sha256: None,
        },
        &CancellationToken::new(),
    )
    .unwrap();
    started.elapsed()
}

#[test]
#[ignore = "writes gigabytes at library scale; set OTZARIA_SCALE_N and OTZARIA_SCALE_DIR"]
fn a_library_scale_set_installs_scans_applies_and_compacts() {
    let n = env("OTZARIA_SCALE_N", 200_000);
    let threads = env(
        "OTZARIA_SCALE_THREADS",
        std::thread::available_parallelism().map_or(1, |cores| cores.get()) as u64,
    ) as usize;
    let root: PathBuf = std::env::var_os("OTZARIA_SCALE_DIR").map_or_else(
        || std::env::temp_dir().join("otzaria_vector_set_scale"),
        PathBuf::from,
    );
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let per_book = n.div_ceil(BOOKS);
    println!("N = {n} records in {BOOKS} books ({per_book} each), dim {DIM}, {threads} threads");

    let mut sample = vec![vec![0f32; DIM]; 20_000.min(n as usize)];
    for (index, vector_out) in sample.iter_mut().enumerate() {
        vector(index as u64, vector_out);
    }
    let refs: Vec<&[f32]> = sample.iter().map(Vec::as_slice).collect();
    let codec = codec_spec().build(DIM, &refs).unwrap();
    println!("codec {}", codec.name());
    drop(sample);

    let name = |book: u64| format!("id:{book:05}");
    let base_books: Vec<(String, Book)> = (0..BOOKS)
        .map(|b| (name(b), base_book(b, per_book)))
        .collect();
    let started = Instant::now();
    let base_path = root.join("base.oxv");
    let base = write(
        &base_path,
        PackageKind::Base,
        0,
        29,
        &base_books,
        Vec::new(),
        &codec,
    );
    let write_time = started.elapsed();
    println!(
        "base: {} slots, {} extras, {:.3} GB, written in {write_time:.2?}",
        base.counts.slots,
        base.counts.extras,
        base.segment.size as f64 / 1e9
    );

    let dir = root.join("vectors");
    let install_time = install(&dir, &base_path, &base);
    std::fs::remove_file(&base_path).unwrap();
    println!("install (copy, SHA-256, every block CRC): {install_time:.2?}");

    let started = Instant::now();
    let set = SegmentSet::open(&dir).unwrap();
    println!("open: {:.2?}", started.elapsed());

    let cancel = CancellationToken::new();
    let measure = |set: &SegmentSet, threads: usize, books: Option<&BookSet>| {
        let mut query = vec![0f32; DIM];
        vector(u64::MAX / 3, &mut query);
        for _ in 0..2 {
            set.scan(
                &query,
                &ScanRequest {
                    top_k: 100,
                    books,
                    threads,
                },
                &cancel,
            )
            .unwrap();
        }
        let samples: Vec<Duration> = (0..9)
            .map(|round| {
                vector(round * 7919 + 1, &mut query);
                let started = Instant::now();
                let hits = set
                    .scan(
                        &query,
                        &ScanRequest {
                            top_k: 100,
                            books,
                            threads,
                        },
                        &cancel,
                    )
                    .unwrap();
                assert!(!hits.is_empty());
                started.elapsed()
            })
            .collect();
        median(samples)
    };
    let filter: BookSet = (0..BOOKS).step_by(20).map(name).collect();
    println!("warm scan, 1 thread: {:.2?}", measure(&set, 1, None));
    println!(
        "warm scan, {threads} threads: {:.2?}",
        measure(&set, threads, None)
    );
    println!(
        "warm scan, {threads} threads, 5% of the books: {:.2?}",
        measure(&set, threads, Some(&filter))
    );
    drop(set);

    // A delta of ~6% new keys: 3% in new books, 3% in new lines of old books; 2% of its
    // records foreign — old keys appearing in new places — and 1.5% of the old keys gone.
    let new_books = (BOOKS * 3).div_ceil(100);
    let mut delta_books: Vec<(String, Book)> = Vec::new();
    let grown = (BOOKS * 3 / 100).max(1);
    for b in 0..BOOKS + new_books {
        let mut records: Book = Vec::new();
        if b >= BOOKS {
            records.extend((0..per_book).map(|i| (i as u32, (b << 32) | i, true)));
            // A few old keys quoted in the new book — foreign records — from a book no
            // tombstone touches.
            let quoted = b - BOOKS + 1;
            if !quoted.is_multiple_of(67) {
                records.extend(
                    (0..per_book / 25)
                        .filter(|i| i % 16 != 7)
                        .map(|i| ((per_book + i) as u32, (quoted << 32) | i, false)),
                );
            }
        } else if b % (BOOKS / grown).max(1) == 0 {
            records.extend(
                (0..per_book).map(|i| ((per_book + i) as u32, (b << 32) | (1 << 31) | i, true)),
            );
        }
        if !records.is_empty() {
            delta_books.push((name(b), records));
        }
    }
    let tombstones: Vec<ChunkKey> = (0..BOOKS)
        .filter(|b| b.is_multiple_of(67))
        .flat_map(|b| {
            (0..per_book)
                .filter(|i| i % 16 != 7)
                .map(move |i| chunk_key((b << 32) | i))
        })
        .collect();
    let delta_path = root.join("delta.oxv");
    let delta = write(
        &delta_path,
        PackageKind::Delta,
        29,
        30,
        &delta_books,
        tombstones,
        &codec,
    );
    println!(
        "delta: {} slots, {} foreign, {} tombstones, {:.1} MB ({:.1}% of the base)",
        delta.counts.slots,
        delta.counts.foreign,
        delta.counts.tombstones,
        delta.segment.size as f64 / 1e6,
        delta.segment.size as f64 * 100.0 / base.segment.size as f64
    );
    let apply_time = install(&dir, &delta_path, &delta);
    std::fs::remove_file(&delta_path).unwrap();
    println!("apply (stage, verify, resolve keys, new generation, flip): {apply_time:.2?}");
    let set = SegmentSet::open(&dir).unwrap();
    println!(
        "after the delta: {} segments, {} live slots, {} dead; warm scan, {threads} threads: {:.2?}",
        set.info().segments.len(),
        set.info().slots_live,
        set.info().slots_dead,
        measure(&set, threads, None)
    );
    drop(set);

    let started = Instant::now();
    let report = compact(
        &dir,
        &CompactionPolicy {
            force: true,
            ..CompactionPolicy::default()
        },
        None,
        &CancellationToken::new(),
    )
    .unwrap();
    println!(
        "compaction: {:.2?}, {} → {} slots, {:.3} → {:.3} GB",
        started.elapsed(),
        report.slots_before,
        report.slots_after,
        report.bytes_before as f64 / 1e9,
        report.bytes_after as f64 / 1e9
    );
    let set = SegmentSet::open(&dir).unwrap();
    println!(
        "after compaction: warm scan, {threads} threads: {:.2?}",
        measure(&set, threads, None)
    );
    drop(set);
    std::fs::remove_dir_all(&root).unwrap();
}
