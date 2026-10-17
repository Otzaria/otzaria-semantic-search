//! Recall of `rot192-i8i4` on a real segment, against the exact scan of its vectors.
//!
//! Ignored: it needs a released `i8-sym-vec` base, decompressed, named by
//! `OTZARIA_OXV_MEASURE`; `OTZARIA_OXV_QUERIES` may name an `.npy` of real query vectors.
//! `cargo test --release --lib oxv::measure -- --ignored --nocapture`.

use super::codec::{Codec, ROT_HEAD, ROT_TAIL};
use super::reader::Segment;
use super::scan::{scan_segments, ScanRequest};
use super::testing::{spec, TempDir};
use super::writer::SegmentBuilder;
use crate::cancellation::CancellationToken;
use crate::distribution::package::PackageKind;
use crate::semantic::chunk_key::ChunkKey;
use std::collections::{HashMap, HashSet};

const DB: usize = 400_000;
const FIT: usize = 200_000;
const QUERIES: usize = 4000;
const K: usize = 50;

/// Every `step`-th slot from `offset`, decoded.
fn decode_slots(source: &Segment, count: usize, offset: usize) -> (Vec<f32>, Vec<ChunkKey>) {
    let slots = source.slot_count() as usize;
    let codec = source.codec();
    let dim = codec.dim();
    let mut vectors = vec![0f32; count * dim];
    let mut keys = Vec::with_capacity(count);
    for (index, out) in vectors.chunks_exact_mut(dim).enumerate() {
        let slot = ((index * slots / count + offset) % slots) as u32;
        codec.decode(source.vector(slot), source.vector_scale(slot), out);
        keys.push(source.key(slot));
    }
    (vectors, keys)
}

/// The exact top `K` of each query over `db`, as indexes, `skip` (the query itself) left out.
fn exact_top(db: &[f32], dim: usize, queries: &[(&[f32], Option<usize>)]) -> Vec<Vec<usize>> {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let chunk = queries.len().div_ceil(threads);
    std::thread::scope(|scope| {
        let handles: Vec<_> = queries
            .chunks(chunk)
            .map(|queries| {
                scope.spawn(move || {
                    let mut scored: Vec<(f32, usize)> = Vec::with_capacity(db.len() / dim);
                    queries
                        .iter()
                        .map(|(query, skip)| {
                            scored.clear();
                            scored.extend(
                                db.chunks_exact(dim)
                                    .enumerate()
                                    .filter(|(index, _)| Some(*index) != *skip)
                                    .map(|(index, v)| {
                                        (v.iter().zip(*query).map(|(a, b)| a * b).sum(), index)
                                    }),
                            );
                            scored.select_nth_unstable_by(K, |a, b| b.0.total_cmp(&a.0));
                            let mut top = scored[..K].to_vec();
                            top.sort_by(|a, b| b.0.total_cmp(&a.0));
                            top.into_iter().map(|(_, index)| index).collect()
                        })
                        .collect::<Vec<Vec<usize>>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|handle| handle.join().unwrap())
            .collect()
    })
}

fn read_npy(path: &str, dim: usize) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(&bytes[..6], b"\x93NUMPY");
    let header = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let data = &bytes[10 + header..];
    let values: Vec<f32> = data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    assert_eq!(values.len() % dim, 0);
    values
}

#[test]
#[ignore = "needs a released segment: OTZARIA_OXV_MEASURE"]
fn rotated_recall_on_a_released_segment() {
    let Ok(path) = std::env::var("OTZARIA_OXV_MEASURE") else {
        return;
    };
    let source = Segment::open(std::path::Path::new(&path)).unwrap();
    let dim = source.codec().dim();
    let (db, keys) = decode_slots(&source, DB, 0);
    // The fit sample lies between the database's slots: none of its vectors is fitted on.
    let (fit, _) = decode_slots(&source, FIT, source.slot_count() as usize / DB / 2);
    let fit_refs: Vec<&[f32]> = fit.chunks_exact(dim).collect();
    let started = std::time::Instant::now();
    let rotated = Codec::fit_rot_i8i4(&fit_refs, ROT_HEAD, ROT_TAIL).unwrap();
    println!("fit on {FIT}: {:.1}s", started.elapsed().as_secs_f64());
    drop(fit_refs);
    drop(fit);

    let by_key: HashMap<ChunkKey, usize> = keys
        .iter()
        .enumerate()
        .map(|(index, key)| (key.narrowed(), index))
        .collect();
    let picks: Vec<usize> = (0..QUERIES).map(|i| i * DB / QUERIES + 7).collect();
    let mut queries: Vec<(&[f32], Option<usize>)> = picks
        .iter()
        .map(|&index| (&db[index * dim..(index + 1) * dim], Some(index)))
        .collect();
    let real = std::env::var("OTZARIA_OXV_QUERIES")
        .ok()
        .map(|path| read_npy(&path, dim));
    let corpus_count = queries.len();
    if let Some(real) = &real {
        queries.extend(real.chunks_exact(dim).map(|query| (query, None)));
    }
    let started = std::time::Instant::now();
    let exact = exact_top(&db, dim, &queries);
    println!("exact reference: {:.1}s", started.elapsed().as_secs_f64());

    let dir = TempDir::new("measure");
    for codec in [Codec::i8_sym_vec(dim).unwrap(), rotated.clone()] {
        let file = dir.join(&format!("{}.oxv", codec.name()));
        let mut builder = SegmentBuilder::new(spec(PackageKind::Base, 0, 31), codec.clone());
        let primary: Vec<(ChunkKey, u32)> = keys
            .iter()
            .enumerate()
            .map(|(index, key)| (*key, index as u32))
            .collect();
        builder.add_book("id:1", &primary, &[], &[]).unwrap();
        let mut sink = builder.write(&file).unwrap();
        for vector in db.chunks_exact(dim) {
            sink.push_f32(vector).unwrap();
        }
        let written = sink.finish().unwrap();
        let segment = Segment::open(&file).unwrap();

        // G5: the decoded vectors against the reference.
        let mut decoded = vec![0f32; dim];
        let mut cosines: Vec<f64> = (0..DB as u32)
            .map(|slot| {
                codec.decode(
                    segment.vector(slot),
                    segment.vector_scale(slot),
                    &mut decoded,
                );
                let original = &db[slot as usize * dim..][..dim];
                let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
                for (a, b) in original.iter().zip(&decoded) {
                    ab += f64::from(*a) * f64::from(*b);
                    aa += f64::from(*a) * f64::from(*a);
                    bb += f64::from(*b) * f64::from(*b);
                }
                ab / (aa.sqrt() * bb.sqrt())
            })
            .collect();
        cosines.sort_by(f64::total_cmp);
        let mean = cosines.iter().sum::<f64>() / cosines.len() as f64;
        let p001 = cosines[(cosines.len() - 1) / 1000];

        let started = std::time::Instant::now();
        let (mut r10, mut r50) = ([0f64; 2], [0f64; 2]);
        let (mut drift, mut flips, mut pairs) = (0f64, 0u64, 0u64);
        for (query_index, ((query, skip), exact)) in queries.iter().zip(&exact).enumerate() {
            let hits = scan_segments(
                &[&segment],
                query,
                &ScanRequest {
                    top_k: K + 1,
                    books: None,
                    threads: 8,
                    window_penalty: 0.0,
                },
                &CancellationToken::new(),
            )
            .unwrap();
            let got: Vec<usize> = hits
                .iter()
                .map(|hit| by_key[&hit.key.narrowed()])
                .filter(|index| Some(*index) != *skip)
                .take(K)
                .collect();
            for hit in &hits {
                let index = by_key[&hit.key.narrowed()];
                let truth: f32 = db[index * dim..][..dim]
                    .iter()
                    .zip(*query)
                    .map(|(a, b)| a * b)
                    .sum();
                drift += f64::from((hit.score - truth).abs());
                flips += u64::from((hit.score >= 0.55) != (truth >= 0.55));
                pairs += 1;
            }
            let kind = usize::from(query_index >= corpus_count);
            for (k, total) in [(10, &mut r10), (50, &mut r50)] {
                let want: HashSet<usize> = exact[..k].iter().copied().collect();
                total[kind] +=
                    got[..k].iter().filter(|index| want.contains(index)).count() as f64 / k as f64;
            }
        }
        let elapsed = started.elapsed().as_secs_f64();
        let real_count = queries.len() - corpus_count;
        println!(
            "{:12} format {} {:>5} B/vector, {} bytes | G5 cosine mean {mean:.6} p0.1 {p001:.6} \
             min {:.6} | clipped {} of {} | corpus r@10 {:.4} r@50 {:.4} | real({real_count}) \
             r@10 {:.4} r@50 {:.4} | score drift {:.5}, threshold flips {flips}/{pairs} | \
             {:.1} ms/query",
            codec.name(),
            segment.format_version(),
            codec.bytes_per_vector() + 4,
            written.size,
            cosines[0],
            written.clipped_components,
            DB * codec.clippable_components(),
            r10[0] / corpus_count as f64,
            r50[0] / corpus_count as f64,
            r10[1] / real_count.max(1) as f64,
            r50[1] / real_count.max(1) as f64,
            drift / pairs as f64,
            elapsed * 1000.0 / queries.len() as f64,
        );
    }
}
