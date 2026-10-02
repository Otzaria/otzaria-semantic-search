//! The scan against brute force: the same hits, in the same order, at any thread count,
//! under any filter, with deleted slots and every kind of record.

use super::codec::Codec;
use super::kernel::{dot_f32, select, PreparedQuery};
use super::reader::Segment;
use super::scan::{
    scan, scan_reporting, Link, ReverseLink, ScanRequest, ScanSegment, ScanSet, LINK_UNRESOLVED,
};
use super::testing::{key, random_books, spec, write_segment, Random, TempDir, TestBook};
use crate::cancellation::{probe, CancellationToken, SCAN_CHECK_INTERVAL};
use crate::distribution::package::PackageKind;
use crate::errors::VectorStoreError;
use crate::semantic::chunk_key::ChunkKey;
use crate::semantic::resolve::{BookSet, MAX_RECORDS_PER_HIT};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

const DIM: usize = 64;

/// A base and a delta over it: the base's books, the delta's new books, extras across
/// books, foreign records linked into the base, a few deleted slots.
struct Fixture {
    _dir: TempDir,
    segments: Vec<Segment>,
    deleted: Vec<Vec<u64>>,
    links: Vec<Vec<Link>>,
    reverse: Vec<ReverseLink>,
}

impl Fixture {
    fn new(books: usize, per_book: usize, seed: u64, codec: impl Fn(&[&[f32]]) -> Codec) -> Self {
        let dir = TempDir::new("scan");
        let mut random = Random(seed);
        let mut base_books = random_books(&mut random, books, per_book, DIM, 0);
        // Book 1 also holds book 0's first two keys, and book 0 holds its own third key a
        // second time further down.
        base_books[1].extras = vec![(1000, 0), (1001, 1)];
        base_books[0].extras = vec![(500, 2)];
        let vectors: Vec<&[f32]> = base_books
            .iter()
            .flat_map(|book| book.primary.iter().map(|(_, _, v)| v.as_slice()))
            .collect();
        let codec = codec(&vectors);
        let base = dir.join("base.oxv");
        write_segment(
            &base,
            spec(PackageKind::Base, 0, 29),
            codec.clone(),
            &base_books,
            &[],
        );

        // The delta: one new book, and an existing one gaining two records of base keys.
        let mut new_book = TestBook::named("id:99999");
        new_book.primary = (0..per_book)
            .map(|i| (key(1_000_000 + i as u64), i as u32, random.unit(DIM)))
            .collect();
        new_book.foreign = vec![(key(5), 77), (key(123_456_789), 3)];
        let mut grown = TestBook::named("id:00002");
        grown.foreign = vec![(key(0), 900)];
        let delta = dir.join("delta.oxv");
        write_segment(
            &delta,
            spec(PackageKind::Delta, 29, 30),
            codec,
            &[grown, new_book],
            &[],
        );

        let segments = vec![
            Segment::open(&base).unwrap(),
            Segment::open(&delta).unwrap(),
        ];
        // Slot 3 of the base is deleted, as a tombstone would; so is slot 0 of the delta.
        let mut deleted = vec![
            vec![0u64; (segments[0].slot_count() as usize).div_ceil(64)],
            vec![0u64; (segments[1].slot_count() as usize).div_ceil(64)],
        ];
        deleted[0][0] |= 1 << 3;
        deleted[1][0] |= 1;
        // The delta's foreign records, resolved against the base by key.
        let base_slots: HashMap<ChunkKey, u32> = (0..segments[0].slot_count())
            .map(|slot| (segments[0].key(slot), slot))
            .collect();
        let links: Vec<Link> = (0..segments[1].foreign_count())
            .map(|index| {
                let (key, _, _) = segments[1].foreign_record(index);
                match base_slots.get(&key) {
                    Some(slot) => Link {
                        seg: 0,
                        flags: 0,
                        slot: *slot,
                    },
                    None => Link {
                        seg: 0,
                        flags: LINK_UNRESOLVED,
                        slot: 0,
                    },
                }
            })
            .collect();
        let mut reverse: Vec<ReverseLink> = links
            .iter()
            .enumerate()
            .filter_map(|(index, link)| {
                link.target().map(|(seg, slot)| ReverseLink {
                    target_seg: seg,
                    target_slot: slot,
                    from_seg: 1,
                    foreign: index as u32,
                })
            })
            .collect();
        reverse.sort();
        Self {
            _dir: dir,
            segments,
            deleted,
            links: vec![Vec::new(), links],
            reverse,
        }
    }

    fn scan_segments(&self) -> Vec<ScanSegment<'_>> {
        self.segments
            .iter()
            .zip(&self.deleted)
            .zip(&self.links)
            .map(|((segment, deleted), links)| ScanSegment {
                segment,
                deleted,
                links,
            })
            .collect()
    }

    /// Every live slot, scored and ordered by the scan's total order, then deduplicated by
    /// key — what the scan must return, computed the slow way.
    fn brute_force(
        &self,
        query: &PreparedQuery,
        admit: impl Fn(u16, u32) -> bool,
    ) -> Vec<(i32, ChunkKey, u16, u32)> {
        let (kernel, _) = select();
        let mut all = Vec::new();
        for (seg, segment) in self.segments.iter().enumerate() {
            for slot in 0..segment.slot_count() {
                let deleted = self.deleted[seg][slot as usize / 64] & (1 << (slot % 64)) != 0;
                if deleted || !admit(seg as u16, slot) {
                    continue;
                }
                let vector = segment.vector(slot);
                let rank = match query {
                    PreparedQuery::Int8 { q16, .. } => kernel(q16, vector),
                    PreparedQuery::Int8PerVector { q16, .. } => super::kernel::ordered(
                        kernel(q16, vector) as f32 * segment.vector_scale(slot).unwrap(),
                    ),
                    PreparedQuery::Float { q } => super::kernel::ordered(dot_f32(q, vector)),
                };
                all.push((rank, segment.key(slot), seg as u16, slot));
            }
        }
        all.sort_by(|a, b| {
            b.0.cmp(&a.0)
                .then(a.1.cmp(&b.1))
                .then((a.2, a.3).cmp(&(b.2, b.3)))
        });
        let mut seen = std::collections::HashSet::new();
        all.retain(|entry| seen.insert(entry.1));
        all
    }
}

fn int8(vectors: &[&[f32]]) -> Codec {
    Codec::calibrate_i8_sym_dim(vectors, 1.0).unwrap()
}

fn per_vector(vectors: &[&[f32]]) -> Codec {
    Codec::i8_sym_vec(vectors[0].len()).unwrap()
}

fn run(
    fixture: &Fixture,
    query: &PreparedQuery,
    top_k: usize,
    books: Option<&BookSet>,
    threads: usize,
) -> Vec<(i32, ChunkKey, u16, u32)> {
    let segments = fixture.scan_segments();
    let set = ScanSet {
        segments: &segments,
        reverse_links: &fixture.reverse,
    };
    let request = ScanRequest {
        top_k,
        books,
        threads,
    };
    let hits = scan(&set, query, &request, &CancellationToken::new()).unwrap();
    let (kernel, _) = select();
    hits.into_iter()
        .map(|hit| {
            let vector = fixture.segments[hit.seg as usize].vector(hit.slot);
            let rank = match query {
                PreparedQuery::Int8 { q16, inv_scale } => {
                    let rank = kernel(q16, vector);
                    assert_eq!(hit.score, rank as f32 * inv_scale);
                    rank
                }
                PreparedQuery::Int8PerVector { q16, inv_scale } => {
                    let scale = fixture.segments[hit.seg as usize]
                        .vector_scale(hit.slot)
                        .unwrap();
                    let product = kernel(q16, vector) as f32 * scale;
                    assert_eq!(hit.score, product * inv_scale);
                    super::kernel::ordered(product)
                }
                PreparedQuery::Float { q } => {
                    assert_eq!(hit.score, dot_f32(q, vector));
                    super::kernel::ordered(hit.score)
                }
            };
            assert_eq!(hit.key, fixture.segments[hit.seg as usize].key(hit.slot));
            (rank, hit.key, hit.seg, hit.slot)
        })
        .collect()
}

/// The top `k` of a scan is the top `k` of a full sort, for every `k` — the threshold, the
/// lazy key and the tie-breaking included — and deleted slots never appear.
#[test]
fn the_top_k_is_the_head_of_a_full_sort() {
    for (seed, codec) in [
        (31u64, int8 as fn(&[&[f32]]) -> Codec),
        (32, |v: &[&[f32]]| Codec::f32(v[0].len()).unwrap()),
        (37, per_vector),
    ] {
        let fixture = Fixture::new(40, 50, seed, codec);
        let mut random = Random(seed + 100);
        let query = PreparedQuery::new(&random.unit(DIM), fixture.segments[0].codec()).unwrap();
        let expected = fixture.brute_force(&query, |_, _| true);
        for k in [1, 2, 10, 137, expected.len(), expected.len() + 50] {
            let got = run(&fixture, &query, k, None, 1);
            assert_eq!(got, expected[..k.min(expected.len())], "k {k}");
        }
        assert!(run(&fixture, &query, 0, None, 1).is_empty());
        let all = run(&fixture, &query, usize::MAX >> 8, None, 3);
        assert!(!all
            .iter()
            .any(|(_, _, seg, slot)| (*seg, *slot) == (0, 3) || (*seg, *slot) == (1, 0)));
    }
}

/// Ties are broken by key, so a vector set holding many equal scores still answers the
/// same way every time.
#[test]
fn equal_scores_are_ordered_by_key() {
    let dir = TempDir::new("ties");
    let vector: Vec<f32> = (0..DIM).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
    let mut book = TestBook::named("id:1");
    book.primary = (0..300)
        .map(|i| (key(i), i as u32, vector.clone()))
        .collect();
    let path = dir.join("ties.oxv");
    write_segment(
        &path,
        spec(PackageKind::Base, 0, 30),
        int8(&[vector.as_slice()]),
        &[book],
        &[],
    );
    let segment = Segment::open(&path).unwrap();
    let segments = [ScanSegment {
        segment: &segment,
        deleted: &[],
        links: &[],
    }];
    let set = ScanSet {
        segments: &segments,
        reverse_links: &[],
    };
    let query = PreparedQuery::new(&vector, segment.codec()).unwrap();
    for threads in [1, 4] {
        let hits = scan(
            &set,
            &query,
            &ScanRequest {
                top_k: 25,
                books: None,
                threads,
            },
            &CancellationToken::new(),
        )
        .unwrap();
        let keys: Vec<ChunkKey> = hits.iter().map(|hit| hit.key).collect();
        let mut expected: Vec<ChunkKey> = (0..300).map(key).collect();
        expected.sort();
        assert_eq!(keys, expected[..25]);
    }
}

/// The partition into threads changes nothing: one thread and eight return the same hits,
/// in the same order, with the same records.
#[test]
fn one_thread_and_eight_return_the_same_hits() {
    for codec in [int8 as fn(&[&[f32]]) -> Codec, per_vector] {
        threads_agree(Fixture::new(200, 300, 33, codec));
    }
}

fn threads_agree(fixture: Fixture) {
    let segments = fixture.scan_segments();
    let set = ScanSet {
        segments: &segments,
        reverse_links: &fixture.reverse,
    };
    let mut random = Random(34);
    for _ in 0..5 {
        let query = PreparedQuery::new(&random.unit(DIM), fixture.segments[0].codec()).unwrap();
        let one = scan(
            &set,
            &query,
            &ScanRequest {
                top_k: 100,
                books: None,
                threads: 1,
            },
            &CancellationToken::new(),
        )
        .unwrap();
        for threads in [2, 3, 8] {
            let many = scan(
                &set,
                &query,
                &ScanRequest {
                    top_k: 100,
                    books: None,
                    threads,
                },
                &CancellationToken::new(),
            )
            .unwrap();
            assert_eq!(many, one, "{threads} threads");
        }
    }
}

/// A filter scans the admitted books' ranges plus what their extras and foreign records
/// point to, and returns exactly what filtering a full scan by records would.
#[test]
fn a_filtered_scan_is_a_full_scan_filtered_by_records() {
    let fixture = Fixture::new(60, 40, 35, int8);
    let mut random = Random(36);
    let query = PreparedQuery::new(&random.unit(DIM), fixture.segments[0].codec()).unwrap();
    // Which books hold each live slot's key, from every kind of record.
    let mut holders: HashMap<(u16, u32), Vec<String>> = HashMap::new();
    for (seg, segment) in fixture.segments.iter().enumerate() {
        let seg = seg as u16;
        for book in segment.books() {
            for slot in book.slots.clone() {
                holders
                    .entry((seg, slot))
                    .or_default()
                    .push(book.name.to_string());
            }
            for extra in book.extras.clone() {
                let (slot, _) = segment.extra(extra);
                holders
                    .entry((seg, slot))
                    .or_default()
                    .push(book.name.to_string());
            }
            for foreign in book.foreign.clone() {
                if let Some(target) = fixture.links[seg as usize][foreign as usize].target() {
                    holders
                        .entry(target)
                        .or_default()
                        .push(book.name.to_string());
                }
            }
        }
    }
    for filter in [
        vec!["id:00001"],
        vec!["id:00000", "id:00007", "id:00031"],
        vec!["id:00002"],
        vec!["id:99999"],
        vec!["id:no-such-book"],
    ] {
        let books: BookSet = filter.iter().copied().collect();
        let expected = fixture.brute_force(&query, |seg, slot| {
            holders
                .get(&(seg, slot))
                .is_some_and(|names| names.iter().any(|name| books.contains(name)))
        });
        for threads in [1, 4] {
            let got = run(&fixture, &query, 1000, Some(&books), threads);
            assert_eq!(got, expected, "{filter:?}, {threads} threads");
        }
    }
}

/// A hit carries its primary record, its extras and the foreign records of later deltas,
/// the admitted books first.
#[test]
fn a_hit_carries_every_record_of_its_key_admitted_books_first() {
    let fixture = Fixture::new(10, 20, 37, int8);
    let segments = fixture.scan_segments();
    let set = ScanSet {
        segments: &segments,
        reverse_links: &fixture.reverse,
    };
    // Key 0: primary in book 0 at hint 0, an extra in book 1 at hint 1000, and a foreign
    // record in the delta's book id:00002 at hint 900. A query equal to its vector puts it
    // first.
    let mut decoded = vec![0f32; DIM];
    fixture.segments[0].codec().decode(
        fixture.segments[0].vector(0),
        fixture.segments[0].vector_scale(0),
        &mut decoded,
    );
    let norm = decoded.iter().map(|x| x * x).sum::<f32>().sqrt();
    let query: Vec<f32> = decoded.iter().map(|x| x / norm).collect();
    let query = PreparedQuery::new(&query, fixture.segments[0].codec()).unwrap();

    let hits = scan(
        &set,
        &query,
        &ScanRequest {
            top_k: 1,
            books: None,
            threads: 1,
        },
        &CancellationToken::new(),
    )
    .unwrap();
    assert_eq!(hits[0].key, key(0));
    let records: Vec<(&str, u32)> = hits[0].records.iter().map(|r| (&*r.book, r.hint)).collect();
    assert_eq!(
        records,
        [("id:00000", 0), ("id:00001", 1000), ("id:00002", 900)]
    );

    let books: BookSet = ["id:00002"].into_iter().collect();
    let hits = scan(
        &set,
        &query,
        &ScanRequest {
            top_k: 1,
            books: Some(&books),
            threads: 1,
        },
        &CancellationToken::new(),
    )
    .unwrap();
    let records: Vec<(&str, u32)> = hits[0].records.iter().map(|r| (&*r.book, r.hint)).collect();
    assert_eq!(
        records,
        [("id:00002", 900), ("id:00000", 0), ("id:00001", 1000)]
    );
}

/// A key held in a thousand books carries 32 of them.
#[test]
fn a_hit_carries_at_most_32_records() {
    let dir = TempDir::new("boilerplate");
    let mut random = Random(38);
    let mut books = vec![TestBook::named("id:0000")];
    books[0].primary = vec![(key(1), 0, random.unit(DIM))];
    for b in 1..1000 {
        let mut book = TestBook::named(&format!("id:{b:04}"));
        book.extras = vec![(5, 0)];
        books.push(book);
    }
    let path = dir.join("many.oxv");
    write_segment(
        &path,
        spec(PackageKind::Base, 0, 30),
        int8(&[books[0].primary[0].2.as_slice()]),
        &books,
        &[],
    );
    let segment = Segment::open(&path).unwrap();
    let segments = [ScanSegment {
        segment: &segment,
        deleted: &[],
        links: &[],
    }];
    let set = ScanSet {
        segments: &segments,
        reverse_links: &[],
    };
    let query = PreparedQuery::new(&random.unit(DIM), segment.codec()).unwrap();
    let wanted: BookSet = ["id:0999"].into_iter().collect();
    let hits = scan(
        &set,
        &query,
        &ScanRequest {
            top_k: 1,
            books: Some(&wanted),
            threads: 1,
        },
        &CancellationToken::new(),
    )
    .unwrap();
    assert_eq!(hits[0].records.len(), MAX_RECORDS_PER_HIT);
    assert_eq!(
        &*hits[0].records[0].book, "id:0999",
        "the admitted book comes first"
    );
}

/// A token cancelled before the scan stops it at its first checkpoint; one cancelled at a
/// checkpoint stops it there, on the calling thread.
#[test]
fn a_cancelled_scan_stops_at_its_next_checkpoint() {
    let fixture = Fixture::new(100, 100, 39, int8);
    let segments = fixture.scan_segments();
    let set = ScanSet {
        segments: &segments,
        reverse_links: &fixture.reverse,
    };
    let query = PreparedQuery::new(&Random(40).unit(DIM), fixture.segments[0].codec()).unwrap();
    let request = ScanRequest {
        top_k: 10,
        books: None,
        threads: 1,
    };

    let token = CancellationToken::new();
    let (result, seen) = probe::checkpoints_of(|| scan(&set, &query, &request, &token));
    assert!(result.is_ok());
    let slots = 100 * 100 + 100;
    let expected: Vec<usize> = std::iter::once(0)
        .chain((0..slots).step_by(SCAN_CHECK_INTERVAL))
        .collect();
    assert_eq!(seen, expected, "one look first, then every interval");

    let token = CancellationToken::new();
    let at = 3 * SCAN_CHECK_INTERVAL;
    let (result, seen) = probe::cancelling_at(&token, at, || scan(&set, &query, &request, &token));
    assert!(matches!(result, Err(VectorStoreError::Cancelled)));
    assert_eq!(seen.last(), Some(&at));

    let token = CancellationToken::new();
    token.cancel();
    let (result, seen) = probe::checkpoints_of(|| scan(&set, &query, &request, &token));
    assert!(matches!(result, Err(VectorStoreError::Cancelled)));
    assert_eq!(seen, [0]);
}

/// Cancelled from another thread while four scan, every one of them stops within two
/// intervals of where it was when the token was cancelled: it notices at its next
/// checkpoint, and the position it last reported is at most one interval behind it.
///
/// On a busy machine the cancelling thread can be descheduled until a worker has all but
/// finished its share — a run that shows nothing, which is run again.
#[test]
fn a_scan_on_four_threads_stops_within_two_intervals_of_a_cancel() {
    let fixture = Fixture::new(200, 1000, 41, int8);
    let segments = fixture.scan_segments();
    let set = ScanSet {
        segments: &segments,
        reverse_links: &fixture.reverse,
    };
    let query = PreparedQuery::new(&Random(42).unit(DIM), fixture.segments[0].codec()).unwrap();
    let request = ScanRequest {
        top_k: 10,
        books: None,
        threads: 4,
    };
    let share = (200 * 1000 + 1000) / 4;

    for attempt in 1.. {
        let progress: Vec<AtomicUsize> = (0..4).map(|_| AtomicUsize::new(0)).collect();
        let token = CancellationToken::new();
        let (result, snapshot) = std::thread::scope(|scope| {
            let scan =
                scope.spawn(|| scan_reporting(&set, &query, &request, &token, Some(&progress)));
            let started = std::time::Instant::now();
            while progress[0].load(Ordering::Relaxed) < 4 * SCAN_CHECK_INTERVAL {
                assert!(started.elapsed().as_secs() < 60, "the scan never got going");
                std::thread::yield_now();
            }
            token.cancel();
            let snapshot: Vec<usize> = progress.iter().map(|p| p.load(Ordering::Relaxed)).collect();
            (scan.join().unwrap(), snapshot)
        });
        let mid_scan = snapshot
            .iter()
            .all(|before| before + 2 * SCAN_CHECK_INTERVAL < share);
        if !mid_scan {
            assert!(
                attempt < 50,
                "in {attempt} runs the cancel never came mid-scan"
            );
            continue;
        }
        assert!(matches!(result, Err(VectorStoreError::Cancelled)));
        for (worker, (counter, before)) in progress.iter().zip(&snapshot).enumerate() {
            let stopped = counter.load(Ordering::Relaxed);
            assert!(
                stopped <= before + 2 * SCAN_CHECK_INTERVAL,
                "worker {worker} went on from {before} to {stopped}"
            );
        }
        break;
    }
}
