//! The exact scan: every live slot of a set scored against one query, the best `k` kept.
//!
//! No index narrows it — at 256 int8 components a vector, a full scan of six million is a
//! memory-bandwidth problem, not an arithmetic one — but three things keep it short:
//!
//! * **threads**, each scanning an equal share of the slots with `std::thread::scope` (no
//!   pool in a library the application links), the calling thread running the first share;
//! * **a book filter** that becomes ranges: a segment stores each book's primary slots
//!   together, so a search restricted to some books scans their ranges, plus the few slots
//!   whose key those books hold as an extra or a foreign record — gathered one by one;
//! * **a lazy key**: the 16-byte key is read only for a vector good enough to enter the
//!   top `k`, where it breaks ties.
//!
//! A filtered scan can also weigh slots its filter does not reach, named one by one
//! ([`SegmentSet::scan_with`](crate::semantic::segment_set::SegmentSet::scan_with)): each
//! is scored as the scan scores, and the best `k` of them are merged into its hits — beside
//! them, never in their place.
//!
//! The ranking is a total order — score, then key, then where the slot is — computed in
//! integers (the `kernel` module), so the same vectors give the same hits in the same order on
//! every CPU, at any thread count and in any segment layout. A cancelled token is noticed at
//! a checkpoint every [`SCAN_CHECK_INTERVAL`] slots, in every thread.

use crate::cancellation::{CancellationToken, SCAN_CHECK_INTERVAL};
use crate::errors::VectorStoreError;
use crate::semantic::chunk_key::ChunkKey;
use crate::semantic::oxv::codec::Codec;
use crate::semantic::oxv::kernel::{
    dot_f32, ordered, select, select_nibbles, unordered, DotKernel, NibbleKernel, PreparedQuery,
};
use crate::semantic::oxv::occurrence::ALIAS_WINDOW;
use crate::semantic::oxv::reader::Segment;
use crate::semantic::resolve::{BookSet, RecordRef, SlotRef, VectorHit, MAX_RECORDS_PER_HIT};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::sync::atomic::AtomicUsize;

/// Below this many slots a share is not worth a thread of its own.
pub(super) const MIN_SLOTS_PER_THREAD: usize = 4096;

/// Keep the per-worker heaps bounded as the number of queries grows. The common
/// passage search (up to twelve windows) fits in one pass.
pub(super) const QUERIES_PER_PASS: usize = 16;

/// Soft budget for the per-worker heaps and one query's merge buffer. A request
/// whose single-query heaps exceed it still gets the top-k it asked for.
const BATCH_CANDIDATE_BUDGET: usize = 32 * 1024 * 1024;

pub(super) fn queries_per_pass(candidates_per_query: usize) -> usize {
    let bytes = candidates_per_query.saturating_mul(size_of::<Candidate>());
    (BATCH_CANDIDATE_BUDGET / bytes.max(1))
        .saturating_sub(1)
        .clamp(1, QUERIES_PER_PASS)
}

/// A foreign record whose key no older segment of the set holds live.
pub(crate) const LINK_UNRESOLVED: u16 = 1;

/// Where one foreign record of a delta resolves: the older segment and slot holding its
/// key, as the set computed when the delta was applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Link {
    pub seg: u16,
    pub flags: u16,
    pub slot: u32,
}

impl Link {
    pub(crate) fn target(&self) -> Option<(u16, u32)> {
        (self.flags & LINK_UNRESOLVED == 0).then_some((self.seg, self.slot))
    }
}

/// A foreign record, seen from the slot it resolves to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct ReverseLink {
    pub target_seg: u16,
    pub target_slot: u32,
    /// The delta holding the foreign record, and the record's index in it.
    pub from_seg: u16,
    pub foreign: u32,
}

/// One segment as the scan sees it.
pub(crate) struct ScanSegment<'a> {
    pub segment: &'a Segment,
    /// One bit per slot, set for a slot no search may return. Words past the end read as
    /// zero.
    pub deleted: &'a [u64],
    /// One entry per foreign record of the segment.
    pub links: &'a [Link],
}

impl ScanSegment<'_> {
    fn is_deleted(&self, slot: u32) -> bool {
        self.deleted
            .get(slot as usize / 64)
            .is_some_and(|word| word & (1 << (slot % 64)) != 0)
    }
}

/// A set as the scan sees it: its segments, oldest first, and the reverse of every link,
/// sorted.
pub(crate) struct ScanSet<'a> {
    pub segments: &'a [ScanSegment<'a>],
    pub reverse_links: &'a [ReverseLink],
}

/// What to scan for.
#[derive(Debug, Clone, Copy)]
pub struct ScanRequest<'a> {
    /// How many hits to return. Zero returns none.
    pub top_k: usize,
    /// Only vectors with a record in one of these books, when given.
    pub books: Option<&'a BookSet>,
    /// Threads to scan with, the calling one included; at least one is used.
    pub threads: usize,
    /// Taken off the cosine of every slot past a line's first window, before the best `k`
    /// are chosen. 0 ranks every slot by its cosine alone.
    pub window_penalty: f32,
}

/// Penalties used when choosing the representative occurrence before top-k selection.
/// The public request keeps its historical window-only shape; callers that know the
/// alias penalty use the explicit-penalty entry points.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ScanPenalties {
    pub window: f32,
    pub alias: f32,
}

#[derive(Clone, Copy)]
struct ScanConfig<'request, 'books> {
    request: &'request ScanRequest<'books>,
    penalties: ScanPenalties,
}

/// The thread count the application gets by default: half the cores, at most eight. A
/// scan is memory-bound, and past the cores of one memory channel more threads add
/// contention, not speed.
pub fn default_scan_threads() -> usize {
    let cores = std::thread::available_parallelism().map_or(1, |cores| cores.get());
    (cores / 2).clamp(1, 8)
}

/// Scan open segments as one set with nothing deleted: every slot of each, scored against
/// `query` — a unit vector of the segments' width — the best `request.top_k` returned.
///
/// The segments must share one codec epoch; a set enforces that, and this is its scan
/// without the set's bookkeeping, for a tool or a benchmark holding segments of its own.
pub fn scan_segments(
    segments: &[&Segment],
    query: &[f32],
    request: &ScanRequest<'_>,
    cancel: &CancellationToken,
) -> Result<Vec<VectorHit>, VectorStoreError> {
    let Some(first) = segments.first() else {
        return Ok(Vec::new());
    };
    if let Some(other) = segments
        .iter()
        .find(|segment| segment.codec_params_sha256() != first.codec_params_sha256())
    {
        return Err(VectorStoreError::SearchFailed {
            reason: format!(
                "{} and {} are of different codec epochs, and their scores do not compare",
                first.path().display(),
                other.path().display()
            ),
        });
    }
    let prepared = PreparedQuery::new(query, first.codec())?;
    let views: Vec<ScanSegment<'_>> = segments
        .iter()
        .map(|segment| ScanSegment {
            segment,
            deleted: &[],
            links: &[],
        })
        .collect();
    let set = ScanSet {
        segments: &views,
        reverse_links: &[],
    };
    scan(&set, &prepared, request, cancel)
}

/// The same hits as [`scan_segments`] for each query, in input order. Large batches are
/// processed in groups to bound the working memory.
pub fn scan_segments_many(
    segments: &[&Segment],
    queries: &[&[f32]],
    request: &ScanRequest<'_>,
    cancel: &CancellationToken,
) -> Result<Vec<Vec<VectorHit>>, VectorStoreError> {
    let Some(first) = segments.first() else {
        return Ok(vec![Vec::new(); queries.len()]);
    };
    if let Some(other) = segments
        .iter()
        .find(|segment| segment.codec_params_sha256() != first.codec_params_sha256())
    {
        return Err(VectorStoreError::SearchFailed {
            reason: format!(
                "{} and {} are of different codec epochs, and their scores do not compare",
                first.path().display(),
                other.path().display()
            ),
        });
    }
    let prepared = prepare_queries(queries, first.codec(), cancel)?;
    let views: Vec<ScanSegment<'_>> = segments
        .iter()
        .map(|segment| ScanSegment {
            segment,
            deleted: &[],
            links: &[],
        })
        .collect();
    let set = ScanSet {
        segments: &views,
        reverse_links: &[],
    };
    scan_many(
        &set,
        &prepared,
        request,
        ScanPenalties {
            window: request.window_penalty,
            alias: request.window_penalty,
        },
        cancel,
    )
}

/// Scan `set` for the `top_k` best vectors under `request`.
pub(crate) fn scan(
    set: &ScanSet<'_>,
    query: &PreparedQuery,
    request: &ScanRequest<'_>,
    cancel: &CancellationToken,
) -> Result<Vec<VectorHit>, VectorStoreError> {
    scan_reporting(set, query, request, cancel, None)
}

/// [`scan`], and the slots of `also` besides: each that is live and holds its key, scored
/// as the scan scores. Of those whose key the scan did not return, the best
/// `request.top_k` are merged into its hits in the scan's order; none of the scan's hits
/// gives way to them, so the result holds up to twice `request.top_k` hits. A slot past
/// its segment's end, deleted, or holding another key is passed over. With `also` empty,
/// exactly [`scan`].
#[cfg(test)]
pub(crate) fn scan_with(
    set: &ScanSet<'_>,
    query: &PreparedQuery,
    request: &ScanRequest<'_>,
    also: &[SlotRef],
    cancel: &CancellationToken,
) -> Result<Vec<VectorHit>, VectorStoreError> {
    scan_with_penalties(
        set,
        query,
        request,
        also,
        ScanPenalties {
            window: request.window_penalty,
            alias: request.window_penalty,
        },
        cancel,
    )
}

pub(crate) fn scan_with_penalties(
    set: &ScanSet<'_>,
    query: &PreparedQuery,
    request: &ScanRequest<'_>,
    also: &[SlotRef],
    penalties: ScanPenalties,
    cancel: &CancellationToken,
) -> Result<Vec<VectorHit>, VectorStoreError> {
    scan_all(set, query, request, also, penalties, cancel, None)
}

/// [`scan`], with each thread's progress — slots visited at its last checkpoint — written
/// to `progress[thread]` when given. What the cross-thread cancellation test watches.
pub(crate) fn scan_reporting(
    set: &ScanSet<'_>,
    query: &PreparedQuery,
    request: &ScanRequest<'_>,
    cancel: &CancellationToken,
    progress: Option<&[AtomicUsize]>,
) -> Result<Vec<VectorHit>, VectorStoreError> {
    scan_all(
        set,
        query,
        request,
        &[],
        ScanPenalties {
            window: request.window_penalty,
            alias: request.window_penalty,
        },
        cancel,
        progress,
    )
}

fn scan_all(
    set: &ScanSet<'_>,
    query: &PreparedQuery,
    request: &ScanRequest<'_>,
    also: &[SlotRef],
    penalties: ScanPenalties,
    cancel: &CancellationToken,
    progress: Option<&[AtomicUsize]>,
) -> Result<Vec<VectorHit>, VectorStoreError> {
    cancel.scan_checkpoint(0)?;
    check_penalties(request, penalties)?;
    if request.top_k == 0 {
        return Ok(Vec::new());
    }
    check_dim(set, query)?;
    let work = Work::plan(set, request.books);
    // No more candidates than there are slots to visit, so a `top_k` sized for "all of
    // them" allocates for what exists.
    let top_k = request.top_k.min(work.total.max(1));
    let threads = request
        .threads
        .max(1)
        .min(work.total.div_ceil(MIN_SLOTS_PER_THREAD).max(1));
    let shares = work.split(threads);
    let config = ScanConfig { request, penalties };

    match query {
        PreparedQuery::Int8 { q16, inv_scale } => {
            let (kernel, _) = select();
            let scorer = Int8 {
                q16,
                inv_scale: *inv_scale,
                kernel,
            };
            let found = run_shares(set, &scorer, config, top_k, &shares, cancel, progress)?;
            let found = merge_also(set, &scorer, request, penalties, found, also, cancel)?;
            Ok(hits(set, found, request, penalties, &scorer))
        }
        PreparedQuery::Int8PerVector { q16, inv_scale } => {
            let (kernel, _) = select();
            let scorer = Int8PerVector {
                q16,
                inv_scale: *inv_scale,
                kernel,
            };
            let found = run_shares(set, &scorer, config, top_k, &shares, cancel, progress)?;
            let found = merge_also(set, &scorer, request, penalties, found, also, cancel)?;
            Ok(hits(set, found, request, penalties, &scorer))
        }
        PreparedQuery::Rotated {
            head,
            inv_head,
            tail_lo,
            tail_hi,
            inv_tail,
            constant,
            ..
        } => {
            let scorer = Rotated {
                head,
                inv_head: *inv_head,
                tail_lo,
                tail_hi,
                inv_tail: *inv_tail,
                constant: *constant,
                dot: select().0,
                nibbles: select_nibbles().0,
            };
            let found = run_shares(set, &scorer, config, top_k, &shares, cancel, progress)?;
            let found = merge_also(set, &scorer, request, penalties, found, also, cancel)?;
            Ok(hits(set, found, request, penalties, &scorer))
        }
        PreparedQuery::Float { q } => {
            let scorer = Float { q };
            let found = run_shares(set, &scorer, config, top_k, &shares, cancel, progress)?;
            let found = merge_also(set, &scorer, request, penalties, found, also, cancel)?;
            Ok(hits(set, found, request, penalties, &scorer))
        }
    }
}

fn check_penalties(
    request: &ScanRequest<'_>,
    penalties: ScanPenalties,
) -> Result<(), VectorStoreError> {
    if !request.window_penalty.is_finite() || request.window_penalty < 0.0 {
        return Err(VectorStoreError::SearchFailed {
            reason: "window penalty must be finite and nonnegative".to_string(),
        });
    }
    if !penalties.window.is_finite() || penalties.window < 0.0 {
        return Err(VectorStoreError::SearchFailed {
            reason: "window penalty must be finite and nonnegative".to_string(),
        });
    }
    if penalties.window != request.window_penalty {
        return Err(VectorStoreError::SearchFailed {
            reason: "scan penalties must match the request's window penalty".to_string(),
        });
    }
    if !penalties.alias.is_finite() || penalties.alias < 0.0 {
        return Err(VectorStoreError::SearchFailed {
            reason: "alias penalty must be finite and nonnegative".to_string(),
        });
    }
    Ok(())
}

fn check_dim(set: &ScanSet<'_>, query: &PreparedQuery) -> Result<(), VectorStoreError> {
    for segment in set.segments {
        if segment.segment.codec().dim() != query.dim() {
            return Err(VectorStoreError::DimensionMismatch {
                store_dim: segment.segment.codec().dim() as u32,
                vector_dim: query.dim() as u32,
            });
        }
    }
    Ok(())
}

/// Prepare a batch for one codec, checking cancellation between queries.
pub(crate) fn prepare_queries(
    queries: &[&[f32]],
    codec: &Codec,
    cancel: &CancellationToken,
) -> Result<Vec<PreparedQuery>, VectorStoreError> {
    cancel.check_scan_cancelled()?;
    queries
        .iter()
        .map(|query| {
            cancel.check_scan_cancelled()?;
            PreparedQuery::new(query, codec)
        })
        .collect()
}

/// The same ordered hits as [`scan_all`] for each query, without additional slots.
/// Queries in a group share the scan plan and reuse each vector while it is in cache.
/// Each query keeps its own ranking, and a slot's role is resolved on demand once per group.
pub(crate) fn scan_many(
    set: &ScanSet<'_>,
    queries: &[PreparedQuery],
    request: &ScanRequest<'_>,
    penalties: ScanPenalties,
    cancel: &CancellationToken,
) -> Result<Vec<Vec<VectorHit>>, VectorStoreError> {
    if let [query] = queries {
        return scan_all(set, query, request, &[], penalties, cancel, None).map(|hits| vec![hits]);
    }

    cancel.scan_checkpoint(0)?;
    check_penalties(request, penalties)?;
    if request.top_k == 0 || queries.is_empty() {
        return Ok(vec![Vec::new(); queries.len()]);
    }
    for query in queries {
        check_dim(set, query)?;
    }
    let work = Work::plan(set, request.books);
    cancel.check_scan_cancelled()?;
    if work.total == 0 {
        return Ok(vec![Vec::new(); queries.len()]);
    }
    let top_k = request.top_k.min(work.total);
    let threads = request
        .threads
        .max(1)
        .min(work.total.div_ceil(MIN_SLOTS_PER_THREAD).max(1));
    // Work shares have equal sizes to within one slot. Each holds at most top_k.
    let group_size = queries_per_pass(top_k.saturating_mul(threads).min(work.total));
    let shares = work.split(threads);
    let config = ScanConfig { request, penalties };
    let mixed = || VectorStoreError::SearchFailed {
        reason: "queries prepared for different codecs cannot be scanned together".to_string(),
    };

    fn finish<S: Scorer>(
        set: &ScanSet<'_>,
        scorers: &[S],
        config: ScanConfig<'_, '_>,
        top_k: usize,
        group_size: usize,
        shares: &[Vec<Piece>],
        cancel: &CancellationToken,
    ) -> Result<Vec<Vec<VectorHit>>, VectorStoreError> {
        let mut hits_by_query = Vec::with_capacity(scorers.len());
        for group in scorers.chunks(group_size) {
            cancel.check_scan_cancelled()?;
            let mut workers = run_shares_many(set, group, config, top_k, shares, cancel)?;
            for scorer in group {
                cancel.check_scan_cancelled()?;
                let heap_len: usize = workers
                    .iter()
                    .map(|worker| worker.as_slice()[0].top.heap.len())
                    .sum();
                let mut heaps = workers.iter_mut().map(|worker| {
                    worker
                        .next()
                        .expect("each worker has a heap per query")
                        .top
                        .heap
                });
                // Reuse the first heap's backing storage. Materialize one query at a time
                // so the other heaps need no duplicate candidate storage.
                let mut found = heaps.next().expect("at least one worker").into_vec();
                found.reserve_exact(heap_len - found.len());
                for heap in heaps {
                    found.extend(heap.into_vec());
                }
                let found = best_of(found, top_k);
                cancel.check_scan_cancelled()?;
                hits_by_query.push(hits(set, found, config.request, config.penalties, scorer));
            }
        }
        cancel.check_scan_cancelled()?;
        Ok(hits_by_query)
    }

    match &queries[0] {
        PreparedQuery::Int8 { .. } => {
            let (kernel, _) = select();
            let scorers = queries
                .iter()
                .map(|query| match query {
                    PreparedQuery::Int8 { q16, inv_scale } => Ok(Int8 {
                        q16,
                        inv_scale: *inv_scale,
                        kernel,
                    }),
                    _ => Err(mixed()),
                })
                .collect::<Result<Vec<_>, _>>()?;
            finish(set, &scorers, config, top_k, group_size, &shares, cancel)
        }
        PreparedQuery::Int8PerVector { .. } => {
            let (kernel, _) = select();
            let scorers = queries
                .iter()
                .map(|query| match query {
                    PreparedQuery::Int8PerVector { q16, inv_scale } => Ok(Int8PerVector {
                        q16,
                        inv_scale: *inv_scale,
                        kernel,
                    }),
                    _ => Err(mixed()),
                })
                .collect::<Result<Vec<_>, _>>()?;
            finish(set, &scorers, config, top_k, group_size, &shares, cancel)
        }
        PreparedQuery::Rotated { .. } => {
            let (dot, nibbles) = (select().0, select_nibbles().0);
            let scorers = queries
                .iter()
                .map(|query| match query {
                    PreparedQuery::Rotated {
                        head,
                        inv_head,
                        tail_lo,
                        tail_hi,
                        inv_tail,
                        constant,
                        ..
                    } => Ok(Rotated {
                        head,
                        inv_head: *inv_head,
                        tail_lo,
                        tail_hi,
                        inv_tail: *inv_tail,
                        constant: *constant,
                        dot,
                        nibbles,
                    }),
                    _ => Err(mixed()),
                })
                .collect::<Result<Vec<_>, _>>()?;
            finish(set, &scorers, config, top_k, group_size, &shares, cancel)
        }
        PreparedQuery::Float { .. } => {
            let scorers = queries
                .iter()
                .map(|query| match query {
                    PreparedQuery::Float { q } => Ok(Float { q }),
                    _ => Err(mixed()),
                })
                .collect::<Result<Vec<_>, _>>()?;
            finish(set, &scorers, config, top_k, group_size, &shares, cancel)
        }
    }
}

/// How one codec turns a stored vector — and its own scale, for a codec with one — into a
/// rank, and a rank back into a score.
trait Scorer: Sync {
    fn rank(&self, vector: &[u8], scale: f32) -> i32;
    fn score(&self, rank: i32) -> f32;
    /// `rank` with `penalty` taken off the score it stands for, still an exact total order.
    fn penalize(&self, rank: i32, penalty: f32) -> i32;
}

struct Int8<'a> {
    q16: &'a [i16],
    inv_scale: f32,
    kernel: DotKernel,
}

impl Scorer for Int8<'_> {
    #[inline]
    fn rank(&self, vector: &[u8], _scale: f32) -> i32 {
        (self.kernel)(self.q16, vector)
    }

    fn score(&self, rank: i32) -> f32 {
        rank as f32 * self.inv_scale
    }

    fn penalize(&self, rank: i32, penalty: f32) -> i32 {
        rank.saturating_sub((penalty / self.inv_scale).round() as i32)
    }
}

/// `i8-sym-vec`: the exact integer sum, then the vector's scale, in `f32` — one product,
/// the same on every CPU — and the rank is that float's order.
struct Int8PerVector<'a> {
    q16: &'a [i16],
    inv_scale: f32,
    kernel: DotKernel,
}

impl Scorer for Int8PerVector<'_> {
    #[inline]
    fn rank(&self, vector: &[u8], scale: f32) -> i32 {
        ordered((self.kernel)(self.q16, vector) as f32 * scale)
    }

    fn score(&self, rank: i32) -> f32 {
        unordered(rank) * self.inv_scale
    }

    fn penalize(&self, rank: i32, penalty: f32) -> i32 {
        ordered(unordered(rank) - penalty / self.inv_scale)
    }
}

/// `rot192-i8i4`: two exact integer sums, combined in `f32` in one fixed order (see the
/// `kernel` module), and the rank is that float's order.
struct Rotated<'a> {
    head: &'a [i16],
    inv_head: f32,
    tail_lo: &'a [i16],
    tail_hi: &'a [i16],
    inv_tail: f32,
    constant: f32,
    dot: DotKernel,
    nibbles: NibbleKernel,
}

impl Scorer for Rotated<'_> {
    #[inline]
    fn rank(&self, vector: &[u8], scale: f32) -> i32 {
        let (codes, packed) = vector.split_at(self.head.len());
        let acc8 = (self.dot)(self.head, codes);
        let acc4 = (self.nibbles)(self.tail_lo, self.tail_hi, packed);
        ordered(acc8 as f32 * scale * self.inv_head + (self.constant + acc4 as f32 * self.inv_tail))
    }

    fn score(&self, rank: i32) -> f32 {
        unordered(rank)
    }

    fn penalize(&self, rank: i32, penalty: f32) -> i32 {
        ordered(unordered(rank) - penalty)
    }
}

struct Float<'a> {
    q: &'a [f32],
}

impl Scorer for Float<'_> {
    #[inline]
    fn rank(&self, vector: &[u8], _scale: f32) -> i32 {
        ordered(dot_f32(self.q, vector))
    }

    fn score(&self, rank: i32) -> f32 {
        unordered(rank)
    }

    fn penalize(&self, rank: i32, penalty: f32) -> i32 {
        ordered(unordered(rank) - penalty)
    }
}

/// A piece of work: a run of slots in one segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Piece {
    seg: u16,
    start: u32,
    end: u32,
}

/// Everything one scan visits, as runs of slots.
struct Work {
    pieces: Vec<Piece>,
    total: usize,
}

impl Work {
    /// Every slot of every segment; or, under a filter, the admitted books' primary
    /// ranges and the slots their extras and foreign records point to elsewhere.
    fn plan(set: &ScanSet<'_>, books: Option<&BookSet>) -> Self {
        let mut pieces = Vec::new();
        match books {
            None => {
                for (seg, segment) in set.segments.iter().enumerate() {
                    let slots = segment.segment.slot_count();
                    if slots > 0 {
                        pieces.push(Piece {
                            seg: seg as u16,
                            start: 0,
                            end: slots,
                        });
                    }
                }
            }
            Some(books) => {
                let admitted: Vec<Vec<bool>> = set
                    .segments
                    .iter()
                    .map(|segment| {
                        segment
                            .segment
                            .books()
                            .iter()
                            .map(|book| books.contains(&book.name))
                            .collect()
                    })
                    .collect();
                let admits = |seg: u16, slot: u32| {
                    let segment = set.segments[seg as usize].segment;
                    admitted[seg as usize][segment.book_of_slot(slot)]
                };
                let mut gathered: HashSet<(u16, u32)> = HashSet::new();
                for (seg, segment) in set.segments.iter().enumerate() {
                    let seg = seg as u16;
                    for (index, book) in segment.segment.books().iter().enumerate() {
                        if !admitted[seg as usize][index] {
                            continue;
                        }
                        if !book.slots.is_empty() {
                            pieces.push(Piece {
                                seg,
                                start: book.slots.start,
                                end: book.slots.end,
                            });
                        }
                        for extra in book.extras.clone() {
                            let (slot, _) = segment.segment.extra(extra);
                            if !admits(seg, slot) {
                                gathered.insert((seg, slot));
                            }
                        }
                        for foreign in book.foreign.clone() {
                            if let Some((target_seg, target_slot)) =
                                segment.links.get(foreign as usize).and_then(Link::target)
                            {
                                if (target_seg as usize) < set.segments.len()
                                    && !admits(target_seg, target_slot)
                                {
                                    gathered.insert((target_seg, target_slot));
                                }
                            }
                        }
                    }
                }
                let mut gathered: Vec<(u16, u32)> = gathered.into_iter().collect();
                gathered.sort_unstable();
                pieces.extend(gathered.into_iter().map(|(seg, slot)| Piece {
                    seg,
                    start: slot,
                    end: slot + 1,
                }));
            }
        }
        let total = pieces
            .iter()
            .map(|piece| (piece.end - piece.start) as usize)
            .sum();
        Self { pieces, total }
    }

    /// Cut the work into `shares` runs of equal slot counts, in order.
    fn split(&self, shares: usize) -> Vec<Vec<Piece>> {
        let mut out: Vec<Vec<Piece>> = vec![Vec::new(); shares];
        let mut pieces = self.pieces.iter().copied();
        let mut current = pieces.next();
        for (index, share) in out.iter_mut().enumerate() {
            let mut want = self.total * (index + 1) / shares - self.total * index / shares;
            while want > 0 {
                let Some(piece) = current.as_mut() else { break };
                let take = want.min((piece.end - piece.start) as usize) as u32;
                share.push(Piece {
                    seg: piece.seg,
                    start: piece.start,
                    end: piece.start + take,
                });
                piece.start += take;
                want -= take as usize;
                if piece.start == piece.end {
                    current = pieces.next();
                }
            }
        }
        out
    }
}

fn run_shares(
    set: &ScanSet<'_>,
    scorer: &impl Scorer,
    config: ScanConfig<'_, '_>,
    k: usize,
    shares: &[Vec<Piece>],
    cancel: &CancellationToken,
    progress: Option<&[AtomicUsize]>,
) -> Result<Vec<Candidate>, VectorStoreError> {
    let report = |share: usize| progress.and_then(|counters| counters.get(share));
    let results: Vec<Result<TopK, VectorStoreError>> = std::thread::scope(|scope| {
        let handles: Vec<_> = shares
            .iter()
            .enumerate()
            .skip(1)
            .map(|(index, share)| {
                let counter = report(index);
                scope.spawn(move || scan_share(set, scorer, config, k, share, cancel, counter))
            })
            .collect();
        let mut results = vec![scan_share(
            set,
            scorer,
            config,
            k,
            &shares[0],
            cancel,
            report(0),
        )];
        results.extend(
            handles
                .into_iter()
                .map(|handle| handle.join().expect("a scan thread must not panic")),
        );
        results
    });
    let mut all = Vec::with_capacity(k * shares.len());
    for result in results {
        all.extend(result?.heap.into_vec());
    }
    Ok(best_of(all, k))
}

/// The best `k` of the threads' candidates, in the scan's order, one per key.
fn best_of(mut all: Vec<Candidate>, k: usize) -> Vec<Candidate> {
    all.sort_unstable_by(Candidate::best_first);
    // One vector per key: a set holds a key live once, and if it ever held it twice the
    // better-ranked copy stands for both.
    let mut seen = HashSet::with_capacity(all.len());
    all.retain(|candidate| seen.insert(candidate.key));
    all.truncate(k);
    all
}

/// [`run_shares`] for several queries: each thread keeps a best `k` per query.
fn run_shares_many<S: Scorer>(
    set: &ScanSet<'_>,
    scorers: &[S],
    config: ScanConfig<'_, '_>,
    k: usize,
    shares: &[Vec<Piece>],
    cancel: &CancellationToken,
) -> Result<Vec<std::vec::IntoIter<BatchTop>>, VectorStoreError> {
    std::thread::scope(|scope| {
        let handles: Vec<_> = shares
            .iter()
            .skip(1)
            .map(|share| {
                scope.spawn(move || scan_share_many(set, scorers, config, k, share, cancel))
            })
            .collect();
        let mut results = vec![scan_share_many(set, scorers, config, k, &shares[0], cancel)];
        results.extend(
            handles
                .into_iter()
                .map(|handle| handle.join().expect("a scan thread must not panic")),
        );
        results
            .into_iter()
            .map(|result| result.map(Vec::into_iter))
            .collect()
    })
}

/// `found` — the scan's winners, best first — with the slots of `also` it did not return
/// merged in: each live slot holding its key, ranked as the scan ranks, the best `k` of them
/// in the scan's order, once per key. Checked for cancellation as the scan is.
fn merge_also(
    set: &ScanSet<'_>,
    scorer: &impl Scorer,
    request: &ScanRequest<'_>,
    penalties: ScanPenalties,
    mut found: Vec<Candidate>,
    also: &[SlotRef],
    cancel: &CancellationToken,
) -> Result<Vec<Candidate>, VectorStoreError> {
    let k = request.top_k;
    if also.is_empty() {
        return Ok(found);
    }
    let returned: HashSet<ChunkKey> = found.iter().map(|candidate| candidate.key).collect();
    let mut weighed = Vec::new();
    for (visited, wanted) in also.iter().enumerate() {
        if visited.is_multiple_of(SCAN_CHECK_INTERVAL) {
            cancel.scan_checkpoint(visited)?;
        }
        let Some(segment) = set.segments.get(wanted.seg as usize) else {
            continue;
        };
        if wanted.slot >= segment.segment.slot_count() || segment.is_deleted(wanted.slot) {
            continue;
        }
        // The key as the segment holds it, at its width: what the scan's hits carry.
        let key = segment.segment.key(wanted.slot);
        if !key.same_text(wanted.key) || returned.contains(&key) {
            continue;
        }
        let scale = segment.segment.vector_scale(wanted.slot).unwrap_or(1.0);
        let mut rank = scorer.rank(segment.segment.vector(wanted.slot), scale);
        let Some(window) = effective_role(set, wanted.seg, wanted.slot, request.books, penalties)
            .or_else(|| effective_role(set, wanted.seg, wanted.slot, None, penalties))
        else {
            continue;
        };
        let applied_penalty = penalty_for(window, penalties);
        if applied_penalty != 0.0 {
            rank = scorer.penalize(rank, applied_penalty);
        }
        weighed.push(Candidate {
            rank,
            key,
            seg: wanted.seg,
            slot: wanted.slot,
            window,
        });
    }
    weighed.sort_unstable_by(Candidate::best_first);
    // A key the set holds live twice, or named twice: the better-ranked copy stands, as
    // in the scan.
    let mut seen = HashSet::with_capacity(weighed.len());
    weighed.retain(|candidate| seen.insert(candidate.key));
    weighed.truncate(k);
    found.extend(weighed);
    // A total order, so the merge is the same whatever order either list came in.
    found.sort_unstable_by(Candidate::best_first);
    Ok(found)
}

fn scan_share(
    set: &ScanSet<'_>,
    scorer: &impl Scorer,
    config: ScanConfig<'_, '_>,
    k: usize,
    share: &[Piece],
    cancel: &CancellationToken,
    progress: Option<&AtomicUsize>,
) -> Result<TopK, VectorStoreError> {
    let mut top = TopK::new(k);
    let mut visited = 0usize;
    for piece in share {
        let segment = &set.segments[piece.seg as usize];
        let width = segment.segment.codec().bytes_per_vector();
        let vectors = &segment.segment.vector_bytes()
            [piece.start as usize * width..piece.end as usize * width];
        let scan = Pass {
            set,
            books: config.request.books,
            segment,
            scorer,
            piece,
            penalties: config.penalties,
            cancel,
            progress,
        };
        // One loop per kind, so the per-vector one reads its scales in step with the codes
        // and the others read nothing.
        match segment.segment.vector_scale_bytes() {
            Some(scales) => {
                let scales = scales[piece.start as usize * 4..piece.end as usize * 4]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|bytes| f32::from_le_bytes(*bytes));
                scan.run(vectors.chunks_exact(width), scales, &mut top, &mut visited)?;
            }
            None => scan.run(
                vectors.chunks_exact(width),
                std::iter::repeat(1.0),
                &mut top,
                &mut visited,
            )?,
        }
    }
    if let Some(progress) = progress {
        progress.store(visited, std::sync::atomic::Ordering::Relaxed);
    }
    Ok(top)
}

/// [`scan_share`] for several queries that reuse the vectors in cache.
fn scan_share_many<S: Scorer>(
    set: &ScanSet<'_>,
    scorers: &[S],
    config: ScanConfig<'_, '_>,
    k: usize,
    share: &[Piece],
    cancel: &CancellationToken,
) -> Result<Vec<BatchTop>, VectorStoreError> {
    cancel.check_scan_cancelled()?;
    let share_slots: usize = share
        .iter()
        .map(|piece| (piece.end - piece.start) as usize)
        .sum();
    let mut tops: Vec<_> = scorers
        .iter()
        .map(|_| BatchTop {
            top: TopK::new(k.min(share_slots)),
            cutoff: i32::MIN,
        })
        .collect();
    // Keep the interval proportional to scoring work as the group grows.
    let checkpoint_mask = SCAN_CHECK_INTERVAL
        .div_ceil(scorers.len())
        .next_power_of_two()
        - 1;
    let mut visited = 0usize;
    for piece in share {
        let segment = &set.segments[piece.seg as usize];
        let width = segment.segment.codec().bytes_per_vector();
        let vectors = &segment.segment.vector_bytes()
            [piece.start as usize * width..piece.end as usize * width];
        let pass = PassMany {
            set,
            books: config.request.books,
            segment,
            scorers,
            piece,
            penalties: config.penalties,
            cancel,
            checkpoint_mask,
        };
        match segment.segment.vector_scale_bytes() {
            Some(scales) => {
                let scales = scales[piece.start as usize * 4..piece.end as usize * 4]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|bytes| f32::from_le_bytes(*bytes));
                pass.run(vectors.chunks_exact(width), scales, &mut tops, &mut visited)?;
            }
            None => pass.run(
                vectors.chunks_exact(width),
                std::iter::repeat(1.0),
                &mut tops,
                &mut visited,
            )?,
        }
    }
    Ok(tops)
}

/// Cache the cutoff so a rejected candidate needs no heap lookup.
struct BatchTop {
    top: TopK,
    cutoff: i32,
}

/// One piece of a share, scanned for several queries.
struct PassMany<'p, 's, S> {
    set: &'p ScanSet<'s>,
    books: Option<&'p BookSet>,
    segment: &'p ScanSegment<'s>,
    scorers: &'p [S],
    piece: &'p Piece,
    penalties: ScanPenalties,
    cancel: &'p CancellationToken,
    checkpoint_mask: usize,
}

impl<S: Scorer> PassMany<'_, '_, S> {
    #[inline]
    fn run<'v>(
        &self,
        vectors: impl Iterator<Item = &'v [u8]>,
        scales: impl Iterator<Item = f32>,
        tops: &mut [BatchTop],
        visited: &mut usize,
    ) -> Result<(), VectorStoreError> {
        for (offset, (vector, scale)) in vectors.zip(scales).enumerate() {
            if *visited & self.checkpoint_mask == 0 {
                self.cancel.scan_checkpoint(*visited)?;
            }
            *visited += 1;
            let slot = self.piece.start + offset as u32;
            if self.segment.is_deleted(slot) {
                continue;
            }
            // the slot's role, resolved once, for the first query it contends for
            let mut role: Option<Option<u8>> = None;
            for (scorer, top) in self.scorers.iter().zip(tops.iter_mut()) {
                let mut rank = scorer.rank(vector, scale);
                // A nonnegative penalty cannot promote this rank past the cutoff.
                if rank < top.cutoff {
                    continue;
                }
                let window = *role.get_or_insert_with(|| {
                    effective_role(self.set, self.piece.seg, slot, self.books, self.penalties)
                });
                let Some(window) = window else {
                    break;
                };
                let applied_penalty = penalty_for(window, self.penalties);
                if applied_penalty != 0.0 {
                    rank = scorer.penalize(rank, applied_penalty);
                }
                top.top.offer(rank, self.piece.seg, slot, window, || {
                    self.segment.segment.key(slot)
                });
                if top.top.heap.len() == top.top.k {
                    top.cutoff = top.top.heap.peek().expect("the heap is full").rank;
                }
            }
        }
        Ok(())
    }
}

/// One piece of a share, scanned.
struct Pass<'p, 's, S> {
    set: &'p ScanSet<'s>,
    books: Option<&'p BookSet>,
    segment: &'p ScanSegment<'s>,
    scorer: &'p S,
    piece: &'p Piece,
    penalties: ScanPenalties,
    cancel: &'p CancellationToken,
    progress: Option<&'p AtomicUsize>,
}

impl<S: Scorer> Pass<'_, '_, S> {
    #[inline]
    fn run<'v>(
        &self,
        vectors: impl Iterator<Item = &'v [u8]>,
        scales: impl Iterator<Item = f32>,
        top: &mut TopK,
        visited: &mut usize,
    ) -> Result<(), VectorStoreError> {
        for (offset, (vector, scale)) in vectors.zip(scales).enumerate() {
            if visited.is_multiple_of(SCAN_CHECK_INTERVAL) {
                if let Some(progress) = self.progress {
                    progress.store(*visited, std::sync::atomic::Ordering::Relaxed);
                }
                self.cancel.scan_checkpoint(*visited)?;
            }
            *visited += 1;
            let slot = self.piece.start + offset as u32;
            if self.segment.is_deleted(slot) {
                continue;
            }
            let mut rank = self.scorer.rank(vector, scale);
            // A nonnegative penalty cannot promote a raw score below the heap's
            // worst. Resolve metadata only for contenders, not every scanned vector.
            if top.heap.len() == top.k && top.heap.peek().is_some_and(|worst| rank < worst.rank) {
                continue;
            }
            let Some(window) =
                effective_role(self.set, self.piece.seg, slot, self.books, self.penalties)
            else {
                continue;
            };
            let applied_penalty = penalty_for(window, self.penalties);
            if applied_penalty != 0.0 {
                rank = self.scorer.penalize(rank, applied_penalty);
            }
            top.offer(rank, self.piece.seg, slot, window, || {
                self.segment.segment.key(slot)
            });
        }
        Ok(())
    }
}

/// One vector in the running: its rank, its key and where it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Candidate {
    rank: i32,
    key: ChunkKey,
    seg: u16,
    slot: u32,
    window: u8,
}

impl Candidate {
    /// The scan's total order: higher rank first, then the smaller key, then the earlier
    /// slot — so the top `k` is one set whatever the partition into threads.
    fn best_first(a: &Self, b: &Self) -> Ordering {
        b.rank
            .cmp(&a.rank)
            .then_with(|| a.key.cmp(&b.key))
            .then_with(|| (a.seg, a.slot).cmp(&(b.seg, b.slot)))
            .then_with(|| a.window.cmp(&b.window))
    }
}

/// Ordered worst first, so a max-heap's top is the candidate to evict.
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        Candidate::best_first(self, other)
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct TopK {
    k: usize,
    heap: BinaryHeap<Candidate>,
}

impl TopK {
    fn new(k: usize) -> Self {
        Self {
            k,
            heap: BinaryHeap::with_capacity(k + 1),
        }
    }

    /// Keep the candidate if it is among the best `k` so far. The key is read only when
    /// the rank alone cannot decide.
    #[inline]
    fn offer(
        &mut self,
        rank: i32,
        seg: u16,
        slot: u32,
        window: u8,
        key: impl FnOnce() -> ChunkKey,
    ) {
        if self.heap.len() < self.k {
            self.heap.push(Candidate {
                rank,
                key: key(),
                seg,
                slot,
                window,
            });
            return;
        }
        let Some(worst) = self.heap.peek() else {
            return;
        };
        if rank < worst.rank {
            return;
        }
        let candidate = Candidate {
            rank,
            key: key(),
            seg,
            slot,
            window,
        };
        if Candidate::best_first(&candidate, worst) == Ordering::Less {
            self.heap.pop();
            self.heap.push(candidate);
        }
    }
}

/// Turn the winners into hits: each with its key, its score and its records.
fn hits(
    set: &ScanSet<'_>,
    found: Vec<Candidate>,
    request: &ScanRequest<'_>,
    penalties: ScanPenalties,
    scorer: &impl Scorer,
) -> Vec<VectorHit> {
    found
        .into_iter()
        .map(|candidate| VectorHit {
            window_penalty: request.window_penalty,
            applied_penalty: penalty_for(candidate.window, penalties),
            score: scorer.score(candidate.rank),
            key: candidate.key,
            records: records(set, candidate.seg, candidate.slot, request.books, penalties),
            window: candidate.window,
            seg: candidate.seg,
            slot: candidate.slot,
        })
        .collect()
}

fn penalty_for(window: u8, penalties: ScanPenalties) -> f32 {
    match window {
        0 => 0.0,
        ALIAS_WINDOW => penalties.alias,
        _ => penalties.window,
    }
}

pub(super) fn penalty_sort_key(window: u8, penalties: ScanPenalties) -> u32 {
    let penalty = penalty_for(window, penalties);
    if penalty == 0.0 {
        0
    } else {
        penalty.to_bits()
    }
}

/// Cheapest role among the current occurrence snapshot admitted by this query. A
/// snapshot supersedes the old vector holder's book and role; a role's penalty, not its
/// ordinal alone, determines which occurrence represents the vector in top-k.
fn effective_role(
    set: &ScanSet<'_>,
    seg: u16,
    slot: u32,
    books: Option<&BookSet>,
    penalties: ScanPenalties,
) -> Option<u8> {
    let source = set.segments[seg as usize].segment;
    // Preserve the common unfiltered newest-segment byte lookup. Only when aliases
    // are cheaper than inner windows can an uncached alias change the stored role.
    if books.is_none() && seg as usize + 1 == set.segments.len() {
        let stored = source.slot_window(slot);
        if penalties.alias >= penalties.window || stored == 0 || stored == ALIAS_WINDOW {
            return Some(stored);
        }
        let key = source.key(slot);
        return source
            .key_occurrences(key)
            .map(|rows| least_role(rows.map(|row| row.window), penalties))
            .unwrap_or(Some(stored));
    }
    let snapshots = if books.is_none() {
        &set.segments[seg as usize + 1..]
    } else {
        set.segments
    };
    let key = source.key(slot);
    for view in snapshots.iter().rev() {
        if let Some(snapshot) = view.segment.key_occurrences(key) {
            return least_role(
                snapshot
                    .filter(|record| {
                        books.is_none_or(|books| {
                            books.contains(&view.segment.books()[record.book as usize].name)
                        })
                    })
                    .map(|record| record.window),
                penalties,
            );
        }
    }
    let stored = source.slot_window(slot);
    if penalties.alias < penalties.window && stored > 0 && stored < ALIAS_WINDOW {
        return source
            .key_occurrences(key)
            .map(|rows| least_role(rows.map(|row| row.window), penalties))
            .unwrap_or(Some(stored));
    }
    Some(stored)
}

fn least_role(windows: impl Iterator<Item = u8>, penalties: ScanPenalties) -> Option<u8> {
    windows.min_by(|a, b| {
        penalty_sort_key(*a, penalties)
            .cmp(&penalty_sort_key(*b, penalties))
            .then_with(|| a.cmp(b))
    })
}

/// Every record of a slot's key: its primary record, its extras, and the foreign records
/// of later deltas that resolve to it — the books `books` admits first, at most
/// [`MAX_RECORDS_PER_HIT`] in all.
fn records(
    set: &ScanSet<'_>,
    seg: u16,
    slot: u32,
    books: Option<&BookSet>,
    penalties: ScanPenalties,
) -> Vec<RecordRef> {
    let segment = set.segments[seg as usize].segment;
    let mut records = Vec::new();
    // A snapshot is complete for its key, including removals. Consult newest first.
    for view in set.segments.iter().rev() {
        if let Some(snapshot) = view.segment.key_occurrences(segment.key(slot)) {
            return snapshot_records(view.segment, snapshot, books, penalties);
        }
    }
    let book = &segment.books()[segment.book_of_slot(slot)];
    records.push(RecordRef {
        book: book.name.clone(),
        hint: segment.hint(slot),
        window: segment.slot_window(slot),
        anchor: None,
    });
    if segment.has_extras(slot) {
        for extra in segment.extras_of_slot(slot) {
            let (_, hint) = segment.extra(extra);
            records.push(RecordRef {
                book: segment.books()[segment.book_of_extra(extra)].name.clone(),
                hint,
                window: segment.slot_window(slot),
                anchor: None,
            });
        }
    }
    let first = set
        .reverse_links
        .partition_point(|link| (link.target_seg, link.target_slot) < (seg, slot));
    for link in &set.reverse_links[first..] {
        if (link.target_seg, link.target_slot) != (seg, slot) {
            break;
        }
        let Some(from) = set.segments.get(link.from_seg as usize) else {
            continue;
        };
        if link.foreign >= from.segment.foreign_count() {
            continue;
        }
        let (_, book, hint) = from.segment.foreign_record(link.foreign);
        if let Some(book) = from.segment.books().get(book as usize) {
            records.push(RecordRef {
                book: book.name.clone(),
                hint,
                window: segment.slot_window(slot),
                anchor: None,
            });
        }
    }
    finish_records(&mut records, books, penalties);
    records
}

/// Select fair representatives while streaming a key's possibly enormous snapshot.
/// At most 32 parents of the current book and 32 records per filter class are held.
fn snapshot_records(
    segment: &Segment,
    snapshot: impl Iterator<Item = super::occurrence::Occurrence>,
    books: Option<&BookSet>,
    penalties: ScanPenalties,
) -> Vec<RecordRef> {
    use super::occurrence::Occurrence;
    type Ranked = ((usize, u32, u8, u32), RecordRef);
    fn flush(
        segment: &Segment,
        group: &mut Vec<Occurrence>,
        books: Option<&BookSet>,
        penalties: ScanPenalties,
        admitted: &mut Vec<Ranked>,
        other: &mut Vec<Ranked>,
    ) {
        let Some(first) = group.first() else {
            return;
        };
        let book = &segment.books()[first.book as usize];
        let selected = if books.is_none_or(|books| books.contains(&book.name)) {
            admitted
        } else {
            other
        };
        for (round, row) in group.drain(..).enumerate() {
            let rank = (
                round,
                penalty_sort_key(row.window, penalties),
                row.window,
                row.book,
            );
            let at = selected.partition_point(|(other, _)| *other < rank);
            if at < MAX_RECORDS_PER_HIT {
                selected.insert(
                    at,
                    (
                        rank,
                        RecordRef {
                            book: book.name.clone(),
                            hint: row.hint,
                            window: row.window,
                            anchor: Some(row.anchor),
                        },
                    ),
                );
                selected.truncate(MAX_RECORDS_PER_HIT);
            }
        }
    }
    let mut admitted = Vec::new();
    let mut other = Vec::new();
    let mut group = Vec::new();
    let mut current = None;
    for row in snapshot {
        if current != Some(row.book) {
            flush(
                segment,
                &mut group,
                books,
                penalties,
                &mut admitted,
                &mut other,
            );
            current = Some(row.book);
        }
        // The least-penalized parent represents a book first; stable hints break ties.
        let row_rank = (penalty_sort_key(row.window, penalties), row.window);
        let at = group.partition_point(|other: &Occurrence| {
            (penalty_sort_key(other.window, penalties), other.window) <= row_rank
        });
        if at < MAX_RECORDS_PER_HIT {
            group.insert(at, row);
            group.truncate(MAX_RECORDS_PER_HIT);
        }
    }
    flush(
        segment,
        &mut group,
        books,
        penalties,
        &mut admitted,
        &mut other,
    );
    admitted
        .into_iter()
        .chain(other)
        .take(MAX_RECORDS_PER_HIT)
        .map(|(_, record)| record)
        .collect()
}

/// A frequent parent must not crowd every other book out of a shared vector's cap.
fn finish_records(records: &mut Vec<RecordRef>, books: Option<&BookSet>, penalties: ScanPenalties) {
    // Prefer the least-penalized representative of each book.
    records.sort_by_key(|record| (penalty_sort_key(record.window, penalties), record.window));
    let mut rounds = std::collections::HashMap::new();
    let mut ranked: Vec<_> = records
        .drain(..)
        .map(|record| {
            let round = rounds.entry(record.book.clone()).or_insert(0usize);
            let rank = (
                books.is_some_and(|books| !books.contains(&record.book)),
                *round,
            );
            *round += 1;
            (rank, record)
        })
        .collect();
    ranked.sort_by_key(|(rank, _)| *rank);
    records.extend(
        ranked
            .into_iter()
            .take(MAX_RECORDS_PER_HIT)
            .map(|(_, record)| record),
    );
}
