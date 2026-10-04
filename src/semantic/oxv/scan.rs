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
use crate::semantic::oxv::kernel::{dot_f32, ordered, select, unordered, DotKernel, PreparedQuery};
use crate::semantic::oxv::reader::Segment;
use crate::semantic::resolve::{BookSet, RecordRef, SlotRef, VectorHit, MAX_RECORDS_PER_HIT};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::sync::atomic::AtomicUsize;

/// Below this many slots a share is not worth a thread of its own.
const MIN_SLOTS_PER_THREAD: usize = 4096;

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
pub(crate) fn scan_with(
    set: &ScanSet<'_>,
    query: &PreparedQuery,
    request: &ScanRequest<'_>,
    also: &[SlotRef],
    cancel: &CancellationToken,
) -> Result<Vec<VectorHit>, VectorStoreError> {
    scan_all(set, query, request, also, cancel, None)
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
    scan_all(set, query, request, &[], cancel, progress)
}

fn scan_all(
    set: &ScanSet<'_>,
    query: &PreparedQuery,
    request: &ScanRequest<'_>,
    also: &[SlotRef],
    cancel: &CancellationToken,
    progress: Option<&[AtomicUsize]>,
) -> Result<Vec<VectorHit>, VectorStoreError> {
    cancel.scan_checkpoint(0)?;
    if request.top_k == 0 {
        return Ok(Vec::new());
    }
    for segment in set.segments {
        if segment.segment.codec().dim() != query.dim() {
            return Err(VectorStoreError::DimensionMismatch {
                store_dim: segment.segment.codec().dim() as u32,
                vector_dim: query.dim() as u32,
            });
        }
    }
    let work = Work::plan(set, request.books);
    // No more candidates than there are slots to visit, so a `top_k` sized for "all of
    // them" allocates for what exists.
    let top_k = request.top_k.min(work.total.max(1));
    let threads = request
        .threads
        .max(1)
        .min(work.total.div_ceil(MIN_SLOTS_PER_THREAD).max(1));
    let shares = work.split(threads);

    match query {
        PreparedQuery::Int8 { q16, inv_scale } => {
            let (kernel, _) = select();
            let scorer = Int8 {
                q16,
                inv_scale: *inv_scale,
                kernel,
            };
            let found = run_shares(set, &scorer, top_k, &shares, cancel, progress)?;
            let found = merge_also(set, &scorer, found, also, request.top_k, cancel)?;
            Ok(hits(set, found, request, &scorer))
        }
        PreparedQuery::Int8PerVector { q16, inv_scale } => {
            let (kernel, _) = select();
            let scorer = Int8PerVector {
                q16,
                inv_scale: *inv_scale,
                kernel,
            };
            let found = run_shares(set, &scorer, top_k, &shares, cancel, progress)?;
            let found = merge_also(set, &scorer, found, also, request.top_k, cancel)?;
            Ok(hits(set, found, request, &scorer))
        }
        PreparedQuery::Float { q } => {
            let scorer = Float { q };
            let found = run_shares(set, &scorer, top_k, &shares, cancel, progress)?;
            let found = merge_also(set, &scorer, found, also, request.top_k, cancel)?;
            Ok(hits(set, found, request, &scorer))
        }
    }
}

/// How one codec turns a stored vector — and its own scale, for a codec with one — into a
/// rank, and a rank back into a score.
trait Scorer: Sync {
    fn rank(&self, vector: &[u8], scale: f32) -> i32;
    fn score(&self, rank: i32) -> f32;
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
                scope.spawn(move || scan_share(set, scorer, k, share, cancel, counter))
            })
            .collect();
        let mut results = vec![scan_share(set, scorer, k, &shares[0], cancel, report(0))];
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
    all.sort_unstable_by(Candidate::best_first);
    // One vector per key: a set holds a key live once, and if it ever held it twice the
    // better-ranked copy stands for both.
    let mut seen = HashSet::with_capacity(all.len());
    all.retain(|candidate| seen.insert(candidate.key));
    all.truncate(k);
    Ok(all)
}

/// `found` — the scan's winners, best first — with the slots of `also` it did not return
/// merged in: each live slot holding its key, ranked as the scan ranks, the best `k` of them
/// in the scan's order, once per key. Checked for cancellation as the scan is.
fn merge_also(
    set: &ScanSet<'_>,
    scorer: &impl Scorer,
    mut found: Vec<Candidate>,
    also: &[SlotRef],
    k: usize,
    cancel: &CancellationToken,
) -> Result<Vec<Candidate>, VectorStoreError> {
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
        if wanted.slot >= segment.segment.slot_count()
            || segment.is_deleted(wanted.slot)
            || returned.contains(&wanted.key)
            || segment.segment.key(wanted.slot) != wanted.key
        {
            continue;
        }
        let scale = segment.segment.vector_scale(wanted.slot).unwrap_or(1.0);
        weighed.push(Candidate {
            rank: scorer.rank(segment.segment.vector(wanted.slot), scale),
            key: wanted.key,
            seg: wanted.seg,
            slot: wanted.slot,
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
            segment,
            scorer,
            piece,
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

/// One piece of a share, scanned.
struct Pass<'p, 's, S> {
    segment: &'p ScanSegment<'s>,
    scorer: &'p S,
    piece: &'p Piece,
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
            top.offer(
                self.scorer.rank(vector, scale),
                self.piece.seg,
                slot,
                || self.segment.segment.key(slot),
            );
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
}

impl Candidate {
    /// The scan's total order: higher rank first, then the smaller key, then the earlier
    /// slot — so the top `k` is one set whatever the partition into threads.
    fn best_first(a: &Self, b: &Self) -> Ordering {
        b.rank
            .cmp(&a.rank)
            .then_with(|| a.key.cmp(&b.key))
            .then_with(|| (a.seg, a.slot).cmp(&(b.seg, b.slot)))
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
    fn offer(&mut self, rank: i32, seg: u16, slot: u32, key: impl FnOnce() -> ChunkKey) {
        if self.heap.len() < self.k {
            self.heap.push(Candidate {
                rank,
                key: key(),
                seg,
                slot,
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
    scorer: &impl Scorer,
) -> Vec<VectorHit> {
    found
        .into_iter()
        .map(|candidate| VectorHit {
            score: scorer.score(candidate.rank),
            key: candidate.key,
            records: records(set, candidate.seg, candidate.slot, request.books),
            seg: candidate.seg,
            slot: candidate.slot,
        })
        .collect()
}

/// Every record of a slot's key: its primary record, its extras, and the foreign records
/// of later deltas that resolve to it — the books `books` admits first, at most
/// [`MAX_RECORDS_PER_HIT`] in all.
fn records(set: &ScanSet<'_>, seg: u16, slot: u32, books: Option<&BookSet>) -> Vec<RecordRef> {
    let segment = set.segments[seg as usize].segment;
    let mut records = Vec::new();
    let book = &segment.books()[segment.book_of_slot(slot)];
    records.push(RecordRef {
        book: book.name.clone(),
        hint: segment.hint(slot),
    });
    if segment.has_extras(slot) {
        for extra in segment.extras_of_slot(slot) {
            let (_, hint) = segment.extra(extra);
            records.push(RecordRef {
                book: segment.books()[segment.book_of_extra(extra)].name.clone(),
                hint,
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
            });
        }
    }
    if let Some(books) = books {
        // Stable, so each half keeps the order above.
        records.sort_by_key(|record| !books.contains(&record.book));
    }
    records.truncate(MAX_RECORDS_PER_HIT);
    records
}
