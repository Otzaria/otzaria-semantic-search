//! Score requested slots with the same kernels and occurrence roles as an unfiltered scan.

use super::{
    check_window_penalty, effective_role, penalty_for, QueryScorer, ScanPenalties, ScanSet, Scorer,
};
use crate::cancellation::{CancellationToken, SCAN_CHECK_INTERVAL};
use crate::errors::VectorStoreError;
use crate::semantic::chunk_key::ChunkKey;
use crate::semantic::oxv::{codec::Codec, kernel::PreparedQuery};
use crate::semantic::resolve::SlotRef;
use std::cmp::Reverse;
use std::collections::{hash_map::Entry, HashMap, HashSet};
use std::ops::Range;
use std::sync::{Arc, Mutex};

type Position = (u16, u32);
type Copies = HashMap<ChunkKey, Vec<Position>>;

/// A generation can contain multiple live slots for one key. Build this index on demand,
/// once per open generation, and retain only keys with more than one live copy.
#[derive(Default)]
pub(crate) struct DuplicateSlots {
    copies: Mutex<Option<Arc<Copies>>>,
}

impl DuplicateSlots {
    fn get(
        &self,
        set: &ScanSet<'_>,
        cancel: &CancellationToken,
    ) -> Result<Arc<Copies>, VectorStoreError> {
        let mut cached = self
            .copies
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        cancel.check_scan_cancelled()?;
        if let Some(copies) = cached.as_ref() {
            return Ok(Arc::clone(copies));
        }
        let copies = Arc::new(find_duplicates(set, cancel)?);
        *cached = Some(Arc::clone(&copies));
        Ok(copies)
    }
}

fn live_keys<'a>(set: &'a ScanSet<'_>) -> impl Iterator<Item = (Position, ChunkKey)> + 'a {
    set.segments.iter().enumerate().flat_map(|(seg, view)| {
        view.segment
            .keys()
            .enumerate()
            .filter_map(move |(slot, key)| {
                let slot = slot as u32;
                (!view.is_deleted(slot)).then_some(((seg as u16, slot), key))
            })
    })
}

fn find_duplicates(
    set: &ScanSet<'_>,
    cancel: &CancellationToken,
) -> Result<Copies, VectorStoreError> {
    // The prefix is enough to rule out duplicates, using eight scratch bytes per slot.
    // Only repeated prefixes need a second pass and comparison at the full stored width.
    let capacity = set
        .segments
        .iter()
        .map(|view| view.segment.slot_count() as usize)
        .sum();
    let mut prefixes = Vec::with_capacity(capacity);
    for (visited, (_, key)) in live_keys(set).enumerate() {
        if visited.is_multiple_of(SCAN_CHECK_INTERVAL) {
            cancel.check_scan_cancelled()?;
        }
        prefixes.push(key.column_value());
    }
    prefixes.sort_unstable();
    cancel.check_scan_cancelled()?;
    let repeated: HashSet<u64> = prefixes
        .windows(2)
        .filter_map(|pair| (pair[0] == pair[1]).then_some(pair[0]))
        .collect();
    drop(prefixes);

    let mut copies = HashMap::<ChunkKey, Vec<Position>>::new();
    if !repeated.is_empty() {
        for (visited, (position, key)) in live_keys(set).enumerate() {
            if visited.is_multiple_of(SCAN_CHECK_INTERVAL) {
                cancel.check_scan_cancelled()?;
            }
            if repeated.contains(&key.column_value()) {
                copies.entry(key).or_default().push(position);
            }
        }
        copies.retain(|_, positions| positions.len() > 1);
    }
    cancel.check_scan_cancelled()?;
    Ok(copies)
}

struct ScoredSlot<'a> {
    position: Position,
    vector: &'a [u8],
    scale: f32,
    penalty: f32,
}

/// Resolve the requested positions and their competing copies once for the query batch.
struct ScorePlan<'a> {
    slots: Vec<Option<(usize, Position)>>,
    groups: Vec<Range<usize>>,
    candidates: Vec<ScoredSlot<'a>>,
}

impl<'a> ScorePlan<'a> {
    fn new(
        set: &ScanSet<'a>,
        duplicate_slots: &DuplicateSlots,
        slots: &[SlotRef],
        penalties: ScanPenalties,
        cancel: &CancellationToken,
    ) -> Result<Self, VectorStoreError> {
        let mut plan = Self {
            slots: Vec::with_capacity(slots.len()),
            groups: Vec::new(),
            candidates: Vec::new(),
        };
        let mut group_by_key = HashMap::new();
        let mut duplicates = None;
        for (visited, wanted) in slots.iter().enumerate() {
            if visited.is_multiple_of(SCAN_CHECK_INTERVAL) {
                cancel.check_scan_cancelled()?;
            }
            let Some(key) = slot_key(set, wanted) else {
                plan.slots.push(None);
                continue;
            };
            let position = (wanted.seg, wanted.slot);
            let group = match group_by_key.entry(key) {
                Entry::Occupied(entry) => *entry.get(),
                Entry::Vacant(entry) => {
                    let copies = match &duplicates {
                        Some(copies) => copies,
                        None => duplicates.insert(duplicate_slots.get(set, cancel)?),
                    };
                    let positions = copies
                        .get(&key)
                        .map_or(std::slice::from_ref(&position), Vec::as_slice);
                    let start = plan.candidates.len();
                    for (visited, &(seg, slot)) in positions.iter().enumerate() {
                        if visited != 0 && visited.is_multiple_of(SCAN_CHECK_INTERVAL) {
                            cancel.check_scan_cancelled()?;
                        }
                        let Some(window) = effective_role(set, seg, slot, None, penalties) else {
                            continue;
                        };
                        let segment = set.segments[seg as usize].segment;
                        plan.candidates.push(ScoredSlot {
                            position: (seg, slot),
                            vector: segment.vector(slot),
                            scale: segment.vector_scale(slot).unwrap_or(1.0),
                            penalty: penalty_for(window, penalties),
                        });
                    }
                    let group = plan.groups.len();
                    plan.groups.push(start..plan.candidates.len());
                    *entry.insert(group)
                }
            };
            plan.slots.push(Some((group, position)));
        }
        Ok(plan)
    }

    fn score(
        &self,
        scorer: &impl Scorer,
        cancel: &CancellationToken,
    ) -> Result<Vec<Option<f32>>, VectorStoreError> {
        let mut winners = Vec::with_capacity(self.groups.len());
        for group in &self.groups {
            let mut best = None;
            for index in group.clone() {
                if index.is_multiple_of(SCAN_CHECK_INTERVAL) {
                    cancel.scan_checkpoint(index)?;
                }
                let slot = &self.candidates[index];
                let mut rank = scorer.rank(slot.vector, slot.scale);
                if slot.penalty != 0.0 {
                    rank = scorer.penalize(rank, slot.penalty);
                }
                // The key is shared, so the scan's tie-break is the earlier position.
                let candidate = (rank, Reverse(slot.position));
                if best.is_none_or(|best| candidate > best) {
                    best = Some(candidate);
                }
            }
            winners.push(best.map(|(rank, Reverse(position))| (position, scorer.score(rank))));
        }
        let mut row = Vec::with_capacity(self.slots.len());
        for (index, slot) in self.slots.iter().enumerate() {
            if index.is_multiple_of(SCAN_CHECK_INTERVAL) {
                cancel.check_scan_cancelled()?;
            }
            row.push(slot.and_then(|(group, position)| {
                winners[group]
                    .filter(|(winner, _)| *winner == position)
                    .map(|(_, score)| score)
            }));
        }
        Ok(row)
    }
}

fn slot_key(set: &ScanSet<'_>, slot: &SlotRef) -> Option<ChunkKey> {
    let segment = set.segments.get(slot.seg as usize)?;
    if slot.slot >= segment.segment.slot_count() || segment.is_deleted(slot.slot) {
        return None;
    }
    let key = segment.segment.key(slot.slot);
    key.same_text(slot.key).then_some(key)
}

pub(crate) fn score_slots_many(
    set: &ScanSet<'_>,
    codec: &Codec,
    duplicate_slots: &DuplicateSlots,
    queries: &[&[f32]],
    slots: &[SlotRef],
    window_penalty: f32,
    cancel: &CancellationToken,
) -> Result<Vec<Vec<Option<f32>>>, VectorStoreError> {
    cancel.check_scan_cancelled()?;
    let mut queries = queries.iter().map(|query| {
        cancel.check_scan_cancelled()?;
        PreparedQuery::new(query, codec)
    });
    let Some(first) = queries.next().transpose()? else {
        return Ok(Vec::new());
    };
    check_window_penalty(window_penalty)?;
    let penalties = ScanPenalties {
        window: window_penalty,
        alias: window_penalty,
    };
    let plan = ScorePlan::new(set, duplicate_slots, slots, penalties, cancel)?;
    std::iter::once(Ok(first))
        .chain(queries)
        .map(|query| match QueryScorer::new(&query?) {
            QueryScorer::Int8(scorer) => plan.score(&scorer, cancel),
            QueryScorer::Int8PerVector(scorer) => plan.score(&scorer, cancel),
            QueryScorer::Rotated(scorer) => plan.score(&scorer, cancel),
            QueryScorer::Float(scorer) => plan.score(&scorer, cancel),
        })
        .collect()
}

/// Score physical slots already selected by a scan, without looking up other copies.
pub(crate) fn score_selected_slots_many(
    set: &ScanSet<'_>,
    codec: &Codec,
    queries: &[&[f32]],
    slots: &[SlotRef],
    cancel: &CancellationToken,
) -> Result<Vec<Vec<Option<f32>>>, VectorStoreError> {
    cancel.check_scan_cancelled()?;
    if queries.is_empty() {
        return Ok(Vec::new());
    }
    let mut vectors = Vec::with_capacity(slots.len());
    for (index, slot) in slots.iter().enumerate() {
        if index.is_multiple_of(SCAN_CHECK_INTERVAL) {
            cancel.check_scan_cancelled()?;
        }
        vectors.push(slot_key(set, slot).map(|_| {
            let segment = set.segments[slot.seg as usize].segment;
            (
                segment.vector(slot.slot),
                segment.vector_scale(slot.slot).unwrap_or(1.0),
            )
        }));
    }
    let mut rows = Vec::with_capacity(queries.len());
    for query in queries {
        cancel.check_scan_cancelled()?;
        let query = PreparedQuery::new(query, codec)?;
        rows.push(match QueryScorer::new(&query) {
            QueryScorer::Int8(scorer) => score_selected(&scorer, &vectors, cancel)?,
            QueryScorer::Int8PerVector(scorer) => score_selected(&scorer, &vectors, cancel)?,
            QueryScorer::Rotated(scorer) => score_selected(&scorer, &vectors, cancel)?,
            QueryScorer::Float(scorer) => score_selected(&scorer, &vectors, cancel)?,
        });
    }
    Ok(rows)
}

fn score_selected(
    scorer: &impl Scorer,
    vectors: &[Option<(&[u8], f32)>],
    cancel: &CancellationToken,
) -> Result<Vec<Option<f32>>, VectorStoreError> {
    let mut row = Vec::with_capacity(vectors.len());
    for (index, vector) in vectors.iter().enumerate() {
        if index.is_multiple_of(SCAN_CHECK_INTERVAL) {
            cancel.scan_checkpoint(index)?;
        }
        row.push(vector.map(|(vector, scale)| scorer.score(scorer.rank(vector, scale))));
    }
    Ok(row)
}
