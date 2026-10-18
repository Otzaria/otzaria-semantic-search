//! Short lines that share a neighbour's vector.
//!
//! A line shorter than the recipe's `min_meaningful_chars` is embedded with the context of
//! its neighbours, so its vector is often the neighbour's own almost exactly. In a
//! format-3 release such a line ships no vector of its own: an **alias** occurrence names
//! the neighbour's key, at the short line's place, with the short line's own key as its
//! anchor ([`ALIAS_WINDOW`]), and a search shows the short line wherever the neighbour's
//! vector is found.
//!
//! **The decision**, made at assembly from the plan and the warehouse alone, so the same
//! inputs give the same aliases in every release:
//!
//! * a source is a record the plan marks [`short`](PlanRecord::short) — always a first
//!   window;
//! * its candidates are the records of the first windows at ordinals −1, +1, −2, +2 of the
//!   same book that are not short, in that order;
//! * the cosine of the two warehouse vectors, summed in `f64` in component order, picks the
//!   best candidate — the first in that order on a tie — and it must be at least the
//!   threshold, an `f32` widened as it is (0.99 is 0.99000000953…);
//! * a key is aliased only if every record of it, in the whole plan, found a target. A
//!   target is never short, so it is never aliased itself: no chains.

use crate::distribution::files::{malformed, KeyMap};
use crate::distribution::plan::{PlanRecord, PlanRecords};
use crate::distribution::warehouse::Warehouse;
use crate::errors::PackError;
use crate::semantic::chunk_key::ChunkKey;
use crate::semantic::oxv::occurrence::{Occurrence, ALIAS_WINDOW};
use std::collections::HashMap;

/// The cosine a short line's vector needs with its neighbour's to share it.
pub const DEFAULT_ALIAS_THRESHOLD: f32 = 0.99;

/// The neighbours a short line may share, in the order they are tried.
const NEIGHBOURS: [i64; 4] = [-1, 1, -2, 2];

/// Which records of a plan are aliases, and of what.
#[derive(Debug, Default)]
pub struct Aliases {
    /// By record index: the target's key, and the cosine.
    targets: HashMap<u64, ([u8; 16], f64)>,
    /// Distinct keys aliased.
    pub keys: u64,
}

impl Aliases {
    /// No alias at all: a plan without short records, or a threshold of 0.
    pub fn none() -> Self {
        Self::default()
    }

    /// Records aliased.
    pub fn len(&self) -> u64 {
        self.targets.len() as u64
    }

    pub fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }

    /// Whether record `index` of the plan is an alias.
    pub fn is_aliased(&self, index: u64) -> bool {
        self.targets.contains_key(&index)
    }

    /// The key whose vector record `index` shares.
    pub fn target(&self, index: u64) -> Option<ChunkKey> {
        self.targets.get(&index).map(|(key, _)| ChunkKey(*key))
    }

    /// The least cosine of an alias to its target.
    pub fn min_cosine(&self) -> Option<f64> {
        self.targets
            .values()
            .map(|(_, cosine)| *cosine)
            .min_by(f64::total_cmp)
    }

    /// The plan's occurrences as a segment stores them: an aliased record's under its
    /// target's key, at its own place, its own key the anchor, window [`ALIAS_WINDOW`].
    pub fn occurrences<'a>(
        &'a self,
        records: &'a PlanRecords,
    ) -> impl Iterator<Item = Occurrence> + 'a {
        records
            .occurrences()
            .enumerate()
            .map(move |(index, occurrence)| match self.target(index as u64) {
                Some(target) => Occurrence {
                    key: target,
                    anchor: occurrence.key,
                    window: ALIAS_WINDOW,
                    ..occurrence
                },
                None => occurrence,
            })
    }
}

/// Decide the aliases of `records` at `threshold`, in `(0, 1]`, on the warehouse's vectors;
/// `None` or 0 decides none.
pub fn decide(
    records: &PlanRecords,
    warehouse: &Warehouse,
    threshold: Option<f32>,
) -> Result<Aliases, PackError> {
    decide_with(records, threshold, warehouse.dim(), |sha256, out| {
        warehouse
            .find(sha256)
            .map(|record| warehouse.vector(record, out))
            .is_some()
    })
}

/// [`decide`] on the vectors `vector` writes for a text's SHA-256, returning `false` for
/// one it lacks.
pub(crate) fn decide_with(
    records: &PlanRecords,
    threshold: Option<f32>,
    dim: usize,
    mut vector: impl FnMut(&[u8; 32], &mut [f32]) -> bool,
) -> Result<Aliases, PackError> {
    let Some(threshold) = threshold.filter(|threshold| *threshold > 0.0) else {
        return Ok(Aliases::none());
    };
    if !(threshold.is_finite() && threshold <= 1.0) {
        return Err(malformed(format!(
            "the alias threshold is {threshold}, and it must be in (0, 1]"
        )));
    }
    if !records.is_windowed() {
        return Ok(Aliases::none());
    }
    let (mut a, mut b) = (vec![0f32; dim], vec![0f32; dim]);
    let mut cosine = |one: &[u8; 32], other: &[u8; 32]| -> Option<f64> {
        if !(vector(one, &mut a) && vector(other, &mut b)) {
            return None;
        }
        let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
        for (x, y) in a.iter().zip(&b) {
            let (x, y) = (f64::from(*x), f64::from(*y));
            ab += x * y;
            aa += x * x;
            bb += y * y;
        }
        (aa > 0.0 && bb > 0.0).then(|| ab / (aa.sqrt() * bb.sqrt()))
    };

    // Every short record's best target, book by book.
    let mut found: HashMap<u64, ([u8; 16], f64)> = HashMap::new();
    let mut sources: KeyMap<[u8; 16], bool> = KeyMap::default();
    let mut book: Vec<(u64, PlanRecord)> = Vec::new();
    let mut decide_book = |book: &[(u64, PlanRecord)]| {
        // First windows by ordinal; a book's records ascend by ordinal.
        let first: Vec<&(u64, PlanRecord)> = book
            .iter()
            .filter(|(_, record)| record.window == 0)
            .collect();
        for (index, record) in first.iter().filter(|(_, record)| record.short) {
            let mut best: Option<([u8; 16], f64)> = None;
            for offset in NEIGHBOURS {
                let Some(ordinal) = u32::try_from(i64::from(record.ordinal) + offset).ok() else {
                    continue;
                };
                let Ok(at) = first.binary_search_by_key(&ordinal, |(_, other)| other.ordinal)
                else {
                    continue;
                };
                let neighbour = &first[at].1;
                if neighbour.short {
                    continue;
                }
                if let Some(value) = cosine(&record.sha256, &neighbour.sha256) {
                    if value >= f64::from(threshold) && best.is_none_or(|(_, top)| value > top) {
                        best = Some((neighbour.key_bytes(), value));
                    }
                }
            }
            if let Some(best) = best {
                found.insert(*index, best);
                sources.insert(record.key_bytes(), true);
            }
        }
    };
    let mut current = None;
    for (index, record) in records.iter().enumerate() {
        if current != Some(record.book) {
            decide_book(&book);
            book.clear();
            current = Some(record.book);
        }
        book.push((index as u64, record));
    }
    decide_book(&book);

    // A key is aliased only where every one of its records is.
    for (index, record) in records.iter().enumerate() {
        if let Some(whole) = sources.get_mut(&record.key_bytes()) {
            if !found.contains_key(&(index as u64)) {
                *whole = false;
            }
        }
    }
    found.retain(|index, _| {
        sources
            .get(&records.get(*index).key_bytes())
            .copied()
            .unwrap_or(false)
    });
    let keys = sources.values().filter(|whole| **whole).count() as u64;
    Ok(Aliases {
        targets: found,
        keys,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::plan::RecordsWriter;
    use crate::distribution::testing::TempDir;

    /// `(book, ordinal, short, text id, vector)` per first window, written as a plan.
    type Line = (u32, u32, bool, u8, [f32; 4]);

    fn decide_on(lines: &[Line], threshold: Option<f32>) -> (Aliases, PlanRecords, TempDir) {
        let dir = TempDir::new("alias");
        let path = dir.join("records.bin");
        let mut writer = RecordsWriter::create_windowed(&path).unwrap();
        let mut vectors = HashMap::new();
        for (book, ordinal, short, text, vector) in lines {
            let sha256 = [*text; 32];
            vectors.insert(sha256, *vector);
            writer
                .push(PlanRecord {
                    book: *book,
                    ordinal: *ordinal,
                    window: 0,
                    short: *short,
                    sha256,
                })
                .unwrap();
        }
        writer.finish().unwrap();
        let records = PlanRecords::open(&path).unwrap();
        let aliases = decide_with(&records, threshold, 4, |sha256, out| {
            vectors
                .get(sha256)
                .map(|vector| out.copy_from_slice(vector))
                .is_some()
        })
        .unwrap();
        (aliases, records, dir)
    }

    fn targets(aliases: &Aliases, records: &PlanRecords) -> Vec<(u64, u8)> {
        (0..records.len())
            .filter_map(|index| aliases.target(index).map(|key| (index, key.0[0])))
            .collect()
    }

    /// A short line takes its neighbour's vector only within its book, only from a line that
    /// is not short, and only past the threshold; the same plan decides the same way twice.
    #[test]
    fn a_short_line_shares_a_close_neighbour_in_its_own_book_and_nothing_chains() {
        let lines: [Line; 8] = [
            (0, 0, false, 1, [1.0, 0.0, 0.0, 0.0]),
            (0, 1, true, 2, [1.0, 0.01, 0.0, 0.0]),
            (0, 2, false, 3, [0.0, 1.0, 0.0, 0.0]),
            // Close only to the short line after it, which is no target.
            (0, 3, true, 4, [0.0, 0.0, 1.0, 0.0]),
            (0, 4, true, 5, [0.0, 0.0, 1.0, 0.001]),
            // Book 1's first line is book 0's first line's twin, and out of reach.
            (1, 0, true, 6, [1.0, 0.0, 0.0, 0.0]),
            (1, 1, false, 7, [0.0, 0.0, 0.0, 1.0]),
            (1, 2, false, 8, [0.0, 1.0, 0.0, 0.0]),
        ];
        let (aliases, records, _dir) = decide_on(&lines, Some(DEFAULT_ALIAS_THRESHOLD));
        assert_eq!(targets(&aliases, &records), [(1, 1)]);
        assert_eq!((aliases.len(), aliases.keys), (1, 1));
        assert!(aliases.min_cosine().unwrap() > 0.9999);
        let (again, _, _again) = decide_on(&lines, Some(DEFAULT_ALIAS_THRESHOLD));
        assert_eq!(targets(&again, &records), targets(&aliases, &records));

        // Its occurrence: the target's key, its own place, its own key the anchor.
        let rows: Vec<Occurrence> = aliases.occurrences(&records).collect();
        assert_eq!(rows[1].key, records.get(0).key());
        assert_eq!(rows[1].anchor, records.get(1).key());
        assert_eq!((rows[1].hint, rows[1].window), (1, ALIAS_WINDOW));
        assert_eq!(rows[3].key, records.get(3).key());

        for none in [None, Some(0.0)] {
            assert!(decide_on(&lines, none).0.is_empty());
        }
    }

    /// The threshold is inclusive, the best neighbour wins, and on a tie the one tried first
    /// — the previous line, then the next, then two away.
    #[test]
    fn the_threshold_is_inclusive_and_a_tie_goes_to_the_line_before() {
        let lines: [Line; 4] = [
            (0, 0, false, 1, [1.0, 1.0, 1.0, 1.0]),
            (0, 1, true, 2, [1.0, 0.0, 0.0, 0.0]),
            (0, 2, false, 3, [1.0, 1.0, 1.0, 1.0]),
            (0, 3, false, 4, [1.0, 1.0, 1.0, 1.0]),
        ];
        // The cosine is exactly 0.5 with each.
        let (aliases, records, _dir) = decide_on(&lines, Some(0.5));
        assert_eq!(targets(&aliases, &records), [(1, 1)]);
        assert!(decide_on(&lines, Some(0.500_001)).0.is_empty());

        let mut better = lines;
        better[2].4 = [1.0, 0.1, 0.0, 0.0];
        let (aliases, records, _dir) = decide_on(&better, Some(0.5));
        assert_eq!(targets(&aliases, &records), [(1, 3)]);

        let error = decide_with(&records, Some(1.5), 4, |_, _| true).unwrap_err();
        assert!(error.to_string().contains("(0, 1]"), "{error}");
    }

    /// A key is aliased only where every record of it finds a target.
    #[test]
    fn a_key_is_aliased_only_where_every_record_of_it_is() {
        let lines: [Line; 4] = [
            (0, 0, false, 1, [1.0, 0.0, 0.0, 0.0]),
            (0, 1, true, 2, [1.0, 0.0, 0.0, 0.0]),
            // The same text in a book where nothing is close to it.
            (1, 0, true, 2, [1.0, 0.0, 0.0, 0.0]),
            (1, 1, false, 3, [0.0, 1.0, 0.0, 0.0]),
        ];
        assert!(decide_on(&lines, Some(0.99)).0.is_empty());
        let mut close = lines;
        close[3].4 = [1.0, 0.0, 0.0, 0.0];
        let (aliases, records, _dir) = decide_on(&close, Some(0.99));
        assert_eq!(targets(&aliases, &records), [(1, 1), (2, 3)]);
        assert_eq!(aliases.keys, 1);
    }
}
