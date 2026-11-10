//! Untimed native lifecycle validation. OXV remains the independent content oracle.
use crate::Result;
use crate::data::key_id;
use otzaria_semantic_search::semantic::segment_set::SegmentSet;
use std::collections::{HashMap, HashSet};
use usearch::ffi::Matches;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

fn check_metadata(index: &Index, dimensions: usize) -> Result<()> {
    if index.dimensions() != dimensions
        || index.scalar_kind() != ScalarKind::F32
        || index.metric_kind() != MetricKind::IP
        || index.multi()
    {
        return Err(
            "ANN no longer has the expected dimensions, F32/IP metric or unique keys".into(),
        );
    }
    Ok(())
}

/// Check every live key and its vector, including keys never returned by a query.
/// The retained OXV generation supplies the expected bytes; regenerating/re-encoding
/// the fixture would add work and would not validate against the installed source.
pub fn verify_contents(index: &Index, source: &SegmentSet) -> Result<()> {
    let codec = source.codec();
    check_metadata(index, codec.dim())?;
    let live_count = usize::try_from(source.info().slots_live)?;
    if index.size() != live_count {
        return Err("ANN live count differs from the retained OXV generation".into());
    }
    let mut expected = vec![0f32; codec.dim()];
    let mut actual = vec![0f32; codec.dim()];
    let mut checked = 0;
    for (seg, segment) in source.segments().iter().enumerate() {
        let seg = u16::try_from(seg)?;
        for slot in 0..segment.slot_count() {
            if !source.is_live(seg, slot) {
                continue;
            }
            let key = key_id(segment.key(slot));
            codec.decode(
                segment.vector(slot),
                segment.vector_scale(slot),
                &mut expected,
            );
            actual.fill(f32::NAN);
            if index.get(key, &mut actual)? != 1 {
                return Err(format!("ANN lost live OXV key {key}").into());
            }
            if actual.iter().zip(&expected).any(|(actual, expected)| {
                !expected.is_finite() || actual.to_bits() != expected.to_bits()
            }) {
                return Err(format!("ANN vector content differs from OXV for key {key}").into());
            }
            checked += 1;
        }
    }
    // OXV's live keys are unique. Finding all of them with the same cardinality
    // leaves no room for unexpected keys, including resurrected tombstones.
    if checked != live_count {
        return Err("OXV live-key traversal differs from its recorded count".into());
    }
    Ok(())
}

fn check_hits(hits: &Matches, count: usize, is_live: &impl Fn(u64) -> bool) -> Result<()> {
    if hits.keys.len() != count || hits.distances.len() != count {
        return Err("native exact search returned the wrong number of keys or distances".into());
    }
    let mut keys = HashSet::with_capacity(count);
    for (&key, &distance) in hits.keys.iter().zip(&hits.distances) {
        if !is_live(key) || !keys.insert(key) || !distance.is_finite() {
            return Err("native exact search returned invalid keys or distances".into());
        }
    }
    if hits.distances.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err("native exact search distances are not sorted".into());
    }
    Ok(())
}

/// Compare top-k results after verifying the index's full contents against OXV.
/// USearch orders by distance only: tied keys can reorder and cutoff-tied keys
/// can substitute. No epsilon may turn genuinely different distances into ties.
pub fn verify_exact_parity(
    index: &Index,
    query: &[f32],
    expected: &Matches,
    actual: &Matches,
    count: usize,
    is_live: &impl Fn(u64) -> bool,
) -> Result<()> {
    check_metadata(index, query.len())?;
    check_hits(expected, count, is_live)?;
    check_hits(actual, count, is_live)?;
    if expected.distances != actual.distances {
        return Err("native exact ranked distances changed".into());
    }
    let Some(&cutoff) = expected.distances.last() else {
        return Ok(());
    };
    let by_key: HashMap<_, _> = expected
        .keys
        .iter()
        .copied()
        .zip(expected.distances.iter().copied())
        .collect();
    let mut substitutes = HashSet::new();
    for (&key, &distance) in actual.keys.iter().zip(&actual.distances) {
        match by_key.get(&key) {
            Some(&before) if before == distance => {}
            Some(_) => return Err(format!("native exact distance changed for key {key}").into()),
            None if distance == cutoff => {
                substitutes.insert(key);
            }
            None => return Err("native exact search lost a strictly better neighbor".into()),
        }
    }
    if substitutes.is_empty() {
        return Ok(());
    }

    // A live key carrying the cutoff score is not necessarily a real tie. Rescore
    // substitutions with the same native F32/IP arithmetic, avoiding a Rust dot
    // product's rounding differences from USearch's fast-math implementation.
    // The usual path allocates no probe; this rare path holds at most k vectors.
    let probe = Index::new(&IndexOptions {
        dimensions: query.len(),
        metric: MetricKind::IP,
        quantization: ScalarKind::F32,
        connectivity: 2,
        expansion_add: 2,
        expansion_search: 1,
        multi: false,
    })?;
    probe.reserve_capacity_and_threads(substitutes.len(), 1)?;
    let mut vector = vec![0f32; query.len()];
    for &key in &substitutes {
        if index.get(key, &mut vector)? != 1 {
            return Err(format!("native exact substitute key {key} is missing").into());
        }
        probe.add(key, &vector)?;
    }
    let rescored = probe.exact_search(query, substitutes.len())?;
    check_hits(&rescored, substitutes.len(), &|key| {
        substitutes.contains(&key)
    })?;
    if rescored
        .distances
        .iter()
        .any(|&distance| distance != cutoff)
    {
        return Err("native exact substitute is not tied at the cutoff".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Result<Index> {
        let index = Index::new(&IndexOptions {
            dimensions: 2,
            metric: MetricKind::IP,
            quantization: ScalarKind::F32,
            connectivity: 2,
            expansion_add: 8,
            expansion_search: 8,
            multi: false,
        })?;
        index.reserve_capacity_and_threads(6, 1)?;
        // For query [1, 0], distances are exactly 0, .25, .25, .5, .5, .75.
        for (key, vector) in [
            (1, [1.0f32, 0.0]),
            (2, [0.75, 0.25]),
            (3, [0.75, -0.25]),
            (4, [0.5, 0.5]),
            (5, [0.5, -0.5]),
            (6, [0.25, 0.0]),
        ] {
            index.add(key, &vector)?;
        }
        Ok(index)
    }

    fn hits(keys: &[u64], distances: &[f32]) -> Matches {
        Matches {
            keys: keys.to_vec(),
            distances: distances.to_vec(),
        }
    }

    fn compare(index: &Index, expected: &Matches, actual: &Matches) -> Result<()> {
        verify_exact_parity(index, &[1.0, 0.0], expected, actual, 4, &|key| {
            (1..=6).contains(&key)
        })
    }

    #[test]
    fn accepts_tied_order_and_boundary_substitution() -> Result<()> {
        let index = fixture()?;
        let expected = hits(&[1, 2, 3, 4], &[0.0, 0.25, 0.25, 0.5]);
        compare(&index, &expected, &hits(&[1, 3, 2, 4], &expected.distances))?;
        compare(&index, &expected, &hits(&[1, 3, 2, 5], &expected.distances))?;
        Ok(())
    }

    #[test]
    fn rejects_false_boundary_tie_even_when_distances_are_unchanged() -> Result<()> {
        let index = fixture()?;
        let expected = hits(&[1, 2, 3, 4], &[0.0, 0.25, 0.25, 0.5]);
        let error =
            compare(&index, &expected, &hits(&[1, 2, 3, 6], &expected.distances)).unwrap_err();
        assert!(error.to_string().contains("not tied at the cutoff"));
        Ok(())
    }

    #[test]
    fn rejects_lost_better_keys_and_changed_key_distance_pairs() -> Result<()> {
        let index = fixture()?;
        let expected = hits(&[1, 2, 3, 4], &[0.0, 0.25, 0.25, 0.5]);
        for keys in [[1, 2, 5, 4], [2, 1, 3, 4]] {
            assert!(compare(&index, &expected, &hits(&keys, &expected.distances)).is_err());
        }
        let next_float = f32::from_bits(0.5f32.to_bits() + 1);
        assert!(
            compare(
                &index,
                &expected,
                &hits(&expected.keys, &[0.0, 0.25, 0.25, next_float])
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn rejects_malformed_reference_and_candidate_results() -> Result<()> {
        let index = fixture()?;
        let expected = hits(&[1, 2, 3, 4], &[0.0, 0.25, 0.25, 0.5]);
        for invalid in [
            hits(&[1, 2, 3], &[0.0, 0.25, 0.25]),
            hits(&[1, 2, 3, 4, 5], &[0.0, 0.25, 0.25, 0.5, 0.5]),
            hits(&[1, 2, 3, 4], &[0.0, 0.25, 0.25]),
            hits(&[1, 2, 2, 4], &expected.distances),
            hits(&[1, 2, 3, 0], &expected.distances),
            hits(&[1, 2, 3, 7], &expected.distances),
            hits(&[1, 2, 3, 4], &[0.0, 0.5, 0.25, 0.25]),
            hits(&[1, 2, 3, 4], &[0.0, 0.25, 0.25, f32::NAN]),
            hits(&[1, 2, 3, 4], &[0.0, 0.25, 0.25, f32::INFINITY]),
            hits(&[1, 2, 3, 4], &[f32::NEG_INFINITY, 0.25, 0.25, 0.5]),
        ] {
            assert!(compare(&index, &expected, &invalid).is_err());
            assert!(compare(&index, &invalid, &expected).is_err());
        }
        Ok(())
    }

    #[test]
    fn accepts_empty_exact_results_only_when_none_are_expected() -> Result<()> {
        let index = fixture()?;
        let empty = hits(&[], &[]);
        verify_exact_parity(&index, &[1.0, 0.0], &empty, &empty, 0, &|_| false)?;
        assert!(verify_exact_parity(&index, &[1.0, 0.0], &empty, &empty, 1, &|_| true).is_err());
        Ok(())
    }
}
