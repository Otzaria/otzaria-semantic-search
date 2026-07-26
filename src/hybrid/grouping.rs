//! Post-fusion grouping.
//!
//! Semantic retrieval readily returns several neighbouring lines from the same
//! passage, which would otherwise fill a page of results with one source.
//! Grouping collapses them behind a representative.

use crate::semantic::types::{FusedCandidate, FusedSibling, GroupedResult, GroupingMode};
use std::collections::HashMap;

/// Upper bound on siblings listed per group.
///
/// Matches the lexical engine's own `MERGED_SIBLINGS_CAP`, so both paths present
/// merged results identically. `group_count` still reports the true group size
/// when the list is truncated.
pub const MAX_SIBLINGS_PER_GROUP: usize = 10;

/// Helper to convert a FusedCandidate into a FusedSibling,
/// preserving identity and location but dropping full text/scores.
fn into_sibling(candidate: FusedCandidate) -> FusedSibling {
    FusedSibling {
        title: candidate.title,
        reference: candidate.reference,
        line_id: candidate.line_id,
        segment: candidate.segment,
        file_path: candidate.file_path,
        is_pdf: candidate.is_pdf,
    }
}

/// Sort candidates best-first, breaking ties on `line_id` so the representative
/// of a group is chosen deterministically rather than by map iteration order.
fn sort_candidates_by_score(candidates: &mut [FusedCandidate]) {
    candidates.sort_by(|a, b| {
        b.fused_score
            .total_cmp(&a.fused_score)
            .then_with(|| a.line_id.cmp(&b.line_id))
    });
}

/// Sort groups best-first, breaking ties on the representative's `line_id`.
fn sort_groups_by_score(groups: &mut [GroupedResult]) {
    groups.sort_by(|a, b| {
        b.representative
            .fused_score
            .total_cmp(&a.representative.fused_score)
            .then_with(|| a.representative.line_id.cmp(&b.representative.line_id))
    });
}

/// Groups candidates by section_id and file_path.
/// The candidate with the highest fused_score becomes the representative.
pub fn group_by_section(candidates: Vec<FusedCandidate>) -> Vec<GroupedResult> {
    let mut groups: HashMap<(u64, String), Vec<FusedCandidate>> = HashMap::new();

    for candidate in candidates {
        let key = (candidate.section_id, candidate.file_path.clone());
        groups.entry(key).or_default().push(candidate);
    }

    let mut results = Vec::new();

    for mut group in groups.into_values() {
        sort_candidates_by_score(&mut group);

        let total_in_group = group.len() as u32;
        let representative = group.remove(0);
        let siblings = group
            .into_iter()
            .take(MAX_SIBLINGS_PER_GROUP)
            .map(into_sibling)
            .collect();

        results.push(GroupedResult {
            representative,
            siblings,
            group_count: total_in_group,
        });
    }

    sort_groups_by_score(&mut results);
    results
}

/// Groups candidates by identical text based on their line_hash.
/// A line_hash of 0 is considered too short for dedup and is never grouped.
/// The candidate with the highest fused_score becomes the representative.
pub fn group_by_identical_text(candidates: Vec<FusedCandidate>) -> Vec<GroupedResult> {
    let mut groups: HashMap<u64, Vec<FusedCandidate>> = HashMap::new();
    let mut ungrouped_results = Vec::new();

    for candidate in candidates {
        let hash = candidate.line_hash;
        if hash == 0 {
            ungrouped_results.push(candidate);
        } else {
            groups.entry(hash).or_default().push(candidate);
        }
    }

    let mut results = Vec::new();

    for mut group in groups.into_values() {
        sort_candidates_by_score(&mut group);

        let total_in_group = group.len() as u32;
        let representative = group.remove(0);
        let siblings = group
            .into_iter()
            .take(MAX_SIBLINGS_PER_GROUP)
            .map(into_sibling)
            .collect();

        results.push(GroupedResult {
            representative,
            siblings,
            group_count: total_in_group,
        });
    }

    for candidate in ungrouped_results {
        results.push(GroupedResult {
            representative: candidate,
            siblings: Vec::new(),
            group_count: 1,
        });
    }

    sort_groups_by_score(&mut results);
    results
}

/// Dispatches grouping to the appropriate method based on the requested GroupingMode.
pub fn group_results(candidates: Vec<FusedCandidate>, mode: GroupingMode) -> Vec<GroupedResult> {
    match mode {
        GroupingMode::SameSection => group_by_section(candidates),
        GroupingMode::IdenticalText => group_by_identical_text(candidates),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic::types::ResultSource;

    fn mock_candidate(
        id: u64,
        section_id: u64,
        line_hash: u64,
        fused_score: f32,
    ) -> FusedCandidate {
        FusedCandidate {
            line_id: id,
            title: format!("Title {id}"),
            reference: format!("Ref {id}"),
            segment: id,
            file_path: "mock_path.txt".to_string(),
            is_pdf: false,
            text: "test text".to_string(),
            raw_bm25_score: Some(1.0),
            normalized_bm25: Some(1.0),
            raw_semantic_score: Some(1.0),
            normalized_semantic: Some(1.0),
            fused_score,
            lexical_weight: 0.5,
            semantic_weight: 0.5,
            needs_hydration: false,
            source: ResultSource::Semantic,
            section_id,
            line_hash,
        }
    }

    /// Same section but a different file: two different books can share a
    /// section id, and collapsing across them would merge unrelated passages.
    fn in_file(mut candidate: FusedCandidate, file_path: &str) -> FusedCandidate {
        candidate.file_path = file_path.to_string();
        candidate
    }

    #[test]
    fn test_group_by_section_groups_same() {
        let c1 = mock_candidate(1, 100, 123, 0.9);
        let c2 = mock_candidate(2, 100, 123, 0.8);

        let result = group_by_section(vec![c1, c2]);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].representative.line_id, 1);
        assert_eq!(result[0].siblings.len(), 1);
        assert_eq!(result[0].siblings[0].line_id, 2);
    }

    #[test]
    fn the_best_scoring_candidate_becomes_the_representative_regardless_of_input_order() {
        let weak = mock_candidate(1, 100, 111, 0.2);
        let strong = mock_candidate(2, 100, 222, 0.9);

        for input in [vec![weak.clone(), strong.clone()], vec![strong, weak]] {
            let result = group_by_section(input);
            assert_eq!(result.len(), 1);
            assert_eq!(result[0].representative.line_id, 2);
            assert_eq!(result[0].representative.fused_score, 0.9);
        }
    }

    #[test]
    fn the_same_section_id_in_different_files_stays_separate() {
        let a = in_file(mock_candidate(1, 100, 111, 0.9), "book_a.txt");
        let b = in_file(mock_candidate(2, 100, 222, 0.8), "book_b.txt");

        let result = group_by_section(vec![a, b]);
        assert_eq!(result.len(), 2, "grouping must key on (section, file)");
    }

    #[test]
    fn group_count_reports_the_real_size_even_when_the_sibling_list_is_capped() {
        let candidates: Vec<FusedCandidate> = (0..25)
            .map(|i| mock_candidate(i, 100, 1000 + i, 1.0 - i as f32 * 0.01))
            .collect();

        let result = group_by_section(candidates);
        assert_eq!(result.len(), 1);
        assert_eq!(
            result[0].siblings.len(),
            MAX_SIBLINGS_PER_GROUP,
            "the listed siblings are capped"
        );
        assert_eq!(
            result[0].group_count, 25,
            "but the reported count stays truthful"
        );
    }

    #[test]
    fn identical_text_grouping_collapses_equal_line_hashes() {
        let candidates = vec![
            in_file(mock_candidate(1, 10, 555, 0.9), "book_a.txt"),
            in_file(mock_candidate(2, 20, 555, 0.7), "book_b.txt"),
            in_file(mock_candidate(3, 30, 999, 0.8), "book_c.txt"),
        ];

        let result = group_by_identical_text(candidates);
        assert_eq!(result.len(), 2);

        let duplicated = result
            .iter()
            .find(|g| g.group_count == 2)
            .expect("the shared line_hash must collapse");
        assert_eq!(duplicated.representative.line_id, 1);
        assert_eq!(duplicated.representative.line_hash, 555);
    }

    /// A `line_hash` of 0 means "no dedup signature". Grouping on it would
    /// collapse every unsigned line into a single result.
    #[test]
    fn a_zero_line_hash_is_never_grouped() {
        let candidates = vec![
            mock_candidate(1, 10, 0, 0.9),
            mock_candidate(2, 20, 0, 0.8),
            mock_candidate(3, 30, 0, 0.7),
        ];

        let result = group_by_identical_text(candidates);
        assert_eq!(result.len(), 3);
        assert!(result.iter().all(|g| g.group_count == 1));
        assert!(result.iter().all(|g| g.siblings.is_empty()));
    }

    #[test]
    fn groups_are_ordered_by_their_representatives_score() {
        let candidates = vec![
            mock_candidate(1, 10, 111, 0.3),
            mock_candidate(2, 20, 222, 0.9),
            mock_candidate(3, 30, 333, 0.6),
        ];

        let result = group_results(candidates, GroupingMode::SameSection);
        let scores: Vec<f32> = result
            .iter()
            .map(|g| g.representative.fused_score)
            .collect();
        assert_eq!(scores, vec![0.9, 0.6, 0.3]);
    }

    /// Grouping runs on every page, so ties must not reshuffle between calls.
    #[test]
    fn tied_groups_are_ordered_deterministically() {
        let candidates: Vec<FusedCandidate> = (1..=8)
            .map(|i| mock_candidate(i, i * 10, i * 100, 0.5))
            .collect();

        let first: Vec<u64> = group_results(candidates.clone(), GroupingMode::SameSection)
            .iter()
            .map(|g| g.representative.line_id)
            .collect();
        assert_eq!(first, vec![1, 2, 3, 4, 5, 6, 7, 8]);

        for _ in 0..5 {
            let again: Vec<u64> = group_results(candidates.clone(), GroupingMode::SameSection)
                .iter()
                .map(|g| g.representative.line_id)
                .collect();
            assert_eq!(again, first);
        }
    }

    #[test]
    fn grouping_an_empty_candidate_list_yields_no_groups() {
        for mode in [GroupingMode::SameSection, GroupingMode::IdenticalText] {
            assert!(group_results(Vec::new(), mode).is_empty());
        }
    }

    #[test]
    fn every_candidate_survives_grouping_exactly_once() {
        let candidates: Vec<FusedCandidate> = (1..=6)
            .map(|i| mock_candidate(i, i % 2, 100 + i % 3, i as f32 / 10.0))
            .collect();

        for mode in [GroupingMode::SameSection, GroupingMode::IdenticalText] {
            let groups = group_results(candidates.clone(), mode);
            let total: u32 = groups.iter().map(|g| g.group_count).sum();
            assert_eq!(total, 6, "{mode:?} lost or duplicated candidates");

            let mut ids: Vec<u64> = groups
                .iter()
                .flat_map(|g| {
                    std::iter::once(g.representative.line_id)
                        .chain(g.siblings.iter().map(|s| s.line_id))
                })
                .collect();
            ids.sort_unstable();
            assert_eq!(ids, vec![1, 2, 3, 4, 5, 6]);
        }
    }
}
