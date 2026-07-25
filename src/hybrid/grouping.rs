use crate::semantic::types::{FusedCandidate, FusedSibling, GroupedResult, GroupingMode};
use std::collections::HashMap;

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
        group.sort_by(|a, b| {
            b.fused_score
                .partial_cmp(&a.fused_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

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

    results.sort_by(|a, b| {
        b.representative
            .fused_score
            .partial_cmp(&a.representative.fused_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
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
        group.sort_by(|a, b| {
            b.fused_score
                .partial_cmp(&a.fused_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

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

    results.sort_by(|a, b| {
        b.representative
            .fused_score
            .partial_cmp(&a.representative.fused_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
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

    fn mock_candidate(id: u64, section_id: u64, line_hash: u64, fused_score: f32) -> FusedCandidate {
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
            source: ResultSource::Semantic,
            section_id,
            line_hash,
        }
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
}
