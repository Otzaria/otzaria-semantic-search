use crate::semantic::types::ResultSource;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub struct FusedEntry {
    pub line_id: u64,
    pub fused_score: f32,
    pub lexical_score: Option<f32>,
    pub semantic_score: Option<f32>,
    pub source: ResultSource,
}

pub fn normalize_bm25_scores(scores: &[f32], k: f32) -> Vec<f32> {
    scores
        .iter()
        .map(|&x| {
            if x.is_nan() || x <= 0.0 {
                0.0
            } else {
                (x / (k + x)).clamp(0.0, 1.0)
            }
        })
        .collect()
}

pub fn normalize_semantic_scores(scores: &[f32]) -> Vec<f32> {
    scores
        .iter()
        .map(|&x| {
            if x.is_nan() {
                0.0
            } else {
                ((x + 1.0) / 2.0).clamp(0.0, 1.0)
            }
        })
        .collect()
}

pub fn fuse_weighted(
    lexical: &[(u64, f32)],
    semantic: &[(u64, f32)],
    alpha: f32,
) -> Vec<FusedEntry> {
    let mut map: HashMap<u64, (Option<f32>, Option<f32>)> =
        HashMap::with_capacity(lexical.len() + semantic.len());

    for &(id, score) in lexical {
        map.entry(id).or_insert((None, None)).0 = Some(score);
    }
    for &(id, score) in semantic {
        map.entry(id).or_insert((None, None)).1 = Some(score);
    }

    let mut result: Vec<FusedEntry> = map
        .into_iter()
        .map(|(id, (l_score, s_score))| {
            let l = l_score.unwrap_or(0.0);
            let s = s_score.unwrap_or(0.0);
            let fused_score = alpha * l + (1.0 - alpha) * s;
            let source = match (l_score, s_score) {
                (Some(_), Some(_)) => ResultSource::Both,
                (Some(_), None) => ResultSource::Lexical,
                (None, Some(_)) => ResultSource::Semantic,
                _ => unreachable!(),
            };
            FusedEntry {
                line_id: id,
                fused_score,
                lexical_score: l_score,
                semantic_score: s_score,
                source,
            }
        })
        .collect();

    // Sort descending by fused score
    result.sort_by(|a, b| {
        b.fused_score
            .partial_cmp(&a.fused_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    result
}

pub fn fuse_rrf(lexical: &[(u64, f32)], semantic: &[(u64, f32)], k: u32) -> Vec<FusedEntry> {
    let mut map: HashMap<u64, (Option<f32>, Option<f32>, f32)> =
        HashMap::with_capacity(lexical.len() + semantic.len());

    for (idx, &(id, score)) in lexical.iter().enumerate() {
        let rank = (idx + 1) as f32;
        let rrf_score = 1.0 / (k as f32 + rank);
        let entry = map.entry(id).or_insert((None, None, 0.0));
        entry.0 = Some(score);
        entry.2 += rrf_score;
    }

    for (idx, &(id, score)) in semantic.iter().enumerate() {
        let rank = (idx + 1) as f32;
        let rrf_score = 1.0 / (k as f32 + rank);
        let entry = map.entry(id).or_insert((None, None, 0.0));
        entry.1 = Some(score);
        entry.2 += rrf_score;
    }

    let mut result: Vec<FusedEntry> = map
        .into_iter()
        .map(|(id, (l_score, s_score, fused_score))| {
            let source = match (l_score, s_score) {
                (Some(_), Some(_)) => ResultSource::Both,
                (Some(_), None) => ResultSource::Lexical,
                (None, Some(_)) => ResultSource::Semantic,
                _ => unreachable!(),
            };
            FusedEntry {
                line_id: id,
                fused_score,
                lexical_score: l_score,
                semantic_score: s_score,
                source,
            }
        })
        .collect();

    // Sort descending by fused score
    result.sort_by(|a, b| {
        b.fused_score
            .partial_cmp(&a.fused_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock ResultSource for tests if it's not actually present, but we assume
    // it's available from crate::semantic::types per instructions.
    // Ensure it derives Debug, Clone, PartialEq.

    #[test]
    fn test_normalize_bm25_scores() {
        let scores = vec![0.0, 10.0, 90.0];
        let normalized = normalize_bm25_scores(&scores, 10.0);
        assert_eq!(normalized, vec![0.0, 0.5, 0.9]);
    }

    #[test]
    fn test_normalize_semantic_scores() {
        let scores = vec![-1.0, 0.0, 1.0];
        let normalized = normalize_semantic_scores(&scores);
        assert_eq!(normalized, vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn test_fuse_weighted_alpha_1() {
        let lexical = vec![(1, 0.8), (2, 0.4)];
        let semantic = vec![(1, 0.9), (3, 0.7)];
        let fused = fuse_weighted(&lexical, &semantic, 1.0);

        assert_eq!(fused.len(), 3);
        let first = fused.iter().find(|e| e.line_id == 1).unwrap();
        assert_eq!(first.fused_score, 0.8);
        assert_eq!(first.source, ResultSource::Both);

        let second = fused.iter().find(|e| e.line_id == 2).unwrap();
        assert_eq!(second.fused_score, 0.4);
        assert_eq!(second.source, ResultSource::Lexical);

        let third = fused.iter().find(|e| e.line_id == 3).unwrap();
        assert_eq!(third.fused_score, 0.0);
        assert_eq!(third.source, ResultSource::Semantic);
    }

    #[test]
    fn test_fuse_weighted_alpha_0() {
        let lexical = vec![(1, 0.8), (2, 0.4)];
        let semantic = vec![(1, 0.9), (3, 0.7)];
        let fused = fuse_weighted(&lexical, &semantic, 0.0);

        assert_eq!(fused.len(), 3);
        let first = fused.iter().find(|e| e.line_id == 1).unwrap();
        assert_eq!(first.fused_score, 0.9);

        let second = fused.iter().find(|e| e.line_id == 2).unwrap();
        assert_eq!(second.fused_score, 0.0);

        let third = fused.iter().find(|e| e.line_id == 3).unwrap();
        assert_eq!(third.fused_score, 0.7);
    }

    #[test]
    fn test_fuse_weighted_merge() {
        let lexical = vec![(1, 0.8)];
        let semantic = vec![(1, 0.6)];
        let fused = fuse_weighted(&lexical, &semantic, 0.5);

        assert_eq!(fused.len(), 1);
        assert!((fused[0].fused_score - 0.7).abs() < 1e-5);
        assert_eq!(fused[0].source, ResultSource::Both);
    }

    #[test]
    fn test_fuse_rrf() {
        let lexical = vec![(1, 0.9), (2, 0.8)];
        let semantic = vec![(2, 0.95), (3, 0.85)];
        let fused = fuse_rrf(&lexical, &semantic, 60);

        assert_eq!(fused.len(), 3);

        let id2 = fused.iter().find(|e| e.line_id == 2).unwrap();
        assert_eq!(id2.source, ResultSource::Both);
        assert_eq!(id2.fused_score, (1.0 / 61.0) + (1.0 / 62.0));

        let id1 = fused.iter().find(|e| e.line_id == 1).unwrap();
        assert_eq!(id1.source, ResultSource::Lexical);
        assert_eq!(id1.fused_score, 1.0 / 61.0);

        let id3 = fused.iter().find(|e| e.line_id == 3).unwrap();
        assert_eq!(id3.source, ResultSource::Semantic);
        assert_eq!(id3.fused_score, 1.0 / 62.0);

        // Check correct descending sort
        assert_eq!(fused[0].line_id, 2);
        assert_eq!(fused[1].line_id, 1);
        assert_eq!(fused[2].line_id, 3);
    }

    #[test]
    fn test_empty_input() {
        let empty: Vec<(u64, f32)> = vec![];
        let fused_w = fuse_weighted(&empty, &empty, 0.5);
        assert!(fused_w.is_empty());

        let fused_rrf = fuse_rrf(&empty, &empty, 60);
        assert!(fused_rrf.is_empty());
    }
}
