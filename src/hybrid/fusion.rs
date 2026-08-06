//! Score normalization and fusion primitives.
//!
//! BM25 and cosine similarity live on different scales, so they are normalized
//! into `[0, 1]` before being combined.
//!
//! Two fusion strategies exist side by side, and the coordinator picks between them
//! per `FusionStrategy` in the active profile. [`fuse_rrf`] is rank-based and needs no
//! score calibration at all, which may well make it the better default. Neither has
//! been measured on Hebrew queries yet — that comparison needs the labelled relevance
//! set from stage S1.

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

/// Map BM25 scores into `[0, 1]` with a saturating curve `x / (k + x)`.
///
/// `k` sets where the curve bends: scores well above `k` all land near 1.0 and
/// stop being distinguishable, so `k` should sit in the same range as the
/// engine's typical scores. NaN and non-positive scores map to 0.
pub fn normalize_bm25_scores(scores: &[f32], k: f32) -> Vec<f32> {
    scores
        .iter()
        .map(|&x| {
            if x.is_nan() || x <= 0.0 {
                0.0
            } else if x.is_infinite() {
                // §1.2: An infinite BM25 score (corrupt data or division edge
                // case) would propagate through the fused score and break
                // downstream confidence computation.
                1.0
            } else {
                (x / (k + x)).clamp(0.0, 1.0)
            }
        })
        .collect()
}

/// Map cosine similarities from `[-1, 1]` into `[0, 1]`.
///
/// Note what this implies: an orthogonal — that is, entirely unrelated — vector
/// normalizes to 0.5, not 0, so on its own this mapping lets a semantic path that
/// found nothing useful contribute mid-range scores. That is why the coordinator
/// calls [`normalize_semantic_with_threshold`] instead; this thresholdless form is
/// kept for callers that want the raw mapping.
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

pub fn normalize_bm25_adaptive(scores: &[f32], k: f32) -> Vec<f32> {
    if scores.is_empty() {
        return Vec::new();
    }

    let mut max_score = f32::NEG_INFINITY;
    let mut min_score = f32::INFINITY;

    for &score in scores {
        if !score.is_nan() && score > 0.0 {
            if score > max_score {
                max_score = score;
            }
            if score < min_score {
                min_score = score;
            }
        }
    }

    if max_score > 2.0 * k && max_score > min_score {
        // Use min-max normalization
        scores
            .iter()
            .map(|&x| {
                if x.is_nan() || x <= 0.0 {
                    0.0
                } else {
                    ((x - min_score) / (max_score - min_score)).clamp(0.0, 1.0)
                }
            })
            .collect()
    } else {
        // Fall back to saturating curve
        normalize_bm25_scores(scores, k)
    }
}

pub fn normalize_semantic_with_threshold(scores: &[f32], threshold: f32) -> Vec<f32> {
    let normalized = normalize_semantic_scores(scores);
    normalized
        .into_iter()
        .map(|x| if x < threshold { 0.0 } else { x })
        .collect()
}

pub fn compute_confidence(sorted_scores: &[f32]) -> Option<f32> {
    if sorted_scores.len() < 2 {
        return None;
    }

    let top = sorted_scores[0];
    let second = sorted_scores[1];

    if top.is_nan() || second.is_nan() || top <= 0.0 {
        return None;
    }

    let confidence = (top - second) / top.max(1e-6);
    Some(confidence.clamp(0.0, 1.0))
}

/// Which retrieval paths produced a candidate.
///
/// Returns `None` for a candidate present in neither, which cannot happen for an
/// entry that exists — but a library must not panic to say so.
fn classify_source(lexical: Option<f32>, semantic: Option<f32>) -> Option<ResultSource> {
    match (lexical, semantic) {
        (Some(_), Some(_)) => Some(ResultSource::Both),
        (Some(_), None) => Some(ResultSource::Lexical),
        (None, Some(_)) => Some(ResultSource::Semantic),
        (None, None) => None,
    }
}

/// Sort fused entries by descending score, breaking ties on `line_id`.
///
/// The tie-break is what makes pagination stable: the entries come out of a
/// `HashMap`, whose iteration order differs between runs.
fn sort_by_score_desc(entries: &mut [FusedEntry]) {
    entries.sort_by(|a, b| {
        b.fused_score
            .total_cmp(&a.fused_score)
            .then_with(|| a.line_id.cmp(&b.line_id))
    });
}

/// Weighted score fusion: `alpha * lexical + (1 - alpha) * semantic`.
///
/// Both score lists must already be normalized to a common scale. A candidate
/// missing from one side contributes 0 for it.
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
        .filter_map(|(id, (lexical_score, semantic_score))| {
            let source = classify_source(lexical_score, semantic_score)?;
            let l = lexical_score.unwrap_or(0.0);
            let s = semantic_score.unwrap_or(0.0);
            Some(FusedEntry {
                line_id: id,
                fused_score: alpha * l + (1.0 - alpha) * s,
                lexical_score,
                semantic_score,
                source,
            })
        })
        .collect();

    sort_by_score_desc(&mut result);
    result
}

/// Reciprocal Rank Fusion: each list contributes `1 / (k + rank)`.
///
/// Uses only the *order* of each list, so the two engines' score scales never
/// have to be reconciled. Both inputs must already be sorted best-first —
/// position is the signal. `k` damps the weight of the top ranks; 60 is the
/// value from the original paper and the usual default.
pub fn fuse_rrf(lexical: &[(u64, f32)], semantic: &[(u64, f32)], k: u32) -> Vec<FusedEntry> {
    let mut map: HashMap<u64, (Option<f32>, Option<f32>, f32)> =
        HashMap::with_capacity(lexical.len() + semantic.len());

    for (idx, &(id, score)) in lexical.iter().enumerate() {
        let rrf_score = 1.0 / (k as f32 + (idx + 1) as f32);
        let entry = map.entry(id).or_insert((None, None, 0.0));
        entry.0 = Some(score);
        entry.2 += rrf_score;
    }

    for (idx, &(id, score)) in semantic.iter().enumerate() {
        let rrf_score = 1.0 / (k as f32 + (idx + 1) as f32);
        let entry = map.entry(id).or_insert((None, None, 0.0));
        entry.1 = Some(score);
        entry.2 += rrf_score;
    }

    let mut result: Vec<FusedEntry> = map
        .into_iter()
        .filter_map(|(id, (lexical_score, semantic_score, fused_score))| {
            Some(FusedEntry {
                line_id: id,
                fused_score,
                lexical_score,
                semantic_score,
                source: classify_source(lexical_score, semantic_score)?,
            })
        })
        .collect();

    sort_by_score_desc(&mut result);
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

    #[test]
    fn test_normalize_bm25_adaptive() {
        let scores = vec![0.0, 10.0, 25.0];
        // max_score is 25.0, which is > 2*10.0 (20.0), so it uses min-max.
        // min is 10.0, max is 25.0.
        // 10.0 -> 0.0
        // 25.0 -> 1.0
        let normalized = normalize_bm25_adaptive(&scores, 10.0);
        assert_eq!(normalized, vec![0.0, 0.0, 1.0]);

        let scores2 = vec![0.0, 5.0, 15.0];
        // max is 15.0, not > 20.0, so it uses saturating curve.
        let normalized2 = normalize_bm25_adaptive(&scores2, 10.0);
        assert_eq!(normalized2, vec![0.0, 5.0 / 15.0, 15.0 / 25.0]);
    }

    #[test]
    fn test_normalize_semantic_with_threshold() {
        let scores = vec![-1.0, 0.0, 1.0];
        // normalizes to 0.0, 0.5, 1.0
        let normalized = normalize_semantic_with_threshold(&scores, 0.6);
        assert_eq!(normalized, vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_compute_confidence() {
        let scores = vec![1.0, 0.8, 0.5];
        let conf = compute_confidence(&scores).unwrap();
        assert!((conf - 0.2).abs() < 1e-6); // (1.0 - 0.8) / 1.0

        let scores2 = vec![0.5, 0.5];
        let conf2 = compute_confidence(&scores2);
        assert_eq!(conf2, Some(0.0));

        let scores3 = vec![1.0];
        assert_eq!(compute_confidence(&scores3), None);
    }
}
