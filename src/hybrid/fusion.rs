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
//!
//! The active profile can be passed with each search
//! ([`HybridSearchParams::ranking`](crate::hybrid::coordinator::HybridSearchParams::ranking)),
//! the strategy, RRF's `k` and BM25's `k` included, so that comparison — and whatever
//! tuning follows it — can run from the application without a release of this crate. The
//! defaults stay as they are until it has.

use crate::semantic::types::{FusedCandidate, ResultSource};
use std::cmp::Ordering;
use std::collections::HashMap;

/// The order of fused results, best first: the score; then the semantic path's own order
/// ([`FusedCandidate::semantic_position`]), a line only the lexical path found after every
/// line it placed; then the line id, then the book — two books can hold lines with the same
/// id, and a line is one per book and id — so the order is total. Fusion, grouping and the
/// groups themselves all sort by it, so a page is the same on every call and pagination
/// neither repeats nor skips a result; `HashMap` iteration order, which every fusion and
/// grouping starts from, differs from one call to the next.
///
/// The semantic order comes before the id because every line one vector resolved to has
/// the same score: the resolver placed them — a line of each book first, then the repeats
/// within a book — and the id would put one book's repeats, numbered in a row, ahead of
/// every other book's copy of the text.
pub(crate) fn best_first(a: &FusedCandidate, b: &FusedCandidate) -> Ordering {
    b.fused_score
        .total_cmp(&a.fused_score)
        .then_with(|| semantic_order(a.semantic_position, b.semantic_position))
        .then_with(|| a.line_id.cmp(&b.line_id))
        .then_with(|| a.file_path.cmp(&b.file_path))
}

/// The semantic path's order, a line it did not place after every line it did.
fn semantic_order(a: Option<u32>, b: Option<u32>) -> Ordering {
    match (a, b) {
        (Some(a), Some(b)) => a.cmp(&b),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

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

    /// A fused candidate with what [`best_first`] reads, and nothing else.
    fn ranked(score: f32, position: Option<u32>, line_id: u64, book: &str) -> FusedCandidate {
        FusedCandidate {
            title: String::new(),
            reference: String::new(),
            text: String::new(),
            line_id,
            section_id: 0,
            line_hash: 0,
            segment: 0,
            is_pdf: false,
            file_path: book.to_string(),
            needs_hydration: position.is_some(),
            source: if position.is_some() {
                ResultSource::Semantic
            } else {
                ResultSource::Lexical
            },
            raw_bm25_score: None,
            normalized_bm25: None,
            raw_semantic_score: None,
            normalized_semantic: None,
            fused_score: score,
            semantic_position: position,
            lexical_weight: 0.5,
            semantic_weight: 0.5,
        }
    }

    /// What `best_first` orders by, a score by its bits.
    fn keys(candidate: &FusedCandidate) -> (u32, Option<u32>, u64, String) {
        (
            candidate.fused_score.to_bits(),
            candidate.semantic_position,
            candidate.line_id,
            candidate.file_path.clone(),
        )
    }

    /// Lines with the same score fall in the semantic path's order — a line it did not place
    /// after every one it did — and only then by id and book; the score comes first.
    #[test]
    fn equal_scores_fall_in_the_semantic_order_before_the_id() {
        let mut lines = [
            ranked(0.5, None, 1, "a.txt"),
            ranked(0.5, Some(3), 2, "a.txt"),
            ranked(0.5, Some(0), 90, "c.txt"),
            ranked(0.9, None, 50, "z.txt"),
            ranked(0.5, Some(1), 3, "a.txt"),
            ranked(0.5, None, 1, "b.txt"),
        ];
        lines.sort_by(best_first);
        let order: Vec<(u64, &str)> = lines
            .iter()
            .map(|line| (line.line_id, line.file_path.as_str()))
            .collect();
        assert_eq!(
            order,
            [
                (50, "z.txt"),
                (90, "c.txt"),
                (3, "a.txt"),
                (2, "a.txt"),
                (1, "a.txt"),
                (1, "b.txt")
            ]
        );
    }

    /// `best_first` is a total order: antisymmetric, transitive, and two candidates compare
    /// equal only when every key it reads is equal — so two lines, one per book and id, never
    /// do, whatever their scores (NaN and both zeros included) and wherever the semantic path
    /// put them. A sort therefore gives one order whatever order the lines came in, which is
    /// what makes a page the same on every call.
    #[test]
    fn best_first_is_a_total_order() {
        let mut lines = Vec::new();
        for score in [0.5, f32::NAN, 0.0, -0.0, 1.0, f32::NEG_INFINITY] {
            for position in [None, Some(0), Some(1), Some(u32::MAX)] {
                for line_id in [1, 2] {
                    for book in ["a.txt", "b.txt"] {
                        lines.push(ranked(score, position, line_id, book));
                    }
                }
            }
        }
        for a in &lines {
            for b in &lines {
                let ab = best_first(a, b);
                assert_eq!(
                    ab,
                    best_first(b, a).reverse(),
                    "{:?} {:?}",
                    keys(a),
                    keys(b)
                );
                assert_eq!(ab == Ordering::Equal, keys(a) == keys(b));
                for c in &lines {
                    if ab.is_le() && best_first(b, c).is_le() {
                        assert!(
                            best_first(a, c).is_le(),
                            "{:?} {:?} {:?}",
                            keys(a),
                            keys(b),
                            keys(c)
                        );
                    }
                }
            }
        }

        let sorted = |mut lines: Vec<FusedCandidate>| {
            lines.sort_by(best_first);
            lines.iter().map(keys).collect::<Vec<_>>()
        };
        let once = sorted(lines.clone());
        let mut shuffled = lines.clone();
        // Fisher–Yates on a fixed linear congruential sequence: arbitrary, and repeatable.
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        for i in (1..shuffled.len()).rev() {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            shuffled.swap(i, (state >> 33) as usize % (i + 1));
        }
        for input in [lines.iter().rev().cloned().collect(), shuffled, {
            let mut rotated = lines.clone();
            rotated.rotate_left(37);
            rotated
        }] {
            assert_eq!(sorted(input), once);
        }
    }
}
