use crate::errors::SemanticSearchError;
use serde::{Deserialize, Serialize};
use std::fmt;

/// `k` in the saturating BM25 normalization `score / (k + score)`, for every preset.
///
/// It sets where the curve bends: a score well above `k` lands near 1.0 and stops being
/// told apart from the scores above it. 10.0 puts the bend inside the range Tantivy's BM25
/// scores typically fall in, 2 to 30, so that top lexical matches are not flattened
/// together. That is reasoned from the range, not measured against relevance judgements —
/// see [`RankingProfile`] for what measuring it needs.
pub const DEFAULT_BM25_SATURATION_K: f32 = 10.0;

fn default_bm25_saturation_k() -> f32 {
    DEFAULT_BM25_SATURATION_K
}

/// The lexical weight `alpha` for each kind of query
/// [`analyze_query`](crate::hybrid::ranking::analyze_query) tells apart; `1 - alpha` goes
/// to the semantic side. Read by
/// [`compute_alpha_with`](crate::hybrid::ranking::compute_alpha_with) whenever
/// [`RankingProfile::alpha_override`] does not fix one alpha for every query.
///
/// The defaults are the numbers the ranking has always used, the same for every preset:
/// lexical-heavy for a lookup, semantic-heavy for a question. Like every number in a
/// [`RankingProfile`] they are unmeasured. Each must be a number from 0 to 1.
///
/// `#[serde(default)]`: a host may send only the types it tunes, and the others keep these
/// values.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct QueryTypeAlphas {
    /// A query with a quoted phrase: a verbatim lookup, where the lexical engine is
    /// authoritative. At 1.0 — the default — a hybrid search does not consult the semantic
    /// path at all, which is what the default buys besides a weight.
    pub quoted_phrase: f32,
    /// One or two words, one of them carrying a digit: a reference such as a page or a
    /// verse number.
    pub exact_reference: f32,
    /// One or two words without a digit.
    pub short: f32,
    /// Three or four words.
    pub mixed: f32,
    /// Five words or more: a question or a description more than a lookup.
    pub conceptual: f32,
    /// No words at all. Such a query cannot be embedded, so a hybrid search degrades to its
    /// lexical results, fused at alpha 1.0 whatever this says. In practice it decides only
    /// whether the semantic path is tried at all: at 1.0 it is not.
    pub unknown: f32,
}

impl Default for QueryTypeAlphas {
    fn default() -> Self {
        Self {
            quoted_phrase: 1.0,
            exact_reference: 0.85,
            short: 0.7,
            mixed: 0.5,
            conceptual: 0.3,
            unknown: 0.5,
        }
    }
}

/// Predefined search profiles that balance speed and quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchProfile {
    Fast,
    Balanced,
    Best,
}

impl fmt::Display for SearchProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SearchProfile::Fast => write!(f, "Fast"),
            SearchProfile::Balanced => write!(f, "Balanced"),
            SearchProfile::Best => write!(f, "Best"),
        }
    }
}

/// Strategy for fusing lexical and semantic scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FusionStrategy {
    Weighted,
    RRF { k: u32 },
    Adaptive,
}

impl fmt::Display for FusionStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FusionStrategy::Weighted => write!(f, "Weighted"),
            FusionStrategy::RRF { k } => write!(f, "RRF(k={k})"),
            FusionStrategy::Adaptive => write!(f, "Adaptive"),
        }
    }
}

/// The complete set of tuning parameters for hybrid ranking.
///
/// Every search ranks by one of these. By default it is the preset the search's
/// [`SearchProfile`] names, with any feature flags applied on top; a host can instead pass
/// a whole profile with the search
/// ([`HybridSearchParams::ranking`](crate::hybrid::coordinator::HybridSearchParams::ranking),
/// [`SearchRequest::ranking`](crate::api::hybrid_search::SearchRequest::ranking)), which is
/// [validated](Self::validate) and then used as it stands.
///
/// # The numbers are placeholders
///
/// None of the defaults has been measured. Each was reasoned from a scale — BM25's typical
/// range, a cosine of about 0.1 meaning unrelated — or carried over from the literature, as
/// RRF's `k = 60` is, and none has been checked against what a reader of this library
/// finds relevant. Calibrating them needs the labelled relevance set stage S1 produces:
/// Hebrew queries of every [`QueryType`](crate::hybrid::ranking::QueryType), each with the
/// lines judged relevant to it; a metric over the page a user sees (nDCG@10, or recall at
/// the page size); and runs over that set that vary one family of parameters at a time —
/// the fusion strategy and RRF's `k` first, because RRF needs no score calibration at all,
/// then alpha per query type, BM25's `k`, the semantic threshold and the bonuses.
///
/// Passing a profile per search is what lets those runs, and whatever tuning follows them,
/// happen from the application without a release of this crate. Until they have happened
/// the defaults stay exactly the ranking this crate has always produced, and a test in the
/// coordinator holds them to it bit for bit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankingProfile {
    /// Which preset this is, or was derived from: a label for telemetry and the result, not
    /// a parameter. Nothing ranks differently by it.
    pub profile: SearchProfile,
    /// How the two sides' scores are combined: weighted by alpha (`Weighted`, and
    /// `Adaptive`, which also min-max normalizes BM25 when its scores run high), or by rank
    /// (`RRF { k }`: `1 / (k + rank)` from each side, with the semantic threshold deciding
    /// only which semantic candidates take part, and no bonus or penalty applied). `k` must
    /// be at least 1.
    pub fusion_strategy: FusionStrategy,
    /// One alpha for every query, in place of [`Self::alpha_by_query_type`]. From 0 to 1.
    pub alpha_override: Option<f32>,
    /// The lexical weight for each query type, when [`Self::alpha_override`] is `None`.
    /// Absent from a profile serialized before it existed, so it defaults to the values
    /// that profile was ranked with.
    #[serde(default)]
    pub alpha_by_query_type: QueryTypeAlphas,
    /// `k` in BM25's normalization `score / (k + score)`; see
    /// [`DEFAULT_BM25_SATURATION_K`]. Positive. Defaulted, like
    /// [`Self::alpha_by_query_type`], when a serialized profile predates it.
    #[serde(default = "default_bm25_saturation_k")]
    pub bm25_saturation_k: f32,
    /// Below this normalized score, suppress semantic contribution.
    ///
    /// A cosine is mapped to `(cosine + 1) / 2`, so an unrelated passage — cosine 0 —
    /// scores 0.5; 0.55 is a cosine of about 0.1. From 0 to 1.
    pub semantic_threshold: f32,
    /// Added to a candidate both sides returned, in a hybrid search fused by weight. From 0
    /// to 1, like the other bonuses and the penalty below.
    pub agreement_bonus: f32,
    pub phrase_match_bonus: f32,
    pub rare_term_bonus: f32,
    pub section_coverage_bonus: f32,
    pub duplicate_penalty: f32,
    pub metadata_ranking_enabled: bool,
    /// How many semantic candidates to fetch per result on the page, from 1 to 10.
    pub candidate_window_multiplier: f32,
    pub query_cache_enabled: bool,
    pub embedding_cache_enabled: bool,
    pub telemetry_enabled: bool,
}

impl RankingProfile {
    /// Creates a ranking profile from a predefined search profile with standard defaults.
    pub fn from_profile(profile: SearchProfile) -> Self {
        match profile {
            SearchProfile::Fast => Self {
                profile,
                fusion_strategy: FusionStrategy::RRF { k: 60 },
                alpha_override: None,
                alpha_by_query_type: QueryTypeAlphas::default(),
                bm25_saturation_k: DEFAULT_BM25_SATURATION_K,
                semantic_threshold: 0.55,
                agreement_bonus: 0.05,
                phrase_match_bonus: 0.0,
                rare_term_bonus: 0.0,
                section_coverage_bonus: 0.0,
                duplicate_penalty: 0.0,
                metadata_ranking_enabled: false,
                candidate_window_multiplier: 1.5,
                query_cache_enabled: true,
                embedding_cache_enabled: true,
                telemetry_enabled: true,
            },
            SearchProfile::Balanced => Self {
                profile,
                fusion_strategy: FusionStrategy::Weighted,
                alpha_override: None,
                alpha_by_query_type: QueryTypeAlphas::default(),
                bm25_saturation_k: DEFAULT_BM25_SATURATION_K,
                // Preserve the pre-profile ranking contract for callers that
                // do not select a profile explicitly.
                semantic_threshold: 0.0,
                agreement_bonus: 0.10,
                phrase_match_bonus: 0.0,
                rare_term_bonus: 0.0,
                section_coverage_bonus: 0.0,
                duplicate_penalty: 0.0,
                metadata_ranking_enabled: false,
                candidate_window_multiplier: 2.0,
                query_cache_enabled: true,
                embedding_cache_enabled: true,
                telemetry_enabled: true,
            },
            SearchProfile::Best => Self {
                profile,
                fusion_strategy: FusionStrategy::Adaptive,
                alpha_override: None,
                alpha_by_query_type: QueryTypeAlphas::default(),
                bm25_saturation_k: DEFAULT_BM25_SATURATION_K,
                semantic_threshold: 0.55,
                agreement_bonus: 0.12,
                phrase_match_bonus: 0.10,
                rare_term_bonus: 0.05,
                section_coverage_bonus: 0.03,
                duplicate_penalty: 0.08,
                metadata_ranking_enabled: true,
                candidate_window_multiplier: 3.0,
                query_cache_enabled: true,
                embedding_cache_enabled: true,
                telemetry_enabled: true,
            },
        }
    }

    /// Refuse a parameter the ranking is not defined for, naming it.
    ///
    /// Every preset passes. What this guards is a profile a host assembled — from a
    /// calibration run, a remote configuration, a settings screen — in which a NaN, a
    /// negative weight or a threshold past 1 would fail nothing: it would rank, silently and
    /// wrongly, and a calibration run is exactly where nobody can tell a bad ranking from a
    /// bad parameter. So nothing is clamped and nothing is substituted, unlike the feature
    /// flags' overrides, which predate this and keep their behaviour.
    ///
    /// The ranges are the ones the ranking already assumes, which it used to enforce by
    /// quietly substituting: RRF ran with `k.max(1)`, alpha and the threshold were clamped
    /// into `[0, 1]`, the window multiplier into `[1, 10]`, a negative bonus counted as 0.
    /// The bonuses and the penalty are held to `[0, 1]` as the feature flags' overrides
    /// are, because the scores they are added to lie in `[0, 1]`: a bonus of 10 is a
    /// percentage typed as a fraction, not a weight. BM25's `k` must be positive, or
    /// `score / (k + score)` is 1 for every score or not a normalization at all.
    ///
    /// The error is [`SemanticSearchError::InvalidRankingParameter`], naming the parameter
    /// as a field path — `alpha_by_query_type.short`, `fusion_strategy.k`.
    pub fn validate(&self) -> Result<(), SemanticSearchError> {
        if let FusionStrategy::RRF { k: 0 } = self.fusion_strategy {
            return Err(SemanticSearchError::InvalidRankingParameter {
                parameter: "fusion_strategy.k",
                value: "0".to_string(),
                requirement: "at least 1",
            });
        }
        if let Some(alpha) = self.alpha_override {
            require_unit("alpha_override", alpha)?;
        }
        let alphas = &self.alpha_by_query_type;
        for (parameter, alpha) in [
            ("alpha_by_query_type.quoted_phrase", alphas.quoted_phrase),
            (
                "alpha_by_query_type.exact_reference",
                alphas.exact_reference,
            ),
            ("alpha_by_query_type.short", alphas.short),
            ("alpha_by_query_type.mixed", alphas.mixed),
            ("alpha_by_query_type.conceptual", alphas.conceptual),
            ("alpha_by_query_type.unknown", alphas.unknown),
        ] {
            require_unit(parameter, alpha)?;
        }
        require(
            "bm25_saturation_k",
            self.bm25_saturation_k,
            self.bm25_saturation_k > 0.0,
            "a positive number",
        )?;
        for (parameter, value) in [
            ("semantic_threshold", self.semantic_threshold),
            ("agreement_bonus", self.agreement_bonus),
            ("phrase_match_bonus", self.phrase_match_bonus),
            ("rare_term_bonus", self.rare_term_bonus),
            ("section_coverage_bonus", self.section_coverage_bonus),
            ("duplicate_penalty", self.duplicate_penalty),
        ] {
            require_unit(parameter, value)?;
        }
        require(
            "candidate_window_multiplier",
            self.candidate_window_multiplier,
            (1.0..=10.0).contains(&self.candidate_window_multiplier),
            "a number from 1 to 10",
        )?;
        Ok(())
    }
}

impl Default for RankingProfile {
    fn default() -> Self {
        Self::from_profile(SearchProfile::Balanced)
    }
}

/// `Err` naming `parameter` unless `value` is a finite number for which `holds` is true.
fn require(
    parameter: &'static str,
    value: f32,
    holds: bool,
    requirement: &'static str,
) -> Result<(), SemanticSearchError> {
    if value.is_finite() && holds {
        return Ok(());
    }
    Err(SemanticSearchError::InvalidRankingParameter {
        parameter,
        value: value.to_string(),
        requirement,
    })
}

fn require_unit(parameter: &'static str, value: f32) -> Result<(), SemanticSearchError> {
    require(
        parameter,
        value,
        (0.0..=1.0).contains(&value),
        "a number from 0 to 1",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_defaults() {
        let fast = RankingProfile::from_profile(SearchProfile::Fast);
        assert_eq!(fast.fusion_strategy, FusionStrategy::RRF { k: 60 });
        assert_eq!(fast.semantic_threshold, 0.55);
        assert!(!fast.metadata_ranking_enabled);

        let balanced = RankingProfile::default();
        assert_eq!(balanced.profile, SearchProfile::Balanced);
        assert_eq!(balanced.fusion_strategy, FusionStrategy::Weighted);
        assert_eq!(balanced.semantic_threshold, 0.0);

        let best = RankingProfile::from_profile(SearchProfile::Best);
        assert_eq!(best.fusion_strategy, FusionStrategy::Adaptive);
        assert_eq!(best.semantic_threshold, 0.55);
        assert!(best.metadata_ranking_enabled);
    }

    #[test]
    fn test_display() {
        assert_eq!(SearchProfile::Fast.to_string(), "Fast");
        assert_eq!(SearchProfile::Balanced.to_string(), "Balanced");
        assert_eq!(SearchProfile::Best.to_string(), "Best");

        assert_eq!(FusionStrategy::Weighted.to_string(), "Weighted");
        assert_eq!(FusionStrategy::RRF { k: 60 }.to_string(), "RRF(k=60)");
        assert_eq!(FusionStrategy::Adaptive.to_string(), "Adaptive");
    }

    /// Becoming parameters changed none of the numbers: every preset carries the constants
    /// the ranking has always used, and passes its own validation.
    #[test]
    fn the_new_parameters_default_to_the_numbers_the_ranking_always_used() {
        assert_eq!(DEFAULT_BM25_SATURATION_K, 10.0);
        assert_eq!(
            QueryTypeAlphas::default(),
            QueryTypeAlphas {
                quoted_phrase: 1.0,
                exact_reference: 0.85,
                short: 0.7,
                mixed: 0.5,
                conceptual: 0.3,
                unknown: 0.5,
            }
        );
        for preset in [
            SearchProfile::Fast,
            SearchProfile::Balanced,
            SearchProfile::Best,
        ] {
            let profile = RankingProfile::from_profile(preset);
            assert_eq!(profile.alpha_by_query_type, QueryTypeAlphas::default());
            assert_eq!(profile.bm25_saturation_k, DEFAULT_BM25_SATURATION_K);
            assert!(profile.alpha_override.is_none());
            assert!(profile.validate().is_ok(), "{preset}");
        }
    }

    /// Each parameter out of its range is refused, and the refusal names that parameter —
    /// a host tuning twenty numbers has to be told which one is wrong.
    #[test]
    fn a_parameter_the_ranking_is_not_defined_for_is_refused_by_name() {
        type Adjust = fn(&mut RankingProfile);
        let cases: [(&str, Adjust); 26] = [
            ("fusion_strategy.k", |p| {
                p.fusion_strategy = FusionStrategy::RRF { k: 0 }
            }),
            ("alpha_override", |p| p.alpha_override = Some(f32::NAN)),
            ("alpha_override", |p| p.alpha_override = Some(1.5)),
            ("alpha_override", |p| p.alpha_override = Some(-0.1)),
            ("alpha_by_query_type.quoted_phrase", |p| {
                p.alpha_by_query_type.quoted_phrase = f32::INFINITY
            }),
            ("alpha_by_query_type.exact_reference", |p| {
                p.alpha_by_query_type.exact_reference = 1.01
            }),
            ("alpha_by_query_type.short", |p| {
                p.alpha_by_query_type.short = -0.2
            }),
            ("alpha_by_query_type.mixed", |p| {
                p.alpha_by_query_type.mixed = 2.0
            }),
            ("alpha_by_query_type.conceptual", |p| {
                p.alpha_by_query_type.conceptual = f32::NAN
            }),
            ("alpha_by_query_type.unknown", |p| {
                p.alpha_by_query_type.unknown = f32::NEG_INFINITY
            }),
            ("bm25_saturation_k", |p| p.bm25_saturation_k = 0.0),
            ("bm25_saturation_k", |p| p.bm25_saturation_k = -10.0),
            ("bm25_saturation_k", |p| p.bm25_saturation_k = f32::NAN),
            ("bm25_saturation_k", |p| p.bm25_saturation_k = f32::INFINITY),
            ("semantic_threshold", |p| p.semantic_threshold = 1.01),
            ("semantic_threshold", |p| p.semantic_threshold = f32::NAN),
            ("semantic_threshold", |p| p.semantic_threshold = -0.5),
            ("agreement_bonus", |p| p.agreement_bonus = -0.1),
            ("agreement_bonus", |p| p.agreement_bonus = 10.0),
            ("phrase_match_bonus", |p| p.phrase_match_bonus = f32::NAN),
            ("rare_term_bonus", |p| p.rare_term_bonus = -1.0),
            ("section_coverage_bonus", |p| p.section_coverage_bonus = 1.5),
            ("duplicate_penalty", |p| p.duplicate_penalty = f32::INFINITY),
            ("candidate_window_multiplier", |p| {
                p.candidate_window_multiplier = 0.5
            }),
            ("candidate_window_multiplier", |p| {
                p.candidate_window_multiplier = 11.0
            }),
            ("candidate_window_multiplier", |p| {
                p.candidate_window_multiplier = f32::NAN
            }),
        ];

        for (expected, adjust) in cases {
            let mut profile = RankingProfile::default();
            adjust(&mut profile);
            match profile.validate() {
                Err(SemanticSearchError::InvalidRankingParameter { parameter, .. }) => {
                    assert_eq!(parameter, expected)
                }
                other => panic!("{expected}: expected a refusal naming it, got {other:?}"),
            }
        }

        let mut profile = RankingProfile::default();
        profile.alpha_by_query_type.short = -0.2;
        assert_eq!(
            profile.validate().unwrap_err().to_string(),
            "Ranking parameter alpha_by_query_type.short is -0.2, and it must be a number \
             from 0 to 1"
        );
    }

    /// The ends of every range are values a calibration may legitimately land on.
    #[test]
    fn the_ends_of_every_range_are_accepted() {
        let alphas = QueryTypeAlphas {
            quoted_phrase: 0.0,
            exact_reference: 1.0,
            short: 0.0,
            mixed: 1.0,
            conceptual: 0.0,
            unknown: 1.0,
        };
        for (zero, one) in [(0.0, 1.0), (1.0, 0.0)] {
            let profile = RankingProfile {
                fusion_strategy: FusionStrategy::RRF { k: 1 },
                alpha_override: Some(zero),
                alpha_by_query_type: alphas,
                bm25_saturation_k: f32::MIN_POSITIVE,
                semantic_threshold: one,
                agreement_bonus: zero,
                phrase_match_bonus: one,
                rare_term_bonus: zero,
                section_coverage_bonus: one,
                duplicate_penalty: zero,
                candidate_window_multiplier: 1.0 + 9.0 * one,
                ..RankingProfile::from_profile(SearchProfile::Best)
            };
            assert!(profile.validate().is_ok(), "{profile:?}");
        }
    }

    /// A profile serialized before the two parameters existed reads back as the profile it
    /// was, ranking as it did; and a host may send only the query types it tunes.
    #[test]
    fn a_profile_serialized_before_the_new_parameters_reads_back_unchanged() {
        for preset in [
            SearchProfile::Fast,
            SearchProfile::Balanced,
            SearchProfile::Best,
        ] {
            let mut json = serde_json::to_value(RankingProfile::from_profile(preset)).unwrap();
            let object = json.as_object_mut().unwrap();
            assert!(object.remove("alpha_by_query_type").is_some());
            assert!(object.remove("bm25_saturation_k").is_some());

            let read: RankingProfile = serde_json::from_value(json).unwrap();
            assert_eq!(read, RankingProfile::from_profile(preset));
        }

        let alphas: QueryTypeAlphas = serde_json::from_str(r#"{"short": 0.6}"#).unwrap();
        assert_eq!(
            alphas,
            QueryTypeAlphas {
                short: 0.6,
                ..QueryTypeAlphas::default()
            }
        );
    }
}
