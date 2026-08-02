use serde::{Deserialize, Serialize};
use std::fmt;

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
            FusionStrategy::RRF { k } => write!(f, "RRF(k={})", k),
            FusionStrategy::Adaptive => write!(f, "Adaptive"),
        }
    }
}

/// The complete set of tuning parameters for hybrid ranking.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankingProfile {
    pub profile: SearchProfile,
    pub fusion_strategy: FusionStrategy,
    pub alpha_override: Option<f32>,
    /// Below this normalized score, suppress semantic contribution.
    pub semantic_threshold: f32,
    pub agreement_bonus: f32,
    pub phrase_match_bonus: f32,
    pub rare_term_bonus: f32,
    pub section_coverage_bonus: f32,
    pub duplicate_penalty: f32,
    pub metadata_ranking_enabled: bool,
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
                semantic_threshold: 0.5,
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
                semantic_threshold: 0.3,
                agreement_bonus: 0.10,
                phrase_match_bonus: 0.08,
                rare_term_bonus: 0.0,
                section_coverage_bonus: 0.0,
                duplicate_penalty: 0.05,
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
                semantic_threshold: 0.2,
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
}

impl Default for RankingProfile {
    fn default() -> Self {
        Self::from_profile(SearchProfile::Balanced)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_defaults() {
        let fast = RankingProfile::from_profile(SearchProfile::Fast);
        assert_eq!(fast.fusion_strategy, FusionStrategy::RRF { k: 60 });
        assert_eq!(fast.semantic_threshold, 0.5);
        assert!(!fast.metadata_ranking_enabled);

        let balanced = RankingProfile::default();
        assert_eq!(balanced.profile, SearchProfile::Balanced);
        assert_eq!(balanced.fusion_strategy, FusionStrategy::Weighted);
        assert_eq!(balanced.semantic_threshold, 0.3);

        let best = RankingProfile::from_profile(SearchProfile::Best);
        assert_eq!(best.fusion_strategy, FusionStrategy::Adaptive);
        assert_eq!(best.semantic_threshold, 0.2);
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
}
