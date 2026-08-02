use crate::config::profiles::{FusionStrategy, RankingProfile, SearchProfile};
use serde::{Deserialize, Serialize};

/// Fine-grained feature flags to optionally override tuning parameters.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct FeatureFlags {
    pub force_rrf: Option<bool>,
    pub force_weighted: Option<bool>,
    pub rrf_k: Option<u32>,
    pub bm25_adaptive_normalization: Option<bool>,
    pub semantic_threshold_override: Option<f32>,
    pub agreement_bonus_override: Option<f32>,
    pub phrase_match_enabled: Option<bool>,
    pub phrase_match_bonus_override: Option<f32>,
    pub rare_term_enabled: Option<bool>,
    pub rare_term_bonus_override: Option<f32>,
    pub section_coverage_enabled: Option<bool>,
    pub duplicate_penalty_enabled: Option<bool>,
    pub metadata_ranking_enabled: Option<bool>,
    pub query_cache_enabled: Option<bool>,
    pub query_cache_capacity: Option<usize>,
    pub embedding_cache_enabled: Option<bool>,
    pub telemetry_enabled: Option<bool>,
    pub telemetry_per_query: Option<bool>,
}

impl FeatureFlags {
    /// Apply any explicitly set feature flags as overrides onto a `RankingProfile`.
    pub fn apply(&self, profile: &mut RankingProfile) {
        if let Some(true) = self.force_rrf {
            let k = self.rrf_k.unwrap_or(60);
            profile.fusion_strategy = FusionStrategy::RRF { k };
        } else if let Some(true) = self.force_weighted {
            profile.fusion_strategy = FusionStrategy::Weighted;
        }

        if let Some(threshold) = self.semantic_threshold_override {
            profile.semantic_threshold = threshold;
        }

        if let Some(bonus) = self.agreement_bonus_override {
            profile.agreement_bonus = bonus;
        }

        if let Some(false) = self.phrase_match_enabled {
            profile.phrase_match_bonus = 0.0;
        } else if let Some(bonus) = self.phrase_match_bonus_override {
            profile.phrase_match_bonus = bonus;
        }

        if let Some(false) = self.rare_term_enabled {
            profile.rare_term_bonus = 0.0;
        } else if let Some(bonus) = self.rare_term_bonus_override {
            profile.rare_term_bonus = bonus;
        }

        if let Some(false) = self.section_coverage_enabled {
            profile.section_coverage_bonus = 0.0;
        }

        if let Some(false) = self.duplicate_penalty_enabled {
            profile.duplicate_penalty = 0.0;
        }

        if let Some(enabled) = self.metadata_ranking_enabled {
            profile.metadata_ranking_enabled = enabled;
        }

        if let Some(enabled) = self.query_cache_enabled {
            profile.query_cache_enabled = enabled;
        }

        if let Some(enabled) = self.embedding_cache_enabled {
            profile.embedding_cache_enabled = enabled;
        }

        if let Some(enabled) = self.telemetry_enabled {
            profile.telemetry_enabled = enabled;
        }

        if let Some(true) = self.bm25_adaptive_normalization {
            // Adaptive normalization implicitly enforces adaptive fusion strategy if weighted was selected
            if profile.fusion_strategy == FusionStrategy::Weighted {
                profile.fusion_strategy = FusionStrategy::Adaptive;
            }
        }
    }

    /// Convenience method to create a profile from a `SearchProfile` and apply these overrides.
    pub fn resolve(profile: SearchProfile, flags: &FeatureFlags) -> RankingProfile {
        let mut base_profile = RankingProfile::from_profile(profile);
        flags.apply(&mut base_profile);
        base_profile
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::profiles::SearchProfile;

    #[test]
    fn test_resolve_empty_flags() {
        let flags = FeatureFlags::default();
        let profile = FeatureFlags::resolve(SearchProfile::Balanced, &flags);
        assert_eq!(
            profile,
            RankingProfile::from_profile(SearchProfile::Balanced)
        );
    }

    #[test]
    fn test_apply_overrides() {
        let mut flags = FeatureFlags::default();
        flags.force_rrf = Some(true);
        flags.rrf_k = Some(100);
        flags.semantic_threshold_override = Some(0.9);
        flags.phrase_match_enabled = Some(false);
        flags.metadata_ranking_enabled = Some(true);

        let mut profile = RankingProfile::from_profile(SearchProfile::Balanced);
        flags.apply(&mut profile);

        assert_eq!(profile.fusion_strategy, FusionStrategy::RRF { k: 100 });
        assert_eq!(profile.semantic_threshold, 0.9);
        assert_eq!(profile.phrase_match_bonus, 0.0);
        assert!(profile.metadata_ranking_enabled);
    }

    #[test]
    fn test_resolve_with_partial_overrides() {
        let flags = FeatureFlags {
            force_weighted: Some(true),
            agreement_bonus_override: Some(0.5),
            ..Default::default()
        };

        let profile = FeatureFlags::resolve(SearchProfile::Fast, &flags);
        assert_eq!(profile.fusion_strategy, FusionStrategy::Weighted);
        assert_eq!(profile.agreement_bonus, 0.5);
        assert_eq!(profile.candidate_window_multiplier, 1.5); // Remains unchanged
    }
}
