use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Configuration for metadata-based ranking signals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataRankingConfig {
    pub primary_source_bonus: f32,
    pub era_affinity_bonus: f32,
    pub category_match_bonus: f32,
    pub enabled: bool,
}

impl Default for MetadataRankingConfig {
    fn default() -> Self {
        Self {
            primary_source_bonus: 0.03,
            era_affinity_bonus: 0.02,
            category_match_bonus: 0.02,
            enabled: true,
        }
    }
}

/// Computed metadata signals for a specific document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataSignal {
    pub primary_source_bonus: f32,
    pub era_affinity_bonus: f32,
    pub category_match_bonus: f32,
    pub total: f32,
}

/// Computes metadata ranking bonuses based on document facets and query context.
#[derive(Debug, Clone)]
pub struct MetadataRanker {
    pub config: MetadataRankingConfig,
}

impl MetadataRanker {
    pub fn new(config: MetadataRankingConfig) -> Self {
        Self { config }
    }
}

impl Default for MetadataRanker {
    fn default() -> Self {
        Self::new(MetadataRankingConfig::default())
    }
}

impl MetadataRanker {
    pub fn compute_signal(
        &self,
        file_path: &str,
        facets: &[String],
        query_facets: &[String],
    ) -> MetadataSignal {
        if !self.config.enabled {
            return MetadataSignal {
                primary_source_bonus: 0.0,
                era_affinity_bonus: 0.0,
                category_match_bonus: 0.0,
                total: 0.0,
            };
        }

        let primary_source_bonus = self.compute_primary_source_bonus(file_path);
        let era_affinity_bonus = self.compute_era_affinity_bonus(facets, query_facets);
        let category_match_bonus = self.compute_category_match_bonus(facets, query_facets);

        MetadataSignal {
            primary_source_bonus,
            era_affinity_bonus,
            category_match_bonus,
            total: primary_source_bonus + era_affinity_bonus + category_match_bonus,
        }
    }

    fn compute_primary_source_bonus(&self, file_path: &str) -> f32 {
        let lower_path = file_path.to_lowercase();
        let primary_keywords = [
            "תנ\"ך",
            "תורה",
            "משנה",
            "תלמוד",
            "tanach",
            "mishna",
            "talmud",
        ];

        for keyword in primary_keywords.iter() {
            if lower_path.contains(keyword) {
                return self.config.primary_source_bonus;
            }
        }
        0.0
    }

    fn compute_era_affinity_bonus(&self, _facets: &[String], _query_facets: &[String]) -> f32 {
        // Placeholder for era affinity computation (future enhancement)
        // Currently returns 0.0 as it requires specific era categorization logic
        0.0
    }

    fn compute_category_match_bonus(&self, facets: &[String], query_facets: &[String]) -> f32 {
        if query_facets.is_empty() || facets.is_empty() {
            return 0.0;
        }

        let query_set: HashSet<&String> = query_facets.iter().collect();
        let match_count = facets.iter().filter(|f| query_set.contains(f)).count();

        if match_count > 0 {
            self.config.category_match_bonus
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_primary_source_detection() {
        let ranker = MetadataRanker::new(MetadataRankingConfig::default());

        let signal = ranker.compute_signal("C:/library/תלמוד/בבלי/ברכות.txt", &[], &[]);
        assert_eq!(signal.primary_source_bonus, 0.03);

        let signal2 = ranker.compute_signal("C:/library/some_modern_book.txt", &[], &[]);
        assert_eq!(signal2.primary_source_bonus, 0.0);
    }

    #[test]
    fn test_category_matching() {
        let ranker = MetadataRanker::new(MetadataRankingConfig::default());

        let facets = vec!["halacha".to_string(), "rambam".to_string()];
        let query_facets = vec!["halacha".to_string()];

        let signal = ranker.compute_signal("path", &facets, &query_facets);
        assert_eq!(signal.category_match_bonus, 0.02);

        let query_facets_no_match = vec!["aggada".to_string()];
        let signal2 = ranker.compute_signal("path", &facets, &query_facets_no_match);
        assert_eq!(signal2.category_match_bonus, 0.0);
    }

    #[test]
    fn test_disabled_config() {
        let mut config = MetadataRankingConfig::default();
        config.enabled = false;
        let ranker = MetadataRanker::new(config);

        let signal =
            ranker.compute_signal("תלמוד", &["halacha".to_string()], &["halacha".to_string()]);
        assert_eq!(signal.total, 0.0);
        assert_eq!(signal.primary_source_bonus, 0.0);
    }
}
