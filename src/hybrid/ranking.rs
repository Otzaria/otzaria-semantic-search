use std::fmt;

/// Type of query based on heuristics
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryType {
    ExactReference,
    Conceptual,
    Mixed,
    Short,
    Unknown,
}

impl fmt::Display for QueryType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            QueryType::ExactReference => "ExactReference",
            QueryType::Conceptual => "Conceptual",
            QueryType::Mixed => "Mixed",
            QueryType::Short => "Short",
            QueryType::Unknown => "Unknown",
        };
        write!(f, "{s}")
    }
}

/// Extracted features from the query string
#[derive(Debug, Clone)]
pub struct QueryFeatures {
    pub token_count: usize,
    pub has_quoted_phrase: bool,
    pub avg_token_length: f32,
    pub estimated_type: QueryType,
}

/// Analyzes a query string to extract features useful for ranking
pub fn analyze_query(query: &str) -> QueryFeatures {
    let has_quoted_phrase = query.contains('"');
    let tokens: Vec<&str> = query.split_whitespace().collect();
    let token_count = tokens.len();

    let total_chars: usize = tokens.iter().map(|t| t.chars().count()).sum();
    let avg_token_length = if token_count > 0 {
        total_chars as f32 / token_count as f32
    } else {
        0.0
    };

    // Basic heuristic for exact references: containing numbers or specific format
    let has_numbers = tokens.iter().any(|t| t.chars().any(|c| c.is_ascii_digit()));

    let estimated_type = if token_count == 0 {
        QueryType::Unknown
    } else if has_quoted_phrase {
        QueryType::ExactReference
    } else if token_count <= 2 {
        if has_numbers {
            QueryType::ExactReference
        } else {
            QueryType::Short
        }
    } else if token_count >= 5 {
        QueryType::Conceptual
    } else {
        QueryType::Mixed
    };

    QueryFeatures {
        token_count,
        has_quoted_phrase,
        avg_token_length,
        estimated_type,
    }
}

/// Computes the alpha weight for lexical search (1 - alpha for semantic)
/// Short/exact -> 0.7-0.9
/// Conceptual -> 0.2-0.4
/// Mixed -> 0.5
pub fn compute_alpha(features: &QueryFeatures) -> f32 {
    match features.estimated_type {
        QueryType::ExactReference => 0.8,
        QueryType::Short => 0.7,
        QueryType::Mixed => 0.5,
        QueryType::Conceptual => 0.3,
        QueryType::Unknown => 0.5,
    }
}

/// Configuration for bonuses and penalties during ranking.
///
/// These values are unmeasured heuristics. Calibrating them — or dropping
/// weighted fusion for RRF, where they would not apply — is roadmap P5.
#[derive(Debug, Clone)]
pub struct BonusConfig {
    /// Added to a candidate that both engines returned.
    ///
    /// Named for what it actually measures: agreement between the two
    /// retrieval paths, not an exact string match. Note that it can push a
    /// fused score above 1.0, so a fused score is a ranking key, not a
    /// normalized confidence.
    pub agreement_bonus: f32,
    /// Reserved for penalizing near-duplicate results. Not applied yet —
    /// duplicates are currently handled by grouping
    /// ([`GroupingMode::IdenticalText`](crate::semantic::types::GroupingMode)).
    pub duplicate_penalty: f32,
    /// Reserved for rewarding a result whose section has several hits. Not
    /// applied yet.
    pub section_coverage_bonus: f32,
}

impl Default for BonusConfig {
    fn default() -> Self {
        Self {
            agreement_bonus: 0.1,
            duplicate_penalty: 0.05,
            section_coverage_bonus: 0.03,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_short_query() {
        let features = analyze_query("שלום עולם");
        assert_eq!(features.estimated_type, QueryType::Short);
        let alpha = compute_alpha(&features);
        assert!((0.7..=0.9).contains(&alpha));
    }

    #[test]
    fn test_long_conceptual_query() {
        let features = analyze_query("מה המשמעות של החיים ביקום לפי הקבלה");
        assert_eq!(features.estimated_type, QueryType::Conceptual);
        let alpha = compute_alpha(&features);
        assert!((0.2..=0.4).contains(&alpha));
    }

    #[test]
    fn test_quoted_phrase() {
        let features = analyze_query("\"בראשית ברא\"");
        assert_eq!(features.estimated_type, QueryType::ExactReference);
        assert!(features.has_quoted_phrase);
        let alpha = compute_alpha(&features);
        assert!((0.7..=0.9).contains(&alpha));
    }

    #[test]
    fn test_analyze_query_tokenization() {
        let features = analyze_query("א ב ג");
        assert_eq!(features.token_count, 3);
        assert_eq!(features.avg_token_length, 1.0);
    }

    #[test]
    fn test_default_bonus_config() {
        let config = BonusConfig::default();
        assert_eq!(config.agreement_bonus, 0.1);
        assert_eq!(config.duplicate_penalty, 0.05);
        assert_eq!(config.section_coverage_bonus, 0.03);
    }

    #[test]
    fn an_empty_query_is_classified_without_dividing_by_zero() {
        let features = analyze_query("");
        assert_eq!(features.token_count, 0);
        assert_eq!(features.avg_token_length, 0.0);
        assert_eq!(features.estimated_type, QueryType::Unknown);
        assert_eq!(compute_alpha(&features), 0.5);
    }

    #[test]
    fn every_query_type_produces_a_weight_inside_the_unit_interval() {
        for query in [
            "",
            "שלום",
            "שלום עולם",
            "\"בראשית ברא\"",
            "ברכות דף כ",
            "מה המשמעות של החיים ביקום לפי הקבלה",
        ] {
            let alpha = compute_alpha(&analyze_query(query));
            assert!(
                (0.0..=1.0).contains(&alpha),
                "query {query:?} produced alpha {alpha}"
            );
        }
    }

    #[test]
    fn query_types_render_for_logging() {
        for query_type in [
            QueryType::ExactReference,
            QueryType::Conceptual,
            QueryType::Mixed,
            QueryType::Short,
            QueryType::Unknown,
        ] {
            assert!(!query_type.to_string().is_empty());
        }
    }
}
