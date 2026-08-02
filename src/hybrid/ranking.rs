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

#[derive(Debug, Clone, Default)]
pub struct RankingSignals {
    pub agreement_bonus: f32,
    pub section_coverage_bonus: f32,
    pub duplicate_penalty: f32,
    pub rare_term_bonus: f32,
    pub phrase_match_bonus: f32,
    pub metadata_bonus: f32,
    pub total: f32,
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
    pub has_numbers: bool,
    pub detected_language: String,
    pub rare_tokens: Vec<String>,
    pub quoted_phrases: Vec<String>,
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

    let mut hebrew_chars = 0;
    let mut latin_chars = 0;
    let mut rare_tokens = Vec::new();
    
    for t in &tokens {
        let char_count = t.chars().count();
        for c in t.chars() {
            if c.is_ascii_alphabetic() {
                latin_chars += 1;
            } else if c >= '\u{0590}' && c <= '\u{05FF}' {
                hebrew_chars += 1;
            }
        }
        if char_count > 5 {
            rare_tokens.push(t.to_string());
        }
    }
    
    let detected_language = if hebrew_chars > 0 && latin_chars == 0 {
        "hebrew".to_string()
    } else if latin_chars > 0 && hebrew_chars == 0 {
        "other".to_string()
    } else if hebrew_chars > 0 && latin_chars > 0 {
        "mixed".to_string()
    } else {
        "other".to_string()
    };
    
    let mut quoted_phrases = Vec::new();
    let mut in_quotes = false;
    let mut current_phrase = String::new();
    for c in query.chars() {
        if c == '"' {
            if in_quotes {
                if !current_phrase.trim().is_empty() {
                    quoted_phrases.push(current_phrase.trim().to_string());
                }
                current_phrase.clear();
                in_quotes = false;
            } else {
                in_quotes = true;
            }
        } else if in_quotes {
            current_phrase.push(c);
        }
    }

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
        has_numbers,
        detected_language,
        rare_tokens,
        quoted_phrases,
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

pub fn compute_phrase_match_bonus(text: &str, phrases: &[String]) -> f32 {
    if phrases.is_empty() {
        return 0.0;
    }
    let mut matches = 0;
    for phrase in phrases {
        if text.contains(phrase) {
            matches += 1;
        }
    }
    (matches as f32) / (phrases.len() as f32)
}

pub fn compute_rare_term_bonus(text: &str, rare_tokens: &[String]) -> f32 {
    if rare_tokens.is_empty() {
        return 0.0;
    }
    let mut matches = 0;
    for token in rare_tokens {
        if text.contains(token) {
            matches += 1;
        }
    }
    (matches as f32) / (rare_tokens.len() as f32)
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

    #[test]
    fn test_compute_phrase_match_bonus() {
        let text = "בראשית ברא אלהים את השמים ואת הארץ";
        let phrases = vec!["בראשית ברא".to_string(), "השמים ואת".to_string()];
        let bonus = compute_phrase_match_bonus(text, &phrases);
        assert_eq!(bonus, 1.0);
        
        let phrases_partial = vec!["בראשית ברא".to_string(), "לא קיים".to_string()];
        let bonus_partial = compute_phrase_match_bonus(text, &phrases_partial);
        assert_eq!(bonus_partial, 0.5);
    }

    #[test]
    fn test_compute_rare_term_bonus() {
        let text = "המולקולה מסובכת";
        let rare_tokens = vec!["המולקולה".to_string()];
        let bonus = compute_rare_term_bonus(text, &rare_tokens);
        assert_eq!(bonus, 1.0);
    }
}
