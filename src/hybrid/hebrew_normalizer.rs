use std::borrow::Cow;

/// Represents the detected language of a query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryLanguage {
    Hebrew,
    Aramaic,
    Mixed,
    Other,
}

/// Stateless utility for normalizing Hebrew and Aramaic text.
pub struct HebrewNormalizer;

impl HebrewNormalizer {
    /// Normalizes text for semantic embedding by stripping diacritics and compacting whitespace.
    pub fn normalize_for_embedding(&self, query: &str) -> String {
        let mut text = strip_nikud(query);
        text = strip_taamim(&text);

        // Normalize geresh/gershayim
        let mut normalized = String::with_capacity(text.len());
        let mut chars = text.chars().peekable();

        while let Some(c) = chars.next() {
            if c == '׳' {
                // U+05F3
                normalized.push('\'');
            } else if c == '״' {
                // U+05F4
                normalized.push('"');
            } else {
                normalized.push(c);
            }
        }

        // Collapse whitespace and trim
        let words: Vec<&str> = normalized.split_whitespace().collect();
        words.join(" ")
    }

    /// Normalizes text for cache lookups (same as embedding but lowercase ASCII for mixed queries).
    pub fn normalize_for_cache(&self, query: &str) -> String {
        let embedding_normalized = self.normalize_for_embedding(query);
        embedding_normalized.to_ascii_lowercase()
    }

    /// Detects the primary language of the text.
    pub fn detect_language(&self, text: &str) -> QueryLanguage {
        if text.trim().is_empty() {
            return QueryLanguage::Other;
        }

        let mut hebrew_chars = 0;
        let mut aramaic_indicators = 0;
        let mut latin_chars = 0;
        let mut total_alpha = 0;

        for c in text.chars() {
            if c.is_alphabetic() {
                total_alpha += 1;

                // Hebrew/Aramaic block
                if c >= '\u{05D0}' && c <= '\u{05EA}' {
                    hebrew_chars += 1;

                    // Simple heuristic for Aramaic suffixes (approximate since characters are identical)
                    // e.g., ending in Aleph or containing specific patterns
                    // For character-by-character analysis it's hard, we'll look for characteristic
                    // words below instead, but count block chars here.
                } else if c.is_ascii_alphabetic() {
                    latin_chars += 1;
                }
            }
        }

        if total_alpha == 0 {
            return QueryLanguage::Other;
        }

        // Check for Aramaic specific common words/suffixes
        let words: Vec<&str> = text.split_whitespace().collect();
        for word in words {
            if word.ends_with('א') && word.len() > 2 {
                aramaic_indicators += 1;
            }
            if word == "די" || word == "קא" || word == "לאו" || word == "הכי" || word == "האי"
            {
                aramaic_indicators += 1;
            }
        }

        let hebrew_ratio = hebrew_chars as f32 / total_alpha as f32;
        let latin_ratio = latin_chars as f32 / total_alpha as f32;

        if latin_ratio > 0.0 && hebrew_ratio > 0.0 {
            QueryLanguage::Mixed
        } else if hebrew_ratio > 0.8 {
            if aramaic_indicators > 1
                || (aramaic_indicators > 0 && text.split_whitespace().count() <= 3)
            {
                QueryLanguage::Aramaic
            } else {
                QueryLanguage::Hebrew
            }
        } else if latin_ratio > 0.8 {
            QueryLanguage::Other
        } else {
            QueryLanguage::Other
        }
    }
}

/// Strips Hebrew nikud (vowels) from the given text.
pub fn strip_nikud(text: &str) -> String {
    text.chars()
        .filter(|&c| {
            // Niqqud Unicode block: U+0591 to U+05C7
            // Exclude non-nikud in this range if any, but standard nikud are mostly:
            // U+05B0 to U+05BD, U+05BF, U+05C1 to U+05C2, U+05C4 to U+05C5, U+05C7
            !((c >= '\u{0591}' && c <= '\u{05BD}')
                || c == '\u{05BF}'
                || (c >= '\u{05C1}' && c <= '\u{05C2}')
                || (c >= '\u{05C4}' && c <= '\u{05C5}')
                || c == '\u{05C7}')
        })
        .collect()
}

/// Strips Hebrew cantillation marks (taamim) from the given text.
pub fn strip_taamim(text: &str) -> String {
    text.chars()
        .filter(|&c| {
            // Taamim Unicode block: U+0591 to U+05AF
            !(c >= '\u{0591}' && c <= '\u{05AF}')
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_nikud() {
        assert_eq!(strip_nikud("שָׁלוֹם"), "שלום");
        assert_eq!(strip_nikud("בְּרֵאשִׁית"), "בראשית");
        assert_eq!(strip_nikud("מַיִם"), "מים");
    }

    #[test]
    fn test_strip_taamim() {
        // Includes cantillation
        assert_eq!(strip_taamim("בְּרֵאשִׁ֖ית בָּרָ֣א אֱלֹהִ֑ים"), "בְּרֵאשִׁית בָּרָא אֱלֹהִים");
    }

    #[test]
    fn test_geresh_normalization() {
        let normalizer = HebrewNormalizer;
        // Should convert ׳ to ' and ״ to "
        assert_eq!(normalizer.normalize_for_embedding("רש״י"), "רש\"י");
        assert_eq!(normalizer.normalize_for_embedding("צ׳יפס"), "צ'יפס");
        // Regular quotes should be preserved
        assert_eq!(normalizer.normalize_for_embedding("רש\"י"), "רש\"י");
    }

    #[test]
    fn test_mixed_text_normalization() {
        let normalizer = HebrewNormalizer;
        // Test multiple spaces and mixed text
        assert_eq!(
            normalizer.normalize_for_embedding("   hello   שָׁלוֹם   "),
            "hello שלום"
        );
        assert_eq!(
            normalizer.normalize_for_cache("   Hello   שָׁלוֹם   "),
            "hello שלום"
        );
    }

    #[test]
    fn test_language_detection() {
        let normalizer = HebrewNormalizer;

        assert_eq!(
            normalizer.detect_language("שלום עולם"),
            QueryLanguage::Hebrew
        );
        assert_eq!(
            normalizer.detect_language("hello world"),
            QueryLanguage::Other
        );
        assert_eq!(
            normalizer.detect_language("hello שלום"),
            QueryLanguage::Mixed
        );

        // Aramaic indicators
        assert_eq!(
            normalizer.detect_language("מאי קא משמע לן"),
            QueryLanguage::Aramaic
        );
        assert_eq!(
            normalizer.detect_language("האי דינא"),
            QueryLanguage::Aramaic
        );
    }

    #[test]
    fn test_empty_and_idempotent() {
        let normalizer = HebrewNormalizer;

        assert_eq!(normalizer.normalize_for_embedding(""), "");
        assert_eq!(normalizer.detect_language(""), QueryLanguage::Other);

        let already_normalized = "שלום עולם";
        assert_eq!(
            normalizer.normalize_for_embedding(already_normalized),
            already_normalized
        );
    }
}
