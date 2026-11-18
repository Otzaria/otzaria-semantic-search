//! Token-budget preflight for pending texts in embed.jsonl (warehouse reuse is absent).
use super::files::{malformed, FileDigest};
use super::plan::{read_embed_plan, EmbedManifest, EMBED_PLAN_FILE};
use crate::errors::{EmbeddingError, PackError};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OverBudgetPassage {
    pub record: u64,
    pub key: String,
    pub tokens: usize,
}

/// Full totals and a bounded sample of sixteen overflowing passages.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenBudgetReport {
    pub plan_sha256: String,
    pub tokenizer_sha256: String,
    pub max_tokens: usize,
    pub passages: u64,
    pub over_budget_passages: u64,
    pub discarded_tokens: u64,
    pub longest_passage_tokens: usize,
    pub examples: Vec<OverBudgetPassage>,
}

/// Verify identities and count tokens with truncation/padding disabled, including
/// prefix and special tokens. ONNX's PassageTokenCounter supplies the exact counter.
pub fn audit_plan_tokens(
    dir: &Path,
    tokenizer: &Path,
    mut count: impl FnMut(&str) -> Result<usize, EmbeddingError>,
) -> Result<TokenBudgetReport, PackError> {
    let manifest = EmbedManifest::read(dir)?;
    let plan_digest = FileDigest::of(&dir.join(EMBED_PLAN_FILE))?;
    if plan_digest.sha256 != manifest.plan_sha256 {
        return Err(malformed(
            "token audit: embed.jsonl digest differs from its manifest",
        ));
    }
    let tokenizer_digest = FileDigest::of(tokenizer)?;
    if tokenizer_digest.sha256 != manifest.model.tokenizer_checksum {
        return Err(malformed(
            "token audit: tokenizer checksum differs from the plan's model",
        ));
    }
    let mut report = TokenBudgetReport {
        plan_sha256: manifest.plan_sha256,
        tokenizer_sha256: tokenizer_digest.sha256,
        max_tokens: manifest.model.max_tokens,
        passages: 0,
        over_budget_passages: 0,
        discarded_tokens: 0,
        longest_passage_tokens: 0,
        examples: Vec::new(),
    };
    for item in read_embed_plan(dir, 0, u64::MAX)? {
        let (index, record) = item?;
        record.check(index)?;
        let tokens = count(&record.embedding_text)?;
        report.passages += 1;
        report.longest_passage_tokens = report.longest_passage_tokens.max(tokens);
        if tokens > report.max_tokens {
            report.over_budget_passages += 1;
            report.discarded_tokens += (tokens - report.max_tokens) as u64;
            if report.examples.len() < 16 {
                report.examples.push(OverBudgetPassage {
                    record: index,
                    key: record.key,
                    tokens,
                });
            }
        }
    }
    if report.passages != manifest.records {
        return Err(malformed(format!(
            "token audit: read {} records, manifest declares {}",
            report.passages, manifest.records
        )));
    }
    plan_digest.verify(&dir.join(EMBED_PLAN_FILE))?;
    Ok(report)
}

/// Operational diagnostics for passages actually sent to inference in one build
/// or shard. This sidecar is deliberately outside package identity and manifests.
pub const BUILD_TOKEN_AUDIT_FILE: &str = "token-audit.json";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BuildTokenAudit {
    pub passages: u64,
    pub over_budget_passages: u64,
    pub discarded_tokens: u64,
    pub max_tokens: usize,
    pub max_observed: usize,
}

#[derive(Default)]
pub(crate) struct BuildTokenAuditAccumulator {
    report: Option<BuildTokenAudit>,
    supported: Option<bool>,
}

impl BuildTokenAuditAccumulator {
    pub(crate) fn observe(
        &mut self,
        runtime: &crate::semantic::embedding::EmbeddingRuntime,
        text: &str,
    ) -> Result<(), EmbeddingError> {
        let tokens = runtime.untruncated_token_count(text)?;
        let supported = tokens.is_some();
        if self.supported.is_some_and(|previous| previous != supported) {
            return Err(EmbeddingError::InferenceFailed {
                reason: "backend changed its uncapped token-count capability during a build"
                    .to_string(),
            });
        }
        self.supported = Some(supported);
        let Some(tokens) = tokens else {
            return Ok(());
        };
        let report = self.report.get_or_insert(BuildTokenAudit {
            passages: 0,
            over_budget_passages: 0,
            discarded_tokens: 0,
            max_tokens: runtime.max_tokens(),
            max_observed: 0,
        });
        report.passages += 1;
        report.max_observed = report.max_observed.max(tokens);
        if tokens > report.max_tokens {
            report.over_budget_passages += 1;
            report.discarded_tokens += (tokens - report.max_tokens) as u64;
        }
        Ok(())
    }

    pub(crate) fn finish(self) -> Option<BuildTokenAudit> {
        self.report
    }
}

/// Write diagnostics before the build's completion manifest. Remove a leftover
/// report when a retry uses a backend without the counting capability.
pub(crate) fn write_build_token_audit(
    dir: &Path,
    report: Option<&BuildTokenAudit>,
) -> Result<(), PackError> {
    let path = dir.join(BUILD_TOKEN_AUDIT_FILE);
    if let Some(report) = report {
        super::files::write_json(&path, report)?;
        log::info!("Build token audit: {} passages, {} over budget, {} discarded tokens (max {} against cap {})",
            report.passages, report.over_budget_passages, report.discarded_tokens,
            report.max_observed, report.max_tokens);
    } else {
        match std::fs::remove_file(&path) {
            Ok(()) => (),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => (),
            Err(source) => {
                return Err(PackError::Io {
                    context: format!("removing stale token audit {}", path.display()),
                    source,
                })
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::plan::EmbedPlanWriter;
    use crate::semantic::versioning::test_identity;

    #[test]
    fn operational_report_is_atomic_and_mock_retry_clears_leftovers() {
        let dir = crate::distribution::testing::TempDir::new("token_report_write");
        let report = BuildTokenAudit {
            passages: 2,
            over_budget_passages: 1,
            discarded_tokens: 3,
            max_tokens: 5,
            max_observed: 8,
        };
        write_build_token_audit(dir.path(), Some(&report)).unwrap();
        let written: BuildTokenAudit = serde_json::from_slice(
            &std::fs::read(dir.path().join(BUILD_TOKEN_AUDIT_FILE)).unwrap(),
        )
        .unwrap();
        assert_eq!(written, report);
        write_build_token_audit(dir.path(), None).unwrap();
        assert!(!dir.path().join(BUILD_TOKEN_AUDIT_FILE).exists());
        write_build_token_audit(dir.path(), None).unwrap();
    }

    #[test]
    fn audits_counts_and_rejects_wrong_inputs() {
        let dir = std::env::temp_dir().join(format!("otzaria-token-audit-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let tokenizer = dir.join("tokenizer.json");
        std::fs::write(&tokenizer, b"fixture tokenizer").unwrap();
        let mut model = test_identity().model;
        model.tokenizer_checksum = FileDigest::of(&tokenizer).unwrap().sha256;
        model.max_tokens = 5;
        let mut writer = EmbedPlanWriter::create(&dir, None).unwrap();
        for text in ["one two", "one two three four five six"] {
            writer
                .offer(&crate::distribution::files::sha256(text.as_bytes()), text)
                .unwrap();
        }
        writer
            .finish(&model, model.chunking_identity, &model.query_packages[0])
            .unwrap();
        let count = |text: &str| Ok(text.split_whitespace().count() + 2);
        let report = audit_plan_tokens(&dir, &tokenizer, count).unwrap();
        assert_eq!(
            (
                report.passages,
                report.over_budget_passages,
                report.discarded_tokens
            ),
            (2, 1, 3)
        );
        assert_eq!(report.longest_passage_tokens, 8);
        std::fs::write(&tokenizer, b"wrong tokenizer").unwrap();
        assert!(audit_plan_tokens(&dir, &tokenizer, count)
            .unwrap_err()
            .to_string()
            .contains("tokenizer checksum"));
        std::fs::write(&tokenizer, b"fixture tokenizer").unwrap();
        std::fs::write(dir.join(EMBED_PLAN_FILE), b"{}\n").unwrap();
        assert!(audit_plan_tokens(&dir, &tokenizer, count)
            .unwrap_err()
            .to_string()
            .contains("digest"));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
