# Complete main-text and separate-note recipe (candidate)

This profile keeps QA491 weights, tokenizer and token limits, changes chunking_version to
**3**, embedding_text_version to **4**, and identity to **12470596585207933847**.
Published profiles and packages remain unchanged.

The host supplies semantic_text without note markers/bodies or BOM, and semantic_notes
in source order. Lexical text remains separate. The companion engine branch
codex/issue34-semantic-source requires a rebuilt index with semantic_projection_version=1.
Old plain-text indexes cannot reconstruct unclassified superscript provenance safely.

Short context retains its anchor. Main text and notes are embedded independently;
notes shorter than five Unicode characters remain lexically searchable but have no
standalone semantic vector. Notes share the source line's window budget. Long lines
use at most 255 windows, reducing overlap and occasionally splitting words to preserve
continuous coverage when necessary. Impossible coverage is refused before publication.

Rebuild the lexical index, export a fresh plan, generate vectors under this profile
and validate the set against the same engine recipe. Old vectors cannot be relabelled.
The application retains its legacy default until a matching new package is selected.
Recipe 4 uses prebuilt artifact generation/installation; the development in-memory store
accepts anchored recipes only. Ranking-quality comparison of separate versus inline
notes still requires S1; structural tests do not establish ranking-quality improvement.

Audit pending token lengths without a graph or ONNX Runtime:

```sh
cargo run --features onnx-backend --example audit_plan_tokens -- \
  /path/to/plan /path/to/tokenizer.json > token-budget.json
```

Exit status 2 means a passage would truncate. JSON records counts, discarded tokens and
example keys. Optional --allow-truncation retains the report with a zero exit status.
Tokenizer and plan digests are verified. Reused warehouse texts are absent from pending
plans and must be audited at initial embedding. Builds and shards also write an automatic
token-audit.json with actual inference counts when the backend supports uncapped counting.

See [the conformance guide](../../../docs/SEMANTIC_RECIPE_FIXTURES.md) for the prepared-text
fixture. The companion engine publishes a raw-HTML-to-key fixture as well.
