# Prepared-text semantic recipe conformance

[`tests/data/semantic_recipe_v3.json`](../tests/data/semantic_recipe_v3.json) is a model-free,
machine-readable example of chunking version 3 with embedding text version 4. It includes
the complete configuration and its identity, a small prepared book, and every expected
window's text, ordinal, Unicode character range, and 128-bit hexadecimal chunk key.
The 64-character window size makes boundary cases easy to inspect; it is part of this
fixture's explicit configuration, rather than a production profile default.

The examples cover section boundaries, balanced context around a short anchor, long lines,
Unicode whitespace, a long unbroken word, skipped empty/very short lines, separate notes,
and a line whose first semantic window comes from a note. `BookLine.text` is the lexical
anchor. `semantic_text` and `semantic_notes` are the already prepared semantic projection.
Raw HTML, footnote recognition, data URI cleanup and lexical normalization happen upstream;
the engine's `semantic_recipe_fixtures` example supplies that part of the contract.

`chars` counts Unicode scalar values in the trimmed main text or the individual note text.
For a contextual window it describes the anchor range; neighbours are represented in
`embedding_text`. Notes keep their own window positions even when their text equals a main
window; downstream vector storage deduplicates equal keys. The key is the first 16 bytes of
SHA-256 over the exact UTF-8 `embedding_text`, formatted as 32 lowercase hexadecimal digits.

Verify or reproduce the fixture without an ONNX model, tokenizer or database:

```sh
cargo test --test recipe_conformance
cargo run --example recipe_fixtures -- --check tests/data/semantic_recipe_v3.json
cargo run --example recipe_fixtures -- --write tests/data/semantic_recipe_v3.json
```

Review recipe version changes before regenerating the JSON. Published v1/v2 behaviour
remains pinned by the existing tests, including `long_line_windows_golden.json`. Version 3
retains the requested overlap when the whole line fits. For exceptionally long lines it
reduces overlap and, when required by the remaining window budget, splits a word to preserve
complete coverage. A line physically exceeding the available windows fails explicit coverage
validation; it is never accepted as a complete build. Notes share the line's maximum of 255
window positions. Authoritative builders must call `validate_book_coverage`; a consumer with
only a neighbour slice can call `validate_line_coverage`.

ONNX corpus builds and shard embedding automatically write `token-audit.json` beside their
output. Counts cover only passages actually sent to inference after vector reuse/deduplication:
`passages`, `over_budget_passages`, `discarded_tokens`, `max_tokens`, and `max_observed`.
They include role prefixes and special tokens and use the inference tokenizer with truncation
and padding disabled. This operational file is outside the package's authenticated manifest
and identity. A backend without an exact uncapped tokenizer, including the mock backend,
emits no fabricated report. Ordinary query inference performs no additional tokenization.
The `audit_plan_tokens` example remains available for a preflight before inference.
