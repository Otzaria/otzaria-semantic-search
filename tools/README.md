# `tools/` — golden reference vector generation (roadmap P2)

These scripts produce and verify `tests/data/golden_vectors.json`, the reference data
that the Rust GGUF embedding backend is asserted against.

Nothing here runs in CI or at build time. They are developer tools, invoked by hand when
the model file or the corpus changes. They require a 396 MB model file that is
**gitignored and must never be committed**.

Read `docs/P2_REFERENCE_VECTORS.md` first — in particular the section on what the
goldens do and do not prove.

| File | Purpose |
|---|---|
| `golden_corpus.json` | The input texts. Plain data — extend it without touching code. |
| `generate_golden_vectors.py` | **Reference A.** llama.cpp via `llama-cpp-python`. Writes the golden file. |
| `crosscheck_torch_reference.py` | **Reference B.** An independent PyTorch forward pass over the same GGUF. Verifies, never writes. |

## The model file

```
Otzaria-Embedding-V1-Flash-0.6B-Q4_K_M.gguf
sha256  a1a89520be990087b0a54cc2635513e6eddbfae598fe979b44c52c6bd224b064
size    396,474,560 bytes
```

Both HuggingFace repos (`EMD123/Otzaria-Embedding-V1-Flash-0.6B-GGUF` and
`EMD123/Otzaria-Embedding-V1-Flash-0.6B`) are `gated: manual` and return HTTP 401
without an accepted-terms token, so there is no unattended download path. Obtain the
file by accepting the terms on HuggingFace, then keep it wherever you like and pass
`--model`. The default is the repository root, where `.gitignore` already excludes it.

Do not copy or move the model into a tracked directory.

## Python environments

Python 3.14 has no `llama-cpp-python` wheel and building against it is not worth the
risk; both references are pinned to **CPython 3.12**. Use two separate virtualenvs so
Reference B's torch stack cannot influence Reference A.

Create them outside the repository — a scratch or temp directory, never inside the
working tree.

```sh
SCRATCH=/tmp/otzaria-p2          # anywhere outside the repo

# Reference A
uv venv --python 3.12 "$SCRATCH/venvA"
VIRTUAL_ENV="$SCRATCH/venvA" uv pip install 'llama-cpp-python==0.3.34' 'numpy==2.5.1'

# Reference B
uv venv --python 3.12 "$SCRATCH/venvB"
VIRTUAL_ENV="$SCRATCH/venvB" uv pip install \
  'torch==2.13.0' 'transformers==5.14.1' 'gguf==0.19.0' 'accelerate==1.14.0' \
  'numpy==2.5.1' 'sentencepiece==0.2.2' 'protobuf==7.35.1'
```

`llama-cpp-python` has no macOS wheel on PyPI and builds from source with cmake; expect
a few minutes. `accelerate` is not optional — `from_pretrained(..., gguf_file=...)`
raises `ValueError: accelerate is required when loading a GGUF file` without it.

## Regenerating the goldens

```sh
"$SCRATCH/venvA/bin/python" tools/generate_golden_vectors.py --date "$(date +%F)"
```

Then re-run the cross-check and paste the numbers into
`docs/P2_REFERENCE_VECTORS.md`:

```sh
"$SCRATCH/venvB/bin/python" tools/crosscheck_torch_reference.py
```

Useful flags:

| Flag | Effect |
|---|---|
| `--model PATH` | Model location. Defaults to the repo root. |
| `--date YYYY-MM-DD` | Value for `header.generated_date`. Under `--check` it defaults to the date already recorded in the golden file; when writing it defaults to `1970-01-01` so output is byte-stable. Pass a real date for a golden you intend to commit. |
| `--check` | Recompute and compare against the committed file. Writes nothing. **Needs no other flags** — it inherits `generated_date` from the file, so it cannot fail over the date alone. Verifies the model identity first, then exits non-zero and reports per-record cosines on a mismatch. |
| `--diagnostics` | Also measure run-to-run and batch-vs-single agreement. Slow. |
| `--threads N` | Pinned ggml thread count (default 4). Measured to have no effect on the numbers, but pinned for hygiene. |
| `--gpu-layers N` | Layers to offload. **Keep 0.** CPU is the reproducible reference; Metal diverges by up to 5e-3 in cosine. |
| `--regenerate` | Required to overwrite goldens produced from a *different* model file. |

### Idempotency

Two runs with the same flags on the same model produce a **byte-identical** file. This is
load-bearing, not incidental — the generator embeds one sequence per decode call, pins
the thread count, uses fixed `n_ctx`/`n_batch`/`n_ubatch` constants rather than deriving
them from the corpus, stores vectors as base64 of little-endian f32 (never as decimal),
and writes no timestamp other than `header.generated_date`.

Verify with `--check`.

### The safety interlock

If `tests/data/golden_vectors.json` already exists and its `header.model_sha256` does
not match the SHA-256 of the model you passed, the generator refuses to proceed and exits
non-zero. Silently regenerating against a different model would replace the reference
data that every downstream assertion depends on, and the drift would look like a passing
test rather than a changed model.

Three details worth knowing:

- **A missing `header.model_sha256` counts as a mismatch**, not as permission to proceed.
  A golden file whose provenance cannot be established is exactly the case where
  overwriting it does the most damage. Same for a golden file that will not parse.
- **`--check` is interlocked too.** Pointing `--check` at the wrong model reports "wrong
  model" instead of comparing vectors — otherwise you get a wall of per-record
  differences that reads as "the goldens drifted" when the real fact is "wrong file".
- **`--regenerate` overrides this when writing, but not under `--check`**, since detecting
  precisely that situation is what `--check` is for.

Override with `--regenerate` when writing, and say so in the commit message.

## Extending the corpus

Add an object to `texts` in `golden_corpus.json`, bump `corpus_version`, then regenerate.
`field_docs` in that file documents every field. Three rules:

- **Never rename or delete an existing `id`.** Rust tests may reference ids by name.
- Write invisible or ambiguous characters as `\uXXXX` escapes, so the corpus stays
  reviewable in a diff. A reviewer cannot see a stray U+200F.
- **Adding an entry must not change any existing vector.** The generator embeds one
  sequence per decode call with fixed `n_ctx`/`n_batch`/`n_ubatch`, so it does not — but
  verify it, by diffing the pre-existing records before and after. If one moves, stop:
  the pipeline is not reproducible and that is a bigger problem than whatever you were
  adding.

Read `deliberately_uncovered` in `golden_corpus.json` before adding a degenerate input.
It records inputs that were measured and then intentionally left out, with the numbers —
currently the empty string, which lands outside three of the recommended tolerances.

Boundary fixtures pin exact token counts via `expect_total_tokens`. If a regeneration
fails there, the tokenizer or the fixture text changed. Recalibrate the fixture text
deliberately — do not relax the assertion.
