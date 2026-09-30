# `tools/` — golden reference vector generation (roadmap P2)

These scripts produce and verify `tests/data/golden_vectors.json`, the reference data
that the Rust GGUF embedding backend is asserted against. The ONNX backend's
counterparts — its goldens, the package checksum and the fixture its ordinary tests run —
are in [their own section](#onnx-the-package-checksum-and-the-meivin-round-2-goldens) below.

The golden generators never run in CI or at build time. They are developer tools, invoked
by hand when the model file or the corpus changes. They require a model file that is
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

## ONNX: the package checksum and the Meivin Round 2 goldens

| File | Purpose |
|---|---|
| `onnx_package_checksum.py` | The ONNX package checksum (`otzaria-onnx-package-v1`), computed independently of the crate. Standard library only. CI runs it on the real package. |
| `test_onnx_package_checksum.py` | Reproduces the crate's golden digest from the same bytes, and pins which files are in a package and which references are refused. CI runs it. |
| `onnx_golden_cases.json` | The ONNX goldens' inputs. Most take their text from `golden_corpus.json` by id. Plain data. |
| `generate_onnx_golden_vectors.py` | The Python reference for the ONNX backend. Writes `tests/data/onnx_golden_vectors.json`. |
| `make_onnx_fixture.py` | Writes the tiny ONNX packages in `tests/data/onnx_fixture/` that the ordinary ONNX tests run — five graphs of about 8 KB, a BERT-style and a production-shaped (Unigram) tokenizer, and `expected.json`, the Python references' ids and vectors. Byte-stable for the package versions in its docstring (`onnx`, `tokenizers`, `onnxruntime`, `numpy`); `--check` compares with the committed files and writes nothing. |

### The package checksum

```sh
python3 tools/onnx_package_checksum.py path/to/graph.onnx               # the checksum, alone on stdout
python3 tools/onnx_package_checksum.py --manifest path/to/graph.onnx    # exactly the text that is hashed
python3 tools/onnx_package_checksum.py --expect <hex> path/to/graph.onnx
```

It validates the package the way `src/semantic/model_package.rs` does — the same protobuf
walk, the same limits, the same refusals — so a package one of them refuses, the other
refuses, and one both accept has one checksum. It must print what
`cargo run -- model-checksum --model-file` prints. The walk is hand-written rather than
`onnx.external_data_helper`, which does not look for external data in sparse
initializers or training graphs: for a package that uses them it would name fewer files
and so a different checksum, with nothing to say so.

`python -m pytest tools/test_onnx_package_checksum.py` needs only pytest. With the `onnx`
package installed, one more test cross-checks the walk against onnx's own reading of a
model it saved with external data.

### The model

```
seforim-embed-round2-fp32.onnx   168,177,986 bytes  sha256 1fc2aa8f9e1a85a38c8667b4901205d2cc1f17d1e2e5acc8c676514b2d687948
tokenizer.json                     2,191,362 bytes  sha256 0664287976ecb078bdfd8f5e5515dc87d8cb7f985a79a481aa1cdf7a7321c0e9
```

From `ArieLLL123/judaic-semantic-round2-onnx-zayit` at
`1ec8dc68888bcea774ae9f735b2fe7cd9dc7f3ca` on HuggingFace, gated behind a manual approval
(the project's private mirror, which CI uses, is `otzaria/judaic-semantic-round2-onnx-zayit`).
Keep the two files together in one directory — the tokenizer is found beside the graph —
anywhere outside the repository, and pass the graph as `--model` or through
`OTZARIA_TEST_ONNX_MODEL`. Licence and attribution:
[`config/models/meivin-round2-onnx/README.md`](../config/models/meivin-round2-onnx/README.md).

### Python environment

```sh
uv venv --python 3.12 "$SCRATCH/venv-onnx"
VIRTUAL_ENV="$SCRATCH/venv-onnx" uv pip install \
  'onnxruntime==1.28.0' 'tokenizers==0.23.2' 'numpy==2.5.1'
```

Both pins are the backend's: `tokenizers` 0.23.2 is the Python binding of the same Rust
tokenizer core `Cargo.toml` pins, and 1.28.0 is the backend's reference ONNX Runtime.
The generator refuses any other version of either — a reference on other versions is a
different reference.

One difference between the two sides is deliberate: the Python binding is built with the
`onig` regex engine and the backend with `fancy-regex`. This tokenizer's normalizer is
regular expressions — it strips pointing and cantillation, strips bidi and joiner marks,
and folds quotes and dashes — so the engines could part. Tokenizing every code point of
the Basic Multilingual Plane and a sample of the others, alone after `[QUERY] ` and
between two letters after `[PASSAGE] `, gave the same ids in both (2026-09-30);
`passage_normalizer_classes` keeps a character of every class in the goldens, where the
gate would see a difference.

### Regenerating the goldens

```sh
"$SCRATCH/venv-onnx/bin/python" tools/generate_onnx_golden_vectors.py \
  --model /path/to/seforim-embed-round2-fp32.onnx --date "$(date +%F)"
"$SCRATCH/venv-onnx/bin/python" tools/generate_onnx_golden_vectors.py \
  --model /path/to/seforim-embed-round2-fp32.onnx --check
```

| Flag | Effect |
|---|---|
| `--model PATH` | The fp32 graph; defaults to `$OTZARIA_TEST_ONNX_MODEL`. Any other file name is refused: the int8 graph beside it is a different model. |
| `--date YYYY-MM-DD` | `header.generated_date`, exactly as for the GGUF goldens: `--check` inherits it from the file. |
| `--check` | Recompute and compare byte for byte. Writes nothing. |
| `--regenerate` | Required to overwrite goldens whose `graph_sha256` or `tokenizer_sha256` is not the files'. The interlock is the GGUF generator's, and so are its reasons. |
| `--diagnostics` | Also measure what can move a vector: repeat runs, fresh sessions, thread counts, optimization levels, KleidiAI, the int8 graph. |
| `--boundary-prefix ID ROLE N` | Print the `prefix_chars` at which a corpus text's input is exactly N ids. How the boundary cases are calibrated. |
| `--threads N`, `--optimization LEVEL` | The session's intra-op threads (default 4) and graph optimization level. Keep `ORT_ENABLE_ALL`: it is the backend's. |

The pipeline is in the script's docstring: the role prefix and the text, the model's own
`tokenizer.json` with its padding off and its truncation replaced by 256 on the right —
the total, special tokens and the role token included — one text per run, and the
graph's first output stored raw. The same inputs on the same machine produce a
byte-identical file.

Measured with `--diagnostics` on 2026-09-30 (Apple M4, onnxruntime 1.28.0, all 41 cases):
repeat runs, fresh sessions and 1, 2, 4 or 8 threads are bit-identical;
`ORT_ENABLE_EXTENDED` equals `ORT_ENABLE_ALL` bit for bit, while `ORT_DISABLE_ALL` and
`ORT_ENABLE_BASIC` move components by up to 2.1e-7 and KleidiAI off by 2.7e-7 (cosine 1
to twelve places); the int8 graph agrees at cosine 0.99911 at worst. Against these goldens
the Rust gate (`onnx_backend::golden`) matched every id and reached cosine 1.0000000000 on
every case, with Microsoft's 1.28.0 build on the same machine.

### Extending the cases

The corpus rules apply: never rename a case, write invisible characters as `\uXXXX`
escapes, and adding a case must not move any other — each is its own run, so it does not.
The boundary cases pin exact counts through `expect_total_tokens` and
`expect_untruncated_tokens`; if a regeneration fails there, the tokenizer or the text
changed, and the fix is to recalibrate with `--boundary-prefix`, not to relax them.

Two behaviours the cases pin on purpose, because the tokenizer decides them and the
backend has to agree:

- **Added special tokens are matched anywhere in the input.** That is how `[PASSAGE] ` and
  `[QUERY] ` become the learned role tokens, and it is also why a `[PASSAGE]`, a `[CLS]` or
  a `[קטע]` inside a book's text becomes that token. The reference neither escapes nor
  strips them. `[query]` in lower case is text: the tokens are declared `normalized: false`.
- **The space after the role prefix is not always free.** `"[QUERY] " + text` and
  `"[QUERY]" + text` give identical ids whenever the text begins with anything visible.
  They differ when it begins with whitespace — a space, or a character NFKC makes one, such
  as a no-break space — when it is empty, or when it begins with another special token:
  the spaced form then carries one more `▁` id (10). The recipe uses the spaced form, as
  the model card and the model author's own export checks do, and hands the tokenizer
  neither shape: the chunker trims every line before its character cap, and text recipe 2
  trims the capped passage and every query before prefixing them
  (`EmbeddingTextRecipe::passage_text`, `query_text`) — a cap can end a passage on a space,
  which would be a lone `▁` before `[SEP]`. No passage case has whitespace at either end,
  so every passage input here is exactly what the recipe produces. `query_leading_whitespace` is
  spelled as the tokenizer receives it, so it still pins the tokenizer's side of this —
  what an untrimmed query would cost — though no search produces it any more; its
  `notes`, recorded in the goldens, predate the trim. The generator reports the comparison
  for every case on each run: 36 of the 37 role-prefixed cases tokenize alike either way —
  those beginning with a digit, a quote, a parenthesis and a Latin letter among them — and
  `query_leading_whitespace` does not.
