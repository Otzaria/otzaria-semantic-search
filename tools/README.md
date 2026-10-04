# `tools/` — the ONNX package checksum, the Meivin Round 2 goldens, and the build machine's queue

These scripts produce and verify the reference data the Rust ONNX backend — the crate's only
backend — is asserted against: its goldens, the package checksum, and the fixture its
ordinary tests run. Under `kaggle/` is the work queue a vector build is spread over.

The golden generators never run in CI or at build time. They are developer tools, invoked
by hand when a graph, the tokenizer or the cases change. The model files they read are
**gitignored and must never be committed**. The GGUF goldens and their two references —
`generate_golden_vectors.py`, `crosscheck_torch_reference.py` and
`tests/data/golden_vectors.json` — went with the GGUF backend; the last commit that has
them is `62f0c44`.

| File | Purpose |
|---|---|
| `onnx_package_checksum.py` | The ONNX package checksum (`otzaria-onnx-package-v1`), computed independently of the crate. Standard library only. CI runs it on the real package. |
| `test_onnx_package_checksum.py` | Reproduces the crate's golden digest from the same bytes, and pins which files are in a package and which references are refused. CI runs it. |
| `golden_corpus.json` | The texts `onnx_golden_cases.json` takes by id. Plain data. It was written for the GGUF goldens: its fields beyond `id` and `text` were theirs, and nothing reads them now. |
| `onnx_golden_cases.json` | The ONNX goldens' inputs. Most take their text from `golden_corpus.json` by id. Plain data. |
| `generate_onnx_golden_vectors.py` | The Python reference for the ONNX backend. Writes one golden file per graph: `tests/data/onnx_golden_vectors_int8.json` for the int8 graph (the default identity) and `tests/data/onnx_golden_vectors.json` for the fp32 graph (its reference). |
| `make_onnx_fixture.py` | Writes the tiny ONNX packages in `tests/data/onnx_fixture/` that the ordinary ONNX tests run — five graphs of about 8 KB, a BERT-style and a production-shaped (Unigram) tokenizer, and `expected.json`, the Python references' ids and vectors. Byte-stable for the package versions in its docstring (`onnx`, `tokenizers`, `onnxruntime`, `numpy`); `--check` compares with the committed files and writes nothing. |
| `kaggle/queue.py`, `kaggle/smoke.py` | A work queue over Kaggle sessions, across however many accounts hold quota (`kaggle/ACCOUNTS.md`), and the check that an account gets the GPU and the internet it reports. `kaggle/test_queue.py` proves they fail when Kaggle says no; CI runs it. |

## The package checksum

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

## The model

```
seforim-embed-round2-int8.onnx    42,489,219 bytes  sha256 659226865abd3a1bc833565ae6b2e2f48abdd7136285824a12966d4d3294cbf8
seforim-embed-round2-fp32.onnx   168,177,986 bytes  sha256 1fc2aa8f9e1a85a38c8667b4901205d2cc1f17d1e2e5acc8c676514b2d687948
tokenizer.json                     2,191,362 bytes  sha256 0664287976ecb078bdfd8f5e5515dc87d8cb7f985a79a481aa1cdf7a7321c0e9
```

From `ArieLLL123/judaic-semantic-round2-onnx-zayit` at
`1ec8dc68888bcea774ae9f735b2fe7cd9dc7f3ca` on HuggingFace, gated behind a manual approval;
the project's private mirror, which CI uses, is `otzaria/judaic-semantic-round2-onnx-zayit`
(99b8a61, the same bytes in all eight files). The int8 graph is the default identity and the
fp32 graph the reference it was quantized from: two models, each with its own golden file.
Keep a graph and the tokenizer together in one directory — the tokenizer is found beside
the graph — anywhere outside the repository, and pass the graph as `--model` or through
`OTZARIA_TEST_ONNX_MODEL`; the Rust gate picks the golden file by the graph's SHA-256. Licence and attribution:
[`config/models/meivin-round2-onnx/README.md`](../config/models/meivin-round2-onnx/README.md).

## Python environment

Create it outside the repository — a scratch or temp directory, never inside the working
tree.

```sh
SCRATCH=/tmp/otzaria-onnx        # anywhere outside the repo
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

## Regenerating the goldens

```sh
for graph in int8 fp32; do
  "$SCRATCH/venv-onnx/bin/python" tools/generate_onnx_golden_vectors.py \
    --model /path/to/seforim-embed-round2-$graph.onnx --date "$(date +%F)"
  "$SCRATCH/venv-onnx/bin/python" tools/generate_onnx_golden_vectors.py \
    --model /path/to/seforim-embed-round2-$graph.onnx --check
done
```

| Flag | Effect |
|---|---|
| `--model PATH` | One of the package's graphs; defaults to `$OTZARIA_TEST_ONNX_MODEL`. Its file name picks its golden file (`GRAPHS` in the script); any other file name is refused, since each graph is its own model. |
| `--out PATH` | The golden file; defaults to the graph's own. |
| `--date YYYY-MM-DD` | `header.generated_date`. Under `--check` it is inherited from the file, so a check cannot fail over the date alone; when writing it defaults to `1970-01-01`, so the output is byte-stable. Pass a real date for a golden you intend to commit. |
| `--check` | Recompute and compare byte for byte. Writes nothing. |
| `--regenerate` | Required to overwrite goldens whose `graph_sha256` or `tokenizer_sha256` is not the files'. Replacing them silently would move every assertion to a different model, and the drift would look like a passing test; a golden file that will not parse counts as a mismatch too. `--check` is interlocked the same way, and `--regenerate` does not silence it: pointed at the wrong package it reports the wrong package rather than a wall of vector differences. |
| `--diagnostics` | Also measure what can move a vector: repeat runs, fresh sessions, thread counts, optimization levels, KleidiAI, the other graph. |
| `--boundary-prefix ID ROLE N` | Print the `prefix_chars` at which a corpus text's input is exactly N ids. How the boundary cases are calibrated. |
| `--threads N`, `--optimization LEVEL` | The session's intra-op threads (default 4) and graph optimization level. Keep `ORT_ENABLE_ALL`: it is the backend's. |

The pipeline is in the script's docstring: the role prefix and the text, the model's own
`tokenizer.json` with its padding off and its truncation replaced by 256 on the right —
the total, special tokens and the role token included — one text per run, and the
graph's first output stored raw. The same inputs on the same machine produce a
byte-identical file.

Measured with `--diagnostics` on 2026-09-30 (Apple M4, onnxruntime 1.28.0, all 41 cases):

| | fp32 graph | int8 graph |
|---|---|---|
| repeat runs, fresh sessions, 1/2/4/8 threads | bit-identical | bit-identical |
| `ORT_ENABLE_EXTENDED` against `ORT_ENABLE_ALL` | bit-identical | bit-identical |
| `ORT_DISABLE_ALL`, `ORT_ENABLE_BASIC` | max \|Δ\| 2.1e-7 | max \|Δ\| 1.06e-2, cosine 0.99908 |
| KleidiAI off | max \|Δ\| 2.7e-7 | max \|Δ\| 1.07e-2, cosine 0.99896 |
| the other graph | cosine 0.99911 at worst | the same |

So the fp32 graph's vectors are a function of its inputs to within a few 1e-7 on any
kernel, while the int8 graph's depend on which int8 kernels run: two correct CPUs can part
at cosine ~0.999, the order at which int8 and fp32 themselves part. The Rust gate
(`onnx_backend::golden`) holds fp32 to cosine 0.99999 everywhere and int8 to 0.99999 on
the goldens' machine class, 0.995 elsewhere; it chooses the golden file by the graph's
hash. Against both files the gate matched every id and reproduced all 41 vectors bit for
bit, with Microsoft's 1.28.0 build on the same machine.

## Extending the cases

Never rename a case, or an id in `golden_corpus.json` a case takes its text from: Rust
tests may name them. Write invisible or ambiguous characters as `\uXXXX` escapes, so the
inputs stay reviewable in a diff — a reviewer cannot see a stray U+200F. And adding a case
must not move any other — each is its own run, so it does not; verify it with `--check`.
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
