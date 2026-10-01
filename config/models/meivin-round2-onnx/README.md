# Meivin Round 2 (ONNX), int8 — the default

`model.json` and `chunking.json` here are the identity of an artifact built with the
Meivin Round 2 ONNX model's **int8 graph**, `seforim-embed-round2-int8.onnx`: what such an
artifact declares, and what `build` must be handed to produce one. It is the graph an
application ships and the one the library's vectors are built with. The pair sits beside
the production pair in `config/`, which it does not replace — see `../../README.md`.

The model's fp32 graph, which the int8 graph was quantized from, is the reference: a
different model with its own identity, in
[`../meivin-round2-onnx-fp32/`](../meivin-round2-onnx-fp32/README.md), and an artifact of
its own. Vectors from one never go into an index of the other.

## Why int8

| | int8 | fp32 |
|---|---:|---:|
| on disk | 42,489,219 B (42 MB) | 168,177,986 B (168 MB) |
| peak footprint, one session | 169 MiB | 387 MiB |
| agreement with fp32, 600 library inputs | cosine 0.999067 at worst | — |
| a 256-token passage / a 14-token query, 4 threads, Apple M4 | 16.6 / 1.7 ms | 16.4 / 2.6 ms |

A quarter of the download and less than half the memory, for vectors within cosine 0.999
of the fp32 graph's — about the same speed on the M4 (`docs/ONNX_BACKEND.md` §7.2). Still
to measure, on a weak PC: int8 latency on an old x86 CPU without VNNI, where dynamically
quantized matrix products have no dedicated instructions. One more property to know: int8
vectors depend on which int8 kernels run — on the M4 alone, other kernels moved them to
cosine 0.99896, where fp32 moves by ~1e-7 — so a library built on one CPU family and
queried on another is compared at about that precision, the order at which int8 and fp32
part anyway. On x86 that holds because every int8 product is computed exactly: by the CPU's
VNNI kernels where it has them, and elsewhere because the backend sets
`session.x64quantprecision`, without which ONNX Runtime's kernels for CPUs without VNNI
saturate them — which put CI's x86 runner at cosine 0.9809 (`docs/ONNX_BACKEND.md` §0.1).
Both paths compute the same products, so which one a CPU takes changes no vector. Measuring
it between x86 and ARM belongs to the weak-PC measurement.

| | |
|---|---|
| Model | Meivin Round 2, the Judaic Semantic Embedding project's sentence encoder for Hebrew and Aramaic: a BERT encoder (8 layers, width 512, a 32,000-token Unigram vocabulary), masked mean pooling, a trained 512→256 projection, its LayerNorm and a final L2 normalization — all inside the graph |
| Source | [`ArieLLL123/judaic-semantic-round2-onnx-zayit`](https://huggingface.co/ArieLLL123/judaic-semantic-round2-onnx-zayit) at `1ec8dc68888bcea774ae9f735b2fe7cd9dc7f3ca`, gated behind a manual approval |
| Mirror | `otzaria/judaic-semantic-round2-onnx-zayit` on Hugging Face (99b8a61, the same bytes in all eight files), private: what CI's `golden-onnx` job downloads, so the `OTZARIA_HF_TOKEN` it runs with must be able to read it |
| Vectors | 256 dimensions, unit-norm, from the graph's first output (`embedding`, `float32[1, 256]`) |
| Inputs | `input_ids` and `attention_mask`, `int64[1, sequence_length]` — the batch is fixed at one |

## The package

An ONNX model is a package: the graph `model_path` names, `tokenizer.json` beside it,
and any external-data file the graph's tensors name (this graph has none). The
`model_checksum` is the `otzaria-onnx-package-v1` checksum of exactly these files:

```text
otzaria-onnx-package-v1
seforim-embed-round2-int8.onnx	42489219	659226865abd3a1bc833565ae6b2e2f48abdd7136285824a12966d4d3294cbf8
tokenizer.json	2191362	0664287976ecb078bdfd8f5e5515dc87d8cb7f985a79a481aa1cdf7a7321c0e9

model_checksum = sha256(the text above) = 9e408407922b4aab26dd148cbe4b9a0e573c65591cfc799ba28e991d77d9d065
```

Either implementation recomputes it from the files, and both must agree:

```sh
python3 tools/onnx_package_checksum.py /path/to/seforim-embed-round2-int8.onnx
cargo run -- model-checksum --model-file /path/to/seforim-embed-round2-int8.onnx
```

The rest of the model repository is not part of the package, because none of it reaches a
vector: `README.md`, `LICENSE.md`, the author's `manifest.json`, `.gitattributes`, the
export script — and `seforim-embed-round2-fp32.onnx`, which is a different model, with its
own identity beside this one. The two graphs can share a directory: the package is the
graph `model_path` names, and neither graph's checksum sees the other. Neither is the ONNX
Runtime library part of it, which is found separately (the path the application passes,
`OTZARIA_ONNX_RUNTIME`, or the platform's file name beside the graph) and is code, not
model data.

## The fields

| Field | Value | Why |
|---|---|---|
| `model_checksum` | `9e408407…d9d065` | the package checksum above |
| `model_quantization` | `int8` | the dynamically quantized graph; a declaration, as for GGUF — the checksum is what tells the graphs apart |
| `embedding_backend` | `onnxruntime-sentence-v1` | ONNX Runtime, `tokenizer.json` → `input_ids` + `attention_mask`, the first output as the sentence vector |
| `embedding_dim` | `256` | the graph's output width |
| `pooling` | `in-graph` | the graph pools and normalizes; nothing is pooled outside it. An `.onnx` model declaring anything else is refused |
| `max_tokens` | `256` | the total sequence length, `[CLS]`, `[SEP]` and the role token included — the cap the model's own search encoder uses |
| `embedding_text_version` | `2` | the role prefixes: `[PASSAGE] ` before every stored passage, `[QUERY] ` before every query. They are learned special tokens |
| `normalization_version` | `1` | the text as the corpus supplies it |
| `chunking_identity` | `2685558872390372738` | `ChunkerConfig::identity` of `chunking.json` here: the production chunking, with `embedding_text_version` 2 |

## Open questions

- **The token cap.** `max_tokens` 256 is what the model card says the model's own search
  encoder truncates to. The author's parity checks, though, tokenize their probes at 128
  tokens, and 256 for one long probe only, so the graph's agreement with its PyTorch
  source is measured at 256 on a single input. Whether 256 or a shorter cap retrieves
  better on Otzaria's line-level corpus has not been measured; the card itself asks an
  integration to evaluate a different limit before choosing one. Changing it is a new
  identity.
- **The space after the role prefix.** The recipe writes `[QUERY] ` and `[PASSAGE] ` with
  a trailing space, as the model card and the author's parity checks do (the author's
  `manifest.json` records the bare tokens). For a text that begins with anything visible
  the two spellings give identical ids; for one that begins with whitespace, the spaced
  form adds one `▁` id, and so does whitespace at the end of a text. So neither side hands
  the model either: the chunker trims every line before its character cap, and text recipe
  2 trims the capped passage — the cap can end it on a space — and every query before
  prefixing them — see `tools/README.md`.

## Licence and attribution

The model and its tokenizer are licensed under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/): non-commercial
use, with attribution, and adaptations under the same licence. The attribution the
licence asks for is **"Judaic Semantic Embedding project, Otzaria (https://otzaria.org/)"**;
it goes wherever the model files go.

The model repository's `export_round2_onnx.py` is under a personal-use licence of its
own. Nothing from it is in this repository, and it is not part of the package: distribute
the model without it.

The model files are never committed here; `.gitignore` names them.
