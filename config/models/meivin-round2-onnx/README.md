# Meivin Round 2 (ONNX)

`model.json` and `chunking.json` here are the identity of an artifact built with the
Meivin Round 2 ONNX model: what such an artifact declares, and what `build` must be
handed to produce one. They sit beside the production pair in `config/`, which they do
not replace — see `../../README.md`.

| | |
|---|---|
| Model | Meivin Round 2, the Judaic Semantic Embedding project's sentence encoder for Hebrew and Aramaic: a BERT encoder (8 layers, width 512, a 32,000-token Unigram vocabulary), masked mean pooling, a trained 512→256 projection, its LayerNorm and a final L2 normalization — all inside the graph |
| Source | [`ArieLLL123/judaic-semantic-round2-onnx-zayit`](https://huggingface.co/ArieLLL123/judaic-semantic-round2-onnx-zayit) at `1ec8dc68888bcea774ae9f735b2fe7cd9dc7f3ca`, gated behind a manual approval |
| Mirror | `otzaria/judaic-semantic-round2-onnx-zayit` on Hugging Face, private: what CI's `golden-onnx` job downloads, so the job needs it to exist and an `OTZARIA_HF_TOKEN` that can read it |
| Vectors | 256 dimensions, unit-norm, from the graph's first output (`embedding`, `float32[1, 256]`) |
| Inputs | `input_ids` and `attention_mask`, `int64[1, sequence_length]` — the batch is fixed at one |

## The package

An ONNX model is a package: the graph `model_path` names, `tokenizer.json` beside it,
and any external-data file the graph's tensors name (this graph has none). The
`model_checksum` is the `otzaria-onnx-package-v1` checksum of exactly these files:

```text
otzaria-onnx-package-v1
seforim-embed-round2-fp32.onnx	168177986	1fc2aa8f9e1a85a38c8667b4901205d2cc1f17d1e2e5acc8c676514b2d687948
tokenizer.json	2191362	0664287976ecb078bdfd8f5e5515dc87d8cb7f985a79a481aa1cdf7a7321c0e9

model_checksum = sha256(the text above) = 4a4a2ae88a86f15ffe6069bfcefc3abd13c207cec5d7aaef52c0c59d752ade46
```

Either implementation recomputes it from the files, and both must agree:

```sh
python3 tools/onnx_package_checksum.py /path/to/seforim-embed-round2-fp32.onnx
cargo run -- model-checksum --model-file /path/to/seforim-embed-round2-fp32.onnx
```

The rest of the model repository is not part of the package, because none of it reaches a
vector: `README.md`, `LICENSE.md`, the author's `manifest.json`, `.gitattributes`, the
export script — and `seforim-embed-round2-int8.onnx`, which is a different model. Its
vectors differ from the fp32 graph's (cosine 0.9991 at worst over the golden cases), so
an artifact built with it would need an identity of its own, with its own checksum and
`model_quantization`. Neither is the ONNX Runtime library, which is found separately
(`OTZARIA_ONNX_RUNTIME`, or the platform's file name beside the graph) and is code, not
model data.

## The fields

| Field | Value | Why |
|---|---|---|
| `model_checksum` | `4a4a2ae8…2ade46` | the package checksum above |
| `model_quantization` | `fp32` | the fp32 graph; a declaration, as for GGUF |
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
  form adds one `▁` id. So neither side hands the model one: passages are trimmed by the
  chunker, and text recipe 2 trims a query before prefixing it — see `tools/README.md`.

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
