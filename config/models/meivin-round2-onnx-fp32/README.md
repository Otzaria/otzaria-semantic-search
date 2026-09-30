# Meivin Round 2 (ONNX), fp32 — the reference graph

`model.json` and `chunking.json` here are the identity of an artifact built with the
Meivin Round 2 model's **fp32 graph**, `seforim-embed-round2-fp32.onnx`. The default is the
int8 graph, in [`../meivin-round2-onnx/`](../meivin-round2-onnx/README.md), which says
why, and what the two share: the tokenizer, the recipe, the backend and the licence.

The fp32 graph is what the int8 graph was quantized from, and the reference it is
measured against: its vectors agree with int8's at cosine ≥ 0.999, not 1, so it is a
different model. An artifact built with it declares this identity and nothing else
opens it with the int8 graph — the checksum differs — and vice versa. Build one only when
the reference itself is wanted: for parity checks, or to measure what quantization costs.

The two identities differ in exactly two fields:

| Field | Value | Why |
|---|---|---|
| `model_checksum` | `4a4a2ae8…2ade46` | the package checksum of `seforim-embed-round2-fp32.onnx` and `tokenizer.json` |
| `model_quantization` | `fp32` | the unquantized graph |

Every other field, and `chunking.json`, is the int8 identity's: the same backend, width,
pooling, token cap, text recipe and chunking, whose `chunking_identity` hashes the
configuration and not the graph.

```text
otzaria-onnx-package-v1
seforim-embed-round2-fp32.onnx	168177986	1fc2aa8f9e1a85a38c8667b4901205d2cc1f17d1e2e5acc8c676514b2d687948
tokenizer.json	2191362	0664287976ecb078bdfd8f5e5515dc87d8cb7f985a79a481aa1cdf7a7321c0e9

model_checksum = sha256(the text above) = 4a4a2ae88a86f15ffe6069bfcefc3abd13c207cec5d7aaef52c0c59d752ade46
```

Its goldens are `tests/data/onnx_golden_vectors.json`; CI's `golden-onnx` job checks this
checksum and runs the gate for this graph too.
