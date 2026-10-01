# Model identities

Each directory here holds one pair: `model.json`, the `ModelIdentity` an artifact
declares, and `chunking.json`, the `ChunkerConfig` it was built under. A build is handed
one pair, never a mix of two.

| Directory | Model | Status |
|---|---|---|
| `config/models/meivin-round2-onnx/` | Meivin Round 2, the ONNX **int8** graph through ONNX Runtime, 256 dimensions | **the production identity**: the graph an application ships and the library is built with; see its README for why int8 |
| `config/models/meivin-round2-onnx-fp32/` | Meivin Round 2, the ONNX fp32 graph | the reference graph the int8 one was quantized from — a different model, for reference artifacts and parity checks only |

The pair that sat in `config/` itself — `EMD123/Otzaria-Embedding-V1-Flash-0.6B`, GGUF
`Q4_K_M` through llama.cpp, 1024 dimensions — went with the GGUF backend; the last commit
that has it is `62f0c44`. This crate can neither build an artifact under it nor open one:
an ONNX graph is the only model it reads.

The default is the int8 graph, decided for users on weak PCs: 42 MB against 168 MB on
disk, 169 against 387 MiB of peak memory, vectors within cosine 0.999 of fp32's and about
the same speed on an Apple M4 (the measurements, and what is still to measure on a weak
PC, are in `models/meivin-round2-onnx/README.md` and `docs/ONNX_BACKEND.md` §0). fp32
stays the reference, never mixed into an int8 index.

The two pairs differ in `model_checksum` and `model_quantization` alone. Both are identity
fields an installation compares, so an artifact built under one pair can never pass for
one built under the other.

## Why they are files

They are checked in rather than passed as workflow inputs for one reason: a recipe that
can be typed at dispatch time is a recipe that can be typed wrong, and the failure is
silent — the build succeeds, declares its own hash, and produces vectors nothing else can
use.

`chunking_identity` in `model.json` is the SHA-256 prefix of every field of its
`chunking.json`. `Chunker::new` refuses a configuration whose hash is not the one the
model declares, so the two cannot drift apart unnoticed.

`model_checksum` is the package checksum over the graph, its `tokenizer.json` and any
external-data file. `cargo run -- model-checksum --model-file <path>` prints it, and
`tools/onnx_package_checksum.py` computes it independently.

## Changing them

Changing any field of a pair invalidates every stored vector built under it. That is not
a warning to be careful; it is the mechanism — `docs/ARTIFACT_CONTRACT.md` and
`src/semantic/recipe.rs`. Leave a pair alone once an artifact has been built under it: a
new model is a new directory under `config/models/`, and becomes production by being
chosen, not by overwriting what the stored vectors were built under.
