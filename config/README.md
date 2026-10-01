# Model identities

Each directory here holds one pair: `model.json`, the `ModelIdentity` a vector set
declares — a model family, its text recipe, and the packages of the family a query may be
embedded with — and `chunking.json`, the `ChunkerConfig` it was built under. A build is
handed one pair, never a mix of two.

| Directory | Model | Status |
|---|---|---|
| `config/models/meivin-round2-onnx/` | Meivin Round 2 through ONNX Runtime, 256 dimensions: its int8 and fp32 packages | **the production identity**: the application ships the int8 package and embeds queries with it; the library's passages are embedded with fp32 |

The pair that sat in `config/` itself — `EMD123/Otzaria-Embedding-V1-Flash-0.6B`, GGUF
`Q4_K_M` through llama.cpp, 1024 dimensions — went with the GGUF backend; the last commit
that has it is `62f0c44`. This crate can neither build an artifact under it nor open one:
an ONNX graph is the only model it reads.

The application's default is the int8 package, decided for users on weak PCs: 42 MB
against 168 MB on disk, 169 against 387 MiB of peak memory, vectors within cosine 0.999 of
fp32's and about the same speed on an Apple M4 (the measurements, and what is still to
measure on a weak PC, are in `models/meivin-round2-onnx/README.md` and
`docs/ONNX_BACKEND.md` §0). The fp32 package was a separate identity until the two became
one family; a set now accepts a query from either.

## Why they are files

They are checked in rather than passed as workflow inputs for one reason: a recipe that
can be typed at dispatch time is a recipe that can be typed wrong, and the failure is
silent — the build succeeds, declares its own hash, and produces vectors nothing else can
use.

`chunking_identity` in `model.json` is the SHA-256 prefix of every field of its
`chunking.json`. `Chunker::new` refuses a configuration whose hash is not the one the
model declares, so the two cannot drift apart unnoticed.

A package's checksum is computed over the graph, its `tokenizer.json` and any
external-data file. `cargo run -- model-checksum --model-file <path>` prints it, and
`tools/onnx_package_checksum.py` computes it independently.

## Changing them

Changing any field of a pair invalidates every stored vector built under it. That is not
a warning to be careful; it is the mechanism — `docs/ARTIFACT_CONTRACT.md` and
`src/semantic/recipe.rs`. Leave a pair alone once an artifact has been built under it: a
new model is a new directory under `config/models/`, and becomes production by being
chosen, not by overwriting what the stored vectors were built under.
