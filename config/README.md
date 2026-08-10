# The frozen production identity

`model.json` and `chunking.json` are what the library's vectors were built under, and
what every later release must be built under to reuse any of them. They are checked in
rather than passed as workflow inputs for one reason: a recipe that can be typed at
dispatch time is a recipe that can be typed wrong, and the failure is silent — the build
succeeds, declares its own hash, and produces vectors nothing else can use.

`chunking_identity` in `model.json` is the SHA-256 prefix of every field of
`chunking.json`. `Chunker::new` refuses a configuration whose hash is not the one the
model declares, so the two cannot drift apart unnoticed.

Changing any field here invalidates every stored vector. That is not a warning to be
careful; it is the mechanism — `docs/ARTIFACT_CONTRACT.md` and
`src/semantic/recipe.rs`.
