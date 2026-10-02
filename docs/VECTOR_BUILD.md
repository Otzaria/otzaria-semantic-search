# Building the library's vectors

How a release's vectors are built on the build machine, file by file: the plan, the
external embedding interface, the warehouse, the ledger, assembly and the gates. The
segment and the release manifest an installation takes are in
[`ARTIFACT_CONTRACT.md`](ARTIFACT_CONTRACT.md). The code is in `src/distribution/`.

Every multi-byte integer is **little-endian**. Every digest is SHA-256, written as 64
lowercase hex digits where it is text. A *key* is the first 16 bytes of the SHA-256 of
the exact string the model is given — role prefix, context and cap included — and
`ChunkKey::of` computes it.

```text
plan        index → records.bin, books.json, embed.jsonl (+ manifest), tombstones.bin,
            plan-manifest.json                                    (export_semantic_plan)
embed       embed.jsonl window → a shard: vectors.f32, keys.bin, shard-manifest.json
                                                         (embed-shard, or any worker)
warehouse   shards → append-only f32 masters by full SHA-256      (warehouse-add)
assemble    plan + warehouse (+ previous ledger) → segment, package, release manifest,
            the next ledger                                       (assemble)
verify      the gates                                             (assemble --verify)
```

## 1. The plan

One directory, written by `export_semantic_plan` over the release index (in
`otzaria_search_engine`) or by `otzaria-semantic-search plan` over a transcription.

| File | Layout |
|---|---|
| `records.bin` | `b"OXVREC1\n"` (8 bytes), `u64` count, then per embedded line `{book u32, ordinal u32, sha256 [32]}` (40 bytes), strictly ascending by (book, ordinal) |
| `books.json` | JSON array of the books' stable keys, strictly ascending by bytes; a record's `book` is an index into it |
| `embed.jsonl` | the texts to embed — §2 |
| `embed-manifest.json` | what `embed.jsonl` is — §2 |
| `tombstones.bin` | `b"OXVKEYS1"`, `u64` count, then 16-byte keys strictly ascending: the keys the previous release held and this one does not |
| `plan-manifest.json` | below |

`ordinal` is the line's position in its book, as the index stores it; a segment records it
as the line's hint. A line the recipe does not embed has no record.

`plan-manifest.json`:

```json
{
  "format": "otzaria-vector-plan", "version": 1,
  "identity": { "text": {…}, "model": {…}, "store": {…} },
  "library_version": 30, "library_release_tag": "v30-20261001120000",
  "previous": { "library_version": 29, "ledger_manifest_sha256": "…" },
  "counts": { "records": 0, "books": 0, "unique": 0, "reused": 0, "to_ship": 0,
              "to_embed": 0, "revived": 0, "tombstones": 0, "foreign_pairs": 0 },
  "files": { "records.bin": { "sha256": "…", "size": 0 }, "books.json": {…}, … },
  "parity": { "checked": 6910000, "mismatches": 0 },
  "created_at": "2026-10-01T12:00:00Z"
}
```

`previous` is `null` for a first base. The counts are the split against the previous
ledger (§4): `unique` distinct keys, `reused` of them held by the previous release,
`to_ship` the rest (a delta's slots), `revived` of those the warehouse holds already,
`tombstones`, and `foreign_pairs` — (book, key) records new in this release whose key the
previous release holds. `parity` is the planner's gate G2 — documents whose stored
`chunkKey` was compared with the key recomputed from their text, and how many disagreed —
and a plan with a mismatch is refused. A plan from a transcription reports `checked: 0`.

## 2. The external embedding interface

A worker — `otzaria-semantic-search embed-shard`, or any other program — is given the
plan's `embed.jsonl` and `embed-manifest.json`, and a window of records. It needs nothing
else: not the corpus, not the index.

### Input

`embed.jsonl`: one JSON object per line, LF-terminated, UTF-8:

```json
{"key":"<32 hex>","embedding_text_sha256":"<64 hex>","embedding_text":"[PASSAGE] …"}
```

- `embedding_text` is the exact string to embed. Do not trim, normalize or prefix it.
- `embedding_text_sha256` is the SHA-256 of its UTF-8 bytes; `key` is its first 32 hex
  digits.
- Each text appears once, in order of first appearance in the plan, and only where the
  warehouse holds no vector for it.

`embed-manifest.json`:

```json
{
  "format": "otzaria-embed-plan", "version": 2,
  "records": 6347587,
  "plan_sha256": "<SHA-256 of embed.jsonl's bytes>",
  "model": { "family_id": "…", "tokenizer_checksum": "…", "embedding_dim": 256,
             "pooling": "in-graph", "max_tokens": 256, "embedding_text_version": 2,
             "normalization_version": 1, "chunking_identity": 0,
             "query_packages": [ { "checksum": "…", "quantization": "int8" },
                                 { "checksum": "…", "quantization": "fp32" } ] },
  "chunking_identity": 0,
  "passage_package": { "checksum": "…", "quantization": "fp32" }
}
```

`passage_package` is the package to embed with — one of `model.query_packages`, and the
warehouse's (§3).

### What the worker checks

1. Before embedding a record: SHA-256 of `embedding_text` equals `embedding_text_sha256`,
   and `key` is its prefix. A mismatch means the plan was damaged on its way; stop.
2. Before embedding anything, if it runs the shipped ONNX graph: the package's checksum is
   `passage_package.checksum`, and its tokenizer checksum, width, pooling and token cap are
   the family's.

### Output: a shard directory

| File | Content |
|---|---|
| `vectors.f32` | `records × dim` little-endian `f32`, in plan order. Each vector finite and of unit L2 norm (within 1e-3) |
| `keys.bin` | `records × 32` bytes: each record's `embedding_text_sha256`, raw, in the same order |
| `shard-manifest.json` | below |

Write `vectors.f32` and `keys.bin` under `.partial` names, flush them, rename them, and
write the manifest last: a directory with all three is a finished shard, and nothing
overwrites one.

`shard-manifest.json` (version 2):

```json
{
  "format": "otzaria-embed-shard", "version": 2,
  "plan_sha256": "<the embed manifest's plan_sha256>",
  "skip": 0, "take": 1000000, "records": 1000000,
  "dim": 256,
  "vectors_sha256": "<SHA-256 of vectors.f32>",
  "keys_sha256": "<SHA-256 of keys.bin>",
  "model": { … the embed manifest's model, verbatim … },
  "passage_package": { "checksum": "…", "quantization": "fp32" },
  "worker": { "name": "seforim-gpu-worker", "version": "1.0", "device": "NVIDIA …",
              "ep": "cuda", "mode": "torch" },
  "parity": { "reference": "onnxruntime 1.28.0 cpu fp32", "samples": 1000,
              "min_cosine": 0.9999997, "mean_cosine": 0.99999995,
              "document_sha256": "<optional: SHA-256 of the full certificate>" }
}
```

- `records` is `take`, or what remained of the plan after `skip`.
- `worker.ep` is the execution provider (`cpu`, `cuda`, `dml`, `rocm`, …); `worker.mode` is
  what ran the graph: `onnxruntime` for ONNX Runtime with the shipped graph, anything else
  (e.g. `torch`) for a re-implementation.
- **`parity` is required unless the worker is ONNX Runtime on a CPU** (`ep` `cpu`, `mode`
  `onnxruntime`). It reports the worker's agreement with ONNX Runtime on a CPU: at least
  1,000 texts (the int8 goldens plus plan texts sampled across the plan), and a lowest
  cosine of at least **0.999**.

A whole-library output may be one shard (`skip` 0, `take` = `records` = the plan's
records) or many; the windows must tile the plan.

## 3. Acceptance

`verify_shards` (run by `warehouse-add`) accepts a set of shards when:

- every manifest is version 2 and names one plan, family and package; with the plan, they
  are the plan's;
- each width is the family's, and the package is one of its query packages;
- the windows tile `[0, records)` exactly — no hole, no overlap — and each holds what its
  window owed;
- each file has the length its count implies and the SHA-256 its manifest declares;
- every vector is finite and of unit norm;
- with the plan, every key is the plan's `embedding_text_sha256` at its position;
- each worker is the reference, or carries a parity certificate that passes.

Without the plan — the import of a whole-library worker's output — everything but the
plan's own keys is checked.
