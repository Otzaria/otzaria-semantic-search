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

## 4. The warehouse

Every vector the build machine ever embedded, as `f32`, keyed by the **full SHA-256** of
its text. It makes a codec change, a new base and a revived key free of GPU work.

| File | Layout |
|---|---|
| `warehouse.json` | `{format: "otzaria-vector-warehouse", version: 1, identity, records, batches}` |
| `vectors.f32` | `records × dim` little-endian `f32`, append-only |
| `keys.bin` | `records × 32` bytes, each record's SHA-256, in the same order |
| `index.bin` | `b"OXVWIDX1"`, `u64` count, then `{sha256 [32], record u64}` ascending by sha256 |

- `identity` is what a vector depends on given its exact text: `family_id`,
  `tokenizer_checksum`, `embedding_dim`, `pooling`, `max_tokens` — and **one**
  `passage_package`. A shard of another package is refused: two packages of a family are
  close, not equal, and a segment must not mix them.
- `batches` records every add: its first record and count, the digests of what it
  appended, the plan it was checked against (`null` for an import), the worker and its
  parity certificate.
- `warehouse.json` is the commit point. An add appends both data files, writes a new
  index under a temporary name, renames it, and only then records the new count; the next
  add truncates whatever a crash left past the count and rebuilds an index that is not the
  count's.
- A key the warehouse holds is not added again.

```sh
otzaria-semantic-search warehouse-add --warehouse <dir> [--create --model <model.json>] \
    [--passage-quantization fp32] [--plan <plan dir>] --shards <dir> [--shards <dir> …]
```

With `--plan`, the shards are held to the plan (§3); without it — an import — to
everything but the plan's own keys.

### Importing a whole-library output

The v30 run on winpc (`$W/out/v30-meivin-r2-fp32/`: `vectors.f32` [6,347,587 × 256],
`keys.sha256` with 32 raw bytes a row, `manifest.json`, `certificate.json`) is the shard
layout but for two names. To import it:

```sh
cd $W/out/v30-meivin-r2-fp32
ln keys.sha256 keys.bin
otzaria-semantic-search adopt-shard --dir . \
    --model config/models/meivin-round2-onnx/model.json --passage-quantization fp32 \
    --plan-sha256 <the run manifest's plan_sha256> \
    --worker-name torch_bert --worker-version <the scripts' sha256> \
    --device "AMD Radeon RX 9060 XT" --ep rocm --mode torch-mixed \
    --parity-reference "onnxruntime 1.28.0 cpu fp32" --parity-samples 20480 \
    --parity-min-cosine 0.99999969 --parity-mean-cosine 0.99999986 \
    --parity-document certificate.json
otzaria-semantic-search warehouse-add --warehouse /srv/otzaria-vectors/<id8>/warehouse \
    --create --model config/models/meivin-round2-onnx/model.json \
    --passage-quantization fp32 --shards .
```

`adopt-shard` hashes both files and writes `shard-manifest.json` (window 0..records). The
import then re-reads every row: its digest, its finiteness, its unit norm, and the parity
rule for a worker that is not ONNX Runtime on a CPU. The worker's own plan was a version 1
`plan.jsonl`, so its digest is recorded and not compared. The fp32 package checksum in
`config/models/meivin-round2-onnx/model.json` (`4a4a2ae8…`) is the run's passage package.

## 5. Assembly

```sh
otzaria-semantic-search assemble --kind base --plan $P --warehouse $W --out $R \
    [--codec i8-sym-vec|i8-sym-dim|f32] [--clip-q 1] [--keep-epoch --previous $L] \
    [--created-at T] [--built-by '{"runId":"…"}'] [--verify]
otzaria-semantic-search assemble --kind delta --plan $P --warehouse $W --previous $L \
    [--previous-version N] --out $R [--verify]
```

`$L` is the directory holding the previous release's ledger — the `--out` of its
assembly. The plan must have been split against that ledger.

* **A base** ships every (book, key) record of the plan: books in byte order, lines in
  order, a key's first record its slot, a later one in another book an extra.
* **A delta** ships the keys the previous ledger lacks as slots, their later records as
  extras, new (book, key) pairs of keys it holds as foreign records, and the ledger keys
  the plan no longer has as tombstones. A delta with no new key ships no vector, and its
  provenance names no worker: none embedded anything.
* **Vectors** come from the warehouse by the full SHA-256 of their text, read 4096 at a
  time in warehouse order. A key the warehouse lacks fails the assembly, with the count
  and the first missing digest.
* **The codec epoch.** A base takes a new one: `i8-sym-vec` (the default) needs no
  calibration. `i8-sym-dim` is calibrated here, exactly — the ⌊clip_q·(n−1)⌋-th smallest
  `|x|` per dimension over the slots' vectors, found in two histogram passes over the
  warehouse. A delta always keeps its ledger's epoch. The codec's name is the identity's
  `store.vector_precision`, so a delta in another epoch or identity is refused by the
  ledger.
* **Memory.** Records, ledger and warehouse are mapped. About 70 bytes are held per slot
  shipped (key, hint, warehouse record, the classification's entry), plus 64 MB of
  histograms for `i8-sym-dim`. The test `assembly_memory_is_bounded` holds the growth to
  under 200 bytes a slot.
* **Deterministic.** The same plan, warehouse, ledger and `--created-at` give the same
  bytes.

`--out` receives `segment.oxv`, `manifest.json` and `payloads.json` (the metadata-v3
package), `release.json` (the release manifest of
[`ARTIFACT_CONTRACT.md`](ARTIFACT_CONTRACT.md), with `requires` =
`{indexSchemaVersion: 5, lineTextVersion, keyVersion}` and `builtBy`), and this
version's ledger: `ledger-vN.keys`, `pairs-vN.bin` and `ledger-vN.manifest.json`. The ledger also
records the passage package, and a delta from a warehouse of another package is refused.

**Published files.** After compressing (and splitting) the segment, the manifest that is
published lists the files, each entry in the shape of the updater's `PatchFileEntry`:

```sh
otzaria-semantic-search release-files --release $R/release.json --compression zstd \
    --files otzaria-vectors-<id8>-v29-v30.oxv.zst --out otzaria-vectors-<id8>-v29-v30.manifest.json
```

Each entry is `{file, compression, sha256, size, uncompressedSha256, uncompressedSize}`.
The uncompressed values are the whole segment's, also for one part of a split file. The
command prints the SHA-256 of the manifest it wrote: that is the value published outside
it, which an install checks. `files` is not part of the package digest.

A removed book's records go when their keys go. A record of a removed book whose key
lives on in another book stays in the device's set, and a resolver drops it as stale. A
compaction keeps it too, because a `LiveKeySource` that does not know a book keeps that
book's records. It goes with the next base.

## 6. Gates

`assemble --verify` checks the release in `--out` against its inputs. It writes
`gates.json` and exits with status 2 if a gate fails. Without `--kind`, it checks a
release that was already assembled.

| Gate | Checks |
|------|--------|
| G1 | The identity is complete and the segment's own; the codec is the declared one; every scale is finite and > 0; for `i8-sym-dim`, at most 1e-4 of the components are clipped |
| G5 | Every slot holds its key's warehouse vector, encoded. On a 20,000-slot sample (`--samples`), the decoded vectors' cosine with their f32 originals has a mean ≥ 0.9995 and a 0.1st percentile ≥ 0.998 |
| G7 | A base is ≤ 2.0 × 10⁹ bytes; a delta is ≤ 0.15 × its base, otherwise publish a base |
| G8 | Assembling again into `<out>.g8` gives the same segment, package, release manifest and ledger, byte for byte |
| G9 | `verify_for_install` passes, reading every payload byte, under the manifest's `packageDigest`; the segment is the manifest's |
| G10 | Reported, not enforced: the base plus its deltas as a multiple of the base, with a note past 1.3 |

G2–G4 and G6 need the release index and belong to the plugin's validator. The library
gives it these pieces in `distribution::gates`:

* `simulate_device(dir, chain)` installs releases, oldest first, into a new set and
  opens it with the runtime reader.
* `coverage(set, plan)` counts the plan's records whose (book, key) a scan reaches (G3).
* `book_records(set, book, out)` lists a book's reachable records with their hints, for
  resolving (G4).
* `ExactReference::new(set, warehouse).top_k(query, k, threads)` is the exact f32 scan
  of the set's live keys. `recall(found, exact)` compares it with the set's own scan
  (G6).
