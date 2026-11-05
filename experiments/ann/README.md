# ANN feasibility experiment

This directory is an **experimental, separately built Cargo crate**. The application
continues to use its OXV exact scan. No production dependency, feature, index format,
manifest, migration, worker policy, or default changes when this experiment is merged.

[Measured results](RESULTS.md) cover three independently built 20k × 256 fixtures,
including recall, filter costs, RAM/disk, advisory-cold observations and lifecycle
checks. The [durable evidence](results/2026-11-05/manifest.json) retains complete
raw results and source/binary hashes. The decision is to keep ANN experimental.

## Questions it can answer

For a fixed reproducible synthetic dataset, how much do an HNSW graph's latency,
recall, extra storage and native memory differ from the real OXV reader/scanner?
Does that difference survive book filters? Do deletions, incremental insertion,
compaction, persisted reopening and cancellation preserve the basic contracts?

It uses [USearch 2.26.6](https://github.com/unum-cloud/USearch/tree/v2.26.6), exactly
pinned in its own lockfile. `default-features = false` excludes NumKong and OpenMP;
the measured graph uses **F32 inner product, portable scalar kernels**. It adds
no `target-cpu=native`, AVX2 or other ISA requirement to any production build.
The C++ bridge is compiled only when this separate crate is built. This deliberately
conservative experiment does not qualify optional SIMD kernels, other compilers,
or an older Rust toolchain.

The tested data have independent queries and deterministic noise. `clustered`
combines 128 shared unit centers with independent noise; `uniform` distributes
unit vectors without those centers. There are 100 synthetic books, assigned in
contiguous ranges as OXV scans them. Neither distribution is a real embedding
corpus, a model-quality evaluation, or evidence of full-library performance.

## Run

Build once, then time binaries without Cargo in the timed process:

```sh
cargo build --release --locked --manifest-path experiments/ann/Cargo.toml -j 1
python3 experiments/ann/run.py \
  --dataset /tmp/otzaria-ann-new-fixture \
  --output /tmp/otzaria-ann-report.json \
  --raw-dir /tmp/otzaria-ann-raw \
  --vectors 100000 --dim 256 --queries 100 --top-k 10 \
  --codec i8-sym-vec --distribution clustered
```

When setting `CARGO_TARGET_DIR`, also pass
`--binary /that/directory/release/otzaria-ann-experiment`.
The output file, raw directory and dataset must not exist. The tool never removes
or overwrites a supplied directory. It leaves generated files in the explicitly
chosen directory so the caller can inspect them and remove them later. A failed
build can leave a partial fixture; only a fixture with `build.json` completed.

Repeat with `--distribution uniform` and with `--codec f32`, `i8-sym-dim`, and
`rot192-i8i4`. The rotated codec requires at least 192 dimensions and fits an epoch
on at most 4,096 rows. It does not alter the codec or any real release's calibration.
Each run writes OXV raw-F32, decoded-F32 and encoded segments, an installed encoded
set for lifecycle checks, plus an ANN derivative. Budget several times the vector
payload in disk space; setup reports the separate sizes and build time.

The default matrix measures 1/4/8 OXV threads, 32/128/512 ANN search expansion,
and all/5%/1% book filters. Use `--threads`, `--expansions`, `--filters` and
`--repeats` to narrow it. `--filters all,5pct,1pct,empty` additionally times a filter
admitting nothing (the lifecycle/test suite always checks it). Filter percentages
describe books; an uneven/small corpus can have a different admitted vector count.
`--reuse-dataset --skip-lifecycle` explicitly repeats timings on a completed fixture
after its lifecycle has already run; the report records the stored configuration.

For one measurement or a launch/join proxy:

```sh
experiments/ann/target/release/otzaria-ann-experiment measure \
  --dir /tmp/otzaria-ann-new-fixture --backend oxv --threads 4 --filter 5pct
experiments/ann/target/release/otzaria-ann-experiment threads --samples 100
```

For deterministic correctness checks:

```sh
cargo test --locked --manifest-path experiments/ann/Cargo.toml -j 1
python3 -m unittest discover -s experiments/ann -p 'test_*.py'
```

The separate `ANN experiment checks` workflow runs two 200-vector native smoke fixtures
(int8 and fitted rotated codec)
and Python recall checks on Linux when this experiment changes. Root Cargo CI
does not implicitly include this separate workspace. After the smoke passes,
three observational jobs measure 20k × 256 fixtures (clustered int8, uniform int8,
clustered rotated codec), preserve JSON/raw results as Actions artifacts, and
repeat a smaller matrix with per-file advisory cache eviction. These are measured
observations on shared runners; there is no CI latency, RSS or recall threshold.

## Three separate references

All exact references call the existing `scan_segments`, including its book ranges,
top-k selection and tie order; the experiment does not reimplement that scanner.

| Reference | What it isolates |
| --- | --- |
| `raw` | Exact scan of original synthetic unit F32 vectors; reference for loss from storage codecs. |
| `decoded` | Exact F32 scan of the selected codec's decoded vectors; reference for the ANN graph's own approximation. |
| `oxv` | Exact production scan of the selected encoded segment, including quantized-query arithmetic; reference for replacing the currently shipped retrieval path. |

USearch is built from the decoded vectors with **inner product**, without cosine
renormalization or another storage quantizer. Cosine over normalized decoded
vectors would change the score contract. Its float arithmetic can still differ
from OXV's integer query kernel; comparison to both `decoded` and `oxv` exposes
that difference. Report `recall_vs_raw`, `recall_vs_decoded`, and `recall_vs_oxv`
separately. A low raw score on a reduced codec must not be attributed entirely to
the ANN graph. OXV's equality between thread counts checks keys, scores and order.

Recall is the fraction of the oracle's `min(k, eligible)` keys present in the
candidate set. Missing results are misses; duplicates cannot inflate recall;
an empty oracle is **not applicable**, never a perfect-recall data point. The
report includes the mean, minimum and number of full-recall queries. Both systems
apply the book predicate **before top-k**: OXV plans eligible ranges, USearch
applies `filtered_search` while traversing the graph. Postfiltering an unfiltered
top-k would be an invalid comparison and is not used.

## Time and memory definitions

Each backend/filter/expansion/thread case runs in a **fresh process**, sequentially.
It loads only its own index, not the reference arrays. Model construction, codec
calibration, oracle calculation and query generation are outside measured search.
The first search is timed before warmups; three warmup searches precede the measured
query set. Warm samples include result construction, as production callers need it.
All timings are in microseconds, with nearest-rank p50/p95/p99 and raw samples when
`--raw-dir` is used. Build/calibration time is reported separately in milliseconds.

**Fresh process/open is not OS-cache-cold.** Setup and earlier subprocesses may
have warmed filesystem caches. By default the file cache is uncontrolled. With
`--cache-policy advisory-cold`, the Python driver opens only the selected fixture
index, calls `fsync` and `posix_fadvise(POSIX_FADV_DONTNEED)` before each measurement
child, and records whether that advisory request was accepted, failed, or was
unsupported. This never guarantees eviction or purges global/other files' caches.
Open time and first-query time are separate from warm-query latency. Compare
accepted advisory runs with the default run on the same machine; a verified
storage-device cold measurement remains a production gate.

On Linux, `/proc/self/status` records RSS, anonymous/file-backed RSS, virtual size
and process peak RSS before open, after open, after the first query and after warm
queries, plus cumulative minor/major faults from `/proc/self/stat`. Differences
between checkpoints include library/allocator/thread-stack faults, not just index
file pages, and must not be treated as proof of cache eviction. Unsupported
platforms report null, not invented RAM figures. USearch's
own allocation estimate is additional, not interchangeable with RSS. The peak
belongs to that child process; fixture construction and reference backends are not
resident there. The lifecycle check intentionally holds several generations and
is not used as steady-state search memory evidence. mmap can still consume resident
RAM/page cache; its mapped file size is not a heap allocation estimate.

The host CPU, cgroup limits, load average, source hashes and measured executable's
SHA-256 accompany reports. Rebuild after source changes; an arbitrary `--binary`
is not assumed to match the source files on disk.
The binary does not set affinity or exclude other workloads; scheduling/contention
must be considered before interpreting a latency improvement. The empty-scope
probe is a **launch/join proxy**, not a profiler attributing OXV time to spawning.
OXV already caps workers based on roughly 4,096 slots per share. No persistent
pool is introduced based on that proxy or on a noisy single-host timing.

## Lifecycle and ownership checks

The lifecycle command installs a real OXV delta with 2% tombstones and 6% new keys,
then compacts it using the existing installer and compactor. It verifies exact
keys/scores/order before and after compaction and that readers retaining earlier
generations continue answering their original snapshots. In the separate native
graph it removes/adds the same keys, compacts, saves, loads and mmap-views the file,
checks every key's membership, and compares native exact results across compaction
and both reopen modes. Approximate recall after delta is reported separately.

The Rust wrapper owns USearch's index handle. `add` copies each temporary input;
we use owned file mappings, not `view_from_buffer`, and install no custom metric
pointer. Query and filter closures remain alive synchronously until their native
calls return. No native handle is shared with a mutating worker in this experiment.
The loaded/mapped restoration checks therefore do not qualify concurrent
search/update ownership for a production adapter.

Upstream's C++ build enables `-ffast-math` (or `/fp:fast`). The experiment therefore
does not assert float score-bit equality between USearch and OXV; it compares
neighbor sets separately against the three references. Native exact results are
also checked before/after compaction within the same executable and host.

Pre-cancelled requests fail before entering either search path. **USearch's tested
Rust API has no bounded native search abort**. A deterministic probe sets the
cancellation flag during its first predicate callback; rejecting subsequent keys
does not stop traversal. The report counts callbacks and measures time until the
native call returns; only then can a wrapper discard the result. This is a production
blocker, not successful mid-query cancellation. A corrupt ANN file is refused while
the original exact set stays readable. Measurements fail on ANN errors; they never
silently report an exact fallback time as an ANN success.

## What blocks production adoption

The experiment intentionally leaves exact scan as both the shipping implementation
and the reference. Production adoption additionally requires all of the following:

- Recall/latency/RAM/disk evidence on representative real vectors and queries, all
  supported codecs and full-library scale, including constrained/older CPUs and
  true storage-cold runs. A synthetic speedup alone is insufficient.
- OXV's window and alias penalties **before top-k**, all occurrence records,
  foreign links, moved-text resolution and the host's `scan_with` extra candidates.
  This fixture has one primary occurrence per vector and does not test those.
- A derivative manifest tied to model identity, codec-parameter digest, OXV
  generation and graph parameters; atomic publication with crash recovery; safe
  rejection/rebuild of a stale or corrupt graph. No existing OXV bytes may be
  reinterpreted as an ANN file.
- A collision-free mapping of full production chunk keys into USearch's u64 key
  space. The fixture's synthetic IDs deliberately fit in eight bytes; truncating
  an arbitrary 128-bit key would lose identity and is not an integration strategy.
- Bounded cancellable native search, graph build and compaction; filtered-candidate
  budgets and an explicit exact fallback; ownership/locking for concurrent readers
  during delta installation and generation replacement.
- Measured graph build/maintenance cost, removed-key churn and recall after repeated
  deltas, not just the single delta in this harness. Graph compaction and OXV
  compaction are different operations with different persistence contracts.
- Profiling before considering a bounded reusable worker pool. Thread reuse must
  preserve cancellation, result order, memory limits and unload ownership.

## Source references

- OXV implementation: [`scan.rs`](../../src/semantic/oxv/scan.rs),
  [`codec.rs`](../../src/semantic/oxv/codec.rs),
  [`reader.rs`](../../src/semantic/oxv/reader.rs),
  [`segment_set`](../../src/semantic/segment_set/).
- Existing measurement tools: [`vector_search`](../../benches/vector_search.rs),
  [`vector_set_scale`](../../tests/vector_set_scale.rs). This experiment supplements
  their scope rather than replacing them.
- [USearch 2.26.6 Rust API source](https://docs.rs/crate/usearch/2.26.6/source/rust/lib.rs):
  filtered/exact search, file load/view, compaction, owned native handle and unsafe
  buffer-view contract.
- [USearch 2.26.6 build](https://docs.rs/crate/usearch/2.26.6/source/build.rs) and
  [manifest](https://docs.rs/crate/usearch/2.26.6/source/Cargo.toml): optional native
  features, dynamic dispatch when enabled, and C++ compiler flags.
