# ANN feasibility results — 2026-11-05

## Decision

**Keep OXV exact scan as the production path, oracle and fallback. Keep USearch experimental. Do not introduce a reusable worker pool from these measurements.**

The experiment found an unfiltered synthetic workload on which an ANN graph is faster, but it also found serious counterexamples: poor recall on uniform vectors, much slower filtered searches, additional disk/RAM, and native traversal that continues after cancellation. No application dependency, search default, existing index bytes or worker policy changes in this PR.

These are newly generated baselines from this PR's own workflow. They do not reuse a measurement session from elsewhere. The local release executable also compiled successfully; the heavily contended local host was not used for the reported timing matrix.

## Provenance and scope

All figures below come from [workflow run 42083690818](https://github.com/Otzaria/otzaria-semantic-search/actions/runs/42083690818), where the native correctness job and all three observation jobs succeeded. The functional PR head was `e5fbae04ed9fa654512f86cf6e1527c9cb818299`, based on `7edc614c3746531b21c71ff738162223297831c6`. GitHub checked out the generated PR merge commit `c5c588ecd1bc293dc6faab7b06f8c7f0a3026835`; that is the clean revision recorded in every report. This is a test merge ref, not a merge of the PR into the repository.

All three jobs measured the same executable SHA-256, `02c89348d140901d9968b5741cb0e2252b0f985380481c8e4a890333e41b5d65` (27,527,768 bytes), compiled with Rust 1.99.1. The report records individual source-file hashes. USearch is pinned to **2.26.6**, F32 inner product, connectivity 16, construction expansion 128, **default features disabled**, with `ann_hardware_acceleration = serial`. Optional NumKong/OpenMP and new target-ISA flags are absent. These are portable scalar ANN measurements, not a claim about the best possible optimized ANN implementation.

| Fixture | CPU exposed by runner | Logical / affinity CPUs | Artifact |
| --- | --- | --- | --- |
| i8-clustered | AMD EPYC 7763 64-Core Processor | 4 / 4 | [13440499848](https://github.com/Otzaria/otzaria-semantic-search/actions/runs/42083690818/artifacts/13440499848) |
| i8-uniform | AMD EPYC 7763 64-Core Processor | 4 / 4 | [13441957570](https://github.com/Otzaria/otzaria-semantic-search/actions/runs/42083690818/artifacts/13441957570) |
| rot192-clustered | INTEL(R) XEON(R) PLATINUM 8573C | 4 / 4 | [13442615130](https://github.com/Otzaria/otzaria-semantic-search/actions/runs/42083690818/artifacts/13442615130) |

Each fixture contains **20,000 vectors × 256 dimensions**, 100 synthetic books, 50 independent queries, and `k=10`. Clustered data combine 128 shared centers with independent noise; uniform vectors are generated from independent components and then normalized. Query noise is separate from stored-vector noise. The rotated epoch is fitted on a bounded 4,096-row sample with a 128-component int8 head and 64-component four-bit tail. This does not evaluate any real Otzaria embedding model or released codec epoch.

Each backend/filter/expansion/thread case runs in its own fresh process, sequentially within its job. After a separately timed first query and three warmups, the 50 queries run twice, giving 100 warm timing samples. The two repetitions are not 100 independent queries. Percentiles are nearest-rank; there are no confidence intervals or multi-run significance claims. CPU affinity was observed, not pinned. Separate matrix jobs use separate shared runners, including different CPU models, so compare backends within a fixture rather than attributing cross-fixture latency differences solely to data or codecs.

The durable evidence is in [results/2026-11-05](results/2026-11-05): a compact JSON summary and a deterministic gzip of complete reports, query keys/scores and timing samples for each fixture. [manifest.json](results/2026-11-05/manifest.json) records artifact digests, source revisions, archive/summary hashes and sizes. The downloaded ZIP digests matched GitHub's artifact metadata before extraction. Gzip headers use timestamp zero; archives preserve the original parsed JSON values. The complete evidence remains in this PR after GitHub's 14-day artifact retention ends.

An earlier run had two valid int8 observations but rejected the rotated fixture because the harness called the production named-codec parser for a fitted codec. The harness now uses `Codec::fit_rot_i8i4`, exposes native subprocess errors, and includes a rotated lifecycle smoke test. All three fixtures were rerun on the corrected functional head; earlier timings are not mixed into these tables.

## Warm latency and recall

Latency is in **microseconds**. `load` copies a persisted native graph into allocated memory; `view` uses USearch's file mapping. The exact rows call the existing OXV scanner with one requested worker. ANN rows use one native query context and the stated search expansion. There is no fallback or insertion of exact hits in any measured ANN result.

Recall is measured against three separately built real OXV scans: original F32 vectors (`raw`), F32 reconstruction of the selected codec (`decoded`), and the selected encoded codec with its production query arithmetic (`OXV`). Missing neighbors count as misses; an empty reference is not applicable. The graph consumes decoded vectors without renormalizing their quantization error away.

### No book filter

| Fixture | Backend | p50 µs | p95 µs | p99 µs | Recall raw | Recall decoded | Recall OXV |
| --- | --- | --- | --- | --- | --- | --- | --- |
| i8-clustered | OXV exact | 219.97 | 246.26 | 255.50 | 0.974 | 0.998 | 1.000 |
| i8-clustered | load ef32 | 62.03 | 93.19 | 119.56 | 0.962 | 0.990 | 0.988 |
| i8-clustered | load ef128 | 139.26 | 175.34 | 189.70 | 0.972 | 1.000 | 0.998 |
| i8-clustered | load ef512 | 572.54 | 685.28 | 692.99 | 0.972 | 1.000 | 0.998 |
| i8-clustered | view ef32 | 86.20 | 122.48 | 156.38 | 0.962 | 0.990 | 0.988 |
| i8-clustered | view ef128 | 130.76 | 168.13 | 188.09 | 0.972 | 1.000 | 0.998 |
| i8-clustered | view ef512 | 587.42 | 728.11 | 787.45 | 0.972 | 1.000 | 0.998 |
| i8-uniform | OXV exact | 215.04 | 235.29 | 242.07 | 0.996 | 1.000 | 1.000 |
| i8-uniform | load ef32 | 113.75 | 168.61 | 203.28 | 0.182 | 0.184 | 0.184 |
| i8-uniform | load ef128 | 363.27 | 414.87 | 438.17 | 0.470 | 0.472 | 0.472 |
| i8-uniform | load ef512 | 1,222.51 | 1,282.29 | 1,338.91 | 0.888 | 0.888 | 0.888 |
| i8-uniform | view ef32 | 112.53 | 171.32 | 204.69 | 0.182 | 0.184 | 0.184 |
| i8-uniform | view ef128 | 369.90 | 424.14 | 447.04 | 0.470 | 0.472 | 0.472 |
| i8-uniform | view ef512 | 1,244.41 | 1,304.48 | 1,360.12 | 0.888 | 0.888 | 0.888 |
| rot192-clustered | OXV exact | 278.56 | 310.01 | 320.84 | 0.578 | 1.000 | 1.000 |
| rot192-clustered | load ef32 | 59.54 | 89.19 | 100.55 | 0.578 | 0.994 | 0.994 |
| rot192-clustered | load ef128 | 123.71 | 148.42 | 159.66 | 0.578 | 1.000 | 1.000 |
| rot192-clustered | load ef512 | 578.33 | 645.08 | 677.60 | 0.578 | 1.000 | 1.000 |
| rot192-clustered | view ef32 | 61.29 | 94.71 | 122.48 | 0.578 | 0.994 | 0.994 |
| rot192-clustered | view ef128 | 122.94 | 153.46 | 178.09 | 0.578 | 1.000 | 1.000 |
| rot192-clustered | view ef512 | 579.09 | 653.18 | 706.18 | 0.578 | 1.000 | 1.000 |

For clustered int8, loaded ANN at expansion 128 reduces observed median latency from 219.97 to 139.26 µs while retaining 0.998 recall against OXV. Expansion 32 is faster but misses additional neighbors. For uniform int8, expansion 128 has only 0.472 recall and is slower than exact; even expansion 512 reaches only 0.888 and costs 1.22 ms. Increasing expansion is therefore not a universal performance win.

The rotated case reaches 1.000 recall against its decoded and OXV references at expansion 128. Its 0.578 recall against original F32 vectors is already present in the exact codec reference. That loss must not be described as an ANN graph error, and this newly fitted synthetic epoch must not be described as the accuracy of a released Otzaria index.

### Filters applied before top-k

The 5% book filter admits 1,000 vectors and the 1% filter admits 200. OXV scans eligible book ranges; USearch applies a native traversal predicate. This is not postfiltering an unfiltered top-k. The table uses loaded ANN; mmap-view results for every case are retained in the JSON summaries and show the same key sets here.

| Fixture | Filter | OXV p50 µs | ANN expansion | ANN p50 µs | ANN p95 µs | Recall OXV |
| --- | --- | --- | --- | --- | --- | --- |
| i8-clustered | 5pct | 20.91 | 32 | 554.06 | 710.47 | 1.000 |
| i8-clustered | 5pct | 20.91 | 128 | 1,667.78 | 1,847.38 | 1.000 |
| i8-clustered | 5pct | 20.91 | 512 | 4,812.98 | 6,250.99 | 1.000 |
| i8-clustered | 1pct | 10.57 | 32 | 2,003.72 | 2,573.41 | 1.000 |
| i8-clustered | 1pct | 10.57 | 128 | 4,730.07 | 5,205.32 | 1.000 |
| i8-clustered | 1pct | 10.57 | 512 | 6,696.80 | 8,343.98 | 1.000 |
| i8-uniform | 5pct | 20.33 | 32 | 1,349.68 | 1,722.51 | 0.862 |
| i8-uniform | 5pct | 20.33 | 128 | 2,762.61 | 3,318.64 | 0.998 |
| i8-uniform | 5pct | 20.33 | 512 | 4,745.37 | 5,399.52 | 1.000 |
| i8-uniform | 1pct | 10.82 | 32 | 2,985.70 | 3,277.48 | 1.000 |
| i8-uniform | 1pct | 10.82 | 128 | 5,099.37 | 5,718.97 | 1.000 |
| i8-uniform | 1pct | 10.82 | 512 | 6,218.06 | 7,350.92 | 1.000 |
| rot192-clustered | 5pct | 59.77 | 32 | 598.30 | 747.70 | 1.000 |
| rot192-clustered | 5pct | 59.77 | 128 | 1,724.05 | 1,924.69 | 1.000 |
| rot192-clustered | 5pct | 59.77 | 512 | 4,379.55 | 4,520.32 | 1.000 |
| rot192-clustered | 1pct | 47.92 | 32 | 1,954.94 | 2,264.97 | 1.000 |
| rot192-clustered | 1pct | 47.92 | 128 | 4,956.16 | 5,162.35 | 1.000 |
| rot192-clustered | 1pct | 47.92 | 512 | 6,493.51 | 6,724.22 | 1.000 |

In these fixtures, all tested ANN settings are slower than the eligible-range exact scan under both filters. At expansion 128, the 5% filter costs approximately 29–136 times the exact median; the 1% filter costs approximately 103–471 times. Good filtered recall does not rescue that latency tradeoff. An eventual adapter would need an explicit filter-aware exact fallback policy and measurements of the crossover on representative data.

### Separating codec loss from graph loss

| Fixture | Filter | Decoded exact vs raw | Encoded OXV vs raw | Encoded OXV vs decoded |
| --- | --- | --- | --- | --- |
| i8-clustered | all | 0.972 | 0.974 | 0.998 |
| i8-clustered | 5pct | 1.000 | 1.000 | 1.000 |
| i8-clustered | 1pct | 0.990 | 0.990 | 1.000 |
| i8-uniform | all | 0.996 | 0.996 | 1.000 |
| i8-uniform | 5pct | 0.992 | 0.992 | 1.000 |
| i8-uniform | 1pct | 0.998 | 0.998 | 1.000 |
| rot192-clustered | all | 0.578 | 0.578 | 1.000 |
| rot192-clustered | 5pct | 0.960 | 0.960 | 1.000 |
| rot192-clustered | 1pct | 0.904 | 0.904 | 1.000 |

These comparisons also expose the small difference between decoded-F32 dot products and OXV's quantized-query arithmetic. USearch's native build uses fast floating-point math, so no cross-backend score-bit equality is assumed. Exact OXV keys, scores and order are checked across its requested thread counts; native exact keys/distances are checked internally across compaction and reopen.

## Disk, memory and construction

ANN storage is **additional to OXV** when retaining exact fallback. The graph uses F32 decoded vectors even when OXV uses a compressed codec. The raw and decoded F32 reference segments each occupy 20,893,696 bytes and are benchmark-only files, not a proposed production requirement.

| Fixture | OXV bytes | Extra ANN bytes | ANN / OXV | OXV + ANN / OXV | Codec fit ms | OXV build + install ms | ANN build ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| i8-clustered | 5,613,696 | 23,453,068 | 4.18× | 5.18× | 15.24 | 104.04 | 3,527.09 |
| i8-uniform | 5,613,696 | 23,453,068 | 4.18× | 5.18× | 7.95 | 69.11 | 8,863.73 |
| rot192-clustered | 3,730,560 | 23,453,068 | 6.29× | 7.29× | 1,487.36 | 879.39 | 4,409.36 |

Construction timings cover different operations: codec fitting is separate; OXV includes segment writing and installation; ANN builds its graph from decoded vectors. They are useful maintenance costs, not a controlled comparison of identical algorithms. The uniform graph took substantially longer to build on its runner. No extrapolation to a full library is made.

The following are unfiltered, expansion-128 search children after warm queries. RSS includes the executable, libraries and allocator; the RSS increase subtracts the same child's pre-open checkpoint. It excludes fixture construction and the other reference arrays because those are in other processes. MiB means 1,048,576 bytes.

| Fixture | Backend | RSS MiB | RSS increase MiB | Anonymous MiB | File-backed MiB | Peak RSS MiB | Native estimate MiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| i8-clustered | oxv | 10.63 | 5.85 | 0.44 | 10.20 | 10.63 | - |
| i8-clustered | usearch-load | 29.83 | 24.96 | 24.65 | 5.18 | 29.83 | 32.15 |
| i8-clustered | usearch-view | 28.85 | 24.20 | 1.36 | 27.48 | 28.85 | 0.15 |
| i8-uniform | oxv | 10.63 | 5.77 | 0.44 | 10.19 | 10.63 | - |
| i8-uniform | usearch-load | 29.86 | 25.03 | 24.75 | 5.11 | 29.86 | 32.15 |
| i8-uniform | usearch-view | 29.07 | 24.11 | 1.46 | 27.61 | 29.07 | 0.15 |
| rot192-clustered | oxv | 9.13 | 4.48 | 0.82 | 8.31 | 9.13 | - |
| rot192-clustered | usearch-load | 29.82 | 25.05 | 24.64 | 5.18 | 29.82 | 32.15 |
| rot192-clustered | usearch-view | 28.80 | 24.11 | 1.36 | 27.45 | 28.80 | 0.15 |

Mapping the ANN file reduces anonymous allocation, but does not eliminate resident memory or page-cache use. In these runs `view` still reaches approximately 29 MiB RSS, while its native allocation estimate is only 0.15 MiB. That estimate excludes mapped-file residency and cannot replace process RSS. The native loaded allocation estimate is 32.15 MiB, also not the same measure as observed RSS. Filtered cases allocate additional traversal state; their individual checkpoints are preserved in the summaries.

## Process-first and advisory-cold observations

The default run uses a fresh process and freshly opened handle with **uncontrolled file caches**. The second matrix issues `fsync` followed by `POSIX_FADV_DONTNEED` for only the selected fixture file before each child. Every request was accepted, but eviction was not verified and storage-device caches were not controlled. No global page-cache purge was used.

The next table reports **open time + first-query time**, separately, in milliseconds. Each entry is one first-query observation, not a percentile. Faults are changes during open / first query; both minor and major are shown. Counters can include allocator and library pages. Loaded ANN performs its file reading during open, so zero subsequent major faults does not mean no disk I/O occurred.

| Fixture | Backend | Default open + first ms | Advised open + first ms | Minor faults open / first | Major faults open / first |
| --- | --- | --- | --- | --- | --- |
| i8-clustered | oxv | 0.066 + 0.368 | 0.485 + 8.708 | 13 / 65 | 1 / 7 |
| i8-clustered | usearch-load | 19.661 + 0.297 | 37.628 + 0.275 | 6258 / 0 | 0 / 0 |
| i8-clustered | usearch-view | 1.372 + 1.083 | 4.994 + 55.441 | 354 / 138 | 2 / 165 |
| i8-uniform | oxv | 0.069 + 0.338 | 0.347 + 6.162 | 14 / 66 | 1 / 7 |
| i8-uniform | usearch-load | 18.493 + 0.641 | 35.255 + 0.722 | 6256 / 0 | 0 / 0 |
| i8-uniform | usearch-view | 1.331 + 1.678 | 4.769 + 47.902 | 353 / 259 | 2 / 211 |
| rot192-clustered | oxv | 0.618 + 0.633 | 2.898 + 44.017 | 165 / 41 | 1 / 4 |
| rot192-clustered | usearch-load | 15.089 + 0.221 | 223.088 + 0.338 | 6259 / 0 | 0 / 0 |
| rot192-clustered | usearch-view | 1.228 + 0.732 | 28.564 + 176.239 | 354 / 144 | 2 / 173 |

The advisory runs expose a much larger first-touch cost than the warm medians, especially for ANN mappings. The rotated job also had unusually slow advisory reads of its F32 reference (about 879 ms for the first raw scan); that makes the shared-runner storage variability visible. These observations cannot establish a guaranteed cold-cache SLA or justify comparing storage speeds across the different runners. Repeated controlled storage-cold runs on target hardware remain necessary.

## Delta, deletion, compaction and ownership

Every large fixture passed a real OXV lifecycle: 400 keys deleted, 1,200 added, 20,800 live, generations 1 → 2 → 3. OXV's exact keys/scores/order survive compaction, while retained readers keep their previous generation's answers. The native graph removes and inserts the same keys; every live/deleted key is checked after compact/save/load/view, and its native exact keys/distances remain identical across these operations. These checks exercise the real installer and compactor, not a mock generation counter.

| Fixture | OXV delta install ms | ANN remove + add ms | OXV compaction ms | ANN compaction ms | Post-delta recall all / 5% / 1% |
| --- | --- | --- | --- | --- | --- |
| i8-clustered | 8.18 | 199.97 | 29.00 | 204.30 | 1.000 / 1.000 / 1.000 |
| i8-uniform | 7.22 | 721.22 | 28.53 | 171.99 | 0.665 / 1.000 / 1.000 |
| rot192-clustered | 10.42 | 311.22 | 46.14 | 199.96 | 1.000 / 1.000 / 1.000 |

Post-delta recall uses 20 queries and ANN expansion 256. The empty filter is separately checked and excluded from recall averages. OXV delta timing excludes writing the source delta; native mutation includes encoding the new vectors and graph insertion. OXV compaction publishes a generation; native compaction here is timed before saving its file. These columns do not measure identical durability contracts.

After compaction, the graph file is 24,390,460 bytes in each fixture. OXV is 5,833,984 bytes for int8 and 3,869,952 bytes for the rotated codec. A corrupt ANN derivative is refused while the exact set stays readable. This is evidence that exact fallback remains available, not an implemented production adapter or automatic recovery protocol.

Native indexes own their handles and file mappings; insertion copies temporary vector buffers. Query buffers and filter closures remain alive synchronously through the C++ call. The experiment uses no unsafe buffer-backed view, custom metric pointer or concurrently mutating shared native handle. Read/update concurrency, crashed publication, repeated-delta churn and old-generation reclamation for a production ANN cache remain unqualified.

### Cancellation is a production blocker

Both paths reject pre-cancelled requests with the typed cancellation error. In the in-flight ANN probe, the first filter callback sets cancellation and all subsequent candidates are rejected. The native search still continues; the wrapper can discard its result only after return.

| Fixture | Predicate calls after cancellation | Cancel request → native return ms | Bounded native abort |
| --- | --- | --- | --- |
| i8-clustered | 20,799 | 10.36 | No |
| i8-uniform | 20,799 | 9.81 | No |
| rot192-clustered | 20,799 | 8.78 | No |

Returning `false` from a filter is therefore not a native abort. In these probes it caused traversal through the remaining live keys. The timing at 20,800 live keys is not a bound at library scale. Search, construction and compaction need bounded cancellation before a production backend can be considered.

## Worker reuse decision

The separate empty-scope probe measures launch/join overhead with the caller included in the requested thread count. It does not run OXV work and is not a CPU profile attributing time inside a real scan. Each cell below is p50 / p95 in microseconds over 100 samples.

| Fixture | 1 thread | 2 threads | 4 threads | 8 threads |
| --- | --- | --- | --- | --- |
| i8-clustered | 0.05 / 0.05 | 39.59 / 58.35 | 93.95 / 136.45 | 196.34 / 303.06 |
| i8-uniform | 0.05 / 0.05 | 38.76 / 53.10 | 86.55 / 115.20 | 182.63 / 258.16 |
| rot192-clustered | 0.06 / 0.07 | 38.76 / 53.51 | 72.11 / 88.87 | 145.03 / 190.46 |

| Fixture | Exact p50 µs: request 1 / effective 1 | Exact p50 µs: request 4 / effective 4 | Exact p50 µs: request 8 / effective 5 | 1-thread / 4-thread speed ratio |
| --- | --- | --- | --- | --- |
| i8-clustered | 219.97 | 204.88 | 266.30 | 1.07× |
| i8-uniform | 215.04 | 202.40 | 256.13 | 1.06× |
| rot192-clustered | 278.56 | 248.62 | 289.11 | 1.12× |

OXV's cap is based on roughly 4,096 eligible slots per share. At 20,000 vectors, a request for eight workers becomes five; on these four-CPU runners that can oversubscribe. Both filtered cases clamp to one worker, so reusable workers would not remove launch overhead from those paths.

The empty-scope cost is material at this small full-scan size, and four workers provide only a modest observed improvement. Requesting eight is slower here. It would be incorrect to call thread startup negligible, but equally incorrect to subtract an empty-scope median from the scan median and claim a measured pool benefit. No attributed CPU profile, representative concurrent request load or full-library measurement establishes a persistent pool as the right change. **No pool is added.** If later profiling identifies spawning as a bottleneck, compare a bounded pool against the existing cap while checking result order, cancellation, queue bounds, teardown and native ownership.

## Independent QA and remaining production gates

An independent QA agent reviewed metric/oracle separation, before-top-k filters, native ownership, retained OXV snapshots, compaction, cancellation semantics and cache labeling. The corrected CI smoke covers both an int8 200 × 32 fixture and a fitted rotated 200 × 256 fixture; all four Rust tests and six Python tests passed. Independent QA additionally exercised an uneven 1,001 × 64 `i8-sym-dim` fixture with `k=25`: the 1% filter admits 11 vectors and returns 11, the empty filter returns zero, and a 20-delete/60-add lifecycle produces 1,041 live keys with retained-snapshot and restoration parity.

QA independently downloaded and verified the three final artifact ZIPs, then
recomputed all 360 recall summaries (18,000 per-query comparisons), ordered-key
hashes and warm percentiles from 13,500 timing samples across 135 measurement
cases. The verifier did not import the benchmark runner. Filter membership,
eligible counts, OXV thread parity and recorded lifecycle contracts also passed.
The [independent verifier](verify_evidence.py) and its
[retained-evidence instructions](results/2026-11-05/README.md) are included.

The implementation and results do not meet the production gates in [README.md](README.md). In particular:

- Measure representative real vectors and queries, full-library sizes, repeated runs, older/constrained CPUs, supported operating systems, and verified storage-cold behavior. Synthetic neighbor recall is not a semantic relevance judgment.
- Preserve window/alias penalties before top-k, all occurrence records, foreign links, moved-text resolution and `scan_with` extra candidates. These fixtures contain one primary occurrence per vector.
- Preserve full key identity. The fixture deliberately places a u64 ID in the first eight bytes of a synthetic `ChunkKey`; a real adapter needs a collision-free mapping for full 128-bit keys where used, not truncation into USearch's u64 key space.
- Tie every derivative to model identity, codec epoch/digest, OXV generation and graph parameters, with atomic publication, crash recovery, stale/corrupt-graph rejection and exact fallback. Existing OXV formats must remain readable.
- Bound native cancellation and memory, validate concurrent readers/mutation ownership, measure repeated-delta churn and rebuild costs, and choose a filter-aware exact fallback using evidence.
- Profile actual worker costs before experimenting with a bounded persistent pool.

The unfiltered clustered speedups justify retaining a reproducible experimental benchmark. The complete tradeoff measured here does not justify changing the production search backend or its worker lifecycle.
