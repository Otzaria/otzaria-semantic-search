# Controlled ORT binding upgrade

Date: 2026-11-05 UTC. Change: [PR #29](https://github.com/Otzaria/otzaria-semantic-search/pull/29).

The upgrade preserves embedding results in the measured comparison and passes the
native compatibility gates. Its purpose is native ownership and error-handling safety;
the measurements do not establish a speedup. Machine-readable results, artifact
provenance and limitations are in
[ort-rc15-20261105.json](measurements/ort-rc15-20261105.json).

## Baseline and publication provenance

The investigation established consumption independently, before measuring:

| Layer | Verified source |
| --- | --- |
| Application | [`otzaria` dev `2ba2e6d76b53e7c72f37d048023f41d150f9702f`](https://github.com/Otzaria/otzaria/tree/2ba2e6d76b53e7c72f37d048023f41d150f9702f), locking hosted engine 7.0.1 |
| Hosted engine | [7.0.1 archive](https://pub.dev/api/archives/otzaria_search_engine-7.0.1.tar.gz), SHA-256 `b4896cc59b3cc8e75f31e66580ca8fa3a9f9820785b136994cd09a107841ceb6` |
| Engine source | [`225a6d5d7265c478a64d3a609a536d0deed433e7`](https://github.com/Otzaria/otzaria_search_engine/tree/225a6d5d7265c478a64d3a609a536d0deed433e7); inspected Rust manifests and native-build files match the hosted archive |
| Consumed semantic revision | [`6b006b0fe1440f8204385fbb42ee0d831bc8cfff`](https://github.com/Otzaria/otzaria-semantic-search/tree/6b006b0fe1440f8204385fbb42ee0d831bc8cfff) |
| Measured baseline | [`7edc614c3746531b21c71ff738162223297831c6`](https://github.com/Otzaria/otzaria-semantic-search/tree/7edc614c3746531b21c71ff738162223297831c6) |
| Measured functional candidate | [`a79901ff1a34bc9d57b994b905de7ec9ad69bb7a`](https://github.com/Otzaria/otzaria-semantic-search/tree/a79901ff1a34bc9d57b994b905de7ec9ad69bb7a) |

The consumed revision and measured baseline have byte-identical ONNX backend source;
the intervening changes concern segment cleanup. The publication chain is baseline
`7edc614` → initial draft `07437d6e097edf933881f8ef70e8bde85f5c8c66` → functional
candidate `a79901f`. The second commit fixes a test-fixture Clippy diagnostic and adds
the minimum-runtime and same-runner comparison gates. Independent QA matched all
nine published files to their local Git blob hashes. Author and committer were
verified as `palmoni5`. This report and its measurement summary are a subsequent
documentation addition; the execution evidence below identifies the functional head
explicitly. These changes have not been merged.

## Version choice and compatibility contract

`ort` and `ort-sys` move together from exact pins `2.0.0-rc.13` to
`2.0.0-rc.15`. **rc.15 remains a prerelease; no stable 2.x was available when
checked.** [rc.14](https://github.com/pykeio/ort/releases/tag/v2.0.0-rc.14) introduced
owned environments and raised its MSRV from Rust 1.88 to 1.92.
[rc.15](https://github.com/pykeio/ort/releases/tag/v2.0.0-rc.15) documents fixes for
more than 30 undefined-behavior cases and no new features. The published crate's VCS
metadata matches [upstream commit `5c564c5`](https://github.com/pykeio/ort/tree/5c564c58507f35fa4adc387a79fce1016e73956b).
Only those two packages change in the lockfile.

The dependency disables defaults and explicitly selects `std`, `load-dynamic` and
`api-17`. Dynamic loading, C API 17 and the reference Microsoft runtime 1.28.0 are
retained. No runtime binary, CPU instruction flag, int8 dispatch rule or precision
workaround changes. Tokenizers 0.23.2, model input wiring, pooling, normalization,
session tuning, backend identity `onnxruntime-sentence-v1`, and existing index formats
remain unchanged. No index migration is required for this change.

Rust 1.92 applies to the optional desktop ONNX dependency. No package-level
`rust-version` increase is imposed on default, non-ONNX or mobile builds. The locked
all-target ONNX check passes on Rust 1.92.0.

## Runtime and session ownership

The [backend](../src/semantic/onnx_backend.rs) now constructs sessions with an owned
`Environment`. One environment and the runtime library remain alive for the process;
each session holds its own environment reference. This is necessary because rc.15's
private weak singleton refuses recreation after its last strong reference disappears.
The retained environment contains no model weights and uses no global thread pool;
session resources still drop with the backend. Initialization locking ends before
session construction or inference. Output vectors own copied data before a session
lease returns to the bounded session pool.

Before `ort` can cache a library, preflight verifies its entry table, UTF-8 version
and non-null `GetApi(17)` result. It also compares the requested API table with the
binding's selected table while both library handles are alive. This catches a
different runtime initialized by another component instead of recording the wrong
path. A loaded library with failed environment construction remains distinguishable
from a ready environment. Failures return existing semantic error types, preserving
the caller's ordinary-search fallback contract; no lexical-search behavior changes.

There is an explicit host-compatibility boundary: rc.15 has no public accessor for
an environment already created by another component. Such a host receives a typed
error rather than silently borrowing that environment. The inspected engine has no
second `ort` consumer. See [ONNX_BACKEND.md](ONNX_BACKEND.md) for the full lifetime
and loading contract.

## Executed validation

All nine jobs of [CI run 42081611462](https://github.com/Otzaria/otzaria-semantic-search/actions/runs/42081611462)
completed successfully for `a79901f`: default/mock builds and tests on Linux, macOS
and Windows; ONNX-only and ONNX+mock native suites on those platforms; minimum
versions; actual-model goldens; and emulated x86 CPU goldens.

The [minimum-version job](https://github.com/Otzaria/otzaria-semantic-search/actions/runs/42081611462/job/127435823213)
uses Rust 1.92 and official Linux runtime 1.17.0. Its integration suite reports
11 passed, zero failed, four ignored helpers. Those helpers actually execute in
child processes through the passing parent tests. Coverage includes failed-model
recovery, partially constructed sessions, four complete two-session open/drop/reopen
cycles, concurrent embeddings, runtime-path identity and correction, and API 17.
Process exits are checked, so a crash after successful inference fails the test.
The runtime archive's pinned hash is locally measured from the official asset;
that historical release has no separately published GitHub asset digest.

The [comparison job](https://github.com/Otzaria/otzaria-semantic-search/actions/runs/42081611462/job/127435823597)
builds the identical [probe](../examples/ort_parity.rs) against both immutable
revisions, saving each executable before the next build. Both run on the same
AMD EPYC 7763 runner with Rust 1.99.1 and runtime 1.28.0, one session, one intra-op
thread and a 32-token cap. Each process uses 25 neutral inputs, 20 warm-up calls
and 100 measured calls. Existing 41-case goldens at the production 256-token cap
and an end-to-end artifact build also pass.

For the toy fixture and both real int8/fp32 graphs, token IDs and ordered top-five
neighbors are identical. Raw and normalized vectors are exactly identical, with
maximum absolute change zero. Independent QA recomputed normalization and neighbor
ordering from the raw JSON and confirmed these results.

| Graph | p50 before → after, µs | p95 before → after, µs |
| --- | ---: | ---: |
| Fixture | 26.279 → 26.870 | 42.659 → 35.587 |
| int8 | 6,688.784 → 6,679.105 | 11,289.089 → 11,338.721 |
| fp32 | 12,681.238 → 12,665.479 | 18,722.602 → 18,310.134 |

Rust allocation counts are unchanged: 22,232 per 100 fixture calls and 23,172 per
100 real-model calls. Requested bytes increase by 2,400 per 100 calls, or 24 bytes
per call. The [six-file evidence artifact](https://github.com/Otzaria/otzaria-semantic-search/actions/runs/42081611462/artifacts/13442225785)
contains neutral probe JSON only, with no model weights or tokenizer; its digest
and expiry are recorded in the checked-in measurement summary.

## Limits and disposition

This is one before/after process per graph on a shared runner. Timing is descriptive,
not a gate or a statistically established performance benefit. Rust allocation
counters measure calls and requested bytes, not retained memory, RSS or native
allocations. The 25-input corpus detects regressions; it does not establish retrieval
quality for the full library. Runtime 1.17 native execution was tested on Linux;
the three-platform matrix uses reference runtime 1.28.

Local baseline work and the candidate build encountered severe shared-host
contention. The broad baseline suite and candidate build were interrupted; neither
is counted as a completed local pass. The independent extended local lifecycle
probe was prepared but not executed against the candidate. The successful native
and same-runner comparisons above are actual CI executions, not inferred results.
No ANN default, index layout or exact-scan worker implementation changes belong to
this upgrade.
