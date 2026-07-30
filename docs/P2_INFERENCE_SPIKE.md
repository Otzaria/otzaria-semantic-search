# P2 — Real inference: what was built, what it costs, what was decided

The roadmap's P2 row asks for four things: a backend trait, output compared against a
Python reference, identical tokenizer / EOS / last-token / L2 behaviour, and real batch
inference. All four landed. This document is the decision record — what the numbers
actually are, which decisions were made on evidence and which were made by fiat, and what
P3 and P4 inherit.

Companion documents: [`P2_REFERENCE_VECTORS.md`](P2_REFERENCE_VECTORS.md) for the golden
data and the tolerance policy, and [`CODE_MAP.md`](CODE_MAP.md) for where the code lives.

---

## 1. The decision, and its honest provenance

**The backend is llama.cpp**, via `llama-cpp-2` pinned at exactly `=0.1.153`, behind the
non-default `llama-backend` feature.

The roadmap (§6, the paragraph after the stage-א table) asks for something this document
does **not** contain: a spike comparing Candle and llama.cpp on vector agreement,
performance, and build size. That comparison was not performed. The project owner decided
llama.cpp-only, explicitly, *before* any measurement existed. This is recorded plainly
because a reader six months from now will otherwise assume the choice was measured, and it
was not.

What that decision costs:

- **No measured alternative** for the three things Candle would have been evaluated on:
  cross-compilation risk across the five Cargokit targets, build weight, and pure-Rust
  portability. The llama.cpp figures in §5 are absolute, not comparative.
- **A C++ build dependency** on every target. It builds cleanly here (§5), and
  `llama-cpp-2` exposes Android STL features, but no non-macOS target was exercised.
- **One `unsafe` block** (§6) that a pure-Rust backend would not have needed.

**What would reopen it.** The option is live, not closed:

- `candle-transformers` 0.11 ships `quantized_qwen3`, so the model architecture is
  supported upstream. That was the original blocking uncertainty and it no longer holds.
- A Candle backend would additionally need **its own Qwen2-BPE tokenizer built from the
  GGUF metadata** — `tokenizer.json` exists only in a gated repo (§3), so it cannot simply
  be loaded. The golden `token_ids` make that tokenizer testable, which is most of why
  they exist.
- Concrete triggers: a Cargokit target where the cmake build cannot be made to work; a
  build-time or artifact-size cost that proves unacceptable downstream; or `llama-cpp-2`
  becoming unmaintained at a version this crate cannot stay pinned to.

The `EmbeddingBackend` trait exists so that this decision is reversible. Adding a second
backend is one row in `CANDIDATES` plus a constructor; the golden corpus then measures it
against the same thresholds, with no changes to the engine, the manifest, or the tests.

---

## 2. Correctness — what is established

Full evidence and methodology in [`P2_REFERENCE_VECTORS.md`](P2_REFERENCE_VECTORS.md).
The short version, measured against the 31 committed golden records:

| Check | Result | Threshold |
|---|---|---|
| **`token_ids` exact equality** | **30/30** | exact — this is the primary gate |
| Rejected truncation convention | correctly does **not** match | — |
| Worst cosine vs golden | **0.996144358** (`teamim_shema`) | ≥ 0.99 |
| Worst max component diff | **1.2886e-2** (`unrelated_a`) | ≤ 0.03 |
| Worst relative raw-norm diff | **3.1492e-2** (`over_512_trunc_total_512`) | ≤ 0.05 |
| Batch vs single-sequence | **0.995430753** | ≥ 0.99 |
| Relational ordering margin | **0.800808** | ≥ 0.405 |

Two things about that table are easy to misread.

**`token_ids` carries the weight, not cosine.** A wrongly prepended BOS scores cosine
0.9947938 while a *legitimate* independent reference scores 0.9947909 — the bug outscores
the real thing, so no cosine threshold can separate them. Truncation off-by-one passes at
0.99838. The vector tolerances catch gross errors only (mean pooling 0.28–0.79, CLS
−0.04–0.15, omitted EOS 0.11–0.95). This inverts how the thresholds were originally
presented and is the single most important thing to preserve if the tests are ever
rewritten.

**The forward pass is not independently verified by these numbers.** The Rust backend and
the golden generator both run llama.cpp, so their agreement is largely tautological for the
model itself; what it genuinely proves is the harness — tokenization, EOS, no BOS,
last-token selection, L2, truncation, ordering. The independent evidence is a
PyTorch/`transformers` cross-check that dequantizes the same GGUF to fp32 and runs its own
Qwen3 implementation, agreeing at cosine **0.99479–0.99880**. Both references read the same
quantized file, so **a quantization error in the published GGUF is invisible to both**.

The 0.996144 figure is a *cross-build* measurement, not a same-build one: `llama-cpp-2`
0.1.153 vendors a different llama.cpp commit than `llama-cpp-python` 0.3.34. It is better
than both documented triangulations (CPU-vs-Metal 0.99491, llama.cpp-vs-PyTorch 0.99479),
which is the expected result and closes the open question §8 of the reference document
raised.

### Cross-sequence contamination: ruled out

Worth stating separately because two earlier investigations left it open. Holding token
length constant makes `[A,B]` and `[A,C]` produce batches of identical shape — same
`n_tokens`, tensor dims, GEMM tiling, thread partitioning, and A in the same rows and KV
cells — so every floating-point explanation is held constant and any difference in A could
only come from reading B's or C's tokens. Across seven experiments (A at offset 0; at
offset 40; long vs short neighbour; neighbour that is a prefix of A; batch of 8; batch
filling `n_ctx`; across decode groups): **bit-identical, 0 of 1024 components differing**.
Confirmed at source level too — `llama-kv-cache.cpp:1617-1624` masks every
`(query, kv-cell)` pair outside the sequence with `-INFINITY`, and unlike `causal`/`swa`
that check is not behind the `unified` flag.

---

## 3. Known and accepted properties

These are real, bounded, and deliberately not treated as bugs. They are listed because each
one is an assumption something downstream might otherwise make.

**Vectors depend on `batch_size`, which is not part of index identity.** Re-indexing the
same corpus at `batch_size` 1 versus 32 gives worst cosine 0.996721 and **0 of 32 records
bit-identical**. A sequence's vector is a function of *(its own tokens, its row offset in
the decode group, the group's total token count)* — ggml kernel and reduction-order
selection, not contamination. The divergence is far below any retrieval threshold and
consistent with the 0.9948 cross-build tolerance already accepted, so it is left out of the
manifest rather than forcing a re-index whenever a caller tunes throughput.

**`n_ubatch` is pinned at 256 and is a correctness-relevant knob.** Below 256 it moves
vectors: 128 moves 1 of 65, 64 moves 7 of 65. 256 is the largest reduction that moves
nothing, which is why it is the value. It is *not* in index identity either, so a value
that moved vectors would manufacture exactly the silent-re-index hazard described above.

**Nothing downstream may assume bitwise reproducibility across platforms.** CPU-vs-Metal
agreement is cosine 0.99491. For P9's plan to ship one official pre-built index to every
platform this is fine — 0.995 is far above any retrieval threshold — but a checksum or
equality scheme over vectors would break the moment it crossed a platform boundary.

**Three knobs provably do not move a vector:** `n_threads` ∈ {1,2,4,8}, `n_seqs` at
constant token total, and re-running an identical configuration. Inference is
deterministic.

---

## 4. Performance and resources, as measured

All figures on Apple Silicon, macOS, CPU inference (`n_gpu_layers = 0`), Q4_K_M weights,
`n_ctx = 512`. **No Android, iOS, Windows or Linux measurement was taken**; see the caveat
at the end of this section before budgeting from these numbers.

### Latency and threading

Single-batch latency against thread count: **7.55 s → 3.93 s → 2.18 s** for 1/2/4 threads,
then flat (2.19 s at 6, 2.25 s at 8). `DEFAULT_THREADS_CAP = 4` is where the curve
flattens, which is also llama.cpp's own default.

Two contexts give **1.93×** the throughput of one — the pool genuinely parallelises, and
nothing serialises on the pool mutex (it is held for a `pop`/`push`, never across a decode).

`contexts × n_threads` is clamped to available parallelism, because the two knobs are
independent and the machine is not. Unclamped, 4 contexts × 4 threads on 10 cores measured
**13.58 s / 0.29 batch/s** against 4 × 2's **6.43 s / 0.62 batch/s** — a 2.1× throughput
loss purely from oversubscription. With the clamp, a requested 4 × 4 becomes 4 × 2 and
recovers it. This matters most on a 4-core phone, which would otherwise get thread demand 8
by default.

### Memory

| Component | Cost |
|---|---|
| KV cache, `n_ctx = 512` | 56.0 MiB per context |
| Compute buffer, `n_ubatch = 256` | 283.87 MiB per context (270.61 Metal + 13.26 CPU) |
| Output buffer, `n_seq_max = 32` | 18.64 MiB per context |
| **Peak RSS, default (1 context)** | **~1.20 GB** |
| Peak RSS, 2 contexts (opt-in) | ~1.68 GB |

**The default is one context, and that is a memory decision rather than a
performance one.** Two contexts is the better engineering answer — it is what the
concurrency problem actually is (one caller indexing, one searching) and it measures
1.93× the throughput — but the second context costs roughly half a gigabyte while it
decodes. A Flutter application that also holds a Tantivy index, a rendered book and
the platform web view cannot be handed a 1.7 GB inference floor it never asked for,
and on a 2–3 GB Android device that is a background kill rather than a slow search.
So concurrency is opt-in via `OTZARIA_LLAMA_CONTEXTS` or `LlamaBackendConfig`, and
the host raises it after measuring on the device it ships to. One context is not a
throughput cliff: callers queue on the pool's condvar rather than failing.

Even 1.20 GB is not obviously affordable on the smallest targets, and it is
dominated by buffers this backend does not use (see below). Bringing it down is
genuinely open work, not a solved problem.

Two findings sit behind those numbers, both of the same shape: llama.cpp reserves buffers
sized for *generation* even when the context is embeddings-only, and this backend never
reads logits.

- `n_seq_max` defaulted to 256, costing **149.11 MiB** per context in output buffer.
  Deriving it from `batch_size` brings it to 18.64 MiB at the default 32.
- `n_ubatch = n_ctx` reserved an `n_ubatch × n_vocab` logits tensor — 512 × 151669 × 4 ≈
  296 MiB. Halving `n_ubatch` to 256 removes 162.4 MiB per context.

`kv_unified = true` is load-bearing rather than cosmetic: without it llama.cpp computes
`n_ctx_seq = n_ctx / n_seq_max`, which at 512/256 would cap **every input at 2 tokens** —
a silent catastrophe that no vector test would have attributed correctly.

**The platform caveat, stated because the arithmetic and the observation disagree.** On
Apple Silicon the compute buffer lands in `MTL0`, and Metal's lazy residency meant measured
*peak* RSS moved only ~30 MiB when the reserve dropped 162.4 MiB per context. The reserve
arithmetic is backend-independent, so a CPU-only Android/iOS build should show the full
saving in resident memory — **but that was not verified on-device, and this document does
not claim it.** Anyone budgeting for mobile should measure rather than trust the
extrapolation.

---

## 5. Build cost

`llama-cpp-2` builds llama.cpp and ggml through cmake, which is the reason the feature is
non-default. A default build stays a pure-Rust crate with five small dependencies
(`thiserror`, `serde`, `serde_json`, `log`, `sha2`) and compiles in seconds; enabling
`llama-backend` adds the C++ toolchain to the critical path of every downstream build.

Measured here: a cold build of the feature completes in **~1m13s** on this machine and
resolves **52 packages**. Feature selection is deliberate — `default-features = false`,
adding back only `android-shared-stdcxx`:

- **`openmp` omitted** — ggml has its own thread pool and takes `n_threads` directly, Apple
  clang ships no `libomp`, and thread count was measured not to affect output.
- **`common` omitted** — it gates grammar samplers, `json_schema_to_grammar`, MTP
  speculative decoding, `fit_params` and `print_memory_breakdown`. An embedding backend
  uses none of them. This is the one omission to re-check if a later stage wants generation.
- **`android-shared-stdcxx` kept** — it sets cmake's `ANDROID_STL=c++_shared` and is inert
  elsewhere. A Flutter Android app ships one `libc++_shared.so`, and a process holding both
  a static and a shared libc++ crashes in ways that are very hard to attribute.
- **Metal cannot be turned off on Apple Silicon.** `llama-cpp-2`'s own manifest attaches
  `features = ["metal"]` to `llama-cpp-sys-2` unconditionally there. It is compiled *and
  used* — the 270.61 MiB compute buffer above is `MTL0`, `graph splits = 2`, and
  `ggml_metal_rsets_init` runs — which is also why the exit teardown in §6 exists.

The version is pinned **exactly** rather than with a caret, for a reason that is about
stored data and not about API stability: `llama-cpp-2` vendors a specific llama.cpp commit,
and the vectors depend on it (§3). The manifest records `embedding_backend` as part of the
index's identity, so the backend id can only honestly claim "these vectors are comparable"
if the llama.cpp underneath cannot move without a deliberate edit. A caret would let
`cargo update` change the numeric output of a shipped index with no signal anywhere.

---

## 6. The one `unsafe`, and the exit teardown

Two things a reviewer should look at directly rather than take on trust.

**`tokenizer::RawVocab`.** `llama-cpp-2` 0.1.153 hardcodes `parse_special = true` at both
tokenizing call sites, and the golden contract requires `false` — measured, a literal
`<|endoftext|>` occurring inside a book changes one record from 162 tokens to 158, and with
`parse_special = true` it would be promoted to a control token. There is no safe route:
no `LlamaVocab` type, no vocab accessor, no `parse_special` on any params type, and
`llama-cpp-2` does not re-export `llama-cpp-sys-2`. The alternative — a second `vocab_only`
model load — costs a duplicate 151k-token vocabulary (measured: 35.0 MiB) on a phone.

So the module derives the model pointer from `LlamaModel`'s `#[repr(transparent)]` layout
and calls `llama_tokenize` directly. **That layout is an undocumented implementation
detail.** Upstream's entire doc comment for the type is "A safe wrapper around
llama_model", the field is `pub(crate)`, and the same crate explicitly disclaims layout
stability for sibling types ("Do not rely on `repr(transparent)` for this type… may change
across minor versions"). The exact version pin is what makes this sound and is doing the
real work; the compile-time layout assertion checks size and alignment only, and the
load-time cross-checks run *after* the derived pointer has already reached C, so they are
detection rather than prevention. Soundness of the call itself was audited: lifetime is
structural, only const reads, the tokenize path is `const` throughout `llama-vocab.cpp`,
and 235,200 concurrent tokenizations across 16 threads produced 0 disagreements with a
serial baseline.

**`release_contexts_at_exit`.** ggml aborts at process exit — `GGML_ASSERT([rsets->data
count] == 0)` from a C++ static destructor — if a model is still alive, *after* successful
work. A crash reporter logs it as a crash. The case that matters is the one P6/P7 will hit:
a host holding the engine in a `static`, which is never dropped, aborts even on a normal
return from `main`. An `atexit` teardown registered from `ContextPool::spawn` — after the
first context exists, so it runs before ggml's own device-registry destructor, since
`atexit` runs in reverse registration order — makes all six holding shapes exit cleanly.
A public `shutdown()` is also available for hosts that want to be explicit.

---

## 7. What P3 and P4 inherit

**The correctness gate now runs in CI, with one documented exception.**

The three golden tests are `#[ignore]`d and need `OTZARIA_TEST_MODEL`, so an ordinary
`cargo test` never executes them. Everything they protect — tokenizer parity, EOS, no
BOS, last-token pooling, truncation convention, batch ordering — used to be guarded
only by whoever remembered to run them locally, which made `token_ids` equality (§2,
the primary gate for this whole PR) the one assertion with no automated enforcement.

The `golden-vectors` job closes that. It downloads the gated 396 MB GGUF with the
`OTZARIA_HF_TOKEN` secret (an HF read token granted access to the manually gated repo),
verifies its SHA-256 against `golden_vectors.json`'s header, and runs the tests with
`--ignored`. It fails loudly rather than skipping when the secret is missing, so a green
tick can never mean "the tokenizer was not checked".

The download is deliberately **not** cached. An Actions cache on `main` is readable by a
workflow run from a fork's pull request, so caching the file would republish a
`gated: manual` model to anyone who opens a PR — a licensing breach that needs no access
to the token at all. Paying ~400 MB per run is the cheaper side of that trade.

The HF file was confirmed byte-identical to the model the goldens were generated from
(`x-linked-etag` == `model_sha256` == `a1a89520…b064`, 396,474,560 bytes), which also
rules out the goldens having come from a different local quantization.

**The exception: pull requests from forks.** GitHub deliberately withholds secrets from
them, so the job is skipped there and needs a maintainer `workflow_dispatch` on the
branch before merge. Everything else — pushes, and pull requests from branches in the
repo — is automatic.

**Open, and belonging to a later stage:**

- The manifest records the *requested* token cap, not the backend's *effective* one. If a
  backend ever clamps 8192 down to its own limit, the manifest keeps 8192.
- The output buffer still reserves logits this backend never reads (18.64 MiB per context).
  llama.cpp offers no way to disable it — `has_logits = true` is unconditional.
- The CLI (`src/main.rs`) has no way to pass a model path, so it cannot exercise real
  inference.
- No non-macOS platform has been measured, for memory, latency, or vector agreement.

**Deliberately deferred:**

- The Candle comparison (§1), with its reopening criteria.
- Dimension selection — 1024 versus 512/256/128 via MRL — is P3, and the roadmap's own
  size arithmetic (§4: ~23.1 GiB at f32/1024 for 6,058,210 lines) is why it matters.
- Model licensing and distribution remain a **product** blocker, not a technical one: both
  HuggingFace repos are manually gated, require contact details, and are non-commercial.
  That is why the goldens necessarily come from the local Q4_K_M rather than from the
  original weights or the F16 GGUF.
