# The ONNX backend — `onnxruntime-sentence-v1`

Real inference for ONNX sentence encoders, through ONNX Runtime and the Hugging Face
tokenizer, behind the non-default `onnx-backend` feature — the crate's only backend. The
code is `src/semantic/onnx_backend.rs`. A model is an ONNX graph, a path ending in `.onnx`
in any ASCII case; any other model path is refused as `InvalidModelFile` before a backend
is asked (`model_package::ensure_onnx_model_path`). GGUF and the llama.cpp backend were
removed after `62f0c44`, the last commit that has them. This document is the decision
record: what the backend promises, why it is built the way it is, and what it measured.

The production model it was written for is "Meivin Round 2"
(`ArieLLL123/judaic-semantic-round2-onnx-zayit` @ `1ec8dc6`; CI reads the private mirror
`otzaria/judaic-semantic-round2-onnx-zayit` @ `99b8a61`, the same bytes), verified on the
downloaded files. Of its two graphs, **the int8 graph is the default** — the one the
application ships and the library's vectors are built with; the fp32 graph is the
reference it was quantized from (§0):

| | |
|---|---|
| graphs | `seforim-embed-round2-fp32.onnx` (168,177,986 B, sha256 `1fc2aa8f…7948`), `…-int8.onnx` (42,489,219 B, `65922686…cbf8`) — IR 8, opset 17 |
| inputs | `input_ids`, `attention_mask`: int64 `[1, sequence_length]` — **the batch is fixed at 1**; no `token_type_ids` |
| output | `embedding`, float32; the runtime infers `[?, 256]` and returns `[1, 256]`, unit norm (0.9999997–1.0000003) |
| graph tail | masked mean → 512→256 projection → LayerNorm → `Div(x, Clip(ReduceL2(x)))`: genuinely in-graph pooling |
| int8 | dynamic quantization: 33 `DynamicQuantizeLinear`, 49 `MatMulInteger` |
| positions | 512: a cap of 512 loads, 513 is refused at load (the `Expand` of the position ids fails) |
| tokenizer | `tokenizer.json` (2,191,362 B, `06642879…c0e9`): **Unigram**, 32,000 pieces, Metaspace pre-tokenizer, `[CLS] $A [SEP]`; normalizer NFKC + regex `Replace` rules (§5) |

---

## 0. Which graph: int8, the default

Decided for users on weak PCs, to whom size and memory matter and the accuracy lost is
negligible. Measured on this backend (§7.2):

| | int8 | fp32 |
|---|---:|---:|
| on disk | 42,489,219 B (42 MB) | 168,177,986 B (168 MB) |
| peak footprint, one session | 169 MiB | 387 MiB |
| first `open` in a process | 108 ms | 386 ms |
| agreement with fp32, 600 library inputs | cosine min 0.999067, median 0.999473 | — |
| 14-token query / 256-token passage, 4 threads, Apple M4 (p50) | 1.66 / 16.62 ms | 2.55 / 16.38 ms |

About the same speed on the M4: int8 is faster on short inputs and no faster on long ones,
because dynamic quantization re-quantizes every activation and this CPU's fp32 matrix
multiply is fast. **Still to measure, on a weak PC:** int8 latency on an old x86 CPU without
VNNI, where the quantized products have no dedicated instructions — the case the decision
is for.

One property int8 does not share with fp32: **its vectors depend on which int8 kernels
run.** On the M4 alone, turning KleidiAI off moved them by up to 1.07e-2 (cosine 0.99896),
and an unfused graph by 1.06e-2 (0.99908), where fp32 moves by a few 1e-7 (§7.2); x86 runs
MLAS's own int8 kernels, which compute this graph's products exactly on a CPU with VNNI and
elsewhere only because the backend asks them to (`session.x64quantprecision`, §0.1). So a
library built on one CPU family and queried on another is compared at about cosine 0.999 —
the order at which int8 and fp32 part anyway, and within the int8 golden gate's
cross-machine bound (§8).

The identities: `config/models/meivin-round2-onnx/` (int8, `model_checksum`
`9e408407…d9d065`) and `config/models/meivin-round2-onnx-fp32/` (fp32, `4a4a2ae8…2ade46`),
which differ in those two fields alone. Each graph has its own golden file.

### 0.1 int8 on x86: exact products, by VNNI or by `session.x64quantprecision`

**What CI found.** The int8 gate's first run on an x86-64 runner (`golden-onnx`,
ubuntu-latest; the CPU went unrecorded — the job prints it now) disagreed with the M4's
goldens on every vector: cosine 0.9809 at worst (`query_single_word`), 0.984–0.997
typically, max |Δ| up to 4.2e-2, none of the 41 bit-identical — where two correct int8
kernels part at about 0.999.

**Why: ONNX Runtime's x86 U8S8 kernels saturate at 16 bits.** A dynamically quantized
matrix product multiplies uint8 activations (`DynamicQuantizeLinear`: 0–255, with a zero
point) by int8 weights. On x86, MLAS's kernel for that pairing (U8S8) multiplies with
`VPMADDUBSW`, which adds each pair of neighbouring byte products into a **saturating** int16
before anything widens it: two products of 255 × 127 sum to 64,770, which is clamped to
32,767. Only VNNI and AMX sum in 32 bits. The kernel is chosen per CPU at load
(`core/mlas/lib/platform.cpp:346–642`; the last that applies of AVX2, AVX-VNNI, AVX-512
core, AVX-512 VNNI and AMX wins — so AVX-512 core, checked after AVX-VNNI, replaces its
kernel):

| CPU | U8S8 kernel | the products |
|---|---|---|
| ARM — the goldens' M4 | dot products into 32 bits; KleidiAI for `DynamicQuantizeMatMul` | exact |
| x86 with AVX-512 VNNI (beside AVX-512 BW, DQ and VL), or AMX | `VPDPBUSDS`, AMX tiles | exact |
| x86 with AVX-VNNI and no AVX-512 BW, DQ and VL | `VPDPBUSD` | exact |
| x86 with AVX2 and no VNNI of either kind, or AVX-512 BW, DQ and VL without AVX-512 VNNI — even with AVX-VNNI | `VPMADDUBSW` (`QgemmU8X8KernelAvx2.S`, `QgemmU8X8KernelAvx512Core.S`) | **saturated** |
| x86 with SSE4.1 and no AVX2, Microsoft's Windows build only | `PMADDUBSW` (`qgemm_kernel_sse41.cpp`) | **saturated** |
| x86 without AVX2, the Linux build | SSE2, widened to 16 bits first | exact |

ONNX Runtime says as much where it defines the remedy: "x64 SSE4.1/AVX2/AVX512(with no
VNNI) has overflow problem with quantizied matrix multiplication with U8S8"
(`include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h:248`).

**Why this graph.** It is `quantize_dynamic(weight_type=QuantType.QInt8)` with the
defaults: 49 `MatMulInteger`, each with int8 weights, one scale and an int8 zero point of
0 — per-tensor, symmetric, no `reduce_range` — and every tensor reaching ±127. Values past
±64, without which no pair could reach 32,767 (255 × 64 × 2 = 32,640), are up to 2.1% of
an encoder matrix and 46.6% of the final 512→256 projection. ONNX Runtime's level-2 fusions
make them 25 `DynamicQuantizeMatMul` and 24 `MatMulIntegerToFloat` — Q, K and V share a
`DynamicQuantizeLinear` in each layer, which keeps those three out of the first fusion — and
every one of them runs U8S8.

**Reproduced on the M4.** Each `MatMulInteger` rewritten inside the graph as exact pair
sums clipped to int16 — what `VPMADDUBSW` computes — and run through ONNX Runtime 1.28.0 on
the M4 gives CI's numbers case by case: `query_single_word` 0.9809471732 on both, 18 of the
41 equal to six digits, every one within 5.3e-4 (r = 0.9992). The same rewrite without the
clip equals the M4's KleidiAI-off vectors (cosine 1.000000), so the rewrite adds nothing of
its own.

**The fix: `session.x64quantprecision = 1`** on every session the backend builds where the
CPU's kernels saturate (`X64_QUANT_PRECISION` in `src/semantic/onnx_backend.rs`; where that
is, below). In ONNX Runtime 1.28.0:

- the key is `kOrtSessionOptionsAvx2PrecisionMode`
  (`onnxruntime_session_options_config_keys.h:252`), set through `ort`'s
  `SessionBuilder::with_config_entry` (`AddSessionConfigEntry`);
- it is read in `GenerateTransformers` (`core/optimizer/graph_transformer_utils.cc:378–383`)
  and counts only where `MlasPlatformU8S8Overflow()` is true — the CPU's U8U8 and U8S8
  dispatches differ (`core/mlas/lib/platform.cpp:950–958`), which is every x86 CPU with
  AVX2, VNNI or not, and an SSE4.1 one in the Windows build; on any other architecture the
  code is compiled out;
- it adds `Avx2WeightS8ToU8Transformer` to level 2 (`graph_transformer_utils.cc:448–452`),
  after the `MatMulIntegerToFloat` and `DynamicQuantizeMatMul` fusions (`:399–400`), so it
  meets the fused nodes;
- the transformer (`core/optimizer/qdq_transformer/avx2_weight_s8_to_u8.cc`) knows
  `MatMulInteger`, `MatMulIntegerToFloat`, `DynamicQuantizeMatMul`, `QAttention`, `QGemm`,
  `QLinearMatMul`, `QLinearConv` and `DynamicQuantizeLSTM`. A constant int8 weight tensor
  with any value outside ±64 becomes uint8 by `w XOR 0x80` — `w + 128` — and its zero point
  likewise (`qdq_transformer/s8_to_u8.h`, `s8_to_u8.cc`): (w + 128) − (z + 128) = w − z,
  the same products exactly. All 49 of this graph's tensors qualify;
- the kernels then take the U8U8 dispatch (`core/mlas/lib/qgemm.h:863–878`), which on AVX2
  and AVX-512 widens both operands to 16 bits and multiplies with `VPMADDWD`
  (`QgemmU8X8KernelAvx2.S`, `QgemmU8X8KernelAvx512Core.S`): 255 × 255 × 2 fits a 32-bit
  lane, and nothing saturates.

**Where the backend sets it.** Not everywhere, because ONNX Runtime's own test cannot tell a
VNNI CPU from an AVX2 one: a VNNI kernel is a different *kernel* under the same AVX2
*dispatch* (`platform.cpp:531–537`, `:586–594`), and AMX replaces the U8S8 dispatch alone
(`:624–630`), so on every CPU with AVX2 the two dispatches differ and the entry makes the
runtime trade kernels that were exact already for the U8U8 ones. It did not always: through
1.24 those three branches also pointed the U8U8 dispatch at the U8S8 one, the test was false
there and the entry did nothing; microsoft/onnxruntime#27671, first in 1.25.0, removed the
three assignments, which had sent U8U8 convolutions down the wrong path. So the backend
decides for itself (`X86Int8`), once per process, from the features MLAS's choice turns on —
as the standard library's `is_x86_feature_detected!` reports them, from the same CPUID bits
and, as MLAS does, only where the operating system saves the registers a feature uses (XCR0):

| the CPU, as detected | MLAS's U8S8 kernel | `session.x64quantprecision` |
|---|---|---|
| not x86 (ARM) | its own, exact | not set; compiled out anyway |
| not all of AVX, FMA and AVX2 | SSE: `PMADDUBSW` on SSE4.1 in the Windows build, widened elsewhere | set; a no-op where nothing saturates |
| AVX2, no AVX-VNNI, no AVX-512 BW, DQ and VL (Haswell to Comet Lake, Zen 1 to 3; CI's EPYC 7763) | `VPMADDUBSW` | set |
| AVX2 and AVX-VNNI, no AVX-512 BW, DQ and VL (Alder Lake and its successors) | `VPDPBUSD` | not set |
| AVX-512 F, BW, DQ and VL without AVX-512 VNNI (Skylake-SP), with AVX-VNNI or not | `VPMADDUBSW` | set |
| AVX-512 F, BW, DQ, VL and VNNI (Cascade Lake, Ice Lake, Zen 4 and later) | `VPDPBUSDS`, or AMX's tiles where present | not set |
| macOS, with AVX2 | unknown | set |
| AVX2 without F16C (only a virtual machine presents it) | unknown | set |

The order is MLAS's, and the same in every release from 1.17.0, the oldest the backend
loads, to 1.28.2. The error is one-sided, and so is the rule: set where it was not needed,
the entry costs speed; left off where it was, it changes vectors. So it is left off only
where the kernel is known to be a VNNI one, and set wherever that cannot be settled. Two
cases cannot. macOS creates threads with AVX-512 masked off in XCR0 and unmasks it for a
thread on its first AVX-512 instruction (xnu's `osfmk/i386/fpu.c`, "On-demand AVX512
support"), so what MLAS read when it chose is unknowable — and since no Intel Mac has
AVX-VNNI, only an AVX-512 VNNI one could have done without the entry. And the standard
library reports AVX-512 only alongside F16C, which MLAS does not ask for, so without F16C an
AVX-512 core MLAS uses could be invisible here. The load log says what was decided, e.g.
`int8 on x86: AVX2 kernels (VPMADDUBSW, saturating), so session.x64quantprecision = 1`.

**Why leaving it off changes no vector.** The VNNI kernels sum four products into each
32-bit lane: `VPDPBUSD` wraps and `VPDPBUSDS` saturates at the int32 limits, which this
graph's sums never approach — its weights lie within ±127 and its widest dot product has
2,048 terms, so no sum of products exceeds 2,048 × 255 × 127 = 66,324,480 in magnitude. The
U8U8 kernels widen both operands to 16 bits first, and the rewrite moves each weight and its
zero point by the same 128. So both paths compute Σ(a − za)(w − zw) exactly, in int32, and
the same operator code converts it to float: a VNNI CPU gives the same vectors with the entry
and without it, and a library embedded on one x86 CPU and queried on another has computed
the same int8 products, whichever path each took. The entry's golden test asserts exactly
that wherever the backend leaves the entry off (§8).

**Where it does nothing, and what it costs.** On ARM nothing — compiled out: the M4's
optimized graph is byte-identical with the entry and without, and both golden gates stay 41
of 41 bit-identical. For the fp32 graph nothing — it has no int8 weights. On an x86 CPU whose
kernels saturate, different and correct vectors, from kernels ONNX Runtime documents as
slower: 14% slower on CI's EPYC 7763 (below). On a VNNI CPU it is no longer set, so nothing
is paid there; how fast that CPU's own kernels run this graph has not been measured.

The backend id stays `onnxruntime-sentence-v1`. Nothing had been built with this backend
when the entry was added, so v1 was defined with exact int8 products on x86 (§1), and
deciding per CPU where the entry is needed changes no vector: where it is now left off, the
VNNI kernels compute the very products the U8U8 kernels did, and everywhere else the sessions
are built as before.

**What remains between x86 and the M4: KleidiAI.** Exact, x86 still does not compute what
the M4 computes. On the M4 the 25 `DynamicQuantizeMatMul` run KleidiAI, which quantizes the
activations itself, each row with its own scale (`kai_lhs_quant_pack_qai8dxp_f32`, 8-bit
asymmetric per-row); x86 quantizes the whole tensor with one scale, as
`DynamicQuantizeLinear` specifies (the two paths of
`contrib_ops/cpu/quantization/dynamic_quantize_matmul.cc`, `:231–262` and `:270–311`). The
M4 with KleidiAI off runs x86's arithmetic, and is the nearest prediction to be had
without an x86 CPU — measured on the M4, not on x86:

| on the M4, the 41 golden cases | worst | median |
|---|---:|---:|
| KleidiAI off — x86's arithmetic, exact — against the int8 goldens | 0.998964 | 0.999425 |
| KleidiAI off against the fp32 goldens | 0.998597 | 0.999100 |
| the int8 goldens (KleidiAI on) against the fp32 goldens | 0.999113 | 0.999438 |
| the U8S8 clamp emulated, against the int8 goldens | 0.980947 | 0.991078 |

**Measured on x86, in CI.** `golden-onnx` prints the runner's CPU, the U8S8 kernel it gets
and whether the backend sets the entry there — derived from the kernel's CPU flags in the
backend's order, and handed to the tests as `OTZARIA_EXPECT_X86_INT8_PRECISION`, so that the
backend's own detection must agree; the gate prints every int8 vector against both golden
files; and `what_the_x86_int8_entry_changes_on_this_cpu_is_reported` runs the same ids with
the entry and without it, timed (best of three passes, four intra-op threads, a debug-built
test binary around a release-built runtime). The run below predates the per-CPU decision,
but on its CPU the decision is to set the entry, as it was then set:

| run | CPU, U8S8 kernel | int8 against the int8 goldens, worst | against the fp32 goldens, worst / median | without the entry: worst; vectors identical | 41 texts, with / without the entry | fp32, 41 texts |
|---|---|---:|---:|---:|---:|---:|
| 712f70c, CI run 36829959414 | AMD EPYC 7763, 4 vCPU: AVX2 without VNNI, saturating | 0.9990642 (`query_starts_with_punctuation`) | 0.9985966 (`passage_literal_bert_specials`) / 0.9990811 | 0.9809472 (`query_single_word`); 0 of 41 | 548.7 / 481.1 ms | 741.2 ms |

So on a saturating CPU the entry is the whole difference: without it, the runner reproduces
the first runs' 0.9809471732 to all ten digits — the same arithmetic, which also says what
CPU they ran on — and with it, int8 lies within 0.9986 of fp32 at worst, as on the M4 with
KleidiAI off (0.998597, the prediction above). It costs 14% (548.7 against 481.1 ms), and
int8 still runs 26% faster than the fp32 graph on the same CPU. The fp32 graph came out at
cosine 1.0000000000 against its goldens, and bit-identical with the entry and without.

The cross-machine bound (§8) stays 0.995: the saturated arithmetic misses it by a wide
margin, and the exact arithmetic clears it by 0.004. 0.998 would hold too, and would fail
every saturated case rather than the worst alone (the best of them reached 0.99787) — worth
adopting once a second x86 CPU type has run the gate.

**Emulated, for the CPUs CI does not have.** The one runner CPU recorded so far takes the
"set" branch. The others run in `golden-onnx-emulated`, under Intel's Software Development
Emulator 10.13.1, which runs the same test binary as another CPU — answering CPUID and XCR0
as that CPU would, and emulating the instructions the host lacks: Ice Lake (`-icl`, AVX-512
with VNNI: not set), Alder Lake (`-adl`, AVX-VNNI without AVX-512: not set) and Skylake-SP
(`-skx`, AVX-512 without VNNI: set). Each leg names the decision it expects, so a wrong
detection fails it; runs the gate, holding the backend's vectors to the cross-machine bound;
and runs the entry's test, which on the first two must find the entry changing no vector.
The emulator's timings measure the emulator, so that test times nothing there
(`OTZARIA_X86_INT8_TIMED_PASSES=0`). The numbers are for its first run to fill:

| emulated CPU | decision | int8 against the int8 goldens, worst | with and without the entry: vectors identical |
|---|---|---:|---:|
| Ice Lake | not set | pending | pending (41 of 41 required) |
| Alder Lake | not set | pending | pending (41 of 41 required) |
| Skylake-SP | set | pending | pending |

---

## 1. What the backend id promises

```text
text                                   # already role-prefixed by the text recipe
  -> tokenizer.json                    # the package's own, never a substitute
       padding off, add_special_tokens = true
  -> truncate to max_tokens            # on the right, content first: the total
                                       # length, and the specials survive
  -> input_ids + attention_mask (ones) # token_type_ids as zeros, only if declared
  -> one text per run, [1, len]        # never padded, never batched
  -> the graph's first output          # [1, dim]: pooled inside the graph
  -> return RAW                        # the runtime normalizes and validates
```

The id names that wiring, plus three settings that change the arithmetic and are therefore
part of it: **graph optimization `All`** (`All` and `Disable`/`Level1` measured maxabs
2.9e-6 apart on one graph; `All` is also the Python reference's default), **one text per
run** (§5), and **exact int8 products on x86** — the CPU's VNNI kernels, or
`session.x64quantprecision` where its kernels saturate (§0.1); which of the two a CPU takes
computes the same products, and is not part of the id. It deliberately does not name the
ONNX Runtime version, the thread count or the session count, because §7 measures them.

`EmbeddingBackend::tokenize` and `embed_batch_raw` go through one function, so the ids the
goldens compare are the ids the graph consumes. Pooling is `in-graph` (`Pooling::InGraph`)
and nothing else: a graph whose first output is rank 3 — token states — is refused at
load, because pooling in Rust is not implemented in v1.

---

## 2. Building: nothing is linked

| crate | requirement | features | why |
|---|---|---|---|
| `ort` | `=2.0.0-rc.13` | `std`, `load-dynamic` | the runtime is loaded at run time, never linked |
| `tokenizers` | `=0.23.2` | `fancy-regex` (no defaults) | no C (`onig`) or C++ (`esaxx`) in the build |
| `libloading` | `0.9` | — | reads the runtime's version, checks it before `ort` does (§3) |

Both pins are exact, with the reasons in `Cargo.toml`: `ort`'s API still moves between
release candidates (rc.10 made `Session::run` take `&mut self`), and both crates decide
what a stored vector is. `cargo tree --features onnx-backend -i cc` finds nothing.

**The regex engine is on the path of every id.** `fancy-regex` was chosen on the premise
that a BERT tokenizer uses no regex; the production tokenizer is not BERT's, and its
normalizer runs five regex `Replace` rules, while the Python reference uses onig. The rules
are plain character classes and both engines read them the same way — measured, not
assumed: 19,116 texts sampled from the library (8,991 with niqqud or cantillation, 3,818
with the controls or marks the rules rewrite), each raw, as a passage and as a query —
**57,348 inputs, ids identical** to the Python `tokenizers` 0.23.2 package, 5,567 of them
truncated at 256. The fixture's production-shaped tokenizer keeps checking this in every
test run (§8); changing the engine means re-running the comparison.

**Why `load-dynamic`.** The default `download-binaries` statically links pyke's builds of
ONNX Runtime, fetched from cdn.pyke.io while cargo runs, and each one disqualifies itself:
the Windows x64 build is compiled with `/arch:AVX2` into the plugin's DLL, where AVX2 code
can run at load and take the whole application down on a pre-Haswell CPU; the Linux builds
need glibc ≥ 2.38; the Apple builds raise the application's floor to macOS 13.4 / iOS 15.1
and link CoreML; and the CDN is unreachable from this build network (content filter 418,
Cloudflare 403 through the tunnel). Loaded at run time, a missing or unusable runtime is
`OnnxRuntimeUnavailable` for semantic search and nothing else, and the plugin's builds need
neither network nor minimum-OS changes.

**Targets.** The crates are declared for desktop targets only — the platforms Microsoft
publishes a runtime for. Everywhere else the feature is on with no backend behind it and
an ONNX model gets `BackendUnavailable` — with a reason that says this target has no ONNX
backend in this version (desktop only), rather than asking for the feature, which the
plugin's `semantic` set enables on phones too. The
condition is spelled in `Cargo.toml` (three declarations), `semantic/mod.rs`, the
constructor pair in `semantic::backend` and `tests/onnx_backend.rs`;
`the_target_condition_is_spelled_identically_everywhere` fails if any of the seven drifts
(verified by mutation).

Every target in the plugin's precompiled-binaries matrix, `cargo check --lib --features
onnx-backend` on this Mac:

| target | ONNX crates | backend | how checked |
|---|---|---|---|
| `aarch64-apple-darwin` | yes | yes | host: clippy, tests |
| `x86_64-apple-darwin` | yes | yes | installed std |
| `aarch64-apple-ios`, `aarch64-apple-ios-sim`, `x86_64-apple-ios` | no | — | `-Zbuild-std` |
| `armv7-linux-androideabi`, `aarch64-linux-android`, `i686-linux-android`, `x86_64-linux-android` | no | — | installed std |
| `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu` | yes | yes | `-Zbuild-std`; the Linux x64 test code and clippy too |
| `x86_64-pc-windows-msvc`, `aarch64-pc-windows-msvc` | yes | yes | `-Zbuild-std`; the Windows x64 test code and clippy too |

`-Zbuild-std` ran under `RUSTC_BOOTSTRAP=1` against the installed `rust-src`, for that
command only, since those targets' standard libraries are not installed here. Checking
compiles; it does not link, and nothing here ran on Linux or Windows.

---

## 3. The runtime library

**Where it comes from**, three places in this order:

1. The path the application passes, `EmbeddingDeployment::onnx_runtime` — carried by
   `SemanticConfig::deployment` and `OfficialIndexConfig::deployment` — for an application
   that ships the runtime itself.
2. `OTZARIA_ONNX_RUNTIME`, the path to the shared library: development, the tests, the
   build machine.
3. The platform's file name in the model package, beside the graph:
   `libonnxruntime.dylib`, `libonnxruntime.so`, `onnxruntime.dll`.

The first place that is *set* decides, and a later one is never tried in its place: a path
passed or a variable set that names nothing, or is empty, is refused rather than skipped,
because falling back would load a runtime nobody chose. With a path passed the variable is
not read at all (a log line says so when it is set). With none of the three,
`OnnxRuntimeUnavailable` — "ONNX Runtime could not be loaded: …" — goes through the three
places in order and says what each held.

Every way the library can fail to load is that error: none found, not a runtime, too old,
refused by `ort`, a different one already running — and each names the library with the
place it came from (`passed by the application`, `named by OTZARIA_ONNX_RUNTIME`, `beside
the model`), so that a support message says which one to fix. It is deliberately not
`BackendUnavailable`, whose message says the *build* has no backend for the model: here
the backend is compiled in, and the fix is a file, a path or a variable, not a rebuild.

Where the runtime lives is a deployment fact, not identity. `EmbeddingDeployment` is held
beside `EmbeddingConfig`, never inside it, and no manifest, artifact identity, chunking
identity or backend id reads it, so moving the runtime invalidates nothing. The stand-in
ignores it: it runs nothing.

**Deployment.** The layout the application uses:

```text
<root>/
├── otzaria/                  the data folder
│   ├── seforim.db
│   └── <model>/              the model package
│       ├── seforim-embed-round2-int8.onnx
│       ├── tokenizer.json
│       └── model.json        the model identity file
├── index/                    the Tantivy index
└── <artifact>/               the vectors artifact, a folder of its own beside index/
```

The runtime either ships with the application — the host then passes its path as
`EmbeddingDeployment::onnx_runtime`; on macOS, signed inside the application's bundle, as
library validation requires (below) — or sits beside the graph in `<model>/`, where the
default lookup finds it; it must then be the build for that machine's OS and architecture.
Neither the identity file nor a runtime in that folder is part of the package checksum,
which covers the graph, the external-data files it names and `tokenizer.json` only.

**macOS: library validation.** An application built with the Hardened Runtime — which
notarization requires — loads only libraries signed by Apple or with its own Team ID,
unless it carries the `com.apple.security.cs.disable-library-validation` entitlement;
the App Sandbox further limits what it may read, and a downloaded file carries a
quarantine attribute. So `dlopen` may refuse Microsoft's `libonnxruntime.dylib` from the
model folder even though the file is intact; the refusal arrives as
`OnnxRuntimeUnavailable`, with the loader's reason (a code-signature or Team ID
mismatch) in the message. What works is shipping the library inside the application
bundle, signed with the application's identity, and passing its path as
`EmbeddingDeployment::onnx_runtime` — or, knowingly, the entitlement above.

The runtime is code, not model data: it is platform-specific and not part of the package
checksum (design D4). The reference is **Microsoft's official ONNX Runtime 1.28.0**
release from GitHub (`onnxruntime-osx-arm64-1.28.0.tgz`, SHA-256
`1268b359718099bde2cedb55787f182a130067bc4f31e8c88478c445b850d3d8`; the Linux x64 tgz is
`a3e1b79d7bb1bf09696ce675f49e4064e6c81f6202b8225624fff0e93f8d6407`). It dispatches CPU
features at run time, so it has no AVX2 requirement. The backend needs ONNX Runtime 1.17
or newer — `ort`'s API level with no `api-*` feature enabled.

**One per process.** `ort` holds one runtime and can neither unload nor replace it. The
same library again is a no-op, from whichever place it comes; a different one is refused,
naming both and where each came from. The load log names the runtime as `ONNX Runtime
1.28.0 (git-branch=HEAD, git-commit-id=da9b5e364c, fp8-kv-cache=1, build type=Release)
from <path> (<place>)`.

**`ort` is only handed a library it will accept.** `ort` 2.0.0-rc.13 cannot survive
refusing one: its library slot is a `OnceLock` whose loader runs under
`Once::call_once_force`, so a load that *fails* still completes the `Once` and leaves the
slot marked initialized, with nothing in it. Every later `init_from` then "succeeds"
without loading, and the first `ort` call after that panics — reproduced with `ort`
alone:

```text
first init_from(not a runtime): Err("failed to load from `…/expected.json`: dlopen failed")
second init_from(real runtime): Ok(())
calling ort::info() ...
panicked at ort-2.0.0-rc.13/src/lib.rs:239:14:
`OrtGetApiBase` must be present in ONNX Runtime dylib:
  DlSym { source: "dlsym(0x0, OrtGetApiBase): invalid handle" }
```

When that first call is `Environment::current()` — as it is for the backend — the panic
happens while `ort`'s environment lock is held, so every later ONNX load in the process
panics too. The backend therefore opens the library itself first, with
`libloading`: it must load, export `OrtGetApiBase`, and report a version `ort`'s own rule
accepts. A refusal at that point touches nothing of `ort`'s and can be corrected and
retried; should `ort` still refuse a library the checks passed, that is remembered and
every later load is refused with the reason and "restart". A test runs the
refused-then-correct sequence in a fresh child process — the only kind of process that can
show it — and fails on exactly that panic when `ort` is handed the bad file (verified by
mutation). This is an upstream bug worth reporting against `ort`.

The first load also commits `ort`'s environment: the runtime's log goes to the `log`
facade (target `onnxruntime`) at warning and above — a library linked into a Flutter
application must not write to the host's stderr — and telemetry is off.

---

## 4. Loading: what is proven before the first search

`OnnxBackend::open` checks everything up front, cheapest first, so that a model that cannot
be served fails at load and never in the middle of an index:

| step | refused as |
|---|---|
| pooling other than `in-graph` | `PoolingMismatch` |
| `intra_threads` or `sessions` of 0 in a literal config | `LoadFailed`, naming the field |
| `max_tokens` above 65,536 — past any encoder's context; the probe below is that long. Refused before this too, by `EmbeddingConfig::validate` and the engine's configuration — before a manifest records it | `LoadFailed` (`Config` in the engine) |
| graph / `tokenizer.json` absent | `ModelNotFound` / `TokenizerNotFound` |
| tokenizer not parseable | `InvalidModelFile` (the graph named, the tokenizer in the reason) |
| `max_tokens` ≤ the special tokens the tokenizer adds | `LoadFailed` — also what keeps the tokenizer's own unchecked subtraction from wrapping |
| no loadable runtime (§3) | `OnnxRuntimeUnavailable` |
| graph the runtime cannot load | `InvalidModelFile` |
| an input other than `input_ids`, `attention_mask`, `token_type_ids` | `InvalidModelFile`, naming the input |
| inputs not int64 `[batch, sequence]`, a batch fixed above 1, a fixed sequence length | `InvalidModelFile` |
| first output not float32 `[batch, dim]`; rank 3 | `InvalidModelFile`; rank 3 says Rust-side pooling is not in v1 |
| the probe | `InvalidModelFile`; sessions disagreeing: `LoadFailed` |

**The probe** is one input of exactly `max_tokens` tokens — Hebrew, Latin and digit words
repeated until truncation cuts them to the cap — run through **every** session. It must
come back finite, non-zero, `dim` long and bit-identical across sessions. A cap the graph
cannot run fails here: the fixture's 48-row position table accepts a cap of 48 and refuses
49 at load. Every session rather than one, because each allocates its activations on its
first run, and on a phone that is where memory runs out.

Then one log line at info names the graph, dim, cap, specials, the batch dimension, the
inputs and output, the session and thread counts, the optimization level, the runtime and
the load time.

---

## 5. Tokenization, and the choice of one text per run

**Padding off, truncation replaced.** Whatever padding or truncation `tokenizer.json`
carries is overridden: padding would feed `[PAD]` ids as content, and a stored truncation
would cap inputs at the file's length instead of the configured one. The fixture tokenizer
deliberately pads to 16 and truncates at 512, and the tests prove neither survives.
Truncation is on the right, at `max_tokens` in total: `[CLS] [PASSAGE] …content… [SEP]`
keeps both specials and the prefix, and the content's tail is what is cut.

**Special tokens inside text are matched — deliberately.** The text recipe spells the role
prefix as text (`"[PASSAGE] "`, `"[QUERY] "`), and it becomes the learned token only
because the tokenizer matches added special tokens in its input. The consequence: a book
containing the literal string `[CLS]`, `[SEP]` or `[QUERY]` gets that token too, on
purpose: the prefix is one. The Python
`tokenizers` package (0.23.2) behaves identically, measured on the fixture's reference
cases, all equal id for id:

| case | ids |
|---|---|
| `[PASSAGE] בראשית ברא אלהים` | `[CLS] [PASSAGE] …` — the prefix is one token |
| `the [CLS] fox and the [SEP] dog` | `[CLS]` and `[SEP]` inside the text become ids 2 and 3 |
| `משה [QUERY] ישראל` | a prefix mid-text is matched too |
| `[query] the fox` | lower case is text: `[`, `query`, `]` — matching is on the raw text, case-sensitive |

On the production tokenizer, every added token is `special: true, normalized: false`:
`[PAD]`=0 `[UNK]`=1 `[CLS]`=2 `[SEP]`=3 `[MASK]`=4 **`[QUERY]`=5 `[PASSAGE]`=6**, and
`[שאילתה]`=7, `[קטע]`=8, which the recipe does not use but which a book's text would
match all the same. The file itself ships with padding `BatchLongest` (pad id 0) and
truncation at 512; both are overridden. `"[QUERY] …"` and `"[QUERY]…"` give the same ids —
the Metaspace pre-tokenizer prepends `▁` either way — but a second space after the prefix,
or one at the end, is a lone `▁` id of its own; so text recipe 2 trims both sides before
the prefix — a query as typed, a passage after the character cap, which can end it on a
space. Its normalizer, in order: NFKC; strip
U+0591–U+05BD and U+05BF–U+05C7 (cantillation and points, not the maqaf); strip the
zero-width and bidi controls U+200B–U+200F, U+202A–U+202E, U+2060–U+2069, U+FEFF; fold
geresh and single curly quotes to `'`, gershayim and double curly quotes to `"`, maqaf and
the dashes to `-`; sof pasuq to `:`.

**The cap is 256, and the author's parity check used 128.** The README says the search
encoder truncates every query and passage to 256 tokens, prefix and specials included,
which is what the identity records (`max_tokens` 256); the author's export-parity script
tokenizes at 128 (256 for one long probe). Both are valid inputs to this graph. Which one
retrieves better on this library is a retrieval-quality measurement, not a backend
question — and a change is an index identity change.

**One text per run, never a padded batch.** A vector must depend on its text alone:

- a dynamically quantized graph (`DynamicQuantizeLinear`) computes its scale over the
  whole input tensor, padding included — the same text alone and in a batch measured
  cosine 0.9938–0.9957;
- even in fp32, large padding moved components by ~8.6e-7;
- the production graph declares a batch of 1 anyway.

A batch dimension is therefore only inspected (a fixed batch above 1 is refused), and
`embed_batch_raw` is a loop. Throughput across callers comes from the session pool.

---

## 6. Concurrency and tuning

**Who embeds what.** The library's vectors are built on the build machine only — the
`build`, `export-plan` and `embed-shard` commands, in CI or on Kaggle. The application never
builds them: it opens a prebuilt artifact read-only through `OfficialSemanticIndex` and
embeds one thing, the query (`docs/PRODUCT_CONTRACT.md` §2, §4). `SemanticEngine`'s own
indexing API (`index_books`) is a prototype scaffold, not an application path. So:

- **the application needs one session**, the default: it embeds one query at a time, and
  every further session would hold another copy of the weights (+74 MiB for int8, +202 MiB
  for fp32, §7.2) for nothing;
- **more sessions are a build-machine knob**, and pay only where several callers embed at
  once. The build commands embed one batch at a time per process, so there throughput comes
  from `OTZARIA_ONNX_THREADS` and from running several `embed-shard` windows side by side,
  each with its own session.

`Session::run` takes `&mut self` (upstream considers concurrent `Run` on one session
unsound), while `EmbeddingBackend` is `&self` and `Sync`. The sessions sit in a bounded
pool **leased one text at a time**, first come first served: where one caller embeds a
batch while another embeds a query — the prototype engine indexing while it searches —
the query waits for one inference, not for the batch; with a plain mutex the caller that
just returned a session is already running and takes it straight back. A lease returns its
session when dropped, during unwinding too, so the pool never shrinks and a caller waits
rather than fails. The tokenizer is `Sync` with a `&self` `encode` and is shared without a
lock.

| variable | meaning | default |
|---|---|---|
| `OTZARIA_ONNX_THREADS` | intra-op threads per session | min(4, cores) |
| `OTZARIA_ONNX_SESSIONS` | sessions in the pool: concurrent inferences. Leave it at 1 in the application; a build-machine knob for concurrent callers | 1 |
| `OTZARIA_ONNX_RUNTIME` | the runtime library, when the application passes none (§3) | beside the graph |

Anything but a positive integer is refused, naming the variable. They are deployment
knobs, not identity — measured not to change a vector (§7). Intra-op spinning is off:
spinning threads buy benchmark throughput with a phone's battery and change no vector.
Asking for more `sessions × threads` than the machine has is logged as a warning, not
clamped. Each session holds its own copy of the weights.

---

## 7. Measurements

Apple M4 (4 performance + 6 efficiency cores), 16 GB, macOS 27.0.1; Microsoft ONNX
Runtime 1.28.0 (osx-arm64); release build, through `OnnxBackend::open` and
`embed_batch_raw`.

### 7.1 On the fixture package

`tests/data/onnx_fixture/dynamic.onnx` — 176-token vocabulary, 8-wide hidden state, 4-wide
output. Too small to say anything about speed; it measures the fixed costs and identity.

| | |
|---|---|
| first `open` in a process (runtime load + environment + session + probe) | 13–15 ms |
| a later `open` (runtime already loaded) | ~1 ms |
| process RSS before the first `open` → after | 2 → 34 MiB — the runtime library itself, ~32 MiB |
| second session | +~1 MiB |

Identity, 58 inputs (31 corpus texts as passages and queries), against 1 thread × 1
session: **bit-identical** for threads 1/2/4/8 × sessions 1/2, batched vs one at a time,
and four concurrent callers vs serial. Against the Python references (`onnxruntime` 1.30.0,
`tokenizers` 0.23.2): all 14 cases equal id for id, and every vector **bit-identical**
(max |Δ| 0).

### 7.2 On the production graph

The machine was shared with other builds (load average ~3.5–5.8), so the timings are an
upper bound; p50 and p95 stayed close throughout. Inputs are real library text, prefixed
as the recipe prefixes it.

**Load** (the first `open` in a process also loads the runtime; peak footprint from
`/usr/bin/time -l`, which counts compressed pages — plain RSS under memory compression
understated a second session to +17 MiB):

| | fp32 | int8 |
|---|---:|---:|
| first `open` (runtime, environment, session, 256-token probe) | 386 ms | 108 ms |
| a later `open` in the same process | 105 ms | 80 ms |
| peak footprint, 1 session | 387 MiB | 169 MiB |
| peak footprint, 2 sessions | 589 MiB | 243 MiB |

Each extra session costs about 200 MiB (fp32) or 75 MiB (int8) — its own copy of the
weights, as the research spike found.

**Single-text latency**, p50 / p95 over 100 runs, one session:

| input | threads | fp32 | int8 |
|---|---:|---:|---:|
| query, 14 tokens | 1 | 2.87 / 2.99 ms | 1.81 / 1.86 ms |
| | 2 | 2.63 / 2.72 ms | 1.68 / 1.78 ms |
| | 4 | 2.55 / 2.83 ms | 1.66 / 1.70 ms |
| passage, 128 tokens | 1 | 13.29 / 14.40 ms | 14.13 / 14.89 ms |
| | 2 | 9.89 / 10.02 ms | 10.14 / 10.22 ms |
| | 4 | 8.19 / 8.36 ms | 8.28 / 8.39 ms |
| full, 256 tokens | 1 | 27.89 / 28.46 ms | 30.53 / 31.65 ms |
| | 2 | 20.13 / 20.69 ms | 21.15 / 21.60 ms |
| | 4 | 16.38 / 16.73 ms | 16.62 / 16.80 ms |

Past four threads nothing is gained on this machine (four performance cores): a 256-token
input takes 16.2–16.6 ms at six, eight and ten — hence the default cap of 4. **int8 is not
faster here** except on short inputs: dynamic quantization re-quantizes every activation,
and this CPU's fp32 matrix multiply is fast. Its case is size, not speed.

**A query while indexing runs** — the prototype engine's shape, not the application's,
which never indexes (§6); kept as the measurement of the lease. A 14-token query issued
every 7 ms while another thread embeds 32 × 256-token passages in a loop, 40 queries:

| sessions × threads | p50 | p95 | max |
|---|---:|---:|---:|
| 1 × 4 | 8.54 ms | 14.08 ms | 14.72 ms |
| 2 × 4 | 5.72 ms | 6.16 ms | 6.25 ms |
| 1 × 2 | 12.21 ms | 13.72 ms | 21.74 ms |
| 2 × 2 | 4.69 ms | 5.01 ms | 6.43 ms |

With one session the query waits for at most one passage, not for the batch of 32 (~525 ms
of inference) — the per-text FIFO lease doing its job. A second session removes most of
the wait, for ~200 MiB — a trade no shipped configuration has to make, since the
application embeds only queries.

**Identity.** 600 inputs (300 library texts as passages and queries), against 1 thread × 1
session, for both graphs: **bit-identical** at threads 1/2/4/8 × sessions 1/2, batched vs
one at a time, and concurrent callers vs serial.

**Against the Python reference.** The same 600 inputs through Python `onnxruntime` with
default session options (`ORT_ENABLE_ALL`), fp32: ids identical, and every vector
**bit-identical** (max |Δ| 0) — with the 1.28.0 wheel, and with 1.30.0 as well.

**The golden gates** (`onnx_backend::golden`, against `tests/data/onnx_golden_vectors.json`
for fp32 and `tests/data/onnx_golden_vectors_int8.json` for int8 — Python `onnxruntime`
1.28.0 and `tokenizers` 0.23.2 on Darwin arm64, `ORT_ENABLE_ALL`, one text per run): 41
cases — 26 passages, 11 queries, 4 raw, including niqqud, cantillation, bidi marks,
literal specials and the 256/257-token boundaries — every input's bytes, every id and the
package checksum equal, and, for **each** graph, **41 of 41 vectors bit-identical**
(cosine 1.0000000000, max |Δ| 0); batched equal to single, and four concurrent callers
over two sessions equal to serial.

**What moves an int8 vector** (the generator's `--diagnostics` on the int8 graph): repeat
runs, fresh sessions and 1/2/4/8 threads, nothing — bit-identical; `ORT_ENABLE_EXTENDED`
against `ORT_ENABLE_ALL`, nothing; `ORT_DISABLE_ALL` or `ORT_ENABLE_BASIC`, up to 1.06e-2
(cosine 0.99908); KleidiAI off, up to 1.07e-2 (0.99896). The same levers move fp32 by
2.1e-7 and 2.7e-7. `session.x64quantprecision` on or off moves nothing here, for either
graph — on ARM it is compiled out, and the backend leaves it unset — and every int8 vector
on an x86 CPU whose kernels saturate (§0.1).

**fp32 vs int8**, same 600 inputs: cosine min 0.999067, median 0.999473, max 0.999742 —
in line with the author's own floor of 0.99863 (int8 against PyTorch, four samples), and a
reason the two graphs are different identities (`model_quantization`).

---

## 8. Tests

| where | what | needs |
|---|---|---|
| `onnx_backend::tests` | refusals before any runtime (pooling, tuning, missing files, caps too small and too large, a non-tokenizer), runtime discovery (the three places in order, a place that is set never skipped, each place named in a refusal), the pool's FIFO order and unwinding, the production-shaped tokenizer against Python id for id, a padded query and a passage capped on a space reaching it exactly as the bare text does, the stand-in's stub tokenizer, the x86 int8 decision for every branch of MLAS's kernel choice, and this machine's (§0.1) | nothing |
| `onnx_backend::tests` | the fixture against the Python references; truncation; order; batch = single; concurrency; threads and sessions change nothing; static batch; `token_type_ids` as zeros; rank-3, extra-input, over-cap and non-ONNX refusals; one runtime per process | `OTZARIA_ONNX_RUNTIME` |
| `onnx_backend::golden` | either production graph against its own golden file, chosen by the graph's SHA-256 (a graph no file describes fails loudly): sha256 of the tokenizer, the D4 package checksum, each input's bytes, ids exactly, cosine per graph (and how many are bit-identical), batch = single, concurrent = serial; where the backend leaves `session.x64quantprecision` off, that the entry changes no vector. Reported, not asserted: for the int8 graph each vector against the fp32 graph's golden too, and what the entry changes on this CPU otherwise, timed (§0.1) | `OTZARIA_TEST_ONNX_MODEL` + runtime; `--ignored` |
| `tests/onnx_backend.rs` | the target condition; `select_backend` serving an ONNX package; env refusals through the table; no fallthrough to the stand-in; the stand-in's stub package refused by the real row; a refused runtime then a correct one in a fresh process; the application's runtime path, with the variable removed, in a fresh process, through `EmbeddingRuntime`, `SemanticEngine` and `OfficialSemanticIndex`, then a different path refused; a passed path that names nothing refused by both public paths although the variable names a runtime; `EmbeddingRuntime::load` end to end, with the D4 checksum recomputed | runtime for most |
| `engine::tests`, `official_index::tests` | a changed `EmbeddingDeployment` leaves an index current and an artifact's identity unchanged, and never reaches the manifest | nothing |

The tests that run a graph skip loudly without `OTZARIA_ONNX_RUNTIME`, as the model-gated
tests do without a model. **CI's `onnx-backend` job sets it to Microsoft's ONNX Runtime
1.28.0 for each runner**, fetched by asset name and a pinned SHA-256, with only the
library extracted: `lib/libonnxruntime.so.1.28.0` from `onnxruntime-linux-x64-1.28.0.tgz`,
`lib/libonnxruntime.1.28.0.dylib` from `onnxruntime-osx-arm64-1.28.0.tgz`, and
`lib/onnxruntime.dll`, with `onnxruntime_providers_shared.dll` beside it, from
`onnxruntime-win-x64-1.28.0.zip`.

The golden tests:

```sh
for graph in int8 fp32; do
  OTZARIA_ONNX_RUNTIME=/path/to/libonnxruntime.dylib \
  OTZARIA_TEST_ONNX_MODEL=/path/to/seforim-embed-round2-$graph.onnx \
    cargo test --lib --features onnx-backend onnx_backend::golden -- --ignored --nocapture
done
```

CI adds `--test-threads=1`, so the output reads in order and the timings the entry's test
prints measure that test alone. Two variables shape these runs.
`OTZARIA_EXPECT_X86_INT8_PRECISION` (`set` or `unset`) fails the gate, the entry's test and
`this_machine_gets_the_x86_int8_decision_its_cpu_implies` unless the backend decides that on
this CPU — CI derives it from the runner's flags, and its emulated legs from the CPU they
emulate (§0.1). `OTZARIA_X86_INT8_TIMED_PASSES` sets how many timed passes the entry's test
runs after its warm-up: three by default, none under an emulator.

The cosine bound is per graph. For fp32 it is 0.99999 everywhere, not 1.0, because the
Python reference and this backend need not share an ONNX Runtime build or an instruction
set — kernel choice alone moved components by 2.9e-6, a cosine deficit near 1e-11 — while
every wiring error the ids cannot see (a mask or type ids fed wrong, the wrong output, a
half-precision provider) lands far below it. For int8 it is 0.99999 on the goldens' own CPU
family, where the same kernels run and the vectors came out bit-identical, and 0.995 on
another, whose int8 kernels differ (§0): two correct int8 approximations, each within
0.99911 of fp32 on these cases, can be up to 0.9964 apart. "Correct" is the operative word:
on x86 the products are exact only through the VNNI kernels or `session.x64quantprecision`,
and with neither the first x86 run failed the bound at 0.9809 (§0.1). The bound is to be
revisited with the x86 numbers CI now prints; it is unchanged until then. A graph standing in
for another cannot pass under the looser bound, because the golden file is chosen by the
graph's hash. Truncation and prefix errors change the ids, which are compared exactly and
first.

**The fixture** is regenerated byte for byte by `tools/make_onnx_fixture.py` (the package
versions are in its docstring; `--check` compares without writing): five graphs of about
8 KB — dynamic batch, batch fixed at 1, a rank-3 output, an extra required input, a
declared `token_type_ids` whose type-0 embedding is zero — a BERT-style tokenizer with the
seven special tokens, a second tokenizer shaped like the production one (Unigram,
Metaspace, the NFKC and regex `Replace` normalizer), and `expected.json` with the Python
references' ids for both and vectors for the first.

---

## 9. Open issues

- **The x86 bound, 0.995 or 0.998** (§0.1): one x86 CPU type has run the gate (AMD EPYC
  7763, AVX2 without VNNI: 0.99906 at worst). 0.998 once a second type has — the emulated
  legs are three, once they have run.
- **int8 on a weak PC** (§0): latency on an old x86 CPU without VNNI, through the U8U8
  kernels the entry selects. CI's EPYC 7763 pays 14% for them and still runs int8 26% faster
  than fp32; a desktop CPU of that class has not been measured. Nor has a VNNI CPU, which
  now keeps its own kernels: the emulated legs' timings are the emulator's.
- **Whether the M4 should compute what x86 does.** Exact, x86 and the M4 still differ by
  KleidiAI's per-row activation quantization (§0.1): `mlas.disable_kleidiai = 1` would give
  the M4 x86's per-tensor arithmetic, at a small cost against fp32 (0.998597 at worst, against
  0.999113) and with new int8 goldens. Not decided: as measured, the two lie within 0.99906
  of each other, which no ranking is expected to notice.
- **The cap, 256 vs 128** (§5), for retrieval-quality measurement to settle.
- **The production graph on Windows.** CI's `onnx-backend` job runs the backend's tests on
  Linux, macOS and Windows, all green, but against the fixture package; the production
  graph runs here on Linux x64 only (`golden-onnx`). The plugin's real-model job
  (`otzaria_search_engine`) is what runs it on all three.
- **Report the `ort` rc.13 `OnceLock` bug upstream** (§3); the pre-check stays needed until
  a fixed release is pinned.
- **Mobile.** No runtime ships for iOS or Android in v1; the gating keeps both building.
  Shipping one is future work (iOS cannot load code the app did not ship signed).
- **Rust-side pooling** for token-level graphs is out of scope for v1 and refused.
