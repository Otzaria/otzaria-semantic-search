# The ONNX backend — `onnxruntime-sentence-v1`

Real inference for ONNX sentence encoders, through ONNX Runtime and the Hugging Face
tokenizer, behind the non-default `onnx-backend` feature. The code is
`src/semantic/onnx_backend.rs`; which models reach it is decided in `semantic::backend`
by the model path's format (`.onnx` in any ASCII case). This document is the decision
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
MLAS's own int8 kernels, which compute this graph's products exactly only because the
backend asks for it (`session.x64quantprecision`, §0.1). So a library built on one CPU
family and queried on another is compared at about cosine 0.999 — the order at which int8
and fp32 part anyway, and within the int8 golden gate's cross-machine bound (§8).

The identities: `config/models/meivin-round2-onnx/` (int8, `model_checksum`
`9e408407…d9d065`) and `config/models/meivin-round2-onnx-fp32/` (fp32, `4a4a2ae8…2ade46`),
which differ in those two fields alone. Each graph has its own golden file.

### 0.1 int8 on x86: exact products, by `session.x64quantprecision`

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
(`core/mlas/lib/platform.cpp`; the first that applies of AMX, AVX-512 VNNI, AVX-512 core,
AVX-VNNI and AVX2 wins):

| CPU | U8S8 kernel | the products |
|---|---|---|
| ARM — the goldens' M4 | dot products into 32 bits; KleidiAI for `DynamicQuantizeMatMul` | exact |
| x86 with AMX, AVX-512 VNNI or AVX-VNNI | `VPDPBUSD(S)`, AMX tiles | exact |
| x86 with AVX2, or AVX-512 without VNNI | `VPMADDUBSW` (`QgemmU8X8KernelAvx2.S`, `QgemmU8X8KernelAvx512Core.S`) | **saturated** |
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

**The fix: `session.x64quantprecision = 1`** on every session the backend builds
(`X64_QUANT_PRECISION` in `src/semantic/onnx_backend.rs`). In ONNX Runtime 1.28.0:

- the key is `kOrtSessionOptionsAvx2PrecisionMode`
  (`onnxruntime_session_options_config_keys.h:252`), set through `ort`'s
  `SessionBuilder::with_config_entry` (`AddSessionConfigEntry`);
- it is read in `GenerateTransformers` (`core/optimizer/graph_transformer_utils.cc:378–383`)
  and counts only where `MlasPlatformU8S8Overflow()` is true — the CPU's U8U8 and U8S8
  dispatches differ (`core/mlas/lib/platform.cpp:952–958`), which is every x86 CPU with
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

**Where it does nothing, and what it costs.** On ARM nothing — compiled out: the M4's
optimized graph is byte-identical with the entry and without, and both golden gates stay 41
of 41 bit-identical. For the fp32 graph nothing — it has no int8 weights. On an x86 CPU with
VNNI or AMX the same products, which were exact already, from U8U8 kernels instead of the
VNNI ones: the same vectors, more slowly. On one without, different and correct vectors,
from kernels ONNX Runtime documents as slower. How much slower is for CI to measure (below),
not to assume.

It is set unconditionally, in the application and on the build machine alike, because it
decides what an int8 vector is on x86: the library's passages (embedded on the build
machine, in CI or on Kaggle — x86 CPUs) and the user's queries (embedded on their PC) must
have computed the same products. The backend id stays `onnxruntime-sentence-v1` — nothing
had been built with this backend when the entry was added, so v1 is defined with it (§1).

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

**Measured on x86 — to be filled in from CI.** `golden-onnx` now prints the runner's CPU and
the U8S8 kernel it gets; the gate prints every int8 vector against both golden files; and
`what_the_x86_int8_entry_changes_on_this_cpu_is_reported` runs the same ids with the entry
and without it, timed. The cross-machine bound (§8) is to be set from these numbers, not
from the M4's.

| run | CPU, U8S8 kernel | int8 against the int8 goldens, worst | against the fp32 goldens, worst / median | without the entry: worst; vectors identical | 41 texts, with / without the entry | fp32, 41 texts |
|---|---|---:|---:|---:|---:|---:|
| *pending — the first CI run with the entry* | — | — | — | — | — | — |

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
run** (§5), and **exact int8 products on x86** (`session.x64quantprecision`, §0.1). It
deliberately does not name the ONNX Runtime version, the thread count or the session count
— the same choice `llama-cpp-qwen3-last-v1` makes about the llama.cpp build — because §7
measures them.

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
an ONNX model gets `BackendUnavailable`, as a GGUF model does on 32-bit ARM — with a
reason that says this target has no ONNX backend in this version (desktop only), rather
than asking for the feature, which the plugin's `semantic` set enables on phones too. The
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

**Where it comes from**, first hit wins:

1. `OTZARIA_ONNX_RUNTIME`, the path to the shared library. Set but empty, or naming
   nothing, is refused — not skipped in favour of the next option.
2. The platform's file name in the model package, beside the graph:
   `libonnxruntime.dylib`, `libonnxruntime.so`, `onnxruntime.dll`.
3. Otherwise `OnnxRuntimeUnavailable` — "ONNX Runtime could not be loaded: …" — naming both.

Every way the library can fail to load is that error: none found, not a runtime, too old,
refused by `ort`, a different one already running. It is deliberately not
`BackendUnavailable`, whose message says the *build* has no backend for the model: here
the backend is compiled in, and the fix is a file or a variable, not a rebuild.

**macOS: library validation.** An application built with the Hardened Runtime — which
notarization requires — loads only libraries signed by Apple or with its own Team ID,
unless it carries the `com.apple.security.cs.disable-library-validation` entitlement;
the App Sandbox further limits what it may read, and a downloaded file carries a
quarantine attribute. So `dlopen` may refuse Microsoft's `libonnxruntime.dylib` from the
model folder even though the file is intact; the refusal arrives as
`OnnxRuntimeUnavailable`, with the loader's reason (a code-signature or Team ID
mismatch) in the message. What works is shipping the library inside the application
bundle, signed with the application's identity, and pointing `OTZARIA_ONNX_RUNTIME` at it
— or, knowingly, the entitlement above.

The runtime is code, not model data: it is platform-specific and not part of the package
checksum (design D4). The reference is **Microsoft's official ONNX Runtime 1.28.0**
release from GitHub (`onnxruntime-osx-arm64-1.28.0.tgz`, SHA-256
`1268b359718099bde2cedb55787f182a130067bc4f31e8c88478c445b850d3d8`; the Linux x64 tgz is
`a3e1b79d7bb1bf09696ce675f49e4064e6c81f6202b8225624fff0e93f8d6407`). It dispatches CPU
features at run time, so it has no AVX2 requirement. The backend needs ONNX Runtime 1.17
or newer — `ort`'s API level with no `api-*` feature enabled.

**One per process.** `ort` holds one runtime and can neither unload nor replace it. The
same library again is a no-op; a different one is refused, naming both. The load log names
the runtime as `ONNX Runtime 1.28.0 (git-branch=HEAD, git-commit-id=da9b5e364c,
fp8-kv-cache=1, build type=Release) from <path>`.

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
| `max_tokens` above 65,536 — past any encoder's context; the probe below is that long. Refused before this too, by `EmbeddingConfig::validate` and the engine's configuration — before a manifest records it — while a GGUF cap stays llama.cpp's to clamp | `LoadFailed` (`Config` in the engine) |
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
containing the literal string `[CLS]`, `[SEP]` or `[QUERY]` gets that token too. This is
the opposite of the llama backend's `parse_special = false`, for the opposite reason —
there a control token in a book is an accident, here the prefix is one. The Python
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
| `OTZARIA_ONNX_RUNTIME` | the runtime library (§3) | beside the graph |

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
graph — on ARM it is compiled out — and every int8 vector on an x86 CPU without VNNI
(§0.1).

**fp32 vs int8**, same 600 inputs: cosine min 0.999067, median 0.999473, max 0.999742 —
in line with the author's own floor of 0.99863 (int8 against PyTorch, four samples), and a
reason the two graphs are different identities (`model_quantization`).

---

## 8. Tests

| where | what | needs |
|---|---|---|
| `onnx_backend::tests` | refusals before any runtime (pooling, tuning, missing files, caps too small and too large, a non-tokenizer), runtime discovery, the pool's FIFO order and unwinding, the production-shaped tokenizer against Python id for id, a padded query and a passage capped on a space reaching it exactly as the bare text does, the stand-in's stub tokenizer | nothing |
| `onnx_backend::tests` | the fixture against the Python references; truncation; order; batch = single; concurrency; threads and sessions change nothing; static batch; `token_type_ids` as zeros; rank-3, extra-input, over-cap and non-ONNX refusals; one runtime per process | `OTZARIA_ONNX_RUNTIME` |
| `onnx_backend::golden` | either production graph against its own golden file, chosen by the graph's SHA-256 (a graph no file describes fails loudly): sha256 of the tokenizer, the D4 package checksum, each input's bytes, ids exactly, cosine per graph (and how many are bit-identical), batch = single, concurrent = serial. Reported, not asserted: for the int8 graph each vector against the fp32 graph's golden too, and what `session.x64quantprecision` changes on this CPU, timed (§0.1) | `OTZARIA_TEST_ONNX_MODEL` + runtime; `--ignored` |
| `tests/onnx_backend.rs` | the target condition; `select_backend` serving an ONNX package; env refusals through the table; no fallthrough to the stand-in; the stand-in's stub package refused by the real row; a refused runtime then a correct one in a fresh process; `EmbeddingRuntime::load` end to end, with the D4 checksum recomputed | runtime for most |

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
prints measure that test alone.

The cosine bound is per graph. For fp32 it is 0.99999 everywhere, not 1.0, because the
Python reference and this backend need not share an ONNX Runtime build or an instruction
set — kernel choice alone moved components by 2.9e-6, a cosine deficit near 1e-11 — while
every wiring error the ids cannot see (a mask or type ids fed wrong, the wrong output, a
half-precision provider) lands far below it. For int8 it is 0.99999 on the goldens' own CPU
family, where the same kernels run and the vectors came out bit-identical, and 0.995 on
another, whose int8 kernels differ (§0): two correct int8 approximations, each within
0.99911 of fp32 on these cases, can be up to 0.9964 apart. "Correct" is the operative word:
on x86 the products are exact only through `session.x64quantprecision`, and without it the
first x86 run failed the bound at 0.9809 (§0.1). The bound is to be revisited with the x86
numbers CI now prints; it is unchanged until then. A graph standing in for another
cannot pass under the looser bound, because the golden file is chosen by the graph's hash.
Truncation and prefix errors change the ids, which are compared exactly and first.

**The fixture** is regenerated byte for byte by `tools/make_onnx_fixture.py` (the package
versions are in its docstring; `--check` compares without writing): five graphs of about
8 KB — dynamic batch, batch fixed at 1, a rank-3 output, an extra required input, a
declared `token_type_ids` whose type-0 embedding is zero — a BERT-style tokenizer with the
seven special tokens, a second tokenizer shaped like the production one (Unigram,
Metaspace, the NFKC and regex `Replace` normalizer), and `expected.json` with the Python
references' ids for both and vectors for the first.

---

## 9. Open issues

- **int8 on x86, measured** (§0.1): the first CI run with `session.x64quantprecision` —
  the runner's CPU, int8 against both golden files, and the entry's cost — fills in §0.1's
  table, and the cross-machine bound (§8) is set from it.
- **int8 on a weak PC** (§0): latency on an old x86 CPU without VNNI, now through the U8U8
  kernels the entry selects, and how far its vectors lie from the ones the library is built
  with.
- **Whether the M4 should compute what x86 does.** Exact, x86 and the M4 still differ by
  KleidiAI's per-row activation quantization (§0.1): `mlas.disable_kleidiai = 1` would give
  the M4 x86's per-tensor arithmetic, at a small cost against fp32 (0.998597 at worst, against
  0.999113) and with new int8 goldens. Not decided; the x86 numbers come first.
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
