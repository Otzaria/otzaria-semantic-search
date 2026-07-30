# P2 — Golden reference vectors

Roadmap: section 6, row **P2 — Inference אמיתי** ("backend trait; השוואת פלט מול Python
reference; tokenizer, EOS, last-token ו־L2 זהים; batch inference אמיתי") and section 7
item 2.

This document covers stage 2 of PR2: the golden reference data itself. It states how the
data was produced, how to reproduce it, what it does and does not prove, the tolerance
stage 4 should assert, and the exact JSON contract the Rust test consumes.

Artifacts:

| Path | What |
|---|---|
| `tests/data/golden_vectors.json` | The golden data. Committed. 433 KB, 31 records. |
| `tools/golden_corpus.json` | The input texts, as reviewable data. |
| `tools/generate_golden_vectors.py` | Reference A — regenerates the golden file. |
| `tools/crosscheck_torch_reference.py` | Reference B — independent verification. |
| `tools/README.md` | Environment setup and flags. |

No Rust source was written for this stage. Stage 4 writes the consuming test; its
contract is specified in [§6](#6-contract-for-the-stage-4-rust-test).

---

## 1. The model

```
file    Otzaria-Embedding-V1-Flash-0.6B-Q4_K_M.gguf
sha256  a1a89520be990087b0a54cc2635513e6eddbfae598fe979b44c52c6bd224b064
size    396,474,560 bytes
gguf    v3, general.architecture = qwen3, general.file_type = 15 (Q4_K_M)
dims    embedding_length 1024, block_count 28, feed_forward_length 3072
attn    head_count 16, head_count_kv 8, key_length = value_length = 128
rope    freq_base 1e6;  rms_norm_eps 1e-6
vocab   tokenizer.ggml.model = "gpt2", pre = "qwen2", 151669 tokens, 151387 merges
```

The SHA-256 above was computed locally and is recorded in
`golden_vectors.json → header.model_sha256`. The file is **gitignored and must never be
committed**.

Both HuggingFace repos (`EMD123/Otzaria-Embedding-V1-Flash-0.6B-GGUF` and the
safetensors repo) are `gated: manual` and return HTTP 401 without an accepted-terms
token. There is therefore **no F16 or safetensors copy available** — the local Q4_K_M is
the only weights source, and both references below read that same file. That is a real
limitation on how independent Reference B can be; see [§4](#4-what-the-goldens-do-and-do-not-prove).

### Pipeline

Per the model card: last-token pooling, then L2 normalization, and no instruction or
query prefix. Concretely, and as verified in [§3](#3-empirical-verification):

```
text
  -> tokenize(add_special = true, parse_special = false)
     -> [content tokens ..., EOS(151643)]      # EOS appended, no BOS prepended
  -> optional token-level truncation
  -> decode (one sequence per decode call)
  -> pooling = LAST  (the hidden state of the final token, i.e. the EOS token)
  -> L2 normalize (computed in float64, stored as f32)
```

`parse_special = false` matters: special-token markup occurring inside a *document* —
a line that literally contains `<|endoftext|>` — is treated as ordinary characters rather
than promoted to a control token. For arbitrary library text that is the only safe
choice, and the Rust side must match it.

Two corpus records exist solely to enforce that, because without them a Rust
implementation using `parse_special = true` passes every other record in the file:

| `id` | `token_count` | Under the wrong `parse_special = true` |
|---|---|---|
| `special_token_markup_literal` | 52 | 36 tokens; cosine **0.9028**, max component diff 6.02e-2 — caught by the vector gates too |
| `special_token_markup_in_paragraph` | 162 | 158 tokens; cosine **0.99287**, max component diff 1.40e-2 — **passes both vector gates; only `token_ids` catches it** |

The second record is the realistic case: one stray marker inside a production-length
chunk. Length dilutes the vector signal; it does not dilute the integers. Note also that
under `parse_special = true` the id 151643 appears *inside* the content, so the model
attends over a materially different sequence while still ending in EOS.

---

## 2. The two references

### Reference A (authoritative — produced the goldens)

| | |
|---|---|
| Implementation | `llama-cpp-python` |
| Version | **0.3.34** (PyPI, built from source with cmake) |
| Python | CPython 3.12.13 |
| ggml build | `MTL : EMBED_LIBRARY = 1 \| CPU : NEON = 1 \| ARM_FMA = 1 \| FP16_VA = 1 \| MATMUL_INT8 = 1 \| DOTPROD = 1 \| SME = 1 \| ACCELERATE = 1 \| REPACK = 1` |
| Backend used | CPU (`n_gpu_layers = 0`) |
| Pooling | `LLAMA_POOLING_TYPE_LAST` (= 3), asserted at load |
| Geometry | `n_ctx = n_batch = n_ubatch = 2048`, `n_threads = 4`, `seed = 0` |
| Decode | one sequence per `llama_decode` call |
| Machine | Apple M4, macOS (Darwin 25.6.0) |

llama-cpp-python was chosen over shelling out to `llama-embedding` or `llama-server`
because it runs in-process and hands back the raw pooled `float*` before any JSON
round-trip, so no precision is lost in formatting. The fallback paths in the task brief
were not needed.

`llama.cpp` is reached through llama-cpp-python's vendored submodule; the version pin
that matters for reproduction is the `llama-cpp-python==0.3.34` sdist, which fixes that
submodule.

### Reference B (independent cross-check — attempted and working)

| | |
|---|---|
| Implementation | HuggingFace `transformers`, `Qwen3Model` |
| Versions | **transformers 5.14.1, torch 2.13.0, gguf 0.19.0, accelerate 1.14.0, tokenizers 0.22.2** |
| Python | CPython 3.12.13 |
| Path | `AutoModel.from_pretrained(dir, gguf_file=..., dtype=torch.float32)` |
| Precision | Q4_K_M tensors dequantized to fp32; forward pass entirely in fp32 |

`qwen3` is present in `transformers.integrations.ggml.GGUF_CONFIG_MAPPING`, so the GGUF
loads without any file from the gated repo. The config transformers reconstructs from the
GGUF metadata matches the header facts exactly (28 layers, 16 / 8 heads, `rope_theta`
1e6, `rms_norm_eps` 9.999999974752427e-07, `vocab_size` 151669, `tie_word_embeddings`
true, all layers `full_attention`), and the model loads as `Qwen3Model` with
**595,776,512 parameters**.

Two setup notes, both of which cost time and are worth recording:

- `accelerate` is **required**. Without it `from_pretrained(..., gguf_file=...)` raises
  `ValueError: accelerate is required when loading a GGUF file`.
- transformers needs the GGUF inside a directory it can treat as a model repo. The model
  must not be copied into the repository, so `crosscheck_torch_reference.py` **symlinks**
  it into a scratch directory. No 396 MB copy is made.

This is a genuinely separate forward pass: PyTorch's `Qwen3Model` with `sdpa` attention
over fp32 weights shares no code with ggml. It is the strongest evidence available, and
it is what the tolerance in [§5](#5-recommended-tolerance-for-stage-4) is built on.

Its one unavoidable weakness: it reads the **same quantized weights**. A hypothetical
error in the original F16 → Q4_K_M quantization would be invisible to both references.
Only the gated F16 GGUF or the safetensors could rule that out, and neither is
obtainable. This is stated as a known gap rather than papered over.

---

## 3. Empirical verification

Nothing below was assumed from the model card; each was measured, and the generator
re-asserts all of it on every run and refuses to write if any fails.

### No BOS, EOS appended

```
add_bos_token()  = False
add_eos_token()  = True
token_eos()      = 151643
token_bos()      = 151643
pooling_type()   = 3   (LAST)
```

`add_bos_token` is absent from the GGUF, and llama.cpp reports `False`. `add_eos_token`
is 1, and llama.cpp reports `True`. The tokenizer was then checked directly, rather than
trusting the flags — token counts with and without `add_special`, for three shapes of
input:

| text | `add_special = false` | `add_special = true` | `== content + [EOS]` |
|---|---|---|---|
| `תורה` | `[54514, 124427]` | `[54514, 124427, 151643]` | yes |
| `The quick brown fox` | `[785, 3974, 13876, 38835]` | `[785, 3974, 13876, 38835, 151643]` | yes |
| `" "` (one space) | `[220]` | `[220, 151643]` | yes |

So `add_special` **appends exactly one EOS and prepends nothing**. The generator performs
this comparison for every corpus entry and aborts on any deviation, so a future
llama.cpp change to `add_special` semantics cannot silently alter the goldens.

Independent corroboration: Reference B's tokenizer, reconstructed from the same GGUF
metadata by `transformers`, produces `[54514, 124427]` for `תורה` — the identical content
tokens. Note that it does **not** append EOS: `tokenizer("תורה")["input_ids"]` is
`[54514, 124427]`. The EOS therefore comes from llama.cpp honouring
`tokenizer.ggml.add_eos_token = 1`, and any non-llama.cpp implementation must append it
explicitly. This is exactly the kind of mismatch the goldens exist to catch.

### Pooling really is last-token

Loading the same model twice, once with `POOLING_TYPE_LAST` and once with
`POOLING_TYPE_NONE` (which returns one vector per token), over nine corpus records:

```
max |row[last] - golden_raw| = 0.0    (bit-identical, all 9 records)
argmax over all rows         = the last row  (all 9 records)
```

The pooled vector is **bit-identical** to the final token's hidden state and the last row
is always the closest one. Pooling is last-token, and because `add_eos_token = 1` the
token pooled over is the EOS. Both halves of that sentence are load-bearing and both are
measured.

What is *not* true is that the neighbouring rows are uniformly far away. The
second-to-last row's cosine against the golden ranges widely:

| record | second-to-last row |
|---|---|
| `heb_prose_mishneh_torah` | **0.9012** |
| `medium_paragraph` | 0.8831 |
| `ascii_only_pangram` | 0.8733 |
| `nikud_bereshit` | 0.8287 |
| `teamim_shema` | 0.6987 |
| `short_single_word` | 0.6033 |
| `space_only` | 0.1087 |

So an off-by-one pooling index is **not** reliably a large-cosine error — on ordinary
prose it lands at 0.90, comfortably inside any threshold a reviewer would think to write.
An earlier draft of this document quoted 0.61 / 0.42 from a 13-token ad-hoc sentence;
those numbers were real but unrepresentative, and the range above replaces them. Omitting
the EOS entirely is still caught (0 of 16 sampled texts pass — see
[§5](#5-recommended-tolerance-for-stage-4)), but it is caught with far less margin than
that draft implied.

### Normalization magnitude is real

Raw pre-normalization L2 norms across the corpus range **69.06 to 104.88**. They are not
near 1, so a missing normalization step is not a subtle effect — and conversely, a golden
that stored only normalized vectors would hide a scaling bug completely. The golden
therefore carries the raw vector *and* its norm alongside the normalized vector.

### Determinism, and one thing that is not deterministic

Measured over all texts, on Reference A:

| Varied | Result |
|---|---|
| Repeat run, same config | **bit-identical** (max abs component diff `0.0`) |
| `n_threads` ∈ {1, 2, 4, 8} | **bit-identical** |
| `n_ubatch` ∈ {2048, 512, 256} | **bit-identical** |
| Batch vs single sequence | **not equal** — worst cosine `0.9961`–`0.99668`, max abs component diff `1.3e-2` |

The last row is the interesting one. It is **reproducible** — an independent reviewer
rebuilt the reference and measured worst cosine `0.99668` / max component diff `1.29e-2`
on a batch of 8, against `0.9961` / `1.3e-2` here; the spread is just which texts share
the batch.

Everything below was then re-measured **on the Rust backend against the same model file**,
which is what closed the question this section used to leave open. Read the subsections in
order: what batching does *not* do, what it does, what is deterministic, and what that
means for a shipped index.

**Established.** The divergence tracks a sequence's token offset within the flattened
batch. Sweeping the offset for one fixed text:

| offset (tokens) | 0 | 3 | 6 | 9 | 12 | 15 | 60 |
|---|---|---|---|---|---|---|---|
| cosine vs single | 1.0 | 0.99832 | 0.99906 | 0.99828 | 1.0 | 0.99832 | 1.0 |

Offsets that are a multiple of 4 often reproduce the single-sequence result bit-for-bit;
others do not. It is also **not** a `llama-cpp-python` position bug: `Batch.add_sequence`
assigns per-sequence positions starting at 0 and one `seq_id` per token, which is correct.

#### Cross-sequence contamination is RULED OUT

This was an open question in earlier drafts of this document, twice: once claimed and
withdrawn as false, then left open. It is now **closed, by direct measurement on the Rust
backend against the real model.**

The experiment is designed so that floating-point explanations are held constant and only
contamination can move the answer. Token length is kept **identical** between neighbours,
so `[A, B]` and `[A, C]` produce batches of the same shape: same `n_tokens`, same tensor
dimensions, same GEMM tiling, same thread partitioning, and A occupying the same rows of
the same buffers. Every kernel-selection and reduction-order effect is therefore the same
in both, and any difference in A's vector could only come from A's attention having read
B's or C's tokens.

Seven variants, all against `Otzaria-Embedding-V1-Flash-0.6B-Q4_K_M.gguf`:

| Experiment | Result for A |
|---|---|
| A at offset 0, neighbour swapped | **bit-identical**, 0/1024 components differ |
| A at offset 40, neighbour swapped | **bit-identical** |
| Long vs short neighbour (same token count) | **bit-identical** |
| Neighbour that is a *prefix* of A | **bit-identical** |
| Batch of 8, neighbours swapped | **bit-identical** |
| Batch filling `n_ctx` entirely | **bit-identical** |
| Neighbour in a different decode group | **bit-identical** |

**Neighbour content has exactly zero effect on a sequence's vector.**

Source-level confirmation of the same conclusion, in the vendored llama.cpp:
`llama-kv-cache.cpp:1617-1624` masks every `(query, kv-cell)` pair with
`if (!cells.seq_has(j, seq_id)) goto skip;`, writing `-INFINITY`. That check is **not**
behind the `unified` flag — unlike `causal` and `swa`, which are template-specialised and
can be compiled out — so it cannot be configured away. The cacheless embedding path
enforces the same thing again at `llama-graph.cpp:438-445`.

#### What batching *does* change, and it is not contamination

A sequence's vector is a function of **(its own tokens, its row offset in the group, the
group's total token count)** — that is, of which ggml kernels run and in what order the
reductions accumulate. Content-independent, but not shape-independent. Measured:

| Comparison | cosine |
|---|---|
| A alone vs A in `[A, B₄₀]` | 0.997877 |
| A at offset 0 vs A at offset 40 | 0.999015 |
| `[A, A]` slot 0 vs slot 1 | 0.999015 |

The last row is the one that disproved the original bit-identity claim, and it is
reproduced here rather than taken on trust.

**The mod-4 rule is incomplete.** A sequence at offset 0 still diverges from its
single-sequence result once a second sequence shares the batch, and `space_only` produced
identical results at offsets 0 and 2 — neither is consistent with a purely
offset-determined rule.

**The cause is still not diagnosed.** An earlier draft attributed it to ARM NEON column
blocking in ggml's Q4_K GEMM selecting a different kernel under `REPACK = 1`. That is a
plausible *hypothesis* and nothing more: it was never instrumented. Treat it as unverified
speculation, and do not repeat it as a cause — including in code comments, where it had
leaked.

#### Inference is deterministic; three knobs provably do not move a vector

Measured on the real model through the Rust backend, 32 mixed-length texts:

| Varied | Result |
|---|---|
| Identical config, run twice | **32/32 bit-identical** |
| `n_threads` ∈ {1, 2, 8} vs 4 | **32/32 bit-identical** |
| `n_seqs` varied at constant token total | **bit-identical** |
| `n_ubatch` ∈ {2048, 512, 256} | **bit-identical** (also re-measured at `n_ctx = 512`: 65/65 vectors, two independent processes) |

These are what make `n_threads` and `n_ubatch` safe to treat as deployment knobs rather
than index identity. Note the boundary: `n_ubatch` is bit-identical **down to 256 and no
further** — at 128 one vector in 65 moved (cosine 0.997456) and at 64 seven did (0.997257),
by the same shape-dependent mechanism as the table above. That is why
`llama_backend::DEFAULT_N_UBATCH` is 256 and not lower, even though lower would save more
memory.

#### `batch_size` changes stored vectors, and it is not part of index identity

This is the shipping consequence, and it needs to be written down because nothing in the
system will announce it: **re-indexing the same corpus with a different `batch_size`
produces different stored vectors.** Measured, 32 texts, same model, same everything else:

| | worst cosine | worst max component | bit-identical |
|---|---|---|---|
| `batch_size` 1 vs 32 | 0.996721 | 1.7698 | **0 of 32** |

It follows directly from the section above: `batch_size` determines how many sequences
share a decode and therefore each sequence's offset and its group's token total.

**This is a known, bounded, accepted property — not an open question and not a bug.**

- **Bounded.** 0.9967 is far below any useful retrieval threshold, and consistent with the
  0.99479 cross-implementation floor and the 0.9948 cross-build tolerance this document
  already accepts in [§5](#5-recommended-tolerance-for-stage-4).
- **Silent, by deliberate choice.** `batch_size` is excluded from the manifest's index
  identity (as are `n_threads` and the pool size) because it is a deployment fact: a phone
  and a desktop should be able to index into comparable indexes, and putting it in the
  identity would force a full re-index of a library whenever the batch size was tuned. The
  same argument `LlamaCppBackend::ID` makes for excluding the llama.cpp build.
- **What this rules out.** Do not write a test that asserts a corpus re-indexed at a
  different `batch_size` reproduces stored vectors bit-for-bit, and do not treat a vector
  that fails to reproduce exactly as evidence of a bug without first checking the batch
  geometry it was produced under.

#### Consequences for the goldens and for stage 4

1. **The goldens are generated one sequence per decode call.** Otherwise the committed
   numbers would depend on the order texts happen to appear in the corpus.
2. **Stage 4 must not assert bitwise batch-vs-single equality.** llama.cpp does not
   provide it. P2's "batch inference אמיתי" acceptance gate should be a tolerance-based
   equivalence check — see [§5](#5-recommended-tolerance-for-stage-4). For retrieval this
   is harmless (0.996 cosine is far tighter than any useful similarity threshold), but as
   a test assertion it would flake.
3. **That relaxation rests on the reproduced bound, not on a diagnosed cause.** This is
   worth being explicit about, because a P2 acceptance gate is being weakened here. The
   relaxation is justified — llama.cpp genuinely does not offer bitwise batch equality —
   but it is justified by "we measured the divergence twice, independently, and it is
   small", not by "we know why it happens and the reason is benign". If a future change
   makes batching diverge further, nothing in this analysis would predict it. What *is*
   now diagnosed is the one thing that would have been alarming: it is not contamination,
   so a sequence's vector never depends on what it was indexed alongside.

---

## 4. What the goldens do and do not prove

This needs stating plainly, because the honest answer is narrower than "the Rust backend
is correct".

The owner has decided the Rust backend will be **llama.cpp via `llama-cpp-2`**; Candle is
out of scope. Reference A is also llama.cpp. So for the **forward pass**, "the Rust
backend matches the goldens" is close to tautological: both sides run the same ggml
kernels over the same quantized weights. A bug inside ggml, or an error in the original
quantization, is invisible to the comparison.

### What they genuinely do prove

Everything around the forward pass — which is where implementation bugs actually live:

- **Tokenization.** 31 exact token-id sequences over Hebrew, Aramaic, nikud,
  cantillation, Hebrew punctuation, gematria, invisible bidi marks, mixed script, literal
  special-token markup, and ASCII. Integer-valued, so this is an exact assertion with no
  tolerance. **This is the load-bearing one** — see [§5](#5-recommended-tolerance-for-stage-4).
- **`add_eos` is applied, and no BOS is prepended.** A missing EOS changes which token
  last-token pooling reads. This is the single most likely wiring error. Note carefully
  that it is the *token-id* assertion that catches it: a wrongly prepended BOS is
  measured to be **invisible** to every vector tolerance in this document.
- **`parse_special = false`.** Two records contain literal `<|…|>` markup (§1). The
  realistic one is invisible to the vector tolerances and caught only by `token_ids`.
- **Last-token selection**, as opposed to mean or CLS pooling — including for the
  3-token and 2-token inputs where a wrong index still yields a plausible unit vector.
- **L2 normalization, and its magnitude.** The raw vectors and norms (69–105) are stored,
  so a missing or doubled normalization is visible rather than absorbed.
- **Truncation semantics at the 512-token boundary**, including both plausible
  conventions — but **only through `token_ids`**. Every truncation bug tested passes the
  vector tolerances; see [§6](#truncation-at-the-512-token-boundary).
- **Batch-vs-single equivalence**, to the tolerance llama.cpp actually supports.
- **Relative similarity ordering.** `cos(near-identical) = 0.9409` vs
  `cos(unrelated) = 0.1311`, a margin of `0.8098`. This is the assertion that catches a
  broken model which still emits plausible-looking unit vectors — a per-vector cosine
  threshold can be passed by a subtly wrong implementation, an ordering with a 0.81
  margin cannot.

### The gap Reference B closes

Reference B is a different codebase (PyTorch/`Qwen3Model`, fp32) and therefore does test
the forward pass. Its agreement with Reference A — **cosine 0.99479 to 0.99880, mean
0.99669** over all 31 texts — is real evidence that the goldens describe the *model* and
not one library's quirks. That is the claim to make, and it is worth having.

It is also the ceiling on precision. Because that floor (0.99479) sits *below* the score
a wrongly prepended BOS achieves (0.99479**38**), the agreement figure that makes
Reference B valuable is the same figure that makes cosine unusable as a correctness gate.
[§5](#5-recommended-tolerance-for-stage-4) works through the consequence.

### The gap that remains open

Both references read the same Q4_K_M file. **A quantization error in the published GGUF
would be invisible to both.** Closing that requires the F16 GGUF or the safetensors, both
of which are behind manual gating (HTTP 401). No public mirror was used; had one been
found it would be untrustworthy for exactly this purpose, since an unverified third-party
copy cannot establish what the official weights contain.

---

## 5. Recommended tolerance for stage 4

### `token_ids` is the primary gate; the vector tolerances are secondary

This is the most important paragraph in the document, and an earlier draft had it
backwards — it called cosine the "PRIMARY check" and described `token_ids` as merely
"where add_eos / BOS / truncation bugs surface". The correct split, forced by
measurement:

> **Primary correctness gate: `token_ids` exact equality.** Tokenization is
> integer-valued, so a mismatch is unambiguous and has no tolerance. Every wiring bug
> that actually matters — missing or extra EOS, a prepended BOS, the wrong truncation
> convention, an off-by-one truncation, `parse_special = true` — changes the integers.
>
> **Secondary: the cosine / max-component / raw-norm thresholds.** They are *not* a
> correctness gate on tokenization. They catch **gross pooling and forward-pass errors**
> only: mean or CLS pooling instead of last-token, an omitted EOS, a dead or mis-loaded
> model, a missing normalization. They are measured to be blind to a prepended BOS, to
> off-by-one truncation, and to `parse_special = true`.

#### Why: no cosine threshold can separate a BOS bug from a legitimate backend

This is not a judgement call, it is arithmetic on two measurements:

| | cosine |
|---|---|
| **Wrongly prepended BOS**, on `boundary_exactly_512` | **0.9947938** |
| **Legitimate Reference B** (PyTorch fp32), on `punct_rashi_abbrev_dense` | **0.9947909** |

**The bug scores higher than the legitimate reference.** Any threshold low enough to
accept a correct independent implementation also accepts a prepended BOS, and any
threshold high enough to reject the bug also rejects PyTorch. There is no value in
between. Tightening is not an available fix — 0.995 would fail *both* legitimate
triangulations below — and loosening changes nothing. The only fix is to stop asking
cosine to carry the weight.

The blindness is not marginal. A prepended BOS was tested against every non-truncated
record (27 of them at the time of measurement, before the two `parse_special` records were
added). It slips past `cosine ≥ 0.99` on four, and **all four also pass
`max_abs_component_diff ≤ 0.03`**:

| record | cosine | max abs component diff |
|---|---|---|
| `aramaic_talmud_sugya` | 0.99221 | 0.0107–0.0144 (all four) |
| `medium_paragraph` | 0.99207 | ″ |
| `boundary_exactly_512` | 0.99479 | ″ |
| `over_512_full` | 0.99473 | ″ |

Note what those four have in common: they are the **long** ones. The blindness grows with
text length, because one extra token at the front of a 512-token sequence perturbs the
final hidden state proportionally less. That is precisely the length regime production
chunks live in, so the vector gates are weakest exactly where they would matter most.

#### What the vector tolerances *do* catch

They are not useless, and the doc should not overcorrect. Every gross pooling or
forward-pass error tested is caught decisively — **0 of 16** sampled texts pass
`cosine ≥ 0.99` in any of these cases:

| Wrong implementation | cosine range |
|---|---|
| MEAN pooling instead of last-token | 0.28 – 0.79 |
| CLS (first-token) pooling | −0.04 – 0.15 |
| EOS omitted, pooling reads the last *content* token | 0.11 – 0.95 |

A dead, mis-loaded or wrongly quantized model, and a missing L2 normalization (raw norms
are 69–105, nowhere near 1), are all firmly inside their reach. That is their job.
Tokenization correctness is not.

### The measurements behind the thresholds

| Comparison | Worst cosine | Max abs component diff (normalized) | Max relative raw-norm diff |
|---|---|---|---|
| Repeat run, same build & config | 1.0 (bit-identical) | 0.0 | 0.0 |
| `n_threads` / `n_ubatch` varied | 1.0 (bit-identical) | 0.0 | 0.0 |
| Batched vs single sequence, CPU | 0.9961–0.99668 | 1.3e-2 | — |
| **CPU vs Metal, same build** | **0.99491** | 1.96e-2 | 3.30e-2 |
| **llama.cpp CPU vs PyTorch fp32** | **0.99479** | 1.94e-2 | 3.34e-2 |

Over all 31 records Reference B agrees at cosine min **0.99479088** / mean **0.99669** /
max **0.99880045**. The two records added for `parse_special` coverage do not move the
worst case (`special_token_markup_literal` 0.99693, `special_token_markup_in_paragraph`
0.99641).

The two independent triangulations land on almost the same floor — 0.99491 and 0.99479 —
with the same worst-case text (`punct_rashi_abbrev_dense`, the dense-abbreviation line)
and the same worst norm case (`nikud_bereshit`). That agreement is what makes the number
trustworthy rather than an artifact of one comparison.

**That floor is the limit on what any golden vector for this quantized model can prove.**
A threshold tighter than ~0.995 would be asserting a property of one build on one
backend, not a property of the model, and would break the moment `llama-cpp-2` pins a
different llama.cpp SHA or someone enables Metal.

#### The zero-content boundary, and why the empty string is not a record

One measurement is worth recording because it marks the outer edge of the envelope. The
empty string is a legal input to the reference: content tokens are `[]` and `add_special`
yields the 1-token sequence `[151643]`, so pooling reads a bare EOS with no context at
all. It produces a perfectly finite vector (raw L2 norm 110.44, and cosine only 0.1107
against the `space_only` golden, so `""` and `" "` are genuinely different inputs).

It was added to the corpus, measured, and deliberately **removed**, because it is where
the two references diverge furthest — **outside three recommended tolerances at once**:

| | measured | tolerance |
|---|---|---|
| Reference A vs B, cosine | **0.98860** | ≥ 0.99 |
| relative raw-norm diff | **0.110** | ≤ 0.05 |
| max abs component diff | **7.37e-2** | ≤ 0.03 |

With no content tokens the pooled EOS state carries no context, so quantization error is
proportionally far larger than anywhere else in the corpus. Committing the record would
have forced either a special-case exemption in the Stage 4 test or a loosening of
thresholds that two legitimate triangulations already pin at ~0.9948 — and the crate's
own pipeline rejects empty and whitespace-only text upstream of the embedder anyway, so
the record would have asserted behaviour production never reaches. `space_only` covers
the degenerate path that *is* reachable. The reasoning is recorded in
`golden_corpus.json → deliberately_uncovered` so it is not silently re-added.

It also independently supports the split above: the vector tolerances are the weaker
instrument, tight enough to fail a legitimate reference on a degenerate input while
staying blind to a real BOS bug on a realistic one.

### Recommended assertions

Mirrored in `header.recommended_tolerances`, with the evidence in
`header.tolerance_evidence`. **Every threshold value here is unchanged from the original
draft** — what changed is which assertion carries the weight:

| Assertion | Threshold | Role | Rationale |
|---|---|---|---|
| `token_ids` equal golden exactly | **exact** | **PRIMARY** | Integers, no tolerance. The only gate that catches BOS, truncation and `parse_special` bugs. Tokenize `text` yourself — see [§6](#vectors--one-record-per-text). |
| `cos(near_identical) − cos(unrelated)` | **≥ 0.405** | strongest vector check | Half the measured 0.8098 margin. Reference B independently gives 0.8145. Hard to fake. |
| cosine, any llama.cpp build/backend | **≥ 0.99** | secondary | Observed worst 0.99479, ~2x headroom on (1 − cosine). Catches gross pooling/forward-pass errors only. |
| max abs component diff (normalized) | **≤ 0.03** | secondary | Observed worst 1.96e-2. Catches a few-wrong-components bug that a 1024-dim cosine dilutes. |
| relative raw-norm diff | **≤ 0.05** | secondary | Observed worst 3.34e-2. Catches scaling bugs normalization would hide. |
| cosine, same build, CPU, single-sequence | **≥ 0.9999** | optional tight bound | Observed bit-identical. Use only when the build is known to match. |
| batch vs single, cosine | **≥ 0.99** | secondary | Observed worst 0.9961–0.99668. llama.cpp does not offer equality here ([§3](#determinism-and-one-thing-that-is-not-deterministic)). |

Use **0.99** as the default cosine threshold. Tighten to 0.9999 only in a test that
pins the llama.cpp build and forces CPU with one sequence per decode.

Two notes on the secondary checks. Cosine over 1024 dimensions is a weak detector of
localized errors — a handful of badly wrong components barely moves it — so assert *both*
cosine and max-abs-component. And if you want one vector-level assertion that is genuinely
hard for a subtly wrong implementation to satisfy, it is the **relational ordering**, not
any per-vector threshold: preserving a 0.81 similarity margin between two independent
pairs cannot be faked by an implementation that merely emits plausible unit vectors.

---

## 6. Contract for the stage-4 Rust test

Specified so the test can be written without inspecting the generator.

### Model path

Read the model path from the environment variable **`OTZARIA_TEST_MODEL`**. If it is
unset, the test should **skip** (not fail) — the 396 MB model is gitignored and
unavailable in CI. If it is set, the test should verify the file's SHA-256 against
`header.model_sha256` and fail loudly on a mismatch, rather than producing a confusing
vector-comparison failure.

### File

`tests/data/golden_vectors.json`, UTF-8, 433 KB. Top level:

```jsonc
{
  "schema_version": 1,          // integer; bump on any breaking layout change
  "header":    { ... },         // provenance + tolerances, see below
  "relations": { ... },         // relative-similarity assertions
  "vectors":   [ { ... } ]      // 31 records, corpus order (NOT sorted by id)
}
```

### `vectors[]` — one record per text

Key order within a record is **`id` first, then the remaining keys alphabetically**. In
`header` the order is approximately alphabetical but not strictly (`reference` precedes
`recommended_tolerances`; `tokenizer` precedes `tolerance_evidence`). None of this is part
of the contract — JSON objects are unordered and every field is addressed by name. The
order exists only so that a diff of a regenerated file is readable. (An earlier draft
claimed record keys were "sorted alphabetically"; they are not, `id` comes first.)

| Field | Type | Meaning |
|---|---|---|
| `id` | string | Stable unique identifier. Safe to reference by name from Rust. |
| `text` | string | The exact input, verbatim. Feed this to the embedder unchanged — no trimming, no normalization, no case folding. Some entries contain invisible characters (U+200F RLM, U+200E LRM, U+00A0 NBSP, U+200C ZWNJ, U+200D ZWJ, U+00AD SOFT HYPHEN) and embedded `\n` / `\t`. |
| `text_char_count` | int | `text.chars().count()` — Unicode scalar values, not bytes. |
| `text_utf8_sha256` | string | Lowercase hex SHA-256 of the UTF-8 bytes of `text`. Assert this first: it proves the test read the exact bytes, including the invisible characters a reviewer cannot see in a diff. |
| `token_ids` | int[] | The exact sequence the reference fed to the model, including the trailing EOS and after any truncation. **This is the primary assertion — see the rule directly below, which governs *how* to assert it.** |
| `token_count` | int | `token_ids.len()`. |
| `source_token_count` | int | Token count of the untruncated sequence (content + EOS). Equals `token_count` when `truncated` is false. |
| `truncated` | bool | Whether truncation was applied. |
| `eos_token_appended` | bool | Always `true`. `token_ids.last() == 151643` for every record. |
| `embedding_normalized_f32_b64` | string | Base64 (standard alphabet, padded) of **4096 bytes** = 1024 little-endian IEEE-754 binary32. The L2-normalized vector. This is the primary comparison target. |
| `embedding_raw_f32_b64` | string | Same encoding, the pooled vector **before** normalization, exactly as llama.cpp emitted it. |
| `raw_l2_norm` | float | L2 norm of the raw vector, computed in float64, rounded to 10 decimals. Range across the corpus: 69.06–104.88. |
| `normalized_preview` | float[8] | First 8 components of the normalized vector, rounded to 10 decimals, for eyeballing a diff. Derived from the f32 payload so it matches exactly. **Not authoritative** — decode the base64. |
| `tags` | string[] | Coverage labels. Informational. |
| `notes` | string | Why the entry exists. Informational. |

#### How to assert `token_ids` — read this before writing the test

> **The Rust side must tokenize `text` with its own tokenizer, apply its own truncation,
> and compare the resulting integers against `token_ids`.**
>
> **The golden `token_ids` must never be fed to the model as input.**

This is not a stylistic preference; getting it wrong disarms the whole file. If the test
feeds `token_ids` straight into the model:

1. the token-id assertion becomes a **tautology** — you are comparing the golden array to
   itself, and the Rust tokenizer is never exercised at all; and
2. the vector comparison then passes, because you fed the model the *correct* tokens — so
   cosine cannot see the bug either, and in any case cosine
   [provably cannot](#why-no-cosine-threshold-can-separate-a-bos-bug-from-a-legitimate-backend)
   separate a prepended BOS from a legitimate backend.

Both defences fail at once, silently, and the suite goes green while the Rust tokenizer is
completely unverified. The mirrored rule lives in
`header.recommended_tolerances.token_ids_comparison_rule` so it is present in the data as
well as in this document.

Feeding `token_ids` to the model is legitimate in exactly one place: Reference B
(`crosscheck_torch_reference.py`) does it deliberately, to isolate the *forward pass* from
tokenization when comparing two Python implementations. That is a different question from
the one Stage 4 is asking.

Decoding in Rust:

```rust
let bytes = base64::engine::general_purpose::STANDARD.decode(&rec.embedding_normalized_f32_b64)?;
assert_eq!(bytes.len(), 4096);
let v: Vec<f32> = bytes.chunks_exact(4)
    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
    .collect();
```

### `header`

Addressed by name; see the note on key order above. Fields stage 4 will actually use:

| Field | Value here | Use |
|---|---|---|
| `model_sha256` | `a1a89520…b064` | Verify `OTZARIA_TEST_MODEL` is the right file. |
| `model_size_bytes` | `396474560` | Cheap pre-check before hashing 396 MB. |
| `embedding_dim` | `1024` | Assert the backend's dimension. |
| `recommended_tolerances` | object | Thresholds from [§5](#5-recommended-tolerance-for-stage-4). Read them rather than hardcoding, so retuning is a data change. |
| `tolerance_evidence` | object | Human-readable measurements behind each threshold. |
| `tokenizer` | object | `add_bos_token: false`, `add_eos_token: true`, `eos_token_id: 151643`, `parse_special: false`. Assert the backend is configured this way. |
| `pipeline` | string | The canonical pipeline description. |
| `env_var_for_model_path` | `"OTZARIA_TEST_MODEL"` | The contract, recorded in the data. |

Also present, informational: `architecture`, `corpus_file`, `corpus_version`,
`generated_date`, `generator`, `gguf_file_type` (15), `gguf_file_type_name`
(`MOSTLY_Q4_K_M`), `gguf_version` (3), `model_file`, `reference` (implementation,
version, ggml system info, geometry, pooling), `vector_encoding`.

### `relations`

```jsonc
{
  "pairs": [ { "a": id, "b": id, "cosine": float, "kind": string, "note": string } ],
  "ordering_assertion": {
    "assert": "cosine(near_identical) > cosine(unrelated) + margin_min",
    "cosine_near_identical": 0.9408610058,
    "cosine_unrelated":      0.1310551691,
    "margin_measured":       0.8098058367,
    "margin_min":            0.405
  }
}
```

Stage 4 should assert, on **its own** freshly computed vectors:

```
cos(near_identical_a, near_identical_b) - cos(unrelated_a, unrelated_b) >= margin_min
```

This is the assertion worth writing carefully. A per-vector cosine threshold against a
single golden can be satisfied by an implementation that is subtly wrong; preserving a
0.81 similarity margin between two independent pairs is much harder to fake. Reference B
reproduces it independently (0.8145).

`kind` values present: `near_identical`, `unrelated`, `marks_sensitivity` (informational —
nikud-only vs nikud+cantillation over the same words, cosine 0.7559).

### Truncation at the 512-token boundary

`src/semantic/embedding.rs` sets `max_tokens: 512`. It is ambiguous whether that 512
counts the EOS, so the corpus pins **both** conventions against the same 915-token text
and stage 4 can assert whichever matches the implementation, without guessing:

| `id` | `token_count` | Meaning |
|---|---|---|
| `boundary_exactly_512` | 512 | A different text that naturally tokenizes to exactly 512 including EOS. Must **not** be truncated. Exercises the boundary without involving truncation logic. |
| `over_512_full` | 915 | The long text embedded at full length, no truncation. Pure model behaviour on long input. |
| `over_512_trunc_total_512` | 512 | Same text; content cut to 511 tokens, then EOS appended. The convention where `max_tokens` **includes** EOS. |
| `over_512_trunc_content_512` | 513 | Same text; content cut to 512 tokens, then EOS appended. The convention where `max_tokens` counts **content only**. |

The three long records share identical `text` (and therefore identical
`text_utf8_sha256`); they differ only in `token_ids`. Whichever convention the Rust code
implements, EOS must remain the final token — otherwise last-token pooling reads a
content token and the vector is meaningless.

#### These four records prove nothing at the vector level

All of their value is in `token_ids`. Do **not** read a passing cosine on them as evidence
that your truncation is correct. Every truncation bug that was tested passes both vector
gates:

| Wrong implementation | cosine | max abs component diff | outcome |
|---|---|---|---|
| `content[:510]+EOS` — off by one vs `over_512_trunc_total_512` | 0.99838 | 0.0113 | **passes** |
| `boundary_exactly_512` wrongly truncated to 511 (must not be truncated at all) | 0.99847 | — | **passes** |
| Wrong convention — `total` where the code means `content` | 0.99476 | 0.0136 | **passes** |

An off-by-one truncation is a *one-token* change at the end of a 512-token sequence; the
final hidden state barely moves, so the normalized vectors stay within noise of each other
while the token arrays differ obviously. The boundary records exist to pin the integers,
and only the integers.

### Corpus coverage (31 records)

| Risk | ids |
|---|---|
| Literal special-token markup (`parse_special = false`) | `special_token_markup_literal`, `special_token_markup_in_paragraph` |
| Plain Hebrew prose | `heb_prose_mishneh_torah`, `heb_prose_aggada` |
| Aramaic, Talmudic register | `aramaic_talmud_sugya`, `aramaic_talmud_short` |
| Nikud / vowel points | `nikud_bereshit`, `nikud_qamats_qatan` (incl. U+05C7 qamats qatan) |
| Cantillation marks | `teamim_shema` (incl. U+05C0 paseq), `teamim_bereshit` |
| Hebrew punctuation | `punct_hebrew_abbrev` (maqaf U+05BE, geresh U+05F3, gershayim U+05F4, sof pasuq U+05C3), `punct_rashi_abbrev_dense` |
| Mixed Hebrew + English | `mixed_hebrew_english` |
| Single short word | `short_single_word` (3 tokens) |
| ASCII-only common path | `ascii_only_pangram` |
| Digits and Hebrew numerals | `digits_only`, `digits_and_gematria` |
| Invisible / RTL marks | `invisible_bidi_marks` (RLM, LRM, NBSP, ZWNJ, ZWJ, soft hyphen) |
| Whitespace shapes | `whitespace_shapes` (newlines, tab, leading spaces), `space_only` (the shortest input covered; the empty string is deliberately **not** a record — see [§5](#the-zero-content-boundary-and-why-the-empty-string-is-not-a-record)) |
| Hebrew final letters | `final_letters` |
| Otzaria title+reference+line form | `otzaria_line_with_reference` (feeds P3) |
| Mid-length realistic chunk | `medium_paragraph` (156 tokens) |
| 512-token boundary | `boundary_exactly_512`, and the three `over_512_*` records |
| Relative similarity | `near_identical_a`/`_b`, `unrelated_a`/`_b` |

---

## 7. Reproducing

Full environment setup, pinned install commands, and every flag are in
[`tools/README.md`](../tools/README.md). In short:

```sh
# Reference A — regenerates tests/data/golden_vectors.json
"$SCRATCH/venvA/bin/python" tools/generate_golden_vectors.py --date "$(date +%F)"

# verify the committed file reproduces byte-for-byte (no flags needed; exits 0)
"$SCRATCH/venvA/bin/python" tools/generate_golden_vectors.py --check

# Reference B — independent verification, writes nothing
"$SCRATCH/venvB/bin/python" tools/crosscheck_torch_reference.py
```

`--check` takes `header.generated_date` from the existing golden file, so it needs no
`--date` and cannot raise a false alarm over the date alone. (It used to default to
`1970-01-01`, which made the bare command above exit 1 with "CHECK FAILED" on perfectly
good data — the fastest way to teach a maintainer to ignore an interlock.) Pass `--date`
only when writing a golden you intend to commit.

Two runs on the same model produce a **byte-identical** file; this was verified, and
`--check` exists to keep verifying it.

The generator refuses to write, **and refuses to `--check`**, when the model it was given
is not the model the existing goldens came from. A golden file with a *missing*
`header.model_sha256` is treated as a mismatch rather than as permission to proceed, since
an unidentifiable golden is where a silent overwrite does the most damage. `--regenerate`
overrides this when writing; it deliberately does **not** override it under `--check`,
because detecting exactly that situation is what `--check` is for. Without the check-mode
interlock, pointing `--check` at the wrong model reported a wall of per-record vector
differences — which reads as "the goldens drifted" when the real fact is "wrong file".

The generator also re-asserts, on every run, that `add_bos` is false, `add_eos` is true,
pooling is `LAST`, `add_special` appends exactly one EOS for every corpus text, the
pinned token counts still hold, no vector is zero / NaN / Inf, and the near-identical
pair is more similar than the unrelated pair. Any failure aborts before writing. The
goldens cannot drift quietly.

---

## 8. Open items for later stages

- **Quantization is unverified against the original weights.** Both references read the
  same Q4_K_M file; the F16 GGUF and safetensors are gated (HTTP 401). If a token is ever
  obtained, re-run Reference B against the F16 GGUF and record the agreement.
- **Batch inference cannot be asserted as bit-exact** (§3). P2's "batch inference אמיתי"
  gate needs a tolerance-based equivalence check, not equality. The **cause is still
  undiagnosed** — but note what is no longer open: **cross-sequence contamination has been
  ruled out**, by seven shape-controlled experiments and at source level, so a sequence's
  vector never depends on what it was batched alongside (§3). What remains undiagnosed is
  benign by comparison: the vector depends on the batch's *shape* (a sequence's offset and
  the group's token total), which is why `batch_size` is a documented, bounded and accepted
  source of vector drift across re-indexes. If the batch path is ever load-bearing beyond
  retrieval tolerances, instrument it properly rather than trusting the offset heuristic in
  §3.
- **The Rust tokenizer is the thing most worth testing and the thing this data can only
  test indirectly.** The goldens pin the expected integers; they cannot force Stage 4 to
  actually run its own tokenizer. If a reviewer checks one thing in the Stage 4 test, it
  should be that `text` is tokenized by Rust and `token_ids` is the expected value, not the
  input ([§6](#how-to-assert-token_ids--read-this-before-writing-the-test)).
- ~~**The Rust `llama-cpp-2` build will pin a different llama.cpp SHA** than
  llama-cpp-python 0.3.34. Once stage 4 exists, measure the actual agreement and record
  it here.~~ **Measured.** `llama-cpp-2 =0.1.153` against these goldens: worst cosine
  **0.996144358** (`teamim_shema`), worst max component diff **1.2886e-2** (`unrelated_a`),
  worst relative raw-norm diff **3.1492e-2** (`over_512_trunc_total_512`), batch-vs-single
  **0.995430753**, ordering margin **0.800808**, and **30/30 `token_ids` exact**. So the
  cross-build agreement is better than both triangulations in [§5](#the-measurements-behind-the-thresholds)
  (CPU-vs-Metal 0.99491, llama.cpp-vs-PyTorch 0.99479).

  **The primary threshold was nevertheless left at 0.99, deliberately.** Tightening it
  would be the wrong lesson to draw: [§5](#why-no-cosine-threshold-can-separate-a-bos-bug-from-a-legitimate-backend)
  shows a wrongly prepended BOS scoring 0.9947938 — *above* the legitimate reference floor
  — so a tighter cosine still would not catch the bug class that matters, while it would
  break the moment `llama-cpp-2` pins a different llama.cpp SHA or someone enables Metal.
  The gate that caught everything here is `token_ids`.
- **Model licence and distribution remain a product blocker**, not a technical one: the
  repo is manually gated, requires contact details, and is non-commercial. Roadmap §7
  flags this; nothing in this stage resolves it.
