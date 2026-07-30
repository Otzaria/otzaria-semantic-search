#!/usr/bin/env python3
"""Regenerate tests/data/golden_vectors.json from the real GGUF model.

Reference A of roadmap stage P2: a Python reference implementation over the local
Q4_K_M GGUF, used as the golden data that the Rust llama.cpp backend is asserted
against in stage 4.

Read tools/README.md before running. See docs/P2_REFERENCE_VECTORS.md for what the
generated goldens do and do not prove.

Determinism contract
--------------------
Running this twice on the same model file, with the same flags, on the same machine
MUST produce a byte-identical output file. To make that true:

  * every text is embedded in its own decode call (one sequence per batch), so the
    numbers never depend on how texts happen to group into batches;
  * `--threads` is pinned (default 4) and recorded in the header, because ggml's work
    partitioning is a function of the thread count;
  * `--n-ctx` / `--n-batch` / `--n-ubatch` are fixed constants, not derived from the
    corpus, so adding a text cannot perturb the vectors of every other text;
  * vectors are stored as base64 of little-endian f32 bytes -- exact, not decimal;
  * every key order is fixed by an explicit OrderedDict rather than by json.dumps'
    sort_keys, and the only timestamp is header.generated_date, which `--date` pins.
    Within a `vectors[]` record the order is `id` first and then the remaining keys
    alphabetically; `header` is approximately alphabetical. Nothing may depend on key
    order -- JSON objects are unordered and the contract is by key name. The order
    exists only so a diff of a regenerated file is readable.

Cross-machine reproduction is a weaker claim: see the tolerance discussion in the doc.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import struct
import sys
from collections import OrderedDict
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CORPUS = os.path.join(REPO_ROOT, "tools", "golden_corpus.json")
DEFAULT_OUT = os.path.join(REPO_ROOT, "tests", "data", "golden_vectors.json")
DEFAULT_MODEL = os.path.join(
    REPO_ROOT, "Otzaria-Embedding-V1-Flash-0.6B-Q4_K_M.gguf"
)

SCHEMA_VERSION = 1
EXPECTED_DIM = 1024

# Used for header.generated_date when --date is not passed and there is no existing
# golden file to inherit the date from. A fixed value, so an unattended run is still
# byte-stable.
DEFAULT_DATE = "1970-01-01"

# Fixed context geometry. Deliberately constant so that adding a corpus entry cannot
# change the numbers produced for the existing entries.
N_CTX = 2048
N_BATCH = 2048
N_UBATCH = 2048

# ggml file types, from ggml.h / llama.cpp `enum llama_ftype`.
GGUF_FILE_TYPE_NAMES = {
    0: "ALL_F32",
    1: "MOSTLY_F16",
    2: "MOSTLY_Q4_0",
    3: "MOSTLY_Q4_1",
    7: "MOSTLY_Q8_0",
    8: "MOSTLY_Q5_0",
    9: "MOSTLY_Q5_1",
    10: "MOSTLY_Q2_K",
    11: "MOSTLY_Q3_K_S",
    12: "MOSTLY_Q3_K_M",
    13: "MOSTLY_Q3_K_L",
    14: "MOSTLY_Q4_K_S",
    15: "MOSTLY_Q4_K_M",
    16: "MOSTLY_Q5_K_S",
    17: "MOSTLY_Q5_K_M",
    18: "MOSTLY_Q6_K",
}

# Tolerances recommended for the stage-4 Rust assertions. These are static literals, not
# measurements taken at generation time, so the generated file stays byte-stable. The
# measurements that justify them are recorded in tolerance_evidence below and discussed
# at length in docs/P2_REFERENCE_VECTORS.md.
RECOMMENDED_TOLERANCES = OrderedDict(
    [
        ("primary_gate", "token_ids"),
        ("token_ids_must_match_exactly", True),
        (
            "token_ids_comparison_rule",
            "Tokenize the record's `text` with the backend's OWN tokenizer, apply the "
            "backend's OWN truncation, and compare the resulting integers against the "
            "golden token_ids. NEVER feed the golden token_ids into the model as input. "
            "Doing so turns the token-id assertion into a tautology and leaves the "
            "vector tolerances as the only remaining defence -- and those provably "
            "cannot separate a wrongly prepended BOS from a legitimate backend (see "
            "tolerance_evidence.why_token_ids_is_the_primary_gate). Both defences fail "
            "together, silently.",
        ),
        ("cosine_min_any_llama_cpp_build", 0.99),
        ("cosine_min_same_build_cpu_single_sequence", 0.9999),
        ("batch_vs_single_cosine_min", 0.99),
        ("max_abs_component_diff_normalized", 0.03),
        ("raw_l2_norm_relative_diff_max", 0.05),
        (
            "notes",
            "PRIMARY GATE: token_ids exact equality. Tokenization is integer-valued, so "
            "a mismatch is unambiguous, and every wiring bug that matters -- missing or "
            "extra EOS, a prepended BOS, the wrong truncation convention, an off-by-one "
            "truncation, parse_special=true -- changes the integers. "
            "SECONDARY: the cosine / max-component / raw-norm thresholds below. They are "
            "NOT a correctness gate on tokenization. They catch only GROSS pooling and "
            "forward-pass errors (mean or CLS pooling instead of last-token, an omitted "
            "EOS, a dead or mis-loaded model, a missing normalization). They are measured "
            "to be blind to a prepended BOS, to off-by-one truncation and to "
            "parse_special=true, all of which land inside them. "
            "The thresholds are deliberately NOT tightened: 0.995 would fail both "
            "legitimate triangulations (CPU-vs-Metal 0.99491 and llama.cpp-vs-PyTorch "
            "0.99479), and the worst prepended-BOS case measures 0.9947938 -- ABOVE the "
            "legitimate 0.9947909 floor. No cosine threshold separates the two, so "
            "tightening buys nothing and loosening costs nothing; the weight has to sit "
            "on token_ids instead. "
            "cosine_min_same_build_cpu_single_sequence is the tight bound that holds only "
            "for llama.cpp on CPU, one sequence per decode, same build. "
            "batch_vs_single_cosine_min exists because llama.cpp CPU batching is NOT "
            "bitwise equal to single-sequence inference; see tolerance_evidence. "
            "max_abs_component_diff_normalized is a per-component check, because a "
            "handful of wrong components is diluted by a cosine over 1024 dims. "
            "raw_l2_norm_relative_diff_max guards against a scaling bug that "
            "normalization would otherwise hide entirely. "
            "The strongest vector-level assertion is not any of these thresholds but the "
            "relational ordering in relations.ordering_assertion.",
        ),
    ]
)

# Measured on 2026-07-29, Apple M4, macOS, llama-cpp-python 0.3.34, over this corpus.
# Static literals so the golden file stays byte-stable across regenerations.
TOLERANCE_EVIDENCE = OrderedDict(
    [
        (
            "repeat_run_same_config",
            "bit-identical (max |component diff| = 0.0 over all 29 texts)",
        ),
        (
            "n_threads_1_2_4_8",
            "bit-identical -- ggml's work partitioning does not change the result, so "
            "the goldens are not thread-count dependent",
        ),
        (
            "n_ubatch_2048_512_256",
            "bit-identical -- forcing micro-batch splits does not change the result",
        ),
        (
            "batch_vs_single_cpu",
            "NOT bitwise equal, reproduced independently. Worst cosine 0.9961-0.99668 at "
            "batch size 8 (the exact figure depends on which texts share the batch); max "
            "|component diff| 1.3e-2. The deviation tracks a sequence's token offset "
            "within the flattened batch -- offsets that are a multiple of 4 often "
            "reproduce the single-sequence result bit-for-bit, others do not -- but that "
            "rule is INCOMPLETE and the mechanism is NOT diagnosed: a sequence at offset "
            "0 still diverges from its single-sequence result once a second sequence "
            "shares the batch, and one text measured identical results at offsets 0 and "
            "2. CROSS-SEQUENCE CONTAMINATION HAS NOT BEEN RULED OUT. Placing the same "
            "text twice in one batch does not always give two bit-identical vectors "
            "(aramaic_talmud_sugya: seq0 vs seq1 cosine 0.99942, neither matching the "
            "single-sequence result at 0.99936 / 0.99934), and where the two batch "
            "vectors ARE bit-identical to each other they can still both differ from the "
            "single-sequence result (space_only, 0.99839). What is established is the "
            "MAGNITUDE, not the cause: the divergence is bounded at the figures above and "
            "is reproducible. batch_vs_single_cosine_min rests on that reproduced bound "
            "alone. This is also why the goldens are generated one sequence per decode "
            "call -- it makes the committed numbers independent of corpus order.",
        ),
        (
            "cpu_vs_metal_same_build",
            "worst cosine 0.99491 (punct_rashi_abbrev_dense), max |component diff| "
            "1.96e-2, max relative raw-norm diff 3.30e-2",
        ),
        (
            "llama_cpp_cpu_vs_pytorch_fp32",
            "Reference A vs Reference B (transformers 5.14.1 / torch 2.13.0, same GGUF "
            "dequantized to fp32, independent Qwen3 forward pass): cosine min 0.99479088 "
            "(punct_rashi_abbrev_dense) / mean 0.99669 / max 0.99880045 over all 31 "
            "texts; max |component diff| 1.94e-2; max relative raw-norm diff 3.34e-2 "
            "(nikud_bereshit). The relational ordering is preserved (near-identical "
            "0.9423 vs unrelated 0.1279 under Reference B, margin 0.8145). The two "
            "parse_special records added in corpus_version 2 do not move the worst case "
            "(special_token_markup_literal 0.99693, special_token_markup_in_paragraph "
            "0.99641). "
            "BOUNDARY OF THE ENVELOPE: the one input measured OUTSIDE these tolerances is "
            "the empty string, where the two references agree at only cosine 0.98860, "
            "relative raw-norm diff 0.110 and max |component diff| 7.37e-2 -- outside "
            "three recommended tolerances at once. With no content tokens the pooled EOS "
            "state carries no context, so quantization error is proportionally far larger. "
            "It is deliberately NOT a corpus record; see "
            "golden_corpus.json -> deliberately_uncovered.",
        ),
        (
            "why_token_ids_is_the_primary_gate",
            "MEASURED: there is NO cosine threshold that accepts a legitimate backend and "
            "rejects a wrongly prepended BOS. Prepending BOS to boundary_exactly_512 "
            "scores cosine 0.9947938 against its own golden, while the worst LEGITIMATE "
            "agreement -- Reference B, PyTorch fp32, on punct_rashi_abbrev_dense -- is "
            "0.9947909. The bug scores HIGHER than the legitimate reference. Any threshold "
            "that rejects the bug also rejects a correct independent implementation. "
            "Across the 27 non-truncated records a prepended BOS slips past cosine >= 0.99 "
            "on four texts (aramaic_talmud_sugya 0.99221, medium_paragraph 0.99207, "
            "boundary_exactly_512 0.99479, over_512_full 0.99473) and all four ALSO pass "
            "max_abs_component_diff_normalized <= 0.03 (measured 0.0107-0.0144). The "
            "blindness grows with text length, i.e. it is worst in exactly the length band "
            "production chunks occupy. token_ids catches every one of these immediately "
            "and exactly.",
        ),
        (
            "what_the_vector_tolerances_do_catch",
            "The secondary thresholds are not useless -- they catch gross pooling and "
            "forward-pass errors, with 0 of 16 sampled texts passing cosine >= 0.99 in "
            "each case: MEAN pooling instead of last-token, cosine 0.28-0.79; CLS (first "
            "token) pooling, cosine -0.04-0.15; EOS omitted so pooling reads the last "
            "CONTENT token, cosine 0.11-0.95. A dead, mis-loaded or wrongly quantized "
            "model, and a missing L2 normalization, are also firmly inside their reach. "
            "That is their job. Tokenization correctness is not.",
        ),
        (
            "truncation_bugs_invisible_to_vector_tolerances",
            "MEASURED: the four boundary records prove nothing at the vector level; all of "
            "their value is in token_ids. An off-by-one truncation (content[:510]+EOS "
            "compared against over_512_trunc_total_512, which is content[:511]+EOS) scores "
            "cosine 0.99838 with max component diff 0.0113 -- it PASSES both gates. "
            "Truncating boundary_exactly_512, which must NOT be truncated at all, to 511 "
            "tokens scores 0.99847 -- PASSES. Implementing the wrong convention (max_tokens "
            "counting total vs content) scores 0.99476 with max component diff 0.0136 -- "
            "PASSES. A passing cosine on the boundary records is therefore NOT evidence "
            "that truncation is right; only the exact token_ids are.",
        ),
        (
            "parse_special_true_invisible_to_vector_tolerances",
            "MEASURED on the two corpus records added for this: with a production-length "
            "chunk carrying one stray literal <|endoftext|> "
            "(special_token_markup_in_paragraph), the wrong parse_special=true scores "
            "cosine 0.99287 against the correct vector with max component diff 1.40e-2 -- "
            "it PASSES both vector gates, and only token_ids (162 tokens vs 158) catches "
            "it. On the short markup-dense line (special_token_markup_literal) the same "
            "bug scores 0.9028 / 6.02e-2, which the gates do catch; the difference is "
            "purely how large a fraction of the text the markup occupies. Length dilutes "
            "the vector signal; it does not dilute the integers.",
        ),
        (
            "last_token_pooling_confirmed",
            "With POOLING_TYPE_NONE (one vector per token) over nine corpus records, "
            "max |row[last] - golden_raw| = 0.0 (bit-identical) and the argmax over all "
            "rows is the last row, in every case. Pooling is last-token, and because "
            "add_eos_token=1 the pooled token is the EOS. The second-to-last row is not "
            "uniformly far away, though: across those records its cosine against the "
            "golden ranges 0.11 (space_only) to 0.90 (heb_prose_mishneh_torah), so an "
            "off-by-one pooling index is NOT reliably a large-cosine error -- another "
            "reason the vector thresholds are secondary.",
        ),
        (
            "interpretation",
            "Two fully independent triangulations (CPU-vs-Metal and llama.cpp-vs-PyTorch) "
            "both bottom out at ~0.9948 cosine. That is the floor on what ANY golden "
            "vector for this quantized model can prove; a threshold tighter than that "
            "would be asserting a property of one build rather than of the model, and "
            "0.995 would fail both triangulations outright. 0.99 was chosen to leave "
            "roughly 2x headroom on (1 - cosine). Because a prepended BOS measures "
            "0.9947938 -- above that legitimate floor -- the cosine threshold is a "
            "sanity check on the forward pass and NOT the correctness gate. The gate is "
            "token_ids.",
        ),
    ]
)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_gguf_version(path: str) -> int:
    with open(path, "rb") as fh:
        magic = fh.read(4)
        if magic != b"GGUF":
            raise SystemExit(
                f"{path} does not start with the GGUF magic bytes (got {magic!r}). "
                "This is not a GGUF model file."
            )
        return struct.unpack("<I", fh.read(4))[0]


def f32_b64(values) -> str:
    """base64 of little-endian float32 bytes. Exact and compact."""
    return base64.b64encode(struct.pack("<%df" % len(values), *values)).decode("ascii")


def l2_norm(values) -> float:
    # float64 accumulation: the reference should not inherit f32 summation error.
    return sum(float(v) * float(v) for v in values) ** 0.5


def normalize(values):
    n = l2_norm(values)
    if n == 0.0:
        raise SystemExit(
            "Refusing to normalize a zero vector -- the model returned an all-zero "
            "embedding, which means the forward pass failed rather than produced a "
            "degenerate-but-valid result."
        )
    return [float(v) / n for v in values]


def cosine(a, b) -> float:
    """Cosine of two already-L2-normalized vectors (still divides, to be safe)."""
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    return dot / (l2_norm(a) * l2_norm(b))


def r10(x: float) -> float:
    """Round for storage. Diagnostics only, and keeps the file byte-stable."""
    return round(float(x), 10)


class Reference:
    """Reference A: llama-cpp-python over the local GGUF, pooling type LAST."""

    def __init__(self, model_path: str, n_threads: int, n_gpu_layers: int):
        try:
            import llama_cpp
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                "llama-cpp-python is not installed in this interpreter.\n"
                "See tools/README.md for the pinned install command."
            ) from exc

        self.llama_cpp = llama_cpp
        self.model = Llama(
            model_path=model_path,
            embedding=True,
            pooling_type=llama_cpp.LLAMA_POOLING_TYPE_LAST,
            n_ctx=N_CTX,
            n_batch=N_BATCH,
            n_ubatch=N_UBATCH,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            seed=0,
            verbose=False,
        )
        self.n_embd = self.model.n_embd()
        if self.n_embd != EXPECTED_DIM:
            raise SystemExit(
                f"Model reports embedding dim {self.n_embd}, expected {EXPECTED_DIM}. "
                "Wrong model file?"
            )

        self.eos = self.model.token_eos()
        self.add_bos = bool(self.model._model.add_bos_token())
        self.add_eos = bool(self.model._model.add_eos_token())

        pooling = llama_cpp.llama_pooling_type(self.model._ctx.ctx)
        if pooling != llama_cpp.LLAMA_POOLING_TYPE_LAST:
            raise SystemExit(
                f"Context reports pooling type {pooling}, expected "
                f"{llama_cpp.LLAMA_POOLING_TYPE_LAST} (LAST). Refusing to continue: "
                "the whole point of the goldens is that pooling is last-token."
            )

        # The model card specifies last-token pooling, which is only meaningful if the
        # tokenizer actually terminates the sequence with EOS. Verify, do not assume.
        if self.add_bos:
            raise SystemExit(
                "Model reports add_bos_token=true. The Otzaria model has no "
                "add_bos_token key and must not prepend BOS; refusing to generate "
                "goldens that would bake in the wrong prompt shape."
            )
        if not self.add_eos:
            raise SystemExit(
                "Model reports add_eos_token=false, but the Otzaria GGUF sets "
                "tokenizer.ggml.add_eos_token=1. Refusing to continue."
            )

    def tokenize(self, text: str) -> tuple[list[int], list[int]]:
        """Return (content_tokens, content_tokens + [EOS]).

        add_bos=True is llama.cpp's `add_special`: it appends EOS because the GGUF sets
        add_eos_token=1, and prepends nothing because add_bos_token is absent/false.
        special=False means special-token markup inside the *text* is not parsed, so a
        document containing "<|endoftext|>" is treated as literal characters.
        """
        raw = text.encode("utf-8")
        content = self.model.tokenize(raw, add_bos=False, special=False)
        with_special = self.model.tokenize(raw, add_bos=True, special=False)
        if with_special != content + [self.eos]:
            raise SystemExit(
                "Tokenizer did not behave as expected: add_special did not simply "
                f"append EOS ({self.eos}).\n"
                f"  text        = {text!r}\n"
                f"  content     = {content}\n"
                f"  with_special= {with_special}\n"
                "Refusing to generate goldens against an unverified tokenization."
            )
        return content, with_special

    def embed_tokens(self, tokens: list[int]) -> list[float]:
        """Embed one explicit token sequence. Raw, un-normalized, pooled vector.

        One sequence per decode call, so the result cannot depend on batch packing.
        """
        if len(tokens) > N_UBATCH:
            raise SystemExit(
                f"Sequence of {len(tokens)} tokens exceeds n_ubatch={N_UBATCH}. "
                "Raise the N_CTX/N_BATCH/N_UBATCH constants in this script -- but note "
                "that doing so may perturb every vector, so regenerate the whole file "
                "and re-review it."
            )
        ctx = self.model._ctx
        ctx.kv_cache_clear()
        self.model._batch.reset()
        self.model._batch.add_sequence(tokens, 0, False)
        ctx.decode(self.model._batch)
        ptr = self.llama_cpp.llama_get_embeddings_seq(ctx.ctx, 0)
        if not ptr:
            raise SystemExit(
                "llama_get_embeddings_seq returned NULL -- pooled embeddings were not "
                "produced. Check that the context was created with embedding=True."
            )
        vec = [float(x) for x in ptr[: self.n_embd]]
        ctx.kv_cache_clear()
        self.model.reset()
        if not all(v == v and abs(v) != float("inf") for v in vec):
            raise SystemExit("Model produced NaN or Inf components; refusing to write.")
        return vec

    def embed_batch(self, token_lists: list[list[int]]) -> list[list[float]]:
        """Embed several sequences in ONE decode call. Used only for the batch-vs-single
        diagnostic, never for the golden values themselves."""
        ctx = self.model._ctx
        ctx.kv_cache_clear()
        self.model._batch.reset()
        for i, toks in enumerate(token_lists):
            self.model._batch.add_sequence(toks, i, False)
        ctx.decode(self.model._batch)
        out = []
        for i in range(len(token_lists)):
            ptr = self.llama_cpp.llama_get_embeddings_seq(ctx.ctx, i)
            out.append([float(x) for x in ptr[: self.n_embd]])
        ctx.kv_cache_clear()
        self.model.reset()
        return out

    def system_info(self) -> str:
        return self.llama_cpp.llama_print_system_info().decode("utf-8").strip()

    def metadata(self) -> dict:
        return dict(self.model.metadata)


def apply_truncation(content: list[int], eos: int, spec: dict | None):
    """Return (embedded_tokens, truncated_flag)."""
    if spec is None:
        return content + [eos], False
    tokens = int(spec["tokens"])
    mode = spec.get("mode", "total")
    if mode == "total":
        keep = tokens - 1
    elif mode == "content":
        keep = tokens
    else:
        raise SystemExit(f"Unknown truncate mode {mode!r}; expected 'total' or 'content'.")
    if keep < 0:
        raise SystemExit(f"truncate.tokens={tokens} with mode={mode} leaves no room for EOS.")
    truncated = len(content) > keep
    return content[:keep] + [eos], truncated


def build_records(ref: Reference, corpus: dict) -> list[OrderedDict]:
    by_id = {e["id"]: e for e in corpus["texts"]}
    records: list[OrderedDict] = []

    for entry in corpus["texts"]:
        eid = entry["id"]
        if "text" in entry:
            text = entry["text"]
        elif "same_text_as" in entry:
            src = entry["same_text_as"]
            if src not in by_id:
                raise SystemExit(f"{eid}: same_text_as refers to unknown id {src!r}.")
            text = by_id[src]["text"]
        else:
            raise SystemExit(f"{eid}: corpus entry has neither 'text' nor 'same_text_as'.")

        content, full = ref.tokenize(text)
        tokens, truncated = apply_truncation(content, ref.eos, entry.get("truncate"))

        if tokens[-1] != ref.eos:
            raise SystemExit(f"{eid}: embedded sequence does not end with EOS.")

        expected = entry.get("expect_total_tokens")
        if expected is not None and len(tokens) != int(expected):
            raise SystemExit(
                f"{eid}: expect_total_tokens={expected} but the tokenizer produced "
                f"{len(tokens)} tokens.\n"
                "This fixture pins a token-count boundary on purpose. Either the model "
                "file changed or the corpus text was edited. Recalibrate the fixture "
                "text deliberately -- do not just relax the assertion."
            )

        raw = ref.embed_tokens(tokens)
        norm = l2_norm(raw)
        unit = normalize(raw)

        # Round-trip through f32 before taking the preview, so the decimals a reviewer
        # eyeballs are exactly the values the base64 payload decodes to. Normalization
        # itself is done in f64; only storage is f32.
        unit_b64 = f32_b64(unit)
        unit_f32 = struct.unpack("<%df" % EXPECTED_DIM, base64.b64decode(unit_b64))

        records.append(
            OrderedDict(
                [
                    ("id", eid),
                    ("embedding_normalized_f32_b64", unit_b64),
                    ("embedding_raw_f32_b64", f32_b64(raw)),
                    ("eos_token_appended", True),
                    ("normalized_preview", [r10(v) for v in unit_f32[:8]]),
                    ("notes", entry.get("notes", "")),
                    ("raw_l2_norm", r10(norm)),
                    ("source_token_count", len(full)),
                    ("tags", list(entry.get("tags", []))),
                    ("text", text),
                    ("text_char_count", len(text)),
                    ("text_utf8_sha256", sha256_text(text)),
                    ("token_count", len(tokens)),
                    ("token_ids", tokens),
                    ("truncated", truncated),
                ]
            )
        )

    return records


def build_relations(corpus: dict, unit_by_id: dict) -> OrderedDict:
    pairs = []
    measured: dict[str, float] = {}
    for rel in corpus.get("relations", []):
        a, b = rel.get("a"), rel.get("b")
        if a is None or b is None:
            continue
        if a not in unit_by_id or b not in unit_by_id:
            raise SystemExit(f"relation references unknown id: {a!r} / {b!r}")
        c = cosine(unit_by_id[a], unit_by_id[b])
        measured[rel["kind"]] = c
        pairs.append(
            OrderedDict(
                [
                    ("a", a),
                    ("b", b),
                    ("cosine", r10(c)),
                    ("kind", rel["kind"]),
                    ("note", rel.get("note", "")),
                ]
            )
        )

    out = OrderedDict([("pairs", pairs)])
    if "near_identical" in measured and "unrelated" in measured:
        near = measured["near_identical"]
        unrel = measured["unrelated"]
        if near <= unrel:
            raise SystemExit(
                "Sanity check failed: the near-identical pair is NOT more similar than "
                f"the unrelated pair (near={near:.6f}, unrelated={unrel:.6f}). Either "
                "the model is broken or the corpus pairs are mislabelled. Refusing to "
                "write goldens that would encode a nonsensical ordering."
            )
        out["ordering_assertion"] = OrderedDict(
            [
                (
                    "assert",
                    "cosine(near_identical) > cosine(unrelated) + margin_min",
                ),
                ("cosine_near_identical", r10(near)),
                ("cosine_unrelated", r10(unrel)),
                ("margin_measured", r10(near - unrel)),
                ("margin_min", r10(round((near - unrel) * 0.5, 3))),
                (
                    "note",
                    "margin_min is half the measured margin, rounded -- a deliberately "
                    "slack bound so the test fails on a broken forward pass rather "
                    "than on numerical noise. A subtly wrong implementation can emit "
                    "plausible unit vectors that pass a per-vector cosine threshold; "
                    "it cannot easily preserve this ordering.",
                ),
            ]
        )
    return out


def build_header(args, ref: Reference, corpus: dict, model_sha: str) -> OrderedDict:
    import importlib.metadata as md

    meta = ref.metadata()
    ftype = int(meta.get("general.file_type", -1))

    return OrderedDict(
        [
            ("architecture", meta.get("general.architecture", "unknown")),
            ("corpus_file", "tools/golden_corpus.json"),
            ("corpus_version", corpus.get("corpus_version")),
            ("embedding_dim", ref.n_embd),
            ("env_var_for_model_path", "OTZARIA_TEST_MODEL"),
            ("generated_date", args.date),
            ("generator", "tools/generate_golden_vectors.py"),
            ("gguf_file_type", ftype),
            ("gguf_file_type_name", GGUF_FILE_TYPE_NAMES.get(ftype, "unknown")),
            ("gguf_version", read_gguf_version(args.model)),
            ("model_file", os.path.basename(args.model)),
            ("model_sha256", model_sha),
            ("model_size_bytes", os.path.getsize(args.model)),
            (
                "pipeline",
                "tokenize(text, add_special=true, parse_special=false) -> "
                "[content..., EOS]; optional token-level truncation; single-sequence "
                "decode; pooling=LAST (hidden state of the final token, i.e. EOS); "
                "L2 normalization in float64",
            ),
            (
                "reference",
                OrderedDict(
                    [
                        ("backend", "metal" if args.gpu_layers else "cpu"),
                        ("decode_mode", "one sequence per decode call"),
                        ("implementation", "llama-cpp-python"),
                        ("llama_cpp_system_info", ref.system_info()),
                        ("n_batch", N_BATCH),
                        ("n_ctx", N_CTX),
                        ("n_gpu_layers", args.gpu_layers),
                        ("n_threads", args.threads),
                        ("n_ubatch", N_UBATCH),
                        ("pooling_type", "LAST"),
                        ("version", md.version("llama-cpp-python")),
                    ]
                ),
            ),
            ("recommended_tolerances", RECOMMENDED_TOLERANCES),
            ("tolerance_evidence", TOLERANCE_EVIDENCE),
            (
                "tokenizer",
                OrderedDict(
                    [
                        ("add_bos_token", ref.add_bos),
                        ("add_eos_token", ref.add_eos),
                        ("bos_token_id", ref.model.token_bos()),
                        ("eos_token_id", ref.eos),
                        ("ggml_model", meta.get("tokenizer.ggml.model", "unknown")),
                        ("ggml_pre", meta.get("tokenizer.ggml.pre", "unknown")),
                        ("parse_special", False),
                    ]
                ),
            ),
            (
                "vector_encoding",
                OrderedDict(
                    [
                        (
                            "embedding_normalized_f32_b64",
                            "base64(standard, padded) of 1024 little-endian IEEE-754 "
                            "binary32 values = 4096 bytes. The L2-normalized vector; "
                            "this is what production compares.",
                        ),
                        (
                            "embedding_raw_f32_b64",
                            "Same encoding, but the pooled vector BEFORE "
                            "normalization, exactly as llama.cpp emitted it. Present so "
                            "a scaling bug cannot hide behind normalization.",
                        ),
                        (
                            "normalized_preview",
                            "First 8 components of the normalized vector as decimals, "
                            "for eyeballing in a diff. Not authoritative -- decode the "
                            "base64.",
                        ),
                        (
                            "raw_l2_norm",
                            "L2 norm of the raw vector, computed in float64.",
                        ),
                    ]
                ),
            ),
        ]
    )


def run_diagnostics(ref: Reference, records: list[OrderedDict]) -> None:
    """Print the measurements that justify the recommended tolerances."""
    import itertools

    def unit_of(rec):
        raw = struct.unpack("<1024f", base64.b64decode(rec["embedding_raw_f32_b64"]))
        return normalize(raw)

    print("\n=== diagnostics ===", file=sys.stderr)

    # 1. run-to-run determinism, same config
    worst_cos, worst_abs, worst_id = 1.0, 0.0, None
    for rec in records:
        again = ref.embed_tokens(rec["token_ids"])
        u1, u2 = unit_of(rec), normalize(again)
        c = cosine(u1, u2)
        m = max(abs(x - y) for x, y in zip(u1, u2))
        if c < worst_cos:
            worst_cos, worst_id = c, rec["id"]
        worst_abs = max(worst_abs, m)
    print(
        f"repeat-run (same config): min cosine = {worst_cos:.12f} (worst: {worst_id}), "
        f"max |component diff| = {worst_abs:.3e}",
        file=sys.stderr,
    )

    # 2. batch vs single
    short = [r for r in records if r["token_count"] <= 256]
    for bs in (2, 4, 8):
        worst_cos, worst_abs, worst_id = 1.0, 0.0, None
        for chunk in [short[i : i + bs] for i in range(0, len(short), bs)]:
            if len(chunk) < 2:
                continue
            got = ref.embed_batch([r["token_ids"] for r in chunk])
            for rec, raw in zip(chunk, got):
                u1, u2 = unit_of(rec), normalize(raw)
                c = cosine(u1, u2)
                m = max(abs(x - y) for x, y in zip(u1, u2))
                if c < worst_cos:
                    worst_cos, worst_id = c, rec["id"]
                worst_abs = max(worst_abs, m)
        print(
            f"batch size {bs} vs single: min cosine = {worst_cos:.12f} "
            f"(worst: {worst_id}), max |component diff| = {worst_abs:.3e}",
            file=sys.stderr,
        )

    # 3. norm range, so the raw-norm tolerance is grounded
    norms = [r["raw_l2_norm"] for r in records]
    print(
        f"raw L2 norms: min={min(norms):.4f} max={max(norms):.4f}",
        file=sys.stderr,
    )

    # 4. pairwise cosine spread, to show the vectors are not all collinear
    units = [unit_of(r) for r in records]
    cs = [cosine(a, b) for a, b in itertools.combinations(units, 2)]
    print(
        f"pairwise cosine over {len(records)} vectors: min={min(cs):.4f} "
        f"max={max(cs):.4f} mean={sum(cs)/len(cs):.4f}",
        file=sys.stderr,
    )


def load_existing(out_path: str) -> tuple[Any, str | None]:
    """Return (parsed_golden_or_None, parse_error_or_None).

    (None, None) means the file simply does not exist yet.
    """
    if not os.path.exists(out_path):
        return None, None
    try:
        with open(out_path, encoding="utf-8") as fh:
            return json.load(fh), None
    except (json.JSONDecodeError, OSError) as exc:
        return None, str(exc)


def enforce_model_identity(
    out_path: str,
    existing: Any,
    parse_error: str | None,
    model_sha: str,
    regenerate: bool,
    checking: bool,
) -> None:
    """Refuse to touch a golden file that did not come from the model we just hashed.

    This guards BOTH writing and --check. A --check run against the wrong model would
    otherwise print a wall of per-record vector differences, which reads as "the goldens
    drifted" when the actual fact is "you pointed me at a different model file".

    A golden file that does not record header.model_sha256 at all is treated as a
    MISMATCH, not as permission to proceed: an unidentifiable golden is exactly the case
    where silently overwriting it does the most damage.
    """
    if existing is None and parse_error is None:
        return  # nothing on disk to protect

    if parse_error is not None:
        reason = "could not be parsed, so the model it came from cannot be established"
        detail = f"  parse error : {parse_error}\n"
    else:
        old_sha = existing.get("header", {}).get("model_sha256")
        if old_sha is None:
            reason = "does not record header.model_sha256, so the model it came from cannot be established"
            detail = "  its model   : MISSING (no header.model_sha256 key)\n"
        elif old_sha != model_sha:
            reason = "was produced from a DIFFERENT model file"
            detail = f"  its model   : {old_sha}\n"
        else:
            return  # identity confirmed

    header = (
        "*** REFUSING TO --check AGAINST THIS MODEL ***"
        if checking
        else "*** REFUSING TO OVERWRITE THE GOLDEN FILE ***"
    )
    body = (
        "\n"
        f"{header}\n"
        "\n"
        f"  golden file : {out_path}\n"
        f"{detail}"
        f"  your model  : {model_sha}\n"
        "\n"
        f"The existing golden file {reason}.\n"
    )

    if checking:
        raise SystemExit(
            body
            + "\n"
            "Comparing vectors now would report a wall of per-record differences and\n"
            "hide the one fact that matters: the model is not the one the goldens were\n"
            "produced from. Point --model at the right file. --regenerate does NOT\n"
            "silence this check, because --check exists precisely to detect it.\n"
        )

    if not regenerate:
        raise SystemExit(
            body
            + "\n"
            "Regenerating would silently replace the reference data that every stage-4\n"
            "assertion depends on, and any drift would look like a passing test rather\n"
            "than a changed model.\n"
            "\n"
            "If you genuinely intend to move the goldens to a new model file, rerun\n"
            "with --regenerate and say so in the commit message.\n"
        )

    print(
        f"warning: --regenerate given; the existing golden {reason}. Overwriting it "
        f"with goldens from model {model_sha}.",
        file=sys.stderr,
    )


def dump_json(obj: Any, path: str) -> None:
    # ensure_ascii=False keeps Hebrew readable in a diff; per-record
    # text_utf8_sha256 is what actually pins the exact bytes, including the
    # invisible characters that a reviewer cannot see.
    text = json.dumps(obj, ensure_ascii=False, indent=1) + "\n"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Regenerate tests/data/golden_vectors.json from the real GGUF model."
    )
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Path to the GGUF model file.")
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument(
        "--date",
        default=None,
        help="Value for header.generated_date. Under --check it defaults to the date "
        "recorded in the existing golden file, so `--check` needs no flags and cannot "
        "raise a false alarm over the date alone. When writing, it defaults to "
        f"{DEFAULT_DATE}; pass a real date for a golden you intend to commit.",
    )
    ap.add_argument("--threads", type=int, default=4, help="Pinned ggml thread count.")
    ap.add_argument(
        "--gpu-layers",
        type=int,
        default=0,
        help="Layers to offload. Keep 0: CPU is the reproducible reference.",
    )
    ap.add_argument(
        "--regenerate",
        action="store_true",
        help="Allow overwriting goldens that were produced from a DIFFERENT model file.",
    )
    ap.add_argument(
        "--diagnostics",
        action="store_true",
        help="Also measure run-to-run and batch-vs-single agreement (slow).",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="Do not write. Recompute and compare against the existing golden file.",
    )
    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(
            f"Model not found: {args.model}\n"
            "The 396 MB GGUF is gitignored and must never be committed. Point --model "
            "at your local copy, or set it via the path in tools/README.md."
        )

    print(f"hashing {args.model} ...", file=sys.stderr)
    model_sha = sha256_file(args.model)
    print(f"  sha256 = {model_sha}", file=sys.stderr)

    existing, parse_error = load_existing(args.out)

    # The interlock runs for --check too, so that pointing --check at the wrong model
    # reports "wrong model" rather than 29 unexplained vector differences.
    enforce_model_identity(
        args.out, existing, parse_error, model_sha, args.regenerate, args.check
    )

    if args.check and existing is None:
        raise SystemExit(
            f"--check but {args.out} "
            + ("could not be parsed: " + str(parse_error) if parse_error else "does not exist.")
        )

    # --check must reproduce the committed file exactly, and generated_date is part of
    # that file. Inheriting it means `--check` works with no flags at all; requiring
    # --date would make a maintainer see "CHECK FAILED" on perfectly good data, which is
    # how people learn to ignore an interlock.
    if args.date is None:
        if args.check:
            args.date = existing.get("header", {}).get("generated_date")
            if args.date is None:
                raise SystemExit(
                    f"--check: {args.out} has no header.generated_date to compare "
                    "against. Pass --date explicitly."
                )
            print(
                f"--check: using generated_date={args.date} from the existing golden file",
                file=sys.stderr,
            )
        else:
            args.date = DEFAULT_DATE

    with open(args.corpus, encoding="utf-8") as fh:
        corpus = json.load(fh)

    ids = [e["id"] for e in corpus["texts"]]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise SystemExit(f"Duplicate corpus ids: {sorted(dupes)}")

    ref = Reference(args.model, args.threads, args.gpu_layers)
    print(
        f"loaded reference: llama-cpp-python, pooling=LAST, add_bos={ref.add_bos}, "
        f"add_eos={ref.add_eos}, eos={ref.eos}, dim={ref.n_embd}",
        file=sys.stderr,
    )

    records = build_records(ref, corpus)
    print(f"embedded {len(records)} texts", file=sys.stderr)

    unit_by_id = {
        r["id"]: normalize(
            struct.unpack("<1024f", base64.b64decode(r["embedding_raw_f32_b64"]))
        )
        for r in records
    }
    relations = build_relations(corpus, unit_by_id)

    doc = OrderedDict(
        [
            ("schema_version", SCHEMA_VERSION),
            ("header", build_header(args, ref, corpus, model_sha)),
            ("relations", relations),
            ("vectors", records),
        ]
    )

    if args.diagnostics:
        run_diagnostics(ref, records)

    if args.check:
        with open(args.out, encoding="utf-8") as fh:
            old = fh.read()
        new = json.dumps(doc, ensure_ascii=False, indent=1) + "\n"
        if old == new:
            print("CHECK OK: regenerated output is byte-identical.", file=sys.stderr)
            return 0
        print("CHECK FAILED: regenerated output differs from the committed golden.", file=sys.stderr)
        old_by_id = {v["id"]: v for v in existing.get("vectors", [])}
        new_ids = {r["id"] for r in records}
        for missing in old_by_id.keys() - new_ids:
            print(
                f"  {missing}: present in the committed golden but NOT regenerated "
                "(was it removed from the corpus?)",
                file=sys.stderr,
            )
        for rec in records:
            o = old_by_id.get(rec["id"])
            if o is None:
                print(f"  {rec['id']}: absent from committed golden", file=sys.stderr)
                continue
            if o["token_ids"] != rec["token_ids"]:
                print(f"  {rec['id']}: TOKEN IDS DIFFER", file=sys.stderr)
            if o["embedding_normalized_f32_b64"] != rec["embedding_normalized_f32_b64"]:
                a = normalize(struct.unpack("<1024f", base64.b64decode(o["embedding_normalized_f32_b64"])))
                b = normalize(struct.unpack("<1024f", base64.b64decode(rec["embedding_normalized_f32_b64"])))
                print(
                    f"  {rec['id']}: vector differs, cosine={cosine(a, b):.12f}, "
                    f"max|d|={max(abs(x-y) for x, y in zip(a, b)):.3e}",
                    file=sys.stderr,
                )
        return 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    dump_json(doc, args.out)
    size = os.path.getsize(args.out)
    print(f"wrote {args.out} ({size:,} bytes)", file=sys.stderr)
    if size > 1_000_000:
        print(
            f"warning: golden file is {size:,} bytes, over the ~1 MB reviewability "
            "budget. Consider trimming the corpus.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
