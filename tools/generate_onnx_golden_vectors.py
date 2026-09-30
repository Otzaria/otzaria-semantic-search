#!/usr/bin/env python3
"""Regenerate tests/data/onnx_golden_vectors.json from the real ONNX model package.

The Python reference for the ONNX backend's parity gate (`semantic::onnx_backend::golden`
in src/semantic/onnx_backend.rs). It runs the model's own `tokenizer.json` through the
`tokenizers` package and the fp32 graph through `onnxruntime`: the same tokenizer core
(0.23.2) and the same ONNX Runtime release (1.28.0) the backend is pinned to, reached
through their Python bindings instead of through this crate. What the backend has to
reproduce is therefore the model's own wiring, not a second copy of ours.

Read tools/README.md before running.

The pipeline, per case
----------------------
  input      "[PASSAGE] " + text, "[QUERY] " + text, or text alone -- role passage, query
             or raw. Embedding text version 2 hands the backend exactly these strings.
  token_ids  tokenizer.encode(input, add_special_tokens=True).ids, with the tokenizer's
             own padding turned OFF and its truncation REPLACED by max_length=256 on the
             right, longest_first. 256 is the total -- [CLS], [SEP] and the role token are
             counted -- and content is what gets cut, so the specials survive. Added
             special tokens are matched anywhere in the input (encode_special_tokens off):
             that is how the prefixes become the learned role tokens.
  vector     the graph's first output, for input_ids = [token_ids] and attention_mask =
             ones (token_type_ids = zeros only if the graph declares it), ONE text per
             run. Stored raw, as the graph emitted it: the graph ends in its own L2
             normalization.

Determinism contract
--------------------
Two runs with the same flags, on the same model files and machine, produce a
byte-identical file:

  * one text per run, never a padded batch, so a vector cannot depend on its neighbours;
  * the session's graph optimization level (ORT_ENABLE_ALL, the backend's level), thread
    counts and execution mode are pinned and recorded, and the package versions are
    checked rather than assumed;
  * vectors are base64 of little-endian f32 -- exact, not decimal;
  * key order is fixed, and the only timestamp is header.generated_date, which --date
    pins. Nothing may depend on key order; it exists so a regenerated file diffs cleanly.

Across machines the claim is weaker -- another instruction set runs other kernels -- which
is why the Rust gate compares ids exactly and vectors by cosine.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import math
import os
import platform
import struct
import sys
from collections import OrderedDict
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CASES = os.path.join(REPO_ROOT, "tools", "onnx_golden_cases.json")
DEFAULT_CORPUS = os.path.join(REPO_ROOT, "tools", "golden_corpus.json")
DEFAULT_OUT = os.path.join(REPO_ROOT, "tests", "data", "onnx_golden_vectors.json")

MODEL_ENV = "OTZARIA_TEST_ONNX_MODEL"
RUNTIME_ENV = "OTZARIA_ONNX_RUNTIME"

# What the goldens are pinned to. The source repository is gated; the files are the same
# bytes wherever they come from, which graph_sha256 / tokenizer_sha256 establish.
MODEL_REPO = "ArieLLL123/judaic-semantic-round2-onnx-zayit"
MODEL_REVISION = "1ec8dc68888bcea774ae9f735b2fe7cd9dc7f3ca"
GRAPH_FILE = "seforim-embed-round2-fp32.onnx"
TOKENIZER_FILE = "tokenizer.json"

# The backend's pins (Cargo.toml: tokenizers =0.23.2, and the reference ONNX Runtime).
# A reference on other versions is a different reference, so they are checked, not
# recorded after the fact.
TOKENIZERS_VERSION = "0.23.2"
ONNXRUNTIME_VERSION = "1.28.0"

MAX_TOKENS = 256
EXPECTED_DIM = 256
PREFIXES = {"passage": "[PASSAGE] ", "query": "[QUERY] ", "raw": ""}
ROLE_TOKENS = {"passage": "[PASSAGE]", "query": "[QUERY]"}

DEFAULT_THREADS = 4
DEFAULT_DATE = "1970-01-01"

GENERATOR = "tools/generate_onnx_golden_vectors.py"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def f32_b64(values) -> str:
    """base64 of little-endian float32 bytes. Exact and compact."""
    return base64.b64encode(struct.pack("<%df" % len(values), *values)).decode("ascii")


def f32_unb64(encoded: str) -> list[float]:
    raw = base64.b64decode(encoded)
    return list(struct.unpack("<%df" % (len(raw) // 4), raw))


def l2_norm(values) -> float:
    return math.sqrt(sum(float(v) * float(v) for v in values))


def cosine(a, b) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b)) / (l2_norm(a) * l2_norm(b))


def load_package_checksum():
    """tools/onnx_package_checksum.py -- the independent D4 implementation."""
    path = os.path.join(REPO_ROOT, "tools", "onnx_package_checksum.py")
    spec = importlib.util.spec_from_file_location("onnx_package_checksum", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_versions():
    try:
        import numpy  # noqa: F401
        import onnxruntime
        import tokenizers
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            f"{exc.name} is not installed in this interpreter.\n"
            "See tools/README.md for the pinned install command."
        ) from exc
    wrong = [
        f"{name} {have} (the reference is {want})"
        for name, have, want in (
            ("tokenizers", tokenizers.__version__, TOKENIZERS_VERSION),
            ("onnxruntime", onnxruntime.__version__, ONNXRUNTIME_VERSION),
        )
        if have != want
    ]
    if wrong:
        raise SystemExit(
            "Refusing to generate with " + ", ".join(wrong) + ". The goldens are the answers "
            "of the versions the backend is pinned to; a reference on other versions is a "
            "different reference. Install the pins from tools/README.md."
        )
    return tokenizers, onnxruntime


class Reference:
    """The tokenizer and the graph, configured as the backend configures them."""

    def __init__(self, graph: str, tokenizer_path: str, threads: int, optimization: str):
        tokenizers, ort = require_versions()
        import numpy as np

        self.np = np
        self.ort = ort
        self.tokenizers = tokenizers

        self.tokenizer = tokenizers.Tokenizer.from_file(tokenizer_path)
        # What the file itself asks for, recorded, then overridden: the caller owns
        # truncation, and the tokenizer's own padding would put pad ids in a sequence
        # that is never batched.
        self.file_padding = self.tokenizer.padding
        self.file_truncation = self.tokenizer.truncation
        self.tokenizer.no_padding()
        self.tokenizer.enable_truncation(
            max_length=MAX_TOKENS, stride=0, strategy="longest_first", direction="right"
        )
        if self.tokenizer.encode_special_tokens:
            raise SystemExit(
                "tokenizer.json turns encode_special_tokens on, so the role prefixes would "
                "be spelled out as text instead of becoming their tokens. Refusing."
            )
        self.untruncated = tokenizers.Tokenizer.from_file(tokenizer_path)
        self.untruncated.no_padding()
        self.untruncated.no_truncation()

        # Every added special token, by id: each is matched wherever it is spelled in an
        # input, the two role tokens and [CLS]/[SEP] included.
        added = self.tokenizer.get_added_tokens_decoder()
        self.special = OrderedDict(
            (added[i].content, i) for i in sorted(added) if added[i].special
        )
        for token in ("[CLS]", "[SEP]", "[QUERY]", "[PASSAGE]"):
            if token not in self.special:
                raise SystemExit(f"tokenizer.json has no special {token} token. Wrong tokenizer?")
        self.role_token_flags = OrderedDict()
        for token in ("[QUERY]", "[PASSAGE]"):
            info = added[self.special[token]]
            if not info.special:
                raise SystemExit(
                    f"{token} is not an added special token in tokenizer.json, so it would be "
                    "split like text. The model's role prefixes are learned special tokens."
                )
            self.role_token_flags[token] = OrderedDict(
                [
                    ("id", self.special[token]),
                    ("lstrip", info.lstrip),
                    ("normalized", info.normalized),
                    ("rstrip", info.rstrip),
                    ("single_word", info.single_word),
                    ("special", info.special),
                ]
            )

        self.optimization = optimization
        self.threads = threads
        options = ort.SessionOptions()
        options.graph_optimization_level = getattr(ort.GraphOptimizationLevel, optimization)
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(
            graph, sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.inputs = OrderedDict((i.name, i) for i in self.session.get_inputs())
        self.output = self.session.get_outputs()[0]
        for name in ("input_ids", "attention_mask"):
            if name not in self.inputs:
                raise SystemExit(f"The graph has no {name} input: {list(self.inputs)}")
        extra = [n for n in self.inputs if n not in ("input_ids", "attention_mask", "token_type_ids")]
        if extra:
            raise SystemExit(f"The graph needs inputs the backend does not feed: {extra}")
        for name, info in self.inputs.items():
            if info.type != "tensor(int64)":
                raise SystemExit(f"Input {name} is {info.type}, not tensor(int64)")
        if self.output.type != "tensor(float)" or len(self.output.shape) != 2:
            raise SystemExit(
                f"The first output is {self.output.type} {self.output.shape}; the backend "
                "serves only a finished [batch, dim] float32 sentence vector."
            )
        if self.output.shape[1] != EXPECTED_DIM:
            raise SystemExit(f"The first output is {self.output.shape}, expected dim {EXPECTED_DIM}")

    def tokenize(self, text: str) -> tuple[list[int], int]:
        """(ids as the backend must produce them, the untruncated id count)."""
        encoding = self.tokenizer.encode(text, add_special_tokens=True)
        ids = list(encoding.ids)
        if any(m != 1 for m in encoding.attention_mask):
            raise SystemExit(f"{text!r}: the attention mask is not all ones; padding is on")
        full = len(self.untruncated.encode(text, add_special_tokens=True).ids)
        return ids, full

    def run(self, ids: list[int], session=None) -> list[float]:
        np = self.np
        feeds = {
            "input_ids": np.array([ids], dtype=np.int64),
            "attention_mask": np.ones((1, len(ids)), dtype=np.int64),
        }
        if "token_type_ids" in self.inputs:
            feeds["token_type_ids"] = np.zeros((1, len(ids)), dtype=np.int64)
        out = (session or self.session).run([self.output.name], feeds)[0]
        if out.shape != (1, EXPECTED_DIM) or out.dtype != np.float32:
            raise SystemExit(f"The graph returned {out.dtype} {out.shape}")
        vector = [float(v) for v in out[0]]
        if not all(math.isfinite(v) for v in vector):
            raise SystemExit("The graph produced NaN or Inf; refusing to write.")
        return vector

    def session_with(self, *, optimization=None, threads=None, config=None):
        """Another session over the same graph, for the diagnostics."""
        ort = self.ort
        options = ort.SessionOptions()
        options.graph_optimization_level = getattr(
            ort.GraphOptimizationLevel, optimization or self.optimization
        )
        options.intra_op_num_threads = threads or self.threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        for key, value in (config or {}).items():
            options.add_session_config_entry(key, value)
        return ort.InferenceSession(
            self.session._model_path, sess_options=options, providers=["CPUExecutionProvider"]
        )


def resolve_cases(cases_doc: dict, corpus: dict) -> list[dict]:
    by_id = {e["id"]: e for e in corpus["texts"]}
    resolved = []
    names = set()
    for case in cases_doc["cases"]:
        name = case["name"]
        if name in names:
            raise SystemExit(f"Duplicate case name {name!r}")
        names.add(name)
        role = case["role"]
        if role not in PREFIXES:
            raise SystemExit(f"{name}: role {role!r} is not passage, query or raw")
        if ("text" in case) == ("corpus_id" in case):
            raise SystemExit(f"{name}: exactly one of 'text' or 'corpus_id' is required")
        if "text" in case:
            text = case["text"]
            source = "tools/onnx_golden_cases.json"
        else:
            entry = by_id.get(case["corpus_id"])
            if entry is None:
                raise SystemExit(f"{name}: no corpus entry {case['corpus_id']!r}")
            if "text" not in entry:
                raise SystemExit(f"{name}: corpus entry {case['corpus_id']!r} has no own text")
            text = entry["text"]
            source = f"tools/golden_corpus.json#{case['corpus_id']}"
            if "prefix_chars" in case:
                cut = int(case["prefix_chars"])
                if not 0 < cut < len(text):
                    raise SystemExit(f"{name}: prefix_chars {cut} is outside the text")
                text = text[:cut]
                source += f"[:{cut}]"
        resolved.append(dict(case, text=text, source=source, input=PREFIXES[role] + text))
    return resolved


def report_prefix_spacing(ref: Reference, cases: list[dict]) -> None:
    """Whether the space after the role prefix changes the ids, case by case.

    The recipe writes "[PASSAGE] " and "[QUERY] " with the space, as the model card and
    the author's own parity checks do; the author's manifest records the bare tokens.
    For a text that begins with something visible the two spellings tokenize alike; one
    that begins with whitespace, or with another special token, gets one more id from
    the spaced form. Reported, not asserted: which texts do that is the tokenizer's
    fact, and the goldens pin the spaced form either way.
    """
    same, differ = 0, []
    for case in cases:
        prefix = PREFIXES[case["role"]]
        if not prefix:
            continue
        spaced = ref.tokenize(case["input"])[0]
        bare = ref.tokenize(prefix.rstrip(" ") + case["text"])[0]
        if spaced == bare:
            same += 1
        else:
            differ.append(case["name"])
    print(
        f"prefix spacing: {same} of {same + len(differ)} role-prefixed cases give the same "
        f"ids without the space after the prefix; different: {differ or 'none'}",
        file=sys.stderr,
    )


def build_records(ref: Reference, cases: list[dict]) -> list[OrderedDict]:
    records = []
    cls, sep = ref.special["[CLS]"], ref.special["[SEP]"]
    for case in cases:
        name, role, text_input = case["name"], case["role"], case["input"]
        ids, full = ref.tokenize(text_input)
        truncated = full > len(ids)

        if not ids or ids[0] != cls or ids[-1] != sep:
            raise SystemExit(f"{name}: the ids do not run [CLS] ... [SEP]: {ids[:3]}...{ids[-3:]}")
        if role in ROLE_TOKENS and ids[1] != ref.special[ROLE_TOKENS[role]]:
            raise SystemExit(
                f"{name}: the second id is {ids[1]}, not {ROLE_TOKENS[role]} "
                f"({ref.special[ROLE_TOKENS[role]]}): the prefix did not become its token"
            )
        if role == "raw" and ids[1] in (ref.special["[QUERY]"], ref.special["[PASSAGE]"]):
            raise SystemExit(f"{name}: a raw case begins with a role token")
        if len(ids) > MAX_TOKENS or (truncated and len(ids) != MAX_TOKENS):
            raise SystemExit(f"{name}: {len(ids)} ids from {full}; truncation is not at {MAX_TOKENS}")
        for key, got in (("expect_total_tokens", len(ids)), ("expect_untruncated_tokens", full)):
            if key in case and got != int(case[key]):
                raise SystemExit(
                    f"{name}: {key}={case[key]} but the tokenizer produced {got}.\n"
                    "This case pins a token-count boundary on purpose: the tokenizer or the "
                    "text changed. Recalibrate it deliberately (--boundary-prefix) -- do not "
                    "relax the expectation."
                )

        vector = ref.run(ids)
        records.append(
            OrderedDict(
                [
                    ("name", name),
                    ("role", role),
                    ("input", text_input),
                    ("token_ids", ids),
                    ("vector_f32_le_base64", f32_b64(vector)),
                    ("input_utf8_sha256", sha256_text(text_input)),
                    ("notes", case.get("notes", "")),
                    ("source", case["source"]),
                    ("truncated", truncated),
                    ("untruncated_token_count", full),
                ]
            )
        )
    return records


def build_header(args, ref: Reference, package, cases_doc: dict, corpus: dict) -> OrderedDict:
    graph_file = next(f for f in package.files if f.path == package.graph)
    tokenizer_file = next(f for f in package.files if f.relpath == TOKENIZER_FILE)
    tokenizer_json = json.loads(open(ref_tokenizer_path(args), encoding="utf-8").read())

    def kind(component):
        return None if component is None else component.get("type")

    return OrderedDict(
        [
            # The schema the Rust gate reads.
            ("model_repo", MODEL_REPO),
            ("model_revision", MODEL_REVISION),
            ("graph_file", os.path.basename(args.model)),
            ("graph_sha256", graph_file.sha256),
            ("tokenizer_sha256", tokenizer_file.sha256),
            ("package_checksum", package.checksum),
            ("max_tokens", MAX_TOKENS),
            ("dim", EXPECTED_DIM),
            ("generator", GENERATOR),
            ("onnxruntime_version", ref.ort.__version__),
            ("tokenizers_version", ref.tokenizers.__version__),
            # Provenance.
            ("cases_file", "tools/onnx_golden_cases.json"),
            ("cases_version", cases_doc.get("cases_version")),
            ("corpus_file", "tools/golden_corpus.json"),
            ("corpus_version", corpus.get("corpus_version")),
            ("env_var_for_model_path", MODEL_ENV),
            ("env_var_for_runtime", RUNTIME_ENV),
            ("generated_date", args.date),
            ("graph_size_bytes", graph_file.size),
            (
                "pipeline",
                "input = role prefix + text; tokenizer.encode(input, add_special_tokens=true) "
                f"with padding off and truncation replaced by max_length={MAX_TOKENS}, right, "
                "longest_first (the total, specials and role token included); one text per "
                "run; input_ids and an all-ones attention_mask (token_type_ids zeros only if "
                "declared); the first output, raw",
            ),
            (
                "reference",
                OrderedDict(
                    [
                        ("execution_mode", "ORT_SEQUENTIAL"),
                        ("execution_provider", "CPUExecutionProvider"),
                        ("graph_inputs", list(ref.inputs)),
                        ("graph_optimization_level", ref.optimization),
                        ("graph_output", ref.output.name),
                        ("implementation", "onnxruntime and tokenizers, Python bindings"),
                        ("inter_op_num_threads", 1),
                        ("intra_op_num_threads", ref.threads),
                        ("machine", f"{platform.system()} {platform.machine()}"),
                        ("run", "one text per run"),
                    ]
                ),
            ),
            (
                "tokenizer",
                OrderedDict(
                    [
                        ("add_special_tokens", True),
                        ("encode_special_tokens", False),
                        ("file_padding", ref.file_padding),
                        ("file_truncation", ref.file_truncation),
                        ("model", kind(tokenizer_json.get("model"))),
                        ("normalizer", kind(tokenizer_json.get("normalizer"))),
                        ("padding", None),
                        ("post_processor", kind(tokenizer_json.get("post_processor"))),
                        ("pre_tokenizer", kind(tokenizer_json.get("pre_tokenizer"))),
                        ("role_tokens", ref.role_token_flags),
                        ("special_token_ids", ref.special),
                        (
                            "truncation",
                            OrderedDict(
                                [
                                    ("direction", "right"),
                                    ("max_length", MAX_TOKENS),
                                    ("strategy", "longest_first"),
                                    ("stride", 0),
                                ]
                            ),
                        ),
                    ]
                ),
            ),
            ("tokenizer_size_bytes", tokenizer_file.size),
            (
                "vector_encoding",
                f"vector_f32_le_base64 is base64 (standard, padded) of {EXPECTED_DIM} "
                f"little-endian IEEE-754 binary32 values = {EXPECTED_DIM * 4} bytes: the "
                "graph's first output exactly as it was emitted. The graph normalizes, so "
                "it is unit-norm to float precision; nothing was done to it after the run.",
            ),
        ]
    )


def ref_tokenizer_path(args) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(args.model)), TOKENIZER_FILE)


def run_diagnostics(ref: Reference, records: list[OrderedDict]) -> None:
    """Print the measurements the tolerance in the Rust gate rests on. Writes nothing."""
    np = ref.np

    def spread(label, other):
        worst_cos, worst_abs, worst = 1.0, 0.0, None
        identical = 0
        for rec in records:
            golden = f32_unb64(rec["vector_f32_le_base64"])
            got = other(rec)
            if got == golden:
                identical += 1
            c = cosine(golden, got)
            m = max(abs(a - b) for a, b in zip(golden, got))
            if c < worst_cos:
                worst_cos, worst = c, rec["name"]
            worst_abs = max(worst_abs, m)
        print(
            f"{label:<44} bit-identical {identical}/{len(records)}, min cosine "
            f"{worst_cos:.12f} ({worst}), max |component diff| {worst_abs:.3e}",
            file=sys.stderr,
        )

    print("\n=== diagnostics (against the vectors just computed) ===", file=sys.stderr)
    spread("same session, run again", lambda r: ref.run(r["token_ids"]))
    fresh = ref.session_with()
    spread("a fresh session", lambda r: ref.run(r["token_ids"], fresh))
    for threads in (1, 2, 8):
        session = ref.session_with(threads=threads)
        spread(f"intra_op threads {threads}", lambda r, s=session: ref.run(r["token_ids"], s))
    for level in ("ORT_DISABLE_ALL", "ORT_ENABLE_BASIC", "ORT_ENABLE_EXTENDED"):
        session = ref.session_with(optimization=level)
        spread(f"optimization {level}", lambda r, s=session: ref.run(r["token_ids"], s))
    try:
        session = ref.session_with(config={"mlas.disable_kleidiai": "1"})
        spread("KleidiAI disabled", lambda r, s=session: ref.run(r["token_ids"], s))
    except Exception as exc:  # the key is platform-specific
        print(f"KleidiAI toggle unavailable: {exc}", file=sys.stderr)

    batch = ref.inputs["input_ids"].shape[0]
    print(f"declared input shape: {ref.inputs['input_ids'].shape} (batch {batch!r})", file=sys.stderr)

    int8 = os.path.join(os.path.dirname(ref.session._model_path), "seforim-embed-round2-int8.onnx")
    if os.path.exists(int8):
        options = ref.ort.SessionOptions()
        options.intra_op_num_threads = ref.threads
        session = ref.ort.InferenceSession(int8, sess_options=options, providers=["CPUExecutionProvider"])
        spread("the int8 graph (not the goldens' model)", lambda r, s=session: ref.run(r["token_ids"], s))

    norms = [l2_norm(f32_unb64(r["vector_f32_le_base64"])) for r in records]
    print(f"vector L2 norms: min {min(norms):.9f} max {max(norms):.9f}", file=sys.stderr)

    by = {r["name"]: f32_unb64(r["vector_f32_le_base64"]) for r in records}
    pairs = [
        ("passage_near_identical_a", "passage_near_identical_b"),
        ("passage_unrelated_a", "passage_unrelated_b"),
        ("passage_heb_prose_mishneh_torah", "query_heb_prose_mishneh_torah"),
        ("passage_heb_prose_mishneh_torah", "raw_heb_prose_mishneh_torah"),
        ("query_heb_prose_mishneh_torah", "raw_heb_prose_mishneh_torah"),
    ]
    for a, b in pairs:
        if a in by and b in by:
            print(f"cosine {a} ~ {b}: {cosine(by[a], by[b]):.6f}", file=sys.stderr)
    del np


def boundary_prefix(ref: Reference, corpus: dict, corpus_id: str, role: str, total: int) -> None:
    """Print the prefix_chars values, cut at a word's end, whose untruncated input is
    exactly `total` ids -- how the boundary cases are calibrated."""
    text = {e["id"]: e for e in corpus["texts"]}[corpus_id]["text"]
    found = []
    for cut in range(1, len(text)):
        if not text[cut].isspace() or text[cut - 1].isspace():
            continue
        _, full = ref.tokenize(PREFIXES[role] + text[:cut])
        if full == total:
            found.append(cut)
        if full > total:
            break
    print(json.dumps({"corpus_id": corpus_id, "role": role, "total": total, "prefix_chars": found}))


def load_existing(out_path: str):
    if not os.path.exists(out_path):
        return None, None
    try:
        with open(out_path, encoding="utf-8") as fh:
            return json.load(fh), None
    except (json.JSONDecodeError, OSError) as exc:
        return None, str(exc)


def enforce_model_identity(out_path, existing, parse_error, graph_sha, tokenizer_sha, regenerate, checking):
    """Refuse to touch a golden file that did not come from the files just hashed -- for
    --check too, where a wrong model would otherwise read as a wall of drifted vectors.
    A golden that does not record both hashes is a mismatch, not permission."""
    if existing is None and parse_error is None:
        return
    if parse_error is not None:
        reason = f"could not be parsed ({parse_error})"
    else:
        header = existing.get("header", {})
        old = (header.get("graph_sha256"), header.get("tokenizer_sha256"))
        if old == (graph_sha, tokenizer_sha):
            return
        reason = (
            f"records graph {old[0]} and tokenizer {old[1]}, and these files are graph "
            f"{graph_sha} and tokenizer {tokenizer_sha}"
        )
    if checking:
        raise SystemExit(
            f"*** REFUSING TO --check: the golden file {out_path} {reason}. Point --model at "
            "the package the goldens came from; --regenerate does not silence this."
        )
    if not regenerate:
        raise SystemExit(
            f"*** REFUSING TO OVERWRITE: the golden file {out_path} {reason}. Replacing it "
            "would move every assertion to a different model silently. Rerun with "
            "--regenerate if that is the intent, and say so in the commit message."
        )
    print(f"warning: --regenerate given; the existing golden {reason}. Overwriting.", file=sys.stderr)


def dumps(doc) -> str:
    # ensure_ascii=False keeps the Hebrew readable in a diff; input_utf8_sha256 is what
    # pins the exact bytes, invisible characters included.
    return json.dumps(doc, ensure_ascii=False, indent=1) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--model",
        default=os.environ.get(MODEL_ENV),
        help=f"The fp32 graph ({GRAPH_FILE}), with tokenizer.json beside it. Defaults to "
        f"${MODEL_ENV}.",
    )
    ap.add_argument("--cases", default=DEFAULT_CASES)
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument(
        "--date",
        default=None,
        help="header.generated_date. Under --check it defaults to the committed file's; when "
        f"writing, to {DEFAULT_DATE}. Pass a real date for a golden you intend to commit.",
    )
    ap.add_argument("--threads", type=int, default=DEFAULT_THREADS, help="Pinned intra-op threads.")
    ap.add_argument(
        "--optimization",
        default="ORT_ENABLE_ALL",
        choices=["ORT_DISABLE_ALL", "ORT_ENABLE_BASIC", "ORT_ENABLE_EXTENDED", "ORT_ENABLE_ALL"],
        help="Graph optimization level. Keep ORT_ENABLE_ALL: it is the backend's.",
    )
    ap.add_argument("--check", action="store_true", help="Recompute and compare; write nothing.")
    ap.add_argument(
        "--regenerate",
        action="store_true",
        help="Allow overwriting goldens produced from DIFFERENT model files.",
    )
    ap.add_argument("--diagnostics", action="store_true", help="Also measure what can move a vector.")
    ap.add_argument(
        "--boundary-prefix",
        nargs=3,
        metavar=("CORPUS_ID", "ROLE", "TOTAL"),
        help="Print the prefix_chars at which a corpus text's input is exactly TOTAL ids, "
        "and exit. For calibrating the boundary cases.",
    )
    args = ap.parse_args()

    if not args.model:
        raise SystemExit(
            f"No model: pass --model or set {MODEL_ENV} to the fp32 graph ({GRAPH_FILE}). "
            "It is gated and never committed; see tools/README.md."
        )
    if not os.path.isfile(args.model):
        raise SystemExit(f"Model not found: {args.model}")
    if os.path.basename(args.model) != GRAPH_FILE:
        raise SystemExit(
            f"{args.model} is not {GRAPH_FILE}. The goldens and model.json describe the fp32 "
            "graph; the int8 graph is a different model."
        )

    with open(args.corpus, encoding="utf-8") as fh:
        corpus = json.load(fh)

    if args.boundary_prefix:
        corpus_id, role, total = args.boundary_prefix
        ref = Reference(args.model, ref_tokenizer_path(args), args.threads, args.optimization)
        boundary_prefix(ref, corpus, corpus_id, role, int(total))
        return 0

    print(f"validating the package of {args.model} ...", file=sys.stderr)
    checksum_tool = load_package_checksum()
    try:
        package = checksum_tool.validate_onnx_package(args.model)
    except checksum_tool.PackageRefused as refused:
        raise SystemExit(f"The model package is refused: {refused}") from None
    for f in package.files:
        print(f"  {f.relpath}\t{f.size}\t{f.sha256}", file=sys.stderr)
    print(f"  package checksum {package.checksum}", file=sys.stderr)
    graph_sha = next(f.sha256 for f in package.files if f.path == package.graph)
    tokenizer_sha = next(f.sha256 for f in package.files if f.relpath == TOKENIZER_FILE)

    existing, parse_error = load_existing(args.out)
    enforce_model_identity(
        args.out, existing, parse_error, graph_sha, tokenizer_sha, args.regenerate, args.check
    )
    if args.check and existing is None:
        raise SystemExit(f"--check, but {args.out} " + (parse_error or "does not exist"))
    if args.date is None:
        if args.check:
            args.date = existing.get("header", {}).get("generated_date")
            if args.date is None:
                raise SystemExit(f"--check: {args.out} has no header.generated_date; pass --date")
        else:
            args.date = DEFAULT_DATE

    with open(args.cases, encoding="utf-8") as fh:
        cases_doc = json.load(fh)
    cases = resolve_cases(cases_doc, corpus)

    ref = Reference(args.model, ref_tokenizer_path(args), args.threads, args.optimization)
    print(
        f"reference: onnxruntime {ref.ort.__version__}, tokenizers {ref.tokenizers.__version__}, "
        f"{ref.optimization}, {ref.threads} thread(s); inputs {list(ref.inputs)}, output "
        f"{ref.output.name} {ref.output.shape}",
        file=sys.stderr,
    )
    records = build_records(ref, cases)
    print(f"embedded {len(records)} cases", file=sys.stderr)
    report_prefix_spacing(ref, cases)

    doc = OrderedDict(
        [
            ("header", build_header(args, ref, package, cases_doc, corpus)),
            ("cases", records),
        ]
    )

    if args.diagnostics:
        run_diagnostics(ref, records)

    text = dumps(doc)
    if args.check:
        with open(args.out, encoding="utf-8") as fh:
            old = fh.read()
        if old == text:
            print("CHECK OK: regenerated output is byte-identical.", file=sys.stderr)
            return 0
        print("CHECK FAILED: regenerated output differs from the committed golden.", file=sys.stderr)
        old_by = {c["name"]: c for c in existing.get("cases", [])}
        for rec in records:
            o = old_by.pop(rec["name"], None)
            if o is None:
                print(f"  {rec['name']}: absent from the committed golden", file=sys.stderr)
                continue
            if o.get("token_ids") != rec["token_ids"]:
                print(f"  {rec['name']}: TOKEN IDS DIFFER", file=sys.stderr)
            if o.get("vector_f32_le_base64") != rec["vector_f32_le_base64"]:
                a = f32_unb64(o["vector_f32_le_base64"])
                b = f32_unb64(rec["vector_f32_le_base64"])
                print(
                    f"  {rec['name']}: vector differs, cosine {cosine(a, b):.12f}, max |d| "
                    f"{max(abs(x - y) for x, y in zip(a, b)):.3e}",
                    file=sys.stderr,
                )
        for name in old_by:
            print(f"  {name}: in the committed golden but not regenerated", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    size = os.path.getsize(args.out)
    print(f"wrote {args.out} ({size:,} bytes)", file=sys.stderr)
    if size > 1_000_000:
        print(f"warning: {size:,} bytes is over the ~1 MB reviewability budget", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
