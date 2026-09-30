#!/usr/bin/env python3
"""Generate the tiny ONNX model packages under `tests/data/onnx_fixture/`.

The ONNX backend's in-crate tests and `tests/onnx_backend.rs` run against these instead
of the 168 MB production graph, so the ordinary test suite needs nothing gated — only
an ONNX Runtime shared library (`OTZARIA_ONNX_RUNTIME`). They are small enough to commit
(a few KB each), and this script regenerates them byte for byte.

What is written:

* `tokenizer.json` — a BERT-style WordPiece tokenizer: `[PAD] [UNK] [CLS] [SEP] [MASK]
  [QUERY] [PASSAGE]` as special tokens, some Hebrew and English words, and single
  letters and `##` continuations so an unknown word still splits. Its own padding and
  truncation are deliberately **on** (padding to 16, truncation at 512): the backend must
  turn padding off and replace the truncation, and the tests prove it did.
* `dynamic.onnx` — `input_ids`/`attention_mask` `int64[batch, sequence]`, a token and a
  position embedding table (48 positions), masked mean pooling, a projection to 4
  components: `sentence_embedding float[batch, 4]`. The position table makes a cap above
  48 tokens fail inside the graph, which is what the load-time probe must catch.
* `static_batch.onnx` — the same with the batch dimension fixed at 1.
* `token_level.onnx` — the projection applied per token, output `float[batch, sequence,
  4]`: the rank-3 graph the backend must refuse.
* `extra_input.onnx` — a third required input, `position_ids`, which the backend cannot
  feed and must refuse by name.
* `token_types.onnx` — `dynamic.onnx` plus a declared `token_type_ids` input and a type
  embedding whose row 0 is zero, so its vectors equal `dynamic.onnx`'s exactly when, and
  only when, the backend feeds zeros.
* `expected.json` — the Python references' answers for a set of texts: the token ids
  from the `tokenizers` package at a given cap, and `dynamic.onnx`'s vector from the
  `onnxruntime` package. The Rust tests assert the ids exactly and the vectors to a
  tolerance, which is also where special-token matching inside text is compared
  against the reference.

Weights come from integer formulas, not a random generator, so no random state is
involved. The output is byte-stable for the package versions below — `expected.json`
records them, and a different `onnxruntime` or `tokenizers` could legitimately change a
reference, which is the point of recording it. Run from the repository root, in a
throwaway virtualenv outside the repository:

    python3 -m venv /tmp/onnx-fixture-venv
    /tmp/onnx-fixture-venv/bin/pip install onnx==1.23.1 tokenizers==0.23.2 \
        onnxruntime==1.30.0 numpy==2.5.3
    /tmp/onnx-fixture-venv/bin/python tools/make_onnx_fixture.py

`--check` regenerates into memory and fails if anything differs from the committed files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import tokenizers
from onnx import TensorProto, helper
from tokenizers import AddedToken, Tokenizer, decoders, models, normalizers, pre_tokenizers, processors

OUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "data" / "onnx_fixture"

# ONNX IR 8 / opset 17: old enough for every ONNX Runtime from 1.17 on, the backend's
# API floor, so the fixture never tests the runtime's version instead of the backend.
IR_VERSION = 8
OPSET = 17

HIDDEN = 8
DIM = 4
MAX_POSITIONS = 48

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[QUERY]", "[PASSAGE]"]
WORDS = [
    # English, lowercase: the normalizer lowercases before the vocabulary is consulted.
    "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "in", "beginning",
    "god", "created", "heaven", "earth", "and", "query", "passage", "cls",
    # Hebrew.
    "בראשית", "ברא", "אלהים", "את", "השמים", "ואת", "הארץ", "תורה", "משה", "ישראל",
    "שלום", "ספר",
]
HEBREW_LETTERS = list("אבגדהוזחטיכךלמםנןסעפףצץקרשת")
LATIN_LETTERS = list("abcdefghijklmnopqrstuvwxyz")
DIGITS = list("0123456789")
PUNCTUATION = list(".,!?:;-()[]'\"")


def vocabulary() -> list[str]:
    pieces = SPECIALS + WORDS
    for letters in (HEBREW_LETTERS, LATIN_LETTERS, DIGITS):
        pieces += letters
        pieces += ["##" + letter for letter in letters]
    pieces += PUNCTUATION
    assert len(pieces) == len(set(pieces)), "duplicate vocabulary entry"
    return pieces


VOCAB = vocabulary()


def build_tokenizer() -> Tokenizer:
    vocab = {piece: index for index, piece in enumerate(VOCAB)}
    tokenizer = Tokenizer(models.WordPiece(vocab, unk_token="[UNK]", max_input_chars_per_word=100))
    tokenizer.normalizer = normalizers.BertNormalizer(
        clean_text=True, handle_chinese_chars=True, strip_accents=None, lowercase=True
    )
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )
    tokenizer.decoder = decoders.WordPiece(prefix="##")
    # The same flags a BERT export gives its specials: matched on the raw text, not
    # normalized, no stripping.
    tokenizer.add_special_tokens(
        [AddedToken(token, special=True, normalized=False) for token in SPECIALS]
    )
    # Deliberately on: the backend must override both.
    tokenizer.enable_padding(pad_id=vocab["[PAD]"], pad_token="[PAD]", length=16)
    tokenizer.enable_truncation(max_length=512)
    return tokenizer


# ── weights ─────────────────────────────────────────────────────────────────────────


def formula_matrix(rows: int, cols: int, salt: int) -> np.ndarray:
    """Small, distinct, deterministic values in [-0.5, 0.5): integer arithmetic only."""
    values = [
        [((r * 7919 + c * 104729 + salt * 1299709) % 997) / 997.0 - 0.5 for c in range(cols)]
        for r in range(rows)
    ]
    return np.array(values, dtype=np.float32)


def tensor(name: str, array: np.ndarray) -> TensorProto:
    return helper.make_tensor(
        name,
        helper.np_dtype_to_tensor_dtype(array.dtype),
        array.shape,
        array.flatten().tolist(),
    )


def int64_scalar(name: str, value: int) -> TensorProto:
    return helper.make_tensor(name, TensorProto.INT64, [], [value])


def int64_vector(name: str, values: list[int]) -> TensorProto:
    return helper.make_tensor(name, TensorProto.INT64, [len(values)], values)


TOKEN_EMBEDDING = formula_matrix(len(VOCAB), HIDDEN, 1)
POSITION_EMBEDDING = formula_matrix(MAX_POSITIONS, HIDDEN, 2)
PROJECTION = formula_matrix(HIDDEN, DIM, 3)
PROJECTION_BIAS = formula_matrix(1, DIM, 4).reshape(DIM)
# Row 0 zero: fed zeros, the type embedding adds nothing.
TYPE_EMBEDDING = np.concatenate(
    [np.zeros((1, HIDDEN), dtype=np.float32), formula_matrix(1, HIDDEN, 5)], axis=0
)


# ── graphs ──────────────────────────────────────────────────────────────────────────


def int64_input(name: str, batch: int | str) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.INT64, [batch, "sequence"])


def model(name: str, nodes, inputs, outputs, initializers) -> onnx.ModelProto:
    graph = helper.make_graph(nodes, name, inputs, outputs, initializers)
    built = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", OPSET)],
        producer_name="otzaria-semantic-search/tools/make_onnx_fixture.py",
    )
    built.ir_version = IR_VERSION
    onnx.checker.check_model(built, full_check=True)
    return built


def hidden_states(positions_from_input: bool, token_types: bool):
    """Nodes and initializers producing `hidden` = token + position (+ type) embedding."""
    nodes = [helper.make_node("Gather", ["token_embedding", "input_ids"], ["tokens"])]
    initializers = [
        tensor("token_embedding", TOKEN_EMBEDDING),
        tensor("position_embedding", POSITION_EMBEDDING),
    ]
    if positions_from_input:
        nodes.append(helper.make_node("Gather", ["position_embedding", "position_ids"], ["positions"]))
    else:
        nodes += [
            helper.make_node("Shape", ["input_ids"], ["input_shape"]),
            helper.make_node("Gather", ["input_shape", "one"], ["sequence_length"], axis=0),
            helper.make_node("Range", ["zero", "sequence_length", "one"], ["position_index"]),
            helper.make_node("Gather", ["position_embedding", "position_index"], ["positions"]),
        ]
        initializers += [int64_scalar("zero", 0), int64_scalar("one", 1)]
    nodes.append(helper.make_node("Add", ["tokens", "positions"], ["embedded"]))
    if token_types:
        nodes += [
            helper.make_node("Gather", ["type_embedding", "token_type_ids"], ["types"]),
            helper.make_node("Add", ["embedded", "types"], ["hidden"]),
        ]
        initializers.append(tensor("type_embedding", TYPE_EMBEDDING))
    else:
        nodes.append(helper.make_node("Identity", ["embedded"], ["hidden"]))
    return nodes, initializers


def pooled_projection():
    """Masked mean over the sequence, then the projection: `hidden` -> `sentence_embedding`."""
    nodes = [
        helper.make_node("Cast", ["attention_mask"], ["mask"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["mask", "last_axis"], ["mask_column"]),
        helper.make_node("Mul", ["hidden", "mask_column"], ["masked"]),
        helper.make_node("ReduceSum", ["masked", "sequence_axis"], ["summed"], keepdims=0),
        helper.make_node("ReduceSum", ["mask_column", "sequence_axis"], ["count"], keepdims=0),
        helper.make_node("Max", ["count", "epsilon"], ["safe_count"]),
        helper.make_node("Div", ["summed", "safe_count"], ["mean"]),
        helper.make_node("MatMul", ["mean", "projection"], ["projected"]),
        helper.make_node("Add", ["projected", "projection_bias"], ["sentence_embedding"]),
    ]
    initializers = [
        int64_vector("last_axis", [-1]),
        int64_vector("sequence_axis", [1]),
        helper.make_tensor("epsilon", TensorProto.FLOAT, [1], [1e-9]),
        tensor("projection", PROJECTION),
        tensor("projection_bias", PROJECTION_BIAS),
    ]
    return nodes, initializers


def sentence_graph(name: str, batch: int | str, *, token_types=False, extra_input=False):
    nodes, initializers = hidden_states(positions_from_input=extra_input, token_types=token_types)
    pool_nodes, pool_initializers = pooled_projection()
    inputs = [int64_input("input_ids", batch), int64_input("attention_mask", batch)]
    if token_types:
        inputs.append(int64_input("token_type_ids", batch))
    if extra_input:
        inputs.append(int64_input("position_ids", batch))
    outputs = [helper.make_tensor_value_info("sentence_embedding", TensorProto.FLOAT, [batch, DIM])]
    return model(name, nodes + pool_nodes, inputs, outputs, initializers + pool_initializers)


def token_level_graph():
    nodes, initializers = hidden_states(positions_from_input=False, token_types=False)
    nodes += [
        helper.make_node("MatMul", ["hidden", "projection"], ["projected"]),
        helper.make_node("Add", ["projected", "projection_bias"], ["last_hidden_state"]),
    ]
    initializers += [tensor("projection", PROJECTION), tensor("projection_bias", PROJECTION_BIAS)]
    inputs = [int64_input("input_ids", "batch"), int64_input("attention_mask", "batch")]
    outputs = [
        helper.make_tensor_value_info(
            "last_hidden_state", TensorProto.FLOAT, ["batch", "sequence", DIM]
        )
    ]
    # `attention_mask` is declared but unused, as in many token-level exports; keep it
    # required so the refusal is about the output and nothing else.
    return model("token_level", nodes, inputs, outputs, initializers)


# ── references ──────────────────────────────────────────────────────────────────────

# Every case is embedded alone, as the backend runs it: one text per `run`.
CASES = [
    ("hebrew", "בראשית ברא אלהים את השמים ואת הארץ", 32),
    ("english", "The quick brown fox jumps over the lazy dog.", 32),
    ("passage_prefix", "[PASSAGE] בראשית ברא אלהים", 32),
    ("query_prefix", "[QUERY] the quick fox", 32),
    # A special token's spelling inside a book is matched as that token, exactly as
    # the prefixes are — the consequence the backend documents.
    ("literal_cls_inside_text", "the [CLS] fox and the [SEP] dog", 32),
    ("literal_prefix_mid_text", "משה [QUERY] ישראל", 32),
    # Matching is on the raw text and case-sensitive: this is not the special token.
    ("lowercase_prefix_is_text", "[query] the fox", 32),
    ("unknown_words_split", "zebra xylophone שמואל", 32),
    ("empty", "", 32),
    ("whitespace_only", "   ", 32),
    ("niqqud_is_stripped", "בְּרֵאשִׁית", 32),
    # Longer than every cap below: truncated on the right, the specials kept.
    ("truncated_to_8", "the quick brown fox jumps over the lazy dog and the fox", 8),
    ("truncated_to_12_with_prefix", "[PASSAGE] " + " ".join(["תורה"] * 30), 12),
    ("exactly_at_cap_10", "a b c d e f g h", 10),
]


def reference_cases(tokenizer_json: str, graph: bytes) -> list[dict]:
    session = onnxruntime.InferenceSession(graph, providers=["CPUExecutionProvider"])
    out = []
    for name, text, cap in CASES:
        tokenizer = Tokenizer.from_str(tokenizer_json)
        tokenizer.no_padding()
        tokenizer.enable_truncation(max_length=cap, strategy="longest_first", direction="right")
        ids = tokenizer.encode(text, add_special_tokens=True).ids
        feed = {
            "input_ids": np.array([ids], dtype=np.int64),
            "attention_mask": np.ones((1, len(ids)), dtype=np.int64),
        }
        (vector,) = session.run(["sentence_embedding"], feed)
        out.append(
            {
                "name": name,
                "text": text,
                "max_tokens": cap,
                "token_ids": ids,
                "vector": [float(x) for x in vector[0]],
            }
        )
    return out


# ── output ──────────────────────────────────────────────────────────────────────────


def generate() -> dict[str, bytes]:
    tokenizer = build_tokenizer()
    tokenizer_json = tokenizer.to_str(pretty=True) + "\n"

    graphs = {
        "dynamic.onnx": sentence_graph("dynamic", "batch"),
        "static_batch.onnx": sentence_graph("static_batch", 1),
        "token_level.onnx": token_level_graph(),
        "extra_input.onnx": sentence_graph("extra_input", "batch", extra_input=True),
        "token_types.onnx": sentence_graph("token_types", "batch", token_types=True),
    }
    files = {name: graph.SerializeToString() for name, graph in graphs.items()}
    files["tokenizer.json"] = tokenizer_json.encode("utf-8")

    expected = {
        "generator": "tools/make_onnx_fixture.py",
        "graph": "dynamic.onnx",
        "dim": DIM,
        "max_positions": MAX_POSITIONS,
        "vocab_size": len(VOCAB),
        "special_token_ids": {token: VOCAB.index(token) for token in SPECIALS},
        "onnx_version": onnx.__version__,
        "tokenizers_version": tokenizers.__version__,
        "onnxruntime_version": onnxruntime.__version__,
        "cases": reference_cases(tokenizer_json, files["dynamic.onnx"]),
    }
    files["expected.json"] = (
        json.dumps(expected, ensure_ascii=False, indent=1) + "\n"
    ).encode("utf-8")
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--check", action="store_true", help="compare, do not write")
    args = parser.parse_args()

    files = generate()
    if args.check:
        stale = [
            name
            for name, content in files.items()
            if not (OUT_DIR / name).is_file() or (OUT_DIR / name).read_bytes() != content
        ]
        if stale:
            print(f"out of date: {', '.join(stale)}", file=sys.stderr)
            return 1
        print(f"{len(files)} files up to date in {OUT_DIR}")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, content in sorted(files.items()):
        (OUT_DIR / name).write_bytes(content)
        print(f"{len(content):>7} bytes  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
