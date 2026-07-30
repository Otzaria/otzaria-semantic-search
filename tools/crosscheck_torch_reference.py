#!/usr/bin/env python3
"""Reference B: verify tests/data/golden_vectors.json against an INDEPENDENT forward pass.

Reference A (tools/generate_golden_vectors.py) uses llama.cpp via llama-cpp-python.
The Rust backend chosen for this project is also llama.cpp (via llama-cpp-2), so
"Rust matches Reference A" is close to tautological for the forward pass itself.

This script closes that gap. It loads the SAME GGUF file through HuggingFace
transformers, which dequantizes the Q4_K_M tensors to fp32 and runs its own PyTorch
Qwen3 implementation -- a completely separate codebase from ggml. Agreement between the
two is real evidence that the golden vectors describe the model rather than one
library's quirks.

It deliberately does NOT regenerate the goldens. It reads the token_ids straight out of
the committed file, so the comparison isolates the forward pass from tokenization.

Run it in a separate virtualenv from the generator; see tools/README.md.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import struct
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_GOLDEN = os.path.join(REPO_ROOT, "tests", "data", "golden_vectors.json")
DEFAULT_MODEL = os.path.join(REPO_ROOT, "Otzaria-Embedding-V1-Flash-0.6B-Q4_K_M.gguf")


def l2(v):
    return sum(float(x) * float(x) for x in v) ** 0.5


def unit(v):
    n = l2(v)
    return [float(x) / n for x in v]


def cosine(a, b):
    return sum(float(x) * float(y) for x, y in zip(a, b)) / (l2(a) * l2(b))


def decode_b64_f32(s: str, dim: int):
    raw = base64.b64decode(s)
    if len(raw) != dim * 4:
        raise SystemExit(f"expected {dim * 4} bytes, got {len(raw)}")
    return list(struct.unpack("<%df" % dim, raw))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=DEFAULT_GOLDEN)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument(
        "--link-dir",
        default=None,
        help="Scratch directory to symlink the GGUF into. transformers requires the "
        "gguf_file to sit inside a directory it can treat as a model repo, and the "
        "model must not be copied or moved into the repo. Defaults to a temp dir.",
    )
    ap.add_argument("--limit", type=int, default=0, help="Only check the first N records.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    import tempfile

    import torch
    from transformers import AutoModel

    with open(args.golden, encoding="utf-8") as fh:
        golden = json.load(fh)
    dim = golden["header"]["embedding_dim"]
    records = golden["vectors"]
    if args.limit:
        records = records[: args.limit]

    link_dir = args.link_dir or tempfile.mkdtemp(prefix="otzaria-gguf-link-")
    os.makedirs(link_dir, exist_ok=True)
    basename = os.path.basename(args.model)
    link = os.path.join(link_dir, basename)
    if not os.path.exists(link):
        os.symlink(os.path.abspath(args.model), link)

    print(f"loading {basename} through transformers (dequantizing to fp32) ...", file=sys.stderr)
    model = AutoModel.from_pretrained(link_dir, gguf_file=basename, dtype=torch.float32)
    model.eval()
    print(f"loaded {type(model).__name__}, {sum(p.numel() for p in model.parameters()):,} params", file=sys.stderr)

    rows = []
    worst_cos, worst_id = 1.0, None
    worst_abs = 0.0
    norm_rel_max, norm_rel_id = 0.0, None

    with torch.no_grad():
        for rec in records:
            ids = torch.tensor([rec["token_ids"]], dtype=torch.long)
            out = model(input_ids=ids, attention_mask=torch.ones_like(ids))
            # Last-token pooling: the hidden state of the final position, which the
            # goldens' pipeline guarantees is the EOS token.
            h = out.last_hidden_state[0, -1].to(torch.float64).tolist()

            b_unit = unit(h)
            b_norm = l2(h)
            a_unit = decode_b64_f32(rec["embedding_normalized_f32_b64"], dim)
            a_norm = rec["raw_l2_norm"]

            c = cosine(a_unit, b_unit)
            mx = max(abs(x - y) for x, y in zip(a_unit, b_unit))
            nrel = abs(b_norm - a_norm) / a_norm

            if c < worst_cos:
                worst_cos, worst_id = c, rec["id"]
            worst_abs = max(worst_abs, mx)
            if nrel > norm_rel_max:
                norm_rel_max, norm_rel_id = nrel, rec["id"]

            rows.append((rec["id"], rec["token_count"], c, mx, a_norm, b_norm, nrel))
            if args.verbose:
                print(
                    f"  {rec['id']:32s} n={rec['token_count']:4d} cos={c:.8f} "
                    f"max|d|={mx:.2e} normA={a_norm:9.4f} normB={b_norm:9.4f} "
                    f"relnorm={nrel:.2e}",
                    file=sys.stderr,
                )

    print("\n=== Reference A (llama.cpp) vs Reference B (transformers/PyTorch) ===")
    print(f"{'id':34s} {'ntok':>5s} {'cosine':>12s} {'max|dcomp|':>11s} {'relnorm':>9s}")
    for rid, n, c, mx, an, bn, nrel in sorted(rows, key=lambda r: r[2]):
        print(f"{rid:34s} {n:5d} {c:12.8f} {mx:11.2e} {nrel:9.2e}")

    cs = [r[2] for r in rows]
    print(f"\nrecords compared        : {len(rows)}")
    print(f"cosine  min / mean / max: {min(cs):.8f} / {sum(cs)/len(cs):.8f} / {max(cs):.8f}")
    print(f"worst cosine            : {worst_cos:.8f}  ({worst_id})")
    print(f"max |component diff|    : {worst_abs:.3e}")
    print(f"max relative norm diff  : {norm_rel_max:.3e}  ({norm_rel_id})")

    # The relational ordering must survive an independent implementation. If it only held
    # under llama.cpp it would be an artefact of that library rather than a property of
    # the model, and stage 4 would be asserting the wrong thing.
    by_id = {r["id"]: r for r in records}
    needed = ("near_identical_a", "near_identical_b", "unrelated_a", "unrelated_b")
    if not all(k in by_id for k in needed):
        print(
            "\nskipping the ordering check: the golden file does not contain all of "
            f"{needed} (was --limit used?)",
            file=sys.stderr,
        )
        return 0

    with torch.no_grad():

        def bvec(rid):
            ids = torch.tensor([by_id[rid]["token_ids"]], dtype=torch.long)
            o = model(input_ids=ids, attention_mask=torch.ones_like(ids))
            return unit(o.last_hidden_state[0, -1].to(torch.float64).tolist())

        near = cosine(bvec("near_identical_a"), bvec("near_identical_b"))
        unrel = cosine(bvec("unrelated_a"), bvec("unrelated_b"))

    gold = golden["relations"].get("ordering_assertion", {})
    print("\n=== relational ordering under Reference B ===")
    print(f"cos(near_identical)  B={near:.8f}   A={gold.get('cosine_near_identical')}")
    print(f"cos(unrelated)       B={unrel:.8f}   A={gold.get('cosine_unrelated')}")
    print(f"margin               B={near - unrel:.8f}   A={gold.get('margin_measured')}")

    margin_min = gold.get("margin_min", 0.0)
    ordering_ok = (near - unrel) >= margin_min
    print(f"margin >= margin_min ({margin_min}) : {ordering_ok}")
    if not ordering_ok:
        print(
            "FAIL: the independent implementation does not reproduce the similarity "
            "ordering the goldens assert.",
            file=sys.stderr,
        )
    return 0 if ordering_ok else 1


if __name__ == "__main__":
    sys.exit(main())
