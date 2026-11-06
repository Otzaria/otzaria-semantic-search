#!/usr/bin/env python3
"""Compare independent ort_parity processes; timing is reported, never a CI gate."""

import argparse
import json
import math
from pathlib import Path


def compare(before, after, max_abs_delta=1e-6):
    if not math.isfinite(max_abs_delta) or max_abs_delta < 0:
        raise ValueError("max_abs_delta must be finite and nonnegative")
    for field in ("schema", "backend", "max_tokens", "texts", "tokens", "neighbours_at_5"):
        if before[field] != after[field]:
            raise ValueError(f"{field} changed")
    if before["schema"] != 1 or not before["texts"]:
        raise ValueError("expected a nonempty schema-1 probe")
    count = len(before["texts"])
    if len(before["tokens"]) != count or len(before["neighbours_at_5"]) != count:
        raise ValueError("token/neighbor row count does not match the corpus")
    report = {"cases": count, "tokens_identical": True, "ordered_top5_identical": True}
    for field in ("raw_vectors", "normalized_vectors"):
        old, new = before[field], after[field]
        if len(old) != count or len(new) != count:
            raise ValueError(f"{field}: row count does not match the corpus")
        width = len(old[0])
        if width == 0:
            raise ValueError(f"{field}: empty vector")
        delta = 0.0
        worst_cosine = 1.0
        for index, (left, right) in enumerate(zip(old, new)):
            if len(left) != width or len(right) != width:
                raise ValueError(f"{field}[{index}]: vector width changed")
            if not all(math.isfinite(value) for value in left + right):
                raise ValueError(f"{field}[{index}]: non-finite component")
            delta = max(delta, max(abs(a - b) for a, b in zip(left, right)))
            left_norm = math.sqrt(sum(value * value for value in left))
            right_norm = math.sqrt(sum(value * value for value in right))
            if left_norm <= 1e-12 or right_norm <= 1e-12:
                raise ValueError(f"{field}[{index}]: zero vector")
            cosine = sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
            worst_cosine = min(worst_cosine, max(-1.0, min(1.0, cosine)))
        if delta > max_abs_delta:
            raise ValueError(f"{field}: maximum absolute delta {delta} exceeds {max_abs_delta}")
        report[field] = {
            "dimensions": width,
            "exactly_identical": old == new,
            "max_abs_delta": delta,
            "worst_cosine": worst_cosine,
        }
    for field in ("iterations", "threads", "sessions", "native_allocations_measured"):
        if before["measurement"][field] != after["measurement"][field]:
            raise ValueError(f"measurement protocol changed: {field}")
    report["before_measurement"] = before["measurement"]
    report["after_measurement"] = after["measurement"]
    report["timing_is_a_gate"] = False
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--max-abs-delta", type=float, default=1e-6)
    args = parser.parse_args()
    before = json.loads(args.before.read_text())
    after = json.loads(args.after.read_text())
    print(json.dumps(compare(before, after, args.max_abs_delta), indent=2))


if __name__ == "__main__":
    main()
