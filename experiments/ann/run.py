#!/usr/bin/env python3
"""Run each backend in a fresh child; aggregate recall against the real OXV scan.

No third-party Python packages. Build the Rust executable once before running.
The driver is deliberately sequential: running competitors concurrently would
contaminate latency and peak-RSS measurements.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def execute(binary: Path, arguments: list[str], raw_dir: Path | None, label: str, metadata: dict | None = None) -> dict:
    print(label, file=sys.stderr, flush=True)
    command = [str(binary), *arguments]
    process = subprocess.run(command, check=True, capture_output=True, text=True, timeout=1800)
    if process.stderr:
        print(process.stderr, file=sys.stderr, end="")
    value = json.loads(process.stdout)
    value.update(metadata or {})
    if raw_dir:
        with (raw_dir / f"{label}.json").open("x", encoding="utf-8") as out:
            json.dump(value, out, indent=2, allow_nan=False)
            out.write("\n")
    return value


def recall(expected: list[int], actual: list[int]) -> float | None:
    """Use the oracle's actual count (min(k, eligible)); empty is not applicable.

    Missing ANN results stay misses. Duplicates cannot inflate recall, and no
    exact hits are inserted to rescue a low-recall approximate result.
    """
    if not expected:
        if actual:
            raise ValueError("nonempty results for an empty oracle")
        return None
    return len(set(expected) & set(actual)) / len(expected)


def recall_summary(reference: dict, candidate: dict) -> dict:
    if len(reference["queries"]) != len(candidate["queries"]):
        raise ValueError("different query counts")
    values = []
    for oracle, result in zip(reference["queries"], candidate["queries"]):
        value = recall(oracle["keys"], result["keys"])
        if value is not None:
            values.append(value)
    return {
        "queries": len(values),
        "mean": sum(values) / len(values) if values else None,
        "min": min(values) if values else None,
        "full_recall_queries": sum(value == 1.0 for value in values),
        "per_query": values,
    }


def read_if_present(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_advice(dataset: Path, backend: str, policy: str) -> dict:
    """Evict only our own completed fixture file, and never claim guaranteed cold."""
    advice = {"policy": policy, "eviction_verified": False}
    if policy == "uncontrolled":
        advice["result"] = "not_requested"
        return advice
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        advice["result"] = "unsupported"
        return advice
    filename = {"raw": "raw.oxv", "decoded": "decoded.oxv", "oxv": "encoded.oxv", "usearch-load": "index.usearch", "usearch-view": "index.usearch"}[backend]
    advice["fixture_file"] = filename
    try:
        fd = os.open(dataset / filename, os.O_RDONLY)
        try:
            os.fsync(fd)  # Only clean pages are candidates for advisory eviction.
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        advice["result"] = "accepted_advisory_no_guarantee"
    except OSError as error:
        advice["result"] = "failed"
        advice["errno"] = error.errno
        advice["error"] = str(error)
    return advice


def environment(binary: Path) -> dict:
    cpu = read_if_present("/proc/cpuinfo")
    model = next((line.split(":", 1)[1].strip() for line in (cpu or "").splitlines() if line.startswith("model name")), None)
    files = [ROOT / "Cargo.toml", ROOT / "Cargo.lock", ROOT / "run.py", *sorted((ROOT / "src").glob("*.rs"))]
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files if path.is_file()}
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": model,
        "logical_cpus": os.cpu_count(),
        "affinity_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "cgroup_cpu_max": read_if_present("/sys/fs/cgroup/cpu.max"),
        "cgroup_memory_max": read_if_present("/sys/fs/cgroup/memory.max"),
        "loadavg": read_if_present("/proc/loadavg"),
        "base_git_revision": revision,
        "worktree_dirty": dirty,
        "experiment_source_sha256": hashes,
        "measured_binary_sha256": sha256_file(binary),
        "measured_binary_bytes": binary.stat().st_size,
    }


def compact_measurement(measured: dict) -> dict:
    result = {key: value for key, value in measured.items() if key not in {"queries", "samples_us", "first_query_keys", "config"}}
    result["ordered_query_keys_sha256"] = hashlib.sha256(json.dumps([query["keys"] for query in measured["queries"]], separators=(",", ":")).encode()).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/otzaria-ann-experiment")
    parser.add_argument("--dataset", type=Path, required=True, help="new directory to create, or explicit --reuse-dataset")
    parser.add_argument("--output", type=Path, required=True, help="new report file; never overwritten")
    parser.add_argument("--raw-dir", type=Path, help="new directory for full query keys/scores/timing samples")
    parser.add_argument("--reuse-dataset", action="store_true")
    parser.add_argument("--vectors", type=int, default=20_000)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--codec", choices=["f32", "i8-sym-vec", "i8-sym-dim", "rot192-i8i4"], default="i8-sym-vec")
    parser.add_argument("--distribution", choices=["uniform", "clustered"], default="clustered")
    parser.add_argument("--threads", default="1,4,8")
    parser.add_argument("--expansions", default="32,128,512")
    parser.add_argument("--filters", default="all,5pct,1pct")
    parser.add_argument("--cache-policy", choices=["uncontrolled", "advisory-cold"], default="uncontrolled")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--skip-lifecycle", action="store_true", help="only for an explicitly reused dataset whose lifecycle already ran")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if args.skip_lifecycle and not args.reuse_dataset:
        parser.error("--skip-lifecycle requires --reuse-dataset")
    if not args.binary.is_file():
        parser.error("build the Rust release executable first, or supply --binary")
    threads = sorted(set(int(value) for value in args.threads.split(",")))
    expansions = sorted(set(int(value) for value in args.expansions.split(",")))
    if min(threads + expansions + [args.repeats]) <= 0:
        parser.error("threads, expansions, and repeats must be positive")
    filters = args.filters.split(",")
    if len(set(filters)) != len(filters) or not set(filters) <= {"all", "5pct", "1pct", "empty"}:
        parser.error("filters must be unique choices of all,5pct,1pct,empty")
    if args.raw_dir:
        args.raw_dir.mkdir()  # Refuse to overwrite earlier evidence.
    args.binary = args.binary.resolve()
    args.dataset = args.dataset.resolve()
    env = environment(args.binary)
    if args.reuse_dataset:
        build = json.loads((args.dataset / "build.json").read_text(encoding="utf-8"))
    else:
        build = execute(args.binary, ["build", "--dir", str(args.dataset), "--vectors", str(args.vectors), "--dim", str(args.dim), "--queries", str(args.queries), "--top-k", str(args.top_k), "--codec", args.codec, "--distribution", args.distribution], args.raw_dir, "build")
    measurements = []
    for filter_name in filters:
        references = {}
        plans = [("raw", 1, 128), ("decoded", 1, 128), ("oxv", 1, 128)]
        plans += [("oxv", count, 128) for count in threads if count != 1]
        plans += [(backend, 1, expansion) for backend in ["usearch-load", "usearch-view"] for expansion in expansions]
        for backend, count, expansion in plans:
            label = f"{filter_name}-{backend}-t{count}-ef{expansion}"
            advice = cache_advice(args.dataset, backend, args.cache_policy)
            value = execute(args.binary, ["measure", "--dir", str(args.dataset), "--backend", backend, "--filter", filter_name, "--threads", str(count), "--expansion", str(expansion), "--repeats", str(args.repeats)], args.raw_dir, label, {"cache_advice": advice})
            if backend in {"raw", "decoded", "oxv"} and count == 1:
                references[backend] = value
            summary = compact_measurement(value)
            for ref_name, reference in references.items():
                summary[f"recall_vs_{ref_name}"] = recall_summary(reference, value)
            if backend == "oxv" and count != 1:
                if value["queries"] != references["oxv"]["queries"]:
                    raise ValueError("OXV thread counts changed keys, scores or order")
            measurements.append(summary)
    lifecycle = None if args.skip_lifecycle else execute(args.binary, ["lifecycle", "--dir", str(args.dataset)], args.raw_dir, "lifecycle")
    thread_probe = execute(args.binary, ["threads", "--samples", "100"], args.raw_dir, "threads")
    report = {
        "schema": 1,
        "purpose": "synthetic ANN feasibility experiment; not production qualification",
        "environment": env,
        "environment_end": {"loadavg": read_if_present("/proc/loadavg")},
        "build": build,
        "measurements": measurements,
        "lifecycle": lifecycle,
        "thread_launch_proxy": thread_probe,
        "limitations": [
            "Fresh process/open is measured; per-file advisory cache policy is recorded and eviction is never guaranteed.",
            "Synthetic independent queries are not a real embedding corpus or semantic-quality test.",
            "USearch is pinned 2.26.6, scalar F32/IP over OXV-decoded vectors; graph memory is extra disk/RAM.",
            "OXV codec scan includes quantized-query arithmetic; decoded-f32 separates its metric difference from ANN recall.",
            "Production alias/window penalties, foreign links, moved-text resolution, atomic ANN generation publication, and bounded native cancellation are not implemented.",
            "One Linux host cannot qualify mobile, Windows, macOS, old CPUs, or full-library scale.",
            "Empty-scope timings are a launch/join proxy, not a CPU profile attributing OXV time to spawning.",
        ],
    }
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(report, out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
