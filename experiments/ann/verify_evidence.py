#!/usr/bin/env python3
"""Independently verify ANN artifact arithmetic without importing the experiment.

Usage: python3 experiments/ann/verify_evidence.py ARTIFACT.zip --sha256 EXPECTED
Also accepts the retained *-raw.json.gz bundles with their manifest SHA-256.
Files are read in place, never extracted. This validates the published 20k x 256
measurement protocol. The expected gzip digest is cases[].raw.sha256 in the
retained manifest; a ZIP digest comes from GitHub artifact metadata. No benchmark
modules are imported. Print a compact review to stdout.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import zipfile


class JsonBundle:
    """Expose retained parsed JSON through the same read interface as the ZIP."""
    def __init__(self, data):
        assert data["schema"] == 1
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def namelist(self):
        return list(self.data["original_json_sha256"])

    def read(self, name):
        if "/" not in name:
            case = "advised" if "-advised" in name else "default"
            value = self.data["reports"][case]
        else:
            folder, filename = name.split("/", 1)
            case = "advised" if "-advised" in folder else "default"
            value = self.data[case][filename]
        return json.dumps(value, allow_nan=False).encode()


def assert_close(actual, expected, context):
    if expected is None:
        assert actual is None, (context, actual, expected)
    else:
        assert math.isfinite(actual) and abs(actual - expected) <= 1e-12, (context, actual, expected)


def check(path, digest):
    actual_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual_digest == digest, ("archive digest", actual_digest, digest)
    verified = {"archive": path.name, "sha256": actual_digest, "reports": []}
    container = (JsonBundle(json.loads(gzip.decompress(path.read_bytes())))
                 if path.name.endswith(".json.gz") else zipfile.ZipFile(path))
    with container as archive:
        names = archive.namelist()
        top = sorted(name for name in names if "/" not in name and name.endswith(".json"))
        assert len(top) == 2, ("expected uncontrolled and advisory reports", top)
        for name in top:
            report = json.loads(archive.read(name))
            assert report["schema"] == 1
            config = report["build"]["config"]
            count, queries, k = (config[field] for field in ("vectors", "queries", "top_k"))
            assert count == 20_000 and queries == 50 and k == 10
            assert config["dim"] == 256
            assert report["build"]["usearch_version"] == "2.26.6"
            assert report["build"]["usearch_features"] == []
            raw_prefix = name[:-5] + "-raw/"
            raw = {}
            for measured in report["measurements"]:
                label = f"{measured['filter']}-{measured['backend']}-t{measured['threads']}-ef{measured['expansion_search']}"
                assert label not in raw
                raw[label] = json.loads(archive.read(raw_prefix + label + ".json"))

            recall_summaries = 0
            query_comparisons = 0
            rows = []
            per_book = math.ceil(count / 100)
            for measured in report["measurements"]:
                label = f"{measured['filter']}-{measured['backend']}-t{measured['threads']}-ef{measured['expansion_search']}"
                source = raw[label]
                assert source["config"] == config
                assert len(source["queries"]) == queries
                filter_name = measured["filter"]

                def admitted(key):
                    book = ((key - 1) // per_book) % 100
                    return (filter_name == "all" or
                            (filter_name == "5pct" and book % 20 == 0) or
                            (filter_name == "1pct" and book == 0))

                eligible = sum(admitted(key) for key in range(1, count + 1))
                assert eligible == measured["eligible_vectors"]
                for result in source["queries"]:
                    keys = result["keys"]
                    assert len(keys) == len(result["scores"])
                    assert len(keys) == len(set(keys)) <= min(k, eligible)
                    assert all(1 <= key <= count and admitted(key) for key in keys)
                    assert all(math.isfinite(score) for score in result["scores"])
                    if measured["backend"] in ("raw", "decoded", "oxv"):
                        assert len(keys) == min(k, eligible)

                ordered_hash = hashlib.sha256(json.dumps(
                    [query["keys"] for query in source["queries"]], separators=(",", ":")
                ).encode()).hexdigest()
                assert ordered_hash == measured["ordered_query_keys_sha256"]
                samples = sorted(source["samples_us"])
                assert len(samples) == queries * measured["repeats"]
                assert all(math.isfinite(value) and value >= 0 for value in samples)
                assert measured["warm_us"]["n"] == len(samples)
                for field, expected in (("min", samples[0]), ("max", samples[-1])):
                    assert_close(measured["warm_us"][field], expected, (label, field))
                for field, q in (("p50", .5), ("p95", .95), ("p99", .99)):
                    expected = samples[max(0, math.ceil(len(samples) * q) - 1)]
                    assert_close(measured["warm_us"][field], expected, (label, field))
                for field in ("cache_advice", "memory_before_open", "memory_after_open", "memory_after_first", "memory_after_warm"):
                    assert measured[field] == source[field]
                assert measured["cache_advice"]["eviction_verified"] is False
                for reference_name in ("raw", "decoded", "oxv"):
                    field = "recall_vs_" + reference_name
                    if field not in measured:
                        continue
                    reference = raw[f"{filter_name}-{reference_name}-t1-ef128"]
                    values = []
                    for expected, result in zip(reference["queries"], source["queries"]):
                        if expected["keys"]:
                            values.append(len(set(expected["keys"]) & set(result["keys"])) / len(expected["keys"]))
                        else:
                            assert not result["keys"]
                    saved = measured[field]
                    assert len(saved["per_query"]) == len(values) == saved["queries"]
                    for got, expected in zip(saved["per_query"], values):
                        assert_close(got, expected, (label, field, "per query"))
                    assert_close(saved["mean"], sum(values)/len(values) if values else None, (label, field, "mean"))
                    assert_close(saved["min"], min(values) if values else None, (label, field, "minimum"))
                    assert saved["full_recall_queries"] == sum(value == 1.0 for value in values)
                    recall_summaries += 1
                    query_comparisons += len(values)
                if measured["backend"] == "oxv" and measured["threads"] != 1:
                    assert source["queries"] == raw[f"{filter_name}-oxv-t1-ef128"]["queries"]

                if ((measured["backend"] == "oxv" and measured["threads"] in (1, 4)) or
                    (measured["backend"] in ("usearch-load", "usearch-view") and measured["expansion_search"] == 128)):
                    rows.append({
                        "filter": filter_name, "backend": measured["backend"], "threads": measured["threads"],
                        "effective_threads": measured["oxv_effective_threads"],
                        "p50_us": measured["warm_us"]["p50"], "p95_us": measured["warm_us"]["p95"],
                        "open_us": measured["open_us"], "first_query_us": measured["first_query_us"],
                        "recall_vs_oxv": measured["recall_vs_oxv"]["mean"],
                        "recall_vs_decoded": measured["recall_vs_decoded"]["mean"],
                        "rss_after_warm": measured["memory_after_warm"]["rss_bytes"],
                        "native_memory_estimate": measured["ann_memory_usage_bytes"],
                        "cache_advice": measured["cache_advice"],
                    })

            lifecycle = report["lifecycle"]
            if lifecycle is not None:
                assert lifecycle["status"] == "passed"
                assert lifecycle["deleted"] == 400 and lifecycle["added"] == 1200
                assert lifecycle["live_after"] == 20800
                assert lifecycle["native_ann_abort_supported"] is False
                for flag in ("exact_oxv_compaction_score_and_order_parity", "retained_oxv_snapshot_parity",
                             "ann_exact_compaction_and_restore_parity", "ann_all_key_membership_after_load_and_view",
                             "pre_cancelled_oxv", "pre_cancelled_ann_wrapper", "cancelled_result_discarded",
                             "corrupt_ann_refused_exact_fallback_available"):
                    assert lifecycle[flag] is True, flag
            verified["reports"].append({
                "name": name, "config": config, "measurement_cases": len(raw),
                "recall_summaries_recomputed": recall_summaries,
                "per_query_recalls_recomputed": query_comparisons,
                "warm_percentiles_and_order_hashes_recomputed": True,
                "oxv_disk_bytes": report["build"]["oxv_bytes"],
                "ann_extra_disk_bytes": report["build"]["ann_bytes"],
                "oxv_plus_ann_disk_bytes": report["build"]["oxv_bytes"] + report["build"]["ann_bytes"],
                "binary_sha256": report["environment"]["measured_binary_sha256"],
                "lifecycle_passed": lifecycle is not None,
                "selected_rows": rows,
            })
    return verified


if __name__ == "__main__":
    if not __debug__:
        raise SystemExit("Verification requires assertions; run Python without -O/PYTHONOPTIMIZE")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.archive, args.sha256), indent=2, allow_nan=False))
