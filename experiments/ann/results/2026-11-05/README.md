# Retained ANN observations

These files preserve all three successful fixtures from
[run 42083690818](https://github.com/Otzaria/otzaria-semantic-search/actions/runs/42083690818).
The functional head, generated merge ref, executable hash, original GitHub ZIP
digests and hashes of the retained files are in [manifest.json](manifest.json).
The analysis and measurement limits are in [RESULTS.md](../../RESULTS.md).

| Fixture | Compact summary | Complete raw archive |
| --- | --- | --- |
| Clustered int8 | [i8-clustered-summary.json](i8-clustered-summary.json) | [i8-clustered-raw.json.gz](i8-clustered-raw.json.gz) |
| Uniform int8 | [i8-uniform-summary.json](i8-uniform-summary.json) | [i8-uniform-raw.json.gz](i8-uniform-raw.json.gz) |
| Clustered rotated codec | [rot192-clustered-summary.json](rot192-clustered-summary.json) | [rot192-clustered-raw.json.gz](rot192-clustered-raw.json.gz) |

Each summary contains `default` and `advised` reports. It omits per-query recall
arrays and individual empty-scope timing samples; all other report values are
retained. Those omitted values, every query's keys/scores, all warm timing samples
and each original report remain in the raw archive.

Each gzip contains a UTF-8 JSON object with this layout:

- `reports.default` and `reports.advised`: the complete original top-level reports.
- `default`: the 36 original normal-run JSON documents, keyed by their filenames.
- `advised`: the 13 original advisory-run JSON documents, keyed by their filenames.
- `artifact`: GitHub artifact metadata, including the verified original ZIP digest.
- `original_json_sha256`: hashes of the exact original JSON members before bundling.

Bundling changes JSON whitespace/key order, while preserving parsed values. Gzip
headers have timestamp zero and no embedded filename. The manifest records both
the compressed and uncompressed retained-file hashes. Independent QA compared all
153 original JSON members against a separately downloaded set of artifact ZIPs;
all parsed values and original-member hashes matched. It also recomputed recall,
percentiles and ordered-key hashes directly from the retained archives.

The [independent verifier](../../verify_evidence.py) repeats those arithmetic and
contract checks without importing benchmark modules. For example, from the
repository root:

```sh
python3 experiments/ann/verify_evidence.py \
  experiments/ann/results/2026-11-05/i8-clustered-raw.json.gz \
  --sha256 f9e248d7781eb84c0cf01a5e14eb354014524794910c1097d0cc717882272ee5 \
  > /tmp/ann-i8-clustered-verified.json
```

Use each case's `raw.sha256` from the manifest for the other retained archives.

Read a verified archive using only Python's standard library, from the repository
root:

```sh
python3 - <<'PY'
from pathlib import Path
import gzip, hashlib, json

folder = Path("experiments/ann/results/2026-11-05")
manifest = json.loads((folder / "manifest.json").read_text())
case = manifest["cases"][0]
compressed = (folder / case["raw"]["file"]).read_bytes()
assert hashlib.sha256(compressed).hexdigest() == case["raw"]["sha256"]
payload = gzip.decompress(compressed)
assert hashlib.sha256(payload).hexdigest() == case["raw"]["uncompressed_sha256"]
evidence = json.loads(payload)
reference = evidence["default"]["all-oxv-t1-ef128.json"]
candidate = evidence["default"]["all-usearch-load-t1-ef128.json"]
recalls = [
    len(set(oracle["keys"]) & set(found["keys"])) / len(oracle["keys"])
    for oracle, found in zip(reference["queries"], candidate["queries"])
]
print(case["name"], "mean Recall@10 vs OXV:", sum(recalls) / len(recalls))
PY
```

The example uses the unfiltered 50-query reference, where all oracle lists have
ten elements. General recall accounting must exclude empty oracles and retain
missing neighbors as misses, as described in the experiment README.
