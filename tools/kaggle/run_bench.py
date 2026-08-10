"""Measure vectors per charged session second, using the binary already compiled.

The compile happened once, in `otzaria-embed-bench-t4`, and this attaches that
kernel's output rather than paying nvcc again — which is the whole reason the build
publishes a binary instead of just a number.

Inputs are *located* rather than assumed. The first attempt hard-coded the path a
`--dir-mode zip` upload does not produce, and three configurations failed in a tenth
of a second each; a `find` costs nothing and cannot be wrong about it.

Written as functions so `verdict` can be tested without a GPU. It is the part that
decides whether an hour of quota measured anything, and it was wrong: see below.
"""

import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

OUT = Path("/kaggle/working")
INPUT = Path("/kaggle/input")

CONFIGS = [
    {"name": "n_ctx512-batch32", "n_ctx": 512, "batch": 32, "sequences": 32},
    {"name": "n_ctx2048-batch64", "n_ctx": 2048, "batch": 64, "sequences": 64},
    {"name": "n_ctx4096-batch128", "n_ctx": 4096, "batch": 128, "sequences": 128},
]


def run(cmd, **kw):
    print(f"\n$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, **kw)


def verdict(completed, elapsed):
    """What one configuration measured, and whether it measured anything at all.

    `returncode == 0` is not the bar. `build` prints `Vectors: N` from its own report,
    after it has packed and verified the artifact, so that line is the only evidence the
    session did the work — and a run without one used to be recorded as `ok` with a null
    rate. Three such runs then left the kernel finishing successfully with no measurement
    in it, which is exactly the failure this tool exists to make loud.
    """
    vectors = next(
        (
            int(line.split()[-1])
            for line in completed.stdout.splitlines()
            if line.startswith("Vectors:")
        ),
        None,
    )
    seconds = round(elapsed, 2)
    if completed.returncode != 0:
        return {"ok": False, "seconds": seconds, "reason": f"exit {completed.returncode}"}
    if not vectors:
        return {"ok": False, "seconds": seconds, "reason": "no vector count was printed"}
    return {
        "ok": True,
        "seconds": seconds,
        "vectors": vectors,
        "vectors_per_second": round(vectors / elapsed, 2) if elapsed > 0 else None,
    }


def locate():
    """The binary, the model and the corpus, wherever the attachment put them."""
    # Attached inputs are two levels deeper than the docs suggest:
    # /kaggle/input/{notebooks,datasets}/<owner>/<slug>/...
    binary = next(INPUT.glob("**/otzaria-semantic-search"), None)
    model = next(INPUT.glob("**/*.gguf"), None)
    if binary is None or model is None:
        raise SystemExit(f"binary={binary} model={model}: an input is missing")
    os.chmod(binary, 0o755)

    # A directory uploaded with `--dir-mode zip` arrives as an archive, and Kaggle does
    # not always unpack it. Unpack to /kaggle/temp, which is not part of the output.
    corpus = next((p.parent for p in INPUT.glob("**/corpus-lines.jsonl")), None)
    if corpus is None:
        archive = next(INPUT.glob("**/*.zip"), None)
        if archive is None:
            raise SystemExit("no corpus and no archive to unpack it from")
        unpacked = Path("/kaggle/temp/corpus")
        unpacked.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(unpacked)
        corpus = next((p.parent for p in unpacked.glob("**/corpus-lines.jsonl")), None)
    if corpus is None:
        raise SystemExit("the corpus is not in the archive either")
    return binary, model, corpus


def run_configuration(binary, model, corpus, config, runner=subprocess.run):
    """Build the whole 24k corpus once, under one pair of knob settings."""
    artifact = OUT / f"artifact-{config['name']}"
    run(f"rm -rf {artifact}")

    env = dict(os.environ)
    env.update({
        "OTZARIA_LLAMA_GPU_LAYERS": "99",
        "OTZARIA_LLAMA_CONTEXTS": "1",
        "OTZARIA_LLAMA_N_CTX": str(config["n_ctx"]),
        "OTZARIA_LLAMA_MAX_SEQUENCES": str(config["sequences"]),
        # One process per card. Splitting a 0.6B model across two by layer spends
        # more on the interconnect than it saves; production runs two processes,
        # and this measures one of them.
        "CUDA_VISIBLE_DEVICES": "0",
    })

    started = time.time()
    completed = runner(
        f"{binary} build"
        f" --corpus-identity {corpus}/corpus-identity.json"
        f" --corpus-lines {corpus}/corpus-lines.jsonl"
        f" --model {corpus}/model.json"
        f" --model-file {model}"
        f" --chunking {corpus}/chunking.json"
        f" --out {artifact}"
        f" --batch {config['batch']}",
        shell=True, env=env, capture_output=True, text=True,
    )
    elapsed = time.time() - started

    print(f"\n=== {config['name']} ===", flush=True)
    print(completed.stdout[-1500:], flush=True)
    result = {**config, **verdict(completed, elapsed)}
    if not result["ok"]:
        print(completed.stderr[-3000:], flush=True)
        print(f"--> FAILED: {result['reason']}", flush=True)
    else:
        print(f"--> {result['vectors_per_second']} vectors/s", flush=True)
    run(f"rm -rf {artifact}")
    return result


def main():
    run(f"find {INPUT} -maxdepth 3 | head -40")
    run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    binary, model, corpus = locate()
    print(f"\nbinary {binary}\nmodel  {model}\ncorpus {corpus}", flush=True)

    gpu = subprocess.run(
        "nvidia-smi --query-gpu=name --format=csv,noheader",
        shell=True, capture_output=True, text=True,
    ).stdout.strip()

    results = [run_configuration(binary, model, corpus, config) for config in CONFIGS]
    (OUT / "bench-results.json").write_text(
        json.dumps({"gpu": gpu, "runs": results}, indent=2)
    )
    print(json.dumps(results, indent=2), flush=True)

    # Written first, then refused: Kaggle keeps the output of a kernel that fails, so
    # exiting non-zero costs nothing and is the only thing that gets read. A benchmark
    # that reports three failures and still succeeds is one somebody has to read out of
    # the logs by eye — which is how a stale binary produced three usage screens and
    # looked like a completed run.
    failed = [f"{run['name']} ({run['reason']})" for run in results if not run["ok"]]
    if failed:
        raise SystemExit(f"configuration(s) failed: {', '.join(failed)}")


if __name__ == "__main__":
    sys.exit(main())
