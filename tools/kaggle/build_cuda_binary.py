"""Build the embedding builder with CUDA offload, then measure it on this GPU.

Build and benchmark are one session on purpose. The compile has to happen where
`nvcc` is, and Kaggle's CPU image does not ship it — a CPU session reports
`nvcc: not found` — so there is no free session to hide the compile in. Since the
quota is being spent anyway, the same session goes on to produce the number that
decides the hardware, and publishes the binary so that no later session pays for
the compile again.

The number is not batch latency. It is

    effective_rate = accepted_vectors / session_seconds

over a real `build` — tokenization, inference, hashing, packing, verification — on
real Otzaria lines, and the artifact is verified before any rate is reported. A run
that produced fast garbage reports nothing.

Two knobs are swept, because they pull against each other. `n_ctx` is the token
budget for one decode call, and llama.cpp sizes the compute buffer from
`n_ubatch = n_ctx / 2` — including an `n_ubatch x n_vocab` logits tensor this
backend never reads. Raising it feeds the GPU more work per call and costs roughly
600 MB of VRAM for a tensor nobody looks at. On a 16 GB Mac that trade measured
flat (11.86 -> 12.03 vectors/s for 3.8x the memory); a T4 has dedicated headroom,
so it is measured here rather than assumed.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = "https://github.com/Otzaria/otzaria-semantic-search.git"
REV = os.environ.get("OTZARIA_REV", "main")
SRC = "/kaggle/temp/src"
OUT = Path("/kaggle/working")
DATA = "/kaggle/input/otzaria-embedding-bench"
MODEL = f"{DATA}/Otzaria-Embedding-V1-Flash-0.6B-Q4_K_M.gguf"
CORPUS = f"{DATA}/bench24k"

CONFIGS = [
    {"name": "n_ctx512-batch32", "n_ctx": 512, "batch": 32, "sequences": 32},
    {"name": "n_ctx2048-batch64", "n_ctx": 2048, "batch": 64, "sequences": 64},
    {"name": "n_ctx4096-batch128", "n_ctx": 4096, "batch": 128, "sequences": 128},
]


def run(cmd, check=True, **kw):
    print(f"\n$ {cmd}", flush=True)
    started = time.time()
    result = subprocess.run(cmd, shell=True, **kw)
    print(f"[{time.time() - started:.1f}s, exit {result.returncode}]", flush=True)
    if check and result.returncode != 0:
        sys.exit(result.returncode)
    return result


run("nvidia-smi")
run("nvcc --version", check=False)

# bindgen needs libclang, which the image does not carry.
run("apt-get -qq update && apt-get -qq install -y libclang-dev", check=False)
run("ls /usr/lib/llvm-*/lib/libclang.so* /usr/lib/x86_64-linux-gnu/libclang*", check=False)

run(
    "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
    "| sh -s -- -y --default-toolchain stable --profile minimal"
)
os.environ["PATH"] = f"{os.path.expanduser('~/.cargo/bin')}:{os.environ['PATH']}"

run(f"rm -rf {SRC} && git clone {REPO} {SRC}")
run(f"cd {SRC} && git checkout {REV} && git rev-parse HEAD")

env = dict(os.environ)
# 75 == Turing (T4): int8 tensor cores and __dp4a, which is what llama.cpp's
# quantized matmul needs. 60 == P100, which has neither. Both, so the sibling run
# on a P100 uses the same binary and the comparison is of hardware only.
# `cuda-no-vmm`, not `cuda`: llama.cpp links `CUDA::cuda_driver` for its virtual-memory
# allocator, and that CMake target only exists when the driver *stub* `libcuda.so` is
# installed. Kaggle's image ships `libcuda.so.1` — the runtime library — and no stub, so
# `cuda` fails configure at ggml-cuda/CMakeLists.txt:182 with "the target was not found".
# `GGML_CUDA_NO_VMM=ON` drops that link. It costs a pooled-allocator optimization and
# changes no arithmetic.
env["CUDAARCHS"] = "60-real;75-real"
env["CMAKE_BUILD_PARALLEL_LEVEL"] = str(os.cpu_count() or 4)
run(
    f"cd {SRC} && cargo build --release --locked "
    '--features "llama-backend,llama-cpp-2/cuda-no-vmm" '
    "--bin otzaria-semantic-search",
    env=env,
)

binary = OUT / "otzaria-semantic-search"
run(f"cp {SRC}/target/release/otzaria-semantic-search {binary}")
run(f"cd {SRC} && git rev-parse HEAD > {OUT}/BUILT_FROM.txt")
run(f"sha256sum {binary} | tee -a {OUT}/BUILT_FROM.txt")

gpu = subprocess.run(
    "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
    shell=True,
    capture_output=True,
    text=True,
).stdout.strip()

results = []
for config in CONFIGS:
    artifact = OUT / f"artifact-{config['name']}"
    run(f"rm -rf {artifact}", check=False)

    env = dict(os.environ)
    env.update(
        {
            "OTZARIA_LLAMA_GPU_LAYERS": "99",
            "OTZARIA_LLAMA_CONTEXTS": "1",
            "OTZARIA_LLAMA_N_CTX": str(config["n_ctx"]),
            "OTZARIA_LLAMA_MAX_SEQUENCES": str(config["sequences"]),
            # One process per GPU. Splitting a 0.6B model across two cards by layer
            # spends more on the interconnect than it saves; production runs two
            # processes, and this measures one of them.
            "CUDA_VISIBLE_DEVICES": "0",
        }
    )

    started = time.time()
    completed = subprocess.run(
        f"{binary} build"
        f" --corpus-identity {CORPUS}/corpus-identity.json"
        f" --corpus-lines {CORPUS}/corpus-lines.jsonl"
        f" --model {CORPUS}/model.json"
        f" --model-file {MODEL}"
        f" --chunking {CORPUS}/chunking.json"
        f" --out {artifact}"
        f" --batch {config['batch']}",
        shell=True,
        env=env,
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - started

    print(f"\n=== {config['name']} ===", flush=True)
    print(completed.stdout[-2000:], flush=True)
    if completed.returncode != 0:
        print(completed.stderr[-4000:], flush=True)
        results.append({**config, "ok": False, "seconds": round(elapsed, 2)})
        continue

    vectors = next(
        (
            int(line.split()[-1])
            for line in completed.stdout.splitlines()
            if line.startswith("Vectors:")
        ),
        None,
    )
    results.append(
        {
            **config,
            "ok": True,
            "seconds": round(elapsed, 2),
            "vectors": vectors,
            "vectors_per_second": round(vectors / elapsed, 2) if vectors else None,
        }
    )
    print(f"--> {results[-1]['vectors_per_second']} vectors/s", flush=True)
    # The rate is the deliverable, not this artifact; /kaggle/working is capped.
    run(f"rm -rf {artifact}", check=False)

(OUT / "bench-results.json").write_text(
    json.dumps({"gpu": gpu, "revision": REV, "runs": results}, indent=2)
)
print(json.dumps({"gpu": gpu, "runs": results}, indent=2), flush=True)
run(f"ls -la {OUT}")
