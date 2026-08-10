"""Does this account actually get what a production shard needs?

Kaggle gates both GPU attachment and notebook internet behind phone verification, and
neither refusal is visible from the quota API: an unverified account reports the same
30 hours and then runs the notebook on a CPU with no network. Since a shard needs a
GPU to be worth running and the internet only to fetch its inputs, both are checked
here, cheaply, before any real work is scheduled onto the account.

The check ends in an exit code, not a printed verdict. That is the whole point: the
subject of this test looks healthy from outside, so a tool that reports and returns
zero is one somebody has to read by eye — and during the v1 campaign four accounts were
scheduled that way before anyone noticed.
"""

import json
import subprocess
import urllib.request


def unusable(verdict):
    """Why this account cannot run a shard, or `None` if it can.

    Separated from the probing so it can be tested without a Kaggle session.
    """
    if not verdict["gpus"]:
        return f"no GPU was attached (internet={verdict['internet']})"
    if verdict["internet"] is not True:
        return f"the network is blocked: {verdict['internet']}"
    return None


def probe():
    """Ask the session what it has."""
    gpu = subprocess.run(
        "nvidia-smi --query-gpu=name --format=csv,noheader",
        shell=True,
        capture_output=True,
        text=True,
    )
    names = [name for name in gpu.stdout.strip().splitlines() if name]

    try:
        urllib.request.urlopen("https://pypi.org/simple/", timeout=20).read(64)
        internet = True
    except Exception as error:  # noqa: BLE001 - any failure means "no network here"
        internet = f"{type(error).__name__}: {error}"

    return {"gpus": names, "gpu_count": len(names), "internet": internet}


def main():
    verdict = probe()
    print(json.dumps(verdict, indent=2))
    with open("/kaggle/working/smoke.json", "w") as handle:
        json.dump(verdict, handle, indent=2)

    reason = unusable(verdict)
    if reason is not None:
        raise SystemExit(f"this account cannot run a shard: {reason}")


if __name__ == "__main__":
    main()
