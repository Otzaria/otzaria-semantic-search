"""Does this account actually get what a production shard needs?

Kaggle gates both GPU attachment and notebook internet behind phone verification, and
neither refusal is visible from the quota API: an unverified account reports the same
30 hours and then runs the notebook on a CPU with no network. Since a shard needs a
GPU to be worth running and the internet only to fetch its inputs, both are checked
here, cheaply, before any real work is scheduled onto the account.
"""
import subprocess, json, os, urllib.request

gpu = subprocess.run("nvidia-smi --query-gpu=name --format=csv,noheader",
                     shell=True, capture_output=True, text=True)
names = [n for n in gpu.stdout.strip().splitlines() if n]

try:
    urllib.request.urlopen("https://pypi.org/simple/", timeout=20).read(64)
    internet = True
except Exception as error:
    internet = f"{type(error).__name__}: {error}"

verdict = {"gpus": names, "gpu_count": len(names), "internet": internet}
print(json.dumps(verdict, indent=2))
with open("/kaggle/working/smoke.json", "w") as handle:
    json.dump(verdict, handle, indent=2)
