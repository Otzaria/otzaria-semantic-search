#!/usr/bin/env python3
"""A work queue over Kaggle sessions, across however many accounts hold quota.

One job is one kernel run. A job names the account that owns it, the accelerator it
asks for, and the directory holding its `kernel-metadata.json`; everything else —
which shard it takes, where its output lands — travels in the kernel's own script.

Accounts are directories, not global state. `kaggle` reads its credentials from
`$KAGGLE_CONFIG_DIR/kaggle.json`, so a second account is a second directory and
never a re-login: `accounts/<name>/kaggle.json`. That is what lets the pool grow
without any job knowing about it.

    ./queue.py add   <name> <account> <accelerator> <dir>
    ./queue.py push  [name ...]     # default: everything queued
    ./queue.py poll                 # one line per live job
    ./queue.py fetch [name ...]     # download output of everything complete
    ./queue.py quota                # remaining GPU hours, per account

Scheduling is deliberately dumb: `push` starts every job whose account has quota
left and fewer than MAX_CONCURRENT running. Kaggle enforces its own ceiling anyway,
and a job that is refused stays queued rather than being lost.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

# Both outside the repository, and deliberately. A credential must not be one
# `git add -A` away from being published, and run state must not be one cleaned
# temp directory away from losing track of which shards have already been paid for.
ACCOUNTS = Path.home() / ".kaggle-accounts"
STATE = Path.home() / ".otzaria-kaggle"
JOBS = STATE / "jobs.json"
MAX_CONCURRENT = 2  # per account; raise once Kaggle's real ceiling is known

DONE = {"COMPLETE", "ERROR", "CANCEL_REQUESTED", "CANCEL_ACKNOWLEDGED"}


def load():
    return json.loads(JOBS.read_text()) if JOBS.exists() else []


def save(jobs):
    STATE.mkdir(parents=True, exist_ok=True)
    JOBS.write_text(json.dumps(jobs, indent=2))


def kaggle(account, *args, capture=True):
    env = dict(os.environ)
    # The account that holds the local token keeps using it; anything else is a
    # directory that was dropped in later.
    config = ACCOUNTS / account
    if config.exists():
        env["KAGGLE_CONFIG_DIR"] = str(config)
    return subprocess.run(
        ["kaggle", *args], env=env, capture_output=capture, text=True
    )


def status_of(job):
    out = kaggle(job["account"], "kernels", "status", job["slug"]).stdout
    for state in ("COMPLETE", "ERROR", "RUNNING", "QUEUED", "CANCEL"):
        if state in out:
            return state
    return out.strip()[-60:] or "UNKNOWN"


def cmd_add(name, account, accelerator, directory):
    jobs = load()
    if any(j["name"] == name for j in jobs):
        sys.exit(f"{name} already exists")
    meta = json.loads((Path(directory) / "kernel-metadata.json").read_text())
    jobs.append(
        {
            "name": name,
            "account": account,
            "accelerator": accelerator,
            "dir": str(Path(directory).resolve()),
            "slug": meta["id"],
            "state": "queued",
        }
    )
    save(jobs)
    print(f"added {name} -> {meta['id']} on {account} ({accelerator})")


def cmd_push(names):
    jobs = load()
    running = {}
    for job in jobs:
        if job["state"] == "pushed":
            running[job["account"]] = running.get(job["account"], 0) + 1

    for job in jobs:
        if names and job["name"] not in names:
            continue
        if job["state"] != "queued":
            continue
        if running.get(job["account"], 0) >= MAX_CONCURRENT:
            print(f"hold  {job['name']}: {job['account']} already has "
                  f"{MAX_CONCURRENT} in flight")
            continue
        args = ["kernels", "push", "-p", job["dir"]]
        if job["accelerator"]:
            args += ["--accelerator", job["accelerator"]]
        result = kaggle(job["account"], *args)
        line = (result.stdout + result.stderr).strip().splitlines()[-1:]
        print(f"push  {job['name']}: {line[0] if line else '?'}")
        if result.returncode == 0:
            job["state"] = "pushed"
            running[job["account"]] = running.get(job["account"], 0) + 1
    save(jobs)


def cmd_poll():
    jobs = load()
    for job in jobs:
        if job["state"] in ("queued", "fetched"):
            print(f"{job['name']:28s} {job['state']}")
            continue
        state = status_of(job)
        job["state"] = "complete" if state == "COMPLETE" else (
            "error" if state == "ERROR" else "pushed"
        )
        print(f"{job['name']:28s} {state:10s} {job['accelerator']:16s} {job['slug']}")
    save(jobs)


def cmd_fetch(names):
    jobs = load()
    for job in jobs:
        if names and job["name"] not in names:
            continue
        if job["state"] not in ("complete", "error"):
            continue
        target = STATE / "output" / job["name"]
        target.mkdir(parents=True, exist_ok=True)
        kaggle(job["account"], "kernels", "output", job["slug"], "-p", str(target))
        print(f"fetch {job['name']} -> {target}")
        if job["state"] == "complete":
            job["state"] = "fetched"
    save(jobs)


def cmd_quota():
    accounts = {job["account"] for job in load()} or {"otzaria"}
    for account in sorted(accounts):
        result = kaggle(account, "quota")
        gpu = [l for l in result.stdout.splitlines() if l.startswith("GPU")]
        print(f"{account:16s} {gpu[0] if gpu else result.stdout.strip()}")


if __name__ == "__main__":
    command, *rest = sys.argv[1:] or ["poll"]
    {
        "add": lambda: cmd_add(*rest),
        "push": lambda: cmd_push(rest),
        "poll": cmd_poll,
        "fetch": lambda: cmd_fetch(rest),
        "quota": cmd_quota,
    }[command]()
