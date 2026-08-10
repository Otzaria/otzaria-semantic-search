#!/usr/bin/env python3
"""A work queue over Kaggle sessions, across however many accounts hold quota.

One job is one kernel run. A job names the account that owns it, the accelerator it
asks for, and the directory holding its `kernel-metadata.json`; everything else —
which shard it takes, where its output lands — travels in the kernel's own script.

Accounts are directories, not global state: `~/.kaggle-accounts/<name>/access_token`,
one per account, so a second account is a second directory and never a re-login. That
is what lets the pool grow without any job knowing about it. See ACCOUNTS.md.

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
import shutil
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

# GPU hours this campaign will not touch, per account. An account someone else works
# from is not a pool of 30 hours; it is a pool of 30 minus whatever they need that
# week, and taking the last hour of it is a way to break their day rather than a way
# to finish sooner. Checked against the quota the API reports, before each push.
RESERVE_HOURS = {"otzaria": 2.0}
DEFAULT_RESERVE = 0.0

DONE = {"COMPLETE", "ERROR", "CANCEL_REQUESTED", "CANCEL_ACKNOWLEDGED"}


def load():
    return json.loads(JOBS.read_text()) if JOBS.exists() else []


def save(jobs):
    """Write through a temporary file and rename.

    The state file records which shards have already been paid for in GPU hours. A
    truncated write — an interrupt in the middle of `write_text` — loses that, and the
    recovery is to re-run work that was already done.
    """
    STATE.mkdir(parents=True, exist_ok=True)
    temporary = JOBS.with_suffix(".tmp")
    with open(temporary, "w") as handle:
        json.dump(jobs, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(JOBS)


def kaggle(account, *args, capture=True):
    env = dict(os.environ)
    # The account whose token is installed globally keeps using it; anything else is a
    # directory that was dropped in later.
    #
    # A *credential* has to be there, not just the directory. An empty folder — one
    # created in advance of a teammate's token — would otherwise redirect `kaggle` at
    # nothing and fail as "not authenticated" rather than as "that account has no key
    # yet", which is a much longer way round to the same fix.
    #
    # Two shapes, and the order matters. `KAGGLE_API_TOKEN` accepts a *path*, which is
    # the only per-process way to point CLI 2.2 at a specific token: it reads
    # `~/.kaggle/access_token` from an absolute path that `KAGGLE_CONFIG_DIR` does not
    # move. `KAGGLE_CONFIG_DIR` still serves an OAuth `credentials.json`, which is what
    # `kaggle auth login` leaves behind for a teammate who would rather not hand over a
    # token at all.
    config = ACCOUNTS / account
    token = config / "access_token"
    if token.exists():
        env["KAGGLE_API_TOKEN"] = str(token)
    elif (config / "credentials.json").exists():
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


def remaining_hours(account):
    """GPU hours the API says are left, or None if it will not say.

    None is not zero: a quota that cannot be read is a reason to ask rather than a
    reason to spend, so `push` holds the job instead of guessing either way.
    """
    for line in kaggle(account, "quota").stdout.splitlines():
        if line.startswith("GPU"):
            for field in line.split():
                if field.endswith("h"):
                    try:
                        float(field[:-1])
                    except ValueError:
                        continue
                    # used, remaining, total — the second is the one to act on.
                    parts = [f for f in line.split() if f.endswith("h")]
                    return float(parts[1][:-1]) if len(parts) > 1 else None
    return None


def cmd_push(names):
    jobs = load()
    running = {}
    for job in jobs:
        if job["state"] == "pushed":
            running[job["account"]] = running.get(job["account"], 0) + 1
    budget = {}

    for job in jobs:
        if names and job["name"] not in names:
            continue
        if job["state"] != "queued":
            continue
        if running.get(job["account"], 0) >= MAX_CONCURRENT:
            print(f"hold  {job['name']}: {job['account']} already has "
                  f"{MAX_CONCURRENT} in flight")
            continue
        account = job["account"]
        if account not in budget:
            budget[account] = remaining_hours(account)
        left, reserve = budget[account], RESERVE_HOURS.get(account, DEFAULT_RESERVE)
        if left is None:
            print(f"hold  {job['name']}: {account}'s quota could not be read")
            continue
        if left <= reserve:
            print(f"hold  {job['name']}: {account} has {left:.2f}h left and "
                  f"{reserve:.2f}h is reserved")
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
        staging = target.with_suffix(".partial")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)

        result = kaggle(job["account"], "kernels", "output", job["slug"], "-p", str(staging))
        # Checked, not assumed. Marking a job `fetched` on a failed download is how a hole
        # in the vectors reaches the merge looking like a complete set of shards.
        if result.returncode != 0 or not any(staging.iterdir()):
            print(f"fetch {job['name']}: FAILED ({result.stderr.strip()[-120:] or 'no files'})")
            shutil.rmtree(staging, ignore_errors=True)
            continue
        if target.exists():
            shutil.rmtree(target)
        staging.rename(target)
        print(f"fetch {job['name']} -> {target}")
        if job["state"] == "complete":
            job["state"] = "fetched"
    save(jobs)


def cmd_quota():
    # Every account that has a credential, not only the ones already carrying work:
    # the number this prints is the pool, and the pool is what the schedule is built
    # from. `otzaria` is included by name because its token is the globally installed
    # one and it therefore has no directory here.
    accounts = {directory.name for directory in ACCOUNTS.glob("*") if directory.is_dir()}
    accounts |= {job["account"] for job in load()} | {"otzaria"}
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
