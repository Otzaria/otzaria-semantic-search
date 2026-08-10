"""What the operational tools do when Kaggle says no.

`python -m py_compile` proves these files parse. It does not prove the thing that
actually cost time: a tool that treats failure as success. Every case here is one that
happened during the v1 campaign —

* a fetch that failed and left the job looking downloaded,
* an account that reported thirty hours of quota and then ran without a GPU,
* a benchmark that printed three failures and exited zero.

Run with `python -m pytest tools/kaggle/test_queue.py`. No network, no Kaggle: the CLI
is replaced by a stub, because a test that needs the real thing is a test nobody runs.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent


def load(name, home):
    """Import one of the tools with `$HOME` pointed at a temporary directory.

    The module resolves `ACCOUNTS` and `STATE` from `Path.home()` at import time, so the
    redirection has to happen before the import rather than after.
    """
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    original = Path.home
    Path.home = staticmethod(lambda: home)
    try:
        spec.loader.exec_module(module)
    finally:
        Path.home = original
    return module


@pytest.fixture
def queue(tmp_path, monkeypatch):
    module = load("queue", tmp_path)
    (tmp_path / ".kaggle-accounts" / "acct").mkdir(parents=True)
    (tmp_path / ".kaggle-accounts" / "acct" / "access_token").write_text("t")
    return module


def stub(module, monkeypatch, *, returncode=0, stdout="", stderr="", writes=None):
    """Replace the `kaggle` CLI with a recorded call and a fixed answer."""
    calls = []

    def fake(account, *args, capture=True):
        calls.append(args)
        if writes is not None and args[:2] == ("kernels", "output"):
            target = Path(args[args.index("-p") + 1])
            for name, body in writes.items():
                (target / name).write_text(body)
        return subprocess.CompletedProcess(args, returncode, stdout, stderr)

    monkeypatch.setattr(module, "kaggle", fake)
    return calls


def one_job(module, state="complete"):
    module.save(
        [
            {
                "name": "embed-000",
                "account": "acct",
                "accelerator": "NvidiaTeslaT4",
                "dir": "/nowhere",
                "slug": "acct/embed-000",
                "state": state,
            }
        ]
    )


def test_a_failed_fetch_does_not_mark_the_job_fetched(queue, monkeypatch, capsys):
    """The one that would put a hole in the vectors and call it a complete set."""
    one_job(queue)
    stub(queue, monkeypatch, returncode=1, stderr="403 Forbidden")

    queue.cmd_fetch([])

    assert queue.load()[0]["state"] == "complete", "a failed download is not a fetch"
    assert "FAILED" in capsys.readouterr().out
    assert not (queue.STATE / "output" / "embed-000").exists()


def test_a_fetch_that_downloads_nothing_is_a_failure(queue, monkeypatch, capsys):
    """Exit code zero and an empty directory is what a deleted kernel output looks like."""
    one_job(queue)
    stub(queue, monkeypatch, returncode=0)

    queue.cmd_fetch([])

    assert queue.load()[0]["state"] == "complete"
    assert "FAILED" in capsys.readouterr().out


def test_a_successful_fetch_lands_atomically(queue, monkeypatch):
    one_job(queue)
    stub(queue, monkeypatch, writes={"vectors.f32": "x", "records.jsonl": "y"})

    queue.cmd_fetch([])

    target = queue.STATE / "output" / "embed-000"
    assert queue.load()[0]["state"] == "fetched"
    assert (target / "vectors.f32").read_text() == "x"
    assert not target.with_suffix(".partial").exists(), "the staging directory is gone"


def test_a_reserve_holds_the_last_hours_of_an_account(queue, monkeypatch, capsys):
    """`otzaria` is somebody's working account, not thirty hours of campaign budget."""
    one_job(queue, state="queued")
    jobs = queue.load()
    jobs[0]["account"] = "otzaria"
    queue.save(jobs)
    stub(queue, monkeypatch, stdout="GPU       28.50h  1.50h     30.00h  2026-08-15\n")

    queue.cmd_push([])

    assert queue.load()[0]["state"] == "queued", "held, not pushed"
    assert "reserved" in capsys.readouterr().out


def test_a_quota_that_cannot_be_read_holds_the_job(queue, monkeypatch, capsys):
    """None is not zero. Not knowing is a reason to ask, not a reason to spend."""
    one_job(queue, state="queued")
    stub(queue, monkeypatch, stdout="something else entirely\n")

    queue.cmd_push([])

    assert queue.load()[0]["state"] == "queued"
    assert "could not be read" in capsys.readouterr().out


def test_the_state_file_survives_being_written(queue):
    """It records which shards have already been paid for in GPU hours."""
    one_job(queue)
    assert json.loads(queue.JOBS.read_text())[0]["name"] == "embed-000"
    assert not queue.JOBS.with_suffix(".tmp").exists(), "the temporary file was renamed"


@pytest.mark.parametrize(
    "gpus,internet,expected",
    [
        ([], True, "no GPU"),
        (["Tesla T4"], "URLError: blocked", "network is blocked"),
        ([], "URLError: blocked", "no GPU"),
    ],
)
def test_smoke_refuses_an_account_that_cannot_run_a_shard(tmp_path, gpus, internet, expected):
    """An unverified account reports a full quota and then runs on a CPU with no network.

    The real function from `smoke.py`, not a restatement of it here.
    """
    smoke = load("smoke", tmp_path)
    verdict = {"gpus": gpus, "gpu_count": len(gpus), "internet": internet}
    reason = smoke.unusable(verdict)
    assert reason is not None and expected in reason


def test_smoke_accepts_only_a_gpu_and_a_network(tmp_path):
    smoke = load("smoke", tmp_path)
    assert (
        smoke.unusable({"gpus": ["Tesla T4", "Tesla T4"], "gpu_count": 2, "internet": True})
        is None
    )


def test_the_tools_that_report_a_failure_also_exit_non_zero():
    """A benchmark that prints three failures and exits zero gets read by eye."""
    for name in ("smoke", "run_bench", "build_cuda_binary"):
        body = (HERE / f"{name}.py").read_text()
        assert "SystemExit" in body, f"{name}.py can still finish successfully after failing"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
