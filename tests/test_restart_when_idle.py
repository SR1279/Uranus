"""The miner is restarted only when no task, verdict or request could be lost."""

from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path

import pytest

from rlvr.neurons.journal import SolveJournal, utc

NOW = 1_800_000_000.0


def load_script():
    spec = importlib.util.spec_from_file_location(
        "restart_when_idle", Path(__file__).parent.parent / "scripts" / "restart_when_idle.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def quiet(tmp_path):
    """A journal with one finished, graded task and a log untouched for an hour."""
    journal = SolveJournal(tmp_path / "journal")
    ids = {"challenge_id": "c1", "task_id": "1" * 64}
    folder = journal.open_task("1" * 64, "c1", NOW - 3000, ids)
    journal.solve({**ids, "time": utc(NOW - 3000), "latency_ms": 300_000, "status": 200})
    journal.outcome({**ids, "reason_code": "passed"})
    os.utime(folder, (NOW - 2700, NOW - 2700))
    log = tmp_path / "miner-out.log"
    log.write_text("INFO: POST /solve 200\n")
    os.utime(log, (NOW - 3600, NOW - 3600))
    work = tmp_path / "work"
    work.mkdir()
    return journal, work, log


def reasons(quiet):
    journal, work, log = quiet
    return load_script().busy_reasons(journal.directory, work, [log], now=NOW, quiet_s=180, verdict_wait_s=1500)


def test_a_quiet_miner_may_be_restarted(quiet):
    assert reasons(quiet) == []


def test_a_task_being_solved_holds_the_restart(quiet):
    journal, work, _log = quiet
    folder = journal.open_task("2" * 64, "c2", NOW - 60, {"challenge_id": "c2", "task_id": "2" * 64})
    os.utime(folder, (NOW - 60, NOW - 60))
    (work / "hone-v3-miner-abc").mkdir()
    assert reasons(quiet) == [f"task {'2' * 16} is being solved", "a task's workspace is unpacked"]


def test_a_rerun_in_progress_holds_the_restart(quiet):
    (quiet[1] / "hone-rerun-abc").mkdir()
    assert reasons(quiet) == ["a failed check is being rerun"]


def test_recent_traffic_holds_the_restart(quiet):
    os.utime(quiet[2], (NOW - 30, NOW - 30))
    assert reasons(quiet) == ["the miner received something 30s ago"]


def test_a_verdict_that_is_due_holds_the_restart_but_one_long_overdue_does_not(quiet):
    journal = quiet[0]
    journal.solve({"challenge_id": "c3", "task_id": "3" * 64, "time": utc(NOW - 900), "latency_ms": 300_000, "status": 200})
    journal.solve({"challenge_id": "c4", "task_id": "4" * 64, "time": utc(NOW - 7200), "latency_ms": 300_000, "status": 200})
    journal.solve({"challenge_id": "c5", "task_id": "5" * 64, "time": utc(NOW - 900), "latency_ms": 300_000, "status": 500})
    assert reasons(quiet) == [f"the verdict for task {'3' * 16} is due (answered 10 min ago)"]


def test_check_mode_reports_without_restarting(quiet, monkeypatch, capsys):
    script = load_script()
    journal, work, log = quiet
    os.utime(log, (time.time() - 10, time.time() - 10))
    monkeypatch.setattr(script, "_running_app", lambda _base: "sn5-miner-a")
    monkeypatch.setattr(script, "_pm2_logs", lambda app: [log] if app == "sn5-miner-a" else [])
    monkeypatch.setattr(script, "restart", lambda *_args, **_kwargs: pytest.fail("a check must never restart"))
    monkeypatch.setenv("MINER_RECORD_DIR", str(journal.directory))
    monkeypatch.setenv("MINER_WORK_DIR", str(work))
    assert script.main(["--check"]) == 1
    assert "Not a quiet moment: the miner received something" in capsys.readouterr().out
    os.utime(log, (time.time() - 3600, time.time() - 3600))
    assert script.main(["--check"]) == 0
    assert json.dumps("Quiet")[1:-1] in capsys.readouterr().out
