#!/usr/bin/env python
"""Summarise what the miner's solves cost and why each failed task failed.

    python scripts/solve_report.py                          # MINER_RECORD_DIR from .env
    python scripts/solve_report.py --dir data/solves-local  # runs from scripts/try_miner.py
    python scripts/solve_report.py --rerun                  # rerun failed checks not yet rerun

Run it from the repository root. It also rewrites LOG.md and tasks.csv in the journal
directory, which list every task: passed or failed, why, what kind it was and what it cost.
The miner rewrites them itself after every task and verdict.

Failures are sorted by cause. Only "reasoning" failures, where a hidden check failed after a
normal finish, are ones a stronger model could fix; the others are harness, time or limit
problems to fix first. --rerun runs each failed repository check again on the miner's own
answer, in the validators' sandbox image, to show which hidden tests failed and why; the
miner does this by itself when a verdict arrives, so it is only needed after a restart.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from rlvr.neurons.demo_miner import DemoMinerSettings
from rlvr.neurons.journal import WORKSPACE_FILE, SolveJournal, load_tasks, report, utc


def rerun_pending(journal: SolveJournal, work_dir: Path | None) -> int:
    """Rerun every failed check that has a kept workspace and no rerun yet; returns how many ran."""
    from rlvr.neurons.diagnosis import DiagnosisError, rerun_failed_check
    from rlvr.neurons.workbench import Sandbox

    sandbox, status = Sandbox.detect(concurrency=1)
    if sandbox is None:
        print(f"Failed checks cannot be rerun here: {status}")
        return 0
    tasks, _refused = load_tasks(journal.directory)
    ran = 0
    for task in tasks:
        outcome = task.outcome or {}
        if (
            task.diagnosis is not None
            or task.folder is None
            or outcome.get("reason_code") != "check_failed"
            or not outcome.get("failed_check")
            or not (task.folder / WORKSPACE_FILE).is_file()
        ):
            continue
        record = {"challenge_id": task.solve.get("challenge_id"), "task_id": task.solve.get("task_id")}
        try:
            record.update(rerun_failed_check(task.folder, outcome["failed_check"], sandbox, work_dir))
        except DiagnosisError as error:
            record["error"] = str(error)
        record["time"] = utc(time.time())
        journal.diagnosis(record)
        print(f"Reran the failed check of task {str(record['task_id'])[:16]}: {record.get('summary') or record.get('error')}")
        ran += 1
    return ran


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, help="journal directory (default: MINER_RECORD_DIR from .env)")
    parser.add_argument("--failures", type=int, default=40, help="how many recent failures to list")
    parser.add_argument("--rerun", action="store_true", help="rerun failed checks that have not been rerun yet")
    parser.add_argument("--no-write", action="store_true", help="print the summary without rewriting LOG.md and tasks.csv")
    parser.add_argument("--since", help="count only tasks received at or after this UTC time, such as 2026-11-10T10:00:00Z")
    args = parser.parse_args(argv)
    settings = DemoMinerSettings()
    directory = (args.dir or Path(settings.miner_record_dir or "data/solves")).resolve()
    if not directory.is_dir():
        print(f"No solves recorded in {directory} yet.")
        return 0
    if not args.no_write or args.rerun:
        journal = SolveJournal(directory)
        journal.backfill()
        if args.rerun:
            rerun_pending(journal, Path(settings.miner_work_dir).resolve() if settings.miner_work_dir else None)
        if not args.no_write:
            journal.write_log()
    print(report(directory, listed=args.failures, since=args.since))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
