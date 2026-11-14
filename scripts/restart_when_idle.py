#!/usr/bin/env python
"""Restart the miner only at a quiet moment, so the restart costs no task and no verdict.

A restart takes the miner off the network until it is serving again, and pm2 kills a solve
that is still running. A task a validator sends in that gap is lost and scores zero. This
waits until the miner is quiet:

- no task is in progress: every task folder has its record, no workspace is unpacked and no
  rerun of a failed check is running;
- nothing has reached the miner for --quiet seconds: no task, no verdict, no request of any
  kind (the miner logs every request it receives);
- no verdict is due: every task answered within the last --verdict-wait seconds has its
  verdict (they come 10 to 20 minutes after the answer);

then restarts it with pm2, waits until it answers again and reports how long it was away.
It never checks the miner's health before the restart, since that request would itself be
logged as activity.

    python scripts/restart_when_idle.py --check     # is now a quiet moment? exit 0 if so
    python scripts/restart_when_idle.py             # wait for one, then restart

Run it from the repository root.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from rlvr.neurons.demo_miner import DemoMinerSettings
from rlvr.neurons.journal import TASK_FILE, parse_utc, read_records

APP = "sn5-miner"
# Task folders older than this are finished whatever their records say.
RECENT_FOLDER_S = 3 * 3600


def _folder_ids(folder: Path) -> tuple[str, str] | None:
    try:
        details = json.loads((folder / TASK_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return details.get("challenge_id"), details.get("task_id")


def busy_reasons(
    journal: Path,
    work_dir: Path | None,
    logs: list[Path],
    *,
    now: float,
    quiet_s: float,
    verdict_wait_s: float,
) -> list[str]:
    """Why now is not a quiet moment to restart; an empty list means it is."""
    reasons: list[str] = []
    solves = read_records(journal / "solves.jsonl")
    recorded = {(record.get("challenge_id"), record.get("task_id")) for record in solves}
    tasks = journal / "tasks"
    for folder in sorted(tasks.iterdir()) if tasks.is_dir() else ():
        try:
            if not folder.is_dir() or now - folder.stat().st_mtime > RECENT_FOLDER_S:
                continue
        except OSError:
            continue
        ids = _folder_ids(folder)
        if ids is not None and ids not in recorded:
            reasons.append(f"task {str(ids[1])[:16]} is being solved")
    if work_dir is not None and work_dir.is_dir():
        for entry in work_dir.iterdir():
            if entry.name.startswith("hone-v3-miner-"):
                reasons.append("a task's workspace is unpacked")
            elif entry.name.startswith("hone-rerun-"):
                reasons.append("a failed check is being rerun")
    newest = max((path.stat().st_mtime for path in logs if path.is_file()), default=None)
    if newest is not None and now - newest < quiet_s:
        reasons.append(f"the miner received something {now - newest:.0f}s ago")
    heard = {(record.get("challenge_id"), record.get("task_id")) for record in read_records(journal / "outcomes.jsonl")}
    for solve in solves:
        started = parse_utc(solve.get("time"))
        latency = solve.get("latency_ms")
        if solve.get("status") != 200 or started is None or not isinstance(latency, (int, float)):
            continue
        age = now - started - latency / 1000
        if 0 <= age < verdict_wait_s and (solve.get("challenge_id"), solve.get("task_id")) not in heard:
            reasons.append(f"the verdict for task {str(solve.get('task_id'))[:16]} is due (answered {age / 60:.0f} min ago)")
    return list(dict.fromkeys(reasons))


def _running_app(base: str) -> str | None:
    """The miner's online pm2 app: the single app, or the copy scripts/swap_miner.py runs."""
    try:
        listed = json.loads(subprocess.run(["pm2", "jlist"], capture_output=True, text=True, timeout=30, check=True).stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    online = {process.get("name") for process in listed if (process.get("pm2_env") or {}).get("status") == "online"}
    return next((name for name in (base, f"{base}-a", f"{base}-b") if name in online), None)


def _pm2_logs(app: str) -> list[Path]:
    """The app's log files, as pm2 reports them; the process environment is never read."""
    try:
        listed = json.loads(subprocess.run(["pm2", "jlist"], capture_output=True, text=True, timeout=30, check=True).stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return []
    for process in listed:
        if process.get("name") == app:
            environment = process.get("pm2_env") or {}
            return [Path(environment[key]) for key in ("pm_out_log_path", "pm_err_log_path") if environment.get(key)]
    return []


def _healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
            return response.status == 200
    except OSError:
        return False


def restart(app: str, port: int, *, timeout_s: float = 600.0) -> float:
    """Restart the app and wait until it answers; returns the seconds it was away."""
    started = time.monotonic()
    subprocess.run(["pm2", "restart", app], capture_output=True, timeout=60, check=True)
    while time.monotonic() - started < timeout_s:
        if _healthy(port):
            return time.monotonic() - started
        time.sleep(0.25)
    raise TimeoutError(f"{app} did not answer within {timeout_s:.0f}s of the restart")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--app", default=None, help=f"pm2 app name (default: whichever of {APP}, {APP}-a, {APP}-b runs)")
    parser.add_argument("--quiet", type=float, default=180.0, help="seconds without any request first (default 180)")
    parser.add_argument("--verdict-wait", type=float, default=1500.0,
                        help="wait for verdicts of tasks answered this many seconds ago (default 1500)")
    parser.add_argument("--max-wait", type=float, default=4 * 3600.0, help="give up after this many seconds (default 4 h)")
    parser.add_argument("--check", action="store_true", help="only say whether now is a quiet moment")
    args = parser.parse_args(argv)
    settings = DemoMinerSettings()
    journal = Path(settings.miner_record_dir or "data/solves").resolve()
    work_dir = Path(settings.miner_work_dir).resolve() if settings.miner_work_dir else None
    args.app = args.app or _running_app(APP) or APP
    logs = _pm2_logs(args.app)
    if not logs:
        print(f"pm2 has no app named {args.app}, or pm2 could not be asked")
        return 2
    deadline = time.monotonic() + args.max_wait
    last: list[str] | None = None
    while True:
        reasons = busy_reasons(journal, work_dir, logs, now=time.time(), quiet_s=args.quiet, verdict_wait_s=args.verdict_wait)
        if not reasons:
            break
        if args.check:
            print("Not a quiet moment: " + "; ".join(reasons))
            return 1
        changed = [re.sub(r"\d+", "#", reason) for reason in reasons]  # a new line only for a new reason, not a new count
        if changed != last:
            print(time.strftime("%H:%M:%S", time.gmtime()) + " waiting: " + "; ".join(reasons), flush=True)
            last = changed
        if time.monotonic() > deadline:
            print("No quiet moment came; the miner was not restarted")
            return 1
        time.sleep(10)
    if args.check:
        print("Quiet: no task in progress, nothing received recently and no verdict due")
        return 0
    away = restart(args.app, settings.axon_port)
    print(f"{time.strftime('%H:%M:%S', time.gmtime())} restarted {args.app}; it answered again after {away:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
