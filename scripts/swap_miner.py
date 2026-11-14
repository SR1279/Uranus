#!/usr/bin/env python
"""Replace the running miner with a fresh copy, with no moment when validators cannot reach it.

Two pm2 apps take turns, sn5-miner-a and sn5-miner-b. The script:

1. starts the idle one, which opens the same port beside the running copy (SO_REUSEPORT), so
   the kernel shares new connections between the two;
2. waits until the new copy itself answers /health; a copy that never does is removed again
   and the running copy keeps serving;
3. stops the old copy through pm2. It closes its port at once, so every new connection goes
   to the new copy, finishes the solves it is running and only then exits; pm2 waits for that;
4. removes the old app and saves pm2's process list, so a reboot starts the new copy.

Connections still waiting in the old copy's queue when it closes are handed to the new copy
when net.ipv4.tcp_migrate_req is 1, which the script turns on.

The first time, the miner still runs as the single app sn5-miner, whose port cannot be
shared. That one switch stops it and starts sn5-miner-a, a gap of a few seconds, so it waits
for a quiet moment as restart_when_idle.py does.

    python scripts/swap_miner.py            # replace the running copy
    python scripts/swap_miner.py --status   # which copy is serving

Run it from the repository root, as the user that runs pm2.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MIGRATE = Path("/proc/sys/net/ipv4/tcp_migrate_req")


def pm2(*arguments: str, timeout: float = 60.0) -> subprocess.CompletedProcess:
    return subprocess.run(["pm2", *arguments], capture_output=True, text=True, timeout=timeout)


def apps(base: str) -> dict[str, dict]:
    """The miner's pm2 apps by name, with only the fields used here; their environment is never read."""
    listed = json.loads(pm2("jlist").stdout or "[]")
    found = {}
    for process in listed:
        name = process.get("name")
        if name in (base, f"{base}-a", f"{base}-b"):
            environment = process.get("pm2_env") or {}
            found[name] = {
                "status": environment.get("status"),
                "pid": process.get("pid"),
                "restarts": environment.get("restart_time"),
                "logs": [environment.get(key) for key in ("pm_out_log_path", "pm_err_log_path") if environment.get(key)],
            }
    return found


def health(port: int) -> dict | None:
    """One /health answer on a new connection, from whichever copy the kernel picks."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as response:
            value = json.loads(response.read())
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def listening_pids(port: int) -> set[int]:
    """The processes listening on the port."""
    try:
        listed = subprocess.run(["ss", "-Hltnp", f"sport = :{port}"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return set()
    return {int(found) for found in re.findall(r"pid=(\d+)", listed)}


def allow_queue_handover() -> str:
    """Turn on the kernel's handover of queued connections from a closing copy to the other one."""
    try:
        if MIGRATE.read_text().strip() == "1":
            return "already on"
        MIGRATE.write_text("1\n")
        return "turned on"
    except OSError as error:
        return f"could not be turned on ({error.strerror}); a connection arriving at the instant the old copy closes may be reset"


def start(name: str, script: Path, cwd: Path, kill_timeout_ms: int) -> None:
    pm2("delete", name)  # a stopped or failed app from an earlier attempt
    started = pm2(
        "start", str(script), "--name", name, "--interpreter", "bash", "--cwd", str(cwd),
        "--kill-timeout", str(kill_timeout_ms), "--time",
    )
    if started.returncode != 0:
        raise RuntimeError(f"pm2 could not start {name}: {started.stderr.strip()[-300:]}")


def wait_until_serving(base: str, name: str, port: int, timeout_s: float) -> int:
    """The new copy's pid once it has answered on the port itself; raises if it never does."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        found = apps(base).get(name) or {}
        if found.get("status") in ("errored", "stopped") or (found.get("restarts") or 0) > 0:
            raise RuntimeError(f"{name} failed to start; its output is in {Path.home() / '.pm2' / 'logs' / (name + '-error.log')}")
        pid = found.get("pid")
        if pid:
            for _attempt in range(8):  # new connections, so the kernel tries both copies
                answer = health(port)
                if answer is not None and answer.get("pid") == pid:
                    return pid
        time.sleep(0.5)
    raise RuntimeError(f"{name} did not answer on port {port} within {timeout_s:.0f}s")


class Probe:
    """Asks /health over and over on new connections and counts every failure."""

    def __init__(self, port: int, interval_s: float = 0.2):
        self.port, self.interval_s = port, interval_s
        self.asked = self.failed = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop.is_set():
            self.asked += 1
            if health(self.port) is None:
                self.failed += 1
            time.sleep(self.interval_s)

    def __enter__(self) -> Probe:
        self.thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop.set()
        self.thread.join(10)


def _wait_for_quiet(args: argparse.Namespace, logs: list[Path]) -> None:
    spec = importlib.util.spec_from_file_location("restart_when_idle", REPO / "scripts" / "restart_when_idle.py")
    idle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(idle)
    deadline = time.monotonic() + args.max_wait
    last = None
    while True:
        reasons = idle.busy_reasons(
            args.journal, args.work_dir, logs, now=time.time(), quiet_s=args.quiet, verdict_wait_s=args.verdict_wait
        )
        if not reasons:
            return
        changed = [re.sub(r"\d+", "#", reason) for reason in reasons]  # a new line only for a new reason, not a new count
        if changed != last:
            print(time.strftime("%H:%M:%S", time.gmtime()) + " waiting for a quiet moment: " + "; ".join(reasons), flush=True)
            last = changed
        if time.monotonic() > deadline:
            raise RuntimeError("no quiet moment came; nothing was changed")
        time.sleep(10)


def switch_from_single_app(args: argparse.Namespace, single: dict) -> None:
    """The one switch with a gap: the single app's port cannot be shared, so it stops first."""
    new = f"{args.name}-a"
    _wait_for_quiet(args, [Path(path) for path in single["logs"]])
    began = time.monotonic()
    pm2("stop", args.name)
    try:
        start(new, args.script, args.cwd, args.kill_timeout_ms)
        pid = wait_until_serving(args.name, new, args.port, args.ready_timeout)
    except Exception:
        pm2("delete", new)
        pm2("restart", args.name)
        raise
    away = time.monotonic() - began
    pm2("delete", args.name)
    print(f"Switched {args.name} to {new} (pid {pid}); the miner was away for {away:.1f}s, this once")


def swap(args: argparse.Namespace, old: str) -> None:
    """Start the idle copy beside the running one, then let the old one finish and stop."""
    new = f"{args.name}-b" if old.endswith("-a") else f"{args.name}-a"
    old_pid = apps(args.name)[old]["pid"]
    print(f"Kernel handover of queued connections: {allow_queue_handover()}")
    with Probe(args.port) as probe:
        began = time.monotonic()
        try:
            start(new, args.script, args.cwd, args.kill_timeout_ms)
            pid = wait_until_serving(args.name, new, args.port, args.ready_timeout)
        except Exception:
            pm2("delete", new)
            print(f"{new} did not come up; {old} keeps serving")
            raise
        print(f"{new} (pid {pid}) is serving beside {old} (pid {old_pid}) after {time.monotonic() - began:.1f}s")
        # pm2 sends SIGINT and waits up to the kill timeout: the old copy closes its port now
        # and exits once its solves are done.
        stopping = subprocess.Popen(["pm2", "stop", old], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        closed_by = time.monotonic() + 60
        while old_pid in listening_pids(args.port) and time.monotonic() < closed_by:
            time.sleep(0.1)
        time.sleep(1.0)  # a few more probes with only the new copy listening
    print(f"{old} closed its port; while both copies were up, {probe.asked} health checks were made and {probe.failed} failed")
    print(f"{old} is finishing its solves, if it has any; it exits when they are done", flush=True)
    stopping.wait(timeout=args.kill_timeout_ms / 1000 + 120)
    pm2("delete", old)
    print(f"{old} has stopped; {new} serves alone")


def status(args: argparse.Namespace) -> None:
    for name, found in sorted(apps(args.name).items()):
        print(f"{name}: {found['status']}, pid {found['pid']}")
    answers = [health(args.port) for _attempt in range(6)]
    pids = sorted({answer.get("pid") for answer in answers if answer})
    print(f"Port {args.port}: listening pids {sorted(listening_pids(args.port))}; answered by {pids or 'nothing'}")


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(REPO))
    from rlvr.neurons.demo_miner import DRAIN_MARGIN_S, DemoMinerSettings

    settings = DemoMinerSettings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="sn5-miner", help="pm2 name; the copies are NAME-a and NAME-b")
    parser.add_argument("--port", type=int, default=settings.axon_port)
    parser.add_argument("--script", type=Path, default=REPO / "start_demo_miner.sh")
    parser.add_argument("--cwd", type=Path, default=REPO)
    parser.add_argument("--kill-timeout-ms", type=int,
                        default=int((settings.openrouter_request_timeout_s + DRAIN_MARGIN_S + 60) * 1000),
                        help="how long pm2 lets a stopping copy finish its solves")
    parser.add_argument("--ready-timeout", type=float, default=300.0, help="seconds for a new copy to start answering")
    parser.add_argument("--journal", type=Path, default=Path(settings.miner_record_dir or "data/solves").resolve())
    parser.add_argument("--work-dir", type=Path, default=Path(settings.miner_work_dir).resolve() if settings.miner_work_dir else None)
    parser.add_argument("--quiet", type=float, default=180.0, help="for the first switch: seconds without any request")
    parser.add_argument("--verdict-wait", type=float, default=1500.0, help="for the first switch: wait for verdicts this recent")
    parser.add_argument("--max-wait", type=float, default=4 * 3600.0, help="for the first switch: give up after this long")
    parser.add_argument("--no-save", action="store_true", help="do not save pm2's process list afterwards")
    parser.add_argument("--status", action="store_true", help="only show which copy is serving")
    args = parser.parse_args(argv)
    if args.status:
        status(args)
        return 0
    running = {name: found for name, found in apps(args.name).items() if found["status"] == "online"}
    copies = sorted(name for name in running if name != args.name)
    try:
        if len(copies) > 1:
            print(f"Both {copies[0]} and {copies[1]} are running: an earlier swap is still finishing. Nothing was changed.")
            return 1
        if copies:
            swap(args, copies[0])
        elif args.name in running:
            switch_from_single_app(args, running[args.name])
        else:
            start(f"{args.name}-a", args.script, args.cwd, args.kill_timeout_ms)
            pid = wait_until_serving(args.name, f"{args.name}-a", args.port, args.ready_timeout)
            print(f"Nothing was running; started {args.name}-a (pid {pid})")
    except RuntimeError as error:
        print(f"Stopped: {error}")
        return 1
    if not args.no_save:
        saved = pm2("save")
        print("Saved pm2's process list, so a reboot starts the new copy" if saved.returncode == 0 else "pm2 save failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
