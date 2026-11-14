"""A new copy of the miner takes over its port without refusing or dropping a request."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from rlvr.neurons.serving import shared_socket

REPO = Path(__file__).resolve().parent.parent
# A stand-in for the miner: the same serving code, a health check naming the copy, and a slow
# request standing in for a solve.
APP = """
import asyncio, os, sys
from fastapi import FastAPI
from rlvr.neurons.serving import serve

app = FastAPI()
port, name, slow_s = int(sys.argv[1]), sys.argv[2], float(sys.argv[3])

@app.get("/health")
async def health():
    return {"name": name, "pid": os.getpid()}

@app.get("/slow")
async def slow():
    await asyncio.sleep(slow_s)
    return {"name": name, "done": True}

serve(app, "127.0.0.1", port, drain_s=30, log_level="warning")
"""


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def get(port: int, path: str, timeout: float = 5.0) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout) as response:
        return json.loads(response.read())


def launch(tmp_path: Path, port: int, name: str, slow_s: float) -> subprocess.Popen:
    script = tmp_path / "app.py"
    script.write_text(APP)
    return subprocess.Popen(
        [sys.executable, str(script), str(port), name, str(slow_s)],
        env={**os.environ, "PYTHONPATH": str(REPO)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def wait_for(port: int, name: str, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if get(port, "/health", timeout=1)["name"] == name:
                return
        except (OSError, ValueError):
            pass
        time.sleep(0.05)
    raise AssertionError(f"{name} never answered")


def test_two_copies_can_listen_on_one_port():
    port = free_port()
    first = shared_socket("127.0.0.1", port)
    second = shared_socket("127.0.0.1", port)
    first.close()
    second.close()


def test_a_new_copy_takes_over_without_refusing_or_dropping_a_request(tmp_path):
    port = free_port()
    old = launch(tmp_path, port, "old", slow_s=4)
    new = None
    stop = threading.Event()
    try:
        wait_for(port, "old")
        answered, failures = [], []

        def probe():
            while not stop.is_set():
                try:
                    answered.append(get(port, "/health")["name"])
                except (OSError, ValueError) as error:
                    failures.append(error)
                time.sleep(0.01)

        prober = threading.Thread(target=probe, daemon=True)
        prober.start()
        slow: dict = {}

        def solve_slowly():
            try:
                slow.update(get(port, "/slow", timeout=30))
            except (OSError, ValueError) as error:
                slow["error"] = repr(error)

        solve = threading.Thread(target=solve_slowly, daemon=True)
        solve.start()  # only the old copy is listening, so it serves this "solve"
        time.sleep(0.5)
        new = launch(tmp_path, port, "new", slow_s=4)
        wait_for(port, "new")
        old.send_signal(signal.SIGINT)  # as pm2 stops an app
        solve.join(30)
        assert slow == {"name": "old", "done": True}  # the stopping copy finished its work
        # uvicorn re-raises the stop signal once it has drained, so the copy ends by that signal.
        assert old.wait(30) in (0, -signal.SIGINT)
        time.sleep(0.5)
        stop.set()
        prober.join(10)
        # No connection was ever refused: a copy was always listening.
        assert not [error for error in failures if isinstance(getattr(error, "reason", error), ConnectionRefusedError)]
        handover = Path("/proc/sys/net/ipv4/tcp_migrate_req")
        if handover.is_file() and handover.read_text().strip() == "1":
            assert failures == []  # and with the kernel's handover, nothing was reset either
        else:
            assert len(failures) <= 2  # without it, a connection queued at the closing instant may be reset
        assert "old" in answered and answered[-1] == "new"
        assert {get(port, "/health")["name"] for _ in range(20)} == {"new"}
    finally:
        stop.set()
        for process in (old, new):
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(10)
