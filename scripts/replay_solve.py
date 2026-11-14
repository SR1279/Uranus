"""Replay a recorded solve's tool calls through the current Workbench, then rerun the validator's hidden check.

Usage: .venv/bin/python scripts/replay_solve.py data/solves/tasks/<folder> <output dir>

Every tool call of the trajectory (the model's and the harness's) runs again, in order, on the task's
original workspace, with the current workbench code. The model's decisions are fixed, so this measures what
the answer would have been with the same actions, not how the model would have reacted to new notices.
"""

from __future__ import annotations

import asyncio
import base64
import gzip
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlvr.neurons.diagnosis import rerun_failed_check  # noqa: E402
from rlvr.neurons.workbench import Sandbox, Workbench  # noqa: E402
from rlvr.v3.miner_workspace import WorkspaceReader  # noqa: E402


def decode(event: dict, key: str):
    raw = event.get(key)
    if not raw:
        return None
    data = base64.b64decode(raw)
    try:
        return json.loads(data)
    except ValueError:
        return None


async def main(folder: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    details = json.loads((folder / "task.json").read_text())
    root = out / "workspace"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(mode=0o700)
    subprocess.run(f"zstd -dc {folder / 'workspace.tar.zst'} | tar -x -C {root}", shell=True, check=True)
    sandbox, status = Sandbox.detect(concurrency=1)
    if sandbox is None:
        raise SystemExit(f"no sandbox: {status}")
    trajectory = json.load(gzip.open(folder / "trajectory.json.gz"))
    calls = [e for e in trajectory["events"] if e["event_type"] == "tool_call"]
    log: list[str] = []
    started = time.monotonic()
    working = details["identity"].get("working_directory", ".")
    with Workbench(WorkspaceReader(root), terminal=False, working_directory=working, sandbox=sandbox) as bench:
        for event in calls:
            arguments = decode(event, "input_body_b64")
            if type(arguments) is not dict or "tool" not in arguments:
                continue
            if arguments["tool"] == "run_command":
                arguments["timeout_s"] = min(int(arguments.get("timeout_s", 120)), 120)
            output = json.loads(await bench.execute(arguments, time_left=600.0))
            tool = arguments["tool"]
            if "error" in output and tool in ("edit_file", "write_file", "revert_file"):
                log.append(f"{event['call_id']} {tool} {arguments.get('path')}: ERROR {output['error']}")
            if tool == "run_command" and ({"answer_updated", "not_in_answer", "answer_kept"} & set(output)):
                shown = {k: output[k] for k in ("answer_updated", "not_in_answer", "answer_kept") if k in output}
                log.append(f"{event['call_id']} run_command exit={output.get('exit_code')} {json.dumps(shown)}")
        patch = bench.submission()
    (out / "answer.diff").write_bytes(patch)
    print(f"replayed {len(calls)} calls in {time.monotonic() - started:.0f}s; answer {len(patch)} bytes "
          f"(recorded answer {len((folder / 'answer.diff').read_bytes())} bytes)")
    for line in log:
        print("  " + line)
    shutil.rmtree(root, ignore_errors=True)
    if not (folder / "failed_check.txt").is_file():
        print("no failed check kept; nothing to rerun")
        return
    for name in ("task.json", "workspace.tar.zst"):
        target = out / name
        if not target.exists():
            os.symlink(folder / name, target)
    display = (folder / "failed_check.txt").read_text()
    result = await asyncio.to_thread(rerun_failed_check, out, display, sandbox)
    print("hidden check on the replayed answer:", result["summary"])
    for test in result["failed"][:12]:
        print("   ", test)


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()))
