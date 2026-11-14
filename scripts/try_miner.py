#!/usr/bin/env python
"""Run the miner on retired tasks with the real model and sandbox, and grade each answer as a validator would.

    python scripts/try_miner.py                    # both sample tasks, once each
    python scripts/try_miner.py --runs 3           # three runs of each
    OPENROUTER_MODEL=anthropic/claude-opus-5.5 python scripts/try_miner.py

Run it from the repository root. Settings come from .env, as for the live miner; an
environment variable overrides its line there. Every run is recorded like a live solve,
with the local grade as its verdict, then summarised by the same report as
scripts/solve_report.py. This spends real OpenRouter credit. Two sample tasks are a
smoke test, not a measurement.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

from rlvr.neurons import demo_miner
from rlvr.neurons.journal import report, utc
from rlvr.neurons.workbench import Sandbox
from rlvr.policy import RELEASE_POLICY
from rlvr.v3.api import MinerSlotSet, MinerTaskRequest
from rlvr.v3.archive import ArchiveLimits, extract_archive
from rlvr.v3.artifacts import ArtifactRef, UploadSlot
from rlvr.v3.identity import RepositoryTaskIdentity, TerminalScriptTaskIdentity, compute_task_id
from rlvr.v3.miner_workspace import WorkspaceReader

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = (
    ROOT / "tests" / "fixtures" / "v3-yamlcpp-e6225d04",
    ROOT / "tests" / "fixtures" / "v3-terminal-urlooker-651460ec",
)
LIMITS = ArchiveLimits(
    max_compressed_bytes=RELEASE_POLICY.v3_workspace_compressed_bytes,
    max_expanded_bytes=RELEASE_POLICY.v3_workspace_bytes,
    max_file_bytes=RELEASE_POLICY.v3_max_file_bytes,
    max_entries=200_000,
    max_path_bytes=4_096,
    max_zstd_window_bytes=128 * 1024**2,
)


def request_for(fixture: Path) -> MinerTaskRequest:
    """The task as a validator would send it, with uploads that go nowhere."""
    raw = json.loads((fixture / "identity.json").read_text())
    terminal = raw.get("task_type") == "terminal_script_v1"
    identity = (TerminalScriptTaskIdentity if terminal else RepositoryTaskIdentity)(**raw)
    references = json.loads((fixture / "artifact_refs.json").read_text())
    workspace = next(
        ArtifactRef(**value) for value in references.values()
        if value.get("artifact_role") in ("workspace", "terminal_environment")
    )
    task_id = compute_task_id(identity)
    challenge_id = f"local-{uuid.uuid4().hex[:16]}"
    expires_at = int(time.time()) + 3_600

    def slot(role: str, artifact_format: str) -> UploadSlot:
        return UploadSlot(
            challenge_id=challenge_id, task_id=task_id, uid=0, hotkey="local-miner",
            artifact_role=role, artifact_format=artifact_format, upload_id=f"{challenge_id}-{role}",
            upload_url="https://local.invalid/upload", expires_at=expires_at, max_bytes=64 * 1024**2,
        )

    return MinerTaskRequest(
        protocol_version=3, challenge_id=challenge_id, task_id=task_id, identity=identity,
        workspace=workspace, workspace_url="https://local.invalid/workspace", expires_at=expires_at,
        slots=MinerSlotSet(
            submission=slot("script", "bash_script_v1") if terminal else slot("patch", "unified_diff_v1"),
            trajectory=slot("trajectory", "trajectory_v1"),
        ),
    )


def grade(fixture: Path, answer: bytes) -> dict:
    """Grade with scripts/try_task.py, as an unprivileged user when run as root."""
    with tempfile.TemporaryDirectory(prefix="try-miner-grade-") as temporary:
        Path(temporary).chmod(0o755)
        path = Path(temporary) / "answer"
        path.write_bytes(answer)
        path.chmod(0o644)
        completed = subprocess.run(
            [str(ROOT / "scripts" / "run_try_task.sh"), "--fixture", str(fixture), "grade", str(path)],
            cwd=ROOT, capture_output=True, text=True, timeout=1_800, check=False,
        )
    text = completed.stdout
    status = re.search(r"^status: (\w+)(?: \((\w+)\))?", text, re.M)
    if status is None:
        raise RuntimeError("the grader gave no status: " + (completed.stderr or text)[-500:])
    summary, _, display = text.partition("failed check:\n")
    checks = re.findall(r"^  (\S+): (passed|failed|skipped)$", summary, re.M)
    return {
        "status": status.group(1),
        "reason_code": "passed" if status.group(1) == "passed" else status.group(2) or "evaluation_failed",
        "failed_check": "\n".join(line[2:] for line in display.splitlines()) or None,
        "checks": f"{sum(outcome == 'passed' for _, outcome in checks)}/{len(checks)}",
    }


async def run_once(miner: demo_miner.DemoMiner, fixture: Path) -> str:
    request = request_for(fixture)
    captured: dict[str, bytes] = {}
    with tempfile.TemporaryDirectory(prefix="try-miner-", dir=miner.work_dir) as temporary:
        root = Path(temporary) / "workspace"
        extract_archive(fixture / "workspace.tar.zst", request.workspace, root, LIMITS, scratch_dir=Path(temporary))

        @asynccontextmanager
        async def workspace(*_args, **_kwargs):
            yield WorkspaceReader(root)

        async def upload(_http, _request, submission, _trajectory, *, allowed_origins):
            captured["submission"] = submission

        demo_miner.open_miner_workspace, demo_miner.upload_miner_result = workspace, upload
        notes = demo_miner.SolveNotes()
        started_at, started = time.time(), time.monotonic()
        status, error = 200, None
        try:
            await miner.solve(request, miner.settings.openrouter_request_timeout_s, notes)
        except Exception as exc:  # noqa: BLE001 - recorded like a live solve that failed
            status, error = 500, f"{type(exc).__name__}: {exc}"[:500]
        elapsed = time.monotonic() - started
        await miner.record_solve(
            "local-fixture", request, notes, status=status, error=error, started_at=started_at, elapsed_s=elapsed,
        )
    spent = f"{elapsed:.0f}s, ${notes.spend.cost_so_far():.3f}, ended {notes.ended_by or 'with an error'}"
    if "submission" not in captured:
        return f"no answer ({error}); {spent}"
    verdict = await asyncio.to_thread(grade, fixture, captured["submission"])
    if verdict["status"] == "abandoned":
        return f"not graded: the local grader could not run; {spent}"
    miner.journal.outcome({
        "time": utc(time.time()), "validator": "local-fixture", "challenge_id": request.challenge_id,
        "task_id": request.task_id, "uid": 0, "reason_code": verdict["reason_code"],
        "failed_check": verdict["failed_check"], "checks": verdict["checks"],
    })
    return f"{verdict['reason_code']}, checks {verdict['checks']}; {spent}, gate {notes.gate or 'not run'}"


async def run_all(settings: demo_miner.DemoMinerSettings, sandbox: Sandbox | None, fixtures: list[Path], runs: int) -> None:
    client = demo_miner.OpenRouterClient(settings)
    miner = demo_miner.DemoMiner(
        settings, client, wallet=SimpleNamespace(hotkey=SimpleNamespace(ss58_address="local-miner")), sandbox=sandbox,
    )
    try:
        for run in range(1, runs + 1):
            for fixture in fixtures:
                print(f"[try-miner] {fixture.name} run {run}: {await run_once(miner, fixture)}", flush=True)
    finally:
        await client.aclose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixture", type=Path, action="append", help="a task fixture directory (default: both samples)")
    parser.add_argument("--runs", type=int, default=1, help="runs of each task")
    parser.add_argument("--record-dir", type=Path, default=ROOT / "data" / "solves-local", help="journal directory")
    parser.add_argument("--cost-cap", type=float, help="spending cap per solve in USD (default: MINER_MAX_SOLVE_COST_USD)")
    args = parser.parse_args(argv)
    overrides: dict = {"miner_record_dir": str(args.record_dir)}
    if args.cost_cap is not None:
        overrides["miner_max_solve_cost_usd"] = args.cost_cap
    settings = demo_miner.DemoMinerSettings(**overrides)
    if not settings.openrouter_api_key:
        print("set OPENROUTER_API_KEY in .env first")
        return 2
    sandbox, status = (
        Sandbox.detect(concurrency=settings.miner_sandbox_concurrency) if settings.miner_sandbox
        else (None, "turned off by MINER_SANDBOX")
    )
    print(f"[try-miner] model {settings.openrouter_model}, effort {settings.openrouter_reasoning_effort}, sandbox: {status}")
    asyncio.run(run_all(settings, sandbox, args.fixture or list(FIXTURES), args.runs))
    print(report(args.record_dir.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
