#!/usr/bin/env python
"""Solve kept failed tasks again with the current miner and grade each answer with the task's own hidden check.

    .venv/bin/python scripts/regression_solve.py data/solves/tasks/<folder> [<folder> ...]
    .venv/bin/python scripts/regression_solve.py --jobs 2 data/solves/tasks/*a6da5e0782c4* ...

Every failed verdict embeds the validator's whole hidden check, and the journal keeps it in the task's
folder (failed_check.txt, check_files/) next to the task (task.json, instruction.md) and its workspace
(workspace.tar.zst). This sends the task to the current DemoMiner exactly as a validator did, with the same
identity and the same lease, sends the uploads nowhere, and reruns that hidden check on the new answer in the
validators' image, as the miner does for a live failure. It is the test that a fix holds on the failure it
was made for, and that a change does not break one fixed earlier.

Runs are recorded in data/solves-regression (never the live journal), answer sharing is off, and each run's
answer, notes and grade are kept in data/solves-regression/runs/<folder>. Run it from the repository root.
This spends real OpenRouter credit, about $1 to $2 a task: check the credit first. One run of a task is
one sample of a model that does not answer the same way twice; a pass shows the fix can hold, not that it
always will.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

from rlvr.neurons import demo_miner
from rlvr.neurons.diagnosis import DiagnosisError, rerun_failed_check
from rlvr.neurons.journal import parse_utc
from rlvr.neurons.workbench import Sandbox
from rlvr.policy import RELEASE_POLICY
from rlvr.v3.api import MinerSlotSet, MinerTaskRequest
from rlvr.v3.archive import ArchiveLimits, extract_archive
from rlvr.v3.artifacts import ArtifactRef, UploadSlot
from rlvr.v3.identity import RepositoryTaskIdentity, compute_task_id
from rlvr.v3.miner_workspace import WorkspaceReader

ROOT = Path(__file__).resolve().parent.parent
LIMITS = ArchiveLimits(
    max_compressed_bytes=RELEASE_POLICY.v3_workspace_compressed_bytes,
    max_expanded_bytes=RELEASE_POLICY.v3_workspace_bytes,
    max_file_bytes=RELEASE_POLICY.v3_max_file_bytes,
    max_entries=200_000,
    max_path_bytes=4_096,
    max_zstd_window_bytes=128 * 1024**2,
)
NEEDED = ("task.json", "instruction.md", "workspace.tar.zst", "failed_check.txt")
# The miner opens a task's workspace and uploads its answer through two module functions. They are replaced
# once, by functions that look the task up by its challenge id, so that tasks solved at the same time never
# read another task's workspace or hand their answer to another task.
ROOTS: dict[str, Path] = {}
CAPTURED: dict[str, bytes] = {}


@asynccontextmanager
async def _workspace(_http, request, *_args, **_kwargs):
    yield WorkspaceReader(ROOTS[request.challenge_id])


async def _upload(_http, request, submission, _trajectory, **_kwargs):
    CAPTURED[request.challenge_id] = submission


def code_version() -> str:
    """A short digest of the solver's source, so runs under different code can be told apart."""
    import hashlib

    digest = hashlib.sha256()
    for path in sorted((ROOT / "rlvr" / "neurons").glob("*.py")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()[:12]


def load_task(folder: Path) -> tuple[dict, RepositoryTaskIdentity, float]:
    """The kept task: its details, its identity with the task text, and the lease the validator gave."""
    missing = [name for name in NEEDED if not (folder / name).is_file()]
    if missing:
        raise ValueError(f"{folder.name}: not kept: {', '.join(missing)}")
    details = json.loads((folder / "task.json").read_text(encoding="utf-8"))
    if details.get("task_type") != "repository_patch_v1":
        raise ValueError(f"{folder.name}: only repository tasks carry a hidden check that can be rerun")
    instruction = (folder / "instruction.md").read_text(encoding="utf-8")
    identity = RepositoryTaskIdentity(**details["identity"], instruction=instruction)
    if compute_task_id(identity) != details["task_id"]:
        # The journal may have changed the text's ending; the task is the same, only its id differs.
        stripped = RepositoryTaskIdentity(**details["identity"], instruction=instruction.rstrip("\n"))
        if compute_task_id(stripped) == details["task_id"]:
            identity = stripped
    lease = parse_utc(details["expires_at"]) - parse_utc(details["received"])
    return details, identity, lease


def request_for(details: dict, identity: RepositoryTaskIdentity, lease: float) -> MinerTaskRequest:
    task_id = compute_task_id(identity)
    challenge_id = f"regression-{uuid.uuid4().hex[:16]}"
    expires_at = int(time.time() + lease)

    def slot(role: str, artifact_format: str) -> UploadSlot:
        return UploadSlot(
            challenge_id=challenge_id, task_id=task_id, uid=0, hotkey="regression-miner",
            artifact_role=role, artifact_format=artifact_format, upload_id=f"{challenge_id}-{role}",
            upload_url="https://local.invalid/upload", expires_at=expires_at, max_bytes=64 * 1024**2,
        )

    return MinerTaskRequest(
        protocol_version=3, challenge_id=challenge_id, task_id=task_id, identity=identity,
        workspace=ArtifactRef.model_validate(details["workspace"]),
        workspace_url="https://local.invalid/workspace", expires_at=expires_at,
        slots=MinerSlotSet(
            submission=slot("patch", "unified_diff_v1"), trajectory=slot("trajectory", "trajectory_v1"),
        ),
    )


async def run_task(miner: demo_miner.DemoMiner, folder: Path, runs_dir: Path, sandbox: Sandbox) -> dict:
    details, identity, lease = load_task(folder)
    request = request_for(details, identity, lease)
    out = runs_dir / f"{folder.name}-{time.strftime('%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    notes = demo_miner.SolveNotes()
    started = time.monotonic()
    error = None
    with tempfile.TemporaryDirectory(prefix="regression-", dir=miner.work_dir) as temporary:
        root = Path(temporary) / "workspace"
        await asyncio.to_thread(
            extract_archive, folder / "workspace.tar.zst", request.workspace, root, LIMITS, scratch_dir=Path(temporary),
        )
        ROOTS[request.challenge_id] = root
        timeout = min(max(0.0, lease - demo_miner.LEASE_MARGIN_S), miner.settings.openrouter_request_timeout_s)
        try:
            await asyncio.wait_for(miner.solve(request, timeout, notes), timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - kept like a live solve that failed
            error = f"{type(exc).__name__}: {exc}"[:300]
        finally:
            ROOTS.pop(request.challenge_id, None)
    elapsed = time.monotonic() - started
    captured = {"submission": CAPTURED.pop(request.challenge_id)} if request.challenge_id in CAPTURED else {}
    fields = notes.fields()
    result = {
        "task": folder.name, "code": code_version(), "seconds": round(elapsed),
        "cost_usd": round(notes.spend.cost_so_far(), 3),
        "ended_by": notes.ended_by or "error", "error": error,
        "stages": {
            key: fields.get(key) for key in ("acceptance", "adjudication", "audit", "surface", "lint", "reviewed")
        },
    }
    if "submission" not in captured:
        result["grade"] = "no answer"
    else:
        (out / "answer.diff").write_bytes(captured["submission"])
        # The probe phase's two versions (miner_probe_phase), kept beside the answer so each can be graded with the
        # task's hidden check (data/research/hidden-suites-2026-11-13/scripts/grade_answer.py).
        for name, patch in (("probe_base.diff", notes.probe_base_patch), ("probed.diff", notes.probed_patch)):
            if patch:
                (out / name).write_bytes(patch)
        if fields.get("probe"):
            result["probe"] = {key: fields.get(key) for key in (
                "probe", "probe_changed", "probe_quotes", "probe_seconds", "probe_cost", "probe_snapshot_used")}
        for name in ("task.json", "workspace.tar.zst"):
            os.symlink(folder / name, out / name)
        try:
            found = await asyncio.to_thread(
                rerun_failed_check, out, (folder / "failed_check.txt").read_text(encoding="utf-8"), sandbox,
            )
            result["grade"] = "passed" if not found.get("failed") and "exactly the required output" in found.get(
                "summary", ""
            ) else "failed"
            result["summary"] = found.get("summary")
            result["failed_tests"] = [
                f"{test.get('test')}: {str(test.get('message', ''))[:200]}" for test in found.get("failed", [])[:8]
            ]
        except DiagnosisError as exc:
            result["grade"] = f"not graded: {exc}"
    (out / "result.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    return result


async def run_all(folders: list[Path], jobs: int, record_dir: Path) -> list[dict]:
    demo_miner.open_miner_workspace, demo_miner.upload_miner_result = _workspace, _upload
    settings = demo_miner.DemoMinerSettings(miner_record_dir=str(record_dir), miner_shared_dir="")
    sandbox, status = Sandbox.detect(concurrency=settings.miner_sandbox_concurrency)
    if sandbox is None:
        raise SystemExit(f"no sandbox: {status}")
    client = demo_miner.OpenRouterClient(settings)
    miner = demo_miner.DemoMiner(
        settings, client, wallet=SimpleNamespace(hotkey=SimpleNamespace(ss58_address="regression-miner")),
        sandbox=sandbox,
    )
    runs_dir = record_dir / "runs"
    gate = asyncio.Semaphore(jobs)
    results: list[dict] = []

    async def one(folder: Path) -> None:
        async with gate:
            try:
                result = await run_task(miner, folder, runs_dir, sandbox)
            except Exception as exc:  # noqa: BLE001 - one task's failure must not stop the others
                result = {"task": folder.name, "grade": f"not run: {type(exc).__name__}: {exc}"[:300]}
            results.append(result)
            line = f"[regression] {result['task']}: {result['grade']}"
            if "cost_usd" in result:
                line += f", ${result['cost_usd']:.2f}, {result['seconds']}s, ended {result['ended_by']}"
            if result.get("summary") and result["grade"] != "passed":
                line += f"\n    {result['summary']}"
            for test in result.get("failed_tests", []):
                line += f"\n      {test}"
            print(line, flush=True)

    try:
        await asyncio.gather(*(one(folder) for folder in folders))
    finally:
        await client.aclose()
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", type=Path, nargs="+", help="kept task folders under data/solves/tasks")
    parser.add_argument("--jobs", type=int, default=1, help="tasks solved at once (default 1)")
    parser.add_argument("--record-dir", type=Path, default=ROOT / "data" / "solves-regression")
    args = parser.parse_args(argv)
    folders = [folder.resolve() for folder in args.folders]
    for folder in folders:
        load_task(folder)  # refuse early when a folder lacks what a regression run needs
    shutil.os.makedirs(args.record_dir, exist_ok=True)
    results = asyncio.run(run_all(folders, max(1, args.jobs), args.record_dir.resolve()))
    passed = sum(result["grade"] == "passed" for result in results)
    spent = sum(result.get("cost_usd", 0.0) for result in results)
    print(f"[regression] passed {passed} of {len(results)}; spent ${spent:.2f}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
