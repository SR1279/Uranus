"""What each solve did and cost, and how validators graded it.

The miner keeps everything in one directory, MINER_RECORD_DIR:

- solves.jsonl, one line per task it answered or failed to answer;
- outcomes.jsonl, one line per verdict a validator sent back: passed, or why not;
- diagnoses.jsonl, one line per failed check the miner reran on its own answer;
- refused.jsonl, one line per task it was sent but could not take on;
- tasks/, one folder per task: what it asked, the answer, the trajectory, the failing
  check and what a rerun of it found, and the workspace: until the verdict, then the
  newest failed ones for a rerun and the newest passed ones as known-good cases;
- LOG.md and tasks.csv: every task, passed or failed, why, what kind it was and what
  it cost, rewritten after every task, verdict and rerun.

`report` and `render_log` join them and sort every failure by cause, so the model is
only changed for failures a better model could fix.
"""

from __future__ import annotations

import calendar
import csv
import gzip
import io
import json
import math
import os
import re
import shutil
import time
import zlib
from base64 import b64decode
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

KEEP_TRAJECTORIES = 1_000
KEEP_TASKS = 1_000
# A failed task's workspace with the hidden tests its verdict returned lets any later answer be graded exactly
# as a validator would; at about 40 MB each, 120 is under 5 GB.
KEEP_FAILED_WORKSPACES = 120
# A passed task's workspace and answer are a known-good case: a suite written from the documentation alone
# must pass on them, which is how the miner's own test writer is measured. The newest are kept, within a
# disk budget, since a workspace can be hundreds of megabytes.
KEEP_PASSED_WORKSPACES = 60
PASSED_WORKSPACE_BYTES = 3 * 1024**3
# A workspace kept for a rerun is dropped when no verdict has come this long after the task.
WORKSPACE_WAIT_S = 24 * 3600
LOGGED_TASKS = 1_000
LISTED_FILES = 8
LISTED_TESTS = 12

TASK_FILE = "task.json"
INSTRUCTION_FILE = "instruction.md"
WORKSPACE_FILE = "workspace.tar.zst"
CREDIT_FILE = "credit.json"
CREDIT_LOW_USD = 25.0


def _credit_line(directory: Path) -> str | None:
    """What LOG.md says first about the account's OpenRouter credit, from the miner's latest reading."""
    try:
        state = json.loads((directory / CREDIT_FILE).read_text(encoding="utf-8"))
        remaining = float(state["remaining_usd"])
        checked = str(state["checked"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    hours = state.get("hours_left")
    pace = f", about {hours:g} hours at the recent pace" if isinstance(hours, (int, float)) else ""
    if remaining <= 0:
        return (f"**OpenRouter credit is used up** (read {checked}): every model call fails, so every task fails, "
                "until the account is topped up. The SN67 miners use the same account.")
    if remaining < CREDIT_LOW_USD:
        return (f"**OpenRouter credit is low: ${remaining:.2f} left** (read {checked}){pace}. Every task fails once it "
                "is gone, and the SN67 miners use the same account.")
    return f"OpenRouter credit: ${remaining:.2f} left (read {checked}){pace}."
TRAJECTORY_FILE = "trajectory.json.gz"
CHECK_FILE = "failed_check.txt"
DIAGNOSIS_FILE = "diagnosis.txt"
CHECK_FILES_DIR = "check_files"
PATCH_FILE = "answer.diff"
SCRIPT_FILE = "answer.sh"
FOLDER_FILES = (
    INSTRUCTION_FILE, PATCH_FILE, SCRIPT_FILE, TRAJECTORY_FILE, CHECK_FILE, DIAGNOSIS_FILE,
    CHECK_FILES_DIR, WORKSPACE_FILE,
)

# The answer never reached the hidden checks, so a better model would not have helped.
HARNESS_REASONS = frozenset({
    "not_serving", "dispatch_failed", "response_unavailable", "slot_mismatch",
    "artifact_invalid", "trajectory_invalid", "patch_rejected", "patch_static_rejected",
    "patch_apply_failed", "script_rejected", "working_directory_invalid", "result_tree_invalid",
})
LIMIT_REASONS = frozenset({"timeout", "memory_limit", "output_limit"})
STOPPED_EARLY = frozenset({"deadline", "wrap_up", "cost_cap", "tool_budget", "model_error", "model_refused"})
FAILURE_CAUSES = ("reasoning", "stopped early", "harness", "miner error", "grading limits", "build failed", "other")

REASONS = {
    "passed": "passed every hidden check",
    "check_failed": "a hidden check printed something other than the required output",
    "timeout": "a check or the script ran past its time limit (hidden tests get about 120 s, compilation included), so the answer was too slow at the documented sizes or hung",
    "memory_limit": "a check or the script ran out of memory",
    "output_limit": "a check or the script printed more than the grader allows",
    "setup_failed": "the task's setup step failed with the answer applied",
    "patch_rejected": "the patch was refused before it was applied",
    "patch_static_rejected": "the patch holds something the grader refuses: binary data, a rename or a copy",
    "patch_apply_failed": "git apply could not apply the patch to the workspace",
    "script_rejected": "the script was refused before it ran",
    "working_directory_invalid": "the answer removed or replaced the task's working directory",
    "result_tree_invalid": "the answer left something the grader refuses, such as a link or a special file",
    "not_serving": "the validator could not reach the miner",
    "dispatch_failed": "the task could not be delivered to the miner",
    "response_unavailable": "the validator did not get a usable answer in time",
    "slot_mismatch": "the answer was uploaded for a different task or miner",
    "artifact_invalid": "the uploaded answer was missing or did not match its digest",
    "trajectory_invalid": "the uploaded trajectory was refused",
    "evaluation_failed": "the validator could not finish grading",
}
CAUSES = {
    "reasoning": "the answer was wrong after a normal finish; with grading limits, the kind a better loop or a stronger model can improve",
    "stopped early": "time, the spending cap, the tool budget, a model error or the host's refusal cut the solve short",
    "harness": "the answer was refused before any check ran",
    "miner error": "the miner returned no answer",
    "grading limits": "the answer hit the grader's time, memory or output limit: nearly always a design too slow or too hungry at the documented limits, which the loop or the model fixes, not the validator",
    "build failed": "the task's setup failed with the answer applied",
    "other": "the validator could not finish grading",
}


def utc(timestamp: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp))


def parse_utc(text: Any) -> float | None:
    if not isinstance(text, str):
        return None
    try:
        return float(calendar.timegm(time.strptime(text, "%Y-%m-%dT%H:%M:%SZ")))
    except ValueError:
        return None


def _label(value: Any, size: int) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(value))[:size]


def folder_name(started_at: float, task_id: str, challenge_id: str) -> str:
    """A task's folder: when it started, then its task and challenge, so names sort by time."""
    return f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime(started_at))}-{_label(task_id, 12)}-{_label(challenge_id, 8)}"


def answer_file(task_type: Any) -> str:
    return SCRIPT_FILE if task_type == "terminal_script_v1" else PATCH_FILE


def instruction_title(instruction: str, limit: int = 200) -> str:
    """The first line of a task's instruction, which names the task."""
    for line in instruction.splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            return line if len(line) <= limit else line[: limit - 1] + "…"
    return ""


def changed_files(patch: bytes) -> list[dict[str, Any]]:
    """Each file a Git diff changes, with how many lines it adds and removes."""
    files: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    in_hunk = False
    for raw in patch.split(b"\n"):
        if raw.startswith(b"diff --git "):
            line = raw.decode("utf-8", "replace")
            match = re.fullmatch(r"diff --git a/(.*) b/(.*)", line)
            current = {"path": match.group(2) if match else line[11:], "added": 0, "removed": 0}
            files.append(current)
            in_hunk = False
        elif current is None:
            continue
        elif raw.startswith(b"@@"):
            in_hunk = True
        elif in_hunk and raw.startswith(b"+"):
            current["added"] += 1
        elif in_hunk and raw.startswith(b"-"):
            current["removed"] += 1
    return files


def script_lines(script: bytes) -> int:
    return script.count(b"\n") + (1 if script and not script.endswith(b"\n") else 0)


def _model_requests(trajectory: dict[str, Any]):
    for event in trajectory.get("events") or ():
        if isinstance(event, dict) and event.get("event_type") == "model_turn":
            try:
                body = json.loads(b64decode(event.get("request_body_b64") or "", validate=True))
            except (ValueError, TypeError):
                continue
            if isinstance(body, dict):
                yield body


def instruction_from_trajectory(trajectory: dict[str, Any]) -> str | None:
    """The task's instruction: the first user message of a recorded model request."""
    for body in _model_requests(trajectory):
        for message in body.get("messages") or ():
            if isinstance(message, dict) and message.get("role") == "user" and isinstance(message.get("content"), str):
                return message["content"]
    return None


def answer_from_trajectory(trajectory: dict[str, Any]) -> bytes | None:
    """The submitted answer, which every trajectory ends by validating."""
    for event in reversed(trajectory.get("events") or ()):
        if isinstance(event, dict) and event.get("tool_name") == "validate_submission":
            try:
                return b64decode(event.get("input_body_b64") or "", validate=True)
            except (ValueError, TypeError):
                return None
    return None


def _write(path: Path, data: bytes) -> None:
    path.write_bytes(data)


def _write_atomic(path: Path, text: str) -> None:
    """Replace a file whole, so a reader never sees half of it."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _key(record: dict[str, Any]) -> tuple[Any, Any]:
    return record.get("challenge_id"), record.get("task_id")


def _latest(records: list[dict[str, Any]]) -> dict[tuple[Any, Any], dict[str, Any]]:
    return {_key(record): record for record in records}


def _folder_ids(folder: Path) -> tuple[Any, Any] | None:
    try:
        details = json.loads((folder / TASK_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return _key(details) if isinstance(details, dict) else None


def _note_verdict(folder: Path, outcome: dict[str, Any]) -> None:
    """Keep the failing check beside the task. Which workspaces stay is left to SolveJournal.prune."""
    display = outcome.get("failed_check")
    if isinstance(display, str) and display and not (folder / CHECK_FILE).exists():
        _write(folder / CHECK_FILE, display.encode("utf-8"))


def task_folder(directory: Path, solve: dict[str, Any]) -> Path | None:
    """The folder that holds a task's files, when it has one."""
    relative = solve.get("task_dir")
    if isinstance(relative, str) and relative:
        candidate = directory / relative
    else:
        # Solves recorded before task folders existed get one named from the record.
        started = parse_utc(solve.get("time"))
        if started is None or not solve.get("task_id"):
            return None
        candidate = directory / "tasks" / folder_name(started, str(solve["task_id"]), str(solve.get("challenge_id", "")))
    try:
        if not candidate.resolve().is_relative_to(directory.resolve()):
            return None
    except OSError:
        return None
    return candidate if candidate.is_dir() else None


class SolveJournal:
    """Append-only records and per-task folders in one directory."""

    def __init__(
        self,
        directory: Path,
        *,
        keep_trajectories: int = KEEP_TRAJECTORIES,
        keep_tasks: int = KEEP_TASKS,
        keep_failed_workspaces: int = KEEP_FAILED_WORKSPACES,
        keep_passed_workspaces: int = KEEP_PASSED_WORKSPACES,
        passed_workspace_bytes: int = PASSED_WORKSPACE_BYTES,
    ):
        self.directory = directory
        self.trajectories = directory / "trajectories"
        self.tasks = directory / "tasks"
        self.keep_trajectories = keep_trajectories
        self.keep_tasks = keep_tasks
        self.keep_failed_workspaces = keep_failed_workspaces
        self.keep_passed_workspaces = keep_passed_workspaces
        self.passed_workspace_bytes = passed_workspace_bytes
        self.trajectories.mkdir(parents=True, exist_ok=True)
        self.tasks.mkdir(parents=True, exist_ok=True)

    def _append(self, name: str, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=str) + "\n"
        with (self.directory / name).open("a", encoding="utf-8") as handle:
            handle.write(line)

    def solve(self, record: dict[str, Any]) -> None:
        self._append("solves.jsonl", record)

    def outcome(self, record: dict[str, Any], folder: Path | None = None) -> None:
        self._append("outcomes.jsonl", record)
        if folder is not None:
            _note_verdict(folder, record)

    def diagnosis(self, record: dict[str, Any]) -> None:
        self._append("diagnoses.jsonl", record)

    def refused(self, record: dict[str, Any]) -> None:
        self._append("refused.jsonl", record)

    def credit(self, state: dict[str, Any]) -> None:
        """The latest reading of the account's OpenRouter credit, for the top of LOG.md."""
        _write_atomic(self.directory / CREDIT_FILE, json.dumps(state, sort_keys=True) + "\n")

    def relative(self, path: Path) -> str:
        return path.relative_to(self.directory).as_posix()

    def open_task(
        self, task_id: str, challenge_id: str, started_at: float, details: dict[str, Any], instruction: str = ""
    ) -> Path:
        """A new folder for one task, holding what it asked; the oldest folders are dropped."""
        name = folder_name(started_at, task_id, challenge_id)
        folder = self.tasks / name
        for suffix in range(1, 100):
            try:
                folder.mkdir(parents=True)
                break
            except FileExistsError:
                folder = self.tasks / f"{name}-{suffix}"
        else:
            raise OSError("no free name for the task folder")
        _write(folder / TASK_FILE, json.dumps(details, ensure_ascii=False, indent=1, default=str).encode("utf-8"))
        if instruction:
            _write(folder / INSTRUCTION_FILE, instruction.encode("utf-8"))
        self.prune()
        return folder

    def keep(self, folder: Path, name: str, data: bytes) -> str:
        _write(folder / name, data)
        return self.relative(folder / name)

    def trajectory(self, task_id: str, challenge_id: str, data: bytes, folder: Path | None = None) -> str:
        """Keep a trajectory, in the task's folder or else oldest dropped first; returns its path in the journal."""
        if folder is not None:
            path = folder / TRAJECTORY_FILE
            path.write_bytes(gzip.compress(data, compresslevel=6))
            return self.relative(path)
        label = re.sub(r"[^A-Za-z0-9_.-]", "_", challenge_id)[:40]
        path = self.trajectories / f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-{task_id[:16]}-{label}.json.gz"
        path.write_bytes(gzip.compress(data, compresslevel=6))
        kept = sorted(self.trajectories.glob("*.json.gz"))
        for old in kept[: max(0, len(kept) - self.keep_trajectories)]:
            old.unlink(missing_ok=True)
        return path.relative_to(self.directory).as_posix()

    def find_solve(self, challenge_id: str, task_id: str) -> dict[str, Any] | None:
        for record in reversed(read_records(self.directory / "solves.jsonl")):
            if _key(record) == (challenge_id, task_id):
                return record
        return None

    def folder_of(self, solve: dict[str, Any] | None) -> Path | None:
        return None if solve is None else task_folder(self.directory, solve)

    def forget_workspace(self, folder: Path) -> None:
        (folder / WORKSPACE_FILE).unlink(missing_ok=True)

    def prune(self, now: float | None = None) -> None:
        """Drop the oldest task folders, and every workspace neither a rerun nor a known-good case needs.

        A workspace is kept until its verdict. A failure keeps it for a rerun and a pass keeps it as a
        known-good case, only the newest few of each (passes also within a disk budget, newest first);
        one with no verdict after a day is dropped.
        """
        now = time.time() if now is None else now
        folders = sorted(path for path in self.tasks.iterdir() if path.is_dir())
        cut = max(0, len(folders) - self.keep_tasks)
        for old in folders[:cut]:
            shutil.rmtree(old, ignore_errors=True)
        verdicts = _latest(read_records(self.directory / "outcomes.jsonl"))
        failed: list[Path] = []
        passed: list[tuple[Path, int]] = []
        for folder in folders[cut:]:
            workspace = folder / WORKSPACE_FILE
            try:
                status = workspace.stat()
            except OSError:
                continue
            verdict = verdicts.get(_folder_ids(folder) or ("", ""))
            if verdict is None:
                if now - status.st_mtime > WORKSPACE_WAIT_S:
                    workspace.unlink(missing_ok=True)
            elif verdict.get("reason_code") == "passed":
                passed.append((workspace, status.st_size))
            else:
                failed.append(workspace)
        for workspace in failed[: max(0, len(failed) - self.keep_failed_workspaces)]:
            workspace.unlink(missing_ok=True)
        kept = kept_bytes = 0
        for workspace, size in reversed(passed):
            if kept < self.keep_passed_workspaces and kept_bytes + size <= self.passed_workspace_bytes:
                kept += 1
                kept_bytes += size
            else:
                workspace.unlink(missing_ok=True)

    def backfill(self) -> None:
        """Give every recorded task its folder and keep each failing check beside its task.

        Tasks recorded before folders existed get one rebuilt from their trajectory,
        which holds the instruction and the answer.
        """
        verdicts = _latest(read_records(self.directory / "outcomes.jsonl"))
        for solve in read_records(self.directory / "solves.jsonl"):
            folder = task_folder(self.directory, solve)
            if folder is None and not solve.get("task_dir"):
                folder = self._rebuild_folder(solve)
            verdict = verdicts.get(_key(solve))
            if folder is not None and verdict is not None:
                _note_verdict(folder, verdict)

    def _rebuild_folder(self, solve: dict[str, Any]) -> Path | None:
        started = parse_utc(solve.get("time"))
        trajectory = solve.get("trajectory")
        if started is None or not solve.get("task_id") or not isinstance(trajectory, str):
            return None
        source = self.directory / trajectory
        try:
            data = json.loads(gzip.decompress(source.read_bytes()))
        except (OSError, EOFError, ValueError, zlib.error):
            return None
        if not isinstance(data, dict):
            return None
        folder = self.tasks / folder_name(started, str(solve["task_id"]), str(solve.get("challenge_id", "")))
        folder.mkdir(parents=True, exist_ok=True)
        details = {name: solve.get(name) for name in ("validator", "challenge_id", "task_id", "task_type", "task_kind", "language")}
        details["rebuilt_from"] = trajectory
        _write(folder / TASK_FILE, json.dumps(details, ensure_ascii=False, indent=1).encode("utf-8"))
        instruction = instruction_from_trajectory(data)
        if instruction:
            _write(folder / INSTRUCTION_FILE, instruction.encode("utf-8"))
        answer = answer_from_trajectory(data)
        if answer is not None:
            _write(folder / answer_file(solve.get("task_type")), answer)
        try:
            os.link(source, folder / TRAJECTORY_FILE)
        except OSError:
            shutil.copyfile(source, folder / TRAJECTORY_FILE)
        return folder

    def write_log(self, now: float | None = None) -> None:
        """Rewrite LOG.md and tasks.csv from the records."""
        self.backfill()
        tasks, refused = load_tasks(self.directory)
        now = time.time() if now is None else now
        _write_atomic(self.directory / "LOG.md", _render_log(self.directory, tasks, refused, now))
        _write_atomic(self.directory / "tasks.csv", _render_csv(self.directory, tasks))


def read_records(path: Path) -> list[dict[str, Any]]:
    """Every whole record in a JSONL file; a line cut short by a crash is skipped."""
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def cause(solve: dict[str, Any] | None, outcome: dict[str, Any] | None) -> str:
    """Why a task ended as it did, in terms of what would fix it."""
    if solve is not None and solve.get("status") != 200:
        return "miner error"
    if outcome is None:
        return "no verdict"
    reason = outcome.get("reason_code")
    if reason == "passed":
        return "passed"
    if solve is not None and solve.get("ended_by") in STOPPED_EARLY:
        return "stopped early"
    if reason in LIMIT_REASONS:
        return "grading limits"
    if reason in HARNESS_REASONS:
        return "harness"
    if reason == "setup_failed":
        return "build failed"
    if reason == "check_failed":
        return "reasoning"
    return "other"


def wilson(passes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """A 95% range for a pass rate measured on `total` tasks."""
    if total == 0:
        return 0.0, 1.0
    rate = passes / total
    scale = 1 + z * z / total
    centre = (rate + z * z / (2 * total)) / scale
    half = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / scale
    return max(0.0, centre - half), min(1.0, centre + half)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))] if ordered else 0.0


@dataclass
class Task:
    """One recorded solve, with its verdict, its rerun and its folder when there are any."""

    solve: dict[str, Any]
    outcome: dict[str, Any] | None = None
    diagnosis: dict[str, Any] | None = None
    folder: Path | None = None

    @property
    def cause(self) -> str:
        return cause(self.solve, self.outcome)

    @property
    def verdict(self) -> str:
        return {"passed": "PASSED", "no verdict": "WAITING", "miner error": "MINER ERROR"}.get(self.cause, "FAILED")

    @property
    def title(self) -> str:
        title = self.solve.get("title")
        if isinstance(title, str) and title:
            return title
        if self.folder is not None:
            try:
                return instruction_title((self.folder / INSTRUCTION_FILE).read_text(encoding="utf-8", errors="replace"))
            except OSError:
                pass
        return ""

    @property
    def changes(self) -> list[dict[str, Any]]:
        recorded = self.solve.get("changed_files")
        if isinstance(recorded, list):
            return recorded
        if self.folder is not None and self.solve.get("task_type") != "terminal_script_v1":
            try:
                return changed_files((self.folder / PATCH_FILE).read_bytes())
            except OSError:
                pass
        return []


def load_tasks(directory: Path) -> tuple[list[Task], list[dict[str, Any]]]:
    """Every recorded solve joined to its verdict and rerun, oldest first, and the refused tasks."""
    outcomes = _latest(read_records(directory / "outcomes.jsonl"))
    diagnoses = _latest(read_records(directory / "diagnoses.jsonl"))
    tasks = [
        Task(solve, outcomes.get(_key(solve)), diagnoses.get(_key(solve)), task_folder(directory, solve))
        for solve in read_records(directory / "solves.jsonl")
    ]
    return tasks, read_records(directory / "refused.jsonl")


def kind_of(record: dict[str, Any]) -> str:
    """The kind of task, such as "python feature", "rust bug fix" or "terminal script"."""
    if record.get("task_type") == "terminal_script_v1":
        return "terminal script"
    kind = str(record.get("task_kind") or "task").replace("_", " ")
    return f"{record.get('language') or 'unknown-language'} {kind}"


def _cost(solve: dict[str, Any]) -> float | None:
    cost = solve.get("cost_usd")
    return float(cost) if isinstance(cost, (int, float)) and not isinstance(cost, bool) else None


def _seconds(solve: dict[str, Any]) -> float:
    latency = solve.get("latency_ms")
    return latency / 1000 if isinstance(latency, (int, float)) else 0.0


def _short(value: Any, size: int = 12) -> str:
    return str(value)[:size] if value else "?"


def _when(record: dict[str, Any]) -> str:
    return str(record.get("time") or "?").replace("T", " ").replace("Z", " UTC")


def _duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def _group_text(name: str, group: list[Task]) -> str:
    graded = [task for task in group if task.cause != "no verdict"]
    passed = sum(task.cause == "passed" for task in graded)
    costs = [cost for cost in (_cost(task.solve) for task in group) if cost is not None]
    text = f"{name}: {len(group)} received, passed {passed} of {len(graded)} graded"
    if costs:
        text += f", ${sum(costs) / len(costs):.3f} per solve"
    return text + f", median {_percentile([_seconds(task.solve) for task in group], 0.5):.0f}s"


def _groups(tasks: list[Task], key: Callable[[Task], str]) -> list[tuple[str, list[Task]]]:
    groups: dict[str, list[Task]] = {}
    for task in tasks:
        groups.setdefault(key(task), []).append(task)
    return sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))


def _overview(tasks: list[Task], refused: list[dict[str, Any]], now: float) -> list[str]:
    """The summary: indented lines belong to the line above them."""
    causes = Counter(task.cause for task in tasks)
    graded = len(tasks) - causes["no verdict"]
    passed = causes["passed"]
    failed = graded - passed
    lines = [f"Solves recorded: {len(tasks)}; graded: {graded}; waiting for a verdict: {causes['no verdict']}"]
    starts = [start for start in (parse_utc(task.solve.get("time")) for task in tasks) if start is not None]
    if starts:
        received = f"Received from {utc(min(starts))} to {utc(max(starts))}"
        if refused:
            received += f"; {len(refused)} more could not be taken on (listed at the end of LOG.md)"
        lines.append(received)
    if graded:
        low, high = wilson(passed, graded)
        lines.append(f"Passed: {passed} of {graded}, {passed / graded:.1%} (95% range {low:.1%} to {high:.1%})")
    costs = [cost for cost in (_cost(task.solve) for task in tasks) if cost is not None]
    if costs:
        total = sum(costs)
        per_solve = total / len(costs)
        per_pass = f", ${total / passed:.3f} per pass" if passed else ""
        lines.append(f"Spend: ${total:.2f} in total, ${per_solve:.3f} per solve{per_pass}")
        recent = [task for task in tasks if (parse_utc(task.solve.get("time")) or 0) >= now - 86_400]
        lines.append(f"  last 24 hours: ${sum(_cost(task.solve) or 0.0 for task in recent):.2f} on {len(recent)} tasks")
        if starts:
            per_day = len(tasks) / max(1.0, (now - min(starts)) / 3600) * 24
            lines.append(f"  pace so far: about {per_day:.0f} tasks a day, which at ${per_solve:.2f} each is about ${per_day * per_solve:.0f} a day")
        input_tokens = sum(int(task.solve.get("input_tokens") or 0) for task in tasks)
        if input_tokens:
            cached = sum(int(task.solve.get("cached_tokens") or 0) for task in tasks)
            output = sum(int(task.solve.get("output_tokens") or 0) for task in tasks)
            lines.append(f"  tokens: {input_tokens:,} input, {cached / input_tokens:.0%} of it from the cache; {output:,} output")
    seconds = [_seconds(task.solve) for task in tasks]
    timing = (
        f"Time per solve: median {_percentile(seconds, 0.5):.0f}s, 90th percentile {_percentile(seconds, 0.9):.0f}s, "
        f"longest {max(seconds):.0f}s"
    )
    allowed = [float(task.solve["time_allowed_s"]) for task in tasks if isinstance(task.solve.get("time_allowed_s"), (int, float))]
    if allowed:
        timing += f"; median time allowed {_percentile(allowed, 0.5):.0f}s"
    lines.append(timing)
    lines.append("By kind of task:")
    lines.extend("  " + _group_text(name, group) for name, group in _groups(tasks, lambda task: kind_of(task.solve)))
    lines.append("By validator:")
    lines.extend("  " + _group_text(name, group) for name, group in _groups(tasks, lambda task: _short(task.solve.get("validator"), 16)))
    ended = Counter(task.solve.get("ended_by") or "error" for task in tasks)
    lines.append("How solves ended: " + ", ".join(f"{name} {count}" for name, count in ended.most_common()))
    lines.append(f"Failures by cause: {failed}")
    lines.extend(f"  {name}: {causes[name]}" for name in FAILURE_CAUSES if causes[name])
    answer_errors = causes["reasoning"] + causes["grading limits"]
    if graded < 100:
        lines.append(f"Decision: only {graded} graded tasks; measure at least 100 before changing the model.")
    elif answer_errors * 2 > failed:
        lines.append("Decision: most failures are reasoning or performance errors in the answers, which a better loop or a stronger model may fix.")
    else:
        lines.append("Decision: most failures are harness or early-stop problems; fix those before paying for a stronger model.")
    return lines


def _rerun_lines(diagnosis: dict[str, Any], indent: str) -> list[str]:
    if diagnosis.get("error"):
        return [f"the failing check could not be rerun: {diagnosis['error']}"]
    lines = [str(diagnosis.get("summary") or "the rerun found nothing to report")]
    tests = diagnosis.get("failed") or []
    for item in tests[:LISTED_TESTS]:
        if isinstance(item, dict):
            message = f" — {item['message']}" if item.get("message") else ""
            lines.append(f"{indent}{item.get('test')}: {item.get('result')}{message}")
    total = diagnosis.get("failed_total", len(tests))
    if isinstance(total, int) and total > min(len(tests), LISTED_TESTS):
        lines.append(f"{indent}and {total - min(len(tests), LISTED_TESTS)} more in {DIAGNOSIS_FILE}")
    return lines


def _verdict_text(task: Task) -> str:
    solve, outcome = task.solve, task.outcome
    if task.cause == "miner error":
        text = f"none; the miner returned no answer (HTTP {solve.get('status')}: {solve.get('error') or 'no detail'})"
        if outcome:
            text += f"; the validator reported {outcome.get('reason_code')}"
        return text
    if outcome is None:
        return "none yet"
    reason = str(outcome.get("reason_code"))
    text = f"{reason}: {REASONS.get(reason, 'see the validator')}"
    received = parse_utc(outcome.get("time"))
    started = parse_utc(solve.get("time"))
    if received is not None:
        text += f"; received {_when(outcome)}"
        if started is not None:
            text += f", {_duration(max(0.0, received - started - _seconds(solve)))} after the answer"
    if task.cause != "passed":
        text += f". Cause: {task.cause}, {CAUSES.get(task.cause, '')}"
    return text


def _why_lines(task: Task) -> list[str]:
    if task.cause in ("passed", "no verdict", "miner error"):
        return []
    lines = []
    if task.cause == "stopped early":
        lines.append(f"- Why: the solve stopped early ({task.solve.get('ended_by')}), so the answer may be unfinished")
    outcome = task.outcome or {}
    if task.diagnosis is not None:
        first, *rest = _rerun_lines(task.diagnosis, "  - ")
        lines += [f"- Rerun of the failing check: {first}", *rest]
    elif outcome.get("reason_code") == "check_failed":
        if outcome.get("failed_check"):
            pending = task.folder is not None and (task.folder / WORKSPACE_FILE).is_file()
            lines.append(
                f"- Failing check: in {CHECK_FILE}, with the output it required"
                + ("; a rerun on the answer is pending" if pending else "; the workspace was not kept, so it was not rerun")
            )
        else:
            lines.append("- Failing check: not shown by the validator (a terminal task's checks never are)")
    return lines


def _work_text(solve: dict[str, Any]) -> str:
    allowed = solve.get("time_allowed_s")
    parts = [
        f"ended {solve.get('ended_by') or 'with an error'} after {_seconds(solve):.0f}s"
        + (f" of {allowed:.0f}s allowed" if isinstance(allowed, (int, float)) else ""),
        f"{solve.get('model_calls', 0)} model calls",
    ]
    tools = solve.get("tools") or {}
    if isinstance(tools, dict) and tools:
        parts.append("tools " + ", ".join(f"{name} {count}" for name, count in sorted(tools.items(), key=lambda item: (-item[1], item[0]))))
    if solve.get("command_failures"):
        parts.append(f"{solve['command_failures']} commands exited non-zero")
    errors = solve.get("tool_errors") or {}
    if isinstance(errors, dict) and errors:
        parts.append("tool errors " + ", ".join(f"{name} {count}" for name, count in sorted(errors.items())))
    if solve.get("reasks"):
        parts.append(f"answer sent back {solve['reasks']} times to fix")
    if solve.get("reviewed"):
        parts.append("changed in review" if solve.get("changed_in_review") else "review changed nothing")
    if solve.get("gate"):
        parts.append(f"clean-copy check {solve['gate']}")
    if solve.get("acceptance"):
        summary = solve.get("acceptance_summary")
        parts.append(f"independent suite {solve['acceptance']}" + (f" ({summary})" if summary else ""))
        runs = solve.get("acceptance_runs") or 0
        if runs > 1:
            final = solve.get("acceptance_final") or "unknown"
            refuted = solve.get("acceptance_refuted") or 0
            rejected = solve.get("acceptance_rejected") or 0
            parts.append(
                f"suite ran {runs} times, {final} at the end"
                + (f", {refuted} case(s) refuted from the documentation" if refuted else "")
                + (f", {rejected} quotation(s) not found in it" if rejected else "")
            )
        if solve.get("clean_copy") == "worse":
            parts.append("did worse on a clean copy than in the working copy")
        if solve.get("kept_best"):
            parts.append(f"sent an earlier version the suite scored better ({solve['kept_best']})")
    if solve.get("retest_reminded"):
        parts.append("sent back to test an edit it had not run")
    if solve.get("left_out"):
        parts.append(f"asked about {solve['left_out']} file(s) only a command created")
    if solve.get("adjudication"):
        parts.append(f"open readings: {solve['adjudication']}")
    if solve.get("surface"):
        sent_back = solve.get("surface_failures") or 0
        parts.append(f"documented imports {solve['surface']}" + (f" after {sent_back} sent back" if sent_back and solve["surface"] == "passed" else ""))
    if solve.get("lint"):
        sent_back = solve.get("lint_findings") or 0
        parts.append(f"diff check {solve['lint']}" + (f" after {sent_back} sent back" if sent_back and solve["lint"] == "passed" else ""))
    if solve.get("own_tests"):
        parts.append(f"own tests on the final version: {solve['own_tests']}")
    if solve.get("imports") and solve["imports"] not in ("same", "nothing to compare"):
        parts.append(f"import check: {solve['imports']}")
    if solve.get("toolchain") and solve["toolchain"] not in ("parses", "nothing to parse", "no project parser"):
        parts.append(f"toolchain check: {solve['toolchain']}")
    if solve.get("shared_from"):
        parts.append(f"answer reused from the solve by hotkey {str(solve['shared_from'])[:16]}, no model calls here")
    if solve.get("audit"):
        parts.append(f"audit of the diff: {solve['audit']}")
    if solve.get("refused"):
        parts.append(f"the host refused the {solve['refused']} prompt, withdrawn")
    if solve.get("cut_off"):
        parts.append(f"{solve['cut_off']} replies cut at the token limit")
    if solve.get("command_edits_taken"):
        parts.append(f"{solve['command_edits_taken']} answer files changed by commands, taken into the answer")
    if solve.get("command_edits_undone"):
        parts.append(f"{solve['command_edits_undone']} command changes to files outside the answer undone")
    if solve.get("script_exit") is not None:
        parts.append(f"last script run exited {solve['script_exit']}")
    return "; ".join(parts)


def _cost_text(solve: dict[str, Any]) -> str:
    cost = _cost(solve)
    text = f"${cost:.3f}" if cost is not None else "unknown"
    if solve.get("cost_complete") is False:
        text += " (some calls reported no cost)"
    input_tokens = int(solve.get("input_tokens") or 0)
    if input_tokens:
        cached = int(solve.get("cached_tokens") or 0)
        text += f"; {input_tokens:,} input tokens, {cached / input_tokens:.0%} from the cache; {int(solve.get('output_tokens') or 0):,} output tokens"
    hosts = solve.get("hosts") or {}
    if isinstance(hosts, dict) and hosts:
        text += "; served by " + ", ".join(sorted(hosts))
    return text


def _answer_text(task: Task) -> str | None:
    solve = task.solve
    size = solve.get("submission_bytes") or 0
    if solve.get("task_type") == "terminal_script_v1":
        lines = solve.get("answer_lines")
        if lines is None and task.folder is not None:
            try:
                lines = script_lines((task.folder / SCRIPT_FILE).read_bytes())
            except OSError:
                lines = None
        return f"a script of {lines} lines, {size:,} bytes" if lines is not None else (f"{size:,} bytes" if size else None)
    files = task.changes
    if not files:
        return f"{size:,} bytes" if size else None
    added = sum(int(item.get("added", 0)) for item in files)
    removed = sum(int(item.get("removed", 0)) for item in files)
    shown = ", ".join(f"{item.get('path')} (+{item.get('added', 0)} -{item.get('removed', 0)})" for item in files[:LISTED_FILES])
    more = f" and {len(files) - LISTED_FILES} more" if len(files) > LISTED_FILES else ""
    return f"{size:,} bytes changing {len(files)} files, +{added} -{removed} lines: {shown}{more}"


def _entry(directory: Path, task: Task) -> list[str]:
    solve = task.solve
    lines = [f"### {_when(solve)} · {task.verdict} · {kind_of(solve)}", ""]
    if task.title:
        lines.append(f"- Asked: {task.title}")
    lines.append(
        f"- From validator {solve.get('validator')}; challenge {solve.get('challenge_id')}; "
        f"task {solve.get('task_id')}; model {solve.get('model')}"
    )
    lines.append(f"- Verdict: {_verdict_text(task)}")
    lines.extend(_why_lines(task))
    lines.append(f"- Work: {_work_text(solve)}")
    lines.append(f"- Cost: {_cost_text(solve)}")
    answer = _answer_text(task)
    if answer:
        lines.append(f"- Answer: {answer}")
    if task.folder is not None:
        present = [name for name in FOLDER_FILES if (task.folder / name).exists()]
        lines.append(f"- Folder: {task.folder.relative_to(directory).as_posix()}/ ({', '.join(present)})")
    elif solve.get("trajectory"):
        lines.append(f"- Trajectory: {solve['trajectory']}")
    return lines


def _render_log(directory: Path, tasks: list[Task], refused: list[dict[str, Any]], now: float) -> str:
    lines = [
        "# Miner log",
        "",
        f"Written {utc(now)}. The miner rewrites this file and tasks.csv after every task, verdict and rerun of a "
        "failed check. Each task's folder under tasks/ holds its instruction, the answer and the trajectory and, "
        "for a failure, the failing check and what a rerun of it found.",
        "",
    ]
    credit = _credit_line(directory)
    if credit:
        lines += [credit, ""]
    if not tasks and not refused:
        return "\n".join([*lines, "No tasks recorded yet.", ""])
    if tasks:
        lines += ["## Summary", ""]
        for line in _overview(tasks, refused, now):
            lines.append("  - " + line.strip() if line.startswith("  ") else "- " + line)
        lines += ["", "## Every task, newest first", ""]
        shown = tasks[-LOGGED_TASKS:]
        for task in reversed(shown):
            lines += [*_entry(directory, task), ""]
        if len(tasks) > len(shown):
            lines += [f"{len(tasks) - len(shown)} older tasks are in tasks.csv.", ""]
    if refused:
        lines += ["## Tasks the miner could not take on", ""]
        for record in reversed(refused[-200:]):
            lines.append(
                f"- {_when(record)} · {kind_of(record)} · task {_short(record.get('task_id'), 16)} from validator "
                f"{_short(record.get('validator'), 16)}: {record.get('reason')}"
            )
        lines.append("")
    return "\n".join(lines)


def render_log(directory: Path, *, now: float | None = None) -> str:
    tasks, refused = load_tasks(directory)
    return _render_log(directory, tasks, refused, time.time() if now is None else now)


CSV_COLUMNS = (
    "time", "verdict", "cause", "reason_code", "verdict_time", "kind", "task_type", "task_kind", "language", "title",
    "validator", "challenge_id", "task_id", "model", "status", "error", "ended_by", "seconds", "time_allowed_s",
    "cost_usd", "model_calls", "input_tokens", "cached_tokens", "output_tokens", "commands", "command_failures",
    "tool_errors", "reviewed", "changed_in_review", "gate", "acceptance", "acceptance_summary", "acceptance_runs",
    "acceptance_final", "acceptance_refuted", "clean_copy", "kept_best", "time_checks", "first_finish_s",
    "turns_after_finish", "adjudication",
    "retest_reminded", "surface", "surface_failures", "lint", "lint_findings", "diff_questions", "own_tests", "imports", "toolchain", "left_out", "audit", "choice_check", "probe", "probe_changed", "probe_quotes", "probe_seconds", "probe_cost", "probe_snapshot_used", "shared_from", "refused", "cut_off",
    "command_edits_taken",
    "command_edits_undone", "files_changed",
    "lines_added", "lines_removed", "submission_bytes", "rerun", "tests_failed", "folder",
)


def _render_csv(directory: Path, tasks: list[Task]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for task in tasks:
        solve, outcome, diagnosis = task.solve, task.outcome or {}, task.diagnosis or {}
        tools = solve.get("tools") if isinstance(solve.get("tools"), dict) else {}
        errors = solve.get("tool_errors") if isinstance(solve.get("tool_errors"), dict) else {}
        files = task.changes
        failed_tests = [str(item.get("test")) for item in diagnosis.get("failed") or () if isinstance(item, dict)]
        row = {
            "time": solve.get("time"), "verdict": task.verdict, "cause": task.cause,
            "reason_code": outcome.get("reason_code"), "verdict_time": outcome.get("time"),
            "kind": kind_of(solve), "title": task.title, "seconds": round(_seconds(solve)),
            "commands": tools.get("run_command", 0) + tools.get("run_script", 0),
            "tool_errors": sum(errors.values()),
            "files_changed": len(files) if files else "",
            "lines_added": sum(int(item.get("added", 0)) for item in files) if files else "",
            "lines_removed": sum(int(item.get("removed", 0)) for item in files) if files else "",
            "rerun": diagnosis.get("error") or diagnosis.get("summary") or "",
            "tests_failed": "; ".join(failed_tests),
            "folder": task.folder.relative_to(directory).as_posix() if task.folder is not None else "",
        }
        writer.writerow([row[name] if name in row else _csv_value(solve.get(name)) for name in CSV_COLUMNS])
    return buffer.getvalue()


def _csv_value(value: Any) -> Any:
    return "" if value is None else value


def report(directory: Path, *, listed: int = 40, now: float | None = None, since: str | None = None) -> str:
    """The summary and the failures; with `since`, only for tasks received at or after that UTC time."""
    tasks, refused = load_tasks(directory)
    if since:
        tasks = [task for task in tasks if str(task.solve.get("time", "")) >= since]
        refused = [record for record in refused if str(record.get("time", "")) >= since]
    if not tasks:
        return f"No solves recorded in {directory}{' since ' + since if since else ''} yet."
    lines = ([f"Only tasks received since {since}:"] if since else []) + _overview(tasks, refused, time.time() if now is None else now)
    shown = 0
    for task in reversed(tasks):
        kind = task.cause
        if kind in ("passed", "no verdict"):
            continue
        if shown == listed:
            lines.append("  (older failures omitted)")
            break
        shown += 1
        solve, outcome = task.solve, task.outcome or {}
        reason = outcome.get("reason_code") or solve.get("error") or "no verdict"
        lines.append(
            f"- {str(solve.get('time', '?'))[:16]} {kind_of(solve)} "
            f"task {str(solve.get('task_id'))[:12]} from {str(solve.get('validator'))[:12]}: {kind} ({reason}); "
            f"ended {solve.get('ended_by') or 'with an error'}, ${_cost(solve) or 0:.3f}, "
            f"{_seconds(solve):.0f}s, failed commands {solve.get('command_failures', 0)}"
        )
        if task.title:
            lines.append(f"    asked: {task.title}")
        if task.diagnosis is not None:
            first, *rest = _rerun_lines(task.diagnosis, "      ")
            lines += [f"    rerun: {first}", *rest]
        elif outcome.get("failed_check"):
            lines.append("    failed check: " + " | ".join(str(outcome["failed_check"]).splitlines()[:4])[:300])
        if task.folder is not None:
            lines.append(f"    folder: {task.folder}")
        elif solve.get("trajectory"):
            lines.append(f"    trajectory: {directory / solve['trajectory']}")
    lines.append(f"Every task, passed or failed, is in {directory / 'LOG.md'} and {directory / 'tasks.csv'}.")
    return "\n".join(lines)
