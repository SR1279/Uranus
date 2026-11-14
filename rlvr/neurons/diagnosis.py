"""Rerun a check that failed on the miner's own answer, to find out why it failed.

A validator's failure notice shows the first check that failed: its command, the working
directory and the exact output it required, but never what the answer printed. A
repository task's check is self-contained (it brings its hidden tests and builds the
project itself), so it can be run again in the validators' image, on the workspace the
task supplied with the miner's patch applied, the way the validator ran it. Comparing
the output with the required output shows which hidden tests failed, and the test
runner's own output shows why.

Terminal tasks' checks inspect the result with the validator's own files and are never
shown, so a terminal failure cannot be rerun.
"""

from __future__ import annotations

import json
import os
import re
import stat
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..policy import RELEASE_POLICY, ValidatorPolicy
from ..v3.archive import ArchiveLimits, extract_archive
from ..v3.artifacts import ArtifactRef
from ..v3.feedback import INLINE_SCRIPT_PLACEHOLDER
from ..v3.patch import PatchLimits, apply_patch_in_container
from ..v3.supervisor import ContainerRequest, ContainerResult, Mount, run_container
from .journal import (
    CHECK_FILES_DIR,
    DIAGNOSIS_FILE,
    LISTED_TESTS,
    PATCH_FILE,
    TASK_FILE,
    WORKSPACE_FILE,
    utc,
)
from .workbench import Sandbox

# The validator gives a check about 90 seconds; a busy miner host gets some slack.
RERUN_TIMEOUT_S = 180
RERUN_OUTPUT_BYTES = 1 << 20
RAW_OUTPUT_CHARS = 20_000
RAW_MARKER = "RAW OUTPUT TAIL:\n"
KEPT_FILES = 20
KEPT_FILE_BYTES = 512 * 1024
KEPT_TOTAL_BYTES = 2 * 1024**2
MESSAGE_CHARS = 300
_HEADERS = (
    "Command (argv): ",
    "Working directory: ",
    "Stdin: ",
    "Required exit code: ",
    "Required stdout: ",
    "Required stderr: ",
)
# Checks written by the task generator print the test runner's raw output only when
# this switch is on; a rerun turns it on to see the failures' tracebacks.
_QUIET_RAW_OUTPUT = re.compile(r'^if False:\n(\s+print\("RAW OUTPUT TAIL:\\n" \+ out\[-)\d+(:\]\))', re.M)
# The per-test lines those checks print: "<runner> <test> <result>".
_TEST_LINES = (
    re.compile(r"^(unittest) (.+?) (ok|FAIL|ERROR|skipped|expected failure|unexpected success)$"),
    re.compile(r"^(go) (\S+) (PASS|FAIL|SKIP)$"),
    re.compile(r"^(tap) (.+?) (not ok|ok)$"),
    re.compile(r"^(cargo) (\S+) (ok|FAILED|ignored)$"),
    re.compile(r"^(pass) (\S+)()$"),
)


class DiagnosisError(Exception):
    """The failed check cannot be rerun; the message says why."""


@dataclass(frozen=True)
class FailedCheck:
    argv: tuple[str, ...]
    cwd: str
    stdin: bytes
    exit_code: int
    stdout: str
    stderr: str | None


def parse_failed_check(display: str) -> FailedCheck | None:
    """Read back a check display as rlvr.v3.feedback.render_failed_check writes it."""
    lines = display.split("\n")
    if len(lines) < len(_HEADERS) or any(not line.startswith(prefix) for line, prefix in zip(lines, _HEADERS)):
        return None
    values = [line[len(prefix):] for line, prefix in zip(lines, _HEADERS)]
    try:
        argv = json.loads(values[0])
        cwd, stdin, stdout = json.loads(values[1]), json.loads(values[2]), json.loads(values[4])
        exit_code = int(values[3])
        stderr = None if values[5] == "not checked" else json.loads(values[5])
    except ValueError:
        return None
    if (
        type(argv) is not list
        or not argv
        or any(type(item) is not str for item in argv)
        or not all(type(item) is str for item in (cwd, stdin, stdout))
        or (stderr is not None and type(stderr) is not str)
    ):
        return None
    if INLINE_SCRIPT_PLACEHOLDER in argv:
        if len(lines) < len(_HEADERS) + 2 or lines[len(_HEADERS)] != "Inline script:":
            return None
        script = "\n".join(lines[len(_HEADERS) + 1:])
        argv = [script if item == INLINE_SCRIPT_PLACEHOLDER else item for item in argv]
    return FailedCheck(tuple(argv), cwd, stdin.encode("utf-8"), exit_code, stdout, stderr)


def louder(argv: tuple[str, ...]) -> tuple[tuple[str, ...], bool]:
    """The check with its raw test output switched on, when it has that switch."""
    if len(argv) != 3 or argv[1] != "-c":
        return argv, False
    script, found = _QUIET_RAW_OUTPUT.subn(rf"if True:\n\g<1>{RAW_OUTPUT_CHARS}\g<2>", argv[2], count=1)
    return (argv[0], argv[1], script), bool(found)


def _tests(text: str) -> dict[str, str]:
    """Each test a check's output reports, with its result."""
    found: dict[str, str] = {}
    for line in text.splitlines():
        for pattern in _TEST_LINES:
            match = pattern.match(line)
            if match:
                found[f"{match.group(1)} {match.group(2)}"] = match.group(3)
                break
    return found


def _test_name(key: str) -> str:
    """"unittest test_x (module.Class.test_x)" becomes "test_x"."""
    return key.split(" ", 1)[-1].split(" (", 1)[0]


# Test runners' own result lines, for a check whose required output has no per-test lines.
_RUNNER_LINES = (
    # unittest -v, including a failing subtest's indented line
    re.compile(r"^\s*(test\S*) \([\w.]+\)(?: \(.*\))? \.\.\. (ok|FAIL|ERROR|skipped|expected failure|unexpected success)\b"),
    re.compile(r"^\s*--- (?P<status>PASS|FAIL|SKIP): (?P<name>\S+)"),  # go test -v
    re.compile(r"^test (\S+) \.\.\. (ok|FAILED|ignored)$"),  # cargo test
    re.compile(r"^(?P<status>not ok|ok) \d+ - (?P<name>.+?)(?:\s+#.*)?$"),  # TAP
)
_NOT_FAILED = frozenset({"ok", "PASS", "SKIP", "skipped", "ignored", "expected failure"})


def _runner_results(raw: str) -> dict[str, str]:
    """Each test the runner's own output reports; a test with any failing part counts as failed."""
    found: dict[str, str] = {}
    for line in raw.splitlines():
        for pattern in _RUNNER_LINES:
            match = pattern.match(line)
            if match is None:
                continue
            groups = match.groupdict()
            name, status = (groups["name"], groups["status"]) if groups else match.groups()[:2]
            if found.get(name) in (None, *_NOT_FAILED):
                found[name] = status
            break
    return found


def _messages(raw: str) -> dict[str, list[str]]:
    """The error each failing test reported, from the test runner's own output."""
    found: dict[str, list[str]] = {}

    def add(name: str, message: str) -> None:
        messages = found.setdefault(name, [])
        message = message.strip()[:MESSAGE_CHARS]
        if message and message not in messages:
            messages.append(message)

    # unittest: a "FAIL: name (id)" block, whose exception line follows the traceback. Some
    # projects indent everything written to stderr (Kivy does), so a block keeps its indentation.
    for block in re.split(r"^[ \t]*={20,}[ \t]*$", raw, flags=re.M):
        header = re.search(r"^([ \t]*)(?:FAIL|ERROR): (\S+)", block, re.M)
        if header is None:
            continue
        indent = header.group(1)
        exception = None
        after_traceback = False
        for line in block.splitlines():
            line = line[len(indent):] if line.startswith(indent) else line
            if line.startswith("Traceback (most recent call last)"):
                after_traceback, exception = True, None
            elif after_traceback and exception is None and line and not line[0].isspace():
                exception = line
        if exception:
            add(header.group(2), exception)
    # go: "--- FAIL: TestName" followed by indented messages.
    for match in re.finditer(r"^\s*--- FAIL: (\S+).*\n\s+(.+)$", raw, re.M):
        add(match.group(1), match.group(2))
    # cargo: "thread 'name' panicked at place:" followed by the message.
    for match in re.finditer(r"^\s*thread '([^']+)' panicked at [^\n]*\n\s*(.+)$", raw, re.M):
        add(match.group(1).rsplit("::", 1)[-1], match.group(2))
    return found


@dataclass
class Comparison:
    matched: bool
    exit_code: int | None
    required_exit_code: int
    tests: int = 0
    required_lines: int = 0
    failed: list[dict[str, str]] = field(default_factory=list)
    missing_lines: list[str] = field(default_factory=list)
    unexpected_lines: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        if self.matched:
            return "the rerun printed exactly the required output, so the failure did not happen again here; it may depend on timing or load"
        exit_note = (
            f"the check exited {self.exit_code}, not {self.required_exit_code}"
            if self.exit_code != self.required_exit_code
            else ""
        )
        if self.failed:
            silent = sum(item["result"] == "no result" for item in self.failed)
            parts = []
            if len(self.failed) - silent:
                parts.append(f"{len(self.failed) - silent} failed or errored")
            if silent:
                parts.append(f"{silent} printed no result (a subtest failed, it crashed or it never ran)")
            text = f"{len(self.failed)} of {self.tests} hidden tests did not pass: {', '.join(parts)}"
        elif self.missing_lines:
            text = f"{len(self.missing_lines)} of {self.required_lines} required output lines were missing or different"
        elif self.unexpected_lines:
            text = f"the output had {len(self.unexpected_lines)} lines that were not required"
        else:
            return exit_note or "the output differed from the required output"
        return f"{text}; {exit_note}" if exit_note else text


def split_raw_output(printed: str) -> tuple[str, str]:
    """What the unmodified check prints, and the runner's raw output that follows the marker."""
    if printed.startswith(RAW_MARKER):
        return "", printed[len(RAW_MARKER):]
    index = printed.find("\n" + RAW_MARKER)
    if index < 0:
        return printed, ""
    return printed[: index + 1], printed[index + 1 + len(RAW_MARKER):]


def compare(check: FailedCheck, stdout: str, exit_code: int | None, raw: str = "") -> Comparison:
    """How the rerun's output differs from what the check required."""
    matched = stdout == check.stdout and exit_code == check.exit_code
    comparison = Comparison(matched, exit_code, check.exit_code, required_lines=len(check.stdout.splitlines()))
    required_lines, printed_lines = check.stdout.splitlines(), stdout.splitlines()
    required_set, printed_set = set(required_lines), set(printed_lines)
    comparison.missing_lines = [line for line in required_lines if line not in printed_set]
    comparison.unexpected_lines = [line for line in printed_lines if line not in required_set]
    required_tests, printed_tests = _tests(check.stdout), _tests(stdout)
    messages = _messages(raw)
    found: list[tuple[str, str]] = []
    if required_tests:
        comparison.tests = len(required_tests)
        for key, wanted in required_tests.items():
            got = printed_tests.get(key)
            if got != wanted:
                found.append((_test_name(key), "no result" if got is None else got))
    elif not matched:
        # The check only states an overall result, so the runner's own output names the failures.
        results = _runner_results(raw)
        comparison.tests = len(results)
        found = [(name, result) for name, result in results.items() if result not in _NOT_FAILED]
    for name, result in found:
        item = {"test": name, "result": result}
        if messages.get(name):
            item["message"] = " | ".join(messages[name][:3])
        comparison.failed.append(item)
    return comparison


def _hand_over(root: Path, uid: int, gid: int) -> None:
    """Give a tree to the unprivileged sandbox user, as the workbench does for its scratch copy."""
    for directory, _directories, files in os.walk(root):
        os.chown(directory, uid, gid)
        for name in files:
            os.chown(os.path.join(directory, name), uid, gid, follow_symlinks=False)


def _regular_files(root: Path) -> set[str]:
    found = set()
    for directory, directories, files in os.walk(root, followlinks=False):
        directories[:] = [name for name in directories if name not in ("__pycache__", ".git")]
        for name in files:
            path = Path(directory, name)
            if stat.S_ISREG(path.lstat().st_mode):
                found.add(path.relative_to(root).as_posix())
    return found


def _keep_check_files(workspace: Path, before: set[str], destination: Path) -> list[str]:
    """Copy the text files the check added (its hidden tests), bounded in number and size."""
    added = sorted(_regular_files(workspace) - before, key=lambda path: ("test" not in path.lower(), path))
    kept: list[str] = []
    total = 0
    for relative in added:
        if len(kept) == KEPT_FILES:
            break
        source = workspace / relative
        try:
            info = source.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_size > KEPT_FILE_BYTES or total + info.st_size > KEPT_TOTAL_BYTES:
                continue
            data = source.read_bytes()
        except OSError:
            continue
        if b"\x00" in data[:8192]:
            continue  # a build product, not a test
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        kept.append(relative)
        total += len(data)
    return kept


def _report_text(check: FailedCheck, comparison: Comparison, result: ContainerResult, raw: str, kept: list[str]) -> str:
    lines = [
        f"Rerun of the failing check, {utc(time.time())}",
        "",
        "The check ran in the validators' image, offline and as an unprivileged user, on the",
        "workspace the task supplied with the miner's patch applied, as the validator runs it.",
        "",
        f"Result: {comparison.summary}.",
        f"Exit code: {result.exit_code} (required {check.exit_code})"
        + ("; it ran out of time" if result.timed_out else "")
        + ("; it ran out of memory" if result.oom_killed else ""),
        "",
    ]
    if comparison.failed:
        lines.append("Tests that did not pass:")
        for item in comparison.failed:
            lines.append(f"- {item['test']}: {item['result']}" + (f" — {item['message']}" if item.get("message") else ""))
        lines.append("")
    if comparison.missing_lines:
        lines += ["Required lines missing from the output:", *(f"  {line}" for line in comparison.missing_lines), ""]
    if comparison.unexpected_lines:
        lines += ["Printed lines that were not required:", *(f"  {line}" for line in comparison.unexpected_lines), ""]
    if raw:
        lines += ["The test runner's output (its end):", raw.rstrip("\n"), ""]
    stderr = result.stderr.decode("utf-8", "replace")
    if stderr.strip():
        lines += ["Standard error (its end):", stderr[-RAW_OUTPUT_CHARS:].rstrip("\n"), ""]
    if kept:
        lines += [f"Files the check added, copied to {CHECK_FILES_DIR}/:", *(f"- {path}" for path in kept), ""]
    return "\n".join(lines)


def rerun_failed_check(
    folder: Path,
    display: str,
    sandbox: Sandbox,
    work_dir: Path | None = None,
    *,
    policy: ValidatorPolicy = RELEASE_POLICY,
) -> dict[str, Any]:
    """Rerun the check in `display` on the answer kept in a task's folder.

    Writes diagnosis.txt and the check's own files to the folder, and returns what the
    rerun found, for the journal. Blocking: run it off the event loop.
    """
    check = parse_failed_check(display)
    if check is None:
        raise DiagnosisError("the validator's check display could not be read")
    try:
        details = json.loads((folder / TASK_FILE).read_text(encoding="utf-8"))
        patch = (folder / PATCH_FILE).read_bytes()
    except (OSError, ValueError):
        raise DiagnosisError("the task's details or answer were not kept") from None
    if details.get("task_type") != "repository_patch_v1":
        raise DiagnosisError("only repository tasks' checks can be rerun")
    archive = folder / WORKSPACE_FILE
    if not archive.is_file():
        raise DiagnosisError("the workspace was not kept")
    try:
        reference = ArtifactRef.model_validate(details.get("workspace"))
    except ValueError:
        raise DiagnosisError("the workspace's reference was not kept") from None
    limits = ArchiveLimits(
        max_compressed_bytes=policy.v3_workspace_compressed_bytes,
        max_expanded_bytes=policy.v3_workspace_bytes,
        max_file_bytes=policy.v3_max_file_bytes,
        max_entries=200_000,
        max_path_bytes=4_096,
        max_zstd_window_bytes=128 * 1024**2,
    )
    prefix = f"hone-rerun-{uuid.uuid4().hex[:16]}"
    with tempfile.TemporaryDirectory(prefix="hone-rerun-", dir=work_dir) as temporary:
        root = Path(temporary)
        workspace = root / "workspace"
        extract_archive(archive, reference, workspace, limits, scratch_dir=root)
        if sandbox.owns_files:
            _hand_over(workspace, sandbox.uid, sandbox.gid)
        applied = apply_patch_in_container(
            workspace,
            patch,
            PatchLimits(max_patch_bytes=policy.v3_patch_bytes, git_timeout_s=30),
            supervisor_policy=sandbox.supervisor,
            docker_binary=sandbox.docker_binary,
            run_prefix=prefix,
        )
        if applied.status != "applied":
            raise DiagnosisError(f"the patch did not apply here: {applied.reason[:200]}")
        before = _regular_files(workspace)
        argv, raw_switched_on = louder(check.argv)
        result = run_container(
            ContainerRequest(
                name=f"{prefix}-check",
                argv=argv,
                cwd=check.cwd,
                mounts=(Mount(workspace, "/work", False),),
                stdin=check.stdin,
                timeout_s=RERUN_TIMEOUT_S,
                max_stdout_bytes=RERUN_OUTPUT_BYTES,
                max_stderr_bytes=RERUN_OUTPUT_BYTES,
                trusted=False,
            ),
            sandbox.supervisor,
            sandbox.docker_binary,
        )
        printed, runner = result.stdout.decode("utf-8", "replace"), ""
        if raw_switched_on:
            printed, runner = split_raw_output(printed)
        stderr = result.stderr.decode("utf-8", "replace")[-RAW_OUTPUT_CHARS:]
        comparison = compare(check, printed, result.exit_code, runner or stderr)
        kept = _keep_check_files(workspace, before, folder / CHECK_FILES_DIR)
    (folder / DIAGNOSIS_FILE).write_text(_report_text(check, comparison, result, runner, kept), encoding="utf-8")
    return {
        "reproduced": not comparison.matched,
        "summary": comparison.summary,
        "tests": comparison.tests,
        "failed_total": len(comparison.failed),
        "failed": comparison.failed[:LISTED_TESTS * 4],
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "check_files": kept,
    }
