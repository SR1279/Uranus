"""Where the demo miner's model edits and tests its answer to one task.

Three trees:

- the extracted workspace stays exactly as the validator supplied it;
- ``edits`` holds the submission, and the edit tools write to it;
- ``scratch`` is a disposable copy that sandboxed commands may do anything to.

On a repository task, what a command did to the repository's files is settled after it runs, because the
model tests what the copy holds (2026-11-11: a solve applied the adjudicator's corrections with a Python
script, its tests passed on the copy, and the submitted diff had none of them). A command's change to a
file the answer already changes is taken into the answer, read without following links; any other change
to a file of the repository is put back before the next command. Both are reported to the model.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

from ..policy import RELEASE_POLICY, ValidatorPolicy
from ..v3.grading import SCRIPT_OUTPUT_BYTES, SCRIPT_TIMEOUT_S
from ..v3.identity import validate_relative_path
from ..v3.miner_workspace import WorkspaceReader
from ..v3.supervisor import (
    ContainerRequest,
    ContainerResult,
    Mount,
    SupervisorError,
    SupervisorPolicy,
    run_container,
)

SCRIPT_PATH = "submission.sh"
SANDBOX_MAX_WORKSPACE_BYTES = 2 * 1024**3
COMMAND_TIMEOUT_S = 120
COMMAND_MAX_CHARS = 64 * 1024
WRITE_MAX_BYTES = 1024**2
OUTPUT_HEAD_BYTES = 4 * 1024
OUTPUT_TAIL_BYTES = 12 * 1024
SEARCH_MAX_MATCHES = 100
SEARCH_MAX_LINE_CHARS = 240
SEARCH_READ_BYTES = 1024**2
SEARCH_TIMEOUT_S = 10
CHANGES_LISTED = 60
TMP_KEPT_MB = 256
# A clean-copy check never runs into the last seconds before the deadline.
GATE_RESERVE_S = 30

# A command's change to a file of the repository is looked for under these names only
# when the answer does not already change the file: build and cache trees are rewritten
# by ordinary builds and say nothing about the answer.
_UNWATCHED = (".git", "node_modules", "__pycache__", "target", ".prebuilt", ".gradle", ".cache", ".venv", "venv",
              ".tox", ".pytest_cache", ".mypy_cache", ".nyc_output", "coverage")
_UNWATCHED_FIND = " -o ".join(f"-name '{name}'" for name in _UNWATCHED)
WATCHED_MAX_BYTES = 1024**2
# Only source and configuration files a command created are asked about before submitting, and not
# tests or scratch files.
_SOURCE_SUFFIXES = (
    ".py", ".pyi", ".go", ".js", ".mjs", ".cjs", ".ts", ".mts", ".cts", ".tsx", ".jsx", ".rs", ".c", ".h", ".cc",
    ".cpp", ".hpp", ".hh", ".java", ".kt", ".rb", ".php", ".cs", ".swift", ".scala", ".sh", ".json", ".toml",
    ".yaml", ".yml", ".cfg", ".ini", ".mod", ".sql",
)
_SCRATCH_RE = re.compile(
    r"(^|/)(tests?|__tests__|testdata|spec|fixtures?)/|^(tmp|out|dist|build)/"
    r"|(^|/)(test_[^/]*|[^/]*_test\.\w+|[^/]*\.(test|spec)\.\w+|conftest\.py)$"
    r"|(^|/)(t|tt|tmp|scratch|debug|dbg|probe|repro|try|ex)\d*\.\w+$"
)
CHANGES_REPORTED = 20
# The longest the look at the repository's files may take after a command, added to its container's limit.
WATCH_S = 8

# Runs inside the container, so a link planted by an earlier command can only
# ever redirect these writes within the sandbox. Standard input lists the
# edited paths the scratch copy has not seen yet, an empty entry, then every path
# the answer changes. /tmp is new in every container, so it is carried between
# commands, where test programs usually live.
#
# With "watch" as the third argument, the runner then lists what the command did to
# the repository's files in /state/changed, NUL-separated kind and path pairs: M, an
# answer file whose content now differs from the answer; D, an answer file the
# command removed or replaced with something other than a file; O, any other file of
# the repository whose content the command changed; N, a text file the command created
# that neither the repository nor the answer has. Commands work on a copy, so without
# this a fix applied with sed or a script would be tested and never submitted.
_RUNNER = f"""
if [ -d /edits ] && ! {{ [ -r /edits ] && [ -x /edits ]; }}; then
  echo "sandbox error: the edited files cannot be read here, so this command did not run" >&2
  exit 126
fi
while IFS= read -r -d '' path && [ -n "$path" ]; do
  rm -rf -- "/work/$path"
  if [ -f "/edits/$path" ]; then
    mkdir -p -- "$(dirname -- "/work/$path")" && cp -p -- "/edits/$path" "/work/$path" || {{ echo "sandbox error: $path could not be copied in" >&2; exit 126; }}
  fi
done
declare -A answer=()
while IFS= read -r -d '' path; do
  [ -n "$path" ] && answer["$path"]=1
done
[ -f /state/tmp.tar ] && tar -xf /state/tmp.tar -C /tmp 2>/dev/null
rm -f /state/changed /state/changed.new
[ "$3" = watch ] && touch /state/.runner-stamp
timeout --signal=TERM --kill-after=2 "$2" /usr/bin/bash --noprofile --norc -c "$1" >/tmp/.runner-output 2>&1 </dev/null
status=$?
if [ "$3" = watch ] && [ -d /edits ] && [ -f /state/.runner-stamp ]; then
  report=/state/changed.new
  rm -f "$report"
  : > "$report"
  for path in "${{!answer[@]}}"; do
    [ -f "/edits/$path" ] || continue
    if [ -L "/work/$path" ] || [ ! -f "/work/$path" ]; then
      printf 'D\\0%s\\0' "$path" >> "$report"
    elif ! cmp -s -- "/work/$path" "/edits/$path"; then
      printf 'M\\0%s\\0' "$path" >> "$report"
    fi
  done
  timeout {WATCH_S} find /work -xdev \\( {_UNWATCHED_FIND} \\) -prune -o -type f -newer /state/.runner-stamp -size -{WATCHED_MAX_BYTES // 1024 + 1}k -print0 2>/dev/null | head -z -n 2000 |
  while IFS= read -r -d '' file; do
    path=${{file#/work/}}
    [ -z "${{answer[$path]+set}}" ] || continue
    if [ ! -e "/edits/$path" ] && [ ! -L "/edits/$path" ]; then
      tr -d '\\000' < "$file" | cmp -s - "$file" || continue
      printf 'N\\0%s\\0' "$path" >> "$report"
      continue
    fi
    [ -f "/edits/$path" ] && [ ! -L "/edits/$path" ] || continue
    cmp -s -- "$file" "/edits/$path" && continue
    tr -d '\\000' < "$file" | cmp -s - "$file" || continue  # a binary file is a build product
    printf 'O\\0%s\\0' "$path" >> "$report"
  done
  mv -f "$report" /state/changed
fi
if [ "$(du -sm /tmp | cut -f1)" -le {TMP_KEPT_MB} ]; then
  tar -cf /state/tmp.new --exclude=./.runner-output -C /tmp . 2>/dev/null && mv -f /state/tmp.new /state/tmp.tar
fi
size=$(stat -c %s /tmp/.runner-output)
if [ "$size" -le {OUTPUT_HEAD_BYTES + OUTPUT_TAIL_BYTES} ]; then
  cat /tmp/.runner-output
else
  head -c {OUTPUT_HEAD_BYTES} /tmp/.runner-output
  printf '\\n[... %s bytes omitted ...]\\n' "$((size - {OUTPUT_HEAD_BYTES + OUTPUT_TAIL_BYTES}))"
  tail -c {OUTPUT_TAIL_BYTES} /tmp/.runner-output
fi
exit "$status"
"""


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


_PATH = {"type": "string", "description": "Path relative to the workspace root; \".\" is the root."}
_ENTRY = {"type": "integer", "description": "Entry index to start from; 0 for the first page."}

WORKSPACE_TOOLS: tuple[dict[str, Any], ...] = (
    _tool("list_files", "List the entries of one directory.", {"path": _PATH, "offset": _ENTRY}, ["path", "offset"]),
    _tool(
        "find_files",
        "List every file under a directory, recursively, sorted by path.",
        {"path": _PATH, "offset": _ENTRY},
        ["path", "offset"],
    ),
    _tool(
        "read_file",
        "Read a file as UTF-8 text, 32 KiB at a time.",
        {"path": _PATH, "offset": {"type": "integer", "description": "Byte offset to start from; 0 for the beginning."}},
        ["path", "offset"],
    ),
    _tool(
        "search_text",
        "Find the lines matching a case-sensitive POSIX extended regular expression in text files under a "
        "directory. Each match gives its path, line number and byte offset; read_file accepts that offset.",
        {"pattern": {"type": "string", "description": "The regular expression."}, "path": _PATH},
        ["pattern", "path"],
    ),
)
EDIT_TOOLS: tuple[dict[str, Any], ...] = (
    _tool(
        "edit_file",
        "Replace exact text in a file. old_text must match the file exactly, including whitespace, and must "
        "occur once unless replace_all is true.",
        {
            "path": _PATH,
            "old_text": {"type": "string", "description": "The text to replace."},
            "new_text": {"type": "string", "description": "The replacement text."},
            "replace_all": {"type": "boolean", "description": "Replace every occurrence."},
        },
        ["path", "old_text", "new_text"],
    ),
    _tool(
        "write_file",
        f"Create a file or replace its whole content. On a script task, the only writable path is {SCRIPT_PATH}.",
        {"path": _PATH, "content": {"type": "string", "description": "The complete new content."}},
        ["path", "content"],
    ),
)
REVERT_TOOL = _tool(
    "revert_file",
    "Undo every change you made to a file: restore its original content, or remove it if you created it.",
    {"path": _PATH},
    ["path"],
)
COMMAND_TOOL = _tool(
    "run_command",
    "Run a Bash command in a disposable copy of the workspace that includes your latest edits, in the grader's "
    f"image and under its limits, with no network access. Files you create persist between commands, including "
    f"those in /tmp. Returns the exit code and combined output. At most {COMMAND_TIMEOUT_S} seconds. Only "
    "edit_file and write_file change your answer: when a command changes a file your answer already changes, the "
    "result is taken into your answer, and when it changes any other file of the repository, that change is "
    "undone before the next command. Both are reported.",
    {
        "command": {"type": "string", "description": "The Bash command."},
        "timeout_s": {"type": "integer", "description": f"Seconds to allow, at most {COMMAND_TIMEOUT_S}."},
    },
    ["command", "timeout_s"],
)
SCRIPT_TOOL = _tool(
    "run_script",
    f"Run {SCRIPT_PATH} exactly as the grader will: from a clean copy of the workspace, in the result directory, "
    "with its time and output limits. Returns the exit code, the output and the files it created or changed. "
    "run_command can then inspect the result.",
    {},
    [],
)


def _shorten(data: bytes) -> str:
    """Bytes for the model, keeping the end, where a failure usually is."""
    if len(data) > OUTPUT_HEAD_BYTES + OUTPUT_TAIL_BYTES:
        omitted = len(data) - OUTPUT_HEAD_BYTES - OUTPUT_TAIL_BYTES
        data = data[:OUTPUT_HEAD_BYTES] + f"\n[... {omitted} bytes omitted ...]\n".encode() + data[-OUTPUT_TAIL_BYTES:]
    return data.decode("utf-8", "replace")


def _timed_out(result: ContainerResult) -> bool:
    return result.timed_out or result.exit_code in (124, 137)


def _read_regular(path: Path, limit: int) -> bytes | None:
    """A regular file's bytes, or None when it is a link, a device, a pipe, missing or over the limit."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return None
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            return None
        chunks: list[bytes] = []
        remaining = limit + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(remaining, 1024**2))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        return None if len(data) > limit else data
    except OSError:
        return None
    finally:
        os.close(descriptor)


class Sandbox:
    """Runs task code in the validators' image, under their limits and with no network."""

    def __init__(self, docker_binary: str, *, concurrency: int, policy: ValidatorPolicy = RELEASE_POLICY):
        self.docker_binary = docker_binary
        running_as_root = os.getuid() == 0
        self.uid = 65_534 if running_as_root else os.getuid()
        self.gid = 65_534 if running_as_root else os.getgid()
        self.owns_files = running_as_root  # only root can hand the scratch copy to another user
        self.supervisor = SupervisorPolicy(
            image=policy.v3_image,
            candidate_uid=self.uid,
            candidate_gid=self.gid,
            memory_bytes=policy.v3_memory_bytes,
            cpus=policy.v3_cpus,
            pids_limit=policy.v3_pids_limit,
            tmpfs_bytes=policy.v3_tmpfs_bytes,
            max_file_bytes=policy.v3_max_file_bytes,
            watchdog_slack_s=5,
            max_workspace_bytes=policy.v3_workspace_bytes,
        )
        self.slots = asyncio.Semaphore(concurrency)

    @classmethod
    def detect(cls, *, concurrency: int, policy: ValidatorPolicy = RELEASE_POLICY) -> tuple[Sandbox | None, str]:
        """The sandbox, or None and why it cannot be used on this host."""
        docker = shutil.which("docker")
        if docker is None:
            return None, "docker is not installed"
        docker = os.path.realpath(docker)
        try:
            inspected = subprocess.run(
                [docker, "image", "inspect", policy.v3_image],
                capture_output=True, timeout=30, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None, "docker could not be run"
        if inspected.returncode != 0:
            return None, f"image is not pulled; run: docker pull {policy.v3_image}"
        try:
            return cls(docker, concurrency=concurrency, policy=policy), "ready"
        except ValueError as error:
            return None, str(error)

    async def run(self, request: ContainerRequest, *, wait_s: float) -> ContainerResult:
        await asyncio.wait_for(self.slots.acquire(), wait_s)
        try:
            return await asyncio.to_thread(run_container, request, self.supervisor, self.docker_binary)
        finally:
            self.slots.release()

    async def selfcheck(self, work_dir: Path | None) -> str | None:
        """Prove that a command sees the edit tools' work, as a task would; the problem found, or None.

        The miner and the sandbox run as different users on the live host, and the test
        suite runs as one user, so a permission mistake between them shows up only here.
        """
        with tempfile.TemporaryDirectory(prefix="hone-selfcheck-", dir=work_dir) as temporary:
            root = Path(temporary) / "workspace"
            root.mkdir(mode=0o700)  # as the extracted workspace is created
            (root / "probe.txt").write_text("before\n", encoding="utf-8")
            with Workbench(WorkspaceReader(root), terminal=False, working_directory=".", sandbox=self) as bench:
                try:
                    bench._edit({"path": "probe.txt", "old_text": "before", "new_text": "after"})
                    bench._write({"path": "made/new.txt", "content": "made\n"})
                    result = await bench._run_command({"command": "cat probe.txt made/new.txt", "timeout_s": 30}, 90.0)
                except (ValueError, OSError, asyncio.TimeoutError, SupervisorError, subprocess.SubprocessError) as error:
                    return f"a command could not run on the edits: {error}"
        if result["exit_code"] != 0 or "after" not in result["output"] or "made" not in result["output"]:
            return "commands do not see the edit tools' work: " + result["output"].strip()[:200]
        return None


class Workbench:
    """One task's editable answer and its test runs."""

    def __init__(
        self,
        reader: WorkspaceReader,
        *,
        terminal: bool,
        working_directory: str,
        sandbox: Sandbox | None = None,
    ):
        self.pristine = reader.root
        self.reader = reader
        self.terminal = terminal
        self.sandbox = sandbox
        self.cwd = "/work" if working_directory == "." else f"/work/{working_directory}"
        self.base = Path(tempfile.mkdtemp(prefix="hone-bench-", dir=reader.root.parent))
        self.edits = self.base / "edits"
        self.scratch = self.base / "scratch"
        self.control = self.base / "control"
        self.control.mkdir(mode=0o755)
        self.state = self.base / "state"
        self.state.mkdir(mode=0o755)
        self.script = self.control / "script.sh"
        self.touched: set[str] = set()
        self.unsynced: set[str] = set()
        self.created: set[str] = set()  # text files commands created in the copy, outside the answer
        self.version = 0  # bumped by every edit
        self.rehearsed = -1  # the script version a clean run last ran without failing

    def __enter__(self) -> Workbench:
        return self

    def __exit__(self, *_exc: object) -> None:
        shutil.rmtree(self.base, ignore_errors=True)

    @property
    def can_run(self) -> bool:
        return self.sandbox is not None

    @property
    def tools(self) -> tuple[dict[str, Any], ...]:
        tools = WORKSPACE_TOOLS + EDIT_TOOLS + (() if self.terminal else (REVERT_TOOL,))
        if self.can_run:
            tools += (SCRIPT_TOOL, COMMAND_TOOL) if self.terminal else (COMMAND_TOOL,)
        return tools

    def has_work(self) -> bool:
        return self.script.is_file() if self.terminal else bool(self.touched)

    def submission(self) -> bytes:
        """The answer so far: the script, or every edit as a Git unified diff."""
        if self.terminal:
            return self.script.read_bytes() if self.script.is_file() else b""
        return self._patch()

    async def execute(self, arguments: dict[str, Any], *, time_left: float) -> bytes:
        name = arguments.get("tool")
        path = arguments.get("path")
        if type(path) is str and (path == "/work" or path.startswith(("/work/", "./"))):
            # Commands show the workspace at /work, so the model often names a file the way a command
            # does; the file tools take paths relative to the workspace root.
            relative = path[len("/work"):] if path.startswith("/work") else path
            while relative.startswith(("/", "./")):
                relative = relative[1:] if relative.startswith("/") else relative[2:]
            arguments = {**arguments, "path": relative or "."}
        try:
            if name == "run_command":
                value = await self._run_command(arguments, time_left)
            elif name == "run_script":
                value = await self._run_script(time_left)
            elif name == "search_text":
                value = await asyncio.to_thread(self._search, arguments)
            elif name == "edit_file":
                value = await asyncio.to_thread(self._edit, arguments)
            elif name == "write_file":
                value = await asyncio.to_thread(self._write, arguments)
            elif name == "revert_file" and not self.terminal:
                value = await asyncio.to_thread(self._revert, arguments)
            elif name == "read_file" and self.terminal and arguments.get("path") == SCRIPT_PATH:
                # The script is kept outside the workspace, but the model may read it back.
                return WorkspaceReader(self.control).execute({**arguments, "path": self.script.name})
            else:
                return self.reader.execute(arguments)
        except ValueError as error:
            value = {"error": str(error)}
        except (OSError, asyncio.TimeoutError, subprocess.SubprocessError, SupervisorError):
            value = {"error": f"{name} could not complete"}
        return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    async def rerun(self, command: str, *, limit_s: int, time_left: float) -> dict[str, Any]:
        """Run an earlier command of the model's again on the copy as it stands, taking nothing it changes.

        Nothing it writes enters the answer: the change report is not even made. What it leaves in the copy stays
        there until the caller's reset_to_answer, and what it writes to /tmp until the caller's restore_tmp. A
        command that could not run comes back as {"error": ...}, as from execute.
        """
        try:
            return await self._run_command({"command": command, "timeout_s": limit_s}, time_left, adopt=False)
        except ValueError as error:
            return {"error": str(error)}
        except (OSError, asyncio.TimeoutError, subprocess.SubprocessError, SupervisorError):
            return {"error": "run_command could not complete"}

    def _ensure_edits(self) -> None:
        """Link the workspace into the edit tree, once. Edits replace a file, so a link never changes the original."""
        if not self.edits.exists():
            shutil.copytree(self.pristine, self.edits, symlinks=True, copy_function=os.link)
            self.reader = WorkspaceReader(self.edits)
            self._open_to_sandbox(self.edits)

    def _open_to_sandbox(self, root: Path) -> None:
        """Let the sandbox user read the whole tree.

        The miner runs as root and the sandbox as another user. The extracted workspace's root
        is private to the miner (mode 0700) and copytree keeps that mode, so without this the
        sandbox could not open /edits at all, and every command ran on a copy with the edited
        files missing.
        """
        if self.sandbox is None or not self.sandbox.owns_files:
            return
        for directory, _directories, files in os.walk(root):
            os.chmod(directory, os.stat(directory).st_mode | 0o555)
            for name in files:
                path = os.path.join(directory, name)
                info = os.lstat(path)
                if stat.S_ISREG(info.st_mode) and not info.st_mode & 0o004:
                    os.chmod(path, info.st_mode | 0o444)  # the inode is the workspace copy's too; a mode is not content

    def _open_path_to_sandbox(self, directory: Path) -> None:
        """Let the sandbox user enter a directory the edit tools just created, and its parents."""
        if self.sandbox is None or not self.sandbox.owns_files or self.terminal:
            return
        while True:
            os.chmod(directory, os.stat(directory).st_mode | 0o555)
            if directory == self.edits or directory.parent == directory:
                return
            directory = directory.parent

    def _target(self, relative: Any) -> Path:
        if type(relative) is not str:
            raise ValueError("path must be a string")
        if self.terminal:
            if relative != SCRIPT_PATH:
                raise ValueError(f"only {SCRIPT_PATH} can be written; the script itself must produce every other file")
            return self.script
        validate_relative_path(relative)
        if relative == ".":
            raise ValueError("path must name a file")
        if ".git" in relative.split("/"):
            raise ValueError(".git cannot be part of a submission")
        self._ensure_edits()
        current = self.edits
        for segment in relative.split("/")[:-1]:
            current = current / segment
            if current.is_symlink() or (current.exists() and not current.is_dir()):
                raise ValueError(f"{current.relative_to(self.edits).as_posix()} is not a directory")
        target = current / relative.split("/")[-1]
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise ValueError("path is not a regular file")
        return target

    def _store(self, relative: str, target: Path, data: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        self._open_path_to_sandbox(target.parent)
        executable = target.exists() and bool(target.stat().st_mode & stat.S_IXUSR)
        temporary = target.with_name(f".{uuid.uuid4().hex}.tmp")
        temporary.write_bytes(data)
        temporary.chmod(0o755 if executable else 0o644)
        os.replace(temporary, target)
        self.version += 1
        if not self.terminal:
            self.touched.add(relative)
            self.unsynced.add(relative)

    def _edit(self, arguments: dict[str, Any]) -> dict[str, Any]:
        relative, old_text, new_text = (arguments.get(key) for key in ("path", "old_text", "new_text"))
        replace_all = arguments.get("replace_all", False)
        if type(old_text) is not str or type(new_text) is not str or type(replace_all) is not bool:
            raise ValueError("old_text and new_text must be strings and replace_all a boolean")
        if not old_text:
            raise ValueError("old_text must not be empty")
        target = self._target(relative)
        if not target.is_file():
            raise ValueError("file does not exist; write_file creates new files")
        old, new = old_text.encode("utf-8"), new_text.encode("utf-8")
        if len(new) > WRITE_MAX_BYTES:
            raise ValueError("new_text is too large")
        data = target.read_bytes()
        count = data.count(old)
        if count == 0:
            raise ValueError("old_text was not found; read the file and copy the text exactly")
        if count > 1 and not replace_all:
            raise ValueError(f"old_text occurs {count} times; add surrounding text, or set replace_all")
        self._store(relative, target, data.replace(old, new))
        return {"replaced": count}

    def _write(self, arguments: dict[str, Any]) -> dict[str, Any]:
        relative, content = arguments.get("path"), arguments.get("content")
        if type(content) is not str:
            raise ValueError("content must be a string")
        data = content.encode("utf-8")
        if len(data) > WRITE_MAX_BYTES:
            raise ValueError("content is too large")
        self._store(relative, self._target(relative), data)
        return {"written_bytes": len(data)}

    def _revert(self, arguments: dict[str, Any]) -> dict[str, Any]:
        relative = arguments.get("path")
        target = self._target(relative)
        if relative not in self.touched:
            raise ValueError("this file has no changes to undo")
        original = self.pristine / relative
        if original.is_file():
            temporary = target.with_name(f".{uuid.uuid4().hex}.tmp")
            shutil.copy2(original, temporary)
            os.replace(temporary, target)
        elif target.exists():
            target.unlink()
        self.touched.discard(relative)
        self.unsynced.add(relative)  # the sandbox copy is put back too
        self.version += 1
        return {"reverted": relative}

    def snapshot(self) -> dict[str, bytes | None]:
        """Every answer file as it stands (None for one the answer deletes), to put back later with restore."""
        if self.terminal:
            return {}
        held: dict[str, bytes | None] = {}
        for relative in sorted(self.touched):
            target = self.edits / relative
            held[relative] = target.read_bytes() if target.is_file() and not target.is_symlink() else None
        return held

    def restore(self, snapshot: dict[str, bytes | None]) -> list[str]:
        """Put the answer back exactly as a snapshot held it; the files that changed, in order.

        A file the answer has now and the snapshot did not is reverted to the repository as supplied; every file the
        snapshot held gets its bytes back through _store, so the version moves on and the sandbox copy follows.
        """
        if self.terminal:
            return []
        changed: list[str] = []
        for relative in sorted(self.touched - set(snapshot)):
            self._revert({"path": relative})
            changed.append(relative)
        for relative, data in sorted(snapshot.items()):
            target = self._target(relative)
            current = target.read_bytes() if target.is_file() else None
            if current == data:
                self.touched.add(relative)
                continue
            if data is None:
                if target.exists():
                    target.unlink()
                self.touched.add(relative)
                self.unsynced.add(relative)
                self.version += 1
            else:
                self._store(relative, target, data)
            changed.append(relative)
        return changed

    def _search(self, arguments: dict[str, Any]) -> dict[str, Any]:
        pattern, relative = arguments.get("pattern"), arguments.get("path", ".")
        if type(pattern) is not str or not pattern or "\x00" in pattern or type(relative) is not str:
            raise ValueError("pattern and path must be non-empty strings")
        validate_relative_path(relative)
        # -H: a single file named as the path is searched too; without it grep leaves the name out, every hit was
        # dropped as malformed, and the model fell back to reading whole 32 KiB pages (6 of 14 Opus solves).
        command = ["grep", "-rnIbEH", "--null", "--exclude-dir=.git", "-e", pattern, "--", relative]
        environment = {"PATH": "/usr/bin:/bin", "LC_ALL": "C"}
        root = self.reader.root
        with subprocess.Popen(
            command, cwd=root, env=environment, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        ) as finder, subprocess.Popen(
            ["head", "-c", str(SEARCH_READ_BYTES)], env=environment,
            stdin=finder.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        ) as limiter:
            finder.stdout.close()  # so grep stops when head has read enough
            try:
                raw, _ = limiter.communicate(timeout=SEARCH_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                limiter.kill()
                finder.kill()
                raise ValueError("search took too long; search a smaller directory") from None
            finally:
                finder.kill()
        lines = raw.split(b"\n")
        cut_short = len(raw) >= SEARCH_READ_BYTES
        if cut_short:
            lines = lines[:-1]  # the last line may be partial
        matches: list[dict[str, Any]] = []
        for entry in lines:
            name, separator, rest = entry.partition(b"\x00")
            parts = rest.split(b":", 2)
            if not separator or len(parts) != 3:
                continue
            if len(matches) == SEARCH_MAX_MATCHES:
                cut_short = True
                break
            path = name.decode("utf-8", "replace")
            matches.append({
                "path": path[2:] if path.startswith("./") else path,
                "line": int(parts[0]),
                "offset": int(parts[1]),
                "text": parts[2].decode("utf-8", "replace")[:SEARCH_MAX_LINE_CHARS],
            })
        return {"matches": matches, "truncated": cut_short}

    def _patch(self) -> bytes:
        """Git, not the model, writes the diff, against only the files that were edited."""
        if not self.touched:
            return b""
        git = shutil.which("git")
        if git is None:
            raise ValueError("git is not available")
        with tempfile.TemporaryDirectory(dir=self.base) as temporary:
            repository, home = Path(temporary, "repository"), Path(temporary, "home")
            repository.mkdir()
            home.mkdir()
            environment = {
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "HOME": str(home),
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull,
                "LC_ALL": "C.UTF-8",
            }

            def run(*arguments: str) -> bytes:
                return subprocess.run(
                    [git, "-c", "core.autocrlf=false", "-c", "user.name=miner", "-c", "user.email=miner@invalid", *arguments],
                    cwd=repository, env=environment, stdin=subprocess.DEVNULL,
                    capture_output=True, timeout=30, check=True,
                ).stdout

            def place(tree: Path) -> None:
                for relative in sorted(self.touched):
                    source = tree / relative
                    if source.is_file():
                        destination = repository / relative
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, destination)

            run("init", "-q", "--template=")
            place(self.pristine)
            run("add", "-f", "-A")
            run("commit", "-q", "--allow-empty", "-m", "workspace")
            # A new index makes Git read every file. It would otherwise trust size and
            # timestamps, and miss an edit that kept the file the same size.
            (repository / ".git" / "index").unlink()
            place(self.edits)
            run("add", "-f", "-A")
            return run("diff", "--cached", "--no-renames", "--no-color", "--no-ext-diff", "HEAD")

    def _reset_scratch(self) -> None:
        if self.scratch.exists():
            shutil.rmtree(self.scratch)
        shutil.copytree(self.pristine, self.scratch, symlinks=True)
        self.unsynced = set(self.touched)
        if self.sandbox is not None and self.sandbox.owns_files:
            os.chown(self.state, self.sandbox.uid, self.sandbox.gid)
            for directory, _directories, files in os.walk(self.scratch):
                os.chown(directory, self.sandbox.uid, self.sandbox.gid)
                for name in files:
                    os.chown(os.path.join(directory, name), self.sandbox.uid, self.sandbox.gid, follow_symlinks=False)

    def reset_to_answer(self) -> None:
        """Make the sandbox copy the repository with only the answer applied, the state the grader starts from.

        Whatever commands left in the copy (files they created, generated output) is gone; /tmp, which the
        runner keeps under /state, is not touched. The answer's files reach the copy with the next command.
        """
        if not self.terminal:
            self._ensure_edits()
        self._reset_scratch()

    def save_tmp(self) -> Path | None:
        """Keep /tmp as the commands so far left it, for restore_tmp; None when no command has left one.

        The runner replaces its archive of /tmp after each command instead of rewriting it, so a link keeps this
        one as it is. OSError when it can be neither linked nor copied.
        """
        archive = self.state / "tmp.tar"
        if not archive.is_file() or archive.is_symlink():
            return None
        saved = self.base / "tmp-saved.tar"
        saved.unlink(missing_ok=True)
        try:
            os.link(archive, saved)
        except OSError:
            shutil.copy2(archive, saved)
        return saved

    def restore_tmp(self, saved: Path | None) -> None:
        """Put /tmp back as save_tmp kept it, so the commands run since then leave nothing in it."""
        archive = self.state / "tmp.tar"
        if saved is None:
            archive.unlink(missing_ok=True)
        else:
            os.replace(saved, archive)

    def put_scratch_file(self, relative: str, data: bytes) -> None:
        """Write one file into the sandbox copy only, never into the answer (the harness's own test file)."""
        validate_relative_path(relative)
        if relative == "." or ".git" in relative.split("/"):
            raise ValueError("path must name a file outside .git")
        target = self.scratch
        for segment in relative.split("/")[:-1]:
            target = target / segment
            if target.is_symlink() or (target.exists() and not target.is_dir()):
                raise ValueError(f"{relative}: a parent is not a directory")
        target = target / relative.split("/")[-1]
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise ValueError(f"{relative} is not a regular file")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        if self.sandbox is not None and self.sandbox.owns_files:
            os.chown(target.parent, self.sandbox.uid, self.sandbox.gid)
            os.chown(target, self.sandbox.uid, self.sandbox.gid)

    def _time_limit(self, wanted: int, time_left: float) -> int:
        limit = min(wanted, int(time_left) - 5)
        if limit < 1:
            raise ValueError("there is not enough time left to run this")
        return limit

    async def _run_command(self, arguments: dict[str, Any], time_left: float, adopt: bool = True) -> dict[str, Any]:
        command, wanted = arguments.get("command"), arguments.get("timeout_s", COMMAND_TIMEOUT_S)
        if self.sandbox is None:
            raise ValueError("commands are not available for this task")
        if type(command) is not str or not command or type(wanted) is not int or wanted < 1:
            raise ValueError("command must be a non-empty string and timeout_s a positive integer")
        if len(command) > COMMAND_MAX_CHARS:
            raise ValueError("command is too long")
        limit = self._time_limit(min(wanted, COMMAND_TIMEOUT_S), time_left)
        # On a repository task the answer is the edit tree, so what a command does to the
        # repository's files is looked at afterwards. A script task's answer is the script.
        # A rerun (adopt=False) is not looked at: nothing it does is taken, and the caller resets the copy.
        watching = adopt and not self.terminal
        if watching:
            await asyncio.to_thread(self._ensure_edits)
        if not self.scratch.exists():
            await asyncio.to_thread(self._reset_scratch)
        for attempt in range(2):
            stale = sorted(self.unsynced)
            mounts = [Mount(self.scratch, "/work", False), Mount(self.state, "/state", False)]
            if self.edits.exists():
                mounts.append(Mount(self.edits, "/edits", True))
            listed = b"".join(path.encode("utf-8") + b"\x00" for path in stale)
            if watching:
                listed += b"\x00" + b"".join(path.encode("utf-8") + b"\x00" for path in sorted(self.touched))
            result = await self.sandbox.run(
                ContainerRequest(
                    name=f"hone-miner-{uuid.uuid4().hex[:24]}",
                    argv=(
                        "/usr/bin/bash", "--noprofile", "--norc", "-c", _RUNNER, "runner", command, str(limit),
                        *(("watch",) if watching else ()),
                    ),
                    cwd=self.cwd,
                    mounts=tuple(mounts),
                    stdin=listed,
                    timeout_s=limit + 4 + (WATCH_S if watching else 0),
                    max_stdout_bytes=2 * (OUTPUT_HEAD_BYTES + OUTPUT_TAIL_BYTES),
                    max_stderr_bytes=OUTPUT_TAIL_BYTES,
                    trusted=False,
                ),
                wait_s=max(1.0, time_left - limit),
            )
            output = result.stdout.decode("utf-8", "replace") + result.stderr.decode("utf-8", "replace")
            if result.exit_code != 126 or "sandbox error:" not in output:
                break
            if attempt == 0 and "could not be copied in" in output:
                # An earlier command left something in the way, such as a link where a directory
                # was. Start again from a clean copy, which gets every edit, and run once more.
                await asyncio.to_thread(self._reset_scratch)
                continue
            raise ValueError(output.strip()[:300])  # the copy did not get the edits, so they stay unsynced
        self.unsynced.difference_update(stale)
        value: dict[str, Any] = {
            "exit_code": result.exit_code,
            "timed_out": _timed_out(result),
            "output": output,
        }
        if result.oom_killed:
            value["memory_limit_exceeded"] = True
        if watching:
            value.update(await asyncio.to_thread(self._take_changes, cut_short=_timed_out(result) or result.oom_killed))
        return value

    def _take_changes(self, *, cut_short: bool) -> dict[str, Any]:
        """Settle what the last command did to the repository's files, and what to tell the model.

        A changed answer file is taken into the answer as the command left it, since the model
        tests what the copy holds; a file put back to its original content, as git checkout or
        stash does, is not, so a stray command cannot wipe the answer. Everything else the
        command changed (another file of the repository, an answer file it removed, or a change
        from a command that was cut short) is put back from the answer before the next command,
        so that every test runs on what will be submitted.
        """
        updated: list[str] = []
        not_kept: list[str] = []
        kept: list[str] = []
        created: list[str] = []
        for kind, relative in self._read_changed():
            if kind == "N":
                # Left in the copy for the commands that follow, and named, since it is not submitted.
                self.created.add(relative)
                created.append(relative)
                continue
            if kind == "M" and not cut_short and self._adopt(relative):
                updated.append(relative)
                continue
            self.unsynced.add(relative)
            (not_kept if kind == "O" else kept).append(relative)
        value: dict[str, Any] = {}
        notes: list[str] = []
        if updated:
            value["answer_updated"] = updated[:CHANGES_REPORTED]
            notes.append(
                "This command changed files that your answer already changes (answer_updated). Your answer now has "
                "them exactly as the command left them. If a change was only for a test, undo it with edit_file."
            )
        if not_kept:
            value["not_in_answer"] = not_kept[:CHANGES_REPORTED]
            notes.append(
                "This command changed files of the repository that your answer does not change (not_in_answer). "
                "Commands run on a copy, so those changes are not submitted and are undone before the next command. "
                "Make every change you want submitted with edit_file or write_file."
            )
        if kept:
            value["answer_kept"] = kept[:CHANGES_REPORTED]
            notes.append(
                "This command removed, restored or rewrote files of your answer in a way that was not taken "
                + ("because it was cut short " if cut_short else "")
                + "(answer_kept). Your answer keeps its own version, which the copy gets back before the next "
                "command. Use revert_file to drop a file's changes from your answer."
            )
        if created:
            value["created_not_in_answer"] = created[:CHANGES_REPORTED]
            notes.append(
                "This command created files that are not part of your answer (created_not_in_answer). They stay in "
                "this copy for your next commands, but a file only a command created is never submitted. Write each "
                "one your change needs with write_file; leave throwaway tests and scratch files out."
            )
        if notes:
            value["note"] = " ".join(notes)
        return value

    def left_out(self) -> list[str]:
        """Files a command created that the copy still holds and the answer does not, tests and scratch files aside.

        4dc6113b5960 (2026-11-12) wrote its documented package pkg/fixerreply with `cat >`, tested it in the copy,
        and submitted without it: all 101 hidden tests failed to build. db84cf7b56bf (2026-11-11) lost its
        package's __init__.py the same way.
        """
        found = []
        for relative in sorted(self.created - self.touched):
            if _SCRATCH_RE.search(relative) or not relative.endswith(_SOURCE_SUFFIXES):
                continue
            current = self.scratch
            try:
                for segment in relative.split("/"):
                    current = current / segment
                    info = os.lstat(current)
                    if stat.S_ISLNK(info.st_mode):
                        break
                else:
                    if stat.S_ISREG(info.st_mode):
                        found.append(relative)
            except OSError:
                continue
        return found[:CHANGES_REPORTED]

    def _read_changed(self) -> list[tuple[str, str]]:
        """The runner's list of changed files. The command could have written it, so every entry is checked."""
        try:
            data = _read_regular(self.state / "changed", 4 * 1024**2)
        finally:
            try:
                os.unlink(self.state / "changed")
            except OSError:
                pass
        if data is None:
            return []
        fields = data.split(b"\x00")
        records: list[tuple[str, str]] = []
        seen: set[str] = set()
        for kind, name in zip(fields[0::2], fields[1::2]):
            try:
                relative = validate_relative_path(name.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                continue
            if relative == "." or ".git" in relative.split("/") or relative in seen:
                continue
            answer_file = relative in self.touched
            created = kind == b"N" and not answer_file and not os.path.lexists(self.pristine / relative)
            if (kind in (b"M", b"D") and answer_file) or (kind == b"O" and not answer_file) or created:
                seen.add(relative)
                records.append((kind.decode("ascii"), relative))
        return records

    def _adopt(self, relative: str) -> bool:
        """Take a command's version of an answer file into the answer, when it is text that edit_file could have written."""
        data = self._read_scratch(relative)
        if data is None:
            return False
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            return False
        original = self.pristine / relative
        try:
            if stat.S_ISREG(os.lstat(original).st_mode) and original.read_bytes() == data:
                return False  # put back to the original: the answer keeps its change
        except OSError:
            pass
        try:
            target = self._target(relative)
        except ValueError:
            return False
        self._store(relative, target, data)
        self.unsynced.discard(relative)  # the copy already holds exactly this
        return True

    def _read_scratch(self, relative: str) -> bytes | None:
        """A file as a command left it in the copy, read without following any link the command may have planted."""
        current = self.scratch
        parts = relative.split("/")
        try:
            for segment in parts[:-1]:
                current = current / segment
                if not stat.S_ISDIR(os.lstat(current).st_mode):
                    return None
        except OSError:
            return None
        return _read_regular(current / parts[-1], WRITE_MAX_BYTES)

    async def _run_script(self, time_left: float) -> dict[str, Any]:
        if self.sandbox is None:
            raise ValueError("scripts cannot be run for this task")
        if not self.script.is_file():
            raise ValueError(f"write {SCRIPT_PATH} first")
        limit = self._time_limit(SCRIPT_TIMEOUT_S, time_left)
        version = self.version
        await asyncio.to_thread(self._reset_scratch)
        result = await self.sandbox.run(
            ContainerRequest(
                name=f"hone-miner-{uuid.uuid4().hex[:24]}",
                argv=("/usr/bin/bash", "--noprofile", "--norc", "/submission/script.sh"),
                cwd=self.cwd,
                mounts=(Mount(self.scratch, "/work", False), Mount(self.control, "/submission", True)),
                stdin=b"",
                timeout_s=limit,
                max_stdout_bytes=SCRIPT_OUTPUT_BYTES,
                max_stderr_bytes=SCRIPT_OUTPUT_BYTES,
                trusted=False,
            ),
            wait_s=max(1.0, time_left - limit),
        )
        value: dict[str, Any] = {
            "exit_code": result.exit_code,
            "timed_out": _timed_out(result),
            "stdout": _shorten(result.stdout),
            "stderr": _shorten(result.stderr),
            **await asyncio.to_thread(self._changes),
        }
        if result.oom_killed:
            value["memory_limit_exceeded"] = True
        if result.stdout_overflow or result.stderr_overflow:
            value["output_limit_exceeded"] = True
        if result.exit_code == 0 and not (_timed_out(result) or result.oom_killed or "output_limit_exceeded" in value):
            self.rehearsed = version
        return value

    async def gate(self, time_left: float) -> tuple[bool, str | None]:
        """Check the answer on a clean copy, the way the grader starts, before it is submitted.

        A script must run cleanly from scratch, unless that exact version already did. A
        repository change must still build with the task's own .rlvr/build.py, when it has
        one. Returns whether a check ran and the problem it found.
        """
        if self.sandbox is None or not self.has_work():
            return False, None
        budget = time_left - GATE_RESERVE_S
        try:
            if self.terminal:
                if self.rehearsed == self.version:
                    return False, None
                result = await self._run_script(budget)
                if self.rehearsed == self.version:
                    return True, None
                limits = [name for name in ("timed_out", "memory_limit_exceeded", "output_limit_exceeded") if result.get(name)]
                return True, (
                    f"run from a clean copy as the grader will, the script failed with exit code {result['exit_code']}"
                    + (f" ({', '.join(limits)})" if limits else "")
                    + f". Its stderr ends with:\n{result['stderr'][-2000:]}"
                )
            if not (self.pristine / ".rlvr" / "build.py").is_file():
                return False, None
            await asyncio.to_thread(self._reset_scratch)
            result = await self._run_command(
                {"command": "cd /work && python3 .rlvr/build.py", "timeout_s": COMMAND_TIMEOUT_S}, budget
            )
            if result["exit_code"] == 0:
                return True, None
            return True, f"with the change applied to a clean copy, .rlvr/build.py fails. Its output ends with:\n{result['output'][-2000:]}"
        except (ValueError, OSError, asyncio.TimeoutError, SupervisorError, subprocess.SubprocessError):
            return False, None  # it cannot be checked in the time left; submit it as it is

    def _changes(self) -> dict[str, Any]:
        """What a script left behind. Metadata only: nothing in this tree is trusted enough to open."""
        created: list[str] = []
        modified: list[str] = []
        for directory, _directories, files in os.walk(self.scratch, followlinks=False):
            for name in files:
                path = Path(directory, name)
                relative = path.relative_to(self.scratch).as_posix()
                info = path.lstat()
                try:
                    original = (self.pristine / relative).lstat()
                except OSError:
                    created.append(relative)
                    continue
                if (info.st_size, info.st_mtime_ns) != (original.st_size, original.st_mtime_ns):
                    modified.append(relative)
        return {
            "files_created": sorted(created)[:CHANGES_LISTED],
            "files_created_total": len(created),
            "existing_files_changed": sorted(modified)[:CHANGES_LISTED],
            "existing_files_changed_total": len(modified),
        }
