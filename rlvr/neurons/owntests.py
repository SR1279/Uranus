"""The own-test regression net: the model's earlier test commands, run again on the final version.

A change made late in a solve, after a suite case or a probe, can break behaviour that one of the model's own
earlier tests had shown working. 966efbb1635f (2026-11-13) is the recorded case: its throwaway test
`/tmp/zz2.go` printed `ok` at workspace-17 and workspace-18, encoding lanes that are global by name, which the
answer key confirms. A wrong suite case then moved it to per-key lanes, the same test ended in a goroutine trace
at workspace-27, and the answer failed. The model saw that output and set the test aside as outdated.

So the harness remembers the model's test commands as they run (`OwnRuns.record`, at no cost), and after the
final check it runs the last few of them once more on the final version, in a copy reset to the answer, taking
nothing they change into the answer (`Workbench.rerun`). Each rerun is compared with that command's own earlier
output, never with marker words alone, because the model's negative probes print the very errors they expect
(review.md). Only an unambiguous marker counts as newly broken: a build, compile or import error, or an uncaught
exception, that an earlier run of the same command did not print. A test runner's FAIL after an earlier ok is
recorded as ambiguous, because a right fix makes exactly the old tests fail that encoded the old reading
(4f5bc0cfb1eb's first test asserted that a limit of 0 means unlimited). An assertion's own traceback or panic is
such a failure, not a crash.

The result goes into the solve record (`own_tests` in tasks.csv). Asking the model about a breakage is a later
switch (setting `ask`), for unambiguous breakage only, once the record shows no false alarm on passed solves.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

# Distinct qualifying commands remembered, the most recent kept.
KEPT = 6
# At most this many are run again, the most recent among those last run on an older version.
RERUNS = 3
# Each rerun's limit: twice its original duration, at most this long.
RERUN_MAX_S = 30
# The reruns start only with this much time left beyond RESERVE_S, which always stays for the upload.
MIN_TIME_S = 60.0
RESERVE_S = 30.0
# A rerun also needs its container started and its result read.
START_S = 5.0
# In ask mode, a breakage goes back only with this much time left before the wrap-up.
ASK_MIN_S = 60.0
HEAD_CHARS = 100
TAIL_CHARS = 1500

_ANSWER_CHANGES = ("answer_updated", "not_in_answer", "answer_kept")

# Here-document bodies are data, not commands: a Python script written to /tmp says nothing about what runs.
_HEREDOC_RE = re.compile(r"(?<!<)<<(?!<)(-?)\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\2")
# A test program at a command position: after a separator (a command substitution's $( too), a keyword,
# environment assignments and wrappers. A closing ) or } is no command position (the shell wants a separator after
# it), nor is the { of ${...}, nor a keyword inside a word (a=do).
_COMMAND_START = r"(?:^|[;&|(!]|(?<!\$)\{|(?<![^\s;&|({!])(?:then|do|else)\b)"
# A word stops at a separator outside $(...) and ${...}, and the wrappers at the end of the line, so that each
# command position is scanned only up to the next one: with \S and \s here, a 64 KiB command of `a=b;a=b;...` or
# of `a=b` lines took seconds to minutes, on the event loop.
_WORD = r"(?:[^\s;&|(){}!$]|\$\([^\s()]*\)|\$\{[^\s{}]*\}|\$)*"
_WRAPPERS = (
    r"(?:[ \t]*(?:[A-Za-z_][A-Za-z0-9_]*=" + _WORD + r"|time|exec|nice|command|env"
    r"|timeout(?:[ \t]+-{1,2}[\w=-]+)*[ \t]+\d+(?:\.\d+)?[smhd]?)(?=[ \t]))*"
)
_PROGRAMS = (
    r"(?:" + _WORD + r"/)?python(?:3(?:\.\d+)?)?(?![\w.-])"  # python3, also unittest and pytest run with -m
    r"|(?:" + _WORD + r"/)?node(?![\w.-])"
    r"|go\s+test\b"
    r"|cargo\s+(?:\+\S+\s+)?test\b"
    r"|/tmp/\S"  # a program or script under /tmp
    r"|(?:ba|da|z)?sh\s+(?:-\w+\s+)*/tmp/\S"
    r"|go\s+run\s+(?:-\S+\s+)*/tmp/\S"
)
_TEST_PROGRAM_RE = re.compile(_COMMAND_START + _WRAPPERS + r"[ \t]*(?:" + _PROGRAMS + ")", re.MULTILINE)
# The suite's Go test functions are named TestAcceptance... and run by that filter (acceptance.AUTHOR_PROMPT).
_GO_SUITE_FILTER_RE = re.compile(r"-run[= ]\s*['\"]?\^?\(?TestAcceptance")

# Unambiguous markers, in the order a report names the first one found.
# A vet report has a second header line, `# [pkg]`, and is a warning unless go test stops on it, which then says
# [build failed]; a compile error follows its package line directly.
_GO_BUILD_RE = re.compile(r"(?m)^# [^\s\[]+(?: \[[^\]\n]+\])?\n\S+\.go:\d+:\d+: ")
_NODE_ERROR_RE = re.compile(r"(?m)^(SyntaxError|ReferenceError|TypeError)(?: \[\w+\])?: [^\n]*\n[ \t]+at \S")
# A caught error printed with console.log(e) shows the same name and stack, and a negative probe prints exactly
# that. Node marks an uncaught one: the source line with a caret under it and a blank line just before the error
# (a compile error carries these even when caught), and `Node.js vN` after its stack and the dump of its
# properties. A pipe may cut one of the two, so either is enough.
_NODE_ARROW_RE = re.compile(r"(?:\A|\n)[ \t]*\^+[ \t]*\n[ \t]*\n\Z")
_NODE_TRAILER_RE = re.compile(r"[^\n]*\n(?:(?:[ \t][^\n]*|\}?)\n){0,400}Node\.js v\d")
# An error line followed by a JavaScript stack frame is a JavaScript error, which only the Node rule judges.
_NOT_JS = r"(?![^\n]*\n[ \t]+at \S)"
_MARKERS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("SyntaxError", re.compile(r"(?m)^(?:E\s+)?(?:SyntaxError|IndentationError|TabError)\b" + _NOT_JS)),
    ("ModuleNotFoundError", re.compile(r"(?m)^(?:E\s+)?ModuleNotFoundError\b")),
    ("ImportError", re.compile(r"(?m)^(?:E\s+)?ImportError\b")),
    ("Go build error", _GO_BUILD_RE),
    ("build failed", re.compile(r"\[build failed\]")),
    ("panic", re.compile(r"(?m)^panic: ")),
    ("goroutine trace", re.compile(r"(?m)^goroutine \d+ \[running\]|^[ \t]+\S+\.go:\d+ \+0x[0-9a-f]+[ \t]*$")),
    ("Rust compile error", re.compile(r"error\[E\d{4}\]")),
    ("could not compile", re.compile(r"\bcould not compile\b")),
)
_TRACEBACK = "Traceback (most recent call last):"
_PANICKED_RE = re.compile(r"panicked at[^\n]*(?:\n[^\n]*)?")

# A test runner's verdict lines: Go, unittest, pytest, node:test and TAP, cargo test.
_PASSED_RE = re.compile(r"(?m)^(?:ok\s+\S|PASS\s*$|OK\b|test result: ok\.|# fail 0\s*$)|\b[1-9]\d* passed\b")
_FAILED_RE = re.compile(
    r"(?m)^(?:--- FAIL|FAIL\b|FAILED\b|FAIL: |ERROR: |not ok \d|# fail [1-9]|test result: FAILED)|\b[1-9]\d* failed\b"
)


def program_text(command: str) -> str:
    """The command without its here-document bodies."""
    lines = command.split("\n")
    kept: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        kept.append(line)
        index += 1
        for match in _HEREDOC_RE.finditer(line):
            strip_tabs, delimiter = match.group(1) == "-", match.group(3)
            while index < len(lines):
                body = lines[index].lstrip("\t") if strip_tabs else lines[index]
                index += 1
                if body == delimiter:
                    break
    return "\n".join(kept)


def runs_tests(command: str) -> bool:
    """Whether the command runs a test program: python3, node, go test, cargo test, unittest, or a /tmp script."""
    return _TEST_PROGRAM_RE.search(program_text(command)) is not None


def names_suite(command: str, suite_path: str = "", suite_run: str = "") -> bool:
    """Whether the command runs the harness's own suite: by its path, its file or module name, or its command."""
    flat = " ".join(command.split())
    if suite_run and " ".join(suite_run.split()) in flat:
        return True
    if not suite_path:
        return False
    names = {suite_path}
    if suite_path.startswith("/work/"):
        names.add(suite_path[len("/work/"):])
    base = suite_path.rsplit("/", 1)[-1]
    stem = base.rsplit(".", 1)[0]
    # A short file name would also match unrelated text; the suite's own names are long (test_x_acceptance).
    names.update(name for name in (base, stem) if len(name) >= 6)
    if any(name in command for name in names):
        return True
    return base.endswith("_test.go") and _GO_SUITE_FILTER_RE.search(command) is not None


def qualifies(command: Any, result: Any, suite_path: str = "", *, suite_run: str = "") -> bool:
    """Whether a command's run can be compared later: it ran a test program to its end, did not run the harness's
    suite, and changed no file of the answer (answer_updated, not_in_answer and answer_kept are all empty).

    A run cut short by its limit or by memory decides nothing: its output stops wherever it was cut.
    """
    if type(command) is not str or not command.strip() or type(result) is not dict or "error" in result:
        return False
    if type(result.get("exit_code")) is not int or result.get("timed_out") or result.get("memory_limit_exceeded"):
        return False
    if any(result.get(key) for key in _ANSWER_CHANGES):
        return False
    return runs_tests(command) and not names_suite(command, suite_path, suite_run)


def _tracebacks(output: str) -> tuple[bool, bool]:
    """(an uncaught exception's traceback, an assertion's traceback) among the Python tracebacks printed."""
    crashed = asserted = False
    for chunk in output.split(_TRACEBACK)[1:]:
        exception = next((line for line in chunk.split("\n")[1:] if line.strip() and not line[0].isspace()), "")
        if re.match(r"(?:\w+\.)*AssertionError\b", exception):
            asserted = True
        else:
            crashed = True
    return crashed, asserted


def _panics(output: str) -> tuple[bool, bool]:
    """(a Rust panic, an assertion's panic): a failed assert! panics too, and says so in its message."""
    crashed = asserted = False
    for match in _PANICKED_RE.finditer(output):
        if "assertion" in match.group(0):
            asserted = True
        else:
            crashed = True
    return crashed, asserted


def markers(output: str) -> set[str]:
    """The unambiguous breakage markers in one run's output.

    Python: an uncaught exception's traceback (not an assertion's), SyntaxError, ModuleNotFoundError, ImportError.
    Go: a panic, a goroutine trace or one of its frame lines, a `# pkg` line followed by `file.go:N:M:`,
    `[build failed]`. Node: an uncaught SyntaxError, ReferenceError or TypeError with its stack, not one a probe
    caught and printed. Rust: `error[E....]`, `could not compile`, a panic that is not an assertion's.
    """
    found = {name for name, pattern in _MARKERS if pattern.search(output)}
    if _tracebacks(output)[0]:
        found.add("Traceback")
    for match in _NODE_ERROR_RE.finditer(output):
        before = output[max(0, match.start() - 400):match.start()]
        if _NODE_ARROW_RE.search(before) or _NODE_TRAILER_RE.match(output, match.start()):
            found.add(f"uncaught {match.group(1)}")
    if _panics(output)[0]:
        found.add("panicked")
    return found


@dataclass(frozen=True)
class Observation:
    """What one run of a command showed."""

    markers: frozenset[str]
    passed: bool  # a test runner reported a pass and no failure
    failed: bool  # a test runner reported a failure, or an assertion failed
    exit_code: int | None
    cut_short: bool  # stopped by its time limit or memory, so it decides nothing


def observe(result: dict[str, Any]) -> Observation:
    output = str(result.get("output", ""))
    failed = (
        _FAILED_RE.search(output) is not None or _tracebacks(output)[1] or _panics(output)[1]
    )
    exit_code = result.get("exit_code")
    return Observation(
        markers=frozenset(markers(output)),
        passed=_PASSED_RE.search(output) is not None and not failed,
        failed=failed,
        exit_code=exit_code if type(exit_code) is int else None,
        cut_short=bool(result.get("timed_out") or result.get("memory_limit_exceeded")),
    )


@dataclass(frozen=True)
class OwnRun:
    """One distinct qualifying command and its recorded runs."""

    command: str
    version: int  # the edit tree's version at its latest run
    duration: float  # seconds its latest run took
    markers: frozenset[str]  # the markers every recorded run printed: one run without a marker makes it new later
    passed: bool  # a recorded run passed
    exit_zero: bool  # a recorded run exited 0
    runs: int = 1


class OwnRuns:
    """The last KEPT distinct qualifying commands the model ran, oldest first."""

    def __init__(self, kept: int = KEPT):
        self.kept = kept
        self.entries: dict[str, OwnRun] = {}

    def record(
        self, command: Any, version: int, duration: float, result: Any, *, suite_path: str = "", suite_run: str = "",
    ) -> bool:
        """Remember one run of a model's command; False when it does not qualify."""
        if not qualifies(command, result, suite_path, suite_run=suite_run):
            return False
        seen = observe(result)
        earlier = self.entries.pop(command, None)
        if earlier is None:
            entry = OwnRun(command, version, duration, seen.markers, seen.passed, seen.exit_code == 0)
        else:
            entry = OwnRun(
                command, version, duration, earlier.markers & seen.markers, earlier.passed or seen.passed,
                earlier.exit_zero or seen.exit_code == 0, earlier.runs + 1,
            )
        self.entries[command] = entry
        while len(self.entries) > self.kept:
            del self.entries[next(iter(self.entries))]
        return True

    def due(self, version: int, *, limit: int = RERUNS, exclude: Callable[[str], bool] | None = None) -> list[OwnRun]:
        """The most recent commands last run on a version older than this one, oldest first."""
        older = [
            entry for entry in self.entries.values()
            if entry.version < version and not (exclude is not None and exclude(entry.command))
        ]
        return older[-limit:] if limit > 0 else []


def limit_s(duration: float) -> int:
    """A rerun's limit: twice what the command took before, at most RERUN_MAX_S."""
    return max(1, min(RERUN_MAX_S, math.ceil(2 * max(0.0, duration))))


def breakage(before: OwnRun, after: Observation) -> str | None:
    """The first unambiguous marker the rerun printed that an earlier run of the same command did not."""
    if after.cut_short:
        return None
    new = after.markers - before.markers
    for name in ("Traceback", *(name for name, _pattern in _MARKERS), "panicked"):
        if name in new:
            return name
    return min(new) if new else None  # the uncaught Node errors


def ambiguous(before: OwnRun, after: Observation) -> str | None:
    """A change that may be a right fix as well as a regression: FAIL after ok, or a non-zero exit after 0."""
    if after.cut_short:
        return None
    if before.passed and after.failed:
        return "FAIL after ok"
    if before.exit_zero and after.exit_code not in (0, None):
        return f"exit {after.exit_code} after 0"
    return None


@dataclass(frozen=True)
class Rerun:
    entry: OwnRun
    result: dict[str, Any]
    broken: str | None = None
    unclear: str | None = None

    @property
    def ran(self) -> bool:
        return "error" not in self.result

    @property
    def cut_short(self) -> bool:
        return self.ran and bool(self.result.get("timed_out") or self.result.get("memory_limit_exceeded"))


def compare(entry: OwnRun, result: dict[str, Any]) -> Rerun:
    if "error" in result:
        return Rerun(entry, result)
    seen = observe(result)
    broken = breakage(entry, seen)
    return Rerun(entry, result, broken, None if broken else ambiguous(entry, seen))


def head(command: str, limit: int = HEAD_CHARS) -> str:
    flat = " ".join(command.split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


def tail(result: dict[str, Any], limit: int = TAIL_CHARS) -> str:
    output = str(result.get("output", "")).rstrip()
    return output if len(output) <= limit else "[...]\n" + output[-limit:]


def summary(reruns: Iterable[Rerun], *, skipped: int = 0) -> str:
    """For the solve record, e.g. "3 rerun, 1 newly broken (goroutine trace): cp /tmp/zz2.go ..."; `skipped` were
    due but left out for lack of time."""
    reruns = list(reruns)
    ran = [rerun for rerun in reruns if rerun.ran]
    broken = [rerun for rerun in ran if rerun.broken]
    unclear = [rerun for rerun in ran if rerun.unclear]
    cut = sum(rerun.cut_short for rerun in ran)
    parts = [f"{len(ran)} rerun"]
    if broken:
        parts.append(f"{len(broken)} newly broken ({broken[0].broken}): {head(broken[0].entry.command)}")
    elif ran:
        parts.append("none newly broken")
    if unclear:
        parts.append(f"{len(unclear)} ambiguous ({unclear[0].unclear}): {head(unclear[0].entry.command)}")
    if cut:
        parts.append(f"{cut} cut short")
    if len(ran) < len(reruns):
        parts.append(f"{len(reruns) - len(ran)} could not run")
    if skipped:
        parts.append(f"{skipped} left out for time")
    return ", ".join(parts)


def prompt(rerun: Rerun) -> str:
    """The one question of ask mode, for an unambiguous breakage."""
    return (
        f"Your earlier command `{head(rerun.entry.command)}` ran cleanly on an earlier version; on the final "
        f"version it now ends in {rerun.broken}:\n\n```\n{tail(rerun.result)}\n```\n\n"
        "Either a later change broke it or the command is out of date: decide from the documentation which; "
        "where a passage supports a deliberate change, keep it and correct the command, otherwise fix the code; "
        "run it again and reply with a one-line summary."
    )
