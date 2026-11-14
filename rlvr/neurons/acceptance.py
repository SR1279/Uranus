"""An acceptance suite written from the documentation alone, run before the answer is submitted.

The hidden checks of every repository task seen so far come from one task generator and follow one
template: one strict case per rule the documentation states, grouped into tests named after the
rules. The model that writes the answer also writes its own tests, so a rule it misreads passes its
tests and fails the hidden one; every reasoning failure after the sandbox fix was of that kind. Here
a second model call that sees only the task text and the documentation, never the code, writes such
a suite in the authors' style while the answer is being written. The harness runs it against the
answer, and where the two readings differ, the difference is settled against the documentation before
the verdict instead of after it.
"""

from __future__ import annotations

import json
import re
import shlex
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .adjudicate import (
    EDGE_RULES, ENCODING_RULE, ERROR_SCOPE, EXAMPLE_RULE, GO_ERROR_FORM, ONE_TYPE, STRICT_LIMIT, WRITER_ENCODING_RULE,
)

# "documented in docs/x.md (already in the repository)", or several files: "documented in CMakeLists.txt,
# docs/environment-plans.md (already in the repository)".
DOC_CLAUSE_RE = re.compile(r"documented in ([^()\n]+?)\s*\(already in the repository\)")
DOC_PATH_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./-]*\.[A-Za-z0-9]+")
DOC_REFERENCE_RE = re.compile(r"documented in ([A-Za-z0-9_][A-Za-z0-9_./-]*\.[A-Za-z0-9]+)")
MAX_DOC_CHARS = 160_000
MAX_SUITE_CHARS = 48_000  # with the heredoc around it, well inside a 64 KiB command
SUITE_DELIMITER = "__HONE_ACCEPTANCE_SUITE__"
RUN_TIMEOUT_S = 120
OUTPUT_LIMIT_CHARS = 9_000
# The suite is skipped when less than this is left before the wrap-up, so that fixing what it finds still fits.
MIN_TIME_S = 240.0
# How long the first finished answer waits for the author's call when that is still running.
AUTHOR_WAIT_S = 90.0
# After its first run the suite runs again on every finished version, and the cases still failing go back to
# the model until each passes or is refuted with a passage of the documentation, or until this share of the
# solve's time has passed: about 660 of the 880 seconds a 900-second lease leaves, so the review still fits.
# On 2026-11-13 the median solve ended at about 250 seconds, and on 11 of the 24 tasks the fleet passed and we
# failed the suite was still red when the answer went out; the top solvers keep working to about 10 minutes.
SETTLE_FRACTION = 0.75
# The suite runs at most this many times in one solve, unless MINER_ACCEPTANCE_RUNS says otherwise.
MAX_RUNS = 5
# A rerun is skipped when less than this is left before the wrap-up.
RERUN_MIN_TIME_S = 60.0
# A quotation settles a case only when it is in the documentation word for word and this long once normalised,
# each part between ellipses at least FRAGMENT_CHARS: a few words are found anywhere.
MIN_QUOTE_CHARS = 24
FRAGMENT_CHARS = 8
# At most this many open cases are named in one message.
MAX_LISTED = 30
# Before the answer goes out, the suite runs once more on a clean copy of the repository with only the answer
# applied (the state the grader starts from), when at least this much is left before the deadline.
FINAL_CHECK_MIN_S = 45.0
_HASH_RE = re.compile(r"\A\s*([0-9a-f]{64})\b")
# Each run prints, after its whole output, the runner's failure lines again under this mark, so that the tail the
# workbench keeps (the first 4 KiB and the last 12 KiB) always holds every failing case; a long run's middle used to
# hide some, and a suite was called settled while cases still failed (review of 2026-11-13).
FAIL_LIST_MARK = "=== failing cases ==="
FAIL_LIST_LINES = 50
_FAIL_LIST_RE = re.compile(r"^=== failing cases === (\d+)\s*$", re.MULTILINE)
_FAIL_LINE_GREP = (
    r"^[[:space:]]*--- FAIL|^(FAIL|ERROR): |^[[:space:]]*not ok |^test .* \.\.\. (FAILED|FAIL|ERROR)[[:space:]]*$"
    r"|^panic: |^FAILED \(|^# (fail|tests) |\[no tests to run\]|^Ran [0-9]+ tests?"
)
# A unittest fixture that failed stopped the cases under it: no quotation can settle that.
_UNITTEST_FIXTURES = frozenset({"setUpClass", "tearDownClass", "setUpModule", "tearDownModule"})
# A panic aborts the Go test binary, so the cases after it never ran.
_GO_PANIC_RE = re.compile(r"^panic: ", re.MULTILINE)
PANIC_CASE = "(a panic stopped the Go test binary, so the cases after it did not run)"
# A run that ran no case at all (the suite file is missing, or no test matched) never counts as passing.
_NO_CASES_RE = re.compile(r"\[no tests? to run\]|\[no test files\]|^Ran 0 tests|^# tests 0\b", re.MULTILINE)

# What the nine hidden suites held so far check, in the authors' own groupings.
HIDDEN_TEST_PATTERNS = (
    "Every hidden suite seen so far checks the same things, usually several per test: the public surface "
    "exactly (exported names, error class identity and inheritance, read-only properties); each validation "
    "rule with its exact error code or exception type, a grammar stated for a kind of value (\"header names must "
    "...\", \"values must contain no ...\") checked again at every other input whose documented type is that kind "
    "of value (a record's header patch as well as the profile's header config) with that input's own code and "
    "path, and the exact-type traps (a bool where an int is "
    "required, a bytearray where bytes is required, a string where an object is required, a key present with the "
    "value undefined or None); the order in which checks run, with inputs that break two rules at once; every "
    "limit at its inclusive boundary and one past it; empty, whitespace-only and after-normalization-empty "
    "values; Unicode (NFC against NFKC, case folding only where the documentation says, case-sensitive "
    "identifiers, no trimming where none is documented); atomicity, where a rejected operation leaves content, "
    "counters, revisions and rendered output unchanged, checked by comparing the whole state before and after; "
    "detachment, where returned collections and arguments share nothing with internal state, checked by "
    "mutating them; deterministic ordering of results, diagnostics and serialized keys; checkpoint or snapshot "
    "round trips, after which the restored object answers every public call as the original did (values "
    "recorded as of an earlier step included) and accepts or rejects the same next operations, and a matrix of "
    "malformed checkpoints (each key missing, extra, of the wrong type, wrong "
    "version), each rejected with the documented code; memoization keyed on the documented identity, with "
    "cached results detached; parity with the project's own behaviour for the same input (what its parser, "
    "renderer or encoder produces natively), and the parts a feature copies from the repository's shipped files "
    "compared with == to those files loaded the documented way (runpy, json, the project's loader); the documentation's own examples verbatim; collaborators replaced "
    "by minimal stand-ins with only the documented methods, by overriding subclasses, or helpers swapped after "
    "import; callbacks that record how often and with what they are called; streamed input split at every "
    "byte with the same results and offsets; fields that must agree, made to disagree; inputs that change "
    "between two calls (environment, files) read afresh; exceptions from caller code passed through as the "
    "same object; integration files (exports, __all__, payload lists) parsed for the new names; structures "
    "deeper than the recursion limit; a command line's exit code and what stdout and stderr hold for each outcome, exactly as the documentation states them; and, where "
    "promised, concurrent use."
)

AUTHOR_SYSTEM_PROMPT = (
    "You write the hidden acceptance test suite for a programming task, the way the task's author did: from the "
    "documentation and the public interface alone. You never see the implementation and must not guess at it; "
    "every expected value comes from the documentation's words and examples. Where the documentation can be "
    "read two ways, take the stricter reading: an input at the edge of a rule violates the rule (empty after "
    "decoding is empty, whitespace-only is empty where empty is invalid, a key present with the value undefined "
    "or None is not an absent key, a selector or name that fails its grammar fails even when its target does not "
    "exist), unless an example or the project's native behaviour shows otherwise. "
    + EDGE_RULES
    + " "
    + STRICT_LIMIT
    + " So never expect a decoder to refuse such whitespace, and assert that a valid input in a layout the "
    "encoder never writes is accepted (spaces and newlines around tokens and after the document, members in "
    "another order, an escaped member name, a document padded with spaces to exactly its size limit) unless a "
    "documented decoder rule rejects it. "
    + GO_ERROR_FORM
    + " "
    + ONE_TYPE
    + " A table of error codes is read "
    "row by row and literally: a code covers only the fields and conditions its own row names, even where another "
    "section gives more fields the same bound, and a defect that no row names takes the table's general code, "
    "such as the schema or type error. "
    + ERROR_SCOPE
    + "\n\n"
    + HIDDEN_TEST_PATTERNS
    + "\n\nTest only what the documentation states. Each case checks a rule that a sentence of the documentation "
    "states or an example shows, and that sentence is copied word for word into a comment beginning RULE: above "
    "the test or table row that checks it; a behaviour that no sentence or example states gets no case, however "
    "usual it is elsewhere, because the list above says where hidden suites look, not what to assume. Never "
    "leave a case disabled (a skip decorator, skipTest, t.Skip, test.skip, if False): decide its expected value or "
    "leave the case out. An assertion about stdout after an error needs a RULE: sentence that names stdout for "
    "that same outcome; without one, assert only the exit code and that stderr is not empty. Begin with "
    "one test of the public interface that uses every name the documentation gives it, with the signature it "
    "gives: in Go, assign each documented function and method to a variable of its documented type (var _ "
    "func(string, int) (*Thing, error) = pkg.NewThing; var _ func(*pkg.Thing, string) error = "
    "(*pkg.Thing).Add), so that a different signature fails the build; in Python, import each name as the "
    "documentation's import lines do and, where the documentation writes a signature out, compare the parameter "
    "names, order and defaults of inspect.signature with it; in JavaScript or TypeScript, import each documented "
    "export and check that it is a function or a class."
    "\n\nWrite one strict case per rule the documentation states, grouped into tests named after the rules, "
    "with exact assertions on whole values (never substring checks), table-driven where the documentation lists "
    "cases. "
    "A case builds only the state its own rule needs: give each case, and each table row that changes state, its own fresh instance (store, tape, queue, session, file), made by one prefixed builder helper so the suite stays short. Where one case must hold several records, give each record that the case is not about its own key, lane, group or quota. That way no ordering, head-of-line, exhaustion, sharing or limit rule the case does not test can decide its outcome. An interaction between two rules gets a case only where a RULE: sentence states that interaction itself. "
    "Test precedence with inputs that break two rules at once. Where the documentation says the feature "
    "reads files that ship in the repository, add one case that runs it on those real files under /work and "
    "compares each part it says is copied or kept verbatim, with ==, to a copy.deepcopy of the same part loaded "
    "the way the documentation loads it; in a fixture you build, give each verbatim-kept value a tuple, a nested "
    "container and a non-string scalar, because == tells a tuple from a list. Derive every expected number from the "
    "inputs you build: set each limit to the exact size of an input you feed, accepted at that size and rejected "
    "at one less, and measure sizes on the input as given, counting exactly the unit the documentation names "
    "(a delimiter it keeps out of that unit is not counted). Include the documentation's examples verbatim and "
    "assert them exactly as printed, weighed by this rule: " + EXAMPLE_RULE + " Where a sentence decides against an "
    "example, assert the sentence with inputs of your own and leave that example out, with a comment beginning "
    "CONFLICT: that quotes both passages. Keep the suite "
    "deterministic, offline, free of new dependencies and under 400 lines. It runs in "
    "the task's sandbox as an unprivileged user, from the workspace root /work, and must be runnable by one "
    "shell command that exits non-zero when any case fails and prints one line per case:\n"
    "- Python: a unittest module placed under /tmp, run as `cd /tmp && PYTHONPATH=/work:/work/src python3 -m "
    "unittest -v MODULE`, keeping whatever PYTHONPATH or import rule the task states;\n"
    "- JavaScript or TypeScript: a Node 18 node:test file under /tmp that imports the project's files by "
    "absolute /work paths, run with `node --test FILE`;\n"
    "- Go: a `_test.go` file in the package directory under /work, in the external package `NAME_test`, with "
    "every helper prefixed so it cannot clash with other test files, written for the language version the "
    "go directive of go.mod names (below go 1.18: interface{} rather than any, and no generics), run with the "
    "toolchain flags the task "
    "names as `cd /work && go test ./PKG/ -run TestAcceptance -count=1 -v`;\n"
    "- Rust: a file under /work/tests run with `cd /work && cargo test --offline --test NAME`;\n"
    "- C++ or C: one source file compiled with g++ (gcc for C) against the project's sources, then run; for "
    "these and any other language, print exactly one TAP line per case, `ok N - name` or `not ok N - name`, and "
    "exit non-zero when any case fails.\n"
    "If no such single command is possible for this task, reply with the single word NONE.\n\n"
    "Reply in exactly this form and nothing else:\n"
    "RUN: <the one shell command>\n"
    "PATH: <absolute path of the file to create, under /tmp or /work>\n"
    "```<language>\n<the file>\n```"
)


@dataclass(frozen=True)
class Suite:
    run: str
    path: str
    content: str


def doc_paths(instruction: str) -> list[str]:
    """The documentation files the task text names, in order, without duplicates or escapes."""
    found: list[str] = []
    candidates: list[str] = []
    for clause in DOC_CLAUSE_RE.findall(instruction):
        candidates.extend(DOC_PATH_RE.findall(clause))
    candidates.extend(DOC_REFERENCE_RE.findall(instruction))  # a reference without the parenthesis
    for path in candidates:
        if path not in found and not path.startswith("/") and ".." not in path.split("/"):
            found.append(path)
    return found


def read_docs(instruction: str, root: Path) -> dict[str, str]:
    """The named documentation files that exist in the workspace, up to a total size."""
    docs: dict[str, str] = {}
    total = 0
    base = root.resolve()
    for relative in doc_paths(instruction):
        target = root / relative
        try:
            if target.is_symlink() or not target.is_file() or not target.resolve().is_relative_to(base):
                continue
            text = target.read_text("utf-8")
        except (OSError, UnicodeDecodeError, ValueError):
            continue
        total += len(text)
        if total > MAX_DOC_CHARS:
            break
        docs[relative] = text
    return docs


def author_messages(instruction: str, docs: dict[str, str], sandbox_tools: str) -> list[dict[str, str]]:
    """The author's conversation: the task text and its documentation, and nothing of the answer."""
    parts = [f"The task, as the miner received it:\n\n{instruction}\n"]
    for path, text in docs.items():
        parts.append(f"The documentation at {path}:\n\n{text}\n")
    parts.append(
        "Write the acceptance suite now, in the form stated. Name each test after the rules it checks, and "
        "put the rule's words in a comment where an expected value is not obvious."
    )
    return [
        {"role": "system", "content": AUTHOR_SYSTEM_PROMPT + "\n\nThe sandbox:\n" + sandbox_tools},
        {"role": "user", "content": "\n".join(parts)},
    ]


_RUN_RE = re.compile(r"^RUN:[ \t]*(.+?)[ \t]*$", re.MULTILINE)
_PATH_RE = re.compile(r"^PATH:[ \t]*(\S+)[ \t]*$", re.MULTILINE)
# The author's reasoning effort and output budget. The file is a few thousand tokens; at effort "high" the
# reasoning alone ran to 25,000 tokens on one task, so the shared 32,000 cap would cut the file short.
AUTHOR_REASONING_EFFORT = "medium"
AUTHOR_MAX_TOKENS = 48_000


def _fenced_file(text: str) -> str | None:
    """The file between the first opening fence and the last closing one.

    A test file may itself contain fences inside strings (a Markdown suite does), so the block ends at the
    reply's last fence line, not the first one met.
    """
    lines = text.splitlines()
    opening = next((index for index, line in enumerate(lines) if line.startswith("```")), None)
    if opening is None:
        return None
    closing = next((index for index in range(len(lines) - 1, opening, -1) if lines[index].strip() == "```"), None)
    if closing is None:
        return None
    return "\n".join(lines[opening + 1 : closing])


def stopped_for_length(response_body: bytes) -> bool:
    """Whether the provider cut the reply at the token limit, which leaves a file that only looks complete."""
    try:
        choices = json.loads(response_body).get("choices") or []
        return any(str(choice.get("finish_reason", "")).lower() == "length" for choice in choices if isinstance(choice, dict))
    except (ValueError, AttributeError):
        return False


def _safe_path(path: str) -> bool:
    if not (path.startswith("/tmp/") or path.startswith("/work/")):
        return False
    if any(part in ("", ".", "..") for part in path.split("/")[1:]):
        return False
    return re.fullmatch(r"[A-Za-z0-9_./+-]+", path) is not None


def parse_suite(output: str) -> Suite | None:
    """The suite the author replied with, or None when it declined or the reply is unusable."""
    text = output.strip()
    if not text or text.splitlines()[0].strip() == "NONE":
        return None
    run, path, content = _RUN_RE.search(text), _PATH_RE.search(text), _fenced_file(text)
    if run is None or path is None or content is None:
        return None
    command, target = run.group(1), path.group(1)
    if not _safe_path(target) or not content.strip() or len(content) > MAX_SUITE_CHARS:
        return None
    if SUITE_DELIMITER in content or SUITE_DELIMITER in command or "\n" in command:
        return None
    return Suite(command, target, content.rstrip("\n") + "\n")


_GO_DIRECTIVE = re.compile(r"^go\s+(\d+)\.(\d+)", re.MULTILINE)
# `any` used as a type: not a field (x.any), not assigned or called as a name of its own.
_ANY = re.compile(r"(?<![\w.])any\b(?!\s*(?::=|=(?!=)|\())")


def go_language_version(root: Path) -> tuple[int, int] | None:
    """The language version the module's go.mod declares, such as (1, 17); None without one."""
    try:
        text = (root / "go.mod").read_text("utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    match = _GO_DIRECTIVE.search(text)
    return (int(match.group(1)), int(match.group(2))) if match else None


_GO_TEST_RE = re.compile(r"\bgo test\b")


def fit_go_suite(suite: Suite, version: tuple[int, int] | None) -> Suite:
    """The suite fitted to the module: `go test -vet=off`, and `any` spelled interface{} before go 1.18.

    On 2026-11-11 two Go suites written with `any` failed to build under `go 1.17` and cost a round of
    reconciliation each; strings and comments are left as written. go test also runs vet on the package, and a
    printf warning in a file nobody touched failed a suite's build (a82b2ffec36e): the hidden suites run where
    that cannot happen, and the suite now runs until it passes, so vet is left to the model's own test runs.
    """
    if not suite.path.endswith(".go"):
        return suite
    run = suite.run
    if "-vet" not in run:
        run = _GO_TEST_RE.sub("go test -vet=off", run, count=1)
    content = suite.content
    if version is not None and version < (1, 18):
        content = _outside_literals(content, lambda code: _ANY.sub("interface{}", code))
    return suite if (run, content) == (suite.run, suite.content) else Suite(run, suite.path, content)


def _outside_literals(text: str, change) -> str:
    """Apply `change` to Go code only, leaving comments, strings, runes and raw strings as they are."""
    out: list[str] = []
    start = index = 0
    length = len(text)
    while index < length:
        if text.startswith("//", index):
            end = text.find("\n", index)
            end = length if end < 0 else end
        elif text.startswith("/*", index):
            end = text.find("*/", index + 2)
            end = length if end < 0 else end + 2
        elif text[index] == "`":
            end = text.find("`", index + 1)
            end = length if end < 0 else end + 1
        elif text[index] in "\"'":
            quote, end = text[index], index + 1
            while end < length and text[end] != quote and text[end] != "\n":
                end += 2 if text[end] == "\\" else 1
            end = min(end + 1, length)
        else:
            index += 1
            continue
        out.append(change(text[start:index]))
        out.append(text[index:end])
        start = index = end
    out.append(change(text[start:]))
    return "".join(out)


def placement_command(suite: Suite) -> str:
    """One shell command that writes the suite file verbatim where the author asked for it."""
    directory = shlex.quote(str(Path(suite.path).parent))
    return (
        f"mkdir -p -- {directory} && cat > {shlex.quote(suite.path)} <<'{SUITE_DELIMITER}'\n"
        f"{suite.content}{SUITE_DELIMITER}\n"
    )


def run_summary(output: str) -> str:
    """How many cases failed, in the runner's own terms, when the output says."""
    match = re.search(r"^FAILED \((.+)\)$", output, re.MULTILINE)  # unittest
    if match:
        return match.group(1)
    fails = re.findall(r"^# fail (\d+)", output, re.MULTILINE)  # node:test
    if fails:
        return f"{fails[-1]} failed"
    match = re.search(r"test result: FAILED\. (\d+) passed; (\d+) failed", output)  # cargo
    if match:
        return f"{match.group(2)} failed"
    go = len(re.findall(r"^\s*--- FAIL", output, re.MULTILINE))
    if go:
        return f"{go} failed"
    not_ok = len(re.findall(r"^not ok ", output, re.MULTILINE))  # TAP
    if not_ok:
        return f"{not_ok} failed"
    return ""


_UNITTEST_FAIL_RE = re.compile(r"^(?:FAIL|ERROR): (\S+) \(([^)\s]*)\)", re.MULTILINE)
# unittest -v also prints each case on one line: "test_x (module.Class.test_x) ... FAIL".
_UNITTEST_LINE_RE = re.compile(r"^(\S+) \(([^)\s]*)\)[^\n]*? \.\.\. (?:FAIL|ERROR)[ \t]*$", re.MULTILINE)
_GO_FAIL_RE = re.compile(r"^\s*--- FAIL: (\S+)", re.MULTILINE)
_CARGO_FAIL_RE = re.compile(r"^test (\S+) \.\.\. FAILED\s*$", re.MULTILINE)
_TAP_RE = re.compile(r"^([ \t]*)(not ok|ok) \d+(?: - (.*?))?(?:[ \t]+#[ \t]*(\w+)[^\n]*)?[ \t]*$", re.MULTILINE)


def _tap_failures(output: str) -> list[str]:
    """The failing TAP cases without the parents that failed only because a subtest did.

    node:test prints a subtest's result before its parent's, indented deeper.
    """
    entries = [
        (
            len(match.group(1).expandtabs()),
            # a TODO or SKIP directive marks a result that does not count
            match.group(2) == "not ok" and (match.group(4) or "").upper() not in ("TODO", "SKIP"),
            (match.group(3) or "").strip(),
        )
        for match in _TAP_RE.finditer(output)
    ]
    failing = []
    for index, (indent, failed, name) in enumerate(entries):
        if not failed:
            continue
        before = index - 1
        child_failed = False
        while before >= 0 and entries[before][0] > indent:
            child_failed = child_failed or entries[before][1]
            before -= 1
        if not child_failed:
            failing.append(name)
    return failing


def _failure_section(output: str) -> tuple[str, int | None]:
    """The complete failure list the run printed after its output, and how many lines it had; else the output."""
    match = None
    for match in _FAIL_LIST_RE.finditer(output):
        pass
    if match is None:
        return output, None
    return output[match.end():], int(match.group(1))


def unlisted_failures(output: str) -> int:
    """Failure lines the run counted beyond those it printed in its list (FAIL_LIST_LINES at most)."""
    _section, total = _failure_section(output)
    return max(0, (total or 0) - FAIL_LIST_LINES)


def failing_cases(output: str, suite_path: str = "") -> list[str] | None:
    """The names of the failing cases as the runner printed them; None when it failed without naming one.

    A build or import error, a crash or a time-out names no case, and no quotation can settle it: the suite
    has to run. A unittest module that did not import, a failed unittest fixture (setUpClass and the like) and a
    node:test file that failed as a whole count so.
    """
    output, _total = _failure_section(output)
    names: list[str] = []
    for name, ident in _UNITTEST_FAIL_RE.findall(output) + _UNITTEST_LINE_RE.findall(output):
        if "_FailedTest" in ident or name in _UNITTEST_FIXTURES:
            return None
        names.append(name)
    if not names:
        go = _GO_FAIL_RE.findall(output)
        names = [name for name in go if not any(other.startswith(name + "/") for other in go)]
    if not names:
        names = _CARGO_FAIL_RE.findall(output)
    if not names:
        names = _tap_failures(output)
        base = suite_path.rsplit("/", 1)[-1]
        if base and any(name == suite_path or name.endswith("/" + base) or name == base for name in names):
            return None
    names = [name for name in dict.fromkeys(names) if name]
    return names or None


_REFUTED_RE = re.compile(
    r"^[ \t>*_`+-]*(?:\d+[.)][ \t]*)?[*_`]*REFUTED[*_`]*:?[*_`]*[ \t]*(.+?)[ \t]*\|[ \t]*(.+?)[ \t`]*$", re.MULTILINE
)


def refutations(text: str) -> list[tuple[str, str]]:
    """The REFUTED lines of a reply: each case with the passage quoted for it."""
    found = []
    for case, quote in _REFUTED_RE.findall(text):
        case = case.strip().strip("`'\"*").strip()
        if case and quote.strip():
            found.append((case, quote.strip()))
    return found


_QUOTE_TABLE = str.maketrans({
    "“": '"', "”": '"', "‘": "'", "’": "'", "`": "", "*": "", "|": " ", " ": " ",
})
# List markers and block quotes at the start of a line, which a quotation spanning lines leaves out.
_LINE_MARK_RE = re.compile(r"^[ \t]*(?:[-*+]|\d+[.)]|>)[ \t]+", re.MULTILINE)
_ELLIPSIS_RE = re.compile(r"\s*(?:\[\.\.\.\]|\.\.\.)\s*")


def normalise(text: str) -> str:
    """Text as a quotation is compared: Markdown marks, quote styles, line breaks and case set aside."""
    text = unicodedata.normalize("NFKC", text)
    text = _LINE_MARK_RE.sub("", text).translate(_QUOTE_TABLE)
    return re.sub(r"\s+", " ", text).strip().lower()


def quote_found(quote: str, sources: Iterable[str]) -> bool:
    """Whether a quotation is in one of the (normalised) documents word for word; ... may join its parts."""
    text = normalise(quote).strip(" \"'")
    parts = [part.strip(" \"'.,;:") for part in _ELLIPSIS_RE.split(text)]
    parts = [part for part in parts if part]
    if not parts or sum(len(part) for part in parts) < MIN_QUOTE_CHARS:
        return False
    if any(len(part) < FRAGMENT_CHARS for part in parts):
        return False
    return any(all(part in source for part in parts) for source in sources)


def _case_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().strip("`'\"").strip()).lower()


_UNITTEST_ID_RE = re.compile(r"[\w.]+ \([\w.]+\)|\w+(?:\.\w+)+")


def covered(case: str, refuted: Iterable[str]) -> bool:
    """Whether a REFUTED line names this case: as printed, or by a unittest id ("test_x (mod.Class.test_x)",
    "Class.test_x") whose parts include it. A Go parent covers its subtests only through Settling.take_reply, which
    expands it into the subtests failing when it was written; a word inside another case's name covers nothing."""
    key = _case_key(case)
    for name in refuted:
        other = _case_key(name)
        if not other:
            continue
        if other == key:
            return True
        if re.fullmatch(r"\w+", key) and _UNITTEST_ID_RE.fullmatch(other) and key in re.split(r"[\s().]+", other):
            return True
    return False


def _shorten(text: str, limit: int = 300) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


@dataclass
class Settling:
    """The independent suite's runs in one solve, and the cases the model refuted from the documentation."""

    suite: Suite
    # The task text and the documentation, normalised, which a refutation's quotation must come from.
    sources: tuple[str, ...]
    refuted: dict[str, str] = field(default_factory=dict)
    # Refutations since the last message whose quotation is not in the documentation word for word.
    rejected: list[tuple[str, str]] = field(default_factory=list)
    rejected_total: int = 0
    runs: int = 0
    # The failing cases no refutation covers after the latest run; None when it failed without naming any.
    open: list[str] | None = None
    # After the latest run: passed, settled (every failing case refuted), open, or unreadable.
    state: str = ""
    # The output of the latest run, its end, and its exit code.
    output: str = ""
    exit_code: object = None
    # Set once no more messages about the suite will be sent in this solve.
    finished: bool = False
    # The sha256 of the suite file at the latest run: runs compare only when the suite file is the same.
    suite_hash: str = ""
    last_unreadable: str = ""
    stuck: bool = False
    # Harness suite commands attempted, recorded or not: each gets its own trajectory call id (a failed run
    # used to leave its number to the next one, and a duplicate id loses the whole answer).
    calls: int = 0
    # The failing cases of the latest run before refutations, for expanding a Go parent a REFUTED line names.
    failing: list[str] = field(default_factory=list)
    timed_out: bool = False
    # The workbench version of the latest recorded run, so that a score is stored only for the version it ran on.
    run_version: int = -1

    @classmethod
    def start(cls, suite: Suite, instruction: str, docs: dict[str, str]) -> Settling:
        return cls(suite, tuple(normalise(text) for text in (instruction, *docs.values())))

    @property
    def settled(self) -> bool:
        return self.state in ("passed", "settled")

    def score(self) -> tuple[int, int]:
        """How good the latest run was, comparable across versions: every case passing, every failing case
        refuted, fewer open cases, and last a run that named no case."""
        if self.state == "passed":
            return (3, 0)
        if self.state == "settled":
            return (2, 0)
        if self.state == "open":
            return (1, -len(self.open or ()))
        return (0, 0)

    def describe(self) -> str:
        """The latest run's result in a few words, for a message to the model."""
        if self.state == "passed":
            return "every case passed"
        if self.state == "settled":
            return "every failing case was refuted from the documentation"
        if self.state == "open":
            shown = ", ".join((self.open or [])[:8])
            more = len(self.open or ()) - 8
            return f"{len(self.open or ())} case(s) failed ({shown}{f' and {more} more' if more > 0 else ''})"
        if self.timed_out:
            return "it timed out before naming its cases"
        return "it failed without naming a case in a form the harness reads (a build or import error, a crash, no case run)"

    def take_reply(self, text: str) -> None:
        """Keep the REFUTED lines of one reply whose quotation is in the documentation; note the others."""
        for case, quote in refutations(text):
            if quote_found(quote, self.sources):
                # A Go parent stands for the subtests failing when it was written, never for later ones.
                key = _case_key(case)
                leaves = [name for name in self.failing if _case_key(name).startswith(key + "/")]
                for name in leaves or [case]:
                    self.refuted[name] = quote
            else:
                self.rejected.append((case, quote))
                self.rejected_total += 1

    def command(self) -> str:
        """The suite's run command: the suite file's sha256 first (runs compare only on the same file), then the
        run's whole output, then its failure lines again under FAIL_LIST_MARK, where the kept tail always has them."""
        return (
            f"sha256sum -- {shlex.quote(self.suite.path)} 2>/dev/null | cut -c1-64; "
            f"( {self.suite.run} ) >/tmp/.acceptance-run 2>&1; s=$?; cat /tmp/.acceptance-run; echo; "
            f"echo '{FAIL_LIST_MARK}' $(grep -cE '{_FAIL_LINE_GREP}' /tmp/.acceptance-run); "
            f"grep -E '{_FAIL_LINE_GREP}' /tmp/.acceptance-run | cut -c1-160 | head -n {FAIL_LIST_LINES}; "
            "rm -f /tmp/.acceptance-run; exit $s"
        )

    def record(self, output: str, exit_code: object, timed_out: bool) -> str:
        """Take one run's result; the state after it."""
        self.runs += 1
        match = _HASH_RE.match(output)
        self.suite_hash = match.group(1) if match else ""
        self.output, self.exit_code, self.timed_out = output[-OUTPUT_LIMIT_CHARS:], exit_code, timed_out
        # A suite file that was not there (no hash) or a run in which no case ran never passes.
        ran_nothing = not self.suite_hash or bool(_NO_CASES_RE.search(output))
        if exit_code == 0 and not timed_out and not ran_nothing:
            self.open, self.state, self.failing = [], "passed", []
            return self.state
        failing = None if timed_out or ran_nothing else failing_cases(output, self.suite.path)
        if failing is None:
            # The same failure that names no case, twice in a row: nothing the model did reached it.
            self.stuck = self.state == "unreadable" and output[-OUTPUT_LIMIT_CHARS:] == self.last_unreadable
            self.last_unreadable = output[-OUTPUT_LIMIT_CHARS:]
            self.open, self.state, self.failing = None, "unreadable", []
        else:
            self.failing = failing
            self.open = [case for case in failing if not covered(case, self.refuted)]
            # Neither a panic nor failures the list could not show can be settled by a quotation.
            if _GO_PANIC_RE.search(output):
                self.open.append(PANIC_CASE)
            unlisted = unlisted_failures(output)
            if unlisted:
                self.open.append(f"(and {unlisted} more failure lines than the list above shows)")
            self.state = "open" if self.open else "settled"
        return self.state


# The edge rules as the solver reads them when it settles the suite's cases: the writer's, except that the
# percent-encoding clause is the one the solver's own instructions hold (prompt audit finding 7, plan.md step 2).
# With the writer's wording here, a red suite on a task that names url.PathEscape or encodeURIComponent would be
# answered with "never +" and "every byte outside the unreserved set" against the system message's "use exactly
# that" (8724e1fd451e), on the very case where the solver decides whether to change its code.
_SOLVER_EDGE_RULES = EDGE_RULES.replace(WRITER_ENCODING_RULE, ENCODING_RULE)
_SETTLE_RULES = (
    "Keep your code where a passage of the documentation supports what it does now, such as an operation's own "
    "description of its steps, and refute the suite's case. Change your code where the documentation requires "
    "the suite's expectation and no passage supports yours; at the edge of a rule the hidden tests take the "
    "stricter reading, except that a decoder accepts insignificant whitespace around JSON tokens (before and "
    "after the whole document included: whitespace is not trailing content), members in any order and escaped "
    "member names unless a rule written for the decoder rejects them, so a suite case that refuses such input is "
    "the suite's error. " + EXAMPLE_RULE + " Where the suite and your code disagree on a value's type, the signature or attribute table "
    "decides it, and a value is never made two types at once to satisfy both sides. " + _SOLVER_EDGE_RULES + " "
    + ERROR_SCOPE + " A case that fails to build "
    "or import because your code lacks a name the documentation gives, or gives it another signature, is your "
    "code's to fix."
)
_REFUTE_FORM = (
    "For each case you refute, quote the passage your code follows, on a line of its own in your reply, in "
    "exactly this form:\n"
    "REFUTED: <the case's name as the runner printed it> | \"<the passage, copied word for word from the "
    "documentation>\"\n"
    "A case is settled when it passes, or when a REFUTED line names it and quotes a passage that is in the "
    "documentation word for word; a paraphrase settles nothing. If the suite itself is broken (an import or "
    "syntax error of its own, a wrong path or command, a name or signature the documentation does not give), or "
    "a case contradicts one of the rules above (a decoder refusing insignificant whitespace, for one), repair the "
    "suite file with a command rather than bending your code to it, and never edit a case only to make it pass."
)


def _settle_time(seconds: int, total: int | None) -> str:
    """Both time figures: the settling window alone was read as the time left for the whole task (966efbb1635f).

    `total` is the time left to the wrap-up; the window is never said to be longer than it.
    """
    if total is None:
        return (
            f"About {max(seconds, 0)} seconds are set aside for settling these cases; the review and the final "
            "checks have their own time after that."
        )
    total = max(total, 0)
    return (
        f"About {min(max(seconds, 0), total)} seconds are set aside for settling these cases, out of about {total} "
        "seconds left for the whole task; the review and the final checks have their own time after that."
    )


def reconcile_prompt(
    suite: Suite,
    output: str,
    exit_code: object,
    seconds: int | None = None,
    again: bool = True,
    total: int | None = None,
) -> str:
    """What the answer's author is told when the independent suite first disagrees with its work.

    `again` says the suite runs again on the reply (MINER_ACCEPTANCE_RUNS above 1); otherwise the model reruns it.
    `seconds` is the settling window and `total` the time left to the wrap-up; without `seconds` neither is named.
    """
    time_left = f" {_settle_time(seconds, total)}" if seconds is not None else ""
    if again:
        ending = (
            f"The suite is at {suite.path}. When you reply it runs again, and the cases still failing that no "
            f"REFUTED line settles come back to you.{time_left} Run your own tests again after your fixes, then "
            "reply with a one-line summary and a REFUTED line for each case you keep your code against."
        )
    else:
        ending = (
            f"The suite is at {suite.path}; rerun it with the command above after your fixes, then run your own "
            "tests again and reply with a one-line summary and a REFUTED line for each case you keep your code against."
        )
    return (
        "Before your work is reviewed, an acceptance suite written independently from the documentation alone, "
        "by a second model call that never saw your code, ran against it:\n\n"
        f"  {suite.run}\n\n"
        f"Exit code {exit_code}. Its output (the end):\n{output}\n\n"
        "The suite is often wrong, so neither side is presumed right: for every failing case, decide from the "
        "documentation which side is. " + _SETTLE_RULES + "\n\n" + _REFUTE_FORM + "\n\n" + ending
    )


def settle_prompt(settling: Settling, seconds: int, total: int | None = None) -> str:
    """What the model is told when the suite, run again, still fails a case that no quotation settles.

    `seconds` is what is left of the settling window and `total` the time left to the wrap-up.
    """
    parts = [
        "The acceptance suite ran again on your work:\n\n"
        f"  {settling.suite.run}\n\n"
        f"Exit code {settling.exit_code}. Its output (the end):\n{settling.output}"
    ]
    if settling.open is None:
        if settling.timed_out:
            reason = "It timed out before naming its cases: a case that never returns (a loop, a wait, a deadlock) or work far too slow."
        elif not settling.suite_hash:
            reason = f"Its file {settling.suite.path} was not there, so no case ran; it is put back before the next run."
        else:
            reason = (
                "It failed without naming its cases in a form the harness reads: a build or import error, a failed "
                "fixture, a crash, no case run, or a runner whose output names none."
            )
        parts.append(
            reason + " Fix your code where it lacks a name the documentation gives or gives it another signature; "
            "repair the suite file with a command where the fault is the suite's own."
        )
    else:
        listed = settling.open[:MAX_LISTED]
        more = len(settling.open) - len(listed)
        parts.append(
            "These cases still fail, and no REFUTED line settles them:\n"
            + "\n".join(f"- {case}" for case in listed)
            + (f"\n- and {more} more in the output above" if more > 0 else "")
        )
    rejected = [(case, quote) for case, quote in settling.rejected if case not in settling.refuted]
    if rejected:
        parts.append(
            "These REFUTED lines settle nothing, because their quotation is not in the documentation word for word "
            "(copy the passage exactly; ... may join the parts you quote):\n"
            + "\n".join(f"- {case}: {_shorten(quote)}" for case, quote in rejected[:MAX_LISTED])
        )
    if settling.refuted:
        parts.append(f"The {len(settling.refuted)} case(s) already refuted with a passage of the documentation stay settled.")
    parts.append(
        "Settle each open case as before: change your code where the documentation requires what the case "
        "expects, or keep your code and write, on a line of its own, REFUTED: <case> | \"<the passage, word for "
        f"word>\". {_settle_time(seconds, total)} Run your own tests again after any fix, then reply with a "
        "one-line summary."
    )
    return "\n\n".join(parts)



def final_check_prompt(now: str, output: str, before: str | None, itself: str | None) -> str:
    """What the model is told when the suite, run again before the answer goes out, does worse than before."""
    lines = []
    if before is not None:
        lines.append(f"An earlier version of your work had this result: {before}.")
    if itself is not None:
        lines.append(f"In your working copy the same code had this result: {itself}.")
    return (
        "Before the answer goes out, the acceptance suite ran again, the last time on a clean copy of the repository "
        f"with only your answer applied, the way the grader starts: {now}.\n" + "\n".join(lines) + "\n\n"
        f"Its output (the end):\n{output}\n\n"
        "Either a recent change broke something, or the answer relies on something that exists only in your "
        "working copy (a file a command created or changed, generated output, state an earlier run left behind), "
        "or a deliberate recent change is right and the suite's case is wrong. The working copy is now that clean "
        "copy. Fix what broke, undo a change that was not needed (revert_file puts a file back to the repository's "
        "version), and write every file the answer needs with write_file or edit_file; where the documentation "
        "supports a deliberate change, keep it and settle the failing case as before.\n\n" + _REFUTE_FORM + "\n\n"
        "Run your tests again, then reply with a one-line summary."
    )
