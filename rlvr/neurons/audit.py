"""A stronger reader goes through the finished diff against the documentation, before the review.

Measured on 2026-11-11 on the 17 failed repository tasks kept with their hidden suites: given the task text,
the documentation and the submitted diff, Opus 5.5 at medium effort named the defect behind at least one
failing hidden test in 10 of the 17, with about four findings each, for about $0.22 a review; Sonnet 5.5 with
the same prompt did so in 5 of 17, with about nine findings each, nearly all about something else. At high
effort Opus did worse, because four of its replies spent the token limit on reasoning and listed nothing.

So the implementer stays on the cheaper model for the long tool loop, where most of the money goes, and the
stronger model reads once, at the end, where reading decides the grade.

The first live version broke a correct answer (fdc7aafc1920, JSON Patch, 07:13): the code raised "move" as the
operation's own description of its steps says, the auditor cited the general precedence list and asked for
"target", the implementer deferred to "a senior engineer", and the hidden test wanted "move"; the pre-audit
answer, replayed, passes every hidden test. So the auditor now reports a point only when no passage of the
documentation supports what the code does (an operation's own description of its steps decides for it over a
general list), and the implementer is told the findings may be wrong and keeps its code where a passage
supports it. On the 19 answers that had passed their hidden tests, the first prompt made 27 findings on 14 of
them, this one 7 on 6; on the broken JSON Patch answer it made none in two runs.

The wording follows the lesson of adjudicate.py: never ask the implementer to catalogue its own earlier
statements, and never describe its work as ruled on by another model call.
"""

from __future__ import annotations

import re

# Skipped when less than this is left before the wrap-up: the review call takes about a minute, and the
# findings need time to be checked and fixed.
MIN_TIME_S = 240.0
MAX_TOKENS = 16_000
REASONING_EFFORT = "medium"
MAX_FINDINGS = 12
FINDING_CHARS = 900
MAX_DIFF_CHARS = 120_000

AUDITOR_SYSTEM_PROMPT = (
    "You review a change to a codebase against the documentation of the feature it implements. Hidden automated "
    "tests, written from that documentation by its author, will grade the change: every documented rule, error "
    "code, precedence, limit, example and public import has its own test, and the tests compare results "
    "exactly. Find the places where the change will fail such a test: a documented rule it does not implement "
    "or implements differently, a wrong error code or exception type, a wrong check order, an off-by-one at a "
    "limit, a documented import or name that is not reachable, a state change on a failure that should leave "
    "state untouched, or a crash. Read the documentation line by line against the code.\n\n"
    "Report a point only when the documentation requires the expected behaviour and no passage of it supports "
    "what the code does now. Before you report a point, look for a passage the code follows, such as an "
    "operation's own description of its steps and the errors they raise; if there is one, the point is open "
    "and you leave it out, because an operation's own description decides for that operation over a general "
    "list. Never report style.\n\n"
    "Reply with at most 12 findings, the most certain first, numbered, each in this form:\n"
    "N. RULE: <the documentation's words, quoted> | CALL: <a concrete call or input> | EXPECTED: <what the "
    "documentation requires> | ACTUAL: <what this code does instead, naming the function>\n"
    "Write NONE if you find nothing."
)

_ITEM_RE = re.compile(r"^\s*(\d{1,3})[.)]\s+(\S.*)$")
# Only ** bold: a double underscore is part of names such as __init__.py and __all__.
_EMPHASIS_RE = re.compile(r"\*\*")


def auditor_messages(instruction: str, docs: dict[str, str], diff: str) -> list[dict[str, str]]:
    """The reviewer's conversation: the task text, its documentation and the diff, nothing else."""
    parts = [f"The task:\n\n{instruction}\n"]
    for path, text in docs.items():
        parts.append(f"The documentation at {path}:\n\n{text}\n")
    if len(diff) > MAX_DIFF_CHARS:
        diff = diff[:MAX_DIFF_CHARS] + "\n[... the rest of the diff is omitted ...]\n"
    parts.append(f"The change, as a unified diff against the repository:\n\n```diff\n{diff}\n```\n")
    parts.append("List the findings now, in the form stated.")
    return [{"role": "system", "content": AUDITOR_SYSTEM_PROMPT}, {"role": "user", "content": "\n".join(parts)}]


def parse_findings(output: str) -> list[str]:
    """The numbered findings, continuation lines joined, at most MAX_FINDINGS; [] for NONE or nothing."""
    findings: list[str] = []
    current: list[str] | None = None
    for raw in (output or "").splitlines():
        line = _EMPHASIS_RE.sub("", raw).rstrip()
        match = _ITEM_RE.match(line)
        if match:
            if current:
                findings.append(" ".join(current))
            current = [match.group(2).strip()]
        elif current is not None and line.strip() and not line.lstrip().startswith("```"):
            current.append(line.strip())
    if current:
        findings.append(" ".join(current))
    cleaned = []
    for finding in findings:
        if finding.strip().upper().rstrip(".") == "NONE":
            continue
        cleaned.append(finding if len(finding) <= FINDING_CHARS else finding[:FINDING_CHARS] + " ...")
    return cleaned[:MAX_FINDINGS]


def findings_prompt(findings: list[str]) -> str:
    listed = "\n".join(f"{number}. {finding}" for number, finding in enumerate(findings, 1))
    return (
        "Before the review, a second reader went through the documentation and your change, without running "
        f"anything, and noted these possible violations:\n\n{listed}\n\n"
        "They may be wrong. For each one, first look for a passage of the documentation that supports what your "
        "code does now, such as the operation's own description of its steps; if there is one, keep your code. "
        "Otherwise run its CALL against your code in a throwaway test, and fix it where the result differs from "
        "EXPECTED. A fix must keep every other documented requirement: follow a changed value through everything "
        "that records or reproduces it (a checkpoint, a restore, a replay, an export) and test that round trip "
        "too. Then run your tests again and reply with a one-line summary."
    )
