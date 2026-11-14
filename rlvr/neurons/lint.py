"""Checks of the finished diff for mistakes with an exact signature, each one a hidden-test failure seen live.

A reading rule in a prompt makes a mistake less likely; a check of the diff makes it impossible to submit
unnoticed. Each check here encodes one failure the hidden tests caught, needs no model call and no sandbox,
and runs on every finished version of the work, like the documentation's import lines. A finding goes back
to the model with the evidence and the fix; nothing is changed behind its back.

Go error values in pointer form. The larksuite SDK (11 of the first 88 live tasks) declares
`type CodeError struct` with value-receiver methods, so the value and the pointer both satisfy `error`, but
its own code returns `&CodeError{...}` and its hidden tests match `err.(*larkcore.CodeError)` or
`errors.As(err, &target)` with a `*CodeError` target, which a value never matches. On 2026-11-11 two answers
returned or stored the response's embedded value (`return resp.CodeError`, `e.Cause = resp.CodeError`) and
failed; earlier answers that returned a pointer to a copy passed. The check finds every error type a project
declares with a value-receiver `Error` method and builds with `&T{...}` in its own code, and reports an added
line that returns or stores such a value. It stays quiet when the documentation asks for the value.

Python error attributes set at some raise sites only. An error class whose documentation defines an attribute
for the class as a whole (a position, a line) is raised with it at every code; 5fbaf671fe6a (2026-11-12) set
`position=self._position` at four CassetteError sites and left it out at six, and lost 13 of 29 hidden tests
("None != 0"). The check groups the added raise sites of each error class across every changed non-test .py
file, and reports the sites that pass no keyword at all when others pass one taken from the raising object's
own state. It stays quiet for a keyword the documentation gives as None, so a code documented without it
(5723230bca9e: "'input' with line=None") is left alone, and the prompt asks for the change only where the
documentation defines the attribute for the class as a whole.

argparse private parser state. 1b907aee92e2 (2026-11-13) set `parser._negative_number_matcher` so that `--root -y=b`
would reach its own check; `--resolve -1` then became a usage error instead of the documented invalid-resolve.
8724e1fd451e's suite also passes a negative number to an integer option and expects the program's own error. The
check reports an added line that touches argparse's private attributes, in a file that uses argparse.

Four diff questions, in every language. Some constructs are right in one documentation and wrong in another,
so they are asked about rather than reported: a form encoder used to re-encode (966efbb1635f), a single-base
subclass of a builtin (a9a5461324c6), a JavaScript destructuring default on an options object (a30b799785d4)
and math.fsum (0af90493237d). On the record each matched exactly one failed answer and none of 104 passed
answers or 28 right diffs. Each question names the line and the one input that decides it, quotes the code
and nothing said about it, and asks for a change only where the documentation's sentence requires one. The
encoder is asked about only where the documentation speaks of escapes and names neither that function nor a
form encoding; math.fsum only where it speaks of finite values or overflow. Retire a pattern the first time
it fires on a passed answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# A finding goes back at most this often per solve; a model that keeps the value has a reason to.
MAX_PROMPTS = 2
# The project scan reads at most this many Go files, each at most this large.
MAX_SCANNED_FILES = 6_000
MAX_FILE_BYTES = 512 * 1024
SKIPPED_DIRS = {"vendor", "testdata", "node_modules", ".git", "third_party"}

_VALUE_ERROR_METHOD = re.compile(r"^func \(\s*\w+\s+([A-Z]\w*)\s*\)\s*Error\(\)\s*string", re.MULTILINE)
_DIFF_FILE = re.compile(r"^\+\+\+ b/(.+)$")


@dataclass(frozen=True)
class Finding:
    path: str
    line: str
    type_name: str


def pointer_error_types(root: Path) -> dict[str, str]:
    """Error types the project returns as pointers: {type name: "path: line" where its own code builds &T{...}}.

    A type qualifies when its Error method has a value receiver (so a value also satisfies `error` and the
    compiler cannot catch the slip) and the project's own non-test code constructs it with `&`.
    """
    declared: set[str] = set()
    files: list[Path] = []
    for path in _go_files(root):
        files.append(path)
        try:
            text = path.read_text("utf-8", errors="replace")
        except OSError:
            continue
        declared.update(_VALUE_ERROR_METHOD.findall(text))
    if not declared:
        return {}
    construct = re.compile(r"&(?:\w+\.)?(" + "|".join(sorted(map(re.escape, declared))) + r")\{")
    found: dict[str, str] = {}
    for path in files:
        if path.name.endswith("_test.go"):
            continue
        try:
            lines = path.read_text("utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            match = construct.search(line)
            if match and match.group(1) not in found:
                found[match.group(1)] = f"{path.relative_to(root)}: {line.strip()[:160]}"
        if len(found) == len(declared):
            break
    return found


def _go_files(root: Path):
    count = 0
    for path in sorted(root.rglob("*.go")):
        relative = path.relative_to(root).parts
        if any(part in SKIPPED_DIRS or part.startswith(".") for part in relative[:-1]):
            continue
        try:
            if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        count += 1
        if count > MAX_SCANNED_FILES:
            return
        yield path


def documentation_asks_for_value(docs: dict[str, str], type_name: str) -> bool:
    """True when a sentence asks for the error as a value rather than a pointer."""
    near = re.compile(
        rf"{re.escape(type_name)}\W{{0,60}}(?:by value|as a value|value,? not a pointer|not a pointer)"
        rf"|(?:by value|as a value|not as a pointer)\W{{0,60}}{re.escape(type_name)}",
        re.IGNORECASE,
    )
    return any(near.search(text) for text in docs.values())


def value_uses(diff: str, types: dict[str, str]) -> list[Finding]:
    """Added Go lines that return or store a value of one of `types` where an error is expected."""
    if not types:
        return []
    names = "|".join(sorted(map(re.escape, types)))
    field = re.compile(rf"[a-z_]\w*(?:\.\w+)*\.({names})")  # resp.CodeError: an embedded field read as a value
    literal = re.compile(rf"(?:\w+\.)?({names})\{{")  # larkcore.CodeError{...} without &
    stored = re.compile(r"^\s*(?:[\w.]+\.)?(?:Cause|Err|err|cause)\s*(?::|=)(?!=)\s*(.+?)\s*,?\s*$")
    assigned = re.compile(r"^\s*(?:var\s+)?([a-z_]\w*)\s*(?::=|=)\s*(.+?)\s*$")
    returned = re.compile(r"^\s*return\b(.*)$")
    # `var ce larkcore.CodeError` and a parameter `ce larkcore.CodeError` hold values too.
    declared = re.compile(rf"\b([a-z_]\w*)\s+(\*?)(?:\w+\.)?({names})\b(?!\s*\{{)")
    signature = re.compile(r"^\s*func\b")
    declaration = re.compile(r"^\s*var\s")
    findings: list[Finding] = []
    path = None
    values: dict[str, str] = {}  # variable -> type, for `ce := resp.CodeError` followed by `return ce`

    def value_of(expression: str) -> str | None:
        expression = expression.strip()
        if not expression or expression.startswith("&"):
            return None
        hit = field.fullmatch(expression) or literal.match(expression)
        return hit.group(1) if hit else values.get(expression)

    for raw in diff.splitlines():
        header = _DIFF_FILE.match(raw)
        if header:
            path = header.group(1)
            values = {}
            continue
        if path is None or not path.endswith(".go") or path.endswith("_test.go"):
            continue
        if not raw.startswith("+") or raw.startswith("+++"):
            continue
        line = raw[1:].split("//", 1)[0].rstrip()
        if not line.strip():
            continue
        if signature.match(line) or (declaration.match(line) and "=" not in line):
            for name, pointer, type_name in declared.findall(line):
                if pointer:
                    values.pop(name, None)  # a pointer of the same name in another function
                else:
                    values[name] = type_name
            if signature.match(line):
                continue
        match = stored.match(line)
        if match and (name := value_of(match.group(1))):
            findings.append(Finding(path, line.strip(), name))
            continue
        match = assigned.match(line)
        if match:
            name = value_of(match.group(2))
            if name:
                values[match.group(1)] = name
            else:
                values.pop(match.group(1), None)
            continue
        match = returned.match(line)
        if match:
            for part in _split_operands(match.group(1)):
                name = value_of(part)
                if name:
                    findings.append(Finding(path, line.strip(), name))
                    break
    return findings


def _split_operands(text: str) -> list[str]:
    """`nil, resp.CodeError` -> ["nil", "resp.CodeError"], splitting on top-level commas only."""
    parts, depth, current = [], 0, []
    for char in text:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    parts.append("".join(current).strip())
    return [part for part in parts if part]


def check(diff: str, types: dict[str, str], docs: dict[str, str]) -> list[Finding]:
    """The findings for this diff, leaving out types the documentation asks to pass by value."""
    allowed = {name: where for name, where in types.items() if not documentation_asks_for_value(docs, name)}
    return value_uses(diff, allowed)


def prompt(findings: list[Finding], types: dict[str, str]) -> str:
    listed = "\n".join(f"- {finding.path}: `{finding.line}`" for finding in findings[:8])
    names = sorted({finding.type_name for finding in findings})
    evidence = "\n".join(f"- {types[name]}" for name in names if name in types)
    shown = ", ".join(f"`{name}`" for name in names)
    return (
        f"A check of your change found an error of the project's type {shown} returned or stored as a value:\n"
        f"{listed}\n\n"
        f"The project's own code returns this type as a pointer, for example:\n{evidence}\n\n"
        "Its methods have value receivers, so the value compiles as an error too, but callers and tests match "
        "it with a type assertion or errors.As against the pointer type, which a value never matches. Return "
        "or store a pointer to a copy instead (ce := resp.CodeError; return &ce), keeping it unwrapped where "
        "the documentation says so, check that every documented way of reading the error still works, then "
        "run your tests again and reply with a one-line summary."
    )


@dataclass(frozen=True)
class Question:
    kind: str
    path: str
    line: str


# Constructs whose result on one input the hidden tests checked against the documentation, each the defect of
# one failed answer and found in none of the 104 passed answers or the 28 right diffs on record (2026-11-13).
# A match is asked about, never reported as a mistake: kind -> (pattern on an added line, the file suffixes it
# applies to, or every non-test file when empty).
QUESTION_PATTERNS: dict[str, tuple[re.Pattern[str], tuple[str, ...]]] = {
    # 966efbb1635f: sb.WriteString(url.QueryEscape(p.k)) re-encoded a query with + for a space, where the right
    # diff wraps the call in strings.ReplaceAll(..., "+", "%20"). Re-encoding only: a read such as
    # new URLSearchParams(...).get(...) (c6c48b966bb1 and its right diff) is not matched.
    "form_encoder": (
        re.compile(
            r"url\.QueryEscape\(|\bquote_plus\(|\burlencode\(|URLSearchParams\([^)]*\)\s*\.toString\("
            r"|String\(\s*new URLSearchParams|`\$\{\s*new URLSearchParams"
        ),
        (),
    ),
    # a9a5461324c6: class _RenderedReport(str), a value built as two types at once.
    "hybrid_type": (re.compile(r"^\s*class\s+\w+\(\s*(str|bytes|int|float|dict|list|tuple)\s*\)\s*:"), (".py",)),
    # a30b799785d4: const { first = '0' } = options(opt), whose default also replaces a key present as undefined.
    "js_destructure_default": (
        re.compile(r"(?:const|let|var)\s*\{[^}]*\b\w+\s*=\s*[^},]+[^}]*\}\s*=\s*\w"),
        (".js", ".mjs", ".cjs", ".ts"),
    ),
    # 0af90493237d: math.fsum(weights) raises OverflowError on finite weights whose sum passes the largest float.
    "fsum": (re.compile(r"math\.fsum\("), ()),
}
# The encoder is asked about only where the documentation speaks of escapes or percent-encoding, and never where
# it names that function or a form encoding itself (8724e1fd451e's encodeURIComponent).
# A written escape such as %20 or %2B counts as speaking of percent-encoding.
_ESCAPE_TALK = re.compile(
    r"\bescap(?:e|es|ed|ing)\b|\bpercent[- ]?(?:en|de)cod|\bpercent[- ]escap|%HH\b|(?-i:%[0-9A-F]{2})", re.IGNORECASE
)
_FORM_ENCODING = re.compile(
    r"encodeURIComponent|x-www-form-urlencoded"
    r"|`?\+`?\s+for\s+(?:a\s+|each\s+)?spaces?\b"
    r"|\bspaces?`?\s+(?:(?:is|are)\s+)?(?:(?:written|encoded|escaped|serialized|sent)\s+as|becomes?|turns?\s+into|as)"
    r"\s+(?:a\s+)?`?\+",
    re.IGNORECASE,
)
_ENCODER_NAME = re.compile(r"QueryEscape|quote_plus|urlencode|URLSearchParams")
# A line that already writes a space as %20 (Go's strings.ReplaceAll(..., "+", "%20") in 966efbb1635f's right diff,
# JavaScript's .replace(/\+/g, "%20"), Python's urlencode(..., quote_via=quote)) is not asked about.
_SPACE_HANDLED = re.compile(r"ReplaceAll|%20|\bquote_via\s*=\s*(?:\w+\.)*quote\b")
# A mention in a clause that rejects it ("uppercase escapes, never + for a space", "don't use url.QueryEscape")
# names nothing. The clause ends at a comma, so "no padding, + for a space" still names the form encoding.
_NEGATION = re.compile(
    r"(?:\b(?:never|not|no|nor|without|instead of|rather than|unlike)\b|n['\u2019]t\b)[^.;:,]{0,40}$", re.IGNORECASE
)
# math.fsum is asked about only where the documentation speaks of finite values or overflow.
_FINITE_TALK = re.compile(r"\b(?:non-?)?finite\b|\boverflow", re.IGNORECASE)
_DESTRUCTURED_KEY = re.compile(r"\{[^}]*?\b(\w+)\s*(?::\s*\w+\s*)?=(?!=)")  # the key, not a renamed binding
# Prose the answer edits (a README's code sample) is not its code.
_PROSE_SUFFIXES = (".md", ".markdown", ".rst", ".txt", ".adoc")
# A longer added line is generated or minified code, and the destructuring pattern's backtracking grows steeply
# with the length of a line (a 2,000-character line takes most of a second), so it is not read.
MAX_QUESTION_LINE = 400


def _source_file(path: str) -> bool:
    """Not a test file, by the rule of _python_source and the other languages' test-file names."""
    name = path.split("/")[-1]
    return (
        _python_source(path)
        and name != "conftest.py"
        and not re.search(r"_test\.\w+$|\.(?:test|spec)\.\w+$", name)
        and "__tests__" not in path.split("/")[:-1]
        and not name.lower().endswith(_PROSE_SUFFIXES)
    )


def _names(pattern: re.Pattern[str], documentation: str) -> bool:
    """True when the documentation mentions the pattern other than in a clause that rejects it."""
    return any(
        not _NEGATION.search(documentation[max(0, match.start() - 60):match.start()])
        for match in pattern.finditer(documentation)
    )


# 121c13873690 (2026-11-14, 95% of the other miners passed it): resolve_argv(plan, step_id, outputs) raised
# BatchError('reference', ..., step_id=None) for an unknown step id, and the hidden test read the id the caller
# gave. An error raised with K=None, or K=<value> if <condition> else None, inside a function that received a
# parameter named K: found in that one failed answer and in none of the 49 passed Python answers or the 28 right
# diffs on record. Unlike the other questions its default is to pass the value: see question_prompt.
DROPPED_ARGUMENT = "dropped_argument"
_DEF_START = re.compile(r"^(\s*)(?:async\s+)?def\s+\w+\s*\(")
_RAISE_START = re.compile(r"^\s*raise\s+[A-Za-z_][\w.]*\(")
_NONE_KEYWORD = re.compile(r"(?<![\w.])([A-Za-z_]\w*)\s*=\s*None\b")
_ELSE_NONE_KEYWORD = re.compile(r"(?<![\w.])([A-Za-z_]\w*)\s*=\s*[^,()=]+\sif\s[^,]+?\selse\s+None\b")
_STRING_LITERAL = re.compile(r"\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'")


def _dropped_keywords(call: str) -> list[str]:
    """The keywords a call passes as None, or as a value that falls back to None; text inside strings is not read."""
    bare = _STRING_LITERAL.sub('""', call)
    return list(dict.fromkeys(_NONE_KEYWORD.findall(bare) + _ELSE_NONE_KEYWORD.findall(bare)))


def _balanced(lines: list[str], start: int, limit: int) -> str:
    """The statement that opens on lines[start], joined until its parentheses close or `limit` more lines."""
    text = lines[start][1:]
    end = start
    while end - start < limit and end + 1 < len(lines) and lines[end + 1][:1] in ("+", " "):
        bare = _STRING_LITERAL.sub('""', text)
        if bare.count("(") <= bare.count(")"):
            break
        end += 1
        text += " " + lines[end][1:].strip()
    return text


def _parameters(signature: str) -> list[str]:
    """The parameter names of a def statement's text."""
    inside = signature[signature.index("(") + 1:]
    depth, piece, pieces = 0, "", []
    for char in inside:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            if depth == 0:
                break
            depth -= 1
        if char == "," and depth == 0:
            pieces.append(piece)
            piece = ""
        else:
            piece += char
    pieces.append(piece)
    names = []
    for item in pieces:
        match = re.match(r"\s*\*{0,2}([A-Za-z_]\w*)", item)
        if match and match.group(1) not in ("self", "cls"):
            names.append(match.group(1))
    return names


def dropped_arguments(diff: str, docs: dict[str, str]) -> list[Question]:
    """Added raises that pass K=None, or K=... else None, inside a function that received a parameter named K."""
    documentation = "\n".join(docs.values())
    lines = diff.splitlines()
    found: list[Question] = []
    path = None
    scopes: list[tuple[int, list[str]]] = []
    for index, raw in enumerate(lines):
        if raw.startswith("+++ "):
            header = _DIFF_FILE.match(raw)
            path = header.group(1) if header and header.group(1).endswith(".py") and _source_file(header.group(1)) else None
            scopes = []
            continue
        if raw.startswith("@@"):
            # The lines between two hunks are not shown. Where the hunk header names the function the hunk opens
            # in, that function's parameters are known; otherwise the enclosing function is not.
            heading = raw.split("@@", 2)[2] if raw.count("@@") >= 2 else ""
            scopes = [(0, _parameters(heading))] if _DEF_START.match(heading.lstrip()) else []
            continue
        if path is None or raw[:1] not in ("+", " ") or len(raw) > MAX_QUESTION_LINE + 1:
            continue
        text = raw[1:]
        if not text.strip():
            continue
        indent = len(text) - len(text.lstrip())
        opened = _DEF_START.match(text)
        if opened:
            scopes = [scope for scope in scopes if scope[0] < indent]
            scopes.append((indent, _parameters(_balanced(lines, index, 12))))
            continue
        if not text.lstrip().startswith(("#", ")", "]", "}", '"', "'")):
            scopes = [scope for scope in scopes if scope[0] < indent]
        if not scopes or not raw.startswith("+") or not _RAISE_START.match(text):
            continue
        call = _balanced(lines, index, 8)
        for keyword in _dropped_keywords(call):
            if keyword not in scopes[-1][1]:
                continue
            if re.search(rf"`?{re.escape(keyword)}`?\s+is\s+`?None`?\s+(?:when|for|if|where|unless|on)\b", documentation):
                continue  # the documentation says when this error carries none
            found.append(Question(DROPPED_ARGUMENT, path, " ".join(call.split())[:160]))
            break
    return found


def questions(diff: str, docs: dict[str, str]) -> list[Question]:
    """Added lines of non-test files that hold a construct of QUESTION_PATTERNS the documentation leaves open."""
    documentation = "\n".join(docs.values())
    escapes = bool(_ESCAPE_TALK.search(documentation)) and not _names(_FORM_ENCODING, documentation)
    finite = bool(_FINITE_TALK.search(documentation))
    found: list[Question] = []
    seen: set[tuple[str, str, str]] = set()
    path = None
    for raw in diff.splitlines():
        if raw.startswith("+++ "):
            header = _DIFF_FILE.match(raw)
            path = header.group(1) if header and _source_file(header.group(1)) else None
            continue
        if path is None or not raw.startswith("+") or len(raw) > MAX_QUESTION_LINE + 1:
            continue
        line = raw[1:]
        for kind, (pattern, suffixes) in QUESTION_PATTERNS.items():
            if suffixes and not path.endswith(suffixes):
                continue
            if not pattern.search(line):
                continue
            if kind == "form_encoder":
                if _SPACE_HANDLED.search(line) or not escapes:
                    continue
                name = _ENCODER_NAME.search(line)  # url.QueryEscape, named with its package or without
                if name and _names(re.compile(rf"\b(?:\w+\.)?{name.group(0)}\b"), documentation):
                    continue
            if kind == "fsum" and not finite:
                continue
            key = (kind, path, line.strip()[:160])
            if key not in seen:
                seen.add(key)
                found.append(Question(*key))
    for question in dropped_arguments(diff, docs):
        key = (question.kind, question.path, question.line)
        if key not in seen:
            seen.add(key)
            found.append(question)
    return found


def question_prompt(found: list[Question]) -> str:
    parts = []
    for kind in dict.fromkeys(question.kind for question in found):
        lines = [question for question in found if question.kind == kind][:3]
        listed = "\n".join(f"- {question.path}: `{question.line}`" for question in lines)
        if kind == "form_encoder":
            parts.append(
                f"A form encoder re-encodes here:\n{listed}\nIt writes a space as + (Go url.QueryEscape, Python "
                "quote_plus and urlencode, and JavaScript URLSearchParams all do). Run a value that holds both a "
                "space and a literal + through this code and compare what comes out with the form the "
                "documentation gives for escaped values."
            )
        elif kind == "hybrid_type":
            bases = dict.fromkeys(
                match.group(1) for question in lines if (match := QUESTION_PATTERNS[kind][0].search(question.line))
            )
            named = " or ".join(f"`{base}`" for base in bases) or "type"
            parts.append(
                f"A class subclasses the builtin {named} here:\n{listed}\nTake the "
                "value this code produces and compare it, with type() and ==, against a plain value of the type "
                "the documentation gives for it (a plain dict, str or list holding the same content), the way a "
                "test of the documented value would."
            )
        elif kind == "js_destructure_default":
            key = _DESTRUCTURED_KEY.search(lines[0].line)
            name = key.group(1) if key else "key"
            parts.append(
                f"A destructuring default reads an options object here:\n{listed}\nIt applies both when the key "
                f"is left out and when it is present with the value undefined. Call this code once with "
                f"`{{{name}: undefined}}` and once with `{name}` left out, and compare each result with what the "
                "documentation says for that input."
            )
        elif kind == "fsum":
            parts.append(
                f"math.fsum sums here:\n{listed}\nIt raises OverflowError when finite values add up past the "
                "largest float, where sum() returns inf. Run it on [1e308, 1e308] and compare the result with "
                "what the documentation says for finite input."
            )
    dropped = [question for question in found if question.kind == DROPPED_ARGUMENT][:3]
    message = ""
    if parts:
        message = (
            "Questions from a check of your change, each about one line and the one input that decides it.\n\n"
            + "\n\n".join(parts)
            + "\n\nThese questions are not findings: change code only where a sentence or example of the "
            "documentation requires a different result for that input, and where none decides it, keep the code as "
            "it is."
        )
    if dropped:
        listed = "\n".join(f"- {question.path}: `{question.line}`" for question in dropped)
        names = dict.fromkeys(keyword for question in dropped for keyword in _dropped_keywords(question.line)[:1])
        named = " and ".join(f"`{name}`" for name in names) or "that value"
        message += ("\n\n" if message else "") + (
            f"{'One more line, with the opposite default. ' if parts else ''}An error is raised here without a value "
            f"the function was given:\n{listed}\nThe function received {named}, and this raise passes None for it. "
            "A caller reads that attribute of a documented error to learn which item the error is about, also when "
            "the item is unknown, missing or invalid: that is exactly when it is needed. Unless a sentence of the "
            "documentation says this error carries none, pass the value the function received, at every raise site "
            "of that function, then run the input that reaches the line and check the attribute."
        )
    return message + " Then run your tests again and reply with a one-line summary."


_PY_FILE = re.compile(r"^\+\+\+ b/(.+\.py)$")
_RAISE = re.compile(r"\braise\s+([A-Z]\w*)\((.*)$")
_SELF_KEYWORD = re.compile(r"\b(\w+)\s*=\s*self\.\w+")
_ANY_KEYWORD = re.compile(r"(?<![=!<>])\b([a-z_]\w*)\s*=(?!=)")


@dataclass(frozen=True)
class AttributeGap:
    error: str
    keyword: str
    with_it: int
    without: tuple[str, ...]  # "path: line" for each site that passes no keyword


def _python_source(path: str) -> bool:
    parts = path.split("/")
    name = parts[-1]
    return not (name.startswith("test") or name.endswith("_test.py") or "tests" in parts[:-1] or "test" in parts[:-1])


def changed_python_sources(diff: str) -> list[str]:
    """The non-test .py files the diff adds or changes."""
    return [m.group(1) for line in diff.splitlines() if (m := _PY_FILE.match(line)) and _python_source(m.group(1))]


def attribute_gaps(diff: str, docs: dict[str, str]) -> list[AttributeGap]:
    """Error classes raised with a keyword from the object's own state at some added sites and none at others."""
    documentation = "\n".join(docs.values())
    sites: dict[str, list[tuple[str, set[str], set[str]]]] = {}
    lines, path = diff.splitlines(), None
    for index, raw in enumerate(lines):
        if raw.startswith("+++ "):
            match = _PY_FILE.match(raw)
            path = match.group(1) if match and _python_source(match.group(1)) else None
            continue
        if path is None or not raw.startswith("+") or not (match := _RAISE.search(raw)):
            continue
        text, end = match.group(2), index
        while text.count(")") <= text.count("(") and end < index + 4 and end + 1 < len(lines) and lines[end + 1].startswith("+"):
            end += 1
            text += " " + lines[end][1:]
        sites.setdefault(match.group(1), []).append(
            (f"{path}: `{raw[1:].strip()[:120]}`", set(_SELF_KEYWORD.findall(text)), set(_ANY_KEYWORD.findall(text)))
        )
    gaps = []
    for error, group in sites.items():
        for keyword in sorted(set().union(*(state for _, state, _ in group))):
            if re.search(rf"\b{keyword}\s*=\s*None\b|`{keyword}`\s+is\s+`?None|\b{keyword}\s+is\s+`?None", documentation):
                continue
            without = tuple(site for site, _, any_keyword in group if not any_keyword)
            with_it = sum(keyword in state for _, state, _ in group)
            if without and with_it:
                gaps.append(AttributeGap(error, keyword, with_it, without[:8]))
    return gaps


def attribute_prompt(gaps: list[AttributeGap]) -> str:
    parts = []
    for gap in gaps[:4]:
        listed = "\n".join(f"- {site}" for site in gap.without)
        parts.append(
            f"Raises of `{gap.error}` pass `{gap.keyword}=` from the raising object's own state at {gap.with_it} "
            f"site(s) and leave it out at these:\n{listed}"
        )
    return (
        "A check of your change found error attributes set at some raise sites only.\n\n"
        + "\n\n".join(parts)
        + "\n\nIf the documentation defines the attribute for the class as a whole, set it from that state on every "
        "code that object raises, through one helper, unless a sentence says that code lacks it; otherwise leave "
        "these as they are. Then run your tests again and reply with a one-line summary."
    )


_ARGPARSE_PRIVATE = re.compile(
    r"(?<!self)(?<!cls)\.(_negative_number_matcher|_has_negative_number_optionals|_option_string_actions|_parse_optional"
    r"|_get_option_tuples|_match_argument|_get_nargs_pattern)\b"
)
_ARGPARSE_USE = re.compile(r"add_argument\(|ArgumentParser|\bargparse\b")


def argparse_private_uses(diff: str) -> list[str]:
    """Added lines that read or change argparse's private parser state, in files that use argparse."""
    files: dict[str, list[str]] = {}
    path = None
    for raw in diff.splitlines():
        if raw.startswith("+++ "):
            match = _PY_FILE.match(raw)
            path = match.group(1) if match and _python_source(match.group(1)) else None
            continue
        if path is not None and raw.startswith("+") and not raw.startswith("+++"):
            files.setdefault(path, []).append(raw[1:])
    sites = []
    for path, added in files.items():
        if not any(_ARGPARSE_USE.search(line) for line in added):
            continue
        for line in added:
            if not line.lstrip().startswith("#") and _ARGPARSE_PRIVATE.search(line):
                sites.append(f"{path}: `{line.strip()[:160]}`")
    return sites[:8]


def argparse_prompt(sites: list[str]) -> str:
    listed = "\n".join(f"- {site}" for site in sites)
    return (
        "A check of your change found code that reads or changes argparse's private parser state:\n"
        f"{listed}\n\n"
        "That state decides how argparse reads every option, not only the case you aimed at. A changed "
        "_negative_number_matcher, for example, decides which words starting with '-' an option takes as its value, "
        "so `--opt -1` can become a usage error before your own range check runs. Hidden tests pass negative numbers "
        "to integer options and expect the documented result. Remove it and keep argparse's own reading of every "
        "option. An option value that starts with '-' and is not a number is then argparse's own usage error, which "
        "is what the documentation means where it lists unknown or missing arguments as usage errors; where a test "
        "expects such a value to reach your own check, answer that test with the documentation's sentence instead of "
        "changing the parser. Only if a sentence of the documentation itself says such a value reaches your check, "
        "rewrite that option and the word after it into the `--opt=value` form before parse_args reads them. Then "
        "run each integer option with -1 through the real command line, run your tests again and reply with a "
        "one-line summary."
    )
