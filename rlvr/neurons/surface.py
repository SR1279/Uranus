"""Run the import lines of the task's documentation against the answer, before the review.

Three of the eight failures of 2026-11-10/11 after the sandbox fix were a documented public surface that was
never wired: a package directory without __init__.py (every hidden test failed at its first import), a
package __init__ without the documented re-exports, and an entry module the documentation named that the
answer never touched. Each documentation file writes out the import lines a user types, and the hidden suite
begins with the same lines. This collects those lines, runs them in the sandbox from the repository root and
sends back the ones that fail, naming the missing file or name.

Only a module whose files exist somewhere in the workspace is held against the answer: an import of a
third-party or standard library module that fails says nothing about the work. The grader's image has only
the standard library, so a project whose package __init__ imports a dependency cannot be imported plainly;
its hidden tests put an empty module in place of the top-level package (marimo, 2026-11-11: msgspec was
missing) and import the documented submodules beneath it. When a plain import fails in code the answer did
not touch, the check does the same, and only a failure in the answer's own files, or a missing module of the
project itself, is reported.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

# Languages whose documentation import lines this reads.
LANGUAGES = frozenset({"python"})
MAX_SPECS = 40
TIMEOUT_S = 60
# Skipped when less than this is left before the wrap-up, so that a fix still fits.
MIN_TIME_S = 150.0
# A solve is sent back for failed imports at most this many times.
MAX_PROMPTS = 2
DELIMITER = "__HONE_SURFACE__"
# The validators' second interpreter (checked in the pinned image on 2026-11-14): Python 3.12 with pydantic 2, httpx,
# anyio, cryptography, zstandard and a few more. 4 of 59 stored hidden checks ran with it, and only the documentation
# can say so: every instruction says "Python 3.12 (image interpreter)", and the project's own imports do not predict
# it (16 of 27 plain-python3 checks were of projects importing one of its packages). Evalscope's documentation gave
# its test command (5fbaf671fe6a), and the plain interpreter reported pydantic missing from the answer's own file.
VENV_PYTHON = "/opt/rlvr-venv/bin/python"

Spec = tuple[str, tuple[str, ...]]

_FROM_RE = re.compile(r"^\s*(?:>>>\s*)?from\s+([A-Za-z_][\w.]*)\s+import\s+(.*)$")
_NAME_RE = re.compile(r"^[A-Za-z_]\w*$")

_SCRIPT = '''
import importlib, os, sys, traceback, types
SPECS = __SPECS__
ATTRIBUTES = __ATTRIBUTES__
ROOT = __ROOT__
ANSWER = set(__ANSWER__)
# Import lines each edited file already had in the repository as supplied.
PRISTINE = {path: set(lines) for path, lines in __PRISTINE__.items()}
SKIP = {".git", "node_modules", "__pycache__", ".venv", "venv", ".tox", ".mypy_cache", ".pytest_cache", "site-packages", "dist", "build"}
dirs = []
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = sorted(d for d in dirnames if d not in SKIP and not d.startswith("."))
    dirs.append(dirpath)
    if len(dirs) > 20000:
        break


def roots(module):
    parts = module.split(".")
    found = []
    for d in dirs:
        base = os.path.join(d, *parts)
        if os.path.isfile(base + ".py") or os.path.isdir(base):
            found.append(d)
    return sorted(found, key=len)


def in_answer(exc):
    """Whether the failure was raised in, or is a syntax error of, a file the answer changes.

    Only the innermost frame inside the workspace counts: an import chain may pass through the answer on
    its way to a dependency the image lacks, and that is not the answer's failure.
    """
    if isinstance(exc, SyntaxError) and exc.filename:
        name = os.path.abspath(exc.filename)
        if name.startswith(ROOT + os.sep):
            return os.path.relpath(name, ROOT) in ANSWER
    frames = [frame for frame in traceback.extract_tb(exc.__traceback__)
              if os.path.abspath(frame.filename).startswith(ROOT + os.sep)]
    if not frames:
        return False
    relative = os.path.relpath(os.path.abspath(frames[-1].filename), ROOT)
    if relative not in ANSWER:
        return False
    # An import line the file already had, left as it was, fails the same way in the repository as supplied
    # (an edited msgspec/__init__.py still starts with `from ._core import ...`): not the answer's failure.
    return (frames[-1].line or "").strip() not in PRISTINE.get(relative, set())


def of_project(exc, module):
    """A missing module the answer is answerable for, rather than a dependency or the repository as supplied.

    The documented module and its packages always count, and so does any module whose source is in the workspace.
    A project module with no source anywhere (a compiled extension the image never builds, such as msgspec._core)
    counts only when the answer's own files import it: raised from untouched code, the repository as supplied
    fails the same way, and making it import changed which hidden tests ran (7efd1df0229c).
    """
    if not isinstance(exc, ModuleNotFoundError) or not exc.name:
        return False
    # A top-level name found only as a file deeper in the tree (piccolo/utils/pydantic.py) is another module of
    # the project, not the missing dependency (piccolo, 2026-11-14).
    if exc.name == module or module.startswith(exc.name + ".") or ("." in exc.name and roots(exc.name)):
        return True
    if exc.name.split(".")[0] != module.split(".")[0]:
        return False
    return in_answer(exc)


def forget(top):
    for key in [key for key in sys.modules if key == top or key.startswith(top + ".")]:
        del sys.modules[key]


def resolve(mod, module, name):
    """Whether `from module import name` works: True, False, or the error of a submodule that exists but fails.

    Python falls back to importing module.name when the package has no such attribute, so a documented
    submodule nobody imported yet is present (marimo, 2026-11-11: cell_runner was reported missing).
    """
    if hasattr(mod, name):
        return True
    try:
        importlib.import_module(module + "." + name)
        return True
    except ModuleNotFoundError as exc:
        if exc.name == module + "." + name:
            return False
        return exc
    except BaseException as exc:
        return exc


def attempt(module, found, stub):
    forget(module.split(".")[0])
    # The hidden tests run with PYTHONPATH=/work:/work/src.
    sys.path[:] = list(dict.fromkeys(found + [ROOT, os.path.join(ROOT, "src")] + original))
    if stub:
        top = module.split(".")[0]
        package = types.ModuleType(top)
        package.__path__ = [os.path.join(found[0], top)]
        sys.modules[top] = package
    return importlib.import_module(module)


original = list(sys.path)
failed = []
skipped = 0
checked = 0
for module, names in SPECS:
    found = roots(module)
    checked += 1
    mod = None
    try:
        mod = attempt(module, found, False)
    except BaseException as exc:
        error = exc
        if found and not in_answer(error) and not of_project(error, module) and "." in module:
            # The way the hidden tests import a package whose __init__ needs what the image lacks.
            try:
                mod = attempt(module, found, True)
            except BaseException as retry:
                error = retry
        if mod is None:
            if not found:
                continue  # not part of this workspace: a dependency or the standard library
            if in_answer(error) or of_project(error, module):
                detail = str(error)[:200]
                if isinstance(error, ModuleNotFoundError):
                    detail += " (the files of %s are under %s)" % (module, found[0])
                failed.append("from %s import %s: %s: %s" % (module, ", ".join(names), type(error).__name__, detail))
            else:
                skipped += 1  # broken outside the answer, as in the untouched repository
            continue
    missing = []
    for name in names:
        present = resolve(mod, module, name)
        if present is False:
            missing.append(name)
        elif present is not True:
            # The submodule exists but does not import: the answer's failure only if it is raised in the
            # answer's own files or names a missing module of the project.
            if in_answer(present) or of_project(present, module):
                failed.append("from %s import %s: %s: %s" % (module, name, type(present).__name__, str(present)[:200]))
            else:
                skipped += 1
    if missing:
        hint = ""
        if getattr(mod, "__file__", None) is None:
            hint = " (the package directory has no __init__.py, so it is a namespace package with no names)"
        failed.append("from %s import %s: not in the module%s" % (module, ", ".join(missing), hint))
for owner, name, via in ATTRIBUTES:
    # A dotted name the documentation states (nltk.BigramLedger), looked up after the documented modules
    # beneath it are imported, as a hidden test does.
    found = roots(owner)
    if not found:
        continue
    checked += 1
    stubbed = False
    try:
        mod = attempt(owner, found, False)
    except BaseException as exc:
        if in_answer(exc) or of_project(exc, owner):
            failed.append("%s.%s: importing %s fails: %s: %s" % (owner, name, owner, type(exc).__name__, str(exc)[:200]))
            continue
        reason = "%s: %s" % (type(exc).__name__, str(exc)[:120])
        try:
            # The image lacks what the package's __init__ needs, so the hidden tests import an empty package
            # in its place (feb50a592a31: `import nltk` fails on regex), and only what the documented modules
            # set on it when they are imported is there.
            mod = attempt(owner, found, True)
            stubbed = True
        except BaseException:
            skipped += 1
            continue
    for sub in via:
        try:
            importlib.import_module(sub)
        except BaseException:
            pass
    if not hasattr(mod, name):
        text = "%s.%s: %s has no attribute %s" % (owner, name, owner, name)
        if via:
            text += " after importing " + ", ".join(via)
        if stubbed:
            text += (" (in this image `import %s` fails outside your change, %s, so the hidden tests put an empty "
                     "%s package in its place: the attribute is there only if the documented modules set it on "
                     "the package when they are imported)" % (owner, reason, owner))
        failed.append(text)
for line in failed:
    print("FAIL " + line)
print("documented imports: %d checked, %d failed, %d could not be judged" % (checked, len(failed), skipped))
sys.exit(1 if failed else 0)
'''


def documented_imports(docs: dict[str, str]) -> list[Spec]:
    """Every `from X import a, b` line in the documentation, absolute modules only, names in order of appearance."""
    specs: dict[str, list[str]] = {}
    for text in docs.values():
        lines = text.splitlines()
        index = 0
        while index < len(lines):
            match = _FROM_RE.match(lines[index])
            index += 1
            if match is None:
                continue
            module, rest = match.group(1), match.group(2)
            if module.startswith(".") or module.startswith("__"):
                continue
            if "(" in rest and ")" not in rest:
                # A parenthesised list continues on the following lines.
                while index < len(lines) and ")" not in lines[index]:
                    rest += " " + lines[index]
                    index += 1
                if index < len(lines):
                    rest += " " + lines[index]
                    index += 1
            rest = rest.split("#", 1)[0].replace("(", " ").replace(")", " ").replace("\\", " ")
            names = []
            for part in rest.split(","):
                tokens = part.split()
                if tokens and _NAME_RE.match(tokens[0]):  # "name as alias" keeps the name
                    names.append(tokens[0])
            if not names:
                continue
            listed = specs.setdefault(module, [])
            listed.extend(name for name in names if name not in listed)
    return [(module, tuple(names)) for module, names in specs.items()][:MAX_SPECS]


_DOTTED_RE = re.compile(r"(?<![\w./-])([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)(?![\w(/-])")
_SENTENCE_RE = re.compile(r"(?<=[.!?:])\s+(?=[A-Z`*_(\[])|\n\s*\n")
_NEGATION_RE = re.compile(r"\b(not|never|no longer|removed|without|nor|cannot)\b|n't\b", re.IGNORECASE)
MAX_ATTRIBUTES = 12

Attribute = tuple[str, str, tuple[str, ...]]


def documented_attributes(docs: dict[str, str], specs: list[Spec]) -> list[Attribute]:
    """Dotted names the documentation states for a documented name under another module of the project.

    feb50a592a31 (2026-11-13): the documentation imported BigramLedger from nltk.collocation_ledger and said
    importing it also publishes `nltk.BigramLedger`; the answer never set it and the hidden test of that failed.
    Only a name the documentation's import lines bring in counts, under a module of the same top-level
    package, and a sentence that says what is not there is skipped. Each comes with the documented modules
    beneath its owner, imported first as a hidden test does.
    """
    names = {name for _module, listed in specs for name in listed}
    modules = [module for module, _names in specs]
    tops = {module.split(".")[0] for module in modules}
    stated = {(module, name) for module, listed in specs for name in listed}
    found: dict[tuple[str, str], None] = {}
    for text in docs.values():
        for sentence in _SENTENCE_RE.split(text):
            if _NEGATION_RE.search(sentence):
                continue
            for match in _DOTTED_RE.finditer(sentence):
                parts = match.group(1).split(".")
                owner, name = ".".join(parts[:-1]), parts[-1]
                if parts[0] not in tops or name not in names or (owner, name) in stated or owner + "." + name in modules:
                    continue
                if not (owner in modules or any(module.startswith(owner + ".") for module in modules)):
                    continue
                found[(owner, name)] = None
    attributes = []
    for owner, name in list(found)[:MAX_ATTRIBUTES]:
        via = tuple(module for module in modules if module.startswith(owner + "."))
        attributes.append((owner, name, via))
    return attributes


def check_script(
    specs: list[Spec], root: str = "/work", answer: Iterable[str] = (), attributes: Iterable[Attribute] = (),
    pristine: dict[str, Iterable[str]] | None = None,
) -> str:
    """The Python that runs the imports with the workspace's own roots on sys.path and prints FAIL lines.

    `answer` lists the files the answer changes, relative to the root: a failure raised in one of them is
    always the answer's.
    """
    return (
        _SCRIPT.replace("__SPECS__", repr([(module, list(names)) for module, names in specs]))
        .replace("__ATTRIBUTES__", repr([(owner, name, list(via)) for owner, name, via in attributes]))
        .replace("__ROOT__", repr(root))
        .replace("__ANSWER__", repr(sorted(answer)))
        .replace("__PRISTINE__", repr({path: sorted(lines) for path, lines in (pristine or {}).items()}))
    )


def interpreter(docs: Mapping[str, str] | None, instruction: str = "") -> str:
    """python3, or the validators' venv interpreter when the task's own text names it."""
    return VENV_PYTHON if any(VENV_PYTHON in text for text in (instruction, *(docs or {}).values())) else "python3"


def python_command(python: str = "python3") -> str:
    """The interpreter as a shell word: the one named when the sandbox has it, python3 otherwise."""
    if python == "python3":
        return "python3"
    return f'"$([ -x {python} ] && echo {python} || echo python3)"'


def check_command(
    specs: list[Spec], root: str = "/work", answer: Iterable[str] = (), attributes: Iterable[Attribute] = (),
    pristine: dict[str, Iterable[str]] | None = None, python: str = "python3",
) -> str:
    """The sandbox command: the script fed to the interpreter from the repository root."""
    script = check_script(specs, root, answer, attributes, pristine).strip()
    return f"cd {root} && {python_command(python)} - <<'{DELIMITER}'\n{script}\n{DELIMITER}\n"


def failures(output: str) -> list[str]:
    return [line[len("FAIL "):].strip() for line in output.splitlines() if line.startswith("FAIL ")]


def prompt(failed: list[str]) -> str:
    return (
        "The import lines written in the task's documentation were run against your work from the repository root, "
        "and these failed:\n" + "\n".join(f"- {line}" for line in failed) + "\n\n"
        "A hidden test begins with the documentation's own import lines, so every documented name must be importable "
        "exactly as written: add the missing __init__.py, re-export or module rather than renaming anything. A dotted "
        "name the documentation states (package.Name) must be an attribute of that package once the documented "
        "modules are imported. Then run your tests again and reply with a one-line summary."
    )
