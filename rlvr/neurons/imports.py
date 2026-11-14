"""Import parity: each existing Python module the answer edits must import in the validators' image as it did.

7efd1df0229c (2026-11-12, msgspec schema catalog) failed 17 of 18 hidden tests. msgspec's compiled extension
`msgspec._core` is not built in the image, so `import msgspec.json` fails there with ModuleNotFoundError. To
test its new class anyway, the model rewrote the project's own `msgspec/__init__.py` and `msgspec/json.py` to
import without the extension. The hidden suite guards on exactly that import (`except ModuleNotFoundError as
exc: if exc.name != 'msgspec._core': raise`) and expected its installed-API tests to be skipped in this image;
with the workaround they ran instead, and the output no longer matched. The task had said to keep the rest of
the project's behaviour unchanged.

So, before the review, every existing module the answer edits is imported twice in the sandbox, each in a fresh
interpreter with the task's PYTHONPATH: once from a tree with the files as the repository supplied them (a link
farm over /work with the edited files' original contents put back and the answer's new files left out) and once
from /work with the answer. A different outcome (imports in one and not the other, or another error) goes back
to the model once. A module that failed only because it imports a file the answer adds is not held against it.
"""

from __future__ import annotations

import base64
import json
import re
import zlib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from rlvr.neurons.surface import python_command  # the same choice of interpreter as the surface check

LANGUAGES = frozenset({"python"})
MAX_MODULES = 12
MAX_PROMPTS = 1
MIN_TIME_S = 120.0
TIMEOUT_S = 100
PER_IMPORT_S = 20
# The sandbox takes commands up to 64 KiB; the original contents ride inside the command, compressed.
MAX_PAYLOAD_CHARS = 56 * 1024
MAX_FILE_BYTES = 1024**2
DELIMITER = "__HONE_IMPORTS__"
MARKER = "IMPORT-PARITY "

_PYTHONPATH_RE = re.compile(r"PYTHONPATH=([^\s;`'\"]+)")


@dataclass(frozen=True)
class Difference:
    module: str
    before: str
    after: str


def python_roots(instruction: str, working_directory: str = ".") -> list[str]:
    """Import roots relative to the workspace root ("" is the root), longest first."""
    roots: set[str] = set()
    match = _PYTHONPATH_RE.search(instruction or "")
    entries = match.group(1).split(":") if match else ["/work", "/work/src"]
    for entry in entries:
        if entry == "/work" or entry.startswith("/work/"):
            roots.add(entry[len("/work/"):].strip("/") if entry != "/work" else "")
    working = (working_directory or ".").strip("/")
    roots.add("" if working in ("", ".") else working)
    return sorted(roots, key=lambda root: (-len(root), root))


def module_name(relative: str, is_package: Callable[[str], bool]) -> tuple[str, str] | None:
    """(dotted module, its import root) for a .py file, named the way its users import it.

    The module's package chain is followed upwards while each directory holds an __init__.py; the directory
    above the top package is the root (a1b8ebf33ba5: a package under api/python_api/ imports itself as
    kubeflow_spark_api, and naming it api.python_api.kubeflow_spark_api from /work broke its absolute imports).
    """
    if not relative.endswith(".py"):
        return None
    parts = relative[: -len(".py")].split("/")
    names = parts[-1:] if parts[-1] != "__init__" else []
    directory = parts[:-1]
    while directory and is_package("/".join(directory)):
        names.insert(0, directory.pop())
    if not names or not all(name.isidentifier() for name in names):
        return None
    return ".".join(names), "/".join(directory)


_SCRIPT = r'''
import base64, json, os, shutil, subprocess, sys, tempfile, zlib
MODULES = __MODULES__
ORIGINALS = json.loads(zlib.decompress(base64.b64decode(__ORIGINALS__)).decode("utf-8"))
NEW = set(__NEW__)
NEW_MODULES = __NEW_MODULES__
ROOTS = __ROOTS__
CWD = __CWD__
PER_IMPORT_S = __PER_IMPORT_S__
SOURCE = "/work"
farm = tempfile.mkdtemp(prefix="hone-original-")
need = set()
for rel in list(ORIGINALS) + sorted(NEW):
    parts = rel.split("/")
    for i in range(1, len(parts)):
        need.add("/".join(parts[:i]))

def build(rel):
    src = os.path.join(SOURCE, rel) if rel else SOURCE
    dst = os.path.join(farm, rel) if rel else farm
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        r = rel + "/" + name if rel else name
        path = os.path.join(src, name)
        if r in NEW:
            continue
        if r in ORIGINALS:
            with open(os.path.join(dst, name), "wb") as handle:
                handle.write(base64.b64decode(ORIGINALS[r]))
        elif r in need and os.path.isdir(path) and not os.path.islink(path):
            build(r)
        else:
            os.symlink(path, os.path.join(dst, name))

build("")
CODE = (
    "import importlib, json, sys\n"
    "try:\n"
    "    importlib.import_module(sys.argv[1])\n"
    "    print('RESULT ' + json.dumps({'ok': True}))\n"
    "except BaseException as e:\n"
    "    print('RESULT ' + json.dumps({'ok': False, 'type': type(e).__name__, 'name': getattr(e, 'name', None), 'msg': str(e)[:300]}))\n"
)

def probe(tree, module, own_root):
    env = dict(os.environ)
    roots = [own_root] + [root for root in ROOTS if root != own_root]
    env["PYTHONPATH"] = ":".join(os.path.join(tree, root) if root else tree for root in roots)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    cwd = os.path.join(tree, CWD) if CWD else tree
    try:
        run = subprocess.run([sys.executable, "-B", "-c", CODE, module], cwd=cwd, env=env,
                             capture_output=True, text=True, timeout=PER_IMPORT_S)
    except subprocess.TimeoutExpired:
        return {"ok": False, "type": "timeout", "name": None, "msg": ""}
    for line in run.stdout.splitlines()[::-1]:
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    return {"ok": False, "type": "exit %d" % run.returncode, "name": None, "msg": run.stderr[-300:]}

def describe(result):
    if result["ok"]:
        return "imports"
    text = result["type"]
    if result.get("name"):
        text += " (%s)" % result["name"]
    elif result.get("msg"):
        text += ": " + result["msg"].replace(farm, SOURCE).splitlines()[0][:160]
    return text

def from_new_file(result):
    blob = "%s %s" % (result.get("name") or "", result.get("msg") or "")
    return any(new == (result.get("name") or "") or new in blob for new in NEW_MODULES)

for module, own_root in MODULES:
    before, after = probe(farm, module, own_root), probe(SOURCE, module, own_root)
    same = before["ok"] == after["ok"] and (before["ok"] or (before["type"], before.get("name")) == (after["type"], after.get("name")))
    # A module that could not load before only because it needs a file the answer adds compares nothing.
    if same or (not before["ok"] and from_new_file(before)):
        continue
    print("IMPORT-PARITY " + json.dumps({"module": module, "before": describe(before), "after": describe(after)}))
shutil.rmtree(farm, ignore_errors=True)
print("IMPORT-PARITY-DONE")
'''


def check_command(
    touched: Iterable[str],
    originals: Mapping[str, bytes],
    roots: list[str],
    working_directory: str = ".",
    is_package: Callable[[str], bool] = lambda directory: False,
    python: str = "python3",
) -> str | None:
    """The sandbox command, or None when there is nothing to compare or the originals do not fit.

    `touched` lists every file the answer changes; `originals` holds the repository's content of those that
    existed before (the others are the answer's new files); `is_package` says whether a directory of the
    answered tree holds an __init__.py.
    """
    touched = sorted(set(touched))
    new = [path for path in touched if path not in originals]
    modules: list[tuple[str, str]] = []
    for path in touched:
        if path in originals:
            named = module_name(path, is_package)
            if named and named not in modules:
                modules.append(named)
    if not modules:
        return None
    modules = modules[:MAX_MODULES]
    new_modules = sorted({named[0] for named in (module_name(path, is_package) for path in new) if named})
    originals = {path: content for path, content in originals.items() if len(content) <= MAX_FILE_BYTES}
    payload = base64.b64encode(
        zlib.compress(
            json.dumps({path: base64.b64encode(content).decode("ascii") for path, content in originals.items()}).encode("utf-8"),
            9,
        )
    ).decode("ascii")
    if len(payload) > MAX_PAYLOAD_CHARS:
        return None
    working = (working_directory or ".").strip("/")
    script = (
        _SCRIPT.replace("__MODULES__", repr(modules))
        .replace("__ORIGINALS__", repr(payload))
        .replace("__NEW_MODULES__", repr(new_modules))
        .replace("__NEW__", repr(new))
        .replace("__ROOTS__", repr(roots))
        .replace("__CWD__", repr("" if working in ("", ".") else working))
        .replace("__PER_IMPORT_S__", repr(PER_IMPORT_S))
    )
    return f"cd /work && {python_command(python)} - <<'{DELIMITER}'\n{script.strip()}\n{DELIMITER}\n"


def differences(output: str) -> list[Difference] | None:
    """What changed, or None when the check did not finish."""
    if "IMPORT-PARITY-DONE" not in output:
        return None
    found = []
    for line in output.splitlines():
        if line.startswith(MARKER):
            item = json.loads(line[len(MARKER):])
            found.append(Difference(item["module"], item["before"], item["after"]))
    return found


def prompt(found: list[Difference]) -> str:
    lines = "\n".join(f"- {d.module}: as the repository supplied it: {d.before}; with your change: {d.after}" for d in found)
    return (
        "A check imported each existing module your change edits in the validators' image, once as the "
        "repository supplied it and once with your change, each in a fresh interpreter with the task's "
        f"PYTHONPATH, and the outcome differs:\n{lines}\n\n"
        "The hidden tests run in this same image, and a suite can depend on how the project's existing modules "
        "import there: it may skip or branch on an import that fails because a compiled extension or an "
        "optional dependency is missing from the image. Unless the task asks for that change, give each module "
        "back its import behaviour exactly: undo what you changed to work around the environment (a guarded "
        "import, a fallback, a stub) and test new code from your throwaway test instead, for example with a stub "
        "for the missing module placed in sys.modules there, never in the project's files. A module that no "
        "longer imports at all is broken by your change and must be fixed. Then run your tests again and reply "
        "with a one-line summary."
    )
