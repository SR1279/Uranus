"""The documentation's import lines, run against the work before the review."""

from __future__ import annotations

import subprocess
import sys

import pytest

from rlvr.neurons import surface
from rlvr.neurons.demo_miner import DemoMiner
from tests.test_acceptance import RUN, ScriptedSandbox, documented_task, result, run_loop
from tests.test_demo_miner_workbench import DONE, FINISHED, Scripted, edit, workspace  # noqa: F401

DOC = (
    "# Pkg\n\nUsers write:\n\n```python\n"
    "from pkg.core import Thing, helper as h  # the two public names\n"
    "from pkg import (\n    Thing,\n    OtherError,\n)\n"
    "from .private import x\n"
    "from pkg.core import *\n"
    ">>> from pkg.cli import main\n"
    "from __future__ import annotations\n"
    "import os\n"
    "```\n"
)


def test_documented_imports_are_collected_in_order_without_relative_or_star_lines():
    assert surface.documented_imports({"d.md": DOC}) == [
        ("pkg.core", ("Thing", "helper")),
        ("pkg", ("Thing", "OtherError")),
        ("pkg.cli", ("main",)),
    ]
    assert surface.documented_imports({"d.md": "no import lines here\n"}) == []
    many = "\n".join(f"from m{i} import a" for i in range(50))
    assert len(surface.documented_imports({"d.md": many})) == surface.MAX_SPECS
    assert surface.documented_imports({"a.md": "from p import a\n", "b.md": "from p import b, a\n"}) == [("p", ("a", "b"))]


def test_the_check_names_a_missing_init_a_missing_name_and_a_broken_module_but_not_third_party_code(tmp_path):
    root = tmp_path / "work"
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "src" / "pkg" / "__init__.py").write_text("from .core import Thing\n")
    (root / "src" / "pkg" / "core.py").write_text("class Thing:\n    pass\n")
    (root / "scripts" / "ns").mkdir(parents=True)  # a package directory without __init__.py
    (root / "scripts" / "ns" / "session.py").write_text("class Session:\n    pass\n")
    (root / "broken.py").write_text("def (\n")
    specs = [
        ("pkg", ("Thing", "OtherError")), ("pkg.core", ("Thing",)), ("ns", ("Session",)), ("broken", ("x",)),
        ("no_such_third_party_module", ("load",)), ("json", ("loads",)),
    ]
    answer = ["src/pkg/__init__.py", "src/pkg/core.py", "broken.py"]
    proc = subprocess.run([sys.executable, "-I", "-"], input=surface.check_script(specs, str(root), answer), text=True, capture_output=True)
    failed = surface.failures(proc.stdout)
    assert proc.returncode == 1, proc.stderr
    assert failed[0] == "from pkg import OtherError: not in the module"
    assert failed[1] == (
        "from ns import Session: not in the module (the package directory has no __init__.py, so it is a "
        "namespace package with no names)"
    )
    assert failed[2].startswith("from broken import x: SyntaxError:")
    assert len(failed) == 3 and "documented imports: 6 checked, 3 failed" in proc.stdout
    # A syntax error in a file the answer does not change is the repository's, not the answer's.
    untouched = subprocess.run([sys.executable, "-I", "-"], input=surface.check_script([("broken", ("x",))], str(root)), text=True, capture_output=True)
    assert untouched.returncode == 0 and "1 could not be judged" in untouched.stdout
    passing = subprocess.run([sys.executable, "-I", "-"], input=surface.check_script([("pkg.core", ("Thing",))], str(root)), text=True, capture_output=True)
    assert passing.returncode == 0 and surface.failures(passing.stdout) == []


def run_check(specs, root, answer=()):
    proc = subprocess.run(
        [sys.executable, "-I", "-"], input=surface.check_script(specs, str(root), answer), text=True, capture_output=True,
    )
    return proc, surface.failures(proc.stdout)


def test_a_package_whose_init_needs_a_missing_dependency_is_imported_the_way_the_hidden_tests_do(tmp_path):
    # marimo, 2026-11-11: marimo/__init__.py needs msgspec, which the image lacks; the hidden tests put an
    # empty module in place of marimo and import marimo._convert beneath it.
    root = tmp_path / "work"
    (root / "pkg" / "conv").mkdir(parents=True)
    (root / "pkg" / "__init__.py").write_text("import not_installed_dependency\n")
    (root / "pkg" / "conv" / "__init__.py").write_text("from .tape import Tape\n")
    (root / "pkg" / "conv" / "tape.py").write_text("class Tape:\n    pass\n")
    answer = ["pkg/conv/__init__.py", "pkg/conv/tape.py"]
    proc, failed = run_check([("pkg.conv", ("Tape",)), ("pkg.conv.tape", ("Tape",))], root, answer)
    assert proc.returncode == 0 and failed == [] and "2 checked, 0 failed, 0 could not be judged" in proc.stdout
    # Beneath the empty package, a name the answer never wrote is still found missing.
    proc, failed = run_check([("pkg.conv.tape", ("Tape", "Missing"))], root, answer)
    assert failed == ["from pkg.conv.tape import Missing: not in the module"]
    # The top-level package itself cannot be imported another way: not the answer's failure.
    proc, failed = run_check([("pkg", ("anything",))], root, answer)
    assert proc.returncode == 0 and failed == [] and "1 could not be judged" in proc.stdout


def test_a_documented_submodule_is_present_even_when_the_package_does_not_import_it(tmp_path):
    # marimo, 2026-11-11: `from marimo._runtime.runner import cell_runner` names a submodule; it was sent back
    # twice as "not in the module", and the model rewrote the package's __init__ for nothing.
    root = tmp_path / "work"
    (root / "pkg" / "runner").mkdir(parents=True)
    (root / "pkg" / "__init__.py").write_text("")
    (root / "pkg" / "runner" / "__init__.py").write_text("")
    (root / "pkg" / "runner" / "cell_runner.py").write_text("VALUE = 1\n")
    (root / "pkg" / "runner" / "needs_dep.py").write_text("import not_installed_dependency\n")
    (root / "pkg" / "runner" / "broken.py").write_text("def (\n")
    specs = [("pkg.runner", ("cell_runner", "needs_dep", "absent")), ("pkg.runner", ("broken",))]
    proc, failed = run_check(specs, root, ["pkg/runner/broken.py"])
    assert proc.returncode == 1, proc.stderr
    assert failed[0] == "from pkg.runner import absent: not in the module"
    assert failed[1].startswith("from pkg.runner import broken: SyntaxError")
    assert len(failed) == 2 and "1 could not be judged" in proc.stdout  # needs_dep: a dependency the image lacks


def test_a_failure_raised_in_the_answers_own_code_is_reported(tmp_path):
    root = tmp_path / "work"
    (root / "pkg" / "conv").mkdir(parents=True)
    (root / "pkg" / "__init__.py").write_text("import not_installed_dependency\n")
    (root / "pkg" / "conv" / "__init__.py").write_text("")
    (root / "pkg" / "conv" / "tape.py").write_text("import not_installed_numpy\n\nclass Tape:\n    pass\n")
    (root / "pkg" / "conv" / "broken.py").write_text("def (\n")
    (root / "pkg" / "conv" / "uses.py").write_text("from pkg.conv.helpers import h\n")
    answer = ["pkg/conv/tape.py", "pkg/conv/broken.py", "pkg/conv/uses.py"]
    specs = [("pkg.conv.tape", ("Tape",)), ("pkg.conv.broken", ("x",)), ("pkg.conv.uses", ("h",))]
    proc, failed = run_check(specs, root, answer)
    assert proc.returncode == 1, proc.stderr
    assert failed[0].startswith("from pkg.conv.tape import Tape: ModuleNotFoundError: No module named 'not_installed_numpy'")
    assert failed[1].startswith("from pkg.conv.broken import x: SyntaxError:")
    # A module of the project that does not exist is the answer's to create.
    assert failed[2].startswith("from pkg.conv.uses import h: ModuleNotFoundError: No module named 'pkg.conv.helpers'")
    assert len(failed) == 3


def test_an_import_chain_through_the_answer_into_a_missing_dependency_is_not_the_answers_failure(tmp_path):
    # The answer's file is on the way, but the line that failed is in code the answer did not touch.
    root = tmp_path / "work"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "__init__.py").write_text("from . import conv\n")
    (root / "pkg" / "conv.py").write_text("from . import messaging\n\nclass Tape:\n    pass\n")
    (root / "pkg" / "messaging.py").write_text("import not_installed_dependency\n")
    proc, failed = run_check([("pkg", ("Tape",))], root, ["pkg/conv.py"])
    assert proc.returncode == 0 and failed == [] and "1 could not be judged" in proc.stdout


def test_a_dependency_named_like_a_file_deeper_in_the_project_is_not_the_projects(tmp_path):
    # piccolo, 2026-11-14: commands/shared.py, which the answer did not touch, imports pydantic, which the image
    # lacks, and piccolo/utils/pydantic.py made the missing top-level pydantic look like a module of the project:
    # dump.py was sent back twice for an import the hidden tests never make.
    root = tmp_path / "work"
    for package in ("pkg", "pkg/utils", "pkg/commands"):
        (root / package).mkdir(parents=True, exist_ok=True)
        (root / package / "__init__.py").write_text("")
    (root / "pkg" / "utils" / "not_installed_dependency.py").write_text("")
    (root / "pkg" / "commands" / "shared.py").write_text("import not_installed_dependency\n")
    (root / "pkg" / "commands" / "dump.py").write_text(
        "from pkg.commands.shared import (\n    x,\n)\n\n\nasync def dump_schema():\n    pass\n"
    )
    proc, failed = run_check([("pkg.commands.dump", ("dump_schema",))], root, ["pkg/commands/dump.py"])
    assert proc.returncode == 0 and failed == [] and "1 could not be judged" in proc.stdout, proc.stdout
    # A module of the project that the answer's own file imports and nobody wrote is still the answer's.
    (root / "pkg" / "commands" / "load.py").write_text("from pkg.utils.missing_helper import y\n\n\ndef load():\n    pass\n")
    proc, failed = run_check([("pkg.commands.load", ("load",))], root, ["pkg/commands/load.py"])
    assert len(failed) == 1 and "pkg.utils.missing_helper" in failed[0], proc.stdout


LEDGER_DOC = (
    "# Ledger\n\n```python\nfrom lib.ledger import Ledger, Event\nfrom lib.checkpoint import dump\n```\n\n"
    "Importing `lib.ledger` also publishes `Ledger` as an attribute of the top-level `lib` package, so "
    "`lib.Ledger is lib.ledger.Ledger`.  `lib.dump` is not published at the top level. "
    "Call `ledger.finder()` and read `lib.ledger.Ledger.revision`; `other.Ledger` is unrelated.\n"
)


def test_dotted_names_the_documentation_states_are_collected_without_negated_or_foreign_ones():
    specs = surface.documented_imports({"d.md": LEDGER_DOC})
    assert surface.documented_attributes({"d.md": LEDGER_DOC}, specs) == [("lib", "Ledger", ("lib.ledger", "lib.checkpoint"))]
    assert surface.documented_attributes({"d.md": DOC}, surface.documented_imports({"d.md": DOC})) == []


def test_a_documented_package_attribute_is_checked_on_the_empty_package_the_hidden_tests_use(tmp_path):
    # feb50a592a31 (2026-11-13): `import nltk` fails in the image (regex is missing), the hidden tests put an
    # empty nltk package in its place, and nltk.BigramLedger was never set by the documented module.
    root = tmp_path / "work"
    (root / "lib").mkdir(parents=True)
    (root / "lib" / "__init__.py").write_text("import no_such_dependency_in_this_image\nfrom lib.ledger import *\n")
    (root / "lib" / "ledger.py").write_text("__all__ = ['Ledger', 'Event']\nclass Ledger: pass\nclass Event: pass\n")
    (root / "lib" / "checkpoint.py").write_text("def dump(x): return x\n")
    specs = surface.documented_imports({"d.md": LEDGER_DOC})
    attributes = surface.documented_attributes({"d.md": LEDGER_DOC}, specs)

    def check():
        script = surface.check_script(specs, str(root), ["lib/ledger.py"], attributes)
        proc = subprocess.run([sys.executable, "-I", "-"], input=script, text=True, capture_output=True)
        return proc, surface.failures(proc.stdout)

    proc, failed = check()
    assert proc.returncode == 1 and len(failed) == 1, proc.stdout + proc.stderr
    assert failed[0].startswith("lib.Ledger: lib has no attribute Ledger after importing lib.ledger, lib.checkpoint")
    assert "`import lib` fails outside your change" in failed[0] and "no_such_dependency_in_this_image" in failed[0]
    (root / "lib" / "ledger.py").write_text(
        "import sys\nclass Ledger: pass\nclass Event: pass\nsys.modules['lib'].Ledger = Ledger\n"
    )
    proc, failed = check()
    assert proc.returncode == 0 and failed == [], proc.stdout + proc.stderr


def test_the_command_and_the_prompt():
    command = surface.check_command([("pkg", ("Thing",))])
    assert command.startswith("cd /work && python3 - <<'__HONE_SURFACE__'\n") and command.endswith("\n__HONE_SURFACE__\n")
    assert "[('pkg', ['Thing'])]" in command and "ROOT = '/work'" in command
    text = surface.prompt(["from pkg import OtherError: not in the module"])
    assert text.startswith("The import lines written in the task's documentation were run against your work")
    assert "- from pkg import OtherError: not in the module" in text and "run your tests again" in text
    assert "model" not in text.lower()


def test_the_checks_use_the_venv_interpreter_only_where_the_task_names_it(tmp_path):
    # evalscope 5fbaf671fe6a: its documentation gave the test command with /opt/rlvr-venv/bin/python, which has
    # pydantic, and the plain interpreter reported pydantic missing from the answer's own file.
    named = {"docs/a.md": "PYTHONPATH=/work:/work/src /opt/rlvr-venv/bin/python -m unittest -v\n"}
    assert surface.interpreter(named) == surface.VENV_PYTHON == "/opt/rlvr-venv/bin/python"
    assert surface.interpreter({"docs/a.md": "uses pydantic"}, "Python 3.12 (image interpreter), offline.") == "python3"
    assert surface.interpreter(None, "Run the tests with /opt/rlvr-venv/bin/python.") == surface.VENV_PYTHON
    assert surface.interpreter(None) == "python3"
    command = surface.check_command([("pkg", ("Thing",))], python=surface.VENV_PYTHON)
    assert command.startswith(
        'cd /work && "$([ -x /opt/rlvr-venv/bin/python ] && echo /opt/rlvr-venv/bin/python || echo python3)" - <<'
    )
    # Run for real: an interpreter that exists is used, and one the sandbox lacks falls back to python3.
    root = tmp_path / "work"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "__init__.py").write_text("import sys\nclass Thing:\n    pass\nprint('ran with', sys.executable)\n")
    for python, expected in [(sys.executable, sys.executable), ("/nonexistent/bin/python", "")]:
        command = surface.check_command([("pkg", ("Thing",))], str(root), python=python)
        proc = subprocess.run(["bash", "-c", command], text=True, capture_output=True, timeout=60)
        assert proc.returncode == 0 and "1 checked, 0 failed" in proc.stdout, proc.stdout + proc.stderr
        assert ("ran with " + expected) in proc.stdout

@pytest.fixture
def importing(workspace):
    (workspace / "docs").mkdir()
    (workspace / "docs" / "widgets.md").write_text("# Widgets\n\n```python\nfrom widgets import Widget\n```\n")
    return workspace


async def no_suite(self, request, docs, deadline):
    return None, None


FAILED = result(1, b"FAIL from widgets import Widget: not in the module (the package directory has no __init__.py, so it is a namespace package with no names)\ndocumented imports: 1 checked, 1 failed\n")
PASSED = result(0, b"documented imports: 1 checked, 0 failed\n")


async def test_failed_documented_imports_go_back_and_are_checked_again_after_the_fix(importing, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", no_suite)
    sandbox = ScriptedSandbox(result(0), FAILED, result(0), PASSED)
    provider = Scripted(edit(), RUN, FINISHED, edit("src/b.py", "return 1", "return 2"), RUN, FINISHED, DONE)
    events, submission, notes = await run_loop(importing, provider, sandbox)
    assert b"gamma" in submission and b"return 2" in submission
    sent_back = provider.seen[3][-1]["content"]
    assert sent_back.startswith("The import lines written in the task's documentation were run against your work")
    assert "from widgets import Widget: not in the module (the package directory has no __init__.py" in sent_back
    assert provider.seen[6][-1]["content"].startswith("Before this is submitted")
    assert notes.surface == "passed" and notes.surface_failures == 1 and notes.surface_checks == 2
    assert [e["call_id"] for e in events if e["event_type"] == "tool_call" and e["call_id"].startswith("surface-")] == ["surface-0", "surface-0", "surface-1", "surface-1"][::2]
    commands = [" ".join(request.argv) for request in sandbox.requests]
    assert surface.DELIMITER in commands[1] and surface.DELIMITER in commands[3] and "from widgets" not in commands[0]


async def test_the_documentation_naming_the_venv_interpreter_sends_the_check_through_it(importing, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", no_suite)
    (importing / "docs" / "widgets.md").write_text(
        "# Widgets\n\n```python\nfrom widgets import Widget\n```\n\nThe tests run with `/opt/rlvr-venv/bin/python -m unittest`.\n"
    )
    sandbox = ScriptedSandbox(result(0), PASSED)
    _, _, notes = await run_loop(importing, Scripted(edit(), RUN, FINISHED, DONE), sandbox)
    commands = [" ".join(request.argv) for request in sandbox.requests]
    assert notes.surface == "passed" and len(commands) == 2
    assert surface.DELIMITER in commands[1] and "[ -x /opt/rlvr-venv/bin/python ]" in commands[1]
    (importing / "docs" / "widgets.md").write_text("# Widgets\n\n```python\nfrom widgets import Widget\n```\n")
    sandbox = ScriptedSandbox(result(0), PASSED)
    await run_loop(importing, Scripted(edit(), RUN, FINISHED, DONE), sandbox)
    assert "rlvr-venv" not in " ".join(sandbox.requests[1].argv)

async def test_nothing_runs_without_import_lines_a_python_task_or_the_setting(importing, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", no_suite)
    sandbox = ScriptedSandbox(result(0))
    _, _, notes = await run_loop(importing, Scripted(edit(), RUN, FINISHED, DONE), sandbox, miner_surface_check=False)
    assert notes.surface == "" and len(sandbox.requests) == 1
    request = documented_task()
    go = request.model_copy(update={"identity": request.identity.model_copy(update={"primary_language": "go"})})
    sandbox = ScriptedSandbox(result(0))
    _, _, notes = await run_loop(importing, Scripted(edit(), RUN, FINISHED, DONE), sandbox, request=go)
    assert notes.surface == "" and len(sandbox.requests) == 1
    (importing / "docs" / "widgets.md").write_text("# Widgets\n\nNo import lines.\n")  # the same workspace
    sandbox = ScriptedSandbox(result(0))
    _, _, notes = await run_loop(importing, Scripted(edit(), RUN, FINISHED, DONE), sandbox)
    assert notes.surface == "" and len(sandbox.requests) == 1


async def test_a_check_that_cannot_run_is_not_held_against_the_answer(importing, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", no_suite)
    sandbox = ScriptedSandbox(result(0), result(127, b"python3: not found\n"))
    provider = Scripted(edit(), RUN, FINISHED, DONE)
    _, submission, notes = await run_loop(importing, provider, sandbox)
    assert b"gamma" in submission and notes.surface == "could not run" and notes.surface_failures == 0
    assert provider.seen[3][-1]["content"].startswith("Before this is submitted")


def test_a_compiled_extension_the_repository_never_had_is_not_held_against_the_answer(tmp_path):
    # 7efd1df0229c: msgspec/__init__.py imports the compiled msgspec._core, which the image never builds. The
    # repository as supplied fails the same way, so the answer is not told to make it import; a missing module
    # the answer's own files import still is.
    root = tmp_path / "work"
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "src" / "pkg" / "__init__.py").write_text("from ._core import Struct\n")
    (root / "src" / "pkg" / "catalog.py").write_text("Catalog = 1\n")
    _proc, failed = run_check([("pkg.catalog", ("Catalog",))], root, answer=["src/pkg/catalog.py"])
    assert failed == []
    (root / "src" / "pkg" / "catalog.py").write_text("from ._helper import x\nCatalog = 1\n")
    (root / "src" / "pkg" / "__init__.py").write_text("")
    _proc, failed = run_check([("pkg.catalog", ("Catalog",))], root, answer=["src/pkg/catalog.py"])
    assert failed and "pkg._helper" in failed[0]


def test_an_unchanged_original_import_line_in_an_edited_file_is_not_held_against_the_answer(tmp_path):
    # Review of 2026-11-13: the answer adds a re-export to src/pkg/__init__.py, whose first line still imports the
    # compiled ._core the image never built; that line fails exactly as in the repository as supplied.
    root = tmp_path / "work"
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "src" / "pkg" / "__init__.py").write_text("from ._core import Struct\nfrom .catalog import Catalog\n")
    (root / "src" / "pkg" / "catalog.py").write_text("Catalog = 1\n")
    answer = ["src/pkg/__init__.py", "src/pkg/catalog.py"]
    pristine = {"src/pkg/__init__.py": ["from ._core import Struct"]}
    proc = subprocess.run(
        [sys.executable, "-I", "-"], input=surface.check_script([("pkg.catalog", ("Catalog",))], str(root), answer, (), pristine),
        text=True, capture_output=True,
    )
    assert surface.failures(proc.stdout) == []
    # without that knowledge the same failure is the answer's, as before
    _proc, failed = run_check([("pkg.catalog", ("Catalog",))], root, answer=answer)
    assert failed and "pkg._core" in failed[0]
