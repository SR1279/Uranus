"""Import parity of the existing Python modules an answer edits (imports.py)."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from rlvr.neurons import imports
from rlvr.neurons.demo_miner import DemoMiner
from tests.test_acceptance import RUN, ScriptedSandbox, documented_task, result, run_loop
from tests.test_demo_miner_workbench import DONE, FINISHED, Scripted, edit, workspace  # noqa: F401


def test_a_module_is_named_by_its_package_chain():
    packages = {"src/msgspec", "api/python_api/kit", "mpmath", "mpmath/matrices"}
    is_package = packages.__contains__
    assert imports.module_name("src/msgspec/json.py", is_package) == ("msgspec.json", "src")
    assert imports.module_name("src/msgspec/__init__.py", is_package) == ("msgspec", "src")
    # Not api.python_api.kit: a package imports itself by its own name (a1b8ebf33ba5).
    assert imports.module_name("api/python_api/kit/__init__.py", is_package) == ("kit", "api/python_api")
    assert imports.module_name("mpmath/matrices/mmio.py", is_package) == ("mpmath.matrices.mmio", "")
    assert imports.module_name("tools/build.py", is_package) == ("build", "tools")
    assert imports.module_name("docs/x.md", is_package) is None
    assert imports.module_name("my-tool/x-y.py", is_package) is None


def test_the_roots_come_from_the_tasks_pythonpath_and_working_directory():
    assert imports.python_roots("Run project code with PYTHONPATH=/work:/work/src; offline.", ".") == ["src", ""]
    assert imports.python_roots("no rule stated", "client") == ["client", "src", ""]


def test_nothing_is_compared_without_an_edited_existing_module_or_when_the_originals_do_not_fit():
    assert imports.check_command(["pkg/new.py"], {}, [""]) is None
    assert imports.check_command(["README.md"], {"README.md": b"x"}, [""]) is None
    noise = os.urandom(200_000)  # does not compress
    assert imports.check_command(["big.py"], {"big.py": noise}, [""]) is None


def test_the_output_is_read_and_an_unfinished_check_is_not_a_verdict():
    output = 'IMPORT-PARITY {"module": "m", "before": "ModuleNotFoundError (m._core)", "after": "imports"}\nIMPORT-PARITY-DONE\n'
    assert imports.differences(output) == [imports.Difference("m", "ModuleNotFoundError (m._core)", "imports")]
    assert imports.differences("IMPORT-PARITY-DONE\n") == []
    assert imports.differences("Traceback ...") is None
    text = imports.prompt(imports.differences(output))
    assert "m: as the repository supplied it: ModuleNotFoundError (m._core); with your change: imports" in text
    assert "sys.modules" in text and "never in the project's files" in text


def test_the_venv_interpreter_is_used_when_asked_and_present(tmp_path):
    originals = {"pkg/mod.py": b"VALUE = 1\n"}
    command = imports.check_command(["pkg/mod.py"], originals, [""], ".", lambda directory: True, "/opt/rlvr-venv/bin/python")
    assert command.startswith('cd /work && "$([ -x /opt/rlvr-venv/bin/python ] && echo /opt/rlvr-venv/bin/python || echo python3)" - <<')
    assert imports.check_command(["pkg/mod.py"], originals, [""], ".", lambda directory: True).startswith("cd /work && python3 - <<")
    tree = tmp_path / "work"
    (tree / "pkg").mkdir(parents=True)
    (tree / "pkg" / "__init__.py").write_text("")
    (tree / "pkg" / "mod.py").write_text("VALUE = 2\n")
    for python in (sys.executable, "/nonexistent/bin/python"):
        command = imports.check_command(["pkg/mod.py"], originals, [""], ".", lambda directory: True, python)
        completed = subprocess.run(["bash", "-c", command.replace("/work", str(tree))], capture_output=True, text=True, timeout=120)
        assert imports.differences(completed.stdout + completed.stderr) == [], completed.stdout + completed.stderr

def run_check(tree, touched, originals):
    """Run the generated sandbox script on a local tree that stands in for /work."""
    is_package = lambda directory: (tree / directory / "__init__.py").is_file()  # noqa: E731
    command = imports.check_command(touched, originals, ["src", ""], ".", is_package)
    completed = subprocess.run(["bash", "-c", command.replace("/work", str(tree))], capture_output=True, text=True, timeout=120)
    return imports.differences(completed.stdout + completed.stderr)


def test_a_module_made_importable_around_a_missing_extension_is_found(tmp_path):
    # 7efd1df0229c: msgspec/__init__.py imports the compiled ._core, absent from the image.
    original = b"from ._core import Struct\n"
    tree = tmp_path / "work"
    (tree / "src" / "pkg").mkdir(parents=True)
    (tree / "src" / "pkg" / "__init__.py").write_text(
        "try:\n    from ._core import Struct\nexcept ModuleNotFoundError:\n    Struct = None\n"
    )
    (tree / "src" / "pkg" / "catalog.py").write_text("class Catalog: pass\n")
    found = run_check(tree, ["src/pkg/__init__.py", "src/pkg/catalog.py"], {"src/pkg/__init__.py": original})
    assert found == [imports.Difference("pkg", "ModuleNotFoundError (pkg._core)", "imports")]


def test_an_import_of_a_module_the_answer_adds_and_an_unchanged_failure_are_not_held_against_it(tmp_path):
    tree = tmp_path / "work"
    (tree / "pkg").mkdir(parents=True)
    # Before: the package re-exports a feature module the task removed; the answer adds it back.
    (tree / "pkg" / "__init__.py").write_text("from .feature import Widget\nVERSION = 2\n")
    (tree / "pkg" / "feature.py").write_text("class Widget: pass\n")
    # A module that failed before and fails the same way after is no difference.
    (tree / "pkg" / "fast.py").write_text("from ._speedups import run\nX = 1\n")
    found = run_check(
        tree,
        ["pkg/__init__.py", "pkg/feature.py", "pkg/fast.py"],
        {"pkg/__init__.py": b"from .feature import Widget\n", "pkg/fast.py": b"from ._speedups import run\n"},
    )
    assert found == []


def test_a_module_the_answer_breaks_is_found(tmp_path):
    tree = tmp_path / "work"
    (tree / "pkg").mkdir(parents=True)
    (tree / "pkg" / "__init__.py").write_text("")
    (tree / "pkg" / "util.py").write_text("import json\nfrom .missing import thing\n")
    found = run_check(tree, ["pkg/util.py"], {"pkg/util.py": b"import json\n"})
    assert found == [imports.Difference("pkg.util", "imports", "ModuleNotFoundError (pkg.missing)")]


async def no_suite(self, request, docs, deadline):
    return None, None


async def test_a_difference_goes_back_once_and_the_solve_continues(workspace, monkeypatch):  # noqa: F811
    monkeypatch.setattr(DemoMiner, "_author_suite", no_suite)
    report = (
        b'{"exit_code": 0}\nIMPORT-PARITY {"module": "b", "before": "ModuleNotFoundError (b._core)", '
        b'"after": "imports"}\nIMPORT-PARITY-DONE\n'
    )
    sandbox = ScriptedSandbox(result(0), result(0, stdout=report), result(0))
    provider = Scripted(
        edit("src/b.py", "return 1", "return 2"), RUN, FINISHED,
        edit("src/b.py", "return 2", "return 3"), RUN, FINISHED,
        DONE,
    )
    _, submission, notes = await run_loop(workspace, provider, sandbox, request=documented_task(), miner_import_check=True)
    sent_back = provider.seen[3][-1]["content"]
    assert sent_back.startswith("A check imported each existing module your change edits")
    assert "b: as the repository supplied it: ModuleNotFoundError (b._core); with your change: imports" in sent_back
    assert notes.imports == "1 differ"
    assert b"return 3" in submission
    assert provider.seen[6][-1]["content"].startswith("Before this is submitted")
    # The check ran in the sandbox as a harness command.
    assert any("IMPORT-PARITY-DONE" in " ".join(request.argv) for request in sandbox.requests)


async def test_nothing_is_checked_with_the_setting_off(workspace, monkeypatch):  # noqa: F811
    monkeypatch.setattr(DemoMiner, "_author_suite", no_suite)
    sandbox = ScriptedSandbox(result(0))
    provider = Scripted(edit("src/b.py", "return 1", "return 2"), RUN, FINISHED, DONE)
    _, _, notes = await run_loop(workspace, provider, sandbox, request=documented_task(), miner_import_check=False)
    assert notes.imports == "" and len(sandbox.requests) == 1


@pytest.mark.parametrize("instruction", ["Run project code with PYTHONPATH=/work:/work/src.", ""])
def test_the_roots_always_include_the_repository_root(instruction):
    assert "" in imports.python_roots(instruction, ".")


async def test_a_check_on_every_finished_version_records_unique_tool_call_ids(workspace, monkeypatch):  # noqa: F811
    # 1f8cc1857dc0 (2026-11-12): the check found nothing on two finished versions, both runs were recorded under
    # one call ID, and the trajectory, which refuses duplicate IDs, cost the answer.
    monkeypatch.setattr(DemoMiner, "_author_suite", no_suite)
    same = result(0, stdout=b'{"exit_code": 0}\nIMPORT-PARITY-DONE\n')
    sandbox = ScriptedSandbox(result(0), same, result(0), same)
    provider = Scripted(
        edit("src/b.py", "return 1", "return 2"), RUN, FINISHED,
        edit("src/b.py", "return 2", "return 3"), RUN, FINISHED,
        DONE,
    )
    events, _, notes = await run_loop(workspace, provider, sandbox, request=documented_task(), miner_import_check=True)
    ids = [event["call_id"] for event in events if event.get("event_type") == "tool_call"]
    assert len(ids) == len(set(ids))
    assert notes.import_checks >= 2 and notes.imports == "same"
