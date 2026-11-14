"""The project's own parsers on the JavaScript files an answer touches (toolchain.py)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

from rlvr.neurons import toolchain
from rlvr.neurons.demo_miner import DemoMiner
from tests.test_acceptance import RUN, ScriptedSandbox, documented_task, result, run_loop
from tests.test_demo_miner_workbench import DONE, FINISHED, Scripted, edit, workspace  # noqa: F401

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")

# A stand-in for browserify's parser that, like detective 5 on acorn-node, does not know private methods.
OLD_PARSER = (
    "module.exports = function (source) {\n"
    "  var lines = source.split('\\n');\n"
    "  for (var i = 0; i < lines.length; i += 1) {\n"
    "    var at = lines[i].search(/#[A-Za-z_]\\w*\\s*\\(/);\n"
    "    if (/^\\s*(static\\s+|async\\s+)*#[A-Za-z_]\\w*\\s*\\(/.test(lines[i])) {\n"
    "      throw new SyntaxError('Unexpected token (' + (i + 1) + ':' + at + ')');\n"
    "    }\n"
    "  }\n"
    "  return [];\n"
    "};\n"
)
OLD_STYLE = "class Timers {\n  constructor () { this._events = {}; }\n  _request (event) { return event; }\n}\nmodule.exports = Timers;\n"
PRIVATE_METHOD = "class Datasets {\n  constructor () { this.count = 0; }\n  #request (event) { return event; }\n}\nmodule.exports = Datasets;\n"


def project(tmp_path, *, bundler="browserify", parser=OLD_PARSER):
    """A small project whose package.json lists `bundler`, with a parser where browserify keeps its own."""
    tree = tmp_path / "work"
    (tree / "lib").mkdir(parents=True)
    (tree / "lib" / "timers.js").write_text(OLD_STYLE)
    (tree / "package.json").write_text(json.dumps({"name": "p", "devDependencies": {bundler: "^16.5.2"} if bundler else {}}))
    for name in ("browserify", "module-deps", "detective"):
        (tree / "node_modules" / name).mkdir(parents=True)
        (tree / "node_modules" / name / "package.json").write_text(json.dumps({"name": name, "version": "1.0.0", "main": "index.js"}))
        (tree / "node_modules" / name / "index.js").write_text(parser if name == "detective" else "module.exports = {};\n")
    return tree


def run_check(tree, touched, originals, working_directory="."):
    command = toolchain.check_command(touched, originals, working_directory)
    assert command is not None
    done = subprocess.run(["bash", "-c", command.replace("cd /work", f"cd {tree}", 1)], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    return toolchain.rejections(done.stdout), toolchain.parsers_tried(done.stdout)


def test_only_javascript_sources_are_parsed_and_tests_are_left_out():
    touched = ["lib/a.js", "lib/b.mjs", "lib/c.cjs", "ui/d.jsx", "test/unit/a.test.js", "lib/a.spec.js", "__tests__/x.js",
               "README.md", "lib/e.py", "src/f.ts"]
    assert toolchain.source_files(touched) == ["lib/a.js", "lib/b.mjs", "lib/c.cjs", "ui/d.jsx"]
    assert toolchain.check_command(["README.md", "test/a.test.js"], {}) is None
    noise = os.urandom(toolchain.MAX_FILE_BYTES + 1)
    assert toolchain.check_command(["lib/big.js"], {"lib/big.js": noise}) is None  # no original to compare with


def test_the_output_is_read_and_an_unfinished_check_is_not_a_verdict():
    output = (
        'TOOLCHAIN-PARSE {"path": "lib/sandbox/datasets.js", "tool": "browserify (its parser, detective)", '
        '"message": "Unexpected token (54:4)"}\nTOOLCHAIN-PARSE-DONE 1\n'
    )
    found = toolchain.rejections(output)
    assert found == [toolchain.Rejection("lib/sandbox/datasets.js", "browserify (its parser, detective)", "Unexpected token (54:4)")]
    assert toolchain.parsers_tried(output) == 1 and toolchain.parsers_tried("TOOLCHAIN-PARSE-DONE 0\n") == 0
    assert toolchain.rejections("TOOLCHAIN-PARSE-DONE 2\n") == []
    assert toolchain.rejections("node: command not found") is None
    text = toolchain.prompt(found)
    assert "- lib/sandbox/datasets.js: browserify (its parser, detective) says: Unexpected token (54:4)" in text
    assert "underscore-prefixed member" in text and "keep the behaviour exactly as it is" in text


@needs_node
def test_a_new_file_the_projects_bundler_cannot_parse_is_found_when_its_neighbours_parse(tmp_path):
    tree = project(tmp_path)
    (tree / "lib" / "datasets.js").write_text(PRIVATE_METHOD)
    found, tried = run_check(tree, ["lib/datasets.js"], {})
    assert tried == 1
    assert found == [toolchain.Rejection("lib/datasets.js", "browserify (its parser, detective)", "Unexpected token (3:2)")]
    # Written in the neighbours' syntax, the same file parses.
    (tree / "lib" / "datasets.js").write_text(OLD_STYLE)
    assert run_check(tree, ["lib/datasets.js"], {}) == ([], 1)


@needs_node
def test_an_edited_file_counts_only_when_the_repositorys_own_version_parsed(tmp_path):
    tree = project(tmp_path)
    (tree / "lib" / "timers.js").write_text(PRIVATE_METHOD)
    found, _ = run_check(tree, ["lib/timers.js"], {"lib/timers.js": OLD_STYLE.encode()})
    assert [rejection.path for rejection in found] == ["lib/timers.js"]
    # The repository's own version already had the construct: this parser is not in that file's path.
    assert run_check(tree, ["lib/timers.js"], {"lib/timers.js": PRIVATE_METHOD.encode()}) == ([], 1)


@needs_node
def test_a_new_file_is_not_held_to_a_parser_its_neighbours_already_fail(tmp_path):
    tree = project(tmp_path)
    (tree / "lib" / "timers.js").write_text(PRIVATE_METHOD)  # the project's own sources use the syntax
    (tree / "lib" / "datasets.js").write_text(PRIVATE_METHOD)
    assert run_check(tree, ["lib/datasets.js"], {}) == ([], 1)


@needs_node
def test_without_a_bundler_in_package_json_nothing_is_parsed(tmp_path):
    tree = project(tmp_path, bundler="")
    (tree / "lib" / "datasets.js").write_text(PRIVATE_METHOD)
    assert run_check(tree, ["lib/datasets.js"], {}) == ([], 0)


@needs_node
def test_the_check_runs_from_the_tasks_working_directory(tmp_path):
    tree = project(tmp_path)
    nested = tmp_path / "outer"
    nested.mkdir()
    shutil.move(str(tree), str(nested / "client"))
    (nested / "client" / "lib" / "datasets.js").write_text(PRIVATE_METHOD)
    command = toolchain.check_command(["client/lib/datasets.js", "other/x.js"], {}, "client")
    assert command.startswith("cd /work/client && node - <<")
    done = subprocess.run(["bash", "-c", command.replace("cd /work/client", f"cd {nested / 'client'}", 1)], capture_output=True, text=True, timeout=60)
    assert [rejection.path for rejection in toolchain.rejections(done.stdout)] == ["lib/datasets.js"]


def javascript_task():
    request = documented_task()
    return request.model_copy(update={"identity": request.identity.model_copy(update={"primary_language": "javascript"})})


async def no_suite(self, request, docs, deadline):
    return None, None


async def test_a_rejection_goes_back_and_the_solve_continues(workspace, monkeypatch):  # noqa: F811
    monkeypatch.setattr(DemoMiner, "_author_suite", no_suite)
    (workspace / "src" / "widget.js").write_text("module.exports = function () { return 1; };\n")
    report = (
        b'TOOLCHAIN-PARSE {"path": "src/widget.js", "tool": "browserify (its parser, detective)", '
        b'"message": "Unexpected token (1:28)"}\nTOOLCHAIN-PARSE-DONE 1\n'
    )
    sandbox = ScriptedSandbox(result(0), result(0, stdout=report), result(0), result(0, stdout=b"TOOLCHAIN-PARSE-DONE 1\n"))
    provider = Scripted(
        edit("src/widget.js", "return 1", "return 2"), RUN, FINISHED,
        edit("src/widget.js", "return 2", "return 3"), RUN, FINISHED,
        DONE,
    )
    _, submission, notes = await run_loop(workspace, provider, sandbox, request=javascript_task())
    sent_back = provider.seen[3][-1]["content"]
    assert sent_back.startswith("A check parsed each JavaScript file your change adds or edits")
    assert "- src/widget.js: browserify (its parser, detective) says: Unexpected token (1:28)" in sent_back
    assert notes.toolchain == "parses" and notes.toolchain_checks == 2 and notes.fields()["toolchain"] == "parses"
    assert b"return 3" in submission
    assert provider.seen[6][-1]["content"].startswith("Before this is submitted")
    commands = [" ".join(request.argv) for request in sandbox.requests]
    assert sum("TOOLCHAIN-PARSE-DONE" in command for command in commands) == 2


async def test_a_python_task_or_the_setting_off_checks_nothing(workspace, monkeypatch):  # noqa: F811
    monkeypatch.setattr(DemoMiner, "_author_suite", no_suite)
    (workspace / "src" / "widget.js").write_text("module.exports = function () { return 1; };\n")
    for request, updates in ((documented_task(), {}), (javascript_task(), {"miner_toolchain_check": False})):
        sandbox = ScriptedSandbox(result(0))
        provider = Scripted(edit("src/widget.js", "return 1", "return 2"), RUN, FINISHED, DONE)
        _, _, notes = await run_loop(workspace, provider, sandbox, request=request, miner_import_check=False, **updates)
        assert notes.toolchain == "" and notes.toolchain_checks == 0
        assert not any("TOOLCHAIN-PARSE" in " ".join(r.argv) for r in sandbox.requests)
        (workspace / "src" / "widget.js").write_text("module.exports = function () { return 1; };\n")


async def test_no_project_parser_and_an_unfinished_check_send_nothing(workspace, monkeypatch):  # noqa: F811
    monkeypatch.setattr(DemoMiner, "_author_suite", no_suite)
    for stdout, expected in ((b"TOOLCHAIN-PARSE-DONE 0\n", "no project parser"), (b"node: not found\n", "could not run")):
        (workspace / "src" / "widget.js").write_text("module.exports = function () { return 1; };\n")
        sandbox = ScriptedSandbox(result(0), result(0, stdout=stdout))
        provider = Scripted(edit("src/widget.js", "return 1", "return 2"), RUN, FINISHED, DONE)
        _, submission, notes = await run_loop(workspace, provider, sandbox, request=javascript_task())
        assert notes.toolchain == expected and b"return 2" in submission
        assert provider.seen[3][-1]["content"].startswith("Before this is submitted")
