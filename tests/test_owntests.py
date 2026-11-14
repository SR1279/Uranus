"""The own-test regression net: the model's earlier test commands, run again on the final version (owntests.py)."""

from __future__ import annotations

import time

import pytest
from pydantic import ValidationError

from rlvr.neurons import owntests
from rlvr.neurons.acceptance import Settling, Suite
from rlvr.neurons.demo_miner import DemoMiner, SolveNotes
from rlvr.neurons.owntests import OwnRun, OwnRuns, markers, observe, qualifies
from rlvr.neurons.workbench import Workbench
from rlvr.v3.miner_workspace import WorkspaceReader
from tests.test_acceptance import documented, fixed_author, result, run_loop, suite_ran  # noqa: F401
from tests.test_demo_miner import WalletLike, completion_with_calls, settings
from tests.test_demo_miner_workbench import DONE, FINISHED, Clock, FakeSandbox, Scripted, edit, workspace  # noqa: F401


def ran(output="", exit_code=0, **fields):
    return {"exit_code": exit_code, "timed_out": False, "output": output, **fields}


# --- which commands count -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "python3 /tmp/t.py",
        "cd /work && PYTHONPATH=/work:/work/src python3 -m unittest -v tests.test_x",
        "cd /tmp && cat > t.py <<'EOF'\nprint(1)\nEOF\nPYTHONPATH=/work python3 t.py",
        "cd /work; timeout 100 python3 -m pytest -q -x 2>&1 | tail -5",
        "/opt/rlvr-venv/bin/python -c 'import pkg'",
        "export GOFLAGS=-mod=vendor && cp /tmp/zz2.go zz_scen_test.go; go test -run ZZScen2 -v . | grep ok; rm zz_scen_test.go",
        "cd service/v1 && GOFLAGS=-mod=vendor GOCACHE=/tmp/gocache go test ./... -count=1",
        "node --test test/logic.test.mjs 2>&1 | head -30",
        "cd /tmp/w && sed -i 's/a/b/' t.mjs && node t.mjs",
        "cd /work; export CARGO_HOME=/tmp/c; cargo test --offline --test lane 2>&1 | tail",
        "/tmp/rec_lua /tmp/rec_test2.lua 2>&1 | cut -c1-200",
        "chmod +x /tmp/run.sh && bash /tmp/run.sh",
        "for f in a b; do python3 /tmp/t.py $f; done",
        "for f in a b;do python3 /tmp/t.py $f; done",
        "pwd; ls; PYTHONPATH=$(pwd):$(pwd)/src python3 /tmp/t/t.py",  # 5a2ddda8d61c
        "PYTHONPATH=${PYTHONPATH}:/work ${VENV}/bin/python -m unittest -v",
        "{ python3 /tmp/t.py; } 2>&1 | tail -3",
        "echo result: $(python3 /tmp/t.py)",
    ],
)
def test_a_test_program_at_a_command_position_qualifies(command):
    assert owntests.runs_tests(command) and qualifies(command, ran("ok\n"))


@pytest.mark.parametrize(
    "command",
    [
        "cd /work; ls; grep -rn python3 docs | head",
        "export GOFLAGS=-mod=vendor && go vet . && go build ./... && gofmt -l *.go; echo done",
        "cat > /tmp/t.py <<'EOF'\nimport subprocess\nnode = 1\npython3 = 2\n go test\nEOF",
        "cat > /tmp/t.sh <<-EOF\n\tpython3 x\n\tEOF\necho written",
        "sed -n 1,40p /tmp/t.py; cat /tmp/.runner-output",
        "ls node_modules | head",
        "gcc -o /tmp/t t.c && echo built",
        "echo x=do python3 is the interpreter",  # a keyword inside a word is no command position
        "echo ${python3}",
    ],
)
def test_reading_building_and_a_heredoc_body_are_not_test_runs(command):
    assert not owntests.runs_tests(command)


def test_only_a_finished_run_that_changed_no_answer_file_qualifies():
    command = "python3 /tmp/t.py"
    assert qualifies(command, ran("ok\n", exit_code=1))  # a failing run is compared later like any other
    assert qualifies(command, ran("ok\n", created_not_in_answer=["zz_test.go"]))
    for changed in ("answer_updated", "not_in_answer", "answer_kept"):
        assert not qualifies(command, ran("ok\n", **{changed: ["src/a.py"]}))
    assert not qualifies(command, ran("partial", timed_out=True))
    assert not qualifies(command, ran("partial", exit_code=137, memory_limit_exceeded=True))
    assert not qualifies(command, {"error": "there is not enough time left to run this"})
    assert not qualifies(None, ran()) and not qualifies("  ", ran()) and not qualifies(command, "ok")


def test_the_harness_suite_is_never_one_of_the_models_own_tests():
    python_suite = "/tmp/acceptance_from_docs.py"
    run = "cd /tmp && PYTHONPATH=/work:/work/src python3 -m unittest -v acceptance_from_docs"
    assert not qualifies(run, ran("OK\n"), python_suite)
    assert not qualifies("python3 /tmp/acceptance_from_docs.py", ran(), python_suite)
    assert not qualifies("cd /tmp && python3 -m unittest acceptance_from_docs 2>&1 | tail -3", ran(), python_suite)
    assert qualifies("python3 /tmp/t.py", ran(), python_suite)
    go_suite = "/work/rfc5424/outbox_acceptance_test.go"
    assert not qualifies("cd /work && go test ./rfc5424/ -run TestAcceptance -count=1 -v", ran(), go_suite)
    assert not qualifies("go test -run 'TestAcceptanceLimits' ./rfc5424/", ran(), go_suite)
    assert not qualifies("sed -i s/x/y/ rfc5424/outbox_acceptance_test.go && go test ./rfc5424/", ran(), go_suite)
    assert qualifies("cd /work && go test ./rfc5424/ -run TestZZ -count=1", ran(), go_suite)
    # the suite's own run command, word for word, whatever its file is called
    suite_run = "cd /work && cargo test --offline --test t"
    assert not qualifies("cd /work &&  cargo test --offline --test t 2>&1 | tail", ran(), "/work/tests/t.rs", suite_run=suite_run)
    # a short suite file name is not looked for in other commands
    assert qualifies("python3 /tmp/t2.py", ran(), "/tmp/t.py")


# --- what counts as broken ------------------------------------------------------------------------------------

PY_CRASH = 'Traceback (most recent call last):\n  File "/tmp/t.py", line 3, in <module>\n    f()\nValueError: boom\n'
PY_ASSERT = (
    "FAIL: test_limit (t.T.test_limit)\n----\nTraceback (most recent call last):\n  File \"/tmp/t.py\", line 9, in "
    "test_limit\n    self.assertEqual(1, 2)\nAssertionError: 1 != 2\n\nFAILED (failures=1)\n"
)
GO_PANIC = (
    "--- FAIL: TestZZScen2 (0.00s)\npanic: runtime error: invalid memory address or nil pointer dereference [recovered]\n"
    "goroutine 7 [running]:\ntesting.tRunner.func1.2({0x8a3a40, 0xc8e1c0})\n\t/usr/lib/go-1.22/src/testing/testing.go:1631 +0x24a\n"
    "FAIL\tgithub.com/ozontech/cute\t0.006s\n"
)
NODE_UNCAUGHT = (
    "file:///work/src/p.js:55\n      throw new TypeError(\n            ^\n\nTypeError: Value must be a string\n"
    "    at #validate (file:///work/src/p.js:55:13)\n    at file:///tmp/t.mjs:11:101\n\nNode.js v18.19.1\n"
)


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        (PY_CRASH, {"Traceback"}),
        (PY_ASSERT, set()),
        ('  File "/tmp/t.py", line 2\n    x = (\n        ^\nSyntaxError: \'(\' was never closed\n', {"SyntaxError"}),
        ("Traceback (most recent call last):\n  File \"/tmp/t.py\", line 1, in <module>\n    import pkg.mod\n"
         "ModuleNotFoundError: No module named 'pkg.mod'\n", {"Traceback", "ModuleNotFoundError"}),
        ("ImportError while importing test module '/work/tests/test_a.py'.\nE   ModuleNotFoundError: No module named 'x'\n",
         {"ImportError", "ModuleNotFoundError"}),
        (GO_PANIC, {"panic", "goroutine trace"}),
        ("# github.com/x/y [github.com/x/y.test]\n./a_test.go:10:2: undefined: Foo\nFAIL\tgithub.com/x/y [build failed]\n",
         {"Go build error", "build failed"}),
        # an explicit go vet's warning is not a build error
        ("# github.com/kakuilan/kgo\n# [github.com/kakuilan/kgo]\n./function.go:945:14: possible misuse of reflect.SliceHeader\n"
         "ok  \tgithub.com/kakuilan/kgo\t0.009s\n", set()),
        (NODE_UNCAUGHT, {"uncaught TypeError"}),
        ("TypeError: Value must be a string\nrejected as documented\n", set()),  # a probe that prints what it caught
        ("error[E0308]: mismatched types\nerror: could not compile `lane` (lib test) due to 1 previous error\n",
         {"Rust compile error", "could not compile"}),
        ("thread 'tests::t' panicked at src/lib.rs:10:5:\nassertion `left == right` failed\n  left: 1\n right: 2\n", set()),
        ("thread 'main' panicked at src/main.rs:4:22:\ncalled `Result::unwrap()` on an `Err` value: Parse\n", {"panicked"}),
        ("ok  \tgithub.com/x/y\t0.01s\n", set()),
    ],
)
def test_markers_are_unambiguous_and_an_assertion_is_not_a_crash(output, expected):
    assert markers(output) == expected


# Node 20 output, from `node` on this host. console.log(e) of a caught error prints its name and stack exactly as an
# uncaught one does; only an uncaught one has the caret under its source line and the `Node.js vN` line after.
NODE_CAUGHT = (
    "SyntaxError: Expected property name or '}' in JSON at position 1\n    at JSON.parse (<anonymous>)\n"
    "    at file:///tmp/t.mjs:1:12\n    at ModuleJob.run (node:internal/modules/esm/module_job:325:25)\n"
    "TypeError: Value must be a string\n    at #validate (file:///work/src/p.js:3:59)\n    at file:///tmp/t.mjs:4:15\n"
    "done\n"
)


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        (NODE_CAUGHT, set()),
        # tail cut the source line (d9cc80a4abd4 workspace-9), but the trailer is there
        ("TypeError: `throws` validation object must not be empty\n    at Test.throws (/work/node_modules/tape/lib/test.js:937:12)\n"
         "    at process.processImmediate (node:internal/timers:476:21)\n\nNode.js v18.19.1\n", {"uncaught TypeError"}),
        # the trailer comes after the dump of the error's properties
        ("TypeError [ERR_INVALID_ARG_TYPE]: The \"path\" argument must be of type string. Received undefined\n"
         "    at new NodeError (node:internal/errors:405:5)\n    at validateString (node:internal/validators:162:11) {\n"
         "  code: 'ERR_INVALID_ARG_TYPE'\n}\n\nNode.js v18.19.1\n", {"uncaught TypeError"}),
        # a compile error carries its caret even where the trailer is cut (653cee6dd16e workspace-19)
        ("/work/dist/module/event.js:2\nimport { EventOptions } from \"./type.js\";\n^^^^^^\n\nSyntaxError: Cannot use import "
         "statement outside a module\n    at internalCompileFunction (node:internal/vm:73:18)\n", {"uncaught SyntaxError"}),
        ("[eval]:1\nundefinedName()\n^\n\nReferenceError: undefinedName is not defined\n    at [eval]:1:1\n",
         {"uncaught ReferenceError"}),
        # Python's own SyntaxError has no JavaScript frame after it
        ('  File "/tmp/t.py", line 2\n    x = (\n        ^\nSyntaxError: \'(\' was never closed\n', {"SyntaxError"}),
    ],
)
def test_a_node_error_counts_only_when_uncaught(output, expected):
    assert markers(output) == expected


def test_a_negative_node_probe_that_now_catches_and_prints_an_error_is_not_broken():
    # A right fix makes an old probe's call throw; the probe catches it and prints it with console.log(e).
    runs = OwnRuns()
    runs.record("node /tmp/neg.mjs", 1, 1.0, ran("accepted\ndone\n"))
    rerun = owntests.compare(runs.entries["node /tmp/neg.mjs"], ran(NODE_CAUGHT))
    assert rerun.broken is None
    rerun = owntests.compare(runs.entries["node /tmp/neg.mjs"], ran(NODE_UNCAUGHT, exit_code=1))
    assert rerun.broken == "uncaught TypeError"


def test_parsing_stays_fast_on_the_largest_commands_and_outputs():
    # Recording runs on the event loop after every command. Before 2026-11-13's review a 64 KiB command of `a=b`
    # lines took about a minute here, and 16 KiB of blank-looking output lines 1.5 s.
    began = time.monotonic()
    for command in (
        "a=b\n" * 16_000, "a=b;" * 16_000, "a=b{" * 16_000, "(" * 60_000, "; " + "A=B " * 16_000, "a=do " * 13_000,
        ")" + "$(x)" * 16_000, "{" + "a=${x} " * 9_000,
    ):
        owntests.runs_tests(command)
    for output in ("\n \n" * 15_000, "TypeError: x\n" + "        at f (file:///a.js:1:1)\n" * 1_500):
        observe(ran(output))
    assert time.monotonic() - began < 2.0


def test_a_test_runner_verdict_is_read_from_each_runner():
    assert observe(ran("ok  \tgithub.com/x/y\t0.01s\n")).passed
    assert observe(ran("Ran 3 tests in 0.001s\n\nOK\n")).passed
    assert observe(ran("test result: ok. 4 passed; 0 failed; 0 ignored\n")).passed
    assert observe(ran("# tests 3\n# pass 3\n# fail 0\n")).passed
    assert observe(ran(PY_ASSERT)).failed and not observe(ran(PY_ASSERT)).passed
    assert observe(ran(GO_PANIC)).failed
    assert observe(ran("test result: FAILED. 4 passed; 1 failed\n")).failed
    assert observe(ran("1 failed, 2 passed in 0.1s\n")).failed
    assert observe(ran("thread 't' panicked at src/lib.rs:1:1:\nassertion failed: x\n")).failed
    seen = observe(ran("x", exit_code=124, timed_out=True))
    assert seen.cut_short and seen.exit_code == 124


# Recorded outputs of 966efbb1635f (2026-11-13): its throwaway test /tmp/zz2.go printed ok at workspace-18 and,
# after the suite moved it to per-key lanes, a goroutine frame at workspace-27, under the model's own grep.
ZZ2_BEFORE = (
    "    zz_scen_test.go:20: c ok <nil> true\n    zz_scen_test.go:21: d ok <nil>\n"
    "    zz_scen_test.go:42: 100000 <nil> 957.883618ms\n    zz_scen_test.go:46: 243.360603ms\n"
    "ok  \tgithub.com/ozontech/cute\t1.380s\nok  \tgithub.com/ozontech/cute\t0.008s\n"
)
ZZ2_AFTER = (
    "    zz_scen_test.go:15: 4 <nil>\n    zz_scen_test.go:19: blocked? false\n    zz_scen_test.go:20: c ok <nil> true\n"
    "    zz_scen_test.go:24: <nil>\n--- FAIL: TestZZScen2 (0.00s)\n\t/work/zz_scen_test.go:25 +0x7e1\n"
    "FAIL\tgithub.com/ozontech/cute\t0.006s\nFAIL\n"
)
ZZ2 = (
    "export GOFLAGS=-mod=vendor GOTOOLCHAIN=local GOPROXY=off GOCACHE=/tmp/gocache && cp /tmp/zz2.go zz_scen_test.go; "
    "go test -run ZZScen2 -v . 2>&1 | grep -E \"100000|ms|ok|FAIL\" ; rm zz_scen_test.go; go test . 2>&1 | tail -3"
)


def test_the_recorded_lane_reversal_shows_as_newly_broken():
    runs = OwnRuns()
    assert runs.record(ZZ2, 10, 3.2, ran(ZZ2_BEFORE), suite_path="/work/scenario_tape_acceptance_test.go")
    rerun = owntests.compare(runs.entries[ZZ2], ran(ZZ2_AFTER))
    assert rerun.broken == "goroutine trace" and rerun.unclear is None
    assert owntests.summary([rerun]).startswith("1 rerun, 1 newly broken (goroutine trace): export GOFLAGS=-mod=vendor")


def test_a_marker_is_new_only_when_an_earlier_run_of_the_same_command_lacked_it():
    probe = "python3 /tmp/neg.py"
    runs = OwnRuns()
    runs.record(probe, 1, 1.0, ran(PY_CRASH, exit_code=1))  # a negative probe that prints the error it expects
    assert owntests.compare(runs.entries[probe], ran(PY_CRASH, exit_code=1)).broken is None
    runs.record(probe, 2, 1.0, ran("raised as documented\n"))  # one run without the traceback is enough
    assert runs.entries[probe].markers == frozenset() and runs.entries[probe].runs == 2
    assert owntests.compare(runs.entries[probe], ran(PY_CRASH, exit_code=1)).broken == "Traceback"


def test_fail_after_ok_and_a_changed_exit_code_are_ambiguous_not_broken():
    runs = OwnRuns()
    runs.record("go test ./lane/", 1, 2.0, ran("ok  \tx/lane\t0.01s\n"))
    rerun = owntests.compare(runs.entries["go test ./lane/"], ran("--- FAIL: TestLane (0.00s)\nFAIL\tx/lane\t0.01s\n", 1))
    assert rerun.broken is None and rerun.unclear == "FAIL after ok"
    runs.record("python3 /tmp/p.py", 1, 1.0, ran("4\n"))
    rerun = owntests.compare(runs.entries["python3 /tmp/p.py"], ran("5\n", exit_code=2))
    assert rerun.broken is None and rerun.unclear == "exit 2 after 0"
    assert owntests.compare(runs.entries["python3 /tmp/p.py"], ran("4\n")).unclear is None


def test_a_rerun_cut_short_or_not_run_decides_nothing():
    entry = OwnRun("python3 /tmp/t.py", 1, 1.0, frozenset(), True, True)
    cut = owntests.compare(entry, ran(PY_CRASH, exit_code=124, timed_out=True))
    assert cut.broken is None and cut.unclear is None and cut.cut_short
    missing = owntests.compare(entry, {"error": "run_command could not complete"})
    assert not missing.ran and missing.broken is None
    assert owntests.summary([cut, missing]) == "1 rerun, none newly broken, 1 cut short, 1 could not run"


def test_the_last_six_distinct_commands_are_kept_and_the_three_most_recent_older_ones_are_due():
    runs = OwnRuns()
    for index in range(8):
        runs.record(f"python3 /tmp/t{index}.py", index, 1.0, ran())
    assert list(runs.entries) == [f"python3 /tmp/t{index}.py" for index in range(2, 8)]
    assert not runs.record("cat /tmp/t1.py", 8, 1.0, ran())  # not a test run: nothing changes
    runs.record("python3 /tmp/t3.py", 6, 1.0, ran())  # run again: now among the most recent
    assert list(runs.entries)[-1] == "python3 /tmp/t3.py" and runs.entries["python3 /tmp/t3.py"].runs == 2
    # version 7 is the final one: what already ran on it is not run again; the rest go oldest first
    assert [entry.command for entry in runs.due(7)] == ["python3 /tmp/t5.py", "python3 /tmp/t6.py", "python3 /tmp/t3.py"]
    excluded = runs.due(7, exclude=lambda command: "t6" in command)
    assert [entry.command for entry in excluded] == ["python3 /tmp/t4.py", "python3 /tmp/t5.py", "python3 /tmp/t3.py"]
    assert runs.due(2) == [] and runs.due(7, limit=0) == []


def test_each_rerun_gets_twice_its_original_time_and_at_most_30_seconds():
    assert [owntests.limit_s(seconds) for seconds in (0.0, 0.4, 3.2, 14.9, 200.0)] == [1, 1, 7, 30, 30]


def test_the_record_names_the_first_breakage_and_counts_the_rest():
    entry = OwnRun("python3 /tmp/t.py " + "x" * 200, 1, 1.0, frozenset(), True, True)
    other = OwnRun("go test ./a/", 1, 1.0, frozenset(), True, True)
    reruns = [owntests.compare(entry, ran(PY_CRASH, 1)), owntests.compare(other, ran("--- FAIL: TestA\nFAIL\ta\n", 1))]
    text = owntests.summary(reruns, skipped=1)
    assert text.startswith("2 rerun, 1 newly broken (Traceback): python3 /tmp/t.py xxx")
    assert "..., 1 ambiguous (FAIL after ok): go test ./a/, 1 left out for time" in text
    assert owntests.summary([owntests.compare(other, ran("ok  \ta\t0.1s\n"))]) == "1 rerun, none newly broken"


def test_the_question_of_ask_mode_quotes_the_command_and_the_end_of_its_new_output():
    entry = OwnRun(ZZ2, 10, 3.0, frozenset(), True, True)
    text = owntests.prompt(owntests.compare(entry, ran(ZZ2_AFTER)))
    assert text.startswith("Your earlier command `export GOFLAGS=-mod=vendor")
    assert "ran cleanly on an earlier version; on the final version it now ends in goroutine trace:" in text
    assert "/work/zz_scen_test.go:25 +0x7e1" in text
    assert "where a passage supports a deliberate change, keep it and correct the command, otherwise fix the code" in text
    assert text.endswith("run it again and reply with a one-line summary.")


def test_the_setting_takes_only_its_three_modes():
    assert settings().miner_own_tests == "record"
    assert settings(miner_own_tests="ask").miner_own_tests == "ask"
    with pytest.raises(ValidationError):
        settings(miner_own_tests="always")


# --- in the solve loop ----------------------------------------------------------------------------------------

TEST_RUN = completion_with_calls(("run_command", {"command": "python3 /tmp/t.py", "timeout_s": 30}))
TRUE_RUN = completion_with_calls(("run_command", {"command": "true", "timeout_s": 10}))


class CommandSandbox(FakeSandbox):
    """Answers by command: the harness suite passes, the model's test prints `first`, and a rerun (the only command
    run without the change report) prints `rerun`, taking `rerun_s` on the clock when one is given."""

    def __init__(self, first=b"ok\n", rerun=PY_CRASH.encode(), rerun_exit=1, clock=None, rerun_s=0.0):
        super().__init__()
        self.first, self.rerun, self.rerun_exit = first, rerun, rerun_exit
        self.clock, self.rerun_s = clock, rerun_s
        self.commands: list[tuple[str, bool]] = []

    async def run(self, request, *, wait_s):
        self.requests.append(request)
        command, watched = request.argv[6], request.argv[-1] == "watch"
        self.commands.append((command, watched))
        if command.startswith("sha256sum --"):
            return suite_ran(0, b"OK\n")
        if not watched:
            if self.clock is not None:
                self.clock.now += self.rerun_s
            return result(self.rerun_exit, self.rerun)
        return result(0, self.first) if command.startswith("python3 /tmp/t") else result(0)


def reruns_of(sandbox):
    return [command for command, watched in sandbox.commands if not watched]


async def test_record_mode_reruns_the_earlier_test_on_the_final_version_and_sends_nothing(workspace):
    sandbox = CommandSandbox()
    provider = Scripted(edit(), TEST_RUN, edit("src/b.py", "return 1", "return 2"), TRUE_RUN, FINISHED, DONE)
    events, submission, notes = await run_loop(workspace, provider, sandbox)
    assert notes.own_tests == "1 rerun, 1 newly broken (Traceback): python3 /tmp/t.py"
    assert len(provider.seen) == 6 and not provider.replies  # no message about it
    assert reruns_of(sandbox) == ["python3 /tmp/t.py"] and sandbox.commands[-1] == ("python3 /tmp/t.py", False)
    # recorded in the trajectory like the harness's other runs, and nothing it did is in the answer
    calls = [event for event in events if event["event_type"] == "tool_call"]
    assert calls[-1]["call_id"] == "own-test-1"
    assert b"gamma" in submission and b"return 2" in submission
    assert notes.fields()["own_tests"] == notes.own_tests


async def test_a_command_that_already_ran_on_the_final_version_is_not_rerun(workspace):
    sandbox = CommandSandbox()
    provider = Scripted(edit(), TEST_RUN, FINISHED, DONE)
    _, _, notes = await run_loop(workspace, provider, sandbox)
    assert notes.own_tests == "none to rerun" and reruns_of(sandbox) == []


async def test_off_mode_records_and_reruns_nothing(workspace):
    sandbox = CommandSandbox()
    provider = Scripted(edit(), TEST_RUN, edit("src/b.py", "return 1", "return 2"), TRUE_RUN, FINISHED, DONE)
    _, _, notes = await run_loop(workspace, provider, sandbox, miner_own_tests="off")
    assert notes.own_tests == "" and reruns_of(sandbox) == []


async def test_ask_mode_sends_an_unambiguous_breakage_back_once(workspace):
    sandbox = CommandSandbox()
    provider = Scripted(
        edit(), TEST_RUN, edit("src/b.py", "return 1", "return 2"), TRUE_RUN, FINISHED, DONE,
        edit("src/b.py", "return 2", "return 3"), TEST_RUN, FINISHED,
    )
    _, submission, notes = await run_loop(workspace, provider, sandbox, miner_own_tests="ask")
    asked = provider.seen[6][-1]["content"]
    assert asked.startswith("Your earlier command `python3 /tmp/t.py` ran cleanly on an earlier version")
    assert "ValueError: boom" in asked
    assert notes.own_tests.endswith("; sent back") and b"return 3" in submission
    assert len(provider.seen) == 9 and reruns_of(sandbox) == ["python3 /tmp/t.py"]  # once per solve


async def test_ask_mode_keeps_an_ambiguous_change_to_the_record(workspace):
    sandbox = CommandSandbox(first=b"Ran 1 test\n\nOK\n", rerun=b"FAIL: test_a\n\nFAILED (failures=1)\n")
    provider = Scripted(edit(), TEST_RUN, edit("src/b.py", "return 1", "return 2"), TRUE_RUN, FINISHED, DONE)
    _, _, notes = await run_loop(workspace, provider, sandbox, miner_own_tests="ask")
    assert notes.own_tests == "1 rerun, none newly broken, 1 ambiguous (FAIL after ok): python3 /tmp/t.py"
    assert len(provider.seen) == 6


async def test_after_the_final_check_and_with_the_suite_left_out(documented, monkeypatch):
    # The suite passes; the final check runs it on a clean copy, then the model's own test is run once more.
    monkeypatch.setattr(DemoMiner, "_author_suite", fixed_author())
    sandbox = CommandSandbox()
    suite_run = "cd /tmp && PYTHONPATH=/work:/work/src python3 -m unittest -v acceptance_from_docs"
    own_suite_run = completion_with_calls(("run_command", {"command": suite_run, "timeout_s": 30}))
    provider = Scripted(
        edit(), TEST_RUN, own_suite_run, edit("src/b.py", "return 1", "return 2"), TRUE_RUN, FINISHED, DONE,
    )
    _, _, notes = await run_loop(documented, provider, sandbox)
    assert notes.clean_copy == "same" and notes.own_tests.startswith("1 rerun, 1 newly broken (Traceback)")
    ran_after = [command for command, _watched in sandbox.commands]
    assert ran_after.index("python3 /tmp/t.py", 1) > max(
        index for index, command in enumerate(ran_after) if command.startswith("sha256sum --")
    )
    assert reruns_of(sandbox) == ["python3 /tmp/t.py"]  # the model's run of the suite is not its own test


async def test_ask_mode_lets_the_final_check_run_again_after_the_question(documented, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", fixed_author())
    sandbox = CommandSandbox()
    provider = Scripted(
        edit(), TEST_RUN, edit("src/b.py", "return 1", "return 2"), TRUE_RUN, FINISHED, DONE,
        edit("src/b.py", "return 2", "return 3"), TRUE_RUN, FINISHED,
    )
    _, submission, notes = await run_loop(documented, provider, sandbox, miner_own_tests="ask")
    assert provider.seen[6][-1]["content"].startswith("Your earlier command")
    suite_runs = [command for command, _watched in sandbox.commands if command.startswith("sha256sum --")]
    # the first run and the final check's clean copy, then after the question the changed version and its clean copy
    assert len(suite_runs) == 4
    assert b"return 3" in submission


# --- the stage on its own -------------------------------------------------------------------------------------


def own_tests_stage(workspace, sandbox):
    miner = DemoMiner(settings(), Scripted(), wallet=WalletLike("miner"), sandbox=sandbox)
    bench = Workbench(WorkspaceReader(workspace), terminal=False, working_directory=".", sandbox=sandbox)
    return miner, bench


async def test_the_stage_needs_a_minute_beyond_its_reserve_and_stops_when_time_runs_short(workspace, monkeypatch):
    clock = Clock()
    monkeypatch.setattr("rlvr.neurons.demo_miner.time", clock)
    sandbox = CommandSandbox(clock=clock, rerun_s=25.0)
    miner, bench = own_tests_stage(workspace, sandbox)
    runs = OwnRuns()
    for index in range(3):
        runs.record(f"python3 /tmp/t{index}.py", 0, 20.0, ran("ok\n"))
    with bench:
        await bench.execute({"tool": "edit_file", "path": "a.txt", "old_text": "beta", "new_text": "gamma"}, time_left=60)
        notes = SolveNotes()
        assert await miner._own_tests(runs, bench, [], notes, clock.now + 85.0, None) is None
        assert notes.own_tests == "not run: too little time left" and sandbox.requests == []
        # 65 s beyond the reserve: two reruns of 30 s fit, the third would eat into the reserve
        notes = SolveNotes()
        found = await miner._own_tests(runs, bench, [], notes, clock.now + 95.0, None)
    assert found is not None and found.broken == "Traceback" and found.entry.command == "python3 /tmp/t0.py"
    assert reruns_of(sandbox) == ["python3 /tmp/t0.py", "python3 /tmp/t1.py"]
    assert notes.own_tests == "2 rerun, 2 newly broken (Traceback): python3 /tmp/t0.py, 1 left out for time"
    assert [request.argv[7] for request in sandbox.requests] == ["30", "30"]


async def test_the_stage_leaves_out_the_suite_and_resets_the_copy_around_the_reruns(workspace):
    calls = []
    sandbox = CommandSandbox()
    miner, bench = own_tests_stage(workspace, sandbox)
    runs = OwnRuns()
    suite = Suite("cd /work && go test ./ -run TestAcceptance -count=1 -v", "/work/x_acceptance_test.go", "package x\n")
    runs.record("go test ./ -run TestAcceptance", 0, 1.0, ran("ok  \tx\t0.1s\n"))  # recorded before the suite existed
    runs.record("python3 /tmp/t.py", 0, 1.0, ran("ok\n"))
    with bench:
        await bench.execute({"tool": "edit_file", "path": "a.txt", "old_text": "beta", "new_text": "gamma"}, time_left=60)
        original = bench.reset_to_answer

        def reset():
            calls.append(len(sandbox.requests))
            original()

        bench.reset_to_answer = reset
        notes = SolveNotes()
        await miner._own_tests(runs, bench, [], notes, time.monotonic() + 300.0, Settling.start(suite, "", {}))
    assert reruns_of(sandbox) == ["python3 /tmp/t.py"] and calls == [0, 1]  # reset before and after
    assert notes.own_tests.startswith("1 rerun, 1 newly broken")


async def test_the_reset_after_the_reruns_keeps_its_time_before_the_reserve(workspace, monkeypatch):
    # A large repository takes a while to copy, and the copy is reset again after the reruns: the reruns stop in
    # time for that second reset to end before the 30 s reserve.
    clock = Clock()
    monkeypatch.setattr("rlvr.neurons.demo_miner.time", clock)
    sandbox = CommandSandbox(clock=clock, rerun_s=30.0)
    miner, bench = own_tests_stage(workspace, sandbox)
    runs = OwnRuns()
    for index in range(3):
        runs.record(f"python3 /tmp/t{index}.py", 0, 20.0, ran("ok\n"))
    with bench:
        await bench.execute({"tool": "edit_file", "path": "a.txt", "old_text": "beta", "new_text": "gamma"}, time_left=60)
        original = bench.reset_to_answer

        def slow_reset():
            clock.now += 20.0
            original()

        bench.reset_to_answer = slow_reset
        notes, deadline = SolveNotes(), clock.now + 150.0
        await miner._own_tests(runs, bench, [], notes, deadline, None)
    # without the second reset's time kept free, the third rerun starts at 70 s left and the reset ends at 20 s
    assert reruns_of(sandbox) == ["python3 /tmp/t0.py", "python3 /tmp/t1.py"]
    assert notes.own_tests.endswith("1 left out for time")
    assert deadline - clock.now >= owntests.RESERVE_S


async def test_the_copy_is_reset_even_when_tmp_cannot_be_put_back(workspace):
    calls = []
    sandbox = CommandSandbox()
    miner, bench = own_tests_stage(workspace, sandbox)
    runs = OwnRuns()
    runs.record("python3 /tmp/t.py", 0, 1.0, ran("ok\n"))
    with bench:
        await bench.execute({"tool": "edit_file", "path": "a.txt", "old_text": "beta", "new_text": "gamma"}, time_left=60)
        original = bench.reset_to_answer

        def reset():
            calls.append(len(sandbox.requests))
            original()

        def broken_restore(saved):
            raise OSError("disk full")

        bench.reset_to_answer, bench.restore_tmp = reset, broken_restore
        with pytest.raises(OSError):  # the finish pipeline records it as "failed: OSError" and submits
            await miner._own_tests(runs, bench, [], SolveNotes(), time.monotonic() + 300.0, None)
    assert calls == [0, 1]


async def test_a_recording_failure_never_costs_the_answer(workspace, monkeypatch):
    def broken(self, *arguments, **options):
        raise RuntimeError("parser bug")

    monkeypatch.setattr(OwnRuns, "record", broken)
    sandbox = CommandSandbox()
    provider = Scripted(edit(), TEST_RUN, edit("src/b.py", "return 1", "return 2"), TEST_RUN, FINISHED, DONE)
    events, submission, notes = await run_loop(workspace, provider, sandbox)
    assert b"gamma" in submission and b"return 2" in submission and not provider.replies
    assert notes.own_tests == "failed: RuntimeError" and reruns_of(sandbox) == []
    # the model still got both results
    assert sum(event["call_id"].startswith("workspace-") for event in events if event["event_type"] == "tool_result") == 4


@pytest.mark.parametrize("with_sandbox", [True, False])
async def test_a_script_task_or_one_without_a_sandbox_records_and_reruns_nothing(workspace, with_sandbox):
    from rlvr.neurons.workbench import SCRIPT_PATH
    from tests.test_demo_miner_workbench import terminal_task

    write = completion_with_calls(("write_file", {"path": SCRIPT_PATH, "content": "#!/bin/bash\necho done\n"}))
    later = completion_with_calls(("write_file", {"path": SCRIPT_PATH, "content": "#!/bin/bash\necho again\n"}))
    requests = {}
    for mode in ("record", "off"):
        sandbox = FakeSandbox() if with_sandbox else None
        replies = (write, TEST_RUN, later, FINISHED, DONE) if with_sandbox else (write, later, FINISHED, DONE)
        provider = Scripted(*replies, FINISHED, DONE)
        notes = SolveNotes()
        miner = DemoMiner(settings(miner_own_tests=mode), provider, wallet=WalletLike("miner"), sandbox=sandbox)
        _, events, submission = await miner._generate(
            terminal_task(), WorkspaceReader(workspace), time.monotonic() + 900.0, notes=notes
        )
        assert submission == b"#!/bin/bash\necho again\n" and notes.own_tests == ""
        assert not any(event["call_id"].startswith("own-test") for event in events if event["event_type"] == "tool_call")
        requests[mode] = len(sandbox.requests) if sandbox is not None else 0
    assert requests["record"] == requests["off"]  # not one container more than with the net off
