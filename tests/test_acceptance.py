"""The acceptance suite a second model writes from the documentation alone, and how the loop uses it."""

from __future__ import annotations

import asyncio
import json
import subprocess
import time

import pytest

from rlvr.neurons import acceptance
from rlvr.neurons.acceptance import Suite, author_messages, parse_suite, placement_command, read_docs, run_summary
from rlvr.neurons.demo_miner import DemoMiner, SolveNotes
from rlvr.v3.miner_workspace import WorkspaceReader
from rlvr.v3.supervisor import ContainerResult
from tests.test_demo_miner import WalletLike, completion_for, completion_with_calls, settings, task
from tests.test_demo_miner_workbench import DONE, FINISHED, FakeSandbox, Scripted, edit, workspace  # noqa: F401

# The model runs something after its edit, so the answer is not sent back as untested before the suite runs.
RUN = completion_with_calls(("run_command", {"command": "true", "timeout_s": 10}))

INSTRUCTION = (
    "Add a new feature to this repository: Resumable widgets.\n\nPublic interface: Widget(limit).\n\n"
    "The design and exact behavior are documented in docs/widgets.md (already in the repository). "
    "Python 3.12 (image interpreter), offline. Run project code with PYTHONPATH=/work:/work/src."
)
REPLY = (
    "RUN: cd /tmp && PYTHONPATH=/work:/work/src python3 -m unittest -v acceptance_from_docs\n"
    "PATH: /tmp/acceptance_from_docs.py\n"
    "```python\nimport unittest\n\n\nclass Rules(unittest.TestCase):\n    def test_limit_is_inclusive(self):\n"
    "        self.assertEqual(1, 1)\n\n\nif __name__ == '__main__':\n    unittest.main()\n```\n"
)


def result(exit_code=0, stdout=b"", timed_out=False):
    return ContainerResult(
        exit_code=exit_code, oom_killed=False, timed_out=timed_out, stdout=stdout, stderr=b"",
        stdout_overflow=False, stderr_overflow=False,
    )


# A real suite run prints the suite file's sha256 first (Settling.command); a run without it reads as a missing file.
SUITE_HASH = b"a" * 64 + b"\n"


def suite_ran(exit_code=0, stdout=b"OK\n", timed_out=False):
    """A suite run's result, as the harness's run command prints it."""
    return result(exit_code, SUITE_HASH + stdout, timed_out)


class ScriptedSandbox(FakeSandbox):
    """Each command gets the next result, so placing the suite and running it can differ."""

    def __init__(self, *results):
        super().__init__(stdout=SUITE_HASH + b"ok\n")
        self.results = list(results)

    async def run(self, request, *, wait_s):
        self.requests.append(request)
        return self.results.pop(0) if self.results else self.result


def documented_task(instruction=INSTRUCTION):
    request = task()
    return request.model_copy(update={"identity": request.identity.model_copy(update={"instruction": instruction})})


@pytest.fixture
def documented(workspace):
    (workspace / "docs").mkdir()
    (workspace / "docs" / "widgets.md").write_text("# Widgets\n\nA limit of 0 is invalid; 1..10 inclusive.\n")
    return workspace


async def run_loop(workspace, provider, sandbox, *, request=None, deadline=900.0, **updates):
    notes = SolveNotes()
    updates.setdefault("miner_adjudicator", False)  # these tests are about the suite; the reader has its own
    miner = DemoMiner(settings(**updates), provider, wallet=WalletLike("miner"), sandbox=sandbox)
    _, events, submission = await miner._generate(
        request or documented_task(), WorkspaceReader(workspace), time.monotonic() + deadline, notes=notes
    )
    return events, submission, notes


def test_the_documentation_is_found_from_the_task_text_and_read_safely(documented, tmp_path):
    assert acceptance.doc_paths(INSTRUCTION) == ["docs/widgets.md"]
    assert acceptance.doc_paths("documented in ../x.md (already) and documented in /etc/x.md") == []
    assert acceptance.doc_paths("documented in a.md, and also documented in a.md") == ["a.md"]
    # task 2bbb94e2828b named two files, the first not Markdown, and the suite never ran
    several = "The design is documented in CMakeLists.txt, docs/environment-plans.md (already in the repository). Implement"
    assert acceptance.doc_paths(several) == ["CMakeLists.txt", "docs/environment-plans.md"]
    docs = read_docs(INSTRUCTION, documented)
    assert list(docs) == ["docs/widgets.md"] and "1..10 inclusive" in docs["docs/widgets.md"]
    assert read_docs(INSTRUCTION.replace("docs/widgets.md", "docs/missing.md"), documented) == {}
    outside = tmp_path / "secret.md"
    outside.write_text("secret")
    (documented / "docs" / "link.md").symlink_to(outside)
    assert read_docs("documented in docs/link.md (already in the repository)", documented) == {}


def test_the_author_sees_the_documentation_and_the_task_but_never_the_code():
    messages = author_messages(INSTRUCTION, {"docs/widgets.md": "# Widgets\nrule text"}, "tools here")
    assert [message["role"] for message in messages] == ["system", "user"]
    system, user = messages[0]["content"], messages[1]["content"]
    assert "never see the implementation" in system and "one strict case per rule" in system
    assert "inputs that break two rules at once" in system and "tools here" in system
    assert "whitespace-only is empty where empty is invalid" in system
    assert "Resumable widgets" in user and "rule text" in user and "docs/widgets.md" in user


def test_the_reply_shape_is_parsed_and_unusable_replies_are_refused():
    suite = parse_suite(REPLY)
    assert suite == Suite(
        "cd /tmp && PYTHONPATH=/work:/work/src python3 -m unittest -v acceptance_from_docs",
        "/tmp/acceptance_from_docs.py",
        REPLY.split("```python\n")[1].split("```")[0],
    )
    assert suite.content.endswith("unittest.main()\n")
    assert parse_suite("NONE") is None and parse_suite("  NONE\nbecause C++") is None and parse_suite("") is None
    assert parse_suite(REPLY.replace("PATH: /tmp/acceptance_from_docs.py\n", "")) is None
    assert parse_suite(REPLY.replace("/tmp/acceptance_from_docs.py", "/etc/cron.d/x.py")) is None
    assert parse_suite(REPLY.replace("/tmp/acceptance_from_docs.py", "/work/../x.py")) is None
    assert parse_suite(REPLY.replace("import unittest", acceptance.SUITE_DELIMITER)) is None
    assert parse_suite(REPLY.replace("import unittest", "x" * (acceptance.MAX_SUITE_CHARS + 1))) is None
    # a Markdown suite holds fences inside strings; the file runs to the reply's last fence, not the first met
    fenced = REPLY.replace("import unittest", 'import unittest\nSOURCE = "```\\n# not a heading\\n```\\n"')
    assert parse_suite(fenced).content.endswith("unittest.main()\n") and "not a heading" in parse_suite(fenced).content
    assert acceptance.stopped_for_length(b'{"choices":[{"finish_reason":"length"}]}') is True
    assert acceptance.stopped_for_length(b'{"choices":[{"finish_reason":"stop"}]}') is False
    assert acceptance.stopped_for_length(b"not json") is False


async def test_the_author_call_has_its_own_token_budget_and_effort():
    seen = {}

    class Provider:
        async def complete(self, messages, *, timeout_s, tools=(), **options):
            seen.update(options)
            return completion_for(REPLY)

    miner = DemoMiner(settings(), Provider(), wallet=WalletLike("miner"))
    _completion, suite = await miner._author_suite(documented_task(), {"docs/widgets.md": "rules"}, time.monotonic() + 600)
    assert suite is not None
    assert seen == {"max_tokens": acceptance.AUTHOR_MAX_TOKENS, "reasoning_effort": "medium"}


def test_the_placement_command_writes_the_file_verbatim(tmp_path):
    content = "a = 'single' \"double\" $HOME `date` \\n\n\tindented\nEOF\n__END__\n"
    suite = Suite("true", str(tmp_path / "deep" / "suite.py"), content)
    subprocess.run(["bash", "-c", placement_command(suite)], check=True)
    assert (tmp_path / "deep" / "suite.py").read_text() == content


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("test_a ... FAIL\n\nFAILED (failures=2, errors=1)\n", "failures=2, errors=1"),
        ("# tests 10\n# pass 7\n# fail 3\n", "3 failed"),
        ("--- FAIL: TestA (0.00s)\n--- FAIL: TestB (0.00s)\nFAIL\n", "2 failed"),
        ("test result: FAILED. 4 passed; 1 failed; 0 ignored\n", "1 failed"),
        ("not ok 1 - a\nok 2 - b\n", "1 failed"),
        ("OK\n", ""),
    ],
)
def test_the_failure_count_is_read_from_each_runner(output, expected):
    assert run_summary(output) == expected


def test_the_reconciliation_message_names_the_command_the_path_and_the_rule_for_settling():
    suite = parse_suite(REPLY)
    text = acceptance.reconcile_prompt(suite, "FAILED (failures=1)", 1)
    assert suite.run in text and suite.path in text and "Exit code 1" in text
    assert "never saw your code" in text and "quote the passage your code follows" in text and "repair the suite file" in text
    # 2026-11-11: in 16 of 26 graded solves whose suite failed, the hidden tests passed; the suite is not presumed right.
    assert "neither side is presumed right" in text and "Keep your code where a passage" in text
    assert "usually right" not in text
    # Each refuted case comes back as a line the harness can read, with a quotation it can check.
    assert 'REFUTED: <the case\'s name as the runner printed it> | "<the passage' in text
    assert "word for word" in text and "runs again" in text
    assert "lacks a name the documentation gives, or gives it another signature, is your code's to fix" in text
    assert "seconds are left" not in text and "set aside" not in text
    # Both figures, so the settling window is not read as the time left for the whole task (966efbb1635f).
    assert (
        "About 420 seconds are set aside for settling these cases, out of about 700 seconds left for the whole task; "
        "the review and the final checks have their own time after that."
    ) in acceptance.reconcile_prompt(suite, "x", 1, seconds=420, total=700)
    alone = acceptance.reconcile_prompt(suite, "x", 1, seconds=420)
    assert "About 420 seconds are set aside for settling these cases; the review" in alone and "whole task" not in alone


async def test_the_author_call_carries_no_tools_and_yields_the_suite(documented):
    provider = Scripted(completion_for(REPLY))
    miner = DemoMiner(settings(), provider, wallet=WalletLike("miner"))
    completion, suite = await miner._author_suite(documented_task(), read_docs(INSTRUCTION, documented), time.monotonic() + 600)
    assert completion is not None and suite == parse_suite(REPLY)
    assert provider.tools == [()] and "1..10 inclusive" in provider.seen[0][1]["content"]


def fixed_author(reply=REPLY, completion=None):
    async def _author_suite(self, request, docs, deadline):
        return completion, parse_suite(reply)
    return _author_suite


async def test_a_failing_independent_suite_is_settled_before_the_review(documented, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", fixed_author(completion=completion_for(REPLY)))
    sandbox = ScriptedSandbox(result(0), result(0), suite_ran(1, b"test_limit_is_inclusive ... FAIL\nFAILED (failures=1)\n"))
    provider = Scripted(edit(), RUN, FINISHED, FINISHED, DONE)
    events, submission, notes = await run_loop(documented, provider, sandbox)
    assert b"gamma" in submission
    settle = provider.seen[3][-1]["content"]
    assert settle.startswith("Before your work is reviewed, an acceptance suite written independently")
    assert "FAILED (failures=1)" in settle and "/tmp/acceptance_from_docs.py" in settle
    assert provider.seen[4][-1]["content"].startswith("Before this is submitted")
    placed, ran = sandbox.requests[1].argv[6], sandbox.requests[2].argv[6]  # after the runner's own name
    assert placed.startswith("mkdir -p -- /tmp && cat > /tmp/acceptance_from_docs.py <<") and "class Rules" in placed
    assert ran.startswith("sha256sum -- /tmp/acceptance_from_docs.py 2>/dev/null | cut -c1-64; ( cd /tmp && ")
    assert "python3 -m unittest -v acceptance_from_docs ) >/tmp/.acceptance-run 2>&1" in ran
    assert acceptance.FAIL_LIST_MARK in ran and ran.endswith("exit $s")  # the failure lines again, where the tail keeps them
    assert (notes.acceptance, notes.acceptance_summary) == ("failed", "failures=1")
    # The model's answer to the failures is checked by running the suite again; here every case passes.
    assert sandbox.requests[3].argv[6] == ran
    # ... and once more on a clean copy before the answer goes out (the final check), which also passes.
    assert (notes.acceptance_runs, notes.acceptance_final, notes.clean_copy) == (3, "passed", "same")
    ids = [event["call_id"] for event in events if event["event_type"] == "tool_call"]
    assert ids[-4:] == ["acceptance-0", "acceptance-1", "acceptance-2", "acceptance-3"]
    turns = [event for event in events if event["event_type"] == "model_turn"]
    assert len(turns) == 6  # five answers and the author's call, all recorded
    assert len(provider.seen) == 5


async def test_a_passing_independent_suite_adds_no_turn(documented, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", fixed_author())
    sandbox = ScriptedSandbox(result(0), result(0), suite_ran(0, b"OK\n"))
    provider = Scripted(edit(), RUN, FINISHED, DONE)
    _, submission, notes = await run_loop(documented, provider, sandbox)
    assert b"gamma" in submission and notes.acceptance == "passed"
    assert provider.seen[3][-1]["content"].startswith("Before this is submitted")


async def test_without_documentation_or_sandbox_nothing_is_authored(workspace, documented, monkeypatch):
    calls = []

    async def counting(self, request, docs, deadline):
        calls.append(docs)
        return None, None

    monkeypatch.setattr(DemoMiner, "_author_suite", counting)
    _, _, notes = await run_loop(workspace, Scripted(edit(), RUN, FINISHED, DONE), FakeSandbox(), request=task())
    assert notes.acceptance == "no documentation" and calls == []
    _, _, notes = await run_loop(documented, Scripted(edit(), FINISHED, DONE), None)
    assert notes.acceptance == "" and calls == []
    _, _, notes = await run_loop(documented, Scripted(edit(), RUN, FINISHED, DONE), FakeSandbox(), miner_acceptance_suite=False)
    assert notes.acceptance == "" and calls == []


async def test_a_slow_author_is_abandoned_and_the_answer_still_goes_out(documented, monkeypatch):
    async def slow(self, request, docs, deadline):
        await asyncio.sleep(30)
        return None, None

    monkeypatch.setattr(DemoMiner, "_author_suite", slow)
    monkeypatch.setattr(acceptance, "AUTHOR_WAIT_S", 0.05)
    provider = Scripted(edit(), RUN, FINISHED, DONE)
    _, submission, notes = await run_loop(documented, provider, FakeSandbox())
    assert b"gamma" in submission and notes.acceptance == "author too slow"
    # the fake completions report no usage, so the four answers count as unreported too; the cancelled call adds one
    assert notes.spend.unreported_calls == len(provider.seen) + 1 and len(provider.seen) == 4


async def test_a_declined_or_broken_suite_never_costs_the_answer(documented, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", fixed_author(reply="NONE", completion=completion_for("NONE")))
    _, submission, notes = await run_loop(documented, Scripted(edit(), RUN, FINISHED, DONE), FakeSandbox())
    assert b"gamma" in submission and notes.acceptance == "no suite"
    monkeypatch.setattr(DemoMiner, "_author_suite", fixed_author())
    sandbox = ScriptedSandbox(result(0), result(1, b"bash: cannot create file\n"))
    _, submission, notes = await run_loop(documented, Scripted(edit(), RUN, FINISHED, DONE), sandbox)
    assert b"gamma" in submission and notes.acceptance == "could not run" and len(sandbox.requests) == 2


async def test_when_the_answer_takes_all_the_time_the_suite_is_skipped_but_its_call_is_recorded(documented, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", fixed_author(completion=completion_for(REPLY)))
    sandbox = ScriptedSandbox()
    events, submission, notes = await run_loop(documented, Scripted(edit(), RUN, FINISHED, DONE), sandbox, deadline=200.0)
    assert b"gamma" in submission and len(sandbox.requests) == 1  # only the model's own run
    assert notes.acceptance.startswith("not run")
    assert sum(event["event_type"] == "model_turn" for event in events) == 5  # four answers plus the author's call
    assert not [event for event in events if event["event_type"] == "tool_call" and event["call_id"].startswith("acceptance-")]


def test_a_go_suite_is_fitted_to_a_module_older_than_go_1_18(tmp_path):
    # Two live suites of 2026-11-11 used `any` under `go 1.17` and failed to build.
    content = (
        'package x_test\n\n// any comment stays\nfunc h(v any, m map[string]any, xs []any) (any, error) {\n'
        '\tvar any2 = 1\n\t_ = any2\n\tt := "any string stays"\n\tr := `raw any`\n\t_ = r\n\t_ = t\n'
        '\tvar box interface{} = v\n\t_ = box\n\treturn m["k"], nil\n}\n'
    )
    suite = acceptance.Suite("go test ./x/", "/work/x/acc_test.go", content)
    (tmp_path / "go.mod").write_text("module example.com/x\n\ngo 1.17\n")
    version = acceptance.go_language_version(tmp_path)
    assert version == (1, 17)
    fitted = acceptance.fit_go_suite(suite, version).content
    assert "func h(v interface{}, m map[string]interface{}, xs []interface{}) (interface{}, error)" in fitted
    assert "// any comment stays" in fitted and '"any string stays"' in fitted and "`raw any`" in fitted
    assert "var any2 = 1" in fitted
    # vet is off in every Go suite run: a printf warning in a file nobody touched failed a build (a82b2ffec36e)
    modern = acceptance.fit_go_suite(suite, (1, 21))
    assert (modern.run, modern.content) == ("go test -vet=off ./x/", content)
    assert acceptance.fit_go_suite(suite, None).run == "go test -vet=off ./x/"
    assert acceptance.fit_go_suite(modern, (1, 21)) is modern  # already fitted
    stated = acceptance.Suite("cd /work && go test -vet=all ./x/", "/work/x/acc_test.go", content)
    assert acceptance.fit_go_suite(stated, (1, 21)) is stated  # a vet setting of the task's own is kept
    assert acceptance.fit_go_suite(acceptance.Suite("x", "/tmp/t.py", "any(xs)\n"), (1, 17)).content == "any(xs)\n"
    assert acceptance.go_language_version(tmp_path / "missing") is None


def test_the_author_tests_only_documented_rules_each_with_its_sentence_and_the_interface_first():
    system = author_messages(INSTRUCTION, {"docs/widgets.md": "rules"}, "tools")[0]["content"]
    assert "Test only what the documentation states" in system and "comment beginning RULE:" in system
    assert "gets no case, however usual it is elsewhere" in system
    assert "var _ func(string, int) (*Thing, error) = pkg.NewThing" in system and "inspect.signature" in system


UNITTEST_FAILURE = (
    b"test_limit_is_inclusive (acceptance_from_docs.Rules.test_limit_is_inclusive) ... FAIL\n\n"
    b"======================================================================\n"
    b"FAIL: test_limit_is_inclusive (acceptance_from_docs.Rules.test_limit_is_inclusive)\n"
    b"----------------------------------------------------------------------\n"
    b"AssertionError: 2 != 1\n\nFAILED (failures=1)\n"
)
GOOD_QUOTE = 'REFUTED: test_limit_is_inclusive | "A limit of 0 is invalid; 1..10 inclusive."'


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        (UNITTEST_FAILURE.decode(), ["test_limit_is_inclusive"]),
        ("ERROR: test_a (m.C.test_a)\nFAIL: test_b (m.C.test_b) [case 3]\nFAILED (failures=1, errors=1)\n", ["test_a", "test_b"]),
        # the module did not import, so no case ran, and no quotation settles that
        ("ERROR: m (unittest.loader._FailedTest.m)\nImportError: cannot import name 'X'\nFAILED (errors=1)\n", None),
        (
            "--- FAIL: TestA (0.00s)\n    --- FAIL: TestA/zero (0.00s)\n    --- PASS: TestA/one (0.00s)\n"
            "--- FAIL: TestB (0.00s)\nFAIL\n",
            ["TestA/zero", "TestB"],
        ),
        ("# example.com/x\n./acc_test.go:9:2: undefined: x.New\nFAIL\texample.com/x [build failed]\n", None),
        ("test tests::limit ... FAILED\ntest tests::ok ... ok\ntest result: FAILED. 1 passed; 1 failed\n", ["tests::limit"]),
        (
            "TAP version 13\n# Subtest: /tmp/acc.test.mjs\n    # Subtest: limits\n        not ok 1 - zero is invalid\n"
            "        ok 2 - ten is valid\n    not ok 1 - limits\n    ok 2 - exports\n    not ok 3 - parses a header\n"
            "    not ok 4 - later # TODO not written yet\nnot ok 1 - /tmp/acc.test.mjs\n",
            ["zero is invalid", "parses a header"],
        ),
        # the file failed as a whole (a syntax error): its only result is the file's own
        ("TAP version 13\nnot ok 1 - /tmp/acc.test.mjs\n  ---\n  error: 'SyntaxError'\n", None),
        ("Segmentation fault\n", None),
    ],
)
def test_failing_cases_are_read_from_each_runner(output, expected):
    assert acceptance.failing_cases(output, "/tmp/acc.test.mjs") == expected


def test_refutations_are_read_in_the_stated_form_and_its_usual_variants():
    reply = (
        "Fixed the limit check.\n"
        'REFUTED: test_limit_is_inclusive | "A limit of 0 is invalid; 1..10 inclusive."\n'
        "- **REFUTED:** `TestA/zero` | \u201cZero means no limit.\u201d\n"
        'REFUTED: table | "| code | meaning |"\n'
        "NOT REFUTED: x | y\n"
        "REFUTED: no quotation here\n"
    )
    assert acceptance.refutations(reply) == [
        ("test_limit_is_inclusive", '"A limit of 0 is invalid; 1..10 inclusive."'),
        ("TestA/zero", "\u201cZero means no limit.\u201d"),
        ("table", '"| code | meaning |"'),
    ]


def test_a_quotation_counts_only_when_it_is_in_the_documentation_word_for_word():
    docs = {
        "docs/w.md": (
            "# Widgets\n\n- A limit of 0 is *invalid*; limits\n  run 1..10 inclusive.\n\n| code | meaning |\n"
            "|---|---|\n| E_LIMIT | the limit is out of range |\n"
        )
    }
    settling = acceptance.Settling.start(Suite("run", "/tmp/s.py", "x"), "Task: widgets resume from a checkpoint.", docs)

    def found(quote):
        return acceptance.quote_found(quote, settling.sources)

    assert found('"A limit of 0 is invalid; limits run 1..10 inclusive."')  # emphasis, list mark and line break set aside
    assert found("\u201ca LIMIT of 0 is invalid\u201d\u2026 run 1..10 inclusive")  # quote style, case, an ellipsis joining parts
    assert found("E_LIMIT the limit is out of range")  # a table row without its bars
    assert found("widgets resume from a checkpoint")  # the task text counts too
    assert not found("A limit of 0 is never accepted by widgets")  # a paraphrase
    assert not found("is invalid")  # too short to tell where it is from
    assert not found("A limit of 0 is invalid ... 10")  # one part too short


def test_a_refutation_covers_the_case_it_names_and_no_other():
    assert acceptance.covered("test_limit", ["test_limit (acc.Rules.test_limit)"])
    assert acceptance.covered("test_limit", ["Rules.test_limit"])
    assert not acceptance.covered("test_limit_zero", ["test_limit"])
    # a Go parent covers no subtest by itself (Settling.take_reply expands it into the subtests failing then)
    assert not acceptance.covered("TestA/zero", ["TestA"]) and not acceptance.covered("TestAB/zero", ["TestA"])
    assert not acceptance.covered("exports", ["default exports are functions"])  # a word inside another name
    assert acceptance.covered("zero is invalid", ["`Zero is invalid`"])
    assert not acceptance.covered("zero is invalid", ["zero"])


async def test_failures_go_back_until_a_quotation_from_the_documentation_settles_them(documented, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", fixed_author(completion=completion_for(REPLY)))
    failing = suite_ran(1, UNITTEST_FAILURE)
    sandbox = ScriptedSandbox(result(0), result(0), failing, failing, failing, failing)
    paraphrase = completion_for('Kept it.\nREFUTED: test_limit_is_inclusive | "zero is not a valid limit for any widget"')
    quoted = completion_for(f"Kept it.\n{GOOD_QUOTE}")
    provider = Scripted(edit(), RUN, FINISHED, paraphrase, quoted, DONE)
    _, submission, notes = await run_loop(documented, provider, sandbox)
    assert b"gamma" in submission
    first, second = provider.seen[3][-1]["content"], provider.seen[4][-1]["content"]
    assert first.startswith("Before your work is reviewed")
    assert second.startswith("The acceptance suite ran again")
    assert "- test_limit_is_inclusive" in second and "settle nothing" in second and "zero is not a valid limit" in second
    assert provider.seen[5][-1]["content"].startswith("Before this is submitted")
    assert (notes.acceptance_runs, notes.acceptance_final) == (4, "settled")  # with the final check's clean run
    assert (notes.acceptance_refuted, notes.acceptance_rejected) == (1, 1)
    assert len(sandbox.requests) == 6


async def test_the_suite_stops_coming_back_once_its_share_of_the_time_is_used(documented, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", fixed_author())
    monkeypatch.setattr(acceptance, "SETTLE_FRACTION", 0.0)
    failing = suite_ran(1, UNITTEST_FAILURE)
    sandbox = ScriptedSandbox(result(0), result(0), failing, failing, failing)
    provider = Scripted(edit(), RUN, FINISHED, FINISHED, DONE)
    _, submission, notes = await run_loop(documented, provider, sandbox)
    assert b"gamma" in submission
    assert provider.seen[3][-1]["content"].startswith("Before your work is reviewed")  # the first failures always go back
    assert provider.seen[4][-1]["content"].startswith("Before this is submitted")
    assert (notes.acceptance_runs, notes.acceptance_final) == (3, "open")  # the final check sees the same open case


async def test_the_suite_runs_at_most_as_often_as_the_setting_allows(documented, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", fixed_author())
    failing = suite_ran(1, UNITTEST_FAILURE)
    sandbox = ScriptedSandbox(result(0), result(0), failing, failing, failing, failing)
    provider = Scripted(edit(), RUN, FINISHED, FINISHED, FINISHED, DONE)
    _, _, notes = await run_loop(documented, provider, sandbox, miner_acceptance_runs=3)
    assert provider.seen[4][-1]["content"].startswith("The acceptance suite ran again")
    assert provider.seen[5][-1]["content"].startswith("Before this is submitted")
    assert (notes.acceptance_runs, notes.acceptance_final) == (4, "open")  # three settling runs, then the final check


async def test_the_settle_messages_name_the_window_and_the_time_to_the_wrap_up(documented, monkeypatch):
    # 966efbb1635f read "About 280 seconds are left for this" (the settling window) as the time left for the task.
    from tests.test_demo_miner_workbench import Clock, Timed
    monkeypatch.setattr(DemoMiner, "_author_suite", fixed_author())
    clock = Clock()
    monkeypatch.setattr("rlvr.neurons.demo_miner.time", clock)
    failing = suite_ran(1, UNITTEST_FAILURE)
    sandbox = ScriptedSandbox(result(0), result(0), failing, failing, failing, failing)
    provider = Timed(clock, 10.0, edit(), RUN, FINISHED, FINISHED, FINISHED, DONE)
    miner = DemoMiner(
        settings(miner_adjudicator=False, miner_acceptance_runs=3), provider, wallet=WalletLike("miner"), sandbox=sandbox
    )
    await miner._generate(documented_task(), WorkspaceReader(documented), clock.now + 900.0, notes=SolveNotes())
    reconcile, settle = provider.seen[3][-1]["content"], provider.seen[4][-1]["content"]
    # The wrap-up is at 1810 (900 s less the 90 s reserve) and settling ends at 1675 (SETTLE_FRACTION of 900 s);
    # each model call takes 10 s, so the first failures go back at 1030 and the rerun's at 1040.
    assert reconcile.startswith("Before your work is reviewed")
    assert (
        "About 645 seconds are set aside for settling these cases, out of about 780 seconds left for the whole task; "
        "the review and the final checks have their own time after that." in reconcile
    )
    assert reconcile.endswith("\n\nClock: 30 s used, about 780 s left for this task.")
    assert settle.startswith("The acceptance suite ran again")
    assert "About 635 seconds are set aside for settling these cases, out of about 770 seconds left" in settle
    assert settle.endswith("\n\nClock: 40 s used, about 770 s left for this task.")
    assert "seconds are left for this" not in reconcile + settle


def test_the_settle_message_never_names_a_window_longer_than_the_time_left():
    settling = acceptance.Settling(parse_suite(REPLY), ("a limit of 0 is invalid",))
    settling.output, settling.exit_code, settling.open = "FAILED (failures=1)", 1, ["test_limit_is_inclusive"]
    text = acceptance.settle_prompt(settling, 300, total=200)
    assert "About 200 seconds are set aside for settling these cases, out of about 200 seconds left" in text
    assert "About 300 seconds are set aside for settling these cases; the review" in acceptance.settle_prompt(settling, 300)


async def test_a_suite_that_fails_without_naming_a_case_is_not_settled_by_a_quotation(documented, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", fixed_author())
    def broken(name):
        return suite_ran(
            1, b"ERROR: acceptance_from_docs (unittest.loader._FailedTest.acceptance_from_docs)\n"
               b"ImportError: cannot import name '" + name + b"'\nFAILED (errors=1)\n"
        )

    # the model's change moved the failure on, so it is not stuck; a quotation still settles nothing
    sandbox = ScriptedSandbox(result(0), result(0), broken(b"Widget"), broken(b"Gadget"), suite_ran(0, b"OK\n"))
    provider = Scripted(edit(), RUN, FINISHED, completion_for(GOOD_QUOTE), FINISHED, DONE)
    _, _, notes = await run_loop(documented, provider, sandbox)
    second = provider.seen[4][-1]["content"]
    assert second.startswith("The acceptance suite ran again") and "without naming its cases" in second
    assert provider.seen[5][-1]["content"].startswith("Before this is submitted")
    assert (notes.acceptance_runs, notes.acceptance_final) == (4, "passed")  # the final check passes too


async def test_a_suite_written_into_the_answer_is_taken_out_before_the_review(documented, monkeypatch):
    go_reply = REPLY.replace("PATH: /tmp/acceptance_from_docs.py", "PATH: /work/zz_acceptance_test.go")
    monkeypatch.setattr(DemoMiner, "_author_suite", fixed_author(reply=go_reply))
    written = completion_with_calls(("write_file", {"path": "zz_acceptance_test.go", "content": "package x_test\n"}))
    sandbox = ScriptedSandbox(result(0), result(0), suite_ran(1, b"--- FAIL: TestLimit (0.00s)\nFAIL\n"), result(0), suite_ran(0, b"ok\n"))
    provider = Scripted(edit(), RUN, FINISHED, written, FINISHED, RUN, FINISHED, DONE)
    _, submission, notes = await run_loop(documented, provider, sandbox)
    assert b"gamma" in submission and b"zz_acceptance_test.go" not in submission
    assert notes.acceptance_final == "passed"
    assert "zz_acceptance_test.go" not in provider.seen[7][-1]["content"]  # the review's diff


async def test_an_unchanged_failure_that_names_no_case_stops_coming_back(documented, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", fixed_author())
    broken = suite_ran(1, b"ERROR: acceptance_from_docs (unittest.loader._FailedTest.acceptance_from_docs)\nFAILED (errors=1)\n")
    sandbox = ScriptedSandbox(result(0), result(0), broken, broken, broken)
    provider = Scripted(edit(), RUN, FINISHED, FINISHED, DONE)
    _, _, notes = await run_loop(documented, provider, sandbox)
    # the first failure goes back; the same one again means nothing the model did reached it, so it stops
    assert provider.seen[4][-1]["content"].startswith("Before this is submitted")
    assert (notes.acceptance_runs, notes.acceptance_final) == (3, "unreadable")  # the final check's clean run fails the same way


async def test_one_run_restores_the_single_send_back(documented, monkeypatch):
    # MINER_ACCEPTANCE_RUNS=1: the failures go back once, the model reruns the suite itself, and the review follows.
    monkeypatch.setattr(DemoMiner, "_author_suite", fixed_author())
    failing = suite_ran(1, UNITTEST_FAILURE)
    sandbox = ScriptedSandbox(result(0), result(0), failing)
    provider = Scripted(edit(), RUN, FINISHED, FINISHED, DONE)
    _, _, notes = await run_loop(documented, provider, sandbox, miner_acceptance_runs=1)
    first = provider.seen[3][-1]["content"]
    assert "rerun it with the command above" in first and "runs again" not in first and "seconds are left" not in first
    # No settling window is named when the suite does not run again, but the clock still ends the message.
    from tests.test_demo_miner_workbench import clock_figures
    assert "set aside" not in first and clock_figures(provider.seen[3][-1]) is not None
    assert provider.seen[4][-1]["content"].startswith("Before this is submitted")
    assert (notes.acceptance_runs, notes.acceptance_final) == (1, "open") and len(sandbox.requests) == 3


async def test_a_time_check_follows_a_tool_batch_once_its_share_of_the_time_has_passed(documented, monkeypatch):
    from rlvr.neurons import demo_miner
    monkeypatch.setattr(DemoMiner, "_author_suite", fixed_author())
    monkeypatch.setattr(demo_miner, "TIME_CHECKS", (0.0,))
    provider = Scripted(edit(), RUN, FINISHED, DONE)
    _, _, notes = await run_loop(documented, provider, ScriptedSandbox(result(0), result(0), suite_ran(0, b"OK\n")))
    note = provider.seen[1][-1]["content"]
    assert note.startswith("Time check: about ") and "seconds are left" in note
    assert notes.time_checks == 1
    assert not any(str(m.get("content", "")).startswith("Time check") for m in provider.seen[2][-1:])  # sent once


async def test_the_final_check_on_a_clean_copy_sends_back_once_when_it_does_worse(documented, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", fixed_author())
    failing = suite_ran(1, UNITTEST_FAILURE)
    # the suite passes in the working copy, then fails on the clean copy, twice
    sandbox = ScriptedSandbox(result(0), result(0), suite_ran(0, b"OK\n"), failing, failing)
    provider = Scripted(edit(), RUN, FINISHED, DONE, FINISHED)
    _, submission, notes = await run_loop(documented, provider, sandbox)
    assert provider.seen[3][-1]["content"].startswith("Before this is submitted")  # the review first
    asked = provider.seen[4][-1]["content"]
    assert asked.startswith("Before the answer goes out, the acceptance suite ran again, the last time on a clean copy")
    assert "In your working copy the same code had this result: every case passed." in asked
    assert "REFUTED: <the case's name" in asked  # a deliberate change can be kept and its case refuted
    assert b"gamma" in submission and notes.kept_best == ""
    assert notes.acceptance_runs == 3 and len(provider.seen) == 5  # asked once; the second check went out as it is


async def test_a_final_version_that_still_runs_its_cases_goes_out_after_one_question(documented, monkeypatch):
    # The review's change made a case fail. The model is asked once, may fix, undo or refute; when it does none
    # of these, its version still goes out: only a version that no longer builds or loads is ever replaced.
    monkeypatch.setattr(DemoMiner, "_author_suite", fixed_author())
    failing = suite_ran(1, UNITTEST_FAILURE)
    sandbox = ScriptedSandbox(
        result(0), result(0), failing, result(0), suite_ran(0, b"OK\n"),  # settled: the rerun passes (version 2)
        failing, failing,                                                 # final check: working copy, clean copy
        failing, failing,                                                 # and again after the question
    )
    provider = Scripted(
        edit(), RUN, FINISHED,
        edit("a.txt", "gamma", "delta"), FINISHED, RUN, FINISHED,
        edit("a.txt", "delta", "epsilon"), FINISHED,
        FINISHED,
    )
    _, submission, notes = await run_loop(documented, provider, sandbox)
    asked = provider.seen[9][-1]["content"]
    assert asked.startswith("Before the answer goes out") and "An earlier version of your work had this result: every case passed." in asked
    assert b"epsilon" in submission and notes.kept_best == ""


async def test_an_earlier_version_is_kept_when_the_final_one_no_longer_builds(documented, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", fixed_author())
    failing = suite_ran(1, UNITTEST_FAILURE)
    broken = suite_ran(1, b"ERROR: acceptance_from_docs (unittest.loader._FailedTest.acceptance_from_docs)\nSyntaxError\n")
    sandbox = ScriptedSandbox(
        result(0), result(0), failing, result(0), suite_ran(0, b"OK\n"),
        broken, broken,
        broken, broken,
    )
    provider = Scripted(
        edit(), RUN, FINISHED,
        edit("a.txt", "gamma", "delta"), FINISHED, RUN, FINISHED,
        edit("a.txt", "delta", "epsilon"), FINISHED,
        FINISHED,
    )
    _, submission, notes = await run_loop(documented, provider, sandbox)
    assert provider.seen[9][-1]["content"].startswith("Before the answer goes out")
    assert b"delta" in submission and b"epsilon" not in submission
    assert notes.kept_best.startswith("version ") and "every case passed" in notes.kept_best
    assert notes.own_tests == "not run: an earlier version goes out"  # the reruns would test another version


# Review of 2026-11-13: each test below pins a defect a reviewer reproduced and a verifier confirmed.

async def test_a_suite_run_that_errors_does_not_leave_its_call_id_to_the_next(documented, monkeypatch):
    # A rerun whose sandbox call failed used to give its number to the final check's run: a duplicate call id
    # makes the trajectory invalid and the whole answer is lost.
    monkeypatch.setattr(DemoMiner, "_author_suite", fixed_author())
    failing = suite_ran(1, UNITTEST_FAILURE)
    sandbox = ScriptedSandbox(result(0), result(0), failing, result(126, b"sandbox error: boom"), failing)
    provider = Scripted(edit(), RUN, FINISHED, FINISHED, DONE)
    events, submission, _notes = await run_loop(documented, provider, sandbox)
    ids = [event["call_id"] for event in events if event["event_type"] == "tool_call"]
    assert len(ids) == len(set(ids)) and "acceptance-3" in ids
    assert b"gamma" in submission


def _state(output, refuted=(), exit_code=1):
    settling = acceptance.Settling.start(Suite("run", "/tmp/s.py", "x"), "task", {"d.md": "A limit of 0 admits nothing at all here."})
    for case in refuted:
        settling.refuted[case] = "quote"
    return settling, settling.record(output, exit_code, False)


def test_runs_that_did_not_really_run_their_cases_are_never_passed_or_settled():
    hashed = SUITE_HASH.decode()
    # a failed fixture stopped every case under it
    assert _state(hashed + "ERROR: setUpClass (acc.Rules)\nRan 0 tests\nFAILED (errors=1)\n")[1] == "unreadable"
    # a panic aborted the Go binary: refuting the panicking case does not settle the rest
    settling, state = _state(hashed + "--- FAIL: TestA/zero (0.00s)\npanic: boom [recovered]\nFAIL\n", refuted=["TestA/zero"])
    assert state == "open" and settling.open == [acceptance.PANIC_CASE]
    # nothing ran: a Go package without the suite file, or a missing file (no hash at all)
    assert _state(hashed + "ok  \texample.com/x\t0.01s [no tests to run]\n", exit_code=0)[1] == "unreadable"
    assert _state("OK\n", exit_code=0)[1] == "unreadable"
    # more failure lines than the list could show stay open whatever is refuted
    listed = "".join(f"FAIL: test_{i} (acc.R.test_{i})\n" for i in range(acceptance.FAIL_LIST_LINES))
    output = hashed + "...\n" + acceptance.FAIL_LIST_MARK + f" {acceptance.FAIL_LIST_LINES + 7}\n" + listed
    settling, state = _state(output, refuted=[f"test_{i}" for i in range(acceptance.FAIL_LIST_LINES)])
    assert state == "open" and settling.open == ["(and 7 more failure lines than the list above shows)"]


def test_the_failure_list_after_the_output_names_cases_the_kept_tail_would_lose():
    # the middle of a long run is cut; the list printed after it is complete
    output = SUITE_HASH.decode() + "test_a ... ok\n[... output omitted ...]\nOK?\n" + acceptance.FAIL_LIST_MARK + " 2\nFAIL: test_hidden (acc.R.test_hidden)\nFAILED (failures=1)\n"
    assert acceptance.failing_cases(output) == ["test_hidden"]


def test_refuted_lines_in_a_numbered_list_or_inline_code_are_read():
    reply = '1. REFUTED: test_a | "one passage"\n`REFUTED: test_b | "two passage"`\n- 2) **REFUTED:** test_c | "three"\n'
    assert [case for case, _quote in acceptance.refutations(reply)] == ["test_a", "test_b", "test_c"]


def test_a_go_parent_refutes_only_the_subtests_failing_when_it_was_written():
    docs = {"d.md": "An empty name is rejected with ErrName before anything else happens."}
    settling = acceptance.Settling.start(Suite("run", "/work/x/acc_test.go", "x"), "task", docs)
    hashed = SUITE_HASH.decode()
    settling.record(hashed + "--- FAIL: TestErrors (0.00s)\n    --- FAIL: TestErrors/empty (0.00s)\nFAIL\n", 1, False)
    settling.take_reply('REFUTED: TestErrors | "An empty name is rejected with ErrName before anything else happens."')
    later = hashed + "--- FAIL: TestErrors (0.00s)\n    --- FAIL: TestErrors/empty (0.00s)\n    --- FAIL: TestErrors/limit (0.00s)\nFAIL\n"
    assert settling.record(later, 1, False) == "open" and settling.open == ["TestErrors/limit"]


def test_a_stored_version_never_holds_the_harness_suite_file():
    from rlvr.neurons.demo_miner import _without_file
    patch = (b"diff --git a/a.txt b/a.txt\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-beta\n+gamma\n"
             b"diff --git a/x/zz_acceptance_test.go b/x/zz_acceptance_test.go\nnew file mode 100644\n--- /dev/null\n"
             b"+++ b/x/zz_acceptance_test.go\n@@ -0,0 +1 @@\n+package x_test\n")
    kept = _without_file(patch, "x/zz_acceptance_test.go")
    assert b"zz_acceptance_test.go" not in kept and b"+gamma" in kept


async def test_a_clean_run_that_timed_out_decides_nothing(documented, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", fixed_author())
    sandbox = ScriptedSandbox(result(0), result(0), suite_ran(0, b"OK\n"), suite_ran(124, b"", timed_out=True))
    provider = Scripted(edit(), RUN, FINISHED, DONE)
    _, submission, notes = await run_loop(documented, provider, sandbox)
    assert b"gamma" in submission and notes.kept_best == "" and len(provider.seen) == 4  # no question was asked


async def test_a_refutation_in_reply_to_the_final_check_settles_the_case(documented, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", fixed_author())
    failing = suite_ran(1, UNITTEST_FAILURE)
    sandbox = ScriptedSandbox(
        result(0), result(0), failing, result(0), suite_ran(0, b"OK\n"),  # settled on version 2
        failing, failing,                                                 # the review's change fails a case
        failing, failing,                                                 # rerun with the refutation, then clean
    )
    provider = Scripted(
        edit(), RUN, FINISHED,
        edit("a.txt", "gamma", "delta"), FINISHED, RUN, FINISHED,
        edit("a.txt", "delta", "epsilon"), FINISHED,
        completion_for(f"Kept the change.\n{GOOD_QUOTE}"),
    )
    _, submission, notes = await run_loop(documented, provider, sandbox)
    assert b"epsilon" in submission and notes.kept_best == "" and notes.acceptance_refuted >= 1


async def test_the_first_finish_with_time_to_spare_asks_once_for_a_test_of_every_documented_rule(documented, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", fixed_author())
    sandbox = ScriptedSandbox(result(0), result(0), suite_ran(0, b"OK\n"))
    provider = Scripted(edit(), RUN, FINISHED, FINISHED, DONE)
    _, submission, notes = await run_loop(documented, provider, sandbox, miner_coverage_prompt=True)
    asked = provider.seen[3][-1]["content"]
    assert asked.startswith("Before this goes further: about ") and "every rule it states" in asked
    assert provider.seen[4][-1]["content"].startswith("Before this is submitted")  # asked once, then the review
    assert notes.coverage_asked and b"gamma" in submission


async def test_no_coverage_question_when_the_time_is_short(documented, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", fixed_author())
    provider = Scripted(edit(), RUN, FINISHED, DONE)
    _, _, notes = await run_loop(documented, provider, ScriptedSandbox(), deadline=300.0, miner_coverage_prompt=True)
    assert not notes.coverage_asked
