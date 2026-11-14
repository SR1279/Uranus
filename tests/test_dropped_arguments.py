"""An error raised without a value its function was given (lint.dropped_arguments, asked with the diff questions)."""

from __future__ import annotations

import pytest

from rlvr.neurons import lint
from rlvr.neurons.demo_miner import DemoMiner
from tests.test_acceptance import RUN, ScriptedSandbox, result, run_loop
from tests.test_demo_miner_workbench import DONE, FINISHED, Scripted, edit, workspace  # noqa: F401

# The two lines of 121c13873690, in the function that received the step id.
RESOLVE = (
    "def resolve_argv(plan: Plan, step_id: str, outputs: Mapping[str, Any]) -> list[str]:",
    "    if not isinstance(plan, Plan):",
    "        raise BatchError('reference', 'not a compiled plan', step_id=step_id if is_step_id(step_id) else None)",
    "    if step_id not in plan.by_id:",
    "        raise BatchError('reference', f'unknown step id {step_id!r}', step_id=None)",
    "    return list(plan.by_id[step_id].argv)",
)


def added(path, *lines):
    return f"--- a/{path}\n+++ b/{path}\n@@ -0,0 +1,{len(lines)} @@\n" + "".join(f"+{line}\n" for line in lines)


def kinds(diff, docs=None):
    return [(question.kind, question.path, question.line) for question in lint.dropped_arguments(diff, docs or {})]


def test_the_recorded_lines_are_found():
    found = kinds(added("pixeltable_cli/client/batch_plan.py", *RESOLVE))
    assert found == [
        (lint.DROPPED_ARGUMENT, "pixeltable_cli/client/batch_plan.py",
         "raise BatchError('reference', 'not a compiled plan', step_id=step_id if is_step_id(step_id) else None)"),
        (lint.DROPPED_ARGUMENT, "pixeltable_cli/client/batch_plan.py",
         "raise BatchError('reference', f'unknown step id {step_id!r}', step_id=None)"),
    ]
    # They come with the other diff questions, each line once.
    asked = lint.questions(added("pkg/plan.py", *RESOLVE, *RESOLVE[3:5]), {})
    assert [question.kind for question in asked] == [lint.DROPPED_ARGUMENT, lint.DROPPED_ARGUMENT]


@pytest.mark.parametrize(
    "lines",
    [
        # The value is passed on.
        ("def resolve(plan, step_id):", "    raise BatchError('reference', 'unknown', step_id=step_id)"),
        # The function was never given a value of that name.
        ("def compile_plan(raw):", "    raise BatchError('plan', 'not an object', step_id=None)"),
        # Another keyword is None; the parameter is not among the keywords.
        ("def resolve(plan, step_id):", "    raise BatchError('reference', 'unknown', pointer=None)"),
        # The words stand inside the message, not among the arguments.
        ("def resolve(plan, step_id):", "    raise BatchError('reference', 'step_id=None is not a step')"),
        # A comparison is not a keyword argument.
        ("def resolve(plan, step_id):", "    raise BatchError('reference', str(step_id == None))"),
        # A call that is not raised, and an attribute of something else.
        ("def resolve(plan, step_id):", "    log(step_id=None)", "    raise BatchError('x', step.step_id)"),
        # The raise stands after the function has ended.
        ("def resolve(plan, step_id):", "    return plan", "", "raise BatchError('reference', 'x', step_id=None)"),
        # Only the innermost function counts: the helper inside was not given the value.
        ("def resolve(plan, step_id):", "    def fail(message):", "        raise BatchError('reference', message, step_id=None)",
         "    return fail"),
        # self and cls are not values a caller reads back.
        ("class A:", "    def check(self, cls):", "        raise BatchError('x', 'y', self=None)"),
    ],
)
def test_other_raises_stay_quiet(lines):
    assert kinds(added("pkg/plan.py", *lines)) == []


def test_methods_async_functions_and_raises_over_several_lines_are_read():
    method = (
        "class Runner:",
        "    async def run_step(",
        "        self,",
        "        step_id: str,",
        "        *,",
        "        attempt: int = 0,",
        "    ) -> None:",
        "        if attempt > 3:",
        "            raise StepError(",
        "                'too many attempts (limit 3)',",
        "                step_id=None,",
        "                attempt=attempt if attempt else None,",
        "            )",
    )
    found = kinds(added("pkg/runner.py", *method))
    assert len(found) == 1 and found[0][2].startswith("raise StepError( 'too many attempts (limit 3)', step_id=None,")
    # A keyword-only or starred parameter is a received value too.
    assert len(kinds(added("pkg/a.py", "def f(*, key):", "    raise E('x', key=None)"))) == 1
    assert len(kinds(added("pkg/a.py", "def f(**options):", "    raise E('x', options=None)"))) == 1


def test_only_python_sources_are_read_and_documented_none_is_left_alone():
    assert kinds(added("tests/test_plan.py", *RESOLVE)) == []
    assert kinds(added("lib/plan.js", *RESOLVE)) == []
    assert kinds(added("pkg/plan.py", *RESOLVE), {"docs/batch.md": "`step_id` is None when the plan itself is invalid."}) == []
    assert len(kinds(added("pkg/plan.py", *RESOLVE), {"docs/batch.md": "Every error names its `step_id`."})) == 2
    # Removed lines are not read, and a removed def does not lend its parameters.
    removed = "--- a/pkg/plan.py\n+++ b/pkg/plan.py\n@@ -1,2 +1,1 @@\n-def resolve(plan, step_id):\n+    raise E('x', step_id=None)\n"
    assert kinds(removed) == []


def test_a_hunk_header_that_names_the_function_lends_its_parameters():
    def hunk(heading, *lines):
        body = "".join(f"{line}\n" for line in lines)
        return f"--- a/pkg/plan.py\n+++ b/pkg/plan.py\n@@ -40,3 +40,5 @@{heading}\n{body}"

    inside = ("     plan = load(plan)", "+    if step_id not in plan.by_id:", "+        raise BatchError('reference', 'unknown', step_id=None)")
    assert len(kinds(hunk(" def resolve_argv(plan, step_id, outputs):", *inside))) == 1
    # Without the function in the header, or once a later function has begun, the parameters are not known.
    assert kinds(hunk("", *inside)) == []
    assert kinds(hunk(" class Plan:", *inside)) == []
    later = (" def other(plan):", "+    raise BatchError('reference', 'unknown', step_id=None)")
    assert kinds(hunk(" def resolve_argv(plan, step_id, outputs):", *later)) == []
    # A second hunk starts again from its own header.
    two = hunk(" def resolve_argv(plan, step_id, outputs):", "     return plan") + "@@ -90,2 +92,3 @@\n" + "".join(f"{line}\n" for line in inside)
    assert kinds(two) == []


def test_the_prompt_asks_for_the_value_unless_the_documentation_says_none():
    found = lint.dropped_arguments(added("pkg/plan.py", *RESOLVE), {})
    text = lint.question_prompt(found)
    assert text.startswith("An error is raised here without a value the function was given:\n- pkg/plan.py: `raise BatchError(")
    assert "The function received `step_id`, and this raise passes None for it." in text
    assert "Unless a sentence of the documentation says this error carries none, pass the value the function received" in text
    assert text.endswith("Then run your tests again and reply with a one-line summary.")
    # The other questions keep their own default; this one is set apart from it.
    both = lint.question_prompt(lint.questions(added("pkg/plan.py", "class Report(str):", "    pass", *RESOLVE), {}))
    assert both.startswith("Questions from a check of your change, each about one line and the one input")
    assert "keep the code as it is.\n\nOne more line, with the opposite default. An error is raised here without" in both
    assert both.count("reply with a one-line summary.") == 1
    # The other questions alone read as they did.
    alone = lint.question_prompt(lint.questions(added("pkg/plan.py", "class Report(str):"), {}))
    assert alone.endswith("keep the code as it is. Then run your tests again and reply with a one-line summary.")
    assert "opposite default" not in alone and "without a value" not in alone
    # It quotes the code and nothing said about it.
    for word in ("model", "catalogue", "you said", "your summary", "assum"):
        assert word not in text.lower()


async def no_suite(self, request, docs, deadline):
    return None, None


DROPS = "def b(step_id=None):\n    if step_id != 'known':\n        raise KeyError('unknown step', step_id=None)\n    return 1\n"


async def test_the_question_goes_back_once_and_the_fixed_answer_is_submitted(workspace, monkeypatch):  # noqa: F811
    monkeypatch.setattr(DemoMiner, "_author_suite", no_suite)
    provider = Scripted(
        edit("src/b.py", "def b():\n    return 1\n", DROPS), RUN, FINISHED,
        edit("src/b.py", "'unknown step', step_id=None)", "'unknown step', step_id=step_id)"), RUN, FINISHED,
        DONE,
    )
    _, submission, notes = await run_loop(workspace, provider, ScriptedSandbox(result(0), result(0)))
    asked = provider.seen[3][-1]["content"]
    assert asked.startswith("An error is raised here without a value the function was given:")
    assert "- src/b.py: `raise KeyError('unknown step', step_id=None)`" in asked
    assert sum("without a value the function was given" in str(m[-1]["content"]) for m in provider.seen) == 1
    assert provider.seen[6][-1]["content"].startswith("Before this is submitted")
    assert b"step_id=step_id)" in submission and notes.diff_questions == lint.DROPPED_ARGUMENT
