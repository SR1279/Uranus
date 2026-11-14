"""The five steps of the first deploy (plan.md steps 1, 2, 3, 5 and 6) working together in one solve.

Each step has its own tests; these run one scripted solve through every stage they added or changed, so that a
merge that breaks how the stages fit together (the clock on each message, the conversation staying append-only,
a refused stage withdrawn with its reply, the settle rules agreeing with the system message) fails here.
"""

from __future__ import annotations

import time
from dataclasses import replace

from rlvr.neurons import adjudicate
from rlvr.neurons.demo_miner import VALUE_STRING_RULE, DemoMiner, SolveNotes
from rlvr.v3.miner_workspace import WorkspaceReader
from tests.test_acceptance import documented, documented_task, fixed_author, result, suite_ran  # noqa: F401
from tests.test_adjudicate import unclocked
from tests.test_demo_miner import WalletLike, completion_for, completion_with_calls, settings
from tests.test_demo_miner_workbench import (  # noqa: F401
    DONE,
    FINISHED,
    FakeSandbox,
    Scripted,
    clock_figures,
    edit,
    sent_bytes,
    workspace,
)

# A builtin subclass (lint.questions: hybrid_type) in a non-test Python file.
HYBRID = edit("src/b.py", "return 1", "return 2\n\n\nclass Rendered(str):\n    pass")
OWN_TEST = completion_with_calls(("run_command", {"command": "python3 /tmp/t.py", "timeout_s": 30}))
TRUE_RUN = completion_with_calls(("run_command", {"command": "true", "timeout_s": 10}))
# The first finished reply names a reading under a choices heading that FALLBACK_RULE decides.
CHOICES = completion_for("Added the limit.\n\nWhere the doc was silent, I chose:\n- A missing limit falls back to 10.\n")
REFUSED = replace(completion_for(""), reasoning="", finish_reason="content_filter", refusal="blocked")


class DeploySandbox(FakeSandbox):
    """The suite fails its first run and passes after; every other command, and a rerun, prints ok."""

    def __init__(self):
        super().__init__()
        self.commands: list[tuple[str, bool]] = []
        self.suite_runs = 0

    async def run(self, request, *, wait_s):
        self.requests.append(request)
        command, watched = request.argv[6], request.argv[-1] == "watch"
        self.commands.append((command, watched))
        if command.startswith("sha256sum --"):
            self.suite_runs += 1
            if self.suite_runs == 1:
                return suite_ran(1, b"test_limit_is_inclusive ... FAIL\nFAILED (failures=1)\n")
            return suite_ran(0, b"OK\n")
        return result(0, b"ok\n")


async def test_every_stage_of_the_first_deploy_in_one_solve(documented, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", fixed_author())
    sandbox = DeploySandbox()
    provider = Scripted(HYBRID, OWN_TEST, edit(), TRUE_RUN, CHOICES, FINISHED, FINISHED, REFUSED, DONE)
    notes = SolveNotes()
    # The production defaults of the new settings (diff questions on, own tests recorded, choice check on); the
    # readers stay off, as in production.
    miner = DemoMiner(settings(miner_adjudicator=False), provider, wallet=WalletLike("miner"), sandbox=sandbox)
    assert (miner.settings.miner_lint, miner.settings.miner_own_tests, miner.settings.miner_choice_check) == (
        True, "record", True)
    _, events, submission = await miner._generate(
        documented_task(), WorkspaceReader(documented), time.monotonic() + 900.0, notes=notes
    )
    assert len(provider.seen) == 9 and not provider.replies
    assert b"gamma" in submission and b"class Rendered(str):" in submission

    question, reconcile, choice, review = (provider.seen[index][-1]["content"] for index in (5, 6, 7, 8))
    # Step 3: the diff question goes back at the first finish, before the suite.
    assert question.startswith("Questions from a check of your change") and "class Rendered(str):" in question
    assert notes.diff_questions == "hybrid_type"
    # Step 1: the reconcile message names the settling window and the time to the wrap-up. It holds the solver's
    # own percent-encoding clause, never the writer's (prompt audit finding 7).
    assert reconcile.startswith("Before your work is reviewed, an acceptance suite written independently")
    assert "seconds are set aside for settling these cases, out of about" in reconcile
    assert reconcile.count(adjudicate.ENCODING_RULE) == 1 and "never +" not in reconcile
    # Step 6: the matched choice is quoted back once; refused, it is withdrawn with its reply and the review follows.
    assert unclocked(choice) == adjudicate.choice_prompt(adjudicate.match_choices(adjudicate.named_choices(CHOICES.output)))
    assert notes.choice_check == "matched fallback (1 of 1 bullets)" and notes.refused == "choice check"
    assert review.startswith("Before this is submitted")

    # Step 1: every request ends with exactly one clock line, and no message carries a second one.
    for request in provider.seen:
        assert clock_figures(request[-1]) is not None
        assert all(str(message.get("content")).count("Clock:") <= 1 for message in request)
    # One clock per message that ended a request; the withdrawn choice message took its clock with it.
    clocks = [sum("Clock:" in str(message.get("content")) for message in request) for request in provider.seen]
    assert clocks == [1, 2, 3, 4, 5, 6, 7, 8, 8]
    # Append-only: every request repeats the one before it byte for byte, except that the refused choice message
    # is replaced by the review after the same reply.
    for index, (previous, current) in enumerate(zip(provider.seen, provider.seen[1:]), 1):
        if index == 8:
            assert sent_bytes(current[:-1]) == sent_bytes(previous[:-1])
        else:
            assert sent_bytes(current[:len(previous)]) == sent_bytes(previous)
    # The cached prefix: one system message, the same in every request, with the instruction after it.
    assert all(sent_bytes(request[:2]) == sent_bytes(provider.seen[0][:2]) for request in provider.seen)
    assert [message["role"] for message in provider.seen[-1]].count("system") == 1
    # Step 2, in the system message this solve sends: the settle rules above agree with it only if it holds the same
    # percent-encoding clause, once, and none of the writer's (finding 7); finding 3 is there once, followed by the
    # live spelling sentence as its stated exception (second merge, first-deploy-2). The live texts of 00:22 (Nov 14)
    # sit beside step 2's, once each, so a later merge that drops either side fails here too.
    system = provider.seen[0][0]["content"]
    assert system.count(adjudicate.ENCODING_RULE) == 1 and "never +" not in system
    assert system.count("and give each place a case of its own in your test. A spelling the documentation gives") == 1
    assert system.count("is not a rule of the value's kind: it belongs to that format's parser alone.") == 1
    for live in (VALUE_STRING_RULE, "find that kind with errors.Is or errors.As on each error's chain",
                 "keeps every check of build, a wrong-type x included, ahead of x.commit"):
        assert system.count(live) == 1, live[:60]
    # The live Python checks (argparse private state, then error attributes) still run on this Python answer after
    # step 3's question has taken one of the shared lint.MAX_PROMPTS, and find nothing.
    assert notes.lint == "passed" and notes.lint_findings == 0

    # Step 5: after the review and the final check's clean-copy run, the earlier test runs once more on the final
    # version, with nothing sent to the model; it now prints ok, so nothing is newly broken.
    assert notes.clean_copy == "same"
    assert notes.own_tests == "1 rerun, none newly broken"
    reruns = [index for index, (_command, watched) in enumerate(sandbox.commands) if not watched]
    suite_runs = [index for index, (command, _watched) in enumerate(sandbox.commands) if command.startswith("sha256sum --")]
    assert [sandbox.commands[index][0] for index in reruns] == ["python3 /tmp/t.py"] and reruns[0] > max(suite_runs)
    assert [event["call_id"] for event in events if event["event_type"] == "tool_call"][-1] == "own-test-1"

    # Step 1's records: the first finish came with the fifth reply, and four model calls followed it.
    assert notes.first_finish_s is not None and notes.turns_after_finish == 4
    fields = notes.fields()
    for name in ("first_finish_s", "turns_after_finish", "diff_questions", "own_tests", "choice_check", "refused"):
        assert fields[name] == getattr(notes, name), name
