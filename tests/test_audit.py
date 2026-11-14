"""The stronger reader's pass over the finished diff, and how its findings reach the implementer."""

from __future__ import annotations

import httpx

from rlvr.neurons import audit
from rlvr.neurons.demo_miner import DemoMiner
from tests.test_acceptance import INSTRUCTION, RUN, documented, documented_task, run_loop  # noqa: F401
from tests.test_adjudicate import LISTING, RULINGS
from tests.test_demo_miner import completion_for
from tests.test_demo_miner_workbench import DONE, FINISHED, FakeSandbox, Scripted, edit, workspace  # noqa: F401

FINDINGS = (
    "1. **RULE:** \"A limit of 0 is invalid\" | CALL: Widget(limit=0) | EXPECTED: ValueError | ACTUAL: accepted\n"
    "   by `Widget.__init__`, which checks only the upper bound\n"
    "2. RULE: \"1..10 inclusive\" | CALL: Widget(limit=10) | EXPECTED: accepted | ACTUAL: rejected by `check`\n"
)
OPUS = "anthropic/claude-opus-5.5"


class AuditedProvider(Scripted):
    """The implementer's scripted replies, with the stronger model's calls answered outside the conversation."""

    def __init__(self, *replies, audit_reply=FINDINGS, rulings=RULINGS, fail=False):
        super().__init__(*replies)
        self.audit_reply = audit_reply
        self.rulings = rulings
        self.fail = fail
        self.audits = []

    async def complete(self, messages, *, timeout_s, tools=(), **options):
        if options.get("model") == OPUS:
            if messages[0]["content"] == audit.AUDITOR_SYSTEM_PROMPT:
                self.audits.append((messages, options))
                if self.fail:
                    raise httpx.ConnectError("unreachable")
                return completion_for(self.audit_reply)
            return completion_for(self.rulings)
        return await super().complete(messages, timeout_s=timeout_s, tools=tools, **options)


async def no_suite(self, request, docs, deadline):
    return None, None


def test_findings_are_parsed_with_their_continuation_lines_and_capped():
    assert audit.parse_findings(FINDINGS) == [
        "RULE: \"A limit of 0 is invalid\" | CALL: Widget(limit=0) | EXPECTED: ValueError | ACTUAL: accepted "
        "by `Widget.__init__`, which checks only the upper bound",
        "RULE: \"1..10 inclusive\" | CALL: Widget(limit=10) | EXPECTED: accepted | ACTUAL: rejected by `check`",
    ]
    assert audit.parse_findings("NONE") == [] and audit.parse_findings("") == [] and audit.parse_findings("1. NONE.") == []
    many = "\n".join(f"{n}. RULE: r{n} | CALL: c | EXPECTED: e | ACTUAL: a" for n in range(1, 20))
    assert len(audit.parse_findings(many)) == audit.MAX_FINDINGS
    long = audit.parse_findings("1. " + "x" * 2000)[0]
    assert len(long) == audit.FINDING_CHARS + 4 and long.endswith(" ...")


def test_the_reviewer_leaves_out_points_a_passage_of_the_documentation_supports():
    # 2026-11-11, JSON Patch: a general precedence list against an operation's own steps broke a correct answer.
    assert "no passage of it supports what the code does now" in audit.AUDITOR_SYSTEM_PROMPT
    assert "an operation's own description decides for that operation over a general list" in audit.AUDITOR_SYSTEM_PROMPT


def test_the_reviewer_sees_the_task_the_documentation_and_the_diff_only():
    messages = audit.auditor_messages("Build widgets", {"docs/widgets.md": "rule text"}, "+new line\n")
    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[0]["content"] == audit.AUDITOR_SYSTEM_PROMPT
    user = messages[1]["content"]
    assert "Build widgets" in user and "docs/widgets.md" in user and "rule text" in user and "+new line" in user
    big = audit.auditor_messages("t", {}, "x" * (audit.MAX_DIFF_CHARS + 10))[1]["content"]
    assert "the rest of the diff is omitted" in big


def test_the_findings_prompt_asks_for_checks_by_running_and_avoids_the_refused_wordings():
    text = audit.findings_prompt(["RULE: a | CALL: f(0) | EXPECTED: error | ACTUAL: accepted"])
    assert text.startswith("Before the review, a second reader went through the documentation and your change")
    assert "1. RULE: a | CALL: f(0)" in text and "run its CALL" in text and "They may be wrong" in text
    assert "keep your code" in text and "senior" not in text
    assert "model" not in text.lower() and "ruled" not in text.lower() and "you have called" not in text.lower()


async def test_findings_go_back_after_the_finish_and_before_the_review(documented, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", no_suite)
    provider = AuditedProvider(edit(), RUN, FINISHED, edit("src/b.py", "return 1", "return 2"), RUN, FINISHED, DONE)
    events, submission, notes = await run_loop(documented, provider, FakeSandbox(), miner_audit=True)
    assert b"gamma" in submission and b"return 2" in submission
    sent = provider.seen[3][-1]["content"]
    assert sent.startswith("Before the review, a second reader") and "Widget(limit=0)" in sent
    assert provider.seen[6][-1]["content"].startswith("Before this is submitted")
    assert notes.audit == "2 findings" and len(provider.audits) == 1
    messages, options = provider.audits[0]
    assert "gamma" in messages[1]["content"] and "1..10 inclusive" in messages[1]["content"]
    assert options["reasoning_effort"] == "medium" and options["max_tokens"] == audit.MAX_TOKENS
    assert any(event["event_type"] == "model_turn" for event in events)


async def test_no_findings_or_a_failed_call_go_straight_to_the_review(documented, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", no_suite)
    provider = AuditedProvider(edit(), RUN, FINISHED, DONE, audit_reply="NONE")
    _, submission, notes = await run_loop(documented, provider, FakeSandbox(), miner_audit=True)
    assert notes.audit == "no findings" and provider.seen[3][-1]["content"].startswith("Before this is submitted")
    provider = AuditedProvider(edit(), RUN, FINISHED, DONE, fail=True)
    _, submission, notes = await run_loop(documented, provider, FakeSandbox(), miner_audit=True)
    assert notes.audit == "auditor failed" and b"gamma" in submission
    assert provider.seen[3][-1]["content"].startswith("Before this is submitted")


async def test_the_audit_follows_the_readings_when_both_run(documented, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", no_suite)
    provider = AuditedProvider(edit(), RUN, FINISHED, LISTING, FINISHED, FINISHED, DONE)
    _, _, notes = await run_loop(documented, provider, FakeSandbox(), miner_audit=True, miner_adjudicator=True)
    stages = [seen[-1]["content"][:40] for seen in provider.seen[3:]]
    assert stages[0].startswith("Before the review, which questions")
    assert stages[1].startswith("Before the review, a colleague")
    assert stages[2].startswith("Before the review, a second reader")
    assert stages[3].startswith("Before this is submitted")
    assert notes.adjudication == "2 readings, 1 overruled" and notes.audit == "2 findings"


async def test_nothing_is_audited_off_without_documentation_or_time(documented, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", no_suite)
    provider = AuditedProvider(edit(), RUN, FINISHED, DONE)
    _, _, notes = await run_loop(documented, provider, FakeSandbox(), miner_audit=False)
    assert notes.audit == "" and provider.audits == []
    (documented / "docs" / "widgets.md").unlink()  # the task names it, but it is not in the workspace
    provider = AuditedProvider(edit(), RUN, FINISHED, DONE)
    _, _, notes = await run_loop(documented, provider, FakeSandbox(), miner_audit=True)
    assert notes.audit == "" and provider.audits == []
    (documented / "docs" / "widgets.md").write_text("# Widgets\n\nA limit of 0 is invalid; 1..10 inclusive.\n")
    provider = AuditedProvider(edit(), RUN, FINISHED, DONE)
    _, _, notes = await run_loop(documented, provider, FakeSandbox(), miner_audit=True, deadline=300.0)
    assert notes.audit == "not run: too little time left" and provider.audits == []


async def test_a_refused_findings_prompt_is_withdrawn(documented, monkeypatch):
    import dataclasses

    monkeypatch.setattr(DemoMiner, "_author_suite", no_suite)
    refused = dataclasses.replace(completion_for(""), reasoning="", finish_reason="content_filter", refusal="blocked")
    provider = AuditedProvider(edit(), RUN, FINISHED, refused, DONE)
    _, submission, notes = await run_loop(documented, provider, FakeSandbox(), miner_audit=True)
    assert notes.refused == "audit" and b"gamma" in submission
    assert provider.seen[4][-1]["content"].startswith("Before this is submitted")
