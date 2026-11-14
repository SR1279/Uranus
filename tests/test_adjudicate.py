"""The model's open questions about the documentation, answered by a fresh reader before the review."""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import pytest

from rlvr.neurons import adjudicate
from rlvr.neurons.adjudicate import Question, Ruling, adjudicator_messages, parse_questions, parse_rulings, verdict_prompt
from rlvr.neurons.demo_miner import DemoMiner, SolveNotes
from rlvr.v3.miner_workspace import WorkspaceReader
from tests.test_acceptance import INSTRUCTION, RUN, documented, documented_task  # noqa: F401
from tests.test_demo_miner import WalletLike, completion_for, settings
from tests.test_demo_miner_workbench import DONE, FINISHED, FakeSandbox, Scripted, edit, workspace  # noqa: F401

LISTING = completion_for(
    "These are the questions I would put to the author.\n\n"
    "1. **MaxMessageBytes.** Does it count every byte of the request as fed, or only the field contents?\n"
    "   My code counts field contents without separators.\n"
    "2. **Two failing requests in one Feed.** Which error is returned? My code returns the first.\n"
)
OLD_FORM = (
    "1. QUESTION: what MaxMessageBytes counts | CHOSEN: field contents without separators | "
    "ALTERNATIVE: every byte of the request as fed\n"
    "2. QUESTION: which error when two requests in one Feed fail | CHOSEN: the first | ALTERNATIVE: the last\n"
)
RULINGS = (
    "1. RULING: every byte of the request as fed | WHY: \"the sum of the request line, header bytes, body bytes "
    "and trailer bytes\"; a test sets the limit to len(wire) | AGREES: no\n"
    "2. RULING: the first | WHY: not stated; stricter reading | AGREES: yes\n"
)
REFUSED = dataclasses.replace(
    completion_for(""), reasoning="", finish_reason="content_filter",
    refusal="This request was blocked as it seems to violate Anthropic's Terms of Service",
)


def test_numbered_questions_are_parsed_with_their_continuation_lines():
    questions = parse_questions(LISTING.output)
    assert [q.number for q in questions] == [1, 2]
    assert questions[0].question == (
        "MaxMessageBytes. Does it count every byte of the request as fed, or only the field contents? "
        "My code counts field contents without separators."
    )
    assert questions[0].chosen == "" and questions[1].question.endswith("My code returns the first.")
    assert parse_questions("NONE") == [] and parse_questions("None.\n") == [] and parse_questions("") == []
    assert parse_questions("I think everything is clear.") == []
    assert [q.number for q in parse_questions("1) a\n1) b\n3. c\n")] == [1, 3]
    assert len(parse_questions("\n".join(f"{n}. question {n}" for n in range(1, 40)))) == adjudicate.MAX_QUESTIONS


def test_the_earlier_fixed_form_is_still_read():
    assert parse_questions(OLD_FORM) == [
        Question(1, "what MaxMessageBytes counts", "field contents without separators", "every byte of the request as fed"),
        Question(2, "which error when two requests in one Feed fail", "the first", "the last"),
    ]
    assert parse_questions("1. QUESTION: a | CHOSEN: b | ALTERNATIVE: c\n1. QUESTION: d | CHOSEN: e | ALTERNATIVE: f\n") == [Question(1, "a", "b", "c")]


def test_rulings_are_matched_to_the_questions_by_number():
    questions = parse_questions(LISTING.output)
    rulings = parse_rulings(RULINGS, questions)
    assert [(r.number, r.agrees) for r in rulings] == [(1, False), (2, True)]
    assert rulings[0].ruling == "every byte of the request as fed" and "len(wire)" in rulings[0].why
    assert parse_rulings("3. RULING: x | WHY: y | AGREES: no\n", questions) == []
    assert parse_rulings("nothing in the form", questions) == []


def test_the_reader_sees_the_documentation_and_the_questions_but_no_code():
    questions = parse_questions(LISTING.output)
    messages = adjudicator_messages(INSTRUCTION, {"docs/widgets.md": "# Widgets\nthe rule text"}, questions)
    system, user = messages[0]["content"], messages[1]["content"]
    assert "never the code" in system and "stricter reading" in system and "speaks to the disputed case more directly" in system
    assert "measured on the input as given" in system
    assert "the rule text" in user and "Resumable widgets" in user
    assert "1. MaxMessageBytes. Does it count every byte" in user and "2. Two failing requests" in user
    old = adjudicator_messages(INSTRUCTION, {}, parse_questions(OLD_FORM))[1]["content"]
    assert "1. what MaxMessageBytes counts The code assumes: field contents without separators. The other reading: every byte" in old


def test_the_prompts_the_engineer_sees_avoid_what_the_host_refuses():
    # Replayed against the refused conversations of 2026-11-10: a request to catalogue the model's own earlier
    # statements, or a description of another model call ruling on its output, is refused as reverse engineering.
    for text in (adjudicate.LIST_PROMPT, verdict_prompt(parse_questions(LISTING.output), parse_rulings(RULINGS, parse_questions(LISTING.output)))):
        assert "model" not in text.lower() and "you have called" not in text and "RULING" not in text
    assert adjudicate.LIST_PROMPT.startswith("Before the review, which questions would you put to the author")
    assert "Do not call any tool" in adjudicate.LIST_PROMPT


def test_the_verdict_marks_the_points_that_differ_and_says_what_to_do():
    questions = parse_questions(LISTING.output)
    text = verdict_prompt(questions, parse_rulings(RULINGS, questions))
    assert text.startswith("Before the review, a colleague who read only the documentation and your questions")
    assert "1. MaxMessageBytes. Does it count" in text and "They read it as: every byte of the request as fed" in text
    assert "This differs from your reading." in text and "This matches your reading." in text
    assert "change your code and your throwaway test to the colleague's reading" in text and "run your tests again" in text
    old = parse_questions(OLD_FORM)
    assert "This differs from your reading (field contents without separators)." in verdict_prompt(old, parse_rulings(RULINGS, old))


def test_a_long_question_is_shortened_when_quoted_back():
    long = parse_questions("1. " + "word " * 300)
    text = verdict_prompt(long, parse_rulings("1. RULING: x | WHY: y | AGREES: no", long))
    assert "…" in text and len(text) < 1500


async def test_the_reader_call_uses_its_own_model_budget_and_effort(documented):
    seen = {}

    class Provider:
        async def complete(self, messages, *, timeout_s, tools=(), **options):
            seen.update(options)
            return completion_for(RULINGS)

    miner = DemoMiner(settings(), Provider(), wallet=WalletLike("miner"))
    notes, events = SolveNotes(), []
    verdict = await miner._adjudicate(documented_task(), {"docs/widgets.md": "rules"}, LISTING.output, events, notes.spend, notes, time.monotonic() + 600)
    assert "This differs from your reading" in verdict and notes.adjudication == "2 readings, 1 overruled"
    assert seen == {"model": "anthropic/claude-opus-5.5", "max_tokens": adjudicate.MAX_TOKENS, "reasoning_effort": "medium"}
    assert len(events) == 1 and events[0]["event_type"] == "model_turn"
    seen.clear()
    miner = DemoMiner(settings(miner_adjudicator_model=""), Provider(), wallet=WalletLike("miner"))
    await miner._adjudicate(documented_task(), {"docs/widgets.md": "rules"}, LISTING.output, [], notes.spend, notes, time.monotonic() + 600)
    assert seen["model"] is None  # the same model as the answer's


async def run_loop(workspace, provider, sandbox, *, request=None, deadline=900.0, **updates):
    notes = SolveNotes()
    miner = DemoMiner(settings(**updates), provider, wallet=WalletLike("miner"), sandbox=sandbox)
    _, events, submission = await miner._generate(
        request or documented_task(), WorkspaceReader(workspace), time.monotonic() + deadline, notes=notes
    )
    return events, submission, notes


async def no_suite(self, request, docs, deadline):
    return None, None


class ReaderProvider(Scripted):
    """The engineer's scripted replies, with the reader's call answered outside the conversation."""

    def __init__(self, *replies, rulings=RULINGS):
        super().__init__(*replies)
        self.rulings = rulings

    async def complete(self, messages, *, timeout_s, tools=(), **options):
        if options.get("model") == "anthropic/claude-opus-5.5":
            return completion_for(self.rulings)
        return await super().complete(messages, timeout_s=timeout_s, tools=tools, **options)


async def test_differing_answers_go_back_before_the_review(documented, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", no_suite)
    provider = ReaderProvider(edit(), RUN, FINISHED, LISTING, FINISHED, DONE)
    _, submission, notes = await run_loop(documented, provider, FakeSandbox())
    assert b"gamma" in submission
    assert provider.seen[3][-1]["content"].startswith("Before the review, which questions would you put to the author")
    verdict = provider.seen[4][-1]["content"]
    assert verdict.startswith("Before the review, a colleague who read only the documentation") and "This differs" in verdict
    assert provider.seen[5][-1]["content"].startswith("Before this is submitted")
    assert notes.adjudication == "2 readings, 1 overruled" and notes.acceptance == "author failed"


async def test_agreed_or_absent_questions_add_no_turn(documented, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", no_suite)
    provider = ReaderProvider(edit(), RUN, FINISHED, LISTING, DONE, rulings=RULINGS.replace("AGREES: no", "AGREES: yes"))
    _, submission, notes = await run_loop(documented, provider, FakeSandbox())
    assert b"gamma" in submission and notes.adjudication == "2 readings, 0 overruled"
    assert provider.seen[4][-1]["content"].startswith("Before this is submitted")
    provider = Scripted(edit(), RUN, FINISHED, completion_for("NONE"), DONE)
    _, submission, notes = await run_loop(documented, provider, FakeSandbox())
    assert b"gamma" in submission and notes.adjudication == "no open readings listed"
    assert provider.seen[4][-1]["content"].startswith("Before this is submitted")


async def test_without_documentation_or_when_off_nothing_is_asked(workspace, documented, monkeypatch):
    from tests.test_demo_miner import task

    monkeypatch.setattr(DemoMiner, "_author_suite", no_suite)
    _, _, notes = await run_loop(workspace, Scripted(edit(), RUN, FINISHED, DONE), FakeSandbox(), request=task())
    assert notes.adjudication == ""
    _, _, notes = await run_loop(documented, Scripted(edit(), RUN, FINISHED, DONE), FakeSandbox(), miner_adjudicator=False)
    assert notes.adjudication == ""


async def test_a_refused_prompt_is_withdrawn_and_the_solve_goes_on_to_the_review(documented, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", no_suite)
    provider = ReaderProvider(edit(), RUN, FINISHED, REFUSED, DONE)
    events, submission, notes = await run_loop(documented, provider, FakeSandbox())
    assert b"gamma" in submission and notes.ended_by == "finished" and notes.reviewed
    listing, review = provider.seen[3], provider.seen[4]
    assert listing[-1]["content"].startswith("Before the review, which questions")
    # The refused prompt and the finished reply it followed are gone; the review follows that reply instead.
    assert review[-1]["content"].startswith("Before this is submitted") and len(review) == len(listing)
    assert review[:-2] == listing[:-2] and review[-2]["content"] == FINISHED.output
    assert notes.refused == "open points" and notes.adjudication == ""
    # The refused turn is still a recorded, billed model turn.
    assert sum(e["event_type"] == "model_turn" for e in events) == 5


async def test_a_refused_verdict_or_review_is_withdrawn_too(documented, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", no_suite)
    provider = ReaderProvider(edit(), RUN, FINISHED, LISTING, REFUSED, REFUSED)
    _, submission, notes = await run_loop(documented, provider, FakeSandbox())
    assert b"gamma" in submission and notes.ended_by == "finished"
    assert provider.seen[4][-1]["content"].startswith("Before the review, a colleague")
    assert provider.seen[5][-1]["content"].startswith("Before this is submitted")
    assert len(provider.seen) == 6 and notes.refused == "second reading, review"
    assert notes.adjudication == "2 readings, 1 overruled"


async def test_a_refusal_of_the_models_own_work_is_retried_once_then_the_work_is_submitted(documented, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", no_suite)
    provider = Scripted(edit(), REFUSED, REFUSED)
    _, submission, notes = await run_loop(documented, provider, FakeSandbox())
    assert b"gamma" in submission and notes.ended_by == "model_refused" and notes.refused == ""
    assert provider.seen[1] == provider.seen[2]  # the same request, once more
    provider = Scripted(edit(), REFUSED, RUN, FINISHED, completion_for("NONE"), DONE)
    _, submission, notes = await run_loop(documented, provider, FakeSandbox())
    assert b"gamma" in submission and notes.ended_by == "finished"


async def test_a_reply_cut_at_the_token_limit_is_continued(documented, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", no_suite)
    cut = dataclasses.replace(completion_for("I will now edit the"), finish_reason="length")
    provider = Scripted(cut, edit(), RUN, FINISHED, completion_for("NONE"), DONE)
    _, submission, notes = await run_loop(documented, provider, FakeSandbox())
    assert b"gamma" in submission and notes.cut_off == 1
    assert provider.seen[1][-1]["content"].startswith("That reply was cut at the token limit")
    assert provider.seen[1][-2] == {"role": "assistant", "content": "I will now edit the", "reasoning": "Inspect the file."}


def test_names_with_double_underscores_survive_the_cleanup():
    # Only ** bold is stripped: __init__.py and __all__ are names, not emphasis.
    questions = parse_questions("1. **Does `pkg/__init__.py` re-export `Thing` in `__all__`?** My code assumes yes.\n")
    assert questions[0].question.startswith("Does `pkg/__init__.py` re-export `Thing` in `__all__`?")


def test_the_stricter_reading_has_its_two_limits_everywhere_it_is_taught():
    # 2026-11-11: the strict reading was stretched to refuse JSON whitespace and member order in a decoder whose
    # rules were headed "canonical" (a6da5e0782c4) and to make absent record fields a schema error that hid the
    # documented "stored record invalid" error (9b572d481b23). Every prompt that teaches the strict reading
    # carries both limits; the missing member still fails, only its error is routed.
    from rlvr.neurons import acceptance
    from rlvr.neurons.demo_miner import REPOSITORY_SYSTEM_PROMPT

    limit = adjudicate.STRICT_LIMIT
    assert "whitespace is not trailing content" in limit and "members in any order" in limit
    assert "still fails" in limit and "zero value" in limit
    assert "never invents" not in limit  # the broad first wording loosened 13 strict rulings on passed answers
    for prompt in (adjudicate.ADJUDICATOR_SYSTEM_PROMPT, acceptance.AUTHOR_SYSTEM_PROMPT, REPOSITORY_SYSTEM_PROMPT):
        assert limit in prompt
    reconcile = acceptance.reconcile_prompt(acceptance.Suite("go test ./x/", "/work/x/a_test.go", ""), "FAIL", 1)
    assert "whitespace is not trailing content" in reconcile and "the suite's error" in reconcile
    assert "padded with spaces to exactly its size limit" in acceptance.AUTHOR_SYSTEM_PROMPT


def test_go_error_values_are_returned_in_pointer_form():
    # The larksuite SDK's hidden tests match err.(*larkcore.CodeError) and errors.As against *CodeError; two
    # answers that returned the embedded value failed after the reader confirmed the value (ace15dab231c,
    # 9b572d481b23).
    from rlvr.neurons import acceptance
    from rlvr.neurons.demo_miner import REPOSITORY_SYSTEM_PROMPT

    assert adjudicate.GO_ERROR_FORM in adjudicate.ADJUDICATOR_SYSTEM_PROMPT
    assert adjudicate.GO_ERROR_FORM in acceptance.AUTHOR_SYSTEM_PROMPT
    # The solver keeps the general rule; the larksuite example left its prompt (prompt audit finding 10), because
    # lint.check now catches a returned resp.CodeError value mechanically.
    assert ("an error is returned and stored in the form the project's own code returns that type (a pointer where "
            "the project returns &T{...}), because the tests match it with err.(*T) or errors.As.") in REPOSITORY_SYSTEM_PROMPT
    assert "resp.CodeError" not in REPOSITORY_SYSTEM_PROMPT
    assert "interface{} rather than any" in acceptance.AUTHOR_SYSTEM_PROMPT


def test_a_value_keeps_its_one_documented_type_everywhere_readings_are_taught():
    # a9a5461324c6 (2026-11-11 18:01): a violation's `report` was documented as "the deterministic report dict"
    # and, for raw logs, as the committed prefix "rendered with the supplied options". The answer stored the text,
    # then a str subclass that also answered report["key"], and three readers approved it under "requirements
    # that seem to conflict both hold"; the hidden test compared it with == to json.loads(render_contract_report()).
    from rlvr.neurons import acceptance
    from rlvr.neurons.demo_miner import REPOSITORY_SYSTEM_PROMPT, _review_prompt

    rule = adjudicate.ONE_TYPE
    assert "exactly one type" in rule and "json.loads of the rendered text" in rule and "str subclass" in rule
    review = _review_prompt(False, b"--- a/x.py\n+++ b/x.py\n@@ -0,0 +1 @@\n+x = 1\n")
    for prompt in (adjudicate.ADJUDICATOR_SYSTEM_PROMPT, acceptance.AUTHOR_SYSTEM_PROMPT, REPOSITORY_SYSTEM_PROMPT, review):
        assert rule in prompt
    # "Both hold" is about behaviour over time and no longer reaches a value's type.
    for prompt in (adjudicate.ADJUDICATOR_SYSTEM_PROMPT, REPOSITORY_SYSTEM_PROMPT):
        assert "requirements on behaviour that seem to conflict both hold" in prompt.lower()
        assert "Requirements that seem to conflict both hold" not in prompt
    reconcile = acceptance.reconcile_prompt(acceptance.Suite("python3 -m unittest t", "/tmp/t.py", ""), "FAIL", 1)
    assert "never made two types at once" in reconcile


def test_the_majority_of_the_readers_decides_each_question_and_a_tie_keeps_the_engineers_reading():
    def run(*agrees):
        return [Ruling(number, f"ruling {number} {'yes' if ok else 'no'}", "why", ok) for number, ok in enumerate(agrees, 1)]

    combined = adjudicate.combine_rulings([run(True, False, False), run(True, False, True), run(False, True, True)])
    assert [ruling.agrees for ruling in combined] == [True, False, True]
    assert combined[1].ruling == "ruling 2 no"  # the first of the majority
    # Two readers left (one failed): a split keeps what the engineer's code assumes.
    assert [ruling.agrees for ruling in adjudicate.combine_rulings([run(True, False), run(False, False)])] == [True, False]
    # A question only some readers answered is decided by those that did.
    assert [r.agrees for r in adjudicate.combine_rulings([run(False), run(False, True)])] == [False, True]


async def test_three_readers_rule_and_the_majority_goes_back(documented, monkeypatch):
    monkeypatch.setattr(adjudicate, "VOTE_STAGGER_S", 0.0)
    # The engineer's second reading is overruled by one reader of three: the majority keeps it.
    lone = "1. RULING: zero is invalid | WHY: \"A limit of 0 is invalid\" | AGREES: yes\n2. RULING: exclusive | WHY: stricter reading | AGREES: no\n"
    agreeing = "1. RULING: zero is invalid | WHY: \"A limit of 0 is invalid\" | AGREES: yes\n2. RULING: inclusive | WHY: \"1..10 inclusive\" | AGREES: yes\n"
    replies = [lone, agreeing, agreeing]
    calls = []

    class Provider:
        async def complete(self, messages, *, timeout_s, tools=(), **options):
            calls.append(options["model"])
            return completion_for(replies.pop(0))

    miner = DemoMiner(settings(miner_adjudicator_votes=3), Provider(), wallet=WalletLike("miner"))
    notes, events = SolveNotes(), []
    verdict = await miner._adjudicate(documented_task(), {"docs/widgets.md": "rules"}, LISTING.output, events, notes.spend, notes, time.monotonic() + 600)
    assert verdict is None and notes.adjudication == "2 readings, 0 overruled by 3 readers"
    assert calls == ["anthropic/claude-opus-5.5"] * 3 and len(events) == 3


async def test_a_failed_reader_drops_out_of_the_vote(documented, monkeypatch):
    monkeypatch.setattr(adjudicate, "VOTE_STAGGER_S", 0.0)
    replies = [RuntimeError("host down"), RULINGS, RULINGS]

    class Provider:
        async def complete(self, messages, *, timeout_s, tools=(), **options):
            reply = replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return completion_for(reply)

    miner = DemoMiner(settings(miner_adjudicator_votes=3), Provider(), wallet=WalletLike("miner"))
    notes, events = SolveNotes(), []
    verdict = await miner._adjudicate(documented_task(), {"docs/widgets.md": "rules"}, LISTING.output, events, notes.spend, notes, time.monotonic() + 600)
    assert "This differs from your reading" in verdict and notes.adjudication == "2 readings, 1 overruled by 2 readers"
    assert len(events) == 2
    replies[:] = [RuntimeError("a"), RuntimeError("b"), RuntimeError("c")]
    verdict = await miner._adjudicate(documented_task(), {"docs/widgets.md": "rules"}, LISTING.output, [], notes.spend, notes, time.monotonic() + 600)
    assert verdict is None and notes.adjudication == "2 readings, reader failed"


def test_the_implicit_rules_reach_the_implementer_the_suite_author_and_the_reader():
    # 27 kept hidden suites check the same implicit things in task after task; each clause failed a live answer
    # or recurs in several suites (8a8f1ffe85b6, d3ae29cad54b, 77faeb4213a5, c8c45028ca97, 0a2b25a81843,
    # 085b42460719, ae3f85baaf88).
    from rlvr.neurons import acceptance
    from rlvr.neurons.demo_miner import IMPLICIT_RULES, REPOSITORY_RUN_PROMPT, REPOSITORY_SYSTEM_PROMPT

    assert IMPLICIT_RULES in REPOSITORY_SYSTEM_PROMPT
    for clause in ("looked up on its object or module at call time", "No shortcut skips validation",
                   "never repaired by rebuilding one", "afresh at", "even when the rest is missing",
                   "names the line where the problem is found", "installer payload",
                   "Traverse recursive structures iteratively",
                   "wrapped with %w", "json.loads accepts duplicate keys",
                   "exactly what the documentation puts on stdout and on stderr", "run it in a finally"):
        assert clause in IMPLICIT_RULES, clause
    assert "a streamed input fed one byte at a time" in REPOSITORY_RUN_PROMPT
    assert "helpers swapped after import" in acceptance.AUTHOR_SYSTEM_PROMPT
    assert adjudicate.READING_RULES in adjudicate.ADJUDICATOR_SYSTEM_PROMPT
    assert "model" not in IMPLICIT_RULES.lower() and "model" not in adjudicate.READING_RULES.lower()


def test_the_2026_11_13_rules_reach_the_prompts_they_were_written_for():
    from rlvr.neurons import acceptance
    from rlvr.neurons.demo_miner import REPOSITORY_SYSTEM_PROMPT

    for text in (REPOSITORY_SYSTEM_PROMPT, acceptance.AUTHOR_SYSTEM_PROMPT, acceptance._SETTLE_RULES):
        assert adjudicate.EXAMPLE_RULE in text and adjudicate.ERROR_SCOPE in text and "example wins" not in text
    # The settle rules hold the edge rules with the solver's own percent-encoding clause (acceptance._SOLVER_EDGE_RULES).
    assert adjudicate.EDGE_RULES in acceptance.AUTHOR_SYSTEM_PROMPT and acceptance._SOLVER_EDGE_RULES in acceptance._SETTLE_RULES
    assert adjudicate.LIFECYCLE_RULES in REPOSITORY_SYSTEM_PROMPT
    assert adjudicate.ERROR_SCOPE in adjudicate.ADJUDICATOR_SYSTEM_PROMPT
    for rule in (adjudicate.EXAMPLE_RULE, adjudicate.ERROR_SCOPE, adjudicate.EDGE_RULES, adjudicate.LIFECYCLE_RULES):
        assert "model" not in rule.lower()  # nothing that reads as rules about the model's own output


def test_a_grammar_stated_once_for_a_kind_of_value_reaches_every_prompt():
    # 9fd0ec887882 (2026-11-13): the header grammar was stated under the profile's header config; record header
    # patches were typed only as "patch object", and the hidden test rejected a bad name and a LF value there.
    from rlvr.neurons import acceptance
    from rlvr.neurons.demo_miner import REPOSITORY_SYSTEM_PROMPT

    assert "binds every input that carries a value of that same kind" in adjudicate.EDGE_RULES
    # The settle rules hold the edge rules with the solver's own percent-encoding clause (acceptance._SOLVER_EDGE_RULES).
    assert adjudicate.EDGE_RULES in acceptance.AUTHOR_SYSTEM_PROMPT and acceptance._SOLVER_EDGE_RULES in acceptance._SETTLE_RULES
    assert "every input that carries a value of that same kind" in REPOSITORY_SYSTEM_PROMPT
    assert "a record's header patch as well as the profile's header config" in acceptance.AUTHOR_SYSTEM_PROMPT
    assert "model" not in adjudicate.EDGE_RULES.lower()


def test_percent_encoding_and_fresh_state_rules_reach_every_prompt():
    from rlvr.neurons import acceptance, demo_miner

    # The suite author keeps the wording from before prompt audit finding 7 (its prompts have their own owner).
    encoding = "a space becomes %20 and a literal + becomes %2B, never +"
    for text in (adjudicate.EDGE_RULES, acceptance.AUTHOR_SYSTEM_PROMPT):
        assert encoding in text and text.count(adjudicate.WRITER_ENCODING_RULE) == 1
        assert "url.QueryEscape" in text
    # The solver reads the settle rules beside its instructions, so they hold the solver's clause, once, and never
    # the writer's: two versions would disagree on + under a named url.PathEscape and on ' under encodeURIComponent.
    for text in (demo_miner.REPOSITORY_SYSTEM_PROMPT, acceptance._SETTLE_RULES):
        assert text.count(adjudicate.ENCODING_RULE) == 1, text[:60]
        assert adjudicate.WRITER_ENCODING_RULE not in text and "never +" not in text
    # Apart from that clause, the settle rules carry the writer's edge rules unchanged.
    assert acceptance._SETTLE_RULES.count(
        adjudicate.EDGE_RULES.replace(adjudicate.WRITER_ENCODING_RULE, adjudicate.ENCODING_RULE)
    ) == 1
    # The solver's clause puts a documented encoding function or encoded form first (8724e1fd451e documents
    # encodeURIComponent, which keeps ' literal and escapes /), and keeps %20 and %2B otherwise (966efbb1635f).
    solver = demo_miner.REPOSITORY_SYSTEM_PROMPT
    named = "Where the documentation names the encoding function or shows the encoded form (encodeURIComponent, "
    otherwise = "a space becomes %20 and a literal + becomes %2B, even where the decoder reads + as a space"
    assert solver.count(named) == 1 and solver.count(otherwise) == 1
    assert solver.index(named) < solver.index(otherwise) < solver.index("url.QueryEscape")
    assert "A lowercase escape on input comes out uppercase." in solver
    fresh = "give each record that the case is not about its own key, lane, group or quota"
    assert fresh in acceptance.AUTHOR_SYSTEM_PROMPT
    assert "model" not in fresh and "model" not in adjudicate.EDGE_RULES.lower()


def test_fallback_rule_reaches_every_prompt_and_keeps_named_exceptions():
    from rlvr.neurons import acceptance, demo_miner

    rule = adjudicate.FALLBACK_RULE
    assert "takes B only when A is not given" in rule
    assert "never replaced by B" in rule
    assert "absent or empty defaults to" in rule  # a sentence that names the state still sends it to B
    assert "is-None test" in rule
    assert "model" not in rule
    for text in (adjudicate.EDGE_RULES, acceptance._SETTLE_RULES, acceptance.AUTHOR_SYSTEM_PROMPT,
                 demo_miner.REPOSITORY_SYSTEM_PROMPT):
        assert text.count(rule) == 1
    solver = demo_miner.REPOSITORY_SYSTEM_PROMPT
    assert solver.index("does not transfer to any other input.") < solver.index(rule) < solver.index(
        "A documented TypeError, ValueError or error code must be that one")


def test_rules_mined_from_the_hidden_suites_reach_the_solver_only():
    from rlvr.neurons import acceptance, demo_miner

    solver, probes = demo_miner.REPOSITORY_SYSTEM_PROMPT, demo_miner.REPOSITORY_RUN_PROMPT
    for rule in (adjudicate.GRAMMAR_RULE, adjudicate.SECTION_RULE, adjudicate.SHAPE_RULE, adjudicate.ERROR_ID_RULE):
        assert solver.count(rule) == 1
        assert rule not in acceptance.AUTHOR_SYSTEM_PROMPT  # the test writer has its own owner
        for word in ("catalogue", "restate", "another model"):
            assert word not in rule.lower()
    # The exceptions the skeptics found in real suites stay part of each rule.
    assert "re.fullmatch" in adjudicate.GRAMMAR_RULE and "\\d" in adjudicate.GRAMMAR_RULE
    assert "where the documentation names it" in adjudicate.GRAMMAR_RULE
    assert "bind the encoder only" in adjudicate.GRAMMAR_RULE
    assert "is a tuple in that order" in adjudicate.SHAPE_RULE
    assert "never makes the operation call a collaborator" in adjudicate.SECTION_RULE
    assert solver.index(adjudicate.ERROR_SCOPE) < solver.index(adjudicate.SECTION_RULE)
    # 121c13873690: an unknown id passed to the call is reported as passed; the skeptics' limits stay part of it.
    assert solver.index(adjudicate.ERROR_SCOPE) < solver.index(adjudicate.ERROR_ID_RULE) < solver.index(adjudicate.SECTION_RULE)
    for limit in ("a well-formed id or key argument", "also when no element has it",
                  "unless a sentence, table row or example gives that code or case another value",
                  "does not cover such a call"):
        assert limit in adjudicate.ERROR_ID_RULE
    for wider in ("missing member", "invalid element", "or input"):
        assert wider not in adjudicate.ERROR_ID_RULE
    assert adjudicate.ERROR_ID_RULE not in acceptance._SETTLE_RULES
    assert solver.index(adjudicate.ONE_TYPE) < solver.index(adjudicate.SHAPE_RULE)
    assert "an earlier item's error at its own step still comes first" in solver
    assert "stays the identical object" in solver
    assert "real output beside the hand-written wire form" in probes


def test_a_byte_order_mark_is_one_only_where_a_document_starts():
    # 4c763391ef17: a decoder rejected U+FEFF inside a JSON string ("a line has a BOM"); the hidden test accepts it
    # there and rejects it at a line's start. 1cab26e45bdf keeps U+FEFF at a later CSV row's start as text.
    from rlvr.neurons import acceptance, demo_miner

    solver = demo_miner.REPOSITORY_SYSTEM_PROMPT
    clause = ("a byte order mark is U+FEFF at the start of the input, or at the start of a line only where each line "
              "is decoded as its own document (JSON Lines), and U+FEFF anywhere else is ordinary content.")
    assert solver.count(clause) == 1
    assert solver.index("checks them itself; a byte order mark") < solver.index(clause) + 1
    assert clause not in acceptance.AUTHOR_SYSTEM_PROMPT  # the test writer has its own owner


def test_public_functions_defined_by_a_formula_keep_the_modules_errors_and_bounds():
    from rlvr.neurons import acceptance, demo_miner

    solver = demo_miner.REPOSITORY_SYSTEM_PROMPT
    rule = "They also call each documented public function directly"
    assert solver.count(rule) == 1 and rule not in acceptance.AUTHOR_SYSTEM_PROMPT
    start = solver.index(rule)
    assert solver.index("stays the identical object") < start < solver.index("Each input is read when the documentation")
    sentence = solver[start:solver.index("Each input is read when the documentation")]
    # The exclusions found in the 50 kept hidden suites stay part of it.
    for kept in ("A predicate (is_x(...) -> bool) still answers False", "a Go func returning only a string",
                 "each still reporting the code its own section gives", "gains no bound from a table",
                 "counted on its result when the bound is stated for the canonical or normalized form"):
        assert kept in sentence
    for word in ("catalogue", "restate", "another model"):
        assert word not in sentence


def test_wrapper_order_text_spelling_and_object_string_rules_reach_the_solver_only():
    from rlvr.neurons import acceptance, demo_miner

    solver = demo_miner.REPOSITORY_SYSTEM_PROMPT
    assert "keeps every check of build, a wrong-type x included, ahead of x.commit" in solver
    assert "bind plan = build(x, ...) first" in solver
    assert "belongs to that format's parser alone" in solver
    assert "never the decoding of its spellings" in solver
    rule = demo_miner.VALUE_STRING_RULE
    assert solver.count(rule) == 1 and rule not in acceptance.AUTHOR_SYSTEM_PROMPT
    assert solver.index(adjudicate.SHAPE_RULE) < solver.index(rule) < solver.index(demo_miner.IMPLICIT_RULES)
    assert ".value or isoformat()" in rule and "plain Enum" in rule and "lock or cache writer" in rule
    for text in (rule, "keeps every check of build", "belongs to that format's parser alone"):
        for word in ("catalogue", "restate", "another model"):
            assert word not in text.lower()
    assert "model" not in rule.lower()


def test_go_error_kinds_are_matched_on_the_chain_unless_provenance_decides():
    from rlvr.neurons import acceptance, demo_miner

    solver = demo_miner.REPOSITORY_SYSTEM_PROMPT
    rule = "find that kind with errors.Is or errors.As on each error's chain"
    assert solver.count(rule) == 1 and rule not in acceptance.AUTHOR_SYSTEM_PROMPT
    assert "a collaborator's error keeps its documented handling whatever it wraps" in solver
    assert solver.index("typed error's Unwrap returns its documented cause") < solver.index(rule) < solver.index(
        "Python's json.loads accepts duplicate keys")


def test_deduplication_comes_before_each_items_work():
    from rlvr.neurons import acceptance, demo_miner

    solver = demo_miner.REPOSITORY_SYSTEM_PROMPT
    rule = "the deduplication comes before each item's work, not after it"
    assert solver.count(rule) == 1 and rule not in acceptance.AUTHOR_SYSTEM_PROMPT
    assert solver.index("No shortcut skips validation") < solver.index(rule) < solver.index("Where the documentation ties two fields")
    for kept in ("is called once per distinct key, in first-occurrence order", "\"sink still runs\"",
                 "\"filtering happens before duplicate suppression\"", "every item is still validated and keyed in order"):
        assert kept in solver


def test_steps_are_the_mechanism_follows_the_example_rule_in_the_solver_only():
    from rlvr.neurons import acceptance, demo_miner

    solver, rule = demo_miner.REPOSITORY_SYSTEM_PROMPT, demo_miner.STEPS_RULE
    assert solver.count(rule) == 1 and rule not in acceptance.AUTHOR_SYSTEM_PROMPT
    assert rule not in adjudicate.EXAMPLE_RULE
    assert solver.index(adjudicate.EXAMPLE_RULE) < solver.index(rule) < solver.index("A test can only compute its")
    assert "is a definition and reaches only as far as those words" in rule
    for word in ("model", "catalogue", "restate"):
        assert word not in rule.lower()


# The two-rule choice matcher: the readings the first finished reply names under a heading of its own, matched
# against FALLBACK_RULE and the percent-encoding clause. The fixtures are the first finished replies of three failed
# Opus solves of 2026-11-13 (59143559837f and 966efbb1635f hold their hidden defect in a matched bullet; ed00cb107823's
# matched bullet already follows the rule) and of one passed Opus solve (85b8b47fb30d), as recorded.
FIRST_REPLIES = Path(__file__).parent / "fixtures" / "first_replies"


def first_reply(task_id: str) -> str:
    return (FIRST_REPLIES / f"{task_id}.md").read_text(encoding="utf-8")


def test_only_the_bullets_under_a_choices_heading_are_read():
    click = adjudicate.named_choices(first_reply("59143559837f"))
    assert len(click) == 5
    assert click[0].startswith("**Badly spelled `extends` entries:**")
    assert click[2].startswith("**Route segment fallback:**") and click[4].startswith("**Rules matching a group:**")
    # A behaviour description under another heading is not a named choice, whatever its words.
    assert "The fallback policy skips keys present in the baseline." in first_reply("59143559837f")
    assert not any("fallback policy" in bullet for bullet in click)
    tapes = adjudicate.named_choices(first_reply("966efbb1635f"))
    assert len(tapes) == 6 and tapes[0].startswith("**Lanes are global by name, not per URL.**")
    assert tapes[-1].startswith("**A body over 1 MiB in a record is `ErrScenarioLimit`")
    face = adjudicate.named_choices(first_reply("ed00cb107823"))
    assert len(face) == 10 and face[0].startswith("**`\"data_root\": null` and source-level `null` values")
    assert "**Report wording:**" in face[5]  # nested bullets are read too
    assert not any("lock_bytes" in bullet for bullet in face)  # the closing paragraph is not a bullet
    assert len(adjudicate.named_choices(first_reply("85b8b47fb30d"))) == 6


def test_a_reply_without_a_choices_heading_has_no_named_choices():
    assert adjudicate.named_choices("Fixed the defect.") == []
    assert adjudicate.named_choices("") == []
    # The same recorded reply with its heading taken out: its bullets are no longer read, the fallback ones included.
    text = first_reply("59143559837f").replace("**Decisions you may want to check:**\n", "")
    assert "**Route segment fallback:**" in text and adjudicate.named_choices(text) == []
    assert adjudicate.match_choices(adjudicate.named_choices(text)) == []
    # The section ends at the next heading.
    text = "Where the doc was silent, I chose:\n- one\n- two\n\nNot covered:\n- falls back to the default\n"
    assert adjudicate.named_choices(text) == ["one", "two"]
    assert adjudicate.named_choices("Choices:\n- one\n## Testing\n- falls back\n") == ["one"]


def test_a_heading_word_inside_a_name_or_a_longer_word_is_not_a_choices_heading():
    # 8f865e72eb1f (passed): "open" inside `openmed` made a line naming files a heading, and its four bullets, which
    # describe what the scripts covered, were read as named choices.
    files = (
        "The modules are `openmed/onnx/maple_transfer_plan.py`, `maple_transfer.py` and `maple_transfer_session.py`:\n"
        "- **Planning:** an empty plan falls back to the defaults.\n"
    )
    assert adjudicate.named_choices(files) == []
    assert adjudicate.named_choices("Where the doc was unambiguous:\n- a missing key falls back\n") == []
    assert adjudicate.named_choices("The OpenAPI schema now has:\n- a null limit defaults to 10\n") == []
    # The whole words still make a heading, "chosen" among them.
    for heading in ("Readings I have chosen:", "**Choices the doc leaves open:**", "Where the doc was silent, I chose:",
                    "I made three smaller calls where the doc is silent:", "Decisions where the guide is silent:"):
        assert adjudicate.named_choices(f"{heading}\n- one\n") == ["one"], heading


def test_a_reply_of_many_blank_or_heading_lines_is_read_at_once():
    # The patterns' \s* crosses lines: before blank runs were collapsed, 8,000 lines of " \n" took 14 seconds on the
    # event loop, before any time check, and 20,000 about a minute and a half.
    text = (
        "Where the doc was silent, I chose:\n- an empty name falls back to the command name\n"
        + " \n" * 20_000 + "\t\n" * 20_000 + "- second\n"
        + "Decisions:\n" * 20_000
        + "".join(f"- reading {n}\n" for n in range(5_000))
    )
    began = time.monotonic()
    bullets = adjudicate.named_choices(text)
    assert time.monotonic() - began < 2.0
    assert bullets[:3] == ["an empty name falls back to the command name", "second", "reading 0"]
    assert len(bullets) == 5_002
    assert adjudicate.matched_rules(adjudicate.match_choices(bullets)) == ["fallback"]
    # Blank lines, however many and whatever they hold, end a section exactly as one does.
    for blank in ("\n", " \n", "\n \t\n\n", " \n" * 50):
        text = f"Where the doc was silent, I chose:\n- one\n{blank}\nNot covered:\n- falls back\n"
        assert adjudicate.named_choices(text) == ["one"], repr(blank)
        assert adjudicate.named_choices(f"Choices:\n{blank}\n- one\n{blank}- two\n") == ["one", "two"], repr(blank)


def test_the_two_rules_match_the_three_recorded_opus_choices_and_nothing_on_the_passed_reply():
    expected = {
        "59143559837f": ("fallback", "**Route segment fallback:**"),
        "966efbb1635f": ("encoding", "**Re-encoding uses Go's escapers.**"),
        "ed00cb107823": ("fallback", "**`\"data_root\": null` and source-level `null` values are rejected.**"),
    }
    for task_id, (name, start) in expected.items():
        matches = adjudicate.match_choices(adjudicate.named_choices(first_reply(task_id)))
        assert len(matches) == 1 and adjudicate.matched_rules(matches) == [name]
        bullet, rule, probe = matches[0]
        assert bullet.startswith(start)
        assert rule == {"fallback": adjudicate.FALLBACK_RULE, "encoding": adjudicate.ENCODING_RULE}[name]
        assert probe == next(item.probe for item in adjudicate.CHOICE_RULES if item.name == name)
    assert adjudicate.match_choices(adjudicate.named_choices(first_reply("85b8b47fb30d"))) == []


def test_the_patterns_are_the_two_precise_rules_and_nothing_wider():
    def names(bullet):
        return adjudicate.matched_rules(adjudicate.match_choices([bullet]))

    for bullet in ("An empty name falls back to the command name.", "A missing key falls back.", "`limit` defaults to 10.",
                   "Falling back to the default", "a null is treated as missing", "the context name, or the "
                   "command's when there is no context name", "a fallback route"):
        assert names(bullet) == ["fallback"], bullet
    for bullet in ("query values go through url.QueryEscape", "I used quote_plus", "urllib.parse.urlencode builds it",
                   "`URLSearchParams` writes the query", "it writes + for a space", "a space becomes `+`",
                   "a space becomes +"):
        assert names(bullet) == ["encoding"], bullet
    for bullet in ("Lanes are global by name, not per URL.", "a limit of 0 admits nothing", "percent signs are kept",
                   "Ports are compared by number.", "members in any order are accepted", ""):
        assert names(bullet) == [], bullet
    both = "An empty query falls back to the path, and values go through QueryEscape."
    assert names(both) == ["fallback", "encoding"]
    many = [f"case {n} falls back" for n in range(10)]
    assert len(adjudicate.match_choices(many)) == adjudicate.MAX_CHOICE_MATCHES


def test_the_choice_message_quotes_the_bullet_the_whole_rule_and_the_input_to_run():
    matches = adjudicate.match_choices(adjudicate.named_choices(first_reply("59143559837f")))
    text = adjudicate.choice_prompt(matches)
    assert text.startswith('Your summary says: "**Route segment fallback:** if `info_name` is not a usable segment')
    assert 'I read "neither supplies" literally here." Your instructions hold this rule for that case: "A value' in text
    # The rule is quoted whole, so that its own exceptions travel with it.
    assert f'"{adjudicate.FALLBACK_RULE}"' in text and "B replaces a given A only where a sentence" in text
    assert "Run a call that gives the first source as None or null, as an empty value and as a value that breaks" in text
    assert text.endswith(
        ", and keep or change your code as that rule and the documentation decide; quote the passage for any change, "
        "then reply with a one-line summary."
    )
    tapes = adjudicate.choice_prompt(adjudicate.match_choices(adjudicate.named_choices(first_reply("966efbb1635f"))))
    assert f'"{adjudicate.ENCODING_RULE}"' in tapes and adjudicate.FALLBACK_RULE not in tapes
    assert "Run the encoding on a path segment, a query key and a query value that each hold a space, a literal +" in tapes
    assert tapes.count("Your summary says:") == 1 and tapes.count("quote the passage") == 1


def test_the_choice_message_avoids_what_the_host_refuses():
    # adjudicate.py's docstring: a request to catalogue the model's own earlier statements, or a description of its
    # output being read and ruled on by another model call, is refused. The message quotes and describes no call.
    for task_id in ("59143559837f", "966efbb1635f", "ed00cb107823"):
        text = adjudicate.choice_prompt(adjudicate.match_choices(adjudicate.named_choices(first_reply(task_id)))).lower()
        for word in ("model", "you have called", "every point", "catalog", "list ", "colleague", "reader", "reviewer"):
            assert word not in text, (task_id, word)


def test_several_matches_make_one_message_with_each_rule_quoted_once():
    bullets = [
        "**Empty names:** an empty `name` falls back to the command name.",
        "**Null options:** `null` is treated as missing.",
        "**Queries:** values go through `QueryEscape`",
    ]
    text = adjudicate.choice_prompt(adjudicate.match_choices(bullets))
    assert text.count(adjudicate.FALLBACK_RULE) == 1 and text.count(adjudicate.ENCODING_RULE) == 1
    assert text.startswith(f'Your summary says: "{bullets[0]}" It also says: "{bullets[1]}" Your instructions hold this '
                           "rule for those cases:")
    assert f'Your summary says: "{bullets[2]}". Your instructions hold this rule for that case:' in text
    assert text.count("Run ") == 2 and text.count("quote the passage for any change") == 1
    assert text.index(adjudicate.FALLBACK_RULE) < text.index("\n\nYour summary says:") < text.index(adjudicate.ENCODING_RULE)
    long = adjudicate.choice_prompt(adjudicate.match_choices(["falls back " + "word " * 300]))
    assert "…" in long and len(long) < len(adjudicate.FALLBACK_RULE) + 1200


def test_the_quoted_rules_are_the_text_the_solvers_instructions_hold():
    # "Your instructions hold this rule": each quoted rule must be in the solver's system message word for word.
    # When the solver's percent-encoding clause is reworded, ENCODING_RULE takes the new wording.
    from rlvr.neurons import demo_miner

    for rule in adjudicate.CHOICE_RULES:
        assert demo_miner.REPOSITORY_SYSTEM_PROMPT.count(rule.rule) == 1, rule.name
        assert "model" not in rule.rule.lower() and "model" not in rule.probe.lower()


SUMMARY_5914 = completion_for(first_reply("59143559837f"))
SUMMARY_85B8 = completion_for(first_reply("85b8b47fb30d"))


def unclocked(content: str) -> str:
    """A request's newest message without the clock line that ends it.

    Since plan step 1 (merged with this step for the first deploy) the newest message of every request ends with
    "\n\nClock: N s used, about M s left for this task.", so a harness prompt sent as that message is the prompt
    followed by the clock line; the exact comparisons below take the line off, after checking it is there.
    """
    from tests.test_demo_miner_workbench import CLOCK

    assert CLOCK.search(content), content[-200:]
    return CLOCK.sub("", content)


async def test_a_matched_choice_is_quoted_back_once_before_the_review(documented, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", no_suite)
    provider = Scripted(edit(), RUN, SUMMARY_5914, FINISHED, DONE)
    _, submission, notes = await run_loop(documented, provider, FakeSandbox(), miner_adjudicator=False)
    assert b"gamma" in submission and notes.ended_by == "finished" and len(provider.seen) == 5
    asked = provider.seen[3]
    assert asked[-2]["content"] == SUMMARY_5914.output
    assert unclocked(asked[-1]["content"]) == adjudicate.choice_prompt(adjudicate.match_choices(adjudicate.named_choices(SUMMARY_5914.output)))
    # Once per solve: the reply to it goes on to the review.
    assert provider.seen[4][-1]["content"].startswith("Before this is submitted")
    assert provider.seen[4][:-2] == asked  # the conversation only grows
    assert notes.choice_check == "matched fallback (1 of 5 bullets)" and notes.fields()["choice_check"] == notes.choice_check


async def test_the_choice_check_reads_the_first_finished_reply_after_the_readings(documented, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", no_suite)
    provider = ReaderProvider(edit(), RUN, SUMMARY_5914, LISTING, FINISHED, DONE, rulings=RULINGS.replace("AGREES: no", "AGREES: yes"))
    _, submission, notes = await run_loop(documented, provider, FakeSandbox())
    assert b"gamma" in submission and notes.adjudication == "2 readings, 0 overruled"
    assert unclocked(provider.seen[3][-1]["content"]) == adjudicate.LIST_PROMPT
    # The listing is the latest reply, but the bullets quoted are the first finished reply's.
    asked = provider.seen[4]
    assert asked[-2]["content"] == LISTING.output and asked[-1]["content"].startswith('Your summary says: "**Route segment fallback:**')
    assert provider.seen[5][-1]["content"].startswith("Before this is submitted")
    assert notes.choice_check == "matched fallback (1 of 5 bullets)"


async def test_named_choices_without_a_match_or_without_a_heading_add_no_turn(documented, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", no_suite)
    provider = Scripted(edit(), RUN, SUMMARY_85B8, DONE)
    _, submission, notes = await run_loop(documented, provider, FakeSandbox(), miner_adjudicator=False)
    assert b"gamma" in submission and len(provider.seen) == 4
    assert provider.seen[3][-1]["content"].startswith("Before this is submitted")
    assert notes.choice_check == "none (6 bullets)"
    provider = Scripted(edit(), RUN, FINISHED, DONE)
    _, _, notes = await run_loop(documented, provider, FakeSandbox(), miner_adjudicator=False)
    assert len(provider.seen) == 4 and notes.choice_check == "none (no choices section)"


async def test_the_choice_check_is_off_for_terminal_tasks_when_turned_off_or_short_of_time(workspace, documented, monkeypatch):
    from rlvr.neurons.workbench import SCRIPT_PATH
    from tests.test_demo_miner import completion_with_calls
    from tests.test_demo_miner_workbench import terminal_task

    monkeypatch.setattr(DemoMiner, "_author_suite", no_suite)
    provider = Scripted(edit(), RUN, SUMMARY_5914, DONE)
    _, submission, notes = await run_loop(documented, provider, FakeSandbox(), miner_adjudicator=False, miner_choice_check=False)
    assert b"gamma" in submission and len(provider.seen) == 4 and notes.choice_check == ""
    assert provider.seen[3][-1]["content"].startswith("Before this is submitted")
    # Too little time before the wrap-up: the match is recorded, not sent.
    provider = Scripted(edit(), RUN, SUMMARY_5914, DONE)
    _, submission, notes = await run_loop(documented, provider, FakeSandbox(), deadline=150.0, miner_adjudicator=False)
    assert b"gamma" in submission and len(provider.seen) == 4
    assert notes.choice_check == "matched fallback (1 of 5 bullets), not sent: too little time left"
    # A terminal task's script is not graded against these rules.
    write = completion_with_calls(("write_file", {"path": SCRIPT_PATH, "content": "echo one\n"}))
    rehearse = completion_with_calls(("run_script", {}))
    provider = Scripted(write, rehearse, SUMMARY_5914, DONE)
    _, submission, notes = await run_loop(workspace, provider, FakeSandbox(), request=terminal_task(), miner_adjudicator=False)
    assert submission == b"echo one\n" and len(provider.seen) == 4 and notes.choice_check == ""


async def test_the_choice_check_needs_a_sandbox_but_not_documentation(workspace, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", no_suite)
    # Without a sandbox the input it names cannot be run: no turn, nothing recorded.
    provider = Scripted(edit(), SUMMARY_5914, DONE)
    _, submission, notes = await run_loop(workspace, provider, None, miner_adjudicator=False)
    assert b"gamma" in submission and len(provider.seen) == 3 and notes.choice_check == ""
    assert provider.seen[2][-1]["content"].startswith("Before this is submitted")
    # Without documentation (the task names docs/widgets.md, which this workspace lacks) both rules are still in the
    # solver's instructions, so a matched choice still goes back.
    provider = Scripted(edit(), RUN, SUMMARY_5914, FINISHED, DONE)
    _, submission, notes = await run_loop(workspace, provider, FakeSandbox(), miner_adjudicator=False)
    assert b"gamma" in submission and len(provider.seen) == 5
    assert provider.seen[3][-1]["content"].startswith('Your summary says: "**Route segment fallback:**')
    assert provider.seen[4][-1]["content"].startswith("Before this is submitted")
    assert notes.choice_check == "matched fallback (1 of 5 bullets)"


async def test_a_refused_choice_message_is_withdrawn_and_the_solve_goes_on_to_the_review(documented, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", no_suite)
    provider = Scripted(edit(), RUN, SUMMARY_5914, REFUSED, DONE)
    events, submission, notes = await run_loop(documented, provider, FakeSandbox(), miner_adjudicator=False)
    assert b"gamma" in submission and notes.ended_by == "finished" and notes.reviewed
    asked, review = provider.seen[3], provider.seen[4]
    assert asked[-1]["content"].startswith("Your summary says:")
    # The refused message and the reply it followed are gone; the review follows that reply instead, and the check
    # is not asked again.
    assert review[-1]["content"].startswith("Before this is submitted") and len(review) == len(asked)
    assert review[:-2] == asked[:-2] and review[-2]["content"] == SUMMARY_5914.output
    assert notes.refused == "choice check" and notes.choice_check == "matched fallback (1 of 5 bullets)"
    assert len(provider.seen) == 5


def test_consistency_fixes_between_rules_added_one_at_a_time():
    from rlvr.neurons import acceptance, demo_miner

    solver = demo_miner.REPOSITORY_SYSTEM_PROMPT
    # A state the documentation tests first answers first, even under "no shortcut skips validation".
    assert "a state the documentation tests before the arguments" in solver and "still answers first" in solver
    # A bound stated for a step's result is counted on that result; a bound on the input still counts it as fed.
    assert "is counted on that result" in solver and "counts it as fed, whitespace included" in solver
    assert "measured on the input as given" in solver
    # The decoder clause applies only where the decoder's own section makes the shape/lookup split.
    assert "where a decoder's section gives its shape code for shape problems" in adjudicate.SECTION_RULE
    assert "Where its section makes no such split" in adjudicate.SECTION_RULE
    assert adjudicate.SECTION_RULE not in acceptance.AUTHOR_SYSTEM_PROMPT
