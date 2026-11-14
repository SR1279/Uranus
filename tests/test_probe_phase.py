"""The probe phase (miner_probe_phase): one timed message after the review, step 8 of the 2026-11-13 plan."""

from __future__ import annotations

import dataclasses

from rlvr.neurons import demo_miner
from rlvr.neurons.demo_miner import DemoMiner, changed_lines, probe_mode, repository_arm
from tests.test_acceptance import RUN, ScriptedSandbox, run_loop
from tests.test_demo_miner import completion_for, completion_with_calls
from tests.test_demo_miner_workbench import DONE, FINISHED, Scripted, bench, edit, terminal_task, workspace  # noqa: F401

# A sentence of the task text (the instruction is one of the sources a CHANGED quotation is checked against).
QUOTE = '"The design and exact behavior are documented in docs/widgets.md"'


async def no_suite(self, request, docs, deadline):
    return None, None


def probe_turns(*after):
    """First finish, review, then the probe phase: one probe edit of a.txt and its run, and what follows."""
    return (edit(), RUN, FINISHED, DONE, edit("a.txt", "gamma", "delta"), RUN, *after)


def sent(provider, start):
    return [seen[-1]["content"] for seen in provider.seen if seen[-1]["content"].startswith(start)]


async def test_live_mode_sends_one_probe_message_after_the_review_and_keeps_a_quoted_change(workspace, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", no_suite)
    provider = Scripted(*probe_turns(completion_for(f"Probes done.\nCHANGED: a.txt | {QUOTE}"), DONE, DONE))
    _, submission, notes = await run_loop(workspace, provider, ScriptedSandbox(), miner_probe_phase="live")
    assert provider.seen[3][-1]["content"].startswith("Before this is submitted")  # the review comes first
    probe = provider.seen[4][-1]["content"]
    assert probe.startswith("Your answer is finished, the acceptance suite has had its say and you have reviewed it")
    assert "change code only where a sentence or example of the documentation decides against it" in probe
    assert "zz_probe_test.go or tests/zz_probe.rs created, run and removed in that same command" in probe
    assert "where the documentation names an encoding function, that function is the algorithm" in probe
    assert "The final checks and the upload have their own time after this." in probe
    assert len(sent(provider, "Your answer is finished")) == 1
    assert b"+delta" in submission
    assert notes.probe == "live: ran" and notes.probe_changed == 1 and notes.probe_quotes == "1 valid, 0 invalid"


async def test_shadow_mode_records_the_probe_and_submits_the_version_from_before_it(workspace, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", no_suite)
    provider = Scripted(*probe_turns(completion_for(f"Probes done.\nCHANGED: a.txt | {QUOTE}"), DONE, DONE))
    _, submission, notes = await run_loop(workspace, provider, ScriptedSandbox(), miner_probe_phase="shadow")
    assert b"+gamma" in submission and b"delta" not in submission
    assert notes.probe == "shadow: ran" and notes.probe_changed == 1


async def test_off_and_terminal_tasks_send_no_probe_message(workspace, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", no_suite)
    provider = Scripted(edit(), RUN, FINISHED, DONE, DONE)
    _, submission, notes = await run_loop(workspace, provider, ScriptedSandbox())  # the default is off
    assert not sent(provider, "Your answer is finished") and notes.probe == "" and b"+gamma" in submission
    write = completion_with_calls(("write_file", {"path": demo_miner.SCRIPT_PATH, "content": "#!/bin/bash\necho done\n"}))
    provider = Scripted(write, FINISHED, DONE, DONE)
    _, _, notes = await run_loop(workspace, provider, ScriptedSandbox(), request=terminal_task(), miner_probe_phase="live")
    assert not sent(provider, "Your answer is finished") and notes.probe == ""


async def test_too_little_time_or_money_skips_the_phase_and_says_why(workspace, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", no_suite)
    monkeypatch.setattr(demo_miner, "PROBE_START_S", 10_000.0)
    provider = Scripted(edit(), RUN, FINISHED, DONE, DONE)
    _, _, notes = await run_loop(workspace, provider, ScriptedSandbox(), miner_probe_phase="live")
    assert notes.probe == "live: skipped for time" and not sent(provider, "Your answer is finished")
    monkeypatch.setattr(demo_miner, "PROBE_START_S", 300.0)
    provider = Scripted(edit(), RUN, FINISHED, DONE, DONE)
    _, _, notes = await run_loop(workspace, provider, ScriptedSandbox(), miner_probe_phase="live",
                                 miner_max_solve_cost_usd=0.5)
    assert notes.probe == "live: skipped for cost"


async def test_a_change_without_a_quotation_is_asked_about_once_and_never_reverted(workspace, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", no_suite)
    provider = Scripted(*probe_turns(completion_for("Probes done."), DONE, DONE, DONE))
    _, submission, notes = await run_loop(workspace, provider, ScriptedSandbox(), miner_probe_phase="live")
    asked = sent(provider, "These files changed during the probe step")
    assert len(asked) == 1 and "a.txt" in asked[0] and "revert_file would put the whole file back" in asked[0]
    assert b"+delta" in submission  # the harness never reverts a probe change on its own
    assert notes.probe == "live: ran" and notes.probe_quotes == "0 valid, 0 invalid"


async def test_the_stop_line_comes_once_after_the_window(workspace, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", no_suite)
    monkeypatch.setattr(demo_miner, "PROBE_WINDOW_S", 0.0)
    provider = Scripted(
        edit(), RUN, FINISHED, DONE, edit("a.txt", "gamma", "delta"), RUN, RUN,
        completion_for(f"Probes done.\nCHANGED: a.txt | {QUOTE}"), DONE, DONE,
    )
    await run_loop(workspace, provider, ScriptedSandbox(), miner_probe_phase="live")
    stops = [seen[-1]["content"] for seen in provider.seen if demo_miner.PROBE_STOP_LINE in seen[-1]["content"]]
    assert len(stops) == 1


async def test_a_forced_stop_inside_the_phase_sends_the_version_from_before_it(workspace, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", no_suite)
    # Three tool calls: the first edit, its run, and the probe edit, after which the budget forces the answer out.
    provider = Scripted(edit(), RUN, FINISHED, DONE, edit("a.txt", "gamma", "delta"), DONE, DONE)
    _, submission, notes = await run_loop(
        workspace, provider, ScriptedSandbox(), miner_probe_phase="live", miner_max_workspace_tool_calls=3,
    )
    assert b"+gamma" in submission and b"delta" not in submission
    assert notes.probe_snapshot_used and notes.probe.startswith("live: stopped")


async def test_a_refused_probe_message_is_withdrawn_and_the_answer_goes_on(workspace, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", no_suite)
    refused = dataclasses.replace(completion_for(""), reasoning="", finish_reason="content_filter", refusal="blocked")
    provider = Scripted(edit(), RUN, FINISHED, DONE, refused, DONE, DONE)
    _, submission, notes = await run_loop(workspace, provider, ScriptedSandbox(), miner_probe_phase="live")
    assert notes.probe == "live: withdrawn" and "probes" in notes.refused
    assert b"+gamma" in submission and len(sent(provider, "Your answer is finished")) == 1


def test_changed_lines_are_read_like_refutations():
    text = 'Fixed.\nCHANGED: `/work/pkg/a.py` | "the value must be an exact int"\n- CHANGED: ./b.go | "x"\nnot a line'
    assert changed_lines(text) == [("pkg/a.py", '"the value must be an exact int"'), ("b.go", '"x"')]


def test_alternate_mode_splits_tasks_independently_of_the_model_test():
    ids = [f"{n:064x}" for n in range(400)]
    modes = [probe_mode("alternate", task_id) for task_id in ids]
    assert set(modes) == {"live", "shadow"} and 150 < modes.count("live") < 250
    same = sum((mode == "live") == bool(repository_arm(task_id)) for mode, task_id in zip(modes, ids))
    assert 150 < same < 250  # not the model test's own split
    assert probe_mode("off", ids[0]) == "off" and probe_mode("shadow", ids[0]) == "shadow"


async def test_a_snapshot_puts_the_answer_back_exactly(bench):
    from tests.test_demo_miner_workbench import call
    await call(bench, "edit_file", path="a.txt", old_text="beta", new_text="gamma")
    held = bench.snapshot()
    await call(bench, "edit_file", path="a.txt", old_text="gamma", new_text="delta")
    await call(bench, "write_file", path="new.txt", content="probe\n")
    version = bench.version
    changed = bench.restore(held)
    assert sorted(changed) == ["a.txt", "new.txt"] and bench.version > version
    assert bench.touched == {"a.txt"} and b"+gamma" in bench.submission() and b"new.txt" not in bench.submission()
    assert bench.restore(held) == []  # nothing left to put back


async def test_the_notes_carry_both_versions_for_grading(workspace, monkeypatch):
    monkeypatch.setattr(DemoMiner, "_author_suite", no_suite)
    provider = Scripted(*probe_turns(completion_for(f"Probes done.\nCHANGED: a.txt | {QUOTE}"), DONE, DONE))
    _, submission, notes = await run_loop(workspace, provider, ScriptedSandbox(), miner_probe_phase="shadow")
    assert b"+gamma" in notes.probe_base_patch and b"+delta" in notes.probed_patch
    assert notes.probe_base_patch == submission
