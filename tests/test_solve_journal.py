from __future__ import annotations

import base64
import csv
import gzip
import io
import json
import os
import time
from contextlib import asynccontextmanager
from dataclasses import replace

import pytest

from rlvr import protocol
from rlvr.neurons import demo_miner
from rlvr.neurons.demo_miner import DemoMiner, SolveNotes, TokenUsage, build_model_messages
from rlvr.neurons.diagnosis import DiagnosisError
from rlvr.neurons.journal import (
    WORKSPACE_FILE,
    WORKSPACE_WAIT_S,
    SolveJournal,
    cause,
    changed_files,
    parse_utc,
    read_records,
    report,
    utc,
    wilson,
)
from rlvr.neurons.workbench import SCRIPT_PATH
from rlvr.v3.miner_workspace import WorkspaceReader
from tests.test_demo_miner import WalletLike, completion_for, completion_with_calls, settings, task
from tests.test_demo_miner_feedback import deliver_notice, make_receiver, notice_body, remember_notice
from tests.test_demo_miner_workbench import (  # noqa: F401 - workspace is a fixture
    DONE,
    FINISHED,
    FakeSandbox,
    Scripted,
    edit,
    terminal_task,
    workspace,
)


def test_journal_appends_records_and_keeps_only_the_newest_trajectories(tmp_path):
    journal = SolveJournal(tmp_path, keep_trajectories=2)
    journal.solve({"task_id": "a"})
    journal.outcome({"task_id": "a", "reason_code": "passed"})
    paths = [journal.trajectory(f"{index:064d}", f"challenge/{index}", b'{"n":%d}' % index) for index in range(3)]
    kept = sorted(path.name for path in (tmp_path / "trajectories").iterdir())
    assert len(kept) == 2 and paths[0].rsplit("/", 1)[1] not in kept
    assert gzip.decompress((tmp_path / paths[2]).read_bytes()) == b'{"n":2}'
    assert paths[1].count("/") == 1  # a slash in a challenge id cannot make a directory
    assert read_records(tmp_path / "solves.jsonl") == [{"task_id": "a"}]
    assert read_records(tmp_path / "outcomes.jsonl") == [{"task_id": "a", "reason_code": "passed"}]


def test_a_line_cut_short_by_a_crash_is_skipped(tmp_path):
    (tmp_path / "solves.jsonl").write_text('{"task_id":"a"}\n{"task_id":"b", "cost\n')
    assert read_records(tmp_path / "solves.jsonl") == [{"task_id": "a"}]


@pytest.mark.parametrize(
    ("solve", "outcome", "expected"),
    [
        ({"status": 200, "ended_by": "finished"}, {"reason_code": "passed"}, "passed"),
        ({"status": 200, "ended_by": "finished"}, {"reason_code": "check_failed"}, "reasoning"),
        ({"status": 200, "ended_by": "wrap_up"}, {"reason_code": "check_failed"}, "stopped early"),
        ({"status": 200, "ended_by": "cost_cap"}, {"reason_code": "check_failed"}, "stopped early"),
        ({"status": 200, "ended_by": "finished"}, {"reason_code": "patch_apply_failed"}, "harness"),
        ({"status": 200, "ended_by": "finished"}, {"reason_code": "timeout"}, "grading limits"),
        ({"status": 200, "ended_by": "finished"}, {"reason_code": "setup_failed"}, "build failed"),
        ({"status": 200, "ended_by": "finished"}, {"reason_code": "evaluation_failed"}, "other"),
        ({"status": 500, "ended_by": None}, None, "miner error"),
        ({"status": 200, "ended_by": "finished"}, None, "no verdict"),
    ],
)
def test_each_failure_is_sorted_by_what_would_fix_it(solve, outcome, expected):
    assert cause(solve, outcome) == expected


def test_the_pass_rate_range_reflects_how_few_tasks_were_seen():
    low, high = wilson(90, 100)
    assert 0.82 < low < 0.83 and 0.94 < high < 0.95
    assert wilson(0, 0) == (0.0, 1.0)


def test_the_report_joins_verdicts_to_solves_and_lists_each_failure(tmp_path):
    journal = SolveJournal(tmp_path)
    base = {"status": 200, "ended_by": "finished", "task_type": "repository_patch_v1", "language": "cpp",
            "validator": "5Validator", "latency_ms": 120_000, "cost_usd": 0.3, "command_failures": 1}
    for index, reason in enumerate(["passed", "passed", "check_failed", None]):
        journal.solve({**base, "time": f"2026-11-10T00:0{index}:00Z", "challenge_id": f"c{index}", "task_id": f"{index:064d}",
                       "trajectory": f"trajectories/t{index}.json.gz"})
        if reason:
            journal.outcome({"challenge_id": f"c{index}", "task_id": f"{index:064d}", "reason_code": reason,
                             "failed_check": "Required stdout: \"x\"\nActual: \"y\"" if reason == "check_failed" else None})
    text = report(tmp_path)
    assert "graded: 3; waiting for a verdict: 1" in text
    assert "Passed: 2 of 3, 66.7%" in text
    assert "$1.20 in total, $0.300 per solve, $0.600 per pass" in text
    assert "reasoning: 1" in text and "measure at least 100" in text
    assert 'failed check: Required stdout: "x" | Actual: "y"' in text
    assert str(tmp_path / "trajectories/t2.json.gz") in text
    assert report(tmp_path / "empty").startswith("No solves recorded")


@pytest.mark.parametrize(
    ("finished", "reason", "expected"),
    [
        (80, "check_failed", "most failures are reasoning or performance errors in the answers"),
        # a check that times out is the answer's slowness, which the loop or the model fixes (task 6e3fe877063d)
        (80, "timeout", "most failures are reasoning or performance errors in the answers"),
        (20, "check_failed", "fix those before paying for a stronger model"),
    ],
)
def test_the_decision_follows_the_rule_once_there_are_enough_tasks(tmp_path, finished, reason, expected):
    journal = SolveJournal(tmp_path)
    for index in range(120):
        failed = index < 100
        journal.solve({"status": 200, "ended_by": "finished" if index < finished else "wrap_up",
                       "challenge_id": f"c{index}", "task_id": f"{index:064d}"})
        journal.outcome({"challenge_id": f"c{index}", "task_id": f"{index:064d}",
                         "reason_code": reason if failed else "passed"})
    assert expected in report(tmp_path)


async def test_every_solve_is_recorded_even_when_it_fails(tmp_path, monkeypatch):
    monkeypatch.setattr("rlvr.neurons.demo_miner.verify_signature", lambda *_args, **_kwargs: True)
    miner = DemoMiner(settings(miner_record_dir=str(tmp_path)), object(), wallet=WalletLike("miner"))

    async def solved(request, timeout_s, notes):
        notes.spend.add(TokenUsage("Anthropic", 100, 50, 50, 10, 0.25))
        notes.ended_by = "finished"
        return {"ok": True}

    async def broken(request, timeout_s, notes):
        raise RuntimeError("provider down")

    body = task().model_dump_json().encode()
    for nonce, solve in (("one", solved), ("two", broken)):
        monkeypatch.setattr(miner, "solve", solve)
        await miner.handle_request({"Epistula-Uuid": nonce, "Epistula-Signed-By": "5Validator"}, body)
    await miner.idle()
    first, second = read_records(tmp_path / "solves.jsonl")
    assert (first["status"], first["cost_usd"], first["ended_by"], first["validator"]) == (200, 0.25, "finished", "5Validator")
    assert first["task_type"] == "repository_patch_v1" and first["model"] == "anthropic/claude-sonnet-5.5"
    assert first["title"] == "Fix it" and first["time_allowed_s"] > 0 and first["workspace_bytes"] > 0
    assert second["status"] == 500 and "provider down" in second["error"]
    folder = tmp_path / first["task_dir"]
    assert (folder / "instruction.md").read_text() == "Fix it"
    details = json.loads((folder / "task.json").read_text())
    assert details["validator"] == "5Validator" and "instruction" not in details["identity"]
    assert "workspace_url" not in json.dumps(details)  # a signed download link is never kept
    log = (tmp_path / "LOG.md").read_text()
    assert "· WAITING · " in log and "· MINER ERROR · " in log and "provider down" in log


async def test_each_validator_verdict_is_recorded_once(tmp_path, monkeypatch):
    monkeypatch.setattr(protocol, "_HAVE_CRYPTO", False)
    monkeypatch.setattr(demo_miner, "print_feedback", lambda _lines: None)
    miner = make_receiver(miner_record_dir=str(tmp_path))
    body = notice_body()
    remember_notice(miner, body)
    assert (await deliver_notice(miner, body))[0] == 200
    assert (await deliver_notice(miner, body))[0] == 200
    [outcome] = read_records(tmp_path / "outcomes.jsonl")
    assert (outcome["reason_code"], outcome["validator"], outcome["task_id"]) == ("check_failed", "validator", "a" * 64)
    assert outcome["failed_check"] == 'Required stdout: "red-fox\\n"'


async def generate_with_notes(workspace, provider, *, deadline=30.0, sandbox=None, request=None, **updates):
    miner = DemoMiner(settings(**updates), provider, wallet=WalletLike("miner"), sandbox=sandbox)
    notes = SolveNotes()
    _, _, submission = await miner._generate(
        request or task(), WorkspaceReader(workspace), time.monotonic() + deadline, None, notes
    )
    return notes, submission


@pytest.mark.parametrize(
    ("replies", "updates", "expected"),
    [
        ((edit(), FINISHED, DONE), {}, "finished"),
        ((edit(), RuntimeError("provider down")), {}, "model_error"),
        ((edit(), TimeoutError("deadline")), {}, "deadline"),
        ((replace(edit(), usage=TokenUsage("Anthropic", 1, 0, 0, 1, 2.0)), FINISHED), {"miner_max_solve_cost_usd": 1.0}, "cost_cap"),
        ((edit(), FINISHED), {"miner_max_workspace_tool_calls": 1}, "tool_budget"),
        ((completion_for("```diff\n--- a/a.txt\n+++ b/a.txt\n@@ -1,2 +1,2 @@\n-alpha\n+omega\n beta\n```"),), {}, "text_submission"),
    ],
)
async def test_the_record_says_how_each_solve_ended(workspace, replies, updates, expected):
    notes, submission = await generate_with_notes(workspace, Scripted(*replies), **updates)
    assert notes.ended_by == expected and submission


async def test_the_record_shows_tool_use_and_whether_the_review_changed_anything(workspace):
    notes, _ = await generate_with_notes(workspace, Scripted(edit(), FINISHED, edit("src/b.py", "return 1", "return 2"), DONE))
    assert notes.reviewed and notes.changed_in_review and notes.tools == {"edit_file": 2}
    notes, _ = await generate_with_notes(workspace, Scripted(edit(), FINISHED, DONE))
    assert notes.reviewed and not notes.changed_in_review


def test_the_work_line_counts_what_commands_changed():
    from rlvr.neurons.demo_miner import SolveNotes
    from rlvr.neurons.journal import _work_text

    notes = SolveNotes()
    notes.note_tool("run_command", b'{"exit_code":0,"answer_updated":["a.py","b.py"],"not_in_answer":["c.py"]}')
    notes.note_tool("run_command", b'{"exit_code":1,"answer_kept":["a.py"]}')
    assert (notes.command_edits_taken, notes.command_edits_undone, notes.command_failures) == (2, 2, 1)
    fields = notes.fields()
    assert fields["command_edits_taken"] == 2 and fields["command_edits_undone"] == 2
    text = _work_text({**fields, "ended_by": "finished", "seconds": 1})
    assert "2 answer files changed by commands, taken into the answer" in text
    assert "2 command changes to files outside the answer undone" in text


def test_the_own_tests_record_reaches_the_log_and_the_csv(tmp_path):
    from rlvr.neurons.journal import _work_text

    notes = SolveNotes()
    assert notes.fields()["own_tests"] is None
    notes.own_tests = "3 rerun, 1 newly broken (panic): cp /tmp/zz2.go zz_scen_test.go; go test -run ZZScen2 ."
    fields = notes.fields()
    assert "own tests on the final version: 3 rerun, 1 newly broken (panic)" in _work_text({**fields, "seconds": 1})
    journal = SolveJournal(tmp_path)
    journal.solve({"time": "2026-11-13T22:00:00Z", "challenge_id": "c1", "task_id": "1" * 64, "status": 200,
                   "task_type": "repository_patch_v1", **fields})
    journal.write_log(now=parse_utc("2026-11-13T23:00:00Z"))
    rows = list(csv.DictReader(io.StringIO((tmp_path / "tasks.csv").read_text())))
    assert rows[0]["own_tests"] == notes.own_tests


class Sequenced(FakeSandbox):
    """Exit codes in turn, one per container run."""

    def __init__(self, *exit_codes):
        super().__init__(stdout=b"build output")
        self.exit_codes = list(exit_codes)

    async def run(self, request, *, wait_s):
        self.requests.append(request)
        return replace(self.result, exit_code=self.exit_codes.pop(0))


def build_runs(sandbox):
    return [request for request in sandbox.requests if ".rlvr/build.py" in " ".join(request.argv)]


# The model ran something after its last edit; without that, a finished answer is sent back to be tested.
RUN = completion_with_calls(("run_command", {"command": "true", "timeout_s": 10}))


async def test_a_change_that_breaks_the_task_build_is_sent_back_before_submitting(workspace):
    (workspace / ".rlvr").mkdir()
    (workspace / ".rlvr" / "build.py").write_text("print('build')\n")
    sandbox = Sequenced(0, 1, 0, 0)  # the model's run, the failing build, the model's run, the passing build
    provider = Scripted(edit(), RUN, FINISHED, edit("src/b.py", "return 1", "return 2"), RUN, FINISHED, DONE)
    notes, submission = await generate_with_notes(workspace, provider, deadline=300.0, sandbox=sandbox)
    assert len(build_runs(sandbox)) == 2 and notes.gate == "passed" and notes.reasks == 1
    sent_back = provider.seen[3][-1]["content"]
    assert sent_back.startswith("A check on a clean copy found a problem") and "build output" in sent_back
    assert b"return 2" in submission and notes.ended_by == "finished"


async def test_without_a_task_build_helper_nothing_extra_runs(workspace):
    sandbox = Sequenced(0)
    notes, _ = await generate_with_notes(workspace, Scripted(edit(), RUN, FINISHED, DONE), deadline=300.0, sandbox=sandbox)
    assert build_runs(sandbox) == [] and len(sandbox.requests) == 1 and notes.gate == ""


async def test_an_edit_nothing_ran_after_is_sent_back_once_to_be_tested(workspace):
    sandbox = Sequenced(0)
    provider = Scripted(edit(), FINISHED, RUN, edit("src/b.py", "return 1", "return 2"), FINISHED, DONE)
    notes, submission = await generate_with_notes(workspace, provider, deadline=300.0, sandbox=sandbox)
    assert provider.seen[2][-1]["content"].startswith("Your last edit came after your last test run")
    assert provider.seen[5][-1]["content"].startswith("Before this is submitted")  # the second time it is not sent back
    assert notes.retest_reminded and b"return 2" in submission and notes.ended_by == "finished"


async def test_a_script_edited_after_its_last_clean_run_is_run_again_before_submitting(workspace):
    write = completion_with_calls(("write_file", {"path": SCRIPT_PATH, "content": "echo one\n"}))
    rewrite = completion_with_calls(("edit_file", {"path": SCRIPT_PATH, "old_text": "one", "new_text": "two"}))
    sandbox = Sequenced(1, 0)
    provider = Scripted(write, FINISHED, rewrite, FINISHED, DONE)
    notes, submission = await generate_with_notes(workspace, provider, deadline=400.0, sandbox=sandbox, request=terminal_task())
    assert len(sandbox.requests) == 2 and notes.gate == "passed" and submission == b"echo two\n"
    assert "the script failed with exit code 1" in provider.seen[2][-1]["content"]


async def test_a_script_already_run_cleanly_is_not_run_again(workspace):
    write = completion_with_calls(("write_file", {"path": SCRIPT_PATH, "content": "echo one\n"}))
    rehearse = completion_with_calls(("run_script", {}))
    sandbox = Sequenced(0)
    notes, _ = await generate_with_notes(
        workspace, Scripted(write, rehearse, FINISHED, DONE), deadline=400.0, sandbox=sandbox, request=terminal_task()
    )
    assert len(sandbox.requests) == 1 and notes.gate == "" and notes.script_exit == 0


def test_the_model_is_told_what_the_sandbox_has_only_when_it_can_run_code():
    with_sandbox = build_model_messages(task(), can_run=True)[0]["content"]
    without = build_model_messages(task(), can_run=False)[0]["content"]
    assert "make, cmake, tsc, pytest, jq and curl are not installed" in with_sandbox
    assert "not installed" not in without


# Task folders, the log, reruns of failed checks, and verdicts kept across a restart.

PATCH = (
    b"diff --git a/src/a.py b/src/a.py\n--- a/src/a.py\n+++ b/src/a.py\n@@ -1,2 +1,3 @@\n-old\n+new\n+more\n keep\n"
    b"diff --git a/docs/b.md b/docs/b.md\nnew file mode 100644\n--- /dev/null\n+++ b/docs/b.md\n@@ -0,0 +1 @@\n"
    b"+--- a heading rule, not a file header\n"
)


def trajectory_with(instruction: str, answer: bytes) -> bytes:
    body = {"model": "m", "messages": [{"role": "system", "content": "rules"}, {"role": "user", "content": instruction}]}
    events = [
        {"event_type": "model_turn", "request_body_b64": ""},  # a request dropped to fit the size limit
        {"event_type": "model_turn", "request_body_b64": base64.b64encode(json.dumps(body).encode()).decode()},
        {"event_type": "tool_call", "tool_name": "validate_submission", "input_body_b64": base64.b64encode(answer).decode()},
    ]
    return json.dumps({"events": events}).encode()


def test_changed_files_counts_the_lines_each_file_gains_and_loses():
    assert changed_files(PATCH) == [
        {"path": "src/a.py", "added": 2, "removed": 1},
        {"path": "docs/b.md", "added": 1, "removed": 0},
    ]
    assert changed_files(b"") == []


def test_a_task_folder_keeps_the_workspace_until_the_verdict(tmp_path):
    journal = SolveJournal(tmp_path)
    ids = {"challenge_id": "c1", "task_id": "1" * 64}
    folder = journal.open_task("1" * 64, "c1", 1_800_000_000, ids, "Add a feature: radix\n\nDetails.")
    assert folder.name.endswith("-111111111111-c1") and folder.parent == tmp_path / "tasks"
    assert (folder / "instruction.md").read_text().startswith("Add a feature")
    (folder / WORKSPACE_FILE).write_bytes(b"archive")
    journal.outcome({**ids, "reason_code": "check_failed", "failed_check": 'Required stdout: "x"'}, folder)
    assert (folder / "failed_check.txt").read_text() == 'Required stdout: "x"' and (folder / WORKSPACE_FILE).exists()
    passed = journal.open_task("2" * 64, "c2", 1_800_000_100, {"challenge_id": "c2", "task_id": "2" * 64})
    (passed / WORKSPACE_FILE).write_bytes(b"archive")
    journal.outcome({"challenge_id": "c2", "task_id": "2" * 64, "reason_code": "passed"}, passed)
    # A passed task keeps its workspace too: with its answer it is a known-good case for the test writer.
    assert (passed / WORKSPACE_FILE).exists()
    journal.prune(now=time.time() + WORKSPACE_WAIT_S + 60)
    assert (passed / WORKSPACE_FILE).exists() and (folder / WORKSPACE_FILE).exists()


def test_only_the_newest_passed_workspaces_are_kept_within_the_disk_budget(tmp_path):
    journal = SolveJournal(tmp_path, keep_passed_workspaces=2, passed_workspace_bytes=10)
    folders = []
    for index, size in enumerate((3, 3, 8, 4)):
        task_id = str(index) * 64
        folder = journal.open_task(task_id, f"c{index}", 1_800_000_000 + index, {"challenge_id": f"c{index}", "task_id": task_id})
        (folder / WORKSPACE_FILE).write_bytes(b"x" * size)
        folders.append(folder)
    for index, folder in enumerate(folders):
        journal.outcome({"challenge_id": f"c{index}", "task_id": str(index) * 64, "reason_code": "passed"}, folder)
    journal.prune(now=time.time())
    # Newest first: the 4-byte one is kept, the 8-byte one would take the total past the 10-byte budget and
    # goes, the 3-byte one before it fits, and the oldest goes because two are already kept.
    assert [(folder / WORKSPACE_FILE).exists() for folder in folders] == [False, True, False, True]
    assert all((folder / "task.json").exists() for folder in folders)
    # A failure's workspace is never counted against the passes' budget.
    failed = journal.open_task("9" * 64, "c9", 1_800_000_009, {"challenge_id": "c9", "task_id": "9" * 64})
    (failed / WORKSPACE_FILE).write_bytes(b"x" * 50)
    journal.outcome({"challenge_id": "c9", "task_id": "9" * 64, "reason_code": "check_failed"}, failed)
    journal.prune(now=time.time())
    assert (failed / WORKSPACE_FILE).exists()
    assert [(folder / WORKSPACE_FILE).exists() for folder in folders] == [False, True, False, True]


def test_old_task_folders_and_workspaces_no_rerun_needs_are_dropped(tmp_path):
    journal = SolveJournal(tmp_path, keep_tasks=3, keep_failed_workspaces=1)
    folders = []
    for index in range(4):
        task_id = str(index) * 64
        folder = journal.open_task(task_id, f"c{index}", 1_800_000_000 + index, {"challenge_id": f"c{index}", "task_id": task_id})
        (folder / WORKSPACE_FILE).write_bytes(b"archive")
        folders.append(folder)
    assert not folders[0].exists() and all(folder.exists() for folder in folders[1:])
    for index in (1, 2):
        journal.outcome({"challenge_id": f"c{index}", "task_id": str(index) * 64, "reason_code": "check_failed"})
    journal.prune(now=time.time())
    # Only the newest failure keeps its workspace; a task still waiting keeps it for a day.
    assert [(folder / WORKSPACE_FILE).exists() for folder in folders[1:]] == [False, True, True]
    journal.prune(now=time.time() + WORKSPACE_WAIT_S + 60)
    assert [(folder / WORKSPACE_FILE).exists() for folder in folders[1:]] == [False, True, False]


def test_tasks_recorded_before_folders_existed_get_one_from_their_trajectory(tmp_path):
    journal = SolveJournal(tmp_path)
    path = journal.trajectory("3" * 64, "c3", trajectory_with("Add a new feature: radix\nmore", PATCH))
    journal.solve({
        "time": "2026-11-10T05:38:27Z", "challenge_id": "c3", "task_id": "3" * 64, "status": 200, "ended_by": "finished",
        "task_type": "repository_patch_v1", "task_kind": "feature", "language": "python", "trajectory": path,
        "cost_usd": 0.64, "latency_ms": 252_000, "submission_bytes": len(PATCH), "validator": "5Validator",
    })
    journal.outcome({"time": "2026-11-10T05:55:57Z", "challenge_id": "c3", "task_id": "3" * 64,
                     "reason_code": "check_failed", "failed_check": 'Required stdout: "x"'})
    journal.write_log(now=parse_utc("2026-11-10T06:00:00Z"))
    folder = tmp_path / "tasks" / "20261110T053827-333333333333-c3"
    assert (folder / "instruction.md").read_text() == "Add a new feature: radix\nmore"
    assert (folder / "answer.diff").read_bytes() == PATCH
    assert (folder / "failed_check.txt").read_text() == 'Required stdout: "x"'
    assert gzip.decompress((folder / "trajectory.json.gz").read_bytes()) == gzip.decompress((tmp_path / path).read_bytes())
    log = (tmp_path / "LOG.md").read_text()
    assert "### 2026-11-10 05:38:27 UTC · FAILED · python feature" in log
    assert "- Asked: Add a new feature: radix" in log
    assert "changing 2 files, +3 -1 lines: src/a.py (+2 -1), docs/b.md (+1 -0)" in log
    assert "received 2026-11-10 05:55:57 UTC, 13 min after the answer. Cause: reasoning" in log
    assert "- Failing check: in failed_check.txt, with the output it required; the workspace was not kept" in log


def test_the_log_shows_what_a_rerun_found_and_the_csv_has_a_row_per_task(tmp_path):
    journal = SolveJournal(tmp_path)
    base = {"status": 200, "ended_by": "finished", "task_type": "repository_patch_v1", "task_kind": "bug_fix",
            "language": "rust", "validator": "5Validator", "latency_ms": 100_000, "cost_usd": 0.5, "title": "Fix the parser"}
    journal.solve({**base, "time": "2026-11-10T01:00:00Z", "challenge_id": "c1", "task_id": "1" * 64})
    journal.solve({**base, "time": "2026-11-10T02:00:00Z", "challenge_id": "c2", "task_id": "2" * 64,
                   "diff_questions": "form_encoder"})
    journal.solve({"time": "2026-11-10T03:00:00Z", "challenge_id": "c3", "task_id": "3" * 64, "status": 200,
                   "task_type": "terminal_script_v1", "ended_by": "wrap_up", "cost_usd": 1.5, "answer_lines": 12,
                   "submission_bytes": 300})
    journal.outcome({"time": "2026-11-10T01:20:00Z", "challenge_id": "c1", "task_id": "1" * 64, "reason_code": "passed"})
    journal.outcome({"time": "2026-11-10T02:30:00Z", "challenge_id": "c2", "task_id": "2" * 64,
                     "reason_code": "check_failed", "failed_check": "the check"})
    journal.outcome({"time": "2026-11-10T03:20:00Z", "challenge_id": "c3", "task_id": "3" * 64, "reason_code": "check_failed"})
    journal.diagnosis({"challenge_id": "c2", "task_id": "2" * 64, "failed_total": 1,
                       "summary": "1 of 4 hidden tests did not pass: 1 failed or errored",
                       "failed": [{"test": "test_parse", "result": "FAIL", "message": "AssertionError: 1 != 2"}]})
    journal.refused({"time": "2026-11-10T04:00:00Z", "task_id": "4" * 64, "validator": "5Other",
                     "task_type": "terminal_script_v1", "reason": "it arrived too close to its deadline to be solved"})
    journal.write_log(now=parse_utc("2026-11-10T05:00:00Z"))
    log = (tmp_path / "LOG.md").read_text()
    assert "- Passed: 1 of 3, 33.3%" in log
    assert "  - rust bug fix: 2 received, passed 1 of 2 graded, $0.500 per solve, median 100s" in log
    assert "- Rerun of the failing check: 1 of 4 hidden tests did not pass: 1 failed or errored" in log
    assert "  - test_parse: FAIL — AssertionError: 1 != 2" in log
    assert "· FAILED · terminal script" in log and "a script of 12 lines" in log
    assert "- Why: the solve stopped early (wrap_up)" in log and "Cause: stopped early" in log
    assert "- Failing check: not shown by the validator" in log
    assert "## Tasks the miner could not take on" in log and "it arrived too close to its deadline" in log
    rows = list(csv.DictReader(io.StringIO((tmp_path / "tasks.csv").read_text())))
    assert [row["verdict"] for row in rows] == ["PASSED", "FAILED", "FAILED"]
    assert rows[1]["tests_failed"] == "test_parse" and rows[1]["rerun"].startswith("1 of 4") and rows[1]["cost_usd"] == "0.5"
    assert [row["diff_questions"] for row in rows] == ["", "form_encoder", ""]  # the diff questions asked (lint.py)
    text = report(tmp_path)
    assert "rerun: 1 of 4 hidden tests did not pass" in text and "test_parse: FAIL" in text


def test_the_csv_has_the_first_finish_and_the_turns_after_it(tmp_path):
    # Plan step 1: the clock line's effect shows in tasks.csv, next to seconds and cached_tokens.
    journal = SolveJournal(tmp_path)
    notes = SolveNotes(first_finish_s=231.4, turns_after_finish=9)
    journal.solve({"time": "2026-11-13T23:00:00Z", "challenge_id": "c1", "task_id": "1" * 64, "status": 200,
                   "latency_ms": 400_000, **notes.fields()})
    journal.solve({"time": "2026-11-13T23:10:00Z", "challenge_id": "c2", "task_id": "2" * 64, "status": 200})
    journal.write_log(now=parse_utc("2026-11-13T23:30:00Z"))
    rows = list(csv.DictReader(io.StringIO((tmp_path / "tasks.csv").read_text())))
    assert [(row["first_finish_s"], row["turns_after_finish"]) for row in rows] == [("231.4", "9"), ("", "")]
    assert SolveNotes().fields()["first_finish_s"] is None and SolveNotes().fields()["turns_after_finish"] == 0


def test_tasks_answered_before_a_restart_still_accept_their_verdict(tmp_path):
    now = time.time()
    journal = SolveJournal(tmp_path)

    def answered(index, minutes_ago, status=200):
        journal.solve({"time": utc(now - minutes_ago * 60 - 100), "latency_ms": 100_000, "status": status,
                       "validator": "validator", "challenge_id": f"c{index}", "task_id": str(index) * 64,
                       "uid": 7, "hotkey": "miner"})

    answered(1, 10)
    answered(2, 30)
    answered(3, 200)  # past the two-hour window for notices
    answered(4, 5, status=500)
    journal.outcome({"validator": "validator", "challenge_id": "c2", "task_id": "2" * 64, "reason_code": "passed"})
    miner = DemoMiner(settings(miner_record_dir=str(tmp_path)), object(), wallet=WalletLike("miner"))
    assert miner.restore_served_tasks(now) == 2
    seen = [miner.served_tasks.classify("validator", f"c{i}", str(i) * 64, 7, "miner", time.monotonic()) for i in (1, 2, 3, 4)]
    assert seen == ["new", "duplicate", "unknown", "unknown"]


@pytest.mark.parametrize("fails", [False, True])
async def test_a_failed_check_is_kept_beside_its_task_and_rerun_on_the_answer(tmp_path, monkeypatch, fails):
    monkeypatch.setattr(protocol, "_HAVE_CRYPTO", False)
    monkeypatch.setattr(demo_miner, "print_feedback", lambda _lines: None)
    seen = []

    def rerun(folder, display, sandbox, work_dir):
        seen.append((folder.name, display, sandbox))
        if fails:
            raise DiagnosisError("the workspace was not kept")
        return {"reproduced": True, "summary": "1 of 2 hidden tests did not pass: 1 failed or errored",
                "failed": [{"test": "test_mul", "result": "FAIL"}], "failed_total": 1}

    monkeypatch.setattr(demo_miner, "rerun_failed_check", rerun)
    miner = make_receiver(miner_record_dir=str(tmp_path))
    miner.sandbox = "sandbox"
    ids = {"challenge_id": "challenge", "task_id": "a" * 64}
    folder = miner.journal.open_task("a" * 64, "challenge", time.time(), ids)
    miner.journal.solve({**ids, "time": utc(time.time()), "status": 200, "ended_by": "finished",
                         "task_dir": miner.journal.relative(folder)})
    body = notice_body()
    remember_notice(miner, body)
    assert (await deliver_notice(miner, body))[0] == 200
    await miner.idle()
    assert seen == [(folder.name, 'Required stdout: "red-fox\\n"', "sandbox")]
    assert (folder / "failed_check.txt").read_text() == 'Required stdout: "red-fox\\n"'
    [found] = read_records(tmp_path / "diagnoses.jsonl")
    log = (tmp_path / "LOG.md").read_text()
    if fails:
        assert found["error"] == "the workspace was not kept"
        assert "- Rerun of the failing check: the failing check could not be rerun: the workspace was not kept" in log
    else:
        assert found["summary"].startswith("1 of 2") and found["task_id"] == "a" * 64
        assert "- Rerun of the failing check: 1 of 2 hidden tests did not pass" in log and "  - test_mul: FAIL" in log


async def test_a_task_the_miner_cannot_take_on_is_logged(tmp_path, monkeypatch):
    monkeypatch.setattr("rlvr.neurons.demo_miner.verify_signature", lambda *_args, **_kwargs: True)
    miner = DemoMiner(settings(miner_record_dir=str(tmp_path)), object(), wallet=WalletLike("miner"))
    expired = task().model_copy(update={"expires_at": int(time.time())})
    status, _ = await miner.handle_request(
        {"Epistula-Uuid": "late", "Epistula-Signed-By": "5Validator"}, expired.model_dump_json().encode()
    )
    await miner.idle()
    assert status == 410
    [refused] = read_records(tmp_path / "refused.jsonl")
    assert (refused["reason"], refused["validator"]) == ("it arrived too close to its deadline to be solved", "5Validator")
    assert "## Tasks the miner could not take on" in (tmp_path / "LOG.md").read_text()


async def test_a_solve_keeps_its_answer_trajectory_and_workspace_in_the_task_folder(workspace, tmp_path, monkeypatch):
    kept = {}

    @asynccontextmanager
    async def opened(*_args, keep_archive=None, **_kwargs):
        kept["archive"] = keep_archive
        keep_archive.write_bytes(b"archive")
        yield WorkspaceReader(workspace)

    async def upload(_http, request, submission, trajectory, *, allowed_origins):
        kept.update(submission=submission)
        return {"uploaded": True}

    monkeypatch.setattr("rlvr.neurons.demo_miner.open_miner_workspace", opened)
    monkeypatch.setattr("rlvr.neurons.demo_miner.upload_miner_result", upload)
    journal_dir = tmp_path / "journal"
    miner = DemoMiner(settings(miner_record_dir=str(journal_dir)), Scripted(edit(), FINISHED, DONE), wallet=WalletLike("miner"))
    request = task()
    notes = SolveNotes(folder=await miner._open_task("5Validator", request, time.time(), 600.0))
    await miner.solve(request, 60, notes)
    folder = notes.folder
    assert kept["archive"] == folder / WORKSPACE_FILE and (folder / WORKSPACE_FILE).exists()
    assert (folder / "answer.diff").read_bytes() == kept["submission"] and b"+gamma" in kept["submission"]
    assert notes.trajectory == miner.journal.relative(folder / "trajectory.json.gz")
    assert notes.changed_files == [{"path": "a.txt", "added": 1, "removed": 1}]


async def test_a_verdict_for_a_task_the_other_copy_answered_is_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr(protocol, "_HAVE_CRYPTO", False)
    monkeypatch.setattr(demo_miner, "print_feedback", lambda _lines: None)
    miner = make_receiver(miner_record_dir=str(tmp_path))
    rereads = []
    restore = miner.restore_served_tasks
    monkeypatch.setattr(miner, "restore_served_tasks", lambda *args: (rereads.append(1), restore(*args))[1])
    # While one copy replaces the other, the other copy answered this task: it is only in the journal.
    miner.journal.solve({"time": utc(time.time() - 120), "latency_ms": 60_000, "status": 200, "validator": "validator",
                         "challenge_id": "challenge", "task_id": "a" * 64, "uid": 7, "hotkey": "miner"})
    assert (await deliver_notice(miner))[0] == 200
    await miner.idle()
    [outcome] = read_records(tmp_path / "outcomes.jsonl")
    assert outcome["task_id"] == "a" * 64 and rereads == [1]
    # A task neither copy answered is still refused, and rereads are rationed.
    assert (await deliver_notice(miner, notice_body(task_id="b" * 64)))[0] == 404
    assert rereads == [1]
    miner.journal_reread_at = 0.0
    assert (await deliver_notice(miner, notice_body(task_id="c" * 64)))[0] == 404
    assert rereads == [1, 1]


def test_health_says_which_copy_answered(tmp_path):
    from fastapi.testclient import TestClient

    miner = make_receiver(miner_record_dir=str(tmp_path))
    answer = TestClient(demo_miner.build_demo_miner_app(miner)).get("/health").json()
    assert answer["status"] == "ok" and answer["pid"] == os.getpid() and answer["started"] == miner.started


def test_the_csv_has_the_choice_check_column(tmp_path):
    journal = SolveJournal(tmp_path)
    journal.solve({"time": "2026-11-13T22:00:00Z", "challenge_id": "c1", "task_id": "1" * 64, "status": 200,
                   "task_type": "repository_patch_v1", "ended_by": "finished",
                   "choice_check": "matched fallback (1 of 5 bullets)"})
    journal.solve({"time": "2026-11-13T22:10:00Z", "challenge_id": "c2", "task_id": "2" * 64, "status": 200,
                   "task_type": "repository_patch_v1", "ended_by": "finished", "choice_check": None})
    journal.write_log(now=parse_utc("2026-11-13T23:00:00Z"))
    rows = list(csv.DictReader(io.StringIO((tmp_path / "tasks.csv").read_text())))
    assert [row["choice_check"] for row in rows] == ["matched fallback (1 of 5 bullets)", ""]
