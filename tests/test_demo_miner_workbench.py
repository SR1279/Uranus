from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import pytest

from rlvr.neurons.demo_miner import DemoMiner, SolveNotes, TokenUsage, build_model_messages
from rlvr.neurons.workbench import _RUNNER, COMMAND_TIMEOUT_S, SCRIPT_PATH, Sandbox, Workbench
from rlvr.policy import RELEASE_POLICY
from rlvr.v3.download import ArtifactDownloadError
from rlvr.v3.identity import TerminalScriptTaskIdentity
from rlvr.v3.miner_workspace import WorkspaceReader, open_miner_workspace
from rlvr.v3.patch import PatchLimits, static_rejection
from rlvr.v3.supervisor import ContainerResult, SupervisorError
from tests.test_demo_miner import WalletLike, completion_for, completion_with_calls, settings, task

FINISHED = completion_for("Fixed the defect.")
DONE = completion_for("DONE")


def terminal_task():
    base = task()
    identity = TerminalScriptTaskIdentity(
        instruction="Produce the report",
        environment_sha256="a" * 64,
        verifier_sha256="b" * 64,
        execution_profile_id=base.identity.execution_profile_id,
        result_tree_path=".",
        verifier_policy=base.identity.verifier_policy,
        authoring_version="test",
    )
    return base.model_copy(update={"identity": identity})


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "workspace"
    (root / "src").mkdir(parents=True)
    (root / "a.txt").write_text("alpha\nbeta\n")
    (root / "src" / "b.py").write_text("def b():\n    return 1\n")
    (root / "no_newline.txt").write_text("last line")
    (root / "run.sh").write_text("#!/bin/sh\necho old\n")
    (root / "run.sh").chmod(0o755)
    return root


@pytest.fixture
def bench(workspace):
    with Workbench(WorkspaceReader(workspace), terminal=False, working_directory=".") as value:
        yield value


async def call(bench, tool, time_left=60.0, **arguments):
    return json.loads(await bench.execute({"tool": tool, **arguments}, time_left=time_left))


def snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def applies(patch: bytes, workspace: Path, tmp_path: Path) -> Path:
    """A fresh copy with the patch applied the way a validator applies it."""
    copy = tmp_path / "applied"
    shutil.copytree(workspace, copy)
    (tmp_path / "submission.diff").write_bytes(patch)
    for check in (["--check"], []):
        subprocess.run(["git", "apply", *check, "-p1", str(tmp_path / "submission.diff")], cwd=copy, check=True)
    return copy


async def test_edits_are_visible_to_reads_but_leave_the_workspace_alone(bench, workspace):
    before = snapshot(workspace)
    assert (await call(bench, "edit_file", path="a.txt", old_text="beta", new_text="gamma"))["replaced"] == 1
    assert "gamma" in (await call(bench, "read_file", path="a.txt", offset=0))["content"]
    assert snapshot(workspace) == before


@pytest.mark.parametrize(
    ("old_text", "extra", "expected"),
    [("missing", {}, "not found"), ("a", {}, "occurs"), ("", {}, "empty")],
)
async def test_edit_needs_one_exact_match(bench, old_text, extra, expected):
    assert expected in (await call(bench, "edit_file", path="a.txt", old_text=old_text, new_text="x", **extra))["error"]


async def test_edit_can_replace_every_occurrence(bench):
    assert (await call(bench, "edit_file", path="a.txt", old_text="a", new_text="o", replace_all=True))["replaced"] == 3


async def test_editing_a_missing_file_points_to_write_file(bench):
    assert "write_file" in (await call(bench, "edit_file", path="new.txt", old_text="a", new_text="b"))["error"]


@pytest.mark.parametrize("path", ["../x", "/etc/passwd", "a/../b", ".", ".git/config", "src/.git/hooks/x"])
async def test_paths_outside_the_submission_are_refused(bench, path):
    assert "error" in await call(bench, "write_file", path=path, content="x")
    assert not bench.touched


async def test_write_creates_nested_files_but_not_over_a_directory(bench):
    assert (await call(bench, "write_file", path="deep/er/new.txt", content="hello"))["written_bytes"] == 5
    assert "error" in await call(bench, "write_file", path="src", content="x")
    assert "error" in await call(bench, "write_file", path="a.txt/child", content="x")


async def test_edits_become_a_patch_git_applies(bench, workspace, tmp_path):
    await call(bench, "edit_file", path="src/b.py", old_text="return 1", new_text="return 2")
    await call(bench, "edit_file", path="no_newline.txt", old_text="last line", new_text="last line\nand more")
    await call(bench, "edit_file", path="run.sh", old_text="old", new_text="new")
    await call(bench, "write_file", path="added/new.txt", content="created\n")
    patch = bench.submission()
    assert static_rejection(patch, PatchLimits(RELEASE_POLICY.v3_patch_bytes, 30)) is None
    assert b"rename" not in patch and b"mode" not in patch.replace(b"new file mode 100644", b"")
    applied = applies(patch, workspace, tmp_path)
    assert (applied / "src" / "b.py").read_text() == "def b():\n    return 2\n"
    assert (applied / "no_newline.txt").read_text() == "last line\nand more"
    assert (applied / "added" / "new.txt").read_text() == "created\n"
    assert os.access(applied / "run.sh", os.X_OK)


async def test_a_rewrite_of_a_binary_looking_file_is_caught_before_upload(bench):
    await call(bench, "write_file", path="blob.bin", content="a\x00b")
    assert static_rejection(bench.submission(), PatchLimits(RELEASE_POLICY.v3_patch_bytes, 30)) is not None


async def test_undoing_an_edit_leaves_nothing_to_submit(bench):
    await call(bench, "edit_file", path="a.txt", old_text="beta", new_text="gamma")
    await call(bench, "edit_file", path="a.txt", old_text="gamma", new_text="beta")
    assert bench.submission() == b""


async def test_search_gives_lines_and_offsets_read_file_accepts(bench):
    found = await call(bench, "search_text", pattern="return [0-9]", path=".")
    assert found["truncated"] is False
    match = found["matches"][0]
    assert (match["path"], match["line"], match["text"]) == ("src/b.py", 2, "    return 1")
    assert (await call(bench, "read_file", path="src/b.py", offset=match["offset"]))["content"] == "    return 1\n"


async def test_search_sees_edits_and_stays_in_its_directory(bench):
    await call(bench, "edit_file", path="a.txt", old_text="beta", new_text="return 9")
    assert [m["path"] for m in (await call(bench, "search_text", pattern="return", path="src"))["matches"]] == ["src/b.py"]
    assert len((await call(bench, "search_text", pattern="return", path="."))["matches"]) == 2


async def test_search_skips_binary_files_and_caps_its_results(bench, workspace):
    (workspace / "blob.bin").write_bytes(b"needle\x00needle")
    (workspace / "many.txt").write_text("needle\n" * 150)
    found = await call(bench, "search_text", pattern="needle", path=".")
    assert len(found["matches"]) == 100 and found["truncated"] is True
    assert {match["path"] for match in found["matches"]} == {"many.txt"}


@pytest.mark.parametrize("arguments", [{"pattern": "", "path": "."}, {"pattern": "x", "path": "../.."}, {"path": "."}])
async def test_search_refuses_bad_arguments(bench, arguments):
    assert "error" in await call(bench, "search_text", **arguments)


async def test_a_script_task_can_only_write_its_script(workspace):
    with Workbench(WorkspaceReader(workspace), terminal=True, working_directory=".") as bench:
        assert "error" in await call(bench, "write_file", path="a.txt", content="x")
        assert not bench.has_work()
        await call(bench, "write_file", path=SCRIPT_PATH, content="#!/bin/bash\necho old\n")
        await call(bench, "edit_file", path=SCRIPT_PATH, old_text="old", new_text="new")
        assert bench.has_work() and bench.submission() == b"#!/bin/bash\necho new\n"


class FakeSandbox:
    owns_files = False
    uid = os.getuid()
    gid = os.getgid()

    def __init__(self, *, stdout=b"ok", effect=None, **fields):
        self.requests = []
        self.effect = effect
        self.result = ContainerResult(
            exit_code=fields.get("exit_code", 0),
            oom_killed=fields.get("oom_killed", False),
            timed_out=fields.get("timed_out", False),
            stdout=stdout,
            stderr=b"",
            stdout_overflow=fields.get("stdout_overflow", False),
            stderr_overflow=False,
        )

    async def run(self, request, *, wait_s):
        self.requests.append(request)
        if self.effect is not None:
            self.effect(request)
        return self.result


def sandboxed(workspace, sandbox, *, terminal=False):
    return Workbench(WorkspaceReader(workspace), terminal=terminal, working_directory=".", sandbox=sandbox)


@pytest.mark.parametrize(
    ("terminal", "has_sandbox", "extra"),
    [
        (False, False, ["revert_file"]),
        (False, True, ["revert_file", "run_command"]),
        (True, False, []),
        (True, True, ["run_script", "run_command"]),
    ],
)
def test_tools_offered_match_the_task(workspace, terminal, has_sandbox, extra):
    with sandboxed(workspace, FakeSandbox() if has_sandbox else None, terminal=terminal) as bench:
        names = [tool["function"]["name"] for tool in bench.tools]
    assert names == ["list_files", "find_files", "read_file", "search_text", "edit_file", "write_file", *extra]


async def test_each_command_gets_the_edits_it_has_not_seen(workspace):
    sandbox = FakeSandbox()
    with sandboxed(workspace, sandbox) as bench:
        await call(bench, "edit_file", path="a.txt", old_text="beta", new_text="gamma")
        await call(bench, "edit_file", path="src/b.py", old_text="1", new_text="2")
        assert (await call(bench, "run_command", command="make", timeout_s=30))["output"] == "ok"
        await call(bench, "run_command", command="make", timeout_s=30)
        await call(bench, "edit_file", path="a.txt", old_text="gamma", new_text="delta")
        await call(bench, "run_command", command="make", timeout_s=30)
    # The paths not yet copied in, an empty entry, then every file of the answer, for the look afterwards.
    answer = b"a.txt\x00src/b.py\x00"
    assert [request.stdin for request in sandbox.requests] == [
        b"a.txt\x00src/b.py\x00\x00" + answer, b"\x00" + answer, b"a.txt\x00\x00" + answer,
    ]
    assert sandbox.requests[0].argv[5:] == ("runner", "make", "30", "watch")
    assert {mount.container: mount.read_only for mount in sandbox.requests[0].mounts} == {"/work": False, "/state": False, "/edits": True}


async def test_commands_cannot_outlast_the_time_left(workspace):
    sandbox = FakeSandbox()
    with sandboxed(workspace, sandbox) as bench:
        await call(bench, "run_command", command="x", timeout_s=9999)
        await call(bench, "run_command", time_left=20.0, command="x", timeout_s=9999)
        assert "time" in (await call(bench, "run_command", time_left=3.0, command="x", timeout_s=9999))["error"]
    assert [request.argv[7] for request in sandbox.requests] == [str(min(COMMAND_TIMEOUT_S, 55)), "15"]


async def test_commands_report_limits(workspace):
    with sandboxed(workspace, FakeSandbox(exit_code=137, oom_killed=True)) as bench:
        result = await call(bench, "run_command", command="x", timeout_s=5)
    assert result["memory_limit_exceeded"] is True and result["exit_code"] == 137


async def test_without_a_sandbox_commands_explain_why_not(bench):
    assert "not available" in (await call(bench, "run_command", command="x", timeout_s=5))["error"]


async def test_run_script_starts_clean_and_reports_what_changed(workspace):
    leftovers = []

    def effect(request):
        scratch = request.mounts[0].host
        leftovers.append((scratch / "made.txt").exists())
        (scratch / "made.txt").write_text("x")
        (scratch / "a.txt").write_text("input overwritten")

    sandbox = FakeSandbox(effect=effect)
    with sandboxed(workspace, sandbox, terminal=True) as bench:
        assert "first" in (await call(bench, "run_script"))["error"]
        await call(bench, "write_file", path=SCRIPT_PATH, content="echo hi\n")
        await call(bench, "run_script")
        report = await call(bench, "run_script")
    assert leftovers == [False, False]
    assert report["files_created"] == ["made.txt"] and report["existing_files_changed"] == ["a.txt"]
    request = sandbox.requests[0]
    assert request.argv == ("/usr/bin/bash", "--noprofile", "--norc", "/submission/script.sh")
    assert (request.mounts[1].container, request.mounts[1].read_only) == ("/submission", True)


async def test_run_script_reports_an_output_that_the_grader_would_fail(workspace):
    with sandboxed(workspace, FakeSandbox(stdout=b"x" * 100_000, stdout_overflow=True), terminal=True) as bench:
        await call(bench, "write_file", path=SCRIPT_PATH, content="echo hi\n")
        report = await call(bench, "run_script")
    assert report["output_limit_exceeded"] is True and "bytes omitted" in report["stdout"]


class Scripted:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.seen = []
        self.tools = []
        self.options = []

    async def complete(self, messages, *, timeout_s, tools=(), **options):
        self.seen.append([dict(message) for message in messages])
        self.tools.append(tools)
        self.options.append(options)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def edit(path="a.txt", old_text="beta", new_text="gamma"):
    return completion_with_calls(("edit_file", {"path": path, "old_text": old_text, "new_text": new_text}))


async def generate(workspace, provider, *, deadline=30.0, sandbox=None, request=None, **updates):
    miner = DemoMiner(settings(**updates), provider, wallet=WalletLike("miner"), sandbox=sandbox)
    return await miner._generate(request or task(), WorkspaceReader(workspace), time.monotonic() + deadline)


async def test_finished_edits_get_one_review_and_become_the_submission(workspace, tmp_path):
    provider = Scripted(edit(), FINISHED, DONE)
    _, events, submission = await generate(workspace, provider)
    review = provider.seen[2][-1]["content"]
    assert review.startswith("Before this is submitted") and "+gamma" in review
    assert (applies(submission, workspace, tmp_path) / "a.txt").read_text() == "alpha\ngamma\n"
    assert not provider.replies
    assert [e["tool_name"] for e in events if e["event_type"] == "tool_call"] == ["workspace_edit"]


async def test_a_problem_found_in_review_can_still_be_fixed(workspace, tmp_path):
    provider = Scripted(edit(), FINISHED, edit("src/b.py", "return 1", "return 2"), DONE)
    _, _, submission = await generate(workspace, provider)
    assert b"return 2" in submission and b"gamma" in submission
    applies(submission, workspace, tmp_path)
    assert sum(message[-1]["content"].startswith("Before this is") for message in provider.seen) == 1


async def test_the_edit_tools_are_offered_to_the_model(workspace):
    provider = Scripted(completion_for("```diff\n--- a/a.txt\n+++ b/a.txt\n@@ -1,2 +1,2 @@\n-alpha\n+omega\n beta\n```"))
    await generate(workspace, provider)
    assert "edit_file" in [tool["function"]["name"] for tool in provider.tools[0]]


async def test_a_model_that_writes_a_diff_itself_is_still_submitted(workspace):
    diff = "--- a/a.txt\n+++ b/a.txt\n@@ -1,2 +1,2 @@\n-alpha\n+omega\n beta\n"
    provider = Scripted(completion_for(f"```diff\n{diff}```"))
    _, _, submission = await generate(workspace, provider)
    assert submission == diff.encode() and len(provider.seen) == 1


async def test_a_script_task_submits_the_script_it_wrote(workspace):
    write = completion_with_calls(("write_file", {"path": SCRIPT_PATH, "content": "#!/bin/bash\necho done\n"}))
    provider = Scripted(write, FINISHED, DONE)
    _, _, submission = await generate(workspace, provider, request=terminal_task())
    assert submission == b"#!/bin/bash\necho done\n"
    assert "run_script" not in provider.seen[2][-1]["content"].split("clean copy")[0]


@pytest.mark.parametrize("failure", [RuntimeError("provider down"), TimeoutError("deadline")])
async def test_finished_work_survives_a_model_failure(workspace, failure):
    _, _, submission = await generate(workspace, Scripted(edit(), failure))
    assert b"gamma" in submission


async def test_a_model_failure_before_any_edit_is_still_an_error(workspace):
    with pytest.raises(RuntimeError, match="provider down"):
        await generate(workspace, Scripted(RuntimeError("provider down")))


class Clock:
    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now

    def time(self):
        return time.time()


class Timed(Scripted):
    """Each model call takes `step` seconds on the clock the miner reads."""

    def __init__(self, clock, step, *replies):
        super().__init__(*replies)
        self.clock = clock
        self.step = step

    async def complete(self, messages, *, timeout_s, tools=(), **options):
        self.clock.now += self.step
        return await super().complete(messages, timeout_s=timeout_s, tools=tools, **options)


async def timed(workspace, provider, clock, monkeypatch):
    monkeypatch.setattr("rlvr.neurons.demo_miner.time", clock)
    miner = DemoMiner(settings(miner_wrap_up_reserve_s=90.0), provider, wallet=WalletLike("miner"))
    return await miner._generate(task(), WorkspaceReader(workspace), clock.now + 400.0)


async def test_the_wrap_up_time_submits_without_another_model_call(workspace, monkeypatch):
    clock = Clock()
    provider = Timed(clock, 320.0, edit(), FINISHED)
    _, _, submission = await timed(workspace, provider, clock, monkeypatch)
    assert b"gamma" in submission and len(provider.seen) == 1


async def test_the_model_is_urged_once_when_time_runs_out_with_no_answer(workspace, monkeypatch):
    clock = Clock()
    read = completion_with_calls(("read_file", {"path": "a.txt", "offset": 0}))
    provider = Timed(clock, 160.0, read, read, edit())
    _, _, submission = await timed(workspace, provider, clock, monkeypatch)
    assert b"gamma" in submission and len(provider.seen) == 3
    assert provider.seen[-1][-1]["content"].startswith("Time is almost up")


async def test_a_short_lease_is_not_told_to_stop_at_once(workspace):
    provider = Scripted(edit(), FINISHED, DONE)
    await generate(workspace, provider, deadline=30.0, miner_wrap_up_reserve_s=90.0)
    assert not any(m["content"].startswith("Time is almost") for m in provider.seen[0] if type(m.get("content")) is str)


async def test_the_spending_cap_stops_further_model_calls(workspace):
    costly = replace(edit(), usage=TokenUsage("Anthropic", 1, 0, 0, 1, 2.0))
    provider = Scripted(costly, FINISHED)
    _, _, submission = await generate(workspace, provider, miner_max_solve_cost_usd=1.0)
    assert b"gamma" in submission and len(provider.seen) == 1


async def test_running_out_of_tool_calls_submits_the_edits(workspace):
    provider = Scripted(edit(), FINISHED)
    _, _, submission = await generate(workspace, provider, miner_max_workspace_tool_calls=1)
    assert b"gamma" in submission and len(provider.seen) == 1


async def test_a_patch_git_rejects_is_sent_back_to_be_fixed(workspace, monkeypatch):
    provider = Scripted(edit(), FINISHED, DONE, edit("a.txt", "gamma", "delta"), FINISHED)
    miner = DemoMiner(settings(), provider, wallet=WalletLike("miner"))
    checks = []
    real = miner._check_bytes

    async def first_fails(request, submission, reader, deadline):
        checks.append(submission)
        return (submission, "patch does not apply") if len(checks) == 1 else await real(request, submission, reader, deadline)

    monkeypatch.setattr(miner, "_check_bytes", first_fails)
    _, _, submission = await miner._generate(task(), WorkspaceReader(workspace), time.monotonic() + 30)
    assert b"delta" in submission and "cannot be submitted" in provider.seen[3][-1]["content"]


async def test_large_workspaces_are_not_offered_commands(workspace):
    provider = Scripted(completion_for("```diff\n--- a/a.txt\n+++ b/a.txt\n@@ -1,2 +1,2 @@\n-alpha\n+omega\n beta\n```"))
    large = task()
    large = large.model_copy(update={"workspace": large.workspace.model_copy(update={"expanded_size_bytes": 3 * 1024**3})})
    await generate(workspace, provider, sandbox=FakeSandbox(), request=large)
    assert "run_command" not in [tool["function"]["name"] for tool in provider.tools[0]]
    assert "cannot run code" in provider.seen[0][0]["content"]


@pytest.mark.parametrize(
    ("request_factory", "can_run", "expected", "absent"),
    [
        (task, True, "run_command", "cannot run"),
        (task, False, "cannot run code", "run_command"),
        (terminal_task, True, "run_script", "cannot run"),
        (terminal_task, False, "cannot run anything", "run_script"),
    ],
)
def test_prompts_match_the_task_and_what_can_run(request_factory, can_run, expected, absent):
    system = build_model_messages(request_factory(), can_run=can_run)[0]["content"]
    assert expected in system and absent not in system


async def test_a_lease_about_to_end_is_refused_in_time_for_the_commit(monkeypatch):
    monkeypatch.setattr("rlvr.neurons.demo_miner.verify_signature", lambda *_args, **_kwargs: True)
    miner = DemoMiner(settings(), Scripted(), wallet=WalletLike("miner"))
    closing = task().model_copy(update={"expires_at": int(time.time()) + 30})
    status, _ = await miner.handle_request({"Epistula-Uuid": "one"}, closing.model_dump_json().encode())
    assert status == 410


async def test_a_lost_workspace_download_is_retried_on_disk(monkeypatch, tmp_path):
    attempts = []

    async def flaky(_http, _url, _reference, destination, _limits, *, allowed_origins):
        attempts.append(destination.name)
        if len(attempts) == 1:
            raise ArtifactDownloadError("connection lost")
        destination.write_bytes(b"x")
        return destination

    def extract(_archive, _reference, workspace, _limits, *, scratch_dir):
        workspace.mkdir()
        (workspace / "a.txt").write_text("a")

    monkeypatch.setattr("rlvr.v3.miner_workspace.download_artifact", flaky)
    monkeypatch.setattr("rlvr.v3.miner_workspace.extract_archive", extract)
    async with open_miner_workspace(None, task(), RELEASE_POLICY, work_dir=tmp_path) as reader:
        assert (reader.root / "a.txt").read_text() == "a" and reader.root.is_relative_to(tmp_path)
    assert attempts == ["workspace-0.tar.zst", "workspace-1.tar.zst"]


@pytest.fixture
def real_sandbox():
    sandbox, status = Sandbox.detect(concurrency=2)
    if sandbox is None:
        pytest.skip(f"the sandbox is not available here: {status}")
    return sandbox


async def test_real_sandbox_runs_edits_offline_as_an_unprivileged_user(workspace, real_sandbox):
    with sandboxed(workspace, real_sandbox) as bench:
        await call(bench, "edit_file", path="a.txt", old_text="beta", new_text="gamma")
        await call(bench, "write_file", path="deep/new.txt", content="created\n")
        result = await call(
            bench, "run_command",
            command="cat a.txt deep/new.txt; id -u; touch /etc/x 2>&1; getent hosts example.com || echo offline",
            timeout_s=30,
        )
    assert result["exit_code"] == 0 and result["timed_out"] is False
    assert "gamma" in result["output"] and "created" in result["output"]
    assert str(real_sandbox.uid) in result["output"] and "Read-only" in result["output"] and "offline" in result["output"]


async def test_real_sandbox_keeps_files_between_commands_but_never_submits_them(workspace, real_sandbox):
    with sandboxed(workspace, real_sandbox) as bench:
        await call(bench, "edit_file", path="a.txt", old_text="beta", new_text="gamma")
        first = await call(bench, "run_command", command="echo built > artifact.o; echo x > src/b.py", timeout_s=30)
        second = await call(bench, "run_command", command="cat artifact.o src/b.py", timeout_s=30)
        patch = bench.submission()
    # A new file stays in the copy and out of the answer; a change to a file outside the answer is
    # reported and undone before the next command, so every test runs on what will be submitted.
    assert first["not_in_answer"] == ["src/b.py"] and "edit_file or write_file" in first["note"]
    assert "answer_updated" not in first and second["output"] == "built\ndef b():\n    return 1\n"
    assert b"gamma" in patch and b"artifact.o" not in patch and b"src/b.py" not in patch


async def test_real_sandbox_takes_a_commands_change_to_an_answer_file_into_the_answer(workspace, real_sandbox, tmp_path):
    with sandboxed(workspace, real_sandbox) as bench:
        await call(bench, "edit_file", path="src/b.py", old_text="return 1", new_text="return 2")
        await call(bench, "write_file", path="src/new.py", content="VALUE = 1\n")
        script = "sed -i 's/return 2/return 3/' src/b.py && python3 - <<'EOF'\np='src/new.py'\ns=open(p).read()\nopen(p,'w').write(s.replace('1','4'))\nEOF\ncat src/b.py src/new.py"
        result = await call(bench, "run_command", command=script, timeout_s=30)
        after = await call(bench, "run_command", command="cat src/b.py src/new.py", timeout_s=30)
        patch = bench.submission()
    assert result["exit_code"] == 0 and sorted(result["answer_updated"]) == ["src/b.py", "src/new.py"]
    assert "Your answer now has them exactly as the command left them" in result["note"]
    assert "answer_updated" not in after and "return 3" in after["output"] and "VALUE = 4" in after["output"]
    assert b"+    return 3" in patch and b"+VALUE = 4" in patch and b"return 2" not in patch
    applies(patch, workspace, tmp_path)


async def test_real_sandbox_never_lets_a_command_wipe_or_drop_an_answer_file(workspace, real_sandbox):
    with sandboxed(workspace, real_sandbox) as bench:
        await call(bench, "edit_file", path="src/b.py", old_text="return 1", new_text="return 2")
        await call(bench, "edit_file", path="a.txt", old_text="beta", new_text="gamma")
        # As git checkout would: back to the original content. And a removal.
        result = await call(bench, "run_command", command="sed -i 's/return 2/return 1/' src/b.py; rm a.txt", timeout_s=30)
        after = await call(bench, "run_command", command="cat src/b.py a.txt", timeout_s=30)
        patch = bench.submission()
    assert sorted(result["answer_kept"]) == ["a.txt", "src/b.py"] and "revert_file" in result["note"]
    assert "answer_updated" not in result and "return 2" in after["output"] and "gamma" in after["output"]
    assert b"+    return 2" in patch and b"gamma" in patch


async def test_a_forged_change_report_cannot_bring_host_files_into_the_answer(workspace, tmp_path):
    # The runner writes its report from inside the sandbox, so a command could forge it and plant links.
    secret = tmp_path / "secret.txt"
    secret.write_text("host data\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "b.py").write_text("stolen\n")
    held = {}

    def plant(_request):
        bench = held["bench"]
        (bench.scratch / "a.txt").unlink()
        (bench.scratch / "a.txt").symlink_to(secret)
        shutil.rmtree(bench.scratch / "src")
        (bench.scratch / "src").symlink_to(outside)
        (bench.state / "changed").write_bytes(
            b"M\x00a.txt\x00M\x00src/b.py\x00M\x00../secret.txt\x00O\x00.git/config\x00M\x00run.sh\x00O\x00src/b.py\x00"
        )

    with sandboxed(workspace, FakeSandbox(effect=plant)) as bench:
        held["bench"] = bench
        await call(bench, "edit_file", path="a.txt", old_text="beta", new_text="gamma")
        await call(bench, "edit_file", path="src/b.py", old_text="return 1", new_text="return 2")
        result = await call(bench, "run_command", command="x", timeout_s=5)
        patch = bench.submission()
        # A report that is itself a link is never read.
        (bench.state / "changed").unlink(missing_ok=True)
        (tmp_path / "forged").write_bytes(b"O\x00no_newline.txt\x00")
        held["bench"] = None
        relinked = FakeSandbox(effect=lambda _request: (bench.state / "changed").symlink_to(tmp_path / "forged"))
        bench.sandbox = relinked
        second = await call(bench, "run_command", command="x", timeout_s=5)
    # run.sh is not part of the answer and src/b.py is, so those two entries are not believed either.
    assert "answer_updated" not in result and sorted(result["answer_kept"]) == ["a.txt", "src/b.py"]
    assert "not_in_answer" not in result and "note" not in second
    assert b"host data" not in patch and b"stolen" not in patch and b"gamma" in patch and b"return 2" in patch


async def test_real_sandbox_ignores_build_products_and_new_files(workspace, real_sandbox):
    (workspace / "target").mkdir()
    (workspace / "target" / "out.txt").write_text("old\n")
    (workspace / "blob.bin").write_bytes(b"\x00\x01")
    with sandboxed(workspace, real_sandbox) as bench:
        await call(bench, "edit_file", path="a.txt", old_text="beta", new_text="gamma")
        result = await call(
            bench, "run_command",
            command="echo new > target/out.txt; printf '\\000\\002' > blob.bin; mkdir -p __pycache__ && echo c > __pycache__/x.pyc; echo t > made.txt",
            timeout_s=30,
        )
    assert result["exit_code"] == 0 and not {"answer_updated", "not_in_answer", "answer_kept"} & set(result)
    # Only the new text file is named, as created and not submitted.
    assert result["created_not_in_answer"] == ["made.txt"] and "never submitted" in result["note"]


async def test_real_sandbox_names_source_files_only_a_command_created_until_the_answer_has_them(workspace, real_sandbox):
    # 4dc6113b5960 (2026-11-12): pkg/fixerreply/replies.go was written with cat >, tested in the copy and
    # never submitted; every hidden test failed to build.
    with sandboxed(workspace, real_sandbox) as bench:
        await call(bench, "edit_file", path="a.txt", old_text="beta", new_text="gamma")
        result = await call(
            bench, "run_command",
            command="mkdir -p pkg/feature && echo 'package feature' > pkg/feature/feature.go"
                    " && echo 'package feature' > pkg/feature/feature_test.go && echo 'print(1)' > t.py",
            timeout_s=30,
        )
        assert sorted(result["created_not_in_answer"]) == ["pkg/feature/feature.go", "pkg/feature/feature_test.go", "t.py"]
        assert bench.left_out() == ["pkg/feature/feature.go"]
        await call(bench, "write_file", path="pkg/feature/feature.go", content="package feature\n")
        assert bench.left_out() == []
        # A file the copy no longer holds is not asked about.
        bench.created.add("pkg/gone.go")
        assert bench.left_out() == []
        patch = bench.submission()
    assert b"pkg/feature/feature.go" in patch and b"feature_test.go" not in patch and b"t.py" not in patch


async def test_real_sandbox_a_command_cut_short_changes_nothing_in_the_answer(workspace, real_sandbox):
    with sandboxed(workspace, real_sandbox) as bench:
        await call(bench, "edit_file", path="src/b.py", old_text="return 1", new_text="return 2")
        result = await call(bench, "run_command", command="sed -i 's/return 2/return 5/' src/b.py; sleep 30", timeout_s=1)
        patch = bench.submission()
    assert result["timed_out"] is True and result["answer_kept"] == ["src/b.py"] and "cut short" in result["note"]
    assert b"+    return 2" in patch


async def test_real_sandbox_links_planted_by_a_command_cannot_reach_the_host(workspace, real_sandbox, tmp_path):
    secret = tmp_path / "host_secret.txt"
    secret.write_text("host data")
    before = snapshot(workspace)
    with sandboxed(workspace, real_sandbox) as bench:
        await call(bench, "edit_file", path="a.txt", old_text="beta", new_text="first")
        planted = f"rm -f a.txt; ln -s {secret} a.txt; rm -rf src; ln -s {tmp_path} src"
        await call(bench, "run_command", command=planted, timeout_s=30)
        await call(bench, "edit_file", path="a.txt", old_text="first", new_text="second")
        await call(bench, "edit_file", path="src/b.py", old_text="return 1", new_text="return 2")
        result = await call(bench, "run_command", command="ls -ld a.txt src; cat a.txt src/b.py", timeout_s=30)
    # The planted links are swept away by a fresh copy, so both edits reach the command.
    assert "second" in result["output"] and "return 2" in result["output"]
    assert secret.read_text() == "host data" and snapshot(workspace) == before


async def test_real_sandbox_shows_output_from_a_command_that_timed_out(workspace, real_sandbox):
    with sandboxed(workspace, real_sandbox) as bench:
        result = await call(bench, "run_command", command="echo started; sleep 30", timeout_s=1)
    assert result["timed_out"] is True and result["output"] == "started\n"


async def test_real_sandbox_keeps_the_end_of_long_output(workspace, real_sandbox):
    with sandboxed(workspace, real_sandbox) as bench:
        result = await call(bench, "run_command", command="seq 1 200000; exit 3", timeout_s=30)
    assert result["exit_code"] == 3 and "bytes omitted" in result["output"]
    assert result["output"].startswith("1\n") and result["output"].endswith("200000\n")


async def test_real_sandbox_runs_a_script_like_the_grader(workspace, real_sandbox):
    script = "#!/bin/bash\nset -euo pipefail\nmkdir -p output\npwd > output/where.txt\nid -u >> output/where.txt\necho out\n"
    with sandboxed(workspace, real_sandbox, terminal=True) as bench:
        await call(bench, "write_file", path=SCRIPT_PATH, content=script)
        report = await call(bench, "run_script")
        inspected = await call(bench, "run_command", command="cat output/where.txt", timeout_s=30)
    assert report["exit_code"] == 0 and report["stdout"] == "out\n"
    assert report["files_created"] == ["output/where.txt"] and report["existing_files_changed"] == []
    assert inspected["output"].startswith("/work\n")


async def test_a_docker_failure_is_reported_to_the_model_not_raised(workspace):
    class Broken(FakeSandbox):
        async def run(self, request, *, wait_s):
            raise SupervisorError("container launch failed")

    with sandboxed(workspace, Broken()) as bench:
        assert "could not complete" in (await call(bench, "run_command", command="x", timeout_s=5))["error"]


async def test_a_rerun_is_not_looked_at_and_takes_nothing_into_the_answer(workspace):
    def effect(request):
        if request.argv[-1] != "watch":  # the rerun rewrites an answer file in the copy
            (request.mounts[0].host / "a.txt").write_text("rewritten by the rerun\n")

    sandbox = FakeSandbox(effect=effect)
    with sandboxed(workspace, sandbox) as bench:
        await call(bench, "edit_file", path="a.txt", old_text="beta", new_text="gamma")
        await call(bench, "run_command", command="python3 t.py", timeout_s=30)
        version, patch = bench.version, bench.submission()
        value = await bench.rerun("python3 t.py", limit_s=7, time_left=60.0)
        assert value == {"exit_code": 0, "timed_out": False, "output": "ok"}
        assert bench.version == version and bench.submission() == patch
        bench.reset_to_answer()  # as the caller does afterwards: the copy starts over and gets the answer back
        assert (bench.scratch / "a.txt").read_text() == "alpha\nbeta\n" and "a.txt" in bench.unsynced
        short = await bench.rerun("python3 t.py", limit_s=7, time_left=3.0)
    assert "time" in short["error"]
    assert sandbox.requests[1].argv[5:] == ("runner", "python3 t.py", "7") and sandbox.requests[1].stdin == b""
    assert [mount.container for mount in sandbox.requests[1].mounts] == ["/work", "/state", "/edits"]

    class Broken(FakeSandbox):
        async def run(self, request, *, wait_s):
            raise SupervisorError("container launch failed")

    with sandboxed(workspace, Broken()) as bench:
        await call(bench, "edit_file", path="a.txt", old_text="beta", new_text="gamma")
        assert (await bench.rerun("x", limit_s=5, time_left=60.0)) == {"error": "run_command could not complete"}


def test_tmp_is_put_back_as_it_was_before_the_reruns(workspace):
    with sandboxed(workspace, FakeSandbox()) as bench:
        assert bench.save_tmp() is None  # no command has left a /tmp yet
        archive = bench.state / "tmp.tar"
        archive.write_bytes(b"before")
        saved = bench.save_tmp()
        # the runner writes a new archive after each command and moves it over the old one
        (bench.state / "tmp.new").write_bytes(b"after the reruns")
        os.replace(bench.state / "tmp.new", archive)
        bench.restore_tmp(saved)
        assert archive.read_bytes() == b"before" and not saved.exists()
        bench.restore_tmp(None)
        assert not archive.exists()


async def test_real_sandbox_a_rerun_leaves_the_answer_and_tmp_as_they_were(workspace, real_sandbox):
    with sandboxed(workspace, real_sandbox) as bench:
        await call(bench, "edit_file", path="a.txt", old_text="beta", new_text="gamma")
        await call(bench, "run_command", command="echo kept > /tmp/note; cat a.txt", timeout_s=30)
        patch = bench.submission()
        saved = bench.save_tmp()
        value = await bench.rerun(
            "echo rerun > /tmp/note; sed -i 's/gamma/delta/' a.txt; echo made > new.txt; cat a.txt",
            limit_s=20, time_left=60.0,
        )
        bench.restore_tmp(saved)
        bench.reset_to_answer()
        after = await call(bench, "run_command", command="cat /tmp/note a.txt; ls new.txt 2>&1", timeout_s=30)
        assert bench.submission() == patch
    assert value["exit_code"] == 0 and value["output"] == "alpha\ndelta\n"
    assert not {"answer_updated", "answer_kept", "created_not_in_answer"} & set(value)
    assert after["output"].startswith("kept\nalpha\ngamma\n") and "No such file" in after["output"]
    assert "answer_updated" not in after


def test_a_trajectory_over_its_slot_loses_only_its_oldest_requests():
    from rlvr.neurons.demo_miner import _OMITTED_REQUEST, _drop_oldest_requests

    events = [
        {"event_type": "model_turn", "request_body_b64": "A" * 1000},
        {"event_type": "tool_call"},
        {"event_type": "model_turn", "request_body_b64": "B" * 1000},
        {"event_type": "model_turn", "request_body_b64": "C" * 1000},
    ]
    _drop_oldest_requests(events, 1200)
    assert [event.get("request_body_b64") for event in events] == [_OMITTED_REQUEST, None, _OMITTED_REQUEST, "C" * 1000]


async def test_an_edit_that_keeps_the_size_and_timestamp_is_never_lost(bench, workspace):
    original = workspace / "src" / "b.py"
    for value in range(1, 16):
        await call(bench, "edit_file", path="src/b.py", old_text=f"return {value}", new_text=f"return {value + 1}")
        edited = bench.edits / "src" / "b.py"
        timestamp = edited.stat().st_mtime_ns
        os.utime(original, ns=(timestamp, timestamp))
        assert f"+    return {value + 1}".encode() in bench.submission()


async def test_real_sandbox_keeps_tmp_between_commands_but_not_for_the_graded_script(workspace, real_sandbox):
    script = "#!/bin/bash\nif [ -e /tmp/note ]; then echo dirty; else echo clean; fi\n"
    with sandboxed(workspace, real_sandbox, terminal=True) as bench:
        await call(bench, "write_file", path=SCRIPT_PATH, content=script)
        first = await call(bench, "run_command", command="echo kept > /tmp/note; echo made", timeout_s=30)
        second = await call(bench, "run_command", command="cat /tmp/note", timeout_s=30)
        report = await call(bench, "run_script")
    assert first["output"] == "made\n" and second["output"] == "kept\n"
    assert report["stdout"] == "clean\n"


def test_the_script_prompt_says_where_the_script_runs():
    assert "script's own location" in build_model_messages(terminal_task(), can_run=True)[0]["content"]


async def test_a_created_file_can_be_taken_back_out_of_the_submission(workspace, tmp_path):
    sandbox = FakeSandbox()
    with sandboxed(workspace, sandbox) as bench:
        await call(bench, "edit_file", path="a.txt", old_text="beta", new_text="gamma")
        await call(bench, "write_file", path="probe/tmp_probe.txt", content="x\n")
        await call(bench, "run_command", command="true", timeout_s=5)
        assert (await call(bench, "revert_file", path="probe/tmp_probe.txt")) == {"reverted": "probe/tmp_probe.txt"}
        await call(bench, "run_command", command="true", timeout_s=5)
        patch = bench.submission()
    assert b"tmp_probe" not in patch and b"gamma" in patch
    assert sandbox.requests[1].stdin == b"probe/tmp_probe.txt\x00\x00a.txt\x00"  # removed from the sandbox copy too
    applies(patch, workspace, tmp_path)


async def test_an_edited_file_can_be_restored_and_nothing_is_left_to_submit(bench, workspace):
    await call(bench, "edit_file", path="run.sh", old_text="old", new_text="new")
    await call(bench, "revert_file", path="run.sh")
    assert bench.submission() == b"" and not bench.has_work()
    assert (await call(bench, "read_file", path="run.sh", offset=0))["content"] == "#!/bin/sh\necho old\n"
    assert os.access(bench.edits / "run.sh", os.X_OK)


async def test_file_tools_take_a_path_written_the_way_commands_see_it(bench):
    # 2026-11-11: three solves lost turns to edit_file("/work/src/...") being refused as absolute.
    assert (await call(bench, "edit_file", path="/work/src/b.py", old_text="return 1", new_text="return 2"))["replaced"] == 1
    assert (await call(bench, "edit_file", path="./a.txt", old_text="beta", new_text="gamma"))["replaced"] == 1
    assert (await call(bench, "read_file", path="/work/src/b.py", offset=0))["content"] == "def b():\n    return 2\n"
    assert (await call(bench, "write_file", path="/work/new/c.py", content="x = 1\n"))["written_bytes"] == 6
    assert "error" not in await call(bench, "list_files", path="/work", offset=0)
    assert (await call(bench, "revert_file", path="/work/new/c.py")) == {"reverted": "new/c.py"}
    assert "error" in await call(bench, "edit_file", path="/etc/passwd", old_text="x", new_text="y")
    assert "error" in await call(bench, "read_file", path="/work/../etc/passwd", offset=0)
    patch = bench.submission()
    assert b"+    return 2" in patch and b"gamma" in patch and b"new/c.py" not in patch


@pytest.mark.parametrize("path", ["a.txt", "../x", ".git/config"])
async def test_only_a_changed_file_inside_the_workspace_can_be_reverted(bench, path):
    assert "error" in await call(bench, "revert_file", path=path)


async def test_a_script_task_can_read_its_script_back(workspace):
    with Workbench(WorkspaceReader(workspace), terminal=True, working_directory=".") as bench:
        await call(bench, "write_file", path=SCRIPT_PATH, content="#!/bin/bash\necho hi\n")
        assert (await call(bench, "read_file", path=SCRIPT_PATH, offset=0))["content"] == "#!/bin/bash\necho hi\n"
        assert "error" in await call(bench, "revert_file", path=SCRIPT_PATH)


async def test_the_review_lists_every_file_the_patch_changes(workspace):
    probe = completion_with_calls(("write_file", {"path": "tmp_probe.txt", "content": "x\n"}))
    provider = Scripted(edit(), probe, FINISHED, completion_with_calls(("revert_file", {"path": "tmp_probe.txt"})), DONE)
    _, _, submission = await generate(workspace, provider)
    review = provider.seen[3][-1]["content"]
    assert "It changes 2 file(s): a.txt, tmp_probe.txt" in review and "revert_file" in review
    assert b"tmp_probe" not in submission


def test_the_edit_tree_is_readable_by_the_sandbox_user_even_when_the_workspace_root_is_private(workspace):
    # Live, the miner is root and the sandbox another user, and rlvr.v3.archive.extract_archive
    # makes the workspace root mode 0700. The sandbox must still be able to read every edit.
    workspace.chmod(0o700)
    (workspace / "secret.txt").write_text("s")
    (workspace / "secret.txt").chmod(0o600)
    sandbox = FakeSandbox()
    sandbox.owns_files = True
    with sandboxed(workspace, sandbox) as bench:
        bench._edit({"path": "a.txt", "old_text": "beta", "new_text": "gamma"})
        bench._write({"path": "deep/er/new.py", "content": "x = 1\n"})
        for directory, _directories, files in os.walk(bench.edits):
            assert os.stat(directory).st_mode & 0o055 == 0o055, directory
            for name in files:
                assert os.stat(os.path.join(directory, name)).st_mode & 0o004, name
    assert workspace.stat().st_mode & 0o777 == 0o700  # the workspace copy itself is left private
    assert (workspace / "secret.txt").stat().st_mode & 0o004  # a linked file's mode is shared, and that is accepted


def test_without_a_separate_sandbox_user_nothing_is_made_readable(workspace):
    workspace.chmod(0o700)
    with sandboxed(workspace, FakeSandbox()) as bench:
        bench._edit({"path": "a.txt", "old_text": "beta", "new_text": "gamma"})
        assert bench.edits.stat().st_mode & 0o077 == 0


async def test_a_command_that_could_not_get_the_edits_is_an_error_not_a_result(workspace):
    sandbox = FakeSandbox(exit_code=126, stdout=b"sandbox error: the edited files cannot be read here, so this command did not run\n")
    with sandboxed(workspace, sandbox) as bench:
        await call(bench, "edit_file", path="a.txt", old_text="beta", new_text="gamma")
        result = await call(bench, "run_command", command="cat a.txt")
        assert result["error"].startswith("sandbox error:") and bench.unsynced == {"a.txt"}
    assert "cannot be read here" in _RUNNER and "could not be copied in" in _RUNNER


async def test_the_sandbox_self_check_proves_that_commands_see_the_edits(tmp_path):
    assert await Sandbox.selfcheck(FakeSandbox(stdout=b"after\nmade\n"), tmp_path) is None
    blind = await Sandbox.selfcheck(FakeSandbox(stdout=b"before\n"), tmp_path)
    assert blind.startswith("commands do not see the edit tools' work")
    refused = FakeSandbox(exit_code=126, stdout=b"sandbox error: the edited files cannot be read here, so this command did not run\n")
    assert (await Sandbox.selfcheck(refused, tmp_path)).startswith("a command could not run on the edits: sandbox error")
    assert list(tmp_path.iterdir()) == []  # the probe workspace is removed again


async def test_real_sandbox_passes_its_self_check(real_sandbox, tmp_path):
    assert await real_sandbox.selfcheck(tmp_path) is None


def test_the_model_is_told_to_list_the_rules_first_test_each_one_and_review_them_one_by_one():
    from rlvr.neurons import demo_miner as dm

    system = dm.build_model_messages(task(), can_run=True)[0]["content"]
    assert "rules checklist" in system and "each error code and exception type" in system
    assert "bool is an int subclass and bytearray is not bytes" in system and "new String('0')" in system
    assert "Test each rule" in system and "Create it from run_command with a heredoc, never with write_file" in system
    # Go and Rust scratch tests cannot live in /tmp: they go in the package and leave in the same command
    assert "zz_scratch_test.go, created, run and removed in the same command" in system
    # a correct answer that is too slow at the documented limits times out in the hidden checks (task 6e3fe877063d)
    assert "Limits are test sizes, and the checks are timed" in system and "never rebuild an index" in system
    assert "Test at the limits" in system and "go test -race" in system and "interleave updates with" in system
    # one throwaway script covering a dozen behaviours lost to a 121-case hidden suite (task 206530f9571a):
    # the test needs one case per documented rule, and a present-but-undefined key is not an absent key
    assert "one case per rule" in system and "Print how many cases ran" in system
    assert "present with the value undefined or None is not an absent key" in system
    assert "does not transfer to any other input" in system
    # three post-fix failures were lenient readings at the edge of a rule, and a precedence rule tested
    # only one-sidedly (task 6a90b2e053a5: whitespace-only title, label entity escaped twice, selector
    # validated after the document lookup)
    assert "Read edge cases strictly" in system and "whitespace-only value is empty" in system
    assert "breaks both rules at once" in system and "escaped exactly once" in system
    # task 1a5ef7c7d48b measured MaxMessageBytes on selected fields instead of the fed bytes; the first live
    # independent suite saw a documented example overridden as a "typo"
    assert "speaks to the disputed case more directly decides" in system and "example wins" not in system
    assert "measured on the input as given" in system and "accepted at that size and rejected at one less" in system
    blind = dm.build_model_messages(task(), can_run=False)[0]["content"]
    assert "trace each checklist item" in blind and "Test each rule" not in blind and "Test at the limits" not in blind
    review = dm._review_prompt(False, b"diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n")
    assert "error and validation sections" in review and "bytearray is not bytes" in review and "x.py" in review
    assert "run your throwaway test again" in review and "proportional to the whole data set" in review
    assert "Name every documented rule your throwaway test has not covered" in review
    assert "think like the author of the hidden tests" in review and "stricter reading" in review
    briefing = dm._briefing(task(), budget=80, seconds=700)
    assert "about 700 seconds and 80 tool calls" in briefing and "ends with a clock line" in briefing
    # The plan instead of "use most of the time": a first tested version by the middle, so the suite's first run
    # (skipped when the first finish comes less than acceptance.MIN_TIME_S before the wrap-up) is not starved.
    assert "wanted by about 350 seconds from now" in briefing and "with rules untested earns nothing" in briefing
    assert "independent acceptance suite, further checks and a review" in briefing
    assert "Use them" not in briefing and "most of the time" not in briefing and "most of its time" not in briefing


async def test_a_source_file_only_a_command_created_is_asked_about_once_before_submitting(workspace, monkeypatch):
    def left_out(self):
        return [] if "src/new.py" in self.touched else ["src/new.py"]

    monkeypatch.setattr(Workbench, "left_out", left_out)
    write = completion_with_calls(("write_file", {"path": "src/new.py", "content": "X = 1\n"}))
    provider = Scripted(edit(), FINISHED, write, FINISHED, DONE, DONE, DONE)
    notes = SolveNotes()
    miner = DemoMiner(settings(), provider, wallet=WalletLike("miner"), sandbox=None)
    _, _, submission = await miner._generate(task(), WorkspaceReader(workspace), time.monotonic() + 30, notes=notes)
    sent = provider.seen[2][-1]["content"]
    assert sent.startswith("These files exist only in the sandbox copy") and "- src/new.py" in sent
    assert notes.left_out == 1 and b"src/new.py" in submission and b"gamma" in submission
    assert sum("These files exist only" in str(m.get("content")) for m in provider.seen[-1]) == 1


async def test_search_in_a_single_file_returns_its_hits_with_offsets(workspace):
    # 2026-11-13 cost analysis: without -H grep leaves the file name out for a single-file path, every hit was
    # dropped, and Opus read whole 32 KiB pages instead (6 of 14 solves).
    bench = sandboxed(workspace, None)
    found = await call(bench, "search_text", pattern="beta", path="a.txt")
    assert found["matches"] and found["matches"][0]["path"] == "a.txt" and found["matches"][0]["offset"] >= 0


async def test_terminal_tasks_run_on_the_terminal_model_and_repository_tasks_on_the_default(workspace):
    sonnet = "anthropic/claude-sonnet-5.5"
    write = completion_with_calls(("write_file", {"path": SCRIPT_PATH, "content": "#!/bin/bash\necho done\n"}))
    provider = Scripted(write, FINISHED, DONE)
    await generate(workspace, provider, request=terminal_task(), miner_terminal_model=sonnet,
                   openrouter_model="anthropic/claude-opus-5.5")
    assert provider.options and all(options.get("model") == sonnet for options in provider.options)

    provider = Scripted(edit(), FINISHED, DONE)
    await generate(workspace, provider, miner_terminal_model=sonnet, openrouter_model="anthropic/claude-opus-5.5")
    assert provider.options and all("model" not in options for options in provider.options)

    provider = Scripted(write, FINISHED, DONE)
    await generate(workspace, provider, request=terminal_task(), openrouter_model="anthropic/claude-opus-5.5")
    assert all("model" not in options for options in provider.options)  # unset: nothing changes


def test_the_solve_record_names_the_model_that_wrote_the_answer():
    miner = DemoMiner(settings(miner_terminal_model="anthropic/claude-sonnet-5.5",
                               openrouter_model="anthropic/claude-opus-5.5"), None, wallet=WalletLike("miner"))
    assert miner.solver_model(terminal_task()) == "anthropic/claude-sonnet-5.5"
    assert miner.solver_model(task()) == "anthropic/claude-opus-5.5"


# --- The real clock (plan step 1): one line at the end of every new message, measured to the wrap-up ---

CLOCK = re.compile(r"\n\nClock: (\d+) s used, about (\d+) s left for this task\.\Z")


def sent_bytes(messages) -> list[bytes]:
    """Each message as the client serializes it, so 'unchanged' means byte for byte."""
    return [json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode() for message in messages]


def clock_figures(message) -> tuple[int, int] | None:
    found = CLOCK.search("\n\n" + message["content"]) if type(message.get("content")) is str else None
    return None if found is None else (int(found.group(1)), int(found.group(2)))


async def test_every_request_ends_with_the_clock_and_nothing_already_sent_changes(workspace, monkeypatch):
    clock = Clock()
    monkeypatch.setattr("rlvr.neurons.demo_miner.time", clock)
    monkeypatch.setattr("rlvr.neurons.demo_miner.TIME_CHECKS", (0.0,))  # a time check after the first batch
    read = completion_for('```workspace\n{"tool":"read_file","path":"a.txt","offset":0}\n```')  # a text tool call
    provider = Timed(clock, 10.0, read, edit(), FINISHED, DONE)  # then a provider tool call, a finish, the review
    notes = SolveNotes()
    miner = DemoMiner(settings(miner_wrap_up_reserve_s=90.0), provider, wallet=WalletLike("miner"))
    _, _, submission = await miner._generate(task(), WorkspaceReader(workspace), clock.now + 400.0, notes=notes)
    assert b"gamma" in submission and len(provider.seen) == 4
    # The wrap-up is at 1310 (400 s minus the 90 s reserve); each call takes 10 s, and the line is written just
    # before it, counted from the start of the work (no solve() here).
    assert [clock_figures(request[-1]) for request in provider.seen] == [(0, 310), (10, 300), (20, 290), (30, 280)]
    briefing, batch, tools, review = (request[-1] for request in provider.seen)
    assert briefing["content"].startswith("Workspace sha256") and "ends with a clock line" in briefing["content"]
    # A text tool batch carries the time check and then the clock, in one user message.
    assert batch["role"] == "user" and batch["content"].startswith("Workspace tool result:")
    assert "\n\nTime check: about 300 seconds are left" in batch["content"]
    assert batch["content"].index("Time check") < batch["content"].index("Clock:")
    # After a provider tool turn the clock is a user message of its own, right after the tool result.
    assert provider.seen[2][-2]["role"] == "tool" and tools == {"role": "user", "content": "Clock: 20 s used, about 290 s left for this task."}
    assert review["content"].startswith("Before this is submitted")
    for previous, current in zip(provider.seen, provider.seen[1:]):
        # Append-only: every message of the previous request goes out again byte for byte, its clock included.
        assert sent_bytes(current[:len(previous)]) == sent_bytes(previous)
    final = provider.seen[-1]
    assert [message["role"] for message in final].count("system") == 1 and final[0]["role"] == "system"
    # One line per message that ended a request, and none anywhere else.
    assert sum(str(message.get("content")).count("Clock:") for message in final) == 4
    assert (notes.first_finish_s, notes.turns_after_finish) == (30.0, 1)
    assert notes.fields()["first_finish_s"] == 30.0 and notes.fields()["turns_after_finish"] == 1


async def test_a_refused_harness_prompt_is_withdrawn_with_its_reply_and_its_clock(workspace, monkeypatch):
    def left_out(self):
        return ["src/new.py"]

    monkeypatch.setattr(Workbench, "left_out", left_out)
    refused = replace(completion_for(""), reasoning="", finish_reason="content_filter", refusal="blocked")
    provider = Scripted(edit(), FINISHED, refused, DONE)
    notes = SolveNotes()
    miner = DemoMiner(settings(), provider, wallet=WalletLike("miner"), sandbox=None)
    _, _, submission = await miner._generate(task(), WorkspaceReader(workspace), time.monotonic() + 30, notes=notes)
    assert b"gamma" in submission and notes.refused == "files left out" and len(provider.seen) == 4
    asked, after = provider.seen[2], provider.seen[3]
    assert asked[-1]["content"].startswith("These files exist only") and clock_figures(asked[-1]) is not None
    # The prompt and the finished reply it followed are gone as two messages, clock and all, and the review
    # follows that reply with a clock of its own; what came before them is as it was sent.
    assert len(after) == len(asked) and after[-2]["content"] == FINISHED.output
    assert after[-1]["content"].startswith("Before this is submitted") and clock_figures(after[-1]) is not None
    assert sent_bytes(after[:-2]) == sent_bytes(asked[:-2]) == sent_bytes(provider.seen[1])
    assert not any("These files exist only" in str(message.get("content")) for message in after)
    assert sum(str(message.get("content")).count("Clock:") for message in after) == 3


async def test_a_retried_request_after_a_refusal_goes_out_unchanged(workspace):
    refused = replace(completion_for(""), reasoning="", finish_reason="content_filter", refusal="blocked")
    provider = Scripted(edit(), refused, FINISHED, DONE)
    await generate(workspace, provider)
    assert sent_bytes(provider.seen[1]) == sent_bytes(provider.seen[2])  # no second clock on the same message
    assert clock_figures(provider.seen[2][-1]) is not None


async def test_the_clock_and_the_first_finish_count_from_the_start_of_the_solve(workspace, monkeypatch):
    clock = Clock()
    monkeypatch.setattr("rlvr.neurons.demo_miner.time", clock)
    provider = Timed(clock, 10.0, edit(), FINISHED, DONE)
    notes = SolveNotes(started=clock.now - 15.0)  # the workspace took 15 s to download and unpack
    miner = DemoMiner(settings(miner_wrap_up_reserve_s=90.0), provider, wallet=WalletLike("miner"))
    await miner._generate(task(), WorkspaceReader(workspace), clock.now + 400.0, notes=notes)
    assert [clock_figures(request[-1]) for request in provider.seen] == [(15, 310), (25, 300), (35, 290)]
    assert (notes.first_finish_s, notes.turns_after_finish) == (35.0, 1)
    assert "started" not in notes.fields()


async def test_no_finish_leaves_the_first_finish_unset(workspace, monkeypatch):
    clock = Clock()
    read = completion_with_calls(("read_file", {"path": "a.txt", "offset": 0}))
    provider = Timed(clock, 160.0, read, read, edit())
    monkeypatch.setattr("rlvr.neurons.demo_miner.time", clock)
    notes = SolveNotes()
    miner = DemoMiner(settings(miner_wrap_up_reserve_s=90.0), provider, wallet=WalletLike("miner"))
    await miner._generate(task(), WorkspaceReader(workspace), clock.now + 400.0, notes=notes)
    # The wrap-up prompt carries the clock too (past the wrap-up, nothing is left), and the work was taken at the
    # deadline without a finished reply.
    assert provider.seen[-1][-1]["content"].startswith("Time is almost up") and clock_figures(provider.seen[-1][-1]) == (320, 0)
    assert (notes.first_finish_s, notes.turns_after_finish, notes.ended_by) == (None, 0, "deadline")


async def test_a_script_task_without_a_sandbox_and_a_cut_off_reply_get_the_clock_too(workspace):
    # Review of plan step 1: the paths the clock must not disturb. A terminal task with no sandbox (nothing runs, no
    # suite, no readings) and a reply cut at the token limit, whose continuation is a harness message like the rest.
    cut = replace(completion_for("I will now write the"), finish_reason="length")
    write = completion_with_calls(("write_file", {"path": SCRIPT_PATH, "content": "#!/bin/bash\necho done\n"}))
    provider = Scripted(cut, write, FINISHED, DONE)
    notes = SolveNotes()
    miner = DemoMiner(settings(), provider, wallet=WalletLike("miner"), sandbox=None)
    _, _, submission = await miner._generate(terminal_task(), WorkspaceReader(workspace), time.monotonic() + 30, notes=notes)
    assert submission == b"#!/bin/bash\necho done\n" and notes.cut_off == 1 and len(provider.seen) == 4
    assert all(clock_figures(request[-1]) is not None for request in provider.seen)
    briefing, continued, tools, review = (request[-1] for request in provider.seen)
    assert briefing["content"].startswith("Workspace sha256") and "Task kind" not in briefing["content"]
    assert continued["content"].startswith("That reply was cut at the token limit. Continue in smaller steps.\n\nClock: ")
    assert provider.seen[1][-2]["role"] == "assistant" and provider.seen[1][-2]["content"] == "I will now write the"
    assert provider.seen[2][-2]["role"] == "tool" and tools["content"].startswith("Clock: ")
    assert review["content"].startswith("Before this is submitted")
    for previous, current in zip(provider.seen, provider.seen[1:]):
        assert sent_bytes(current[:len(previous)]) == sent_bytes(previous)
    assert all(message["role"] != "system" for request in provider.seen for message in request[1:])
    assert notes.first_finish_s is not None and notes.turns_after_finish == 1


def test_repository_tasks_split_between_two_models_by_task_id_and_the_author_keeps_the_default(workspace):
    from rlvr.neurons.demo_miner import repository_arm

    ids = [f"{n:064x}" for n in range(40)]
    arms = [repository_arm(i) for i in ids]
    assert 10 < sum(arms) < 30 and repository_arm(ids[0]) == repository_arm(ids[0])
    odd, even = ids[arms.index(1)], ids[arms.index(0)]
    sonnet, opus = "anthropic/claude-sonnet-5.5", "anthropic/claude-opus-5.5"
    miner = DemoMiner(settings(openrouter_model=opus, miner_repository_alt_model=sonnet), None, wallet=WalletLike("miner"))
    assert miner.solver_model(task().model_copy(update={"task_id": odd})) == sonnet
    assert miner.solver_model(task().model_copy(update={"task_id": even})) == opus
    assert miner.solver_model(terminal_task().model_copy(update={"task_id": odd})) == opus  # terminal tasks: not split
    off = DemoMiner(settings(openrouter_model=opus), None, wallet=WalletLike("miner"))
    assert off.solver_model(task().model_copy(update={"task_id": odd})) == opus  # unset: no split


async def test_the_alternate_arm_names_its_model_on_every_solver_call(workspace):
    from rlvr.neurons.demo_miner import repository_arm

    odd = next(f"{n:064x}" for n in range(40) if repository_arm(f"{n:064x}") == 1)
    provider = Scripted(edit(), FINISHED, DONE)
    await generate(workspace, provider, request=task().model_copy(update={"task_id": odd}),
                   openrouter_model="anthropic/claude-opus-5.5", miner_repository_alt_model="anthropic/claude-sonnet-5.5")
    assert provider.options and all(o.get("model") == "anthropic/claude-sonnet-5.5" for o in provider.options)
