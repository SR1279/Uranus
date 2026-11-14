from __future__ import annotations

import asyncio
import base64
import dataclasses
import json
import subprocess
import time
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from rlvr.neurons.demo_miner import (
    DemoMiner,
    DemoMinerSettings,
    ModelCompletion,
    OpenRouterClient,
    SolveNotes,
    SolveSpend,
    TokenUsage,
    build_model_messages,
    extract_submission,
    extract_tool_calls,
    parse_usage,
)
from rlvr.v3.api import MinerTaskRequest, MinerTaskResponse
from rlvr.v3.miner_workspace import WorkspaceReader
from rlvr.v3.trajectory import parse_trajectory
from tests.test_v3_api import lease, slot_set


class WalletLike:
    class Hotkey:
        def __init__(self, address: str):
            self.ss58_address = address

    def __init__(self, address: str):
        self.hotkey = self.Hotkey(address)


def settings(**updates) -> DemoMinerSettings:
    # The audit adds a model call at the end of every documented repository solve; the tests of other
    # stages script their conversations without it, and tests/test_audit.py turns it on.
    updates.setdefault("miner_audit", False)
    # Three readers vote in production; one keeps the scripted conversations of other tests unchanged, and
    # tests/test_adjudicate.py tests the vote.
    updates.setdefault("miner_adjudicator_votes", 1)
    # The import check runs one more sandbox command on Python answers that edit existing modules; its own
    # tests (test_imports.py) turn it on, so scripted sandboxes elsewhere keep their counts.
    updates.setdefault("miner_import_check", False)
    # The readers are off in production (2026-11-12); the tests of their stage keep them on.
    updates.setdefault("miner_adjudicator", True)
    # The first-finish coverage question adds a turn to every solve with time to spare; its own tests turn it on.
    updates.setdefault("miner_coverage_prompt", False)
    return DemoMinerSettings(_env_file=None, openrouter_api_key="key", **updates)


@pytest.fixture(autouse=True)
def _no_reasks_by_default(request, monkeypatch):
    # Loop tests that predate the pre-upload check use empty diffs; keep them focused.
    if "reask" not in request.node.name:
        monkeypatch.setattr("rlvr.neurons.demo_miner.SUBMISSION_REASK_LIMIT", 0)


def task(hotkey="miner") -> MinerTaskRequest:
    value = lease()
    slots = slot_set(hotkey=hotkey)
    roomy = slots.trajectory.model_copy(update={"max_bytes": 64 * 1024**2})
    return MinerTaskRequest(
        protocol_version=3,
        challenge_id=value["challenge_id"],
        task_id=value["task_id"],
        identity=value["identity"],
        workspace=value["workspace"],
        workspace_url=value["workspace_url"],
        expires_at=2**53 - 1,
        slots=slots.model_copy(update={"trajectory": roomy}),
    )


def test_prompt_and_submission_are_v3_shapes():
    messages = build_model_messages(task())
    assert "Git unified diff" in messages[0]["content"]
    assert messages[1]["content"] == "Fix it"
    assert extract_submission("```diff\n--- a/a\n+++ b/a\n```") == b"--- a/a\n+++ b/a\n"
    assert extract_submission("```diff\n context line \n```") == b" context line \n"
    assert extract_submission("```diff\n+value\r\n```") == b"+value\r\n"
    assert extract_submission(" --- a/a\n+++ b/a\n") == b"--- a/a\n+++ b/a\n"  # provider-added leading space
    # a raw diff that adds a fenced block keeps the fence as content
    fenced = "--- a/README.md\n+++ b/README.md\n@@ -1 +1,4 @@\n x\n+```python\n+print('hi')\n+```\n"
    assert extract_submission(fenced) == fenced.encode()
    assert extract_submission("  #!/bin/bash\necho hi\n") == b"#!/bin/bash\necho hi\n"


def test_extracted_repository_patch_remains_git_applicable(tmp_path):
    (tmp_path / "a.txt").write_text("old\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    patch = extract_submission(
        "```diff\n"
        "diff --git a/a.txt b/a.txt\n"
        "--- a/a.txt\n"
        "+++ b/a.txt\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
        "```"
    )
    completed = subprocess.run(
        ["git", "apply", "--check", "-"],
        cwd=tmp_path,
        input=patch,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr


async def test_provider_response_records_exact_bodies_and_five_alternatives():
    token = {
        "token": "x",
        "bytes": [120],
        "token_id": 7,
        "logprob": -0.1,
        "top_logprobs": [
            {"token": value, "bytes": [ord(value)], "token_id": index, "logprob": -index}
            for index, value in enumerate("abcde")
        ],
    }

    async def handler(request):
        body = json.loads(request.content)
        assert body["logprobs"] is True and body["top_logprobs"] == 5
        assert body["reasoning"] == {"effort": "high"}
        return httpx.Response(
            200,
            json={"choices": [{"finish_reason": "stop", "message": {"content": "x", "reasoning_content": "r"}, "logprobs": {"content": [token]}}]},
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenRouterClient(
        settings(openrouter_base_url="https://provider.invalid", openrouter_logprobs=True), http=http
    )
    result = await client.complete([{"role": "user", "content": "x"}], timeout_s=10)
    await http.aclose()
    assert result.generated_bytes == b"x"
    assert len(result.tokens[0]["top_logprobs"]) == 5
    assert json.loads(result.request_body)["model"] == "anthropic/claude-sonnet-5.5"


def _records(pieces: list[str]) -> list[dict]:
    alternatives = [{"token": "a", "bytes": [97], "token_id": 1, "logprob": -1.0}] * 5
    return [
        {"token": piece, "bytes": list(piece.encode()), "token_id": 1, "logprob": -0.5, "top_logprobs": alternatives}
        for piece in pieces
    ]


async def _complete_with(records: list[dict] | None, content: str = " 391", reasoning: str | None = " thinking"):
    async def handler(request):
        message = {"content": content}
        if reasoning is not None:
            message["reasoning"] = reasoning
        choice = {"finish_reason": "stop", "message": message}
        if records is not None:
            choice["logprobs"] = {"content": records}
        return httpx.Response(200, json={"choices": [choice]})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenRouterClient(settings(openrouter_base_url="https://provider.invalid"), http=http)
    try:
        return await client.complete([{"role": "user", "content": "x"}], timeout_s=10)
    finally:
        await http.aclose()


async def test_provider_tokens_cover_reasoning_and_answer():
    result = await _complete_with(_records(["thinking", "</think>", "391", "<|im_end|>"]))
    assert result.generated_bytes == b"thinking</think>391<|im_end|>"
    assert result.output == " 391" and result.reasoning == " thinking"


async def test_provider_tokens_cover_a_multiword_answer():
    records = _records(["thinking", "</think>", "the", " answer", " is", " 391", "<|im_end|>"])
    result = await _complete_with(records, content=" the answer is 391\n")
    assert result.output == " the answer is 391\n"


async def test_provider_tokens_are_recorded_as_returned_even_when_they_differ_from_the_text():
    result = await _complete_with(_records(["thinking", "</think>", "392", "<|im_end|>"]))
    assert result.output == " 391" and result.generated_bytes == b"thinking</think>392<|im_end|>"


NATIVE_CALL = (
    " <|tool_calls_section_begin|> <|tool_call_begin|> functions.list_files:0 "
    "<|tool_call_argument_begin|> {\"path\": \".\", \"offset\": 0} <|tool_call_end|> <|tool_calls_section_end|>"
)


@pytest.mark.parametrize("output, expected", [
    ("```workspace\n{\"tool\":\"read_file\",\"path\":\"a.py\",\"offset\":0}\n```", {"tool": "read_file", "path": "a.py", "offset": 0}),
    (NATIVE_CALL, {"path": ".", "offset": 0, "tool": "list_files"}),
    (NATIVE_CALL.replace("list_files", "ReadFile").replace('"path": "."', '"file_path": "slug/core.py"'), {"file_path": "slug/core.py", "offset": 0, "tool": "read_file"}),
    (NATIVE_CALL.replace("list_files", "command").replace('"offset": 0', '"tool": "read_file"'), {"path": ".", "tool": "read_file"}),
    (NATIVE_CALL.replace('{"path": ".", "offset": 0}', "{not json}"), {"tool": "list_files"}),
    ("--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n", None),
    # markup inside a submission stays a submission
    ("--- a/t.py\n+++ b/t.py\n@@ -1 +1,2 @@\n a\n+MARK = '" + NATIVE_CALL.strip() + "'\n", None),
    ("```diff\n" + NATIVE_CALL + "\n```", None),
    # non-object arguments keep the tool name only
    (NATIVE_CALL.replace('{"path": ".", "offset": 0}', "[]"), {"tool": "list_files"}),
    # the first call wins
    (NATIVE_CALL.replace("<|tool_calls_section_end|>", " <|tool_call_begin|> functions.read_file:1 <|tool_call_argument_begin|> {\"path\": \"x\"} <|tool_call_end|> <|tool_calls_section_end|>"), {"path": ".", "offset": 0, "tool": "list_files"}),
])
def test_extract_tool_call_reads_the_fence_or_native_markup(output, expected):
    from rlvr.neurons.demo_miner import extract_tool_call

    found = extract_tool_call(output)
    assert (None if found is None else json.loads(found)) == expected


async def test_demo_uploads_submission_and_genuine_canonical_trajectory(monkeypatch, tmp_path):
    completion = ModelCompletion(
        request_body=b"{}",
        response_body=b'{"ok":true}',
        output="```diff\n\n```",
        reasoning="No change is needed.",
        generated_bytes=b"x",
        tokens=[{
            "token_bytes_b64": "eA==",
            "token_id": None,
            "logprob": 0,
            "top_logprobs": [
                {"token_bytes_b64": "eA==", "token_id": None, "logprob": 0}
            ] * 5,
        }],
    )

    class Provider:
        calls = 0

        async def complete(self, messages, *, timeout_s, tools=()):
            self.calls += 1
            if self.calls == 1:
                assert "Task working directory" in messages[-1]["content"]
                return replace(
                    completion,
                    output='```workspace\n{"tool":"read_file","path":"a.txt","offset":0}\n```',
                    request_body=json.dumps({"messages": messages}).encode(),
                )
            assert "existing repository contents" in messages[-1]["content"]
            return replace(completion, request_body=json.dumps({"messages": messages}).encode())

        async def aclose(self):
            return None

    captured = {}

    (tmp_path / "a.txt").write_text("existing repository contents")

    @asynccontextmanager
    async def workspace(*_args, **_kwargs):
        yield WorkspaceReader(tmp_path)

    async def upload(_http, request, submission, trajectory, *, allowed_origins):
        captured.update(submission=submission, trajectory=trajectory, origins=allowed_origins)
        return MinerTaskResponse(
            protocol_version=3,
            challenge_id=request.challenge_id,
            task_id=request.task_id,
            response_type=request.identity.task_type,
            submission={
                "artifact_role": "patch",
                "artifact_format": "unified_diff_v1",
                "upload_id": request.slots.submission.upload_id,
                "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "size_bytes": 0,
            },
            trajectory={
                "artifact_role": "trajectory",
                "artifact_format": "trajectory_v1",
                "upload_id": request.slots.trajectory.upload_id,
                "sha256": "1" * 64,
                "size_bytes": len(trajectory),
            },
        )

    monkeypatch.setattr("rlvr.neurons.demo_miner.upload_miner_result", upload)
    monkeypatch.setattr("rlvr.neurons.demo_miner.open_miner_workspace", workspace)
    miner = DemoMiner(settings(), Provider(), wallet=WalletLike("miner"))
    response = await miner.solve(task(), 90)  # past the 45-second upload reserve
    trajectory = parse_trajectory(captured["trajectory"])
    assert response.response_type == "repository_patch_v1"
    assert captured["submission"] == b""
    assert trajectory.miner_hotkey == "miner"
    assert trajectory.events[-1].submission_sha256 == trajectory.submission_sha256
    turns = [event for event in trajectory.events if event.event_type == "model_turn"]
    assert len(turns) == 2
    assert b"existing repository contents" in base64.b64decode(turns[-1].request_body_b64)
    assert [event.sequence for event in trajectory.events] == list(range(len(trajectory.events)))
    reads = [event for event in trajectory.events if event.event_type == "tool_result"]
    assert b"existing repository contents" in base64.b64decode(reads[0].output_body_b64)


async def test_openrouter_client_requires_https():
    client = OpenRouterClient(settings(openrouter_base_url="http://provider.invalid"))
    try:
        with pytest.raises(RuntimeError, match="HTTPS"):
            _ = client.completion_url
    finally:
        await client.aclose()


def completion_for(output):
    encoded = base64.b64encode(output.encode()).decode()
    return ModelCompletion(
        request_body=b"{}", response_body=b"{}", output=output, reasoning="Inspect the file.",
        generated_bytes=output.encode(),
        tokens=[{
            "token_bytes_b64": encoded, "token_id": None, "logprob": 0,
            "top_logprobs": [{"token_bytes_b64": encoded, "token_id": None, "logprob": 0}] * 5,
        }],
    )


@pytest.mark.parametrize("keep_calling", [False, True])
async def test_workspace_tool_budget_allows_one_final_turn(tmp_path, keep_calling):
    class Provider:
        calls = 0

        async def complete(self, messages, *, timeout_s, tools=()):
            self.calls += 1
            if self.calls == 2:
                assert "Tool budget exhausted" in messages[-1]["content"]
                if not keep_calling:
                    return completion_for("```diff\n\n```")
            return completion_for('```workspace\n{"tool":"list_files","path":".","offset":0}\n```')

    import time

    provider = Provider()
    miner = DemoMiner(settings(miner_max_workspace_tool_calls=1), provider)
    if keep_calling:
        with pytest.raises(ValueError, match="tool call limit"):
            await miner._generate(task(), WorkspaceReader(tmp_path), time.monotonic() + 10)
    else:
        completion, _, _ = await miner._generate(task(), WorkspaceReader(tmp_path), time.monotonic() + 10)
        assert extract_submission(completion.output) == b""
    assert provider.calls == 2


async def test_workspace_reads_share_one_model_deadline(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("rlvr.neurons.demo_miner.time", SimpleNamespace(monotonic=lambda: clock[0]))
    budgets = []

    class Provider:
        async def complete(self, messages, *, timeout_s, tools=()):
            budgets.append(timeout_s)
            clock[0] += 25
            if len(budgets) == 1:
                return completion_for('```workspace\n{"tool":"read_file","path":"missing","offset":0}\n```')
            assert "error" in messages[-1]["content"]
            return completion_for("```diff\n\n```")

    miner = DemoMiner(settings(), Provider())
    await miner._generate(task(), WorkspaceReader(tmp_path), 200)
    assert budgets == [100, 75]
    with pytest.raises(TimeoutError):
        await miner._generate(task(), WorkspaceReader(tmp_path), 100)
    assert len(budgets) == 2


async def test_provider_call_is_bounded_by_total_time(monkeypatch):
    async def handler(request):
        await asyncio.sleep(5)
        return httpx.Response(200, json={})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenRouterClient(settings(openrouter_base_url="https://provider.invalid", openrouter_max_retries=0), http=http)
    started = time.monotonic()
    with pytest.raises(asyncio.TimeoutError):
        await client.complete([{"role": "user", "content": "x"}], timeout_s=0.2)
    await http.aclose()
    assert time.monotonic() - started < 2


async def test_provider_reply_without_reasoning_is_recorded_with_empty_reasoning():
    result = await _complete_with(_records(["391", "<|im_end|>"]), content=" 391", reasoning=None)
    assert result.reasoning == "" and result.output == " 391"


async def test_malformed_reasoning_is_rejected_not_blanked():
    with pytest.raises(RuntimeError, match="invalid response"):
        await _complete_with(_records(["391", "<|im_end|>"]), content=" 391", reasoning=False)


async def test_empty_reply_is_recorded_and_the_loop_reasks(tmp_path):
    import time

    empty = await _complete_with(None, content="", reasoning="thinking only")
    assert empty.output == "" and empty.tool_calls == ()
    replies = [empty, completion_for("```diff\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-a\n+b\n```")]
    (tmp_path / "a.txt").write_text("a\n")
    prompts = []

    class Provider:
        async def complete(self, messages, *, timeout_s, tools=()):
            prompts.append(messages[-1]["content"])
            return replies.pop(0)

    miner = DemoMiner(settings(), Provider())
    _, _, submission = await miner._generate(task(), WorkspaceReader(tmp_path), time.monotonic() + 10)
    assert submission.startswith(b"--- a/a.txt") and prompts[-1].startswith("That reply was not accepted:")


async def test_provider_reply_without_logprobs_is_recorded_without_tokens():
    result = await _complete_with(None, content=" 391")
    assert result.tokens == [] and result.generated_bytes == b"" and result.output == " 391"


async def test_logprobs_that_do_not_fit_the_trajectory_cost_the_evidence_not_the_solve(capsys):
    records = _records(["391"])
    records[0]["top_logprobs"] = records[0]["top_logprobs"][:3]
    result = await _complete_with(records, content=" 391")
    assert result.tokens == [] and result.generated_bytes == b"" and result.output == " 391"
    assert "dropped unusable logprobs" in capsys.readouterr().out


@pytest.mark.parametrize("records", [{}, "", 0])
async def test_logprobs_with_wrong_type_content_are_dropped(records):
    result = await _complete_with(records, content=" 391")
    assert result.tokens == [] and result.output == " 391"


async def test_logprobs_with_too_few_values_for_the_trajectory_are_dropped():
    records = _records(["391"])
    records[0]["top_logprobs"] = [{"token": "a", "bytes": [97], "token_id": 1, "logprob": None}] * 5
    result = await _complete_with(records, content=" 391")
    assert result.tokens == [] and result.output == " 391"


def _tool_call_message(*calls):
    return {"content": "", "tool_calls": [
        {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
        for i, (name, args) in enumerate(calls)
    ]}


async def test_provider_tool_calls_are_parsed_and_the_request_declares_tools():
    seen = {}

    async def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": _tool_call_message(("list_files", {"path": ".", "offset": 0}))}]})

    from rlvr.neurons.workbench import WORKSPACE_TOOLS

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenRouterClient(settings(openrouter_base_url="https://provider.invalid"), http=http)
    result = await client.complete([{"role": "user", "content": "x"}], timeout_s=10, tools=WORKSPACE_TOOLS)
    await http.aclose()
    assert result.tool_calls == (("call_0", "list_files", '{"path": ".", "offset": 0}'),)
    assert result.output == "" and result.tokens == []
    assert [tool["function"]["name"] for tool in seen["body"]["tools"]] == ["list_files", "find_files", "read_file", "search_text"]


def completion_with_calls(*calls):
    base = completion_for("")
    return dataclasses.replace(base, tool_calls=tuple((f"call_{i}", name, json.dumps(args)) for i, (name, args) in enumerate(calls)))


async def test_loop_executes_provider_tool_calls_and_answers_in_tool_messages(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.txt").write_text("hello")
    replies = [
        completion_with_calls(("find_files", {"path": ".", "offset": 0}), ("read_file", {"path": "sub/a.txt", "offset": 0})),
        completion_for("```diff\n\n```"),
    ]
    seen_messages = []

    class Provider:
        async def complete(self, messages, *, timeout_s, tools=()):
            seen_messages.append([dict(m) for m in messages])
            return replies.pop(0)

    import time

    miner = DemoMiner(settings(miner_max_workspace_tool_calls=2), Provider())
    completion, events, _ = await miner._generate(task(), WorkspaceReader(tmp_path), time.monotonic() + 10)
    assert extract_submission(completion.output) == b""
    second = seen_messages[1]
    assistant = next(m for m in second if m["role"] == "assistant")
    assert [c["id"] for c in assistant["tool_calls"]] == ["call_0", "call_1"]
    tool_replies = [m for m in second if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_replies] == ["call_0", "call_1"]
    assert json.loads(tool_replies[0]["content"])["files"] == ["sub/a.txt"]
    assert json.loads(tool_replies[1]["content"])["content"] == "hello"
    assert "Tool budget exhausted" in second[-1]["content"]  # both calls used the budget of two
    call_ids = [e["call_id"] for e in events if e["event_type"] == "tool_call"]
    assert call_ids == ["workspace-0", "workspace-1"]


async def test_loop_reasks_when_the_submission_is_not_a_valid_patch(tmp_path):
    (tmp_path / "a.txt").write_text("a\n")
    bad = "```diff\n<|tool_call_begin|> functions.read_file:0 {\"path\": \"a.txt\"}\n```"
    good = "```diff\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-a\n+b\n```"
    replies = [completion_for(bad), completion_for(good)]
    prompts = []

    class Provider:
        async def complete(self, messages, *, timeout_s, tools=()):
            prompts.append(messages[-1]["content"])
            return replies.pop(0)

    import time

    miner = DemoMiner(settings(), Provider())
    _, _, submission = await miner._generate(task(), WorkspaceReader(tmp_path), time.monotonic() + 10)
    assert submission.startswith(b"--- a/a.txt")
    assert prompts[-1].startswith("That reply was not accepted:")


async def test_loop_reask_gives_up_after_the_limit_and_uploads_the_last_reply(tmp_path):
    bad = completion_for("```diff\nnot a diff\n```")

    class Provider:
        calls = 0

        async def complete(self, messages, *, timeout_s, tools=()):
            self.calls += 1
            return bad

    import time

    provider = Provider()
    miner = DemoMiner(settings(), provider)
    completion, _, submission = await miner._generate(task(), WorkspaceReader(tmp_path), time.monotonic() + 10)
    assert completion is bad and provider.calls == 3  # first try plus two re-asks
    assert submission == b"not a diff\n"  # uploaded as is


@pytest.mark.skipif(__import__("shutil").which("git") is None, reason="git required")
async def test_check_submission_uses_git_and_repairs_only_what_git_accepts(tmp_path):
    import time

    (tmp_path / "a.txt").write_text("a\n")
    miner = DemoMiner(settings(), object())
    deadline = time.monotonic() + 30
    good = "```diff\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-a\n+b\n```"
    wrong_context = "```diff\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-zzz\n+b\n```"
    assert await miner._check_submission(task(), good, WorkspaceReader(tmp_path), deadline) == (extract_submission(good), None)
    submission, problem = await miner._check_submission(task(), wrong_context, WorkspaceReader(tmp_path), deadline)
    assert submission == extract_submission(wrong_context) and problem and "error" in problem
    assert (await miner._check_submission(task(), "```diff\n\x00\n```", WorkspaceReader(tmp_path), deadline))[1] == "patch contains a NUL byte"


@pytest.mark.skipif(__import__("shutil").which("git") is None, reason="git required")
async def test_check_submission_ignores_git_config_inside_the_workspace(tmp_path):
    import time

    (tmp_path / "a.txt").write_text("a\n")
    (tmp_path / ".gitattributes").write_text("* filter=evil\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text(f'[filter "evil"]\n\tclean = touch {tmp_path}/FILTER_RAN\n')
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    miner = DemoMiner(settings(), object())
    good = "```diff\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-a\n+b\n```"
    assert (await miner._check_submission(task(), good, WorkspaceReader(tmp_path), time.monotonic() + 30))[1] is None
    assert not (tmp_path / "FILTER_RAN").exists()


def test_find_files_is_recursive_in_directory_order_and_paged(tmp_path, monkeypatch):
    from rlvr.v3 import miner_workspace

    for name in ("b/z.txt", "b/a/x.txt", "a.txt"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
    (tmp_path / "link").symlink_to(tmp_path / "b")
    monkeypatch.setattr(miner_workspace, "LIST_ENTRIES", 2)
    reader = WorkspaceReader(tmp_path)
    # a directory's files come before its subdirectories; the symlink is skipped
    first = json.loads(reader.execute({"tool": "find_files", "path": ".", "offset": 0}))
    assert first == {"files": ["a.txt", "b/z.txt"], "next_offset": 2}
    second = json.loads(reader.execute({"tool": "find_files", "path": ".", "offset": 2}))
    assert second == {"files": ["b/a/x.txt"], "next_offset": None}
    assert "error" in json.loads(reader.execute({"tool": "find_files", "path": "a.txt", "offset": 0}))


@pytest.mark.parametrize("header, body, expected", [
    ("@@ -1,8 +1,17 @@", [" a", "-b", "+c", "+d"], "@@ -1,2 +1,3 @@"),
    ("@@ -1 +1 @@ fn main()", ["-a", "+b"], "@@ -1,1 +1,1 @@ fn main()"),
    ("@@ -3,2 +3,2 @@", [" x", "-y", "+z", "\\ No newline at end of file"], "@@ -3,2 +3,2 @@"),
])
def test_recount_hunks_fixes_only_the_counts(header, body, expected):
    from rlvr.neurons.demo_miner import recount_hunks

    patch = "--- a/f\n+++ b/f\n" + header + "\n" + "\n".join(body) + "\n"
    fixed = recount_hunks(patch.encode()).decode()
    assert fixed == "--- a/f\n+++ b/f\n" + expected + "\n" + "\n".join(body) + "\n"


@pytest.mark.skipif(__import__("shutil").which("git") is None, reason="git required")
async def test_recount_is_applied_only_when_git_accepts_the_repair(tmp_path):
    import time

    (tmp_path / "a.txt").write_text("one\ntwo\nthree\n")
    miner = DemoMiner(settings(), object())
    miscounted = "```diff\n--- a/a.txt\n+++ b/a.txt\n@@ -1,9 +1,12 @@\n one\n-two\n+TWO\n+more\n three\n```"
    submission, problem = await miner._check_submission(task(), miscounted, WorkspaceReader(tmp_path), time.monotonic() + 30)
    assert problem is None and submission.startswith(b"--- a/a.txt\n+++ b/a.txt\n@@ -1,3 +1,4 @@\n")


def test_recount_leaves_a_multi_file_diff_without_git_headers_alone_when_valid():
    from rlvr.neurons.demo_miner import recount_hunks

    # Two files, no "diff --git" lines: the second file's headers start with - and +.
    patch = b"--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n--- a/y\n+++ b/y\n@@ -1 +1 @@\n-c\n+d\n"
    assert recount_hunks(patch) != patch  # the naive recount would change it ...
    # ... which is why the miner only uses a recount that git has accepted.


async def test_a_repair_is_not_used_when_git_could_not_run(tmp_path, monkeypatch):
    import time

    (tmp_path / "a.txt").write_text("one\ntwo\nthree\n")
    miscounted = "```diff\n--- a/a.txt\n+++ b/a.txt\n@@ -1,9 +1,12 @@\n one\n-two\n+TWO\n+more\n three\n```"
    calls = []

    def fake_git_check(git, patch, workspace, timeout):
        calls.append(patch)
        if len(calls) == 1:
            return True, "error: corrupt patch at line 3"
        raise_timeout = __import__("subprocess").TimeoutExpired
        raise raise_timeout(cmd="git", timeout=timeout)

    def guarded(git, patch, workspace, timeout):
        try:
            return fake_git_check(git, patch, workspace, timeout)
        except __import__("subprocess").SubprocessError:
            return False, None

    monkeypatch.setattr("rlvr.neurons.demo_miner.shutil.which", lambda _: "/usr/bin/git")
    monkeypatch.setattr("rlvr.neurons.demo_miner._git_check", guarded)
    miner = DemoMiner(settings(), object())
    submission, problem = await miner._check_submission(task(), miscounted, WorkspaceReader(tmp_path), time.monotonic() + 30)
    assert submission == extract_submission(miscounted)  # the raw bytes, not the unverified repair
    assert problem == "error: corrupt patch at line 3"
    assert len(calls) == 2


async def test_no_git_check_runs_when_the_deadline_has_passed(tmp_path, monkeypatch):
    import time

    (tmp_path / "a.txt").write_text("a\n")
    monkeypatch.setattr("rlvr.neurons.demo_miner.shutil.which", lambda _: "/usr/bin/git")
    monkeypatch.setattr("rlvr.neurons.demo_miner._git_check", lambda *a: (_ for _ in ()).throw(AssertionError("must not run")))
    miner = DemoMiner(settings(), object())
    bad = "```diff\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-zzz\n+b\n```"
    assert await miner._check_submission(task(), bad, WorkspaceReader(tmp_path), time.monotonic() - 1) == (extract_submission(bad), None)


async def test_reask_is_skipped_and_the_reply_uploaded_when_time_runs_out(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("rlvr.neurons.demo_miner.time", SimpleNamespace(monotonic=lambda: clock[0]))
    (tmp_path / "a.txt").write_text("a\n")
    bad = "```diff\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-zzz\n+b\n```"
    calls = []

    class Provider:
        async def complete(self, messages, *, timeout_s, tools=()):
            calls.append(timeout_s)
            clock[0] += 60  # the model reply consumed the rest of the budget
            return completion_for(bad)

    async def failing_check(request, output, reader, deadline):
        return extract_submission(output), "error: patch failed: a.txt:1"

    miner = DemoMiner(settings(), Provider())
    monkeypatch.setattr(miner, "_check_submission", failing_check)
    _, _, submission = await miner._generate(task(), WorkspaceReader(tmp_path), 150)
    assert submission == extract_submission(bad) and calls == [50]  # no second call, no TimeoutError


async def test_reask_call_timeout_uploads_the_previous_reply(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("a\n")
    bad = "```diff\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-zzz\n+b\n```"
    calls = []

    class Provider:
        async def complete(self, messages, *, timeout_s, tools=()):
            calls.append(len(messages))
            if len(calls) == 1:
                return completion_for(bad)
            raise TimeoutError("OpenRouter request deadline exceeded")

    async def failing_check(request, output, reader, deadline):
        return extract_submission(output), "error: patch failed: a.txt:1"

    import time

    miner = DemoMiner(settings(), Provider())
    monkeypatch.setattr(miner, "_check_submission", failing_check)
    _, events, submission = await miner._generate(task(), WorkspaceReader(tmp_path), time.monotonic() + 10)
    assert submission == extract_submission(bad) and len(calls) == 2
    assert [e["event_type"] for e in events] == ["model_turn"]  # the failed correction attempt left no event


async def test_reask_then_tool_call_then_timeout_still_uploads_the_checked_reply(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("a\n")
    bad = "```diff\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-zzz\n+b\n```"
    replies = [completion_for(bad), completion_for('```workspace\n{"tool":"read_file","path":"a.txt","offset":0}\n```')]

    class Provider:
        async def complete(self, messages, *, timeout_s, tools=()):
            if replies:
                return replies.pop(0)
            raise TimeoutError("OpenRouter request deadline exceeded")

    async def failing_check(request, output, reader, deadline):
        return extract_submission(output), "error: patch failed: a.txt:1"

    import time

    miner = DemoMiner(settings(), Provider())
    monkeypatch.setattr(miner, "_check_submission", failing_check)
    _, events, submission = await miner._generate(task(), WorkspaceReader(tmp_path), time.monotonic() + 10)
    assert submission == extract_submission(bad)
    assert [e["event_type"] for e in events] == ["model_turn"]


async def test_over_budget_tool_call_reasks_for_the_submission_instead_of_failing(tmp_path):
    (tmp_path / "a.txt").write_text("a\n")
    call = completion_for('```workspace\n{"tool":"read_file","path":"a.txt","offset":0}\n```')
    diff = completion_for("```diff\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-a\n+b\n```")
    replies = [call, call, diff]  # second call is over a budget of one
    prompts = []

    class Provider:
        async def complete(self, messages, *, timeout_s, tools=()):
            prompts.append(messages[-1]["content"])
            return replies.pop(0)

    import time

    miner = DemoMiner(settings(miner_max_workspace_tool_calls=1), Provider())
    _, events, submission = await miner._generate(task(), WorkspaceReader(tmp_path), time.monotonic() + 10)
    assert submission.startswith(b"--- a/a.txt")
    assert sum(1 for e in events if e["event_type"] == "tool_call") == 1  # the over-budget call was not run
    assert sum(1 for e in events if e["event_type"] == "model_turn") == 3  # every model reply is still recorded
    assert "budget is exhausted" in prompts[-1]


async def test_over_budget_tool_calls_fail_after_the_reask_limit(tmp_path):
    call = completion_for('```workspace\n{"tool":"list_files","path":".","offset":0}\n```')

    class Provider:
        async def complete(self, messages, *, timeout_s, tools=()):
            return call  # never stops calling tools

    import time

    miner = DemoMiner(settings(miner_max_workspace_tool_calls=1), Provider())
    with pytest.raises(ValueError, match="tool call limit"):
        await miner._generate(task(), WorkspaceReader(tmp_path), time.monotonic() + 10)


def _provider_settings(**updates):
    return settings(openrouter_base_url="https://provider.invalid", **updates)


def _reply(**extra):
    return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": "x"}}], **extra})


async def _call(client_settings, handler, timeout_s=30.0, messages=None):
    """One completion against ``handler``; returns it and the request bodies sent."""
    bodies = []

    async def record(request):
        body = json.loads(request.content)
        bodies.append(body)
        return await handler(body)

    http = httpx.AsyncClient(transport=httpx.MockTransport(record))
    client = OpenRouterClient(client_settings, http=http)
    try:
        return await client.complete(messages or [{"role": "user", "content": "x"}], timeout_s=timeout_s), bodies
    finally:
        await http.aclose()


@pytest.fixture
def no_retry_wait(monkeypatch):
    monkeypatch.setattr("rlvr.neurons.demo_miner._retry_delay", lambda *_args: 0.0)


async def test_requests_go_to_openrouter_with_the_key():
    seen = {}

    async def handler(request):
        seen.update(url=str(request.url), auth=request.headers["authorization"], body=json.loads(request.content))
        return _reply()

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenRouterClient(settings(), http=http)
    await client.complete([{"role": "user", "content": "x"}], timeout_s=10)
    await http.aclose()
    assert seen["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert seen["auth"] == "Bearer key"
    assert not {"service_tier", "reasoning_effort", "include_reasoning"} & seen["body"].keys()


async def test_every_turn_of_a_solve_shares_one_session_and_solves_differ():
    async def handler(body):
        return _reply()

    opening = [{"role": "system", "content": "s"}, {"role": "user", "content": "task one"}]
    _, first = await _call(_provider_settings(), handler, messages=opening)
    _, later = await _call(_provider_settings(), handler, messages=[*opening, {"role": "assistant", "content": "a"}])
    _, other = await _call(_provider_settings(), handler, messages=[opening[0], {"role": "user", "content": "task two"}])
    assert first[0]["session_id"] == later[0]["session_id"] != other[0]["session_id"]
    assert len(first[0]["session_id"]) <= 256


async def test_each_call_logs_its_host_tokens_and_charge(capsys):
    async def handler(body):
        return _reply(provider="Fireworks", usage={"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.0123})

    result, _ = await _call(_provider_settings(), handler)
    assert result.usage == TokenUsage("Fireworks", 10, 0, 0, 2, 0.0123)
    out = capsys.readouterr().out
    assert "[demo-miner] model call: host=Fireworks input=10" in out and "cost=$0.0123" in out


@pytest.mark.parametrize("status", [408, 429, 502, 503])
async def test_transient_statuses_are_retried(status, no_retry_wait):
    replies = [httpx.Response(status, json={"error": {"code": status, "message": "busy"}}), _reply()]

    async def handler(body):
        return replies.pop(0)

    result, bodies = await _call(_provider_settings(), handler)
    assert result.output == "x" and len(bodies) == 2


def _payment_required(**metadata):
    return httpx.Response(402, json={"error": {"code": 402, "message": "credits", "metadata": metadata}})


async def test_the_in_flight_budget_402_is_retried(no_retry_wait):
    replies = [_payment_required(limit_source="openrouter_in_flight_budget"), _reply()]

    async def handler(body):
        return replies.pop(0)

    result, bodies = await _call(_provider_settings(), handler)
    assert result.output == "x" and len(bodies) == 2


@pytest.mark.parametrize("response", [
    lambda: _payment_required(),
    lambda: _payment_required(limit_source="openrouter_credits"),
    lambda: httpx.Response(402, text="not json"),
    lambda: httpx.Response(401, json={"error": {"code": 401, "message": "User not found."}}),
])
async def test_credit_and_key_errors_fail_at_once_and_log_why(response, capsys):
    calls = []

    async def handler(body):
        calls.append(body)
        return response()

    with pytest.raises(httpx.HTTPStatusError):
        await _call(_provider_settings(), handler)
    assert len(calls) == 1
    assert "[demo-miner] OpenRouter HTTP 40" in capsys.readouterr().out


@pytest.mark.parametrize("failed", [
    {"error": {"code": 502, "message": "Provider disconnected"}},
    {"choices": [{"finish_reason": "error", "message": {"content": "partial"}, "error": {"code": 502, "message": "gone"}}]},
])
async def test_a_failure_inside_a_200_body_is_retried(failed, no_retry_wait, capsys):
    replies = [httpx.Response(200, json=failed), _reply()]

    async def handler(body):
        return replies.pop(0)

    result, bodies = await _call(_provider_settings(), handler)
    assert result.output == "x" and len(bodies) == 2
    assert "generation failed" in capsys.readouterr().out


async def test_a_generation_that_keeps_failing_raises_after_the_retries(no_retry_wait):
    calls = []

    async def handler(body):
        calls.append(body)
        return httpx.Response(200, json={"error": {"code": 502, "message": "down"}})

    with pytest.raises(RuntimeError, match="could not complete"):
        await _call(_provider_settings(openrouter_max_retries=1), handler)
    assert len(calls) == 2


@pytest.mark.parametrize("headers, attempt, remaining, expected", [
    ({"Retry-After": "7"}, 0, 30.0, 7.0),
    ({"Retry-After": "90"}, 0, 30.0, 30.0),
    ({"Retry-After": "soon"}, 1, 30.0, 2.0),
    ({}, 2, 30.0, 4.0),
    ({}, 0, -1.0, 0.0),
])
def test_retry_delay_follows_the_server_within_the_time_left(headers, attempt, remaining, expected):
    from rlvr.neurons.demo_miner import _retry_delay

    assert _retry_delay(attempt, remaining, httpx.Response(429, headers=headers)) == expected


async def test_reasoning_goes_back_on_an_earlier_tool_call_turn(tmp_path):
    (tmp_path / "a.txt").write_text("alpha")
    read = completion_with_calls(("read_file", {"path": "a.txt", "offset": 0}))
    replies = [replace(read, reasoning="I need a.txt."), completion_for("```diff\n\n```")]
    seen = []

    class Provider:
        async def complete(self, messages, *, timeout_s, tools=()):
            seen.append([dict(m) for m in messages])
            return replies.pop(0)

    miner = DemoMiner(settings(), Provider())
    await miner._generate(task(), WorkspaceReader(tmp_path), time.monotonic() + 10)
    assistant = next(m for m in seen[1] if m["role"] == "assistant")
    assert assistant["reasoning"] == "I need a.txt." and assistant["tool_calls"][0]["id"] == "call_0"


def test_a_turn_without_reasoning_sends_none_back():
    from rlvr.neurons.demo_miner import _assistant_message

    message = _assistant_message(replace(completion_for("x"), reasoning=""), "x")
    assert message == {"role": "assistant", "content": "x"}


def test_parse_usage_reads_cache_fields_cost_and_host():
    data = {
        "provider": "Wafer",
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "cost": 0.25,
            "prompt_tokens_details": {"cached_tokens": 60, "cache_write_tokens": 30},
        },
    }
    assert parse_usage(data) == TokenUsage("Wafer", 100, 60, 30, 5, 0.25)


@pytest.mark.parametrize("data", [{}, {"usage": None}, {"usage": []}, [], None])
def test_parse_usage_returns_none_without_a_usage_object(data):
    assert parse_usage(data) is None


@pytest.mark.parametrize("cost", ["0.1", True, -1, float("inf"), None])
def test_parse_usage_ignores_malformed_counts_and_costs(cost):
    data = {"usage": {"prompt_tokens": "9", "completion_tokens": True, "prompt_tokens_details": [], "cost": cost}}
    assert parse_usage(data) == TokenUsage()


def test_solve_spend_totals_tokens_hosts_and_cost():
    spend = SolveSpend()
    spend.add(TokenUsage("Wafer", 1000, 700, 300, 100, 0.001))
    spend.add(TokenUsage("Fireworks", 1000, 1000, 0, 100, 0.002))
    line = spend.summary("abcdef" * 4)
    assert "task abcdefabcdefabcd calls=2 input=2000 cached=1700 (85%) written=300 output=200" in line
    assert "hosts=Fireworks:1,Wafer:1 cost=$0.0030" in line


def test_solve_spend_does_not_guess_when_a_call_went_unreported():
    spend = SolveSpend()
    spend.add(TokenUsage("Wafer", 1000, 0, 0, 100, 0.001))
    spend.add(None)
    line = spend.summary("task")
    assert "calls=1 unreported=1" in line and line.endswith("cost=unknown")


def test_solve_spend_does_not_guess_when_a_charge_went_unreported():
    spend = SolveSpend()
    spend.add(TokenUsage("Wafer", 1000, 0, 0, 100))
    assert spend.summary("task").endswith("cost=unknown")


def test_solve_spend_survives_a_provider_that_reports_no_usage_at_all():
    spend = SolveSpend()
    spend.add(None)
    assert "calls=0 unreported=1 input=0 cached=0 written=0" in spend.summary("task")


async def test_the_upload_reserve_is_45_seconds_and_the_solve_clock_starts_with_the_solve(monkeypatch, tmp_path):
    # Uploads have taken up to 38.8 s, and an upload cut by the solve's wait_for returns nothing (plan step 1).
    assert DemoMinerSettings(_env_file=None).openrouter_upload_reserve_s == 45.0

    class Clock:
        now = 5000.0

        def monotonic(self):
            return self.now

        def time(self):
            return time.time()

    clock = Clock()
    monkeypatch.setattr("rlvr.neurons.demo_miner.time", clock)

    @asynccontextmanager
    async def workspace(*_args, **_kwargs):
        clock.now += 15.0  # the download and the unpacking
        yield WorkspaceReader(tmp_path)

    seen = {}

    async def generate(self, request, reader, deadline, spend=None, notes=None):
        seen.update(deadline=deadline, started=notes.started, now=clock.now)
        raise RuntimeError("stop here")

    monkeypatch.setattr("rlvr.neurons.demo_miner.open_miner_workspace", workspace)
    monkeypatch.setattr(DemoMiner, "_generate", generate)
    miner = DemoMiner(settings(), None, wallet=WalletLike("miner"))
    with pytest.raises(TimeoutError, match="insufficient time"):
        await miner.solve(task(), 45.0)
    notes = SolveNotes()
    with pytest.raises(RuntimeError, match="stop here"):
        await miner.solve(task(), 900.0, notes)
    # The model works until 855 s after receipt, and its clock (and first_finish_s) counts from receipt, so the
    # workspace's 15 s are already used when the first message goes out.
    assert seen == {"deadline": 5855.0, "started": 5000.0, "now": 5015.0} and notes.started == 5000.0


async def test_solve_reports_spend_even_when_it_fails(monkeypatch, tmp_path, capsys):
    @asynccontextmanager
    async def workspace(*_args, **_kwargs):
        yield WorkspaceReader(tmp_path)

    class Provider:
        calls = 0

        async def complete(self, messages, *, timeout_s, tools=()):
            self.calls += 1
            if self.calls == 1:
                reply = completion_for('```workspace\n{"tool":"list_files","path":".","offset":0}\n```')
                return replace(reply, usage=TokenUsage("Wafer", 1000, 0, 1000, 10))
            raise RuntimeError("provider down")

    monkeypatch.setattr("rlvr.neurons.demo_miner.open_miner_workspace", workspace)
    miner = DemoMiner(settings(), Provider(), wallet=WalletLike("miner"))
    with pytest.raises(RuntimeError, match="provider down"):
        await miner.solve(task(), 90)
    out = capsys.readouterr().out
    assert "solve spend" in out and "calls=1 input=1000" in out


def test_prompt_asks_for_reads_to_be_batched():
    system = build_model_messages(task())[0]["content"]
    assert "request them together as separate tool calls in one reply." in system
    assert "up to 8" not in system  # the code has no cap of 8 (prompt audit finding 14)


# The repository system message with a sandbox before the prompt replacements of audit findings 3, 5, 6, 7, 10, 11,
# 12 and 14 (plan.md step 2). It was 26,935 characters at 22:23:43 on 2026-11-13; the live file of 22:49:10 added
# the public-function sentence to IMPLICIT_RULES (1,493 characters), and step 2 was merged onto that file for the
# first deploy, so the pin is re-measured there: 28,428. The plan expected about the same size; its texts, applied
# verbatim, make it 684 characters longer (findings 3, 5, 6, 7, 11 and 12 add 995, finding 10 and finding 14's trims
# take away 311). The merge then made finding 3 point to the public-function sentence instead of repeating it ("a
# public helper function" and "or a formula" became "a public function as the rule on public functions below sets
# out"), 27 more, so the step adds 711 characters, about 180 cached tokens.
# Re-measured again for the second merge (branch first-deploy-2, 2026-11-14): the live file of 00:22:22 on Nov 14 added
# VALUE_STRING_RULE after SHAPE_RULE, the text-format spelling sentence after finding 3's checklist sentence, the
# wrapper-precedence clause and the Go error-kind sentence in IMPLICIT_RULES, 3,264 characters in all, so the system
# message before step 2 is now 31,692 (28,428 + 3,264). Step 2 replaces the same old finding 3 text there as before,
# so it still adds 711. The merge adds 38 more of its own: "is not a rule of the value's kind: it " makes the
# spelling sentence the stated exception to finding 3, which now names a size or count and a grammar (see the
# comment beside the sentence in demo_miner.py). With a sandbox the merged message is 32,441 = 31,692 + 711 + 38.
# Third merge (branch first-deploy-3, 2026-11-14): the live file of 01:46:35 added the deduplication rule and
# STEPS_RULE, 1,945 characters, so the message before step 2 is 33,637 and the merged one 34,386 = 33,637 + 711 + 38.
# Fourth change (session c65b0987, 2026-11-14 ~03:00): three consistency fixes (a state tested first answers first;
# a bound on a step's result is counted on that result; SECTION_RULE's decoder clause only where the section splits
# shape and lookup codes) add 686 characters, so the base is re-measured on the merged file: 35,072 - 711 - 38.
# Fifth change (session c65b0987, 2026-11-14 ~05:50): ERROR_ID_RULE after ERROR_SCOPE (an unknown id passed to a call
# is reported as passed; 121c13873690) adds 473 characters, so the base is re-measured: 35,545 - 711 - 38.
# Sixth change (session c65b0987, 2026-11-14 ~15:00): the byte-order-mark clause on the json.loads sentence
# (B23, 4c763391ef17) and the text deployed since, so the base is re-measured: 35,738 - 711 - 38.
SYSTEM_CHARS_BEFORE_STEP_2 = 34_989
STEP_2_ADDS = 711
SPELLING_EXCEPTION_ADDS = 38


def test_the_prompt_replacements_of_the_audit_reach_the_system_message_once_and_the_old_texts_are_gone():
    import re

    from rlvr.neurons import adjudicate

    system = build_model_messages(task(), can_run=True)[0]["content"]
    # No task id, no "up to 8", no "never write a diff yourself", and none of the replaced instance texts.
    assert not re.search(r"\(\s*[0-9a-f]{12}\s*\)", system)
    for gone in ("up to 8", "never write a diff yourself", "grammar or forbidden-character rule",
                 "ce := resp.CodeError", "the embedded resp.CodeError", "against a *T target",
                 "deeper than Python's recursion limit", "never check its class",
                 "every example in the documentation verbatim", "copies or preserves data from files",
                 "with a heredoc in /tmp", "so it cannot end up in your patch",
                 "A standard form encoder is not that encoding unless", "never +, even where"):
        assert gone not in system, gone
    new_clauses = (
        # finding 3: every rule of a named kind of value, ranges and length bounds included, at every carrier. Merged
        # with the live public-function sentence of 22:49 (same task, b4ab73f800ed): the carrier "a public helper
        # function" and "or a formula" became a pointer to that sentence, so its exclusions decide public functions.
        "Note also where each rule stated for a named kind of value applies, whatever the rule is (a grammar, a set "
        "of forbidden characters, a range, a length bound, a normalization) and whatever the kind (a header name or "
        "value, an identifier, a channel name, a path, a size or count): every input that carries a value of that "
        "same kind (a profile, a record, a patch object, a defaults object, a constructor argument, a public "
        "function as the rule on public functions below sets out), even when its own section gives only the "
        "value's type, unless that input's own section states a different rule or a sentence exempts it; check each "
        "place through the one helper the first place uses, with that place's own error code and pointer, and give "
        # The live spelling sentence of 00:03 (Nov 14) now follows finding 3 in the same bullet, so the clause ends at
        # its full stop rather than at the newline; the order is checked below.
        "each place a case of its own in your test.",
        # finding 5: every documented equality becomes one == assertion
        "for every sentence that says one value equals, copies, preserves or reproduces another (\"a deep copy of\", "
        "\"the same as\", \"preserved verbatim\", \"yields the same ... as\", \"the identical object\"), one assertion "
        "that compares the two whole values with == (is, for identity) on real inputs, the repository's shipped "
        "files among them (every variant the documentation names), with the source loaded the way the "
        "documentation loads it (runpy, json, the project's loader), never through your own helpers;",
        # finding 6: a type check only where the documentation requires one
        "so check its type only where the documentation requires one, and then with isinstance against the "
        "documented base, so that subclasses and stand-ins built on it pass while a class passed for an instance "
        "still fails; never call its private methods or keep a reference taken at import.",
        # finding 7: a named encoding function or a shown encoded form wins
        "Where the documentation names the encoding function or shows the encoded form (encodeURIComponent, "
        "url.PathEscape, quote with safe=''), use exactly that. Otherwise, re-encoding \"with escapes\" (canonical "
        "URLs, paths, query keys and values) writes every byte outside A-Z a-z 0-9 - . _ ~ as an uppercase %HH "
        "escape, except the characters a sentence or example keeps literal (a structural /, the ? & = separators): "
        "a space becomes %20 and a literal + becomes %2B, even where the decoder reads + as a space, so Go "
        "url.QueryEscape, Python quote_plus or urlencode and JavaScript URLSearchParams, which write + for a space, "
        "are not that encoding. A lowercase escape on input comes out uppercase.",
        # finding 10: the general Go error form, and the short recursion clause
        "In Go, an empty value is not nil, and an error is returned and stored in the form the project's own code "
        "returns that type (a pointer where the project returns &T{...}), because the tests match it with err.(*T) "
        "or errors.As. ",
        "Traverse recursive structures iteratively and compute documented sizes without building the data.",
        # finding 11 with the critic's addition: Go and Rust scratch tests in the package, removed in the same command
        "Create it from run_command with a heredoc, never with write_file: in /tmp, or for Go and Rust in the "
        "package directory or tests/ under a name such as zz_scratch_test.go, created, run and removed in the same "
        "command, so that no later build or suite run picks it up. Files that commands create are never submitted.",
        # finding 12: examples are a check of the reading, weighed by the example rule
        "and every example in the documentation as a check of your reading, weighed by the rule on examples above: "
        "where an example and a sentence that spells out the same case disagree, the test follows the sentence.",
        # finding 14's trims keep the rest of their sentences
        "Your edits become a Git unified diff automatically. Every file you create or change with them is submitted",
    )
    for clause in new_clauses:
        assert system.count(clause) == 1, clause
    # The run prompt's example clause points back at the example rule, which comes before it.
    assert system.index(adjudicate.EXAMPLE_RULE) < system.index("weighed by the rule on examples above")
    # Finding 3 and the public-function sentence state the public-function case once: finding 3 only points to the
    # sentence, which comes after it and keeps its exclusions.
    pointer = "a public function as the rule on public functions below sets out"
    public = "They also call each documented public function directly"
    assert system.count(pointer) == 1 and system.count(public) == 1
    assert system.index(pointer) < system.index(public)
    assert "public helper function" not in system and "only the value's type or a formula" not in system
    for kept in ("A predicate (is_x(...) -> bool) still answers False", "a Go func returning only a string",
                 "such a function gains no bound from a table"):
        assert system.count(kept) == 1, kept
    # The live spelling sentence follows finding 3 directly, as the stated exception to it, once.
    spelling = ("each place a case of its own in your test. A spelling the documentation gives for how one text "
                "format writes a value whose own type is not text")
    assert system.count(spelling) == 1
    assert system.count("is not a rule of the value's kind: it belongs to that format's parser alone.") == 1
    assert system.index(spelling) < system.index("keeps its one grammar everywhere, as above.\n- Exact means exact.")
    # The replacements keep the system message's size near the size before them. The pin is absolute on purpose:
    # text another change adds to the system message moves it, and then SYSTEM_CHARS_BEFORE_STEP_2 is re-measured
    # on the merged base (as it was for both merges, above). The upper margin is the step's own 711 characters, the
    # second merge's 38 and the same small slack as before (14).
    assert SYSTEM_CHARS_BEFORE_STEP_2 - 300 <= len(system) <= (
        SYSTEM_CHARS_BEFORE_STEP_2 + STEP_2_ADDS + SPELLING_EXCEPTION_ADDS + 14), (
        f"the system message is {len(system)} characters; re-measure SYSTEM_CHARS_BEFORE_STEP_2 if other text was "
        "added to it")
    # Finding 14's summary change and finding 8's reframing are not part of this step.
    assert "reply with a one-line summary and no tool call" in system
    assert "fixing a real codebase" in system and "Keep the change minimal" in system
    # Without a sandbox the run prompt's clauses stay out, and the shared system clauses are the same.
    blind = build_model_messages(task(), can_run=False)[0]["content"]
    assert "zz_scratch_test.go" not in blind and "the identical object\")" not in blind
    assert blind.count(new_clauses[0]) == 1 and blind.count(new_clauses[3]) == 1


def _native_batch(*paths):
    calls = " ".join(
        f'<|tool_call_begin|> functions.read_file:{i} <|tool_call_argument_begin|> {{"path": "{path}", "offset": 0}} <|tool_call_end|>'
        for i, path in enumerate(paths)
    )
    return f" <|tool_calls_section_begin|> {calls} <|tool_calls_section_end|>"


def test_extract_tool_calls_returns_every_native_call_in_order():
    found = [json.loads(call) for call in extract_tool_calls(_native_batch("a.txt", "b.txt"))]
    assert [call["path"] for call in found] == ["a.txt", "b.txt"]
    assert all(call["tool"] == "read_file" for call in found)
    assert extract_tool_calls("--- a/x\n+++ b/x\n") == []


async def test_loop_runs_every_call_in_a_native_markup_batch(tmp_path):
    (tmp_path / "a.txt").write_text("alpha")
    (tmp_path / "b.txt").write_text("beta")
    replies = [completion_for(_native_batch("a.txt", "b.txt")), completion_for("```diff\n\n```")]
    prompts = []

    class Provider:
        async def complete(self, messages, *, timeout_s, tools=()):
            prompts.append(messages[-1]["content"])
            return replies.pop(0)

    miner = DemoMiner(settings(), Provider())
    _, events, _ = await miner._generate(task(), WorkspaceReader(tmp_path), time.monotonic() + 10)
    assert sum(1 for e in events if e["event_type"] == "tool_call") == 2
    assert prompts[-1].startswith("Workspace tool results, in the order requested:\n")
    assert prompts[-1].index("alpha") < prompts[-1].index("beta")


async def test_batch_over_the_budget_runs_what_fits_and_answers_the_rest(tmp_path):
    (tmp_path / "a.txt").write_text("alpha")
    read = ("read_file", {"path": "a.txt", "offset": 0})
    replies = [completion_with_calls(read, read, read), completion_for("```diff\n\n```")]
    seen = []

    class Provider:
        async def complete(self, messages, *, timeout_s, tools=()):
            seen.append([dict(m) for m in messages])
            return replies.pop(0)

    miner = DemoMiner(settings(miner_max_workspace_tool_calls=2), Provider())
    _, events, _ = await miner._generate(task(), WorkspaceReader(tmp_path), time.monotonic() + 10)
    assert sum(1 for e in events if e["event_type"] == "tool_call") == 2
    tool_replies = [m for m in seen[1] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_replies] == ["call_0", "call_1", "call_2"]
    assert "alpha" in tool_replies[1]["content"] and "not run" in tool_replies[2]["content"]
    assert "Tool budget exhausted" in seen[1][-1]["content"]


async def _request_body(**updates) -> dict:
    seen = {}

    async def handler(request):
        seen.update(json.loads(request.content))
        return _reply()

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenRouterClient(settings(**updates), http=http)
    await client.complete([{"role": "user", "content": "x"}], timeout_s=10)
    await http.aclose()
    return seen


@pytest.mark.parametrize("model", ["anthropic/claude-sonnet-5.5", "~anthropic/claude-sonnet-latest"])
async def test_claude_requests_turn_on_prompt_caching(model):
    body = await _request_body(openrouter_model=model)
    assert body["cache_control"] == {"type": "ephemeral"}


async def test_other_models_are_not_sent_the_claude_cache_field():
    assert "cache_control" not in await _request_body(openrouter_model="moonshotai/kimi-k3")


async def test_the_default_request_leaves_out_what_sonnet_does_not_accept():
    body = await _request_body()
    assert body["model"] == "anthropic/claude-sonnet-5.5" and body["reasoning"] == {"effort": "high"}
    assert not {"temperature", "top_p", "top_k", "logprobs", "top_logprobs"} & body.keys()


async def test_temperature_and_logprobs_are_sent_only_when_configured():
    body = await _request_body(
        openrouter_model="moonshotai/kimi-k3", openrouter_temperature=0.0, openrouter_logprobs=True
    )
    assert (body["temperature"], body["logprobs"], body["top_logprobs"]) == (0.0, True, 5)


async def test_xhigh_effort_is_allowed():
    assert (await _request_body(openrouter_reasoning_effort="xhigh"))["reasoning"] == {"effort": "xhigh"}


SIGNED_REASONING = [
    {"type": "reasoning.text", "text": "summary", "signature": "sig", "format": "anthropic-claude-v1", "index": 0}
]


async def test_structured_reasoning_is_kept_from_the_reply():
    async def handler(request):
        message = {"content": "x", "reasoning": "summary", "reasoning_details": SIGNED_REASONING}
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": message}]})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenRouterClient(settings(), http=http)
    result = await client.complete([{"role": "user", "content": "x"}], timeout_s=10)
    await http.aclose()
    assert result.reasoning == "summary" and result.reasoning_details == SIGNED_REASONING


async def test_signed_reasoning_goes_back_unchanged_instead_of_the_text(tmp_path):
    (tmp_path / "a.txt").write_text("alpha")
    read = replace(
        completion_with_calls(("read_file", {"path": "a.txt", "offset": 0})),
        reasoning="summary",
        reasoning_details=SIGNED_REASONING,
    )
    replies = [read, completion_for("```diff\n\n```")]
    seen = []

    class Provider:
        async def complete(self, messages, *, timeout_s, tools=()):
            seen.append([dict(m) for m in messages])
            return replies.pop(0)

    await DemoMiner(settings(), Provider())._generate(task(), WorkspaceReader(tmp_path), time.monotonic() + 10)
    assistant = next(m for m in seen[1] if m["role"] == "assistant")
    assert assistant["reasoning_details"] == SIGNED_REASONING and "reasoning" not in assistant
    assert assistant["tool_calls"][0]["id"] == "call_0"


def test_an_empty_reply_is_not_sent_back_with_only_its_reasoning():
    from rlvr.neurons.demo_miner import _assistant_message

    turn = replace(completion_for(""), reasoning="thinking", reasoning_details=SIGNED_REASONING)
    assert _assistant_message(turn, "") == {"role": "assistant", "content": "(empty reply)"}


async def test_over_budget_reask_answers_provider_calls_and_keeps_them_as_generated(tmp_path):
    from rlvr.neurons.demo_miner import _TOOL_NOT_RUN

    (tmp_path / "a.txt").write_text("a\n")
    read = replace(
        completion_with_calls(("read_file", {"path": "a.txt", "offset": 0})),
        reasoning_details=SIGNED_REASONING,
    )
    replies = [read, read, completion_for("```diff\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-a\n+b\n```")]
    seen = []

    class Provider:
        async def complete(self, messages, *, timeout_s, tools=()):
            seen.append([dict(m) for m in messages])
            return replies.pop(0)

    miner = DemoMiner(settings(miner_max_workspace_tool_calls=1), Provider())
    _, events, _ = await miner._generate(task(), WorkspaceReader(tmp_path), time.monotonic() + 10)
    assistant, tool_reply, request = seen[-1][-3:]
    assert assistant["tool_calls"][0]["id"] == "call_0" and assistant["reasoning_details"] == SIGNED_REASONING
    assert tool_reply == {"role": "tool", "tool_call_id": "call_0", "content": _TOOL_NOT_RUN.decode()}
    assert "budget is exhausted" in request["content"]
    assert sum(e["event_type"] == "tool_call" for e in events) == 1


async def test_a_reply_cut_at_the_token_limit_keeps_its_text_but_not_a_half_tool_call(capsys):
    async def handler(request):
        message = {"content": "partial", "tool_calls": [{"id": "c", "function": {"name": "read_file", "arguments": "{\"pa"}}]}
        return httpx.Response(200, json={"choices": [{"finish_reason": "length", "message": message}]})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    completion = await OpenRouterClient(settings(), http=http).complete([{"role": "user", "content": "x"}], timeout_s=10)
    await http.aclose()
    assert completion.cut_off and not completion.refused
    assert completion.output == "partial" and completion.tool_calls == ()
    assert "cut at the token limit" in capsys.readouterr().out


async def test_a_refused_reply_is_returned_as_such_with_the_hosts_reason(capsys):
    async def handler(request):
        choice = {
            "finish_reason": "content_filter", "native_finish_reason": "refusal",
            "message": {"content": None, "refusal": "This request was blocked as it seems to violate the terms"},
        }
        return httpx.Response(200, json={"choices": [choice]})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    completion = await OpenRouterClient(settings(), http=http).complete([{"role": "user", "content": "x"}], timeout_s=10)
    await http.aclose()
    assert completion.refused and completion.output == "" and completion.tool_calls == ()
    assert completion.refusal.startswith("This request was blocked")
    assert "the host refused the reply (refusal): This request was blocked" in capsys.readouterr().out


async def test_an_unknown_finish_reason_names_itself_in_the_error():
    async def handler(request):
        return httpx.Response(200, json={"choices": [{"finish_reason": "paused", "message": {"content": "partial"}}]})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenRouterClient(settings(), http=http)
    with pytest.raises(RuntimeError, match="invalid response: OpenRouter completion did not finish normally: paused"):
        await client.complete([{"role": "user", "content": "x"}], timeout_s=10)
    await http.aclose()
