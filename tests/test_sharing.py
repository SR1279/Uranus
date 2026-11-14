"""One task reaching several of this host's hotkeys is solved once, and each UID still answers as itself."""

from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager

import pytest

from rlvr.neurons import demo_miner
from rlvr.neurons.demo_miner import DemoMiner, SolveNotes
from rlvr.neurons.sharing import CLAIM_GRACE_S, SharedAnswer, SharedSolves
from rlvr.v3.api import MinerTaskResponse
from rlvr.v3.miner_workspace import WorkspaceReader
from rlvr.v3.trajectory import parse_trajectory
from tests.test_demo_miner import WalletLike, completion_for, settings, task

EVENTS = [{"event_type": "model_turn", "request_body_b64": "e30=", "response_body_b64": "e30="}]


def answer(hotkey="hot1", submission=b"diff"):
    return SharedAnswer(submission=submission, events=EVENTS, model_name="m", hotkey=hotkey, fields={"ended_by": "finished"})


def test_one_process_claims_the_others_wait_and_read_the_published_answer(tmp_path):
    first, second = SharedSolves(tmp_path, "hot1"), SharedSolves(tmp_path, "hot2")
    until = time.time() + 600
    assert first.claim("k", until) == 0
    assert second.claim("k", until) is None and second.state("k", 0) == "working"
    first.publish("k", 0, answer())
    assert second.state("k", 0) == "answered" and second.claim("k", until) is None
    read = second.read("k", 0)
    assert (read.submission, read.events, read.hotkey, read.fields) == (b"diff", EVENTS, "hot1", {"ended_by": "finished"})


def test_a_failed_or_dead_holder_passes_the_work_to_the_next_generation(tmp_path):
    store = SharedSolves(tmp_path, "hot2")
    until = time.time() + 600
    assert store.claim("k", until) == 0
    store.fail("k", 0, "RuntimeError: no model")
    assert store.state("k", 0) == "failed" and store.claim("k", until) == 1
    # A holder whose process is gone, or that is past its own deadline, is dead.
    (tmp_path / "d.0.claim").write_text(json.dumps({"pid": 2**22 + 12345, "until": until}))
    assert store.state("d", 0) == "dead" and store.claim("d", until) == 1
    (tmp_path / "e.0.claim").write_text(json.dumps({"pid": os.getpid(), "until": time.time() - CLAIM_GRACE_S - 1}))
    assert store.state("e", 0) == "dead"


def test_everything_is_forgotten_after_the_time_to_live(tmp_path):
    clock = [1_000_000.0]
    store = SharedSolves(tmp_path, "hot1", ttl_s=100, clock=lambda: clock[0])
    assert store.claim("k", clock[0] + 50) == 0
    store.publish("k", 0, answer())
    for path in tmp_path.iterdir():
        os.utime(path, (clock[0], clock[0]))
    clock[0] += 101
    # The same task seen again later is solved afresh, not answered from an old round.
    assert store.claim("k", clock[0] + 50) == 0 and store.state("k", 0) == "working"


def test_the_key_is_the_task_not_the_challenge_or_the_uid():
    one, other = task("hot1"), task("hot2")
    other = other.model_copy(update={"challenge_id": "another-round"})
    assert SharedSolves.key(one) == SharedSolves.key(other)
    changed = one.model_copy(update={"task_id": "f" * 64})
    assert SharedSolves.key(changed) != SharedSolves.key(one)


async def test_waiting_ends_with_the_answer_a_failure_or_the_deadline(tmp_path):
    first, second = SharedSolves(tmp_path, "hot1"), SharedSolves(tmp_path, "hot2")
    assert first.claim("k", time.time() + 600) == 0
    assert await second.wait("k", time.monotonic() + 0.05) == ("timeout", None)
    first.publish("k", 0, answer())
    state, found = await second.wait("k", time.monotonic() + 5)
    assert state == "answered" and found.submission == b"diff"
    assert first.claim("j", time.time() + 600) == 0
    first.fail("j", 0, "boom")
    assert await second.wait("j", time.monotonic() + 5) == ("failed", None)


class Gate:
    """A provider whose replies wait for a signal, counting the model calls."""

    def __init__(self, fail=False):
        self.calls = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.fail = fail

    async def complete(self, messages, *, timeout_s, tools=(), **options):
        self.calls += 1
        self.started.set()
        await self.release.wait()
        if self.fail:
            raise RuntimeError("the model is unreachable")
        return completion_for("All done.")

    async def aclose(self):
        return None


@pytest.fixture
def fakes(tmp_path, monkeypatch):
    monkeypatch.setattr(demo_miner, "SUBMISSION_REASK_LIMIT", 0)
    (tmp_path / "ws").mkdir()
    (tmp_path / "ws" / "a.txt").write_text("content\n")
    opened, uploads = [], []

    @asynccontextmanager
    async def workspace(*_args, **_kwargs):
        opened.append(1)
        yield WorkspaceReader(tmp_path / "ws")

    async def upload(_http, request, submission, trajectory, *, allowed_origins):
        uploads.append((request, submission, trajectory))
        return MinerTaskResponse(
            protocol_version=3, challenge_id=request.challenge_id, task_id=request.task_id,
            response_type=request.identity.task_type,
            submission={"artifact_role": "patch", "artifact_format": "unified_diff_v1",
                        "upload_id": request.slots.submission.upload_id, "sha256": "0" * 64, "size_bytes": len(submission)},
            trajectory={"artifact_role": "trajectory", "artifact_format": "trajectory_v1",
                        "upload_id": request.slots.trajectory.upload_id, "sha256": "1" * 64, "size_bytes": len(trajectory)},
        )

    monkeypatch.setattr(demo_miner, "open_miner_workspace", workspace)
    monkeypatch.setattr(demo_miner, "upload_miner_result", upload)
    return opened, uploads


def miner(tmp_path, hotkey, provider):
    return DemoMiner(settings(miner_shared_dir=str(tmp_path / "shared")), provider, wallet=WalletLike(hotkey))


async def test_two_hotkeys_one_model_run_each_uid_uploads_and_names_itself(tmp_path, fakes):
    opened, uploads = fakes
    worker_model, waiter_model = Gate(), Gate()
    first, second = miner(tmp_path, "hot1", worker_model), miner(tmp_path, "hot2", waiter_model)
    second_request = task("hot2").model_copy(update={"challenge_id": task("hot2").challenge_id})
    first_notes, second_notes = SolveNotes(), SolveNotes()
    solving = asyncio.create_task(first.solve(task("hot1"), 60, first_notes))
    await worker_model.started.wait()
    waiting = asyncio.create_task(second.solve(second_request, 60, second_notes))
    await asyncio.sleep(0.2)
    worker_model.release.set()
    await asyncio.gather(solving, waiting)
    assert waiter_model.calls == 0 and worker_model.calls >= 1 and len(opened) == 1  # the model and the download, once
    assert second_notes.ended_by == "shared" and second_notes.shared_from == "hot1" and first_notes.shared_from == ""
    by_hotkey = {parse_trajectory(trajectory).miner_hotkey: (request, submission, parse_trajectory(trajectory))
                 for request, submission, trajectory in uploads}
    assert set(by_hotkey) == {"hot1", "hot2"}
    (r1, s1, t1), (r2, s2, t2) = by_hotkey["hot1"], by_hotkey["hot2"]
    assert s1 == s2 and t1.submission_sha256 == t2.submission_sha256
    assert r1.slots.submission.hotkey == "hot1" and r2.slots.submission.hotkey == "hot2"  # each UID's own slots
    assert t2.challenge_id == r2.challenge_id and t2.task_id == r2.task_id
    assert [e.event_type for e in t1.events] == [e.event_type for e in t2.events]
    assert second_notes.spend.cost_so_far() == 0


async def test_when_the_solving_process_fails_a_waiter_with_time_solves_it_itself(tmp_path, fakes, monkeypatch):
    opened, uploads = fakes
    monkeypatch.setattr(demo_miner, "SHARE_TAKEOVER_MIN_S", 1.0)
    failing, healthy = Gate(fail=True), Gate()
    healthy.release.set()
    first, second = miner(tmp_path, "hot1", failing), miner(tmp_path, "hot2", healthy)
    solving = asyncio.create_task(first.solve(task("hot1"), 60, SolveNotes()))
    await failing.started.wait()
    waiting = asyncio.create_task(second.solve(task("hot2"), 60, SolveNotes()))
    await asyncio.sleep(0.2)
    failing.release.set()
    with pytest.raises(RuntimeError):
        await solving
    await waiting
    assert healthy.calls >= 1 and len(opened) == 2
    assert [parse_trajectory(t).miner_hotkey for _r, _s, t in uploads] == ["hot2"]


async def test_without_a_shared_directory_nothing_changes(tmp_path, fakes):
    opened, uploads = fakes
    model = Gate()
    model.release.set()
    alone = DemoMiner(settings(), model, wallet=WalletLike("hot1"))
    assert alone.shared is None
    notes = SolveNotes()
    await alone.solve(task("hot1"), 60, notes)
    assert notes.shared_from == "" and len(opened) == 1 and len(uploads) == 1


async def test_a_broken_store_never_costs_the_task(tmp_path, fakes, monkeypatch):
    opened, uploads = fakes
    model = Gate()
    model.release.set()
    alone = miner(tmp_path, "hot1", model)

    def broken(*_args, **_kwargs):
        raise PermissionError("read-only file system")

    monkeypatch.setattr(alone.shared, "claim", broken)
    notes = SolveNotes()
    await alone.solve(task("hot1"), 60, notes)
    assert model.calls >= 1 and len(uploads) == 1 and notes.shared_from == ""
    assert not any(name.endswith("answer.gz") for name in os.listdir(tmp_path / "shared"))  # nothing published
