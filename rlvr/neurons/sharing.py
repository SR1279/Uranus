"""Answers shared between this host's miner processes, one process per hotkey, keyed by the task.

A validator sends each task to a random pool of serving miners at once, so an operator with several
registered hotkeys often receives the same task on more than one of them within seconds. The validator
grades byte-identical submissions once and gives every miner that sent them the same verdict ("it changes no
score", rlvr/v3/round.py), and the release policy pays every pass the same whatever its latency. So the
model's work is done once: the first process to receive a task claims it and solves it, and the others wait
for its answer. Each process still writes its own trajectory (its own hotkey and challenge), uploads to its own
UID's slots and signs its response with its own hotkey; only the model's work is shared.

The store is a directory every process can write (MINER_SHARED_DIR). A claim is a file created with O_EXCL, so
exactly one process wins it; the answer is written beside it atomically. If the solving process fails, or
dies, the claim moves to the next generation and a waiter with enough time left solves the task itself.
Everything is kept only SHARE_TTL_S after its claim: a task seen again later is solved afresh.
"""

from __future__ import annotations

import asyncio
import base64
import gzip
import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SHARE_TTL_S = 1800.0
POLL_S = 1.0
# A claim whose holder neither answered nor failed this long after its own deadline is treated as dead.
CLAIM_GRACE_S = 60.0
MAX_GENERATIONS = 4


@dataclass
class SharedAnswer:
    submission: bytes
    events: list[dict[str, Any]]
    model_name: str
    hotkey: str
    # The solving process's record of the work, for each waiter's journal.
    fields: dict[str, Any] = field(default_factory=dict)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, ValueError, OverflowError):
        return False
    return True


class SharedSolves:
    """The shared store; one instance per process."""

    def __init__(self, root: Path, hotkey: str, *, ttl_s: float = SHARE_TTL_S, clock: Any = time.time):
        self.root = root
        self.hotkey = hotkey
        self.ttl_s = ttl_s
        self.clock = clock
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(request: Any) -> str:
        """The task itself: its id and the digests of what it is made of, the same on every UID it reaches."""
        identity = request.identity
        material = "\n".join(
            str(part) for part in (
                request.task_id,
                identity.task_type,
                getattr(identity, "workspace_sha256", ""),
                getattr(identity, "verifier_sha256", ""),
                request.workspace.sha256,
            )
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:40]

    def _path(self, key: str, generation: int, suffix: str) -> Path:
        return self.root / f"{key}.{generation}.{suffix}"

    def _sweep(self, key: str | None = None) -> None:
        """Forget what is older than the time to live: this key's files, or anything twice that old."""
        now = self.clock()
        try:
            names = os.listdir(self.root)
        except OSError:
            return
        for name in names:
            mine = key is not None and name.startswith(key + ".")
            path = self.root / name
            try:
                age = now - path.stat().st_mtime
            except OSError:
                continue
            if (mine and age > self.ttl_s) or age > 2 * self.ttl_s:
                try:
                    path.unlink()
                except OSError:
                    pass

    def _claim_info(self, key: str, generation: int) -> dict[str, Any] | None:
        try:
            return json.loads(self._path(key, generation, "claim").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def state(self, key: str, generation: int) -> str:
        """answered, failed, working, dead (the holder is gone without a word) or open (nobody claimed it)."""
        if self._path(key, generation, "answer.gz").exists():
            return "answered"
        if self._path(key, generation, "failed").exists():
            return "failed"
        if not self._path(key, generation, "claim").exists():
            return "open"
        info = self._claim_info(key, generation)
        if info is None:
            return "working"  # being written this instant
        if not _pid_alive(int(info.get("pid", 0))) or self.clock() > float(info.get("until", 0)) + CLAIM_GRACE_S:
            return "dead"
        return "working"

    def claim(self, key: str, until: float) -> int | None:
        """Claim the task's work: the generation claimed, or None when another process is on it or has answered.

        `until` is this process's own deadline (epoch seconds), after which its claim may be taken over.
        """
        self._sweep(key)
        for generation in range(MAX_GENERATIONS):
            state = self.state(key, generation)
            if state in ("failed", "dead"):
                continue
            if state in ("answered", "working"):
                return None
            path = self._path(key, generation, "claim")
            try:
                descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                return None  # another process won this generation just now
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump({"hotkey": self.hotkey, "pid": os.getpid(), "claimed": self.clock(), "until": until}, handle)
            return generation
        return None

    def current(self, key: str) -> int:
        """The latest generation anyone claimed."""
        latest = 0
        for generation in range(MAX_GENERATIONS):
            if self._path(key, generation, "claim").exists():
                latest = generation
        return latest

    def publish(self, key: str, generation: int, answer: SharedAnswer) -> None:
        payload = {
            "submission_b64": base64.b64encode(answer.submission).decode("ascii"),
            "events": answer.events,
            "model_name": answer.model_name,
            "hotkey": answer.hotkey,
            "fields": answer.fields,
        }
        data = gzip.compress(json.dumps(payload, separators=(",", ":")).encode("utf-8"), compresslevel=3)
        target = self._path(key, generation, "answer.gz")
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_bytes(data)
        os.replace(temporary, target)

    def fail(self, key: str, generation: int, reason: str) -> None:
        target = self._path(key, generation, "failed")
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(reason[:500], encoding="utf-8")
        os.replace(temporary, target)

    def read(self, key: str, generation: int) -> SharedAnswer | None:
        try:
            payload = json.loads(gzip.decompress(self._path(key, generation, "answer.gz").read_bytes()))
            return SharedAnswer(
                submission=base64.b64decode(payload["submission_b64"]),
                events=payload["events"],
                model_name=str(payload["model_name"]),
                hotkey=str(payload["hotkey"]),
                fields=dict(payload.get("fields") or {}),
            )
        except (OSError, ValueError, KeyError, TypeError, EOFError):
            return None

    async def wait(self, key: str, deadline: float, *, monotonic: Any = time.monotonic) -> tuple[str, SharedAnswer | None]:
        """Wait for the answer of whoever holds the claim: ("answered", answer), ("failed" | "dead", None) or
        ("timeout", None) at `deadline` (monotonic seconds)."""
        while True:
            generation = self.current(key)
            state = self.state(key, generation)
            if state == "answered":
                answer = await asyncio.to_thread(self.read, key, generation)
                if answer is not None:
                    return "answered", answer
                state = "failed"  # unreadable: as good as no answer
            if state in ("failed", "dead", "open"):
                return ("failed" if state == "open" else state), None
            if monotonic() >= deadline:
                return "timeout", None
            await asyncio.sleep(POLL_S)
