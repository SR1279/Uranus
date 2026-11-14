"""The account's OpenRouter credit: read, logged, and said at the top of LOG.md."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from rlvr.neurons import demo_miner
from rlvr.neurons.demo_miner import DemoMiner, OpenRouterClient, credit_hours_left
from rlvr.neurons.journal import SolveJournal, render_log
from tests.test_demo_miner import WalletLike, settings


def client_answering(status, body):
    def handler(request):
        assert request.url.path.endswith("/credits") and request.headers["authorization"] == "Bearer key"
        return httpx.Response(status, json=body)

    return OpenRouterClient(settings(), http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


async def test_the_credit_left_is_read_from_the_account():
    assert await client_answering(200, {"data": {"total_credits": 910, "total_usage": 897.4}}).remaining_credit() == pytest.approx(12.6)
    assert await client_answering(401, {"error": {"message": "no"}}).remaining_credit() is None
    assert await client_answering(200, {"data": {}}).remaining_credit() is None


def test_hours_left_come_from_the_rate_over_the_window():
    assert credit_hours_left([(0.0, 20.0)]) is None
    assert credit_hours_left([(0.0, 20.0), (3600.0, 15.0)]) == 3.0  # $5 an hour, $15 left
    assert credit_hours_left([(0.0, 10.0), (3600.0, 50.0)]) is None  # topped up
    assert credit_hours_left([(0.0, 10.0), (1800.0, 10.0)]) is None  # nothing spent


class CreditClient:
    def __init__(self, *values):
        self.values = list(values)

    async def remaining_credit(self):
        return self.values.pop(0)

    async def aclose(self):
        pass


async def test_the_watch_logs_the_credit_and_keeps_it_for_the_log(tmp_path, monkeypatch, capsys):
    sleeps = []

    async def sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 3:
            raise asyncio.CancelledError

    clock = iter([0.0, 3600.0, 7200.0])
    monkeypatch.setattr(demo_miner.asyncio, "sleep", sleep)
    miner = DemoMiner(settings(miner_record_dir=str(tmp_path)), CreditClient(40.0, 20.0, -0.5), wallet=WalletLike("miner"))
    with pytest.raises(asyncio.CancelledError):
        await miner.watch_credit(clock=lambda: next(clock))
    out = capsys.readouterr().out
    assert "OpenRouter credit: $40.00 left" in out
    assert "WARNING: OpenRouter credit is low: $20.00 left, about 1 hours at the recent pace" in out
    assert "WARNING: OpenRouter credit is used up" in out
    state = json.loads((tmp_path / "credit.json").read_text())
    assert state["remaining_usd"] == -0.5 and sleeps == [demo_miner.CREDIT_CHECK_S] * 3


def test_log_md_says_the_credit_first(tmp_path):
    journal = SolveJournal(tmp_path)
    journal.credit({"checked": "2026-11-11T06:00:00Z", "remaining_usd": 12.6, "hours_left": 4.2})
    text = render_log(tmp_path)
    assert "**OpenRouter credit is low: $12.60 left** (read 2026-11-11T06:00:00Z), about 4.2 hours at the recent pace" in text
    journal.credit({"checked": "2026-11-11T09:00:00Z", "remaining_usd": 0.0, "hours_left": None})
    assert "**OpenRouter credit is used up**" in render_log(tmp_path)
    journal.credit({"checked": "2026-11-11T10:00:00Z", "remaining_usd": 300.0, "hours_left": None})
    assert "OpenRouter credit: $300.00 left (read 2026-11-11T10:00:00Z)." in render_log(tmp_path)


def test_the_audit_is_off_by_default():
    # 10 of 13 live audits found nothing and one finding re-broke a test, at an eighth of a solve's cost.
    assert demo_miner.DemoMinerSettings(_env_file=None, openrouter_api_key="key").miner_audit is False


async def test_low_credit_leaves_out_the_optional_calls(monkeypatch):
    from rlvr.neurons import adjudicate
    from rlvr.neurons.demo_miner import SolveNotes
    from tests.test_acceptance import documented_task
    from tests.test_adjudicate import LISTING, RULINGS
    from tests.test_demo_miner import completion_for

    monkeypatch.setattr(adjudicate, "VOTE_STAGGER_S", 0.0)
    calls = []

    class Provider:
        async def complete(self, messages, *, timeout_s, tools=(), **options):
            calls.append(options.get("model"))
            return completion_for(RULINGS)

    miner = DemoMiner(settings(miner_adjudicator_votes=3), Provider(), wallet=WalletLike("miner"))
    assert not miner.economy()  # no reading yet: never assume the credit is low
    miner.credit_left = 40.0
    assert not miner.economy()
    miner.credit_left = 9.5
    assert miner.economy()
    notes = SolveNotes()
    await miner._adjudicate(documented_task(), {"d.md": "rules"}, LISTING.output, [], notes.spend, notes, 1e12)
    assert len(calls) == 1 and notes.adjudication == "2 readings, 1 overruled"

    class Bench:
        terminal, can_run = False, True

    assert miner._start_author(documented_task(), Bench(), 1e12, notes, {"d.md": "rules"}) is None
    assert notes.acceptance == "skipped: low credit"
    off = DemoMiner(settings(miner_economy_below_usd=0), Provider(), wallet=WalletLike("miner"))
    off.credit_left = 1.0
    assert not off.economy()


def test_the_readers_are_off_by_default():
    # 2026-11-12: replays showed the Opus readers' majority overruling Sonnet's correct reading in 2 of 14
    # failures (8e60bf78d322, 015ff5c79492) and rescuing none; see the comment on miner_adjudicator.
    from rlvr.neurons.demo_miner import DemoMinerSettings

    assert DemoMinerSettings(_env_file=None, openrouter_api_key="key").miner_adjudicator is False


def test_provider_routing_applies_only_to_other_models_and_only_when_set():
    from rlvr.neurons.demo_miner import DemoMinerSettings, _provider_routing

    plain = DemoMinerSettings(_env_file=None, openrouter_api_key="key")
    assert _provider_routing(plain, "moonshotai/kimi-k3.1-pro") is None
    pinned = DemoMinerSettings(
        _env_file=None, openrouter_api_key="key",
        openrouter_provider_order="Moonshot AI, Fireworks", openrouter_provider_quantizations="bf16,fp8,unknown",
    )
    assert _provider_routing(pinned, "moonshotai/kimi-k3.1-pro") == {
        "order": ["Moonshot AI", "Fireworks"], "quantizations": ["bf16", "fp8", "unknown"], "allow_fallbacks": True,
    }
    assert _provider_routing(pinned, "anthropic/claude-sonnet-5.5") is None
