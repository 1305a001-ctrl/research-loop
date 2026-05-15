"""Tests for the halt-fast venue-wide circuit breaker."""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from research_loop import halt_fast, halts
from research_loop.halt_fast import (
    HaltFastDecision,
    decide_halt_fast,
    fetch_venue_pnl_24h,
    run_halt_fast_cycle,
)

# ─── Pure: decide_halt_fast ────────────────────────────────────────


def test_decide_halt_fast_trips_when_pnl_below_floor() -> None:
    d = decide_halt_fast(
        venue="polymarket",
        realized_pnl_24h_usd=-100.0,
        n_closed_24h=10,
        threshold_usd=-50.0,
    )
    assert d.trip is True
    assert d.venue == "polymarket"
    assert "halt_fast_24h_pnl_floor_breach" in d.reason


def test_decide_halt_fast_no_trip_when_pnl_above_floor() -> None:
    d = decide_halt_fast(
        venue="polymarket",
        realized_pnl_24h_usd=-30.0,   # better than -50
        n_closed_24h=10,
        threshold_usd=-50.0,
    )
    assert d.trip is False
    assert "pnl_above_floor" in d.reason


def test_decide_halt_fast_no_trip_when_pnl_equal_to_floor() -> None:
    """Exactly at the floor is NOT a trip — strict less-than."""
    d = decide_halt_fast(
        venue="polymarket",
        realized_pnl_24h_usd=-50.0,
        n_closed_24h=10,
        threshold_usd=-50.0,
    )
    assert d.trip is False


def test_decide_halt_fast_no_trip_when_insufficient_closes() -> None:
    """Below the floor but not enough closed positions — still no trip."""
    d = decide_halt_fast(
        venue="polymarket",
        realized_pnl_24h_usd=-100.0,
        n_closed_24h=0,
        threshold_usd=-50.0,
        min_n_closed=1,
    )
    assert d.trip is False
    assert "insufficient_closed_count" in d.reason


def test_decide_halt_fast_min_n_closed_configurable() -> None:
    """Bump min_n_closed to 5 to suppress single-trade spurious trips."""
    # Below floor + only 3 closes — does NOT trip with min=5
    d = decide_halt_fast(
        venue="polymarket",
        realized_pnl_24h_usd=-100.0,
        n_closed_24h=3,
        threshold_usd=-50.0,
        min_n_closed=5,
    )
    assert d.trip is False
    # Same PnL + 5 closes → trips
    d2 = decide_halt_fast(
        venue="polymarket",
        realized_pnl_24h_usd=-100.0,
        n_closed_24h=5,
        threshold_usd=-50.0,
        min_n_closed=5,
    )
    assert d2.trip is True


def test_decide_halt_fast_positive_pnl_never_trips() -> None:
    d = decide_halt_fast(
        venue="polymarket",
        realized_pnl_24h_usd=200.0,
        n_closed_24h=10,
        threshold_usd=-50.0,
    )
    assert d.trip is False


def test_halt_fast_decision_is_frozen_dataclass() -> None:
    """frozen=True dataclasses raise FrozenInstanceError on assignment."""
    d = HaltFastDecision(
        venue="x", realized_pnl_24h_usd=0.0, threshold_usd=0.0,
        trip=False, n_closed_24h=0, reason="x",
    )
    from dataclasses import FrozenInstanceError
    with pytest.raises(FrozenInstanceError):
        d.trip = True   # type: ignore[misc]


# ─── DB read (mocked pool) ─────────────────────────────────────────


class _FakePool:
    def __init__(self, row: dict | None = None, raise_exc: Exception | None = None):
        self.row = row
        self.raise_exc = raise_exc
        self.queries: list[tuple] = []

    async def fetchrow(self, sql: str, *params):
        self.queries.append((sql, params))
        if self.raise_exc:
            raise self.raise_exc
        return self.row


@pytest.mark.asyncio
async def test_fetch_venue_pnl_24h_returns_row_values() -> None:
    pool = _FakePool(row={
        "realized_pnl_24h_usd": -123.45,
        "n_closed": 7,
        "slugs": ["poly-sell-wings", "poly-cross-market-arb"],
    })
    pnl, n, slugs = await fetch_venue_pnl_24h(pool, "polymarket")
    assert pnl == -123.45
    assert n == 7
    assert sorted(slugs) == ["poly-cross-market-arb", "poly-sell-wings"]


@pytest.mark.asyncio
async def test_fetch_venue_pnl_24h_returns_zero_on_empty() -> None:
    pool = _FakePool(row=None)
    pnl, n, slugs = await fetch_venue_pnl_24h(pool, "polymarket")
    assert pnl == 0.0
    assert n == 0
    assert slugs == []


@pytest.mark.asyncio
async def test_fetch_venue_pnl_24h_swallows_db_error() -> None:
    """DB blip → return zeros so the cycle no-ops rather than crashing."""
    pool = _FakePool(raise_exc=RuntimeError("connection refused"))
    pnl, n, slugs = await fetch_venue_pnl_24h(pool, "polymarket")
    assert (pnl, n, slugs) == (0.0, 0, [])


@pytest.mark.asyncio
async def test_fetch_venue_pnl_24h_filters_none_slugs() -> None:
    """If the array_agg returns nulls, they must be filtered out."""
    pool = _FakePool(row={
        "realized_pnl_24h_usd": -10.0, "n_closed": 1, "slugs": [None, "valid", None],
    })
    _, _, slugs = await fetch_venue_pnl_24h(pool, "polymarket")
    assert slugs == ["valid"]


@pytest.mark.asyncio
async def test_fetch_venue_pnl_24h_uses_24h_cutoff() -> None:
    """Verify the cutoff parameter is now - 24h."""
    pool = _FakePool(row=None)
    fixed_now = datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)
    await fetch_venue_pnl_24h(pool, "polymarket", now=fixed_now)
    sql, params = pool.queries[0]
    cutoff = params[0]
    # cutoff should be exactly 24h before fixed_now
    delta = fixed_now - cutoff
    assert delta.total_seconds() == 24 * 3600
    assert params[1] == "polymarket"


# ─── Orchestrator (with mock halts) ────────────────────────────────


class _StubHalts:
    """Replaces halts.actuate to record calls instead of touching Redis."""
    def __init__(self, already_halted: set | None = None) -> None:
        self.calls: list[tuple[str, str]] = []
        self.already = already_halted or set()

    async def actuate(self, slug: str, reason: str) -> bool:
        self.calls.append((slug, reason))
        return slug not in self.already


@pytest.mark.asyncio
async def test_run_cycle_no_trip_when_pnl_above_floor(monkeypatch) -> None:
    pool = _FakePool(row={
        "realized_pnl_24h_usd": -10.0, "n_closed": 5,
        "slugs": ["poly-sell-wings"],
    })
    stub = _StubHalts()
    monkeypatch.setattr(halts, "actuate", stub.actuate)
    stats = await run_halt_fast_cycle(
        pool, venue="polymarket", threshold_usd=-50.0,
    )
    assert stats["trip"] is False
    assert stats["n_halted_now"] == 0
    assert stub.calls == []


@pytest.mark.asyncio
async def test_run_cycle_trips_and_halts_all_venue_slugs(monkeypatch) -> None:
    pool = _FakePool(row={
        "realized_pnl_24h_usd": -100.0, "n_closed": 6,
        "slugs": ["poly-sell-wings", "poly-cross-market-arb", "poly-settlement-momentum"],
    })
    stub = _StubHalts()
    monkeypatch.setattr(halts, "actuate", stub.actuate)
    stats = await run_halt_fast_cycle(
        pool, venue="polymarket", threshold_usd=-50.0,
    )
    assert stats["trip"] is True
    assert stats["n_halted_now"] == 3
    assert stats["n_already_halted"] == 0
    halted = {c[0] for c in stub.calls}
    assert halted == {
        "poly-sell-wings", "poly-cross-market-arb", "poly-settlement-momentum",
    }


@pytest.mark.asyncio
async def test_run_cycle_counts_already_halted_separately(monkeypatch) -> None:
    pool = _FakePool(row={
        "realized_pnl_24h_usd": -100.0, "n_closed": 6,
        "slugs": ["poly-sell-wings", "poly-cross-market-arb"],
    })
    stub = _StubHalts(already_halted={"poly-cross-market-arb"})
    monkeypatch.setattr(halts, "actuate", stub.actuate)
    stats = await run_halt_fast_cycle(
        pool, venue="polymarket", threshold_usd=-50.0,
    )
    assert stats["trip"] is True
    assert stats["n_halted_now"] == 1
    assert stats["n_already_halted"] == 1


@pytest.mark.asyncio
async def test_run_cycle_with_no_slugs(monkeypatch) -> None:
    """No closes in 24h → no halt attempts even if PnL would trip."""
    pool = _FakePool(row={
        "realized_pnl_24h_usd": 0.0, "n_closed": 0, "slugs": [],
    })
    stub = _StubHalts()
    monkeypatch.setattr(halts, "actuate", stub.actuate)
    stats = await run_halt_fast_cycle(
        pool, venue="polymarket", threshold_usd=-50.0,
    )
    assert stats["trip"] is False
    assert stub.calls == []


@pytest.mark.asyncio
async def test_loop_respects_stop_event(monkeypatch) -> None:
    """halt_fast_loop runs at least once, then exits when stop_event set."""
    pool = _FakePool(row={
        "realized_pnl_24h_usd": 0.0, "n_closed": 0, "slugs": [],
    })

    async def get_pool() -> object:
        return pool

    stub = _StubHalts()
    monkeypatch.setattr(halts, "actuate", stub.actuate)

    stop = asyncio.Event()

    async def kill_in(seconds: float) -> None:
        await asyncio.sleep(seconds)
        stop.set()

    # Loop with very short interval so the wait_for exits quickly.
    # Schedule a stop in 0.1s; the loop should have done >=1 iteration.
    await asyncio.gather(
        halt_fast.halt_fast_loop(
            stop, get_pool, venues=["polymarket"],
            threshold_usd=-50.0, interval_sec=1, min_n_closed=1,
        ),
        kill_in(0.15),
    )
