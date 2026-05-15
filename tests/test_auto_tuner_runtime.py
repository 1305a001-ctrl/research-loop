"""Tests for the auto-tuner runtime orchestrator (pure + async)."""
from __future__ import annotations

import pytest

from research_loop.auto_tuner_runtime import (
    TUNABLE_PARAMS,
    parse_opted_in_strategies,
    run_tune_cycle,
)
from research_loop.learning_loop import ClosedOutcome

# ─── parse_opted_in_strategies (pure) ──────────────────────────────


def test_parse_empty_returns_empty() -> None:
    assert parse_opted_in_strategies("") == []
    assert parse_opted_in_strategies("   ") == []


def test_parse_all_returns_all_keys() -> None:
    out = parse_opted_in_strategies("all")
    assert set(out) == set(TUNABLE_PARAMS.keys())


def test_parse_all_case_insensitive() -> None:
    assert set(parse_opted_in_strategies("ALL")) == set(TUNABLE_PARAMS.keys())
    assert set(parse_opted_in_strategies("All")) == set(TUNABLE_PARAMS.keys())


def test_parse_single_strategy() -> None:
    assert parse_opted_in_strategies("poly-sell-wings") == ["poly-sell-wings"]


def test_parse_csv() -> None:
    out = parse_opted_in_strategies("poly-sell-wings, poly-publisher-taker")
    assert sorted(out) == ["poly-publisher-taker", "poly-sell-wings"]


def test_parse_drops_unknown_strategies() -> None:
    """Unknown slugs are silently dropped (logged) — won't crash the loop."""
    out = parse_opted_in_strategies("poly-sell-wings,not-a-real-strategy,poly-publisher-taker")
    assert "not-a-real-strategy" not in out
    assert "poly-sell-wings" in out
    assert "poly-publisher-taker" in out


def test_parse_strips_whitespace() -> None:
    assert parse_opted_in_strategies("  poly-sell-wings  ,  poly-publisher-taker  ") == [
        "poly-sell-wings", "poly-publisher-taker",
    ]


def test_parse_empty_entries_skipped() -> None:
    assert parse_opted_in_strategies("poly-sell-wings,,,poly-publisher-taker") == [
        "poly-sell-wings", "poly-publisher-taker",
    ]


def test_tunable_params_registry_invariants() -> None:
    """Every registered strategy must have at least one ParamBounds, and
    each bounds entry must have min<max, default in range, bins>0."""
    assert len(TUNABLE_PARAMS) >= 4
    for slug, bounds_list in TUNABLE_PARAMS.items():
        assert bounds_list, f"{slug} has no bounds"
        for b in bounds_list:
            assert b.bound_min < b.bound_max, f"{slug}.{b.name} bounds inverted"
            assert b.bound_min <= b.default <= b.bound_max, \
                f"{slug}.{b.name} default {b.default} outside [{b.bound_min}, {b.bound_max}]"
            assert b.n_bins > 0
            assert 0 < b.max_step_pct <= 0.5, \
                f"{slug}.{b.name} max_step_pct {b.max_step_pct} unreasonable"


# ─── run_tune_cycle (orchestrator) ─────────────────────────────────


class _FakePool:
    """Stub asyncpg pool that returns predetermined ClosedOutcomes."""

    def __init__(self, closed: dict[str, list[ClosedOutcome]] | None = None,
                 raise_for: set[str] | None = None) -> None:
        self.closed = closed or {}
        self.raise_for = raise_for or set()


class _FakeRedis:
    """Stub Redis: stores values in a dict; records all SET calls."""

    def __init__(self, current: dict[tuple[str, str], float] | None = None) -> None:
        self.current = current or {}
        self.sets: list[tuple[str, str, float, str]] = []
        self.fail_set: bool = False

    async def hget(self, key: str, name: str):
        # key format: params:<slug>
        slug = key.split(":", 1)[1] if ":" in key else key
        v = self.current.get((slug, name))
        return None if v is None else str(v)

    async def hset(self, key: str, name: str, value: str):
        if self.fail_set:
            raise RuntimeError("redis down")
        slug = key.split(":", 1)[1] if ":" in key else key
        self.current[(slug, name)] = float(value)
        return 1

    async def xadd(self, *args, **kwargs):
        return "stream-id"


def _outcome(*, pnl: float, params: dict[str, float], asset: str = "x") -> ClosedOutcome:
    return ClosedOutcome(
        alpha_id="00000000-0000-0000-0000-000000000000",
        strategy_slug="x",
        asset=asset,
        venue="polymarket",
        emitted_at_iso="2026-05-15T00:00:00Z",
        features={},
        params_snapshot=params,
        outcome="win" if pnl > 0 else "loss",
        realized_pnl_usd=pnl,
        notional_usd=100.0,
        closed_at_iso="2026-05-15T01:00:00Z",
    )


async def _fake_fetch(pool, *, strategy_slug, limit, days):
    """Monkeypatch target for learning_loop.fetch_recent_closed."""
    if strategy_slug in pool.raise_for:
        raise RuntimeError("simulated db error")
    return pool.closed.get(strategy_slug, [])


@pytest.mark.asyncio
async def test_run_cycle_no_strategies_returns_empty() -> None:
    """No opted-in strategies → empty stats."""
    pool = _FakePool()
    redis = _FakeRedis()
    out = await run_tune_cycle(pool, redis, strategies=[], min_samples=30)
    assert out == {}


@pytest.mark.asyncio
async def test_run_cycle_insufficient_samples_no_nudge(monkeypatch) -> None:
    """< min_samples → no nudge, but stats reflect the fetch."""
    import research_loop.learning_loop as ll
    monkeypatch.setattr(ll, "fetch_recent_closed", _fake_fetch)

    closed = [_outcome(pnl=1.0, params={"yes_upper": 0.90}) for _ in range(5)]
    pool = _FakePool(closed={"poly-sell-wings": closed})
    redis = _FakeRedis(current={("poly-sell-wings", "yes_upper"): 0.90})

    out = await run_tune_cycle(
        pool, redis, strategies=["poly-sell-wings"], min_samples=30,
    )
    assert out["poly-sell-wings"]["n_closed"] == 5
    assert out["poly-sell-wings"]["n_nudged"] == 0
    assert redis.sets == []


@pytest.mark.asyncio
async def test_run_cycle_nudges_when_data_supports_it(monkeypatch) -> None:
    """When one bin has clearly better Sharpe, nudge toward its center."""
    import research_loop.learning_loop as ll
    monkeypatch.setattr(ll, "fetch_recent_closed", _fake_fetch)

    # Construct outcomes where high-yes_upper values had positive PnL +
    # low-yes_upper had negative — should nudge yes_upper UP from 0.85
    closed = []
    # 20 winners at yes_upper=0.93 (high bin, mean=$2, stdev~$1)
    for i in range(20):
        closed.append(_outcome(pnl=2.0 + (i % 3) * 0.5, params={"yes_upper": 0.93}))
    # 20 losers at yes_upper=0.86 (low bin, mean=-$1, stdev~$0.5)
    for i in range(20):
        closed.append(_outcome(pnl=-1.0 - (i % 3) * 0.2, params={"yes_upper": 0.86}))

    pool = _FakePool(closed={"poly-sell-wings": closed})
    redis = _FakeRedis(current={
        ("poly-sell-wings", "yes_upper"): 0.85,
        ("poly-sell-wings", "yes_lower"): 0.10,
        ("poly-sell-wings", "min_volume"): 5000.0,
    })

    out = await run_tune_cycle(
        pool, redis, strategies=["poly-sell-wings"], min_samples=30,
        dry_run=False,
    )

    nudges = out["poly-sell-wings"]["per_param"]
    # yes_upper should have moved upward (toward 0.93 bin)
    assert nudges["yes_upper"]["delta"] > 0
    assert nudges["yes_upper"]["proposed"] > 0.85
    # And the registry should have the new value
    new_val = redis.current[("poly-sell-wings", "yes_upper")]
    assert new_val > 0.85


@pytest.mark.asyncio
async def test_run_cycle_dry_run_does_not_write(monkeypatch) -> None:
    """dry_run=True: decisions computed, but no Redis writes."""
    import research_loop.learning_loop as ll
    monkeypatch.setattr(ll, "fetch_recent_closed", _fake_fetch)

    # Need variance in PnL within each bin or sharpe_proxy returns None
    closed = []
    for i in range(20):
        closed.append(_outcome(pnl=5.0 + (i % 3) * 0.5, params={"yes_upper": 0.93}))
    for i in range(20):
        closed.append(_outcome(pnl=-2.0 - (i % 3) * 0.3, params={"yes_upper": 0.86}))

    pool = _FakePool(closed={"poly-sell-wings": closed})
    redis = _FakeRedis(current={("poly-sell-wings", "yes_upper"): 0.85})

    out = await run_tune_cycle(
        pool, redis, strategies=["poly-sell-wings"], min_samples=30,
        dry_run=True,
    )
    # Stats show what would happen
    assert out["poly-sell-wings"]["per_param"]["yes_upper"]["delta"] > 0
    # But Redis stays at 0.85
    assert redis.current[("poly-sell-wings", "yes_upper")] == 0.85


@pytest.mark.asyncio
async def test_run_cycle_swallows_fetch_errors(monkeypatch) -> None:
    """DB blip on one strategy: skip that strategy, continue with others."""
    import research_loop.learning_loop as ll
    monkeypatch.setattr(ll, "fetch_recent_closed", _fake_fetch)

    pool = _FakePool(
        closed={"poly-publisher-taker": []},
        raise_for={"poly-sell-wings"},
    )
    redis = _FakeRedis()

    out = await run_tune_cycle(
        pool, redis,
        strategies=["poly-sell-wings", "poly-publisher-taker"],
        min_samples=30,
    )
    assert out["poly-sell-wings"]["error"] == "fetch_closed_failed"
    assert "error" not in out["poly-publisher-taker"]


@pytest.mark.asyncio
async def test_run_cycle_skips_unknown_strategy(monkeypatch) -> None:
    """If a strategy is in the list but not in TUNABLE_PARAMS, skip silently
    (parse_opted_in_strategies should have filtered it, but we're defensive)."""
    import research_loop.learning_loop as ll
    monkeypatch.setattr(ll, "fetch_recent_closed", _fake_fetch)

    pool = _FakePool()
    redis = _FakeRedis()
    out = await run_tune_cycle(
        pool, redis,
        strategies=["not-a-real-strategy"], min_samples=30,
    )
    assert out == {}


@pytest.mark.asyncio
async def test_run_cycle_below_epsilon_no_write(monkeypatch) -> None:
    """A tiny delta (< NUDGE_EPSILON) is skipped to keep audit log clean."""
    import research_loop.learning_loop as ll
    monkeypatch.setattr(ll, "fetch_recent_closed", _fake_fetch)

    # All in the same bin → tune_param returns proposed=current effectively
    closed = [_outcome(pnl=1.0, params={"yes_upper": 0.90}) for _ in range(50)]
    pool = _FakePool(closed={"poly-sell-wings": closed})
    redis = _FakeRedis(current={("poly-sell-wings", "yes_upper"): 0.90})

    out = await run_tune_cycle(
        pool, redis, strategies=["poly-sell-wings"], min_samples=30,
    )
    # n_nudged is 0 because delta is at-or-below epsilon
    assert out["poly-sell-wings"]["n_nudged"] == 0


@pytest.mark.asyncio
async def test_run_cycle_redis_error_falls_back_to_default(monkeypatch) -> None:
    """If Redis hget fails, param_registry.get_param defensively returns the
    default. The bound is still processed, just starting from the default."""
    import research_loop.learning_loop as ll
    monkeypatch.setattr(ll, "fetch_recent_closed", _fake_fetch)

    class _BadRedis(_FakeRedis):
        async def hget(self, key: str, name: str):
            if name == "yes_upper":
                raise RuntimeError("redis flake on this key")
            return await super().hget(key, name)

    closed = [_outcome(pnl=1.0, params={"yes_lower": 0.10}) for _ in range(50)]
    pool = _FakePool(closed={"poly-sell-wings": closed})
    redis = _BadRedis(current={("poly-sell-wings", "yes_lower"): 0.10})

    out = await run_tune_cycle(
        pool, redis, strategies=["poly-sell-wings"], min_samples=30,
    )
    # yes_upper IS in per_param — fallback to default kicks in
    assert "yes_upper" in out["poly-sell-wings"]["per_param"]
    # And current matches the bounds default (0.90), not the missing redis value
    assert out["poly-sell-wings"]["per_param"]["yes_upper"]["current"] == 0.90
    # yes_lower also processed normally
    assert "yes_lower" in out["poly-sell-wings"]["per_param"]
