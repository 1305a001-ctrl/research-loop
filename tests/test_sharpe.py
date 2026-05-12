"""Tests for sharpe + auto-halt math (pure, no I/O)."""
from __future__ import annotations

import pytest

from research_loop.sharpe import (
    HaltDecision,
    StrategyStats,
    health_badge,
    sharpe_proxy,
    should_halt,
)

# ─── sharpe_proxy ──────────────────────────────────────────────────────────


def test_sharpe_none_for_empty():
    assert sharpe_proxy([]) is None


def test_sharpe_none_for_one_sample():
    assert sharpe_proxy([5.0]) is None


def test_sharpe_none_when_all_identical():
    """Zero stddev → undefined Sharpe → None."""
    assert sharpe_proxy([3.0, 3.0, 3.0, 3.0]) is None


def test_sharpe_positive_for_winning_strategy():
    # Mostly positive returns
    s = sharpe_proxy([2.0, 3.0, 1.5, 2.5, 1.0, 4.0])
    assert s is not None
    assert s > 0


def test_sharpe_negative_for_losing_strategy():
    s = sharpe_proxy([-2.0, -3.0, -1.5, -2.5, -1.0, -4.0])
    assert s is not None
    assert s < 0


def test_sharpe_near_zero_for_random():
    """Mean ≈ 0 → Sharpe ≈ 0."""
    s = sharpe_proxy([1, -1, 2, -2, 1, -1])
    assert s is not None
    assert abs(s) < 0.5


# ─── should_halt ───────────────────────────────────────────────────────────


def _stats(*, n=50, pnl=10.0, sharpe=0.5, days=14) -> StrategyStats:
    return StrategyStats(
        slug="test", n_closed=n, total_pnl_usd=pnl, sharpe=sharpe, days_observed=days,
    )


def test_catastrophic_loss_halts_even_with_small_sample():
    """Total PnL ≤ catastrophic floor → halt regardless of n_closed."""
    s = _stats(n=3, pnl=-500.0, sharpe=None, days=1)
    d = should_halt(s)
    assert d.halt
    assert "catastrophic" in d.reason
    assert d.severity == "critical"


def test_insufficient_sample_does_not_halt():
    """n_closed < min_n_closed → don't halt (need more data)."""
    s = _stats(n=10, pnl=-50.0, sharpe=-0.5, days=20)
    d = should_halt(s, min_n_closed=30)
    assert not d.halt
    assert "insufficient_sample" in d.reason


def test_insufficient_days_does_not_halt():
    """days_observed < min_days → don't halt even with bad sharpe."""
    s = _stats(n=50, pnl=-50.0, sharpe=-0.5, days=3)
    d = should_halt(s, min_days_observed=7)
    assert not d.halt
    assert "insufficient_days" in d.reason


def test_bad_sharpe_halts_when_sample_sufficient():
    s = _stats(n=50, pnl=-50.0, sharpe=-0.3, days=14)
    d = should_halt(s)
    assert d.halt
    assert "sharpe_below_threshold" in d.reason
    assert d.severity == "warn"


def test_healthy_strategy_not_halted():
    s = _stats(n=50, pnl=100.0, sharpe=0.5, days=14)
    d = should_halt(s)
    assert not d.halt
    assert "healthy" in d.reason


def test_sharpe_unavailable_does_not_halt():
    s = _stats(n=50, pnl=10.0, sharpe=None, days=14)
    d = should_halt(s)
    assert not d.halt
    assert "sharpe_unavailable" in d.reason


def test_custom_threshold_more_strict():
    """Higher sharpe threshold halts borderline performers."""
    s = _stats(n=50, pnl=10.0, sharpe=0.2, days=14)
    # default threshold (0.0) — passes
    assert not should_halt(s).halt
    # stricter threshold (0.3) — halts
    assert should_halt(s, sharpe_threshold=0.3).halt


def test_catastrophic_floor_threshold():
    """Custom catastrophic floor."""
    s = _stats(n=3, pnl=-100.0, sharpe=None, days=1)
    # default floor ($-200) — not catastrophic yet
    assert not should_halt(s).halt
    # stricter floor ($-50) — halts
    assert should_halt(s, catastrophic_pnl_floor_usd=-50.0).halt


# ─── health_badge ─────────────────────────────────────────────────────────


def test_badge_small_n():
    s = _stats(n=3, pnl=0, sharpe=0, days=1)
    assert health_badge(s) == "small-N"


def test_badge_healthy():
    s = _stats(n=50, pnl=100.0, sharpe=0.5, days=14)
    assert health_badge(s) == "healthy"


def test_badge_mixed():
    s = _stats(n=50, pnl=10.0, sharpe=0.1, days=14)   # positive but low sharpe
    assert health_badge(s) == "mixed"


def test_badge_bleed():
    s = _stats(n=50, pnl=-50.0, sharpe=-0.5, days=14)
    assert health_badge(s) == "bleed"


def test_decision_dataclass_immutable():
    """HaltDecision is frozen — guards against mutation in caller."""
    d = HaltDecision(halt=True, reason="test", severity="warn")
    with pytest.raises(Exception):  # noqa: B017, BLE001 — any exception is fine here
        d.halt = False   # type: ignore[misc]
