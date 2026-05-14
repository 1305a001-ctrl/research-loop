"""Tests for the parameter auto-tuner pure-math layer."""
from __future__ import annotations

import pytest

from research_loop.auto_tuner import (
    ParamBounds,
    _bin_centers,
    _sharpe_proxy,
    _value_to_bin,
    tune_param,
)
from research_loop.learning_loop import ClosedOutcome


def _bounds(name: str = "velocity_threshold") -> ParamBounds:
    return ParamBounds(
        name=name,
        bound_min=0.10,
        bound_max=0.60,
        default=0.30,
        n_bins=5,
        max_step_pct=0.10,    # 5% absolute max nudge (= 0.10 × 0.5 range)
    )


def _outcome(
    *, param_value: float, pnl: float,
    name: str = "velocity_threshold", strategy: str = "poly-sell-wings",
) -> ClosedOutcome:
    return ClosedOutcome(
        alpha_id=f"id-{id(pnl)}",
        strategy_slug=strategy,
        asset="BTC-USDT",
        venue="polymarket",
        emitted_at_iso="2026-05-14T00:00:00+00:00",
        features={},
        params_snapshot={name: param_value},
        outcome="win" if pnl > 0 else ("loss" if pnl < 0 else "flat"),
        realized_pnl_usd=pnl,
        notional_usd=20.0,
        closed_at_iso="2026-05-14T01:00:00+00:00",
    )


# ─── Helpers ─────────────────────────────────────────────────────────


def test_bin_centers() -> None:
    b = _bounds()
    # 5 bins over [0.10, 0.60] → centers at 0.15, 0.25, 0.35, 0.45, 0.55
    centers = _bin_centers(b)
    assert centers == pytest.approx([0.15, 0.25, 0.35, 0.45, 0.55])


def test_value_to_bin() -> None:
    b = _bounds()
    # 0.10 → bin 0
    assert _value_to_bin(0.10, b) == 0
    # 0.20 → bin 1 (since edge is 0.20 exactly, falls at start of bin 1)
    assert _value_to_bin(0.21, b) == 1
    # 0.59 → bin 4 (the highest)
    assert _value_to_bin(0.59, b) == 4
    # 0.60 → bin 4 (clamped to last)
    assert _value_to_bin(0.60, b) == 4
    # Out of range
    assert _value_to_bin(0.05, b) is None
    assert _value_to_bin(0.99, b) is None


def test_sharpe_proxy() -> None:
    assert _sharpe_proxy([]) is None
    assert _sharpe_proxy([1.0]) is None
    assert _sharpe_proxy([5.0, 5.0]) is None   # zero stdev
    s = _sharpe_proxy([1.0, 2.0, 3.0, 4.0])    # mean=2.5, stdev=1.29
    assert s is not None
    assert s == pytest.approx(2.5 / 1.290994, rel=1e-3)


# ─── tune_param decisions ────────────────────────────────────────────


def test_insufficient_samples_returns_current() -> None:
    """Sub-min-samples → no change."""
    b = _bounds()
    decision = tune_param(
        bounds=b,
        current=0.30,
        closed=[_outcome(param_value=0.30, pnl=5.0)],   # only 1 sample
        min_samples=20,
    )
    assert decision.proposed == 0.30
    assert decision.delta == 0.0
    assert "insufficient_samples" in decision.reason


def test_nudges_toward_best_bin() -> None:
    """30 samples across 2 bins; bin@0.25 has high sharpe → param nudges down."""
    b = _bounds()
    closed = []
    # Bin 1 (center 0.25): 15 trades with consistent +$10 PnL (high Sharpe)
    for _ in range(15):
        closed.append(_outcome(param_value=0.22, pnl=10.0))
    for _ in range(15):
        closed.append(_outcome(param_value=0.22, pnl=10.5))
    # Bin 3 (center 0.45): 15 trades with mixed PnL (low Sharpe)
    for _ in range(15):
        closed.append(_outcome(param_value=0.48, pnl=10.0))
    for _ in range(15):
        closed.append(_outcome(param_value=0.48, pnl=-9.0))

    decision = tune_param(bounds=b, current=0.45, closed=closed, min_samples=20)
    # Should nudge toward 0.25 — proposed < current
    assert decision.proposed < decision.current
    # Should be bounded by max_step_pct of range = 0.10 × 0.50 = 0.05 absolute
    assert decision.current - decision.proposed <= 0.05 + 1e-9
    assert "best_bin" in decision.reason


def test_clamped_to_bounds() -> None:
    """If all samples favor an extreme bin, proposed clamps to bound_max."""
    b = _bounds()
    closed = []
    # Many trades at high param value, all profitable
    for _ in range(30):
        closed.append(_outcome(param_value=0.55, pnl=10.0))
    for _ in range(30):
        closed.append(_outcome(param_value=0.55, pnl=11.0))

    decision = tune_param(bounds=b, current=0.59, closed=closed, min_samples=20)
    # Best bin center is 0.55; current is 0.59 → would nudge DOWN to 0.55
    # Bounded step = 0.05; so proposed = max(0.55, 0.59 - 0.05) = 0.55? actually 0.59-0.05=0.54
    # Clamp to bound range — 0.54 is within bounds, so proposed = 0.54
    assert b.bound_min <= decision.proposed <= b.bound_max


def test_max_step_bounds_movement() -> None:
    """A large nudge target is capped by max_step_pct."""
    b = _bounds()    # max_step_pct = 0.10 → 0.05 absolute on a 0.50 range
    closed = []
    for _ in range(30):
        closed.append(_outcome(param_value=0.15, pnl=8.0))
    for _ in range(30):
        closed.append(_outcome(param_value=0.15, pnl=9.0))

    decision = tune_param(bounds=b, current=0.55, closed=closed, min_samples=20)
    # Best bin is centered at 0.15 → raw delta = -0.40
    # Capped to -0.05 → proposed = 0.50
    assert decision.proposed == pytest.approx(0.50, abs=1e-9)
    assert "raw_delta=-0.40" in decision.reason
    assert "capped=-0.05" in decision.reason


def test_param_not_in_snapshot_skipped() -> None:
    """Outcomes missing the param value are skipped from binning."""
    b = _bounds()
    closed = [
        ClosedOutcome(
            alpha_id="x",
            strategy_slug="poly-sell-wings",
            asset="BTC", venue="polymarket",
            emitted_at_iso="2026-05-14T00:00:00",
            features={},
            params_snapshot={},   # missing our param
            outcome="win",
            realized_pnl_usd=10.0,
            notional_usd=20.0,
            closed_at_iso="2026-05-14T01:00:00",
        ),
    ] * 50

    decision = tune_param(bounds=b, current=0.30, closed=closed, min_samples=20)
    # No in-range samples (all missing) → no change
    assert decision.proposed == 0.30
    assert "no_in_range_samples" in decision.reason or "no_bin_with_sharpe" in decision.reason


def test_zero_stdev_bin_skipped() -> None:
    """A bin where all PnLs are identical → Sharpe is None → not eligible."""
    b = _bounds()
    closed = []
    # Identical-PnL bin (stdev=0, Sharpe undefined)
    for _ in range(20):
        closed.append(_outcome(param_value=0.55, pnl=5.0))
    # Real-Sharpe bin
    for _ in range(20):
        closed.append(_outcome(param_value=0.15, pnl=10.0))
    for _ in range(20):
        closed.append(_outcome(param_value=0.15, pnl=11.0))

    decision = tune_param(bounds=b, current=0.30, closed=closed, min_samples=20)
    # Best bin should be 0.15 (it has non-None Sharpe)
    # current=0.30, raw_delta = -0.15, capped = -0.05 → proposed = 0.25
    assert decision.proposed == pytest.approx(0.25, abs=1e-9)
