"""Parameter auto-tuner — Bayesian-style nudges with bounds.

For each strategy + param, the tuner:
  1. Reads recent ClosedOutcome rows (last 30-100 closes)
  2. Bins by the param's value-at-emit (from params_snapshot)
  3. Computes Sharpe per bin
  4. Nudges the param toward the best-Sharpe bin, bounded by max_step
  5. Writes the new value to the param registry (Redis hash)

Safety:
  - Each param has explicit min/max bounds; values clamped
  - Max step per cycle = max_step_pct of (bound_max - bound_min); default 10%
  - Hard prior: if sample size < min_samples, return current value unchanged
  - Per-strategy opt-in via setting

PURE math + IO-free decision logic. Caller is responsible for reading
ClosedOutcome rows and writing nudged params.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from .learning_loop import ClosedOutcome


@dataclass(frozen=True)
class ParamBounds:
    """The allowable range + tuning sensitivity for one named param."""
    name: str                          # e.g. "velocity_threshold"
    bound_min: float
    bound_max: float
    default: float
    n_bins: int = 5                    # how many bins for value attribution
    max_step_pct: float = 0.10         # nudge ≤ 10% of (max-min) per cycle


@dataclass(frozen=True)
class TuneDecision:
    """Result of one tune call. Always returns a value within bounds."""
    name: str
    current: float
    proposed: float
    delta: float
    reason: str
    n_samples: int
    by_bin: dict[float, float] = None  # bin → sharpe; populated when we have data


def _sharpe_proxy(pnls: list[float]) -> float | None:
    """Pure: mean(pnl) / stddev(pnl). None for <2 samples or zero stddev."""
    if len(pnls) < 2:
        return None
    avg = statistics.mean(pnls)
    stdev = statistics.stdev(pnls)
    if stdev <= 0:
        return None
    return avg / stdev


def _bin_edges(bounds: ParamBounds) -> list[float]:
    """Pure: n_bins+1 edge points evenly spaced over [bound_min, bound_max]."""
    step = (bounds.bound_max - bounds.bound_min) / bounds.n_bins
    return [bounds.bound_min + step * i for i in range(bounds.n_bins + 1)]


def _bin_centers(bounds: ParamBounds) -> list[float]:
    """Pure: bin centers (midpoints between edges)."""
    edges = _bin_edges(bounds)
    return [(edges[i] + edges[i + 1]) / 2.0 for i in range(bounds.n_bins)]


def _value_to_bin(value: float, bounds: ParamBounds) -> int | None:
    """Pure: which bin does this param value fall in? None if out of range."""
    if value < bounds.bound_min or value > bounds.bound_max:
        return None
    step = (bounds.bound_max - bounds.bound_min) / bounds.n_bins
    if step <= 0:
        return None
    idx = int((value - bounds.bound_min) / step)
    return min(idx, bounds.n_bins - 1)


def tune_param(
    *,
    bounds: ParamBounds,
    current: float,
    closed: list[ClosedOutcome],
    min_samples: int = 20,
) -> TuneDecision:
    """Pure: pick the next value for this param based on recent outcomes.

    Algorithm:
      1. Group `closed` by which bin their params_snapshot[bounds.name] fell into.
      2. Compute Sharpe per bin (need ≥2 samples per bin to score).
      3. If no bin has enough samples → leave current unchanged.
      4. Find the bin with highest Sharpe.
      5. Nudge toward that bin's center, bounded by max_step_pct of range.
      6. Clamp to [bound_min, bound_max].
    """
    if len(closed) < min_samples:
        return TuneDecision(
            name=bounds.name,
            current=current,
            proposed=current,
            delta=0.0,
            reason=f"insufficient_samples_n={len(closed)}<{min_samples}",
            n_samples=len(closed),
        )

    by_bin: dict[int, list[float]] = {}
    for outcome in closed:
        raw = outcome.params_snapshot.get(bounds.name)
        if not isinstance(raw, int | float):
            continue
        b = _value_to_bin(float(raw), bounds)
        if b is None:
            continue
        by_bin.setdefault(b, []).append(outcome.realized_pnl_usd)

    if not by_bin:
        return TuneDecision(
            name=bounds.name,
            current=current,
            proposed=current,
            delta=0.0,
            reason="no_in_range_samples",
            n_samples=len(closed),
        )

    centers = _bin_centers(bounds)
    bin_sharpes: dict[float, float] = {}
    for bin_idx, pnls in by_bin.items():
        s = _sharpe_proxy(pnls)
        if s is None:
            continue
        bin_sharpes[centers[bin_idx]] = s

    if not bin_sharpes:
        return TuneDecision(
            name=bounds.name,
            current=current,
            proposed=current,
            delta=0.0,
            reason="no_bin_with_sharpe",
            n_samples=len(closed),
            by_bin=bin_sharpes,
        )

    best_center = max(bin_sharpes.keys(), key=lambda c: bin_sharpes[c])
    target = best_center
    max_step = (bounds.bound_max - bounds.bound_min) * bounds.max_step_pct

    raw_delta = target - current
    delta = math.copysign(min(abs(raw_delta), max_step), raw_delta)
    proposed = current + delta
    proposed = max(bounds.bound_min, min(bounds.bound_max, proposed))

    return TuneDecision(
        name=bounds.name,
        current=current,
        proposed=proposed,
        delta=proposed - current,
        reason=(
            f"best_bin_center={best_center:.4f} sharpe={bin_sharpes[best_center]:.3f} "
            f"raw_delta={raw_delta:+.4f} capped={delta:+.4f}"
        ),
        n_samples=len(closed),
        by_bin=bin_sharpes,
    )


__all__ = [
    "ParamBounds",
    "TuneDecision",
    "tune_param",
]
