"""Sharpe + auto-halt math.

Two responsibilities, both pure (no I/O):
  1. `sharpe_proxy(pnl_list)` — per-trade returns → unitless Sharpe proxy.
     Not annualized (we'd need a return-rate, not raw $); useful for
     RELATIVE ranking between strategies. Same metric as the
     control-plane dashboard exposes.
  2. `should_halt(stats)` — given (n_closed, total_pnl, sharpe_proxy,
     days_since_first_trade), decide whether the strategy should be
     auto-halted. Conservative — defaults err toward keeping strategies
     RUNNING (false negatives cheap, false positives expensive).
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass


def sharpe_proxy(pnls: list[float]) -> float | None:
    """Pure: mean(pnl) / stddev(pnl). Returns None for <2 samples or
    zero-stddev (all identical returns).

    NOT annualized. Useful only for relative ranking — same metric the
    control-plane dashboard's StrategyFitnessGrid uses, so the two views
    line up."""
    if len(pnls) < 2:
        return None
    avg = statistics.mean(pnls)
    stdev = statistics.stdev(pnls)
    if stdev <= 0:
        return None
    return avg / stdev


@dataclass(frozen=True)
class StrategyStats:
    """Aggregated stats for one strategy over a measurement window."""
    slug: str
    n_closed: int
    total_pnl_usd: float
    sharpe: float | None        # may be None if too few samples
    days_observed: int          # days since first closed trade in the window
    open_positions: int = 0     # informational; not a halt criterion


@dataclass(frozen=True)
class HaltDecision:
    """Output of should_halt — decision + reason for the audit log."""
    halt: bool
    reason: str
    severity: str               # "info" | "warn" | "critical"


def should_halt(
    stats: StrategyStats,
    *,
    min_n_closed: int = 30,                   # need at least N trades to judge
    min_days_observed: int = 7,               # and at least N days
    sharpe_threshold: float = 0.0,            # below 0 = losing money on avg
    catastrophic_pnl_floor_usd: float = -200, # immediate halt if total < this
) -> HaltDecision:
    """Pure: should this strategy be auto-halted?

    Rules (in order; first match wins):
      1. Catastrophic loss → halt regardless of sample size
      2. Insufficient sample → don't halt (need data first)
      3. Sharpe < threshold → halt (with explanation)
      4. Otherwise → don't halt

    Default thresholds err on the side of LETTING strategies run.
    Auto-halt is conservative; manual review still has the final say.
    """
    # Rule 1: catastrophic loss is an immediate halt regardless of sample
    if stats.total_pnl_usd <= catastrophic_pnl_floor_usd:
        return HaltDecision(
            halt=True,
            reason=(
                f"catastrophic_loss total_pnl=${stats.total_pnl_usd:.2f} "
                f"≤ floor ${catastrophic_pnl_floor_usd:.2f}"
            ),
            severity="critical",
        )

    # Rule 2: insufficient data — leave running
    if stats.n_closed < min_n_closed:
        return HaltDecision(
            halt=False,
            reason=f"insufficient_sample n_closed={stats.n_closed} < {min_n_closed}",
            severity="info",
        )
    if stats.days_observed < min_days_observed:
        return HaltDecision(
            halt=False,
            reason=f"insufficient_days days={stats.days_observed} < {min_days_observed}",
            severity="info",
        )

    # Rule 3: bad Sharpe → halt
    if stats.sharpe is None:
        return HaltDecision(
            halt=False,
            reason="sharpe_unavailable",
            severity="info",
        )
    if stats.sharpe < sharpe_threshold:
        return HaltDecision(
            halt=True,
            reason=(
                f"sharpe_below_threshold sharpe={stats.sharpe:.3f} "
                f"< {sharpe_threshold:.3f} (n={stats.n_closed}, "
                f"pnl=${stats.total_pnl_usd:.2f}, days={stats.days_observed})"
            ),
            severity="warn",
        )

    return HaltDecision(
        halt=False,
        reason=f"healthy sharpe={stats.sharpe:.3f} pnl=${stats.total_pnl_usd:.2f}",
        severity="info",
    )


def health_badge(stats: StrategyStats) -> str:
    """Pure: short status label for reports. Matches control-plane's badge
    logic so the two views agree visually."""
    if stats.n_closed < 5:
        return "small-N"
    if stats.total_pnl_usd > 0 and (stats.sharpe or 0) >= 0.3:
        return "healthy"
    if stats.total_pnl_usd >= 0:
        return "mixed"
    return "bleed"


__all__ = ["HaltDecision", "StrategyStats", "health_badge", "sharpe_proxy", "should_halt"]
