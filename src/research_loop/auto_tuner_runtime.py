"""Auto-tuner runtime — orchestrator that ties learning_loop, auto_tuner,
and param_registry into a periodic cycle.

Architecture:

    auto_tuner_loop (hourly by default)
        │
        ▼  for each opted-in strategy:
    fetch_recent_closed(pool, strategy_slug, limit=100, days=14)
        │
        ▼
    for each ParamBounds in TUNABLE_PARAMS[strategy_slug]:
        ├── get_param (Redis) → current value
        ├── tune_param (pure) → TuneDecision
        └── if |delta| > epsilon: set_param + audit log

Safety:
  - Default OFF. Settings.auto_tuner_enabled gate.
  - min_samples=30 default → won't nudge with < 30 closed trades.
  - Per-param bounds defined in TUNABLE_PARAMS; clamps everything.
  - max_step_pct=0.10 → no parameter moves more than 10% of range per cycle.
  - Audit log entry written on every nudge — full history in
    `params:audit` Redis stream.

To opt a new strategy in: add an entry to TUNABLE_PARAMS with conservative
ParamBounds and enable in production via settings.auto_tuner_strategies.
"""
from __future__ import annotations

import logging
from typing import Any

from . import learning_loop
from .auto_tuner import ParamBounds, tune_param
from .param_registry import get_param, set_param
from .settings import settings

log = logging.getLogger(__name__)


# Per-strategy tunable params + safe bounds. Bounds are deliberately
# conservative — even at maximum step (10% of range per cycle), nothing
# can drift more than ~3-4 cycles' worth without hitting a clamp.
#
# When adding a new strategy: pick a single most-impactful param to start;
# expand once the loop has run cleanly for ≥7 days.
TUNABLE_PARAMS: dict[str, list[ParamBounds]] = {
    "poly-sell-wings": [
        ParamBounds(name="yes_upper", bound_min=0.85, bound_max=0.95,
                    default=0.90, n_bins=5, max_step_pct=0.10),
        ParamBounds(name="yes_lower", bound_min=0.05, bound_max=0.15,
                    default=0.10, n_bins=5, max_step_pct=0.10),
        ParamBounds(name="min_volume", bound_min=1_000.0, bound_max=20_000.0,
                    default=5_000.0, n_bins=5, max_step_pct=0.10),
    ],
    "poly-publisher-taker": [
        ParamBounds(name="yes_upper", bound_min=0.75, bound_max=0.95,
                    default=0.85, n_bins=5, max_step_pct=0.10),
        ParamBounds(name="yes_lower", bound_min=0.05, bound_max=0.25,
                    default=0.15, n_bins=5, max_step_pct=0.10),
        ParamBounds(name="min_depth_usd", bound_min=500.0, bound_max=10_000.0,
                    default=1_000.0, n_bins=5, max_step_pct=0.10),
    ],
    "poly-cross-market-arb": [
        ParamBounds(name="min_inversion_pp", bound_min=0.01, bound_max=0.10,
                    default=0.02, n_bins=5, max_step_pct=0.10),
        ParamBounds(name="min_volume", bound_min=1_000.0, bound_max=20_000.0,
                    default=5_000.0, n_bins=5, max_step_pct=0.10),
    ],
    "poly-settlement-momentum": [
        ParamBounds(name="trigger_pp", bound_min=0.01, bound_max=0.10,
                    default=0.03, n_bins=5, max_step_pct=0.10),
        ParamBounds(name="min_volume", bound_min=1_000.0, bound_max=20_000.0,
                    default=5_000.0, n_bins=5, max_step_pct=0.10),
    ],
}


# Floor on |delta| below which we don't bother writing — saves audit-log churn.
NUDGE_EPSILON: float = 1e-6


# ─── Pure helpers ──────────────────────────────────────────────────


def parse_opted_in_strategies(csv: str) -> list[str]:
    """Pure: parse a CSV of strategy slugs into a list, intersect with
    TUNABLE_PARAMS keys (silently drop unknown slugs).

    Empty CSV → empty list (no strategies opted in).
    "all"     → all strategies in TUNABLE_PARAMS.
    """
    if not csv or not csv.strip():
        return []
    if csv.strip().lower() == "all":
        return list(TUNABLE_PARAMS.keys())
    out: list[str] = []
    for s in csv.split(","):
        slug = s.strip()
        if not slug:
            continue
        if slug not in TUNABLE_PARAMS:
            log.warning(
                "auto_tuner_runtime.unknown_strategy_in_opt_in slug=%s — skipping",
                slug,
            )
            continue
        out.append(slug)
    return out


# ─── Async orchestrator ────────────────────────────────────────────


async def run_tune_cycle(
    pool: Any,
    redis_client: Any,
    *,
    strategies: list[str],
    min_samples: int,
    days: int = 14,
    limit: int = 100,
    dry_run: bool = False,
) -> dict[str, dict[str, Any]]:
    """One pass over the opted-in strategies. Returns per-strategy stats.

    {
      strategy_slug: {
        n_closed: int,
        per_param: {param_name: {current, proposed, delta, reason}},
        n_nudged: int,
      }
    }

    `dry_run`: when True, decisions computed but NOT written to Redis.
    The audit-log still NOT written. Useful for first-N cycles to validate
    behavior before flipping enforcement.
    """
    out: dict[str, dict[str, Any]] = {}
    for slug in strategies:
        bounds_list = TUNABLE_PARAMS.get(slug)
        if not bounds_list:
            continue
        try:
            closed = await learning_loop.fetch_recent_closed(
                pool, strategy_slug=slug, limit=limit, days=days,
            )
        except Exception:
            log.exception("auto_tuner_runtime.fetch_closed_failed slug=%s", slug)
            out[slug] = {
                "n_closed": 0, "per_param": {}, "n_nudged": 0,
                "error": "fetch_closed_failed",
            }
            continue
        n_nudged = 0
        per_param: dict[str, Any] = {}
        for bounds in bounds_list:
            try:
                current = await get_param(
                    redis_client, slug, bounds.name, default=bounds.default,
                )
            except Exception:
                log.exception("auto_tuner_runtime.get_param_failed slug=%s name=%s",
                              slug, bounds.name)
                continue
            decision = tune_param(
                bounds=bounds,
                current=current,
                closed=closed,
                min_samples=min_samples,
            )
            per_param[bounds.name] = {
                "current": decision.current,
                "proposed": decision.proposed,
                "delta": decision.delta,
                "reason": decision.reason,
                "n_samples": decision.n_samples,
            }
            if abs(decision.delta) < NUDGE_EPSILON:
                continue
            if dry_run:
                log.info(
                    "auto_tuner_runtime.dry_run_nudge slug=%s name=%s "
                    "current=%.4f proposed=%.4f reason=%s",
                    slug, bounds.name, decision.current, decision.proposed,
                    decision.reason,
                )
                continue
            try:
                await set_param(
                    redis_client, slug, bounds.name, decision.proposed,
                    old_value=decision.current, reason=decision.reason,
                )
                n_nudged += 1
                log.warning(
                    "auto_tuner_runtime.nudged slug=%s name=%s %.4f→%.4f reason=%s",
                    slug, bounds.name, decision.current, decision.proposed,
                    decision.reason,
                )
            except Exception:
                log.exception(
                    "auto_tuner_runtime.set_param_failed slug=%s name=%s",
                    slug, bounds.name,
                )
        out[slug] = {
            "n_closed": len(closed),
            "per_param": per_param,
            "n_nudged": n_nudged,
        }
    return out


async def auto_tuner_loop(
    stop_event: Any,
    pool_getter: Any,
    redis_client_getter: Any,
) -> None:
    """Forever loop. Runs on settings.auto_tuner_interval_sec cadence.
    Settings.auto_tuner_dry_run flips enforcement on/off independently of
    the loop being enabled."""
    import asyncio

    strategies = parse_opted_in_strategies(settings.auto_tuner_strategies)
    if not strategies:
        log.info("auto_tuner_runtime.no_opted_in_strategies — loop will idle")
        await stop_event.wait()
        return

    log.info(
        "auto_tuner_runtime.starting strategies=%s interval=%ds dry_run=%s "
        "min_samples=%d days=%d",
        strategies,
        settings.auto_tuner_interval_sec,
        settings.auto_tuner_dry_run,
        settings.auto_tuner_min_samples,
        settings.auto_tuner_days_window,
    )
    while not stop_event.is_set():
        try:
            pool = await pool_getter()
            redis_client = await redis_client_getter()
            stats = await run_tune_cycle(
                pool, redis_client,
                strategies=strategies,
                min_samples=settings.auto_tuner_min_samples,
                days=settings.auto_tuner_days_window,
                limit=settings.auto_tuner_closed_limit,
                dry_run=settings.auto_tuner_dry_run,
            )
            log.info("auto_tuner_runtime.cycle %s", stats)
        except Exception:
            log.exception("auto_tuner_runtime.cycle_failed")
        try:
            await asyncio.wait_for(
                stop_event.wait(), timeout=settings.auto_tuner_interval_sec,
            )
        except TimeoutError:
            pass


__all__ = [
    "NUDGE_EPSILON",
    "TUNABLE_PARAMS",
    "auto_tuner_loop",
    "parse_opted_in_strategies",
    "run_tune_cycle",
]
