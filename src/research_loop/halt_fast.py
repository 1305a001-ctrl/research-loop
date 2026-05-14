"""Halt-fast trigger — venue-scoped 24h PnL circuit breaker.

When the realised PnL on a venue (e.g. polymarket) over the last 24h
drops below a floor (e.g. -$50), HALT every strategy currently active
on that venue. Prevents the death-spiral pattern we saw last week:
poly-crypto-momentum + poly-politics-momentum bled $42k between them
over ~36h before manual intervention.

This is independent of the per-strategy `should_halt` decision in
`sharpe.py` — that one is slow-moving (30 closed trades, 7 days
observed). Halt-fast is the fire-alarm: trip immediately on bleed,
even if no single strategy has enough closed trades to be evaluated.

Pure decision (`decide_halt_fast`) is tested in isolation; the I/O
glue is thin.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from . import halts
from .settings import settings

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class HaltFastDecision:
    """One venue's halt-fast verdict."""
    venue: str
    realized_pnl_24h_usd: float
    threshold_usd: float
    trip: bool
    n_closed_24h: int
    reason: str


# ─── Pure decision ─────────────────────────────────────────────────


def decide_halt_fast(
    *,
    venue: str,
    realized_pnl_24h_usd: float,
    n_closed_24h: int,
    threshold_usd: float,
    min_n_closed: int = 1,
) -> HaltFastDecision:
    """Pure: should we trip the venue-wide halt?

    Trips when BOTH:
      - realized_pnl_24h_usd < threshold_usd (negative threshold, e.g. -50)
      - n_closed_24h >= min_n_closed (don't trip on a single noisy close)

    `min_n_closed=1` by default — even one big realized loss should
    trigger, because the realised PnL itself is the signal we care
    about (not statistical confidence in the strategy's expected
    return).

    Returns a structured decision with the reason filled in.
    """
    if realized_pnl_24h_usd >= threshold_usd:
        return HaltFastDecision(
            venue=venue,
            realized_pnl_24h_usd=realized_pnl_24h_usd,
            threshold_usd=threshold_usd,
            trip=False,
            n_closed_24h=n_closed_24h,
            reason=f"pnl_above_floor (${realized_pnl_24h_usd:.2f} >= ${threshold_usd:.2f})",
        )
    if n_closed_24h < min_n_closed:
        return HaltFastDecision(
            venue=venue,
            realized_pnl_24h_usd=realized_pnl_24h_usd,
            threshold_usd=threshold_usd,
            trip=False,
            n_closed_24h=n_closed_24h,
            reason=f"insufficient_closed_count ({n_closed_24h} < {min_n_closed})",
        )
    return HaltFastDecision(
        venue=venue,
        realized_pnl_24h_usd=realized_pnl_24h_usd,
        threshold_usd=threshold_usd,
        trip=True,
        n_closed_24h=n_closed_24h,
        reason=(
            f"halt_fast_24h_pnl_floor_breach "
            f"(${realized_pnl_24h_usd:.2f} < ${threshold_usd:.2f}, "
            f"n_closed={n_closed_24h})"
        ),
    )


# ─── DB read (thin) ────────────────────────────────────────────────


_FETCH_24H_PNL_SQL = """
    SELECT
      COUNT(*)::int                                       AS n_closed,
      COALESCE(SUM(p.realized_pnl_usd), 0)::float         AS realized_pnl_24h_usd,
      array_agg(DISTINCT s.slug)                          AS slugs
    FROM positions p
    JOIN strategies s ON s.id = p.strategy_id
    WHERE p.status = 'closed'
      AND p.closed_at >= $1
      AND p.venue = $2
"""


async def fetch_venue_pnl_24h(
    pool: Any,
    venue: str,
    *,
    now: datetime | None = None,
) -> tuple[float, int, list[str]]:
    """Read realized PnL + closed count + distinct slugs for a venue
    over the last 24h.

    Returns (realized_pnl_24h_usd, n_closed, distinct_slugs). On error,
    returns (0.0, 0, []) so the caller can no-op rather than crash —
    we'd rather miss a halt than wrongly halt-everything on a DB blip.
    """
    cutoff = (now or datetime.now(UTC)) - timedelta(hours=24)
    try:
        row = await pool.fetchrow(_FETCH_24H_PNL_SQL, cutoff, venue)
    except Exception:
        log.exception("halt_fast.fetch_venue_pnl_failed venue=%s", venue)
        return (0.0, 0, [])
    if row is None:
        return (0.0, 0, [])
    slugs = [s for s in (row["slugs"] or []) if s]
    return (
        float(row["realized_pnl_24h_usd"] or 0.0),
        int(row["n_closed"] or 0),
        slugs,
    )


# ─── Async orchestrator ────────────────────────────────────────────


async def run_halt_fast_cycle(
    pool: Any,
    *,
    venue: str,
    threshold_usd: float,
    min_n_closed: int = 1,
    now: datetime | None = None,
) -> dict[str, Any]:
    """One iteration: fetch venue 24h PnL → decide → actuate halts.

    Returns a stats dict for logging:
        venue, realized_pnl_24h_usd, n_closed_24h, trip, n_strategies,
        n_halted_now, n_already_halted, slugs.

    Calls halts.actuate (which respects settings.enforce_halts) on
    EVERY active slug for the venue when the decision trips. Idempotent
    by virtue of `actuate` using SADD and returning 0 if the slug was
    already in the set.
    """
    pnl, n_closed, slugs = await fetch_venue_pnl_24h(pool, venue, now=now)
    decision = decide_halt_fast(
        venue=venue,
        realized_pnl_24h_usd=pnl,
        n_closed_24h=n_closed,
        threshold_usd=threshold_usd,
        min_n_closed=min_n_closed,
    )

    stats: dict[str, Any] = {
        "venue": venue,
        "realized_pnl_24h_usd": round(pnl, 4),
        "n_closed_24h": n_closed,
        "trip": decision.trip,
        "n_strategies": len(slugs),
        "n_halted_now": 0,
        "n_already_halted": 0,
        "slugs": slugs,
        "reason": decision.reason,
    }

    if not decision.trip:
        return stats

    log.warning(
        "halt_fast.tripping venue=%s pnl_24h=%.2f n_closed=%d strategies=%d",
        venue, pnl, n_closed, len(slugs),
    )
    for slug in slugs:
        ok = await halts.actuate(slug, decision.reason)
        if ok:
            stats["n_halted_now"] += 1
        else:
            stats["n_already_halted"] += 1
    return stats


async def halt_fast_loop(
    stop_event: Any,
    pool_getter: Any,
    *,
    venues: list[str],
    threshold_usd: float,
    interval_sec: int,
    min_n_closed: int = 1,
) -> None:
    """Forever loop with stop_event support. One iteration per venue per
    interval. Picks up settings.enforce_halts via halts.actuate, so
    in dry-run mode you get logs but no SADD.
    """
    import asyncio

    log.info(
        "halt_fast.loop_starting venues=%s threshold=%.2f interval=%ds enforce=%s",
        venues, threshold_usd, interval_sec, settings.enforce_halts,
    )
    while not stop_event.is_set():
        try:
            pool = await pool_getter()
            for v in venues:
                stats = await run_halt_fast_cycle(
                    pool, venue=v,
                    threshold_usd=threshold_usd,
                    min_n_closed=min_n_closed,
                )
                log.info("halt_fast.cycle %s", stats)
        except Exception:
            log.exception("halt_fast.cycle_failed")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_sec)
        except TimeoutError:
            pass


__all__ = [
    "HaltFastDecision",
    "decide_halt_fast",
    "fetch_venue_pnl_24h",
    "halt_fast_loop",
    "run_halt_fast_cycle",
]
