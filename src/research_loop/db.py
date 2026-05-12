"""Postgres reader for the research loop.

Reads per-strategy aggregated stats from the `positions` table (same DB
the trading stack writes to). Read-only.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from .settings import settings
from .sharpe import StrategyStats, sharpe_proxy

log = logging.getLogger(__name__)

_pool: Any = None


async def _get_pool() -> Any:
    global _pool
    if _pool is None:
        import asyncpg
        _pool = await asyncpg.create_pool(settings.postgres_dsn, min_size=1, max_size=4)
    return _pool


async def close() -> None:
    global _pool
    if _pool is not None:
        try:
            await _pool.close()
        except Exception:
            log.exception("research_loop.db.close_failed")
        _pool = None


async def fetch_strategy_stats(
    *, window_days: int | None = None,
) -> list[StrategyStats]:
    """Read per-strategy aggregated stats over the configured window.

    Returns one StrategyStats per (strategy, venue). Sharpe is computed
    from the per-trade pnl distribution. Strategies with no closed
    positions in the window are omitted.
    """
    window = window_days or settings.window_days
    since = datetime.now(UTC) - timedelta(days=window)

    pool = await _get_pool()
    rows = await pool.fetch(
        """
        SELECT s.slug AS slug,
               p.venue AS venue,
               COUNT(*)::int AS n_closed,
               COALESCE(SUM(p.realized_pnl_usd), 0)::float AS total_pnl_usd,
               EXTRACT(EPOCH FROM (NOW() - MIN(p.opened_at))) / 86400 AS days_observed,
               array_agg(p.realized_pnl_usd) AS pnls
        FROM positions p
        JOIN strategies s ON s.id = p.strategy_id
        WHERE p.status = 'closed'
          AND p.closed_at >= $1
        GROUP BY s.slug, p.venue
        """,
        since,
    )

    open_counts = await pool.fetch(
        """
        SELECT s.slug AS slug,
               p.venue AS venue,
               COUNT(*)::int AS n_open
        FROM positions p
        JOIN strategies s ON s.id = p.strategy_id
        WHERE p.status IN ('open', 'partial')
        GROUP BY s.slug, p.venue
        """,
    )
    open_map = {(r["slug"], r["venue"]): int(r["n_open"]) for r in open_counts}

    out: list[StrategyStats] = []
    for r in rows:
        pnls = [float(x) for x in (r["pnls"] or []) if x is not None]
        out.append(StrategyStats(
            slug=f"{r['slug']}@{r['venue']}",
            n_closed=int(r["n_closed"]),
            total_pnl_usd=float(r["total_pnl_usd"]),
            sharpe=sharpe_proxy(pnls),
            days_observed=int(r["days_observed"] or 0),
            open_positions=open_map.get((r["slug"], r["venue"]), 0),
        ))
    return out


__all__ = ["close", "fetch_strategy_stats"]
