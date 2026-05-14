"""Learning loop — feature capture, outcome join, and the shared substrate
that powers both the parameter auto-tuner and the ML training-prep export.

Architecture:

    strategy emits alpha
            │
            ▼
    capture_emit(alpha_id, strategy, features, params_snapshot)
            │  (writes feature_outcomes row, outcome=NULL)
            ▼
    [time passes — position opens, closes, PnL realized]
            │
            ▼
    run_outcome_join()   (joins feature_outcomes rows to closed positions
                          via alpha_id → oms_intents.metadata.alpha_id →
                          positions.open_intent_id)
            │
            ▼
    feature_outcomes row updated with outcome + realized_pnl + closed_at
            │
            ▼
    Two consumers read the same table:
        1. auto_tuner — short-horizon (last 50-100 closes) param nudge
        2. training_export — nightly Parquet snapshot for offline ML

PURE schema + capture + join logic. Auto-tuner math lives in auto_tuner.py.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


# ─── Schema ──────────────────────────────────────────────────────────


FEATURE_OUTCOMES_DDL = """
CREATE TABLE IF NOT EXISTS feature_outcomes (
    alpha_id          UUID PRIMARY KEY,
    strategy_slug     TEXT NOT NULL,
    asset             TEXT NOT NULL,
    venue             TEXT NOT NULL,
    emitted_at        TIMESTAMPTZ NOT NULL,

    -- Feature snapshot at emit time (strategy-specific JSON dict).
    -- The auto-tuner reads named scalars; the ML export reads the whole dict.
    features          JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- Strategy params at emit time. Used by auto-tuner to attribute outcomes
    -- to the specific param values in use when the alpha fired.
    params_snapshot   JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- Outcome — populated by run_outcome_join when position closes.
    -- States: NULL (awaiting outcome), 'open', 'win', 'loss', 'flat',
    -- 'rejected' (oms-gateway rejected the alpha, no position opened).
    outcome           TEXT,
    position_id       UUID,
    realized_pnl_usd  DOUBLE PRECISION,
    notional_usd      DOUBLE PRECISION,
    closed_at         TIMESTAMPTZ,

    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- For the auto-tuner: fetch recent closes for one strategy
CREATE INDEX IF NOT EXISTS feature_outcomes_strategy_emitted_idx
    ON feature_outcomes (strategy_slug, emitted_at DESC);

-- For the outcome-joiner: find open rows efficiently
CREATE INDEX IF NOT EXISTS feature_outcomes_open_idx
    ON feature_outcomes (emitted_at)
    WHERE outcome IS NULL OR outcome = 'open';

-- For backfill: join via position_id
CREATE INDEX IF NOT EXISTS feature_outcomes_position_idx
    ON feature_outcomes (position_id)
    WHERE position_id IS NOT NULL;
"""


async def ensure_schema(pool: Any) -> None:
    """Idempotent — runs DDL. Safe to call on every boot."""
    async with pool.acquire() as conn:
        await conn.execute(FEATURE_OUTCOMES_DDL)
    log.info("learning_loop.schema_ensured")


# ─── Capture API ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class EmitRecord:
    """One emit-time snapshot. Producer side (strategy-runners) constructs
    this and calls capture_emit. Consumer side (auto-tuner) reads back."""
    alpha_id: str          # UUID string
    strategy_slug: str
    asset: str
    venue: str
    emitted_at_iso: str    # ISO-8601 UTC
    features: dict[str, Any]
    params_snapshot: dict[str, Any]


async def capture_emit(pool: Any, record: EmitRecord) -> None:
    """Insert a feature_outcomes row at alpha-emit time.

    Idempotent on alpha_id — replays during pipeline reruns won't double-write.
    Producer (strategy-runners) calls this synchronously inside the emit path;
    failure logs but doesn't break the strategy (the alpha still emits).
    """
    sql = """
        INSERT INTO feature_outcomes
            (alpha_id, strategy_slug, asset, venue, emitted_at,
             features, params_snapshot)
        VALUES ($1, $2, $3, $4, $5::timestamptz, $6::jsonb, $7::jsonb)
        ON CONFLICT (alpha_id) DO NOTHING
    """
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                sql,
                record.alpha_id,
                record.strategy_slug,
                record.asset,
                record.venue,
                record.emitted_at_iso,
                json.dumps(record.features),
                json.dumps(record.params_snapshot),
            )
    except Exception:
        log.exception("learning_loop.capture_failed alpha_id=%s", record.alpha_id)


# ─── Outcome joiner ──────────────────────────────────────────────────


# Joins feature_outcomes to closed positions via the alpha_id → intent →
# position chain. Run periodically (every 5 min). One-pass UPDATE.
#
# - alpha_id lives in oms_intents.metadata->>'alpha_id'
# - intent → position via positions.open_intent_id
# - Only update rows still NULL/open and the position is closed
OUTCOME_JOIN_SQL = """
WITH closed AS (
    SELECT
        (i.metadata->>'alpha_id')::uuid                 AS alpha_id,
        p.id                                            AS position_id,
        p.realized_pnl_usd                              AS pnl,
        (p.qty * p.avg_entry_price)                     AS notional,
        p.closed_at                                     AS closed_at,
        CASE
            WHEN p.realized_pnl_usd > 0  THEN 'win'
            WHEN p.realized_pnl_usd < 0  THEN 'loss'
            ELSE 'flat'
        END AS outcome
    FROM positions p
    JOIN oms_intents i ON i.id = p.open_intent_id
    WHERE p.status = 'closed'
      AND p.closed_at >= NOW() - INTERVAL '7 days'
      AND (i.metadata->>'alpha_id') IS NOT NULL
)
UPDATE feature_outcomes fo
SET
    outcome           = c.outcome,
    position_id       = c.position_id,
    realized_pnl_usd  = c.pnl,
    notional_usd      = c.notional,
    closed_at         = c.closed_at,
    updated_at        = NOW()
FROM closed c
WHERE fo.alpha_id = c.alpha_id
  AND (fo.outcome IS NULL OR fo.outcome = 'open')
"""


async def run_outcome_join(pool: Any) -> int:
    """Update feature_outcomes with newly-closed position outcomes.

    Returns number of rows updated. Called periodically by the loop.
    """
    async with pool.acquire() as conn:
        result = await conn.execute(OUTCOME_JOIN_SQL)
    # asyncpg returns string like 'UPDATE 42' — parse the count
    try:
        n = int(result.rsplit(" ", 1)[-1])
    except (ValueError, IndexError):
        n = 0
    log.info("learning_loop.outcome_join updated=%d", n)
    return n


# ─── Reader API (for auto-tuner + training export) ──────────────────


@dataclass(frozen=True)
class ClosedOutcome:
    """One closed (features, outcome) pair for downstream consumers."""
    alpha_id: str
    strategy_slug: str
    asset: str
    venue: str
    emitted_at_iso: str
    features: dict[str, Any]
    params_snapshot: dict[str, Any]
    outcome: str               # 'win' | 'loss' | 'flat'
    realized_pnl_usd: float
    notional_usd: float
    closed_at_iso: str


async def fetch_recent_closed(
    pool: Any,
    *,
    strategy_slug: str | None = None,
    limit: int = 100,
    days: int = 14,
) -> list[ClosedOutcome]:
    """Pull recent closed (features, outcome) rows for analysis.

    Filters:
      - strategy_slug=None → all strategies; otherwise just the one
      - days: time window (default 14d)
      - limit: max rows returned (newest first)
    """
    if strategy_slug:
        sql = """
            SELECT alpha_id::text, strategy_slug, asset, venue,
                   emitted_at::text, features, params_snapshot,
                   outcome, realized_pnl_usd, notional_usd, closed_at::text
            FROM feature_outcomes
            WHERE strategy_slug = $1
              AND outcome IN ('win', 'loss', 'flat')
              AND closed_at >= NOW() - ($3 || ' days')::interval
            ORDER BY emitted_at DESC
            LIMIT $2
        """
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, strategy_slug, limit, str(days))
    else:
        sql = """
            SELECT alpha_id::text, strategy_slug, asset, venue,
                   emitted_at::text, features, params_snapshot,
                   outcome, realized_pnl_usd, notional_usd, closed_at::text
            FROM feature_outcomes
            WHERE outcome IN ('win', 'loss', 'flat')
              AND closed_at >= NOW() - ($2 || ' days')::interval
            ORDER BY emitted_at DESC
            LIMIT $1
        """
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, limit, str(days))
    out: list[ClosedOutcome] = []
    for r in rows:
        features_raw = r["features"]
        if isinstance(features_raw, dict):
            features = features_raw
        else:
            features = json.loads(features_raw or "{}")
        params_raw = r["params_snapshot"]
        if isinstance(params_raw, dict):
            params = params_raw
        else:
            params = json.loads(params_raw or "{}")
        out.append(ClosedOutcome(
            alpha_id=str(r["alpha_id"]),
            strategy_slug=str(r["strategy_slug"]),
            asset=str(r["asset"]),
            venue=str(r["venue"]),
            emitted_at_iso=str(r["emitted_at"]),
            features=features,
            params_snapshot=params,
            outcome=str(r["outcome"]),
            realized_pnl_usd=float(r["realized_pnl_usd"] or 0.0),
            notional_usd=float(r["notional_usd"] or 0.0),
            closed_at_iso=str(r["closed_at"] or ""),
        ))
    return out


__all__ = [
    "ClosedOutcome",
    "EmitRecord",
    "FEATURE_OUTCOMES_DDL",
    "OUTCOME_JOIN_SQL",
    "capture_emit",
    "ensure_schema",
    "fetch_recent_closed",
    "run_outcome_join",
]
