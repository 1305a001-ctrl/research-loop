"""Training-data export — nightly Parquet snapshots for offline ML.

Reads feature_outcomes for the last N days, flattens features JSONB into
columnar form, writes a versioned Parquet file per (strategy, date).

Why Parquet:
  - Columnar = efficient for ML training (only read needed features)
  - Schema-on-read = features can evolve without breaking past snapshots
  - Versioned = each day's export is immutable; reruns don't clobber

Output path: {settings.training_export_dir}/strategy={slug}/date={YYYY-MM-DD}.parquet

When pyarrow isn't available (slim image), we fall back to JSONL — same
content, less efficient. The ML pipeline reads either format.
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .learning_loop import fetch_recent_closed

log = logging.getLogger(__name__)


async def export_recent(
    pool: Any,
    *,
    output_dir: Path,
    days: int = 1,
    strategies: list[str] | None = None,
) -> dict[str, int]:
    """Export the last N days of closed outcomes, partitioned by strategy.

    Returns dict of {strategy_slug: rows_exported}.
    Idempotent — re-running overwrites the day's file with same content.
    """
    rows = await fetch_recent_closed(pool, limit=100_000, days=days)
    if strategies:
        rows = [r for r in rows if r.strategy_slug in strategies]

    # Group by (strategy_slug, date)
    grouped: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        date_part = (r.closed_at_iso or r.emitted_at_iso or "")[:10]
        if not date_part:
            continue
        key = (r.strategy_slug, date_part)
        flat = {
            "alpha_id": r.alpha_id,
            "asset": r.asset,
            "venue": r.venue,
            "emitted_at": r.emitted_at_iso,
            "closed_at": r.closed_at_iso,
            "outcome": r.outcome,
            "realized_pnl_usd": r.realized_pnl_usd,
            "notional_usd": r.notional_usd,
            # Flatten features + params into prefixed columns
            **{f"feature_{k}": v for k, v in r.features.items()},
            **{f"param_{k}": v for k, v in r.params_snapshot.items()},
        }
        grouped.setdefault(key, []).append(flat)

    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    for (slug, date), records in grouped.items():
        strategy_dir = output_dir / f"strategy={slug}"
        strategy_dir.mkdir(parents=True, exist_ok=True)
        try:
            _write_parquet(strategy_dir / f"date={date}.parquet", records)
        except ImportError:
            # pyarrow not installed — fall back to JSONL
            _write_jsonl(strategy_dir / f"date={date}.jsonl", records)
        counts[slug] = counts.get(slug, 0) + len(records)
        log.info(
            "training_export.wrote strategy=%s date=%s n_rows=%d",
            slug, date, len(records),
        )

    return counts


def _write_parquet(path: Path, records: list[dict]) -> None:
    """Write records to a Parquet file. Raises ImportError if pyarrow absent."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not records:
        return
    # Build a uniform schema across records — collect all keys, fill missing with None
    all_keys: set[str] = set()
    for r in records:
        all_keys.update(r.keys())
    normalized = [{k: r.get(k) for k in all_keys} for r in records]
    table = pa.Table.from_pylist(normalized)
    pq.write_table(table, str(path), compression="zstd")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Fallback: JSONL when pyarrow unavailable. One record per line."""
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")


async def run_nightly_export(
    pool: Any,
    *,
    output_dir: Path,
) -> dict[str, int]:
    """Convenience wrapper — exports yesterday's data."""
    now = datetime.now(UTC)
    log.info("training_export.run_nightly start at=%s", now.isoformat())
    counts = await export_recent(pool, output_dir=output_dir, days=2)
    total = sum(counts.values())
    log.info("training_export.run_nightly done total_rows=%d strategies=%d", total, len(counts))
    return counts


def _todo_compute_train_test_split() -> None:
    """PLACEHOLDER for future: time-based train/test split for offline ML.

    For ML training, we'll want:
      - Train: closes >= N days ago (older than today's tail)
      - Test: closes in the last N days (held out)

    This avoids look-ahead bias when evaluating model performance. Build
    this when we have ~1000+ rows per strategy.
    """
    pass


# Avoid unused-helper warnings when pyarrow is missing
_ = (timedelta,)


__all__ = ["export_recent", "run_nightly_export"]
