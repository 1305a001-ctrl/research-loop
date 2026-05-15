"""research-loop entrypoint.

One cycle per `cycle_interval_sec` (default 1h):
  1. Read per-strategy aggregated stats from Postgres
  2. Run `should_halt` on each
  3. For strategies meeting halt criteria, actuate via Redis (or log
     when enforce=False)
  4. Print summary so we have an in-stdout audit trail

FastAPI /health on :8015 for observability.
"""
from __future__ import annotations

import asyncio
import signal
import time
from pathlib import Path

import structlog
import uvicorn
from fastapi import FastAPI, Response

from . import (
    auto_tuner_runtime,
    db,
    halt_fast,
    halts,
    learning_loop,
    training_export,
)
from .settings import settings
from .sharpe import health_badge, should_halt

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)
log = structlog.get_logger(__name__)


# Snapshot of last cycle for /health
_last_cycle = {"at_unix": 0, "n_evaluated": 0, "n_halted": 0, "n_would_halt": 0}


async def run_cycle() -> None:
    """One iteration of the loop."""
    try:
        stats = await db.fetch_strategy_stats()
    except Exception:
        log.exception("research_loop.fetch_stats_failed")
        return

    n_halted = 0
    n_would_halt = 0
    summary_lines = []
    for s in stats:
        decision = should_halt(
            s,
            min_n_closed=settings.min_n_closed_for_halt,
            min_days_observed=settings.min_days_observed_for_halt,
            sharpe_threshold=settings.sharpe_halt_threshold,
            catastrophic_pnl_floor_usd=settings.catastrophic_pnl_floor_usd,
        )
        badge = health_badge(s)
        already = await halts.is_halted(s.slug.split("@")[0])
        summary_lines.append(
            f"  {s.slug:50s} n={s.n_closed:>4d} pnl=${s.total_pnl_usd:>10.2f} "
            f"sharpe={s.sharpe if s.sharpe is not None else 'NA':>7} "
            f"[{badge}] {'HALT' if decision.halt else 'ok'}{' (already)' if already else ''}",
        )
        if decision.halt and not already:
            base_slug = s.slug.split("@")[0]   # drop @venue suffix
            actuated = await halts.actuate(base_slug, decision.reason)
            if actuated:
                n_halted += 1
            else:
                n_would_halt += 1

    log.info(
        "research_loop.cycle n_evaluated=%d n_halted=%d n_would_halt=%d",
        len(stats), n_halted, n_would_halt,
    )
    if summary_lines:
        for line in summary_lines[:50]:
            log.info("research_loop.strategy %s", line)

    _last_cycle["at_unix"] = int(time.time())
    _last_cycle["n_evaluated"] = len(stats)
    _last_cycle["n_halted"] = n_halted
    _last_cycle["n_would_halt"] = n_would_halt


async def cycle_loop(stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        await run_cycle()
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=settings.cycle_interval_sec)
        except TimeoutError:
            pass


async def outcome_join_loop(stop_event: asyncio.Event) -> None:
    """Periodically join newly-closed positions into feature_outcomes."""
    while not stop_event.is_set():
        try:
            pool = await db._get_pool()
            updated = await learning_loop.run_outcome_join(pool)
            _last_cycle["outcome_join_updated"] = updated
        except Exception:
            log.exception("learning_loop.outcome_join_failed")
        try:
            await asyncio.wait_for(
                stop_event.wait(), timeout=settings.outcome_join_interval_sec,
            )
        except TimeoutError:
            pass


async def training_export_loop(stop_event: asyncio.Event) -> None:
    """Nightly export of (features, outcomes) for ML training prep."""
    while not stop_event.is_set():
        try:
            pool = await db._get_pool()
            counts = await training_export.run_nightly_export(
                pool, output_dir=Path(settings.training_export_dir),
            )
            _last_cycle["training_export_rows"] = sum(counts.values())
        except Exception:
            log.exception("training_export.failed")
        try:
            await asyncio.wait_for(
                stop_event.wait(), timeout=settings.training_export_interval_sec,
            )
        except TimeoutError:
            pass


def make_app() -> FastAPI:
    app = FastAPI(title="research-loop", version="0.1.0")

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {
            "status": "ok",
            "last_cycle": _last_cycle,
            "enforce_halts": settings.enforce_halts,
        }

    @app.get("/metrics")
    async def metrics() -> Response:
        body = "\n".join([
            "# HELP research_loop_strategies_evaluated Strategies evaluated last cycle",
            "# TYPE research_loop_strategies_evaluated gauge",
            f"research_loop_strategies_evaluated {_last_cycle['n_evaluated']}",
            "# HELP research_loop_halts_actuated Strategies halted last cycle",
            "# TYPE research_loop_halts_actuated counter",
            f"research_loop_halts_actuated {_last_cycle['n_halted']}",
            "# HELP research_loop_halts_would_actuate Halts logged but not actuated (dry-run)",
            "# TYPE research_loop_halts_would_actuate counter",
            f"research_loop_halts_would_actuate {_last_cycle['n_would_halt']}",
            "",
        ])
        return Response(content=body, media_type="text/plain")

    return app


async def main() -> None:
    log.info(
        "research_loop.start version=0.2.0 enforce_halts=%s window_days=%d "
        "learning_loop=%s training_export=%s",
        settings.enforce_halts, settings.window_days,
        settings.learning_loop_enabled, settings.training_export_enabled,
    )

    # Bootstrap learning-loop schema (idempotent)
    if settings.learning_loop_enabled:
        try:
            pool = await db._get_pool()
            await learning_loop.ensure_schema(pool)
        except Exception:
            log.exception("learning_loop.bootstrap_failed — continuing")

    stop_event = asyncio.Event()
    app = make_app()
    server = uvicorn.Server(uvicorn.Config(
        app,
        host=settings.http_host,
        port=settings.http_port,
        log_level=settings.log_level.lower(),
    ))

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    tasks: list = [cycle_loop(stop_event), server.serve()]
    if settings.learning_loop_enabled:
        tasks.append(outcome_join_loop(stop_event))
    if settings.training_export_enabled:
        tasks.append(training_export_loop(stop_event))
    if settings.halt_fast_enabled:
        venues = [v.strip() for v in settings.halt_fast_venues.split(",") if v.strip()]
        if venues:
            tasks.append(halt_fast.halt_fast_loop(
                stop_event,
                db._get_pool,
                venues=venues,
                threshold_usd=settings.halt_fast_pnl_floor_usd_24h,
                interval_sec=settings.halt_fast_interval_sec,
                min_n_closed=settings.halt_fast_min_n_closed,
            ))
    if settings.auto_tuner_enabled:
        # Redis client getter for the auto-tuner — uses the existing halts
        # module's singleton so we share the connection pool.
        tasks.append(auto_tuner_runtime.auto_tuner_loop(
            stop_event,
            db._get_pool,
            halts._get_client,
        ))

    await asyncio.gather(*tasks, return_exceptions=True)

    server.should_exit = True
    await db.close()
    await halts.close()
    log.info("research_loop.shutdown_complete")


if __name__ == "__main__":
    asyncio.run(main())
