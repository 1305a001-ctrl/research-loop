"""Halt actuation — writes to Redis `strategy:halts` SET.

The trading stack already reads this SET (halts.is_halted in
strategy-runners + liquidation-bot). Adding an entry here causes
the strategy to stop firing on its next cycle.

GATED by settings.enforce_halts. When False, we LOG would-be halts
without actually writing. Useful for the first few days of running
the research loop in observation mode before turning enforcement on.
"""
from __future__ import annotations

import logging
from typing import Any

from .settings import settings

log = logging.getLogger(__name__)

_client: Any = None


async def _get_client() -> Any:
    global _client
    if _client is None:
        from redis.asyncio import Redis
        _client = Redis.from_url(settings.redis_url, decode_responses=True)
    return _client


async def close() -> None:
    global _client
    if _client is not None:
        try:
            await _client.aclose()
        except Exception:
            log.exception("halts.close_failed")
        _client = None


async def is_halted(slug: str) -> bool:
    """Check whether a slug is currently in the halt set."""
    try:
        r = await _get_client()
        return bool(await r.sismember(settings.halts_redis_set, slug))
    except Exception:
        log.exception("halts.is_halted_failed slug=%s", slug)
        # Fail open: if we can't read, don't claim it's halted
        return False


async def actuate(slug: str, reason: str, *, dry_run: bool | None = None) -> bool:
    """Add `slug` to `strategy:halts`. Returns True iff an actual halt
    write happened (i.e. enforcement is on AND the strategy wasn't
    already halted).

    `dry_run` overrides settings.enforce_halts when explicitly passed.
    """
    do_enforce = (not dry_run) if dry_run is not None else settings.enforce_halts

    if not do_enforce:
        log.info(
            "research_loop.would_halt slug=%s reason=%s (enforce=False)",
            slug, reason,
        )
        return False

    try:
        r = await _get_client()
        added = await r.sadd(settings.halts_redis_set, slug)
        if added:
            log.warning(
                "research_loop.halted slug=%s reason=%s",
                slug, reason,
            )
            return True
        log.info(
            "research_loop.already_halted slug=%s reason=%s",
            slug, reason,
        )
        return False
    except Exception:
        log.exception("halts.actuate_failed slug=%s", slug)
        return False


__all__ = ["actuate", "close", "is_halted"]
