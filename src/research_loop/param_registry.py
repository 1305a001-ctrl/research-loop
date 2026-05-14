"""Live param registry — Redis-backed.

Strategies read their tunable params from Redis hashes keyed by slug:
    HGETALL params:<strategy_slug>

The auto-tuner writes nudged values here. On each strategy emit cycle,
the strategy reads the current value, falls back to its hardcoded default
if missing. This decouples auto-tuning from strategy deploys — no rebuild
required to apply a new value.

Each param write is also XADDed to `params:audit` so we have a permanent
log of every nudge (timestamp, strategy, name, old, new, reason).
"""
from __future__ import annotations

import logging
import time
from typing import Any

log = logging.getLogger(__name__)


PARAM_KEY_PREFIX = "params:"
AUDIT_STREAM = "params:audit"
AUDIT_MAXLEN = 100_000


def _param_key(strategy_slug: str) -> str:
    return f"{PARAM_KEY_PREFIX}{strategy_slug}"


async def get_param(
    redis_client: Any,
    strategy_slug: str,
    name: str,
    default: float,
) -> float:
    """Read a param from Redis, falling back to default if missing or malformed."""
    try:
        raw = await redis_client.hget(_param_key(strategy_slug), name)
    except Exception:
        log.warning("param_registry.get_failed strategy=%s name=%s", strategy_slug, name)
        return default
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        log.warning(
            "param_registry.malformed strategy=%s name=%s raw=%s",
            strategy_slug, name, raw,
        )
        return default


async def get_all_params(
    redis_client: Any,
    strategy_slug: str,
) -> dict[str, float]:
    """Read all params for a strategy. Returns dict; missing keys absent."""
    try:
        raw = await redis_client.hgetall(_param_key(strategy_slug))
    except Exception:
        return {}
    out: dict[str, float] = {}
    for k, v in (raw or {}).items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


async def set_param(
    redis_client: Any,
    strategy_slug: str,
    name: str,
    value: float,
    *,
    old_value: float | None = None,
    reason: str = "",
) -> None:
    """Write a param value + audit log entry.

    Caller passes old_value for the audit log (cheap — they just read it).
    `reason` is the auto-tuner's decision text (or 'manual' / 'reset' / etc.).
    """
    try:
        await redis_client.hset(_param_key(strategy_slug), name, str(value))
    except Exception:
        log.exception(
            "param_registry.set_failed strategy=%s name=%s value=%s",
            strategy_slug, name, value,
        )
        return
    audit = {
        "data": _format_audit({
            "ts_unix": time.time(),
            "strategy": strategy_slug,
            "name": name,
            "old": old_value if old_value is not None else "",
            "new": value,
            "reason": reason,
        }),
    }
    try:
        await redis_client.xadd(AUDIT_STREAM, audit, maxlen=AUDIT_MAXLEN, approximate=True)
    except Exception:
        log.debug("param_registry.audit_xadd_failed", exc_info=True)


def _format_audit(entry: dict[str, Any]) -> str:
    """Pure: serialize an audit entry as JSON-ish string for XADD payload."""
    import json
    return json.dumps(entry, default=str)


__all__ = [
    "AUDIT_STREAM",
    "PARAM_KEY_PREFIX",
    "get_all_params",
    "get_param",
    "set_param",
]
