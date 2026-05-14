"""Tests for the Redis param registry."""
from __future__ import annotations

import pytest

from research_loop.param_registry import (
    AUDIT_STREAM,
    PARAM_KEY_PREFIX,
    get_all_params,
    get_param,
    set_param,
)


class _FakeRedis:
    def __init__(self, hash_data: dict[str, dict[str, str]] | None = None):
        self.hashes: dict[str, dict[str, str]] = hash_data or {}
        self.xadds: list[tuple[str, dict]] = []
        self.raise_on_hget = False
        self.raise_on_hset = False

    async def hget(self, key: str, field: str) -> str | None:
        if self.raise_on_hget:
            raise RuntimeError("simulated")
        return self.hashes.get(key, {}).get(field)

    async def hgetall(self, key: str) -> dict[str, str]:
        return dict(self.hashes.get(key, {}))

    async def hset(self, key: str, name: str, value: str):
        if self.raise_on_hset:
            raise RuntimeError("simulated")
        self.hashes.setdefault(key, {})[name] = value
        return 1

    async def xadd(self, stream: str, fields: dict, maxlen: int = 0, approximate: bool = False):
        self.xadds.append((stream, fields))
        return "0-0"


@pytest.mark.asyncio
async def test_get_param_returns_default_when_missing() -> None:
    r = _FakeRedis()
    v = await get_param(r, "poly-sell-wings", "velocity_threshold", default=0.30)
    assert v == 0.30


@pytest.mark.asyncio
async def test_get_param_returns_redis_value() -> None:
    r = _FakeRedis({f"{PARAM_KEY_PREFIX}poly-sell-wings": {"velocity_threshold": "0.42"}})
    v = await get_param(r, "poly-sell-wings", "velocity_threshold", default=0.30)
    assert v == 0.42


@pytest.mark.asyncio
async def test_get_param_malformed_falls_back_to_default() -> None:
    r = _FakeRedis({f"{PARAM_KEY_PREFIX}poly-sell-wings": {"velocity_threshold": "not-a-number"}})
    v = await get_param(r, "poly-sell-wings", "velocity_threshold", default=0.30)
    assert v == 0.30


@pytest.mark.asyncio
async def test_get_param_redis_error_falls_back() -> None:
    r = _FakeRedis()
    r.raise_on_hget = True
    v = await get_param(r, "poly-sell-wings", "velocity_threshold", default=0.30)
    assert v == 0.30


@pytest.mark.asyncio
async def test_get_all_params() -> None:
    r = _FakeRedis({
        f"{PARAM_KEY_PREFIX}poly-sell-wings": {
            "velocity_threshold": "0.42",
            "cooldown_sec": "300",
            "bad_key": "not-a-number",
        },
    })
    out = await get_all_params(r, "poly-sell-wings")
    assert out == {"velocity_threshold": 0.42, "cooldown_sec": 300.0}


@pytest.mark.asyncio
async def test_set_param_writes_value_and_audits() -> None:
    r = _FakeRedis()
    await set_param(r, "poly-sell-wings", "velocity_threshold", 0.42,
                    old_value=0.30, reason="auto-tune nudge")
    # Value persisted
    v = await get_param(r, "poly-sell-wings", "velocity_threshold", default=0.0)
    assert v == 0.42
    # Audit XADD'd
    assert len(r.xadds) == 1
    stream, fields = r.xadds[0]
    assert stream == AUDIT_STREAM
    assert "data" in fields
    import json
    entry = json.loads(fields["data"])
    assert entry["strategy"] == "poly-sell-wings"
    assert entry["name"] == "velocity_threshold"
    assert float(entry["old"]) == 0.30
    assert float(entry["new"]) == 0.42
    assert "auto-tune" in entry["reason"]


@pytest.mark.asyncio
async def test_set_param_resilient_to_redis_error() -> None:
    """If HSET fails, we log + return without raising."""
    r = _FakeRedis()
    r.raise_on_hset = True
    # Should not raise
    await set_param(r, "poly-sell-wings", "velocity_threshold", 0.42)
