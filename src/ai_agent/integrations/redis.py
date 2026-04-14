"""Redis client for sessions, rate limiting, and caching."""

from __future__ import annotations

import json
from typing import Any

import redis.asyncio as aioredis
import structlog

logger = structlog.get_logger()

SESSION_PREFIX = "copilot:session:"
RATE_LIMIT_PREFIX = "copilot:ratelimit:"


class RedisClient:
    """Thin wrapper around aioredis for Copilot-specific operations."""

    def __init__(self, connection: aioredis.Redis | None = None, url: str = "") -> None:
        if connection:
            self._redis = connection
        else:
            self._redis = aioredis.from_url(url, decode_responses=True)

    async def close(self) -> None:
        await self._redis.aclose()

    # --- Sessions ---

    async def set_session(self, session_id: str, data: dict[str, Any], ttl: int = 86400) -> None:
        key = f"{SESSION_PREFIX}{session_id}"
        await self._redis.set(key, json.dumps(data), ex=ttl)

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        key = f"{SESSION_PREFIX}{session_id}"
        raw = await self._redis.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    async def delete_session(self, session_id: str) -> None:
        key = f"{SESSION_PREFIX}{session_id}"
        await self._redis.delete(key)

    # --- Rate Limiting (fixed-window counter) ---

    async def check_rate_limit(self, user_id: str, limit: int, window: int) -> bool:
        key = f"{RATE_LIMIT_PREFIX}{user_id}"
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.incr(key)
            pipe.expire(key, window)
            results = await pipe.execute()
        count = results[0]
        return count <= limit

    # --- Health ---

    async def ping(self) -> bool:
        return await self._redis.ping()
