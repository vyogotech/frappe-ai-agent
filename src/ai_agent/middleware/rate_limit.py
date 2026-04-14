"""Per-user rate limiting backed by Redis."""

from __future__ import annotations

from ai_agent.integrations.redis import RedisClient


class RateLimiter:
    """Simple sliding-window rate limiter."""

    def __init__(self, redis: RedisClient, limit: int, window_seconds: int) -> None:
        self._redis = redis
        self._limit = limit
        self._window = window_seconds

    async def is_allowed(self, user_id: str) -> bool:
        return await self._redis.check_rate_limit(user_id, limit=self._limit, window=self._window)
