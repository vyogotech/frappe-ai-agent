"""Session management backed by Redis."""

from __future__ import annotations

import uuid
from typing import Any

from ai_agent.integrations.redis import RedisClient


class SessionService:
    def __init__(self, redis: RedisClient) -> None:
        self._redis = redis

    async def create(self, user: str, site: str) -> dict[str, Any]:
        session_id = str(uuid.uuid4())
        data = {"session_id": session_id, "user": user, "site": site}
        await self._redis.set_session(session_id, data)
        return data

    async def get(self, session_id: str) -> dict[str, Any] | None:
        return await self._redis.get_session(session_id)

    async def invalidate(self, session_id: str) -> None:
        await self._redis.delete_session(session_id)
