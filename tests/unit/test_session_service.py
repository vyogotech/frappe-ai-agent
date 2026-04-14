import pytest
from fakeredis import aioredis as fakeredis_aio

from copilot_agent.integrations.redis import RedisClient
from copilot_agent.services.session import SessionService


@pytest.fixture
async def session_service():
    fake = fakeredis_aio.FakeRedis()
    redis = RedisClient(connection=fake)
    svc = SessionService(redis=redis)
    yield svc
    await fake.aclose()


class TestSessionService:
    async def test_create_session(self, session_service):
        session = await session_service.create(user="admin@test.com", site="site1.local")
        assert session["user"] == "admin@test.com"
        assert "session_id" in session

    async def test_get_session(self, session_service):
        session = await session_service.create(user="admin@test.com", site="site1.local")
        retrieved = await session_service.get(session["session_id"])
        assert retrieved["user"] == "admin@test.com"

    async def test_get_missing_session(self, session_service):
        result = await session_service.get("nonexistent")
        assert result is None

    async def test_invalidate(self, session_service):
        session = await session_service.create(user="admin@test.com", site="site1.local")
        await session_service.invalidate(session["session_id"])
        result = await session_service.get(session["session_id"])
        assert result is None
