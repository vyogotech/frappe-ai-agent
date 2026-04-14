import pytest
from fakeredis import aioredis as fakeredis_aio

from ai_agent.integrations.redis import RedisClient


@pytest.fixture
async def redis_client():
    fake = fakeredis_aio.FakeRedis()
    client = RedisClient(connection=fake)
    yield client
    await fake.aclose()


class TestRedisClient:
    async def test_set_and_get_session(self, redis_client):
        await redis_client.set_session("sess-1", {"user": "admin@test.com"})
        data = await redis_client.get_session("sess-1")
        assert data["user"] == "admin@test.com"

    async def test_get_missing_session(self, redis_client):
        data = await redis_client.get_session("nonexistent")
        assert data is None

    async def test_delete_session(self, redis_client):
        await redis_client.set_session("sess-1", {"user": "admin@test.com"})
        await redis_client.delete_session("sess-1")
        data = await redis_client.get_session("sess-1")
        assert data is None

    async def test_rate_limit_under(self, redis_client):
        allowed = await redis_client.check_rate_limit("user@test.com", limit=5, window=60)
        assert allowed is True

    async def test_rate_limit_exceeded(self, redis_client):
        for _ in range(5):
            await redis_client.check_rate_limit("user@test.com", limit=5, window=60)
        allowed = await redis_client.check_rate_limit("user@test.com", limit=5, window=60)
        assert allowed is False
