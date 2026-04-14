import pytest
from fakeredis import aioredis as fakeredis_aio

from copilot_agent.integrations.redis import RedisClient
from copilot_agent.middleware.rate_limit import RateLimiter


@pytest.fixture
async def rate_limiter():
    fake = fakeredis_aio.FakeRedis()
    client = RedisClient(connection=fake)
    limiter = RateLimiter(redis=client, limit=3, window_seconds=60)
    yield limiter
    await fake.aclose()


class TestRateLimiter:
    async def test_allows_under_limit(self, rate_limiter):
        assert await rate_limiter.is_allowed("user@test.com") is True

    async def test_blocks_over_limit(self, rate_limiter):
        for _ in range(3):
            await rate_limiter.is_allowed("user@test.com")
        assert await rate_limiter.is_allowed("user@test.com") is False

    async def test_separate_users(self, rate_limiter):
        for _ in range(3):
            await rate_limiter.is_allowed("user1@test.com")
        assert await rate_limiter.is_allowed("user2@test.com") is True
