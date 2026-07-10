import logging

import redis.asyncio as aioredis

from app.config.settings import settings

logger = logging.getLogger(__name__)

# One connection pool shared for the lifetime of the process.
# decode_responses=True ensures all values are returned as str, never raw bytes.
_client: aioredis.Redis = aioredis.from_url(
    settings.redis_url,
    decode_responses=True,
)


def get_redis() -> aioredis.Redis:
    """Return the shared async Redis client."""
    return _client


async def check_redis() -> None:
    """Verify that Redis is reachable. Raises on failure — never swallows errors."""
    await _client.ping()
    logger.info("Redis connection OK")
