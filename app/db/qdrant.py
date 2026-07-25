import logging

from qdrant_client import AsyncQdrantClient

from app.config.settings import settings

logger = logging.getLogger(__name__)

_client: AsyncQdrantClient | None = None


def get_qdrant() -> AsyncQdrantClient:
    """Return the shared async Qdrant client."""
    global _client
    if _client is None:
        if settings.QDRANT_HOST == ":memory:":
            _client = AsyncQdrantClient(location=":memory:")
        else:
            _client = AsyncQdrantClient(url=settings.qdrant_url)
    return _client


async def check_qdrant() -> None:
    """Verify that Qdrant is reachable. Raises on failure — never swallows errors."""
    client = get_qdrant()
    await client.get_collections()
    logger.info("Qdrant connection OK")
