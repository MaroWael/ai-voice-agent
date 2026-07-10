import logging

from qdrant_client import AsyncQdrantClient

from app.config.settings import settings

logger = logging.getLogger(__name__)

# One client shared for the lifetime of the process.
_client: AsyncQdrantClient = AsyncQdrantClient(url=settings.qdrant_url)


def get_qdrant() -> AsyncQdrantClient:
    """Return the shared async Qdrant client."""
    return _client


async def check_qdrant() -> None:
    """Verify that Qdrant is reachable. Raises on failure — never swallows errors."""
    await _client.get_collections()
    logger.info("Qdrant connection OK")
