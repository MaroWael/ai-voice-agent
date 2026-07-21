"""
Vector store factory.

Centralizes creation of QdrantProvider so that no call site needs
to import or configure the Qdrant client directly.
"""

from app.db.qdrant import get_qdrant
from app.vector_store.providers.qdrant_provider import QdrantProvider


def build_qdrant_provider() -> QdrantProvider:
    """Return a QdrantProvider wrapping the shared AsyncQdrantClient."""
    return QdrantProvider(get_qdrant())
