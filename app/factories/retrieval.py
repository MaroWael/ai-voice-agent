"""
Retrieval factory.

Composes EmbeddingService and QdrantProvider into a RetrievalService.
All callers receive a ready-to-use RetrievalService without knowing
the construction details of its dependencies.
"""

from app.factories.embeddings import build_embedding_service
from app.factories.vector_store import build_qdrant_provider
from app.retrieval.services.retrieval_service import RetrievalService


def build_retrieval_service() -> RetrievalService:
    """Return a fully wired RetrievalService."""
    return RetrievalService(
        embedding_service=build_embedding_service(),
        qdrant_provider=build_qdrant_provider(),
    )
