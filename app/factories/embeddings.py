"""
Embedding factories.

Centralizes creation of SentenceTransformerProvider and EmbeddingService
so that no call site needs to know the construction details.
"""

from app.config.settings import settings
from app.embeddings.providers.sentence_transformer_provider import SentenceTransformerProvider
from app.embeddings.services.embedding_service import EmbeddingService


def build_sentence_transformer_provider() -> SentenceTransformerProvider:
    """Return a SentenceTransformerProvider configured from Settings."""
    return SentenceTransformerProvider(
        model_name=settings.EMBEDDING_MODEL,
        batch_size=settings.EMBEDDING_BATCH_SIZE,
    )


def build_embedding_service() -> EmbeddingService:
    """Return a fully wired EmbeddingService backed by SentenceTransformerProvider."""
    provider = build_sentence_transformer_provider()
    return EmbeddingService(provider)
