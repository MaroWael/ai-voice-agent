"""
Query Normalization — Interface

Defines the abstract contract for query normalizers.
Higher layers depend on QueryNormalizer, never on concrete implementations.
"""

from abc import ABC, abstractmethod


class QueryNormalizer(ABC):
    """
    Abstract base for query normalizers used by the RAG pipeline.

    Implementations receive a raw user query and return a clean, normalized
    version suitable for semantic retrieval.

    Language preservation rule:
    User language is strictly preserved (Arabic stays Arabic, English stays English).
    No keyword expansion or translation is performed.
    """

    @abstractmethod
    async def normalize(self, query: str) -> str:
        """
        Return a normalized version of *query* for semantic retrieval.

        Args:
            query: Raw user question, potentially in Arabic, English, or mixed.

        Returns:
            A clean normalized query string.
        """
        pass

    async def optimize(self, query: str) -> str:
        """Backwards-compatible alias for normalize."""
        return await self.normalize(query)


# Backwards compatibility alias for existing callers
QueryOptimizer = QueryNormalizer

