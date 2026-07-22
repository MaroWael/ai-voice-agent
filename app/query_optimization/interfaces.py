"""
Query Optimization — Interface

Defines the abstract contract for query optimizers.
Higher layers depend on QueryOptimizer, never on concrete implementations.
"""

from abc import ABC, abstractmethod


class QueryOptimizer(ABC):
    """
    Abstract base for query optimizers used by the RAG pipeline.

    Implementations receive a raw user query and return an optimized version
    suitable for semantic retrieval. They MUST preserve the user's language —
    Arabic input stays Arabic; optimization only improves token coverage.

    The optimized query is used exclusively by RetrievalService.
    PromptBuilder always receives the original, unmodified question.
    """

    @abstractmethod
    async def optimize(self, query: str) -> str:
        """
        Return an optimized version of *query* for semantic retrieval.

        Args:
            query: Raw user question, potentially in Arabic, English, or mixed.

        Returns:
            An optimized query string. If no improvement is possible (empty
            input, whitespace-only), returns an empty string.
        """
