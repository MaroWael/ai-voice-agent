import asyncio
import logging
import time
from dataclasses import dataclass

from app.embeddings.services.embedding_service import EmbeddingService
from app.retrieval.models.search_result import SearchResult
from app.vector_store.providers.vector_store_provider import VectorStoreProvider

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalTiming:
    """
    Per-request timing breakdown for a single retrieve_timed() call.

    Fields:
        embedding_time: Seconds spent generating the query embedding.
        search_time:    Seconds spent querying the vector store.
        total_time:     embedding_time + search_time.
    """

    embedding_time: float
    search_time: float

    @property
    def total_time(self) -> float:
        return self.embedding_time + self.search_time


class RetrievalService:
    """
    Orchestrates semantic retrieval by coordinating EmbeddingService and VectorStoreProvider.

    Workflow:
    Question → EmbeddingService (Query Vector) → VectorStoreProvider (Search) → list[SearchResult]

    Constructor injection only.
    RetrievalService is responsible only for orchestration.

    candidate_k:
    When provided, the vector store is queried for a larger candidate pool
    (candidate_k results). Today the first top_k results are returned directly.
    When a reranker is added in a future milestone, it will be applied to
    the candidate pool before slicing to top_k, without changing this API.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        qdrant_provider: VectorStoreProvider,
    ) -> None:
        self._embedding_service = embedding_service
        self._qdrant_provider = qdrant_provider

    async def retrieve(
        self,
        question: str,
        top_k: int = 5,
        candidate_k: int | None = None,
    ) -> list[SearchResult]:
        """
        Retrieve top-k semantically relevant KnowledgeDocuments for a user question.

        Args:
            question:    User question string.
            top_k:       Number of final results to return (default: 5).
            candidate_k: Size of the initial candidate pool fetched from the vector
                         store. When None, defaults to top_k. Intended for future
                         reranking: fetch candidate_k results, rerank, return top_k.
                         Today the first top_k results from the candidate pool are
                         returned directly.

        Returns:
            List of SearchResult objects ordered by relevance score descending,
            with rank assigned (1-based).
        """
        results, _ = await self.retrieve_timed(question, top_k=top_k, candidate_k=candidate_k)
        return results

    async def retrieve_timed(
        self,
        question: str,
        top_k: int = 5,
        candidate_k: int | None = None,
    ) -> tuple[list[SearchResult], RetrievalTiming]:
        """
        Retrieve top-k results and return per-stage timing alongside them.

        Identical behavior to retrieve(), but additionally measures and returns:
          - embedding_time: time spent generating the query vector.
          - search_time:    time spent querying the vector store.

        Intended for benchmarking and diagnostics. Business logic is identical
        to retrieve() — no retrieval behavior is changed.

        Args:
            question:    User question string.
            top_k:       Number of final results to return (default: 5).
            candidate_k: Size of the initial candidate pool (see retrieve()).

        Returns:
            Tuple of (list[SearchResult], RetrievalTiming).
        """
        if not question or not question.strip():
            timing = RetrievalTiming(embedding_time=0.0, search_time=0.0)
            return [], timing

        search_k = candidate_k or top_k

        logger.info(
            "Executing semantic retrieval for question: '%s' (top_k=%d, search_k=%d)",
            question,
            top_k,
            search_k,
        )

        # ── Embed ────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        query_vector = await self._embedding_service.embed_query(question)
        embedding_time = time.perf_counter() - t0

        # ── Search ───────────────────────────────────────────────────────────
        t1 = time.perf_counter()
        candidates = await self._qdrant_provider.search(vector=query_vector, top_k=search_k)
        search_time = time.perf_counter() - t1

        logger.info(
            "Retrieved %d candidate(s) for question: '%s' "
            "(embed=%.3fs, search=%.3fs)",
            len(candidates),
            question,
            embedding_time,
            search_time,
        )

        # No reranker yet — return the top_k highest-scoring candidates directly.
        # Assign 1-based rank to each result.
        results = [
            result.model_copy(update={"rank": rank})
            for rank, result in enumerate(candidates[:top_k], start=1)
        ]

        timing = RetrievalTiming(embedding_time=embedding_time, search_time=search_time)
        return results, timing
