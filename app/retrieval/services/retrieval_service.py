import logging
import time
from dataclasses import dataclass

from app.config.settings import settings
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
    Orchestrates semantic dense retrieval by coordinating EmbeddingService and VectorStoreProvider.

    Workflow:
        Question -> EmbeddingService (Query Vector) -> VectorStoreProvider (Search) -> list[SearchResult]

    Strictly dense similarity retrieval using BAAI/bge-m3 and Qdrant.
    No score filtering is performed inside RetrievalService — Unknown Detection
    is responsible for evaluating context sufficiency.
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
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """
        Retrieve top-k semantically relevant KnowledgeDocuments for a user question.

        Args:
            question: User question string.
            top_k: Optional override for top-k count. Defaults to settings.RAG_TOP_K.

        Returns:
            List of SearchResult objects ordered by relevance score descending.
        """
        results, _ = await self.retrieve_timed(question, top_k=top_k)
        return results

    async def retrieve_timed(
        self,
        question: str,
        top_k: int | None = None,
    ) -> tuple[list[SearchResult], RetrievalTiming]:
        """
        Retrieve top-k results and return per-stage timing alongside them.
        """
        if not question or not question.strip():
            timing = RetrievalTiming(embedding_time=0.0, search_time=0.0)
            return [], timing

        effective_top_k = top_k if top_k is not None else settings.RAG_TOP_K

        logger.info(
            "Executing semantic dense retrieval for question: %r (top_k=%d)",
            question,
            effective_top_k,
        )

        # ── Embed ────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        query_vector = await self._embedding_service.embed_query(question)
        embedding_time = time.perf_counter() - t0

        # ── Search ───────────────────────────────────────────────────────────
        t1 = time.perf_counter()
        candidates = await self._qdrant_provider.search(vector=query_vector, top_k=effective_top_k)
        search_time = time.perf_counter() - t1

        logger.info(
            "Retrieved %d document(s) (embed=%.3fs, search=%.3fs)",
            len(candidates),
            embedding_time,
            search_time,
        )

        # Assign 1-based rank to each result
        results = [
            result.model_copy(update={"rank": rank})
            for rank, result in enumerate(candidates, start=1)
        ]

        timing = RetrievalTiming(embedding_time=embedding_time, search_time=search_time)
        return results, timing
