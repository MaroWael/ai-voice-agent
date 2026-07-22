"""
RAG Service

Coordinates the full RAG quality pipeline:

  1. QueryOptimizer     — produces a retrieval-optimized query
  2. RetrievalService   — retrieves relevant documents using the optimized query
  3. UnknownAnswerDetector — evaluates whether context is sufficient
  4. ContextBuilder     — formats retrieved documents into a context block
  5. PromptBuilder      — constructs the final LLM prompt with the ORIGINAL question
  6. LLMProvider        — generates the natural-language answer

The original user question is always forwarded to PromptBuilder unchanged.
The optimized query is used only by RetrievalService.
"""

import logging

from app.query_optimization.interfaces import QueryOptimizer
from app.rag.builders.context_builder import ContextBuilder
from app.rag.builders.prompt_builder import PromptBuilder
from app.rag.models.response import RagResponse
from app.rag.models.status import RagStatus
from app.rag.providers.base import LLMProvider
from app.retrieval.services.retrieval_service import RetrievalService
from app.unknown_detection.interfaces import UnknownAnswerDetector

logger = logging.getLogger(__name__)

# Standard reply returned when retrieved documents do not provide sufficient
# evidence to answer the user's question.
# Defined here (application layer) because message content is a business concern,
# not a configuration concern.
_INSUFFICIENT_CONTEXT_MSG = (
    "I don't have enough information to answer this question."
)


class RagService:
    """
    Orchestrates the RAG quality pipeline.

    All dependencies are injected at construction time.
    RagService never instantiates its own collaborators.
    """

    def __init__(
        self,
        retrieval_service: RetrievalService,
        context_builder: ContextBuilder,
        prompt_builder: PromptBuilder,
        llm_provider: LLMProvider,
        query_optimizer: QueryOptimizer,
        unknown_detector: UnknownAnswerDetector,
    ) -> None:
        self._retrieval_service = retrieval_service
        self._context_builder = context_builder
        self._prompt_builder = prompt_builder
        self._llm_provider = llm_provider
        self._query_optimizer = query_optimizer
        self._unknown_detector = unknown_detector

    async def initialize(self) -> None:
        """Initializes owned resources (e.g. LLMProvider)."""
        await self._llm_provider.initialize()

    async def close(self) -> None:
        """Closes/releases owned resources."""
        await self._llm_provider.close()

    async def answer(self, question: str, top_k: int | None = None) -> RagResponse:
        """
        Execute the RAG quality pipeline and return a structured response.

        Args:
            question: The original user question. Always used by PromptBuilder.
            top_k:    Optional override for the number of documents to retrieve.

        Returns:
            RagResponse with status SUCCESS or INSUFFICIENT_CONTEXT.
        """
        logger.info("RAG pipeline started for question: %r", question)

        # ── Step 1: Query Optimization ────────────────────────────────────────
        optimized_query = await self._query_optimizer.optimize(question)
        logger.debug("Original query: %r", question)
        logger.debug("Optimized query: %r", optimized_query)

        # Fall back to the original question if optimization produces empty output
        # (e.g. a query composed entirely of filler phrases).
        retrieval_query = optimized_query if optimized_query else question

        # ── Step 2: Retrieval ─────────────────────────────────────────────────
        if top_k is not None:
            retrieved_docs = await self._retrieval_service.retrieve(
                retrieval_query, top_k=top_k
            )
        else:
            retrieved_docs = await self._retrieval_service.retrieve(retrieval_query)

        logger.debug(
            "Retrieved %d document(s). Scores: %s",
            len(retrieved_docs),
            [round(r.score, 4) for r in retrieved_docs],
        )

        # ── Step 3: Unknown Answer Detection ─────────────────────────────────
        detection = await self._unknown_detector.evaluate(question, retrieved_docs)
        logger.debug(
            "Detection result: has_context=%s, reason=%s, "
            "top_score=%.4f, avg_score=%.4f",
            detection.has_context,
            detection.reason.value,
            detection.top_score,
            detection.average_score,
        )

        if not detection.has_context:
            logger.info(
                "Insufficient context detected (reason=%s). "
                "Returning fallback response without LLM call.",
                detection.reason.value,
            )
            return RagResponse(
                answer=_INSUFFICIENT_CONTEXT_MSG,
                prompt="",
                retrieved_documents=retrieved_docs,
                status=RagStatus.INSUFFICIENT_CONTEXT,
            )

        # ── Step 4: Context + Prompt (original question) ─────────────────────
        context = self._context_builder.build_context(retrieved_docs)
        # PromptBuilder always receives the ORIGINAL question, not the optimized query.
        prompt = self._prompt_builder.build_prompt(question, context)

        # ── Step 5: LLM Generation ────────────────────────────────────────────
        answer = await self._llm_provider.generate(prompt)

        logger.info("RAG pipeline completed successfully.")
        logger.debug("Final response path: SUCCESS")

        return RagResponse(
            answer=answer,
            prompt=prompt,
            retrieved_documents=retrieved_docs,
            status=RagStatus.SUCCESS,
        )
