"""
RAG Service

Coordinates the full RAG pipeline:
  1. QueryNormalizer       — produces a clean, normalized query
  2. TranslationService    — optionally translates query if enabled
  3. RetrievalService      — retrieves relevant documents using dense vector search
  4. UnknownAnswerDetector — evaluates whether context similarity is sufficient
  5. ContextBuilder        — formats retrieved documents into a context block
  6. PromptBuilder         — constructs LLM prompt with original question & language mandate
  7. LLMProvider           — generates the natural-language answer

Forwarding rules:
  - Original user question is always passed to PromptBuilder and UnknownDetector.
  - Normalized / translated query is used exclusively by RetrievalService.
"""

import logging
import time

from app.config.settings import settings
from app.query_optimization.interfaces import QueryNormalizer
from app.rag.builders.context_builder import ContextBuilder
from app.rag.builders.language_detector import detect_query_language
from app.rag.builders.prompt_builder import PromptBuilder
from app.rag.models.response import ChunkDebugInfo, RagDebugInfo, RagResponse
from app.rag.models.status import RagStatus
from app.rag.providers.base import LLMProvider
from app.retrieval.services.retrieval_service import RetrievalService
from app.translation.interfaces import TranslationService
from app.unknown_detection.interfaces import UnknownAnswerDetector

logger = logging.getLogger(__name__)


class RagService:
    """
    Orchestrates the RAG pipeline.
    All collaborators are constructor-injected.
    """

    def __init__(
        self,
        retrieval_service: RetrievalService,
        context_builder: ContextBuilder,
        prompt_builder: PromptBuilder,
        llm_provider: LLMProvider,
        query_normalizer: QueryNormalizer,
        unknown_detector: UnknownAnswerDetector,
        translation_service: TranslationService | None = None,
    ) -> None:
        self._retrieval_service = retrieval_service
        self._context_builder = context_builder
        self._prompt_builder = prompt_builder
        self._llm_provider = llm_provider
        self._query_normalizer = query_normalizer
        self._unknown_detector = unknown_detector
        self._translation_service = translation_service

    async def initialize(self) -> None:
        """Initializes owned resources (e.g. LLMProvider)."""
        await self._llm_provider.initialize()

    async def close(self) -> None:
        """Closes/releases owned resources."""
        await self._llm_provider.close()

    def _get_refusal_message(self, question: str) -> str:
        """Return a polite refusal message in the customer's language."""
        lang = detect_query_language(question)
        if lang == "ar":
            return settings.RAG_REFUSAL_MSG_AR
        return settings.RAG_REFUSAL_MSG_EN

    async def answer(
        self,
        question: str,
        top_k: int | None = None,
        debug: bool | None = None,
    ) -> RagResponse:
        """
        Execute the RAG pipeline and return a structured response.
        """
        is_debug = debug if debug is not None else settings.RAG_DEBUG
        logger.info("RAG pipeline started for question: %r (debug=%s)", question, is_debug)

        t_total_start = time.perf_counter()

        # ── Step 1: Query Normalization ───────────────────────────────────────
        t0 = time.perf_counter()
        normalized_query = await self._query_normalizer.normalize(question)
        t_norm_ms = (time.perf_counter() - t0) * 1000.0

        retrieval_query = normalized_query if normalized_query else question

        # ── Step 2: Translation (Optional abstraction layer) ──────────────────
        t_trans_start = time.perf_counter()
        if self._translation_service and settings.TRANSLATION_ENABLED:
            source_lang = detect_query_language(retrieval_query)
            retrieval_query = await self._translation_service.translate(
                retrieval_query, source_lang=source_lang, target_lang="en"
            )
        t_trans_ms = (time.perf_counter() - t_trans_start) * 1000.0

        # ── Step 3: Dense Retrieval ───────────────────────────────────────────
        t1 = time.perf_counter()
        retrieved_docs, retrieval_timing = await self._retrieval_service.retrieve_timed(
            retrieval_query, top_k=top_k
        )
        t_retrieval_ms = (time.perf_counter() - t1) * 1000.0

        # ── Step 4: Unknown Answer Detection ─────────────────────────────────
        t2 = time.perf_counter()
        detection = await self._unknown_detector.evaluate(question, retrieved_docs)
        t_detect_ms = (time.perf_counter() - t2) * 1000.0

        if not detection.has_context:
            t_total_ms = (time.perf_counter() - t_total_start) * 1000.0
            refusal_msg = self._get_refusal_message(question)

            debug_info = None
            if is_debug:
                debug_info = RagDebugInfo(
                    original_query=question,
                    normalized_query=normalized_query,
                    retrieval_scores=[r.score for r in retrieved_docs],
                    retrieved_chunks=[
                        ChunkDebugInfo(
                            id=r.document.id,
                            product_name=r.document.metadata.product_name if r.document.metadata else "",
                            section=r.document.title,
                            score=r.score,
                        )
                        for r in retrieved_docs
                    ],
                    final_context="",
                    prompt_length_chars=0,
                    latencies_ms={
                        "normalization": round(t_norm_ms, 2),
                        "translation": round(t_trans_ms, 2),
                        "retrieval": round(t_retrieval_ms, 2),
                        "detection": round(t_detect_ms, 2),
                        "generation": 0.0,
                        "total": round(t_total_ms, 2),
                    },
                    detection_reason=detection.reason.value,
                    has_context=False,
                )

            return RagResponse(
                answer=refusal_msg,
                prompt="",
                retrieved_documents=retrieved_docs,
                status=RagStatus.INSUFFICIENT_CONTEXT,
                debug_info=debug_info,
            )

        # ── Step 5: Context & Prompt ──────────────────────────────────────────
        context = self._context_builder.build_context(retrieved_docs, question=question)
        prompt = self._prompt_builder.build_prompt(question, context)

        # ── Step 6: LLM Generation ────────────────────────────────────────────
        t3 = time.perf_counter()
        answer = await self._llm_provider.generate(prompt)
        t_gen_ms = (time.perf_counter() - t3) * 1000.0

        t_total_ms = (time.perf_counter() - t_total_start) * 1000.0

        debug_info = None
        if is_debug:
            debug_info = RagDebugInfo(
                original_query=question,
                normalized_query=normalized_query,
                retrieval_scores=[r.score for r in retrieved_docs],
                retrieved_chunks=[
                    ChunkDebugInfo(
                        id=r.document.id,
                        product_name=r.document.metadata.product_name if r.document.metadata else "",
                        section=r.document.title,
                        score=r.score,
                    )
                    for r in retrieved_docs
                ],
                final_context=context,
                prompt_length_chars=len(prompt),
                latencies_ms={
                    "normalization": round(t_norm_ms, 2),
                    "translation": round(t_trans_ms, 2),
                    "retrieval": round(t_retrieval_ms, 2),
                    "detection": round(t_detect_ms, 2),
                    "generation": round(t_gen_ms, 2),
                    "total": round(t_total_ms, 2),
                },
                detection_reason=detection.reason.value,
                has_context=True,
            )

        return RagResponse(
            answer=answer,
            prompt=prompt,
            retrieved_documents=retrieved_docs,
            status=RagStatus.SUCCESS,
            debug_info=debug_info,
        )
