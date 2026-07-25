"""
RagLanguageModel

Adapter that implements the voice-layer LanguageModel ABC by delegating
all generation work to the existing RagService pipeline.

Responsibilities:
  - Accept a Transcription from the voice Orchestrator.
  - Extract the user text and pass it to RagService.answer().
  - Map the RagResponse back into the AIResponse contract expected by the
    Orchestrator and the HTTP/WebSocket response payloads.

The voice layer stays decoupled from Qdrant, Groq, embeddings, and retrieval.
"""

import logging
from typing import Optional

from llm.base import LanguageModel
from llm.models import AIResponse
from input.models.transcription import Transcription

logger = logging.getLogger(__name__)


class RagLanguageModel(LanguageModel):
    """
    Voice-layer LanguageModel that routes every query through the RAG pipeline.

    Lifecycle:
        rag_llm = RagLanguageModel()
        await rag_llm.initialize()   # boots RagService + GroqProvider once
        ...
        response = await rag_llm.generate(transcription)
        ...
        await rag_llm.close()        # closes GroqProvider HTTP client
    """

    def __init__(self, rag_service=None) -> None:
        """
        Args:
            rag_service: A pre-built RagService instance.
                         If None, one is built from the factory at initialize().
        """
        self._rag_service = rag_service
        self._initialized = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """
        Build (if not injected) and initialize the RagService once.
        Safe to call multiple times — subsequent calls are no-ops.
        """
        if self._initialized:
            return

        if self._rag_service is None:
            from app.factories.rag import build_rag_service
            self._rag_service = build_rag_service()

        await self._rag_service.initialize()
        self._initialized = True
        logger.info("RagLanguageModel initialized — RAG pipeline ready.")

    async def close(self) -> None:
        """Close and release the RagService's HTTP client pool."""
        if self._rag_service is not None:
            await self._rag_service.close()
            logger.info("RagLanguageModel closed — RAG pipeline resources released.")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    async def generate(self, transcription: Transcription) -> AIResponse:
        """
        Passes the transcribed user text through the RAG pipeline and returns
        a fully populated AIResponse compatible with the Orchestrator contract.

        Mapping:
            RagStatus.SUCCESS              → action="rag"
            RagStatus.INSUFFICIENT_CONTEXT → action="rag"  (refusal answer)
            Any exception                  → re-raised

        Args:
            transcription: Output of FasterWhisperSTT containing .text and .language.

        Returns:
            AIResponse with action, department, reason, message fields populated.
        """
        if self._rag_service is None or not self._initialized:
            raise RuntimeError(
                "RagLanguageModel not initialized. Call initialize() first."
            )

        question = transcription.text
        logger.info("RagLanguageModel.generate() — question: %r", question)

        from app.rag.models.status import RagStatus
        from app.config.settings import settings

        rag_response = await self._rag_service.answer(question, debug=True)

        debug = rag_response.debug_info
        min_score = settings.UNKNOWN_DETECTOR_MIN_SCORE
        mean_thresh = settings.UNKNOWN_DETECTOR_MEAN_THRESHOLD
        min_results = settings.UNKNOWN_DETECTOR_MIN_RESULTS

        top_results_str = []
        if debug and debug.retrieved_chunks:
            for idx, chunk in enumerate(debug.retrieved_chunks[:5], 1):
                product = chunk.product_name or "Unknown Product"
                section = chunk.section or "General"
                top_results_str.append(f"  {idx}. {product} — {section} (score={chunk.score:.4f})")
        else:
            top_results_str.append("  (No documents retrieved)")

        top_score = debug.retrieval_scores[0] if (debug and debug.retrieval_scores) else 0.0
        avg_score = (sum(debug.retrieval_scores) / len(debug.retrieval_scores)) if (debug and debug.retrieval_scores) else 0.0
        reason = debug.detection_reason if debug else rag_response.status.value

        if rag_response.status == RagStatus.SUCCESS:
            decision_str = f"Accepted (SUFFICIENT_CONTEXT) — Top score ({top_score:.4f}) >= threshold ({min_score})"
        else:
            if reason == "low_top_score":
                explanation = f"Top score ({top_score:.4f}) < min_score threshold ({min_score})"
            elif reason == "low_mean_score":
                explanation = f"Mean score ({avg_score:.4f}) < mean_threshold ({mean_thresh})"
            elif reason == "insufficient_results":
                explanation = f"Retrieved doc count ({len(debug.retrieved_chunks if debug else [])}) < min_results ({min_results})"
            elif reason == "empty_results":
                explanation = "No documents retrieved from vector store"
            else:
                explanation = f"Reason: {reason}"
            decision_str = f"Rejected because {explanation}"

        diag_log = (
            "\n==================== RAG DIAGNOSTICS ====================\n"
            f"Normalized Query:\n\"{debug.normalized_query if debug else question}\"\n\n"
            f"Top Results:\n" + "\n".join(top_results_str) + "\n\n"
            f"Threshold:\nMin Score: {min_score} | Mean Threshold: {mean_thresh} | Min Results: {min_results}\n\n"
            f"Decision:\n{decision_str}\n"
            "========================================================="
        )
        logger.info(diag_log)

        logger.info(
            "RagLanguageModel.generate() — status: %s, answer length: %d",
            rag_response.status.value,
            len(rag_response.answer),
        )

        return AIResponse(
            action="rag",
            department=None,
            reason=rag_response.status.value,
            message=rag_response.answer,
            language=transcription.language,
        )
