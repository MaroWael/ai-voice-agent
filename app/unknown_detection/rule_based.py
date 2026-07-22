"""
Unknown Answer Detection — Rule-Based Detector

Lightweight, zero-latency gate based on similarity score signals.
No LLM, no cross-encoder, no network calls.

Signal chain (evaluated in order, short-circuits on first failure):
  1. Empty result list            → EMPTY_RESULTS
  2. Result count < min_results   → INSUFFICIENT_RESULTS
  3. Top-1 score < min_score      → LOW_TOP_SCORE
  4. Mean score < mean_threshold  → LOW_MEAN_SCORE
  5. All pass                     → SUFFICIENT_CONTEXT

All thresholds are constructor-injected and never hardcoded here.
"""

import logging

from app.retrieval.models.search_result import SearchResult
from app.unknown_detection.interfaces import (
    DetectionReason,
    DetectionResult,
    UnknownAnswerDetector,
)

logger = logging.getLogger(__name__)


class RuleBasedUnknownDetector(UnknownAnswerDetector):
    """
    Signal-based retrieval quality gate.

    Args:
        min_score:        Minimum acceptable top-1 similarity score.
        min_results:      Minimum number of retrieved documents required.
        mean_threshold:   Minimum acceptable mean score across retrieved docs.
    """

    def __init__(
        self,
        min_score: float,
        min_results: int,
        mean_threshold: float,
    ) -> None:
        self._min_score = min_score
        self._min_results = min_results
        self._mean_threshold = mean_threshold

    async def evaluate(
        self,
        query: str,
        results: list[SearchResult],
    ) -> DetectionResult:
        """
        Evaluate whether *results* provide sufficient context for *query*.

        The *query* parameter is accepted for interface compliance and future
        query-aware extensions, but is not used by this implementation.
        """
        top_score, avg_score = self._compute_scores(results)

        logger.debug(
            "UnknownDetector evaluating: count=%d, top_score=%.4f, avg_score=%.4f "
            "(thresholds: min_score=%.4f, min_results=%d, mean=%.4f)",
            len(results),
            top_score,
            avg_score,
            self._min_score,
            self._min_results,
            self._mean_threshold,
        )

        # Signal chain — short-circuit on first failure.
        if not results:
            return self._reject(DetectionReason.EMPTY_RESULTS, top_score, avg_score)

        if len(results) < self._min_results:
            return self._reject(DetectionReason.INSUFFICIENT_RESULTS, top_score, avg_score)

        if top_score < self._min_score:
            return self._reject(DetectionReason.LOW_TOP_SCORE, top_score, avg_score)

        if avg_score < self._mean_threshold:
            return self._reject(DetectionReason.LOW_MEAN_SCORE, top_score, avg_score)

        result = DetectionResult(
            has_context=True,
            reason=DetectionReason.SUFFICIENT_CONTEXT,
            top_score=top_score,
            average_score=avg_score,
        )
        logger.debug("UnknownDetector decision: SUFFICIENT (reason=%s)", result.reason.value)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_scores(results: list[SearchResult]) -> tuple[float, float]:
        """Return (top_score, average_score). Both are 0.0 for empty lists."""
        if not results:
            return 0.0, 0.0
        scores = [r.score for r in results]
        return scores[0], sum(scores) / len(scores)

    @staticmethod
    def _reject(reason: DetectionReason, top_score: float, avg_score: float) -> DetectionResult:
        result = DetectionResult(
            has_context=False,
            reason=reason,
            top_score=top_score,
            average_score=avg_score,
        )
        logger.debug("UnknownDetector decision: REJECT (reason=%s)", reason.value)
        return result
