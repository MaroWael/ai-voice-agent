"""
Unknown Answer Detection — Interfaces

Defines the public contract for unknown-answer detectors and the structured
result they return.

Design decisions:
  - DetectionResult is a frozen dataclass (not Pydantic) because it is an
    internal pipeline signal, not a public API model.
  - DetectionReason is an enum so callers can branch on specific failure modes
    without comparing strings.
  - UnknownAnswerDetector is purely about retrieval quality — it knows nothing
    about what message to show the user. That responsibility belongs to RagService.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from app.retrieval.models.search_result import SearchResult


class DetectionReason(Enum):
    """Categorical reason produced by an UnknownAnswerDetector evaluation."""

    EMPTY_RESULTS = "empty_results"
    """No documents were retrieved at all."""

    INSUFFICIENT_RESULTS = "insufficient_results"
    """Retrieved document count is below the minimum threshold."""

    LOW_TOP_SCORE = "low_top_score"
    """The highest similarity score is below the minimum acceptable score."""

    LOW_MEAN_SCORE = "low_mean_score"
    """The mean similarity score across top results is below the threshold."""

    DOMAIN_MISMATCH = "domain_mismatch"
    """The query intent asks for a domain/product (e.g. loans) absent from retrieved context."""

    SUFFICIENT_CONTEXT = "sufficient_context"
    """All signals indicate sufficient evidence to answer the query."""


@dataclass(frozen=True)
class DetectionResult:
    """
    Structured outcome of an UnknownAnswerDetector evaluation.

    Fields:
        has_context:   True when there is sufficient evidence to answer.
        reason:        Categorical reason for the decision.
        top_score:     Similarity score of the highest-ranked document (0.0 if empty).
        average_score: Mean similarity score across all evaluated documents (0.0 if empty).
    """

    has_context: bool
    reason: DetectionReason
    top_score: float
    average_score: float


class UnknownAnswerDetector(ABC):
    """
    Abstract interface for retrieval-quality gates.

    An implementation evaluates the retrieved documents and decides whether
    they contain enough evidence to answer the user's question.

    It does NOT:
      - call the LLM
      - use cross-encoders
      - decide what message to return to the user
    """

    @abstractmethod
    async def evaluate(
        self,
        query: str,
        results: list[SearchResult],
    ) -> DetectionResult:
        """
        Evaluate whether *results* provide sufficient context for *query*.

        Args:
            query:   The original user question (may be used by future
                     implementations for query-aware scoring).
            results: Ranked list of SearchResult objects from RetrievalService.

        Returns:
            DetectionResult with the decision and supporting signal values.
        """
