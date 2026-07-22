"""
Arabic / Bilingual RAG Quality Suite

Runs the full RAG quality layer against a benchmark dataset and prints
structured metrics to the logger.

Usage:
    python -m evaluation.run_quality_suite

The script:
  1. Loads benchmark cases from tests/benchmarks/query_optimizer_benchmark.json
  2. For each case:
       a. Optimizes the query via QueryOptimizer
       b. Retrieves documents via RetrievalService
       c. Evaluates retrieval quality via UnknownAnswerDetector
       d. Compares the decision to expected_behavior
  3. Prints per-query results (structured, no print() calls)
  4. Prints aggregate summary metrics

Environment requirements:
  - Qdrant running and populated with knowledge base
  - .env configured with embedding model and Qdrant settings
"""

import asyncio
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

from app.factories.retrieval import build_retrieval_service
from app.query_optimization.factory import build_query_optimizer
from app.unknown_detection.factory import build_unknown_detector
from app.unknown_detection.interfaces import DetectionResult

# ---------------------------------------------------------------------------
# Logging setup — structured output, no print() anywhere in this module.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

BENCHMARK_PATH = (
    Path(__file__).parent.parent / "tests" / "benchmarks" / "query_optimizer_benchmark.json"
)


@dataclass
class CaseResult:
    """Result for a single benchmark case."""

    case_id: str
    query: str
    expected_behavior: str       # "answer" | "reject"
    optimized_query: str
    doc_count: int
    detection: DetectionResult
    predicted_behavior: str      # "answer" | "reject"
    correct: bool


@dataclass
class SuiteMetrics:
    """Aggregated metrics across all benchmark cases."""

    total: int = 0
    correct: int = 0
    retrieval_successes: int = 0   # cases where doc_count >= 1 and top_score above threshold
    false_positives: int = 0       # expected reject → predicted answer
    false_negatives: int = 0       # expected answer → predicted reject
    top_scores: list[float] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def retrieval_success_rate(self) -> float:
        return self.retrieval_successes / self.total if self.total else 0.0

    @property
    def average_top_score(self) -> float:
        return sum(self.top_scores) / len(self.top_scores) if self.top_scores else 0.0


def _load_benchmark() -> list[dict]:
    if not BENCHMARK_PATH.exists():
        raise FileNotFoundError(f"Benchmark file not found: {BENCHMARK_PATH}")
    with BENCHMARK_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


async def run_suite() -> SuiteMetrics:
    """Execute all benchmark cases and return aggregated metrics."""
    cases = _load_benchmark()
    optimizer = build_query_optimizer()
    retrieval_service = build_retrieval_service()
    detector = build_unknown_detector()

    metrics = SuiteMetrics()
    results: list[CaseResult] = []

    logger.info("=" * 60)
    logger.info("RAG Quality Suite — %d cases", len(cases))
    logger.info("=" * 60)

    for case in cases:
        case_id = case["id"]
        query = case["query"]
        expected = case["expected_behavior"]

        # ── Step 1: Query Optimization ────────────────────────────────────
        optimized = await optimizer.optimize(query)

        # ── Step 2: Retrieval ─────────────────────────────────────────────
        retrieval_query = optimized if optimized else query
        docs = await retrieval_service.retrieve(retrieval_query, top_k=5)

        # ── Step 3: Unknown Detection ─────────────────────────────────────
        detection = await detector.evaluate(query, docs)

        # ── Step 4: Compare to expected behavior ──────────────────────────
        predicted = "answer" if detection.has_context else "reject"
        correct = predicted == expected

        result = CaseResult(
            case_id=case_id,
            query=query,
            expected_behavior=expected,
            optimized_query=optimized,
            doc_count=len(docs),
            detection=detection,
            predicted_behavior=predicted,
            correct=correct,
        )
        results.append(result)

        # ── Logging ───────────────────────────────────────────────────────
        status_tag = "PASS" if correct else "FAIL"
        logger.info(
            "[%s] %s | query=%r",
            status_tag,
            case_id,
            query,
        )
        logger.info(
            "       optimized=%r | docs=%d | top_score=%.4f | avg=%.4f",
            optimized,
            len(docs),
            detection.top_score,
            detection.average_score,
        )
        logger.info(
            "       detection=%s (reason=%s) | expected=%s | predicted=%s",
            detection.has_context,
            detection.reason.value,
            expected,
            predicted,
        )

        # ── Metrics accumulation ──────────────────────────────────────────
        metrics.total += 1
        if correct:
            metrics.correct += 1
        if detection.has_context and expected == "answer":
            metrics.retrieval_successes += 1
        elif detection.has_context and expected == "reject":
            metrics.false_positives += 1
        elif not detection.has_context and expected == "answer":
            metrics.false_negatives += 1
        metrics.top_scores.append(detection.top_score)

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("Total cases              : %d", metrics.total)
    logger.info("Unknown Detection Accuracy: %.1f%%", metrics.accuracy * 100)
    logger.info("Retrieval Success Rate   : %.1f%%", metrics.retrieval_success_rate * 100)
    logger.info("Average Top-1 Similarity : %.4f", metrics.average_top_score)
    logger.info("False Positives          : %d", metrics.false_positives)
    logger.info("False Negatives          : %d", metrics.false_negatives)
    logger.info("=" * 60)

    return metrics


def main() -> None:
    asyncio.run(run_suite())


if __name__ == "__main__":
    main()
