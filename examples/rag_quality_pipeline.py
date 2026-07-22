"""
RAG Quality Layer — End-to-End Pipeline Demo

Demonstrates the complete quality pipeline:

    User Question
         ↓
    QueryOptimizer       → optimized query for retrieval
         ↓
    RetrievalService     → retrieved documents + scores
         ↓
    UnknownAnswerDetector → DetectionResult (has_context, reason, scores)
         ↓ (if sufficient)
    ContextBuilder + PromptBuilder + LLMProvider
         ↓
    RagResponse (status: SUCCESS | INSUFFICIENT_CONTEXT)

The script runs 3 scenario groups:
  1. In-domain Arabic/English questions  → expect SUCCESS
  2. Mixed Arabic/English questions      → expect SUCCESS
  3. Out-of-domain questions             → expect INSUFFICIENT_CONTEXT

Prerequisites:
    Knowledge base must be initialized:
        python initialize_knowledge_base.py

Run with:
    python examples/rag_quality_pipeline.py
"""

import asyncio
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.config.settings import settings
from app.db.qdrant import get_qdrant
from app.factories.rag import build_rag_service
from app.query_optimization.factory import build_query_optimizer
from app.rag.models.status import RagStatus
from app.retrieval.services.retrieval_service import RetrievalService
from app.unknown_detection.factory import build_unknown_detector

# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------

SCENARIOS = [
    # ── Group 1: In-domain Arabic ──────────────────────────────────────────
    {
        "group": "In-Domain Arabic",
        "query": "كام رسوم البلاتينيوم؟",
        "expected_status": "SUCCESS",
        "description": "Arabic: fees for the Platinum card",
    },
    {
        "group": "In-Domain Arabic",
        "query": "ايه مزايا بطاقة الجولد؟",
        "expected_status": "SUCCESS",
        "description": "Arabic: benefits of the Gold card",
    },
    {
        "group": "In-Domain Arabic",
        "query": "الحد الائتماني للكلاسيك كام؟",
        "expected_status": "SUCCESS",
        "description": "Arabic: credit limit for Classic card",
    },
    # ── Group 2: In-domain English ─────────────────────────────────────────
    {
        "group": "In-Domain English",
        "query": "What are the fees for the Platinum credit card?",
        "expected_status": "SUCCESS",
        "description": "English: Platinum fees",
    },
    {
        "group": "In-Domain English",
        "query": "What installment options are available for the Gold card?",
        "expected_status": "SUCCESS",
        "description": "English: Gold installments",
    },
    # ── Group 3: Mixed Arabic/English ──────────────────────────────────────
    {
        "group": "Mixed Language",
        "query": "ايه الـ fees على البلاتينيوم؟",
        "expected_status": "SUCCESS",
        "description": "Mixed: fees on Platinum (code-switching)",
    },
    {
        "group": "Mixed Language",
        "query": "the credit limit بتاع الجولد كام؟",
        "expected_status": "SUCCESS",
        "description": "Mixed: credit limit of Gold",
    },
    # ── Group 4: Out-of-domain ─────────────────────────────────────────────
    {
        "group": "Out-of-Domain",
        "query": "ازاي اصلح شاشة الموبايل؟",
        "expected_status": "INSUFFICIENT_CONTEXT",
        "description": "Arabic: out-of-domain (phone repair)",
    },
    {
        "group": "Out-of-Domain",
        "query": "What is the weather forecast for Cairo tomorrow?",
        "expected_status": "INSUFFICIENT_CONTEXT",
        "description": "English: out-of-domain (weather)",
    },
    {
        "group": "Out-of-Domain",
        "query": "ايه أحسن مطعم في القاهرة؟",
        "expected_status": "INSUFFICIENT_CONTEXT",
        "description": "Arabic: out-of-domain (restaurant)",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _separator(char: str = "─", width: int = 65) -> str:
    return char * width


async def _knowledge_base_ready() -> bool:
    """Return True if Qdrant collection exists and is non-empty."""
    client = get_qdrant()
    try:
        if not await client.collection_exists(settings.QDRANT_COLLECTION_NAME):
            return False
        info = await client.get_collection(settings.QDRANT_COLLECTION_NAME)
        return bool(info and info.points_count)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main pipeline demo
# ---------------------------------------------------------------------------

async def main() -> None:
    if not await _knowledge_base_ready():
        print("❌ Knowledge base has not been initialized.")
        print("\nRun first:\n    python initialize_knowledge_base.py\n")
        return

    optimizer = build_query_optimizer()
    rag_service = build_rag_service()
    unknown_detector = build_unknown_detector()

    await rag_service.initialize()

    # We also need a bare RetrievalService to show intermediate retrieval results.
    # build_rag_service() wraps this internally; here we build a separate one
    # for the evaluation display layer.
    from app.factories.retrieval import build_retrieval_service
    retrieval_service = build_retrieval_service()

    print()
    print(_separator("═"))
    print("  RAG QUALITY LAYER — END-TO-END PIPELINE DEMO")
    print(_separator("═"))
    print(f"  Embedding model  : {settings.EMBEDDING_MODEL}")
    print(f"  LLM model        : {settings.LLM_MODEL}")
    print(f"  Detection thresholds:")
    print(f"    min_score      : {settings.UNKNOWN_DETECTOR_MIN_SCORE}")
    print(f"    min_results    : {settings.UNKNOWN_DETECTOR_MIN_RESULTS}")
    print(f"    mean_threshold : {settings.UNKNOWN_DETECTOR_MEAN_THRESHOLD}")
    print(_separator("═"))

    pass_count = 0
    fail_count = 0
    current_group = None

    try:
        for idx, scenario in enumerate(SCENARIOS, start=1):
            group = scenario["group"]
            query = scenario["query"]
            expected = scenario["expected_status"]
            description = scenario["description"]

            # Print group header when group changes
            if group != current_group:
                current_group = group
                print(f"\n{'── ' + group + ' ':─<65}")

            print(f"\n[{idx:02d}] {description}")
            print(f"     Original query  : {query!r}")

            # ── Step 1: Query Optimization ────────────────────────────────
            optimized = await optimizer.optimize(query)
            retrieval_query = optimized if optimized else query
            print(f"     Optimized query : {retrieval_query!r}")

            # ── Step 2: Retrieval ─────────────────────────────────────────
            docs = await retrieval_service.retrieve(retrieval_query, top_k=5)
            scores_display = [f"{d.score:.4f}" for d in docs]
            print(f"     Retrieved docs  : {len(docs)}  scores={scores_display}")

            # ── Step 3: Unknown Detection ─────────────────────────────────
            detection = await unknown_detector.evaluate(query, docs)
            print(
                f"     Detection       : has_context={detection.has_context} "
                f"reason={detection.reason.value} "
                f"top={detection.top_score:.4f} avg={detection.average_score:.4f}"
            )

            # ── Step 4: Full RAG pipeline (for answer) ────────────────────
            response = await rag_service.answer(query, top_k=5)
            actual_status = response.status.value.upper()

            # Result
            match = actual_status == expected
            if match:
                pass_count += 1
                tag = "✅ PASS"
            else:
                fail_count += 1
                tag = "❌ FAIL"

            print(f"     Status          : {actual_status}  (expected {expected})  {tag}")

            if response.status == RagStatus.INSUFFICIENT_CONTEXT:
                print(f"     Response        : {response.answer!r}")
            else:
                # Show a trimmed answer
                answer_preview = response.answer[:120].replace("\n", " ")
                if len(response.answer) > 120:
                    answer_preview += "…"
                print(f"     Answer preview  : {answer_preview!r}")

        # ── Summary ───────────────────────────────────────────────────────
        total = pass_count + fail_count
        print()
        print(_separator("═"))
        print("  SUMMARY")
        print(_separator("─"))
        print(f"  Total scenarios  : {total}")
        print(f"  Passed           : {pass_count}")
        print(f"  Failed           : {fail_count}")
        print(f"  Accuracy         : {pass_count / total * 100:.1f}%")
        print(_separator("═"))

    finally:
        await rag_service.close()


if __name__ == "__main__":
    asyncio.run(main())
