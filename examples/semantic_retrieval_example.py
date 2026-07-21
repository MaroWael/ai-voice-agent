"""
Semantic Retrieval Example.

Workflow:
    Query → EmbeddingService (Query Vector) → QdrantProvider (Search) → list[SearchResult]

Timing stages reported per query:
    Embedding Time  — query vector generation (includes model cold-load on first run)
    Search Time     — Qdrant vector similarity search
    Total Time      — sum of both stages

Prerequisites:
    The knowledge base must be initialized before running this script:
        python initialize_knowledge_base.py

Run with:
    python examples/semantic_retrieval_example.py
"""

import asyncio
from pathlib import Path
import sys

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.config.settings import settings
from app.db.qdrant import get_qdrant
from app.factories.retrieval import build_retrieval_service

EXAMPLE_QUESTIONS = [
    "How much does the Platinum card cost?",
]


async def _knowledge_base_ready() -> bool:
    """Return True if the Qdrant collection exists and contains indexed points."""
    client = get_qdrant()
    try:
        exists = await client.collection_exists(settings.QDRANT_COLLECTION_NAME)
        if not exists:
            return False
        info = await client.get_collection(settings.QDRANT_COLLECTION_NAME)
        points_count = info.points_count if info and info.points_count is not None else 0
        return points_count > 0
    except Exception:
        return False


async def main() -> None:
    if not await _knowledge_base_ready():
        print("Knowledge base has not been initialized.")
        print("\nRun:\n")
        print("python initialize_knowledge_base.py")
        return

    retrieval_service = build_retrieval_service()

    print("\n==================================================")
    print("SEMANTIC RETRIEVAL")
    print("==================================================")

    for question in EXAMPLE_QUESTIONS:
        print(f"\nQuestion:\n{question}")
        print("\n--------------------------------------------------")

        results, timing = await retrieval_service.retrieve_timed(question, top_k=5)

        for result in results:
            print(f"\nRank: {result.rank}")
            print(f"Score: {result.score:.4f}")
            print(f"\nProduct:\n{result.document.metadata.product_name}")
            print(f"\nSection:\n{result.document.metadata.section}")
            print(f"\nTitle:\n{result.document.title}")
            print(f"\nContent:\n{result.document.content}")
            print("\n--------------------------------------------------")

        print(f"\nEmbedding Time:\n{timing.embedding_time:.3f} s")
        print(f"\nSearch Time:\n{timing.search_time:.3f} s")
        print(f"\nTotal Retrieval Time:\n{timing.total_time:.3f} s")


if __name__ == "__main__":
    asyncio.run(main())
