"""
RAG Pipeline Verification Example.

Workflow:
    Question → RetrievalService → ContextBuilder → PromptBuilder → LLMProvider → RagResponse

Prerequisites:
    The knowledge base must be initialized before running this script:
        python initialize_knowledge_base.py

Run with:
    python examples/rag_example.py
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
from app.factories.rag import build_rag_service

EXAMPLE_QUESTIONS = [
    # In-domain question (should answer based on Qdrant contents)
    "How much does the Platinum card cost?",
    # Out-of-domain question (should trigger strict fallback constraint)
    "What is the capital of France?",
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

    rag_service = build_rag_service()

    # Initialize the RAG service and its internal resources
    await rag_service.initialize()

    print("\n==================================================")
    print("RAG PIPELINE INITIALIZED")
    print("==================================================")

    try:
        for idx, question in enumerate(EXAMPLE_QUESTIONS, start=1):
            print(f"\n--- Test Case {idx} ---")
            print(f"Question: '{question}'")
            print("Running pipeline...")

            # Run RAG answer pipeline (using top_k=3 for test case context brevity)
            response = await rag_service.answer(question, top_k=10)

            print("\n>>> Generated Answer:")
            print(response.answer)

            print("\n>>> Prompt Sent to LLM:")
            print("-" * 50)
            print(response.prompt)
            print("-" * 50)

            print("\n>>> Retrieved Documents:")
            for rank, doc_result in enumerate(response.retrieved_documents, start=1):
                doc = doc_result.document
                print(f"  Rank {rank} (Score: {doc_result.score:.4f}): {doc.metadata.product_name} - {doc.title}")

            print("\n" + "=" * 50)

    finally:
        # Guarantee resource release
        await rag_service.close()


if __name__ == "__main__":
    asyncio.run(main())
