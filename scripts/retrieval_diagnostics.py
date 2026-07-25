"""
Phase 6 — Retrieval Diagnostics Script

Executes targeted test queries across English and Arabic product names,
printing full retrieval details (Query, Embedded Query, Top-5 Chunks, Scores, Product, Section).
"""

import asyncio
import sys
from pathlib import Path

# Add project root directory to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

from app.config.settings import settings
from app.factories.embeddings import build_sentence_transformer_provider
from app.factories.vector_store import build_qdrant_provider
from app.query_optimization.factory import build_query_normalizer
from app.retrieval.services.retrieval_service import RetrievalService
from app.startup.knowledge_initializer import initialize_knowledge_base


async def run_diagnostics() -> None:
    print("=" * 80)
    print("PHASE 6 — RETRIEVAL DIAGNOSTICS SUITE")
    print("=" * 80)

    # 1. Initialize knowledge base in Qdrant (in-memory mode for diagnostics)
    await initialize_knowledge_base()

    # 2. Build retrieval service collaborators
    normalizer = build_query_normalizer()
    embedder = build_sentence_transformer_provider()
    qdrant = build_qdrant_provider()

    retriever = RetrievalService(
        embedding_provider=embedder,
        vector_store_provider=qdrant,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        top_k=5,
    )

    test_queries = [
        "Platinum",
        "Platinum Credit Card",
        "بطاقة البلاتينيوم",
        "البلاتينيوم",
        "Gold",
        "بطاقة الجولد",
        "Titanium",
        "بطاقة التيتانيوم",
    ]

    for q in test_queries:
        print("\n" + "=" * 80)
        print(f"QUERY: {q}")
        
        normalized = await normalizer.normalize(q)
        embedded_query = normalized if normalized else q
        print(f"EMBEDDED QUERY: {embedded_query}")
        print("-" * 80)

        results = await retriever.retrieve(embedded_query, top_k=5)

        for rank, res in enumerate(results, start=1):
            product_name = res.document.metadata.product_name if res.document.metadata else "Unknown Product"
            section_name = res.document.title or "Unknown Section"
            aliases = res.document.metadata.aliases if res.document.metadata else []
            score = res.score

            print(f"  [{rank}] Score: {score:.4f} | Product: '{product_name}' | Section: '{section_name}'")
            print(f"      ID: {res.document.id}")
            if aliases:
                print(f"      Aliases: {', '.join(aliases[:4])}")

    print("\n" + "=" * 80)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_diagnostics())
