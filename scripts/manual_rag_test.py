"""
Manual RAG Verification Script

Interactive & CLI manual test script for the refactored RAG pipeline.

Tests:
  - Arabic queries
  - English queries
  - Mixed-language queries
  - Unsupported queries

Usage:
    python scripts/manual_rag_test.py [--interactive]
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root directory to sys.path for direct script execution
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

from app.config.settings import settings
from app.factories.rag import build_rag_service


# Sample test queries covering conversational voice queries and out-of-scope queries
SAMPLE_QUERIES = [
    {
        "category": "Conversational Gold Card Query (Dialectal)",
        "question": "ايه هيه مصاريف الفيزا الجولد",
    },
    {
        "category": "Conversational Platinum Card Query",
        "question": "عايز اسأل عن مصاريف بطاقة البلاتينيوم",
    },
    {
        "category": "Conversational Classic Card Query",
        "question": "كام رسوم بطاقة كلاسيك",
    },
    {
        "category": "Out-of-Scope Loan Query",
        "question": "ما هو سعر الفائدة على القروض الشخصية؟",
    },
]


async def run_query(rag_service, question: str) -> None:
    """Run a single question through RAG with debug enabled and print clean output."""
    print("\n" + "=" * 70)
    print(f"QUESTION: {question}")
    print("=" * 70)

    # Force debug=True for manual inspection script
    response = await rag_service.answer(question, debug=True)

    print(f"\nSTATUS: {response.status.value}")
    print(f"\nANSWER:\n{response.answer}")

    if response.debug_info:
        dbg = response.debug_info
        print("\n--- DEBUG INFORMATION ---")
        print(f"Original Query:    {dbg.original_query}")
        print(f"Normalized Query:  {dbg.normalized_query}")
        print(f"Detection Status:  has_context={dbg.has_context} (reason={dbg.detection_reason})")
        print(f"Retrieval Scores:  {dbg.retrieval_scores}")
        print(f"Prompt Length:     {dbg.prompt_length_chars} chars")
        print("Stage Timings (ms):")
        for stage, ms in dbg.latencies_ms.items():
            print(f"  - {stage:15s}: {ms:.2f} ms")

        print("\nRetrieved Chunks:")
        for idx, chunk in enumerate(dbg.retrieved_chunks, start=1):
            print(f"  [{idx}] Product: '{chunk.product_name}' | Section: '{chunk.section}' | Score: {chunk.score:.4f} | ID: {chunk.id}")

        if dbg.final_context:
            print("\nFinal Context Preview (first 300 chars):")
            preview = dbg.final_context[:300].replace("\n", " ")
            print(f"  {preview}...")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Manual RAG Verification Tool")
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Run in interactive prompt mode"
    )
    args = parser.parse_args()

    print("Initializing RAG service...")
    rag_service = build_rag_service()
    await rag_service.initialize()

    try:
        if args.interactive:
            print("\n=== Interactive RAG Verification Mode ===")
            print("Type your question and press Enter. Type 'exit' or 'quit' to stop.\n")
            while True:
                user_input = input("\nEnter Question: ").strip()
                if user_input.lower() in ("exit", "quit", "q"):
                    break
                if not user_input:
                    continue
                await run_query(rag_service, user_input)
        else:
            print("\n=== Running Sample Verification Test Suite ===")
            for item in SAMPLE_QUERIES:
                print(f"\n>>> Scenario: [{item['category']}]")
                await run_query(rag_service, item["question"])
    finally:
        await rag_service.close()
        print("\nRAG service closed.")


if __name__ == "__main__":
    asyncio.run(main())
