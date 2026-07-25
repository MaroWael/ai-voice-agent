"""
CLI entry point for knowledge base initialization.

Delegates entirely to app.startup.knowledge_initializer.

Run with:
    python initialize_knowledge_base.py [--reindex]
"""

import argparse
import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.startup.knowledge_initializer import initialize_knowledge_base

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize or re-index knowledge base in Qdrant.")
    parser.add_argument("--reindex", action="store_true", help="Force re-indexing by deleting existing Qdrant collection")
    args = parser.parse_args()

    asyncio.run(initialize_knowledge_base(force_reindex=args.reindex))
