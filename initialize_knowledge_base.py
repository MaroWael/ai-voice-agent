"""
CLI entry point for knowledge base initialization.

Delegates entirely to app.startup.knowledge_initializer.
No business logic lives here.

Run with:
    python initialize_knowledge_base.py
"""

import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.startup.knowledge_initializer import initialize_knowledge_base

if __name__ == "__main__":
    asyncio.run(initialize_knowledge_base())
