"""
Inspect exact embedding text generated for Knowledge Base documents.
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
from app.embeddings.providers.embedding_provider import EmbeddingProvider
from app.embeddings.services.embedding_service import EmbeddingService
from app.factories.embeddings import build_sentence_transformer_provider
from app.knowledge.extractors.section_extractor import SectionExtractor
from app.knowledge.loaders.json_loader import JsonKnowledgeLoader
from app.knowledge.normalizers.knowledge_normalizer import KnowledgeNormalizer


class DummyProvider(EmbeddingProvider):
    @property
    def dimension(self) -> int:
        return 1024

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 1024 for _ in texts]


async def inspect() -> None:
    loader = JsonKnowledgeLoader()
    normalizer = KnowledgeNormalizer()
    extractor = SectionExtractor(normalizer)
    service = EmbeddingService(DummyProvider())

    raw_docs = await loader.load_directory(settings.KNOWLEDGE_DATA_PATH)

    target_products = [
        "Classic Credit Card",
        "Gold Credit Cards",
        "Titanium Credit Card",
        "Platinum Visa - Master Credit Card",
    ]

    for raw_doc in raw_docs:
        if raw_doc.name not in target_products:
            continue

        print("\n" + "=" * 80)
        print(f"RAW DOCUMENT NAME: {raw_doc.name}")
        print("=" * 80)

        knowledge_docs = extractor.extract(raw_doc)
        for doc in knowledge_docs:
            if "Fees" in doc.title or "fees" in doc.title.lower():
                embed_text = service._build_embedding_text(doc)
                print(f"\n--- [Document ID: {doc.id}] ---")
                print("VERBATIM EMBEDDING TEXT:")
                print(embed_text)
                print("-" * 50)


if __name__ == "__main__":
    asyncio.run(inspect())
