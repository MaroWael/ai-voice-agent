"""
End-to-end demonstration of the Knowledge Engine ingestion pipeline.

Pipeline:
    JSON files in data/
        → JsonKnowledgeLoader   (parse raw JSON)
        → KnowledgeValidator    (enforce business rules)
        → SectionExtractor      (+ KnowledgeNormalizer internally)
        → InMemoryKnowledgeRepository
        → KnowledgeSearchService (lexical search)

Run with:
    python knowledge_example.py
"""

import asyncio
import sys
from pathlib import Path

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.knowledge.extractors.section_extractor import SectionExtractor
from app.knowledge.loaders.json_loader import JsonKnowledgeLoader
from app.knowledge.normalizers.knowledge_normalizer import KnowledgeNormalizer
from app.knowledge.repository.in_memory_repository import InMemoryKnowledgeRepository
from app.knowledge.search.knowledge_search_service import KnowledgeSearchService
from app.knowledge.search.lexical_search import LexicalSearch
from app.knowledge.validators.knowledge_validator import (
    KnowledgeValidationError,
    KnowledgeValidator,
)

from app.config.settings import settings
from app.embeddings.models.embedded_document import EmbeddedDocument
from app.embeddings.providers.sentence_transformer_provider import SentenceTransformerProvider
from app.embeddings.services.embedding_service import EmbeddingService

DATA_DIR = Path("data")

SEARCH_QUERIES = [
    "installment",
    "lounge access",
    "credit limit fees",
    "cash withdrawal",
    "supplementary card",
]


async def ingest(
    loader: JsonKnowledgeLoader,
    validator: KnowledgeValidator,
    extractor: SectionExtractor,
    repository: InMemoryKnowledgeRepository,
) -> tuple[int, int]:
    """Load, validate, extract, and store all documents. Returns (stored, skipped)."""
    print(f"\n{'-' * 60}")
    print(f"  INGESTION  --  source: {DATA_DIR.resolve()}")
    print(f"{'-' * 60}")

    raw_documents = await loader.load_directory(DATA_DIR)
    print(f"  Loaded {len(raw_documents)} JSON file(s).\n")

    stored = 0
    skipped = 0

    for raw_doc in raw_documents:
        try:
            validator.validate(raw_doc)
        except KnowledgeValidationError as exc:
            print(f"  [SKIP]  {exc}")
            skipped += 1
            continue

        knowledge_docs = extractor.extract(raw_doc)
        await repository.save_many(knowledge_docs)
        stored += len(knowledge_docs)
        print(f"  [OK]    {raw_doc.name!r:45s}  ->  {len(knowledge_docs)} section(s)")

    print(f"\n  Ingestion complete: {stored} section(s) stored, {skipped} skipped.")
    return stored, skipped


async def run_searches(search_service: KnowledgeSearchService) -> None:
    """Run a set of demo queries and print results."""
    print(f"\n{'-' * 60}")
    print("  SEARCH RESULTS")
    print(f"{'-' * 60}")

    for query in SEARCH_QUERIES:
        results = await search_service.search(query)
        print(f"\n  Query: '{query}'  ->  {len(results)} result(s)")
        for doc in results[:5]:
            print(f"    [{doc.metadata.product_name}]  {doc.title}")
            # Show a short excerpt of the normalized content.
            excerpt = doc.content[:120].replace("\n", " ")
            print(f"      > {excerpt}{'...' if len(doc.content) > 120 else ''}")


async def demo_embeddings(
    repository: InMemoryKnowledgeRepository,
    embedding_service: EmbeddingService,
    provider: SentenceTransformerProvider,
) -> tuple[list[EmbeddedDocument], int]:
    """Generate embeddings for all ingested documents and print a summary."""
    print(f"\n{'-' * 60}")
    print("  EMBEDDINGS")
    print(f"{'-' * 60}")

    documents = await repository.list_all()
    embedded = await embedding_service.embed_documents(documents)
    dimension = provider.dimension

    print(f"\n  Loaded model      : {provider.model_name}")
    print(f"  Documents         : {len(documents)}")
    print(f"  Embedding dimension: {dimension}")
    print(f"  Embeddings generated: {len(embedded)}")
    return embedded, dimension


async def demo_qdrant_indexing(
    embedded_docs: list[EmbeddedDocument],
    vector_dimension: int,
) -> None:
    """Index all embedded documents into Qdrant vector store and print summary."""
    print(f"\n{'-' * 60}")
    print("  QDRANT VECTOR INDEXING")
    print(f"{'-' * 60}")

    from app.db.qdrant import get_qdrant
    from app.vector_store.providers.qdrant_provider import QdrantProvider
    from app.vector_store.services.qdrant_indexer import QdrantIndexer

    client = get_qdrant()
    vector_provider = QdrantProvider(client)
    indexer = QdrantIndexer(
        provider=vector_provider,
        collection_name=settings.QDRANT_COLLECTION_NAME,
        batch_size=settings.QDRANT_BATCH_SIZE,
    )

    indexed_count = await indexer.index_documents(embedded_docs, vector_dimension)

    print(f"\n  Qdrant Collection : {settings.QDRANT_COLLECTION_NAME}")
    print(f"  Vector Size       : {vector_dimension}")
    print(f"  Batch Size        : {settings.QDRANT_BATCH_SIZE}")
    print(f"  Indexed Points    : {indexed_count}")


async def demo_document_inspection(repository: InMemoryKnowledgeRepository) -> None:
    """Inspect a single document to show raw vs. normalized content."""
    print(f"\n{'-' * 60}")
    print("  DOCUMENT INSPECTION  --  raw_content vs. content")
    print(f"{'-' * 60}")

    all_docs = await repository.list_all()
    # Pick a document with a table for a meaningful comparison.
    table_doc = next(
        (d for d in all_docs if "fees" in d.title.lower()),
        all_docs[0] if all_docs else None,
    )

    if table_doc is None:
        print("  No documents in repository.")
        return

    print(f"\n  Document ID  : {table_doc.id}")
    print(f"  Product      : {table_doc.metadata.product_name}")
    print(f"  Section      : {table_doc.title}")
    print(f"\n  --- raw_content (first 300 chars) ---")
    print(f"  {table_doc.raw_content[:300].replace(chr(10), chr(10) + '  ')}")
    print(f"\n  --- content / normalized (first 300 chars) ---")
    print(f"  {table_doc.content[:300].replace(chr(10), chr(10) + '  ')}")


async def main() -> None:
    # Wire up the pipeline via constructor injection.
    loader = JsonKnowledgeLoader()
    validator = KnowledgeValidator()
    normalizer = KnowledgeNormalizer()
    extractor = SectionExtractor(normalizer)
    repository = InMemoryKnowledgeRepository()
    search_service = KnowledgeSearchService(repository, LexicalSearch())

    provider = SentenceTransformerProvider(
        model_name=settings.EMBEDDING_MODEL,
        batch_size=settings.EMBEDDING_BATCH_SIZE,
    )
    embedding_service = EmbeddingService(provider)

    await ingest(loader, validator, extractor, repository)
    await run_searches(search_service)
    await demo_document_inspection(repository)
    embedded_docs, dimension = await demo_embeddings(repository, embedding_service, provider)
    await demo_qdrant_indexing(embedded_docs, dimension)

    print(f"\n{'-' * 60}\n")


if __name__ == "__main__":
    asyncio.run(main())
    print("END OF MAIN")