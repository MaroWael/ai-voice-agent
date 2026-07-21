"""
Knowledge base initializer.

Idempotent startup routine that prepares the Qdrant knowledge base.

Public API:

    await initialize_knowledge_base()

Safe to call on every application startup. If the collection already
contains indexed documents the function returns immediately without
touching the vector store.
"""

import logging

from app.config.settings import settings
from app.db.qdrant import get_qdrant
from app.embeddings.services.embedding_service import EmbeddingService
from app.factories.embeddings import build_sentence_transformer_provider
from app.factories.vector_store import build_qdrant_provider
from app.knowledge.extractors.section_extractor import SectionExtractor
from app.knowledge.loaders.json_loader import JsonKnowledgeLoader
from app.knowledge.normalizers.knowledge_normalizer import KnowledgeNormalizer
from app.knowledge.repository.in_memory_repository import InMemoryKnowledgeRepository
from app.knowledge.validators.knowledge_validator import (
    KnowledgeValidationError,
    KnowledgeValidator,
)
from app.vector_store.services.qdrant_indexer import QdrantIndexer

logger = logging.getLogger(__name__)


async def _is_initialized() -> bool:
    """
    Return True if the configured Qdrant collection exists and contains indexed points.

    Private — use `initialize_knowledge_base()` which calls this internally.
    """
    client = get_qdrant()
    collection_name = settings.QDRANT_COLLECTION_NAME
    try:
        exists = await client.collection_exists(collection_name)
        if not exists:
            return False
        info = await client.get_collection(collection_name)
        points_count = info.points_count if info and info.points_count is not None else 0
        return points_count > 0
    except Exception as exc:
        logger.warning("Error checking collection '%s': %s", collection_name, exc)
        return False


async def initialize_knowledge_base() -> None:
    """
    Idempotent knowledge base initialization.

    Workflow:
        1. Check whether the collection already contains indexed documents.
        2. If yes  →  print a short message and return immediately.
        3. If no   →  load JSON files, validate, embed, and index them.
    """
    if await _is_initialized():
        print("Knowledge base already initialized.")
        print("Skipping indexing.")
        return

    collection_name = settings.QDRANT_COLLECTION_NAME

    # ── Load ──────────────────────────────────────────────────────────────────
    loader = JsonKnowledgeLoader()
    validator = KnowledgeValidator()
    normalizer = KnowledgeNormalizer()
    extractor = SectionExtractor(normalizer)
    repository = InMemoryKnowledgeRepository()

    data_dir = settings.KNOWLEDGE_DATA_PATH
    raw_documents = await loader.load_directory(data_dir)

    for raw_doc in raw_documents:
        try:
            validator.validate(raw_doc)
        except KnowledgeValidationError:
            continue
        knowledge_docs = extractor.extract(raw_doc)
        await repository.save_many(knowledge_docs)

    documents = await repository.list_all()

    # ── Embed ─────────────────────────────────────────────────────────────────
    # Use a single provider instance so the model is loaded only once.
    # EmbeddingService is wired from that same provider.
    provider = build_sentence_transformer_provider()
    embedding_service = EmbeddingService(provider)
    embedded_docs = await embedding_service.embed_documents(documents)

    # ── Index ─────────────────────────────────────────────────────────────────
    qdrant_provider = build_qdrant_provider()
    indexer = QdrantIndexer(
        provider=qdrant_provider,
        collection_name=collection_name,
        batch_size=settings.QDRANT_BATCH_SIZE,
    )
    indexed_count = await indexer.index_documents(embedded_docs, provider.dimension)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n==========================================")
    print("KNOWLEDGE BASE INITIALIZATION")
    print("==========================================")
    print(f"\nCollection: {collection_name}")
    print(f"\nDocuments: {len(documents)}")
    print(f"Embeddings: {len(embedded_docs)}")
    print(f"Indexed: {indexed_count}")
    print("\nKnowledge base is ready.")
