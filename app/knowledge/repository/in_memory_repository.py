from app.knowledge.models.knowledge_document import KnowledgeDocument


class InMemoryKnowledgeRepository:
    """
    In-memory store for KnowledgeDocuments.

    Backed by a dict keyed on document ID. Insertion order is preserved
    (Python 3.7+ dict guarantee), which keeps search results stable.

    All methods are async for interface consistency with future repository
    implementations (e.g. Qdrant-backed) that will require real IO.
    """

    def __init__(self) -> None:
        self._store: dict[str, KnowledgeDocument] = {}

    async def save(self, document: KnowledgeDocument) -> None:
        """Persist a single document. Overwrites if the ID already exists."""
        self._store[document.id] = document

    async def save_many(self, documents: list[KnowledgeDocument]) -> None:
        """Persist multiple documents in one call. Preserves insertion order."""
        for document in documents:
            self._store[document.id] = document

    async def list_all(self) -> list[KnowledgeDocument]:
        """Return all stored documents in insertion order."""
        return list(self._store.values())

    async def get_by_id(self, doc_id: str) -> KnowledgeDocument | None:
        """Return the document with the given ID, or None if not found."""
        return self._store.get(doc_id)

    async def clear(self) -> None:
        """Remove all documents from the store."""
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)
