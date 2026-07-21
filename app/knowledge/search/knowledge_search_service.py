from app.knowledge.models.knowledge_document import KnowledgeDocument
from app.knowledge.repository.in_memory_repository import InMemoryKnowledgeRepository
from app.knowledge.search.lexical_search import LexicalSearch


class KnowledgeSearchService:
    """
    Public entry point for knowledge search.

    Owns the repository dependency and is responsible for fetching documents
    before delegating scoring to LexicalSearch. This keeps LexicalSearch
    stateless and independently testable.

    The delegation point is designed for future extension to HybridSearch
    (lexical + semantic via Qdrant) without changing this class's public API.

    Callers should depend on KnowledgeSearchService, not LexicalSearch directly.
    """

    def __init__(
        self,
        repository: InMemoryKnowledgeRepository,
        lexical_search: LexicalSearch,
    ) -> None:
        self._repository = repository
        self._lexical_search = lexical_search

    async def search(self, query: str) -> list[KnowledgeDocument]:
        """
        Search the knowledge base using the query string.

        Returns documents sorted by relevance score (descending).
        Documents with no matching tokens are excluded.
        """
        documents = await self._repository.list_all()
        return self._lexical_search.search(query, documents)
