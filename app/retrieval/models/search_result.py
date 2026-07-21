from pydantic import BaseModel, ConfigDict

from app.knowledge.models.knowledge_document import KnowledgeDocument


class SearchResult(BaseModel):
    """
    Represents a single semantic search result.

    Fields:
        document: The retrieved KnowledgeDocument.
        score:    Similarity score from the vector store (higher is more relevant).
        rank:     1-based position in the result list, assigned by RetrievalService.
                  Defaults to 0 (unranked) when constructed directly by the vector store.
    """

    model_config = ConfigDict(frozen=True)

    document: KnowledgeDocument
    score: float
    rank: int = 0
