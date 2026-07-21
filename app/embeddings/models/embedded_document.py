from pydantic import BaseModel, ConfigDict

from app.knowledge.models.knowledge_document import KnowledgeDocument


class EmbeddedDocument(BaseModel):
    """
    A KnowledgeDocument paired with its embedding vector.

    Produced by EmbeddingService. Passed directly to the Qdrant indexing
    layer in the next epic — no modifications needed on either side.

    The original KnowledgeDocument is preserved intact so that all metadata
    and content remain accessible without re-fetching from the repository.
    """

    model_config = ConfigDict(frozen=True)

    document: KnowledgeDocument

    # Dense embedding produced by the configured SentenceTransformer model.
    # Dimension depends on the model (1024 for BAAI/bge-m3).
    embedding: list[float]
