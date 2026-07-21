from pydantic import BaseModel, ConfigDict

from app.knowledge.models.document_metadata import DocumentMetadata


class KnowledgeDocument(BaseModel):
    """
    A single searchable knowledge section.

    Produced by SectionExtractor. One document per section of a RawDocument.
    Does not include embeddings — those belong to a future epic.

    Both raw_content and content are preserved so that the normalization
    pipeline can be improved later without discarding the original text.
    """

    model_config = ConfigDict(frozen=True)

    # Deterministic ID: slugified product name + zero-padded section index.
    # Example: "al-araby-card_002"
    id: str

    # Section title, e.g. "Fees and charges".
    title: str

    # Assembled section text before normalization (content + bullets + tables).
    # Preserved for auditing and future pipeline improvements.
    raw_content: str

    # Normalized version of raw_content — the field used for search and retrieval.
    content: str

    metadata: DocumentMetadata
