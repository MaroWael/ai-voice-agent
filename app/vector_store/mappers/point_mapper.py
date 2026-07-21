import logging
from typing import Any
import uuid

from app.embeddings.models.embedded_document import EmbeddedDocument
from app.knowledge.models.document_metadata import DocumentMetadata
from app.knowledge.models.knowledge_document import KnowledgeDocument
from app.vector_store.models.vector_point import VectorPoint

# Dedicated namespace UUID for Voice AI Assistant Knowledge Documents
KNOWLEDGE_NAMESPACE_UUID = uuid.UUID("f47ac10b-58cc-4372-a567-0e02b2c3d479")

logger = logging.getLogger(__name__)


class PointMapper:
    """
    Bi-directional mapper between KnowledgeDocument domain models and vector store structures.

    Responsibilities:
    - Derive deterministic UUIDs for vector points using a project-specific namespace.
    - Transform EmbeddedDocument domain models into provider-agnostic VectorPoint models for indexing.
    - Transform vector payload dictionaries back into KnowledgeDocument domain models for retrieval.
    """

    def to_vector_point(self, embedded_doc: EmbeddedDocument) -> VectorPoint:
        """
        Convert a single EmbeddedDocument into a VectorPoint.

        Args:
            embedded_doc: The EmbeddedDocument containing knowledge document & vector.

        Returns:
            A populated VectorPoint domain model.
        """
        doc = embedded_doc.document
        point_id = str(uuid.uuid5(KNOWLEDGE_NAMESPACE_UUID, doc.id))

        payload = {
            "id": doc.id,
            "title": doc.title,
            "raw_content": doc.raw_content,
            "content": doc.content,
            "product_id": doc.metadata.product_id,
            "product_name": doc.metadata.product_name,
            "section": doc.metadata.section,
            "source": str(doc.metadata.source),
            "url": doc.metadata.url,
            "language": doc.metadata.language,
        }

        return VectorPoint(
            id=point_id,
            vector=embedded_doc.embedding,
            payload=payload,
        )

    def to_vector_points(
        self, embedded_docs: list[EmbeddedDocument]
    ) -> list[VectorPoint]:
        """
        Convert a list of EmbeddedDocuments into a list of VectorPoints.

        Args:
            embedded_docs: List of EmbeddedDocuments.

        Returns:
            List of VectorPoint domain models.
        """
        return [self.to_vector_point(doc) for doc in embedded_docs]

    def from_payload(self, payload: dict[str, Any] | None) -> KnowledgeDocument | None:
        """
        Reconstruct a KnowledgeDocument from a vector store payload dictionary.

        Args:
            payload: Payload dictionary retrieved from vector store point.

        Returns:
            KnowledgeDocument instance, or None if payload is empty or invalid.
        """
        if not payload:
            return None

        try:
            metadata = DocumentMetadata(
                product_id=str(payload.get("product_id", "")),
                product_name=str(payload.get("product_name", "")),
                section=str(payload.get("section", "")),
                language=str(payload.get("language", "en")),
                source=str(payload.get("source", "")),
                url=str(payload.get("url", "")),
            )
            return KnowledgeDocument(
                id=str(payload.get("id", "")),
                title=str(payload.get("title", payload.get("section", ""))),
                raw_content=str(payload.get("raw_content", payload.get("content", ""))),
                content=str(payload.get("content", "")),
                metadata=metadata,
            )
        except Exception as exc:
            logger.warning("Failed to deserialize KnowledgeDocument from payload: %s", exc)
            return None

