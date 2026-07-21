import uuid

from app.embeddings.models.embedded_document import EmbeddedDocument
from app.vector_store.models.vector_point import VectorPoint

# Dedicated namespace UUID for Voice AI Assistant Knowledge Documents
KNOWLEDGE_NAMESPACE_UUID = uuid.UUID("f47ac10b-58cc-4372-a567-0e02b2c3d479")


class PointMapper:
    """
    Transforms EmbeddedDocuments into provider-agnostic VectorPoint domain models.

    Responsibilities:
    - Derive deterministic UUIDs for vector points using a project-specific namespace.
    - Extract useful retrieval metadata into the payload.
    - Exclude text content from payload to keep vector storage lightweight and non-redundant.
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
