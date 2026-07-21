from typing import Any

from pydantic import BaseModel, ConfigDict


class VectorPoint(BaseModel):
    """
    Framework-independent representation of a vector point.

    Decouples domain logic from vector database SDKs (such as Qdrant's PointStruct).
    """

    model_config = ConfigDict(frozen=True)

    # Deterministic UUID string derived from the document ID
    id: str

    # Embedded float vector representation
    vector: list[float]

    # Associated metadata fields for filtering and provenance
    payload: dict[str, Any]
