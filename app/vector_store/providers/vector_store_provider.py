from typing import Protocol, runtime_checkable

from app.vector_store.models.vector_point import VectorPoint


@runtime_checkable
class VectorStoreProvider(Protocol):
    """
    Abstract interface for vector database providers.

    Decouples business services from concrete vector database SDKs.
    """

    async def create_collection_if_not_exists(
        self, collection_name: str, vector_size: int
    ) -> None:
        """
        Create a collection in the vector database if it does not already exist.

        Args:
            collection_name: Target collection identifier.
            vector_size: Dimensionality of the vectors stored in the collection.
        """
        ...

    async def upsert_points(
        self, collection_name: str, points: list[VectorPoint]
    ) -> None:
        """
        Upsert a batch of VectorPoints into the specified collection.

        Args:
            collection_name: Target collection identifier.
            points: Batch of VectorPoints to insert or update.
        """
        ...
