import logging
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config.settings import settings
from app.vector_store.exceptions import (
    BatchUploadError,
    CollectionError,
    VectorStoreConnectionError,
)
from app.vector_store.models.vector_point import VectorPoint
from app.vector_store.providers.vector_store_provider import VectorStoreProvider

logger = logging.getLogger(__name__)


class QdrantProvider(VectorStoreProvider):
    """
    Concrete implementation of VectorStoreProvider backed by Qdrant AsyncQdrantClient.

    Encapsulates all Qdrant SDK calls and model conversions (VectorPoint -> PointStruct).
    Domain logic outside this provider never touches the Qdrant SDK.
    """

    def __init__(self, client: AsyncQdrantClient) -> None:
        self._client = client
        self._distance_metric = self._resolve_distance_metric(settings.QDRANT_DISTANCE_METRIC)

    @staticmethod
    def _resolve_distance_metric(metric_name: str) -> Distance:
        """Map setting metric name string to Qdrant SDK Distance enum."""
        try:
            return Distance[metric_name.upper()]
        except KeyError:
            logger.warning(
                "Unknown distance metric '%s'. Defaulting to Cosine.", metric_name
            )
            return Distance.COSINE

    async def create_collection_if_not_exists(
        self, collection_name: str, vector_size: int
    ) -> None:
        """
        Create a collection in Qdrant if it does not already exist.

        Directly attempts creation and gracefully catches existing collection errors.

        Args:
            collection_name: Target Qdrant collection name.
            vector_size: Dimension of stored vectors.

        Raises:
            CollectionError: If creation fails due to server error.
            VectorStoreConnectionError: If Qdrant service is unreachable.
        """
        logger.info(
            "Creating collection '%s' (vector size: %d, distance: %s)...",
            collection_name,
            vector_size,
            self._distance_metric.name,
        )

        try:
            await self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=self._distance_metric,
                ),
            )
            logger.info("Collection '%s' created successfully.", collection_name)
        except UnexpectedResponse as exc:
            # Status 409 or message containing "already exists" indicates existing collection
            if exc.status_code in (400, 409) or "already exists" in str(exc).lower():
                logger.info("Collection '%s' already exists.", collection_name)
            else:
                logger.error("Failed to create collection '%s': %s", collection_name, exc)
                raise CollectionError(
                    f"Failed to create collection '{collection_name}': {exc}"
                ) from exc
        except Exception as exc:
            logger.error(
                "Connection error during collection creation for '%s': %s",
                collection_name,
                exc,
            )
            raise VectorStoreConnectionError(
                f"Failed to connect to Qdrant while creating collection '{collection_name}': {exc}"
            ) from exc

    async def upsert_points(
        self, collection_name: str, points: list[VectorPoint]
    ) -> None:
        """
        Upsert a batch of VectorPoints into Qdrant.

        Converts domain VectorPoint objects into Qdrant PointStruct instances internally.

        Args:
            collection_name: Target Qdrant collection name.
            points: Batch of VectorPoints to upsert.

        Raises:
            BatchUploadError: If the upsert request fails.
            VectorStoreConnectionError: If Qdrant is unreachable.
        """
        if not points:
            return

        point_structs = [
            PointStruct(
                id=point.id,
                vector=point.vector,
                payload=point.payload,
            )
            for point in points
        ]

        try:
            await self._client.upsert(
                collection_name=collection_name,
                points=point_structs,
            )
        except UnexpectedResponse as exc:
            logger.error(
                "Batch upload failed for collection '%s': %s", collection_name, exc
            )
            raise BatchUploadError(
                f"Failed to upsert batch into '{collection_name}': {exc}"
            ) from exc
        except Exception as exc:
            logger.error(
                "Connection error during batch upload for '%s': %s",
                collection_name,
                exc,
            )
            raise VectorStoreConnectionError(
                f"Connection failure during batch upload to '{collection_name}': {exc}"
            ) from exc
