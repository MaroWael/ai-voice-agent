import logging

from app.embeddings.models.embedded_document import EmbeddedDocument
from app.shared.utils.batching import chunk_items
from app.vector_store.mappers.point_mapper import PointMapper
from app.vector_store.providers.vector_store_provider import VectorStoreProvider

logger = logging.getLogger(__name__)


class QdrantIndexer:
    """
    Service responsible for indexing EmbeddedDocuments into a vector store.

    Responsibilities:
    - Ensure vector store collection exists with the required dimensionality.
    - Transform EmbeddedDocuments to VectorPoints via PointMapper.
    - Upload points in configurable batch sizes via VectorStoreProvider.

    Clean Architecture:
    - Depends on VectorStoreProvider abstraction, keeping business logic 100% decoupled
      from third-party vector database SDKs.
    """

    def __init__(
        self,
        provider: VectorStoreProvider,
        mapper: PointMapper | None = None,
        collection_name: str = "knowledge_base",
        batch_size: int = 64,
    ) -> None:
        self._provider = provider
        self._mapper = mapper or PointMapper()
        self._collection_name = collection_name
        self._batch_size = batch_size

    async def ensure_collection(self, vector_size: int) -> None:
        """
        Ensure that the target vector store collection exists.

        Args:
            vector_size: Dimensionality of vectors to be stored.
        """
        await self._provider.create_collection_if_not_exists(
            collection_name=self._collection_name,
            vector_size=vector_size,
        )

    async def index_documents(
        self, documents: list[EmbeddedDocument], vector_size: int
    ) -> int:
        """
        Index a batch of EmbeddedDocuments into the vector store.

        Args:
            documents: Non-empty list of EmbeddedDocuments to index.
            vector_size: Vector dimension of the embeddings.

        Returns:
            The total number of indexed documents.

        Raises:
            ValueError: If documents is empty.
        """
        if not documents:
            raise ValueError("documents must not be empty.")

        await self.ensure_collection(vector_size)

        logger.info(
            "Starting indexing of %d document(s) into '%s'...",
            len(documents),
            self._collection_name,
        )

        vector_points = self._mapper.to_vector_points(documents)
        chunks = list(chunk_items(vector_points, self._batch_size))
        total_chunks = len(chunks)

        indexed_count = 0
        for idx, chunk in enumerate(chunks, start=1):
            logger.info(
                "Uploading batch %d/%d (%d points)...",
                idx,
                total_chunks,
                len(chunk),
            )
            await self._provider.upsert_points(self._collection_name, chunk)
            indexed_count += len(chunk)

        logger.info(
            "Successfully indexed %d document(s) into '%s'.",
            indexed_count,
            self._collection_name,
        )
        return indexed_count
