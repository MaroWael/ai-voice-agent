import asyncio
import logging

from app.embeddings.models.embedded_document import EmbeddedDocument
from app.embeddings.providers.embedding_provider import EmbeddingProvider
from app.knowledge.models.knowledge_document import KnowledgeDocument

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Generates embeddings for KnowledgeDocuments using an EmbeddingProvider.

    Responsibilities:
    - Enrich knowledge documents into semantic embedding texts by combining product name,
      section title, and normalized content.
    - Send all enriched texts to the provider in a single batch call.
    - Pair each embedding with its source document in an EmbeddedDocument.

    Semantic Context Enrichment:
    - Product name (`document.metadata.product_name`): Identifies the banking product.
    - Section title (`document.title`): Provides local semantic context (e.g., distinguishing
      "Fees and Charges" across different credit cards).
    - Normalized content (`document.content`): Contains the searchable knowledge while preserving
      headings, lists, and tables.

    Administrative Metadata Exclusion:
    Administrative metadata (such as `id`, `raw_content`, `product_id`, `source`, `url`, and `language`)
    is intentionally excluded from the embedding text to avoid cluttering the vector space with non-semantic data.
    These fields will be stored later as Qdrant payload metadata for filtering and hybrid retrieval.

    This service knows nothing about vector databases or retrieval.
    It is the bridge between the ingestion pipeline and the indexing layer.
    """

    def __init__(self, provider: EmbeddingProvider) -> None:
        self._provider = provider

    def _build_embedding_text(self, document: KnowledgeDocument) -> str:
        """
        Construct the text representation of a KnowledgeDocument for embedding.

        Enriches document content with semantic context (product name and section title)
        while excluding administrative metadata.

        Args:
            document: The KnowledgeDocument to format.

        Returns:
            The structured embedding text string.
        """
        return (
            f"Product: {document.metadata.product_name}\n\n"
            f"Section: {document.title}\n\n"
            f"Content:\n{document.content}"
        )

    async def embed_documents(
        self, documents: list[KnowledgeDocument]
    ) -> list[EmbeddedDocument]:
        """
        Generate embeddings for a list of KnowledgeDocuments.

        Enriches each KnowledgeDocument into a structured embedding text with semantic context,
        then calls the provider once using batch embedding.

        The provider's encode call is CPU/GPU-bound and may block the event
        loop. It is offloaded to a thread via asyncio.to_thread so that the
        rest of the async pipeline stays unblocked.

        Args:
            documents: Non-empty list of KnowledgeDocuments to embed.

        Returns:
            A list of EmbeddedDocuments in the same order as the input.

        Raises:
            ValueError: If documents is empty.
        """
        if not documents:
            raise ValueError("documents must not be empty.")

        texts = [self._build_embedding_text(doc) for doc in documents]

        logger.info("Generating embeddings for %d document(s)...", len(texts))
        # Offload blocking model inference to a thread pool worker.
        vectors = await asyncio.to_thread(self._provider.embed, texts)
        logger.info("Embeddings generated successfully.")

        return [
            EmbeddedDocument(document=doc, embedding=vector)
            for doc, vector in zip(documents, vectors)
        ]

    async def embed_query(self, text: str) -> list[float]:
        """
        Generate an embedding vector for a query string / user question.

        Offloads the blocking provider call to a thread via asyncio.to_thread.

        Args:
            text: Non-empty query text string.

        Returns:
            A list of float values representing the query embedding vector.

        Raises:
            ValueError: If text is empty or whitespace-only.
        """
        if not text or not text.strip():
            raise ValueError("text must not be empty.")

        vectors = await asyncio.to_thread(self._provider.embed, [text])
        return vectors[0]

