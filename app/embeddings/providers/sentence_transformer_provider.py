import logging

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class SentenceTransformerProvider:
    """
    Embedding provider backed by a SentenceTransformer model.

    The model is loaded lazily: the first call to `embed()` triggers
    model loading. Subsequent calls reuse the same instance. This avoids
    paying the load cost at import time and keeps startup fast.

    The model name is injected at construction time and must come from
    Settings — never hardcoded at the call site.

    Satisfies the EmbeddingProvider protocol via structural subtyping.
    """

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        # Model is None until the first embed() call.
        self._model: SentenceTransformer | None = None

    @property
    def model_name(self) -> str:
        """The configured model identifier."""
        return self._model_name

    @property
    def batch_size(self) -> int:
        """The configured inference batch size."""
        return self._batch_size

    @property
    def is_loaded(self) -> bool:
        """True if the model has been loaded into memory."""
        return self._model is not None

    @property
    def dimension(self) -> int:
        """The output vector dimension of the embedding model."""
        model = self._get_model()
        return model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a batch of texts.

        Loads the model on first call. All texts are embedded using the
        configured batch_size.

        Args:
            texts: Non-empty list of strings to embed.

        Returns:
            A list of float vectors in the same order as the input.

        Raises:
            ValueError: If texts is empty.
        """
        if not texts:
            raise ValueError("texts must not be empty.")

        model = self._get_model()
        # encode() returns a numpy ndarray; convert to plain Python lists
        # so the result is framework-independent and Pydantic-serializable.
        vectors = model.encode(texts, batch_size=self._batch_size, show_progress_bar=False)
        return [vec.tolist() for vec in vectors]

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _get_model(self) -> SentenceTransformer:
        """Return the cached model, loading it on first access."""
        if self._model is None:
            logger.info("Loading embedding model '%s'...", self._model_name)
            self._model = SentenceTransformer(self._model_name)
            logger.info(
                "Embedding model '%s' loaded. Dimension: %d.",
                self._model_name,
                self._model.get_sentence_embedding_dimension(),
            )
        return self._model
