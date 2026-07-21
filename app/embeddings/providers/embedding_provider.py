from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """
    Structural interface for embedding model providers.

    Any class that implements `embed(texts)` satisfies this protocol —
    no inheritance required. This keeps concrete providers decoupled from
    the abstraction while still allowing isinstance checks in tests.

    Implementations must:
    - Accept a batch of texts and return one embedding per text.
    - Never embed texts one by one inside a loop.
    - Be safe to call from an async context (via asyncio.to_thread).
    """

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: Non-empty list of strings to embed.

        Returns:
            A list of float vectors, one per input text, in the same order.
        """
        ...
