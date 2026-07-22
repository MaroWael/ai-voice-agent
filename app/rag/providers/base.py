from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """
    Abstract interface for RAG LLM communication (Asynchronous).
    """

    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """
        Sends the prompt to the language model asynchronously and returns the raw string response.

        Args:
            prompt: Combined system instructions, context, and user question.
        """
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize connection clients or resources."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close and release connection clients or resources."""
        pass
