from abc import ABC, abstractmethod
from llm.models import AIResponse
from input.models.transcription import Transcription

class LanguageModel(ABC):
    """Abstract base class defining the lifecycle and generation interface for language models."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the client or resources. Called once at startup."""
        pass

    @abstractmethod
    async def generate(self, transcription: Transcription) -> AIResponse:
        """Generates an AIResponse for the given Transcription."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close/release any open resources (like HTTP client connection pools). Called on shutdown."""
        pass
