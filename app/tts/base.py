from abc import ABC, abstractmethod


class SpeechSynthesizer(ABC):
    """
    Abstract interface for converting text to speech.
    """

    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        """
        Synthesize text to speech asynchronously, returning the audio bytes.

        Args:
            text: The plain text response to be synthesized.

        Returns:
            The raw audio bytes returned by the TTS provider.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        Close/release any open resources (like HTTP client connection pools). Called on shutdown.
        """
        pass
