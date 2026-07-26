from abc import ABC, abstractmethod
from typing import Optional


class BaseTextNormalizer(ABC):
    """
    Abstract base class for speech text normalizer strategies.
    Each normalizer performs a single responsibility transformation on the input text.
    """

    @abstractmethod
    def normalize(self, text: str, language: Optional[str] = None) -> str:
        """
        Transforms input text into a speech-friendly format.

        Args:
            text: The raw input text.
            language: Resolved ISO language code (e.g., 'en', 'ar') if available.

        Returns:
            The normalized text string.
        """
        pass
