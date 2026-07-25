"""
Translation Service — Interface

Defines abstract contract for query translation services in RAG pipeline.
Decouples application code from translation provider implementations.
"""

from abc import ABC, abstractmethod


class TranslationService(ABC):
    """
    Abstract base for text translation services.
    """

    @abstractmethod
    async def translate(
        self,
        text: str,
        source_lang: str = "auto",
        target_lang: str = "en",
    ) -> str:
        """
        Translate *text* from source_lang to target_lang.

        Args:
            text: Text to translate.
            source_lang: Source language code ('ar', 'en', 'auto').
            target_lang: Target language code ('en', 'ar').

        Returns:
            Translated text string, or original text if translation fails/is disabled.
        """
        pass
