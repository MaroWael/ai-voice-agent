"""
No-Op Translation Service

Returns the input text unchanged. Zero latency, zero external calls.
"""

from app.translation.interfaces import TranslationService


class NoTranslationService(TranslationService):
    """
    Default no-op translation service.
    """

    async def translate(
        self,
        text: str,
        source_lang: str = "auto",
        target_lang: str = "en",
    ) -> str:
        return text
