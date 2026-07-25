"""
Translation Factory

Constructs a TranslationService based on configuration settings.
"""

from app.config.settings import settings
from app.rag.providers.ollama_provider import OllamaRagProvider
from app.translation.interfaces import TranslationService
from app.translation.providers.no_translation import NoTranslationService
from app.translation.providers.qwen_translation import QwenTranslationService


def build_translation_service() -> TranslationService:
    """
    Constructs and returns the configured TranslationService implementation.
    """
    if not settings.TRANSLATION_ENABLED or settings.TRANSLATION_PROVIDER.lower() == "none":
        return NoTranslationService()

    if settings.TRANSLATION_PROVIDER.lower() == "qwen":
        return QwenTranslationService(llm_provider=OllamaRagProvider())

    return NoTranslationService()
