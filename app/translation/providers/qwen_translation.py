"""
Qwen LLM Translation Service

Translates queries using Qwen LLM via Ollama provider.
"""

import logging
from app.rag.providers.base import LLMProvider
from app.translation.interfaces import TranslationService

logger = logging.getLogger(__name__)


class QwenTranslationService(TranslationService):
    """
    Translates text using an underlying LLMProvider.
    """

    def __init__(self, llm_provider: LLMProvider) -> None:
        self._llm_provider = llm_provider

    async def translate(
        self,
        text: str,
        source_lang: str = "auto",
        target_lang: str = "en",
    ) -> str:
        if not text or not text.strip():
            return ""

        prompt = (
            f"You are a professional banking translator.\n"
            f"Translate the following customer query from {source_lang} to {target_lang}.\n"
            f"Preserve product names and financial numbers accurately.\n"
            f"Output ONLY the translated text without explanations or quotes.\n\n"
            f"Query: {text}\nTranslation:"
        )
        try:
            translated = await self._llm_provider.generate(prompt)
            translated_clean = translated.strip().strip('"').strip("'")
            logger.debug("QwenTranslation input: %r -> output: %r", text, translated_clean)
            return translated_clean if translated_clean else text
        except Exception as exc:
            logger.warning("QwenTranslation failed for text %r: %s", text, exc)
            return text
