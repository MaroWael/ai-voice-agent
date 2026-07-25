"""
LLM Provider Factory

Constructs and returns an LLMProvider instance based on configuration settings.
Supports easy A/B comparison between Ollama and Groq providers.
"""

import logging

from app.config.settings import settings
from app.rag.providers.base import LLMProvider
from app.rag.providers.groq_provider import GroqProvider
from app.rag.providers.ollama_provider import OllamaRagProvider

logger = logging.getLogger(__name__)


def build_llm_provider() -> LLMProvider:
    """
    Return a fully constructed LLMProvider instance matching settings.LLM_PROVIDER.
    """
    provider_type = (settings.LLM_PROVIDER or "ollama").strip().lower()

    if provider_type == "groq":
        logger.info("Configuring LLM provider: GroqProvider (model: %s)", settings.GROQ_MODEL)
        return GroqProvider()

    logger.info("Configuring LLM provider: OllamaRagProvider (model: %s)", settings.LLM_MODEL)
    return OllamaRagProvider()
