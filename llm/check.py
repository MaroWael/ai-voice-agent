import logging
from app.config.settings import settings

logger = logging.getLogger(__name__)


async def check_llm() -> None:
    """
    Validates that the Groq LLM provider is properly configured.

    A full connectivity test is not performed here because the GroqProvider
    HTTP client is initialized (and validated) inside RagService.initialize(),
    which is called by RagLanguageModel.initialize() during the first request.

    This check guards against missing configuration at startup time.
    """
    if not settings.GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. "
            "Set GROQ_API_KEY in your .env file before starting the server."
        )

    if not settings.GROQ_MODEL:
        raise RuntimeError(
            "GROQ_MODEL is not configured. "
            "Set GROQ_MODEL in your .env file before starting the server."
        )

    logger.info(
        "LLM provider configuration OK — Groq model: '%s'", settings.GROQ_MODEL
    )
