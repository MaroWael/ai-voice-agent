import logging
from app.config.settings import settings
from app.tts.silma import SilmaTTSError

logger = logging.getLogger(__name__)


async def check_tts() -> None:
    """
    Verify that the Silma TTS configuration settings exist and are valid.
    Does NOT perform any network requests.
    """
    logger.info("Verifying Silma TTS configuration...")

    required_keys = {
        "SILMA_API_KEY": settings.SILMA_API_KEY,
        "SILMA_BASE_URL": settings.SILMA_BASE_URL,
        "SILMA_MODEL_ID": settings.SILMA_MODEL_ID,
        "SILMA_VOICE_ID": settings.SILMA_VOICE_ID,
    }

    missing = [key for key, value in required_keys.items() if not value or not str(value).strip()]

    if missing:
        raise SilmaTTSError(
            f"Silma TTS configuration check failed. Missing or empty settings: {', '.join(missing)}"
        )

    logger.info(
        "Silma TTS configuration check OK (Model: '%s', Voice: '%s')",
        settings.SILMA_MODEL_ID,
        settings.SILMA_VOICE_ID,
    )
