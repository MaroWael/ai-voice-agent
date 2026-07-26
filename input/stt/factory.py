import logging
from typing import Optional

from app.config.settings import settings
from input.models.speech_segment import SpeechSegment
from input.models.transcription import Transcription
from input.stt.base import SpeechRecognizer
from input.stt.exceptions import STTProviderError
from input.stt.faster_whisper import FasterWhisperSTT
from input.stt.groq_whisper import GroqWhisperSTT

logger = logging.getLogger(__name__)


class FallbackSTTWrapper(SpeechRecognizer):
    """
    STT Provider wrapper that attempts primary transcription first,
    and falls back to a secondary provider if primary fails.
    """

    def __init__(self, primary: SpeechRecognizer, fallback: SpeechRecognizer) -> None:
        self.primary = primary
        self.fallback = fallback
        self._fallback_initialized = False

    async def initialize(self) -> None:
        """
        Initialize primary provider. If primary fails during startup, initialize fallback.
        """
        try:
            await self.primary.initialize()
        except Exception as exc:
            logger.warning(
                "Primary STT provider initialization failed: %s. Pre-initializing fallback provider...",
                exc,
            )
            await self.fallback.initialize()
            self._fallback_initialized = True

    async def transcribe(self, segment: SpeechSegment) -> Transcription:
        """
        Attempt transcription with primary provider. Fall back to secondary on failure.
        """
        try:
            return await self.primary.transcribe(segment)
        except (STTProviderError, Exception) as exc:
            logger.warning(
                "Primary STT provider transcription failed (%s). Falling back to secondary provider...",
                exc,
            )
            if not self._fallback_initialized:
                await self.fallback.initialize()
                self._fallback_initialized = True
            return await self.fallback.transcribe(segment)


def build_speech_recognizer(provider: Optional[str] = None) -> SpeechRecognizer:
    """
    Factory function to construct a SpeechRecognizer based on configuration or argument.

    Args:
        provider: Optional override ("groq" or "local"). If None, reads settings.STT_PROVIDER.

    Returns:
        An instance of SpeechRecognizer (or FallbackSTTWrapper if fallback is enabled).
    """
    selected_provider = (provider or settings.STT_PROVIDER).lower()

    if selected_provider == "groq":
        primary = GroqWhisperSTT()
        if settings.STT_FALLBACK_ENABLED:
            logger.info("Building GroqWhisperSTT with local FasterWhisperSTT fallback enabled")
            fallback = FasterWhisperSTT()
            return FallbackSTTWrapper(primary=primary, fallback=fallback)
        logger.info("Building GroqWhisperSTT without fallback")
        return primary

    elif selected_provider == "local":
        logger.info("Building local FasterWhisperSTT")
        return FasterWhisperSTT()

    else:
        logger.warning(
            "Unknown STT provider '%s' configured. Defaulting to local FasterWhisperSTT.",
            selected_provider,
        )
        return FasterWhisperSTT()
