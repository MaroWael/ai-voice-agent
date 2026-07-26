import logging
from typing import List, Optional

from app.speech_formatting.base import BaseTextNormalizer
from app.speech_formatting.normalizers import (
    MarkdownNormalizer,
    AbbreviationAndCurrencyNormalizer,
    PunctuationAndWhitespaceNormalizer,
)

logger = logging.getLogger(__name__)


class SpeechResponseFormatter:
    """
    Generic Speech Response Formatting layer.
    Converts raw LLM text into speech-friendly text for TTS consumption.
    
    Language resolution priority:
      1. AIResponse.language / explicitly passed `language`
      2. transcription.language (`transcription_language`)
      3. Fallback script detection (Arabic vs English characters)
    """

    def __init__(self, normalizers: Optional[List[BaseTextNormalizer]] = None) -> None:
        self.normalizers: List[BaseTextNormalizer] = normalizers if normalizers is not None else [
            MarkdownNormalizer(),
            AbbreviationAndCurrencyNormalizer(),
            PunctuationAndWhitespaceNormalizer(),
        ]

    def format(
        self,
        text: str,
        language: Optional[str] = None,
        transcription_language: Optional[str] = None,
    ) -> str:
        """
        Formats text for TTS synthesis.

        Args:
            text: Raw response message string.
            language: Language code from AIResponse (highest priority).
            transcription_language: Language code from STT Transcription (fallback priority).

        Returns:
            Speech-ready text for TTS.
        """
        if not text:
            return ""

        # Language priority resolution: AIResponse.language -> transcription_language -> None (script fallback)
        resolved_language = language or transcription_language

        logger.debug(
            "SpeechResponseFormatter.format() — input_length=%d, resolved_language=%s",
            len(text),
            resolved_language,
        )

        result = text
        for normalizer in self.normalizers:
            result = normalizer.normalize(result, language=resolved_language)

        logger.debug("SpeechResponseFormatter.format() — formatted text: %r", result)
        return result
