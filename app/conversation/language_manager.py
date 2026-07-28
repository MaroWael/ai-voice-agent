"""
Conversation Language Manager

Per-turn independent language evaluation and fallback manager.
Evaluates the user's CURRENT message language via script analysis and text pattern detection,
falling back to previous response_language only when detection confidence is < 0.60.
"""

import logging
import re
from typing import Optional

from app.conversation.models import ConversationState, Language

logger = logging.getLogger(__name__)

_ARABIC_SCRIPT_RE = re.compile(r"[\u0600-\u06FF]")
_LATIN_SCRIPT_RE = re.compile(r"[a-zA-Z]")


class ConversationLanguageManager:
    """
    Evaluates per-turn language independently.
    Separates detected_language from response_language for clean observability.
    """

    def __init__(self, fallback_threshold: float = 0.60) -> None:
        self.fallback_threshold = fallback_threshold

    def evaluate_turn_language(
        self,
        state: ConversationState,
        text: str,
        whisper_language: Optional[str] = None,
    ) -> tuple[Language, Language, float]:
        """
        Evaluates the current turn's language independently.

        Returns:
            Tuple of (detected_language, response_language, confidence_score)
        """
        clean_text = text.strip() if text else ""

        if not clean_text:
            return state.detected_language, state.response_language, state.language_confidence if hasattr(state, 'language_confidence') else 1.0

        has_arabic = bool(_ARABIC_SCRIPT_RE.search(clean_text))
        has_latin = bool(_LATIN_SCRIPT_RE.search(clean_text))

        detected_lang: Language
        confidence: float

        # ── 1. Per-Turn Script Analysis ─────────────────────────────────────
        if has_arabic:
            # Arabic script is dominant (even with mixed terms like "عايز اعرف Visa Platinum")
            detected_lang = Language.ARABIC
            confidence = 0.98 if not has_latin else 0.90
        elif has_latin and not has_arabic:
            # Pure Latin script (e.g. "What are the fees?", "Hello")
            detected_lang = Language.ENGLISH
            confidence = 0.95
        else:
            # Ambiguous input (e.g. digits or symbols): check whisper metadata
            if whisper_language:
                cleaned_whisper = whisper_language.strip().lower()
                if cleaned_whisper in ("ar", "arabic", "ara"):
                    detected_lang = Language.ARABIC
                    confidence = 0.70
                elif cleaned_whisper in ("en", "english", "eng"):
                    detected_lang = Language.ENGLISH
                    confidence = 0.70
                else:
                    detected_lang = state.response_language
                    confidence = 0.50
            else:
                detected_lang = state.response_language
                confidence = 0.50

        # ── 2. Response Language Resolution ──────────────────────────────────
        # Fall back to previous response_language ONLY when confidence < fallback_threshold
        if confidence < self.fallback_threshold:
            logger.info(
                "Turn language confidence low (%.2f < %.2f). Falling back to previous response_language: %s",
                confidence,
                self.fallback_threshold,
                state.response_language.value,
            )
            response_lang = state.response_language
        else:
            response_lang = detected_lang

        state.detected_language = detected_lang
        state.response_language = response_lang

        logger.info(
            "Per-turn language evaluated -> Detected: %s, Response: %s (confidence: %.2f)",
            detected_lang.value,
            response_lang.value,
            confidence,
        )

        return detected_lang, response_lang, confidence
