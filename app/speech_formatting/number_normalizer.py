import logging
import re
from typing import Optional

from num2words import num2words

from app.config.settings import settings
from app.speech_formatting.base import BaseTextNormalizer

logger = logging.getLogger(__name__)

# Unicode range for Arabic characters
_ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF]")

# Regex to detect keywords preceding ID / Card / Account numbers that should NOT be converted to spoken words
_ID_OR_CARD_PREFIX_RE = re.compile(
    r"\b(?:card(?:\s+number)?|account|transaction|ref(?:erence)?|id|no\.|number|رقم|بطاقة|حساب|معاملة)\s*#?:?\s*$",
    re.IGNORECASE,
)

# Regex to match standalone numeric numbers in text
_NUMBER_TOKEN_RE = re.compile(r"\b\d+(?:,\d+)*(?:\.\d+)?\b")


class NumberSpeechNormalizer(BaseTextNormalizer):
    """
    Converts numeric digits into natural spoken words prior to TTS synthesis.

    Features:
    - Provider independent and language aware.
    - Respects settings.ENABLE_NUMBER_SPEECH_NORMALIZATION flag.
    - Safety rules:
        - Preserves long digits / card numbers (>= 6 digits or preceded by ID/Card keywords).
        - Preserves 4-digit years (e.g. 1900-2099 when preceded by year indicators like "in 2025", "عام 2025", "سنة 2025").
    - Converts numbers to English words when language is 'en' or text is Latin script.
    - Converts numbers to Arabic words when language is 'ar' or text contains Arabic script.
    """

    def normalize(self, text: str, language: Optional[str] = None) -> str:
        if not text or not text.strip():
            return text

        if not getattr(settings, "ENABLE_NUMBER_SPEECH_NORMALIZATION", True):
            return text

        # Language resolution: explicit parameter > text script detection
        is_arabic = (language == "ar") or bool(_ARABIC_CHAR_RE.search(text))
        lang_code = "ar" if is_arabic else "en"

        def _replace_number_match(match: re.Match) -> str:
            raw_str = match.group(0).replace(",", "")
            start_pos = match.start()

            prefix_text = text[:start_pos]

            # Rule 1: Preserve Card numbers / IDs / Long digits (>= 6 digits or preceded by ID keywords)
            if len(raw_str) >= 6 or _ID_OR_CARD_PREFIX_RE.search(prefix_text):
                return match.group(0)

            # Rule 2: Preserve 4-digit years (1900-2099 when preceded by in/year/عام/سنة)
            if len(raw_str) == 4 and raw_str.isdigit():
                val = int(raw_str)
                if 1900 <= val <= 2099:
                    year_prefix_re = re.compile(r"\b(?:in|year|since|عام|سنة)\s*$", re.IGNORECASE)
                    if year_prefix_re.search(prefix_text):
                        return match.group(0)

            try:
                if "." in raw_str:
                    num_val = float(raw_str)
                else:
                    num_val = int(raw_str)

                words = num2words(num_val, lang=lang_code)
                if lang_code == "en":
                    words = words.replace(" and ", " ")
                return words
            except Exception as exc:
                logger.warning("Failed to convert number %r to words (%s); leaving unchanged.", raw_str, exc)
                return match.group(0)

        normalized_text = _NUMBER_TOKEN_RE.sub(_replace_number_match, text)

        if normalized_text != text:
            logger.info(
                "Number speech normalization:\n"
                "  Before: %r\n"
                "  After:  %r",
                text,
                normalized_text,
            )

        return normalized_text
