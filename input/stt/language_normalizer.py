import re
from typing import Optional


# Mapping of raw provider language names/codes to ISO 639-1 2-letter codes
LANGUAGE_CODE_MAP = {
    "english": "en",
    "en": "en",
    "eng": "en",
    "arabic": "ar",
    "ar": "ar",
    "ara": "ar",
}


def _has_arabic_script(text: str) -> bool:
    """Returns True if text contains Arabic characters."""
    return bool(re.search(r"[\u0600-\u06FF]", text))


def _has_latin_script(text: str) -> bool:
    """Returns True if text contains Latin alphabet characters."""
    return bool(re.search(r"[a-zA-Z]", text))


def normalize_stt_language(text: str, raw_language: Optional[str]) -> str:
    """
    Normalizes provider STT language output into a standard 2-letter ISO code ('en' or 'ar').
    Validates provider metadata against the actual script in the transcription text.

    Args:
        text: The transcribed text string.
        raw_language: Raw language string or code returned by STT provider (e.g. 'Arabic', 'English', 'ar', 'en').

    Returns:
        Normalized 2-letter ISO code: 'en' or 'ar'.
    """
    # 1. Normalize raw language code if provided
    candidate_lang: Optional[str] = None
    if raw_language:
        cleaned_raw = raw_language.strip().lower()
        candidate_lang = LANGUAGE_CODE_MAP.get(cleaned_raw)

    # 2. Script detection on transcribed text
    has_ar = _has_arabic_script(text)
    has_latin = _has_latin_script(text)

    # 3. Validation & Conflict Resolution
    # If text is strictly Arabic script (contains Arabic and no Latin), force 'ar'
    if has_ar and not has_latin:
        return "ar"

    # If text is strictly Latin script (contains Latin and no Arabic), force 'en'
    if has_latin and not has_ar:
        return "en"

    # If text contains mixed script or no alphabetic characters (e.g. numbers only):
    # Fallback to candidate_lang if mapped, otherwise default to 'ar'
    if candidate_lang in ("en", "ar"):
        return candidate_lang

    return "ar"
