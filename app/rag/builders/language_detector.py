"""
RAG Language Detector — Lightweight Rule-Based Language Detection

Detects the primary language of a user query to enforce language mirroring
in LLM responses without ML models or external dependencies.
"""

import re

# Unicode range for Arabic characters (including Arabic script extensions)
_ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")


def detect_query_language(text: str) -> str:
    """
    Detects whether the query is Arabic or English using character matching.

    Rules:
      - If the query contains any Arabic characters (even in code-switching / mixed queries),
        return 'ar' (Arabic) to prioritize Arabic for Egyptian market customers.
      - Otherwise, return 'en' (English).

    Args:
        text: The raw user question string.

    Returns:
        'ar' for Arabic/mixed queries, 'en' for English/other.
    """
    if not text or not text.strip():
        return "en"

    if _ARABIC_CHAR_RE.search(text):
        return "ar"

    return "en"


def get_language_instruction(lang_code: str) -> str:
    """
    Returns explicit system prompt language instructions based on the detected language code.
    """
    if lang_code == "ar":
        return (
            "LANGUAGE MANDATE:\n"
            "- The customer asked in Arabic (or Arabic-dominant mixed language).\n"
            "- You MUST answer entirely in Arabic.\n"
            "- Preserve brand and product names (e.g. Platinum, Visa, Mastercard, Gold, Titanium, Infinite, Signature, World) in English/standard form."
        )
    return (
        "LANGUAGE MANDATE:\n"
        "- The customer asked in English.\n"
        "- You MUST answer entirely in English."
    )
