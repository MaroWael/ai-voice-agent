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
            "- Answer ONLY in natural Arabic. Never translate the answer to English.\n"
            "- Answer entirely in natural Egyptian banking Arabic (e.g. use 'رسوم الإصدار', 'رسوم التجديد', 'البطاقة الائتمانية', 'الحد الائتماني').\n"
            "- NEVER use literal or unnatural translations (e.g. NEVER write 'البطاقة التوافرية' or 'المتوازن').\n"
            "- Preserve English product names, brand terms, and currency codes verbatim (e.g. Platinum, Visa, Mastercard, Gold, Titanium, Points, EGP)."
        )
    return (
        "LANGUAGE MANDATE:\n"
        "- The customer asked in English.\n"
        "- Answer ONLY in clear, natural English. Never translate the answer to Arabic.\n"
        "- Do NOT default to Arabic.\n"
        "- Do NOT translate English banking terms into Arabic unless the user asked in Arabic.\n"
        "- Preserve product names, numbers, technical terms, and currency codes verbatim (e.g. EGP 500)."
    )
