"""
Text Normalizer

Provides text normalization for Egyptian Arabic and English speech transcripts.
Prepares clean, normalized text strings for intent routing, entity extraction,
topic classification, and alias matching.
"""

import re


class TextNormalizer:
    """
    Normalizes Arabic and English text for robust downstream NLP matching.
    """

    @staticmethod
    def normalize(text: str) -> str:
        """
        Applies full text normalization pipeline:
        1. Strips leading/trailing whitespace
        2. Lowercases English text
        3. Removes Arabic tatweel (kashida)
        4. Normalizes Arabic alef variants (أ, إ, آ -> ا)
        5. Normalizes Arabic alef maqsura (ى -> ي) and teh marbuta (ة -> ه)
        6. Collapses 3+ character repetitions to single character
        7. Collapses redundant whitespace
        """
        if not text:
            return ""

        res = text.strip()

        # Lowercase English letters
        res = res.lower()

        # Remove Arabic tatweel (kashida: \u0640)
        res = res.replace("\u0640", "")

        # Normalize Alef forms (أ, إ, آ -> ا)
        res = re.sub(r"[\u0622\u0623\u0625]", "\u0627", res)

        # Normalize Alef Maqsura (ى -> ي)
        res = res.replace("\u0649", "\u064a")

        # Normalize Teh Marbuta (ة -> ه) for uniform matching
        res = res.replace("\u0629", "\u0647")

        # Collapse 3 or more repeated characters down to a single character (e.g., "الفييييزااا" -> "الفيزا")
        res = re.sub(r"(.)\1{2,}", r"\1", res)

        # Collapse multiple spaces
        res = re.sub(r"\s+", " ", res).strip()

        return res
