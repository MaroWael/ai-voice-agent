import re
from typing import Optional

from app.speech_formatting.base import BaseTextNormalizer


class MarkdownNormalizer(BaseTextNormalizer):
    """
    Strips speech-unfriendly Markdown formatting (headers, bold, italics, code blocks, links)
    and formats bullet point lists into natural spoken sentences.
    """

    def normalize(self, text: str, language: Optional[str] = None) -> str:
        if not text:
            return ""

        # Remove code blocks
        result = re.sub(r"```[\s\S]*?```", "", text)
        result = re.sub(r"`([^`]+)`", r"\1", result)

        # Remove Markdown headers (# Header)
        result = re.sub(r"^#{1,6}\s*", "", result, flags=re.MULTILINE)

        # Remove Markdown links [text](url) -> text
        result = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", result)

        # Remove bold and italics (**text**, __text__, *text*, _text_)
        result = re.sub(r"(\*\*|__)(.*?)\1", r"\2", result)
        result = re.sub(r"(\*|_)(.*?)\1", r"\2", result)

        # Normalize bullet list items (•, -, *) into clean lines with trailing pause punctuation
        lines = result.splitlines()
        processed_lines = []
        for line in lines:
            stripped = line.strip()
            # Check if line starts with a bullet symbol or list marker (e.g. •, -, *, 1.)
            bullet_match = re.match(r"^(?:[•\-\*]|\d+[\.\)])\s*(.*)$", stripped)
            if bullet_match:
                item_content = bullet_match.group(1).strip()
                if item_content:
                    # Ensure item ends with a pause punctuation if not already present
                    if not item_content[-1] in ".!؟?:;":
                        item_content += "."
                    processed_lines.append(item_content)
            else:
                if stripped:
                    processed_lines.append(stripped)

        return "\n".join(processed_lines)


class AbbreviationAndCurrencyNormalizer(BaseTextNormalizer):
    """
    Generic currency and abbreviation expansion for speech synthesis.
    Converts written currency codes (EGP, LE, USD, EUR) to natural spoken language.
    Does NOT contain product-specific or banking-specific hardcoded entity logic.
    """

    # Generic currency patterns
    # Handles: EGP 500, 500 EGP, EGP500, E.G.P. 500, LE 500, 500 LE, L.E. 500
    EN_CURRENCY_REPLACEMENTS = [
        (r"(?i)\b(?:EGP|E\.G\.P\.|LE|L\.E\.)\s*(\d+(?:\.\d+)?)\b", r"\1 Egyptian Pounds"),
        (r"(?i)\b(\d+(?:\.\d+)?)\s*(?:EGP|E\.G\.P\.|LE|L\.E\.)\b", r"\1 Egyptian Pounds"),
        (r"(?i)\bUSD\s*(\d+(?:\.\d+)?)\b", r"\1 US Dollars"),
        (r"(?i)\b(\d+(?:\.\d+)?)\s*USD\b", r"\1 US Dollars"),
        (r"(?i)\bEUR\s*(\d+(?:\.\d+)?)\b", r"\1 Euros"),
        (r"(?i)\b(\d+(?:\.\d+)?)\s*EUR\b", r"\1 Euros"),
    ]

    AR_CURRENCY_REPLACEMENTS = [
        (r"(?i)\b(?:EGP|E\.G\.P\.|LE|L\.E\.)\s*(\d+(?:\.\d+)?)\b", r"\1 جنيه مصري"),
        (r"(?i)\b(\d+(?:\.\d+)?)\s*(?:EGP|E\.G\.P\.|LE|L\.E\.)\b", r"\1 جنيه مصري"),
        (r"(?i)\bUSD\s*(\d+(?:\.\d+)?)\b", r"\1 دولار أمريكي"),
        (r"(?i)\b(\d+(?:\.\d+)?)\s*USD\b", r"\1 دولار أمريكي"),
        (r"(?i)\bEUR\s*(\d+(?:\.\d+)?)\b", r"\1 يورو"),
        (r"(?i)\b(\d+(?:\.\d+)?)\s*EUR\b", r"\1 يورو"),
    ]

    def _contains_arabic(self, text: str) -> bool:
        """Returns True if the text contains Arabic script characters."""
        return bool(re.search(r"[\u0600-\u06FF]", text))

    def normalize(self, text: str, language: Optional[str] = None) -> str:
        if not text:
            return ""

        # Determine effective language for currency expansion
        is_arabic = False
        if language:
            is_arabic = language.lower().startswith("ar")
        else:
            is_arabic = self._contains_arabic(text)

        replacements = self.AR_CURRENCY_REPLACEMENTS if is_arabic else self.EN_CURRENCY_REPLACEMENTS

        result = text
        for pattern, replacement in replacements:
            result = re.sub(pattern, replacement, result)

        return result


class PunctuationAndWhitespaceNormalizer(BaseTextNormalizer):
    """
    Normalizes whitespace, line breaks, and punctuation spacing for speech synthesis cadence.
    """

    def normalize(self, text: str, language: Optional[str] = None) -> str:
        if not text:
            return ""

        # Replace multiple spaces/tabs with single space
        result = re.sub(r"[ \t]+", " ", text)

        # Replace multiple blank lines with single newline
        result = re.sub(r"\n\s*\n+", "\n", result)

        # Fix spacing around colons for natural pause reading (e.g. "Fee:500" -> "Fee: 500")
        result = re.sub(r"(:)(\S)", r"\1 \2", result)

        # Clean double periods or period spaces
        result = re.sub(r"\.\s*\.", ".", result)

        return result.strip()
