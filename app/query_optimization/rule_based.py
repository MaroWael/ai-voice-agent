"""
Query Normalization — Rule-Based Normalizer

Pure, deterministic, language-agnostic query normalizer. No AI, no network calls.

Pipeline:
  1. Guard: empty / whitespace-only input -> return immediately
  2. Unicode NFC normalization
  3. Safe ASCII punctuation cleanup
  4. Whitespace normalization (collapse runs of whitespace to single space)

Contains ZERO business-specific knowledge, product aliases, or manual dictionaries.
Language is preserved.
"""

import logging
import re
import unicodedata

from app.query_optimization.interfaces import QueryNormalizer

logger = logging.getLogger(__name__)

# ASCII punctuation characters safe to strip.
# Arabic-specific characters (e.g. ، ؟ ؛) are preserved.
_ASCII_PUNCTUATION_RE = re.compile(r'[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_{|}~`]+')

# One or more whitespace characters
_WHITESPACE_RE = re.compile(r"\s+")


class RuleBasedQueryNormalizer(QueryNormalizer):
    """
    Language-agnostic query normalizer for Arabic/English queries.
    """

    async def normalize(self, query: str) -> str:
        """
        Return a clean, normalized version of *query*.
        """
        if not query or not query.strip():
            return ""

        logger.debug("QueryNormalizer input: %r", query)

        text = self._normalize_unicode(query)
        text = self._strip_punctuation(text)
        text = self._normalize_whitespace(text)

        logger.debug("QueryNormalizer output: %r", text)
        return text

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_unicode(text: str) -> str:
        """Apply NFC normalization."""
        return unicodedata.normalize("NFC", text)

    @staticmethod
    def _strip_punctuation(text: str) -> str:
        """Remove safe ASCII punctuation."""
        return _ASCII_PUNCTUATION_RE.sub(" ", text)

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Collapse multiple whitespace characters into a single space."""
        return _WHITESPACE_RE.sub(" ", text).strip()


# Backwards compatibility alias
RuleBasedQueryOptimizer = RuleBasedQueryNormalizer
