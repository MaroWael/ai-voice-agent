"""
Query Optimization — Rule-Based Optimizer

Pure, deterministic query optimizer. No AI, no network calls.

Pipeline (applied in order):
  1. Guard: empty / whitespace-only input → return immediately
  2. Unicode NFC normalization
  3. Punctuation stripping (ASCII punctuation only — Arabic chars are preserved)
  4. Whitespace normalization (collapse runs of whitespace to single space)
  5. Phrase normalization (dictionary lookup, longest match first)
  6. Keyword expansion (append synonyms after each matched token)
  7. Product alias expansion (replace alias with canonical name fragment)
  8. Ordered duplicate-token removal (preserve first occurrence)

The user's language is always preserved — Arabic text is never translated.
The optimizer only widens token coverage for the embedding model.
"""

import logging
import re
import unicodedata

from app.query_optimization.interfaces import QueryOptimizer

logger = logging.getLogger(__name__)

# ASCII punctuation characters that are safe to strip.
# Arabic-specific characters (e.g. ، ؟ ؛) are intentionally not included
# because they carry meaning in Arabic text and their removal could distort
# the query for the embedding model.
_ASCII_PUNCTUATION_RE = re.compile(r'[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_{|}~`]+')

# One or more whitespace characters (including non-breaking spaces)
_WHITESPACE_RE = re.compile(r"\s+")


class RuleBasedQueryOptimizer(QueryOptimizer):
    """
    Lightweight, rule-based optimizer for Arabic/English banking queries.

    All dictionaries are injected at construction time so they can be
    swapped or extended without subclassing.
    """

    def __init__(
        self,
        phrase_normalizations: dict[str, str],
        keyword_expansions: dict[str, list[str]],
        product_aliases: dict[str, str],
    ) -> None:
        # Sort phrase keys by length descending so longer matches take priority.
        self._phrases: list[tuple[str, str]] = sorted(
            phrase_normalizations.items(), key=lambda kv: len(kv[0]), reverse=True
        )
        self._keyword_expansions = keyword_expansions
        # Sort alias keys by length descending for the same reason.
        self._product_aliases: list[tuple[str, str]] = sorted(
            product_aliases.items(), key=lambda kv: len(kv[0]), reverse=True
        )

    async def optimize(self, query: str) -> str:
        """
        Return a retrieval-optimized version of *query*.

        The original query is never modified in place — each step produces
        a new string. The caller (RagService) retains the original for the
        PromptBuilder.
        """
        if not query or not query.strip():
            return ""

        logger.debug("QueryOptimizer input: %r", query)

        text = self._normalize_unicode(query)
        text = self._strip_punctuation(text)
        text = self._normalize_whitespace(text)
        text = self._apply_phrase_normalizations(text)
        text = self._expand_keywords(text)
        text = self._expand_product_aliases(text)
        text = self._remove_duplicates(text)
        text = self._normalize_whitespace(text)

        logger.debug("QueryOptimizer output: %r", text)
        return text

    # ------------------------------------------------------------------
    # Private steps
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_unicode(text: str) -> str:
        """Apply NFC normalization to unify composed/decomposed Arabic forms."""
        return unicodedata.normalize("NFC", text)

    @staticmethod
    def _strip_punctuation(text: str) -> str:
        """Remove ASCII punctuation. Arabic punctuation is preserved."""
        return _ASCII_PUNCTUATION_RE.sub(" ", text)

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Collapse multiple whitespace characters into a single space."""
        return _WHITESPACE_RE.sub(" ", text).strip()

    def _apply_phrase_normalizations(self, text: str) -> str:
        """
        Replace conversational filler phrases with their canonical forms.
        Longest phrases are matched first to avoid partial replacements.
        Case-insensitive match on Latin characters; Arabic is case-invariant.
        """
        lower = text.lower()
        for phrase, replacement in self._phrases:
            phrase_lower = phrase.lower()
            idx = lower.find(phrase_lower)
            if idx != -1:
                text = text[:idx] + replacement + text[idx + len(phrase):]
                lower = text.lower()
        return text

    def _expand_keywords(self, text: str) -> str:
        """
        Append synonym expansions after each recognized keyword token.
        Only exact whole-word matches trigger expansion.
        """
        tokens = text.split()
        result: list[str] = []
        already_expanded: set[str] = set()

        for token in tokens:
            result.append(token)
            token_lower = token.lower()
            if token_lower in self._keyword_expansions and token_lower not in already_expanded:
                expansions = self._keyword_expansions[token_lower]
                result.extend(expansions)
                already_expanded.add(token_lower)

        # Also try multi-word keys (up to 3 tokens).
        return self._expand_multiword_keywords(" ".join(result))

    def _expand_multiword_keywords(self, text: str) -> str:
        """Handle multi-word keyword keys (e.g. 'credit limit', 'cash withdrawal')."""
        for key, expansions in self._keyword_expansions.items():
            if " " not in key:
                continue  # single-word keys already handled
            key_lower = key.lower()
            if key_lower in text.lower():
                # Append expansions to the end (they're not yet in text).
                for exp in expansions:
                    if exp.lower() not in text.lower():
                        text = text + " " + exp
        return text

    def _expand_product_aliases(self, text: str) -> str:
        """
        Replace informal product aliases with the canonical name fragment.
        Longest aliases are matched first; match is case-insensitive.
        """
        lower = text.lower()
        for alias, canonical in self._product_aliases:
            alias_lower = alias.lower()
            idx = lower.find(alias_lower)
            if idx != -1:
                # Replace alias with canonical name in original-case text.
                text = text[:idx] + canonical + text[idx + len(alias):]
                lower = text.lower()
        return text

    @staticmethod
    def _remove_duplicates(text: str) -> str:
        """
        Remove duplicate tokens while preserving the order of first occurrence.
        Case-insensitive comparison so 'Fees' and 'fees' are considered the same.
        """
        seen: set[str] = set()
        unique: list[str] = []
        for token in text.split():
            key = token.lower()
            if key not in seen:
                seen.add(key)
                unique.append(token)
        return " ".join(unique)
