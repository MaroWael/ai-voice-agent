"""
Query Enhancement — LLM-Based Search Query Enhancer

Uses an LLMProvider to rewrite conversational, noisy, or dialectal user queries
into concise search-optimized keyword strings for vector retrieval recovery.
"""

import logging
from app.query_optimization.interfaces import QueryEnhancer
from app.rag.providers.base import LLMProvider

logger = logging.getLogger(__name__)

ENHANCER_SYSTEM_PROMPT = """You are a search query optimizer for a banking customer service vector search system.
Your job is to rewrite conversational, informal, or dialectal user queries into a concise, search-optimized keyword query in English.

Instructions:
1. Strip all conversational filler, politeness phrases, and question frames (e.g. "عايز اسأل عن", "ممكن أعرف", "ايه هي", "لو سمحت", "قولي", "عايز اعرف", "كام", "I want to know", "can you tell me").
2. Retain core entity names, product types, card levels, and financial intent (e.g. Gold, Platinum, Classic, fees, charges, limit, benefits).
3. Translate Arabic product and financial terms to standard English search terms (e.g., "فيزا" / "بطاقة" -> "Credit Card", "مصاريف" / "رسوم" -> "fees charges", "الجولد" -> "Gold", "البلاتينيوم" -> "Platinum", "كلاسيك" -> "Classic").
4. Output ONLY the search query string in English. Do NOT add explanations, quotes, punctuation, or preamble.

Examples:
Input: ايه هي مصاريف الفيزا الجولد
Output: Gold Visa Credit Card fees charges

Input: كام رسوم بطاقة البلاتينيوم
Output: Platinum Credit Card fees charges

Input: عايز اعرف مصاريف بطاقة كلاسيك
Output: Classic Credit Card fees charges

User Query: {query}
Output:"""


class LLMQueryEnhancer(QueryEnhancer):
    """
    LLM-driven query enhancer for low-confidence retrieval recovery.
    """

    def __init__(self, llm_provider: LLMProvider) -> None:
        self._llm_provider = llm_provider

    async def enhance(self, query: str) -> str:
        """
        Rewrite *query* to optimize for vector search retrieval.
        """
        if not query or not query.strip():
            return query

        prompt = ENHANCER_SYSTEM_PROMPT.format(query=query)
        try:
            enhanced = await self._llm_provider.generate(prompt)
            # Clean output of quotes, markdown, or accidental extra lines
            enhanced = enhanced.strip().strip('"').strip("'").split("\n")[0].strip()
            logger.info("LLMQueryEnhancer: original=%r -> enhanced=%r", query, enhanced)
            return enhanced if enhanced else query
        except Exception as exc:
            logger.warning("LLMQueryEnhancer failed (%s). Falling back to raw query.", exc)
            return query
