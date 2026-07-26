"""
Query Enhancement — LLM-Based Search Query Enhancer

Uses an LLMProvider to rewrite conversational, noisy, or dialectal user queries
into concise search-optimized keyword strings for vector retrieval recovery.
"""

import logging
from app.query_optimization.interfaces import QueryEnhancer
from app.rag.providers.base import LLMProvider

logger = logging.getLogger(__name__)

ENHANCER_SYSTEM_PROMPT = """You are a search query optimizer for a customer service vector retrieval system.
Your ONLY responsibility is to rewrite conversational, dialectal, or multi-lingual user queries into concise, search-optimized keyword strings for dense vector retrieval.

CRITICAL RULES:
1. Strip all conversational filler and politeness phrases (e.g., "ايه هي", "عايز اعرف", "ممكن اعرف", "قولي", "can you tell me", "I want to know", "what is the", "tell me about").
2. Normalize dialect Arabic into clear search intent and bridge language differences between user speech and knowledge base content:
   - Convert dialect terms into standard domain concepts and include both English domain terms and Arabic keywords matching potential knowledge base documents.
   - General domain mappings to apply dynamically:
     - Card / Banking terms ("فيزا", "كارت", "بطاقة") -> include "Credit Card" "Visa" / "Card"
     - Fees / charges ("رسوم", "مصاريف", "تكلفة", "كم سعر") -> include "fees charges"
     - Benefits / features ("مميزات", "فوائد", "منافع") -> include "benefits features"
     - Withdrawal limits ("حد السحب", "الحد الأقصى للسحب", "سحب") -> include "cash withdrawal limit"
3. Strictly PRESERVE all product names, tier names, and numbers from the user query EXACTLY as mentioned (e.g., "Gold", "جولد", "Platinum", "بلاتينيوم", "Titanium", "تيتانيوم", "Classic", "كلاسيك").
4. NEVER invent or assume specific products, card tiers, or entities not mentioned or implied by the query.
5. NEVER map unknown or unrelated entities (e.g. "مدرسة بلاتينام" must NOT be mapped to a credit card).
6. NEVER answer the query, summarize information, or output explanations, quotes, punctuation, or preamble.
7. Output ONLY a single line of concise search keywords optimized for vector search.

EXAMPLES:

Input: ايه هي رسوم الفيزا الجولد
Output: Gold Credit Card fees charges رسوم ومصاريف بطاقة جولد

Input: عايز اعرف مميزات الكارت
Output: Credit Card benefits features مميزات البطاقة

Input: كام الحد الأقصى للسحب
Output: Cash withdrawal limit maximum withdrawal amount حد السحب

Input: ممكن اعرف مصاريف بطاقة البلاتينيوم
Output: Platinum Credit Card fees charges مصاريف بطاقة بلاتينيوم

Input: What are gold card fees
Output: Gold Credit Card fees charges

Input: مقر مدرسة بلاتينام
Output: مقر مدرسة بلاتينام

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
