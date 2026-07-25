"""
Query Normalization — Factory

Constructs a fully wired QueryNormalizer.
"""

from app.query_optimization.interfaces import QueryEnhancer, QueryNormalizer
from app.query_optimization.llm_enhancer import LLMQueryEnhancer
from app.query_optimization.rule_based import RuleBasedQueryNormalizer
from app.rag.providers.base import LLMProvider


def build_query_normalizer() -> QueryNormalizer:
    """
    Return a fully wired RuleBasedQueryNormalizer.
    """
    return RuleBasedQueryNormalizer()


def build_query_enhancer(llm_provider: LLMProvider) -> QueryEnhancer:
    """
    Return a fully wired LLMQueryEnhancer.
    """
    return LLMQueryEnhancer(llm_provider=llm_provider)


# Backwards compatibility alias
build_query_optimizer = build_query_normalizer
