"""
Query Normalization — Factory

Constructs a fully wired QueryNormalizer.
"""

from app.query_optimization.interfaces import QueryNormalizer
from app.query_optimization.rule_based import RuleBasedQueryNormalizer


def build_query_normalizer() -> QueryNormalizer:
    """
    Return a fully wired RuleBasedQueryNormalizer.
    """
    return RuleBasedQueryNormalizer()


# Backwards compatibility alias
build_query_optimizer = build_query_normalizer
