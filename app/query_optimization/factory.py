"""
Query Optimization — Factory

Constructs a fully wired QueryOptimizer using the standard banking dictionaries.
Callers never depend on RuleBasedQueryOptimizer directly — only on QueryOptimizer.
"""

from app.query_optimization.dictionaries import (
    BANKING_KEYWORD_EXPANSIONS,
    PHRASE_NORMALIZATIONS,
    PRODUCT_ALIASES,
)
from app.query_optimization.interfaces import QueryOptimizer
from app.query_optimization.rule_based import RuleBasedQueryOptimizer


def build_query_optimizer() -> QueryOptimizer:
    """
    Return a fully wired RuleBasedQueryOptimizer.

    The concrete type is an implementation detail hidden behind QueryOptimizer.
    """
    return RuleBasedQueryOptimizer(
        phrase_normalizations=PHRASE_NORMALIZATIONS,
        keyword_expansions=BANKING_KEYWORD_EXPANSIONS,
        product_aliases=PRODUCT_ALIASES,
    )
