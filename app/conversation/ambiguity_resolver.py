"""
Ambiguity Resolver

Detects multi-attribute collisions (e.g. card annual fees vs branch transaction fees)
when topic transitions collide.
"""

import logging
from typing import Optional

from app.conversation.models import ConversationState, Topic
from app.conversation.text_normalizer import TextNormalizer

logger = logging.getLogger(__name__)


class AmbiguityResolver:
    """
    Detects topic and attribute ambiguity in user queries.
    """

    def resolve_ambiguity(self, query: str, state: ConversationState) -> tuple[bool, Optional[str]]:
        """
        Evaluates *query* against *state* for ambiguity.

        Returns:
            Tuple of (is_ambiguous, ambiguity_reason)
        """
        if not query:
            return False, None

        norm_query = TextNormalizer.normalize(query)

        # Ambiguity Case 1: "fees" / "مصاريف" when active topic was BRANCHES but entity exists
        if "رسوم" in norm_query or "مصاريف" in norm_query or "fees" in norm_query:
            if state.active_topic == Topic.BRANCHES and state.entity_stack.peek() is not None:
                logger.info("AmbiguityResolver: Detected ambiguous fee query (Card fee vs Branch fee)")
                return True, "AMBIGUOUS_CARD_VS_BRANCH_FEE"

        return False, None
