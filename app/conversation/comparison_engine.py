"""
Comparison Engine

Generates structured feature comparison matrices for multi-entity comparisons,
formats multi-product standalone search queries, and validates missing product information.
"""

import logging
from typing import Dict, List, Optional, Tuple

from app.conversation.models import ComparisonState, ConversationEntity, Language, Topic
from app.conversation.text_normalizer import TextNormalizer

logger = logging.getLogger(__name__)


class ComparisonEngine:
    """
    Manages multi-product comparisons.
    """

    def build_comparison_matrix(self, entities: List[ConversationEntity]) -> ComparisonState:
        """
        Builds a ComparisonState matrix for the given entities.
        """
        matrix: Dict[str, Dict[str, str]] = {}
        for ent in entities:
            meta = ent.metadata or {}
            matrix[ent.display_name] = {
                "minimum_salary": f"{meta.get('min_salary', 0):,} EGP",
                "annual_fee": f"{meta.get('annual_fee', 0):,} EGP",
                "lounge_access": f"{meta.get('lounge_access_count', 0)} visits",
                "cashback": f"{meta.get('cashback_rate', 0.0)*100}%",
            }

        logger.info("ComparisonEngine: Built comparison matrix for %d entities", len(entities))
        return ComparisonState(
            compared_entities=entities,
            focused_attribute=Topic.COMPARISON,
            comparison_matrix=matrix,
        )

    def format_comparison_query(self, entities: List[ConversationEntity], lang: Language) -> str:
        """
        Formats a standalone multi-entity comparison query string.
        """
        if len(entities) < 2:
            entity_names = entities[0].display_name if entities else "products"
            return f"Tell me about {entity_names}"

        names = [e.display_name for e in entities]
        if lang == Language.ARABIC:
            names_str = " و ".join(names)
            return f"مقارنة بين {names_str} من حيث الرسوم والمميزات والحد الأدنى للراتب وصالات المطار والتقسيط."
        else:
            names_str = " vs ".join(names)
            return f"Compare {names_str} features, fees, minimum salary, and airport lounge access."

    def validate_comparison_data(
        self,
        requested_entities: List[ConversationEntity],
        retrieved_entity_ids: List[str],
        lang: Language,
    ) -> Tuple[bool, Optional[str]]:
        """
        Checks if any requested comparison entity is missing from retrieved context (BUG #4).
        If missing, returns (is_incomplete=True, user_facing_warning_message).
        """
        if len(requested_entities) < 2:
            return False, None

        present = [e for e in requested_entities if e.id in retrieved_entity_ids or e.display_name.lower() in [r.lower() for r in retrieved_entity_ids]]
        missing = [e for e in requested_entities if e not in present]

        if missing and present:
            pres_name = present[0].display_name
            miss_name = missing[0].display_name
            if lang == Language.ARABIC:
                msg = f"لقد وجدت معلومات عن بطاقة {pres_name} ولكن لم أجد معلومات كافية عن بطاقة {miss_name} لإجراء مقارنة دقيقة."
            else:
                msg = f"I found information about {pres_name} but I couldn't find enough information about {miss_name} to generate a reliable comparison."
            logger.warning("ComparisonEngine: Missing comparison data for %s", miss_name)
            return True, msg

        return False, None
