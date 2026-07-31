"""
Reference & Anaphora Resolver

Resolves conversational pronouns, ordinal references ("the first one", "the second one", "التانية", "الأولى"),
and chronological references ("go back to the first card") against state.comparison_state and state.entity_stack.
"""

import logging
import re
from typing import Optional

from app.conversation.models import ConversationEntity, ConversationState
from app.conversation.text_normalizer import TextNormalizer

logger = logging.getLogger(__name__)

_PRONOUN_PATTERNS = re.compile(
    r"\b(?:it|its|this|that|them|those|بتاعها|بتاعه|بتاعته|دي|ده|هي|هو|بها|عنها|عليها|فيها|منها|الاتنين|كلاهما|both)\b",
    re.IGNORECASE,
)

_ORDINAL_FIRST_PATTERNS = re.compile(
    r"\b(?:first|first one|الأولى|الاولى|الأول|الاول|الأولاني|الاولاني)\b",
    re.IGNORECASE,
)

_ORDINAL_SECOND_PATTERNS = re.compile(
    r"\b(?:second|second one|الثانية|الثانيه|الثاني|التانية|التانيه|التاني|والتانية|والتانيه|والتاني)\b",
    re.IGNORECASE,
)

_ORDINAL_THIRD_PATTERNS = re.compile(
    r"\b(?:third|third one|الثالثة|الثالثه|الثالث|التالتة|التالته|التالت)\b",
    re.IGNORECASE,
)

_CHRONOLOGICAL_FIRST_PATTERNS = re.compile(
    r"(?:go back to the first|first one we discussed|نرجع لأول|نرجع لاول|أول كارت اتكلمنا عليه|اول كارت اتكلمنا عليه)",
    re.IGNORECASE,
)


class ReferenceResolver:
    """
    Deterministically resolves anaphora and ordinal references against state.comparison_state and state.entity_stack.
    """

    def resolve_reference(self, text: str, state: ConversationState) -> tuple[bool, Optional[ConversationEntity]]:
        """
        Resolves reference in *text* against state.comparison_state or state.entity_stack.

        Returns:
            Tuple of (is_reference_resolved, resolved_entity)
        """
        norm_text = TextNormalizer.normalize(text)

        # 1. Chronological First Reference Check BEFORE Ordinal Check
        if _CHRONOLOGICAL_FIRST_PATTERNS.search(text) or _CHRONOLOGICAL_FIRST_PATTERNS.search(norm_text):
            first_entity = state.entity_stack.get_chronological_first()
            if first_entity:
                logger.info("ReferenceResolver: Chronological first resolved -> %s", first_entity.display_name)
                return True, first_entity

        # 2. Active ComparisonState check for ordinals if comparison is active
        comp_ents = state.comparison_state.compared_entities if (state.comparison_state and state.comparison_state.compared_entities) else []

        # 2a. Ordinal First Reference
        if _ORDINAL_FIRST_PATTERNS.search(text) or _ORDINAL_FIRST_PATTERNS.search(norm_text):
            if comp_ents and len(comp_ents) >= 1:
                logger.info("ReferenceResolver: Ordinal first resolved from ComparisonState -> %s", comp_ents[0].display_name)
                return True, comp_ents[0]
            entity = state.entity_stack.get_by_index(0)
            if entity:
                logger.info("ReferenceResolver: Ordinal first resolved -> %s", entity.display_name)
                return True, entity

        # 2b. Ordinal Second Reference
        if _ORDINAL_SECOND_PATTERNS.search(text) or _ORDINAL_SECOND_PATTERNS.search(norm_text):
            if comp_ents and len(comp_ents) >= 2:
                logger.info("ReferenceResolver: Ordinal second resolved from ComparisonState -> %s", comp_ents[1].display_name)
                return True, comp_ents[1]
            entity = state.entity_stack.get_by_index(1)
            if entity:
                logger.info("ReferenceResolver: Ordinal second resolved -> %s", entity.display_name)
                return True, entity

        # 2c. Ordinal Third Reference
        if _ORDINAL_THIRD_PATTERNS.search(text) or _ORDINAL_THIRD_PATTERNS.search(norm_text):
            if comp_ents and len(comp_ents) >= 3:
                logger.info("ReferenceResolver: Ordinal third resolved from ComparisonState -> %s", comp_ents[2].display_name)
                return True, comp_ents[2]
            entity = state.entity_stack.get_by_index(2)
            if entity:
                logger.info("ReferenceResolver: Ordinal third resolved -> %s", entity.display_name)
                return True, entity

        # 3. General Pronoun Reference (Top of Stack)
        if _PRONOUN_PATTERNS.search(text) or _PRONOUN_PATTERNS.search(norm_text):
            top_entity = state.entity_stack.peek()
            if top_entity:
                logger.info("ReferenceResolver: Pronoun resolved -> %s", top_entity.display_name)
                return True, top_entity

        return False, None
