"""
Entity and Topic Extractor

Combines entity matching and extensible topic classification for banking queries.
Extracts products, cards, loans, certificates, and accounts with alias matching
and fuzzy text normalization.

Maintains active entity context preservation across multi-turn exchanges while
continuously updating active topic per turn.
"""

import logging
import re
from typing import Optional

from app.conversation.models import ConversationEntity, ConversationState, EntityType, Topic
from app.conversation.text_normalizer import TextNormalizer

logger = logging.getLogger(__name__)

# Pre-configured banking entities registry
CANONICAL_ENTITIES = [
    {
        "id": "visa_platinum",
        "display_name": "Visa Platinum",
        "entity_type": EntityType.CREDIT_CARD,
        "aliases": ["Platinum", "Platinum Visa", "Visa Plat", "الفيزا البلاتينيوم", "البلاتينيوم", "فيزا بلاتينوم", "بلاتينيوم"],
    },
    {
        "id": "visa_gold",
        "display_name": "Visa Gold",
        "entity_type": EntityType.CREDIT_CARD,
        "aliases": ["Gold", "Gold Visa", "Visa Gold Card", "الفيزا الجولد", "الجولد", "فيزا جولد", "جولد"],
    },
    {
        "id": "visa_signature",
        "display_name": "Visa Signature",
        "entity_type": EntityType.CREDIT_CARD,
        "aliases": ["Signature", "Visa Signature Card", "الفيزا السيجنتشر", "سيجنتشر"],
    },
    {
        "id": "mastercard",
        "display_name": "Mastercard",
        "entity_type": EntityType.CREDIT_CARD,
        "aliases": ["Master Card", "ماستر كارد", "ماستركارد"],
    },
    {
        "id": "al_qemma_certificate",
        "display_name": "Al Qemma Certificate",
        "entity_type": EntityType.CERTIFICATE,
        "aliases": ["Al Qemma", "Qemma Certificate", "شهادة القمة", "شهاده القمه", "القمة", "قمة"],
    },
    {
        "id": "personal_loan",
        "display_name": "Personal Loan",
        "entity_type": EntityType.LOAN,
        "aliases": ["Loan", "Personal Loans", "القرض الشخصي", "قرض شخصي", "القرض", "قرض", "قروض"],
    },
    {
        "id": "savings_account",
        "display_name": "Savings Account",
        "entity_type": EntityType.ACCOUNT,
        "aliases": ["Savings", "Account", "حساب التوفير", "الحساب الشخصي", "الحساب", "توفير"],
    },
]

# Topic Keyword Patterns for Extensible Matching
_TOPIC_PATTERNS = {
    Topic.FEES: re.compile(
        r"\b(?:رسوم|مصاريف|مصروفات|عمولة|عموله|فوائد|فائدة|فائده|كام|مصاريفها|رسومها|fees|fee|cost|charge|charges|interest)\b",
        re.IGNORECASE,
    ),
    Topic.REPLACEMENT: re.compile(
        r"\b(?:ضاعت|ضاع|اتسرقت|اتسرق|فقدت|فقدان|استبدال|بديل|تجديد|lost|stolen|replace|replacement)\b",
        re.IGNORECASE,
    ),
    Topic.REQUIREMENTS: re.compile(
        r"\b(?:الأوراق|الاوراق|المستندات|الشروط|مرتب|المرتب|الحد الأدنى|الحد الادنى|سن|السن|شروط|مستندات|requirements|documents|eligible|eligibility|salary)\b",
        re.IGNORECASE,
    ),
    Topic.BENEFITS: re.compile(
        r"\b(?:مميزات|ميزة|ميزه|كاش باك|خصومات|نقاط|مزايا|benefits|perks|cashback|points|reward|rewards)\b",
        re.IGNORECASE,
    ),
}


class EntityAndTopicExtractor:
    """
    Extracts banking entities and topics from user input.
    """

    def __init__(self) -> None:
        self._registered_entities: list[dict] = []
        for ent in CANONICAL_ENTITIES:
            norm_aliases = [TextNormalizer.normalize(alias) for alias in ent["aliases"]]
            norm_aliases.append(TextNormalizer.normalize(ent["display_name"]))
            self._registered_entities.append(
                {
                    "id": ent["id"],
                    "display_name": ent["display_name"],
                    "entity_type": ent["entity_type"],
                    "aliases": ent["aliases"],
                    "normalized_aliases": list(set(norm_aliases)),
                }
            )

    def extract(
        self,
        text: str,
        state: ConversationState,
    ) -> tuple[Optional[ConversationEntity], Topic, bool, bool]:
        """
        Extracts entity and topic from query text and updates ConversationState.
        Preserves existing active_entity if no explicit entity is present.
        Continuously updates active_topic per turn.

        Returns:
            Tuple of (entity, topic, is_entity_switched, is_entity_preserved)
        """
        normalized_text = TextNormalizer.normalize(text)

        # 1. Match Entity
        entity, entity_switched, entity_preserved = self._match_entity(normalized_text, state)

        # 2. Classify Topic (Continuous Topic Updates per turn)
        topic = self._classify_topic(normalized_text)

        # Update State
        state.active_entity = entity
        state.active_topic = topic

        logger.debug(
            "EntityAndTopicExtractor -> Entity: %s, Topic: %s (switched=%s, preserved=%s)",
            entity.display_name if entity else "None",
            topic.value,
            entity_switched,
            entity_preserved,
        )

        return entity, topic, entity_switched, entity_preserved

    def _match_entity(
        self,
        normalized_text: str,
        state: ConversationState,
    ) -> tuple[Optional[ConversationEntity], bool, bool]:
        """
        Sub-helper for entity matching.
        """
        best_match = None
        longest_match_len = 0

        for ent_def in self._registered_entities:
            for alias in ent_def["normalized_aliases"]:
                if not alias:
                    continue
                if re.search(r"\b" + re.escape(alias) + r"\b", normalized_text) or alias in normalized_text:
                    if len(alias) > longest_match_len:
                        longest_match_len = len(alias)
                        best_match = ent_def

        if best_match is not None:
            new_entity = ConversationEntity(
                id=best_match["id"],
                display_name=best_match["display_name"],
                entity_type=best_match["entity_type"],
                aliases=best_match["aliases"],
                normalized_aliases=best_match["normalized_aliases"],
                confidence=0.95,
            )
            is_switched = (
                state.active_entity is not None and state.active_entity.id != new_entity.id
            )
            state.last_entity_update_turn = state.turn_count
            logger.info("Explicit Entity Detected: %s (switched=%s)", new_entity.display_name, is_switched)
            return new_entity, is_switched, not is_switched

        # Context Preservation: Retain active_entity if present
        if state.active_entity is not None:
            logger.info("Entity Context Preserved: %s", state.active_entity.display_name)
            return state.active_entity, False, True

        return None, False, False

    def _classify_topic(self, normalized_text: str) -> Topic:
        """
        Sub-helper for topic classification.
        """
        for topic, pattern in _TOPIC_PATTERNS.items():
            if pattern.search(normalized_text):
                return topic
        return Topic.GENERAL_INFO
