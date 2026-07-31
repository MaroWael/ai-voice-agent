"""
Entity Resolver

Extracts canonical banking entities from user utterances and pushes them
to the session's EntityStack and EntityTimeline.
Supports multi-entity extraction for comparison requests.
"""

import logging
import re
from typing import List, Optional

from app.conversation.models import ConversationEntity, ConversationState, EntityType
from app.conversation.text_normalizer import TextNormalizer

logger = logging.getLogger(__name__)


CANONICAL_ENTITIES = [
    {
        "id": "visa_platinum",
        "canonical_name": "Visa Platinum Credit Card",
        "display_name": "Visa Platinum",
        "entity_type": EntityType.CREDIT_CARD,
        "aliases": ["Visa Platinum", "Platinum Credit Card", "الفيزا البلاتينيوم", "البلاتينيوم", "فيزا بلاتينوم", "بلاتينيوم"],
        "document_source": "credit_cards/visa_platinum.pdf",
        "metadata": {"min_salary": 15000, "annual_fee": 500, "lounge_access_count": 6, "cashback_rate": 0.015, "foreign_markup": 0.03},
    },
    {
        "id": "visa_gold",
        "canonical_name": "Visa Gold Credit Card",
        "display_name": "Visa Gold",
        "entity_type": EntityType.CREDIT_CARD,
        "aliases": ["Visa Gold", "Gold Credit Card", "الفيزا الجولد", "الجولد", "فيزا جولد", "جولد"],
        "document_source": "credit_cards/visa_gold.pdf",
        "metadata": {"min_salary": 5000, "annual_fee": 200, "lounge_access_count": 0, "cashback_rate": 0.005, "foreign_markup": 0.035},
    },
    {
        "id": "visa_signature",
        "canonical_name": "Visa Signature Credit Card",
        "display_name": "Visa Signature",
        "entity_type": EntityType.CREDIT_CARD,
        "aliases": ["Visa Signature", "Signature", "الفيزا السيجنتشر", "سيجنتشر"],
        "document_source": "credit_cards/visa_signature.pdf",
        "metadata": {"min_salary": 30000, "annual_fee": 1500, "lounge_access_count": 12, "cashback_rate": 0.025, "foreign_markup": 0.02},
    },
    {
        "id": "al_qemma_certificate",
        "canonical_name": "Al Qemma 3-Year Certificate",
        "display_name": "Al Qemma Certificate",
        "entity_type": EntityType.CERTIFICATE,
        "aliases": ["Al Qemma", "Qemma Certificate", "شهادة القمة", "شهاده القمه", "القمة", "قمة"],
        "document_source": "certificates/al_qemma.pdf",
        "metadata": {"duration_years": 3, "interest_rate": 0.19, "payout_frequency": "monthly", "min_amount": 1000},
    },
    {
        "id": "personal_loan",
        "canonical_name": "Banque Misr Personal Loan",
        "display_name": "Personal Loan",
        "entity_type": EntityType.LOAN,
        "aliases": ["Personal Loan", "Loan", "القرض الشخصي", "قرض شخصي", "القرض", "قروض"],
        "document_source": "loans/personal_loan.pdf",
        "metadata": {"min_salary": 3000, "max_amount": 1500000, "max_tenure_years": 10},
    },
]


class EntityResolver:
    """
    Extracts entities from query text and manages EntityStack pushing.
    """

    def __init__(self) -> None:
        self._registered_entities: List[dict] = []
        for ent in CANONICAL_ENTITIES:
            norm_aliases = [TextNormalizer.normalize(alias) for alias in ent["aliases"]]
            norm_aliases.append(TextNormalizer.normalize(ent["display_name"]))
            self._registered_entities.append(
                {
                    "id": ent["id"],
                    "canonical_name": ent["canonical_name"],
                    "display_name": ent["display_name"],
                    "entity_type": ent["entity_type"],
                    "aliases": ent["aliases"],
                    "normalized_aliases": list(set(norm_aliases)),
                    "document_source": ent.get("document_source", ""),
                    "metadata": ent.get("metadata", {}),
                }
            )

    def extract_entities(self, text: str, state: ConversationState) -> List[ConversationEntity]:
        """
        Extracts all canonical entities mentioned in text and pushes them to state.entity_stack.
        Returns list of extracted entities.
        """
        norm_text = TextNormalizer.normalize(text)
        extracted: List[ConversationEntity] = []

        for ent_def in self._registered_entities:
            for alias in ent_def["normalized_aliases"]:
                if not alias:
                    continue
                if re.search(r"\b" + re.escape(alias) + r"\b", norm_text) or alias in norm_text:
                    entity = ConversationEntity(
                        id=ent_def["id"],
                        canonical_name=ent_def["canonical_name"],
                        display_name=ent_def["display_name"],
                        entity_type=ent_def["entity_type"],
                        aliases=ent_def["aliases"],
                        normalized_aliases=ent_def["normalized_aliases"],
                        document_source=ent_def["document_source"],
                        confidence=0.95,
                        metadata=ent_def["metadata"],
                    )
                    extracted.append(entity)
                    state.entity_stack.push(entity, turn=state.turn_count, mention_type="EXPLICIT")
                    state.last_entity_update_turn = state.turn_count
                    break

        if extracted:
            logger.info("EntityResolver extracted %d entities: %s", len(extracted), [e.display_name for e in extracted])
        elif state.entity_stack.peek():
            logger.debug("EntityResolver preserved active entity: %s", state.entity_stack.peek().display_name)

        return extracted
