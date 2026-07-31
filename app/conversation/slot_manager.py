"""
Slot Manager

Extracts customer profile slots from user input using deterministic rules and regex patterns.
Updates state.user_profile and state.slots with confidence tracking and update history.
"""

import logging
import re
from typing import Dict

from app.conversation.models import ConversationState, SlotValue
from app.conversation.text_normalizer import TextNormalizer

logger = logging.getLogger(__name__)

# Salary Extraction Patterns (handles EGP, LE, "ألف", "الف", "k", etc.)
_SALARY_PATTERNS = [
    re.compile(r"(?:مرتبي|مرتب|دخلي|دخل|باخد|بقبض|salary|income)\s*(?:هو|بقيمة)?\s*(\d+[\d,.]*)", re.IGNORECASE),
    re.compile(r"(\d+[\d,.]*)\s*(?:جنيه|جنية|الاف|ألاف|ألف|الف|egp|le)", re.IGNORECASE),
    re.compile(r"\b(\d+)\s*k\b", re.IGNORECASE),
]

# Purpose Patterns
_PURPOSE_PATTERNS = {
    "travel": re.compile(r"\b(?:سفر|بسافر|مطار|مطارات|طيران|travel|airport|flight|lounges)\b", re.IGNORECASE),
    "cashback": re.compile(r"\b(?:كاش باك|كاشباك|استرجاع|cashback|cash back)\b", re.IGNORECASE),
    "installments": re.compile(r"\b(?:تقسيط|قسط|أقساط|اقساط|installments|installment)\b", re.IGNORECASE),
}

# Employment Status Patterns
_EMPLOYMENT_PATTERNS = {
    "salaried": re.compile(r"\b(?:موظف|شركة|حكومة|بشتغل|بقبض|salaried|employee|company)\b", re.IGNORECASE),
    "self_employed": re.compile(r"\b(?:أعمال حرة|اعمال حرة|تاجر|صاحب عمل|خاص|self employed|freelance|business owner)\b", re.IGNORECASE),
    "retired": re.compile(r"\b(?:معاش|معاشات|متقاعد|retired|pension)\b", re.IGNORECASE),
}


class SlotManager:
    """
    Extracts and updates user profile slots deterministically.
    """

    def process_turn(self, query: str, state: ConversationState) -> Dict[str, SlotValue]:
        """
        Processes turn *query* and updates state.user_profile and state.slots.
        Returns dictionary of updated slots.
        """
        if not query:
            return state.slots

        norm_query = TextNormalizer.normalize(query)
        turn = state.turn_count

        # 1. Extract Salary Slot
        self._extract_salary(query, norm_query, state, turn)

        # 2. Extract Primary Purpose Slot
        self._extract_purpose(query, norm_query, state, turn)

        # 3. Extract Employment Status Slot
        self._extract_employment(query, norm_query, state, turn)

        return state.slots

    def _extract_salary(self, raw_query: str, norm_query: str, state: ConversationState, turn: int) -> None:
        for pattern in _SALARY_PATTERNS:
            match = pattern.search(raw_query) or pattern.search(norm_query)
            if match:
                raw_num = match.group(1).replace(",", "")
                try:
                    val = float(raw_num)
                    # Check for thousands multiplier ("ألف", "الف", "k")
                    if "ألف" in raw_query or "الف" in raw_query or "k" in raw_query.lower() or "ألف" in norm_query or "الف" in norm_query:
                        if val < 1000:
                            val *= 1000.0

                    state.user_profile.salary_egp = val
                    slot = state.slots.get("salary_egp")
                    if slot is None:
                        slot = SlotValue(name="salary_egp", value=val, confidence=0.95, updated_at_turn=turn)
                    else:
                        slot.update(val, new_confidence=0.95, turn=turn)

                    state.slots["salary_egp"] = slot
                    logger.info("SlotManager extracted salary_egp: %.2f EGP", val)
                    break
                except ValueError:
                    continue

    def _extract_purpose(self, raw_query: str, norm_query: str, state: ConversationState, turn: int) -> None:
        for purpose, pattern in _PURPOSE_PATTERNS.items():
            if pattern.search(raw_query) or pattern.search(norm_query):
                state.user_profile.primary_purpose = purpose
                slot = state.slots.get("primary_purpose")
                if slot is None:
                    slot = SlotValue(name="primary_purpose", value=purpose, confidence=0.95, updated_at_turn=turn)
                else:
                    slot.update(purpose, new_confidence=0.95, turn=turn)

                state.slots["primary_purpose"] = slot
                logger.info("SlotManager extracted primary_purpose: %s", purpose)
                break

    def _extract_employment(self, raw_query: str, norm_query: str, state: ConversationState, turn: int) -> None:
        for emp, pattern in _EMPLOYMENT_PATTERNS.items():
            if pattern.search(raw_query) or pattern.search(norm_query):
                state.user_profile.employment_status = emp
                slot = state.slots.get("employment_status")
                if slot is None:
                    slot = SlotValue(name="employment_status", value=emp, confidence=0.95, updated_at_turn=turn)
                else:
                    slot.update(emp, new_confidence=0.95, turn=turn)

                state.slots["employment_status"] = slot
                logger.info("SlotManager extracted employment_status: %s", emp)
                break
