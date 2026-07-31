"""
Context Validator

Evaluates state prerequisites prior to retrieval to detect impossible situations
(e.g., comparison requested with < 2 entities, recommendation requested with missing required slots).
"""

import logging
from app.conversation.models import ConversationState, IntentType, ValidationResult

logger = logging.getLogger(__name__)


class ContextValidator:
    """
    Validates state readiness before vector retrieval.
    """

    def validate(self, intent: IntentType, state: ConversationState) -> ValidationResult:
        """
        Validates whether *state* has required prerequisites for *intent*.

        Returns:
            ValidationResult enum.
        """
        if intent == IntentType.COMPARISON:
            if len(state.entity_stack.stack) < 2 and not (state.comparison_state and len(state.comparison_state.compared_entities) >= 2):
                logger.info("ContextValidator: Comparison request has < 2 active entities -> NEEDS_CLARIFICATION_COMPARISON")
                return ValidationResult.NEEDS_CLARIFICATION_COMPARISON

        if intent == IntentType.RECOMMENDATION:
            salary_present = state.user_profile.salary_egp is not None
            purpose_present = state.user_profile.primary_purpose is not None
            if not salary_present and not purpose_present:
                logger.info("ContextValidator: Recommendation request missing required slots -> NEEDS_SLOTS_RECOMMENDATION")
                return ValidationResult.NEEDS_SLOTS_RECOMMENDATION

        return ValidationResult.VALID
