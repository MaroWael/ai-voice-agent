"""
Dialogue Policy Engine

Declarative Table-Driven Policy Rule Matrix for Dialogue Control.
Evaluates turn intent, stage, and STT confidence to deterministically select policy action.
"""

import logging
from typing import List, Optional

from app.conversation.models import ConversationStage, ConversationState, IntentType, PolicyRule

logger = logging.getLogger(__name__)

# Declarative Policy Rule Matrix
POLICY_RULE_MATRIX: List[PolicyRule] = [
    PolicyRule(rule_id="R01", intent=IntentType.GREETING, min_stt_confidence=0.60, requires_rag=False, action_handler="handle_social"),
    PolicyRule(rule_id="R02", intent=IntentType.GOODBYE, min_stt_confidence=0.60, requires_rag=False, action_handler="handle_social"),
    PolicyRule(rule_id="R03", intent=IntentType.SMALL_TALK, min_stt_confidence=0.60, requires_rag=False, action_handler="handle_social"),
    PolicyRule(rule_id="R04", intent=IntentType.THANKS, min_stt_confidence=0.60, requires_rag=False, action_handler="handle_social"),
    PolicyRule(rule_id="R05", intent=IntentType.FRAUD, min_stt_confidence=0.60, requires_rag=False, action_handler="handle_workflow"),
    PolicyRule(rule_id="R06", intent=IntentType.COMPLAINT, min_stt_confidence=0.60, requires_rag=False, action_handler="handle_workflow"),
    PolicyRule(rule_id="R07", intent=IntentType.CUSTOMER_SERVICE, min_stt_confidence=0.60, requires_rag=False, action_handler="handle_workflow"),
    PolicyRule(rule_id="R08", intent=IntentType.TRANSFER_REQUEST, min_stt_confidence=0.60, requires_rag=False, action_handler="handle_workflow"),
    PolicyRule(rule_id="R09", intent=IntentType.COMPARISON, min_stt_confidence=0.60, requires_rag=True, action_handler="handle_comparison"),
    PolicyRule(rule_id="R10", intent=IntentType.RECOMMENDATION, min_stt_confidence=0.60, requires_rag=True, action_handler="handle_recommendation"),
    PolicyRule(rule_id="R11", intent=IntentType.PRODUCT_QUERY, min_stt_confidence=0.60, requires_rag=True, action_handler="handle_query"),
    PolicyRule(rule_id="R12", intent=IntentType.FACT_QUERY, min_stt_confidence=0.60, requires_rag=True, action_handler="handle_query"),
    PolicyRule(rule_id="R13", intent=IntentType.FOLLOW_UP, min_stt_confidence=0.60, requires_rag=True, action_handler="handle_query"),
]


class DialoguePolicyEngine:
    """
    Evaluates policy matrix rules deterministically against current turn state.
    """

    def evaluate_policy(self, state: ConversationState, intent: IntentType, stt_confidence: float = 1.0) -> PolicyRule:
        """
        Selects matching PolicyRule from POLICY_RULE_MATRIX.
        """
        for rule in POLICY_RULE_MATRIX:
            if rule.intent == intent:
                if stt_confidence >= rule.min_stt_confidence:
                    logger.info("DialoguePolicyEngine: Matched Policy Rule %s (%s)", rule.rule_id, rule.action_handler)
                    return rule

        # Default Fallback Rule -> General RAG Query
        logger.info("DialoguePolicyEngine: No specific rule matched. Defaulting to query rule R11.")
        return PolicyRule(
            rule_id="R_FALLBACK",
            intent=intent,
            min_stt_confidence=0.40,
            requires_rag=True,
            action_handler="handle_query",
        )
