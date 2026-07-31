"""
Tests for Dialogue Manager v5 Phase 8 ClarificationEngine, WorkflowManager, and DialoguePolicyEngine

Verifies:
  1. ClarificationEngine localized clarification response generation.
  2. WorkflowManager WorkflowStack push, pop, and workflow resumption.
  3. DialoguePolicyEngine table-driven policy rule matching.
"""

import pytest

from app.conversation.clarification_engine import ClarificationEngine
from app.conversation.models import ConversationState, IntentType, Language, WorkflowType
from app.conversation.policy_engine import DialoguePolicyEngine
from app.conversation.workflow_manager import WorkflowManager


def test_clarification_engine_responses():
    clarifier = ClarificationEngine()
    state = ConversationState(session_id="clar-v5")
    state.response_language = Language.ARABIC

    r_comp = clarifier.generate_clarification("NEEDS_CLARIFICATION_COMPARISON", state)
    assert "المقارنة" in r_comp.message or "مقارنتها" in r_comp.message
    assert r_comp.action == "ROUTE"

    state.response_language = Language.ENGLISH
    r_slots = clarifier.generate_clarification("NEEDS_SLOTS_RECOMMENDATION", state)
    assert "monthly salary" in r_slots.message


def test_workflow_manager_stack_pause_resume():
    wf_mgr = WorkflowManager()
    state = ConversationState(session_id="wf-stack-v5")

    # Start recommendation workflow
    wf_mgr.push_workflow(state, WorkflowType.CREDIT_CARD_RECOMMENDATION)
    assert state.active_workflow == WorkflowType.CREDIT_CARD_RECOMMENDATION

    # Interrupt with FRAUD workflow -> RECOMMENDATION pushed to stack!
    wf_mgr.push_workflow(state, WorkflowType.FRAUD)
    assert state.active_workflow == WorkflowType.FRAUD
    assert len(state.workflow_stack) == 1
    assert state.workflow_stack[0] == WorkflowType.CREDIT_CARD_RECOMMENDATION

    # Complete FRAUD -> Pop stack and resume RECOMMENDATION!
    resumed = wf_mgr.pop_workflow(state)
    assert resumed == WorkflowType.CREDIT_CARD_RECOMMENDATION
    assert state.active_workflow == WorkflowType.CREDIT_CARD_RECOMMENDATION


def test_policy_engine_matrix_matching():
    policy_engine = DialoguePolicyEngine()
    state = ConversationState(session_id="policy-v5")

    rule_greet = policy_engine.evaluate_policy(state, IntentType.GREETING, stt_confidence=0.95)
    assert rule_greet.rule_id == "R01"
    assert rule_greet.requires_rag is False

    rule_comp = policy_engine.evaluate_policy(state, IntentType.COMPARISON, stt_confidence=0.95)
    assert rule_comp.rule_id == "R09"
    assert rule_comp.requires_rag is True
