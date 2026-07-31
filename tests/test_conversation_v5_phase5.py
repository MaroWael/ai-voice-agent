"""
Tests for Dialogue Manager v5 Phase 5 HybridIntentRouter & 17-Intent Taxonomy

Verifies:
  1. Social intent routing (GREETING, GOODBYE, THANKS, SMALL_TALK) with RAG short-circuiting.
  2. Operational workflow routing (FRAUD, COMPLAINT, CUSTOMER_SERVICE, HUMAN_AGENT).
  3. Dynamic response pools in Arabic and English.
"""

import pytest

from app.conversation.models import ConversationState, Department, IntentType, Language, WorkflowType
from app.conversation.router.intent_router import HybridIntentRouter


@pytest.mark.asyncio
async def test_intent_router_greetings_and_goodbye():
    router = HybridIntentRouter(llm_provider=None)
    state = ConversationState(session_id="intent-v5")

    # Turn 1: Greeting English
    state.response_language = Language.ENGLISH
    d1 = await router.route("Hello", state)
    assert d1.intent == IntentType.GREETING
    assert d1.is_rag_required is False
    assert len(d1.message) > 0

    # Turn 2: Greeting Arabic
    state.response_language = Language.ARABIC
    d2 = await router.route("السلام عليكم", state)
    assert d2.intent == IntentType.GREETING
    assert d2.is_rag_required is False
    assert "مرحباً" in d2.message or "أهلاً" in d2.message

    # Turn 3: Goodbye -> sets is_session_idle = True
    d3 = await router.route("Goodbye", state)
    assert d3.intent == IntentType.GOODBYE
    assert d3.is_rag_required is False
    assert state.is_session_idle is True


@pytest.mark.asyncio
async def test_intent_router_operational_workflows():
    router = HybridIntentRouter(llm_provider=None)
    state = ConversationState(session_id="wf-v5")

    # Fraud report
    d_fraud = await router.route("كارتي اتسرق!", state)
    assert d_fraud.intent == IntentType.FRAUD
    assert d_fraud.department == Department.FRAUD
    assert d_fraud.workflow == WorkflowType.FRAUD
    assert d_fraud.is_rag_required is False

    # Complaint report
    d_comp = await router.route("عايز أقدم شكوى", state)
    assert d_comp.intent == IntentType.COMPLAINT
    assert d_comp.department == Department.COMPLAINTS
    assert d_comp.workflow == WorkflowType.COMPLAINT
    assert d_comp.is_rag_required is False
