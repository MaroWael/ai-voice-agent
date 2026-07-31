"""
Tests for Dialogue Manager v5 Phase 6 ContextValidator, AmbiguityResolver, and QueryRewriter

Verifies:
  1. ContextValidator pre-retrieval validation (insufficient entities for comparison, missing recommendation slots).
  2. AmbiguityResolver collision detection (branch fee vs card fee).
  3. QueryRewriter template rewrites for single entity and multi-entity requests.
"""

import pytest

from app.conversation.ambiguity_resolver import AmbiguityResolver
from app.conversation.context_validator import ContextValidator
from app.conversation.models import (
    ConversationEntity,
    ConversationState,
    EntityType,
    IntentType,
    Language,
    Topic,
    ValidationResult,
)
from app.conversation.rewriter.query_rewriter import ConversationQueryRewriter


def test_context_validator():
    validator = ContextValidator()
    state = ConversationState(session_id="val-v5")

    # 1-entity comparison request -> NEEDS_CLARIFICATION_COMPARISON
    e1 = ConversationEntity(id="visa_platinum", display_name="Visa Platinum", entity_type=EntityType.CREDIT_CARD)
    state.entity_stack.push(e1, turn=1)

    res_comp = validator.validate(IntentType.COMPARISON, state)
    assert res_comp == ValidationResult.NEEDS_CLARIFICATION_COMPARISON

    # Missing slots recommendation request -> NEEDS_SLOTS_RECOMMENDATION
    res_rec = validator.validate(IntentType.RECOMMENDATION, state)
    assert res_rec == ValidationResult.NEEDS_SLOTS_RECOMMENDATION

    # Provided slots recommendation request -> VALID
    state.user_profile.salary_egp = 20000.0
    state.user_profile.primary_purpose = "travel"
    res_rec_valid = validator.validate(IntentType.RECOMMENDATION, state)
    assert res_rec_valid == ValidationResult.VALID


def test_ambiguity_resolver():
    resolver = AmbiguityResolver()
    state = ConversationState(session_id="ambig-v5")
    e1 = ConversationEntity(id="visa_platinum", display_name="Visa Platinum", entity_type=EntityType.CREDIT_CARD)
    state.entity_stack.push(e1, turn=1)

    state.active_topic = Topic.BRANCHES
    is_ambig, reason = resolver.resolve_ambiguity("What are the fees?", state)
    assert is_ambig is True
    assert reason == "AMBIGUOUS_CARD_VS_BRANCH_FEE"


@pytest.mark.asyncio
async def test_query_rewriter_templates():
    rewriter = ConversationQueryRewriter(llm_provider=None)
    state = ConversationState(session_id="rewriter-v5")
    state.response_language = Language.ENGLISH
    e1 = ConversationEntity(id="visa_platinum", display_name="Visa Platinum", entity_type=EntityType.CREDIT_CARD)
    state.entity_stack.push(e1, turn=1)
    state.active_topic = Topic.FEES

    sq, is_rew, reason, is_rule = await rewriter.rewrite("What are its fees?", state)
    assert is_rew is True
    assert sq == "What are the Visa Platinum fees?"
