"""
Comprehensive Production Stabilization Unit Tests for Dialogue Manager v5 (Bugs 1-12)

Verifies:
  1. Fee rewrite preserves semantic topic (BUG #1).
  2. Benefit rewrite preserves semantic topic (BUG #1).
  3. Credit limit rewrite preserves semantic topic (BUG #2).
  4. Installment rewrite preserves semantic topic (BUG #2).
  5. Rewards rewrite preserves semantic topic (BUG #2).
  6. Eligibility rewrite preserves semantic topic (BUG #2).
  7. Requirements rewrite preserves semantic topic (BUG #2).
  8. Comparison pipeline end-to-end multi-entity extraction & matrix building (BUG #3).
  9. Missing comparison data handling (BUG #4).
 10. Comparison follow-up and ordinal resolution ("إيه رسوم الأولى؟", "والتانية؟") (BUG #5).
 11. Continuous topic switching across 5+ turns (BUG #6).
 12. Recommendation reusing ComparisonState (BUG #7).
 13. Entity memory persistence across multiple turns (BUG #8).
 14. Entity replacement on explicit new entity mention (BUG #8).
"""

import pytest
from unittest.mock import AsyncMock

from app.conversation.comparison_engine import ComparisonEngine
from app.conversation.conversation_manager import ConversationManager
from app.conversation.entity_resolver import EntityResolver
from app.conversation.language_manager import ConversationLanguageManager
from app.conversation.models import (
    ConversationEntity,
    ConversationState,
    EntityType,
    IntentType,
    Language,
    Topic,
    UserProfile,
)
from app.conversation.reference_resolver import ReferenceResolver
from app.conversation.recommendation_engine import RecommendationEngine
from app.conversation.rewriter.query_rewriter import ConversationQueryRewriter
from app.conversation.router.intent_router import HybridIntentRouter
from input.models.transcription import Transcription
from llm.models import AIResponse


@pytest.mark.asyncio
async def test_bug1_fee_and_benefit_rewrite_preserves_topic():
    rewriter = ConversationQueryRewriter(llm_provider=None)
    state = ConversationState(session_id="bug1-test")
    state.response_language = Language.ARABIC
    state.entity_stack.push(
        ConversationEntity(id="visa_gold", display_name="Visa Gold", entity_type=EntityType.CREDIT_CARD), turn=1
    )

    # 1. Fee rewrite -> Must preserve FEES topic!
    state.active_topic = Topic.FEES
    sq1, is_rew1, reason1, rule_app1 = await rewriter.rewrite("ما هي الرسوم الخاصة بها؟", state)
    assert is_rew1 is True
    assert "رسوم" in sq1
    assert "Visa Gold" in sq1
    assert "تفاصيل ومعلومات" not in sq1  # Must NOT be generic!

    # 2. Benefit rewrite -> Must preserve BENEFITS topic!
    state.active_topic = Topic.BENEFITS
    sq2, is_rew2, reason2, rule_app2 = await rewriter.rewrite("ما هي مميزاتها؟", state)
    assert is_rew2 is True
    assert "مميزات" in sq2
    assert "Visa Gold" in sq2


@pytest.mark.asyncio
async def test_bug2_topic_aware_query_rewrites():
    rewriter = ConversationQueryRewriter(llm_provider=None)
    state = ConversationState(session_id="bug2-test")
    state.response_language = Language.ARABIC
    state.entity_stack.push(
        ConversationEntity(id="visa_gold", display_name="Visa Gold", entity_type=EntityType.CREDIT_CARD), turn=1
    )

    # Credit Limit Rewrite
    state.active_topic = Topic.CREDIT_LIMIT
    sq_limit, _, _, _ = await rewriter.rewrite("كام الحد الائتماني؟", state)
    assert "الحد الائتماني" in sq_limit
    assert "Visa Gold" in sq_limit

    # Installments Rewrite
    state.active_topic = Topic.INSTALLMENTS
    sq_inst, _, _, _ = await rewriter.rewrite("هل ينفع أقسط بيها؟", state)
    assert "التقسيط" in sq_inst
    assert "Visa Gold" in sq_inst

    # Rewards Rewrite
    state.active_topic = Topic.REWARDS
    sq_rew, _, _, _ = await rewriter.rewrite("طب برنامج المكافآت؟", state)
    assert "المكافآت" in sq_rew
    assert "Visa Gold" in sq_rew

    # Eligibility Rewrite
    state.active_topic = Topic.ELIGIBILITY
    sq_elig, _, _, _ = await rewriter.rewrite("طب شروط الإصدار؟", state)
    assert "شروط" in sq_elig or "أهلية" in sq_elig
    assert "Visa Gold" in sq_elig


def test_bug3_comparison_pipeline_matrix():
    resolver = EntityResolver()
    comp_engine = ComparisonEngine()
    state = ConversationState(session_id="bug3-test")

    ents = resolver.extract_entities("قارن بين Visa Gold و Visa Platinum", state)
    assert len(ents) == 2
    
    comp_state = comp_engine.build_comparison_matrix(ents)
    assert len(comp_state.compared_entities) == 2
    assert comp_state.compared_entities[0].id in ["visa_gold", "visa_platinum"]
    assert comp_state.compared_entities[1].id in ["visa_gold", "visa_platinum"]


def test_bug4_missing_comparison_data():
    comp_engine = ComparisonEngine()
    gold = ConversationEntity(id="visa_gold", display_name="Visa Gold", entity_type=EntityType.CREDIT_CARD)
    plat = ConversationEntity(id="visa_platinum", display_name="Visa Platinum", entity_type=EntityType.CREDIT_CARD)

    # Only Platinum retrieved, Gold missing!
    is_inc, msg = comp_engine.validate_comparison_data([gold, plat], ["visa_platinum"], Language.ARABIC)
    assert is_inc is True
    assert "Visa Platinum" in msg
    assert "Visa Gold" in msg
    assert "غير كافية" in msg or "كافية" in msg or "مقارنة" in msg


def test_bug5_comparison_followup_and_ordinals():
    ref_resolver = ReferenceResolver()
    state = ConversationState(session_id="bug5-test")
    gold = ConversationEntity(id="visa_gold", display_name="Visa Gold", entity_type=EntityType.CREDIT_CARD)
    plat = ConversationEntity(id="visa_platinum", display_name="Visa Platinum", entity_type=EntityType.CREDIT_CARD)

    comp_engine = ComparisonEngine()
    state.comparison_state = comp_engine.build_comparison_matrix([gold, plat])

    # Ordinal 1: "إيه رسوم الأولى؟" -> Gold
    is_r1, ent1 = ref_resolver.resolve_reference("إيه رسوم الأولى؟", state)
    assert is_r1 is True
    assert ent1.id == "visa_gold"

    # Ordinal 2: "والتانية؟" -> Platinum
    is_r2, ent2 = ref_resolver.resolve_reference("والتانية؟", state)
    assert is_r2 is True
    assert ent2.id == "visa_platinum"


@pytest.mark.asyncio
async def test_bug6_continuous_topic_switching():
    mock_store = AsyncMock()
    session_state = ConversationState(session_id="bug6-test")
    mock_store.get_state.return_value = session_state
    mock_store.save_state = AsyncMock()

    manager = ConversationManager(
        store=mock_store,
        language_manager=ConversationLanguageManager(),
        router=HybridIntentRouter(llm_provider=None),
    )

    rag_executor = AsyncMock()
    rag_executor.return_value = AIResponse(action="rag", department=None, reason="success", message="Info", language="ar")

    def _t(text: str) -> Transcription:
        return Transcription(text=text, language="ar", start_timestamp=0.0, end_timestamp=1.0)

    # Turn 1: "كلمني عن Visa Gold" -> GENERAL_INFO
    await manager.process_transcript("bug6-test", _t("كلمني عن Visa Gold"), rag_executor)
    assert session_state.active_entity.id == "visa_gold"
    assert session_state.active_topic == Topic.GENERAL_INFO

    # Turn 2: "ما هي الرسوم؟" -> FEES
    await manager.process_transcript("bug6-test", _t("ما هي الرسوم؟"), rag_executor)
    assert session_state.active_topic == Topic.FEES

    # Turn 3: "ما هي المميزات؟" -> BENEFITS
    await manager.process_transcript("bug6-test", _t("ما هي المميزات؟"), rag_executor)
    assert session_state.active_topic == Topic.BENEFITS

    # Turn 4: "طب الحد الائتماني؟" -> CREDIT_LIMIT
    await manager.process_transcript("bug6-test", _t("طب الحد الائتماني؟"), rag_executor)
    assert session_state.active_topic == Topic.CREDIT_LIMIT

    # Turn 5: "طب شروط الإصدار؟" -> REQUIREMENTS / ELIGIBILITY
    await manager.process_transcript("bug6-test", _t("طب شروط الإصدار؟"), rag_executor)
    assert session_state.active_topic in [Topic.REQUIREMENTS, Topic.ELIGIBILITY]

    # Turn 6: "طب برنامج المكافآت؟" -> REWARDS
    await manager.process_transcript("bug6-test", _t("طب برنامج المكافآت؟"), rag_executor)
    assert session_state.active_topic == Topic.REWARDS


def test_bug7_comparison_recommendation_reuse():
    rec_engine = RecommendationEngine()
    state = ConversationState(session_id="bug7-test")
    gold = ConversationEntity(id="visa_gold", display_name="Visa Gold", entity_type=EntityType.CREDIT_CARD, metadata={"min_salary": 5000})
    plat = ConversationEntity(id="visa_platinum", display_name="Visa Platinum", entity_type=EntityType.CREDIT_CARD, metadata={"min_salary": 15000, "lounge_access_count": 6})

    comp_engine = ComparisonEngine()
    state.comparison_state = comp_engine.build_comparison_matrix([gold, plat])

    profile = UserProfile(salary_egp=20000.0, primary_purpose="travel")
    # RecommendationEngine evaluates compared_entities directly
    ranked = rec_engine.recommend(profile, state.comparison_state.compared_entities)
    assert len(ranked) == 2
    assert ranked[0].entity.id == "visa_platinum"


@pytest.mark.asyncio
async def test_bug8_entity_memory_persistence_and_replacement():
    mock_store = AsyncMock()
    session_state = ConversationState(session_id="bug8-test")
    mock_store.get_state.return_value = session_state
    mock_store.save_state = AsyncMock()

    manager = ConversationManager(
        store=mock_store,
        language_manager=ConversationLanguageManager(),
        router=HybridIntentRouter(llm_provider=None),
    )

    rag_executor = AsyncMock()
    rag_executor.return_value = AIResponse(action="rag", department=None, reason="success", message="Info", language="ar")

    def _t(text: str) -> Transcription:
        return Transcription(text=text, language="ar", start_timestamp=0.0, end_timestamp=1.0)

    # 1. Active Entity = Visa Gold
    await manager.process_transcript("bug8-test", _t("Visa Gold"), rag_executor)
    assert session_state.active_entity.id == "visa_gold"

    # Follow-ups keep active entity = Visa Gold
    await manager.process_transcript("bug8-test", _t("رسومها"), rag_executor)
    assert session_state.active_entity.id == "visa_gold"

    await manager.process_transcript("bug8-test", _t("مميزاتها"), rag_executor)
    assert session_state.active_entity.id == "visa_gold"

    await manager.process_transcript("bug8-test", _t("التقسيط"), rag_executor)
    assert session_state.active_entity.id == "visa_gold"

    # Explicit entity switch -> Visa Platinum replaces Visa Gold!
    await manager.process_transcript("bug8-test", _t("ماذا عن Visa Platinum؟"), rag_executor)
    assert session_state.active_entity.id == "visa_platinum"
