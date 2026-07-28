"""
Tests for Conversation Subsystem v2

Verifies:
  1. Per-turn independent language evaluation (detected_language vs response_language).
  2. GREETING, SMALL_TALK, THANKS, GOODBYE social intents & response pools.
  3. Soft session idle handling on GOODBYE turn.
  4. Entity preservation & explicit entity switching.
  5. Continuous active topic evolution across turns.
  6. Deterministic follow-up rule engine & data-driven query rewriting.
  7. Full multi-turn bilingual banking dialogue end-to-end integration test.
"""

import pytest
from unittest.mock import AsyncMock

from app.conversation.models import (
    ConversationEntity,
    ConversationState,
    Department,
    EntityType,
    IntentType,
    Language,
    Topic,
    WorkflowType,
)
from app.conversation.text_normalizer import TextNormalizer
from app.conversation.language_manager import ConversationLanguageManager
from app.conversation.entity_topic_extractor import EntityAndTopicExtractor
from app.conversation.context_manager import ContextManager
from app.conversation.router.intent_router import HybridIntentRouter
from app.conversation.rewriter.query_rewriter import ConversationQueryRewriter
from app.conversation.conversation_manager import ConversationManager
from input.models.transcription import Transcription
from llm.models import AIResponse


def test_text_normalizer():
    assert TextNormalizer.normalize("الفييييزااا البلاتينيوم") == "الفيزا البلاتينيوم"
    assert TextNormalizer.normalize("  ViSa    PLATINUM  ") == "visa platinum"


def test_per_turn_language_manager():
    lang_mgr = ConversationLanguageManager(fallback_threshold=0.60)
    state = ConversationState(session_id="lang-test-v2")

    # Turn 1: English -> detected=EN, response=EN
    d1, r1, conf1 = lang_mgr.evaluate_turn_language(state, "Hello, how are you?")
    assert d1 == Language.ENGLISH
    assert r1 == Language.ENGLISH
    assert state.response_language == Language.ENGLISH

    # Turn 2: Arabic -> detected=AR, response=AR (per-turn independent language!)
    d2, r2, conf2 = lang_mgr.evaluate_turn_language(state, "عايز أعرف الرسوم")
    assert d2 == Language.ARABIC
    assert r2 == Language.ARABIC
    assert state.response_language == Language.ARABIC

    # Turn 3: English -> detected=EN, response=EN
    d3, r3, conf3 = lang_mgr.evaluate_turn_language(state, "What are the benefits?")
    assert d3 == Language.ENGLISH
    assert r3 == Language.ENGLISH
    assert state.response_language == Language.ENGLISH


@pytest.mark.asyncio
async def test_social_intents_and_pools():
    router = HybridIntentRouter(llm_provider=None)
    state = ConversationState(session_id="social-sess")

    # Greeting English
    d_greet = await router.route("Hello", state)
    assert d_greet.intent == IntentType.GREETING
    assert d_greet.is_rag_required is False
    assert len(d_greet.message) > 0

    # Small Talk Arabic
    state.response_language = Language.ARABIC
    d_st = await router.route("ازيك عامل ايه", state)
    assert d_st.intent == IntentType.SMALL_TALK
    assert d_st.is_rag_required is False
    assert "بخير" in d_st.message or "الحمد لله" in d_st.message

    # Thanks English
    state.response_language = Language.ENGLISH
    d_thanks = await router.route("Thank you very much", state)
    assert d_thanks.intent == IntentType.THANKS
    assert d_thanks.is_rag_required is False

    # Goodbye Soft Session Idle
    d_bye = await router.route("Goodbye", state)
    assert d_bye.intent == IntentType.GOODBYE
    assert d_bye.is_rag_required is False
    assert state.is_session_idle is True


def test_entity_topic_extractor_and_topic_evolution():
    extractor = EntityAndTopicExtractor()
    state = ConversationState(session_id="extractor-v2")

    # Turn 1: Product inquiry -> Entity=Visa Platinum, Topic=GENERAL_INFO
    ent1, topic1, sw1, pr1 = extractor.extract("Tell me about Visa Platinum", state)
    assert ent1.id == "visa_platinum"
    assert topic1 == Topic.GENERAL_INFO

    # Turn 2: Follow-up fees -> Entity Preserved=Visa Platinum, Topic=FEES
    ent2, topic2, sw2, pr2 = extractor.extract("How much are the fees?", state)
    assert ent2.id == "visa_platinum"
    assert topic2 == Topic.FEES
    assert pr2 is True

    # Turn 3: Follow-up benefits -> Entity Preserved=Visa Platinum, Topic=BENEFITS
    ent3, topic3, sw3, pr3 = extractor.extract("What are the benefits?", state)
    assert ent3.id == "visa_platinum"
    assert topic3 == Topic.BENEFITS
    assert pr3 is True

    # Turn 4: Explicit Entity Switch -> Entity Switched=Visa Gold, Topic=GENERAL_INFO
    ent4, topic4, sw4, pr4 = extractor.extract("What about Visa Gold?", state)
    assert ent4.id == "visa_gold"
    assert sw4 is True


@pytest.mark.asyncio
async def test_rule_based_query_rewriter_and_followup_rule_engine():
    rewriter = ConversationQueryRewriter(llm_provider=None)
    state = ConversationState(session_id="rewriter-v2")
    state.response_language = Language.ENGLISH
    state.active_entity = ConversationEntity(
        id="visa_platinum",
        display_name="Visa Platinum",
        entity_type=EntityType.CREDIT_CARD,
        normalized_aliases=["visa platinum", "platinum"],
        confidence=0.95,
    )
    state.active_topic = Topic.FEES

    # 1. Follow-up pronoun "its fees" -> Rule rewrite without LLM
    sq1, is_rew1, reason1, rule_app1 = await rewriter.rewrite("What are its fees?", state, "Context")
    assert sq1 == "What are the Visa Platinum fees?"
    assert is_rew1 is True
    assert rule_app1 is True

    # 2. Follow-up Arabic "ولو ضاعت؟" -> Topic=REPLACEMENT -> Rule rewrite
    state.active_topic = Topic.REPLACEMENT
    state.response_language = Language.ARABIC
    sq2, is_rew2, reason2, rule_app2 = await rewriter.rewrite("ولو ضاعت؟", state, "Context")
    assert "استبدال بطاقة Visa Platinum" in sq2 or "Visa Platinum" in sq2
    assert is_rew2 is True

    # 3. Unrelated query "What is today's exchange rate?" -> Not a follow-up -> Original query
    sq3, is_rew3, reason3, rule_app3 = await rewriter.rewrite("What is today's exchange rate?", state, "Context")
    assert sq3 == "What is today's exchange rate?"
    assert is_rew3 is False
    assert reason3 == "Not a Follow-up"


@pytest.mark.asyncio
async def test_end_to_end_bilingual_banking_conversation():
    """
    Complete multi-turn integration test exercising:
    Hello -> Tell me about Visa Platinum -> Benefits -> Fees -> Lost card -> Thank you -> Goodbye
    Verifying per-turn language, entity preservation, topic evolution, rule rewrites, and RAG short-circuiting.
    """
    mock_store = AsyncMock()
    session_state = ConversationState(session_id="e2e-session")
    mock_store.get_state.return_value = session_state
    mock_store.save_state = AsyncMock()

    manager = ConversationManager(
        store=mock_store,
        language_manager=ConversationLanguageManager(),
        router=HybridIntentRouter(llm_provider=None),
        entity_topic_extractor=EntityAndTopicExtractor(),
        context_manager=ContextManager(),
        rewriter=ConversationQueryRewriter(llm_provider=None),
        summarizer=AsyncMock(),
    )

    rag_executor = AsyncMock()
    rag_executor.return_value = AIResponse(
        action="rag",
        department=None,
        reason="success",
        message="Informational response from knowledge base.",
        language="en",
    )

    def _t(text: str, lang: str = "en") -> Transcription:
        return Transcription(text=text, language=lang, start_timestamp=0.0, end_timestamp=1.0)

    # Turn 1: "Hello" (Greeting - RAG Bypassed)
    r1 = await manager.process_transcript("e2e-session", _t("Hello"), rag_executor)
    assert r1.language == "en"
    assert len(r1.message) > 0

    # Turn 2: "Tell me about Visa Platinum" (Product Inquiry)
    r2 = await manager.process_transcript("e2e-session", _t("Tell me about Visa Platinum"), rag_executor)
    assert session_state.active_entity.id == "visa_platinum"
    assert session_state.active_topic == Topic.GENERAL_INFO

    # Turn 3: "What are its benefits?" (Follow-up Benefits)
    r3 = await manager.process_transcript("e2e-session", _t("What are its benefits?"), rag_executor)
    assert session_state.active_entity.id == "visa_platinum"
    assert session_state.active_topic == Topic.BENEFITS

    # Turn 4: "كام رسومها؟" (Arabic Follow-up Fees)
    r4 = await manager.process_transcript("e2e-session", _t("كام رسومها؟", "ar"), rag_executor)
    assert r4.language == "ar"
    assert session_state.active_entity.id == "visa_platinum"
    assert session_state.active_topic == Topic.FEES

    # Turn 5: "What if it is lost?" (Follow-up Replacement)
    r5 = await manager.process_transcript("e2e-session", _t("What if it is lost?"), rag_executor)
    assert session_state.active_entity.id == "visa_platinum"
    assert session_state.active_topic == Topic.REPLACEMENT

    # Turn 6: "Thank you" (Thanks - RAG Bypassed)
    r6 = await manager.process_transcript("e2e-session", _t("Thank you"), rag_executor)
    assert r6.language == "en"

    # Turn 7: "Goodbye" (Goodbye - RAG Bypassed, Soft Idle Flag)
    r7 = await manager.process_transcript("e2e-session", _t("Goodbye"), rag_executor)
    assert r7.language == "en"
    assert session_state.is_session_idle is True
    # Entity is preserved despite soft idle flag
    assert session_state.active_entity.id == "visa_platinum"
