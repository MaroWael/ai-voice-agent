import pytest
import re

from app.factories.rag import build_rag_service
from app.rag.builders.language_detector import detect_query_language
from app.rag.models.status import RagStatus
from app.startup.knowledge_initializer import initialize_knowledge_base
from input.models.transcription import Transcription
from llm.rag_llm import RagLanguageModel

pytestmark = pytest.mark.asyncio

_ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF]")


async def _ensure_kb():
    await initialize_knowledge_base()


async def test_english_query_language_consistency():
    """Verify that English queries receive responses in English and set language='en'."""
    await _ensure_kb()
    rag_llm = RagLanguageModel()
    await rag_llm.initialize()
    try:
        trans = Transcription(
            text="What are the fees for the Gold Credit Card?",
            language="en",
            start_timestamp=0.0,
            end_timestamp=1.0,
        )
        ai_response = await rag_llm.generate(trans)
        assert ai_response.language == "en", f"Expected language='en', got {ai_response.language}"

        # Answer should contain English text and NOT be translated into pure Arabic sentences
        answer_text = ai_response.message
        assert len(answer_text) > 0
        # Verify response contains English characters (e.g. EGP, issuance, fee, card, Gold)
        assert any(c.isalpha() and ord(c) < 128 for c in answer_text), f"Answer is not in English: {answer_text}"
        # Verify answer is not predominantly Arabic
        arabic_chars = len(_ARABIC_CHAR_RE.findall(answer_text))
        total_letters = sum(1 for c in answer_text if c.isalpha())
        assert (arabic_chars / total_letters if total_letters else 0) < 0.3, f"Answer unexpectedly translated to Arabic: {answer_text}"
    finally:
        await rag_llm.close()


async def test_arabic_query_language_consistency():
    """Verify that Arabic queries receive responses in Arabic and set language='ar'."""
    await _ensure_kb()
    rag_llm = RagLanguageModel()
    await rag_llm.initialize()
    try:
        trans = Transcription(
            text="ايه رسوم الفيزا الجولد؟",
            language="ar",
            start_timestamp=0.0,
            end_timestamp=1.0,
        )
        ai_response = await rag_llm.generate(trans)
        assert ai_response.language == "ar", f"Expected language='ar', got {ai_response.language}"

        answer_text = ai_response.message
        assert len(answer_text) > 0
        assert len(_ARABIC_CHAR_RE.findall(answer_text)) > 0, f"Expected Arabic response, got: {answer_text}"
    finally:
        await rag_llm.close()


async def test_mixed_query_language_consistency():
    """Verify that code-switched queries follow dominant language strategy ('ar' when Arabic characters exist)."""
    await _ensure_kb()
    rag_llm = RagLanguageModel()
    await rag_llm.initialize()
    try:
        trans = Transcription(
            text="what are رسوم gold card?",
            language=None,
            start_timestamp=0.0,
            end_timestamp=1.0,
        )
        ai_response = await rag_llm.generate(trans)
        expected_lang = detect_query_language("what are رسوم gold card?")
        assert ai_response.language == expected_lang, f"Expected language='{expected_lang}', got {ai_response.language}"
    finally:
        await rag_llm.close()
