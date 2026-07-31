"""
Tests for Dialogue Manager v5 Phase 2 Text Normalizer & Language Manager

Verifies:
  1. Text Normalization (Arabic tatweel removal, character collapsing, alef/yaa/teh normalization).
  2. Per-turn independent language evaluation for English, Arabic, and mixed code-switching.
"""

import pytest

from app.conversation.models import ConversationState, Language
from app.conversation.text_normalizer import TextNormalizer
from app.conversation.language_manager import ConversationLanguageManager


def test_text_normalizer_arabic():
    assert TextNormalizer.normalize("الفييييزااا البلاتينيوم") == "الفيزا البلاتينيوم"
    assert TextNormalizer.normalize("  إمرأة  آمال  أحمد  ") == "امراه امال احمد"


def test_text_normalizer_english():
    assert TextNormalizer.normalize("  ViSa    PLATINUM  ") == "visa platinum"


def test_text_normalizer_mixed():
    assert TextNormalizer.normalize("  عايز  اعرف   Visa Platinum  ") == "عايز اعرف visa platinum"


def test_language_manager_per_turn_independence():
    lang_mgr = ConversationLanguageManager(fallback_threshold=0.60)
    state = ConversationState(session_id="lang-v5")

    # Turn 1: English
    d1, r1, c1 = lang_mgr.evaluate_turn_language(state, "Hello, how are you?")
    assert d1 == Language.ENGLISH
    assert r1 == Language.ENGLISH

    # Turn 2: Arabic
    d2, r2, c2 = lang_mgr.evaluate_turn_language(state, "عايز أعرف الرسوم")
    assert d2 == Language.ARABIC
    assert r2 == Language.ARABIC

    # Turn 3: Mixed English/Arabic -> Arabic dominant script
    d3, r3, c3 = lang_mgr.evaluate_turn_language(state, "Tell me about الفيزا البلاتينيوم")
    assert d3 == Language.ARABIC
    assert r3 == Language.ARABIC

    # Turn 4: Pure English
    d4, r4, c4 = lang_mgr.evaluate_turn_language(state, "What are the benefits?")
    assert d4 == Language.ENGLISH
    assert r4 == Language.ENGLISH
