"""
Tests for Dialogue Manager v5 Phase 9 ContextCompressor and VoiceDialogueHandler

Verifies:
  1. ContextCompressor 6-turn sliding window history compression and summary update.
  2. VoiceDialogueHandler STT confidence threshold evaluation.
"""

import pytest

from app.conversation.context_compressor import ContextCompressor
from app.conversation.models import ConversationState, VoiceTurnAction
from app.conversation.voice_handler import VoiceDialogueHandler


def test_context_compressor_sliding_window():
    compressor = ContextCompressor()
    state = ConversationState(session_id="comp-v5")

    # Add 10 exchanges (20 messages)
    for i in range(10):
        state.add_exchange(f"User question {i}", f"Assistant answer {i}", max_turns=20)

    assert len(state.history) == 20

    # Compress to 6 turns (12 messages)
    compressor.compress_if_needed(state, max_raw_turns=6)

    assert len(state.history) == 12
    assert "User asked:" in state.conversation_summary


def test_voice_dialogue_handler_thresholds():
    handler = VoiceDialogueHandler()
    state = ConversationState(session_id="voice-v5")

    assert handler.evaluate_voice_turn(0.95, state) == VoiceTurnAction.PROCEED
    assert handler.evaluate_voice_turn(0.75, state) == VoiceTurnAction.PROCEED_UNCERTAIN
    assert handler.evaluate_voice_turn(0.50, state) == VoiceTurnAction.CONFIRM_INTENT
    assert handler.evaluate_voice_turn(0.30, state) == VoiceTurnAction.DISCARD_UTTERANCE
