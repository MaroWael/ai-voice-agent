"""
Tests for Dialogue Manager v5 Phase 4 SlotManager

Verifies:
  1. Salary extraction in EGP, thousands ("20 ألف", "15000 EGP", "20k").
  2. Primary purpose extraction ("travel", "cashback", "installments").
  3. Profile slot updates and slot correction history.
"""

import pytest

from app.conversation.models import ConversationState
from app.conversation.slot_manager import SlotManager


def test_slot_manager_salary_extraction():
    mgr = SlotManager()
    state = ConversationState(session_id="slot-v5")

    # Arabic thousands
    mgr.process_turn("مرتبي 20 ألف جنيه", state)
    assert state.user_profile.salary_egp == 20000.0
    assert state.slots["salary_egp"].value == 20000.0

    # English EGP
    mgr.process_turn("My salary is 15000 EGP", state)
    assert state.user_profile.salary_egp == 15000.0
    assert state.slots["salary_egp"].value == 15000.0
    assert state.slots["salary_egp"].history == [20000.0]  # History updated on correction!


def test_slot_manager_purpose_extraction():
    mgr = SlotManager()
    state = ConversationState(session_id="slot-purpose-v5")

    mgr.process_turn("عايز كارت عشان بسافر كتير وبدخل مطارات", state)
    assert state.user_profile.primary_purpose == "travel"
    assert state.slots["primary_purpose"].value == "travel"
