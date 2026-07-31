"""
Tests for Dialogue Manager v5 Phase 3 EntityResolver & ReferenceResolver

Verifies:
  1. Entity extraction from text and EntityStack pushing.
  2. Multi-entity extraction for comparison requests.
  3. Pronoun reference resolution ("its fees" -> top of EntityStack).
  4. Ordinal reference resolution ("the first one", "the second one").
  5. Chronological reference resolution ("go back to the first card").
"""

import pytest

from app.conversation.entity_resolver import EntityResolver
from app.conversation.models import ConversationState
from app.conversation.reference_resolver import ReferenceResolver


def test_entity_resolver_single_and_multi():
    resolver = EntityResolver()
    state = ConversationState(session_id="ent-v5")

    # Turn 1: Single Entity Mention
    ents1 = resolver.extract_entities("Tell me about Visa Platinum", state)
    assert len(ents1) == 1
    assert ents1[0].id == "visa_platinum"
    assert state.entity_stack.peek().id == "visa_platinum"

    # Turn 2: Multi-Entity Mention
    ents2 = resolver.extract_entities("Compare Visa Gold and Visa Platinum", state)
    assert len(ents2) == 2
    extracted_ids = [e.id for e in ents2]
    assert "visa_gold" in extracted_ids
    assert "visa_platinum" in extracted_ids
    # Visa Platinum was pushed last or Gold pushed last depending on iteration order
    assert len(state.entity_stack.stack) == 2


def test_reference_resolver_pronoun_and_ordinals():
    resolver = EntityResolver()
    ref_resolver = ReferenceResolver()
    state = ConversationState(session_id="ref-v5")

    # Setup stack with 3 products: Platinum -> Gold -> Signature
    resolver.extract_entities("Visa Platinum", state)
    resolver.extract_entities("Visa Gold", state)
    resolver.extract_entities("Visa Signature", state)

    # Stack top (index 0) is Signature, index 1 is Gold, index 2 is Platinum
    assert state.entity_stack.get_by_index(0).id == "visa_signature"
    assert state.entity_stack.get_by_index(1).id == "visa_gold"
    assert state.entity_stack.get_by_index(2).id == "visa_platinum"

    # Test Pronoun Resolution -> Top of stack (Signature)
    is_res1, ent1 = ref_resolver.resolve_reference("What are its fees?", state)
    assert is_res1 is True
    assert ent1.id == "visa_signature"

    # Test Ordinal First -> Index 0 (Signature)
    is_res2, ent2 = ref_resolver.resolve_reference("Tell me about the first one", state)
    assert is_res2 is True
    assert ent2.id == "visa_signature"

    # Test Ordinal Second -> Index 1 (Gold)
    is_res3, ent3 = ref_resolver.resolve_reference("What about the second one?", state)
    assert is_res3 is True
    assert ent3.id == "visa_gold"

    # Test Chronological First -> First discussed in timeline (Platinum)
    is_res4, ent4 = ref_resolver.resolve_reference("Go back to the first card we discussed", state)
    assert is_res4 is True
    assert ent4.id == "visa_platinum"
