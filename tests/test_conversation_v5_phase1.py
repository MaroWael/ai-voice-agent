"""
Tests for Dialogue Manager v5 Phase 1 Models & Redis Store

Verifies:
  1. Strongly typed enums, structs, and ConversationState instantiation.
  2. EntityStack deduplication, LIFO ordering, ordinal lookup, and timeline tracking.
  3. ConversationState transactional snapshot creation and rollback.
  4. RedisConversationStore state serialization and retrieval.
"""

import pytest
from unittest.mock import AsyncMock, patch

from app.conversation.models import (
    ConversationEntity,
    ConversationStage,
    ConversationState,
    EntityStack,
    EntityType,
    IntentType,
    Language,
    SlotValue,
    Topic,
    UserProfile,
)
from app.conversation.storage.redis_store import RedisConversationStore


def test_entity_stack_deduplication_and_ordinals():
    stack = EntityStack()
    
    e1 = ConversationEntity(id="visa_platinum", display_name="Visa Platinum", entity_type=EntityType.CREDIT_CARD)
    e2 = ConversationEntity(id="visa_gold", display_name="Visa Gold", entity_type=EntityType.CREDIT_CARD)
    
    stack.push(e1, turn=1)
    stack.push(e2, turn=2)
    
    # Check top of stack (peek)
    assert stack.peek().id == "visa_gold"
    
    # Check ordinal access
    assert stack.get_by_index(0).id == "visa_gold"
    assert stack.get_by_index(1).id == "visa_platinum"
    
    # Check chronological first
    assert stack.get_chronological_first().id == "visa_platinum"
    
    # Push e1 again -> Deduplicates and moves e1 to top
    stack.push(e1, turn=3)
    assert stack.peek().id == "visa_platinum"
    assert stack.get_by_index(0).id == "visa_platinum"
    assert stack.get_by_index(1).id == "visa_gold"
    assert len(stack.stack) == 2
    assert len(stack.timeline) == 3


def test_state_snapshot_and_rollback():
    state = ConversationState(session_id="snap-test")
    state.active_topic = Topic.GENERAL_INFO
    state.user_profile.salary_egp = 20000.0
    
    # Create snapshot before turn mutation
    state.create_snapshot()
    
    # Mutate state during turn processing
    state.active_topic = Topic.FEES
    state.user_profile.salary_egp = 50000.0
    state.turn_count += 1
    
    assert state.active_topic == Topic.FEES
    assert state.user_profile.salary_egp == 50000.0
    
    # Rollback upon simulated turn processing error
    state.rollback()
    
    # Verify pre-turn state is completely restored
    assert state.active_topic == Topic.GENERAL_INFO
    assert state.user_profile.salary_egp == 20000.0
    assert state.turn_count == 0


def test_conversation_state_json_roundtrip():
    state = ConversationState(session_id="json-test")
    e1 = ConversationEntity(id="visa_platinum", display_name="Visa Platinum", entity_type=EntityType.CREDIT_CARD)
    state.entity_stack.push(e1, turn=1)
    state.user_profile.salary_egp = 30000.0
    state.slots["salary_egp"] = SlotValue(name="salary_egp", value=30000.0, confidence=0.95)
    
    json_str = state.model_dump_json()
    restored = ConversationState.model_validate_json(json_str)
    
    assert restored.session_id == "json-test"
    assert restored.active_entity.id == "visa_platinum"
    assert restored.user_profile.salary_egp == 30000.0
    assert restored.slots["salary_egp"].value == 30000.0


@pytest.mark.asyncio
async def test_redis_store_mock_roundtrip():
    store = RedisConversationStore(key_prefix="test:")
    state = ConversationState(session_id="redis-test")
    e1 = ConversationEntity(id="visa_gold", display_name="Visa Gold", entity_type=EntityType.CREDIT_CARD)
    state.entity_stack.push(e1, turn=1)
    
    mock_redis = AsyncMock()
    mock_redis.get.return_value = state.model_dump_json()
    mock_redis.set.return_value = True
    
    with patch("app.conversation.storage.redis_store.get_redis", return_value=mock_redis):
        await store.save_state(state)
        mock_redis.set.assert_called_once()
        
        loaded = await store.get_state("redis-test")
        assert loaded.session_id == "redis-test"
        assert loaded.active_entity.id == "visa_gold"
