"""
Conversation Domain Models

Defines core data structures for short-term memory, conversation state,
typed banking entities, operational workflows, and intent routing decisions.
"""

from enum import Enum
from typing import Any, Literal, Optional
import time
from pydantic import BaseModel, Field


class Language(str, Enum):
    """Supported conversation languages."""
    ARABIC = "ar"
    ENGLISH = "en"


class WorkflowType(str, Enum):
    """Operational workflow categories."""
    NONE = "NONE"
    FRAUD = "FRAUD"
    COMPLAINT = "COMPLAINT"
    CUSTOMER_SERVICE = "CUSTOMER_SERVICE"
    HUMAN_AGENT = "HUMAN_AGENT"


class EntityType(str, Enum):
    """Banking product/service entity classification."""
    CREDIT_CARD = "CREDIT_CARD"
    DEBIT_CARD = "DEBIT_CARD"
    CERTIFICATE = "CERTIFICATE"
    LOAN = "LOAN"
    ACCOUNT = "ACCOUNT"
    OTHER = "OTHER"


class Topic(str, Enum):
    """Conversation topics for banking inquiries."""
    GENERAL_INFO = "GENERAL_INFO"
    FEES = "FEES"
    REQUIREMENTS = "REQUIREMENTS"
    REPLACEMENT = "REPLACEMENT"
    BENEFITS = "BENEFITS"
    UNKNOWN = "UNKNOWN"


class IntentType(str, Enum):
    """Supported intent classification categories."""
    QUESTION = "QUESTION"
    COMPLAINT = "COMPLAINT"
    CUSTOMER_SERVICE = "CUSTOMER_SERVICE"
    FRAUD = "FRAUD"
    TRANSFER_REQUEST = "TRANSFER_REQUEST"
    GREETING = "GREETING"
    GOODBYE = "GOODBYE"
    THANKS = "THANKS"
    SMALL_TALK = "SMALL_TALK"
    UNKNOWN = "UNKNOWN"


class Department(str, Enum):
    """Target department category for operational routing."""
    FRAUD = "FRAUD"
    COMPLAINTS = "COMPLAINTS"
    CUSTOMER_SERVICE = "CUSTOMER_SERVICE"
    HUMAN_AGENT = "HUMAN_AGENT"
    UNKNOWN = "UNKNOWN"


class ConversationEntity(BaseModel):
    """
    Represents a banking product or service entity tracked across multi-turn exchanges.
    """
    id: str
    display_name: str
    entity_type: EntityType
    aliases: list[str] = Field(default_factory=list)
    normalized_aliases: list[str] = Field(default_factory=list)
    confidence: float = 1.0


class ConversationMessage(BaseModel):
    """
    Represents a single message turn (user or assistant) in conversation history.
    """
    role: Literal["user", "assistant"]
    content: str
    timestamp: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationState(BaseModel):
    """
    Session-wide conversation state stored in Redis.
    Shared across all routing decisions, entity tracking, and workflows.
    """
    session_id: str
    detected_language: Language = Language.ARABIC
    response_language: Language = Language.ARABIC
    active_entity: Optional[ConversationEntity] = None
    active_topic: Topic = Topic.GENERAL_INFO
    active_workflow: WorkflowType = WorkflowType.NONE
    is_session_idle: bool = False
    conversation_summary: str = ""
    history: list[ConversationMessage] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    last_intent: Optional[str] = None
    last_standalone_query: Optional[str] = None
    last_successful_query: Optional[str] = None
    last_answer_source: Optional[str] = None
    turn_count: int = 0
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    last_activity_at: float = Field(default_factory=time.time)
    last_entity_update_turn: int = 0

    def add_exchange(
        self,
        user_content: str,
        assistant_content: str,
        max_turns: int = 3,
        user_meta: Optional[dict[str, Any]] = None,
        assistant_meta: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Appends a complete exchange turn pair (User + Assistant).
        Enforces max_turns retention limit.
        """
        now = time.time()
        self.history.append(
            ConversationMessage(
                role="user",
                content=user_content,
                timestamp=now,
                metadata=user_meta or {},
            )
        )
        self.history.append(
            ConversationMessage(
                role="assistant",
                content=assistant_content,
                timestamp=now,
                metadata=assistant_meta or {},
            )
        )
        self.turn_count += 1
        self.updated_at = now
        self.last_activity_at = now

        # Retain at most max_turns exchange pairs (max_turns * 2 messages)
        max_messages = max_turns * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    def reset(self) -> None:
        """
        Resets conversational context properties while preserving session_id.
        Prevents stale entity context from leaking into new sessions.
        """
        self.active_entity = None
        self.active_topic = Topic.GENERAL_INFO
        self.active_workflow = WorkflowType.NONE
        self.is_session_idle = False
        self.conversation_summary = ""
        self.history = []
        self.last_intent = None
        self.last_standalone_query = None
        self.last_successful_query = None
        self.last_answer_source = None
        self.last_entity_update_turn = 0
        self.updated_at = time.time()
        self.last_activity_at = time.time()


class RoutingDecision(BaseModel):
    """
    Represents the outcome of intent classification and department routing.
    """
    intent: IntentType
    department: Optional[Department] = None
    workflow: WorkflowType = WorkflowType.NONE
    message: Optional[str] = None
    is_rag_required: bool = True
    confidence: float = 1.0
