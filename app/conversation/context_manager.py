"""
Context Manager

Prepares structured conversational context representations from ConversationState.
Formats active entity, active topic, active workflow, conversation summary, and
recent exchanges into a unified context block for LLM query disambiguation and prompt generation.
"""

import logging
from app.conversation.models import ConversationState

logger = logging.getLogger(__name__)


class ContextManager:
    """
    Builds structured conversational context representation for RAG and Query Rewriter.
    """

    @staticmethod
    def build_conversation_context(state: ConversationState) -> str:
        """
        Formats state history, entity, topic, workflow, and summary into a structured context string.

        Format:
        Conversation Context

        Current Active Entity:
        <entity_display_name or None>

        Current Topic:
        <topic_value>

        Current Workflow:
        <workflow_value>

        Conversation Summary:
        <summary or None>

        Recent Exchanges:
        - User: <text>
        - Assistant: <text>
        """
        entity_name = state.active_entity.display_name if state.active_entity else "None"
        topic_name = state.active_topic.value if state.active_topic else "None"
        workflow_name = state.active_workflow.value if state.active_workflow else "None"
        summary_text = state.conversation_summary.strip() if state.conversation_summary else "None"

        parts = [
            "Conversation Context",
            "",
            "Current Active Entity:",
            entity_name,
            "",
            "Current Topic:",
            topic_name,
            "",
            "Current Workflow:",
            workflow_name,
            "",
            "Conversation Summary:",
            summary_text,
        ]

        if state.history:
            parts.extend(["", "Recent Exchanges:"])
            for msg in state.history:
                role_label = "User" if msg.role == "user" else "Assistant"
                parts.append(f"- {role_label}: {msg.content}")

        context_str = "\n".join(parts)
        logger.debug("ContextManager produced conversation context (%d chars)", len(context_str))
        return context_str
