"""
Context Compressor

Implements a 6-turn sliding window history compression engine.
Consolidates older turns into state.conversation_summary while preserving recent raw turns.
"""

import logging
from app.conversation.models import ConversationState

logger = logging.getLogger(__name__)


class ContextCompressor:
    """
    Compresses conversation history to bound token usage.
    """

    def compress_if_needed(self, state: ConversationState, max_raw_turns: int = 6) -> None:
        """
        Trims raw history beyond *max_raw_turns* exchange pairs (max_raw_turns * 2 messages)
        and consolidates key user turn facts into conversation_summary.
        """
        max_messages = max_raw_turns * 2
        if len(state.history) <= max_messages:
            return

        older_messages = state.history[:-max_messages]
        recent_messages = state.history[-max_messages:]

        summary_lines = []
        if state.conversation_summary:
            summary_lines.append(state.conversation_summary)

        for msg in older_messages:
            if msg.role == "user":
                summary_lines.append(f"User asked: {msg.content}")

        state.conversation_summary = " | ".join(summary_lines[-5:])
        state.history = recent_messages

        logger.info(
            "ContextCompressor: Compressed history to %d messages. Summary updated.",
            len(state.history),
        )
