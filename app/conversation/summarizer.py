"""
Conversation Summarizer

Maintains an incremental summary of the conversation session.
Avoids unnecessary LLM generation calls on every single turn. Updates only on:
  1. Significant context/topic shifts
  2. Reaching turn interval thresholds (e.g. every 3 exchange turns)
"""

import logging
from typing import Optional

from app.config.settings import settings
from app.conversation.models import ConversationState
from app.rag.providers.base import LLMProvider

logger = logging.getLogger(__name__)

SUMMARY_UPDATE_PROMPT = """You are an AI conversation summarizer for a customer support agent.
Update the existing conversation summary using the current summary and the latest user/assistant exchange pair.

CRITICAL RULES:
1. Keep the summary under 3 short sentences.
2. Focus ONLY on key products mentioned, active request types (e.g. Visa Gold, Complaint, Lost card), and customer goal.
3. Do NOT include polite greetings, chatter, or redundant phrasing.
4. Output ONLY the updated summary text.

Previous Summary: {previous_summary}
Active Context: {active_context}

Latest User Turn: {user_turn}
Latest Assistant Turn: {assistant_turn}

Updated Summary:"""


class ConversationSummarizer:
    """
    Summarizer for ConversationState.
    """

    def __init__(self, llm_provider: Optional[LLMProvider] = None) -> None:
        self.llm_provider = llm_provider
        self.update_interval = settings.SUMMARY_UPDATE_INTERVAL_TURNS

    async def update_summary_if_needed(
        self,
        state: ConversationState,
        latest_user_text: str,
        latest_assistant_text: str,
        new_active_context: Optional[str] = None,
    ) -> None:
        """
        Evaluates whether a summary update is warranted, and updates state.conversation_summary in-place.
        """
        if self.llm_provider is None:
            return

        context_shifted = bool(
            new_active_context and new_active_context != state.active_context
        )
        interval_reached = bool(
            state.turn_count > 0 and state.turn_count % self.update_interval == 0
        )

        if not (context_shifted or interval_reached or not state.conversation_summary):
            logger.debug(
                "Skipping summary update for session %s (turn %d)",
                state.session_id,
                state.turn_count,
            )
            return

        prev_summary = state.conversation_summary or "None"
        act_ctx = new_active_context or state.active_context or "General Inquiry"

        prompt = SUMMARY_UPDATE_PROMPT.format(
            previous_summary=prev_summary,
            active_context=act_ctx,
            user_turn=latest_user_text,
            assistant_turn=latest_assistant_text,
        )

        try:
            new_summary = await self.llm_provider.generate(prompt)
            clean_summary = new_summary.strip().replace("\n", " ")
            if clean_summary:
                state.conversation_summary = clean_summary
                logger.info(
                    "ConversationSummarizer updated session %s summary: %r",
                    state.session_id,
                    clean_summary,
                )
        except Exception as exc:
            logger.warning("ConversationSummarizer failed for session %s (%s).", state.session_id, exc)
