"""
Conversation Query Rewriter

Transforms multi-turn follow-up questions into explicit, self-contained search queries.
Uses a deterministic Rule Engine for follow-up detection and data-driven pattern matching
without invoking an LLM. Fallbacks to LLM generation only when rule matching yields no result.
"""

import logging
import re
from typing import Optional

from app.conversation.models import ConversationState, Language, Topic
from app.conversation.text_normalizer import TextNormalizer
from app.rag.providers.base import LLMProvider

logger = logging.getLogger(__name__)

_ARABIC_SCRIPT_RE = re.compile(r"[\u0600-\u06FF]")
_LATIN_SCRIPT_RE = re.compile(r"[a-zA-Z]")

# Pronoun & Ambiguous Reference Regex Patterns
_PRONOUN_PATTERNS = re.compile(
    r"\b(?:it|its|this|that|them|those|بتاعها|بتاعه|بتاعته|دي|ده|هي|هو)\b",
    re.IGNORECASE,
)

# Topic Continuation Keywords Regex Patterns
_TOPIC_CONTINUATION_PATTERNS = re.compile(
    r"\b(?:fees|fee|cost|benefits|perks|replacement|renewal|requirements|salary|how much|الرسوم|المصاريف|المميزات|الشروط|الاوراق|المستندات|البديل|التجديد|كام|قد ايه|قد إيه|ولو|طب)\b",
    re.IGNORECASE,
)

# Data-Driven Rewrite Templates (Mapped by Topic and Language)
_REWRITE_TEMPLATES = {
    (Topic.FEES, Language.ENGLISH): "What are the {entity} fees?",
    (Topic.FEES, Language.ARABIC): "ما هي رسوم بطاقة {entity}؟",
    (Topic.BENEFITS, Language.ENGLISH): "What are the benefits of {entity}?",
    (Topic.BENEFITS, Language.ARABIC): "ما هي مميزات بطاقة {entity}؟",
    (Topic.REPLACEMENT, Language.ENGLISH): "What happens if {entity} is lost?",
    (Topic.REPLACEMENT, Language.ARABIC): "ما هي إجراءات ورسوم استبدال بطاقة {entity} عند الفقدان؟",
    (Topic.REQUIREMENTS, Language.ENGLISH): "What are the requirements for {entity}?",
    (Topic.REQUIREMENTS, Language.ARABIC): "ما هي شروط ومتطلبات الحصول على {entity}؟",
    (Topic.GENERAL_INFO, Language.ENGLISH): "Tell me about {entity}.",
    (Topic.GENERAL_INFO, Language.ARABIC): "ما هي تفاصيل ومعلومات بطاقة {entity}؟",
}

QUERY_REWRITER_PROMPT = """You are an expert banking query rewriter.
Your task is to transform the User Query into an EXPLICIT, STANDALONE search query using the provided active banking entity, topic, and conversation context.

Active State:
- Current Active Entity: {active_entity}
- Current Topic: {active_topic}
- Current Workflow: {active_workflow}

Conversation Context:
{context}

CRITICAL INSTRUCTIONS:
1. Context Resolution & Explicit Naming:
   If the User Query lacks an explicit product or service name, YOU MUST EXPLICITLY INJECT the Current Active Entity into the query.
2. Bypass Rewriting for Complete Queries:
   If the current question ALREADY contains a complete product or entity name, DO NOT rewrite it. Return it unchanged.
3. Output Constraints:
   - Output ONLY the final rewritten standalone query string.
   - Do NOT answer the question. Do NOT add preamble or quotes.

User Query:
{query}

Standalone Query:"""


class ConversationQueryRewriter:
    """
    Rewrites conversational follow-up questions into standalone search queries.
    Merges deterministic follow-up detection with data-driven rule rewriting.
    """

    def __init__(self, llm_provider: Optional[LLMProvider] = None) -> None:
        self.llm_provider = llm_provider

    async def rewrite(
        self,
        query: str,
        state: ConversationState,
        conversation_context: str = "",
    ) -> tuple[str, bool, str, bool]:
        """
        Transforms *query* into a standalone search query.

        Returns:
            Tuple of (standalone_query, is_rewritten, rewrite_reason, is_rule_applied)
        """
        if not query or not query.strip():
            return query, False, "Empty Query", False

        clean_query = query.strip()
        norm_query = TextNormalizer.normalize(clean_query)

        # Infer turn query language for template formatting
        if _ARABIC_SCRIPT_RE.search(clean_query):
            query_lang = Language.ARABIC
        elif _LATIN_SCRIPT_RE.search(clean_query):
            query_lang = Language.ENGLISH
        else:
            query_lang = getattr(state, "response_language", Language.ARABIC)

        # ── 1. Check if Query already contains an explicit entity ────────────
        if state.active_entity:
            for alias in state.active_entity.normalized_aliases:
                if alias and (re.search(r"\b" + re.escape(alias) + r"\b", norm_query) or alias in norm_query):
                    logger.info(
                        "ConversationQueryRewriter: Query explicitly contains active entity '%s'. Bypassing rewrite.",
                        state.active_entity.display_name,
                    )
                    return clean_query, False, "Explicit Entity Found", False

        # ── 2. Entity Confidence Guard (Must be >= 0.80) ────────────────────
        if state.active_entity and state.active_entity.confidence < 0.80:
            logger.info(
                "ConversationQueryRewriter: Entity confidence too low (%.2f < 0.80). Bypassing rewrite.",
                state.active_entity.confidence,
            )
            return clean_query, False, "Low Entity Confidence", False

        # ── 3. Deterministic Follow-up Rule Engine ──────────────────────────
        is_followup, followup_reason = self._evaluate_followup_rule_engine(norm_query, clean_query, state)
        if not is_followup:
            logger.info("ConversationQueryRewriter: Query is not a follow-up (%s). Bypassing rewrite.", followup_reason)
            return clean_query, False, "Not a Follow-up", False

        # ── 4. Data-Driven Rule Rewrite ──────────────────────────────────────
        rule_rewritten = self._apply_rule_rewrite(state, query_lang)
        if rule_rewritten:
            logger.info(
                "ConversationQueryRewriter [Rule Match]: Rewrote %r -> %r (Reason: %s)",
                clean_query,
                rule_rewritten,
                followup_reason,
            )
            return rule_rewritten, True, f"Rule-Based Rewrite ({followup_reason})", True

        # ── 5. Fallback: LLM Rewrite ─────────────────────────────────────────
        if self.llm_provider is not None and conversation_context:
            try:
                entity_name = state.active_entity.display_name if state.active_entity else "None"
                topic_name = state.active_topic.value if state.active_topic else "None"
                workflow_name = state.active_workflow.value if state.active_workflow else "None"

                prompt = QUERY_REWRITER_PROMPT.format(
                    active_entity=entity_name,
                    active_topic=topic_name,
                    active_workflow=workflow_name,
                    context=conversation_context,
                    query=clean_query,
                )

                raw_rewritten = await self.llm_provider.generate(prompt)
                clean_rewritten = raw_rewritten.strip().replace("\n", " ").replace('"', "").replace("'", "")

                if clean_rewritten and clean_rewritten != clean_query:
                    return clean_rewritten, True, "LLM Rewrite Fallback", False
            except Exception as exc:
                logger.warning("ConversationQueryRewriter LLM fallback failed (%s).", exc)

        return clean_query, False, "Fallback to Original", False

    def _evaluate_followup_rule_engine(
        self,
        norm_query: str,
        clean_query: str,
        state: ConversationState,
    ) -> tuple[bool, str]:
        """
        Deterministic Rule Engine for follow-up detection:

        IF (PronounDetected) -> True
        ELSE IF (TopicContinuation AND MissingExplicitEntity) -> True
        ELSE IF (ShortQuery AND PreviousEntityExists AND TurnDistance <= 3) -> True
        ELSE -> False
        """
        if not state.active_entity:
            return False, "No Active Entity"

        turn_distance = state.turn_count - state.last_entity_update_turn

        # Rule 1: Pronoun Detected
        if _PRONOUN_PATTERNS.search(clean_query) or _PRONOUN_PATTERNS.search(norm_query):
            return True, "Pronoun Detected"

        # Rule 2: Topic Continuation & Missing Explicit Entity
        if _TOPIC_CONTINUATION_PATTERNS.search(clean_query) or _TOPIC_CONTINUATION_PATTERNS.search(norm_query):
            return True, "Topic Continuation"

        # Rule 3: Very Short Query (<= 3 words) & Previous Entity Exists & TurnDistance <= 3
        word_count = len(clean_query.split())
        if word_count <= 3 and turn_distance <= 3:
            return True, "Short Query Continuation"

        return False, "Not a Follow-up"

    def _apply_rule_rewrite(self, state: ConversationState, lang: Language) -> Optional[str]:
        """
        Executes data-driven template formatting using state.active_topic and state.active_entity.
        """
        if not state.active_entity:
            return None

        topic = state.active_topic or Topic.GENERAL_INFO
        template = _REWRITE_TEMPLATES.get((topic, lang))

        if not template:
            fallback_lang = Language.ENGLISH if lang == Language.ARABIC else Language.ARABIC
            template = _REWRITE_TEMPLATES.get((topic, fallback_lang))

        if template:
            return template.format(entity=state.active_entity.display_name)

        return None
