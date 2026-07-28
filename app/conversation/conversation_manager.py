"""
Conversation Manager

Session-wide orchestrator for the Conversation Layer v2.
Coordinates per-turn independent language evaluation, intent routing with dynamic response pools,
early pipeline short-circuiting, entity preservation & switching, continuous topic updates,
rule-based query rewriting, session memory persistence, and rich turn diagnostics.
"""

from dataclasses import replace
import logging
from typing import Awaitable, Callable, Optional

from app.config.settings import settings
from app.conversation.context_manager import ContextManager
from app.conversation.entity_topic_extractor import EntityAndTopicExtractor
from app.conversation.language_manager import ConversationLanguageManager
from app.conversation.models import ConversationState, RoutingDecision
from app.conversation.rewriter.query_rewriter import ConversationQueryRewriter
from app.conversation.router.intent_router import HybridIntentRouter
from app.conversation.storage.redis_store import RedisConversationStore
from app.conversation.summarizer import ConversationSummarizer
from app.conversation.text_normalizer import TextNormalizer
from input.models.transcription import Transcription
from llm.models import AIResponse

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Session-wide conversation manager coordinating all conversation layer components.
    """

    def __init__(
        self,
        store: Optional[RedisConversationStore] = None,
        language_manager: Optional[ConversationLanguageManager] = None,
        router: Optional[HybridIntentRouter] = None,
        entity_topic_extractor: Optional[EntityAndTopicExtractor] = None,
        context_manager: Optional[ContextManager] = None,
        rewriter: Optional[ConversationQueryRewriter] = None,
        summarizer: Optional[ConversationSummarizer] = None,
    ) -> None:
        self.store = store or RedisConversationStore()
        self.language_manager = language_manager or ConversationLanguageManager()
        self.router = router or HybridIntentRouter()
        self.entity_topic_extractor = entity_topic_extractor or EntityAndTopicExtractor()
        self.context_manager = context_manager or ContextManager()
        self.rewriter = rewriter or ConversationQueryRewriter()
        self.summarizer = summarizer or ConversationSummarizer()

    async def process_transcript(
        self,
        session_id: str,
        transcription: Transcription,
        rag_executor: Callable[[Transcription, str, str], Awaitable[AIResponse]],
    ) -> AIResponse:
        """
        Processes a single conversation turn through the redesigned pipeline.

        Pipeline Order:
          1. Load session state from Redis.
          2. ConversationLanguageManager -> Evaluate turn language independently.
          3. IntentRouter -> Classify intent & check if RAG is required.
          4. Early Short-Circuiting:
             - If is_rag_required is False: Construct routing or pool response (RAG Bypassed).
             - If is_rag_required is True:
                 a. EntityAndTopicExtractor -> Update active_entity and active_topic per turn.
                 b. ContextManager -> Build structured conversation context.
                 c. ConversationQueryRewriter -> Standalone search query generation.
                 d. Execute rag_executor.
          5. Universal Memory Persistence & Summary Update.
          6. Rich Turn Diagnostic Logging.
        """
        user_text = transcription.text.strip() if transcription.text else ""
        norm_text = TextNormalizer.normalize(user_text)

        # 1. Fetch Session State from Redis
        state = await self.store.get_state(session_id)

        # 2. Per-Turn Independent Language Evaluation
        detected_lang, response_lang, lang_conf = self.language_manager.evaluate_turn_language(
            state=state,
            text=user_text,
            whisper_language=transcription.language,
        )

        # 3. Intent Routing & Workflow Decision
        decision: RoutingDecision = await self.router.route(user_text, state)
        state.last_intent = decision.intent.value
        state.active_workflow = decision.workflow

        rag_executed = False
        rewrite_applied = False
        rule_applied = False
        rewrite_reason = "None"
        standalone_query = user_text
        ent_switched = False
        ent_preserved = False

        # 4. Turn Processing
        if not decision.is_rag_required:
            # Operational Workflow / Social Turn Routing (RAG Bypassed)
            rag_executed = False
            rewrite_reason = f"Pipeline Short-Circuited ({decision.intent.value})"
            response = AIResponse(
                action="ROUTE",
                department=decision.department,
                reason=f"Routed via IntentRouter to {decision.intent.value}",
                message=decision.message or "تم تحويل طلبك بنجاح.",
                language=response_lang.value,
            )
        else:
            # Informational Query / RAG Required Path
            rag_executed = True

            # a. Entity & Topic Extraction
            entity, topic, ent_switched, ent_preserved = self.entity_topic_extractor.extract(user_text, state)

            # b. Build Structured Conversation Context
            conv_context = self.context_manager.build_conversation_context(state)

            # c. Query Rewriting with Follow-up Rule Engine & Confidence Guards
            standalone_query, rewrite_applied, rewrite_reason, rule_applied = await self.rewriter.rewrite(
                query=user_text,
                state=state,
                conversation_context=conv_context,
            )

            # d. Execute RAG Pipeline
            raw_response = await rag_executor(transcription, conv_context, standalone_query)
            # Replace language with turn's response_language code
            response = replace(raw_response, language=response_lang.value)

        # 5. Universal Conversation Memory Updates for ALL Turns
        state.add_exchange(
            user_content=user_text,
            assistant_content=response.message,
            max_turns=settings.MAX_CONVERSATION_TURNS,
            user_meta={
                "intent": decision.intent.value,
                "detected_language": detected_lang.value,
                "response_language": response_lang.value,
            },
            assistant_meta={
                "action": response.action,
                "department": response.department,
            },
        )

        # Conditionally Update Conversation Summary
        new_active_ctx = (
            f"Entity: {state.active_entity.display_name if state.active_entity else 'None'}, "
            f"Topic: {state.active_topic.value}, Workflow: {state.active_workflow.value}"
        )
        await self.summarizer.update_summary_if_needed(
            state=state,
            latest_user_text=user_text,
            latest_assistant_text=response.message,
            new_active_context=new_active_ctx,
        )

        # Persist Updated State to Redis
        await self.store.save_state(state)

        # 6. Rich Diagnostic Logging
        ent_display = state.active_entity.display_name if state.active_entity else "None"
        ent_id = state.active_entity.id if state.active_entity else "none"
        ent_conf = state.active_entity.confidence if state.active_entity else 0.0

        diag_log = (
            "\n==================== CONVERSATION TURN DIAGNOSTICS ====================\n"
            f"Session ID:              {session_id}\n"
            f"User Query:              {user_text!r}\n"
            f"Normalized Query:        {norm_text!r}\n"
            f"Detected Language:       {detected_lang.value} (confidence: {lang_conf:.2f})\n"
            f"Response Language:       {response_lang.value}\n"
            f"Current Entity:          {ent_display} [id={ent_id}, confidence={ent_conf:.2f}]\n"
            f"Entity Preserved:        {ent_preserved}\n"
            f"Entity Switched:         {ent_switched}\n"
            f"Current Topic:           {state.active_topic.value}\n"
            f"Current Workflow:        {state.active_workflow.value}\n"
            f"Intent:                  {decision.intent.value}\n"
            f"Department:              {decision.department.value if decision.department else 'None'}\n"
            f"RAG Executed:            {rag_executed}\n"
            f"Rewrite Applied:         {rewrite_applied}\n"
            f"Rule Rewrite Applied:    {rule_applied}\n"
            f"Rewrite Reason:          {rewrite_reason}\n"
            f"Standalone Query:        {standalone_query!r}\n"
            f"Session Idle Flag:       {state.is_session_idle}\n"
            f"Assistant Message:       {response.message!r}\n"
            "======================================================================="
        )
        logger.info(diag_log)

        return response
