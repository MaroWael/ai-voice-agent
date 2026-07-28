"""
Hybrid Intent Router

Combines fast, deterministic rule-based pattern matching with an LLM fallback
classifier to categorize user queries into operational workflows, social turns
(GREETING, GOODBYE, THANKS, SMALL_TALK), or RAG search queries.

Social and operational turns short-circuit the pipeline, bypassing RAG.
Response pools return dynamic, natural responses in the turn's response_language.
"""

import logging
import random
import re
from typing import Optional

from app.conversation.models import (
    ConversationState,
    Department,
    IntentType,
    Language,
    RoutingDecision,
    WorkflowType,
)
from app.conversation.text_normalizer import TextNormalizer
from app.rag.providers.base import LLMProvider

logger = logging.getLogger(__name__)

# Pre-compiled regex patterns for Rule-Based Intent Matching
_GREETING_PATTERNS = re.compile(
    r"\b(?:السلام عليكم|مرحبا|مرحباً|ازيك|كيف حالك|صباح الخير|مساء الخير|أهلاً|اهلاً|أهلا|اهت|hello|hi|hey|good morning|good evening|welcome)\b",
    re.IGNORECASE,
)

_SMALL_TALK_PATTERNS = re.compile(
    r"\b(?:ازيك|عامل ايه|عامل إيه|كيف حالك|أخبارك|اخبارك|كيف الحقيق|how are you|what\'s up|how\'s it going|how are things)\b",
    re.IGNORECASE,
)

_THANKS_PATTERNS = re.compile(
    r"\b(?:شكراً|شكرا|تسلم|ألف شكر|الف شكر|الله يخليك|متشكر|مشكور|thank you|thanks|thanks a lot|appreciate it)\b",
    re.IGNORECASE,
)

_GOODBYE_PATTERNS = re.compile(
    r"\b(?:مع السلامة|مع السلامه|باي|سلام|أشوفك على خير|اشوفك على خير|goodbye|bye|see you|have a good day|see ya)\b",
    re.IGNORECASE,
)

_COMPLAINT_PATTERNS = re.compile(
    r"(?:عايز|عاوز|حابب|أريد|اريد|عندي)?\s*(?:أقدم|اقدم|اعمل|أعمل|اشتكي|أشتكي|تقديم)?\s*(?:شكوى|شكوي|complaint|file a complaint|make a complaint)",
    re.IGNORECASE,
)

_FRAUD_PATTERNS = re.compile(
    r"(?:احتيال|إحتيال|سرقة|سرقه|تزوير|اتسرق|إتسرق|اتسرقت|إتسرقت|ضاعت|ضاع|فقدت|أوقف|اوقف|توقيف|إيقاف|ايقاف|معملتهاش|مش بتاعتي|غير مصرح|unauthorized|stolen|lost card|block card|stolen card|report fraud|fraud)"
    r"|(?:الفيزا|البطاقة|البطاقه|الكارت|كارتي|فيزتي)\s*(?:بتاعتي|بتاعي)?\s*(?:ضاعت|ضاع|اتسرقت|اتسرق|فقدت|أوقف|اوقف)"
    r"|(?:عايز|عاوز|أريد|اريد)?\s*(?:أوقف|اوقف|أبلغ|ابلغ|توقيف|إيقاف|ايقاف)\s*(?:عن)?\s*(?:فقدان|سرقة|سرقه|ضياع)?\s*(?:البطاقة|البطاقه|الفيزا|الكارت)"
    r"|(?:عملية|عمليه|خصم|معاملة|معامله)\s*(?:مش بتاعتي|معملتهاش|غير مصرحة|غير مصرحه|احتيالية|مش أنا اللي عملتها)",
    re.IGNORECASE,
)

_TRANSFER_PATTERNS = re.compile(
    r"(?:حولني|حوّلني|كلمني|أكلم|اكلم)?\s*(?:لـ|ل|مع|إلى)?\s*(?:موظف|ممثل|انسان|إنسان|human|agent|speak to an agent|transfer me)",
    re.IGNORECASE,
)

_CUSTOMER_SERVICE_PATTERNS = re.compile(
    r"(?:اتباع|اتبعت|اتسحب|اتسحبت|بلعت|بلع)\s*(?:في|من)?\s*(?:الماكينة|الماكينه|الـ ATM|ATM|ماكينة)"
    r"|(?:أجدد|أجدّد|اجدد|تجديد)\s*(?:البطاقة|البطاقه|الكارت|الفيزا)"
    r"|(?:أغير|اغير|تغيير|تعديل)\s*(?:عنواني|العنوان|بياناتي)"
    r"|(?:نسيت)\s*(?:الرقم السري|البين كود|كلمة السر|الباسورد)"
    r"|(?:البطاقة|البطاقه|الكارت|الفيزا)\s*(?:مش شغالة|مش شغال|باظت|مكسور|مكسورة)"
    r"|(?:عايز|عاوز|أريد|اريد|كلمني|أكلم|اكلم)?\s*(?:بـ|ب|لـ|ل)?\s*(?:خدمة العملاء|خدمه العملاء|خدمة عملاء|customer service)",
    re.IGNORECASE,
)

# Response Pools
_RESPONSE_POOLS = {
    IntentType.GREETING: {
        Language.ARABIC: [
            "مرحباً، كيف يمكنني مساعدتك اليوم؟",
            "أهلاً وسهلاً، كيف أستطيع مساعدتك اليوم؟",
            "أهلاً بك، كيف يمكنني خدمتك اليوم؟",
        ],
        Language.ENGLISH: [
            "Hello! How can I help you today?",
            "Hi! How may I assist you?",
            "Welcome! What can I do for you today?",
        ],
    },
    IntentType.SMALL_TALK: {
        Language.ARABIC: [
            "أنا بخير، شكراً لسؤالك! كيف يمكنني مساعدتك اليوم؟",
            "بأحسن حال الحمد لله. كيف أستطيع خدمتك اليوم؟",
        ],
        Language.ENGLISH: [
            "I'm doing well, thank you! How can I help you today?",
            "Everything is great, thanks! How may I assist you?",
        ],
    },
    IntentType.THANKS: {
        Language.ARABIC: [
            "العفو، في خدمتك دائماً!",
            "على الرحب والسعة، يسعدني مساعدتك.",
            "العفو، أتمنى لك يوماً سعيداً!",
        ],
        Language.ENGLISH: [
            "You're welcome! Happy to help.",
            "Anytime! Let me know if you need anything else.",
            "My pleasure! Have a great day.",
        ],
    },
    IntentType.GOODBYE: {
        Language.ARABIC: [
            "شكراً لتواصلك معنا، يومك سعيد!",
            "مع السلامة، ونسعد بخدمتك دائماً.",
        ],
        Language.ENGLISH: [
            "Thank you for reaching out! Have a great day.",
            "Goodbye! Feel free to reach out anytime.",
        ],
    },
}

INTENT_LLM_PROMPT = """You are an intent classifier for an enterprise customer service AI agent.
Analyze the user utterance and categorize it into EXACTLY ONE of the following intent categories:

1. QUESTION: The user is asking an informational, banking, technical, product, or account question.
2. COMPLAINT: The user wants to file a complaint or express dissatisfaction with a service.
3. CUSTOMER_SERVICE: The user wants general customer support, card renewal, forgotten PIN, ATM swallowed card, or address update.
4. FRAUD: The user is reporting fraud, stolen card, lost card, card blocking requests, unauthorized transactions, or security emergency.
5. TRANSFER_REQUEST: The user explicitly requests to speak with a human agent or representative.
6. GREETING: The user is greeting the assistant (e.g. Hello, Hi, السلام عليكم, مرحبا).
7. GOODBYE: The user is saying goodbye (e.g. Bye, Goodbye, مع السلامة).
8. THANKS: The user is expressing gratitude (e.g. Thanks, Thank you, شكراً).
9. SMALL_TALK: The user is engaging in casual polite banter (e.g. How are you, ازيك, عامل ايه).
10. UNKNOWN: Unclear input.

Active Context: {active_context}
User Query: {query}

Output ONLY the category name in uppercase.
Category:"""


class HybridIntentRouter:
    """
    Hybrid Intent Router implementing fast rule-based checks + LLM fallback.
    """

    def __init__(self, llm_provider: Optional[LLMProvider] = None) -> None:
        self.llm_provider = llm_provider

    async def route(self, query: str, state: Optional[ConversationState] = None) -> RoutingDecision:
        """
        Classifies *query* and returns a RoutingDecision.
        """
        if not query or not query.strip():
            return RoutingDecision(
                intent=IntentType.UNKNOWN,
                department=None,
                workflow=WorkflowType.NONE,
                is_rag_required=True,
                confidence=1.0,
            )

        lang = state.response_language if state else Language.ARABIC
        text = query.strip()
        norm_text = TextNormalizer.normalize(text)

        # ── Step 1: Rule-Based Pattern Check ─────────────────────────────────
        rule_decision = self._check_rules(text, norm_text, lang, state)
        if rule_decision is not None:
            logger.info(
                "IntentRouter [Rule Match]: %s -> Department: %s, Workflow: %s",
                rule_decision.intent.value,
                rule_decision.department,
                rule_decision.workflow.value,
            )
            return rule_decision

        # ── Step 2: LLM Classifier Fallback ──────────────────────────────────
        if self.llm_provider is not None:
            try:
                active_ctx = state.conversation_summary if state else "None"
                prompt = INTENT_LLM_PROMPT.format(active_context=active_ctx, query=text)
                raw_intent = await self.llm_provider.generate(prompt)
                clean_intent_str = raw_intent.strip().upper().replace('"', "").replace("'", "")

                if clean_intent_str in IntentType.__members__:
                    intent_enum = IntentType[clean_intent_str]
                    decision = self._build_decision_for_intent(intent_enum, lang, state, confidence=0.85)
                    logger.info(
                        "IntentRouter [LLM Match]: %s -> Department: %s, Workflow: %s",
                        decision.intent.value,
                        decision.department,
                        decision.workflow.value,
                    )
                    return decision
            except Exception as exc:
                logger.warning("IntentRouter LLM classification failed (%s). Defaulting to QUESTION.", exc)

        # Default Fallback -> QUESTION / RAG Required
        return RoutingDecision(
            intent=IntentType.QUESTION,
            department=None,
            workflow=WorkflowType.NONE,
            is_rag_required=True,
            confidence=0.5,
        )

    def _check_rules(
        self,
        text: str,
        norm_text: str,
        lang: Language,
        state: Optional[ConversationState],
    ) -> Optional[RoutingDecision]:
        """Runs fast regex pattern matching against raw and normalized text."""
        # Operational Workflows
        if _FRAUD_PATTERNS.search(text) or _FRAUD_PATTERNS.search(norm_text):
            return self._build_decision_for_intent(IntentType.FRAUD, lang, state, confidence=1.0)

        if _COMPLAINT_PATTERNS.search(text) or _COMPLAINT_PATTERNS.search(norm_text):
            return self._build_decision_for_intent(IntentType.COMPLAINT, lang, state, confidence=1.0)

        if _TRANSFER_PATTERNS.search(text) or _TRANSFER_PATTERNS.search(norm_text):
            return self._build_decision_for_intent(IntentType.TRANSFER_REQUEST, lang, state, confidence=1.0)

        if _CUSTOMER_SERVICE_PATTERNS.search(text) or _CUSTOMER_SERVICE_PATTERNS.search(norm_text):
            return self._build_decision_for_intent(IntentType.CUSTOMER_SERVICE, lang, state, confidence=1.0)

        # Social Turns (SMALL_TALK, GREETING, THANKS, GOODBYE)
        if _SMALL_TALK_PATTERNS.search(text) or _SMALL_TALK_PATTERNS.search(norm_text):
            return self._build_decision_for_intent(IntentType.SMALL_TALK, lang, state, confidence=1.0)

        if _GREETING_PATTERNS.search(text) or _GREETING_PATTERNS.search(norm_text):
            return self._build_decision_for_intent(IntentType.GREETING, lang, state, confidence=1.0)

        if _THANKS_PATTERNS.search(text) or _THANKS_PATTERNS.search(norm_text):
            return self._build_decision_for_intent(IntentType.THANKS, lang, state, confidence=1.0)

        if _GOODBYE_PATTERNS.search(text) or _GOODBYE_PATTERNS.search(norm_text):
            return self._build_decision_for_intent(IntentType.GOODBYE, lang, state, confidence=1.0)

        return None

    @staticmethod
    def _build_decision_for_intent(
        intent: IntentType,
        lang: Language,
        state: Optional[ConversationState] = None,
        confidence: float = 1.0,
    ) -> RoutingDecision:
        """Constructs RoutingDecision with responses in response_language."""
        # Social Turns (Pool selection)
        if intent in _RESPONSE_POOLS:
            pool = _RESPONSE_POOLS[intent].get(lang, _RESPONSE_POOLS[intent][Language.ARABIC])
            msg = random.choice(pool)

            if intent == IntentType.GOODBYE and state is not None:
                state.is_session_idle = True

            return RoutingDecision(
                intent=intent,
                department=None,
                workflow=WorkflowType.NONE,
                message=msg,
                is_rag_required=False,
                confidence=confidence,
            )

        # Operational Workflows
        if intent == IntentType.FRAUD:
            msg = (
                "يبدو أنك تبلغ عن فقدان أو سرقة البطاقة أو حالة احتيال. لحماية حسابك وأمانك، سأقوم بتحويل طلبك فوراً إلى قسم مكافحة الاحتيال لإيقاف البطاقة واتخاذ الإجراءات اللازمة."
                if lang == Language.ARABIC
                else "It appears that you are reporting a lost or stolen card or potential fraud. For your security, I will transfer your request to the Fraud Department immediately so they can block the card and take necessary action."
            )
            return RoutingDecision(
                intent=intent,
                department=Department.FRAUD,
                workflow=WorkflowType.FRAUD,
                message=msg,
                is_rag_required=False,
                confidence=confidence,
            )

        if intent == IntentType.COMPLAINT:
            msg = (
                "يؤسفنا مواجهتك لهذه المشكلة. حرصاً منا على تقديم أفضل خدمة، سأقوم بتحويلك إلى قسم الشكاوى لمتابعة طلبك وإيجاد حل في أسرع وقت."
                if lang == Language.ARABIC
                else "We are sorry for the issue you experienced. I will transfer your request to the Complaints Department so our team can follow up and resolve it promptly."
            )
            return RoutingDecision(
                intent=intent,
                department=Department.COMPLAINTS,
                workflow=WorkflowType.COMPLAINT,
                message=msg,
                is_rag_required=False,
                confidence=confidence,
            )

        if intent == IntentType.CUSTOMER_SERVICE:
            msg = (
                "يسعدنا مساعدتك. سأقوم بتحويلك الآن إلى قسم خدمة العملاء لمتابعة وتنفيذ طلبك."
                if lang == Language.ARABIC
                else "I will transfer your request to Customer Service so they can assist you with your request."
            )
            return RoutingDecision(
                intent=intent,
                department=Department.CUSTOMER_SERVICE,
                workflow=WorkflowType.CUSTOMER_SERVICE,
                message=msg,
                is_rag_required=False,
                confidence=confidence,
            )

        if intent == IntentType.TRANSFER_REQUEST:
            msg = (
                "جاري تحويلك الآن للتحدث المباشر مع أحد ممثلي خدمة العملاء."
                if lang == Language.ARABIC
                else "Connecting you now to speak directly with a customer service representative."
            )
            return RoutingDecision(
                intent=intent,
                department=Department.HUMAN_AGENT,
                workflow=WorkflowType.HUMAN_AGENT,
                message=msg,
                is_rag_required=False,
                confidence=confidence,
            )

        return RoutingDecision(
            intent=intent,
            department=None,
            workflow=WorkflowType.NONE,
            message=None,
            is_rag_required=True,
            confidence=confidence,
        )
