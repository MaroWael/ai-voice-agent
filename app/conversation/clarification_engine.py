"""
Clarification Engine

Generates targeted clarifying prompts for missing entities, missing recommendation slots,
and ambiguous topic queries.
"""

import logging
from app.conversation.models import ConversationState, Language
from llm.models import AIResponse

logger = logging.getLogger(__name__)


class ClarificationEngine:
    """
    Generates targeted clarification prompts.
    """

    def generate_clarification(self, reason: str, state: ConversationState) -> AIResponse:
        """
        Generates AIResponse containing localized clarification question.
        """
        lang = state.response_language

        if reason == "NEEDS_CLARIFICATION_COMPARISON":
            entity_name = state.entity_stack.peek().display_name if state.entity_stack.peek() else "البطاقة"
            msg = (
                f"ما هي البطاقة الثانية التي تود مقارنتها مع بطاقة {entity_name}؟ (مثال: Visa Gold أو Visa Signature)؟"
                if lang == Language.ARABIC
                else f"Which second card would you like to compare with {entity_name}? (e.g. Visa Gold or Visa Signature)?"
            )
        elif reason == "NEEDS_SLOTS_RECOMMENDATION":
            msg = (
                "لترشيح البطاقة الأنسب لك، برجاء توضيح قيمة دخل الشهري والهدف الرئيسي من البطاقة (سفر، تسوق، أو كاش باك)؟"
                if lang == Language.ARABIC
                else "To recommend the best card, could you please share your monthly salary and primary purpose for the card (travel, shopping, or cashback)?"
            )
        elif reason == "AMBIGUOUS_CARD_VS_BRANCH_FEE":
            entity_name = state.entity_stack.peek().display_name if state.entity_stack.peek() else "البطاقة"
            msg = (
                f"هل تسأل عن رسوم ومصاريف بطاقة {entity_name} أم عن رسوم الخدمات في الفرع؟"
                if lang == Language.ARABIC
                else f"Are you asking about {entity_name} card annual fees or branch service fees?"
            )
        else:
            msg = (
                "عفواً، هل يمكنك توضيح طلبك بشكل أكثر تفصيلاً؟"
                if lang == Language.ARABIC
                else "Could you please clarify your request?"
            )

        logger.info("ClarificationEngine: Generated clarification response for reason %s", reason)
        return AIResponse(
            action="ROUTE",
            department=None,
            reason=f"CLARIFICATION_{reason}",
            message=msg,
            language=lang.value,
        )
