"""
Voice Dialogue Handler

Evaluates STT confidence scores and acoustic voice triggers.
Determines VoiceTurnAction (PROCEED, PROCEED_UNCERTAIN, CONFIRM_INTENT, DISCARD_UTTERANCE).
"""

import logging
from app.conversation.models import ConversationState, VoiceTurnAction

logger = logging.getLogger(__name__)


class VoiceDialogueHandler:
    """
    Evaluates STT confidence thresholds and real-time voice acoustic signals.
    """

    def __init__(
        self,
        high_threshold: float = 0.90,
        medium_threshold: float = 0.60,
        low_threshold: float = 0.40,
    ) -> None:
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.low_threshold = low_threshold

    def evaluate_voice_turn(self, stt_confidence: float, state: ConversationState) -> VoiceTurnAction:
        """
        Evaluates STT confidence against thresholds.
        """
        if stt_confidence >= self.high_threshold:
            logger.debug("VoiceDialogueHandler: High STT confidence (%.2f) -> PROCEED", stt_confidence)
            return VoiceTurnAction.PROCEED
        elif stt_confidence >= self.medium_threshold:
            logger.info("VoiceDialogueHandler: Medium STT confidence (%.2f) -> PROCEED_UNCERTAIN", stt_confidence)
            return VoiceTurnAction.PROCEED_UNCERTAIN
        elif stt_confidence >= self.low_threshold:
            logger.warning("VoiceDialogueHandler: Low STT confidence (%.2f) -> CONFIRM_INTENT", stt_confidence)
            return VoiceTurnAction.CONFIRM_INTENT
        else:
            logger.warning("VoiceDialogueHandler: Critical low STT confidence (%.2f) -> DISCARD_UTTERANCE", stt_confidence)
            return VoiceTurnAction.DISCARD_UTTERANCE
