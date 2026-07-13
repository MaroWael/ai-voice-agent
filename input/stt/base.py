from abc import ABC, abstractmethod

from input.models.speech_segment import SpeechSegment
from input.models.transcription import Transcription


class SpeechRecognizer(ABC):
    """
    Abstract interface for converting speech segments to transcripts.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """
        Load the recognition model or client. Called once during startup.
        """
        pass

    @abstractmethod
    async def transcribe(self, segment: SpeechSegment) -> Transcription:
        """
        Transcribe a SpeechSegment asynchronously.
        """
        pass
