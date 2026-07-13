from abc import ABC, abstractmethod

from input.models.audio_frame import AudioFrame
from input.models.vad_result import VADResult


class VoiceActivityDetector(ABC):
    """
    Abstract interface for Voice Activity Detection (VAD).

    Determines if a single AudioFrame contains speech.
    This abstraction allows swapping Silero VAD with WebRTC VAD or any other
    implementation without modifying the downstream pipeline components.
    """

    @abstractmethod
    async def detect(self, frame: AudioFrame) -> VADResult:
        """
        Analyze a single AudioFrame and return whether it contains speech.

        Args:
            frame: The canonical AudioFrame to evaluate.

        Returns:
            A VADResult containing is_speech, confidence, and timestamp.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the detector's internal recurrent/temporal state for a new audio session.
        """
        ...
