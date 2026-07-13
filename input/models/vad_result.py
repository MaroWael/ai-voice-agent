from dataclasses import dataclass


@dataclass(frozen=True)
class VADResult:
    """
    Result of Voice Activity Detection (VAD) for a single AudioFrame.

    Carries no processing logic — it is a pure value object.
    """

    is_speech: bool
    confidence: float
    timestamp: float
