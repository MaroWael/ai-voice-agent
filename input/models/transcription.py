from dataclasses import dataclass


@dataclass(frozen=True)
class Transcription:
    """
    Spoken language converted into text.

    This acts as the data transfer contract between the STT layer
    and the downstream orchestrator.
    """

    # The transcribed text
    text: str

    # The language of the transcription (e.g. "en", "ar")
    language: str | None

    # Monotonic timestamp of the start of the segment
    start_timestamp: float

    # Monotonic timestamp of the end of the segment
    end_timestamp: float
