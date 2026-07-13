from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SpeechSegment:
    """
    An accumulated segment of speech samples.

    This represents a completed utterance, ready to be sent to the
    Speech-to-Text (STT) layer. It carries no logic — it is a pure value object.
    """

    # Consolidated raw PCM samples as float32 mono.
    samples: np.ndarray

    # Sample rate of the samples (typically 16000 Hz).
    sample_rate: int

    # Start time of the segment, in seconds, from a monotonic clock.
    # Refers to the timestamp of the first frame included in this segment.
    start_timestamp: float

    # End time of the segment, in seconds, from a monotonic clock.
    # Refers to the timestamp of the last active speech frame included.
    end_timestamp: float
