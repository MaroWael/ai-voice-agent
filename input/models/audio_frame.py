from dataclasses import dataclass

import numpy as np


@dataclass
class AudioFrame:
    """
    A single chunk of raw audio captured from a source.

    This is the only data contract between the Source layer and all downstream
    layers (Adapter, Preprocessing, VAD). It carries no processing logic —
    it is a pure value object.
    """

    # Raw PCM samples as float32. Shape: (frames, channels) or (frames,) for mono.
    samples: np.ndarray

    # Native sample rate of the source, in Hz (e.g. 44100, 48000).
    # The Adapter layer is responsible for normalizing this to the pipeline rate.
    sample_rate: int

    # Number of audio channels (1 = mono, 2 = stereo).
    channels: int

    # Capture time, in seconds, from a monotonic clock.
    # Used for per-stage latency tracking.
    timestamp: float
