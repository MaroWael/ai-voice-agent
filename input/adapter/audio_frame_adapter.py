from math import gcd

import numpy as np
from scipy.signal import resample_poly

from app.config.settings import settings
from input.models.audio_frame import AudioFrame


def _ensure_float32(samples: np.ndarray) -> np.ndarray:
    """
    Return samples cast to float32.

    This is a no-op when the array is already float32, so the check is
    free on the hot path (microphone always produces float32).
    """
    if samples.dtype == np.float32:
        return samples
    return samples.astype(np.float32)


def _to_mono(samples: np.ndarray) -> np.ndarray:
    """
    Average all channels into a single mono channel.

    Accepts:
    - shape (frames,)          — already mono, returned as-is
    - shape (frames, channels) — channels are averaged across axis=1
    """
    if samples.ndim == 1:
        return samples
    # Average across the channel axis to preserve perceived loudness.
    return samples.mean(axis=1)


def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """
    Resample a 1-D float32 array from source_rate to target_rate using a
    polyphase anti-aliasing filter (scipy.signal.resample_poly).

    resample_poly(x, up, down) upsamples by `up` then downsamples by `down`,
    applying a Kaiser-windowed low-pass filter to prevent aliasing. The
    up/down factors are derived from the exact rational ratio
    target_rate / source_rate, reduced by their GCD.

    This is a no-op when the rates are already equal.
    """
    if source_rate == target_rate:
        return samples

    divisor = gcd(source_rate, target_rate)
    up = target_rate // divisor
    down = source_rate // divisor

    resampled = resample_poly(samples, up, down)
    # resample_poly returns float64; cast back to float32 to stay consistent.
    return resampled.astype(np.float32)


class AudioFrameAdapter:
    """
    Normalises any AudioFrame into the canonical pipeline format:

        sample_rate : settings.PIPELINE_SAMPLE_RATE  (default 16 000 Hz)
        channels    : settings.PIPELINE_CHANNELS      (default 1 — mono)
        dtype       : float32
        timestamp   : preserved exactly from the source frame

    The adapter is intentionally synchronous because it performs pure CPU
    computation. Callers that run inside an async pipeline should offload
    via asyncio.get_event_loop().run_in_executor() at the call site.

    Operations are applied in this order:
        1. Ensure float32 dtype
        2. Convert to mono
        3. Resample to target sample rate

    The input frame is never mutated; a new AudioFrame is always returned.
    """

    def __init__(self) -> None:
        # Read pipeline constants once at construction.
        # Avoids repeated attribute lookups on the settings singleton per frame.
        self._target_sample_rate: int = settings.PIPELINE_SAMPLE_RATE
        self._target_channels: int = settings.PIPELINE_CHANNELS

    def adapt(self, frame: AudioFrame) -> AudioFrame:
        """
        Accept a raw AudioFrame from any AudioSource and return a new
        AudioFrame in the canonical pipeline format.

        Args:
            frame: Raw audio as captured by the source layer.

        Returns:
            A new AudioFrame with normalised sample rate, channels, and dtype.
            The original frame is unchanged.
        """
        samples = _ensure_float32(frame.samples)
        samples = _to_mono(samples)
        samples = _resample(samples, frame.sample_rate, self._target_sample_rate)

        return AudioFrame(
            samples=samples,
            sample_rate=self._target_sample_rate,
            channels=self._target_channels,
            timestamp=frame.timestamp,
        )
