import asyncio
import numpy as np
import torch

from app.config.settings import settings
from input.models.audio_frame import AudioFrame
from input.models.vad_result import VADResult
from input.vad.base import VoiceActivityDetector
from silero_vad import load_silero_vad


_REQUIRED_SAMPLES = 512


class SileroVAD(VoiceActivityDetector):
    """
    Silero VAD implementation utilizing the ONNX backend.

    Determines if a single AudioFrame contains speech by checking the
    probability output against the specified threshold.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """
        Initialize the Silero VAD detector.

        Args:
            threshold: Probability threshold above which a frame is classified as speech.
        """
        self.threshold = threshold
        self._sample_rate = settings.PIPELINE_SAMPLE_RATE

        # Load the ONNX model wrapper with onnx=True to use the ONNX runtime.
        # This is a synchronous one-time CPU-bound load (~330ms).
        self._model = load_silero_vad(onnx=True)

    async def detect(self, frame: AudioFrame) -> VADResult:
        """
        Detect speech in a single AudioFrame.

        Args:
            frame: Canonical AudioFrame containing float32 mono PCM samples at 16kHz.

        Returns:
            VADResult containing is_speech, confidence, and timestamp.
        """
        if frame.sample_rate != self._sample_rate:
            raise ValueError(
                f"SileroVAD expects sample rate of {self._sample_rate}Hz, but received {frame.sample_rate}Hz. "
                "Ensure AudioFrameAdapter is used to normalize the audio stream."
            )

        if frame.samples.shape[0] != _REQUIRED_SAMPLES:
            raise ValueError(
                f"SileroVAD expects exactly {_REQUIRED_SAMPLES} samples per frame (32ms at 16kHz), "
                f"but received {frame.samples.shape[0]} samples. Adjust frame duration at the source layer."
            )

        # Offload the inference execution to the event loop's default thread pool
        # to avoid blocking the event loop on CPU-bound operations.
        loop = asyncio.get_running_loop()
        confidence = await loop.run_in_executor(None, self._run_inference, frame.samples)
        is_speech = confidence >= self.threshold

        return VADResult(
            is_speech=is_speech,
            confidence=confidence,
            timestamp=frame.timestamp,
        )

    def _run_inference(self, samples: np.ndarray) -> float:
        # Wrap numpy samples in a CPU PyTorch tensor.
        # OnnxWrapper requires a PyTorch tensor input and updates recurrent states in-place.
        tensor_samples = torch.from_numpy(samples)
        with torch.no_grad():
            prob_tensor = self._model(tensor_samples, self._sample_rate)
            return float(prob_tensor.item())

    def reset(self) -> None:
        """
        Reset the recurrent state of the Silero ONNX model.
        Should be called at the start of a new audio/conversation session.
        """
        self._model.reset_states()
