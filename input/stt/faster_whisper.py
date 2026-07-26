import asyncio
import logging
import time
from typing import Optional
import numpy as np
from faster_whisper import WhisperModel

from app.config.settings import settings
from input.models.speech_segment import SpeechSegment
from input.models.transcription import Transcription
from input.stt.base import SpeechRecognizer
from input.stt.exceptions import STTProviderError

logger = logging.getLogger(__name__)


class FasterWhisperSTT(SpeechRecognizer):
    """
    Faster-Whisper speech recognition runner.

    Loads the Whisper model and runs CPU-bound transcription in an executor
    to prevent event loop blocking.
    """

    def __init__(
        self,
        model_size: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        beam_size: Optional[int] = None,
    ) -> None:
        """
        Initialize FasterWhisperSTT. Configuration is fetched from settings by default.
        """
        self.model_size = model_size or settings.STT_MODEL
        self.device = device or settings.STT_DEVICE
        self.compute_type = compute_type or settings.STT_COMPUTE_TYPE
        self.beam_size = beam_size if beam_size is not None else settings.STT_BEAM_SIZE
        self._model: Optional[WhisperModel] = None

    async def initialize(self) -> None:
        """
        Load the Whisper model offloaded to a worker thread.
        """
        logger.info("STT Provider: FasterWhisperSTT (model: %s, device: %s)", self.model_size, self.device)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_model)

    def _load_model(self) -> None:
        try:
            self._model = WhisperModel(
                model_size_or_path=self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        except Exception as exc:
            logger.error("Failed to load FasterWhisper model: %s", exc)
            raise STTProviderError(f"FasterWhisper initialization failed: {exc}") from exc

    async def transcribe(self, segment: SpeechSegment) -> Transcription:
        """
        Transcribe a SpeechSegment. Offloads CPU-bound inference to a thread pool.
        """
        if self._model is None:
            raise STTProviderError("FasterWhisperSTT not initialized. Call initialize() first.")

        logger.info("Transcription started using Faster Whisper")
        start_time = time.perf_counter()

        loop = asyncio.get_running_loop()
        try:
            text, language = await loop.run_in_executor(
                None,
                self._run_transcription,
                segment.samples,
            )
        except Exception as exc:
            logger.error("FasterWhisper transcription failed: %s", exc)
            raise STTProviderError(f"FasterWhisper transcription failed: {exc}") from exc

        elapsed = time.perf_counter() - start_time
        logger.info("STT completed in %.2f seconds", elapsed)

        from input.stt.language_normalizer import normalize_stt_language
        normalized_language = normalize_stt_language(text, language)

        return Transcription(
            text=text,
            language=normalized_language,
            start_timestamp=segment.start_timestamp,
            end_timestamp=segment.end_timestamp,
        )

    def _run_transcription(self, samples: np.ndarray) -> tuple[str, Optional[str]]:
        segments, info = self._model.transcribe(
            samples,
            beam_size=self.beam_size,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        logger.info("info.language: %s", info.language)
        logger.info("info.language_probability: %s", info.language_probability)

        # Consolidate segments joining with spaces to prevent word merges
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return text, info.language

