import io
import wave
import time
import logging
from typing import Optional
import numpy as np
from groq import AsyncGroq

from app.config.settings import settings
from input.models.speech_segment import SpeechSegment
from input.models.transcription import Transcription
from input.stt.base import SpeechRecognizer
from input.stt.exceptions import STTProviderError

logger = logging.getLogger(__name__)


def audio_samples_to_wav_bytes(samples: np.ndarray, sample_rate: int = 16000) -> bytes:
    """
    Converts 1D float32 numpy PCM samples [-1.0, 1.0] to an in-memory 16-bit WAV file.
    """
    pcm16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


class GroqWhisperSTT(SpeechRecognizer):
    """
    Groq Whisper API speech recognition provider.

    Uses Groq's cloud-accelerated Whisper model to transcribe SpeechSegments.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or settings.GROQ_API_KEY
        self.model_name = model_name or settings.GROQ_STT_MODEL
        self._client: Optional[AsyncGroq] = None

    async def initialize(self) -> None:
        """
        Initialize the AsyncGroq client.
        """
        logger.info("STT Provider: GroqWhisperSTT (model: %s)", self.model_name)
        if not self.api_key:
            logger.warning(
                "GroqWhisperSTT initialized without GROQ_API_KEY. "
                "Ensure GROQ_API_KEY is configured in settings or environment."
            )
        self._client = AsyncGroq(api_key=self.api_key)

    async def transcribe(self, segment: SpeechSegment) -> Transcription:
        """
        Transcribes a SpeechSegment using Groq's Whisper API.
        """
        if self._client is None:
            raise STTProviderError("GroqWhisperSTT not initialized. Call initialize() first.")

        if not self.api_key:
            raise STTProviderError("GROQ_API_KEY is missing. Cannot call Groq Whisper API.")

        logger.info("Transcription started using Groq Whisper")
        start_time = time.perf_counter()

        duration_sec = len(segment.samples) / segment.sample_rate if segment.sample_rate > 0 else 0.0
        first_20 = segment.samples[:20].tolist() if len(segment.samples) >= 20 else segment.samples.tolist()
        last_20 = segment.samples[-20:].tolist() if len(segment.samples) >= 20 else segment.samples.tolist()

        try:
            wav_bytes = audio_samples_to_wav_bytes(segment.samples, segment.sample_rate)
            file_tuple = ("speech.wav", wav_bytes, "audio/wav")

            logger.info(
                "STT DIAGNOSTICS:\n"
                "  Audio Duration:    %.3f seconds\n"
                "  Sample Count:      %d samples\n"
                "  Sample Rate:       %d Hz\n"
                "  WAV Size:          %d bytes\n"
                "  First 20 Samples:  %s\n"
                "  Last 20 Samples:   %s",
                duration_sec,
                len(segment.samples),
                segment.sample_rate,
                len(wav_bytes),
                [round(s, 5) for s in first_20],
                [round(s, 5) for s in last_20],
            )

            response = await self._client.audio.transcriptions.create(
                file=file_tuple,
                model=self.model_name,
                prompt="Transcribe exactly in the spoken language. Do not translate.",
                response_format="verbose_json",
            )
        except Exception as exc:
            logger.error("Groq Whisper API transcription request failed: %s", exc)
            raise STTProviderError(f"Groq Whisper transcription failed: {exc}") from exc

        elapsed = time.perf_counter() - start_time

        # Safe mapping & language normalization
        text = getattr(response, "text", "") or ""
        text = text.strip()
        raw_language = getattr(response, "language", None)

        from input.stt.language_normalizer import normalize_stt_language
        normalized_language = normalize_stt_language(text, raw_language)

        logger.info(
            "STT RESULT:\n"
            "  Transcript:        %r\n"
            "  Detected Language: %s (Raw: %s)\n"
            "  Groq Latency:      %.3f seconds",
            text,
            normalized_language,
            raw_language,
            elapsed,
        )

        start_ts = segment.start_timestamp if segment.start_timestamp is not None else 0.0
        if segment.end_timestamp is not None and segment.end_timestamp > start_ts:
            end_ts = segment.end_timestamp
        else:
            end_ts = start_ts + duration_sec

        return Transcription(
            text=text,
            language=normalized_language,
            start_timestamp=start_ts,
            end_timestamp=end_ts,
        )
