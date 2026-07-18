import logging
import httpx
import struct
from typing import Optional

from app.config.settings import settings
from app.tts.base import SpeechSynthesizer

logger = logging.getLogger(__name__)


class SilmaTTSError(RuntimeError):
    """
    Raised when a Silma TTS operation fails.
    """
    pass


class SilmaTTS(SpeechSynthesizer):
    """
    SILMA Text-to-Speech integration.
    Communicates with SILMA's TTS API to synthesize text to speech.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_id: Optional[str] = None,
        voice_id: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        """
        Initialize SilmaTTS. Configuration is fetched from settings by default.
        """
        self.api_key = api_key if api_key is not None else settings.SILMA_API_KEY
        self.base_url = base_url if base_url is not None else settings.SILMA_BASE_URL
        self.model_id = model_id if model_id is not None else settings.SILMA_MODEL_ID
        self.voice_id = voice_id if voice_id is not None else settings.SILMA_VOICE_ID
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

        if not all([self.api_key, self.base_url, self.model_id, self.voice_id]):
            raise SilmaTTSError(
                "Silma TTS configuration is incomplete. "
                "Ensure SILMA_API_KEY, SILMA_BASE_URL, SILMA_MODEL_ID, and SILMA_VOICE_ID are set."
            )

    def _pcm_to_wav(self, pcm_data: bytes, sample_rate: int = 24000) -> bytes:
        """
        Convert raw IEEE float32 PCM data to signed int16 PCM data,
        and wrap it into a standard WAV container (format 1).
        """
        import numpy as np

        # Convert float32 bytes to numpy array
        samples = np.frombuffer(pcm_data, dtype=np.float32)
        # Clip to safe range [-1.0, 1.0] to prevent overflow distortion
        samples = np.clip(samples, -1.0, 1.0)
        # Scale to 16-bit signed integer range
        int16_samples = (samples * 32767.0).astype(np.int16)
        int16_pcm = int16_samples.tobytes()

        audio_format = 1  # Integer PCM
        num_channels = 1
        bits_per_sample = 16
        
        block_align = num_channels * (bits_per_sample // 8)
        byte_rate = sample_rate * block_align
        
        fmt_chunk_size = 16
        data_chunk_size = len(int16_pcm)
        riff_chunk_size = 4 + (8 + fmt_chunk_size) + (8 + data_chunk_size)
        
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            riff_chunk_size,
            b"WAVE",
            b"fmt ",
            fmt_chunk_size,
            audio_format,
            num_channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            b"data",
            data_chunk_size
        )
        return header + int16_pcm

    async def synthesize(self, text: str) -> bytes:
        """
        Synthesize text to speech asynchronously, returning the audio bytes.
        """
        if not text or not text.strip():
            logger.error("Empty text provided for synthesis.")
            raise ValueError("Text cannot be empty.")

        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)

        url = f"{self.base_url.rstrip('/')}/stream"
        headers = {
            "apiKey": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "model_id": self.model_id,
            "voice_id": self.voice_id,
            "text": text,
        }

        logger.info("Requesting TTS synthesis from Silma: voice_id=%s, text_length=%d", self.voice_id, len(text))
        
        try:
            response = await self._client.post(
                url,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error("Silma TTS API failed with status %d: %s", exc.response.status_code, exc.response.text)
            raise SilmaTTSError(
                f"Silma TTS API error (status {exc.response.status_code}): {exc.response.text}"
            ) from exc
        except httpx.TimeoutException as exc:
            logger.error("Silma TTS synthesis request timed out: %s", exc)
            raise SilmaTTSError(f"Silma TTS synthesis request timed out after {self.timeout} seconds") from exc
        except httpx.RequestError as exc:
            logger.error("Silma TTS synthesis network error: %s", exc)
            raise SilmaTTSError(f"Silma TTS synthesis network error: {exc}") from exc

        # Debug logging as required
        logger.info("Silma headers: %s", response.headers)
        logger.info("Silma content-type: %s", response.headers.get("content-type"))
        logger.info("Silma first bytes: %s", response.content[:32])
        logger.info("Silma content length: %d", len(response.content))

        # Parse sample rate from headers (default to 24000 Hz)
        sample_rate = 24000
        sample_rate_str = response.headers.get("x-audio-sample-rate")
        if sample_rate_str and sample_rate_str.isdigit():
            sample_rate = int(sample_rate_str)

        # Wrap raw PCM into WAV format
        wav_content = self._pcm_to_wav(response.content, sample_rate)
        logger.info("Silma TTS synthesis succeeded: wrapped raw PCM to WAV. Received %d bytes -> Output %d bytes", len(response.content), len(wav_content))

        # Save the final WAV bytes to response.wav
        try:
            with open("response.wav", "wb") as f:
                f.write(wav_content)
            logger.info("Saved final WAV response to response.wav")
        except Exception as exc:
            logger.error("Failed to save response.wav: %s", exc)

        return wav_content

    async def close(self) -> None:
        """
        Close the HTTP client session pool.
        """
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.info("SilmaTTS connection pool closed.")
