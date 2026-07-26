import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock

from input.models.speech_segment import SpeechSegment
from input.stt.groq_whisper import GroqWhisperSTT


@pytest.mark.asyncio
async def test_groq_whisper_english_audio_transcription():
    """Verify that GroqWhisperSTT passes prompt parameter to preserve English audio language as 'en'."""
    stt = GroqWhisperSTT(api_key="mock_key", model_name="whisper-large-v3")
    await stt.initialize()

    # Mock Groq client response for English audio
    mock_response = MagicMock()
    mock_response.text = "What are the fees for the gold visa?"
    mock_response.language = "english"

    stt._client.audio.transcriptions.create = AsyncMock(return_value=mock_response)

    # Synthetic 1-second 16kHz audio segment
    samples = np.zeros(16000, dtype=np.float32)
    segment = SpeechSegment(samples=samples, sample_rate=16000, start_timestamp=0.0, end_timestamp=1.0)

    transcription = await stt.transcribe(segment)

    # 1. Assert transcription text and language
    assert transcription.text == "What are the fees for the gold visa?"
    assert transcription.language == "en"

    # 2. Assert create call arguments (prompt parameter included, no forced Arabic language)
    create_call = stt._client.audio.transcriptions.create.call_args
    assert create_call is not None
    kwargs = create_call.kwargs
    assert kwargs.get("prompt") == "Transcribe exactly in the spoken language. Do not translate."
    assert kwargs.get("response_format") == "verbose_json"
    assert "language" not in kwargs or kwargs["language"] is None


@pytest.mark.asyncio
async def test_groq_whisper_arabic_audio_transcription():
    """Verify that GroqWhisperSTT transcribes Arabic audio accurately with language='ar'."""
    stt = GroqWhisperSTT(api_key="mock_key", model_name="whisper-large-v3")
    await stt.initialize()

    # Mock Groq client response for Arabic audio
    mock_response = MagicMock()
    mock_response.text = "ايه رسوم الفيزا الجولد؟"
    mock_response.language = "arabic"

    stt._client.audio.transcriptions.create = AsyncMock(return_value=mock_response)

    # Synthetic 1-second 16kHz audio segment
    samples = np.zeros(16000, dtype=np.float32)
    segment = SpeechSegment(samples=samples, sample_rate=16000, start_timestamp=0.0, end_timestamp=1.0)

    transcription = await stt.transcribe(segment)

    # 1. Assert transcription text and language
    assert transcription.text == "ايه رسوم الفيزا الجولد؟"
    assert transcription.language == "ar"

    # 2. Assert create call arguments
    create_call = stt._client.audio.transcriptions.create.call_args
    assert create_call is not None
    kwargs = create_call.kwargs
    assert kwargs.get("prompt") == "Transcribe exactly in the spoken language. Do not translate."
    assert kwargs.get("response_format") == "verbose_json"
