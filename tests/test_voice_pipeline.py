import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from input.models.speech_segment import SpeechSegment
from input.models.transcription import Transcription
from llm.models import AIResponse
from orchestration.orchestrator import Orchestrator, OrchestratorResult
from app.speech_formatting import SpeechResponseFormatter
from app.tts.base import SpeechSynthesizer


class DummyTTS(SpeechSynthesizer):
    def __init__(self):
        self.synthesized_texts = []

    async def synthesize(self, text: str) -> bytes:
        self.synthesized_texts.append(text)
        return b"RIFF_HEADER_MOCK_AUDIO_BYTES"

    async def close(self) -> None:
        pass


class TestVoicePipelineIntegration(unittest.TestCase):
    def setUp(self):
        self.mock_adapter = MagicMock()
        self.mock_vad = AsyncMock()
        self.mock_buffer = AsyncMock()
        self.mock_recognizer = AsyncMock()
        self.mock_llm = AsyncMock()
        self.tts = DummyTTS()

        self.orchestrator = Orchestrator(
            audio_source=None,
            adapter=self.mock_adapter,
            vad=self.mock_vad,
            buffer=self.mock_buffer,
            recognizer=self.mock_recognizer,
            llm=self.mock_llm,
            tts=self.tts,
        )

    def test_orchestrator_uncoupled_from_speech_formatter(self):
        """Verify Orchestrator does not have speech formatter attributes or logic injected."""
        self.assertFalse(hasattr(self.orchestrator, "formatter"))
        self.assertFalse(hasattr(self.orchestrator, "speech_formatter"))

    def test_pipeline_tts_boundary_formatting(self):
        async def run_test():
            # Setup mock STT transcription
            segment = SpeechSegment(samples=MagicMock(), sample_rate=16000, start_timestamp=0.0, end_timestamp=1.0)
            transcription = Transcription(text="What are the issuance fees?", language="en", start_timestamp=0.0, end_timestamp=1.0)
            self.mock_recognizer.transcribe.return_value = transcription

            # Setup mock LLM raw response (reading optimized)
            raw_llm_message = "The fees are:\n• Issuance: EGP 500\n• Renewal: EGP 500"
            ai_response = AIResponse(
                action="rag",
                department=None,
                reason="success",
                message=raw_llm_message,
                language="en",
            )
            self.mock_llm.generate.return_value = ai_response

            # 1. Execute Orchestrator
            result: OrchestratorResult = await self.orchestrator.process_speech_segment(segment)

            # 2. Verify AIResponse.message remains untouched for UI/API
            self.assertEqual(result.response.message, raw_llm_message)
            self.assertIn("• Issuance: EGP 500", result.response.message)

            # 3. Format text at final TTS boundary
            formatter = SpeechResponseFormatter()
            speech_text = formatter.format(
                result.response.message,
                language=result.response.language,
                transcription_language=result.transcription.language,
            )

            # 4. Pass formatted text to TTS synthesize
            audio_bytes = await self.tts.synthesize(speech_text)

            # 5. Assertions
            # The UI/API object still holds raw_llm_message
            self.assertEqual(result.response.message, raw_llm_message)

            # Only TTS input received the formatted speech-ready text
            self.assertEqual(len(self.tts.synthesized_texts), 1)
            sent_to_tts = self.tts.synthesized_texts[0]
            self.assertNotIn("•", sent_to_tts)
            self.assertNotIn("EGP 500", sent_to_tts)
            self.assertIn("500 Egyptian Pounds", sent_to_tts)

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
