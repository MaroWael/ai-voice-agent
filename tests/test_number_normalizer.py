import unittest
from unittest.mock import AsyncMock, MagicMock

from app.speech_formatting.formatter import SpeechResponseFormatter
from app.speech_formatting.number_normalizer import NumberSpeechNormalizer
from input.models.speech_segment import SpeechSegment
from input.models.transcription import Transcription
from llm.models import AIResponse
from orchestration.orchestrator import Orchestrator, OrchestratorResult


class TestNumberSpeechNormalizer(unittest.TestCase):

    def setUp(self):
        self.normalizer = NumberSpeechNormalizer()

    def test_english_number_normalization(self):
        text = "The fee is 250 Egyptian Pounds."
        result = self.normalizer.normalize(text, language="en")
        self.assertIn("two hundred", result)
        self.assertIn("fifty Egyptian Pounds", result)
        self.assertNotIn(" 250 ", result)

    def test_english_500_normalization(self):
        text = "The fee is 500 US Dollars."
        result = self.normalizer.normalize(text, language="en")
        self.assertIn("five hundred US Dollars", result)

    def test_arabic_number_normalization(self):
        text = "رسوم الإصدار 250 جنيه مصري"
        result = self.normalizer.normalize(text, language="ar")
        self.assertIn("جنيه مصري", result)
        self.assertNotIn("250", result)
        self.assertTrue("خمسون" in result or "مئتان" in result or "مائتان" in result)

    def test_multiple_numbers_conversion(self):
        text = "Issuance fee is 250 Egyptian Pounds and renewal fee is 500 Egyptian Pounds."
        result = self.normalizer.normalize(text, language="en")
        self.assertIn("two hundred", result)
        self.assertIn("five hundred", result)
        self.assertNotIn("250", result)
        self.assertNotIn("500", result)

    def test_preserve_card_and_id_numbers(self):
        text = "Card number 123456"
        result = self.normalizer.normalize(text, language="en")
        self.assertEqual(result, "Card number 123456")

        text_tx = "Transaction ID 12345"
        result_tx = self.normalizer.normalize(text_tx, language="en")
        self.assertEqual(result_tx, "Transaction ID 12345")

    def test_preserve_years(self):
        text = "Visa launched in 2025"
        result = self.normalizer.normalize(text, language="en")
        self.assertEqual(result, "Visa launched in 2025")


class TestPipelineNumberNormalizerIntegration(unittest.TestCase):

    def test_ai_response_message_remains_unchanged_while_formatter_normalizes_numbers(self):
        async def run_test():
            raw_llm_message = "The issuance fee is EGP 250 and renewal fee is EGP 500."

            mock_adapter = MagicMock()
            mock_vad = AsyncMock()
            mock_buffer = AsyncMock()
            mock_recognizer = AsyncMock()
            mock_llm = AsyncMock()
            mock_tts = AsyncMock()

            orchestrator = Orchestrator(
                audio_source=None,
                adapter=mock_adapter,
                vad=mock_vad,
                buffer=mock_buffer,
                recognizer=mock_recognizer,
                llm=mock_llm,
                tts=mock_tts,
            )

            segment = SpeechSegment(samples=MagicMock(), sample_rate=16000, start_timestamp=0.0, end_timestamp=1.0)
            transcription = Transcription(text="What are the card fees?", language="en", start_timestamp=0.0, end_timestamp=1.0)
            mock_recognizer.transcribe.return_value = transcription

            ai_response = AIResponse(
                action="rag",
                department=None,
                reason="success",
                message=raw_llm_message,
                language="en",
            )
            mock_llm.generate.return_value = ai_response

            result: OrchestratorResult = await orchestrator.process_speech_segment(segment)

            # 1. Assert AIResponse.message remains unchanged for API/UI callers
            self.assertEqual(result.response.message, raw_llm_message)

            # 2. Format speech text at TTS boundary
            formatter = SpeechResponseFormatter()
            speech_text = formatter.format(
                result.response.message,
                language=result.response.language,
                transcription_language=result.transcription.language,
            )

            # 3. Assert currency was expanded AND number was normalized into words
            self.assertIn("two hundred", speech_text)
            self.assertIn("five hundred", speech_text)
            self.assertIn("Egyptian Pounds", speech_text)
            self.assertNotIn("EGP", speech_text)
            self.assertNotIn("250", speech_text)
            self.assertNotIn("500", speech_text)

        import asyncio
        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
