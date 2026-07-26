import asyncio
import unittest
from unittest.mock import AsyncMock

from input.stt.language_normalizer import normalize_stt_language
from app.query_optimization.llm_enhancer import LLMQueryEnhancer
from app.speech_formatting import (
    SpeechResponseFormatter,
    MarkdownNormalizer,
    AbbreviationAndCurrencyNormalizer,
    PunctuationAndWhitespaceNormalizer,
    BaseTextNormalizer,
)


class TestSTTLanguageNormalization(unittest.TestCase):
    def test_stt_language_normalization_english_text_with_arabic_metadata(self):
        text = "What is the fees of the gold visa?"
        raw_lang = "Arabic"
        result = normalize_stt_language(text, raw_lang)
        self.assertEqual(result, "en")

    def test_stt_language_normalization_english_gold_visa_query(self):
        text = "What are the fees of the gold visa?"
        raw_lang = "Arabic"
        result = normalize_stt_language(text, raw_lang)
        self.assertEqual(result, "en")

    def test_stt_language_normalization_arabic_text_with_english_metadata(self):
        text = "ما هي رسوم البطاقة الذهبية؟"
        raw_lang = "English"
        result = normalize_stt_language(text, raw_lang)
        self.assertEqual(result, "ar")

    def test_stt_language_normalization_full_strings(self):
        self.assertEqual(normalize_stt_language("Hello world", "English"), "en")
        self.assertEqual(normalize_stt_language("مرحبا", "arabic"), "ar")
        self.assertEqual(normalize_stt_language("Hello", "en"), "en")
        self.assertEqual(normalize_stt_language("مرحبا", "ar"), "ar")


class TestQueryEnhancerEntityPreservation(unittest.TestCase):
    def test_query_enhancer_preserves_unknown_entities(self):
        async def run():
            mock_llm_provider = AsyncMock()
            mock_llm_provider.generate.return_value = "مقر مدرسة بلاتينام"
            enhancer = LLMQueryEnhancer(llm_provider=mock_llm_provider)

            query = "نعم، أخبرني بأكثر عن مقر مدرسة بلاتينام"
            result = await enhancer.enhance(query)
            
            prompt_arg = mock_llm_provider.generate.call_args[0][0]
            self.assertIn("CRITICAL RULES", prompt_arg)
            self.assertIn("NEVER invent or assume specific products", prompt_arg)
            self.assertNotIn("Platinum Credit Card benefits", result)
            self.assertIn("بلاتينام", result)

        asyncio.run(run())

    def test_query_enhancer_no_unrequested_category_words(self):
        async def run():
            mock_llm_provider = AsyncMock()
            mock_llm_provider.generate.return_value = "gold visa fees"
            enhancer = LLMQueryEnhancer(llm_provider=mock_llm_provider)

            query = "What is the fees of the gold visa?"
            result = await enhancer.enhance(query)

            prompt_arg = mock_llm_provider.generate.call_args[0][0]
            self.assertIn("Credit Card", prompt_arg)
            self.assertNotIn("Credit Card", result)
            self.assertEqual(result, "gold visa fees")

        asyncio.run(run())


class TestSpeechResponseFormatter(unittest.TestCase):
    def setUp(self):
        self.formatter = SpeechResponseFormatter()

    def test_english_currency_expansion_egp_250(self):
        text = "The issuance fee is EGP 250"
        expected = "The issuance fee is 250 Egyptian Pounds"
        result = self.formatter.format(text, language="en")
        self.assertEqual(result, expected)

    def test_english_currency_expansion_500(self):
        text = "Issuance fee: EGP 500"
        expected = "Issuance fee: 500 Egyptian Pounds"
        result = self.formatter.format(text, language="en")
        self.assertEqual(result, expected)

    def test_english_le_expansion(self):
        text = "The renewal fee is 250 LE."
        expected = "The renewal fee is 250 Egyptian Pounds."
        result = self.formatter.format(text, language="en")
        self.assertEqual(result, expected)

    def test_arabic_currency_expansion(self):
        text = "رسوم الإصدار: EGP 500"
        expected = "رسوم الإصدار: 500 جنيه مصري"
        result = self.formatter.format(text, language="ar")
        self.assertEqual(result, expected)

    def test_arabic_currency_expansion_fallback_detection(self):
        text = "رسوم الإصدار: EGP 500"
        expected = "رسوم الإصدار: 500 جنيه مصري"
        result = self.formatter.format(text, language=None)
        self.assertEqual(result, expected)

    def test_language_priority_resolution(self):
        text = "رسوم الإصدار: EGP 500"
        result = self.formatter.format(
            text,
            language="ar",
            transcription_language="en"
        )
        self.assertEqual(result, "رسوم الإصدار: 500 جنيه مصري")

    def test_transcription_language_fallback(self):
        text = "Issuance fee: EGP 500"
        result = self.formatter.format(
            text,
            language=None,
            transcription_language="en"
        )
        self.assertEqual(result, "Issuance fee: 500 Egyptian Pounds")

    def test_markdown_and_bullet_cleanup(self):
        text = "**Fees**:\n• Issuance: EGP 500\n• Renewal: EGP 500"
        result = self.formatter.format(text, language="en")
        self.assertNotIn("**", result)
        self.assertNotIn("•", result)
        self.assertIn("500 Egyptian Pounds", result)

    def test_empty_string_handling(self):
        result = self.formatter.format("")
        self.assertEqual(result, "")

    def test_extensibility_with_custom_normalizer(self):
        class PrefixNormalizer(BaseTextNormalizer):
            def normalize(self, text, language=None):
                return "Spoken: " + text

        custom_formatter = SpeechResponseFormatter(normalizers=[PrefixNormalizer(), MarkdownNormalizer()])
        result = custom_formatter.format("Hello")
        self.assertEqual(result, "Spoken: Hello")


if __name__ == "__main__":
    unittest.main()
