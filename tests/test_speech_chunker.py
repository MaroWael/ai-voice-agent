import io
import unittest
import wave
from unittest.mock import AsyncMock, MagicMock

from app.speech_formatting.chunker import SpeechChunker
from app.tts.audio_utils import merge_audio_chunks
from input.models.speech_segment import SpeechSegment
from input.models.transcription import Transcription
from llm.models import AIResponse
from orchestration.orchestrator import Orchestrator, OrchestratorResult


def create_dummy_wav(duration_frames: int = 1600, sample_rate: int = 16000) -> bytes:
    """Helper to generate a valid 16-bit mono PCM WAV bytes object."""
    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * duration_frames)
    return out.getvalue()


class TestSpeechChunker(unittest.TestCase):

    def test_short_text_single_chunk(self):
        chunker = SpeechChunker(max_chunk_length=180)
        chunks = chunker.split("Hello")
        self.assertEqual(chunks, ["Hello"])

    def test_long_english_response_multiple_chunks(self):
        chunker = SpeechChunker(max_chunk_length=80)
        text = (
            "The Gold Credit Card issuance fee is 250 Egyptian Pounds and the renewal fee is 250 Egyptian Pounds. "
            "The supplementary card fee is 150 Egyptian Pounds."
        )
        chunks = chunker.split(text)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c), 80, f"Chunk length {len(c)} exceeds max_chunk_length 80: {c}")

    def test_arabic_response_splitting(self):
        chunker = SpeechChunker(max_chunk_length=40)
        text = "رسوم الإصدار هي 250 جنيه مصري، ورسوم التجديد هي 250 جنيه مصري."
        chunks = chunker.split(text)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c), 40, f"Chunk length {len(c)} exceeds max_chunk_length 40: {c}")

    def test_number_and_currency_token_preservation(self):
        chunker = SpeechChunker(max_chunk_length=25)
        text = "Fee is 250 Egyptian Pounds."
        chunks = chunker.split(text)
        for c in chunks:
            if "250" in c:
                self.assertIn("Egyptian Pounds", c, f"Number '250' was separated from currency unit in chunk: {c}")

    def test_empty_string_handling(self):
        chunker = SpeechChunker(max_chunk_length=180)
        self.assertEqual(chunker.split(""), [])
        self.assertEqual(chunker.split("   "), [])


class TestAudioMergeUtility(unittest.TestCase):

    def test_merge_empty_chunks(self):
        self.assertEqual(merge_audio_chunks([]), b"")
        self.assertEqual(merge_audio_chunks([b""]), b"")

    def test_merge_single_chunk(self):
        wav = create_dummy_wav(800)
        merged = merge_audio_chunks([wav])
        self.assertEqual(merged, wav)

    def test_merge_multiple_wav_chunks(self):
        wav1 = create_dummy_wav(1600)  # 0.1s
        wav2 = create_dummy_wav(1600)  # 0.1s
        merged = merge_audio_chunks([wav1, wav2])

        self.assertTrue(merged.startswith(b"RIFF"))
        self.assertGreater(len(merged), len(wav1))

        # Inspect frames in merged WAV
        with wave.open(io.BytesIO(merged), "rb") as wf:
            self.assertEqual(wf.getnframes(), 3200)


class TestPipelineTTSChunkerIntegration(unittest.TestCase):

    def test_ai_response_message_remains_unchanged_while_tts_receives_chunked_text(self):
        async def run_test():
            raw_llm_message = (
                "The issuance fee for the Gold Credit Card is 250 Egyptian Pounds. "
                "The renewal fee is 250 Egyptian Pounds. "
                "The supplementary card issuance fee is 150 Egyptian Pounds."
            )

            mock_adapter = MagicMock()
            mock_vad = AsyncMock()
            mock_buffer = AsyncMock()
            mock_recognizer = AsyncMock()
            mock_llm = AsyncMock()

            # Mock TTS synthesizer
            synthesized_chunks = []

            class MockTTS:
                async def synthesize(self, text: str) -> bytes:
                    synthesized_chunks.append(text)
                    return create_dummy_wav(800)

            tts = MockTTS()

            orchestrator = Orchestrator(
                audio_source=None,
                adapter=mock_adapter,
                vad=mock_vad,
                buffer=mock_buffer,
                recognizer=mock_recognizer,
                llm=mock_llm,
                tts=tts,
            )

            segment = SpeechSegment(samples=MagicMock(), sample_rate=16000, start_timestamp=0.0, end_timestamp=1.0)
            transcription = Transcription(text="What are the gold card fees?", language="en", start_timestamp=0.0, end_timestamp=1.0)
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

            # 1. Assert AIResponse.message remains unchanged for API/UI
            self.assertEqual(result.response.message, raw_llm_message)

            # 2. Simulate TTS helper synthesis with SpeechChunker (max 100 chars)
            chunker = SpeechChunker(max_chunk_length=100)
            formatted_speech_text = result.response.message
            split_chunks = chunker.split(formatted_speech_text)

            audio_chunks = []
            for c in split_chunks:
                a = await tts.synthesize(c)
                audio_chunks.append(a)

            merged_audio = merge_audio_chunks(audio_chunks)

            # Assertions
            self.assertGreater(len(synthesized_chunks), 1)
            for chunk_sent in synthesized_chunks:
                self.assertLessEqual(len(chunk_sent), 100)
            self.assertTrue(merged_audio.startswith(b"RIFF"))

        import asyncio
        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
