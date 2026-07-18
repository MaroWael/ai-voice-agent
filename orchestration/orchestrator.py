import logging
from dataclasses import dataclass
from typing import Optional
from input.sources.base import AudioSource
from input.adapter.audio_frame_adapter import AudioFrameAdapter
from input.vad.base import VoiceActivityDetector
from input.buffer.speech_buffer import SpeechBuffer
from input.stt.base import SpeechRecognizer
from input.models.audio_frame import AudioFrame
from input.models.speech_segment import SpeechSegment
from input.models.transcription import Transcription
from llm.base import LanguageModel
from llm.models import AIResponse

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrchestratorResult:
    """Holds both the transcribed user speech and the generated AI response."""
    transcription: Transcription
    response: AIResponse


class Orchestrator:
    """
    Coordinates the audio processing pipeline from input frame ingestion
    to final transcript generation and LLM response execution.
    """

    def __init__(
        self,
        audio_source: Optional[AudioSource],
        adapter: AudioFrameAdapter,
        vad: VoiceActivityDetector,
        buffer: SpeechBuffer,
        recognizer: SpeechRecognizer,
        llm: LanguageModel,
    ) -> None:
        self.audio_source = audio_source
        self.adapter = adapter
        self.vad = vad
        self.buffer = buffer
        self.recognizer = recognizer
        self.llm = llm

    async def receive_audio_frame(self, frame: AudioFrame) -> Optional[SpeechSegment]:
        """
        Ingests a single AudioFrame. Adapts it, runs VAD, and processes the speech buffer.
        If a completed SpeechSegment is emitted, resets the VAD and buffer, and returns it.
        Otherwise, returns None.
        """
        adapted_frame = self.adapter.adapt(frame)
        vad_result = await self.vad.detect(adapted_frame)
        segment = await self.buffer.process(adapted_frame, vad_result)

        if segment is not None:
            self.reset()
            return segment
        return None

    async def process_speech_segment(self, segment: SpeechSegment) -> OrchestratorResult:
        """
        Processes a completed SpeechSegment. Transcribes it and runs the LLM router.
        """
        import time
        start_stt = time.perf_counter()
        transcription = await self.recognizer.transcribe(segment)
        stt_elapsed = time.perf_counter() - start_stt
        logger.info("STT completed in %.2f seconds", stt_elapsed)

        start_llm = time.perf_counter()
        logger.info("=== STT OUTPUT ===")
        logger.info("Text: %s", transcription.text)
        logger.info("Language: %s", transcription.language)
        logger.info("==================")
        response = await self.llm.generate(transcription)
        logger.info("=== LLM OUTPUT ===")
        logger.info(response)
        logger.info("==================")
        llm_elapsed = time.perf_counter() - start_llm
        logger.info("LLM completed in %.2f seconds", llm_elapsed)

        return OrchestratorResult(transcription=transcription, response=response)

    async def process_audio_frame(self, frame: AudioFrame) -> Optional[OrchestratorResult]:
        """
        Processes a single AudioFrame.
        Adapts the frame, runs VAD, and passes it to the speech buffer.
        If a complete SpeechSegment is emitted, transcribes it and runs the LLM,
        returning an OrchestratorResult. Otherwise returns None.
        """
        segment = await self.receive_audio_frame(frame)
        if segment is not None:
            return await self.process_speech_segment(segment)
        return None

    def reset(self) -> None:
        """
        Resets both the VAD and the Speech Buffer states.
        """
        self.vad.reset()
        self.buffer.reset()

    async def run(self) -> OrchestratorResult:
        """
        Executes the audio pipeline continuously until a final Transcription is produced,
        then invokes the language model to generate an AIResponse.
        Guarantees VAD and buffer resets on successful completion, cancellation, or error.
        """
        if self.audio_source is None:
            raise RuntimeError("Cannot call run() without an audio_source.")
        try:
            async for frame in self.audio_source.stream():
                result = await self.process_audio_frame(frame)
                if result is not None:
                    return result
            raise RuntimeError("Audio stream ended before a speech segment could be transcribed.")
        finally:
            self.reset()
