import logging
from dataclasses import dataclass
from input.sources.base import AudioSource
from input.adapter.audio_frame_adapter import AudioFrameAdapter
from input.vad.base import VoiceActivityDetector
from input.buffer.speech_buffer import SpeechBuffer
from input.stt.base import SpeechRecognizer
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
        audio_source: AudioSource,
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

    async def run(self) -> OrchestratorResult:
        """
        Executes the audio pipeline continuously until a final Transcription is produced,
        then invokes the language model to generate an AIResponse.
        Guarantees VAD and buffer resets on successful completion, cancellation, or error.
        """
        try:
            async for frame in self.audio_source.stream():
                adapted_frame = self.adapter.adapt(frame)
                vad_result = await self.vad.detect(adapted_frame)
                segment = await self.buffer.process(adapted_frame, vad_result)

                if segment is not None:
                    import time
                    start_stt = time.perf_counter()
                    transcription = await self.recognizer.transcribe(segment)
                    stt_elapsed = time.perf_counter() - start_stt
                    logger.info("STT completed in %.2f seconds", stt_elapsed)

                    start_llm = time.perf_counter()
                    response = await self.llm.generate(transcription)
                    llm_elapsed = time.perf_counter() - start_llm
                    logger.info("LLM completed in %.2f seconds", llm_elapsed)

                    return OrchestratorResult(transcription=transcription, response=response)

            raise RuntimeError("Audio stream ended before a speech segment could be transcribed.")
        finally:
            self.vad.reset()
            self.buffer.reset()
