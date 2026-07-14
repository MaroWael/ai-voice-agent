import logging
from input.sources.base import AudioSource
from input.adapter.audio_frame_adapter import AudioFrameAdapter
from input.vad.base import VoiceActivityDetector
from input.buffer.speech_buffer import SpeechBuffer
from input.stt.base import SpeechRecognizer
from input.models.transcription import Transcription

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Coordinates the audio processing pipeline from input frame ingestion
    to final transcript generation.
    """

    def __init__(
        self,
        audio_source: AudioSource,
        adapter: AudioFrameAdapter,
        vad: VoiceActivityDetector,
        buffer: SpeechBuffer,
        recognizer: SpeechRecognizer,
    ) -> None:
        self.audio_source = audio_source
        self.adapter = adapter
        self.vad = vad
        self.buffer = buffer
        self.recognizer = recognizer

    async def run(self) -> Transcription:
        """
        Executes the audio pipeline continuously until a final Transcription is produced.
        Guarantees VAD and buffer resets on successful completion, cancellation, or error.
        """
        try:
            async for frame in self.audio_source.stream():
                adapted_frame = self.adapter.adapt(frame)
                vad_result = await self.vad.detect(adapted_frame)
                segment = await self.buffer.process(adapted_frame, vad_result)

                if segment is not None:
                    return await self.recognizer.transcribe(segment)

            raise RuntimeError("Audio stream ended before a speech segment could be transcribed.")
        finally:
            self.vad.reset()
            self.buffer.reset()
