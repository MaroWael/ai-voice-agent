import asyncio
from unittest.mock import MagicMock

import pytest

from input.adapter.audio_frame_adapter import AudioFrameAdapter
from input.buffer.speech_buffer import SpeechBuffer
from input.models.audio_frame import AudioFrame
from input.models.speech_segment import SpeechSegment
from input.models.transcription import Transcription
from input.models.vad_result import VADResult
from input.sources.base import AudioSource
from input.stt.base import SpeechRecognizer
from input.vad.base import VoiceActivityDetector
from orchestration.orchestrator import Orchestrator


class MockAudioSource(AudioSource):
    def __init__(self, frames):
        self.frames = frames
        self.stream_called = False

    async def stream(self):
        self.stream_called = True
        for frame in self.frames:
            yield frame


class MockVoiceActivityDetector(VoiceActivityDetector):
    def __init__(self, results):
        self.results = results
        self.detect_calls = []
        self.reset_called = False

    async def detect(self, frame: AudioFrame) -> VADResult:
        self.detect_calls.append(frame)
        if self.results:
            return self.results.pop(0)
        return VADResult(is_speech=False, confidence=0.0, timestamp=frame.timestamp)

    def reset(self) -> None:
        self.reset_called = True


class MockSpeechBuffer(SpeechBuffer):
    def __init__(self, segments):
        self.segments = segments
        self.process_calls = []
        self.reset_called = False

    async def process(self, frame: AudioFrame, vad: VADResult) -> SpeechSegment | None:
        self.process_calls.append((frame, vad))
        if self.segments:
            return self.segments.pop(0)
        return None

    def reset(self) -> None:
        self.reset_called = True


class MockSpeechRecognizer(SpeechRecognizer):
    def __init__(self, transcription):
        self.transcription = transcription
        self.transcribe_called_with = None
        self.initialize_called = False

    async def initialize(self) -> None:
        self.initialize_called = True

    async def transcribe(self, segment: SpeechSegment) -> Transcription:
        self.transcribe_called_with = segment
        if isinstance(self.transcription, Exception):
            raise self.transcription
        return self.transcription


@pytest.mark.asyncio
async def test_orchestrator_successful_run():
    # Setup mock data
    frame = AudioFrame(samples=None, sample_rate=16000, channels=1, timestamp=1.0)
    source = MockAudioSource([frame])

    adapter = MagicMock(spec=AudioFrameAdapter)
    adapter.adapt.return_value = frame

    vad_result = VADResult(is_speech=True, confidence=0.9, timestamp=1.0)
    vad = MockVoiceActivityDetector([vad_result])

    segment = SpeechSegment(
        samples=None, sample_rate=16000, start_timestamp=1.0, end_timestamp=2.0
    )
    buffer = MockSpeechBuffer([segment])

    transcription = Transcription(
        text="Hello", language="en", start_timestamp=1.0, end_timestamp=2.0
    )
    recognizer = MockSpeechRecognizer(transcription)

    orchestrator = Orchestrator(source, adapter, vad, buffer, recognizer)

    # Run
    result = await orchestrator.run()

    # Assertions
    assert result == transcription
    assert source.stream_called
    adapter.adapt.assert_called_once_with(frame)
    assert len(vad.detect_calls) == 1
    assert len(buffer.process_calls) == 1
    assert recognizer.transcribe_called_with == segment
    assert vad.reset_called
    assert buffer.reset_called


@pytest.mark.asyncio
async def test_orchestrator_resource_cleanup_on_exception():
    frame = AudioFrame(samples=None, sample_rate=16000, channels=1, timestamp=1.0)
    source = MockAudioSource([frame])

    adapter = MagicMock(spec=AudioFrameAdapter)
    adapter.adapt.return_value = frame

    vad_result = VADResult(is_speech=True, confidence=0.9, timestamp=1.0)
    vad = MockVoiceActivityDetector([vad_result])

    segment = SpeechSegment(
        samples=None, sample_rate=16000, start_timestamp=1.0, end_timestamp=2.0
    )
    buffer = MockSpeechBuffer([segment])

    # Recognizer raises an exception
    recognizer = MockSpeechRecognizer(RuntimeError("STT failed"))

    orchestrator = Orchestrator(source, adapter, vad, buffer, recognizer)

    # Run and assert exception propagation
    with pytest.raises(RuntimeError, match="STT failed"):
        await orchestrator.run()

    # Assert cleanups were still called
    assert vad.reset_called
    assert buffer.reset_called


@pytest.mark.asyncio
async def test_orchestrator_cancellation_cleanup():
    frame = AudioFrame(samples=None, sample_rate=16000, channels=1, timestamp=1.0)

    # Infinite stream to allow cancellation to trigger
    async def infinite_stream():
        while True:
            yield frame
            await asyncio.sleep(0.001)

    source = MagicMock(spec=AudioSource)
    source.stream = infinite_stream

    adapter = MagicMock(spec=AudioFrameAdapter)
    adapter.adapt.return_value = frame

    vad = MockVoiceActivityDetector([])
    buffer = MockSpeechBuffer([])
    recognizer = MockSpeechRecognizer(None)

    orchestrator = Orchestrator(source, adapter, vad, buffer, recognizer)

    # Start run task and cancel it immediately
    task = asyncio.create_task(orchestrator.run())
    await asyncio.sleep(0.005)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert vad.reset_called
    assert buffer.reset_called


@pytest.mark.asyncio
async def test_orchestrator_stream_ends_without_segment():
    source = MockAudioSource([])  # Empty stream

    adapter = MagicMock(spec=AudioFrameAdapter)
    vad = MockVoiceActivityDetector([])
    buffer = MockSpeechBuffer([])
    recognizer = MockSpeechRecognizer(None)

    orchestrator = Orchestrator(source, adapter, vad, buffer, recognizer)

    with pytest.raises(
        RuntimeError,
        match="Audio stream ended before a speech segment could be transcribed",
    ):
        await orchestrator.run()

    assert vad.reset_called
    assert buffer.reset_called
