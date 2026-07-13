from collections import deque
from typing import List, Optional
import numpy as np

from input.models.audio_frame import AudioFrame
from input.models.speech_segment import SpeechSegment
from input.models.vad_result import VADResult


class SpeechBuffer:
    """
    Accumulates AudioFrames into complete SpeechSegments based on VADResults.

    Manages a state machine to group streaming frames into clean utterances,
    prepending pre-speech history to mitigate VAD onset latency and trimming
    trailing silence.
    """

    def __init__(
        self,
        max_silence_duration_ms: int = 500,
        pre_speech_padding_ms: int = 200,
    ) -> None:
        """
        Initialize the SpeechBuffer.

        Args:
            max_silence_duration_ms: Continuous silence allowed inside an utterance before finalizing.
            pre_speech_padding_ms: Duration of silent history to prepend to the start of speech.
        """
        self.max_silence_duration_ms = max_silence_duration_ms
        self.pre_speech_padding_ms = pre_speech_padding_ms

        self._is_speaking = False
        self._active_frames: List[AudioFrame] = []
        self._silence_frames: List[AudioFrame] = []
        self._silence_duration = 0.0

        self._pre_speech_buffer: deque[AudioFrame] = deque()
        self._pre_speech_duration = 0.0

    async def process(self, frame: AudioFrame, vad: VADResult) -> Optional[SpeechSegment]:
        """
        Process a single AudioFrame and its VADResult.

        Returns:
            A finalized SpeechSegment if the silence timeout is met, otherwise None.
        """
        frame_duration = frame.samples.shape[0] / frame.sample_rate

        if vad.is_speech:
            if not self._is_speaking:
                self._is_speaking = True
                
                # Prepend pre-speech silent padding
                self._active_frames.extend(self._pre_speech_buffer)
                self._pre_speech_buffer.clear()
                self._pre_speech_duration = 0.0
                
                self._active_frames.append(frame)
            else:
                # Merge tolerated pause frames back into the segment
                if self._silence_frames:
                    self._active_frames.extend(self._silence_frames)
                    self._silence_frames.clear()
                self._active_frames.append(frame)
                self._silence_duration = 0.0
            
            return None
        else:
            if not self._is_speaking:
                # Slide frame into the padding history window
                self._pre_speech_buffer.append(frame)
                self._pre_speech_duration += frame_duration
                
                while self._pre_speech_duration > (self.pre_speech_padding_ms / 1000.0) and self._pre_speech_buffer:
                    removed_frame = self._pre_speech_buffer.popleft()
                    self._pre_speech_duration -= removed_frame.samples.shape[0] / removed_frame.sample_rate
                
                return None
            else:
                # Buffer trailing silence frames
                self._silence_frames.append(frame)
                self._silence_duration += frame_duration
                
                # Finalize if silence threshold exceeded
                if self._silence_duration >= (self.max_silence_duration_ms / 1000.0):
                    return self._finalize_segment()
                
                return None

    def _finalize_segment(self) -> SpeechSegment:
        """
        Consolidate active frames into a finalized SpeechSegment.
        """
        if not self._active_frames:
            self.reset()
            raise RuntimeError("Cannot finalize SpeechSegment with no active frames.")

        all_samples = np.concatenate([f.samples for f in self._active_frames])
        sample_rate = self._active_frames[0].sample_rate
        start_timestamp = self._active_frames[0].timestamp
        
        last_frame = self._active_frames[-1]
        frame_duration = last_frame.samples.shape[0] / last_frame.sample_rate
        end_timestamp = last_frame.timestamp + frame_duration

        segment = SpeechSegment(
            samples=all_samples,
            sample_rate=sample_rate,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
        )

        self.reset()
        return segment

    def reset(self) -> None:
        """
        Reset all internal state.
        """
        self._is_speaking = False
        self._active_frames.clear()
        self._silence_frames.clear()
        self._silence_duration = 0.0
        self._pre_speech_buffer.clear()
        self._pre_speech_duration = 0.0
