from abc import ABC, abstractmethod
from typing import AsyncGenerator

from input.models.audio_frame import AudioFrame


class AudioSource(ABC):
    """
    Abstract interface for any audio source in the pipeline.

    The Input Pipeline depends on this abstraction, not on any concrete
    implementation. This allows MicrophoneSource, TwilioSource, WebRTCSource,
    and any future source to be swapped without touching the rest of the pipeline.
    """

    @abstractmethod
    async def stream(self) -> AsyncGenerator[AudioFrame, None]:
        """
        Continuously yield AudioFrame objects until the source is exhausted
        or the caller cancels the generator.

        Implementations must handle asyncio.CancelledError to release
        any hardware or network resources cleanly.
        """
        ...
