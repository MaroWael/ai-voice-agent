import asyncio
import time
from typing import AsyncGenerator

import sounddevice as sd

from input.models.audio_frame import AudioFrame
from input.sources.base import AudioSource

class MicrophoneSource(AudioSource):
    """
    Captures raw audio from the system's default input device.

    Produces one AudioFrame every `frame_duration_ms` milliseconds at the device's native sample rate.
    Sample-rate normalization is the responsibility of the Audio Adapter layer.

    This class does not perform resampling, normalization, noise reduction,
    VAD, or any form of audio processing. It only reads audio.
    """

    def __init__(
        self,
        frame_duration_ms: int = 20,
        channels: int = 1,
        device: int | str | None = None,
    ) -> None:
        self.frame_duration_ms = frame_duration_ms
        self.channels = channels
        self.device = device

    async def stream(self) -> AsyncGenerator[AudioFrame, None]:
        # Retrieve default or specified device information to read native sample rate
        if self.device is None:
            device_info = sd.query_devices(kind="input")
        else:
            device_info = sd.query_devices(self.device)

        sample_rate = int(device_info["default_samplerate"])
        blocksize = int(sample_rate * self.frame_duration_ms / 1000)

        loop = asyncio.get_running_loop()

        # Open in blocking mode (no callback).
        # The blocking read API accumulates exactly `blocksize` frames inside
        # PortAudio and returns once.
        with sd.InputStream(
            device=self.device,
            samplerate=sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=blocksize,
        ) as mic:
            try:
                while True:
                    # run_in_executor offloads the blocking read to a worker
                    # thread so the event loop is never stalled.
                    # One blocking read = exactly one AudioFrame.
                    data, _ = await loop.run_in_executor(
                        None, mic.read, blocksize
                    )
                    yield AudioFrame(
                        samples=data.copy(),
                        sample_rate=sample_rate,
                        channels=self.channels,
                        timestamp=time.monotonic(),
                    )
            except asyncio.CancelledError:
                # The caller cancelled the generator (e.g. barge-in, shutdown).
                # The `with` block closes the hardware stream on exit.
                return
