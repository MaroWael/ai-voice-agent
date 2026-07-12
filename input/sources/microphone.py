import asyncio
import time
from typing import AsyncGenerator

import sounddevice as sd

from input.models.audio_frame import AudioFrame
from input.sources.base import AudioSource

# ~20 ms of audio at the device's native sample rate.
FRAME_DURATION_MS = 20


class MicrophoneSource(AudioSource):
    """
    Captures raw audio from the system's default input device.

    Produces one AudioFrame every ~20 ms at the device's native sample rate.
    Sample-rate normalization is the responsibility of the Audio Adapter layer.

    This class does not perform resampling, normalization, noise reduction,
    VAD, or any form of audio processing. It only reads audio.
    """

    async def stream(self) -> AsyncGenerator[AudioFrame, None]:
        # Locate the WASAPI host API and its default input device.
        # sounddevice resolves sd.query_devices(kind="input") to the MME
        # default (hostapi=0) on Windows, which can trigger double callbacks.
        # WASAPI has a clean, well-defined scheduling model and avoids this.
        hostapis = sd.query_hostapis()
        wasapi = next(
            api for api in hostapis if "WASAPI" in api["name"]
        )
        device_index: int = wasapi["default_input_device"]

        device_info = sd.query_devices(device_index)
        sample_rate = int(device_info["default_samplerate"])
        # Capture mono. max_input_channels is the device ceiling, not the
        # desired channel count. Downstream layers expect a known channel count.
        channels = 1
        blocksize = int(sample_rate * FRAME_DURATION_MS / 1000)

        loop = asyncio.get_running_loop()

        # Open in blocking mode (no callback).
        # The blocking read API accumulates exactly `blocksize` frames inside
        # PortAudio and returns once — it is immune to WASAPI's double-fire
        # behaviour that affects the callback API on Windows.
        with sd.InputStream(
            device=device_index,
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
            blocksize=blocksize,
        ) as mic:
            try:
                while True:
                    # run_in_executor offloads the blocking read to a worker
                    # thread so the event loop is never stalled.
                    # One blocking read = exactly one AudioFrame. No duplicates.
                    print("READ", time.monotonic())  # DEBUG
                    data, _ = await loop.run_in_executor(
                        None, mic.read, blocksize
                    )
                    frame = AudioFrame(
                        samples=data.copy(),
                        sample_rate=sample_rate,
                        channels=channels,
                        timestamp=time.monotonic(),
                    )
                    print("YIELD", frame.timestamp)  # DEBUG
                    yield frame
            except asyncio.CancelledError:
                # The caller cancelled the generator (e.g. barge-in, shutdown).
                # The `with` block closes the hardware stream on exit.
                return
