import io
import logging
import wave

logger = logging.getLogger(__name__)


def merge_audio_chunks(chunks: list[bytes]) -> bytes:
    """
    Combines multiple audio chunk byte payloads into a single contiguous audio bytes payload.

    Handles:
    - Empty or single-element chunk lists.
    - Standard RIFF/WAV files: concatenates PCM audio frames and updates header sample metadata.
    - Raw unheadered PCM bytes: concatenates directly.

    Args:
        chunks: List of audio chunk bytes returned by TTS providers.

    Returns:
        A single contiguous bytes object representing the merged audio payload.
    """
    if not chunks:
        return b""

    # Filter out empty byte strings
    valid_chunks = [c for c in chunks if c and len(c) > 0]
    if not valid_chunks:
        return b""

    if len(valid_chunks) == 1:
        return valid_chunks[0]

    first_chunk = valid_chunks[0]

    # If first chunk is not WAV headered (doesn't start with b"RIFF"), concatenate raw bytes
    if not first_chunk.startswith(b"RIFF"):
        return b"".join(valid_chunks)

    try:
        # Read parameters from the first WAV header
        with wave.open(io.BytesIO(first_chunk), "rb") as first_wav:
            params = first_wav.getparams()

        out_buf = io.BytesIO()
        with wave.open(out_buf, "wb") as out_wav:
            out_wav.setparams(params)
            for idx, chunk in enumerate(valid_chunks, start=1):
                if chunk.startswith(b"RIFF"):
                    with wave.open(io.BytesIO(chunk), "rb") as w:
                        frames = w.readframes(w.getnframes())
                        out_wav.writeframes(frames)
                else:
                    out_wav.writeframes(chunk)

        merged_bytes = out_buf.getvalue()
        logger.info(
            "Successfully merged %d WAV audio chunk(s) into %d total bytes",
            len(valid_chunks),
            len(merged_bytes),
        )
        return merged_bytes
    except Exception as exc:
        logger.warning(
            "WAV audio chunk merging via wave module failed (%s); falling back to direct byte concatenation.",
            exc,
        )
        return b"".join(valid_chunks)
