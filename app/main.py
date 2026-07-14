import sys
import logging
import asyncio
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.config.settings import settings
from app.db.postgres import check_postgres
from app.db.qdrant import check_qdrant
from app.db.redis import check_redis
from llm.check import check_llm

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────────
    logger.info("Starting %s — verifying database connections...", settings.PROJECT_NAME)

    try:
        await check_postgres()
    except Exception as exc:
        logger.error("PostgreSQL connection failed: %s", exc)
        raise

    try:
        await check_redis()
    except Exception as exc:
        logger.error("Redis connection failed: %s", exc)
        raise

    try:
        await check_qdrant()
    except Exception as exc:
        logger.error("Qdrant connection failed: %s", exc)
        raise

    try:
        await check_llm()
    except Exception as exc:
        logger.error("LLM connection failed: %s", exc)
        raise

    logger.info("%s is ready.", settings.PROJECT_NAME)

    yield  # application runs here

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("%s is shutting down.", settings.PROJECT_NAME)

    global _llm_cache
    if _llm_cache is not None:
        try:
            logger.info("Closing LLM cache...")
            await _llm_cache.close()
        except Exception as exc:
            logger.error("Failed to close LLM cache: %s", exc)


app = FastAPI(
    title=settings.PROJECT_NAME,
    lifespan=lifespan,
)


@app.get("/")
async def root():
    return {
        "project": settings.PROJECT_NAME,
        "status": "running",
    }


from typing import Literal
from pydantic import BaseModel
import contextlib
import sys

class TranscriptionResponse(BaseModel):
    text: str
    language: str | None
    start_timestamp: float
    end_timestamp: float


class LLMStructuredResponse(BaseModel):
    action: Literal["retrieval", "tool", "human"]
    reason: str
    message: str


class DemoOrchestratorResponse(BaseModel):
    transcription: str
    language: str | None
    response: LLMStructuredResponse

_recognizer_cache = None
_llm_cache = None

@app.post("/demo/stt", response_model=TranscriptionResponse)
async def demo_stt():
    from input.sources.microphone import MicrophoneSource
    from input.adapter.audio_frame_adapter import AudioFrameAdapter
    from input.vad.silero import SileroVAD
    from input.buffer.speech_buffer import SpeechBuffer
    from input.stt.faster_whisper import FasterWhisperSTT

    global _recognizer_cache
    
    print("Listening...")
    sys.stdout.flush()

    if _recognizer_cache is None:
        _recognizer_cache = FasterWhisperSTT()
        await _recognizer_cache.initialize()
    recognizer = _recognizer_cache

    source = MicrophoneSource(frame_duration_ms=32)
    adapter = AudioFrameAdapter()
    vad = SileroVAD(threshold=0.5)
    buffer = SpeechBuffer(max_silence_duration_ms=1000, pre_speech_padding_ms=200)

    speech_detected_logged = False
    transcription_result = None

    try:
        async with contextlib.aclosing(source.stream()) as stream:
            async for frame in stream:
                adapted_frame = adapter.adapt(frame)
                vad_result = await vad.detect(adapted_frame)
                
                if vad_result.is_speech and not speech_detected_logged:
                    print("Speech detected...")
                    sys.stdout.flush()
                    speech_detected_logged = True
                
                segment = await buffer.process(adapted_frame, vad_result)
                
                if segment is not None:
                    print("Transcribing...")
                    sys.stdout.flush()
                    transcription_result = await recognizer.transcribe(segment)
                    print("Done.")
                    sys.stdout.flush()
                    break
    except Exception as exc:
        logger.exception("STT pipeline execution failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    if transcription_result is None:
        raise HTTPException(status_code=400, detail="No speech segment detected.")

    return TranscriptionResponse(
        text=transcription_result.text,
        language=transcription_result.language,
        start_timestamp=transcription_result.start_timestamp,
        end_timestamp=transcription_result.end_timestamp
    )


@app.post("/demo/orchestrator", response_model=DemoOrchestratorResponse)
async def demo_orchestrator():
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    from input.sources.microphone import MicrophoneSource
    from input.adapter.audio_frame_adapter import AudioFrameAdapter
    from input.vad.silero import SileroVAD
    from input.buffer.speech_buffer import SpeechBuffer
    from input.stt.faster_whisper import FasterWhisperSTT
    from orchestration.orchestrator import Orchestrator
    from llm.ollama import OllamaLanguageModel

    global _recognizer_cache, _llm_cache

    if _recognizer_cache is None:
        _recognizer_cache = FasterWhisperSTT()
        await _recognizer_cache.initialize()
    recognizer = _recognizer_cache

    if _llm_cache is None:
        _llm_cache = OllamaLanguageModel()
        await _llm_cache.initialize()
    llm = _llm_cache

    source = MicrophoneSource(frame_duration_ms=32)
    adapter = AudioFrameAdapter()
    vad = SileroVAD(threshold=0.5)
    buffer = SpeechBuffer(max_silence_duration_ms=1000, pre_speech_padding_ms=200)

    orchestrator = Orchestrator(source, adapter, vad, buffer, recognizer, llm)

    logger.info("Orchestrator Started")
    logger.info("Listening...")

    import time
    start_pipeline = time.perf_counter()
    try:
        result = await orchestrator.run()
    except Exception as exc:
        logger.exception("Orchestration pipeline execution failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    pipeline_elapsed = time.perf_counter() - start_pipeline

    logger.info("Transcription: %s", result.transcription.text)
    logger.info("Language: %s", result.transcription.language)
    logger.info(
        "AI Response - Action: %s, Reason: %s, Message: %s",
        result.response.action,
        result.response.reason,
        result.response.message
    )
    logger.info("Total pipeline completed in %.2f seconds", pipeline_elapsed)

    return DemoOrchestratorResponse(
        transcription=result.transcription.text,
        language=result.transcription.language,
        response=LLMStructuredResponse(
            action=result.response.action,
            reason=result.response.reason,
            message=result.response.message
        )
    )


if __name__ == "__main__":
    import uvicorn
    config = uvicorn.Config(app, host="127.0.0.1", port=8000, loop="asyncio")
    server = uvicorn.Server(config)
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(server.serve())
    finally:
        loop.close()


