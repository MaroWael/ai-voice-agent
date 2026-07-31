import sys
import logging
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Response

from app.config.settings import settings
from app.db.postgres import check_postgres
from app.db.qdrant import check_qdrant
from app.db.redis import check_redis
from llm.check import check_llm
from app.tts.check import check_tts

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
        logger.warning(
            "PostgreSQL connection failed (non-fatal — not used by active pipeline): %s", exc
        )

    try:
        await check_redis()
    except Exception as exc:
        logger.warning(
            "Redis connection failed (non-fatal — not used by active pipeline): %s", exc
        )

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

    try:
        await check_tts()
    except Exception as exc:
        logger.warning(
            "TTS configuration check failed (non-fatal — add Silma credentials to .env to enable audio output): %s", exc
        )

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

    global _tts_cache
    if _tts_cache is not None:
        try:
            logger.info("Closing TTS cache...")
            await _tts_cache.close()
        except Exception as exc:
            logger.error("Failed to close TTS cache: %s", exc)


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


from llm.models import RoutingAction, CustomerServiceDepartment


class LLMStructuredResponse(BaseModel):
    action: RoutingAction
    department: CustomerServiceDepartment | None
    reason: str
    message: str


class DemoOrchestratorResponse(BaseModel):
    transcription: str
    language: str | None
    response: LLMStructuredResponse

_recognizer_cache = None
_llm_cache = None
_tts_cache = None

@app.post("/demo/stt", response_model=TranscriptionResponse)
async def demo_stt():
    from input.sources.microphone import MicrophoneSource
    from input.adapter.audio_frame_adapter import AudioFrameAdapter
    from input.vad.silero import SileroVAD
    from input.buffer.speech_buffer import SpeechBuffer
    from input.stt import build_speech_recognizer

    global _recognizer_cache
    
    print("Listening...")
    sys.stdout.flush()

    if _recognizer_cache is None:
        _recognizer_cache = build_speech_recognizer()
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


@app.post("/demo/orchestrator")
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
    from input.stt import build_speech_recognizer
    from orchestration.orchestrator import Orchestrator
    from llm.rag_llm import RagLanguageModel

    global _recognizer_cache, _llm_cache, _tts_cache

    if _recognizer_cache is None:
        _recognizer_cache = build_speech_recognizer()
        await _recognizer_cache.initialize()
    recognizer = _recognizer_cache

    if _llm_cache is None:
        _llm_cache = RagLanguageModel()
        await _llm_cache.initialize()
    llm = _llm_cache

    if _tts_cache is None:
        from app.tts import SilmaTTS
        _tts_cache = SilmaTTS()
    tts = _tts_cache

    source = MicrophoneSource(frame_duration_ms=32)
    adapter = AudioFrameAdapter()
    vad = SileroVAD(threshold=0.5)
    buffer = SpeechBuffer(max_silence_duration_ms=1000, pre_speech_padding_ms=200)

    from app.conversation import ConversationManager
    conv_mgr = ConversationManager()

    orchestrator = Orchestrator(source, adapter, vad, buffer, recognizer, llm, tts, conversation_manager=conv_mgr)

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

    # Format text for speech output at final TTS boundary
    from app.speech_formatting import SpeechResponseFormatter
    speech_formatter = SpeechResponseFormatter()
    speech_text = speech_formatter.format(
        result.response.message,
        language=result.response.language,
        transcription_language=result.transcription.language,
    )

    # Synthesize the assistant's response to audio
    logger.info("TTS INPUT: %s", speech_text)
    try:
        audio_bytes = await _synthesize_tts_helper(tts, speech_text)
    except ValueError as exc:
        logger.error("TTS input validation failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("TTS synthesis failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {exc}")

    # Temporary debugging logs
    logger.info("Returned audio first 44 bytes: %s", audio_bytes[:44])
    logger.info("Returned audio length: %d", len(audio_bytes))
    logger.info("Response media type: audio/wav")

    return Response(content=audio_bytes, media_type="audio/wav")


async def _synthesize_tts_helper(tts, text: str) -> bytes:
    """
    Helper function to perform TTS synthesis with SpeechChunker text splitting and audio chunk merging.
    """
    from app.speech_formatting.chunker import SpeechChunker
    from app.tts.audio_utils import merge_audio_chunks

    chunker = SpeechChunker()
    chunks = chunker.split(text)

    if not chunks:
        logger.warning("SpeechChunker returned no chunks for text.")
        return b""

    logger.info("TTS Synthesis Started — Original length: %d chars, Chunks created: %d", len(text), len(chunks))

    audio_chunks = []
    for idx, chunk in enumerate(chunks, start=1):
        logger.info("Synthesizing TTS Chunk %d/%d (%d chars): %r", idx, len(chunks), len(chunk), chunk)
        try:
            audio = await tts.synthesize(chunk)
            audio_chunks.append(audio)
        except ValueError as exc:
            logger.error("TTS chunk %d input validation failed: %s", idx, exc)
            raise
        except Exception as exc:
            logger.exception("TTS chunk %d synthesis failed: %s", idx, exc)
            raise

    merged_audio = merge_audio_chunks(audio_chunks)
    logger.info("TTS Synthesis Completed — Merged %d chunk(s), Total audio bytes: %d", len(audio_chunks), len(merged_audio))
    return merged_audio


@app.websocket("/ws/audio")
async def websocket_audio(
    websocket: WebSocket,
    session_id: str = "default_session",
    sample_rate: int = 16000,
    channels: int = 1,
    format: Literal["int16", "float32"] = "int16"
):
    await websocket.accept()
    logger.info("WebSocket connection accepted: session_id=%s, sample_rate=%d, channels=%d, format=%s", session_id, sample_rate, channels, format)

    from input.adapter.audio_frame_adapter import AudioFrameAdapter
    from input.vad.silero import SileroVAD
    from input.buffer.speech_buffer import SpeechBuffer
    from input.models.audio_frame import AudioFrame
    from orchestration.orchestrator import Orchestrator, OrchestratorResult
    from llm.rag_llm import RagLanguageModel
    from input.stt import build_speech_recognizer
    from app.conversation import ConversationManager
    import numpy as np
    import time
    import asyncio

    global _recognizer_cache, _llm_cache, _tts_cache

    if _recognizer_cache is None:
        _recognizer_cache = build_speech_recognizer()
        await _recognizer_cache.initialize()
    recognizer = _recognizer_cache

    if _llm_cache is None:
        _llm_cache = RagLanguageModel()
        await _llm_cache.initialize()
    llm = _llm_cache

    if _tts_cache is None:
        from app.tts import SilmaTTS
        _tts_cache = SilmaTTS()
    tts = _tts_cache

    adapter = AudioFrameAdapter()
    vad = SileroVAD(threshold=0.5)
    buffer = SpeechBuffer(max_silence_duration_ms=1000, pre_speech_padding_ms=200)
    
    conversation_manager = ConversationManager()
    
    # Instantiate Orchestrator with ConversationManager lifecycle owner
    orchestrator = Orchestrator(None, adapter, vad, buffer, recognizer, llm, tts, conversation_manager=conversation_manager)

    # Bounded queue of size 3 for completed speech segments
    queue = asyncio.Queue(maxsize=3)

    # Background worker task to process enqueued segments
    async def worker():
        try:
            while True:
                segment = await queue.get()
                try:
                    # Execute full Orchestrator turn (STT -> ConversationManager -> Router / RAG)
                    result = await orchestrator.process_speech_segment(segment, session_id=session_id)

                    # Send stt_result event immediately so User message renders with full transcript
                    stt_payload = {
                        "type": "stt_result",
                        "transcription": result.transcription.text,
                        "language": result.transcription.language,
                    }
                    try:
                        await websocket.send_json(stt_payload)
                    except (WebSocketDisconnect, RuntimeError) as send_err:
                        logger.info("Client disconnected while sending STT payload. Worker exiting.")
                        return

                    # 3. Perform TTS Synthesis (with Failsafe Error Handling)
                    audio_bytes = None
                    try:
                        await websocket.send_json({"type": "tts_started"})
                        from app.speech_formatting import SpeechResponseFormatter
                        speech_formatter = SpeechResponseFormatter()
                        speech_text = speech_formatter.format(
                            result.response.message,
                            language=result.response.language,
                            transcription_language=result.transcription.language,
                        )
                        audio_bytes = await _synthesize_tts_helper(tts, speech_text)
                    except Exception as tts_exc:
                        logger.warning("TTS synthesis failed (failsafe mode triggered): %s", tts_exc)
                        audio_bytes = None

                    # 4. Send assistant_response JSON payload (held until audio is ready or TTS finishes)
                    response_payload = {
                        "type": "assistant_response",
                        "transcription": result.transcription.text,
                        "language": result.transcription.language,
                        "response": {
                            "action": result.response.action,
                            "department": result.response.department,
                            "reason": result.response.reason,
                            "message": result.response.message
                        },
                        "has_audio": bool(audio_bytes)
                    }
                    try:
                        await websocket.send_json(response_payload)
                    except (WebSocketDisconnect, RuntimeError) as send_err:
                        logger.info("Client disconnected while sending JSON response. Worker exiting.")
                        return

                    # 5. Send Audio Bytes if available; else send tts_failed
                    if audio_bytes:
                        logger.info("Sending audio")
                        try:
                            await websocket.send_bytes(audio_bytes)
                            logger.info("Audio sent successfully")
                        except (WebSocketDisconnect, RuntimeError) as send_err:
                            logger.info("Client disconnected while sending audio bytes. Worker exiting.")
                            return

                        try:
                            await websocket.send_json({"type": "tts_finished"})
                        except (WebSocketDisconnect, RuntimeError) as send_err:
                            logger.info("Client disconnected while sending tts_finished. Worker exiting.")
                            return
                    else:
                        try:
                            await websocket.send_json({"type": "tts_failed", "reason": "Audio synthesis unavailable"})
                        except (WebSocketDisconnect, RuntimeError) as send_err:
                            logger.info("Client disconnected while sending tts_failed. Worker exiting.")
                            return

                except Exception as exc:
                    logger.exception("Error during background speech processing: %s", exc)
                    try:
                        await websocket.close(code=1011, reason="Internal processing error")
                    except Exception:
                        pass
                    return
                finally:
                    queue.task_done()
        except asyncio.CancelledError:
            logger.info("Background worker task cancelled.")
            raise

    # Start the worker task
    worker_task = asyncio.create_task(worker())

    start_time = time.monotonic()
    processed_samples = 0

    try:
        while True:
            data = await websocket.receive_bytes()
            
            if format == "int16":
                item_size = 2
                remainder = len(data) % item_size
                if remainder != 0:
                    data = data[:-remainder]
                if not data:
                    continue
                samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            elif format == "float32":
                item_size = 4
                remainder = len(data) % item_size
                if remainder != 0:
                    data = data[:-remainder]
                if not data:
                    continue
                samples = np.frombuffer(data, dtype=np.float32)
            else:
                logger.error("Unsupported format in WebSocket: %s", format)
                await websocket.close(code=1003, reason=f"Unsupported format: {format}")
                break

            # Compute continuous monotonic timestamp based on samples processed
            # to keep input timeline perfectly aligned.
            chunk_timestamp = start_time + (processed_samples / sample_rate)
            processed_samples += len(samples)

            frame = AudioFrame(
                samples=samples,
                sample_rate=sample_rate,
                channels=channels,
                timestamp=chunk_timestamp
            )

            # receive_audio_frame only adapts, VAD detects, buffers (extremely fast, non-blocking)
            segment = await orchestrator.receive_audio_frame(frame)
            if segment is not None:
                # Bounded queue non-blocking push with Drop Newest policy
                if queue.full():
                    logger.warning("Speech queue is full. Dropping the newest speech segment to prevent blocking.")
                else:
                    queue.put_nowait(segment)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client")
    except Exception as exc:
        logger.exception("WebSocket connection error: %s", exc)
        try:
            await websocket.close(code=1011, reason=str(exc))
        except Exception:
            pass
    finally:
        # Cancel worker and wait for it to stop
        logger.info("Cleaning up WebSocket session background task...")
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
        # Clear the queue
        while not queue.empty():
            try:
                queue.get_nowait()
                queue.task_done()
            except asyncio.QueueEmpty:
                break



# ── Evaluation endpoint ─────────────────────────────────────────────────────
# This endpoint is used exclusively by the RAG evaluation pipeline.
# It routes a raw text question directly through the full RAG pipeline
# (QueryNormalizer → Qdrant → UnknownAnswerDetector → Groq LLM)
# without going through STT or TTS.

class EvalQueryRequest(BaseModel):
    question: str


class EvalQueryResponse(BaseModel):
    question: str
    answer: str
    status: str


@app.post("/eval/query", response_model=EvalQueryResponse)
async def eval_query(request: EvalQueryRequest):
    """
    Direct text-to-RAG evaluation endpoint.
    Accepts a plain-text question and returns the RAG pipeline answer as JSON.
    Used exclusively by the evaluation pipeline (rag_evaluator.py).
    """
    from app.factories.rag import build_rag_service

    rag_service = build_rag_service()
    await rag_service.initialize()

    try:
        rag_response = await rag_service.answer(
            question=request.question,
            debug=False,
        )
        return EvalQueryResponse(
            question=request.question,
            answer=rag_response.answer,
            status=rag_response.status.value,
        )
    except Exception as exc:
        logger.exception("eval_query pipeline failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        await rag_service.close()


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


