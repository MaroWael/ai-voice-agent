import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config.settings import settings
from app.db.postgres import check_postgres
from app.db.qdrant import check_qdrant
from app.db.redis import check_redis

# ── TEMPORARY: Feature 1 verification — remove after manual test is complete ──
from input.sources.microphone import MicrophoneSource
# ─────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# ── TEMPORARY: Feature 1 verification — remove after manual test is complete ──
async def _print_microphone_frames() -> None:
    """Read frames from the microphone and print lightweight metadata."""
    source = MicrophoneSource()
    async for frame in source.stream():
        print(
            f"AudioFrame("
            f"timestamp={frame.timestamp:.4f}, "
            f"shape={frame.samples.shape}, "
            f"sample_rate={frame.sample_rate}, "
            f"channels={frame.channels}"
            f")"
        )
# ─────────────────────────────────────────────────────────────────────────────


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

    logger.info("%s is ready.", settings.PROJECT_NAME)

    # ── TEMPORARY: Feature 1 verification — remove after manual test is complete ──
    mic_task = asyncio.create_task(_print_microphone_frames())
    # ─────────────────────────────────────────────────────────────────────────────

    yield  # application runs here

    # ── Shutdown ─────────────────────────────────────────────────────────────
    # ── TEMPORARY: Feature 1 verification — remove after manual test is complete ──
    mic_task.cancel()
    # ─────────────────────────────────────────────────────────────────────────────

    logger.info("%s is shutting down.", settings.PROJECT_NAME)


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
