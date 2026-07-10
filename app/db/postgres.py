import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config.settings import settings

logger = logging.getLogger(__name__)

# One engine and one session factory shared for the lifetime of the process.
# echo=False keeps production logs clean; enable only during local debugging.
_engine: AsyncEngine = create_async_engine(settings.postgres_url, echo=False)
_session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
    _engine, expire_on_commit=False
)


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    """Yield an AsyncSession and guarantee it is closed when the caller is done."""
    async with _session_factory() as session:
        yield session


async def check_postgres() -> None:
    """Verify that PostgreSQL is reachable. Raises on failure — never swallows errors."""
    async with get_session() as session:
        await session.execute(text("SELECT 1"))
    logger.info("PostgreSQL connection OK")
