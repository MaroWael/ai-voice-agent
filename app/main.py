from fastapi import FastAPI

from app.config.settings import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
)


@app.get("/")
async def root():
    return {
        "project": settings.PROJECT_NAME,
        "status": "running",
    }