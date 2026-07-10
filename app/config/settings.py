from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ==========================
    # Project
    # ==========================
    PROJECT_NAME: str

    # ==========================
    # PostgreSQL
    # ==========================
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_PORT: int

    # ==========================
    # Redis
    # ==========================
    REDIS_PORT: int

    # ==========================
    # Qdrant
    # ==========================
    QDRANT_HTTP_PORT: int

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()