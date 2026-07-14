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
    POSTGRES_HOST: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_PORT: int

    # ==========================
    # Redis
    # ==========================
    REDIS_HOST: str
    REDIS_PORT: int

    # ==========================
    # Qdrant
    # ==========================
    QDRANT_HOST: str
    QDRANT_HTTP_PORT: int

    # ==========================
    # Audio Pipeline
    # ==========================
    # Canonical format that every layer downstream of the Adapter expects.
    # These are architectural constants, not deployment-specific values.
    # They have default values here and must never be required in .env.
    PIPELINE_SAMPLE_RATE: int = 16_000
    PIPELINE_CHANNELS: int = 1

    # ==========================
    # Speech-to-Text (STT)
    # ==========================
    STT_MODEL: str = "medium"
    STT_DEVICE: str = "cpu"
    STT_COMPUTE_TYPE: str = "float32"
    STT_BEAM_SIZE: int = 5

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    @property
    def postgres_url(self) -> str:
        # postgresql+psycopg uses psycopg3 async driver (psycopg[binary] in requirements.txt)
        return (
            f"postgresql+psycopg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def redis_url(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}"

    @property
    def qdrant_url(self) -> str:
        return f"http://{self.QDRANT_HOST}:{self.QDRANT_HTTP_PORT}"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()