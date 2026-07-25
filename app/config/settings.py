from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ==========================
    # Project
    # ==========================
    PROJECT_NAME: str = "Online RAG Customer Service Assistant"

    # ==========================
    # PostgreSQL
    # ==========================
    POSTGRES_HOST: str = "localhost"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "voice_agent"
    POSTGRES_PORT: int = 5432

    # ==========================
    # Redis
    # ==========================
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379

    # ==========================
    # Qdrant
    # ==========================
    QDRANT_HOST: str = "localhost"
    QDRANT_HTTP_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "knowledge_base"
    QDRANT_DISTANCE_METRIC: str = "Cosine"
    QDRANT_BATCH_SIZE: int = 16

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

    # ==========================
    # Language Model (LLM)
    # ==========================
    LLM_PROVIDER: str = "groq"  # "ollama" | "groq"  — used by RAG factory only

    # ------------------------------------------------------------------
    # DEPRECATED — Ollama/Qwen3 voice-path settings
    # These settings were used by OllamaLanguageModel in the voice pipeline.
    # The voice path now uses RagLanguageModel → GroqProvider exclusively.
    # These fields are retained to avoid breaking imports but are no longer
    # read at runtime. They can be removed after the migration is confirmed.
    # ------------------------------------------------------------------
    LLM_BASE_URL: str = "http://localhost:11434"   # deprecated
    LLM_MODEL: str = "qwen3:1.7b"                 # deprecated
    LLM_TIMEOUT: float = 180.0                     # deprecated
    LLM_NUM_PREDICT: int = 350                     # deprecated
    LLM_TEMPERATURE: float = 0.1                   # deprecated
    LLM_TOP_P: float = 0.2                         # deprecated
    LLM_REPEAT_PENALTY: float = 1.1                # deprecated

    # ==========================
    # Groq Settings
    # ==========================
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    GROQ_TIMEOUT: float = 30.0
    GROQ_TEMPERATURE: float = 0.1
    GROQ_MAX_TOKENS: int = 350

    # ==========================
    # Silma Text-to-Speech (TTS)
    # ==========================
    SILMA_API_KEY: str = ""
    SILMA_BASE_URL: str = ""
    SILMA_MODEL_ID: str = ""
    SILMA_VOICE_ID: str = ""

    # ==========================
    # Embeddings
    # ==========================
    # Model loaded via SentenceTransformer for multilingual semantic embeddings.
    # Override in .env only when switching models.
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_BATCH_SIZE: int = 32

    # ==========================
    # Knowledge Base
    # ==========================
    # Directory containing the raw JSON knowledge files.
    # Override in .env to point at a different data directory.
    KNOWLEDGE_DATA_PATH: Path = Path("data")

    # ==========================
    # RAG Pipeline Settings
    # ==========================
    RAG_TOP_K: int = 5
    RAG_MAX_CONTEXT_CHARS: int = 4000
    RAG_DEBUG: bool = False
    RAG_RECOVERY_MIN_SCORE: float = 0.35  # Cutoff below which low-confidence recovery is skipped
    QUERY_NORMALIZER_ENABLED: bool = True
    QUERY_OPTIMIZER_ENABLED: bool = True  # Backward-compatible alias for QUERY_NORMALIZER_ENABLED
    TRANSLATION_ENABLED: bool = False
    TRANSLATION_PROVIDER: str = "none"  # "none" | "qwen"
    RAG_REFUSAL_MSG_EN: str = "I don't have enough information in the available knowledge base to answer this question."
    RAG_REFUSAL_MSG_AR: str = "لا أملك معلومات كافية في قاعدة المعرفة للإجابة على هذا السؤال."

    # ==========================
    # Unknown Answer Detection
    # ==========================
    # Minimum cosine similarity score for the top-ranked document.
    # Queries whose best match falls below this threshold are rejected.
    UNKNOWN_DETECTOR_MIN_SCORE: float = 0.58
    # Minimum number of retrieved documents required before scoring is applied.
    UNKNOWN_DETECTOR_MIN_RESULTS: int = 1
    # Minimum acceptable mean similarity score across all retrieved documents.
    UNKNOWN_DETECTOR_MEAN_THRESHOLD: float = 0.50

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
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