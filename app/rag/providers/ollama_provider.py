import logging
import httpx
from app.config.settings import settings
from app.rag.providers.base import LLMProvider

logger = logging.getLogger(__name__)


class OllamaRagProvider(LLMProvider):
    """
    Ollama-based LLM provider for RAG plain text generation.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model_name: str | None = None,
    ) -> None:
        self.base_url = base_url or settings.LLM_BASE_URL
        self.model_name = model_name or settings.LLM_MODEL
        self._client: httpx.AsyncClient | None = None

    async def initialize(self) -> None:
        """Create the httpx.AsyncClient connection pool."""
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url)
            logger.info("OllamaRagProvider initialized with base_url: %s", self.base_url)

    async def generate(self, prompt: str) -> str:
        """Sends the prompt to Ollama's /api/generate and returns the raw string response."""
        if self._client is None:
            raise RuntimeError("OllamaRagProvider not initialized. Call initialize() first.")

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "keep_alive": "30m",
            "options": {
                "temperature": 0.2,
                "num_predict": 512,
            }
        }

        logger.info("Generating RAG response with model: %s", self.model_name)
        
        response = await self._client.post(
            "/api/generate",
            json=payload,
            timeout=settings.LLM_TIMEOUT,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama API request failed with status code {response.status_code}: {response.text}"
            )

        try:
            data = response.json()
        except Exception as exc:
            raise RuntimeError(f"Failed to parse Ollama response as JSON: {response.text}") from exc

        if not isinstance(data, dict):
            raise RuntimeError(f"Unexpected response format from Ollama (expected JSON object): {data}")

        if "response" not in data:
            raise RuntimeError(f"Ollama response payload missing expected 'response' key: {data}")

        return data["response"].strip()

    async def close(self) -> None:
        """Close connection client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.info("OllamaRagProvider connection pool closed.")
