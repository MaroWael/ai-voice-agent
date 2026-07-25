"""
Groq LLM Provider

Implements LLMProvider interface for Groq's OpenAI-compatible Chat Completions API.
Uses httpx.AsyncClient for asynchronous, dependency-free execution.
"""

import logging
import httpx

from app.config.settings import settings
from app.rag.providers.base import LLMProvider

logger = logging.getLogger(__name__)


class GroqProvider(LLMProvider):
    """
    Groq-backed LLM provider using OpenAI-compatible Chat Completions API.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.api_key = api_key or settings.GROQ_API_KEY
        self.model_name = model_name or settings.GROQ_MODEL
        self.base_url = (base_url or settings.GROQ_BASE_URL).rstrip("/")
        self.temperature = temperature if temperature is not None else settings.GROQ_TEMPERATURE
        self.max_tokens = max_tokens if max_tokens is not None else settings.GROQ_MAX_TOKENS
        self._client: httpx.AsyncClient | None = None

    async def initialize(self) -> None:
        """Initialize the httpx AsyncClient connection pool with authorization headers."""
        if not self.api_key:
            logger.warning(
                "GroqProvider initialized without GROQ_API_KEY. "
                "Ensure GROQ_API_KEY is configured in settings or environment."
            )

        if self._client is None:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=settings.GROQ_TIMEOUT,
            )
            logger.info(
                "GroqProvider initialized with model '%s' (base_url: %s)",
                self.model_name,
                self.base_url,
            )

    async def generate_with_metadata(
        self,
        prompt: str,
        num_predict: int | None = None,
    ) -> tuple[str, dict]:
        """
        Sends the prompt to Groq's OpenAI-compatible /chat/completions endpoint.

        Returns:
            Tuple of (answer_text, metadata_dict).
        """
        if self._client is None:
            raise RuntimeError("GroqProvider not initialized. Call initialize() first.")

        tokens_limit = num_predict if num_predict is not None else self.max_tokens

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": tokens_limit,
        }

        logger.info("Sending RAG prompt to Groq API (model: %s)", self.model_name)

        try:
            response = await self._client.post(
                "/chat/completions",
                json=payload,
            )
        except httpx.RequestError as exc:
            raise RuntimeError(f"Groq API connection request failed: {exc}") from exc

        if response.status_code != 200:
            raise RuntimeError(
                f"Groq API request failed with status {response.status_code}: {response.text}"
            )

        try:
            data = response.json()
        except Exception as exc:
            raise RuntimeError(f"Failed to parse Groq API response JSON: {response.text}") from exc

        if not isinstance(data, dict):
            raise RuntimeError(f"Unexpected response payload format from Groq: {data}")

        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(f"Groq API response payload missing 'choices': {data}")

        answer = choices[0].get("message", {}).get("content", "").strip()

        usage = data.get("usage", {})
        metadata = {
            "prompt_eval_count": usage.get("prompt_tokens", 0),
            "eval_count": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "raw_response": data,
        }

        return answer, metadata

    async def generate(self, prompt: str) -> str:
        """Sends prompt to Groq API and returns the generated answer text."""
        answer, _ = await self.generate_with_metadata(prompt)
        return answer

    async def close(self) -> None:
        """Close AsyncClient connection pool."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.info("GroqProvider connection pool closed.")
