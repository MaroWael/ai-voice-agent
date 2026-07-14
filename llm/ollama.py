import logging
import httpx
from llm.base import LanguageModel
from llm.models import AIResponse
from input.models.transcription import Transcription
from app.config.settings import settings

logger = logging.getLogger(__name__)

class OllamaLanguageModel(LanguageModel):
    """Integrates local LLMs using Ollama's HTTP endpoints."""

    def __init__(
        self,
        base_url: str | None = None,
        model_name: str | None = None,
    ) -> None:
        self.base_url = base_url or settings.LLM_BASE_URL
        self.model_name = model_name or settings.LLM_MODEL
        self._client: httpx.AsyncClient | None = None

    async def initialize(self) -> None:
        """Explicitly create the AsyncClient connection pool."""
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url)
            logger.info("OllamaLanguageModel initialized with base_url: %s", self.base_url)

    async def generate(self, transcription: Transcription) -> AIResponse:
        if self._client is None:
            raise RuntimeError("LanguageModel not initialized. Call initialize() first.")
            
        from llm.prompts import ROUTER_PROMPT

        prompt = f"""{ROUTER_PROMPT}

Customer:
{transcription.text}

Assistant:
"""

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "keep_alive": "30m",
            "options": {
                "temperature": 0.2,
                "num_predict": 64,
            }
        }
        
        logger.info("Generating response for model: %s", self.model_name)
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

        response_text = data["response"].strip()
        
        try:
            import json
            parsed_data = json.loads(response_text)
        except Exception as exc:
            raise RuntimeError(f"LLM returned invalid JSON: {response_text}") from exc
            
        if not isinstance(parsed_data, dict):
            raise RuntimeError(f"Unexpected response format from LLM (expected JSON object): {response_text}")
            
        for field in ("action", "reason", "message"):
            if field not in parsed_data:
                raise RuntimeError(f"LLM response JSON is missing required field '{field}': {response_text}")
                
        allowed_actions = ("retrieval", "tool", "human")
        if parsed_data["action"] not in allowed_actions:
            raise RuntimeError(
                f"LLM response 'action' must be one of {allowed_actions}, got '{parsed_data['action']}': {response_text}"
            )
        
        return AIResponse(
            action=parsed_data["action"],
            reason=parsed_data["reason"],
            message=parsed_data["message"],
            language=transcription.language,
        )

    async def close(self) -> None:
        """Explicitly close and dispose of the connection pool."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.info("OllamaLanguageModel connection pool closed.")
