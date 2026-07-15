import logging
import httpx
from llm.base import LanguageModel
from llm.models import AIResponse
from input.models.transcription import Transcription
from app.config.settings import settings

logger = logging.getLogger(__name__)


def _has_arabic_characters(text: str) -> bool:
    """Check if the string contains any Arabic Unicode characters (U+0600 to U+06FF)."""
    return any('\u0600' <= char <= '\u06FF' for char in text)


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
            "format": "json",
            "keep_alive": "30m",
            "options": {
                "temperature": 0.2,
                "num_predict": 256,
            }
        }
        
        attempts = 2
        for attempt in range(attempts):
            logger.info("Generating response for model: %s (attempt %d/%d)", self.model_name, attempt + 1, attempts)
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
                
                for field in ("action", "department", "reason", "message"):
                    if field not in parsed_data:
                        raise RuntimeError(f"LLM response JSON is missing required field '{field}'")
                        
                action = parsed_data["action"]
                department = parsed_data["department"]
                message = parsed_data["message"]

                from llm.models import ROUTING_ACTIONS, CUSTOMER_SERVICE_DEPARTMENTS

                if action not in ROUTING_ACTIONS:
                    raise RuntimeError(
                        f"LLM response 'action' must be one of {ROUTING_ACTIONS}, got '{action}'"
                    )

                if action == "customer_service":
                    if department not in CUSTOMER_SERVICE_DEPARTMENTS:
                        raise RuntimeError(
                            f"LLM response 'department' must be one of {CUSTOMER_SERVICE_DEPARTMENTS} when action is 'customer_service', got '{department}'"
                        )
                else:
                    if department is not None:
                        raise RuntimeError(
                            f"LLM response 'department' must be null when action is '{action}', got '{department}'"
                        )
                
                # Language consistency validation
                if transcription.language == "ar":
                    if not _has_arabic_characters(message):
                        raise RuntimeError(
                            f"Language mismatch: Expected Arabic response message for transcription language 'ar', got '{message}'"
                        )
                elif transcription.language == "en":
                    if _has_arabic_characters(message):
                        raise RuntimeError(
                            f"Language mismatch: Expected English response message for transcription language 'en', got '{message}'"
                        )
                
                return AIResponse(
                    action=action,
                    department=department,
                    reason=parsed_data["reason"],
                    message=message,
                    language=transcription.language,
                )
            except Exception as exc:
                if attempt < attempts - 1:
                    logger.warning("LLM returned invalid response or JSON. Retrying generation once. Error: %s", exc)
                    continue
                else:
                    raise RuntimeError(f"LLM validation failed after {attempts} attempts. Final error: {exc}. Response text: {response_text}") from exc

    async def close(self) -> None:
        """Explicitly close and dispose of the connection pool."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.info("OllamaLanguageModel connection pool closed.")
