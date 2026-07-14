import logging
import httpx
from app.config.settings import settings

logger = logging.getLogger(__name__)

async def check_llm() -> None:
    """Verify that the local Ollama LLM provider is reachable and the model is pulled."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{settings.LLM_BASE_URL}/api/tags",
            timeout=settings.LLM_TIMEOUT,
        )
        response.raise_for_status()
        
        data = response.json()
        models = [m["name"] for m in data.get("models", [])]
        
        model_name = settings.LLM_MODEL
        model_found = any(
            m == model_name or m.startswith(f"{model_name}:") or model_name.startswith(f"{m}:")
            for m in models
        )
        
        if not model_found:
            raise RuntimeError(
                f"Configured LLM model '{model_name}' was not found in Ollama. "
                f"Available models: {models}"
            )
            
    logger.info("LLM provider connection OK (model '%s' is active)", model_name)
