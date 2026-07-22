"""
RAG Factory

Constructs and returns a fully wired RagService.
All dependencies — including the new QueryOptimizer and UnknownAnswerDetector —
are assembled here. Callers never see construction details.
"""

from pathlib import Path

from app.factories.retrieval import build_retrieval_service
from app.query_optimization.factory import build_query_optimizer
from app.rag.builders.context_builder import ContextBuilder
from app.rag.builders.prompt_builder import PromptBuilder
from app.rag.providers.ollama_provider import OllamaRagProvider
from app.rag.services.rag_service import RagService
from app.unknown_detection.factory import build_unknown_detector


def build_rag_service(template_path: Path | None = None) -> RagService:
    """
    Return a fully wired RagService with all quality-layer components.

    Args:
        template_path: Optional path to the prompt template file.
                       Defaults to the standard template in app/rag/prompts/.
    """
    if template_path is None:
        template_path = Path(__file__).parent.parent / "rag" / "prompts" / "default_rag.txt"

    try:
        template = template_path.read_text(encoding="utf-8")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read prompt template from {template_path}: {exc}"
        ) from exc

    return RagService(
        retrieval_service=build_retrieval_service(),
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(template),
        llm_provider=OllamaRagProvider(),
        query_optimizer=build_query_optimizer(),
        unknown_detector=build_unknown_detector(),
    )
