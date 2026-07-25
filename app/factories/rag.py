"""
RAG Factory

Constructs and returns a fully wired RagService.
"""

from pathlib import Path

from app.factories.llm import build_llm_provider
from app.factories.retrieval import build_retrieval_service
from app.factories.translation import build_translation_service
from app.query_optimization.factory import build_query_enhancer, build_query_normalizer
from app.rag.builders.context_builder import ContextBuilder
from app.rag.builders.prompt_builder import PromptBuilder
from app.rag.services.rag_service import RagService
from app.unknown_detection.factory import build_unknown_detector


def build_rag_service(template_path: Path | None = None) -> RagService:
    """
    Return a fully wired RagService with all pipeline components.
    """
    if template_path is None:
        template_path = Path(__file__).parent.parent / "rag" / "prompts" / "default_rag.txt"

    try:
        template = template_path.read_text(encoding="utf-8")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read prompt template from {template_path}: {exc}"
        ) from exc

    llm_provider = build_llm_provider()

    return RagService(
        retrieval_service=build_retrieval_service(),
        context_builder=ContextBuilder(),
        prompt_builder=PromptBuilder(template),
        llm_provider=llm_provider,
        query_normalizer=build_query_normalizer(),
        query_enhancer=build_query_enhancer(llm_provider),
        unknown_detector=build_unknown_detector(),
        translation_service=build_translation_service(),
    )
