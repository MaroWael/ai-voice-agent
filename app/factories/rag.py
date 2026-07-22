from pathlib import Path
from app.factories.retrieval import build_retrieval_service
from app.rag.builders.context_builder import ContextBuilder
from app.rag.builders.prompt_builder import PromptBuilder
from app.rag.providers.ollama_provider import OllamaRagProvider
from app.rag.services.rag_service import RagService


def build_rag_service(template_path: Path | None = None) -> RagService:
    """
    Constructs and returns a fully wired RagService.
    Loads the prompt template from the filesystem.
    """
    if template_path is None:
        template_path = Path(__file__).parent.parent / "rag" / "prompts" / "default_rag.txt"

    try:
        template = template_path.read_text(encoding="utf-8")
    except Exception as exc:
        raise RuntimeError(f"Failed to read prompt template from {template_path}: {exc}") from exc

    retrieval_service = build_retrieval_service()
    context_builder = ContextBuilder()
    prompt_builder = PromptBuilder(template)
    llm_provider = OllamaRagProvider()

    return RagService(
        retrieval_service=retrieval_service,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        llm_provider=llm_provider,
    )
