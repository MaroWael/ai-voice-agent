import logging
from app.rag.models.response import RagResponse
from app.rag.builders.context_builder import ContextBuilder
from app.rag.builders.prompt_builder import PromptBuilder
from app.rag.providers.base import LLMProvider
from app.retrieval.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)


class RagService:
    """
    Coordinates RetrievalService, ContextBuilder, PromptBuilder, and LLMProvider
    to execute the RAG pipeline.
    """

    def __init__(
        self,
        retrieval_service: RetrievalService,
        context_builder: ContextBuilder,
        prompt_builder: PromptBuilder,
        llm_provider: LLMProvider,
    ) -> None:
        self._retrieval_service = retrieval_service
        self._context_builder = context_builder
        self._prompt_builder = prompt_builder
        self._llm_provider = llm_provider

    async def initialize(self) -> None:
        """
        Initializes owned resources (e.g. LLMProvider).
        """
        await self._llm_provider.initialize()

    async def close(self) -> None:
        """
        Closes/releases owned resources.
        """
        await self._llm_provider.close()

    async def answer(self, question: str, top_k: int | None = None) -> RagResponse:
        """
        Orchestrates the RAG flow:
        - Retrieves documents via RetrievalService.
        - Formats context block via ContextBuilder (synchronously).
        - Formats the prompt via PromptBuilder (synchronously).
        - Calls the LLMProvider to generate the answer asynchronously.
        - Packages the result into a RagResponse.
        """
        logger.info("RAG pipeline coordinator called with question: '%s'", question)

        # 1. Retrieve documents (delegate default config to RetrievalService if top_k is None)
        if top_k is not None:
            retrieved_docs = await self._retrieval_service.retrieve(question, top_k=top_k)
        else:
            retrieved_docs = await self._retrieval_service.retrieve(question)

        logger.info("Retrieved %d document(s) for RAG context.", len(retrieved_docs))

        # 2. Build context (pure, synchronous, preserves order)
        context = self._context_builder.build_context(retrieved_docs)

        # 3. Build prompt (pure, synchronous, template-driven)
        prompt = self._prompt_builder.build_prompt(question, context)

        # 4. Generate answer (asynchronous LLM generation)
        answer = await self._llm_provider.generate(prompt)

        logger.info("RAG generation complete.")

        # 5. Return RagResponse
        return RagResponse(
            answer=answer,
            prompt=prompt,
            retrieved_documents=retrieved_docs,
        )
