"""
Tests for End-to-End Comparison Entities Propagation

Verifies that comparison_entities propagates without being lost across:
ConversationManager -> Orchestrator._rag_executor -> RagLanguageModel -> RagService -> RetrievalService.retrieve_entity_wise()
and that the old single retrieval path is skipped.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.conversation.conversation_manager import ConversationManager
from app.conversation.models import ConversationEntity, EntityType, IntentType, RoutingDecision
from app.knowledge.models.document_metadata import DocumentMetadata
from app.knowledge.models.knowledge_document import KnowledgeDocument
from app.rag.models.response import RagResponse
from app.rag.models.status import RagStatus
from app.rag.services.rag_service import RagService
from app.retrieval.models.search_result import SearchResult
from app.retrieval.services.retrieval_service import RetrievalService
from input.models.transcription import Transcription
from llm.rag_llm import RagLanguageModel
from orchestration.orchestrator import Orchestrator


def _make_doc(doc_id: str, title: str, content: str, product_name: str) -> KnowledgeDocument:
    return KnowledgeDocument(
        id=doc_id,
        title=title,
        raw_content=content,
        content=content,
        metadata=DocumentMetadata(
            product_id=product_name.lower().replace(" ", "-"),
            product_name=product_name,
            section=title,
            aliases=[],
            arabic_name=product_name,
            language="en",
            source="test.json",
            url="http://test.com",
        ),
    )


@pytest.mark.asyncio
async def test_comparison_entities_end_to_end_propagation():
    """
    Simulates a comparison turn and verifies that retrieve_entity_wise() is called
    and that standard retrieve() is NOT called.
    """
    embedding_mock = AsyncMock()
    qdrant_mock = AsyncMock()

    doc_gold = _make_doc("d1", "Fees", "Visa Gold fee 200 EGP", "Visa Gold")
    doc_plat = _make_doc("d2", "Fees", "Visa Platinum fee 500 EGP", "Visa Platinum")

    # Qdrant return search results per entity
    qdrant_mock.search.side_effect = [
        [SearchResult(document=doc_gold, score=0.88, rank=1)],
        [SearchResult(document=doc_plat, score=0.92, rank=1)],
    ]

    retrieval_service = RetrievalService(embedding_mock, qdrant_mock)
    retrieval_service.retrieve_timed = AsyncMock(side_effect=[
        ([SearchResult(document=doc_gold, score=0.88, rank=1)], MagicMock()),
        ([SearchResult(document=doc_plat, score=0.92, rank=1)], MagicMock()),
    ])

    # Build RAG components
    context_builder = MagicMock()
    context_builder.build_context.return_value = "Retrieved Context"
    prompt_builder = MagicMock()
    prompt_builder.build_prompt.return_value = "Prompt Text"
    llm_provider = AsyncMock()
    llm_provider.generate.return_value = "Comparison Answer text"
    query_normalizer = AsyncMock()
    query_normalizer.normalize.side_effect = lambda q: q
    unknown_detector = AsyncMock()
    detection_mock = MagicMock()
    detection_mock.has_context = True
    reason_mock = MagicMock()
    reason_mock.value = "SUFFICIENT_CONTEXT"
    detection_mock.reason = reason_mock
    unknown_detector.evaluate.return_value = detection_mock

    rag_service = RagService(
        retrieval_service=retrieval_service,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        llm_provider=llm_provider,
        query_normalizer=query_normalizer,
        unknown_detector=unknown_detector,
    )

    rag_llm = RagLanguageModel(rag_service=rag_service)
    await rag_llm.initialize()

    conv_manager = ConversationManager()
    store_mock = AsyncMock()
    state_mock = MagicMock()
    state_mock.response_language.value = "ar"
    state_mock.comparison_state = None
    state_mock.active_entity = None
    store_mock.get_state.return_value = state_mock
    conv_manager.store = store_mock

    # Mock entity extraction for comparison (2 entities)
    gold_ent = ConversationEntity(id="visa-gold", canonical_name="Visa Gold", display_name="Visa Gold", entity_type=EntityType.CREDIT_CARD, confidence=0.95)
    plat_ent = ConversationEntity(id="visa-plat", canonical_name="Visa Platinum", display_name="Visa Platinum", entity_type=EntityType.CREDIT_CARD, confidence=0.95)

    conv_manager.entity_resolver = MagicMock()
    conv_manager.entity_resolver.extract_entities.return_value = [gold_ent, plat_ent]
    conv_manager.entity_resolver._registered_entities = True

    # Router returns COMPARISON intent
    conv_manager.router = AsyncMock()
    conv_manager.router.route.return_value = RoutingDecision(
        intent=IntentType.COMPARISON,
        confidence=0.95,
        reason="Comparison requested",
    )

    orchestrator = Orchestrator(
        audio_source=None,
        adapter=MagicMock(),
        vad=MagicMock(),
        buffer=MagicMock(),
        recognizer=MagicMock(),
        llm=rag_llm,
        tts=MagicMock(),
        conversation_manager=conv_manager,
    )

    segment_mock = MagicMock()
    transcription_obj = Transcription("Compare Visa Gold and Visa Platinum", "en", 0.0, 1.0)
    orchestrator.recognizer.transcribe = AsyncMock(return_value=transcription_obj)

    result = await orchestrator.process_speech_segment(segment_mock)

    # 1. Verify Orchestrator & RagLanguageModel returned RAG AIResponse
    assert result.response.action == "rag"

    # 2. Verify retrieve_timed was called twice (once per entity in retrieve_entity_wise)
    assert retrieval_service.retrieve_timed.call_count == 2
    calls = [c.args[0] for c in retrieval_service.retrieve_timed.call_args_list]
    assert "Visa Gold" in calls
    assert "Visa Platinum" in calls

    # 3. Verify single combined retrieve() was NOT called
    assert qdrant_mock.search.call_count == 0
