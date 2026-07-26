import pytest
from app.factories.rag import build_rag_service
from app.rag.models.status import RagStatus
from app.startup.knowledge_initializer import initialize_knowledge_base

pytestmark = pytest.mark.asyncio


async def _ensure_kb():
    await initialize_knowledge_base()


async def test_arabic_dialect_recovery():
    """Verify that dialectal Arabic queries ('ايه هي رسوم الفيزا الجولد') recover successfully."""
    await _ensure_kb()
    rag_service = build_rag_service()
    await rag_service.initialize()
    try:
        response = await rag_service.answer("ايه هي رسوم الفيزا الجولد", debug=True)
        assert response.status == RagStatus.SUCCESS, f"Expected SUCCESS, got {response.status}"
        assert len(response.retrieved_documents) > 0
        top_doc = response.retrieved_documents[0]
        assert "Gold" in (top_doc.document.metadata.product_name if top_doc.document.metadata else "")
    finally:
        await rag_service.close()


async def test_arabic_conversational_recovery():
    """Verify conversational Arabic queries ('ممكن اعرف مصاريف بطاقة البلاتينيوم') succeed."""
    await _ensure_kb()
    rag_service = build_rag_service()
    await rag_service.initialize()
    try:
        response = await rag_service.answer("ممكن اعرف مصاريف بطاقة البلاتينيوم", debug=True)
        assert response.status == RagStatus.SUCCESS, f"Expected SUCCESS, got {response.status}"
        assert len(response.retrieved_documents) > 0
        top_doc = response.retrieved_documents[0]
        assert "Platinum" in (top_doc.document.metadata.product_name if top_doc.document.metadata else "")
    finally:
        await rag_service.close()


async def test_english_query():
    """Verify standard English query ('What are gold card fees') succeeds directly."""
    await _ensure_kb()
    rag_service = build_rag_service()
    await rag_service.initialize()
    try:
        response = await rag_service.answer("What are gold card fees", debug=True)
        assert response.status == RagStatus.SUCCESS, f"Expected SUCCESS, got {response.status}"
        assert len(response.retrieved_documents) > 0
    finally:
        await rag_service.close()


async def test_out_of_scope_query():
    """Verify out of scope queries ('What is the personal loan interest rate') return INSUFFICIENT_CONTEXT."""
    await _ensure_kb()
    rag_service = build_rag_service()
    await rag_service.initialize()
    try:
        response = await rag_service.answer("What is the personal loan interest rate", debug=True)
        assert response.status == RagStatus.INSUFFICIENT_CONTEXT, f"Expected INSUFFICIENT_CONTEXT, got {response.status}"
    finally:
        await rag_service.close()
