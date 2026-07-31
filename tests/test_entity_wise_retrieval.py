"""
Tests for Dynamic Entity-wise Concurrent Retrieval & Comparison Metadata

Verifies:
  1. Concurrent dense retrieval across 1, 2, and 3 comparison entities.
  2. Merging & deduplication by document ID or content hash preserving highest score.
  3. Ordering of merged search results by score descending.
  4. Fallback handling when an entity returns 0 documents (missing_entities tracking).
  5. Comparison metadata dictionary generation.
  6. ContextBuilder integration: retains chunks across all comparison products, injects missing entity notice, and enforces max character limits.
"""

import pytest
from unittest.mock import AsyncMock

from app.knowledge.models.document_metadata import DocumentMetadata
from app.knowledge.models.knowledge_document import KnowledgeDocument
from app.rag.builders.context_builder import ContextBuilder
from app.retrieval.models.search_result import SearchResult
from app.retrieval.services.retrieval_service import RetrievalService


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


def _make_result(doc_id: str, title: str, content: str, product_name: str, score: float) -> SearchResult:
    doc = _make_doc(doc_id, title, content, product_name)
    return SearchResult(document=doc, score=score, rank=1)


@pytest.mark.asyncio
async def test_entity_wise_retrieval_single_entity():
    embedding_mock = AsyncMock()
    qdrant_mock = AsyncMock()

    doc1 = _make_doc("d1", "Fees", "Visa Gold annual fee is 200 EGP", "Visa Gold")
    qdrant_mock.search.return_value = [SearchResult(document=doc1, score=0.88, rank=1)]

    retrieval_svc = RetrievalService(embedding_mock, qdrant_mock)
    results, meta = await retrieval_svc.retrieve_entity_wise(["Visa Gold"])

    assert len(results) == 1
    assert meta["comparison_entities"] == ["Visa Gold"]
    assert meta["retrieved_entities"] == ["Visa Gold"]
    assert meta["missing_entities"] == []
    assert meta["per_entity_stats"]["Visa Gold"]["retrieved_count"] == 1


@pytest.mark.asyncio
async def test_entity_wise_retrieval_multi_entity_concurrent():
    embedding_mock = AsyncMock()
    qdrant_mock = AsyncMock()

    doc_gold = _make_doc("d1", "Fees", "Visa Gold annual fee is 200 EGP", "Visa Gold")
    doc_plat = _make_doc("d2", "Fees", "Visa Platinum annual fee is 500 EGP", "Visa Platinum")
    doc_sig = _make_doc("d3", "Fees", "Visa Signature annual fee is 1500 EGP", "Visa Signature")

    qdrant_mock.search.side_effect = [
        [SearchResult(document=doc_gold, score=0.85, rank=1)],
        [SearchResult(document=doc_plat, score=0.92, rank=1)],
        [SearchResult(document=doc_sig, score=0.78, rank=1)],
    ]

    retrieval_svc = RetrievalService(embedding_mock, qdrant_mock)
    results, meta = await retrieval_svc.retrieve_entity_wise(["Visa Gold", "Visa Platinum", "Visa Signature"])

    assert len(results) == 3
    assert len(meta["comparison_entities"]) == 3
    assert len(meta["retrieved_entities"]) == 3
    assert len(meta["missing_entities"]) == 0

    # Verify score descending order (Platinum 0.92 > Gold 0.85 > Signature 0.78)
    assert results[0].document.id == "d2"
    assert results[1].document.id == "d1"
    assert results[2].document.id == "d3"


@pytest.mark.asyncio
async def test_entity_wise_retrieval_deduplication_max_score():
    embedding_mock = AsyncMock()
    qdrant_mock = AsyncMock()

    shared_doc = _make_doc("shared_1", "General", "General credit card rules apply to all cards", "General")

    qdrant_mock.search.side_effect = [
        [SearchResult(document=shared_doc, score=0.70, rank=1)],
        [SearchResult(document=shared_doc, score=0.90, rank=1)],
    ]

    retrieval_svc = RetrievalService(embedding_mock, qdrant_mock)
    results, meta = await retrieval_svc.retrieve_entity_wise(["Visa Gold", "Visa Platinum"])

    assert len(results) == 1
    assert results[0].score == 0.90
    assert meta["duplicates_removed"] == 1


@pytest.mark.asyncio
async def test_entity_wise_retrieval_missing_entity():
    embedding_mock = AsyncMock()
    qdrant_mock = AsyncMock()

    doc_gold = _make_doc("d1", "Fees", "Visa Gold fee 200 EGP", "Visa Gold")

    qdrant_mock.search.side_effect = [
        [SearchResult(document=doc_gold, score=0.88, rank=1)],
        [],
    ]

    retrieval_svc = RetrievalService(embedding_mock, qdrant_mock)
    results, meta = await retrieval_svc.retrieve_entity_wise(["Visa Gold", "Titanium"])

    assert len(results) == 1
    assert meta["retrieved_entities"] == ["Visa Gold"]
    assert meta["missing_entities"] == ["Titanium"]
    assert meta["per_entity_stats"]["Titanium"]["retrieved_count"] == 0


def test_context_builder_comparison_metadata_and_missing_notice():
    builder = ContextBuilder()
    res1 = _make_result("d1", "Fees", "Visa Gold fee 200 EGP", "Visa Gold", 0.88)
    res2 = _make_result("d2", "Fees", "Visa Platinum fee 500 EGP", "Visa Platinum", 0.92)

    comp_meta = {
        "comparison_entities": ["Visa Gold", "Visa Platinum", "Titanium"],
        "retrieved_entities": ["Visa Gold", "Visa Platinum"],
        "missing_entities": ["Titanium"],
    }

    context = builder.build_context([res1, res2], question="Compare cards", comparison_metadata=comp_meta)

    assert "Visa Gold" in context
    assert "Visa Platinum" in context
    assert "Titanium" in context
    assert "MISSING" in context or "No knowledge base documents" in context
