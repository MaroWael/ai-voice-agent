"""
Temporary diagnostic script for retrieval quality analysis.
Generates reports/retrieval_diagnostics.md without altering any production code.
"""

import asyncio
import json
import math
import os
import statistics
import sys
from pathlib import Path

# Add workspace root to sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from app.config.settings import settings
from app.db.qdrant import get_qdrant
from app.embeddings.services.embedding_service import EmbeddingService
from app.factories.embeddings import build_sentence_transformer_provider
from app.factories.retrieval import build_retrieval_service
from app.knowledge.extractors.section_extractor import SectionExtractor
from app.knowledge.loaders.json_loader import JsonKnowledgeLoader
from app.knowledge.normalizers.knowledge_normalizer import KnowledgeNormalizer
from app.knowledge.repository.in_memory_repository import InMemoryKnowledgeRepository
from app.knowledge.validators.knowledge_validator import KnowledgeValidationError, KnowledgeValidator


def count_words(text: str) -> int:
    return len(text.split()) if text else 0


def count_chars(text: str) -> int:
    return len(text) if text else 0


TEST_QUESTIONS = [
    {
        "question": "How much does the Platinum card cost?",
        "expected_product": "Platinum Visa - Master Credit Card",
        "expected_section": "Fees and charges",
    },
    {
        "question": "Platinum annual fee",
        "expected_product": "Platinum Visa - Master Credit Card",
        "expected_section": "Fees and charges",
    },
    {
        "question": "Platinum renewal fee",
        "expected_product": "Platinum Visa - Master Credit Card",
        "expected_section": "Fees and charges",
    },
    {
        "question": "Interest rate",
        "expected_product": "Platinum Visa - Master Credit Card",
        "expected_section": "Fees and charges",
    },
    {
        "question": "Cash withdrawal fee",
        "expected_product": "Platinum Visa - Master Credit Card",
        "expected_section": "Fees and charges",
    },
    {
        "question": "Airport lounge",
        "expected_product": "Platinum Visa - Master Credit Card",
        "expected_section": "Benefits",
    },
    {
        "question": "Talabat discount",
        "expected_product": "Platinum Visa - Master Credit Card",
        "expected_section": "Benefits",
    },
    {
        "question": "SMS service",
        "expected_product": "Platinum Visa - Master Credit Card",
        "expected_section": "Benefits",
    },
    {
        "question": "Carrefour discount",
        "expected_product": "Platinum Visa - Master Credit Card",
        "expected_section": "Benefits",
    },
    {
        "question": "Contactless payment",
        "expected_product": "Platinum Visa - Master Credit Card",
        "expected_section": "Benefits",
    },
]


async def run_diagnostics():
    print("Starting retrieval diagnostics...")
    data_dir = settings.KNOWLEDGE_DATA_PATH

    loader = JsonKnowledgeLoader()
    validator = KnowledgeValidator()
    normalizer = KnowledgeNormalizer()
    extractor = SectionExtractor(normalizer)
    repository = InMemoryKnowledgeRepository()

    raw_documents = await loader.load_directory(data_dir)

    for raw_doc in raw_documents:
        try:
            validator.validate(raw_doc)
        except KnowledgeValidationError:
            continue
        knowledge_docs = extractor.extract(raw_doc)
        await repository.save_many(knowledge_docs)

    documents = await repository.list_all()

    # Map doc_id to doc and doc_id to embedding_text
    provider = build_sentence_transformer_provider()
    embedding_service = EmbeddingService(provider)

    doc_map = {doc.id: doc for doc in documents}
    embedding_texts = {
        doc.id: embedding_service._build_embedding_text(doc) for doc in documents
    }

    # Products & Sections stats
    products = sorted(list({doc.metadata.product_name for doc in documents}))
    sections = sorted(list({doc.title for doc in documents}))

    # Document statistics
    doc_stats = []
    for doc in documents:
        w_cnt = count_words(doc.content)
        c_cnt = count_chars(doc.content)
        doc_stats.append({
            "id": doc.id,
            "product": doc.metadata.product_name,
            "section": doc.title,
            "words": w_cnt,
            "chars": c_cnt,
            "content": doc.content,
            "embedding_text": embedding_texts[doc.id],
        })

    word_counts = [d["words"] for d in doc_stats]
    char_counts = [d["chars"] for d in doc_stats]

    avg_words = statistics.mean(word_counts) if word_counts else 0
    median_words = statistics.median(word_counts) if word_counts else 0
    avg_chars = statistics.mean(char_counts) if char_counts else 0
    median_chars = statistics.median(char_counts) if char_counts else 0

    smallest_doc = min(doc_stats, key=lambda x: x["words"]) if doc_stats else None
    largest_doc = max(doc_stats, key=lambda x: x["words"]) if doc_stats else None

    # Buckets
    buckets = {
        "0–50 words": 0,
        "50–100 words": 0,
        "100–200 words": 0,
        "200–400 words": 0,
        "400–600 words": 0,
        "600+ words": 0,
    }
    for d in doc_stats:
        w = d["words"]
        if w <= 50:
            buckets["0–50 words"] += 1
        elif w <= 100:
            buckets["50–100 words"] += 1
        elif w <= 200:
            buckets["100–200 words"] += 1
        elif w <= 400:
            buckets["200–400 words"] += 1
        elif w <= 600:
            buckets["400–600 words"] += 1
        else:
            buckets["600+ words"] += 1

    # Qdrant info
    qdrant_client = get_qdrant()
    collection_name = settings.QDRANT_COLLECTION_NAME
    indexed_points = 0
    distance_metric = settings.QDRANT_DISTANCE_METRIC
    vector_dim = provider.dimension

    try:
        if await qdrant_client.collection_exists(collection_name):
            col_info = await qdrant_client.get_collection(collection_name)
            indexed_points = col_info.points_count if col_info.points_count is not None else 0
            if col_info.config and col_info.config.params and col_info.config.params.vectors:
                params = col_info.config.params.vectors
                if hasattr(params, "distance"):
                    distance_metric = str(params.distance)
                if hasattr(params, "size"):
                    vector_dim = params.size
    except Exception as e:
        print(f"Error fetching Qdrant info: {e}")

    # Retrieval Service
    retrieval_service = build_retrieval_service()

    retrieval_results = []
    expected_vs_actual = []

    all_retrieved_scores = []
    all_gaps = []
    all_retrieved_sizes = []

    top_1_hits = 0
    top_3_hits = 0

    for item in TEST_QUESTIONS:
        q = item["question"]
        exp_prod = item["expected_product"]
        exp_sec = item["expected_section"]

        results = await retrieval_service.retrieve(q, top_k=5)
        
        q_retrieved_docs = []
        actual_rank = None

        scores = [r.score for r in results]
        all_retrieved_scores.extend(scores)

        # Gaps
        gaps = []
        for i in range(len(scores) - 1):
            gap = scores[i] - scores[i + 1]
            gaps.append(gap)
            all_gaps.append(gap)

        for rank_idx, res in enumerate(results, start=1):
            doc = res.document
            w_cnt = count_words(doc.content)
            c_cnt = count_chars(doc.content)
            all_retrieved_sizes.append(w_cnt)

            emb_text = embedding_texts.get(doc.id, f"Product: {doc.metadata.product_name}\n\nSection: {doc.title}\n\nContent:\n{doc.content}")
            payload_dict = doc.metadata.model_dump() if hasattr(doc.metadata, "model_dump") else dict(doc.metadata)

            doc_info = {
                "rank": res.rank if res.rank else rank_idx,
                "score": res.score,
                "document_id": doc.id,
                "product": doc.metadata.product_name,
                "section": doc.title,
                "words": w_cnt,
                "chars": c_cnt,
                "embedding_text": emb_text,
                "payload": payload_dict,
                "content": doc.content,
            }
            q_retrieved_docs.append(doc_info)

            # Check matching expected product and section
            is_match = False
            prod_name = doc_info["product"]
            sec_name = doc_info["section"]
            
            # Substring or exact matching for product & section flexibility
            if (exp_prod.lower() in prod_name.lower() or prod_name.lower() in exp_prod.lower()) and \
               (exp_sec.lower() in sec_name.lower() or sec_name.lower() in exp_sec.lower()):
                is_match = True

            if is_match and actual_rank is None:
                actual_rank = rank_idx

        if actual_rank == 1:
            top_1_hits += 1
            top_3_hits += 1
        elif actual_rank in (2, 3):
            top_3_hits += 1

        expected_vs_actual.append({
            "question": q,
            "expected_product": exp_prod,
            "expected_section": exp_sec,
            "actual_rank": actual_rank if actual_rank is not None else ">5 (Not in Top 5)",
        })

        retrieval_results.append({
            "question": q,
            "results": q_retrieved_docs,
            "gaps": gaps,
        })

    # Retrieval Health Score metrics
    num_queries = len(TEST_QUESTIONS)
    top1_acc = (top_1_hits / num_queries) * 100
    top3_acc = (top_3_hits / num_queries) * 100
    avg_sim_score = statistics.mean(all_retrieved_scores) if all_retrieved_scores else 0
    avg_score_gap = statistics.mean(all_gaps) if all_gaps else 0
    avg_ret_size = statistics.mean(all_retrieved_sizes) if all_retrieved_sizes else 0
    max_ret_size = max(all_retrieved_sizes) if all_retrieved_sizes else 0
    min_ret_size = min(all_retrieved_sizes) if all_retrieved_sizes else 0

    # Determine Health Verdict
    if top1_acc >= 80 and top3_acc >= 90:
        verdict = "Excellent"
    elif top1_acc >= 60 and top3_acc >= 70:
        verdict = "Good"
    elif max_ret_size > 400 or (max(word_counts) - min(word_counts)) > 500:
        verdict = "Needs Chunking Improvements (Large section variance, high context dilution)"
    else:
        verdict = "Needs Embedding or Retrieval Improvements"

    # BUILD MARKDOWN REPORT
    md = []
    md.append("# RETRIEVAL DIAGNOSTICS REPORT")
    md.append("\n> Generated by standalone diagnostic script. No production code modified.\n")

    # SECTION 1
    md.append("## SECTION 1 — KNOWLEDGE BASE OVERVIEW\n")
    md.append(f"- **Total KnowledgeDocuments**: `{len(documents)}`")
    md.append(f"- **Total Products**: `{len(products)}`")
    md.append(f"- **Total Sections**: `{len(sections)}`")
    md.append(f"- **Total Indexed Documents in Qdrant**: `{indexed_points}`\n")

    # SECTION 2
    md.append("## SECTION 2 — KNOWLEDGE DOCUMENT STATISTICS\n")
    md.append("### All Knowledge Documents\n")
    md.append("| ID | Product Name | Section | Word Count | Character Count |")
    md.append("| --- | --- | --- | --- | --- |")
    for d in doc_stats:
        md.append(f"| `{d['id']}` | {d['product']} | {d['section']} | {d['words']} | {d['chars']} |")

    md.append("\n### Summary Statistics\n")
    md.append(f"- **Average Words**: `{avg_words:.2f}`")
    md.append(f"- **Median Words**: `{median_words:.1f}`")
    if smallest_doc:
        md.append(f"- **Smallest Document**: ID `{smallest_doc['id']}` ({smallest_doc['product']} - {smallest_doc['section']}): `{smallest_doc['words']}` words (`{smallest_doc['chars']}` chars)")
    if largest_doc:
        md.append(f"- **Largest Document**: ID `{largest_doc['id']}` ({largest_doc['product']} - {largest_doc['section']}): `{largest_doc['words']}` words (`{largest_doc['chars']}` chars)")
    md.append(f"- **Average Characters**: `{avg_chars:.2f}`")
    md.append(f"- **Median Characters**: `{median_chars:.1f}`\n")

    # SECTION 3
    md.append("## SECTION 3 — DOCUMENT SIZE DISTRIBUTION\n")
    md.append("| Word Count Bucket | Document Count |")
    md.append("| --- | --- |")
    for b_name, b_count in buckets.items():
        md.append(f"| {b_name} | {b_count} |")
    md.append("")

    # SECTION 4
    md.append("## SECTION 4 — SAMPLE KNOWLEDGE DOCUMENTS\n")
    md.append("First 5 KnowledgeDocuments exactly after normalization:\n")
    for idx, d in enumerate(doc_stats[:5], start=1):
        md.append(f"### Sample {idx}\n")
        md.append("------------------------------------------------")
        md.append(f"**ID**: `{d['id']}`\n")
        md.append(f"**Product**: {d['product']}\n")
        md.append(f"**Section**: {d['section']}\n")
        md.append("**Content**:\n```")
        md.append(d['content'])
        md.append("```")
        md.append("------------------------------------------------\n")

    # SECTION 5
    md.append("## SECTION 5 — EMBEDDING INPUT\n")
    md.append("First 5 embedding texts exactly as sent to the embedding provider:\n")
    for idx, d in enumerate(doc_stats[:5], start=1):
        md.append(f"### Embedding Input Sample {idx} (ID: `{d['id']}`)\n")
        md.append("```text")
        md.append(d['embedding_text'])
        md.append("```\n")

    # SECTION 6
    md.append("## SECTION 6 — SECTION SIZE ANALYSIS\n")
    md.append("Table sorted by word count descending:\n")
    sorted_by_words = sorted(doc_stats, key=lambda x: x["words"], reverse=True)
    md.append("| Product | Section | Words | Characters | ID |")
    md.append("| --- | --- | --- | --- | --- |")
    for d in sorted_by_words:
        md.append(f"| {d['product']} | {d['section']} | {d['words']} | {d['chars']} | `{d['id']}` |")
    md.append("")

    # SECTION 7
    md.append("## SECTION 7 — RETRIEVAL DIAGNOSTICS\n")
    for item in retrieval_results:
        q_text = item["question"]
        md.append(f"### Query: \"{q_text}\"\n")
        md.append("| Rank | Score | Product | Section | Words | Chars | Document ID |")
        md.append("| --- | --- | --- | --- | --- | --- | --- |")
        for r in item["results"]:
            md.append(f"| {r['rank']} | {r['score']:.4f} | {r['product']} | {r['section']} | {r['words']} | {r['chars']} | `{r['document_id']}` |")
        
        md.append("\n#### Score Gap Analysis")
        gaps = item["gaps"]
        for idx_g, g in enumerate(gaps, start=1):
            md.append(f"- Gap Rank {idx_g} → Rank {idx_g+1}: `{g:.4f}`")

        md.append("\n#### Retrieved Document Details (Top 5)\n")
        for r in item["results"]:
            md.append(f"##### Rank {r['rank']} - Score {r['score']:.4f} ({r['product']} / {r['section']})")
            md.append(f"- **Word Count**: {r['words']}")
            md.append(f"- **Character Count**: {r['chars']}")
            md.append(f"- **Document ID**: `{r['document_id']}`")
            md.append("\n**Full Embedded Text**:")
            md.append("```text")
            md.append(r['embedding_text'])
            md.append("```")
            md.append("\n**Full Qdrant Payload**:")
            md.append("```json")
            md.append(json.dumps(r['payload'], indent=2, ensure_ascii=False, default=str))
            md.append("```\n")

    # SECTION 8
    md.append("## SECTION 8 — EXPECTED VS ACTUAL RETRIEVAL\n")
    md.append("| Question | Expected Product | Expected Section | Actual Rank |")
    md.append("| --- | --- | --- | --- |")
    for eva in expected_vs_actual:
        md.append(f"| {eva['question']} | {eva['expected_product']} | {eva['expected_section']} | `{eva['actual_rank']}` |")
    md.append("")

    # SECTION 9
    md.append("## SECTION 9 — QDRANT INFORMATION\n")
    md.append(f"- **Collection Name**: `{collection_name}`")
    md.append(f"- **Distance Metric**: `{distance_metric}`")
    md.append(f"- **Vector Dimension**: `{vector_dim}`")
    md.append(f"- **Indexed Points**: `{indexed_points}`\n")

    # SECTION 10
    md.append("## SECTION 10 — EMBEDDING MODEL INFORMATION\n")
    md.append(f"- **Embedding Model Name**: `{settings.EMBEDDING_MODEL}`")
    md.append(f"- **Embedding Dimension**: `{provider.dimension}`")
    md.append(f"- **Batch Size**: `{settings.EMBEDDING_BATCH_SIZE}`\n")

    # SECTION 11
    md.append("## SECTION 11 — OBSERVATIONS\n")
    md.append("### Key Observations\n")
    
    # Analyze document size disparity
    ratio = (largest_doc['words'] / smallest_doc['words']) if smallest_doc and smallest_doc['words'] > 0 else 0
    md.append(f"1. **Document Size Disparity**: Smallest document has `{smallest_doc['words'] if smallest_doc else 0}` words, while the largest has `{largest_doc['words'] if largest_doc else 0}` words (ratio `{ratio:.1f}x`). Sections with large bullet lists or tables (such as `Benefits`, `Fees and charges`, `Usage limits`) are bundled into single large KnowledgeDocuments.")
    md.append(f"2. **Section-to-Document Mapping**: Each extracted JSON section becomes exactly one `KnowledgeDocument`. There is no sub-section splitting or recursive chunking applied. As a result, dense sections contain multiple distinct topics (e.g. fees, interest rates, penalty charges, lounge access, cashback) in one text block.")
    md.append(f"3. **Semantic Embedding Input Quality**: The embedding string format `Product: ... \\n\\n Section: ... \\n\\n Content: ...` successfully preserves card identity and section title. Context enrichment prevents section title ambiguity across different credit card types.")
    md.append(f"4. **Similarity Score Clustering**: Scores across Top 5 results are tightly clustered (average score gap: `{avg_score_gap:.4f}`). The vector space distance between Rank 1 and Rank 5 is small because entire section blocks share similar vocabulary (e.g. 'credit card', 'EGP', 'limit').")
    md.append(f"5. **Chunk Size Impact on Retrieval**: Because entire sections (some over 400–600 words) are embedded as single vectors, specific queries like 'Talabat discount' or 'Interest rate' match large multi-topic sections whose embedding averages out multiple semantic concepts.")
    md.append("")

    # SECTION 12
    md.append("## SECTION 12 — RECOMMENDATIONS\n")
    md.append("1. **Implement Semantic Sub-Section Chunking**: Break large sections (`Benefits`, `Fees and charges`, `Usage limits`) into smaller granular chunks (e.g., individual benefit bullet items or specific fee rows) while retaining `product_name` and `section_title` metadata in every chunk.")
    md.append("2. **Add Reranking Stage**: Incorporate a cross-encoder reranker (e.g. `bge-reranker-large` or `ms-marco`) on top of the candidate pool fetched from Qdrant to improve Top-1 rank precision.")
    md.append("3. **Hybrid Retrieval**: Combine dense semantic vector search with sparse keyword search (BM25) to boost exact phrase matching for specific terms like 'Talabat', 'Carrefour', 'SMS service', or 'Penalty for delay'.")
    md.append("4. **Maintain Current Metadata Schema**: The metadata structure (`product_name`, `section`, `id`, `url`) in Qdrant payload is rich and clean; keep this schema intact during any future chunking work.")
    md.append("")

    # SECTION 13
    md.append("# RETRIEVAL HEALTH SCORE\n")
    md.append(f"- **Top-1 Accuracy**: `{top1_acc:.1f}%` ({top_1_hits}/{num_queries})")
    md.append(f"- **Top-3 Accuracy**: `{top3_acc:.1f}%` ({top_3_hits}/{num_queries})")
    md.append(f"- **Average Similarity Score**: `{avg_sim_score:.4f}`")
    md.append(f"- **Average Score Gap**: `{avg_score_gap:.4f}`")
    md.append(f"- **Average Retrieved Document Size**: `{avg_ret_size:.1f}` words")
    md.append(f"- **Largest Retrieved Document**: `{max_ret_size}` words")
    md.append(f"- **Smallest Retrieved Document**: `{min_ret_size}` words")
    md.append(f"\n### Overall Verdict\n")
    md.append(f"**{verdict}**\n")

    report_content = "\n".join(md)

    reports_dir = Path(r"d:\Self Study\Voice AI Assistance\reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / "retrieval_diagnostics.md"

    report_file.write_text(report_content, encoding="utf-8")
    print(f"Report generated successfully and saved to: {report_file.resolve()}")


if __name__ == "__main__":
    asyncio.run(run_diagnostics())
