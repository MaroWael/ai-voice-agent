import asyncio
import json
import sys
from pathlib import Path

# Add project root directory to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

from app.config.settings import settings
from app.factories.retrieval import build_retrieval_service
from app.factories.llm import build_llm_provider
from app.query_optimization.factory import build_query_normalizer, build_query_enhancer
from app.startup.knowledge_initializer import initialize_knowledge_base


async def run_benchmark():
    print("Initializing Knowledge Base if needed...")
    await initialize_knowledge_base()

    normalizer = build_query_normalizer()
    llm = build_llm_provider()
    await llm.initialize()
    enhancer = build_query_enhancer(llm)
    retriever = build_retrieval_service()

    queries = [
        # English
        "Gold Credit Card fees charges",
        "Gold card fees",
        "Gold Credit Cards Fees and charges",
        # Arabic
        "ايه رسوم الفيزا الجولد",
        "رسوم الفيزا الجولد",
        "مصاريف بطاقة الجولد",
        # Mixed
        "gold card fees",
        "رسوم gold card",
        # Original issue case
        "ايه هي رسوم الفيزا الجولد",
    ]

    print("\n" + "=" * 90)
    print("PART 1: RETRIEVAL BENCHMARK ON TEST QUERIES")
    print("=" * 90)

    for q in queries:
        normalized = await normalizer.normalize(q)
        results = await retriever.retrieve(normalized if normalized else q, top_k=5)

        scores = [r.score for r in results]
        top_score = scores[0] if scores else 0.0
        mean_score = sum(scores) / len(scores) if scores else 0.0

        print(f"\nOriginal Query:   {q}")
        print(f"Normalized Query: {normalized}")
        print(f"Top Score:        {top_score:.4f}")
        print(f"Mean Score:       {mean_score:.4f}")
        print("Top 5 Retrieved Documents:")
        for rank, r in enumerate(results, 1):
            prod = r.document.metadata.product_name if r.document.metadata else "N/A"
            title = r.document.title or "N/A"
            doc_id = r.document.id
            print(f"  [{rank}] Score: {r.score:.4f} | Product: {prod} | Section: {title} | ID: {doc_id}")

    print("\n" + "=" * 90)
    print("PART 2: RAW ARABIC VS IDEAL SEARCH QUERY COMPARISON")
    print("=" * 90)

    raw_query = "ايه هي رسوم الفيزا الجولد"
    ideal_query = "Gold Credit Card fees charges"

    norm_raw = await normalizer.normalize(raw_query)
    norm_ideal = await normalizer.normalize(ideal_query)

    res_raw = await retriever.retrieve(norm_raw, top_k=5)
    res_ideal = await retriever.retrieve(norm_ideal, top_k=5)

    raw_scores = [r.score for r in res_raw]
    ideal_scores = [r.score for r in res_ideal]

    raw_top = raw_scores[0] if raw_scores else 0.0
    ideal_top = ideal_scores[0] if ideal_scores else 0.0
    raw_mean = sum(raw_scores)/len(raw_scores) if raw_scores else 0.0
    ideal_mean = sum(ideal_scores)/len(ideal_scores) if ideal_scores else 0.0

    print(f"Raw Arabic Query:   '{raw_query}' (Normalized: '{norm_raw}')")
    print(f"  Top Score: {raw_top:.4f} | Mean Score: {raw_mean:.4f}")
    if res_raw:
        print(f"  Top Match: Product='{res_raw[0].document.metadata.product_name}' | Section='{res_raw[0].document.title}'")

    print(f"\nIdeal Search Query: '{ideal_query}' (Normalized: '{norm_ideal}')")
    print(f"  Top Score: {ideal_top:.4f} | Mean Score: {ideal_mean:.4f}")
    if res_ideal:
        print(f"  Top Match: Product='{res_ideal[0].document.metadata.product_name}' | Section='{res_ideal[0].document.title}'")

    print(f"\nScore Delta (Ideal - Raw): Top Score Delta = {ideal_top - raw_top:+.4f} | Mean Score Delta = {ideal_mean - raw_mean:+.4f}")

    print("\n" + "=" * 90)
    print("PART 3: ENHANCER BEHAVIOR TEST")
    print("=" * 90)
    for q in ["ايه هي رسوم الفيزا الجولد", "ايه رسوم الفيزا الجولد", "مصاريف بطاقة الجولد"]:
        try:
            enh = await enhancer.enhance(q)
            print(f"Query: '{q}' -> Enhanced: '{enh}'")
            if enh != q:
                res_enh = await retriever.retrieve(enh, top_k=5)
                enh_scores = [r.score for r in res_enh]
                enh_top = enh_scores[0] if enh_scores else 0.0
                enh_mean = sum(enh_scores)/len(enh_scores) if enh_scores else 0.0
                print(f"  Enhanced Retrieval -> Top Score: {enh_top:.4f} | Mean Score: {enh_mean:.4f}")
                if res_enh:
                    print(f"  Enhanced Top Match: Product='{res_enh[0].document.metadata.product_name}' | Section='{res_enh[0].document.title}'")
        except Exception as e:
            print(f"Enhancer failed for '{q}': {e}")

    await llm.close()

if __name__ == "__main__":
    asyncio.run(run_benchmark())
