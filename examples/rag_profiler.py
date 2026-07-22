"""
Voice AI RAG Profiler — Developer Inspection Tool

Measures execution time (in milliseconds), stage percentages, and detailed LLM performance
metrics for every stage of the RAG quality pipeline for a single user query.

Workflow:
    User Question
         ↓ [Query Optimization]
    Optimized Query
         ↓ [Embedding Generation + Vector Search]
    Retrieved Documents
         ↓ [Unknown Answer Detector]
    Detection Result (has_context)
         ├─ True  → [Context Builder] → [Prompt Builder] → [LLM Performance & Generation]
         └─ False → Skip Context/Prompt/LLM → Fallback Reply

Run with:
    python examples/rag_profiler.py
"""

import asyncio
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.config.settings import settings
from app.db.qdrant import get_qdrant
from app.factories.retrieval import build_retrieval_service
from app.query_optimization.factory import build_query_optimizer
from app.rag.builders.context_builder import ContextBuilder
from app.rag.builders.prompt_builder import PromptBuilder
from app.rag.providers.ollama_provider import OllamaRagProvider
from app.rag.services.rag_service import _INSUFFICIENT_CONTEXT_MSG
from app.unknown_detection.factory import build_unknown_detector


async def _knowledge_base_ready() -> bool:
    """Return True if Qdrant collection exists and is non-empty."""
    client = get_qdrant()
    try:
        if not await client.collection_exists(settings.QDRANT_COLLECTION_NAME):
            return False
        info = await client.get_collection(settings.QDRANT_COLLECTION_NAME)
        return bool(info and info.points_count)
    except Exception:
        return False


def _load_prompt_template() -> str:
    """Load the standard RAG prompt template."""
    template_path = PROJECT_ROOT / "app" / "rag" / "prompts" / "default_rag.txt"
    try:
        return template_path.read_text(encoding="utf-8")
    except Exception as exc:
        raise RuntimeError(f"Failed to load prompt template from {template_path}: {exc}") from exc


async def profile_query(
    query: str,
    optimizer,
    retrieval_service,
    unknown_detector,
    context_builder: ContextBuilder,
    prompt_builder: PromptBuilder,
    llm_provider: OllamaRagProvider,
) -> None:
    """Execute the pipeline step-by-step for *query* and print a detailed stage & performance report."""
    pipeline_t0 = time.perf_counter()

    # ---------------------------------------------------------
    # Stage 1: Query Optimization
    # ---------------------------------------------------------
    t0 = time.perf_counter()
    optimized_query = await optimizer.optimize(query)
    query_opt_time = (time.perf_counter() - t0) * 1000.0

    retrieval_query = optimized_query if optimized_query else query

    print("\n---------------------------------------------------------")
    print("Query")
    print("---------------------------------------------------------")
    print("Original Query:")
    print(query)
    print("\nOptimized Query:")
    print(retrieval_query)
    print(f"\nTime:\n{query_opt_time:.1f} ms")

    # ---------------------------------------------------------
    # Stage 2: Retrieval (Embedding Generation + Vector Search split)
    # ---------------------------------------------------------
    retrieved_docs, timing = await retrieval_service.retrieve_timed(retrieval_query, top_k=5)
    embed_gen_time = timing.embedding_time * 1000.0
    vector_search_time = timing.search_time * 1000.0
    total_retrieval_time = embed_gen_time + vector_search_time

    print("\n---------------------------------------------------------")
    print("Retrieval")
    print("---------------------------------------------------------")
    print(f"\nEmbedding Generation:\n{embed_gen_time:.1f} ms")
    print(f"\nVector Search:\n{vector_search_time:.1f} ms")
    print(f"\nTotal Retrieval Time:\n{total_retrieval_time:.1f} ms")
    print(f"\nRetrieved Documents: {len(retrieved_docs)}")
    print("\nScores:")
    if retrieved_docs:
        for doc_res in retrieved_docs:
            print(f"{doc_res.score:.2f}")
    else:
        print("None")

    # ---------------------------------------------------------
    # Stage 3: Unknown Detection
    # ---------------------------------------------------------
    t0 = time.perf_counter()
    detection = await unknown_detector.evaluate(query, retrieved_docs)
    detection_time = (time.perf_counter() - t0) * 1000.0

    print("\n---------------------------------------------------------")
    print("Unknown Detection")
    print("---------------------------------------------------------")
    print(f"\nHas Context:\n{detection.has_context}")
    print(f"\nReason:\n{detection.reason.name}")
    print(f"\nTop Score:\n{detection.top_score:.2f}")
    print(f"\nAverage Score:\n{detection.average_score:.2f}")
    print(f"\nTime:\n{detection_time:.1f} ms")

    # ---------------------------------------------------------
    # Early Exit if INSUFFICIENT_CONTEXT
    # ---------------------------------------------------------
    if not detection.has_context:
        pipeline_total_time = (time.perf_counter() - pipeline_t0) * 1000.0

        print("\n---------------------------------------------------------")
        print("Response (Early Exit)")
        print("---------------------------------------------------------")
        print("Answer:\n" + _INSUFFICIENT_CONTEXT_MSG)

        print("\n=========================================================")
        print("PIPELINE BREAKDOWN")
        print("=========================================================\n")
        print(f"Query Optimization    : {query_opt_time:.1f} ms ({(query_opt_time/pipeline_total_time)*100:.1f}%)")
        print(f"\nEmbedding Generation  : {embed_gen_time:.1f} ms ({(embed_gen_time/pipeline_total_time)*100:.1f}%)")
        print(f"\nVector Search         : {vector_search_time:.1f} ms ({(vector_search_time/pipeline_total_time)*100:.1f}%)")
        print(f"\nUnknown Detection     : {detection_time:.1f} ms ({(detection_time/pipeline_total_time)*100:.1f}%)")
        print("\nContext Builder       : Skipped")
        print("\nPrompt Builder        : Skipped")
        print("\nLLM Prompt Processing : Skipped")
        print("\nLLM Generation        : Skipped")
        print("\n---------------------------------------------------------")
        print(f"\nTOTAL                 : {pipeline_total_time:.1f} ms (100.0%)")
        print("\n=========================================================")
        return

    # ---------------------------------------------------------
    # Stage 4: Context Builder
    # ---------------------------------------------------------
    t0 = time.perf_counter()
    context = context_builder.build_context(retrieved_docs)
    context_time = (time.perf_counter() - t0) * 1000.0

    print("\n---------------------------------------------------------")
    print("Context Builder")
    print("---------------------------------------------------------")
    print(f"\nContext Chunks:\n{len(retrieved_docs)}")
    print(f"\nContext Length:\n{len(context)} chars")
    print(f"\nTime:\n{context_time:.1f} ms")

    # ---------------------------------------------------------
    # Stage 5: Prompt Builder
    # ---------------------------------------------------------
    t0 = time.perf_counter()
    prompt = prompt_builder.build_prompt(query, context)
    prompt_time = (time.perf_counter() - t0) * 1000.0

    approx_prompt_tokens = len(prompt.split())

    print("\n---------------------------------------------------------")
    print("Prompt Builder")
    print("---------------------------------------------------------")
    print(f"\nPrompt Tokens (approx):\n{approx_prompt_tokens}")
    print(f"\nPrompt Length:\n{len(prompt)} chars")
    print(f"\nTime:\n{prompt_time:.1f} ms")

    # ---------------------------------------------------------
    # Stage 6: LLM Provider Execution & Detailed Performance
    # ---------------------------------------------------------
    t0 = time.perf_counter()
    if hasattr(llm_provider, "generate_with_metadata"):
        answer, metadata = await llm_provider.generate_with_metadata(prompt)
    else:
        answer = await llm_provider.generate(prompt)
        metadata = {}
    wall_llm_time_ms = (time.perf_counter() - t0) * 1000.0

    # Extract Ollama metadata metrics (duration is in nanoseconds: 1ms = 1e6 ns)
    prompt_eval_count = metadata.get("prompt_eval_count", approx_prompt_tokens)
    prompt_eval_duration_ns = metadata.get("prompt_eval_duration", 0)
    eval_count = metadata.get("eval_count", len(answer.split()))
    eval_duration_ns = metadata.get("eval_duration", 0)
    total_duration_ns = metadata.get("total_duration", 0)

    if prompt_eval_duration_ns > 0:
        llm_prompt_proc_ms = prompt_eval_duration_ns / 1e6
    else:
        llm_prompt_proc_ms = 0.0

    if eval_duration_ns > 0:
        llm_gen_ms = eval_duration_ns / 1e6
    else:
        llm_gen_ms = wall_llm_time_ms - llm_prompt_proc_ms

    if total_duration_ns > 0:
        total_llm_time_ms = total_duration_ns / 1e6
    else:
        total_llm_time_ms = wall_llm_time_ms

    prompt_speed = (prompt_eval_count / (llm_prompt_proc_ms / 1000.0)) if llm_prompt_proc_ms > 0 else 0.0
    gen_speed = (eval_count / (llm_gen_ms / 1000.0)) if llm_gen_ms > 0 else 0.0

    print("\n---------------------------------------------------------")
    print("LLM PERFORMANCE")
    print("---------------------------------------------------------")
    print(f"\nPrompt Length:\n{len(prompt)} chars")
    print(f"\nPrompt Tokens:\n{prompt_eval_count}")
    print(f"\nAnswer Tokens:\n{eval_count}")
    print(f"\nPrompt Processing:\n{llm_prompt_proc_ms:.1f} ms")
    print(f"\nGeneration:\n{llm_gen_ms:.1f} ms")
    print(f"\nPrompt Speed:\n{prompt_speed:.1f} tokens/sec")
    print(f"\nGeneration Speed:\n{gen_speed:.1f} tokens/sec")
    print(f"\nTotal LLM Time:\n{total_llm_time_ms:.1f} ms")
    print("---------------------------------------------------------")

    print("\nAnswer:\n")
    print(answer)

    # ---------------------------------------------------------
    # Pipeline Breakdown & Summary
    # ---------------------------------------------------------
    pipeline_total_time = (time.perf_counter() - pipeline_t0) * 1000.0

    print("\n=========================================================")
    print("PIPELINE BREAKDOWN")
    print("=========================================================\n")
    print(f"Query Optimization    : {query_opt_time:.1f} ms ({(query_opt_time/pipeline_total_time)*100:.1f}%)")
    print(f"\nEmbedding Generation  : {embed_gen_time:.1f} ms ({(embed_gen_time/pipeline_total_time)*100:.1f}%)")
    print(f"\nVector Search         : {vector_search_time:.1f} ms ({(vector_search_time/pipeline_total_time)*100:.1f}%)")
    print(f"\nUnknown Detection     : {detection_time:.1f} ms ({(detection_time/pipeline_total_time)*100:.1f}%)")
    print(f"\nContext Builder       : {context_time:.1f} ms ({(context_time/pipeline_total_time)*100:.1f}%)")
    print(f"\nPrompt Builder        : {prompt_time:.1f} ms ({(prompt_time/pipeline_total_time)*100:.1f}%)")
    print(f"\nLLM Prompt Processing : {llm_prompt_proc_ms:.1f} ms ({(llm_prompt_proc_ms/pipeline_total_time)*100:.1f}%)")
    print(f"\nLLM Generation        : {llm_gen_ms:.1f} ms ({(llm_gen_ms/pipeline_total_time)*100:.1f}%)")
    print("\n---------------------------------------------------------")
    print(f"\nTOTAL                 : {pipeline_total_time:.1f} ms (100.0%)")
    print("\n=========================================================")


async def main() -> None:
    if not await _knowledge_base_ready():
        print("❌ Knowledge base has not been initialized.")
        print("\nRun first:\n    python initialize_knowledge_base.py\n")
        return

    # Instantiate reusable components
    optimizer = build_query_optimizer()
    retrieval_service = build_retrieval_service()
    unknown_detector = build_unknown_detector()
    context_builder = ContextBuilder()
    prompt_template = _load_prompt_template()
    prompt_builder = PromptBuilder(prompt_template)
    llm_provider = OllamaRagProvider()

    await llm_provider.initialize()

    print("=========================================================")
    print("Voice AI RAG Profiler")
    print("=========================================================")

    try:
        while True:
            print("\nAsk a question (empty = exit):")
            user_input = await asyncio.to_thread(input, "\n> ")
            query = user_input.strip()
            if not query:
                print("Exiting profiler.")
                break

            await profile_query(
                query=query,
                optimizer=optimizer,
                retrieval_service=retrieval_service,
                unknown_detector=unknown_detector,
                context_builder=context_builder,
                prompt_builder=prompt_builder,
                llm_provider=llm_provider,
            )
    finally:
        await llm_provider.close()


if __name__ == "__main__":
    asyncio.run(main())
