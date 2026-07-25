import re
from collections import defaultdict
from typing import Sequence

from app.config.settings import settings
from app.retrieval.models.search_result import SearchResult


def _is_comparison_query(query: str | None) -> bool:
    """Return True if the query explicitly asks to compare variants (e.g. Visa vs Mastercard)."""
    if not query:
        return False
    q_lower = query.lower()
    compare_keywords = ["الفرق", "مقارنة", "اختلاف", "compare", "difference", "versus", "vs"]
    if any(kw in q_lower for kw in compare_keywords):
        return True

    has_visa = "visa" in q_lower or "فيزا" in q_lower
    has_mastercard = "mastercard" in q_lower or "ماستركارد" in q_lower
    if has_visa and has_mastercard:
        return True

    return False


def _parse_base_section_title(title: str) -> tuple[str, str | None]:
    """Extract base section title and variant name if present in title (e.g. 'Fees and charges (Visa)')."""
    match = re.search(r"^(.*?)\s*\((.*?)\)$", title.strip())
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return title.strip(), None


class ContextBuilder:
    """
    Formats retrieved search results into a structured context block.
    Uses section-level grouping, section ranking, context filtering, and standardized formatting.
    """

    def build_context(
        self,
        search_results: Sequence[SearchResult],
        question: str | None = None,
        max_chars: int | None = None,
        section_score_margin: float = 0.05,
    ) -> str:
        """
        Formats search results into a clean, section-structured block of knowledge.

        Args:
            search_results: Sequence of SearchResult objects ordered by score descending.
            question: Optional user query for intent-based variant comparison detection.
            max_chars: Optional character length budget limit. Defaults to settings.RAG_MAX_CONTEXT_CHARS.
            section_score_margin: Margin below top section score to include secondary sections (default 0.05).

        Returns:
            A formatted context string block.
        """
        if not search_results:
            return ""

        limit = max_chars if max_chars is not None else settings.RAG_MAX_CONTEXT_CHARS

        # Step 1: Deduplicate search results by ID & content
        seen_ids: set[str] = set()
        seen_texts: set[str] = set()
        unique_results: list[SearchResult] = []

        for result in search_results:
            doc = result.document
            doc_id = doc.id
            content_key = doc.content.strip().lower()

            if doc_id in seen_ids or content_key in seen_texts:
                continue

            seen_ids.add(doc_id)
            seen_texts.add(content_key)
            unique_results.append(result)

        if not unique_results:
            return ""

        # Step 2: Group chunks by Section (Product, Section Title)
        section_chunks: dict[tuple[str, str], list[SearchResult]] = defaultdict(list)
        section_scores: dict[tuple[str, str], float] = {}

        for result in unique_results:
            doc = result.document
            product_name = doc.metadata.product_name if doc.metadata else "Unknown Product"
            section_title = doc.title or (doc.metadata.section if doc.metadata else "General Info")
            key = (product_name, section_title)

            section_chunks[key].append(result)
            if key not in section_scores or result.score > section_scores[key]:
                section_scores[key] = result.score

        # Step 3: Rank Sections by representative score (max vector score per section)
        ranked_sections = sorted(section_scores.keys(), key=lambda k: section_scores[k], reverse=True)

        if not ranked_sections:
            return ""

        top_section = ranked_sections[0]
        top_product, top_full_title = top_section
        top_base_title, top_variant = _parse_base_section_title(top_full_title)
        top_score = section_scores[top_section]

        is_comparison = _is_comparison_query(question)

        # Step 4: Context filtering (deduplicate variants and filter unrelated topics)
        selected_sections: list[tuple[str, str]] = []

        if is_comparison:
            # Allow multiple variant sections if customer explicitly requested comparison
            selected_sections = [
                sec for sec in ranked_sections
                if (top_score - section_scores[sec]) <= 0.10
            ]
        else:
            # Non-comparison: Keep top product relevant section(s), preferring highest-scoring variant
            seen_base_topics: set[str] = set()

            for sec in ranked_sections:
                p_name, s_title = sec
                b_title, var = _parse_base_section_title(s_title)
                score = section_scores[sec]

                # Filter out unrelated products if top score is strong
                if p_name != top_product:
                    continue

                if (top_score - score) > section_score_margin:
                    continue

                # Ensure section topic matches top base topic
                if b_title != top_base_title and len(selected_sections) > 0:
                    continue

                # Deduplicate variant sections for the same base topic (e.g. Visa vs Mastercard)
                if b_title in seen_base_topics:
                    continue

                seen_base_topics.add(b_title)
                selected_sections.append(sec)

        if not selected_sections:
            selected_sections = [top_section]

        # Step 5: Preserve ALL chunks belonging to selected section(s)
        selected_chunks: list[SearchResult] = []
        for sec in selected_sections:
            selected_chunks.extend(section_chunks[sec])

        # Step 6: Format context consistently
        context_blocks = ["Retrieved Banking Knowledge\n"]
        current_len = len(context_blocks[0])

        for result in selected_chunks:
            doc = result.document
            product_name = doc.metadata.product_name if doc.metadata else "Unknown Product"
            section_title = doc.title or (doc.metadata.section if doc.metadata else "General Info")
            content = doc.content.strip()

            block = (
                f"Product: {product_name}\n"
                f"Section: {section_title}\n"
                f"Content:\n{content}"
            )

            # Enforce max context character budget
            if limit > 0 and (current_len + len(block) + 2 > limit):
                break

            context_blocks.append(block)
            current_len += len(block) + 2

        return "\n\n".join(context_blocks)

