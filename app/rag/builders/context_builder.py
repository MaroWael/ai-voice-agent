"""
RAG Context Builder

Pure synchronous formatter that converts search results into a clean, structured
block of context for the LLM.
Preserves retrieval order exactly.
"""

from app.retrieval.models.search_result import SearchResult


class ContextBuilder:
    """
    Formats retrieved search results into a structured context block.
    """

    def build_context(self, search_results: list[SearchResult]) -> str:
        """
        Formats search results into a clean, source-structured block of knowledge.

        Output format:
            Retrieved Banking Knowledge

            Source 1
            ---------
            [Product Name - Section Title]
            Content: ...

            Source 2
            ---------
            ...

        Args:
            search_results: List of SearchResult objects.

        Returns:
            A formatted string block representing the context.
        """
        if not search_results:
            return ""

        context_blocks = ["Retrieved Banking Knowledge\n"]
        for idx, result in enumerate(search_results, start=1):
            doc = result.document
            product_name = doc.metadata.product_name if doc.metadata else "Unknown Product"
            title = doc.title or "Untitled Section"
            content = doc.content.strip()

            block = (
                f"Source {idx}\n"
                f"---------\n"
                f"[{product_name} - {title}]\n"
                f"Content: {content}"
            )
            context_blocks.append(block)

        return "\n\n".join(context_blocks)
