from app.retrieval.models.search_result import SearchResult


class ContextBuilder:
    """
    Pure synchronous formatter that converts search results into a single context block.
    Preserves retrieval order exactly. Contains no filtering, truncation, or logic.
    """

    def build_context(self, search_results: list[SearchResult]) -> str:
        """
        Formats search results into a clean, text-based block of context.

        Args:
            search_results: List of SearchResult objects.

        Returns:
            A formatted string block representing the context.
        """
        if not search_results:
            return ""

        context_blocks = []
        for idx, result in enumerate(search_results, start=1):
            doc = result.document
            product_name = doc.metadata.product_name if doc.metadata else "Unknown Product"
            title = doc.title or "Untitled Section"
            content = doc.content.strip()

            block = (
                f"[Document {idx}: {product_name} - {title}]\n"
                f"Content: {content}"
            )
            context_blocks.append(block)

        return "\n\n".join(context_blocks)
