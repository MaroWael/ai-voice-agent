from app.knowledge.models.knowledge_document import KnowledgeDocument


class LexicalSearch:
    """
    Token-based lexical search over a list of KnowledgeDocuments.

    Algorithm:
    1. Tokenize the query by splitting on whitespace (case-insensitive).
    2. For each document, count how many query tokens appear in the
       searchable text (title + content). Both fields are lowercased before
       matching, making search case-insensitive.
    3. Return documents sorted by descending score.
    4. Documents with a score of 0 are excluded from results.

    No embeddings. No BM25. No ranking models.
    Stateless: accepts documents as input rather than owning a repository.
    """

    def search(
        self, query: str, documents: list[KnowledgeDocument]
    ) -> list[KnowledgeDocument]:
        """Score and rank documents against the query. Returns best matches first."""
        tokens = self._tokenize(query)
        if not tokens:
            return []

        scored: list[tuple[int, KnowledgeDocument]] = []
        for document in documents:
            score = self._score(document, tokens)
            if score > 0:
                scored.append((score, document))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [doc for _, doc in scored]

    def _tokenize(self, text: str) -> list[str]:
        """Split text into lowercase tokens, discarding empty strings."""
        return [token for token in text.lower().split() if token]

    def _score(self, document: KnowledgeDocument, tokens: list[str]) -> int:
        """Count how many query tokens appear in the document's searchable fields.

        title and content are searched. metadata.section is intentionally
        excluded: it is always identical to title, so including it would
        double-count title tokens and distort scores.
        """
        haystack = f"{document.title} {document.content}".lower()
        return sum(1 for token in tokens if token in haystack)
