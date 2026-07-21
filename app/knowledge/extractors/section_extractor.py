import re

from app.knowledge.models.document_metadata import DocumentMetadata
from app.knowledge.models.knowledge_document import KnowledgeDocument
from app.knowledge.models.raw_document import RawDocument, RawSection
from app.knowledge.normalizers.knowledge_normalizer import KnowledgeNormalizer


def _slugify(text: str) -> str:
    """Convert a display name into a lowercase URL-safe slug.

    Example: "Al-Araby Card" → "al-araby-card"
    """
    text = text.lower()
    # Remove characters that are not alphanumeric, whitespace, or hyphens.
    text = re.sub(r"[^\w\s-]", "", text)
    # Collapse whitespace and underscores into hyphens.
    text = re.sub(r"[\s_]+", "-", text)
    return text.strip("-")


def _assemble_raw_content(section: RawSection) -> str:
    """
    Assemble all parts of a section into raw text before normalization.

    Preserves the original formatting so that the pre-normalization snapshot
    stored in KnowledgeDocument.raw_content is faithful to the source.
    """
    parts: list[str] = []

    if section.content.strip():
        parts.append(section.content.strip())

    if section.bullets:
        bullet_lines = [f"• {b.strip()}" for b in section.bullets if b.strip()]
        if bullet_lines:
            parts.append("\n".join(bullet_lines))

    for table in section.tables:
        if not table:
            continue
        # Represent each table row as pipe-separated cells.
        table_lines = [" | ".join(str(cell) for cell in row) for row in table]
        parts.append("\n".join(table_lines))

    return "\n\n".join(parts)


class SectionExtractor:
    """
    Converts a single RawDocument into a list of KnowledgeDocuments.

    Rules:
    - Exactly one KnowledgeDocument is produced per section.
    - No token-based chunking.
    - Section titles are preserved.
    - Both raw and normalized content are stored.
    - Deterministic IDs: slugified product name + zero-padded section index.
    """

    def __init__(self, normalizer: KnowledgeNormalizer) -> None:
        self._normalizer = normalizer

    def extract(self, document: RawDocument) -> list[KnowledgeDocument]:
        """Extract all sections from a RawDocument into KnowledgeDocuments."""
        product_id = _slugify(document.name)
        documents: list[KnowledgeDocument] = []

        for index, section in enumerate(document.sections):
            raw_content = _assemble_raw_content(section)
            content = self._normalizer.normalize_document(
                content=section.content,
                bullets=section.bullets,
                tables=section.tables,
            )

            metadata = DocumentMetadata(
                product_id=product_id,
                product_name=document.name,
                section=section.title,
                # Fall back to "en" until language detection is implemented.
                language=document.language or "en",
                source=str(document.source_path),
                url=document.url,
            )

            documents.append(
                KnowledgeDocument(
                    id=f"{product_id}_{index:03d}",
                    title=section.title,
                    raw_content=raw_content,
                    content=content,
                    metadata=metadata,
                )
            )

        return documents
