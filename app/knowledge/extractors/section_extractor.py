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
    - Duplicate section titles inside the same product are disambiguated using
      metadata/variant info (e.g. "Visa", "Mastercard") or deterministic fallback ("variant 1").
    - Both raw and normalized content are stored.
    - Chunk IDs incorporate the variant distinction when disambiguated.
    """

    def __init__(self, normalizer: KnowledgeNormalizer) -> None:
        self._normalizer = normalizer

    def _detect_section_variant(self, section: RawSection) -> str | None:
        # 1. Prefer existing structured metadata if available
        if hasattr(section, "variant") and getattr(section, "variant"):
            return str(getattr(section, "variant"))

        # 2. Inspect content when metadata is unavailable
        text_parts = [section.content]
        if section.bullets:
            text_parts.extend(section.bullets)
        for t in section.tables:
            for row in t:
                text_parts.append(" ".join(str(cell) for cell in row))
        text = " ".join(text_parts)

        text_lower = text.lower()
        has_visa = "visa" in text_lower
        has_mastercard = "mastercard" in text_lower or "master card" in text_lower
        has_amex = "amex" in text_lower or "american express" in text_lower

        if has_visa and not has_mastercard:
            return "Visa"
        if has_mastercard and not has_visa:
            return "Mastercard"
        if has_amex and not (has_visa or has_mastercard):
            return "Amex"

        return None

    def extract(self, document: RawDocument) -> list[KnowledgeDocument]:
        """Extract all sections from a RawDocument into KnowledgeDocuments."""
        product_id = _slugify(document.name)
        sections = document.sections

        # Check for duplicate section titles
        title_counts: dict[str, int] = {}
        for s in sections:
            title_counts[s.title] = title_counts.get(s.title, 0) + 1

        has_duplicates = any(count > 1 for count in title_counts.values())

        # Resolve variant names for each occurrence index if duplicate titles exist
        occurrence_variants: dict[int, str] = {}
        section_occurrences: list[int | None] = []

        if has_duplicates:
            title_occurrences: dict[str, int] = {}
            for s in sections:
                if title_counts[s.title] > 1:
                    occ = title_occurrences.get(s.title, 0)
                    title_occurrences[s.title] = occ + 1
                    section_occurrences.append(occ)
                else:
                    section_occurrences.append(None)

            # Discover variants per occurrence group
            for idx, s in enumerate(sections):
                occ = section_occurrences[idx]
                if occ is not None and occ not in occurrence_variants:
                    variant_found = self._detect_section_variant(s)
                    if variant_found:
                        occurrence_variants[occ] = variant_found

            # Fallback to deterministic variant labels if variant info cannot be determined
            max_occ = max(title_occurrences.values()) if title_occurrences else 0
            for occ in range(max_occ):
                if occ not in occurrence_variants:
                    occurrence_variants[occ] = f"variant {occ + 1}"

        documents: list[KnowledgeDocument] = []
        title_current_occ: dict[str, int] = {}

        for index, section in enumerate(sections):
            raw_content = _assemble_raw_content(section)
            content = self._normalizer.normalize_document(
                content=section.content,
                bullets=section.bullets,
                tables=section.tables,
            )

            section_title = section.title
            doc_id = f"{product_id}_{index:03d}"

            if title_counts[section.title] > 1:
                occ = title_current_occ.get(section.title, 0)
                title_current_occ[section.title] = occ + 1
                var_name = occurrence_variants.get(occ, f"variant {occ + 1}")
                section_title = f"{section.title} ({var_name})"
                var_slug = _slugify(var_name)
                doc_id = f"{product_id}_{var_slug}_{index:03d}"

            metadata = DocumentMetadata(
                product_id=product_id,
                product_name=document.name,
                section=section_title,
                aliases=document.aliases,
                arabic_name=document.arabic_name,
                language=document.language or "en",
                source=str(document.source_path),
                url=document.url,
            )

            documents.append(
                KnowledgeDocument(
                    id=doc_id,
                    title=section_title,
                    raw_content=raw_content,
                    content=content,
                    metadata=metadata,
                )
            )

        return documents

