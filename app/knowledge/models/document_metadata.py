from pydantic import BaseModel, ConfigDict


class DocumentMetadata(BaseModel):
    """
    Metadata for a single knowledge section.

    Carries provenance information only — no content.
    """

    model_config = ConfigDict(frozen=True)

    # Normalized slug derived from the product name, e.g. "al-araby-card".
    # Used as a stable identifier for grouping and future Qdrant collection naming.
    product_id: str

    # Human-readable display name, e.g. "Al-Araby Card".
    product_name: str

    # Title of the section this document was extracted from.
    section: str

    # BCP-47 language tag. Defaults to "en" until language detection is added.
    language: str

    # Absolute path to the source JSON file, stored as a string for serialization.
    source: str

    # Original product page URL from the JSON file.
    url: str
