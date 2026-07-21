from pathlib import Path

from pydantic import BaseModel, ConfigDict


class RawSection(BaseModel):
    """One logical section parsed directly from a JSON knowledge file."""

    model_config = ConfigDict(frozen=True)

    title: str
    content: str
    bullets: list[str]
    # Each table is a 2-D array: first row is the header, subsequent rows are data.
    tables: list[list[list[str]]]


class RawDocument(BaseModel):
    """
    Verbatim representation of a single JSON knowledge file.

    Produced by JsonKnowledgeLoader. Contains no normalization or business logic.
    The `source_path` and `language` fields are injected after JSON parsing —
    they are not present in the source files.

    `full_text` is intentionally excluded: it duplicates section data and the
    pipeline uses `sections` as the sole source of truth.
    Extra fields from the JSON (full_text, images, pdfs) are silently ignored.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    name: str
    url: str
    description: str
    sections: list[RawSection]

    # Injected by the loader; not present in source JSON.
    source_path: Path

    # Reserved for future multilingual support.
    # Not detected automatically — callers may set this if known.
    language: str | None = None
