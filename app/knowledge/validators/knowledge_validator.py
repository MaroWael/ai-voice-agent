from app.knowledge.models.raw_document import RawDocument


class KnowledgeValidationError(Exception):
    """Raised when a RawDocument fails business-level validation."""


class KnowledgeValidator:
    """
    Validates business invariants on a RawDocument before normalization.

    This is distinct from Pydantic schema validation.
    Pydantic ensures the JSON is structurally correct.
    KnowledgeValidator enforces domain rules that Pydantic cannot express:
    - A document must have a non-empty product name.
    - A document must contain at least one section.
    - Every section must have a non-empty title.
    """

    def validate(self, document: RawDocument) -> None:
        """
        Validate a document. Raises KnowledgeValidationError on the first failure.
        """
        self._require_product_name(document)
        self._require_at_least_one_section(document)
        self._require_non_empty_section_titles(document)

    def _require_product_name(self, document: RawDocument) -> None:
        if not document.name.strip():
            raise KnowledgeValidationError(
                f"Document from '{document.source_path}' has an empty product name."
            )

    def _require_at_least_one_section(self, document: RawDocument) -> None:
        if not document.sections:
            raise KnowledgeValidationError(
                f"Document '{document.name}' contains no sections."
            )

    def _require_non_empty_section_titles(self, document: RawDocument) -> None:
        for index, section in enumerate(document.sections):
            if not section.title.strip():
                raise KnowledgeValidationError(
                    f"Document '{document.name}' has a section at index {index} "
                    f"with an empty title."
                )
