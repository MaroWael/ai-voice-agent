import re


class KnowledgeNormalizer:
    """
    Stateless text normalizer for knowledge documents.

    All methods are pure functions: same input always produces same output.
    The normalizer never changes the meaning of content — only its presentation.

    Responsibilities are split into focused methods orchestrated by
    normalize_document().
    """

    # Matches lines that contain only a bullet character with optional whitespace.
    _EMPTY_BULLET_LINE = re.compile(r"^[•\-\*]\s*$")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def normalize_text(self, text: str) -> str:
        """
        Clean whitespace and line breaks in a block of text.

        - Normalizes Windows (\\r\\n) and old Mac (\\r) line endings to \\n.
        - Collapses multiple spaces/tabs on a single line into one space.
        - Strips leading and trailing whitespace from each line.
        - Collapses three or more consecutive blank lines into two.
        """
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Collapse horizontal whitespace only, preserving newlines.
        text = re.sub(r"[ \t]+", " ", text)
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)
        # Limit runs of blank lines to two.
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def normalize_bullets(self, bullets: list[str]) -> str:
        """
        Convert a list of raw bullet strings into normalized bullet text.

        Empty bullets and bullets containing only a bullet character are removed.
        Each surviving bullet is prefixed with '• '.
        """
        lines: list[str] = []
        for bullet in bullets:
            stripped = bullet.strip()
            if not stripped or self._EMPTY_BULLET_LINE.match(stripped):
                continue
            lines.append(f"• {self.normalize_text(stripped)}")
        return "\n".join(lines)

    def normalize_table(self, table: list[list[str]]) -> str:
        """
        Format a table into readable, embedding-friendly prose for any column count.

        Every data row is rendered as a multi-line block — one "Header: value" pair
        per line — so that column context is always preserved in the output.
        Rows are separated by a blank line to keep each record visually distinct.

        Special case: when the header is "tenor" (case-insensitive), numeric
        values are suffixed with " months" for semantic clarity (e.g. "3 months").

        Empty cells and empty rows are silently skipped.
        """
        if not table or len(table) < 2:
            return ""

        headers = [self.normalize_text(cell).rstrip(":") for cell in table[0]]
        formatted_rows: list[str] = []

        for row in table[1:]:
            pairs: list[str] = []
            for col_index, cell in enumerate(row):
                header = (
                    headers[col_index]
                    if col_index < len(headers)
                    else f"Column {col_index + 1}"
                )
                value = self.normalize_text(cell)
                if not value:
                    continue
                value = self._format_cell_value(header, value)
                pairs.append(f"{header}: {value}")
            if pairs:
                formatted_rows.append("\n".join(pairs))

        return "\n\n".join(formatted_rows)

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _format_cell_value(self, header: str, value: str) -> str:
        """Apply header-specific formatting to a single cell value.

        Currently handles one special case:
        - "Tenor" columns: appends " months" to values that are purely numeric,
          e.g. "3" becomes "3 months", "2.81%" is left unchanged.
        """
        if header.lower() == "tenor" and value.isdigit():
            return f"{value} months"
        return value

    def normalize_document(
        self,
        content: str,
        bullets: list[str],
        tables: list[list[list[str]]],
    ) -> str:
        """
        Orchestrate normalization of all section parts into one clean string.

        Parts are joined with a blank line separator. Empty parts are omitted.
        Order: content → bullets → tables (mirrors the source JSON structure).
        """
        parts: list[str] = []

        if content.strip():
            parts.append(self.normalize_text(content))

        if bullets:
            normalized_bullets = self.normalize_bullets(bullets)
            if normalized_bullets:
                parts.append(normalized_bullets)

        for table in tables:
            normalized_table = self.normalize_table(table)
            if normalized_table:
                parts.append(normalized_table)

        return "\n\n".join(parts)
