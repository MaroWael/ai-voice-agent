"""
RAG Prompt Builder

Pure synchronous builder that constructs the final prompt string from context,
question, and detected language instructions.
"""

from app.rag.builders.language_detector import (
    detect_query_language,
    get_language_instruction,
)


class PromptBuilder:
    """
    Constructs the final prompt string for LLM consumption.
    Receives prompt template externally.
    """

    def __init__(self, template: str) -> None:
        """
        Initializes the builder with a configurable prompt template string.

        Args:
            template: The raw prompt template containing '{context}' and '{question}' placeholders.
        """
        if "{context}" not in template or "{question}" not in template:
            raise ValueError(
                "Prompt template must contain both '{context}' and '{question}' placeholders."
            )
        self._template = template

    def build_prompt(self, question: str, context: str) -> str:
        """
        Formats the template with the provided context, question, and language instructions.

        Args:
            question: The user query string.
            context: The formatted context block string.

        Returns:
            The complete prompt ready for LLM consumption.
        """
        lang_code = detect_query_language(question)
        lang_instruction = get_language_instruction(lang_code)

        if "{language_instruction}" in self._template:
            return self._template.format(
                language_instruction=lang_instruction,
                context=context,
                question=question,
            )

        return self._template.format(context=context, question=question)
