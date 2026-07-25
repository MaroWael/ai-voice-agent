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
            template: The raw prompt template containing context and question placeholders.
        """
        has_context = "{context}" in template or "{retrieved_context}" in template
        has_question = "{question}" in template or "{user_query}" in template
        if not has_context or not has_question:
            raise ValueError(
                "Prompt template must contain context and question placeholders."
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

        kwargs = {
            "context": context,
            "retrieved_context": context,
            "question": question,
            "user_query": question,
        }
        if "{language_instruction}" in self._template:
            kwargs["language_instruction"] = lang_instruction

        return self._template.format(**kwargs)

