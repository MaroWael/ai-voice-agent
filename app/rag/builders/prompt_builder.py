class PromptBuilder:
    """
    Pure synchronous builder that constructs the final prompt string from context and question.
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
        Formats the template with the provided context and question.

        Args:
            question: The user query string.
            context: The formatted context block string.

        Returns:
            The complete prompt ready for LLM consumption.
        """
        return self._template.format(context=context, question=question)
