"""
RAG Models — Response Status

Enum representing the outcome of a RAG pipeline execution.
Kept in its own module so it can be imported by downstream layers
without pulling in the full RagResponse.
"""

from enum import Enum


class RagStatus(Enum):
    """
    Categorical status of a completed RAG pipeline run.

    SUCCESS:
        The pipeline retrieved sufficient context and generated an answer.

    INSUFFICIENT_CONTEXT:
        The retrieval step did not return enough evidence to answer the
        question. The pipeline returned a standard fallback reply without
        calling the LLM.

    ERROR:
        An unexpected error occurred during pipeline execution.
        Reserved for future use by error-handling middleware.
    """

    SUCCESS = "success"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    ERROR = "error"
