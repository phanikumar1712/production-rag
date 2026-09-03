"""
Custom exception classes for RAG system.
"""


class RetrievalError(Exception):
    """Raised when document retrieval fails."""
    pass


class EmptyContextError(Exception):
    """Raised when no relevant context is found for a query."""
    pass


class LLMGenerationError(Exception):
    """Raised when LLM response generation fails."""
    pass
