"""
RAG Mentor Platform

Production-grade Retrieval-Augmented Generation (RAG) system with hybrid search,
cross-encoder reranking, and citation-enforced answers.
"""

__version__ = "1.0.0"
__author__ = "RAG Team"

from .core import settings, RetrievalError, EmptyContextError, LLMGenerationError

__all__ = [
    "settings",
    "RetrievalError", 
    "EmptyContextError",
    "LLMGenerationError",
]
