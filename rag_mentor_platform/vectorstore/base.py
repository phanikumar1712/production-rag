"""
Abstract base class for swappable vectorstore implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple


class VectorStore(ABC):
    """Base class for vector store implementations."""

    @abstractmethod
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Add documents to the vector store.

        Args:
            documents: List of document texts or document objects
            metadata: Optional list of metadata dictionaries
        """
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents.

        Args:
            query: Search query string
            top_k: Number of top results to return
            filters: Optional filters to apply

        Returns:
            List of search results with scores and metadata
        """
        pass

    @abstractmethod
    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Delete documents from the vector store.

        Args:
            document_ids: List of document IDs to delete
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all documents from the vector store."""
        pass
