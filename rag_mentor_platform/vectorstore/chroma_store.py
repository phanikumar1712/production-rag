"""
Chroma-based vector store implementation for RAG system.
"""

from typing import List, Dict, Any, Optional
from .base import VectorStore


class ChromaVectorStore(VectorStore):
    """Chroma implementation of the VectorStore interface."""

    def __init__(self, collection_name: str = "rag_docs", client=None):
        """
        Initialize Chroma vector store.

        Args:
            collection_name: Name of the Chroma collection
            client: Optional Chroma client instance
        """
        self.collection_name = collection_name
        self.client = client
        if self.client:
            self.collection = self.client.get_or_create_collection(name=collection_name)
        else:
            self.collection = None

    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add documents to Chroma collection."""
        if not self.collection:
            raise RuntimeError("Chroma client not initialized")
        
        # Implementation would add documents to Chroma collection
        self.collection.add(
            documents=[doc.get("text", str(doc)) for doc in documents],
            metadatas=metadata or [{} for _ in documents],
            ids=[doc.get("id", f"doc_{i}") for i, doc in enumerate(documents)],
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for relevant documents in Chroma."""
        if not self.collection:
            raise RuntimeError("Chroma client not initialized")

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=filters,
        )
        
        # Format results
        formatted_results = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                formatted_results.append(
                    {
                        "text": doc,
                        "score": results["distances"][0][i] if results["distances"] else 0.0,
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    }
                )
        
        return formatted_results

    def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents from Chroma."""
        if not self.collection:
            raise RuntimeError("Chroma client not initialized")
        
        self.collection.delete(ids=document_ids)

    def clear(self) -> None:
        """Clear all documents from the collection."""
        if not self.collection:
            raise RuntimeError("Chroma client not initialized")
        
        # Get all IDs and delete them
        all_items = self.collection.get()
        if all_items and all_items["ids"]:
            self.collection.delete(ids=all_items["ids"])
