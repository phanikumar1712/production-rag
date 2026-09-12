"""
Two-stage retrieval:
1. Hybrid search (BM25 + dense) returns top-20 candidates
2. Cohere cross-encoder reranks to top-5

This module is a near-copy of the original `src/retrieval.py` wired
into the new scaffold.
"""

import logging

from langchain_cohere import CohereRerank
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_classic.retrievers import ContextualCompressionRetriever
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from rag_mentor_platform.core import settings, RetrievalError

load_dotenv()

logger = logging.getLogger(__name__)


def get_retriever():
    """
    Build the two-stage retriever.
    Returns ContextualCompressionRetriever (hybrid search + Cohere rerank)
    or base retriever if Cohere key is not set.
    """
    client = QdrantClient(url=settings.qdrant_url)
    dense_emb = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
    )
    sparse_emb = FastEmbedSparse(model_name="Qdrant/BM25")

    # Connect to the already-indexed collection (no data added here)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=settings.collection_name,
        embedding=dense_emb,
        sparse_embedding=sparse_emb,
        retrieval_mode=RetrievalMode.HYBRID,
    )
    base_retriever = vector_store.as_retriever(
        search_kwargs={"k": settings.dense_top_k}
    )

    if settings.cohere_api_key:
        # rerank-v3.5: current recommended model as of 2025
        # multilingual, handles 100+ languages
        compressor = CohereRerank(
            cohere_api_key=settings.cohere_api_key,
            model="rerank-v3.5",
            top_n=settings.rerank_top_k,
        )
        try:
            return ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever,
            )
        except Exception as e:
            # Log warning but fall back to dense-only retrieval
            logger.warning(
                "Cohere reranker failed to initialize (%s) - falling back to dense-only", e
            )
            return base_retriever
    else:
        # No Cohere API key; use dense-only retrieval
        return base_retriever


def retrieve_documents(query: str) -> list:
    """Retrieve top-k document chunks for a query.
    
    Args:
        query: The search query string
        
    Returns:
        List of retrieved documents
        
    Raises:
        RetrievalError: If retrieval fails (e.g., Qdrant unavailable, collection empty)
    """
    try:
        retriever = get_retriever()
        results = retriever.invoke(query)
        if not results:
            raise RetrievalError(f"No documents found for query: {query}")
        return results
    except RetrievalError:
        raise
    except Exception as e:
        raise RetrievalError(f"Retrieval failed: {str(e)}")
