"""
Two-stage retrieval:
1. Hybrid search (BM25 + dense) returns top-20 candidates
2. Cohere cross-encoder reranks to top-5

Why two stages?
- Bi-encoder (stage 1): fast - query and docs embedded independently,
  ANN lookup takes milliseconds. Less accurate because query and doc
  never interact during the embedding computation.
- Cross-encoder (stage 2): accurate - query and doc processed together,
  full attention between all tokens. Cannot be pre-computed (no index).
  Impractical for full corpus search; runs on the top-20 candidates only.

This combination improves context precision by 15-30% over bi-encoder alone.

IMPORTANT: Using rerank-v3.5 model (released 2025).
rerank-english-v3.0 is the older model - still works but v3.5 is better
and multilingual. Always use the current model name.
"""

from langchain_cohere import CohereRerank
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_classic.retrievers import ContextualCompressionRetriever
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from src.config import settings

load_dotenv()


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
            print(f"\nWarning: Cohere reranker failed ({e}) - falling back to dense only")
            return base_retriever
    else:
        print("\nNote: COHERE_API_KEY not set - skipping reranker (lower precision)")
        return base_retriever


def retrieve_documents(query: str) -> list:
    """Retrieve top-k document chunks for a query."""
    try:
        retriever = get_retriever()
        return retriever.invoke(query)
    except Exception as e:
        # Collection may not exist yet, or Qdrant may be unavailable
        print(f"\nWarning: Retrieval failed: {e}")
        return []
