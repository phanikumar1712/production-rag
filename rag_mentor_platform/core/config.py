"""
All settings loaded from environment variables via pydantic-settings.
Never hardcode API keys in code.
Contains RAG-specific configuration knobs only.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openrouter_api_key: str
    cohere_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "rag_docs"

    # Chunking: 512 chosen after testing 256/512/768.
    # 512 gave best RAGAS context precision (0.84) on the golden eval set.
    chunk_size: int = 512
    chunk_overlap: int = 50  # 10% overlap reduces boundary-split failures

    # Retrieval: retrieve 20 for reranker to score, inject top 5
    dense_top_k: int = 20
    rerank_top_k: int = 5

    # Generation
    llm_model: str = "openai/gpt-4o-mini"
    temperature: float = 0.0  # 0 for factual Q&A

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
