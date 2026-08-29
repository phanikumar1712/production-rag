"""
Ingestion pipeline: load → chunk → embed → store in Qdrant.

Why offline (not at query time):
- Embedding 1000 docs takes minutes; do it once
- Indexed vectors persist in Qdrant's volume
- New docs can be added incrementally

Run with: python -m src.ingestion --dir data/
"""

import argparse
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    TextLoader,
)
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from dotenv import load_dotenv
from src.config import settings

load_dotenv()

LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".md": UnstructuredMarkdownLoader,
    ".txt": TextLoader,
}


def load_documents(directory: str) -> list:
    """Load all supported documents from a directory recursively."""
    docs = []
    for ext, loader_cls in LOADER_MAP.items():
        for filepath in Path(directory).rglob(f"*{ext}"):
            try:
                loader = loader_cls(str(filepath))
                loaded = loader.load()
                # Attach filename to every chunk - used for citations
                for doc in loaded:
                    doc.metadata["source_file"] = filepath.name
                docs.extend(loaded)
                print(f"  ✓ {filepath.name} ({len(loaded)} sections)")
            except Exception as e:
                print(f"  ✗ {filepath.name}: {e}")
    if not docs:
        raise ValueError(
            f"No documents found in '{directory}'. Add .pdf, .md, or .txt files."
        )
    return docs


def chunk_documents(docs: list) -> list:
    """
    Split with RecursiveCharacterTextSplitter.
    Tries paragraph → sentence → word → character splits in that order.
    Far better semantic coherence than fixed-size splitting.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"  ✓ {len(chunks)} chunks from {len(docs)} documents")
    return chunks


def build_index(docs_directory: str) -> None:
    """
    Full pipeline: load → chunk → embed → store in Qdrant hybrid collection.
    Uses Qdrant's native hybrid mode (dense + BM25 sparse in one collection).
    Qdrant handles the score fusion internally at query time.
    """
    print(f"\n=== Ingestion: {docs_directory} ===")
    docs = load_documents(docs_directory)
    chunks = chunk_documents(docs)

    # Dense embeddings: OpenAI text-embedding-3-large (3072 dims)
    dense_emb = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
    )
    # Sparse embeddings: BM25 via FastEmbed (no API key needed, runs locally)
    sparse_emb = FastEmbedSparse(model_name="Qdrant/BM25")

    print(f"\nEmbedding {len(chunks)} chunks (takes a few minutes)...")
    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=dense_emb,
        sparse_embedding=sparse_emb,
        url=settings.qdrant_url,
        collection_name=settings.collection_name,
        retrieval_mode=RetrievalMode.HYBRID,
        force_recreate=True,  # fresh ingest each run
    )
    print(f"  ✓ Indexed {len(chunks)} chunks")
    print(f"  ✓ Dashboard: http://localhost:6333/dashboard\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="data/")
    args = parser.parse_args()
    build_index(args.dir)
