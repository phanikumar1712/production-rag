#!/usr/bin/env python
"""
RAG Document Ingestion Script

Run manually to ingest documents from data/ directory into the vector store.

Usage:
    python ingest.py                    # Ingest all documents
    python ingest.py --clear            # Clear vector store and re-ingest
    python ingest.py --file documents.md  # Ingest specific file
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional
import argparse

# Import from RAG platform
from rag_mentor_platform.core import settings
from rag_mentor_platform.ingestion.document_processor import DocumentProcessor
from rag_mentor_platform.vectorstore import ChromaVectorStore

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_documents_from_directory(data_dir: Path) -> List[Path]:
    """Get all supported document files from directory."""
    supported_extensions = {".txt", ".md", ".pdf"}
    documents = []
    
    if not data_dir.exists():
        logger.warning(f"Data directory not found: {data_dir}")
        return documents
    
    for file_path in data_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            documents.append(file_path)
    
    return sorted(documents)


def ingest_documents(
    data_dir: Path = None,
    vector_store: ChromaVectorStore = None,
    clear_first: bool = False,
    specific_file: Optional[str] = None,
) -> dict:
    """
    Ingest documents into vector store.
    
    Args:
        data_dir: Directory containing documents (default: data/)
        vector_store: Vector store instance (default: Chroma)
        clear_first: Clear vector store before ingesting
        specific_file: Only ingest specific file
    
    Returns:
        Dictionary with ingestion statistics
    """
    if data_dir is None:
        data_dir = Path(__file__).parent / "data"
    
    if vector_store is None:
        logger.info(f"Initializing Chroma vector store...")
        # Note: In production, inject actual Chroma client
        vector_store = ChromaVectorStore(collection_name=settings.collection_name)
    
    # Clear if requested
    if clear_first:
        logger.info("Clearing vector store...")
        vector_store.clear()
    
    # Get documents
    if specific_file:
        doc_files = [data_dir / specific_file]
    else:
        doc_files = get_documents_from_directory(data_dir)
    
    if not doc_files:
        logger.error(f"No documents found in {data_dir}")
        return {"success": False, "error": "No documents found"}
    
    logger.info(f"Found {len(doc_files)} document(s) to ingest")
    
    # Initialize processor
    processor = DocumentProcessor(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    
    # Process and ingest
    stats = {
        "total_files": len(doc_files),
        "total_chunks": 0,
        "processed_files": 0,
        "errors": [],
    }
    
    for file_path in doc_files:
        try:
            logger.info(f"Processing: {file_path.name}")
            
            # Load and chunk document
            chunks, metadata = processor.process_document(file_path)
            
            if chunks:
                # Prepare documents for vector store
                documents = [
                    {
                        "id": f"{file_path.stem}_{i}",
                        "text": chunk,
                        "metadata": meta,
                    }
                    for i, (chunk, meta) in enumerate(zip(chunks, metadata))
                ]
                
                # Add to vector store
                vector_store.add_documents(documents)
                
                stats["processed_files"] += 1
                stats["total_chunks"] += len(chunks)
                
                logger.info(
                    f"✓ Ingested {file_path.name}: {len(chunks)} chunks"
                )
            else:
                logger.warning(f"No chunks generated from {file_path.name}")
        
        except Exception as e:
            error_msg = f"Error processing {file_path.name}: {str(e)}"
            logger.error(error_msg)
            stats["errors"].append(error_msg)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("Ingestion Summary:")
    logger.info(f"  Files processed: {stats['processed_files']}/{stats['total_files']}")
    logger.info(f"  Total chunks: {stats['total_chunks']}")
    logger.info(f"  Chunk size: {settings.chunk_size}")
    logger.info(f"  Chunk overlap: {settings.chunk_overlap}")
    if stats["errors"]:
        logger.info(f"  Errors: {len(stats['errors'])}")
        for error in stats["errors"]:
            logger.debug(f"    - {error}")
    logger.info(f"{'='*60}\n")
    
    stats["success"] = len(stats["errors"]) == 0
    return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into RAG vector store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest.py                          # Ingest all documents
  python ingest.py --clear                  # Clear and re-ingest
  python ingest.py --file acme_docs.md     # Ingest specific file
        """,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to documents directory (default: ./data)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear vector store before ingesting",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Ingest specific file only",
    )
    
    args = parser.parse_args()
    
    logger.info("RAG Document Ingestion")
    logger.info(f"Vector DB: Qdrant at {settings.qdrant_url}")
    logger.info(f"Collection: {settings.collection_name}\n")
    
    try:
        stats = ingest_documents(
            data_dir=args.data_dir,
            clear_first=args.clear,
            specific_file=args.file,
        )
        
        if stats["success"]:
            logger.info("✓ Ingestion complete!")
            sys.exit(0)
        else:
            logger.error("✗ Ingestion failed")
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("\n✗ Ingestion cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"✗ Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
