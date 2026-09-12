#!/usr/bin/env python
"""
RAG Document Ingestion Script

Thin CLI wrapper around the ingestion pipeline
(rag_mentor_platform.ingestion.document_processor).

Usage:
    python ingest.py                          # Ingest all documents from data/
    python ingest.py --dir path/to/docs       # Ingest a specific directory
    python ingest.py --file documents.md      # Ingest a specific file
"""

import argparse
import logging
import shutil
import sys
import tempfile
from pathlib import Path

from rag_mentor_platform.ingestion.document_processor import build_index
from rag_mentor_platform.core import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ingest")

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def resolve_documents(data_dir: Path, specific_file: str | None) -> list[Path]:
    """Collect documents to ingest, validating they exist and are supported."""
    if specific_file:
        file_path = data_dir / specific_file
        if not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{file_path.suffix}'. "
                f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
            )
        return [file_path]

    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    documents = sorted(
        p
        for p in data_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not documents:
        raise FileNotFoundError(
            f"No supported documents ({sorted(SUPPORTED_EXTENSIONS)}) in {data_dir}"
        )
    return documents


def ingest(data_dir: Path, specific_file: str | None = None) -> None:
    """Stage the requested docs in a temp dir and run the pipeline.

    A single file is copied into an empty temp dir so that the pipeline's
    force_recreate=True rebuilds the collection with only that file.
    """
    documents = resolve_documents(data_dir, specific_file)
    logger.info("Ingesting %d document(s): %s", len(documents),
                ", ".join(d.name for d in documents))

    # Single file: stage it alone so force_recreate doesn't wipe it with
    # stale data from other files in the directory.
    if specific_file:
        with tempfile.TemporaryDirectory(prefix="rag_ingest_") as tmp:
            staged = Path(tmp) / specific_file
            shutil.copy2(documents[0], staged)
            build_index(str(tmp))
    else:
        build_index(str(data_dir))

    logger.info("Collection: %s | Dashboard: %s/dashboard",
                settings.collection_name, settings.qdrant_url)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest documents into the RAG vector store (Qdrant)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest.py                        # Ingest all documents from data/
  python ingest.py --dir path/to/docs     # Ingest a specific directory
  python ingest.py --file acme_docs.md    # Ingest a single file
        """,
    )
    parser.add_argument(
        "--dir", type=Path, default=Path(__file__).parent / "data",
        help="Path to documents directory (default: ./data)",
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Ingest a single file (relative to --dir) instead of the whole directory",
    )
    args = parser.parse_args()

    logger.info("RAG Document Ingestion | Qdrant: %s | Collection: %s",
                settings.qdrant_url, settings.collection_name)

    try:
        ingest(args.dir, args.file)
    except KeyboardInterrupt:
        logger.info("Ingestion cancelled by user")
        sys.exit(130)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Ingestion failed: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=True)
        sys.exit(1)

    logger.info("Ingestion complete!")


if __name__ == "__main__":
    main()
