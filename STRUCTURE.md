# Project Structure

This document describes the reorganized RAG system folder structure.

## Directory Layout

```
production-rag/
├── ingest.py                      # Run manually: python ingest.py
├── rag_mentor_platform/           # Main application package
│   ├── main.py                    # Entry point / FastAPI app initialization
│   ├── core/                      # Core configuration and exceptions
│   │   ├── config.py              # Settings, environment variables, RAG tuning knobs
│   │   ├── exceptions.py          # Custom exceptions (RetrievalError, EmptyContextError, LLMGenerationError)
│   │   └── __init__.py
│   ├── api/                       # REST API layer
│   │   ├── __init__.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       └── chat.py            # Chat/query endpoint
│   ├── ingestion/                 # Document processing
│   │   ├── __init__.py
│   │   └── document_processor.py  # Chunking, metadata extraction, incremental indexing
│   ├── vectorstore/               # Vector database abstraction
│   │   ├── base.py                # VectorStore interface (swappable implementations)
│   │   ├── chroma_store.py        # Chroma vector store implementation
│   │   └── __init__.py
│   ├── retrieval/                 # Document retrieval
│   │   ├── __init__.py
│   │   └── retriever.py           # Hybrid search, reranking, min-score filtering
│   ├── llm/                       # LLM integration
│   │   ├── __init__.py
│   │   ├── response_generator.py  # Grounded generation, citations, refusal logic
│   │   └── prompts/
│   │       └── rag_prompt_v1.txt  # RAG system prompt template
│   └── evaluation/                # Evaluation tools
│       ├── __init__.py
│       └── ragas_eval.py          # RAGAS evaluation (run manually)
├── data/                          # Document storage
│   ├── acme_api_documentation.md
│   ├── acme_employee_handbook.txt
│   └── company_financial_report.txt
├── golden_dataset.json            # Evaluation golden dataset
├── docker-compose.yml             # Docker setup for Qdrant + app
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variables template
└── README.md                       # Main documentation

## Module Descriptions

### `ingest.py` (Root Level)
Standalone document ingestion script. Run manually to process and store documents in the vector database.

**Usage:**
```bash
python ingest.py                    # Ingest all documents from data/
python ingest.py --clear            # Clear vector store and re-ingest
python ingest.py --file docs.md    # Ingest specific file
```

**What it does:**
1. Loads all documents from `data/` directory (supports `.txt`, `.md`, `.pdf`)
2. Uses `DocumentProcessor` to chunk documents with configured `chunk_size` and `chunk_overlap`
3. Extracts and preserves metadata (filename, chunk ID, etc.)
4. Stores chunks in the vector store (Chroma/Qdrant)
5. Logs ingestion statistics and any errors

### `core/`
Contains all configuration knobs and custom exceptions. This is the single source of truth for:
- RAG tuning parameters (`chunk_size`, `top_k`, model names, etc.)
- Custom exception classes for error handling throughout the system

**Key Files:**
- `config.py`: Settings class with environment variable loading (pydantic)
- `exceptions.py`: RetrievalError, EmptyContextError, LLMGenerationError

### `ingestion/`
Handles document ingestion pipeline.

**Key Files:**
- `document_processor.py`: 
  - Recursive text chunking with configurable size/overlap
  - Metadata extraction (filename, page number, chunk ID)
  - Incremental indexing support for new documents

### `vectorstore/`
Abstraction layer for vector database backends. Supports swapping implementations without code changes.

**Key Files:**
- `base.py`: VectorStore ABC with interface
- `chroma_store.py`: Chroma implementation

### `retrieval/`
Two-stage retrieval pipeline: hybrid search + reranking.

**Key Files:**
- `retriever.py`:
  - Hybrid search (BM25 + dense embedding)
  - Cohere cross-encoder reranking
  - Minimum score filtering
  - Source attribution

### `llm/`
LLM-based response generation.

**Key Files:**
- `response_generator.py`:
  - Prompt construction with retrieved context
  - Grounded answer generation (stays within context)
  - Mandatory source citations
  - Refusal for out-of-scope questions
- `prompts/rag_prompt_v1.txt`: System prompt template

### `evaluation/`
RAGAS-based evaluation framework.

**Key Files:**
- `ragas_eval.py`:
  - Faithfulness scoring
  - Answer relevance
  - Context precision
  - Context recall
  - Run evaluation against `golden_dataset.json`

## Import Pattern

All modules are designed to work with clean imports:

```python
# Import config
from rag_mentor_platform.core import settings, RetrievalError, EmptyContextError

# Import vectorstore
from rag_mentor_platform.vectorstore import ChromaVectorStore

# Import other modules
from rag_mentor_platform.ingestion import document_processor
from rag_mentor_platform.retrieval import retriever
from rag_mentor_platform.llm import response_generator
from rag_mentor_platform.evaluation import run_evaluation
```

## Configuration

All configuration is captured in `core/config.py` and loaded from environment variables (`.env` file):

```env
# API Keys
OPENROUTER_API_KEY=...
COHERE_API_KEY=...

# Vector Database
QDRANT_URL=http://localhost:6333
COLLECTION_NAME=rag_docs

# RAG Tuning Knobs
CHUNK_SIZE=512
CHUNK_OVERLAP=50
DENSE_TOP_K=20
RERANK_TOP_K=5
LLM_MODEL=openai/gpt-4o-mini
TEMPERATURE=0.0
```

## Evaluation

To evaluate your RAG system:

```python
from rag_mentor_platform.evaluation import run_evaluation
from rag_mentor_platform.retrieval import retriever

results = run_evaluation("golden_dataset.json", rag_pipeline)
print(results)  # {faithfulness, answer_relevance, context_precision, context_recall}
```

See `golden_dataset.json` for expected format:
```json
[
  {
    "question": "...",
    "contexts": ["...", "..."],
    "ground_truth": "..."
  },
  ...
]
```

## Development

To add a new feature:

1. **New retrieval strategy?** — Create a new class in `retrieval/` following the same interface
2. **New vector store?** — Implement the `VectorStore` interface in `vectorstore/`
3. **New LLM provider?** — Add logic to `llm/response_generator.py`
4. **New RAG knob?** — Add it to `core/config.py` and update `.env.example`

All modules maintain clean APIs for composition and testing.
