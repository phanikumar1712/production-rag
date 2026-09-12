# Production RAG Pipeline

A production-grade Retrieval-Augmented Generation (RAG) system with hybrid search, cross-encoder reranking, and citation-enforced answers.

Ask questions against your documents and get precise, cited answers with source attribution. Powered by [OpenRouter](https://openrouter.ai/) for flexible model access.

## Architecture

```
┌──────────────┐     ┌──────────────────────────────────────────────┐
│  User Query  │────▶│              FastAPI Server                   │
└──────────────┘     │  POST /query        (full answer)             │
                     │  POST /query/stream  (SSE token streaming)    │
                     └──────────┬───────────────────┬───────────────┘
                                │                   │
                    ┌───────────▼──────┐   ┌───────▼──────────┐
                    │    Retrieval     │   │    Generation     │
                    │                  │   │                   │
                    │  1. Hybrid search│   │  System prompt    │
                    │     (BM25+dense) │   │  with citations   │
                    │     → top 20     │   │  + retrieved      │
                    │  2. Cohere       │   │    context        │
                    │     reranker     │   │  → OpenRouter     │
                    │     → top 5      │   │                   │
                    └───────┬──────────┘   └───────────────────┘
                            │
                    ┌───────▼──────────┐
                    │  Qdrant Vector   │
                    │  Store           │
                    │  (hybrid index)  │
                    │  port 6333       │
                    └──────────────────┘
```

## How It Works

### Ingestion (offline)
1. **Load** — PDF, Markdown, and Text files from a directory
2. **Chunk** — RecursiveCharacterTextSplitter (512 tokens, 50 overlap)
3. **Embed** — OpenAI text-embedding-3-large via OpenRouter (dense) + BM25 (sparse)
4. **Store** — Qdrant hybrid collection (dense + sparse vectors)

### Query (online)
1. **Hybrid search** — BM25 + dense retrieval returns top-20 candidates
2. **Rerank** — Cohere cross-encoder narrows to top-5 most relevant chunks
3. **Generate** — LLM answers with mandatory `[Source: filename]` citations via OpenRouter

### Why Two-Stage Retrieval?
- **Bi-encoder (stage 1):** Fast — query and docs embedded independently, ANN lookup in milliseconds. Less accurate because query and doc never interact during embedding.
- **Cross-encoder (stage 2):** Accurate — query and doc processed together with full token attention. Too slow for full corpus; runs only on the top-20 candidates.
- **Result:** 15-30% improvement in context precision over bi-encoder alone.

## Prerequisites

- Python 3.12+
- Docker (for Qdrant)
- [OpenRouter](https://openrouter.ai/) API key (LLM + embeddings)
- [Cohere](https://cohere.com/) API key (reranker, improves retrieval by 15-30%)

> **Note:** When installing packages, always use `python -m pip` instead of bare `pip` to ensure packages install into the correct virtualenv.

## Quick Start

### 1. Start Qdrant

```bash
docker compose up -d
```

### 2. Set up Python environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install markdown  # required for .md file loading
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and add your API keys:
#   OPENROUTER_API_KEY=sk-or-v1-...   (required: LLM + embeddings)
#   COHERE_API_KEY=cohere_...         (optional: improves retrieval)
```

### 4. Ingest your documents

```bash
# Place .pdf, .md, or .txt files in data/, then:
python ingest.py                    # ingest everything from data/
python ingest.py --file docs.md     # ingest a single file
python ingest.py --dir path/to/docs # ingest a different directory
```

### 5. Start the API server

```bash
python -m rag_mentor_platform.main --host 0.0.0.0 --port 8000
# or:
uvicorn rag_mentor_platform.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Ask questions

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How many PTO days do employees get?"}'
```

## API Endpoints

### `GET /health`
Health check. Returns model name and status.

### `POST /query`
Synchronous Q&A. Returns the full answer at once.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic of the documents?"}'
```

Response:
```json
{
  "answer": "The main topic is... [Source: policy.pdf]",
  "sources": ["policy.pdf"],
  "context_chunks_used": 5
}
```

### `POST /query/stream`
Streaming Q&A via Server-Sent Events. First token in <1s.

```bash
curl -N -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic of the documents?"}'
```

## Configuration

All settings are in `rag_mentor_platform/core/config.py` and configurable via environment variables:

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | *(required)* | OpenRouter API key for LLM and embeddings |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter API endpoint |
| `COHERE_API_KEY` | *(empty)* | Cohere API key for reranker (optional) |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `COLLECTION_NAME` | `rag_docs` | Qdrant collection name |
| `CHUNK_SIZE` | `512` | Tokens per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `DENSE_TOP_K` | `20` | Candidates before reranking |
| `RERANK_TOP_K` | `5` | Chunks sent to LLM |
| `LLM_MODEL` | `openai/gpt-4o-mini` | Generation model (OpenRouter format) |
| `TEMPERATURE` | `0.0` | 0 for factual Q&A |

### Changing Models

OpenRouter supports hundreds of models. Just change `LLM_MODEL` in your `.env`:

```bash
# OpenAI models (via OpenRouter)
LLM_MODEL=openai/gpt-4o
LLM_MODEL=openai/gpt-4o-mini

# Anthropic models (via OpenRouter)
LLM_MODEL=anthropic/claude-3.5-sonnet

# Google models (via OpenRouter)
LLM_MODEL=google/gemini-2.0-flash-001
```

## Evaluation

The project uses [RAGAS](https://docs.ragas.io/) for automated evaluation against `golden_dataset.json`:

```python
from rag_mentor_platform.evaluation.ragas_eval import run_evaluation

results = run_evaluation("golden_dataset.json")
print(results)  # {faithfulness, answer_relevancy, context_precision, context_recall, ...}
```

Or from the command line (requires Qdrant running + API keys configured):

```bash
python -m rag_mentor_platform.evaluation.ragas_eval           # all questions
python -m rag_mentor_platform.evaluation.ragas_eval --limit 4 # first 4 only
```

Metrics tracked:
| Metric | Target | What it measures |
|---|---|---|
| **Faithfulness** | > 0.80 | Are model claims supported by context? |
| **AnswerRelevancy** | > 0.80 | Does the answer address the question? |
| **ContextPrecision** | > 0.80 | Are retrieved chunks actually relevant? |
| **ContextRecall** | > 0.80 | Did retrieval find all needed information? |

## Testing

```bash
python -m pytest tests/ -v
```

## Project Structure

```
.
├── ingest.py                            # CLI: ingest docs into Qdrant
├── rag_mentor_platform/                 # Main application package
│   ├── main.py                          # Server entrypoint (uvicorn)
│   ├── core/
│   │   ├── config.py                    # Settings from env vars (pydantic-settings)
│   │   └── exceptions.py                # RetrievalError, EmptyContextError, LLMGenerationError
│   ├── api/
│   │   ├── app.py                       # FastAPI app factory
│   │   └── routes/
│   │       └── chat.py                  # /query and /query/stream endpoints
│   ├── ingestion/
│   │   └── document_processor.py        # Load → chunk → embed → store in Qdrant
│   ├── retrieval/
│   │   └── retriever.py                 # Hybrid search + Cohere reranking
│   ├── llm/
│   │   ├── response_generator.py        # Grounded generation with citation enforcement
│   │   └── prompts/
│   │       └── rag_prompt_v1.txt        # RAG system prompt template
│   └── evaluation/
│       └── ragas_eval.py                # RAGAS evaluation pipeline
├── data/                                # Documents to ingest (.pdf, .md, .txt)
├── tests/                               # Smoke tests (pytest)
├── golden_dataset.json                  # Q&A pairs for RAGAS evaluation
├── docker-compose.yml                   # Qdrant vector store
├── requirements.txt                     # Pinned Python dependencies
├── .env                                 # Environment variables (not committed)
├── .env.example                         # Environment variable template
└── README.md
```

## Key Design Decisions

**OpenRouter** — Single API key for 100+ models (OpenAI, Anthropic, Google, Meta, etc). Swap models by changing one env var.

**Chunk size 512** — Tested 256/512/768. 512 gave the best RAGAS context precision (0.84) on the golden evaluation set.

**Temperature 0.0** — Factual Q&A requires deterministic outputs. Higher temperatures increase hallucination risk.

**Citation enforcement** — The system prompt mandates `[Source: filename]` for every factual claim. This makes hallucinations visible and gives users a way to verify answers.

**"I don't have enough information" fallback** — Prevents the model from guessing when context is insufficient. Critical for trust.

**Hybrid search over dense-only** — BM25 catches keyword matches that embeddings miss (e.g., exact product names, error codes). Dense captures semantic similarity. Fusion of both improves recall.

**Cohere reranker (optional)** — Cross-encoder reranking improves context precision by 15-30% over bi-encoder alone. Falls back gracefully if `COHERE_API_KEY` is not set.
