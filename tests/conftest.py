"""Shared pytest fixtures: fake retriever/generator so tests run without Qdrant or API keys."""

from langchain_core.documents import Document
import pytest

from rag_mentor_platform.core import settings


@pytest.fixture
def fake_docs():
    """Documents mimicking retriever output (LangChain Documents with metadata)."""
    return [
        Document(
            page_content="Employees are entitled to 20 days of paid time off per year.",
            metadata={"source_file": "acme_employee_handbook.txt", "relevance_score": 0.92},
        ),
        Document(
            page_content="The Enterprise tier allows 10,000 requests per minute.",
            metadata={"source_file": "acme_api_documentation.md", "relevance_score": 0.87},
        ),
    ]


@pytest.fixture
def mock_rag_pipeline(monkeypatch, fake_docs):
    """Patch retrieval and generation so the API layer can be tested in isolation."""
    from rag_mentor_platform.api.routes import chat

    monkeypatch.setattr(chat, "retrieve_documents", lambda query: fake_docs)
    monkeypatch.setattr(
        chat,
        "generate_answer",
        lambda query, docs: {
            "answer": "Employees get 20 days of PTO. [Source: acme_employee_handbook.txt]",
            "sources": ["acme_employee_handbook.txt"],
            "context_used": len(docs),
        },
    )


@pytest.fixture
def client():
    """FastAPI TestClient, with Settings validation satisfied via env vars."""
    import os

    os.environ.setdefault("OPENROUTER_API_KEY", "test-key-for-smoke-tests")
    from fastapi.testclient import TestClient

    from rag_mentor_platform.api.app import app

    return TestClient(app)
