"""Smoke tests: app wiring, config, and API endpoints (pipeline mocked)."""

import pytest
from fastapi.testclient import TestClient


class TestAppWiring:
    def test_app_creation(self):
        from rag_mentor_platform.api.app import create_app

        app = create_app()
        assert app.title == "RAG Q&A API"

    def test_routes_registered(self):
        from rag_mentor_platform.api.app import app

        paths = set(app.openapi()["paths"])
        assert {"/health", "/query", "/query/stream"} <= paths


class TestConfig:
    def test_settings_load(self):
        from rag_mentor_platform.core import settings

        assert settings.openrouter_api_key  # required, must be non-empty
        assert settings.chunk_size > settings.chunk_overlap > 0
        assert settings.llm_model

    def test_custom_exceptions(self):
        from rag_mentor_platform.core import (
            EmptyContextError,
            LLMGenerationError,
            RetrievalError,
        )

        for exc in (RetrievalError, EmptyContextError, LLMGenerationError):
            assert issubclass(exc, Exception)


class TestHealth:
    def test_health(self, client: TestClient):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["model"]


class TestQueryEndpoints:
    def test_query_success(self, client: TestClient, mock_rag_pipeline):
        resp = client.post("/query", json={"query": "How many PTO days?"})
        assert resp.status_code == 200
        body = resp.json()
        assert "20 days" in body["answer"]
        assert body["sources"] == ["acme_employee_handbook.txt"]
        assert body["context_chunks_used"] == 2

    def test_query_rejects_blank(self, client: TestClient, mock_rag_pipeline):
        resp = client.post("/query", json={"query": "   "})
        assert resp.status_code == 422


class TestIngestionHelpers:
    def test_resolve_documents_finds_data(self):
        from pathlib import Path

        from ingest import resolve_documents

        data_dir = Path(__file__).parent.parent / "data"
        docs = resolve_documents(data_dir, specific_file=None)
        names = {d.name for d in docs}
        assert "acme_employee_handbook.txt" in names
        assert "acme_api_documentation.md" in names

    def test_resolve_documents_rejects_missing_file(self, tmp_path):
        from ingest import resolve_documents

        with pytest.raises(FileNotFoundError):
            resolve_documents(tmp_path, specific_file="nope.md")

    def test_resolve_documents_rejects_unsupported_type(self, tmp_path):
        from ingest import resolve_documents

        bad = tmp_path / "malware.exe"
        bad.write_bytes(b"whatever")
        with pytest.raises(ValueError):
            resolve_documents(tmp_path, specific_file="malware.exe")
