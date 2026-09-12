"""FastAPI application factory."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag_mentor_platform.api.routes.chat import router as chat_router
from rag_mentor_platform.core import settings


def create_app() -> FastAPI:
    app = FastAPI(title="RAG Q&A API", version="1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(chat_router)

    @app.get("/health")
    def health():
        return {"status": "healthy", "model": settings.llm_model}

    return app


app = create_app()
