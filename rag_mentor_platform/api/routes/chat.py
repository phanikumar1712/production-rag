"""
Chat/query endpoints for the RAG pipeline.
Two endpoints:
- POST /query: returns full answer at once
- POST /query/stream: streams tokens via Server-Sent Events (SSE)
"""

from typing import AsyncGenerator
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from rag_mentor_platform.retrieval.retriever import retrieve_documents
from rag_mentor_platform.llm.response_generator import generate_answer, format_context, SYSTEM_PROMPT_TEMPLATE
from rag_mentor_platform.core import settings

load_dotenv()

router = APIRouter()


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    context_chunks_used: int


@router.post("/query", response_model=QueryResponse)
def query_sync(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="Query cannot be empty")
    docs = retrieve_documents(request.query)
    result = generate_answer(request.query, docs)
    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        context_chunks_used=result["context_used"],
    )


@router.post("/query/stream")
async def query_stream(request: QueryRequest):
    """
    Streaming endpoint using Server-Sent Events.
    Each token is sent as 'data: <token>\n\n'
    Final messages: [SOURCES]...[/SOURCES] then [DONE]
    """
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="Query cannot be empty")
    docs = retrieve_documents(request.query)
    if not docs:
        async def empty():
            yield "data: No relevant documents found.\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(empty(), media_type="text/event-stream")

    context = format_context(docs)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)

    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.temperature,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        streaming=True,
    )

    async def token_generator() -> AsyncGenerator[str, None]:
        async for chunk in llm.astream([
            SystemMessage(content=system_prompt),
            HumanMessage(content=request.query),
        ]):
            if chunk.content:
                yield f"data:{chunk.content}\n\n"
        sources = ",".join(sorted(set(
            doc.metadata.get("source_file", "Unknown") for doc in docs
        )))
        yield f"data: [SOURCES]{sources}[/SOURCES]\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        token_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
