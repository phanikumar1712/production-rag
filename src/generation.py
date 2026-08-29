"""
Generation with citation enforcement.
Key design: the system prompt mandates citations for every factual claim.
This does three things:
1. Forces the model to ground answers in retrieved context
2. Makes hallucinations visible (uncited claims can be flagged)
3. Gives users a way to verify any answer

The "I don't have enough information" instruction is equally important:
it prevents the model from guessing when the context is insufficient.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from src.config import settings

load_dotenv()

SYSTEM_PROMPT_TEMPLATE = """\
You are a precise document analyst answering questions based on provided context.

MANDATORY RULES:
1. Use ONLY the information from the context documents below.
2. Every factual claim MUST have a citation in the format [Source: <filename>]
3. If the context does not contain enough information to answer, respond:
"I don't have enough information in the provided documents to answer this."
4. Be concise and direct.

CONTEXT:
{context}
"""


def format_context(docs: list) -> str:
    """Format retrieved docs as numbered blocks with source metadata."""
    if not docs:
        return "No documents retrieved."
    blocks = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_file", "Unknown")
        relevance = doc.metadata.get("relevance_score", "")
        note = f" [score:{relevance:.3f}]" if relevance else ""
        blocks.append(
            f"[{i}] Source: {source}{note}\n{doc.page_content.strip()}"
        )
    return "\n\n---\n\n".join(blocks)


def generate_answer(query: str, docs: list) -> dict:
    """
    Generate a cited answer.
    Returns dict with answer, sources list, and context_used count.
    """
    if not docs:
        return {"answer": "No relevant documents found.", "sources": [], "context_used": 0}

    context = format_context(docs)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)

    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.temperature,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=query),
    ])

    sources = sorted(set(
        doc.metadata.get("source_file", "Unknown") for doc in docs
    ))

    return {
        "answer": response.content,
        "sources": sources,
        "context_used": len(docs),
    }
