"""
Generation with citation enforcement.
Key design: the system prompt mandates citations for every factual claim.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from rag_mentor_platform.core import settings, LLMGenerationError, EmptyContextError

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
    Generate a cited answer grounded in provided documents.
    
    Args:
        query: The user's question
        docs: List of retrieved documents with context
        
    Returns:
        Dictionary with answer, sources list, and context_used count
        
    Raises:
        EmptyContextError: If no documents are provided
        LLMGenerationError: If LLM fails to generate response
    """
    if not docs:
        raise EmptyContextError("No documents provided for answer generation")

    try:
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
    except Exception as e:
        raise LLMGenerationError(f"Failed to generate answer: {str(e)}")
