"""
RAGAS evaluation using the v0.2+ API.

IMPORTANT: RAGAS changed its API in v0.2. The old pattern is broken:
OLD (deprecated):
    from datasets import Dataset
    dataset = Dataset.from_dict({"question": [...], "answer": [...]})
    evaluate(dataset, metrics=[faithfulness, ...])

NEW (current v0.2+):
    from ragas import SingleTurnSample, EvaluationDataset, evaluate
    from ragas.metrics import Faithfulness, AnswerRelevancy, ...
    samples = [SingleTurnSample(user_input=..., response=..., retrieved_contexts=..., reference=...)]
    dataset = EvaluationDataset(samples=samples)
    evaluate(dataset=dataset, metrics=[Faithfulness(llm=evaluator_llm)])

Note that metric classes are now capitalized (Faithfulness, not faithfulness)
and take an explicit llm argument.

What each metric tells you:
- Faithfulness (> 0.80 target): are model claims supported by context?
  Low → model is hallucinating or ignoring retrieved docs.
- AnswerRelevancy: does the answer address the question?
  Low → model is going off-topic.
- ContextPrecision: are retrieved chunks actually relevant?
  Low → retrieval is noisy; improve embedding or reranker.
- ContextRecall: did retrieval find all needed information?
  Low → relevant chunks are missing; check chunking.
"""

import json
import asyncio
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from src.retrieval import retrieve_documents
from src.generation import generate_answer
from src.config import settings

load_dotenv()


def load_golden_dataset(path: str = "golden_dataset.json") -> list[dict]:
    """Load golden Q&A pairs from JSON."""
    with open(path) as f:
        return json.load(f)


async def run_evaluation(golden_path: str = "golden_dataset.json") -> dict:
    """Run RAGAS evaluation on the full pipeline using the v0.2 API."""
    # Lazy imports: ragas has heavy deps that may not be installed
    from ragas import SingleTurnSample, EvaluationDataset, evaluate
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall

    golden = load_golden_dataset(golden_path)
    print(f"Evaluating on {len(golden)} questions...")

    # Evaluator LLM and embeddings - used by RAGAS internally to compute metrics
    # (separate from the LLM used to generate answers in your pipeline)
    evaluator_llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
    )
    evaluator_emb = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
    )

    # Initialize metrics with explicit evaluator (v0.2 pattern)
    metrics = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_emb),
        ContextPrecision(llm=evaluator_llm),
        ContextRecall(llm=evaluator_llm),
    ]

    samples = []
    for i, item in enumerate(golden):
        question = item["question"]
        ground_truth = item["ground_truth"]
        print(f"\n  [{i+1}/{len(golden)}] {question[:60]}...")

        docs = retrieve_documents(question)
        result = generate_answer(question, docs)

        # SingleTurnSample is the v0.2 data structure
        sample = SingleTurnSample(
            user_input=question,
            retrieved_contexts=[doc.page_content for doc in docs],
            response=result["answer"],
            reference=ground_truth,  # needed for ContextRecall
        )
        samples.append(sample)

    dataset = EvaluationDataset(samples=samples)

    print("\nRunning RAGAS metrics (makes LLM calls - takes a few minutes)...")
    results = evaluate(dataset=dataset, metrics=metrics)
    score_dict = results.to_pandas().mean().to_dict()

    print("\n" + "=" * 55)
    print("RAGAS RESULTS")
    print("=" * 55)
    for metric, score in score_dict.items():
        status = "✓ PASS" if score >= 0.80 else "✗ BELOW 0.80"
        print(f"{metric:<28}{score:.3f}  {status}")
    print("=" * 55)
    return score_dict


if __name__ == "__main__":
    asyncio.run(run_evaluation())


# golden_dataset.json - starter (replace with your actual Q&A):
# [
#   {
#     "question": "What is the main topic covered in the uploaded documents?",
#     "ground_truth": "Replace this with the actual correct answer from your document."
#   },
#   {
#     "question": "What year was the policy introduced?",
#     "ground_truth": "Replace this with the actual year from your document."
#   }
# ]
