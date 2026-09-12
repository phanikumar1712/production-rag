"""
RAGAS evaluation pipeline (ragas 0.3.x).

End-to-end evaluation: for each question in golden_dataset.json it
1. retrieves context via the production retriever (Qdrant hybrid + Cohere rerank),
2. generates an answer via the production generator (citation-enforced prompt),
3. scores the results with RAGAS:
   - faithfulness        (answer claims supported by context?)
   - answer_relevancy    (answer addresses the question?)
   - context_precision   (retrieved chunks actually relevant?)
   - context_recall      (all needed information retrieved?)

Requires Qdrant running and OPENROUTER_API_KEY set (COHERE_API_KEY optional).
Run manually:

    python -m rag_mentor_platform.evaluation.ragas_eval
    python -m rag_mentor_platform.evaluation.ragas_eval --limit 4
    python -m rag_mentor_platform.evaluation.ragas_eval --dataset path/to/golden_dataset.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Optional

from ragas import RunConfig, evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from rag_mentor_platform.core import (
    EmptyContextError,
    LLMGenerationError,
    RetrievalError,
    settings,
)

logger = logging.getLogger(__name__)

DEFAULT_DATASET_PATH = Path(__file__).resolve().parents[2] / "golden_dataset.json"

METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]

# Metrics considered passing at/above this score (matches README targets).
TARGETS = {
    "faithfulness": 0.80,
    "answer_relevancy": 0.80,
    "context_precision": 0.80,
    "context_recall": 0.80,
}


def load_golden_dataset(path: str | Path) -> list[dict]:
    """Load and validate the golden dataset (question/ground_truth pairs)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Golden dataset not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    if not isinstance(dataset, list) or not dataset:
        raise ValueError(f"Golden dataset must be a non-empty JSON array: {path}")
    for i, item in enumerate(dataset):
        if not item.get("question") or not item.get("ground_truth"):
            raise ValueError(f"Dataset item {i} needs 'question' and 'ground_truth'")
    return dataset


def run_rag_pipeline(questions: list[str]) -> tuple[list[str], list[list[str]], list[str]]:
    """
    Run the production RAG pipeline over all questions.

    Returns:
        (answers, contexts, failures) where contexts[i] is the list of
        retrieved chunk texts for questions[i], and failures lists the
        questions that could not be answered (pipeline errors).
    """
    # Imported lazily so importing this module never requires Qdrant to be up.
    from rag_mentor_platform.llm.response_generator import generate_answer
    from rag_mentor_platform.retrieval.retriever import retrieve_documents

    answers: list[str] = []
    contexts: list[list[str]] = []
    failures: list[str] = []

    for question in questions:
        try:
            docs = retrieve_documents(question)
            result = generate_answer(question, docs)
            answers.append(result["answer"])
            contexts.append([doc.page_content for doc in docs])
        except (RetrievalError, EmptyContextError, LLMGenerationError) as e:
            logger.error("Pipeline failed for question %r: %s", question, e)
            failures.append(question)
            answers.append("")
            contexts.append([])

    return answers, contexts, failures


def build_samples(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> tuple[list[SingleTurnSample], int]:
    """
    Build RAGAS evaluation samples, skipping rows the pipeline could not
    answer (empty response or no retrieved context would fail validation).
    """
    samples: list[SingleTurnSample] = []
    skipped = 0
    for question, answer, ctx, reference in zip(questions, answers, contexts, ground_truths):
        if not answer.strip() or not ctx:
            skipped += 1
            continue
        samples.append(
            SingleTurnSample(
                user_input=question,
                retrieved_contexts=ctx,
                response=answer,
                reference=reference,
            )
        )
    return samples, skipped


def _mean(values: list[float]) -> float:
    """Mean that ignores NaN entries (ragas emits NaN on metric failures)."""
    valid = [v for v in values if v == v]  # NaN != NaN
    return sum(valid) / len(valid) if valid else 0.0


def _scores_from_result(result: Any, metrics: list) -> dict[str, float]:
    """Extract {metric_name: mean_score} from a ragas EvaluationResult."""
    repr_dict = getattr(result, "_repr_dict", None)
    if repr_dict:
        return {name: float(score) for name, score in repr_dict.items()}
    # Fallback for ragas versions without _repr_dict
    return {m.name: _mean(result[m.name]) for m in metrics}


class RAGASEvaluator:
    """RAGAS evaluator wired to the project's OpenRouter-based judge models."""

    def __init__(self, llm=None, embeddings=None, run_config: Optional[RunConfig] = None):
        """
        Args:
            llm: Optional pre-built judge LLM (defaults to settings.llm_model via OpenRouter).
            embeddings: Optional judge embeddings (defaults to text-embedding-3-large).
            run_config: Optional ragas RunConfig (timeouts/retries for the eval run).
        """
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        self.llm = llm or LangchainLLMWrapper(
            ChatOpenAI(
                model=settings.llm_model,
                temperature=0.0,  # judge must be deterministic
                api_key=settings.openrouter_api_key,
                base_url=settings.openrouter_base_url,
            )
        )
        self.embeddings = embeddings or LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model="text-embedding-3-large",
                api_key=settings.openrouter_api_key,
                base_url=settings.openrouter_base_url,
            )
        )
        self.run_config = run_config or RunConfig(timeout=120, max_retries=3)
        self.metrics = METRICS

    def evaluate_samples(self, samples: list[SingleTurnSample]) -> dict[str, Any]:
        """Score pre-built samples; returns metric means plus sample counts."""
        if not samples:
            raise LLMGenerationError(
                "No evaluable samples: every question failed before reaching RAGAS. "
                "Check that Qdrant is running and OPENROUTER_API_KEY is valid."
            )
        result = evaluate(
            dataset=EvaluationDataset(samples=samples),
            metrics=self.metrics,
            llm=self.llm,
            embeddings=self.embeddings,
            run_config=self.run_config,
            show_progress=True,
        )
        report: dict[str, Any] = _scores_from_result(result, self.metrics)
        report["num_samples"] = len(samples)
        report["pass"] = {
            name: report.get(name, 0.0) >= target for name, target in TARGETS.items()
        }
        return report


def run_evaluation(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    answer_pipeline: Optional[Callable[[str], str]] = None,
    limit: Optional[int] = None,
) -> dict[str, Any]:
    """
    Convenience entry point: evaluate the RAG system against a golden dataset.

    Args:
        dataset_path: Path to golden_dataset.json.
        answer_pipeline: Optional custom callable (question -> answer) used
            instead of the built-in retrieve+generate pipeline.
        limit: Optional cap on the number of questions evaluated.

    Returns:
        Report dict: metric means, num_samples, skipped, failures, pass flags.
    """
    dataset = load_golden_dataset(dataset_path)
    if limit:
        dataset = dataset[:limit]

    questions = [item["question"] for item in dataset]
    ground_truths = [item["ground_truth"] for item in dataset]

    if answer_pipeline is not None:
        answers: list[str] = []
        contexts: list[list[str]] = []
        failures: list[str] = []
        for question in questions:
            try:
                answers.append(answer_pipeline(question))
                contexts.append([""])  # unknown context; only relevancy-style metrics valid
            except Exception as e:
                logger.error("Custom pipeline failed for %r: %s", question, e)
                failures.append(question)
                answers.append("")
                contexts.append([])
    else:
        answers, contexts, failures = run_rag_pipeline(questions)

    samples, skipped = build_samples(questions, answers, contexts, ground_truths)
    evaluator = RAGASEvaluator()
    report = evaluator.evaluate_samples(samples)
    report["skipped"] = skipped
    report["failures"] = failures
    return report


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on the golden dataset")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH), help="Path to golden_dataset.json")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N questions")
    args = parser.parse_args()

    logger.info("RAGAS evaluation | dataset=%s | model=%s", args.dataset, settings.llm_model)
    report = run_evaluation(dataset_path=args.dataset, limit=args.limit)

    print("\n=== RAGAS Evaluation Results ===")
    for name in TARGETS:
        score = report.get(name, 0.0)
        flag = "PASS" if report["pass"].get(name) else "FAIL"
        print(f"  {name:<20} {score:>7.3f}  [{flag}] (target >= {TARGETS[name]})")
    print(f"  samples evaluated: {report['num_samples']}, skipped: {report.get('skipped', 0)}")
    if report.get("failures"):
        print(f"  pipeline failures: {len(report['failures'])}")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nEvaluation cancelled")
        sys.exit(130)
