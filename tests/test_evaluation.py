"""Tests for the RAGAS evaluation module (no network: dataset handling, sample building, score extraction)."""

import json
from types import SimpleNamespace

import pytest

from rag_mentor_platform.evaluation.ragas_eval import (
    _mean,
    _scores_from_result,
    build_samples,
    load_golden_dataset,
)


class TestLoadGoldenDataset:
    def test_loads_real_dataset(self):
        from pathlib import Path

        dataset = load_golden_dataset(
            Path(__file__).parent.parent / "golden_dataset.json"
        )
        assert len(dataset) == 8
        assert all("question" in item and "ground_truth" in item for item in dataset)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_golden_dataset(tmp_path / "nope.json")

    def test_invalid_items_raise(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps([{"question": "q only"}]))
        with pytest.raises(ValueError, match="ground_truth"):
            load_golden_dataset(path)

    def test_empty_dataset_raises(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text("[]")
        with pytest.raises(ValueError, match="non-empty"):
            load_golden_dataset(path)


class TestBuildSamples:
    def test_builds_samples_from_valid_rows(self):
        samples, skipped = build_samples(
            questions=["q1", "q2"],
            answers=["a1", "a2"],
            contexts=[["c1"], ["c2"]],
            ground_truths=["g1", "g2"],
        )
        assert skipped == 0
        assert len(samples) == 2
        assert samples[0].user_input == "q1"
        assert samples[0].retrieved_contexts == ["c1"]
        assert samples[0].response == "a1"
        assert samples[0].reference == "g1"

    def test_skips_failed_pipeline_rows(self):
        samples, skipped = build_samples(
            questions=["ok", "failed"],
            answers=["answer", ""],  # empty answer = pipeline failure
            contexts=[["ctx"], []],  # empty context = retrieval failure
            ground_truths=["g1", "g2"],
        )
        assert skipped == 1
        assert len(samples) == 1
        assert samples[0].user_input == "ok"


class TestScoreExtraction:
    def test_mean_ignores_nan(self):
        nan = float("nan")
        assert _mean([0.5, nan, 1.0]) == pytest.approx(0.75)
        assert _mean([nan, nan]) == 0.0
        assert _mean([]) == 0.0

    def test_scores_from_repr_dict(self):
        result = SimpleNamespace(
            _repr_dict={"faithfulness": 0.9, "answer_relevancy": 0.85}
        )
        scores = _scores_from_result(result, metrics=[])
        assert scores == {"faithfulness": 0.9, "answer_relevancy": 0.85}

    def test_scores_fallback_per_question_lists(self):
        class _FakeResult:
            def __getitem__(self, key):
                return {"faithfulness": [1.0, 0.0]}[key]

        metric = SimpleNamespace(name="faithfulness")
        scores = _scores_from_result(_FakeResult(), metrics=[metric])
        assert scores["faithfulness"] == pytest.approx(0.5)
