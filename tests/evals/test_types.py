# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for evals core types."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from uuid import UUID

import pytest

import weakincentives.evals as evals_module
from weakincentives.evals import (
    BASELINE,
    Dataset,
    EvalReport,
    EvalRequest,
    EvalResult,
    Experiment,
    ExperimentComparison,
    Sample,
    Score,
)
from weakincentives.evals._types import _coerce

# =============================================================================
# Module Tests
# =============================================================================


def test_module_dir() -> None:
    """Module __dir__ includes all exports."""
    module_dir = dir(evals_module)
    # Should include all __all__ exports
    assert "Sample" in module_dir
    assert "Dataset" in module_dir
    assert "Score" in module_dir
    assert "EvalResult" in module_dir
    assert "EvalReport" in module_dir
    assert "EvalLoop" in module_dir
    assert "exact_match" in module_dir
    assert "contains" in module_dir
    assert "all_of" in module_dir
    assert "any_of" in module_dir
    assert "llm_judge" in module_dir
    # Should also include module attributes
    assert "__all__" in module_dir


# =============================================================================
# Sample Tests
# =============================================================================


def test_sample_creation() -> None:
    """Sample can be created with typed fields."""
    sample = Sample(id="1", input="What is 2+2?", expected="4")
    assert sample.id == "1"
    assert sample.input == "What is 2+2?"
    assert sample.expected == "4"


def test_sample_is_frozen() -> None:
    """Sample is immutable."""
    sample = Sample(id="1", input="hello", expected="world")
    with pytest.raises(AttributeError):
        sample.id = "2"  # type: ignore[misc]


def test_sample_with_complex_types() -> None:
    """Sample supports complex generic types."""

    @dataclass(slots=True, frozen=True)
    class MathProblem:
        a: int
        b: int
        operation: str

    sample = Sample(
        id="1",
        input=MathProblem(a=2, b=3, operation="+"),
        expected=5,
    )
    assert sample.input.a == 2
    assert sample.input.b == 3
    assert sample.expected == 5


# =============================================================================
# Score Tests
# =============================================================================


def test_score_creation() -> None:
    """Score can be created with default reason."""
    score = Score(value=0.8, passed=True)
    assert score.value == 0.8
    assert score.passed is True
    assert score.reason == ""


def test_score_with_reason() -> None:
    """Score can include an explanation."""
    score = Score(value=0.5, passed=False, reason="Missing key details")
    assert score.reason == "Missing key details"


def test_score_is_frozen() -> None:
    """Score is immutable."""
    score = Score(value=1.0, passed=True)
    with pytest.raises(AttributeError):
        score.value = 0.5  # type: ignore[misc]


# =============================================================================
# Dataset Tests
# =============================================================================


def test_dataset_creation() -> None:
    """Dataset can be created with samples tuple."""
    samples = (
        Sample(id="1", input="a", expected="b"),
        Sample(id="2", input="c", expected="d"),
    )
    dataset = Dataset(samples=samples)
    assert len(dataset) == 2


def test_dataset_iteration() -> None:
    """Dataset supports iteration."""
    samples = (
        Sample(id="1", input="a", expected="b"),
        Sample(id="2", input="c", expected="d"),
    )
    dataset = Dataset(samples=samples)
    ids = [s.id for s in dataset]
    assert ids == ["1", "2"]


def test_dataset_indexing() -> None:
    """Dataset supports index access."""
    samples = (
        Sample(id="1", input="a", expected="b"),
        Sample(id="2", input="c", expected="d"),
    )
    dataset = Dataset(samples=samples)
    assert dataset[0].id == "1"
    assert dataset[1].id == "2"


def test_dataset_is_frozen() -> None:
    """Dataset is immutable."""
    samples = (Sample(id="1", input="a", expected="b"),)
    dataset = Dataset(samples=samples)
    with pytest.raises(AttributeError):
        dataset.samples = ()  # type: ignore[misc]


def test_dataset_load_from_jsonl() -> None:
    """Dataset.load reads JSONL files."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"id": "1", "input": "What is 2+2?", "expected": "4"}\n')
        f.write('{"id": "2", "input": "Capital of France?", "expected": "Paris"}\n')
        path = Path(f.name)

    try:
        dataset = Dataset.load(path, str, str)
        assert len(dataset) == 2
        assert dataset[0].id == "1"
        assert dataset[0].input == "What is 2+2?"
        assert dataset[0].expected == "4"
        assert dataset[1].id == "2"
    finally:
        path.unlink()


def test_dataset_load_with_complex_types() -> None:
    """Dataset.load deserializes complex types."""

    @dataclass(slots=True, frozen=True)
    class Problem:
        question: str
        difficulty: int

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(
            json.dumps(
                {
                    "id": "1",
                    "input": {"question": "What is 2+2?", "difficulty": 1},
                    "expected": "4",
                }
            )
            + "\n"
        )
        path = Path(f.name)

    try:
        dataset = Dataset.load(path, Problem, str)
        assert len(dataset) == 1
        assert dataset[0].input.question == "What is 2+2?"
        assert dataset[0].input.difficulty == 1
    finally:
        path.unlink()


# =============================================================================
# _coerce Tests
# =============================================================================


def test_coerce_string() -> None:
    """_coerce handles string primitives."""
    result = _coerce("hello", str)
    assert result == "hello"


def test_coerce_int() -> None:
    """_coerce handles int primitives."""
    result = _coerce(42, int)
    assert result == 42


def test_coerce_float() -> None:
    """_coerce handles float primitives."""
    result = _coerce(3.14, float)
    assert result == 3.14


def test_coerce_bool() -> None:
    """_coerce handles bool primitives."""
    result = _coerce(True, bool)
    assert result is True


def test_coerce_type_mismatch_raises() -> None:
    """_coerce raises TypeError on type mismatch."""
    with pytest.raises(TypeError, match="expected str, got int"):
        _coerce(42, str)


def test_coerce_mapping_to_dataclass() -> None:
    """_coerce parses mappings into dataclasses."""

    @dataclass(slots=True, frozen=True)
    class Point:
        x: int
        y: int

    result = _coerce({"x": 1, "y": 2}, Point)
    assert result == Point(x=1, y=2)


def test_coerce_invalid_value_type_raises() -> None:
    """_coerce raises TypeError for non-primitive, non-mapping values."""

    @dataclass(slots=True, frozen=True)
    class Point:
        x: int
        y: int

    with pytest.raises(TypeError, match="cannot coerce list to Point"):
        _coerce([1, 2], Point)


# =============================================================================
# EvalResult Tests
# =============================================================================


def test_eval_result_creation() -> None:
    """EvalResult can be created with required fields."""
    result = EvalResult(
        sample_id="1",
        experiment_name="baseline",
        score=Score(value=1.0, passed=True),
        latency_ms=150,
    )
    assert result.sample_id == "1"
    assert result.experiment_name == "baseline"
    assert result.score.value == 1.0
    assert result.latency_ms == 150
    assert result.error is None
    assert result.success is True


def test_eval_result_with_error() -> None:
    """EvalResult with error is not successful."""
    result = EvalResult(
        sample_id="1",
        experiment_name="baseline",
        score=Score(value=0.0, passed=False),
        latency_ms=0,
        error="Execution failed",
    )
    assert result.success is False
    assert result.error == "Execution failed"


def test_eval_result_is_frozen() -> None:
    """EvalResult is immutable."""
    result = EvalResult(
        sample_id="1",
        experiment_name="baseline",
        score=Score(value=1.0, passed=True),
        latency_ms=100,
    )
    with pytest.raises(AttributeError):
        result.sample_id = "2"  # type: ignore[misc]


# =============================================================================
# EvalReport Tests
# =============================================================================


def test_eval_report_empty() -> None:
    """EvalReport handles empty results."""
    report = EvalReport(results=())
    assert report.total == 0
    assert report.successful == 0
    assert report.pass_rate == 0.0
    assert report.mean_score == 0.0
    assert report.mean_latency_ms == 0.0
    assert report.failed_samples() == ()


def test_eval_report_all_passing() -> None:
    """EvalReport correctly calculates metrics for all passing."""
    results = (
        EvalResult(
            sample_id="1",
            experiment_name="baseline",
            score=Score(value=1.0, passed=True),
            latency_ms=100,
        ),
        EvalResult(
            sample_id="2",
            experiment_name="baseline",
            score=Score(value=0.8, passed=True),
            latency_ms=200,
        ),
    )
    report = EvalReport(results=results)
    assert report.total == 2
    assert report.successful == 2
    assert report.pass_rate == 1.0
    assert report.mean_score == 0.9
    assert report.mean_latency_ms == 150.0
    assert report.failed_samples() == ()


def test_eval_report_mixed_results() -> None:
    """EvalReport correctly calculates metrics for mixed results."""
    results = (
        EvalResult(
            sample_id="1",
            experiment_name="baseline",
            score=Score(value=1.0, passed=True),
            latency_ms=100,
        ),
        EvalResult(
            sample_id="2",
            experiment_name="baseline",
            score=Score(value=0.3, passed=False),
            latency_ms=200,
        ),
    )
    report = EvalReport(results=results)
    assert report.total == 2
    assert report.successful == 2
    assert report.pass_rate == 0.5  # 1 of 2 passed
    assert report.mean_score == 0.65  # (1.0 + 0.3) / 2
    assert len(report.failed_samples()) == 1
    assert report.failed_samples()[0].sample_id == "2"


def test_eval_report_with_errors() -> None:
    """EvalReport excludes errors from pass rate calculation."""
    results = (
        EvalResult(
            sample_id="1",
            experiment_name="baseline",
            score=Score(value=1.0, passed=True),
            latency_ms=100,
        ),
        EvalResult(
            sample_id="2",
            experiment_name="baseline",
            score=Score(value=0.0, passed=False),
            latency_ms=0,
            error="Execution failed",
        ),
    )
    report = EvalReport(results=results)
    assert report.total == 2
    assert report.successful == 1  # Only 1 successful
    assert report.pass_rate == 1.0  # 1 of 1 successful passed
    assert report.mean_score == 1.0  # Only successful counts


def test_eval_report_is_frozen() -> None:
    """EvalReport is immutable."""
    report = EvalReport(results=())
    with pytest.raises(AttributeError):
        report.results = ()  # type: ignore[misc]


# =============================================================================
# EvalRequest Tests
# =============================================================================


def test_eval_request_creation() -> None:
    """EvalRequest can be created with sample and experiment."""
    sample = Sample(id="1", input="hello", expected="world")
    request = EvalRequest(sample=sample, experiment=BASELINE)
    assert request.sample is sample
    assert request.experiment is BASELINE
    assert isinstance(request.request_id, UUID)
    assert request.created_at.tzinfo == UTC


def test_eval_request_is_frozen() -> None:
    """EvalRequest is immutable."""
    sample = Sample(id="1", input="hello", expected="world")
    request = EvalRequest(sample=sample, experiment=BASELINE)
    with pytest.raises(AttributeError):
        request.sample = sample  # type: ignore[misc]


# =============================================================================
# Experiment Tests
# =============================================================================


def test_experiment_creation() -> None:
    """Experiment can be created with default values."""
    exp = Experiment(name="test")
    assert exp.name == "test"
    assert exp.overrides_tag == "latest"
    assert exp.flags == {}
    assert exp.owner is None
    assert exp.description is None


def test_experiment_with_all_fields() -> None:
    """Experiment accepts all optional fields."""
    exp = Experiment(
        name="v2-prompts",
        overrides_tag="v2",
        flags={"verbose": True, "retries": 5},
        owner="alice@example.com",
        description="Test concise prompts",
    )
    assert exp.name == "v2-prompts"
    assert exp.overrides_tag == "v2"
    assert exp.flags == {"verbose": True, "retries": 5}
    assert exp.owner == "alice@example.com"
    assert exp.description == "Test concise prompts"


def test_experiment_with_flag() -> None:
    """with_flag returns new experiment with flag added."""
    exp = Experiment(name="test")
    exp2 = exp.with_flag("debug", True)
    assert exp.flags == {}  # Original unchanged
    assert exp2.flags == {"debug": True}


def test_experiment_with_tag() -> None:
    """with_tag returns new experiment with different tag."""
    exp = Experiment(name="test")
    exp2 = exp.with_tag("v2")
    assert exp.overrides_tag == "latest"  # Original unchanged
    assert exp2.overrides_tag == "v2"


def test_experiment_get_flag_existing() -> None:
    """get_flag returns value for existing flag."""
    exp = Experiment(name="test", flags={"retries": 5})
    assert exp.get_flag("retries") == 5


def test_experiment_get_flag_default() -> None:
    """get_flag returns default for missing flag."""
    exp = Experiment(name="test")
    assert exp.get_flag("retries", 3) == 3
    assert exp.get_flag("retries") is None


def test_experiment_has_flag() -> None:
    """has_flag checks flag existence including False/None values."""
    exp = Experiment(name="test", flags={"debug": False, "count": None})
    assert exp.has_flag("debug") is True
    assert exp.has_flag("count") is True
    assert exp.has_flag("missing") is False


def test_baseline_sentinel() -> None:
    """BASELINE sentinel has correct defaults."""
    assert BASELINE.name == "baseline"
    assert BASELINE.overrides_tag == "latest"
    assert BASELINE.flags == {}


# =============================================================================
# ExperimentComparison Tests
# =============================================================================


def test_experiment_comparison_pass_rates() -> None:
    """ExperimentComparison calculates pass rates correctly."""
    baseline_results = (
        EvalResult(
            sample_id="1",
            experiment_name="baseline",
            score=Score(value=1.0, passed=True),
            latency_ms=100,
        ),
        EvalResult(
            sample_id="2",
            experiment_name="baseline",
            score=Score(value=0.5, passed=False),
            latency_ms=100,
        ),
    )
    treatment_results = (
        EvalResult(
            sample_id="1",
            experiment_name="treatment",
            score=Score(value=1.0, passed=True),
            latency_ms=100,
        ),
        EvalResult(
            sample_id="2",
            experiment_name="treatment",
            score=Score(value=1.0, passed=True),
            latency_ms=100,
        ),
    )
    comparison = ExperimentComparison(
        baseline_name="baseline",
        treatment_name="treatment",
        baseline_results=baseline_results,
        treatment_results=treatment_results,
    )
    assert comparison.baseline_pass_rate == 0.5
    assert comparison.treatment_pass_rate == 1.0
    assert comparison.pass_rate_delta == 0.5
    assert comparison.relative_improvement == 1.0  # 100% improvement


def test_experiment_comparison_zero_baseline() -> None:
    """ExperimentComparison handles zero baseline pass rate."""
    baseline_results = (
        EvalResult(
            sample_id="1",
            experiment_name="baseline",
            score=Score(value=0.0, passed=False),
            latency_ms=100,
        ),
    )
    treatment_results = (
        EvalResult(
            sample_id="1",
            experiment_name="treatment",
            score=Score(value=1.0, passed=True),
            latency_ms=100,
        ),
    )
    comparison = ExperimentComparison(
        baseline_name="baseline",
        treatment_name="treatment",
        baseline_results=baseline_results,
        treatment_results=treatment_results,
    )
    assert comparison.baseline_pass_rate == 0.0
    assert comparison.treatment_pass_rate == 1.0
    assert comparison.relative_improvement is None  # Cannot divide by zero


# =============================================================================
# EvalReport Experiment Methods Tests
# =============================================================================


def test_eval_report_by_experiment() -> None:
    """by_experiment groups results by experiment name."""
    results = (
        EvalResult(
            sample_id="1",
            experiment_name="baseline",
            score=Score(value=1.0, passed=True),
            latency_ms=100,
        ),
        EvalResult(
            sample_id="2",
            experiment_name="baseline",
            score=Score(value=0.8, passed=True),
            latency_ms=100,
        ),
        EvalResult(
            sample_id="1",
            experiment_name="treatment",
            score=Score(value=1.0, passed=True),
            latency_ms=100,
        ),
    )
    report = EvalReport(results=results)
    by_exp = report.by_experiment()
    assert len(by_exp) == 2
    assert len(by_exp["baseline"]) == 2
    assert len(by_exp["treatment"]) == 1


def test_eval_report_pass_rate_by_experiment() -> None:
    """pass_rate_by_experiment computes per-experiment pass rates."""
    results = (
        EvalResult(
            sample_id="1",
            experiment_name="baseline",
            score=Score(value=1.0, passed=True),
            latency_ms=100,
        ),
        EvalResult(
            sample_id="2",
            experiment_name="baseline",
            score=Score(value=0.5, passed=False),
            latency_ms=100,
        ),
        EvalResult(
            sample_id="1",
            experiment_name="treatment",
            score=Score(value=1.0, passed=True),
            latency_ms=100,
        ),
        EvalResult(
            sample_id="2",
            experiment_name="treatment",
            score=Score(value=1.0, passed=True),
            latency_ms=100,
        ),
    )
    report = EvalReport(results=results)
    rates = report.pass_rate_by_experiment()
    assert rates["baseline"] == 0.5
    assert rates["treatment"] == 1.0


def test_eval_report_mean_score_by_experiment() -> None:
    """mean_score_by_experiment computes per-experiment mean scores."""
    results = (
        EvalResult(
            sample_id="1",
            experiment_name="baseline",
            score=Score(value=1.0, passed=True),
            latency_ms=100,
        ),
        EvalResult(
            sample_id="2",
            experiment_name="baseline",
            score=Score(value=0.5, passed=False),
            latency_ms=100,
        ),
        EvalResult(
            sample_id="1",
            experiment_name="treatment",
            score=Score(value=0.8, passed=True),
            latency_ms=100,
        ),
    )
    report = EvalReport(results=results)
    scores = report.mean_score_by_experiment()
    assert scores["baseline"] == 0.75
    assert scores["treatment"] == 0.8


def test_eval_report_compare_experiments() -> None:
    """compare_experiments returns ExperimentComparison."""
    results = (
        EvalResult(
            sample_id="1",
            experiment_name="baseline",
            score=Score(value=1.0, passed=True),
            latency_ms=100,
        ),
        EvalResult(
            sample_id="2",
            experiment_name="baseline",
            score=Score(value=0.5, passed=False),
            latency_ms=100,
        ),
        EvalResult(
            sample_id="1",
            experiment_name="treatment",
            score=Score(value=1.0, passed=True),
            latency_ms=100,
        ),
        EvalResult(
            sample_id="2",
            experiment_name="treatment",
            score=Score(value=1.0, passed=True),
            latency_ms=100,
        ),
    )
    report = EvalReport(results=results)
    comparison = report.compare_experiments("baseline", "treatment")
    assert comparison.baseline_name == "baseline"
    assert comparison.treatment_name == "treatment"
    assert comparison.baseline_pass_rate == 0.5
    assert comparison.treatment_pass_rate == 1.0
    assert comparison.pass_rate_delta == 0.5


def test_experiment_comparison_empty_results() -> None:
    """ExperimentComparison handles empty results gracefully."""
    comparison = ExperimentComparison(
        baseline_name="baseline",
        treatment_name="treatment",
        baseline_results=(),
        treatment_results=(),
    )
    assert comparison.baseline_pass_rate == 0.0
    assert comparison.treatment_pass_rate == 0.0
    assert comparison.pass_rate_delta == 0.0
    assert comparison.relative_improvement is None


def test_eval_report_pass_rate_by_experiment_with_errors() -> None:
    """pass_rate_by_experiment handles experiments with all errors."""
    results = (
        EvalResult(
            sample_id="1",
            experiment_name="baseline",
            score=Score(value=0.0, passed=False),
            latency_ms=0,
            error="Failed",
        ),
        EvalResult(
            sample_id="2",
            experiment_name="treatment",
            score=Score(value=1.0, passed=True),
            latency_ms=100,
        ),
    )
    report = EvalReport(results=results)
    rates = report.pass_rate_by_experiment()
    assert rates["baseline"] == 0.0  # All errors -> 0.0
    assert rates["treatment"] == 1.0


def test_eval_report_mean_score_by_experiment_with_errors() -> None:
    """mean_score_by_experiment handles experiments with all errors."""
    results = (
        EvalResult(
            sample_id="1",
            experiment_name="baseline",
            score=Score(value=0.0, passed=False),
            latency_ms=0,
            error="Failed",
        ),
        EvalResult(
            sample_id="2",
            experiment_name="treatment",
            score=Score(value=0.8, passed=True),
            latency_ms=100,
        ),
    )
    report = EvalReport(results=results)
    scores = report.mean_score_by_experiment()
    assert scores["baseline"] == 0.0  # All errors -> 0.0
    assert scores["treatment"] == 0.8


# =============================================================================
# EvalReport Debug Bundle Refs Tests
# =============================================================================


def test_eval_report_debug_bundle_refs_empty() -> None:
    """debug_bundle_refs returns empty dict when no bundles exist."""
    results = (
        EvalResult(
            sample_id="1",
            experiment_name="baseline",
            score=Score(value=1.0, passed=True),
            latency_ms=100,
        ),
    )
    report = EvalReport(results=results)
    refs = report.debug_bundle_refs()
    assert refs == {}


def test_eval_report_debug_bundle_refs_with_bundles() -> None:
    """debug_bundle_refs maps (sample_id, experiment) to bundle paths."""
    results = (
        EvalResult(
            sample_id="1",
            experiment_name="baseline",
            score=Score(value=1.0, passed=True),
            latency_ms=100,
            bundle_path=Path("/bundles/1_baseline.zip"),
        ),
        EvalResult(
            sample_id="2",
            experiment_name="baseline",
            score=Score(value=0.8, passed=True),
            latency_ms=100,
            bundle_path=Path("/bundles/2_baseline.zip"),
        ),
        EvalResult(
            sample_id="1",
            experiment_name="treatment",
            score=Score(value=1.0, passed=True),
            latency_ms=100,
            bundle_path=Path("/bundles/1_treatment.zip"),
        ),
    )
    report = EvalReport(results=results)
    refs = report.debug_bundle_refs()
    assert len(refs) == 3
    assert refs["1", "baseline"] == Path("/bundles/1_baseline.zip")
    assert refs["2", "baseline"] == Path("/bundles/2_baseline.zip")
    assert refs["1", "treatment"] == Path("/bundles/1_treatment.zip")


def test_eval_report_debug_bundle_refs_mixed() -> None:
    """debug_bundle_refs only includes results with bundle_path set."""
    results = (
        EvalResult(
            sample_id="1",
            experiment_name="baseline",
            score=Score(value=1.0, passed=True),
            latency_ms=100,
            bundle_path=Path("/bundles/1_baseline.zip"),
        ),
        EvalResult(
            sample_id="2",
            experiment_name="baseline",
            score=Score(value=0.8, passed=True),
            latency_ms=100,
            # No bundle_path
        ),
    )
    report = EvalReport(results=results)
    refs = report.debug_bundle_refs()
    assert len(refs) == 1
    assert refs["1", "baseline"] == Path("/bundles/1_baseline.zip")
    assert ("2", "baseline") not in refs
