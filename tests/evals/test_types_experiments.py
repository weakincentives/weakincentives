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

"""Tests for evals experiment types: EvalRequest, Experiment, ExperimentComparison, EvalReport experiment methods."""

from __future__ import annotations

from datetime import UTC
from uuid import UUID

import pytest

from weakincentives.evals import (
    BASELINE,
    EvalReport,
    EvalRequest,
    EvalResult,
    Experiment,
    ExperimentComparison,
    Sample,
    Score,
)

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
