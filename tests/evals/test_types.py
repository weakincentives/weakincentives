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

"""Tests for eval core types."""

from __future__ import annotations

from weakincentives.evals import (
    PASSING_RATINGS,
    RATING_VALUES,
    EvalReport,
    EvalResult,
    JudgeOutput,
    Sample,
    SampleEvaluated,
    Score,
)

# =============================================================================
# Sample Tests
# =============================================================================


def test_sample_with_string_types() -> None:
    """Sample works with string types."""
    sample: Sample[str, str] = Sample(id="1", input="hello", expected="world")
    assert sample.id == "1"
    assert sample.input == "hello"
    assert sample.expected == "world"


def test_sample_with_int_types() -> None:
    """Sample works with int types."""
    sample: Sample[int, int] = Sample(id="math-1", input=5, expected=10)
    assert sample.input == 5
    assert sample.expected == 10


def test_sample_frozen() -> None:
    """Sample is frozen (immutable)."""
    sample = Sample(id="1", input="a", expected="b")
    try:
        sample.id = "2"  # type: ignore[misc]
        raise AssertionError("Should have raised")
    except AttributeError:
        pass


# =============================================================================
# Score Tests
# =============================================================================


def test_score_passed() -> None:
    """Score represents a passing result."""
    score = Score(value=1.0, passed=True)
    assert score.value == 1.0
    assert score.passed is True
    assert score.reason == ""


def test_score_failed() -> None:
    """Score represents a failing result."""
    score = Score(value=0.0, passed=False, reason="Did not match")
    assert score.value == 0.0
    assert score.passed is False
    assert score.reason == "Did not match"


def test_score_partial() -> None:
    """Score can have partial values."""
    score = Score(value=0.5, passed=False, reason="Partial match")
    assert score.value == 0.5
    assert score.passed is False


# =============================================================================
# JudgeOutput Tests
# =============================================================================


def test_judge_output_excellent() -> None:
    """JudgeOutput with excellent rating."""
    output = JudgeOutput(rating="excellent", reason="Perfect answer")
    assert output.rating == "excellent"
    assert output.reason == "Perfect answer"


def test_judge_output_wrong() -> None:
    """JudgeOutput with wrong rating."""
    output = JudgeOutput(rating="wrong", reason="Completely incorrect")
    assert output.rating == "wrong"


# =============================================================================
# Rating Constants Tests
# =============================================================================


def test_rating_values() -> None:
    """Rating values are correctly defined."""
    assert RATING_VALUES["excellent"] == 1.0
    assert RATING_VALUES["good"] == 0.75
    assert RATING_VALUES["fair"] == 0.5
    assert RATING_VALUES["poor"] == 0.25
    assert RATING_VALUES["wrong"] == 0.0


def test_passing_ratings() -> None:
    """Passing ratings include excellent and good."""
    assert "excellent" in PASSING_RATINGS
    assert "good" in PASSING_RATINGS
    assert "fair" not in PASSING_RATINGS
    assert "poor" not in PASSING_RATINGS
    assert "wrong" not in PASSING_RATINGS


# =============================================================================
# EvalResult Tests
# =============================================================================


def test_eval_result_success() -> None:
    """EvalResult for successful evaluation."""
    score = Score(value=1.0, passed=True)
    result = EvalResult(sample_id="1", score=score, latency_ms=100)
    assert result.sample_id == "1"
    assert result.score is score
    assert result.latency_ms == 100
    assert result.error is None
    assert result.success is True


def test_eval_result_error() -> None:
    """EvalResult for errored evaluation."""
    score = Score(value=0.0, passed=False, reason="Error occurred")
    result = EvalResult(
        sample_id="2", score=score, latency_ms=50, error="Connection failed"
    )
    assert result.error == "Connection failed"
    assert result.success is False


# =============================================================================
# EvalReport Tests
# =============================================================================


def test_eval_report_empty() -> None:
    """EvalReport with no results."""
    report = EvalReport(results=())
    assert report.total == 0
    assert report.successful == 0
    assert report.pass_rate == 0.0
    assert report.mean_score == 0.0
    assert report.mean_latency_ms == 0.0
    assert report.failed_samples() == ()


def test_eval_report_all_pass() -> None:
    """EvalReport with all passing results."""
    results = (
        EvalResult(sample_id="1", score=Score(value=1.0, passed=True), latency_ms=100),
        EvalResult(sample_id="2", score=Score(value=1.0, passed=True), latency_ms=200),
    )
    report = EvalReport(results=results)
    assert report.total == 2
    assert report.successful == 2
    assert report.pass_rate == 1.0
    assert report.mean_score == 1.0
    assert report.mean_latency_ms == 150.0
    assert report.failed_samples() == ()


def test_eval_report_mixed() -> None:
    """EvalReport with mixed results."""
    results = (
        EvalResult(sample_id="1", score=Score(value=1.0, passed=True), latency_ms=100),
        EvalResult(sample_id="2", score=Score(value=0.0, passed=False), latency_ms=100),
        EvalResult(sample_id="3", score=Score(value=0.5, passed=False), latency_ms=100),
    )
    report = EvalReport(results=results)
    assert report.total == 3
    assert report.successful == 3
    assert report.pass_rate == 1 / 3
    assert report.mean_score == 0.5
    assert len(report.failed_samples()) == 2


def test_eval_report_with_errors() -> None:
    """EvalReport with errored samples."""
    results = (
        EvalResult(sample_id="1", score=Score(value=1.0, passed=True), latency_ms=100),
        EvalResult(
            sample_id="2",
            score=Score(value=0.0, passed=False, reason="error"),
            latency_ms=50,
            error="timeout",
        ),
    )
    report = EvalReport(results=results)
    assert report.total == 2
    assert report.successful == 1  # Only non-error samples
    assert report.pass_rate == 1.0  # Only successful sample passed
    assert report.mean_score == 1.0  # Only from successful sample
    assert report.mean_latency_ms == 75.0  # All results count for latency


def test_eval_report_failed_samples() -> None:
    """EvalReport.failed_samples excludes errors."""
    results = (
        EvalResult(sample_id="1", score=Score(value=1.0, passed=True), latency_ms=100),
        EvalResult(sample_id="2", score=Score(value=0.0, passed=False), latency_ms=100),
        EvalResult(
            sample_id="3",
            score=Score(value=0.0, passed=False, reason="err"),
            latency_ms=100,
            error="timeout",
        ),
    )
    report = EvalReport(results=results)
    failed = report.failed_samples()
    assert len(failed) == 1
    assert failed[0].sample_id == "2"


# =============================================================================
# SampleEvaluated Tests
# =============================================================================


def test_sample_evaluated_event() -> None:
    """SampleEvaluated event stores sample_id and result."""
    result = EvalResult(
        sample_id="1", score=Score(value=1.0, passed=True), latency_ms=100
    )
    event = SampleEvaluated(sample_id="1", result=result)
    assert event.sample_id == "1"
    assert event.result is result


# =============================================================================
# Module dir Tests
# =============================================================================


def test_module_dir() -> None:
    """Module __dir__ returns sorted exports."""
    import weakincentives.evals as evals_module

    exports = dir(evals_module)
    assert "Sample" in exports
    assert "Score" in exports
    assert "run_eval" in exports
