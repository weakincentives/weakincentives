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

"""Serialization/deserialization tests for eval types (Score, Sample, Experiment,
EvalResult, EvalRequest)."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest

from tests.serde.conftest import AnswerExpected, QuestionInput
from weakincentives.evals._types import EvalRequest, EvalResult, Sample, Score
from weakincentives.experiment import Experiment
from weakincentives.serde import dump, parse

pytestmark = pytest.mark.core


# =============================================================================
# Score Tests
# =============================================================================


class TestScoreSerde:
    """Tests for Score serialization/deserialization."""

    def test_score_round_trip(self) -> None:
        """Score serializes and deserializes correctly."""
        score = Score(value=0.85, passed=True, reason="Good answer")

        data = dump(score)
        restored = parse(Score, data)

        assert restored.value == 0.85
        assert restored.passed is True
        assert restored.reason == "Good answer"

    def test_score_minimal(self) -> None:
        """Score with only required fields."""
        score = Score(value=0.5, passed=False)

        data = dump(score)
        restored = parse(Score, data)

        assert restored.value == 0.5
        assert restored.passed is False
        assert restored.reason == ""

    def test_score_parse_from_json(self) -> None:
        """Score parses from JSON-like dict."""
        data = {"value": "0.75", "passed": "true", "reason": "Partial match"}

        restored = parse(Score, data)

        assert restored.value == 0.75
        assert restored.passed is True
        assert restored.reason == "Partial match"


# =============================================================================
# Sample Tests
# =============================================================================


class TestSampleSerde:
    """Tests for Sample serialization/deserialization."""

    def test_sample_with_primitives(self) -> None:
        """Sample with primitive input/expected types."""
        sample: Sample[str, str] = Sample(
            id="test-1",
            input="What is 2+2?",
            expected="4",
        )

        data = dump(sample)
        restored = parse(Sample[str, str], data)

        assert restored.id == "test-1"
        assert restored.input == "What is 2+2?"
        assert restored.expected == "4"

    def test_sample_with_dataclass_types(self) -> None:
        """Sample with dataclass input/expected types."""
        sample: Sample[QuestionInput, AnswerExpected] = Sample(
            id="qa-1",
            input=QuestionInput(question="What is the capital of France?"),
            expected=AnswerExpected(answer="Paris", keywords=("France", "capital")),
        )

        data = dump(sample)
        restored = parse(Sample[QuestionInput, AnswerExpected], data)

        assert restored.id == "qa-1"
        assert isinstance(restored.input, QuestionInput)
        assert restored.input.question == "What is the capital of France?"
        assert isinstance(restored.expected, AnswerExpected)
        assert restored.expected.answer == "Paris"
        assert restored.expected.keywords == ("France", "capital")

    def test_sample_parse_nested_from_json(self) -> None:
        """Sample parses nested dataclass from JSON."""
        data = {
            "id": "qa-2",
            "input": {"question": "What is Python?", "context": "Programming"},
            "expected": {"answer": "A programming language"},
        }

        restored = parse(Sample[QuestionInput, AnswerExpected], data)

        assert restored.id == "qa-2"
        assert restored.input.question == "What is Python?"
        assert restored.input.context == "Programming"
        assert restored.expected.answer == "A programming language"


# =============================================================================
# Experiment Tests
# =============================================================================


class TestExperimentSerde:
    """Tests for Experiment serialization/deserialization."""

    def test_experiment_round_trip(self) -> None:
        """Experiment serializes and deserializes correctly."""
        experiment = Experiment(
            name="v2-prompts",
            overrides_tag="v2",
            flags={"verbose": True, "max_retries": 5},
            owner="alice@example.com",
            description="Test concise prompts",
        )

        data = dump(experiment)
        restored = parse(Experiment, data)

        assert restored.name == "v2-prompts"
        assert restored.overrides_tag == "v2"
        assert restored.flags == {"verbose": True, "max_retries": 5}
        assert restored.owner == "alice@example.com"
        assert restored.description == "Test concise prompts"

    def test_experiment_minimal(self) -> None:
        """Experiment with only required fields."""
        experiment = Experiment(name="baseline")

        data = dump(experiment)
        restored = parse(Experiment, data)

        assert restored.name == "baseline"
        assert restored.overrides_tag == "latest"
        assert restored.flags == {}
        assert restored.owner is None
        assert restored.description is None

    def test_experiment_exclude_none(self) -> None:
        """Experiment dump with exclude_none."""
        experiment = Experiment(name="test")

        data = dump(experiment, exclude_none=True)

        assert "owner" not in data
        assert "description" not in data


# =============================================================================
# EvalResult Tests
# =============================================================================


class TestEvalResultSerde:
    """Tests for EvalResult serialization/deserialization."""

    def test_eval_result_round_trip(self) -> None:
        """EvalResult serializes and deserializes correctly."""
        result = EvalResult(
            sample_id="sample-1",
            experiment_name="baseline",
            score=Score(value=0.9, passed=True),
            latency_ms=150,
        )

        data = dump(result)
        restored = parse(EvalResult, data)

        assert restored.sample_id == "sample-1"
        assert restored.experiment_name == "baseline"
        assert restored.score.value == 0.9
        assert restored.score.passed is True
        assert restored.latency_ms == 150
        assert restored.error is None
        assert restored.success is True

    def test_eval_result_with_error(self) -> None:
        """EvalResult with error field."""
        result = EvalResult(
            sample_id="sample-2",
            experiment_name="test",
            score=Score(value=0.0, passed=False),
            latency_ms=50,
            error="Timeout exceeded",
        )

        data = dump(result)
        restored = parse(EvalResult, data)

        assert restored.error == "Timeout exceeded"
        assert restored.success is False

    def test_eval_result_parse_from_json(self) -> None:
        """EvalResult parses from JSON-like dict with coercion."""
        data = {
            "sample_id": "s1",
            "experiment_name": "exp",
            "score": {"value": "0.5", "passed": "false"},
            "latency_ms": "100",
        }

        restored = parse(EvalResult, data)

        assert restored.sample_id == "s1"
        assert restored.score.value == 0.5
        assert restored.score.passed is False
        assert restored.latency_ms == 100


# =============================================================================
# EvalRequest Tests
# =============================================================================


class TestEvalRequestSerde:
    """Tests for EvalRequest serialization/deserialization."""

    def test_eval_request_with_string_types(self) -> None:
        """EvalRequest with primitive string types."""
        sample: Sample[str, str] = Sample(id="1", input="test", expected="result")
        experiment = Experiment(name="baseline")
        request_id = UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")
        created_at = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)

        request: EvalRequest[str, str] = EvalRequest(
            sample=sample,
            experiment=experiment,
            request_id=request_id,
            created_at=created_at,
        )

        data = dump(request)
        restored = parse(EvalRequest[str, str], data)

        assert restored.sample.id == "1"
        assert restored.sample.input == "test"
        assert restored.sample.expected == "result"
        assert isinstance(restored.experiment, Experiment)
        assert restored.experiment.name == "baseline"
        assert restored.request_id == request_id
        assert restored.created_at == created_at

    def test_eval_request_dump_structure(self) -> None:
        """EvalRequest dump produces correct structure."""
        sample: Sample[str, str] = Sample(id="1", input="test", expected="result")
        experiment = Experiment(name="baseline")
        request_id = UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")
        created_at = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)

        request: EvalRequest[str, str] = EvalRequest(
            sample=sample,
            experiment=experiment,
            request_id=request_id,
            created_at=created_at,
        )

        data = dump(request)

        # Verify structure
        assert data["sample"] == {"id": "1", "input": "test", "expected": "result"}
        assert data["experiment"]["name"] == "baseline"
        assert data["experiment"]["overrides_tag"] == "latest"
        assert data["request_id"] == "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        assert data["created_at"] == "2024-06-15T12:00:00+00:00"

    def test_eval_request_with_dataclass_types(self) -> None:
        """EvalRequest with nested dataclass types."""
        sample: Sample[QuestionInput, AnswerExpected] = Sample(
            id="qa-1",
            input=QuestionInput(question="What is AI?"),
            expected=AnswerExpected(answer="Artificial Intelligence"),
        )
        experiment = Experiment(name="v2", flags={"debug": True})

        request: EvalRequest[QuestionInput, AnswerExpected] = EvalRequest(
            sample=sample,
            experiment=experiment,
        )

        data = dump(request)
        restored = parse(EvalRequest[QuestionInput, AnswerExpected], data)

        assert isinstance(restored.sample.input, QuestionInput)
        assert restored.sample.input.question == "What is AI?"
        assert isinstance(restored.sample.expected, AnswerExpected)
        assert restored.sample.expected.answer == "Artificial Intelligence"
        assert isinstance(restored.experiment, Experiment)
        assert restored.experiment.flags == {"debug": True}

    def test_eval_request_parse_from_json(self) -> None:
        """EvalRequest parses from JSON with string UUIDs and datetimes."""
        data = {
            "sample": {"id": "s1", "input": "hello", "expected": "world"},
            "experiment": {"name": "test"},
            "request_id": "11111111-2222-3333-4444-555555555555",
            "created_at": "2024-06-15T10:30:00+00:00",
        }

        restored = parse(EvalRequest[str, str], data)

        assert restored.sample.id == "s1"
        assert isinstance(restored.experiment, Experiment)
        assert restored.experiment.name == "test"
        assert restored.request_id == UUID("11111111-2222-3333-4444-555555555555")
        assert restored.created_at.year == 2024
