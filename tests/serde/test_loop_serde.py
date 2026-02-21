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

"""Serialization/deserialization tests for EvalLoop and AgentLoop types."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

from weakincentives.evals._types import EvalResult, Sample, Score
from weakincentives.experiment import Experiment
from weakincentives.runtime.agent_loop_types import (
    AgentLoopResult,
)
from weakincentives.runtime.run_context import RunContext
from weakincentives.serde import dump, parse

pytestmark = pytest.mark.core


# =============================================================================
# Test data types for generic parameters
# =============================================================================


@dataclass(slots=True, frozen=True)
class QuestionInput:
    """Sample input for QA-style evaluation."""

    question: str
    context: str | None = None


@dataclass(slots=True, frozen=True)
class AnswerExpected:
    """Expected answer for QA-style evaluation."""

    answer: str
    keywords: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class TaskRequest:
    """User request for AgentLoop."""

    task: str
    priority: int = 1


@dataclass(slots=True, frozen=True)
class TaskOutput:
    """Output from AgentLoop processing."""

    result: str
    success: bool = True


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
# AgentLoopResult Tests
# =============================================================================


class TestAgentLoopResultSerde:
    """Tests for AgentLoopResult serialization/deserialization."""

    def test_agent_loop_result_success_string(self) -> None:
        """AgentLoopResult with string output type (success)."""
        request_id = UUID("11111111-2222-3333-4444-555555555555")
        session_id = UUID("99999999-8888-7777-6666-555555555555")
        completed_at = datetime(2024, 6, 15, 16, 0, 0, tzinfo=UTC)

        result: AgentLoopResult[str] = AgentLoopResult(
            request_id=request_id,
            output="Task completed successfully",
            session_id=session_id,
            completed_at=completed_at,
        )

        data = dump(result)
        restored = parse(AgentLoopResult[str], data)

        assert restored.request_id == request_id
        assert restored.output == "Task completed successfully"
        assert restored.session_id == session_id
        assert restored.completed_at == completed_at
        assert restored.error is None
        assert restored.success is True

    def test_agent_loop_result_success_dataclass(self) -> None:
        """AgentLoopResult with dataclass output type (success)."""
        request_id = UUID("22222222-3333-4444-5555-666666666666")
        output = TaskOutput(result="Report generated", success=True)
        run_ctx = RunContext(worker_id="worker-1")

        result: AgentLoopResult[TaskOutput] = AgentLoopResult(
            request_id=request_id,
            output=output,
            run_context=run_ctx,
        )

        data = dump(result)
        restored = parse(AgentLoopResult[TaskOutput], data)

        assert restored.request_id == request_id
        assert isinstance(restored.output, TaskOutput)
        assert restored.output.result == "Report generated"
        assert restored.output.success is True
        assert restored.run_context is not None
        assert restored.run_context.worker_id == "worker-1"
        assert restored.success is True

    def test_agent_loop_result_error(self) -> None:
        """AgentLoopResult with error."""
        request_id = UUID("33333333-4444-5555-6666-777777777777")

        result: AgentLoopResult[str] = AgentLoopResult(
            request_id=request_id,
            error="Processing failed: timeout",
        )

        data = dump(result)
        restored = parse(AgentLoopResult[str], data)

        assert restored.request_id == request_id
        assert restored.output is None
        assert restored.error == "Processing failed: timeout"
        assert restored.success is False

    def test_agent_loop_result_with_bundle_path(self) -> None:
        """AgentLoopResult with bundle path."""
        request_id = UUID("44444444-5555-6666-7777-888888888888")

        result: AgentLoopResult[str] = AgentLoopResult(
            request_id=request_id,
            output="done",
            bundle_path=Path("/debug/bundle.zip"),
        )

        data = dump(result)
        restored = parse(AgentLoopResult[str], data)

        assert restored.bundle_path == Path("/debug/bundle.zip")

    def test_agent_loop_result_parse_from_json(self) -> None:
        """AgentLoopResult parses from JSON with nested types."""
        data = {
            "request_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "output": {"result": "Success", "success": "true"},
            "session_id": "11111111-2222-3333-4444-555555555555",
            "completed_at": "2024-06-15T17:00:00+00:00",
        }

        restored = parse(AgentLoopResult[TaskOutput], data)

        assert restored.request_id == UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
        assert isinstance(restored.output, TaskOutput)
        assert restored.output.result == "Success"
        assert restored.output.success is True
        assert restored.session_id == UUID("11111111-2222-3333-4444-555555555555")
