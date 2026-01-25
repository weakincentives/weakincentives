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

"""Serialization/deserialization tests for EvalLoop and MainLoop types."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

from weakincentives.budget import Budget
from weakincentives.clock import FakeClock
from weakincentives.deadlines import Deadline
from weakincentives.evals._types import EvalRequest, EvalResult, Sample, Score
from weakincentives.experiment import Experiment
from weakincentives.runtime.main_loop_types import (
    MainLoopConfig,
    MainLoopRequest,
    MainLoopResult,
)
from weakincentives.runtime.run_context import RunContext
from weakincentives.serde import clone, dump, parse

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
    """User request for MainLoop."""

    task: str
    priority: int = 1


@dataclass(slots=True, frozen=True)
class TaskOutput:
    """Output from MainLoop processing."""

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


# =============================================================================
# RunContext Tests
# =============================================================================


class TestRunContextSerde:
    """Tests for RunContext serialization/deserialization."""

    def test_run_context_round_trip(self) -> None:
        """RunContext serializes and deserializes correctly."""
        run_id = UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
        request_id = UUID("11111111-2222-3333-4444-555555555555")
        session_id = UUID("99999999-8888-7777-6666-555555555555")

        ctx = RunContext(
            run_id=run_id,
            request_id=request_id,
            session_id=session_id,
            attempt=2,
            worker_id="worker-1",
            trace_id="trace-abc",
            span_id="span-123",
        )

        data = dump(ctx)
        restored = parse(RunContext, data)

        assert restored.run_id == run_id
        assert restored.request_id == request_id
        assert restored.session_id == session_id
        assert restored.attempt == 2
        assert restored.worker_id == "worker-1"
        assert restored.trace_id == "trace-abc"
        assert restored.span_id == "span-123"

    def test_run_context_minimal(self) -> None:
        """RunContext with defaults."""
        ctx = RunContext()

        data = dump(ctx)
        restored = parse(RunContext, data)

        assert isinstance(restored.run_id, UUID)
        assert isinstance(restored.request_id, UUID)
        assert restored.session_id is None
        assert restored.attempt == 1
        assert restored.worker_id == ""

    def test_run_context_parse_from_json(self) -> None:
        """RunContext parses from JSON with string UUIDs."""
        data = {
            "run_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "request_id": "11111111-2222-3333-4444-555555555555",
            "attempt": "3",
            "worker_id": "w2",
        }

        restored = parse(RunContext, data)

        assert restored.run_id == UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
        assert restored.attempt == 3


# =============================================================================
# Budget Tests
# =============================================================================


class TestBudgetSerde:
    """Tests for Budget serialization/deserialization."""

    def test_budget_with_tokens_round_trip(self) -> None:
        """Budget with token limits."""
        budget = Budget(
            max_total_tokens=10000,
            max_input_tokens=8000,
            max_output_tokens=2000,
        )

        data = dump(budget)
        restored = parse(Budget, data)

        assert restored.max_total_tokens == 10000
        assert restored.max_input_tokens == 8000
        assert restored.max_output_tokens == 2000
        assert restored.deadline is None

    def test_budget_with_deadline(self) -> None:
        """Budget with deadline only."""
        clock = FakeClock()
        clock.set_wall(datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC))

        deadline = Deadline(
            expires_at=datetime(2024, 6, 15, 13, 0, 0, tzinfo=UTC),
            clock=clock,
        )
        budget = Budget(deadline=deadline)

        data = dump(budget)
        # Note: deadline.clock is not serialized (compare=False)
        # For round-trip, we need to parse and verify the expires_at

        assert "deadline" in data
        deadline_data = data["deadline"]
        assert isinstance(deadline_data, dict)
        assert deadline_data["expires_at"] == "2024-06-15T13:00:00+00:00"

    def test_budget_parse_from_json(self) -> None:
        """Budget parses from JSON with coercion."""
        data = {
            "max_total_tokens": "5000",
            "max_input_tokens": "4000",
        }

        restored = parse(Budget, data)

        assert restored.max_total_tokens == 5000
        assert restored.max_input_tokens == 4000


# =============================================================================
# MainLoopConfig Tests
# =============================================================================


class TestMainLoopConfigSerde:
    """Tests for MainLoopConfig serialization/deserialization."""

    def test_main_loop_config_minimal(self) -> None:
        """MainLoopConfig with defaults."""
        config = MainLoopConfig()

        data = dump(config)
        restored = parse(MainLoopConfig, data)

        assert restored.deadline is None
        assert restored.budget is None
        assert restored.resources is None
        assert restored.lease_extender is None
        assert restored.debug_bundle is None

    def test_main_loop_config_with_budget(self) -> None:
        """MainLoopConfig with budget."""
        budget = Budget(max_total_tokens=5000)
        config = MainLoopConfig(budget=budget)

        data = dump(config)
        restored = parse(MainLoopConfig, data)

        assert restored.budget is not None
        assert restored.budget.max_total_tokens == 5000


# =============================================================================
# MainLoopRequest Tests
# =============================================================================


class TestMainLoopRequestSerde:
    """Tests for MainLoopRequest serialization/deserialization."""

    def test_main_loop_request_with_string_type(self) -> None:
        """MainLoopRequest with primitive string type."""
        request_id = UUID("12345678-1234-5678-1234-567812345678")
        created_at = datetime(2024, 6, 15, 14, 0, 0, tzinfo=UTC)

        request: MainLoopRequest[str] = MainLoopRequest(
            request="Process this text",
            request_id=request_id,
            created_at=created_at,
        )

        data = dump(request)
        restored = parse(MainLoopRequest[str], data)

        assert restored.request == "Process this text"
        assert restored.request_id == request_id
        assert restored.created_at == created_at
        assert restored.budget is None
        assert restored.deadline is None
        assert restored.experiment is None

    def test_main_loop_request_dump_structure(self) -> None:
        """MainLoopRequest dump produces correct structure."""
        task = TaskRequest(task="Generate report", priority=2)
        experiment = Experiment(name="fast-mode", flags={"cache": True})
        budget = Budget(max_total_tokens=10000)

        request: MainLoopRequest[TaskRequest] = MainLoopRequest(
            request=task,
            budget=budget,
            experiment=experiment,
        )

        data = dump(request)

        # Verify structure
        assert data["request"] == {"task": "Generate report", "priority": 2}
        assert data["experiment"]["name"] == "fast-mode"
        assert data["experiment"]["flags"] == {"cache": True}
        assert data["budget"]["max_total_tokens"] == 10000

    def test_main_loop_request_with_dataclass_type(self) -> None:
        """MainLoopRequest with nested dataclass type."""
        task = TaskRequest(task="Generate report", priority=2)
        experiment = Experiment(name="fast-mode", flags={"cache": True})
        budget = Budget(max_total_tokens=10000)

        request: MainLoopRequest[TaskRequest] = MainLoopRequest(
            request=task,
            budget=budget,
            experiment=experiment,
        )

        data = dump(request)
        restored = parse(MainLoopRequest[TaskRequest], data)

        assert isinstance(restored.request, TaskRequest)
        assert restored.request.task == "Generate report"
        assert restored.request.priority == 2
        assert restored.budget is not None
        assert restored.budget.max_total_tokens == 10000
        assert isinstance(restored.experiment, Experiment)
        assert restored.experiment.name == "fast-mode"
        assert restored.experiment.flags == {"cache": True}

    def test_main_loop_request_with_run_context(self) -> None:
        """MainLoopRequest with run context."""
        run_ctx = RunContext(worker_id="worker-5", attempt=1)

        request: MainLoopRequest[str] = MainLoopRequest(
            request="test",
            run_context=run_ctx,
        )

        data = dump(request)
        restored = parse(MainLoopRequest[str], data)

        assert restored.run_context is not None
        assert restored.run_context.worker_id == "worker-5"
        assert restored.run_context.attempt == 1

    def test_main_loop_request_parse_from_json(self) -> None:
        """MainLoopRequest parses from JSON with nested types."""
        data = {
            "request": {"task": "Analyze data", "priority": "3"},
            "request_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "created_at": "2024-06-15T15:00:00+00:00",
            "experiment": {"name": "experiment-1"},
            "budget": {"max_total_tokens": "8000"},
        }

        restored = parse(MainLoopRequest[TaskRequest], data)

        assert restored.request.task == "Analyze data"
        assert restored.request.priority == 3
        assert restored.request_id == UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
        assert isinstance(restored.experiment, Experiment)
        assert restored.experiment.name == "experiment-1"

        # Budget is properly parsed since it's not in TYPE_CHECKING
        assert restored.budget is not None
        assert restored.budget.max_total_tokens == 8000


# =============================================================================
# MainLoopResult Tests
# =============================================================================


class TestMainLoopResultSerde:
    """Tests for MainLoopResult serialization/deserialization."""

    def test_main_loop_result_success_string(self) -> None:
        """MainLoopResult with string output type (success)."""
        request_id = UUID("11111111-2222-3333-4444-555555555555")
        session_id = UUID("99999999-8888-7777-6666-555555555555")
        completed_at = datetime(2024, 6, 15, 16, 0, 0, tzinfo=UTC)

        result: MainLoopResult[str] = MainLoopResult(
            request_id=request_id,
            output="Task completed successfully",
            session_id=session_id,
            completed_at=completed_at,
        )

        data = dump(result)
        restored = parse(MainLoopResult[str], data)

        assert restored.request_id == request_id
        assert restored.output == "Task completed successfully"
        assert restored.session_id == session_id
        assert restored.completed_at == completed_at
        assert restored.error is None
        assert restored.success is True

    def test_main_loop_result_success_dataclass(self) -> None:
        """MainLoopResult with dataclass output type (success)."""
        request_id = UUID("22222222-3333-4444-5555-666666666666")
        output = TaskOutput(result="Report generated", success=True)
        run_ctx = RunContext(worker_id="worker-1")

        result: MainLoopResult[TaskOutput] = MainLoopResult(
            request_id=request_id,
            output=output,
            run_context=run_ctx,
        )

        data = dump(result)
        restored = parse(MainLoopResult[TaskOutput], data)

        assert restored.request_id == request_id
        assert isinstance(restored.output, TaskOutput)
        assert restored.output.result == "Report generated"
        assert restored.output.success is True
        assert restored.run_context is not None
        assert restored.run_context.worker_id == "worker-1"
        assert restored.success is True

    def test_main_loop_result_error(self) -> None:
        """MainLoopResult with error."""
        request_id = UUID("33333333-4444-5555-6666-777777777777")

        result: MainLoopResult[str] = MainLoopResult(
            request_id=request_id,
            error="Processing failed: timeout",
        )

        data = dump(result)
        restored = parse(MainLoopResult[str], data)

        assert restored.request_id == request_id
        assert restored.output is None
        assert restored.error == "Processing failed: timeout"
        assert restored.success is False

    def test_main_loop_result_with_bundle_path(self) -> None:
        """MainLoopResult with bundle path."""
        request_id = UUID("44444444-5555-6666-7777-888888888888")

        result: MainLoopResult[str] = MainLoopResult(
            request_id=request_id,
            output="done",
            bundle_path=Path("/debug/bundle.zip"),
        )

        data = dump(result)
        restored = parse(MainLoopResult[str], data)

        assert restored.bundle_path == Path("/debug/bundle.zip")

    def test_main_loop_result_parse_from_json(self) -> None:
        """MainLoopResult parses from JSON with nested types."""
        data = {
            "request_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "output": {"result": "Success", "success": "true"},
            "session_id": "11111111-2222-3333-4444-555555555555",
            "completed_at": "2024-06-15T17:00:00+00:00",
        }

        restored = parse(MainLoopResult[TaskOutput], data)

        assert restored.request_id == UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
        assert isinstance(restored.output, TaskOutput)
        assert restored.output.result == "Success"
        assert restored.output.success is True
        assert restored.session_id == UUID("11111111-2222-3333-4444-555555555555")


# =============================================================================
# Clone Tests
# =============================================================================


class TestLoopTypeClone:
    """Tests for cloning loop types."""

    def test_clone_score(self) -> None:
        """Clone Score with overrides."""
        score = Score(value=0.8, passed=True, reason="Good")
        cloned = clone(score, value=0.9)

        assert cloned.value == 0.9
        assert cloned.passed is True
        assert cloned.reason == "Good"

    def test_clone_experiment(self) -> None:
        """Clone Experiment with overrides."""
        exp = Experiment(name="v1", flags={"a": 1})
        cloned = clone(exp, name="v2")

        assert cloned.name == "v2"
        assert cloned.flags == {"a": 1}

    def test_clone_run_context(self) -> None:
        """Clone RunContext with overrides."""
        ctx = RunContext(worker_id="w1", attempt=1)
        cloned = clone(ctx, attempt=2)

        assert cloned.worker_id == "w1"
        assert cloned.attempt == 2

    def test_clone_eval_result(self) -> None:
        """Clone EvalResult with overrides."""
        result = EvalResult(
            sample_id="s1",
            experiment_name="exp",
            score=Score(value=0.5, passed=False),
            latency_ms=100,
        )
        cloned = clone(result, latency_ms=200)

        assert cloned.sample_id == "s1"
        assert cloned.latency_ms == 200


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestLoopSerdeEdgeCases:
    """Edge cases and error handling for loop type serde."""

    def test_eval_request_unspecialized_generic_error(self) -> None:
        """EvalRequest without type arguments raises error."""
        data = {
            "sample": {"id": "1", "input": "x", "expected": "y"},
            "experiment": {"name": "test"},
        }

        with pytest.raises(TypeError) as exc:
            parse(EvalRequest, data)

        assert "cannot parse TypeVar field" in str(exc.value)

    def test_main_loop_request_unspecialized_generic_error(self) -> None:
        """MainLoopRequest without type arguments raises error."""
        data = {
            "request": "test",
        }

        with pytest.raises(TypeError) as exc:
            parse(MainLoopRequest, data)

        assert "cannot parse TypeVar field" in str(exc.value)

    def test_main_loop_result_unspecialized_generic_error(self) -> None:
        """MainLoopResult without type arguments raises error."""
        data = {
            "request_id": "11111111-2222-3333-4444-555555555555",
            "output": "test",
        }

        with pytest.raises(TypeError) as exc:
            parse(MainLoopResult, data)

        assert "cannot parse TypeVar field" in str(exc.value)

    def test_score_missing_required_field(self) -> None:
        """Score missing required field raises error."""
        data = {"value": 0.5}  # missing 'passed'

        with pytest.raises(ValueError) as exc:
            parse(Score, data)

        assert "passed" in str(exc.value)

    def test_eval_result_missing_score(self) -> None:
        """EvalResult missing score raises error."""
        data = {
            "sample_id": "s1",
            "experiment_name": "exp",
            "latency_ms": 100,
        }

        with pytest.raises(ValueError) as exc:
            parse(EvalResult, data)

        assert "score" in str(exc.value)

    def test_budget_requires_at_least_one_limit(self) -> None:
        """Budget with no limits raises error."""
        data: dict[str, object] = {}

        with pytest.raises(ValueError) as exc:
            parse(Budget, data)

        assert "at least one limit" in str(exc.value)

    def test_budget_negative_tokens_rejected(self) -> None:
        """Budget with negative token limits raises error."""
        data = {"max_total_tokens": -100}

        with pytest.raises(ValueError) as exc:
            parse(Budget, data)

        assert "positive" in str(exc.value)

    def test_experiment_empty_name(self) -> None:
        """Experiment can have empty name (no validation)."""
        data = {"name": ""}

        restored = parse(Experiment, data)
        assert restored.name == ""

    def test_sample_round_trip_with_int_types(self) -> None:
        """Sample with int types for numeric evaluations."""
        sample: Sample[int, int] = Sample(id="num-1", input=42, expected=84)

        data = dump(sample)
        restored = parse(Sample[int, int], data)

        assert restored.input == 42
        assert restored.expected == 84

    def test_dump_exclude_none_on_loop_types(self) -> None:
        """Dump with exclude_none removes None fields."""
        result: MainLoopResult[str] = MainLoopResult(
            request_id=UUID("11111111-2222-3333-4444-555555555555"),
            output="done",
        )

        data = dump(result, exclude_none=True)

        assert "error" not in data
        assert "session_id" not in data
        assert "run_context" not in data
        assert "bundle_path" not in data
        assert data["output"] == "done"


# =============================================================================
# Integration Tests - Full Workflow Round-trips
# =============================================================================


class TestLoopSerdeIntegration:
    """Integration tests for complete request/response workflows."""

    def test_eval_workflow_round_trip(self) -> None:
        """Complete EvalLoop workflow: request -> result."""
        # Create request
        sample: Sample[QuestionInput, AnswerExpected] = Sample(
            id="integration-1",
            input=QuestionInput(
                question="What is machine learning?",
                context="Computer science",
            ),
            expected=AnswerExpected(
                answer="A field of AI",
                keywords=("AI", "learning", "algorithms"),
            ),
        )
        experiment = Experiment(
            name="eval-test",
            flags={"model": "gpt-4", "temperature": 0.7},
        )
        request: EvalRequest[QuestionInput, AnswerExpected] = EvalRequest(
            sample=sample,
            experiment=experiment,
        )

        # Serialize and deserialize request
        request_data = dump(request)
        restored_request = parse(
            EvalRequest[QuestionInput, AnswerExpected], request_data
        )

        assert restored_request.sample.input.question == "What is machine learning?"
        assert isinstance(restored_request.experiment, Experiment)
        assert restored_request.experiment.name == "eval-test"

        # Create result
        result = EvalResult(
            sample_id=restored_request.sample.id,
            experiment_name=restored_request.experiment.name,
            score=Score(value=0.95, passed=True, reason="Excellent answer"),
            latency_ms=250,
        )

        # Serialize and deserialize result
        result_data = dump(result)
        restored_result = parse(EvalResult, result_data)

        assert restored_result.sample_id == "integration-1"
        assert restored_result.score.passed is True

    def test_main_loop_workflow_round_trip(self) -> None:
        """Complete MainLoop workflow: request -> result."""
        # Create request
        task = TaskRequest(task="Generate summary", priority=1)
        budget = Budget(max_total_tokens=5000, max_output_tokens=1000)
        run_ctx = RunContext(worker_id="worker-integration", attempt=1)

        request: MainLoopRequest[TaskRequest] = MainLoopRequest(
            request=task,
            budget=budget,
            run_context=run_ctx,
            experiment=Experiment(name="main-test"),
        )

        # Serialize and deserialize request
        request_data = dump(request)
        restored_request = parse(MainLoopRequest[TaskRequest], request_data)

        assert restored_request.request.task == "Generate summary"
        assert restored_request.budget is not None
        assert restored_request.budget.max_total_tokens == 5000
        assert isinstance(restored_request.experiment, Experiment)
        assert restored_request.experiment.name == "main-test"

        # Create result
        output = TaskOutput(result="Summary: Integration test passed", success=True)
        result: MainLoopResult[TaskOutput] = MainLoopResult(
            request_id=restored_request.request_id,
            output=output,
            session_id=UUID("88888888-7777-6666-5555-444444444444"),
            run_context=restored_request.run_context,
        )

        # Serialize and deserialize result
        result_data = dump(result)
        restored_result = parse(MainLoopResult[TaskOutput], result_data)

        assert restored_result.output is not None
        assert restored_result.output.success is True
        assert restored_result.success is True
