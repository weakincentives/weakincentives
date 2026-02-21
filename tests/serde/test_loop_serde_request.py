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

"""Serialization/deserialization tests for EvalRequest, RunContext, Budget, AgentLoopConfig, and AgentLoopRequest."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

import pytest

from weakincentives.budget import Budget
from weakincentives.clock import FakeClock
from weakincentives.deadlines import Deadline
from weakincentives.evals._types import EvalRequest, EvalResult, Sample, Score
from weakincentives.experiment import Experiment
from weakincentives.runtime.agent_loop_types import (
    AgentLoopConfig,
    AgentLoopRequest,
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
# AgentLoopConfig Tests
# =============================================================================


class TestAgentLoopConfigSerde:
    """Tests for AgentLoopConfig serialization/deserialization."""

    def test_agent_loop_config_minimal(self) -> None:
        """AgentLoopConfig with defaults."""
        config = AgentLoopConfig()

        data = dump(config)
        restored = parse(AgentLoopConfig, data)

        assert restored.budget is None
        assert restored.resources is None
        assert restored.lease_extender is None
        assert restored.debug_bundle is None

    def test_agent_loop_config_with_budget(self) -> None:
        """AgentLoopConfig with budget."""
        budget = Budget(max_total_tokens=5000)
        config = AgentLoopConfig(budget=budget)

        data = dump(config)
        restored = parse(AgentLoopConfig, data)

        assert restored.budget is not None
        assert restored.budget.max_total_tokens == 5000


# =============================================================================
# AgentLoopRequest Tests
# =============================================================================


class TestAgentLoopRequestSerde:
    """Tests for AgentLoopRequest serialization/deserialization."""

    def test_agent_loop_request_with_string_type(self) -> None:
        """AgentLoopRequest with primitive string type."""
        request_id = UUID("12345678-1234-5678-1234-567812345678")
        created_at = datetime(2024, 6, 15, 14, 0, 0, tzinfo=UTC)

        request: AgentLoopRequest[str] = AgentLoopRequest(
            request="Process this text",
            request_id=request_id,
            created_at=created_at,
        )

        data = dump(request)
        restored = parse(AgentLoopRequest[str], data)

        assert restored.request == "Process this text"
        assert restored.request_id == request_id
        assert restored.created_at == created_at
        assert restored.budget is None
        assert restored.deadline is None
        assert restored.experiment is None

    def test_agent_loop_request_dump_structure(self) -> None:
        """AgentLoopRequest dump produces correct structure."""
        task = TaskRequest(task="Generate report", priority=2)
        experiment = Experiment(name="fast-mode", flags={"cache": True})
        budget = Budget(max_total_tokens=10000)

        request: AgentLoopRequest[TaskRequest] = AgentLoopRequest(
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

    def test_agent_loop_request_with_dataclass_type(self) -> None:
        """AgentLoopRequest with nested dataclass type."""
        task = TaskRequest(task="Generate report", priority=2)
        experiment = Experiment(name="fast-mode", flags={"cache": True})
        budget = Budget(max_total_tokens=10000)

        request: AgentLoopRequest[TaskRequest] = AgentLoopRequest(
            request=task,
            budget=budget,
            experiment=experiment,
        )

        data = dump(request)
        restored = parse(AgentLoopRequest[TaskRequest], data)

        assert isinstance(restored.request, TaskRequest)
        assert restored.request.task == "Generate report"
        assert restored.request.priority == 2
        assert restored.budget is not None
        assert restored.budget.max_total_tokens == 10000
        assert isinstance(restored.experiment, Experiment)
        assert restored.experiment.name == "fast-mode"
        assert restored.experiment.flags == {"cache": True}

    def test_agent_loop_request_with_run_context(self) -> None:
        """AgentLoopRequest with run context."""
        run_ctx = RunContext(worker_id="worker-5", attempt=1)

        request: AgentLoopRequest[str] = AgentLoopRequest(
            request="test",
            run_context=run_ctx,
        )

        data = dump(request)
        restored = parse(AgentLoopRequest[str], data)

        assert restored.run_context is not None
        assert restored.run_context.worker_id == "worker-5"
        assert restored.run_context.attempt == 1

    def test_agent_loop_request_parse_from_json(self) -> None:
        """AgentLoopRequest parses from JSON with nested types."""
        data = {
            "request": {"task": "Analyze data", "priority": "3"},
            "request_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "created_at": "2024-06-15T15:00:00+00:00",
            "experiment": {"name": "experiment-1"},
            "budget": {"max_total_tokens": "8000"},
        }

        restored = parse(AgentLoopRequest[TaskRequest], data)

        assert restored.request.task == "Analyze data"
        assert restored.request.priority == 3
        assert restored.request_id == UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
        assert isinstance(restored.experiment, Experiment)
        assert restored.experiment.name == "experiment-1"

        # Budget is properly parsed since it's not in TYPE_CHECKING
        assert restored.budget is not None
        assert restored.budget.max_total_tokens == 8000


# =============================================================================
# Integration Tests
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

    def test_agent_loop_workflow_round_trip(self) -> None:
        """Complete AgentLoop workflow: request -> result."""
        # Create request
        task = TaskRequest(task="Generate summary", priority=1)
        budget = Budget(max_total_tokens=5000, max_output_tokens=1000)
        run_ctx = RunContext(worker_id="worker-integration", attempt=1)

        request: AgentLoopRequest[TaskRequest] = AgentLoopRequest(
            request=task,
            budget=budget,
            run_context=run_ctx,
            experiment=Experiment(name="agent-test"),
        )

        # Serialize and deserialize request
        request_data = dump(request)
        restored_request = parse(AgentLoopRequest[TaskRequest], request_data)

        assert restored_request.request.task == "Generate summary"
        assert restored_request.budget is not None
        assert restored_request.budget.max_total_tokens == 5000
        assert isinstance(restored_request.experiment, Experiment)
        assert restored_request.experiment.name == "agent-test"

        # Create result
        output = TaskOutput(result="Summary: Integration test passed", success=True)
        result: AgentLoopResult[TaskOutput] = AgentLoopResult(
            request_id=restored_request.request_id,
            output=output,
            session_id=UUID("88888888-7777-6666-5555-444444444444"),
            run_context=restored_request.run_context,
        )

        # Serialize and deserialize result
        result_data = dump(result)
        restored_result = parse(AgentLoopResult[TaskOutput], result_data)

        assert restored_result.output is not None
        assert restored_result.output.success is True
        assert restored_result.success is True
