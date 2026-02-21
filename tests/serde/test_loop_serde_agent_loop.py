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

"""Serialization/deserialization tests for AgentLoop types (RunContext, Budget,
AgentLoopConfig, AgentLoopRequest, AgentLoopResult)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

from tests.serde.conftest import TaskOutput, TaskRequest
from weakincentives.budget import Budget
from weakincentives.clock import FakeClock
from weakincentives.deadlines import Deadline
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
