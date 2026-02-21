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

"""Clone tests, edge cases, and integration tests for loop type serde."""

from __future__ import annotations

from uuid import UUID

import pytest

from tests.serde.conftest import AnswerExpected, QuestionInput, TaskOutput, TaskRequest
from weakincentives.budget import Budget
from weakincentives.evals._types import EvalRequest, EvalResult, Sample, Score
from weakincentives.experiment import Experiment
from weakincentives.runtime.agent_loop_types import (
    AgentLoopRequest,
    AgentLoopResult,
)
from weakincentives.runtime.run_context import RunContext
from weakincentives.serde import clone, dump, parse

pytestmark = pytest.mark.core


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

    def test_agent_loop_request_unspecialized_generic_error(self) -> None:
        """AgentLoopRequest without type arguments raises error."""
        data = {
            "request": "test",
        }

        with pytest.raises(TypeError) as exc:
            parse(AgentLoopRequest, data)

        assert "cannot parse TypeVar field" in str(exc.value)

    def test_agent_loop_result_unspecialized_generic_error(self) -> None:
        """AgentLoopResult without type arguments raises error."""
        data = {
            "request_id": "11111111-2222-3333-4444-555555555555",
            "output": "test",
        }

        with pytest.raises(TypeError) as exc:
            parse(AgentLoopResult, data)

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
        result: AgentLoopResult[str] = AgentLoopResult(
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
