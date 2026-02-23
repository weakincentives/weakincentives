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

"""Serialization/deserialization tests for clone and edge-case behaviour of loop types."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

import pytest

from weakincentives.evals._types import EvalResult, Sample, Score
from weakincentives.experiment import Experiment
from weakincentives.runtime.agent_loop_types import (
    AgentLoopResult,
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
    """User request for AgentLoop."""

    task: str
    priority: int = 1


@dataclass(slots=True, frozen=True)
class TaskOutput:
    """Output from AgentLoop processing."""

    result: str
    success: bool = True


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
        from weakincentives.evals._types import EvalRequest as EvalRequestType

        data = {
            "sample": {"id": "1", "input": "x", "expected": "y"},
            "experiment": {"name": "test"},
        }

        with pytest.raises(TypeError) as exc:
            parse(EvalRequestType, data)

        assert "cannot parse TypeVar field" in str(exc.value)

    def test_agent_loop_request_unspecialized_generic_error(self) -> None:
        """AgentLoopRequest without type arguments raises error."""
        from weakincentives.runtime.agent_loop_types import (
            AgentLoopRequest as AgentLoopRequestType,
        )

        data = {
            "request": "test",
        }

        with pytest.raises(TypeError) as exc:
            parse(AgentLoopRequestType, data)

        assert "cannot parse TypeVar field" in str(exc.value)

    def test_agent_loop_result_unspecialized_generic_error(self) -> None:
        """AgentLoopResult without type arguments raises error."""
        from weakincentives.runtime.agent_loop_types import (
            AgentLoopResult as AgentLoopResultType,
        )

        data = {
            "request_id": "11111111-2222-3333-4444-555555555555",
            "output": "test",
        }

        with pytest.raises(TypeError) as exc:
            parse(AgentLoopResultType, data)

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
        from weakincentives.budget import Budget as BudgetType

        data: dict[str, object] = {}

        with pytest.raises(ValueError) as exc:
            parse(BudgetType, data)

        assert "at least one limit" in str(exc.value)

    def test_budget_negative_tokens_rejected(self) -> None:
        """Budget with negative token limits raises error."""
        from weakincentives.budget import Budget as BudgetType

        data = {"max_total_tokens": -100}

        with pytest.raises(ValueError) as exc:
            parse(BudgetType, data)

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
