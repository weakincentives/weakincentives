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

"""Tests for session-aware evaluators: tool checks, token usage, slice_contains."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from tests.evals.conftest import (
    make_mock_session,
    make_prompt_executed,
    make_tool_invoked,
)
from weakincentives.evals import (
    all_tools_succeeded,
    slice_contains,
    token_usage_under,
    tool_call_count,
    tool_called,
    tool_not_called,
)
from weakincentives.runtime.events import PromptExecuted

# =============================================================================
# tool_called Tests
# =============================================================================


def test_tool_called_pass() -> None:
    """tool_called passes when the tool was invoked."""
    session = make_mock_session(
        tool_invocations=[make_tool_invoked("search", params={"query": "test"})]
    )
    evaluator = tool_called("search")
    score = evaluator(None, None, session)
    assert score.passed is True
    assert score.value == 1.0
    assert "called 1 time" in score.reason


def test_tool_called_fail() -> None:
    """tool_called fails when the tool was not invoked."""
    session = make_mock_session(tool_invocations=[])
    evaluator = tool_called("search")
    score = evaluator(None, None, session)
    assert score.passed is False
    assert score.value == 0.0
    assert "called 0 time" in score.reason


def test_tool_called_multiple_times() -> None:
    """tool_called passes and reports count when called multiple times."""
    session = make_mock_session(
        tool_invocations=[
            make_tool_invoked("search"),
            make_tool_invoked("search"),
            make_tool_invoked("fetch"),
        ]
    )
    evaluator = tool_called("search")
    score = evaluator(None, None, session)
    assert score.passed is True
    assert "called 2 time" in score.reason


def test_tool_called_wrong_tool() -> None:
    """tool_called fails when only other tools were called."""
    session = make_mock_session(
        tool_invocations=[make_tool_invoked("fetch"), make_tool_invoked("write")]
    )
    evaluator = tool_called("search")
    score = evaluator(None, None, session)
    assert score.passed is False


# =============================================================================
# tool_not_called Tests
# =============================================================================


def test_tool_not_called_pass() -> None:
    """tool_not_called passes when the tool was not invoked."""
    session = make_mock_session(tool_invocations=[make_tool_invoked("fetch")])
    evaluator = tool_not_called("dangerous_tool")
    score = evaluator(None, None, session)
    assert score.passed is True
    assert score.value == 1.0
    assert score.reason == ""  # No reason on pass


def test_tool_not_called_fail() -> None:
    """tool_not_called fails when the tool was invoked."""
    session = make_mock_session(tool_invocations=[make_tool_invoked("dangerous_tool")])
    evaluator = tool_not_called("dangerous_tool")
    score = evaluator(None, None, session)
    assert score.passed is False
    assert score.value == 0.0
    assert "called 1 time" in score.reason


def test_tool_not_called_empty_session() -> None:
    """tool_not_called passes when no tools were invoked."""
    session = make_mock_session(tool_invocations=[])
    evaluator = tool_not_called("any_tool")
    score = evaluator(None, None, session)
    assert score.passed is True


# =============================================================================
# tool_call_count Tests
# =============================================================================


def test_tool_call_count_exact() -> None:
    """tool_call_count passes when count is exactly within bounds."""
    session = make_mock_session(
        tool_invocations=[make_tool_invoked("search"), make_tool_invoked("search")]
    )
    evaluator = tool_call_count("search", min_count=2, max_count=2)
    score = evaluator(None, None, session)
    assert score.passed is True
    assert "called 2 times" in score.reason


def test_tool_call_count_below_min() -> None:
    """tool_call_count fails when count is below minimum."""
    session = make_mock_session(tool_invocations=[make_tool_invoked("search")])
    evaluator = tool_call_count("search", min_count=2)
    score = evaluator(None, None, session)
    assert score.passed is False
    assert "called 1 times" in score.reason


def test_tool_call_count_above_max() -> None:
    """tool_call_count fails when count exceeds maximum."""
    session = make_mock_session(
        tool_invocations=[
            make_tool_invoked("search"),
            make_tool_invoked("search"),
            make_tool_invoked("search"),
        ]
    )
    evaluator = tool_call_count("search", max_count=2)
    score = evaluator(None, None, session)
    assert score.passed is False
    assert "called 3 times" in score.reason


def test_tool_call_count_no_max() -> None:
    """tool_call_count with no max allows any count above min."""
    session = make_mock_session(
        tool_invocations=[make_tool_invoked("search") for _ in range(10)]
    )
    evaluator = tool_call_count("search", min_count=1)
    score = evaluator(None, None, session)
    assert score.passed is True


def test_tool_call_count_zero_calls() -> None:
    """tool_call_count reports zero calls correctly."""
    session = make_mock_session(tool_invocations=[])
    evaluator = tool_call_count("search", min_count=1)
    score = evaluator(None, None, session)
    assert score.passed is False
    assert "called 0 times" in score.reason


# =============================================================================
# all_tools_succeeded Tests
# =============================================================================


def test_all_tools_succeeded_pass() -> None:
    """all_tools_succeeded passes when all tools return success=True."""
    session = make_mock_session(
        tool_invocations=[
            make_tool_invoked("search", result={"success": True}),
            make_tool_invoked("fetch", result={"success": True}),
        ]
    )
    evaluator = all_tools_succeeded()
    score = evaluator(None, None, session)
    assert score.passed is True
    assert score.value == 1.0
    assert score.reason == ""


def test_all_tools_succeeded_fail() -> None:
    """all_tools_succeeded fails when any tool returns success=False."""
    session = make_mock_session(
        tool_invocations=[
            make_tool_invoked("search", result={"success": True}),
            make_tool_invoked("fetch", result={"success": False, "error": "not found"}),
        ]
    )
    evaluator = all_tools_succeeded()
    score = evaluator(None, None, session)
    assert score.passed is False
    assert score.value == 0.0
    assert "fetch" in score.reason


def test_all_tools_succeeded_no_success_field() -> None:
    """all_tools_succeeded treats missing success field as success."""
    session = make_mock_session(
        tool_invocations=[
            make_tool_invoked("search", result={"data": "result"}),
        ]
    )
    evaluator = all_tools_succeeded()
    score = evaluator(None, None, session)
    assert score.passed is True


def test_all_tools_succeeded_empty_session() -> None:
    """all_tools_succeeded passes when no tools were invoked."""
    session = make_mock_session(tool_invocations=[])
    evaluator = all_tools_succeeded()
    score = evaluator(None, None, session)
    assert score.passed is True


def test_all_tools_succeeded_multiple_failures() -> None:
    """all_tools_succeeded reports all failed tools."""
    session = make_mock_session(
        tool_invocations=[
            make_tool_invoked("search", result={"success": False}),
            make_tool_invoked("fetch", result={"success": False}),
        ]
    )
    evaluator = all_tools_succeeded()
    score = evaluator(None, None, session)
    assert score.passed is False
    assert "search" in score.reason
    assert "fetch" in score.reason


# =============================================================================
# token_usage_under Tests
# =============================================================================


def test_token_usage_under_pass() -> None:
    """token_usage_under passes when usage is under budget."""
    session = make_mock_session(
        prompt_executions=[
            make_prompt_executed(input_tokens=100, output_tokens=50),
        ]
    )
    evaluator = token_usage_under(200)
    score = evaluator(None, None, session)
    assert score.passed is True
    assert score.value == 1.0
    assert "used 150 tokens" in score.reason


def test_token_usage_under_fail() -> None:
    """token_usage_under fails when usage exceeds budget."""
    session = make_mock_session(
        prompt_executions=[
            make_prompt_executed(input_tokens=100, output_tokens=50),
            make_prompt_executed(input_tokens=100, output_tokens=50),
        ]
    )
    evaluator = token_usage_under(200)
    score = evaluator(None, None, session)
    assert score.passed is False
    assert score.value == 0.0
    assert "used 300 tokens" in score.reason


def test_token_usage_under_exact_limit() -> None:
    """token_usage_under passes when usage equals budget exactly."""
    session = make_mock_session(
        prompt_executions=[
            make_prompt_executed(input_tokens=100, output_tokens=100),
        ]
    )
    evaluator = token_usage_under(200)
    score = evaluator(None, None, session)
    assert score.passed is True


def test_token_usage_under_no_usage() -> None:
    """token_usage_under passes when no executions occurred."""
    session = make_mock_session(prompt_executions=[])
    evaluator = token_usage_under(100)
    score = evaluator(None, None, session)
    assert score.passed is True
    assert "used 0 tokens" in score.reason


def test_token_usage_under_with_none_usage() -> None:
    """token_usage_under handles executions with None usage."""
    # Create an execution with usage=None
    execution_with_no_usage = PromptExecuted(
        prompt_name="test",
        adapter="openai",
        result={"output": "test"},
        session_id=uuid4(),
        created_at=datetime.now(UTC),
        usage=None,  # Explicitly None
    )
    session = make_mock_session(
        prompt_executions=[
            make_prompt_executed(input_tokens=50, output_tokens=25),  # 75 tokens
            execution_with_no_usage,  # 0 tokens (usage is None)
        ]
    )
    evaluator = token_usage_under(100)
    score = evaluator(None, None, session)
    assert score.passed is True
    assert "used 75 tokens" in score.reason


# =============================================================================
# slice_contains Tests
# =============================================================================


@dataclass(slots=True, frozen=True)
class PlanStep:
    """Test dataclass for slice_contains tests."""

    name: str
    status: str


def test_slice_contains_pass() -> None:
    """slice_contains passes when matching items exist."""
    steps = (
        PlanStep(name="step1", status="completed"),
        PlanStep(name="step2", status="pending"),
    )
    session = make_mock_session(custom_slices={PlanStep: steps})
    evaluator = slice_contains(PlanStep, lambda s: s.status == "completed")
    score = evaluator(None, None, session)
    assert score.passed is True
    assert "found 1 matching items" in score.reason


def test_slice_contains_fail() -> None:
    """slice_contains fails when no matching items exist."""
    steps = (
        PlanStep(name="step1", status="pending"),
        PlanStep(name="step2", status="pending"),
    )
    session = make_mock_session(custom_slices={PlanStep: steps})
    evaluator = slice_contains(PlanStep, lambda s: s.status == "completed")
    score = evaluator(None, None, session)
    assert score.passed is False
    assert "found 0 matching items" in score.reason


def test_slice_contains_min_count() -> None:
    """slice_contains respects min_count parameter."""
    steps = (
        PlanStep(name="step1", status="completed"),
        PlanStep(name="step2", status="completed"),
    )
    session = make_mock_session(custom_slices={PlanStep: steps})

    # Need at least 2 completed
    evaluator = slice_contains(PlanStep, lambda s: s.status == "completed", min_count=2)
    score = evaluator(None, None, session)
    assert score.passed is True

    # Need at least 3 completed (fail)
    evaluator = slice_contains(PlanStep, lambda s: s.status == "completed", min_count=3)
    score = evaluator(None, None, session)
    assert score.passed is False


def test_slice_contains_empty_slice() -> None:
    """slice_contains fails when slice is empty."""
    session = make_mock_session(custom_slices={PlanStep: ()})
    evaluator = slice_contains(PlanStep, lambda s: True)
    score = evaluator(None, None, session)
    assert score.passed is False
