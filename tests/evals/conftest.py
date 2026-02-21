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

"""Shared test fixtures for eval loop tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from unittest.mock import Mock
from uuid import uuid4

from weakincentives.adapters.core import PromptResponse, ProviderAdapter
from weakincentives.budget import Budget, BudgetTracker
from weakincentives.deadlines import Deadline
from weakincentives.evals import Score, exact_match
from weakincentives.prompt import MarkdownSection, Prompt, PromptTemplate
from weakincentives.runtime import AgentLoop, InMemoryMailbox, Session
from weakincentives.runtime.agent_loop import AgentLoopRequest, AgentLoopResult
from weakincentives.runtime.events import PromptExecuted, TokenUsage, ToolInvoked
from weakincentives.runtime.session import SessionProtocol, SessionViewProtocol

# =============================================================================
# Shared Dataclasses
# =============================================================================


@dataclass(slots=True, frozen=True)
class Params:
    """Test prompt params."""

    content: str


@dataclass(slots=True, frozen=True)
class Output:
    """Test output type."""

    result: str


# =============================================================================
# Mock Adapters
# =============================================================================


class MockAdapter(ProviderAdapter[Output]):
    """Mock adapter for testing."""

    def __init__(
        self,
        *,
        result: str = "success",
        error: Exception | None = None,
    ) -> None:
        self._result = result
        self._error = error
        self.call_count = 0

    def evaluate(
        self,
        prompt: Prompt[Output],
        *,
        session: SessionProtocol,
        deadline: Deadline | None = None,
        budget: Budget | None = None,
        budget_tracker: BudgetTracker | None = None,
        heartbeat: object = None,
        run_context: object = None,
    ) -> PromptResponse[Output]:
        del prompt, session, deadline, budget, budget_tracker, heartbeat, run_context
        self.call_count += 1
        if self._error is not None:
            raise self._error
        return PromptResponse(
            prompt_name="test",
            text=self._result,
            output=Output(result=self._result),
        )


class NoneOutputAdapter(ProviderAdapter[Output]):
    """Mock adapter that returns None output."""

    def evaluate(
        self,
        prompt: Prompt[Output],
        *,
        session: SessionProtocol,
        deadline: Deadline | None = None,
        budget: Budget | None = None,
        budget_tracker: BudgetTracker | None = None,
        heartbeat: object = None,
        run_context: object = None,
    ) -> PromptResponse[Output]:
        del prompt, session, deadline, budget, budget_tracker, heartbeat, run_context
        return PromptResponse(
            prompt_name="test",
            text="no structured output",
            output=None,
        )


# =============================================================================
# Test AgentLoop Implementations
# =============================================================================


class TestLoop(AgentLoop[str, Output]):
    """Test AgentLoop implementation."""

    def __init__(
        self,
        *,
        adapter: ProviderAdapter[Output],
        requests: InMemoryMailbox[AgentLoopRequest[str], AgentLoopResult[Output]],
    ) -> None:
        super().__init__(adapter=adapter, requests=requests)
        self._template = PromptTemplate[Output](
            ns="test",
            key="test-prompt",
            sections=[
                MarkdownSection[Params](
                    title="Test",
                    template="$content",
                    key="test",
                ),
            ],
        )

    def prepare(
        self,
        request: str,
        *,
        experiment: object = None,
    ) -> tuple[Prompt[Output], Session]:
        _ = experiment
        prompt = Prompt(self._template).bind(Params(content=request))
        session = Session(tags={"loop": "test"})
        return prompt, session


class NoneOutputLoop(AgentLoop[str, Output]):
    """AgentLoop that returns None output."""

    def __init__(
        self,
        *,
        adapter: ProviderAdapter[Output],
        requests: InMemoryMailbox[AgentLoopRequest[str], AgentLoopResult[Output]],
    ) -> None:
        super().__init__(adapter=adapter, requests=requests)
        self._template = PromptTemplate[Output](
            ns="test",
            key="test-prompt",
            sections=[
                MarkdownSection[Params](
                    title="Test",
                    template="$content",
                    key="test",
                ),
            ],
        )

    def prepare(
        self,
        request: str,
        *,
        experiment: object = None,
    ) -> tuple[Prompt[Output], Session]:
        _ = experiment
        prompt = Prompt(self._template).bind(Params(content=request))
        session = Session(tags={"loop": "test"})
        return prompt, session


# =============================================================================
# Helper Functions
# =============================================================================


def create_test_loop(
    *,
    result: str = "success",
    error: Exception | None = None,
) -> TestLoop:
    """Create a test AgentLoop with mock adapter."""
    adapter = MockAdapter(result=result, error=error)
    # EvalLoop doesn't use AgentLoop's mailboxes directly, but AgentLoop requires one
    requests: InMemoryMailbox[AgentLoopRequest[str], AgentLoopResult[Output]] = (
        InMemoryMailbox(name="dummy-requests")
    )
    return TestLoop(adapter=adapter, requests=requests)


def output_to_str(output: Output, expected: str) -> Score:
    """Convert Output to string for evaluation."""
    return exact_match(output.result, expected)


def session_aware_evaluator(
    output: object,
    expected: object,
    session: SessionProtocol,
) -> Score:
    """Session-aware evaluator for testing the 3-param path."""
    _ = session  # Use session parameter to mark as session-aware
    if isinstance(output, Output) and isinstance(expected, str):
        return exact_match(output.result, expected)
    return Score(value=0.0, passed=False, reason="Type mismatch")


# =============================================================================
# Session Evaluator Helpers
# =============================================================================


def make_tool_invoked(
    name: str,
    *,
    params: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
) -> ToolInvoked:
    """Create a ToolInvoked event for testing."""
    return ToolInvoked(
        prompt_name="test",
        adapter="openai",
        name=name,
        params=params or {},
        result=result or {"success": True},
        session_id=uuid4(),
        created_at=datetime.now(UTC),
    )


def make_prompt_executed(
    *,
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> PromptExecuted:
    """Create a PromptExecuted event for testing."""
    return PromptExecuted(
        prompt_name="test",
        adapter="openai",
        result={"output": "test"},
        session_id=uuid4(),
        created_at=datetime.now(UTC),
        usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def make_mock_session(
    tool_invocations: list[ToolInvoked] | None = None,
    prompt_executions: list[PromptExecuted] | None = None,
    custom_slices: dict[type, tuple[Any, ...]] | None = None,
) -> SessionViewProtocol:
    """Create a mock session with specified slices.

    Args:
        tool_invocations: ToolInvoked events to populate.
        prompt_executions: PromptExecuted events to populate.
        custom_slices: Additional custom slices as {type: (items,)}.

    Returns:
        Mock session implementing SessionViewProtocol.
    """
    session = Mock(spec=SessionViewProtocol)
    tool_invocations = tool_invocations or []
    prompt_executions = prompt_executions or []
    custom_slices = custom_slices or {}

    def get_slice(slice_type: type) -> Mock:
        slice_mock = Mock()
        if slice_type is ToolInvoked:
            slice_mock.all.return_value = tuple(tool_invocations)
            slice_mock.where.side_effect = lambda pred: iter(
                t for t in tool_invocations if pred(t)
            )
        elif slice_type is PromptExecuted:
            slice_mock.all.return_value = tuple(prompt_executions)
            slice_mock.where.side_effect = lambda pred: iter(
                p for p in prompt_executions if pred(p)
            )
        elif slice_type in custom_slices:
            items = custom_slices[slice_type]
            slice_mock.all.return_value = items
            slice_mock.where.side_effect = lambda pred: iter(
                item for item in items if pred(item)
            )
        else:
            slice_mock.all.return_value = ()
            slice_mock.where.side_effect = lambda pred: iter(())
        return slice_mock

    session.__getitem__ = Mock(side_effect=get_slice)
    return session
