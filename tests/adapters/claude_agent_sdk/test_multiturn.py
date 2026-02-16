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

"""Tests for edge cases in multi-turn continuation loop."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from tests.adapters.claude_agent_sdk.conftest import (
    MockResultMessage,
    MockSDKQuery,
    MockTransport,
    SimpleOutput,
    sdk_patches,
    setup_mock_query,
)
from weakincentives.adapters.claude_agent_sdk import (
    ClaudeAgentSDKAdapter,
    TaskCompletionChecker,
    TaskCompletionContext,
    TaskCompletionResult,
)
from weakincentives.adapters.core import PromptEvaluationError
from weakincentives.deadlines import Deadline
from weakincentives.prompt import MarkdownSection, Prompt, PromptTemplate
from weakincentives.runtime.session import Session


def _make_prompt_with_checker(
    checker: TaskCompletionChecker,
) -> Prompt[None]:
    """Create a minimal prompt with a task_completion_checker."""
    template = PromptTemplate[None](
        ns="test",
        key="with-checker",
        sections=[
            MarkdownSection(title="Task", key="task", template="Do something"),
        ],
        task_completion_checker=checker,
    )
    return Prompt(template)


class TestMultiturnEdgeCases:
    """Tests for edge cases in multi-turn continuation loop."""

    def test_deadline_exceeded_during_continuation(self, session: Session) -> None:
        """Test that continuation stops when deadline expires mid-loop."""
        check_count = 0

        def mock_remaining() -> timedelta:
            nonlocal check_count
            check_count += 1
            if check_count <= 5:
                return timedelta(seconds=10)
            return timedelta(seconds=-1)

        mock_deadline = MagicMock(spec=Deadline)
        mock_deadline.remaining.side_effect = mock_remaining

        class ForceContinuationChecker(TaskCompletionChecker):
            def __init__(self) -> None:
                self.check_count = 0

            def check(self, context: TaskCompletionContext) -> TaskCompletionResult:
                self.check_count += 1
                return TaskCompletionResult.incomplete("Please continue.")

        checker = ForceContinuationChecker()

        class MockClientDeadlineTest:
            def __init__(
                self, options: object | None = None, transport: object | None = None
            ) -> None:
                self.options = options
                self.receive_count = 0
                self._transport = MockTransport()
                MockSDKQuery.captured_options.append(options)

            async def connect(self, prompt: object | None = None) -> None:
                pass

            async def disconnect(self) -> None:
                self._transport = None

            async def query(self, prompt: str, session_id: str = "default") -> None:
                pass

            async def receive_response(self) -> AsyncGenerator[object, None]:
                self.receive_count += 1
                yield MockResultMessage(
                    result=f"Response {self.receive_count}",
                    usage={"input_tokens": 10, "output_tokens": 5},
                )

        setup_mock_query([])
        adapter = ClaudeAgentSDKAdapter()
        prompt = _make_prompt_with_checker(checker)

        with sdk_patches():
            with patch("claude_agent_sdk.ClaudeSDKClient", MockClientDeadlineTest):
                response = adapter.evaluate(
                    prompt, session=session, deadline=mock_deadline
                )

        assert response.text is not None
        assert check_count >= 6

    def test_budget_check_raises_exception_during_continuation(
        self, session: Session
    ) -> None:
        """Test that continuation stops when budget check raises an exception."""
        from weakincentives.budget import Budget, BudgetExceededError, BudgetTracker

        budget = Budget(max_total_tokens=1000)
        budget_tracker = BudgetTracker(budget)

        check_count = 0
        original_check = budget_tracker.check

        def mock_check() -> None:
            nonlocal check_count
            check_count += 1
            if check_count > 1:
                raise BudgetExceededError("Budget exceeded during continuation")
            original_check()

        budget_tracker.check = mock_check  # type: ignore[method-assign]

        class ForceContinuationChecker(TaskCompletionChecker):
            def __init__(self) -> None:
                self.check_count = 0

            def check(self, context: TaskCompletionContext) -> TaskCompletionResult:
                self.check_count += 1
                return TaskCompletionResult.incomplete("Please continue.")

        checker = ForceContinuationChecker()

        class MockClientBudgetTest:
            def __init__(
                self, options: object | None = None, transport: object | None = None
            ) -> None:
                self.options = options
                self.receive_count = 0
                self._transport = MockTransport()
                MockSDKQuery.captured_options.append(options)

            async def connect(self, prompt: object | None = None) -> None:
                pass

            async def disconnect(self) -> None:
                self._transport = None

            async def query(self, prompt: str, session_id: str = "default") -> None:
                pass

            async def receive_response(self) -> AsyncGenerator[object, None]:
                self.receive_count += 1
                yield MockResultMessage(
                    result=f"Response {self.receive_count}",
                    usage={"input_tokens": 100, "output_tokens": 50},
                )

        setup_mock_query([])
        adapter = ClaudeAgentSDKAdapter()
        prompt = _make_prompt_with_checker(checker)

        with sdk_patches():
            with patch("claude_agent_sdk.ClaudeSDKClient", MockClientBudgetTest):
                response = adapter.evaluate(
                    prompt, session=session, budget_tracker=budget_tracker
                )

        assert response.text == "Response 1"
        assert check_count >= 2

    def test_empty_message_stream_breaks_continuation(
        self, session: Session, untyped_prompt: Prompt[None]
    ) -> None:
        """Test that continuation stops when receive_response returns no messages."""

        class MockClientEmptyStream:
            def __init__(
                self, options: object | None = None, transport: object | None = None
            ) -> None:
                self.options = options
                self._transport = MockTransport()
                MockSDKQuery.captured_options.append(options)

            async def connect(self, prompt: object | None = None) -> None:
                pass

            async def disconnect(self) -> None:
                self._transport = None

            async def query(self, prompt: str, session_id: str = "default") -> None:
                pass

            async def receive_response(self) -> AsyncGenerator[object, None]:
                return
                yield  # pragma: no cover - makes this an async generator

        setup_mock_query([])
        adapter = ClaudeAgentSDKAdapter()

        with sdk_patches():
            with patch("claude_agent_sdk.ClaudeSDKClient", MockClientEmptyStream):
                response = adapter.evaluate(untyped_prompt, session=session)

        assert response.text is None

    def test_deadline_timeout_while_waiting_for_response_stream(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        """Deadline should interrupt stalled response streams."""
        deadline = Deadline(datetime.now(UTC) + timedelta(seconds=1.5))

        class MockClientStalledStream:
            def __init__(
                self, options: object | None = None, transport: object | None = None
            ) -> None:
                self.options = options
                self._transport = MockTransport()
                MockSDKQuery.captured_options.append(options)

            async def connect(self, prompt: object | None = None) -> None:
                pass

            async def disconnect(self) -> None:
                self._transport = None

            async def query(self, prompt: str, session_id: str = "default") -> None:
                pass

            async def receive_response(self) -> AsyncGenerator[object, None]:
                await asyncio.sleep(60)
                yield MockResultMessage(result="unreachable")

        setup_mock_query([])
        adapter = ClaudeAgentSDKAdapter()

        with sdk_patches():
            with patch("claude_agent_sdk.ClaudeSDKClient", MockClientStalledStream):
                with pytest.raises(
                    PromptEvaluationError,
                    match="Deadline exceeded while waiting for Claude SDK response stream",
                ):
                    adapter.evaluate(
                        simple_prompt,
                        session=session,
                        deadline=deadline,
                    )

    def test_cleanup_handles_none_transport(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        """Test that cleanup handles the case where _transport is None."""

        class MockClientNoTransport:
            def __init__(
                self, options: object | None = None, transport: object | None = None
            ) -> None:
                self.options = options
                self._transport = None
                MockSDKQuery.captured_options.append(options)

            async def connect(self, prompt: object | None = None) -> None:
                pass

            async def disconnect(self) -> None:
                pass

            async def query(self, prompt: str, session_id: str = "default") -> None:
                pass

            async def receive_response(self) -> AsyncGenerator[object, None]:
                yield MockResultMessage(
                    result="Response with no transport",
                    usage={"input_tokens": 10, "output_tokens": 5},
                )

        setup_mock_query([])
        adapter = ClaudeAgentSDKAdapter()

        with sdk_patches():
            with patch("claude_agent_sdk.ClaudeSDKClient", MockClientNoTransport):
                response = adapter.evaluate(simple_prompt, session=session)

        assert response.text == "Response with no transport"

    def test_structured_output_used_for_completion_check(
        self, session: Session
    ) -> None:
        """Test that structured_output is used for task completion checking."""
        captured_outputs: list[object] = []

        class OutputCapturingChecker(TaskCompletionChecker):
            def check(self, context: TaskCompletionContext) -> TaskCompletionResult:
                captured_outputs.append(context.tentative_output)
                return TaskCompletionResult.ok("Complete")

        checker = OutputCapturingChecker()

        class MockClientStructuredOutput:
            def __init__(
                self, options: object | None = None, transport: object | None = None
            ) -> None:
                self.options = options
                self._transport = MockTransport()
                MockSDKQuery.captured_options.append(options)

            async def connect(self, prompt: object | None = None) -> None:
                pass

            async def disconnect(self) -> None:
                self._transport = None

            async def query(self, prompt: str, session_id: str = "default") -> None:
                pass

            async def receive_response(self) -> AsyncGenerator[object, None]:
                yield MockResultMessage(
                    result="Text result",
                    structured_output={"key": "structured_value"},
                    usage={"input_tokens": 10, "output_tokens": 5},
                )

        setup_mock_query([])
        adapter = ClaudeAgentSDKAdapter()
        prompt = _make_prompt_with_checker(checker)

        with sdk_patches():
            with patch("claude_agent_sdk.ClaudeSDKClient", MockClientStructuredOutput):
                response = adapter.evaluate(prompt, session=session)

        assert len(captured_outputs) == 1
        assert captured_outputs[0] == {"key": "structured_value"}
        assert response.text == "Text result"

    def test_incomplete_without_feedback_exits_loop(self, session: Session) -> None:
        """Test that incomplete result without feedback exits the loop."""

        class NoFeedbackChecker(TaskCompletionChecker):
            def check(self, context: TaskCompletionContext) -> TaskCompletionResult:
                return TaskCompletionResult(complete=False, feedback=None)

        checker = NoFeedbackChecker()

        class MockClientNoFeedback:
            def __init__(
                self, options: object | None = None, transport: object | None = None
            ) -> None:
                self.options = options
                self.receive_count = 0
                self._transport = MockTransport()
                MockSDKQuery.captured_options.append(options)

            async def connect(self, prompt: object | None = None) -> None:
                pass

            async def disconnect(self) -> None:
                self._transport = None

            async def query(self, prompt: str, session_id: str = "default") -> None:
                pass

            async def receive_response(self) -> AsyncGenerator[object, None]:
                self.receive_count += 1
                yield MockResultMessage(
                    result=f"Response {self.receive_count}",
                    usage={"input_tokens": 10, "output_tokens": 5},
                )

        setup_mock_query([])
        adapter = ClaudeAgentSDKAdapter()
        prompt = _make_prompt_with_checker(checker)

        with sdk_patches():
            with patch("claude_agent_sdk.ClaudeSDKClient", MockClientNoFeedback):
                response = adapter.evaluate(prompt, session=session)

        assert response.text == "Response 1"
