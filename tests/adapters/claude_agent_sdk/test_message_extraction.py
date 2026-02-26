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

"""Tests for message content extraction helper functions."""

from __future__ import annotations

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
from weakincentives.prompt import MarkdownSection, Prompt, PromptTemplate
from weakincentives.runtime.session import Session


class TestMessageContentExtraction:
    """Tests for the message content extraction helper functions."""

    def test_extract_content_block_tool_use(self) -> None:
        """Tool use blocks include name, id, and input."""
        from weakincentives.adapters.claude_agent_sdk._message_extraction import (
            _extract_content_block,
        )

        block = {
            "type": "tool_use",
            "name": "my_tool",
            "id": "toolu_123",
            "input": {"path": "/foo"},
        }
        result = _extract_content_block(block)
        assert result["type"] == "tool_use"
        assert result["name"] == "my_tool"
        assert result["id"] == "toolu_123"
        assert result["input"] == {"path": "/foo"}

    def test_extract_content_block_text(self) -> None:
        """Text blocks include full text content."""
        from weakincentives.adapters.claude_agent_sdk._message_extraction import (
            _extract_content_block,
        )

        block = {"type": "text", "text": "Hello world, this is a long message"}
        result = _extract_content_block(block)
        assert result["type"] == "text"
        assert result["text"] == "Hello world, this is a long message"

    def test_extract_content_block_tool_result(self) -> None:
        """Tool result blocks include tool_use_id and full content."""
        from weakincentives.adapters.claude_agent_sdk._message_extraction import (
            _extract_content_block,
        )

        block = {
            "type": "tool_result",
            "tool_use_id": "toolu_123",
            "content": "Full result content here",
            "is_error": False,
        }
        result = _extract_content_block(block)
        assert result["type"] == "tool_result"
        assert result["tool_use_id"] == "toolu_123"
        assert result["content"] == "Full result content here"
        assert result["is_error"] is False

    def test_extract_content_block_tool_result_no_is_error(self) -> None:
        """Tool result blocks without is_error field work correctly."""
        from weakincentives.adapters.claude_agent_sdk._message_extraction import (
            _extract_content_block,
        )

        block = {
            "type": "tool_result",
            "tool_use_id": "toolu_456",
            "content": "Some content",
        }
        result = _extract_content_block(block)
        assert result["type"] == "tool_result"
        assert result["tool_use_id"] == "toolu_456"
        assert result["content"] == "Some content"
        assert "is_error" not in result

    def test_extract_content_block_unknown_type(self) -> None:
        """Unknown block types include all fields."""
        from weakincentives.adapters.claude_agent_sdk._message_extraction import (
            _extract_content_block,
        )

        block = {"type": "image", "data": "base64...", "media_type": "image/png"}
        result = _extract_content_block(block)
        assert result["type"] == "image"
        assert result["data"] == "base64..."
        assert result["media_type"] == "image/png"

    def test_extract_list_content_mixed(self) -> None:
        """List content extracts all blocks with full content."""
        from weakincentives.adapters.claude_agent_sdk._message_extraction import (
            _extract_list_content,
        )

        content = [
            {"type": "text", "text": "Hello"},
            {
                "type": "tool_use",
                "name": "read_file",
                "id": "t1",
                "input": {"path": "/x"},
            },
            {"type": "text", "text": "World"},
        ]
        result = _extract_list_content(content)
        assert len(result) == 3
        assert result[0] == {"type": "text", "text": "Hello"}
        assert result[1]["name"] == "read_file"
        assert result[1]["input"] == {"path": "/x"}
        assert result[2] == {"type": "text", "text": "World"}

    def test_extract_list_content_skips_non_dict(self) -> None:
        """Non-dict blocks are skipped."""
        from weakincentives.adapters.claude_agent_sdk._message_extraction import (
            _extract_list_content,
        )

        content = [
            {"type": "text", "text": "Hello"},
            "not a dict",
            123,
            {"type": "text", "text": "World"},
        ]
        result = _extract_list_content(content)
        assert len(result) == 2
        assert result[0] == {"type": "text", "text": "Hello"}
        assert result[1] == {"type": "text", "text": "World"}

    def test_multiturn_with_task_completion_checker(self, session: Session) -> None:
        """Test multi-turn conversations with task completion checking."""

        class TestChecker(TaskCompletionChecker):
            def __init__(self) -> None:
                self.check_count = 0

            def check(self, context: TaskCompletionContext) -> TaskCompletionResult:
                self.check_count += 1
                if self.check_count == 1:
                    return TaskCompletionResult.incomplete(
                        "Please continue working on the task."
                    )
                return TaskCompletionResult.ok("Task complete.")

        checker = TestChecker()
        template = PromptTemplate[None](
            ns="test",
            key="with-checker",
            sections=[
                MarkdownSection(title="Task", key="task", template="Do something"),
            ],
            task_completion_checker=checker,
        )
        prompt = Prompt(template)
        adapter = ClaudeAgentSDKAdapter()

        class MockClient:
            def __init__(
                self, options: object | None = None, transport: object | None = None
            ) -> None:
                self.options = options
                self.query_count = 0
                self.receive_count = 0
                self.feedback_received: list[str] = []
                self._transport = MockTransport()
                MockSDKQuery.captured_options.append(options)

            async def connect(self, prompt: object | None = None) -> None:
                pass

            async def disconnect(self) -> None:
                self._transport = None

            async def query(self, prompt: str, session_id: str = "default") -> None:
                self.query_count += 1
                self.feedback_received.append(prompt)

            async def receive_response(self) -> AsyncGenerator[object, None]:
                current_receive = self.receive_count
                self.receive_count += 1

                if current_receive == 0:
                    yield MockResultMessage(
                        result="Partial work",
                        usage={"input_tokens": 10, "output_tokens": 5},
                    )
                elif current_receive == 1:
                    yield MockResultMessage(
                        result="Complete work",
                        usage={"input_tokens": 5, "output_tokens": 3},
                    )

        setup_mock_query([])
        with sdk_patches():
            with patch("claude_agent_sdk.ClaudeSDKClient", MockClient):
                response = adapter.evaluate(prompt, session=session)

        assert checker.check_count == 2
        assert response.output is None

    def test_multiturn_deadline_exceeded(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        """Test that evaluation raises error when deadline already expired."""
        from weakincentives.clock import FakeClock
        from weakincentives.deadlines import Deadline

        clock = FakeClock()
        anchor = datetime.now(UTC)
        clock.set_wall(anchor)
        expired_deadline = Deadline(anchor + timedelta(seconds=1), clock=clock)
        clock.advance(2)

        setup_mock_query(
            [
                MockResultMessage(
                    result="Test", usage={"input_tokens": 10, "output_tokens": 5}
                ),
            ]
        )

        adapter = ClaudeAgentSDKAdapter()

        with sdk_patches():
            with pytest.raises(PromptEvaluationError, match="Deadline expired"):
                adapter.evaluate(
                    simple_prompt, session=session, deadline=expired_deadline
                )

    def test_multiturn_budget_exceeded(
        self, session: Session, simple_prompt: Prompt[SimpleOutput]
    ) -> None:
        """Test that multi-turn stops when budget is exceeded."""
        from weakincentives.budget import Budget, BudgetTracker

        budget = Budget(max_output_tokens=1)
        budget_tracker = BudgetTracker(budget)

        from weakincentives.runtime.events import TokenUsage

        budget_tracker.record_cumulative("test", TokenUsage(output_tokens=0))

        setup_mock_query(
            [
                MockResultMessage(
                    result="Test", usage={"input_tokens": 10, "output_tokens": 5}
                ),
            ]
        )

        adapter = ClaudeAgentSDKAdapter()

        with sdk_patches():
            response = adapter.evaluate(
                simple_prompt, session=session, budget_tracker=budget_tracker
            )
            assert response.output is None

    def test_extract_inner_message_content_string(self) -> None:
        """String content is extracted fully."""
        from weakincentives.adapters.claude_agent_sdk._message_extraction import (
            _extract_inner_message_content,
        )

        inner_msg = {"role": "assistant", "content": "Full message content here"}
        result = _extract_inner_message_content(inner_msg)
        assert result == {"role": "assistant", "content": "Full message content here"}

    def test_extract_inner_message_content_list(self) -> None:
        """List content is extracted as content_blocks."""
        from weakincentives.adapters.claude_agent_sdk._message_extraction import (
            _extract_inner_message_content,
        )

        inner_msg = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Thinking..."},
                {"type": "tool_use", "name": "bash", "id": "t1"},
            ],
        }
        result = _extract_inner_message_content(inner_msg)
        assert result["role"] == "assistant"
        assert len(result["content_blocks"]) == 2
        assert result["content_blocks"][0]["text"] == "Thinking..."
        assert result["content_blocks"][1]["name"] == "bash"

    def test_extract_inner_message_content_no_role(self) -> None:
        """Inner message without role skips role field."""
        from weakincentives.adapters.claude_agent_sdk._message_extraction import (
            _extract_inner_message_content,
        )

        inner_msg = {"content": "Hello"}
        result = _extract_inner_message_content(inner_msg)
        assert "role" not in result
        assert result["content"] == "Hello"

    def test_extract_inner_message_content_non_str_non_list(self) -> None:
        """Non-string/non-list content returns only role."""
        from weakincentives.adapters.claude_agent_sdk._message_extraction import (
            _extract_inner_message_content,
        )

        inner_msg = {"role": "user", "content": 12345}
        result = _extract_inner_message_content(inner_msg)
        assert result == {"role": "user"}

    def test_extract_message_content_with_inner_message(self) -> None:
        """Message with inner message dict extracts full content."""
        from weakincentives.adapters.claude_agent_sdk._message_extraction import (
            _extract_message_content,
        )

        message = MagicMock()
        message.message = {"role": "user", "content": "Full user message"}
        message.result = None
        message.structured_output = None
        message.usage = None

        result = _extract_message_content(message)
        assert result["role"] == "user"
        assert result["content"] == "Full user message"

    def test_extract_message_content_with_result(self) -> None:
        """ResultMessage with result field extracts full result."""
        from weakincentives.adapters.claude_agent_sdk._message_extraction import (
            _extract_message_content,
        )

        message = MagicMock()
        message.message = None
        message.result = "Final answer with full content"
        message.structured_output = None
        message.usage = None

        result = _extract_message_content(message)
        assert result["result"] == "Final answer with full content"

    def test_extract_message_content_with_structured_output(self) -> None:
        """Message with structured_output includes full structured output."""
        from weakincentives.adapters.claude_agent_sdk._message_extraction import (
            _extract_message_content,
        )

        message = MagicMock()
        message.message = None
        message.result = None
        message.structured_output = {"summary": "test", "issues": ["a", "b"]}
        message.usage = None

        result = _extract_message_content(message)
        assert result["structured_output"] == {"summary": "test", "issues": ["a", "b"]}

    def test_extract_message_content_with_usage(self) -> None:
        """Message with usage includes usage data."""
        from weakincentives.adapters.claude_agent_sdk._message_extraction import (
            _extract_message_content,
        )

        message = MagicMock()
        message.message = None
        message.result = None
        message.structured_output = None
        message.usage = {"input_tokens": 100, "output_tokens": 50}

        result = _extract_message_content(message)
        assert result["usage"] == {"input_tokens": 100, "output_tokens": 50}

    def test_extract_message_content_no_attrs(self) -> None:
        """Message without expected attributes returns empty dict."""
        from weakincentives.adapters.claude_agent_sdk._message_extraction import (
            _extract_message_content,
        )

        message = MagicMock(spec=[])

        result = _extract_message_content(message)
        assert result == {}
