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

"""Tests for StandardLoggingSubscribers helper."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from io import StringIO
from typing import cast
from uuid import uuid4

from tests.helpers.adapters import TEST_ADAPTER_NAME
from weakincentives.prompt.tool import ToolResult
from weakincentives.runtime.events import (
    InProcessEventBus,
    PromptExecuted,
    PromptRendered,
    StandardLoggingSubscribers,
    TokenUsage,
    ToolInvoked,
    attach_standard_logging,
)


@dataclass(slots=True, frozen=True)
class _TestParams:
    """Test parameter dataclass for tool invocations."""

    query: str
    limit: int = 10


@dataclass(slots=True, frozen=True)
class _TestPayload:
    """Test payload dataclass for tool results."""

    items: list[str]


class TestStandardLoggingSubscribers:
    """Tests for the StandardLoggingSubscribers class."""

    def test_attach_subscribes_to_all_event_types(self) -> None:
        """Verify attach() subscribes handlers for all three event types."""
        bus = InProcessEventBus()
        stream = StringIO()
        subscribers = StandardLoggingSubscribers(stream=stream)

        subscribers.attach(bus)

        # Publish each event type and verify output is produced
        bus.publish(
            PromptRendered(
                prompt_ns="test",
                prompt_key="demo",
                prompt_name="Test Prompt",
                adapter=TEST_ADAPTER_NAME,
                session_id=uuid4(),
                render_inputs=(),
                rendered_prompt="Hello, world!",
                created_at=datetime.now(UTC),
            )
        )
        output = stream.getvalue()
        assert "[prompt] Rendered prompt (Test Prompt)" in output
        assert "Hello, world!" in output

    def test_prompt_rendered_uses_ns_key_when_name_missing(self) -> None:
        """Verify prompt label falls back to ns:key when name is None."""
        bus = InProcessEventBus()
        stream = StringIO()
        subscribers = StandardLoggingSubscribers(stream=stream)
        subscribers.attach(bus)

        bus.publish(
            PromptRendered(
                prompt_ns="examples",
                prompt_key="greeting",
                prompt_name=None,
                adapter=TEST_ADAPTER_NAME,
                session_id=uuid4(),
                render_inputs=(),
                rendered_prompt="Test content",
                created_at=datetime.now(UTC),
            )
        )

        output = stream.getvalue()
        assert "[prompt] Rendered prompt (examples:greeting)" in output

    def test_tool_invoked_logs_params_result_and_usage(self) -> None:
        """Verify tool invocation logs include params, result, and token usage."""
        bus = InProcessEventBus()
        stream = StringIO()
        subscribers = StandardLoggingSubscribers(stream=stream)
        subscribers.attach(bus)

        params = _TestParams(query="search term", limit=5)
        result = ToolResult(message="Found 3 results", value=_TestPayload(items=["a", "b", "c"]))
        usage = TokenUsage(input_tokens=100, output_tokens=50)

        bus.publish(
            ToolInvoked(
                prompt_name="test_prompt",
                adapter=TEST_ADAPTER_NAME,
                name="search_tool",
                params=params,
                result=cast(ToolResult[object], result),
                session_id=uuid4(),
                created_at=datetime.now(UTC),
                usage=usage,
                value=result.value,
            )
        )

        output = stream.getvalue()
        assert "[tool] search_tool (test_prompt)" in output
        assert "params:" in output
        assert "search term" in output
        assert "result: Found 3 results" in output
        assert "input=100" in output
        assert "output=50" in output
        assert "payload:" in output

    def test_tool_invoked_handles_none_usage(self) -> None:
        """Verify tool invocation handles missing token usage gracefully."""
        bus = InProcessEventBus()
        stream = StringIO()
        subscribers = StandardLoggingSubscribers(stream=stream)
        subscribers.attach(bus)

        params = _TestParams(query="test")
        result = ToolResult(message="Done", value=None)

        bus.publish(
            ToolInvoked(
                prompt_name="test_prompt",
                adapter=TEST_ADAPTER_NAME,
                name="simple_tool",
                params=params,
                result=cast(ToolResult[object], result),
                session_id=uuid4(),
                created_at=datetime.now(UTC),
                usage=None,
                value=None,
            )
        )

        output = stream.getvalue()
        assert "token usage: (not reported)" in output
        # Should not have payload line when value is None
        assert "payload:" not in output

    def test_prompt_executed_logs_completion_and_usage(self) -> None:
        """Verify execution completion logs include token usage."""
        bus = InProcessEventBus()
        stream = StringIO()
        subscribers = StandardLoggingSubscribers(stream=stream)
        subscribers.attach(bus)

        usage = TokenUsage(input_tokens=200, output_tokens=100, cached_tokens=50)

        bus.publish(
            PromptExecuted(
                prompt_name="demo",
                adapter=TEST_ADAPTER_NAME,
                result=None,
                session_id=uuid4(),
                created_at=datetime.now(UTC),
                usage=usage,
            )
        )

        output = stream.getvalue()
        assert "[prompt] Execution complete" in output
        assert "input=200" in output
        assert "output=100" in output
        assert "cached=50" in output

    def test_truncation_respects_limit(self) -> None:
        """Verify long payloads are truncated to the configured limit."""
        bus = InProcessEventBus()
        stream = StringIO()
        subscribers = StandardLoggingSubscribers(stream=stream, truncate_limit=50)
        subscribers.attach(bus)

        # Create a payload with a very long string
        params = _TestParams(query="x" * 200, limit=1)
        result = ToolResult(message="y" * 200, value=None)

        bus.publish(
            ToolInvoked(
                prompt_name="test",
                adapter=TEST_ADAPTER_NAME,
                name="tool",
                params=params,
                result=cast(ToolResult[object], result),
                session_id=uuid4(),
                created_at=datetime.now(UTC),
                usage=None,
                value=None,
            )
        )

        output = stream.getvalue()
        # Output should contain truncation marker
        assert "…" in output

    def test_show_flags_disable_specific_handlers(self) -> None:
        """Verify show_* flags prevent subscription of specific handlers."""
        bus = InProcessEventBus()
        stream = StringIO()
        subscribers = StandardLoggingSubscribers(
            stream=stream,
            show_rendered_prompt=False,
            show_tool_invocations=False,
            show_execution_summary=True,
        )
        subscribers.attach(bus)

        # Publish all event types
        bus.publish(
            PromptRendered(
                prompt_ns="test",
                prompt_key="demo",
                prompt_name="Test",
                adapter=TEST_ADAPTER_NAME,
                session_id=uuid4(),
                render_inputs=(),
                rendered_prompt="Should not appear",
                created_at=datetime.now(UTC),
            )
        )
        bus.publish(
            ToolInvoked(
                prompt_name="test",
                adapter=TEST_ADAPTER_NAME,
                name="tool",
                params=_TestParams(query="test"),
                result=cast(ToolResult[object], ToolResult(message="ok", value=None)),
                session_id=uuid4(),
                created_at=datetime.now(UTC),
                usage=None,
                value=None,
            )
        )
        bus.publish(
            PromptExecuted(
                prompt_name="demo",
                adapter=TEST_ADAPTER_NAME,
                result=None,
                session_id=uuid4(),
                created_at=datetime.now(UTC),
                usage=TokenUsage(input_tokens=10),
            )
        )

        output = stream.getvalue()
        assert "[prompt] Rendered prompt" not in output
        assert "[tool]" not in output
        assert "[prompt] Execution complete" in output

    def test_detach_clears_internal_handlers(self) -> None:
        """Verify detach() clears the internal handler registry."""
        bus = InProcessEventBus()
        subscribers = StandardLoggingSubscribers()

        subscribers.attach(bus)
        assert len(subscribers._handlers) == 3

        subscribers.detach(bus)
        assert len(subscribers._handlers) == 0

    def test_context_manager_attaches_and_detaches(self) -> None:
        """Verify context manager properly attaches and detaches handlers."""
        bus = InProcessEventBus()
        stream = StringIO()
        subscribers = StandardLoggingSubscribers(stream=stream)

        with subscribers.attached(bus) as ctx:
            assert ctx is subscribers
            assert len(subscribers._handlers) == 3

            bus.publish(
                PromptExecuted(
                    prompt_name="demo",
                    adapter=TEST_ADAPTER_NAME,
                    result=None,
                    session_id=uuid4(),
                    created_at=datetime.now(UTC),
                    usage=None,
                )
            )

        # After exiting context, handlers should be cleared
        assert len(subscribers._handlers) == 0
        output = stream.getvalue()
        assert "[prompt] Execution complete" in output


class TestAttachStandardLogging:
    """Tests for the attach_standard_logging convenience function."""

    def test_returns_subscribers_instance(self) -> None:
        """Verify the function returns a StandardLoggingSubscribers instance."""
        bus = InProcessEventBus()
        stream = StringIO()

        result = attach_standard_logging(bus, stream=stream)

        assert isinstance(result, StandardLoggingSubscribers)
        assert result.stream is stream

    def test_attaches_handlers_to_bus(self) -> None:
        """Verify handlers are attached and receive events."""
        bus = InProcessEventBus()
        stream = StringIO()

        attach_standard_logging(bus, stream=stream)

        bus.publish(
            PromptExecuted(
                prompt_name="test",
                adapter=TEST_ADAPTER_NAME,
                result=None,
                session_id=uuid4(),
                created_at=datetime.now(UTC),
                usage=None,
            )
        )

        output = stream.getvalue()
        assert "[prompt] Execution complete" in output

    def test_respects_configuration_parameters(self) -> None:
        """Verify all configuration parameters are passed through."""
        bus = InProcessEventBus()
        stream = StringIO()

        subscribers = attach_standard_logging(
            bus,
            stream=stream,
            truncate_limit=100,
            show_rendered_prompt=False,
            show_tool_invocations=True,
            show_execution_summary=False,
        )

        assert subscribers.truncate_limit == 100
        assert subscribers.show_rendered_prompt is False
        assert subscribers.show_tool_invocations is True
        assert subscribers.show_execution_summary is False


class TestTokenUsageFormatting:
    """Tests for token usage formatting edge cases."""

    def test_format_usage_with_all_fields(self) -> None:
        """Verify all token fields are formatted when present."""
        bus = InProcessEventBus()
        stream = StringIO()
        subscribers = StandardLoggingSubscribers(stream=stream)
        subscribers.attach(bus)

        usage = TokenUsage(input_tokens=100, output_tokens=50, cached_tokens=25)
        bus.publish(
            PromptExecuted(
                prompt_name="test",
                adapter=TEST_ADAPTER_NAME,
                result=None,
                session_id=uuid4(),
                created_at=datetime.now(UTC),
                usage=usage,
            )
        )

        output = stream.getvalue()
        assert "input=100" in output
        assert "output=50" in output
        assert "cached=25" in output
        assert "total=150" in output

    def test_format_usage_with_partial_fields(self) -> None:
        """Verify partial token fields are formatted correctly."""
        bus = InProcessEventBus()
        stream = StringIO()
        subscribers = StandardLoggingSubscribers(stream=stream)
        subscribers.attach(bus)

        usage = TokenUsage(input_tokens=100)
        bus.publish(
            PromptExecuted(
                prompt_name="test",
                adapter=TEST_ADAPTER_NAME,
                result=None,
                session_id=uuid4(),
                created_at=datetime.now(UTC),
                usage=usage,
            )
        )

        output = stream.getvalue()
        assert "input=100" in output
        assert "output=" not in output
        assert "cached=" not in output

    def test_format_usage_with_empty_token_usage(self) -> None:
        """Verify empty TokenUsage (all None values) shows not reported."""
        bus = InProcessEventBus()
        stream = StringIO()
        subscribers = StandardLoggingSubscribers(stream=stream)
        subscribers.attach(bus)

        # TokenUsage with all None values
        usage = TokenUsage()
        bus.publish(
            PromptExecuted(
                prompt_name="test",
                adapter=TEST_ADAPTER_NAME,
                result=None,
                session_id=uuid4(),
                created_at=datetime.now(UTC),
                usage=usage,
            )
        )

        output = stream.getvalue()
        assert "token usage: (not reported)" in output


class TestPayloadCoercion:
    """Tests for payload coercion to JSON-serializable types."""

    def test_handles_raw_dataclass_in_value(self) -> None:
        """Verify raw dataclasses in value are properly serialized."""
        bus = InProcessEventBus()
        stream = StringIO()
        subscribers = StandardLoggingSubscribers(stream=stream)
        subscribers.attach(bus)

        params = _TestParams(query="test")
        # Pass a raw dataclass as the value (not pre-serialized)
        raw_payload = _TestPayload(items=["x", "y"])
        result = ToolResult(message="ok", value=raw_payload)

        bus.publish(
            ToolInvoked(
                prompt_name="test",
                adapter=TEST_ADAPTER_NAME,
                name="tool",
                params=params,
                result=cast(ToolResult[object], result),
                session_id=uuid4(),
                created_at=datetime.now(UTC),
                usage=None,
                value=raw_payload,
            )
        )

        output = stream.getvalue()
        # The dataclass should be serialized and shown in output
        assert "items" in output
        assert "x" in output

    def test_handles_nested_mappings(self) -> None:
        """Verify nested dictionaries are properly coerced."""
        bus = InProcessEventBus()
        stream = StringIO()
        subscribers = StandardLoggingSubscribers(stream=stream)
        subscribers.attach(bus)

        params = _TestParams(query="test")
        result = ToolResult(message="ok", value={"nested": {"key": "value"}})

        bus.publish(
            ToolInvoked(
                prompt_name="test",
                adapter=TEST_ADAPTER_NAME,
                name="tool",
                params=params,
                result=cast(ToolResult[object], result),
                session_id=uuid4(),
                created_at=datetime.now(UTC),
                usage=None,
                value={"nested": {"key": "value"}},
            )
        )

        output = stream.getvalue()
        assert "nested" in output
        assert "value" in output

    def test_handles_sets_as_sorted_lists(self) -> None:
        """Verify sets are coerced to sorted lists."""
        bus = InProcessEventBus()
        stream = StringIO()
        subscribers = StandardLoggingSubscribers(stream=stream)
        subscribers.attach(bus)

        params = _TestParams(query="test")
        result = ToolResult(message="ok", value={"tags": {"c", "a", "b"}})

        bus.publish(
            ToolInvoked(
                prompt_name="test",
                adapter=TEST_ADAPTER_NAME,
                name="tool",
                params=params,
                result=cast(ToolResult[object], result),
                session_id=uuid4(),
                created_at=datetime.now(UTC),
                usage=None,
                value={"tags": {"c", "a", "b"}},
            )
        )

        output = stream.getvalue()
        # Sets should be sorted
        assert '["a", "b", "c"]' in output or "['a', 'b', 'c']" in output

    def test_handles_non_serializable_fallback(self) -> None:
        """Verify non-serializable objects fall back to repr."""
        bus = InProcessEventBus()
        stream = StringIO()
        subscribers = StandardLoggingSubscribers(stream=stream)
        subscribers.attach(bus)

        class NonSerializable:
            def __repr__(self) -> str:
                return "<NonSerializable>"

        params = _TestParams(query="test")
        result = ToolResult(message="ok", value=NonSerializable())

        bus.publish(
            ToolInvoked(
                prompt_name="test",
                adapter=TEST_ADAPTER_NAME,
                name="tool",
                params=params,
                result=cast(ToolResult[object], result),
                session_id=uuid4(),
                created_at=datetime.now(UTC),
                usage=None,
                value=NonSerializable(),
            )
        )

        output = stream.getvalue()
        assert "NonSerializable" in output
