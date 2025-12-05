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

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

import pytest

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
from weakincentives.runtime.logging import StructuredLogger, get_logger


@dataclass(slots=True, frozen=True)
class _TestParams:
    """Test parameter dataclass for tool invocations."""

    query: str
    limit: int = 10


@dataclass(slots=True, frozen=True)
class _TestPayload:
    """Test payload dataclass for tool results."""

    items: list[str]


def _find_log_record(
    caplog: pytest.LogCaptureFixture, event_name: str
) -> logging.LogRecord | None:
    """Find a log record with the given event name."""
    for record in caplog.records:
        if getattr(record, "event", None) == event_name:
            return record
    return None


def _get_context(record: logging.LogRecord) -> dict[str, object]:
    """Extract context dict from a log record."""
    return getattr(record, "context", {})


class TestStandardLoggingSubscribers:
    """Tests for the StandardLoggingSubscribers class."""

    def test_attach_subscribes_to_all_event_types(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify attach() subscribes handlers for all three event types."""
        bus = InProcessEventBus()
        subscribers = StandardLoggingSubscribers()
        subscribers.attach(bus)

        with caplog.at_level(logging.INFO):
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

        record = _find_log_record(caplog, "prompt_rendered")
        assert record is not None
        context = _get_context(record)
        assert context["prompt_name"] == "Test Prompt"
        assert context["rendered_prompt"] == "Hello, world!"

    def test_prompt_rendered_uses_ns_key_when_name_missing(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify prompt label falls back to ns:key when name is None."""
        bus = InProcessEventBus()
        subscribers = StandardLoggingSubscribers()
        subscribers.attach(bus)

        with caplog.at_level(logging.INFO):
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

        record = _find_log_record(caplog, "prompt_rendered")
        assert record is not None
        context = _get_context(record)
        assert context["prompt_name"] == "examples:greeting"

    def test_tool_invoked_logs_params_result_and_usage(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify tool invocation logs include params, result, and token usage."""
        bus = InProcessEventBus()
        subscribers = StandardLoggingSubscribers()
        subscribers.attach(bus)

        params = _TestParams(query="search term", limit=5)
        result = ToolResult(
            message="Found 3 results", value=_TestPayload(items=["a", "b", "c"])
        )
        usage = TokenUsage(input_tokens=100, output_tokens=50)

        with caplog.at_level(logging.INFO):
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

        record = _find_log_record(caplog, "tool_invoked")
        assert record is not None
        context = _get_context(record)
        assert context["tool_name"] == "search_tool"
        assert context["prompt_name"] == "test_prompt"
        assert "query" in str(context["params"])
        assert context["result_message"] == "Found 3 results"
        assert context["token_usage"]["input_tokens"] == 100
        assert context["token_usage"]["output_tokens"] == 50

    def test_tool_invoked_handles_none_usage(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify tool invocation handles missing token usage gracefully."""
        bus = InProcessEventBus()
        subscribers = StandardLoggingSubscribers()
        subscribers.attach(bus)

        params = _TestParams(query="test")
        result = ToolResult(message="Done", value=None)

        with caplog.at_level(logging.INFO):
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

        record = _find_log_record(caplog, "tool_invoked")
        assert record is not None
        context = _get_context(record)
        # No token_usage key when usage is None
        assert "token_usage" not in context
        # No result_payload when value is None
        assert "result_payload" not in context

    def test_prompt_executed_logs_completion_and_usage(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify execution completion logs include token usage."""
        bus = InProcessEventBus()
        subscribers = StandardLoggingSubscribers()
        subscribers.attach(bus)

        usage = TokenUsage(input_tokens=200, output_tokens=100, cached_tokens=50)

        with caplog.at_level(logging.INFO):
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

        record = _find_log_record(caplog, "prompt_executed")
        assert record is not None
        context = _get_context(record)
        assert context["prompt_name"] == "demo"
        assert context["token_usage"]["input_tokens"] == 200
        assert context["token_usage"]["output_tokens"] == 100
        assert context["token_usage"]["cached_tokens"] == 50

    def test_truncation_respects_limit(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify long result messages are truncated to the configured limit."""
        bus = InProcessEventBus()
        subscribers = StandardLoggingSubscribers(truncate_limit=50)
        subscribers.attach(bus)

        params = _TestParams(query="test", limit=1)
        long_message = "y" * 200
        result = ToolResult(message=long_message, value=None)

        with caplog.at_level(logging.INFO):
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

        record = _find_log_record(caplog, "tool_invoked")
        assert record is not None
        context = _get_context(record)
        # Result message should be truncated
        assert len(context["result_message"]) == 50
        assert context["result_message"].endswith("…")

    def test_show_flags_disable_specific_handlers(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify show_* flags prevent subscription of specific handlers."""
        bus = InProcessEventBus()
        subscribers = StandardLoggingSubscribers(
            show_rendered_prompt=False,
            show_tool_invocations=False,
            show_execution_summary=True,
        )
        subscribers.attach(bus)

        with caplog.at_level(logging.INFO):
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

        assert _find_log_record(caplog, "prompt_rendered") is None
        assert _find_log_record(caplog, "tool_invoked") is None
        assert _find_log_record(caplog, "prompt_executed") is not None

    def test_detach_clears_internal_handlers(self) -> None:
        """Verify detach() clears the internal handler registry."""
        bus = InProcessEventBus()
        subscribers = StandardLoggingSubscribers()

        subscribers.attach(bus)
        assert len(subscribers._handlers) == 3

        subscribers.detach(bus)
        assert len(subscribers._handlers) == 0

    def test_context_manager_attaches_and_detaches(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify context manager properly attaches and detaches handlers."""
        bus = InProcessEventBus()
        subscribers = StandardLoggingSubscribers()

        with subscribers.attached(bus) as ctx:
            assert ctx is subscribers
            assert len(subscribers._handlers) == 3

            with caplog.at_level(logging.INFO):
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
        assert _find_log_record(caplog, "prompt_executed") is not None

    def test_custom_logger_is_used(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verify a custom logger is used when provided."""
        bus = InProcessEventBus()
        custom_logger = get_logger("custom.test.logger")
        subscribers = StandardLoggingSubscribers(logger=custom_logger)
        subscribers.attach(bus)

        with caplog.at_level(logging.INFO, logger="custom.test.logger"):
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

        # Find log record from custom logger
        record = None
        for rec in caplog.records:
            if rec.name == "custom.test.logger":
                record = rec
                break
        assert record is not None

    def test_stdlib_logger_is_wrapped(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verify a stdlib Logger is wrapped in StructuredLogger."""
        bus = InProcessEventBus()
        stdlib_logger = logging.getLogger("stdlib.test.logger")
        subscribers = StandardLoggingSubscribers(logger=stdlib_logger)
        subscribers.attach(bus)

        # The _resolved_logger should be a StructuredLogger
        assert isinstance(subscribers._resolved_logger, StructuredLogger)

        with caplog.at_level(logging.INFO, logger="stdlib.test.logger"):
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

        record = _find_log_record(caplog, "prompt_executed")
        assert record is not None


class TestAttachStandardLogging:
    """Tests for the attach_standard_logging convenience function."""

    def test_returns_subscribers_instance(self) -> None:
        """Verify the function returns a StandardLoggingSubscribers instance."""
        bus = InProcessEventBus()

        result = attach_standard_logging(bus)

        assert isinstance(result, StandardLoggingSubscribers)

    def test_attaches_handlers_to_bus(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify handlers are attached and receive events."""
        bus = InProcessEventBus()

        attach_standard_logging(bus)

        with caplog.at_level(logging.INFO):
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

        record = _find_log_record(caplog, "prompt_executed")
        assert record is not None

    def test_respects_configuration_parameters(self) -> None:
        """Verify all configuration parameters are passed through."""
        bus = InProcessEventBus()
        custom_logger = get_logger("config.test")

        subscribers = attach_standard_logging(
            bus,
            truncate_limit=100,
            logger=custom_logger,
            show_rendered_prompt=False,
            show_tool_invocations=True,
            show_execution_summary=False,
        )

        assert subscribers.truncate_limit == 100
        assert subscribers.show_rendered_prompt is False
        assert subscribers.show_tool_invocations is True
        assert subscribers.show_execution_summary is False


class TestTokenUsageContext:
    """Tests for token usage context building."""

    def test_full_token_usage_context(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify all token fields are included in context when present."""
        bus = InProcessEventBus()
        subscribers = StandardLoggingSubscribers()
        subscribers.attach(bus)

        usage = TokenUsage(input_tokens=100, output_tokens=50, cached_tokens=25)

        with caplog.at_level(logging.INFO):
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

        record = _find_log_record(caplog, "prompt_executed")
        assert record is not None
        context = _get_context(record)
        token_usage = context["token_usage"]
        assert token_usage["input_tokens"] == 100
        assert token_usage["output_tokens"] == 50
        assert token_usage["cached_tokens"] == 25
        assert token_usage["total_tokens"] == 150

    def test_partial_token_usage_context(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify partial token fields are included correctly."""
        bus = InProcessEventBus()
        subscribers = StandardLoggingSubscribers()
        subscribers.attach(bus)

        usage = TokenUsage(input_tokens=100)

        with caplog.at_level(logging.INFO):
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

        record = _find_log_record(caplog, "prompt_executed")
        assert record is not None
        context = _get_context(record)
        token_usage = context["token_usage"]
        assert token_usage["input_tokens"] == 100
        assert "output_tokens" not in token_usage
        assert "cached_tokens" not in token_usage

    def test_empty_token_usage_not_included(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify empty TokenUsage (all None values) is not included."""
        bus = InProcessEventBus()
        subscribers = StandardLoggingSubscribers()
        subscribers.attach(bus)

        # TokenUsage with all None values
        usage = TokenUsage()

        with caplog.at_level(logging.INFO):
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

        record = _find_log_record(caplog, "prompt_executed")
        assert record is not None
        context = _get_context(record)
        # Empty token usage should not be included
        assert "token_usage" not in context


class TestPayloadCoercion:
    """Tests for payload coercion to JSON-serializable types."""

    def test_handles_raw_dataclass_in_value(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify raw dataclasses in value are properly serialized."""
        bus = InProcessEventBus()
        subscribers = StandardLoggingSubscribers()
        subscribers.attach(bus)

        params = _TestParams(query="test")
        raw_payload = _TestPayload(items=["x", "y"])
        result = ToolResult(message="ok", value=raw_payload)

        with caplog.at_level(logging.INFO):
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

        record = _find_log_record(caplog, "tool_invoked")
        assert record is not None
        context = _get_context(record)
        assert "result_payload" in context
        assert context["result_payload"]["items"] == ["x", "y"]

    def test_handles_nested_mappings(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify nested dictionaries are properly coerced."""
        bus = InProcessEventBus()
        subscribers = StandardLoggingSubscribers()
        subscribers.attach(bus)

        params = _TestParams(query="test")
        result = ToolResult(message="ok", value={"nested": {"key": "value"}})

        with caplog.at_level(logging.INFO):
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

        record = _find_log_record(caplog, "tool_invoked")
        assert record is not None
        context = _get_context(record)
        assert context["result_payload"]["nested"]["key"] == "value"

    def test_handles_sets_as_sorted_lists(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify sets are coerced to sorted lists."""
        bus = InProcessEventBus()
        subscribers = StandardLoggingSubscribers()
        subscribers.attach(bus)

        params = _TestParams(query="test")
        result = ToolResult(message="ok", value={"tags": {"c", "a", "b"}})

        with caplog.at_level(logging.INFO):
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

        record = _find_log_record(caplog, "tool_invoked")
        assert record is not None
        context = _get_context(record)
        # Sets should be sorted lists
        assert context["result_payload"]["tags"] == ["a", "b", "c"]

    def test_handles_non_serializable_fallback(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify non-serializable objects fall back to str representation."""
        bus = InProcessEventBus()
        subscribers = StandardLoggingSubscribers()
        subscribers.attach(bus)

        class NonSerializable:
            def __repr__(self) -> str:
                return "<NonSerializable>"

        params = _TestParams(query="test")
        result = ToolResult(message="ok", value=NonSerializable())

        with caplog.at_level(logging.INFO):
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

        record = _find_log_record(caplog, "tool_invoked")
        assert record is not None
        context = _get_context(record)
        assert "NonSerializable" in str(context["result_payload"])
