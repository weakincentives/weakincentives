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

"""Tests for the prompt event bus primitives."""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast
from uuid import UUID, uuid4

import pytest

from tests.helpers.adapters import TEST_ADAPTER_NAME
from tests.helpers.events import NullEventBus
from weakincentives.adapters.core import PromptResponse
from weakincentives.prompt.tool import ToolResult
from weakincentives.runtime.events import (
    HandlerFailure,
    InProcessEventBus,
    PromptExecuted,
    PromptRendered,
    PublishResult,
    TokenUsage,
    ToolCall,
    ToolCallStatus,
    ToolInvoked,
    compute_correlation_key,
)


def make_prompt_response(prompt_name: str) -> PromptResponse[object]:
    return cast(
        PromptResponse[object],
        PromptResponse(
            prompt_name=prompt_name,
            text="hello",
            output="hello",
        ),
    )


def test_null_event_bus_is_noop() -> None:
    bus = NullEventBus()

    events: list[PromptExecuted] = []

    def record_event(event: object) -> None:
        assert isinstance(event, PromptExecuted)
        events.append(event)

    bus.subscribe(PromptExecuted, record_event)
    result = bus.publish(
        PromptExecuted(
            prompt_name="demo",
            adapter=TEST_ADAPTER_NAME,
            result=make_prompt_response("demo"),
            session_id=uuid4(),
            created_at=datetime.now(UTC),
        )
    )

    assert events == []
    assert isinstance(result, PublishResult)
    assert result.ok
    assert result.handlers_invoked == ()
    assert result.handled_count == 0
    assert result.errors == ()


def test_publish_without_subscribers_returns_success_result() -> None:
    bus = InProcessEventBus()

    event = PromptExecuted(
        prompt_name="demo",
        adapter=TEST_ADAPTER_NAME,
        result=make_prompt_response("demo"),
        session_id=uuid4(),
        created_at=datetime.now(UTC),
    )

    result = bus.publish(event)

    assert result.ok
    assert result.handlers_invoked == ()
    assert result.errors == ()
    assert result.handled_count == 0
    result.raise_if_errors()
    assert isinstance(event.event_id, UUID)


def test_publish_prompt_rendered_returns_success() -> None:
    bus = InProcessEventBus()

    event = PromptRendered(
        prompt_ns="demo",
        prompt_key="greeting",
        prompt_name="Demo Prompt",
        adapter=TEST_ADAPTER_NAME,
        session_id=uuid4(),
        render_inputs=(_Params(value=1),),
        rendered_prompt="Render result",
        created_at=datetime.now(UTC),
    )

    result = bus.publish(event)

    assert result.ok
    assert result.errors == ()
    assert isinstance(event.event_id, UUID)


def test_in_process_bus_delivers_in_order() -> None:
    bus = InProcessEventBus()
    delivered: list[PromptExecuted] = []

    def first_handler(event: object) -> None:
        assert isinstance(event, PromptExecuted)
        delivered.append(event)

    def second_handler(event: object) -> None:
        assert isinstance(event, PromptExecuted)
        delivered.append(event)

    bus.subscribe(PromptExecuted, first_handler)
    bus.subscribe(PromptExecuted, second_handler)

    event = PromptExecuted(
        prompt_name="demo",
        adapter=TEST_ADAPTER_NAME,
        result=make_prompt_response("demo"),
        session_id=uuid4(),
        created_at=datetime.now(UTC),
    )
    result = bus.publish(event)

    assert delivered == [event, event]
    assert result.handlers_invoked == (first_handler, second_handler)
    assert result.errors == ()
    assert result.ok
    assert result.handled_count == 2
    assert isinstance(event.event_id, UUID)


def test_in_process_bus_isolates_handler_exceptions(
    caplog: pytest.LogCaptureFixture,
) -> None:
    bus = InProcessEventBus()
    received: list[PromptExecuted] = []

    def bad_handler(event: object) -> None:
        assert isinstance(event, PromptExecuted)
        raise RuntimeError("boom")

    def good_handler(event: object) -> None:
        assert isinstance(event, PromptExecuted)
        received.append(event)

    bus.subscribe(PromptExecuted, bad_handler)
    bus.subscribe(PromptExecuted, good_handler)

    event = PromptExecuted(
        prompt_name="demo",
        adapter=TEST_ADAPTER_NAME,
        result=make_prompt_response("demo"),
        session_id=uuid4(),
        created_at=datetime.now(UTC),
    )
    with caplog.at_level(logging.ERROR, logger="weakincentives.runtime.events"):
        result = bus.publish(event)

    assert received == [event]
    assert any(
        getattr(record, "event", None) == "event_delivery_failed"
        and getattr(record, "context", {}).get("event_type") == "PromptExecuted"
        for record in caplog.records
    )
    assert result.handlers_invoked == (bad_handler, good_handler)
    assert len(result.errors) == 1
    failure = result.errors[0]
    assert isinstance(failure, HandlerFailure)
    assert failure.handler is bad_handler
    assert isinstance(failure.error, RuntimeError)
    assert not result.ok
    assert result.handled_count == 2


def test_publish_result_raise_if_errors() -> None:
    bus = InProcessEventBus()

    def first_handler(_: object) -> None:
        raise ValueError("first")

    def second_handler(_: object) -> None:
        raise RuntimeError("second")

    bus.subscribe(PromptExecuted, first_handler)
    bus.subscribe(PromptExecuted, second_handler)

    result = bus.publish(
        PromptExecuted(
            prompt_name="demo",
            adapter=TEST_ADAPTER_NAME,
            result=make_prompt_response("demo"),
            session_id=uuid4(),
            created_at=datetime.now(UTC),
        )
    )

    assert not result.ok
    with pytest.raises(ExceptionGroup) as excinfo:
        result.raise_if_errors()

    group = excinfo.value
    assert isinstance(group, ExceptionGroup)
    assert len(group.exceptions) == 2
    assert "first_handler" in str(group)
    assert any(isinstance(error, ValueError) for error in group.exceptions)
    assert any(isinstance(error, RuntimeError) for error in group.exceptions)


@dataclass(slots=True)
class _Params:
    value: int


@dataclass(slots=True)
class _Payload:
    value: str


def test_tool_invoked_event_fields() -> None:
    raw_result = ToolResult(message="ok", value=_Payload(value="data"))
    result = cast(ToolResult[object], raw_result)
    rendered_output = raw_result.render()
    event = ToolInvoked(
        prompt_name="demo",
        adapter=TEST_ADAPTER_NAME,
        name="tool",
        params=_Params(value=1),
        result=result,
        call_id="abc123",
        session_id=uuid4(),
        created_at=datetime.now(UTC),
        rendered_output=rendered_output,
    )

    assert event.prompt_name == "demo"
    assert event.adapter == TEST_ADAPTER_NAME
    assert event.name == "tool"
    assert isinstance(event.params, _Params)
    assert event.params.value == 1
    assert isinstance(event.session_id, UUID)
    assert isinstance(event.created_at, datetime)
    assert isinstance(event.result.value, _Payload)
    assert event.result is result
    assert event.call_id == "abc123"
    assert isinstance(event.event_id, UUID)
    assert event.rendered_output == rendered_output


def test_token_usage_total_tokens_none_when_unset() -> None:
    usage = TokenUsage()

    assert usage.total_tokens is None


def test_token_usage_total_tokens_sums_counts() -> None:
    usage = TokenUsage(input_tokens=5, output_tokens=7)

    assert usage.total_tokens == 12


def test_unsubscribe_removes_handler_and_returns_true() -> None:
    bus = InProcessEventBus()
    delivered: list[PromptExecuted] = []

    def handler(event: object) -> None:
        assert isinstance(event, PromptExecuted)
        delivered.append(event)

    bus.subscribe(PromptExecuted, handler)
    event = PromptExecuted(
        prompt_name="demo",
        adapter=TEST_ADAPTER_NAME,
        result=make_prompt_response("demo"),
        session_id=uuid4(),
        created_at=datetime.now(UTC),
    )

    bus.publish(event)
    assert len(delivered) == 1

    result = bus.unsubscribe(PromptExecuted, handler)
    assert result is True

    bus.publish(event)
    assert len(delivered) == 1


def test_unsubscribe_returns_false_for_nonexistent_handler() -> None:
    bus = InProcessEventBus()

    def handler_a(_: object) -> None:
        pass

    def handler_b(_: object) -> None:
        pass

    bus.subscribe(PromptExecuted, handler_a)

    result = bus.unsubscribe(PromptExecuted, handler_b)
    assert result is False


def test_unsubscribe_returns_false_for_nonexistent_event_type() -> None:
    bus = InProcessEventBus()

    def handler(_: object) -> None:
        pass

    result = bus.unsubscribe(PromptExecuted, handler)
    assert result is False


def test_unsubscribe_does_not_affect_other_handlers() -> None:
    bus = InProcessEventBus()
    delivered_a: list[PromptExecuted] = []
    delivered_b: list[PromptExecuted] = []

    def handler_a(event: object) -> None:
        assert isinstance(event, PromptExecuted)
        delivered_a.append(event)

    def handler_b(event: object) -> None:
        assert isinstance(event, PromptExecuted)
        delivered_b.append(event)

    bus.subscribe(PromptExecuted, handler_a)
    bus.subscribe(PromptExecuted, handler_b)

    event = PromptExecuted(
        prompt_name="demo",
        adapter=TEST_ADAPTER_NAME,
        result=make_prompt_response("demo"),
        session_id=uuid4(),
        created_at=datetime.now(UTC),
    )

    bus.publish(event)
    assert len(delivered_a) == 1
    assert len(delivered_b) == 1

    bus.unsubscribe(PromptExecuted, handler_a)

    bus.publish(event)
    assert len(delivered_a) == 1
    assert len(delivered_b) == 2


def test_null_event_bus_unsubscribe_returns_false() -> None:
    bus = NullEventBus()

    def handler(_: object) -> None:
        pass

    bus.subscribe(PromptExecuted, handler)
    result = bus.unsubscribe(PromptExecuted, handler)
    assert result is False


# --- Trace Correlation Tests ---


class TestComputeCorrelationKey:
    def test_same_inputs_produce_same_key(self) -> None:
        key1 = compute_correlation_key("test_tool", {"a": 1, "b": "hello"})
        key2 = compute_correlation_key("test_tool", {"a": 1, "b": "hello"})
        assert key1 == key2

    def test_different_tool_names_produce_different_keys(self) -> None:
        key1 = compute_correlation_key("tool_a", {"x": 1})
        key2 = compute_correlation_key("tool_b", {"x": 1})
        assert key1 != key2

    def test_different_args_produce_different_keys(self) -> None:
        key1 = compute_correlation_key("test_tool", {"x": 1})
        key2 = compute_correlation_key("test_tool", {"x": 2})
        assert key1 != key2

    def test_key_is_deterministic_with_different_arg_order(self) -> None:
        """Args dict order should not affect the correlation key."""
        key1 = compute_correlation_key("test_tool", {"a": 1, "b": 2})
        key2 = compute_correlation_key("test_tool", {"b": 2, "a": 1})
        assert key1 == key2

    def test_returns_16_char_hex_string(self) -> None:
        key = compute_correlation_key("tool", {"arg": "value"})
        assert len(key) == 16
        assert all(c in "0123456789abcdef" for c in key)

    def test_empty_args_produces_consistent_key(self) -> None:
        key1 = compute_correlation_key("tool", {})
        key2 = compute_correlation_key("tool", {})
        assert key1 == key2


class TestToolInvokedTiming:
    def test_tool_invoked_with_timing_fields(self) -> None:
        started = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        completed = datetime(2025, 1, 1, 12, 0, 1, tzinfo=UTC)
        raw_result = ToolResult(message="ok", value=_Payload(value="data"))
        result = cast(ToolResult[object], raw_result)

        event = ToolInvoked(
            prompt_name="demo",
            adapter=TEST_ADAPTER_NAME,
            name="tool",
            params=_Params(value=1),
            result=result,
            session_id=uuid4(),
            created_at=completed,
            started_at=started,
            completed_at=completed,
            metadata={"source": "test"},
            correlation_key="abc123",
        )

        assert event.started_at == started
        assert event.completed_at == completed
        assert event.metadata == {"source": "test"}
        assert event.correlation_key == "abc123"

    def test_duration_ms_calculation(self) -> None:
        started = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        completed = datetime(2025, 1, 1, 12, 0, 1, 500000, tzinfo=UTC)  # 1.5s later
        raw_result = ToolResult(message="ok", value=None)
        result = cast(ToolResult[object], raw_result)

        event = ToolInvoked(
            prompt_name="demo",
            adapter=TEST_ADAPTER_NAME,
            name="tool",
            params={},
            result=result,
            session_id=uuid4(),
            created_at=completed,
            started_at=started,
            completed_at=completed,
        )

        assert event.duration_ms == 1500.0

    def test_duration_ms_returns_none_without_timing(self) -> None:
        raw_result = ToolResult(message="ok", value=None)
        result = cast(ToolResult[object], raw_result)

        event = ToolInvoked(
            prompt_name="demo",
            adapter=TEST_ADAPTER_NAME,
            name="tool",
            params={},
            result=result,
            session_id=uuid4(),
            created_at=datetime.now(UTC),
        )

        assert event.duration_ms is None


class TestToolCallSlice:
    def test_create_requested(self) -> None:
        requested_at = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        tool_call = ToolCall.create_requested(
            tool_name="test_tool",
            params={"arg": "value"},
            prompt_name="test_prompt",
            adapter=TEST_ADAPTER_NAME,
            requested_at=requested_at,
            call_id="call-123",
            metadata={"version": "1.0"},
        )

        assert tool_call.tool_name == "test_tool"
        assert tool_call.params == {"arg": "value"}
        assert tool_call.status == ToolCallStatus.REQUESTED
        assert tool_call.prompt_name == "test_prompt"
        assert tool_call.adapter == TEST_ADAPTER_NAME
        assert tool_call.requested_at == requested_at
        assert tool_call.call_id == "call-123"
        assert tool_call.metadata == {"version": "1.0"}
        assert tool_call.started_at is None
        assert tool_call.completed_at is None
        assert tool_call.result is None
        assert tool_call.error is None
        assert len(tool_call.correlation_key) == 16

    def test_with_status_running(self) -> None:
        requested_at = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        started_at = datetime(2025, 1, 1, 12, 0, 0, 100000, tzinfo=UTC)
        tool_call = ToolCall.create_requested(
            tool_name="test_tool",
            params={"arg": "value"},
            prompt_name="test_prompt",
            adapter=TEST_ADAPTER_NAME,
            requested_at=requested_at,
        )

        running = tool_call.with_status(ToolCallStatus.RUNNING, started_at=started_at)

        assert running.status == ToolCallStatus.RUNNING
        assert running.started_at == started_at
        assert running.completed_at is None
        # Original should be unchanged
        assert tool_call.status == ToolCallStatus.REQUESTED
        assert tool_call.started_at is None

    def test_with_status_completed(self) -> None:
        requested_at = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        started_at = datetime(2025, 1, 1, 12, 0, 0, 100000, tzinfo=UTC)
        completed_at = datetime(2025, 1, 1, 12, 0, 1, tzinfo=UTC)
        tool_call = ToolCall.create_requested(
            tool_name="test_tool",
            params={},
            prompt_name="test_prompt",
            adapter=TEST_ADAPTER_NAME,
            requested_at=requested_at,
        )

        completed = tool_call.with_status(
            ToolCallStatus.COMPLETED,
            started_at=started_at,
            completed_at=completed_at,
            result={"output": "success"},
        )

        assert completed.status == ToolCallStatus.COMPLETED
        assert completed.started_at == started_at
        assert completed.completed_at == completed_at
        assert completed.result == {"output": "success"}
        assert completed.error is None

    def test_with_status_failed(self) -> None:
        requested_at = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        started_at = datetime(2025, 1, 1, 12, 0, 0, 100000, tzinfo=UTC)
        completed_at = datetime(2025, 1, 1, 12, 0, 1, tzinfo=UTC)
        tool_call = ToolCall.create_requested(
            tool_name="test_tool",
            params={},
            prompt_name="test_prompt",
            adapter=TEST_ADAPTER_NAME,
            requested_at=requested_at,
        )

        failed = tool_call.with_status(
            ToolCallStatus.FAILED,
            started_at=started_at,
            completed_at=completed_at,
            error="Connection timeout",
        )

        assert failed.status == ToolCallStatus.FAILED
        assert failed.error == "Connection timeout"
        assert failed.result is None

    def test_duration_ms(self) -> None:
        requested_at = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        started_at = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        completed_at = datetime(2025, 1, 1, 12, 0, 2, 500000, tzinfo=UTC)  # 2.5s later
        tool_call = ToolCall.create_requested(
            tool_name="test_tool",
            params={},
            prompt_name="test_prompt",
            adapter=TEST_ADAPTER_NAME,
            requested_at=requested_at,
        )

        completed = tool_call.with_status(
            ToolCallStatus.COMPLETED,
            started_at=started_at,
            completed_at=completed_at,
        )

        assert completed.duration_ms == 2500.0

    def test_duration_ms_none_without_timing(self) -> None:
        requested_at = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        tool_call = ToolCall.create_requested(
            tool_name="test_tool",
            params={},
            prompt_name="test_prompt",
            adapter=TEST_ADAPTER_NAME,
            requested_at=requested_at,
        )

        assert tool_call.duration_ms is None

    def test_correlation_key_matches_compute_function(self) -> None:
        params = {"arg1": "value1", "arg2": 42}
        tool_call = ToolCall.create_requested(
            tool_name="test_tool",
            params=params,
            prompt_name="test_prompt",
            adapter=TEST_ADAPTER_NAME,
            requested_at=datetime.now(UTC),
        )

        expected_key = compute_correlation_key("test_tool", params)
        assert tool_call.correlation_key == expected_key

    def test_with_status_can_update_call_id(self) -> None:
        """Test that call_id can be updated via with_status for MCP correlation."""
        tool_call = ToolCall.create_requested(
            tool_name="mcp_tool",
            params={},
            prompt_name="test_prompt",
            adapter=TEST_ADAPTER_NAME,
            requested_at=datetime.now(UTC),
            call_id=None,  # MCP tools start without call_id
        )

        assert tool_call.call_id is None

        # After correlation, we can add the call_id
        correlated = tool_call.with_status(
            ToolCallStatus.RUNNING,
            call_id="toolu_abc123",
        )

        assert correlated.call_id == "toolu_abc123"


class TestToolCallStatusEnum:
    def test_status_values(self) -> None:
        assert ToolCallStatus.REQUESTED.value == "requested"
        assert ToolCallStatus.RUNNING.value == "running"
        assert ToolCallStatus.COMPLETED.value == "completed"
        assert ToolCallStatus.FAILED.value == "failed"

    def test_all_statuses_defined(self) -> None:
        statuses = list(ToolCallStatus)
        assert len(statuses) == 4
