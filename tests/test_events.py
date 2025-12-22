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
    InProcessEventBus,
    PromptExecuted,
    PromptRendered,
    TokenUsage,
    ToolInvoked,
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
    bus.publish(
        PromptExecuted(
            prompt_name="demo",
            adapter=TEST_ADAPTER_NAME,
            result=make_prompt_response("demo"),
            session_id=uuid4(),
            created_at=datetime.now(UTC),
        )
    )

    # NullEventBus discards events
    assert events == []


def test_publish_without_subscribers_succeeds() -> None:
    bus = InProcessEventBus()

    event = PromptExecuted(
        prompt_name="demo",
        adapter=TEST_ADAPTER_NAME,
        result=make_prompt_response("demo"),
        session_id=uuid4(),
        created_at=datetime.now(UTC),
    )

    # Should not raise
    bus.publish(event)
    assert isinstance(event.event_id, UUID)


def test_publish_prompt_rendered_succeeds() -> None:
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

    # Should not raise
    bus.publish(event)
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
    bus.publish(event)

    assert delivered == [event, event]
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
        bus.publish(event)

    # Good handler still received the event despite bad handler raising
    assert received == [event]

    # Error was logged
    assert any(
        getattr(record, "event", None) == "event_delivery_failed"
        and getattr(record, "context", {}).get("event_type") == "PromptExecuted"
        for record in caplog.records
    )


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
