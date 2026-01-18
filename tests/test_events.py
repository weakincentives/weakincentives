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

"""Tests for the prompt event dispatcher primitives."""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast
from uuid import UUID, uuid4

import pytest

from tests.helpers.adapters import TEST_ADAPTER_NAME
from tests.helpers.events import NullDispatcher
from weakincentives.adapters.core import PromptResponse
from weakincentives.prompt.tool import ToolResult
from weakincentives.runtime.events import (
    DispatchResult,
    HandlerFailure,
    InProcessDispatcher,
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


def test_null_event_dispatcher_is_noop() -> None:
    dispatcher = NullDispatcher()

    events: list[PromptExecuted] = []

    def record_event(event: object) -> None:
        assert isinstance(event, PromptExecuted)
        events.append(event)

    dispatcher.subscribe(PromptExecuted, record_event)
    result = dispatcher.dispatch(
        PromptExecuted(
            prompt_name="demo",
            adapter=TEST_ADAPTER_NAME,
            result=make_prompt_response("demo"),
            session_id=uuid4(),
            created_at=datetime.now(UTC),
        )
    )

    assert events == []
    assert isinstance(result, DispatchResult)
    assert result.ok
    assert result.handlers_invoked == ()
    assert result.handled_count == 0
    assert result.errors == ()


def test_publish_without_subscribers_returns_success_result() -> None:
    dispatcher = InProcessDispatcher()

    event = PromptExecuted(
        prompt_name="demo",
        adapter=TEST_ADAPTER_NAME,
        result=make_prompt_response("demo"),
        session_id=uuid4(),
        created_at=datetime.now(UTC),
    )

    result = dispatcher.dispatch(event)

    assert result.ok
    assert result.handlers_invoked == ()
    assert result.errors == ()
    assert result.handled_count == 0
    result.raise_if_errors()
    assert isinstance(event.event_id, UUID)


def test_publish_prompt_rendered_returns_success() -> None:
    dispatcher = InProcessDispatcher()

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

    result = dispatcher.dispatch(event)

    assert result.ok
    assert result.errors == ()
    assert isinstance(event.event_id, UUID)


def test_in_process_dispatcher_delivers_in_order() -> None:
    dispatcher = InProcessDispatcher()
    delivered: list[PromptExecuted] = []

    def first_handler(event: object) -> None:
        assert isinstance(event, PromptExecuted)
        delivered.append(event)

    def second_handler(event: object) -> None:
        assert isinstance(event, PromptExecuted)
        delivered.append(event)

    dispatcher.subscribe(PromptExecuted, first_handler)
    dispatcher.subscribe(PromptExecuted, second_handler)

    event = PromptExecuted(
        prompt_name="demo",
        adapter=TEST_ADAPTER_NAME,
        result=make_prompt_response("demo"),
        session_id=uuid4(),
        created_at=datetime.now(UTC),
    )
    result = dispatcher.dispatch(event)

    assert delivered == [event, event]
    assert result.handlers_invoked == (first_handler, second_handler)
    assert result.errors == ()
    assert result.ok
    assert result.handled_count == 2
    assert isinstance(event.event_id, UUID)


def test_in_process_dispatcher_isolates_handler_exceptions(
    caplog: pytest.LogCaptureFixture,
) -> None:
    dispatcher = InProcessDispatcher()
    received: list[PromptExecuted] = []

    def bad_handler(event: object) -> None:
        assert isinstance(event, PromptExecuted)
        raise RuntimeError("boom")

    def good_handler(event: object) -> None:
        assert isinstance(event, PromptExecuted)
        received.append(event)

    dispatcher.subscribe(PromptExecuted, bad_handler)
    dispatcher.subscribe(PromptExecuted, good_handler)

    event = PromptExecuted(
        prompt_name="demo",
        adapter=TEST_ADAPTER_NAME,
        result=make_prompt_response("demo"),
        session_id=uuid4(),
        created_at=datetime.now(UTC),
    )
    with caplog.at_level(logging.ERROR, logger="weakincentives.runtime.events"):
        result = dispatcher.dispatch(event)

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
    dispatcher = InProcessDispatcher()

    def first_handler(_: object) -> None:
        raise ValueError("first")

    def second_handler(_: object) -> None:
        raise RuntimeError("second")

    dispatcher.subscribe(PromptExecuted, first_handler)
    dispatcher.subscribe(PromptExecuted, second_handler)

    result = dispatcher.dispatch(
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
    raw_result = ToolResult.ok(_Payload(value="data"), message="ok")
    rendered_output = raw_result.render()
    event = ToolInvoked(
        prompt_name="demo",
        adapter=TEST_ADAPTER_NAME,
        name="tool",
        params=_Params(value=1),
        success=raw_result.success,
        message=raw_result.message,
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
    assert event.success is True
    assert event.message == "ok"
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
    dispatcher = InProcessDispatcher()
    delivered: list[PromptExecuted] = []

    def handler(event: object) -> None:
        assert isinstance(event, PromptExecuted)
        delivered.append(event)

    dispatcher.subscribe(PromptExecuted, handler)
    event = PromptExecuted(
        prompt_name="demo",
        adapter=TEST_ADAPTER_NAME,
        result=make_prompt_response("demo"),
        session_id=uuid4(),
        created_at=datetime.now(UTC),
    )

    dispatcher.dispatch(event)
    assert len(delivered) == 1

    result = dispatcher.unsubscribe(PromptExecuted, handler)
    assert result is True

    dispatcher.dispatch(event)
    assert len(delivered) == 1


def test_unsubscribe_returns_false_for_nonexistent_handler() -> None:
    dispatcher = InProcessDispatcher()

    def handler_a(_: object) -> None:
        pass

    def handler_b(_: object) -> None:
        pass

    dispatcher.subscribe(PromptExecuted, handler_a)

    result = dispatcher.unsubscribe(PromptExecuted, handler_b)
    assert result is False


def test_unsubscribe_returns_false_for_nonexistent_event_type() -> None:
    dispatcher = InProcessDispatcher()

    def handler(_: object) -> None:
        pass

    result = dispatcher.unsubscribe(PromptExecuted, handler)
    assert result is False


def test_unsubscribe_does_not_affect_other_handlers() -> None:
    dispatcher = InProcessDispatcher()
    delivered_a: list[PromptExecuted] = []
    delivered_b: list[PromptExecuted] = []

    def handler_a(event: object) -> None:
        assert isinstance(event, PromptExecuted)
        delivered_a.append(event)

    def handler_b(event: object) -> None:
        assert isinstance(event, PromptExecuted)
        delivered_b.append(event)

    dispatcher.subscribe(PromptExecuted, handler_a)
    dispatcher.subscribe(PromptExecuted, handler_b)

    event = PromptExecuted(
        prompt_name="demo",
        adapter=TEST_ADAPTER_NAME,
        result=make_prompt_response("demo"),
        session_id=uuid4(),
        created_at=datetime.now(UTC),
    )

    dispatcher.dispatch(event)
    assert len(delivered_a) == 1
    assert len(delivered_b) == 1

    dispatcher.unsubscribe(PromptExecuted, handler_a)

    dispatcher.dispatch(event)
    assert len(delivered_a) == 1
    assert len(delivered_b) == 2


def test_null_event_dispatcher_unsubscribe_returns_false() -> None:
    dispatcher = NullDispatcher()

    def handler(_: object) -> None:
        pass

    dispatcher.subscribe(PromptExecuted, handler)
    result = dispatcher.unsubscribe(PromptExecuted, handler)
    assert result is False
