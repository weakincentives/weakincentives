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

"""Tests for session event dispatch and reducers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast
from uuid import UUID

import pytest

from tests.helpers.adapters import GENERIC_ADAPTER_NAME
from tests.helpers.session import (
    DEFAULT_SESSION_ID,
    ExampleOutput,
    ExampleParams,
    ExamplePayload,
    make_prompt_event,
    make_prompt_rendered,
    make_tool_event,
)
from weakincentives.adapters.core import PromptResponse
from weakincentives.prompt.tool import ToolResult
from weakincentives.runtime.events import (
    PromptExecuted,
    PromptRendered,
    ToolInvoked,
)
from weakincentives.runtime.session import (
    Append,
    ReducerContextProtocol,
    ReducerEvent,
    Replace,
    SliceView,
    append_all,
    replace_latest,
    upsert_by,
)

if TYPE_CHECKING:
    from tests.conftest import SessionFactory

pytestmark = pytest.mark.core


def test_tool_invoked_appends_payload_every_time(
    session_factory: SessionFactory,
) -> None:
    session, bus = session_factory()

    event = make_tool_event(1)
    first_result = bus.dispatch(event)
    second_result = bus.dispatch(event)

    assert first_result.ok
    assert second_result.ok
    # append_all uses ledger semantics - every publish appends
    assert session[ExamplePayload].all() == (
        ExamplePayload(value=1),
        ExamplePayload(value=1),
    )
    assert isinstance(event.event_id, UUID)


def test_tool_invoked_extracts_value_from_result(
    session_factory: SessionFactory,
) -> None:
    """Session extracts value from result.value for slice dispatch."""
    session, bus = session_factory()

    payload = ExamplePayload(value=7)
    tool_result = cast(ToolResult[object], ToolResult(message="ok", value=payload))
    event = ToolInvoked(
        prompt_name="example",
        adapter=GENERIC_ADAPTER_NAME,
        name="tool",
        params=ExampleParams(value=7),
        result=tool_result,
        session_id=DEFAULT_SESSION_ID,
        created_at=datetime.now(UTC),
        rendered_output=tool_result.render(),
    )

    bus.dispatch(event)

    # Session dispatches result.value to slice reducers
    assert session[ExamplePayload].all() == (payload,)


def test_prompt_rendered_appends_start_event(
    session_factory: SessionFactory,
) -> None:
    from uuid import uuid4

    session, bus = session_factory()

    event = make_prompt_rendered(
        rendered_prompt="Rendered prompt text",
        session_id=uuid4(),
    )

    result = bus.dispatch(event)

    assert result.ok
    lifecycle = session[PromptRendered].all()
    assert lifecycle == (event,)
    assert lifecycle[0].rendered_prompt == "Rendered prompt text"
    assert isinstance(event.event_id, UUID)


def test_prompt_executed_emits_multiple_dataclasses(
    session_factory: SessionFactory,
) -> None:
    outputs = [ExampleOutput(text="first"), ExampleOutput(text="second")]

    session, bus = session_factory()

    result = bus.dispatch(make_prompt_event(outputs))

    assert result.ok
    assert session[ExampleOutput].all() == tuple(outputs)


def test_reducers_run_in_registration_order(session_factory: SessionFactory) -> None:
    @dataclass(slots=True, frozen=True)
    class FirstSlice:
        value: str

    @dataclass(slots=True, frozen=True)
    class SecondSlice:
        value: str

    session, bus = session_factory()

    call_order: list[str] = []

    def first(
        view: SliceView[FirstSlice],
        event: ReducerEvent,
        *,
        context: ReducerContextProtocol,
    ) -> Append[FirstSlice]:
        del context, view
        call_order.append("first")
        value = cast(ExampleOutput, event)
        return Append(FirstSlice(value.text))

    def second(
        view: SliceView[SecondSlice],
        event: ReducerEvent,
        *,
        context: ReducerContextProtocol,
    ) -> Append[SecondSlice]:
        del context, view
        call_order.append("second")
        value = cast(ExampleOutput, event)
        return Append(SecondSlice(value.text))

    session[FirstSlice].register(ExampleOutput, first)
    session[SecondSlice].register(ExampleOutput, second)

    result = bus.dispatch(make_prompt_event(ExampleOutput(text="hello")))

    assert call_order == ["first", "second"]
    assert session[FirstSlice].all() == (FirstSlice("hello"),)
    assert session[SecondSlice].all() == (SecondSlice("hello"),)
    assert result.ok


def test_default_append_used_when_no_custom_reducer(
    session_factory: SessionFactory,
) -> None:
    session, bus = session_factory()

    result = bus.dispatch(make_prompt_event(ExampleOutput(text="hello")))

    assert result.ok
    assert session[ExampleOutput].all() == (ExampleOutput(text="hello"),)


def test_prompt_executed_extracts_value_from_result(
    session_factory: SessionFactory,
) -> None:
    """Session extracts output from result.output for slice dispatch."""
    session, bus = session_factory()

    output = ExampleOutput(text="filled")
    event = PromptExecuted(
        prompt_name="example",
        adapter=GENERIC_ADAPTER_NAME,
        result=cast(
            PromptResponse[object],
            PromptResponse(
                prompt_name="example",
                text="done",
                output=output,
            ),
        ),
        session_id=DEFAULT_SESSION_ID,
        created_at=datetime.now(UTC),
    )

    bus.dispatch(event)

    # Session dispatches result.output to slice reducers
    assert session[ExampleOutput].all() == (output,)


def test_non_dataclass_payloads_are_ignored(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    event = make_tool_event(1)
    non_dataclass_result = cast(
        ToolResult[object], ToolResult(message="ok", value="not a dataclass")
    )
    non_dataclass_event = ToolInvoked(
        prompt_name="example",
        adapter=GENERIC_ADAPTER_NAME,
        name="tool",
        params=ExampleParams(value=2),
        result=non_dataclass_result,
        session_id=DEFAULT_SESSION_ID,
        created_at=datetime.now(UTC),
        rendered_output=non_dataclass_result.render(),
    )

    first_result = bus.dispatch(event)
    second_result = bus.dispatch(non_dataclass_event)

    assert first_result.ok
    assert second_result.ok
    assert session[ExamplePayload].all() == (ExamplePayload(value=1),)


def test_upsert_by_replaces_matching_keys(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    session[ExamplePayload].register(
        ExamplePayload, upsert_by(lambda payload: payload.value)
    )

    first_result = bus.dispatch(make_tool_event(1))
    second_result = bus.dispatch(make_tool_event(1))
    third_result = bus.dispatch(make_tool_event(2))

    assert first_result.ok
    assert second_result.ok
    assert third_result.ok
    assert session[ExamplePayload].all() == (
        ExamplePayload(value=1),
        ExamplePayload(value=2),
    )


def test_replace_latest_keeps_only_newest_value(
    session_factory: SessionFactory,
) -> None:
    session, bus = session_factory()

    session[ExamplePayload].register(ExamplePayload, replace_latest)

    first_result = bus.dispatch(make_tool_event(1))
    second_result = bus.dispatch(make_tool_event(2))

    assert first_result.ok
    assert second_result.ok
    assert session[ExamplePayload].all() == (ExamplePayload(value=2),)


def test_tool_data_slice_records_failures(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    bus.dispatch(make_tool_event(1))

    failure = cast(
        ToolResult[object],
        ToolResult(message="failed", value=None, success=False),
    )
    failure_event = ToolInvoked(
        prompt_name="example",
        adapter=GENERIC_ADAPTER_NAME,
        name="tool",
        params=ExampleParams(value=2),
        result=failure,
        session_id=DEFAULT_SESSION_ID,
        created_at=datetime.now(UTC),
        rendered_output=failure.render(),
    )
    bus.dispatch(failure_event)

    tool_events = session[ToolInvoked].all()
    assert len(tool_events) == 2
    # First event has payload via result.value
    assert tool_events[0].result.value == ExamplePayload(value=1)
    # Failure event has no value
    assert tool_events[1].result.value is None
    assert tool_events[1].result.success is False


def test_reducer_failure_leaves_previous_slice_unchanged(
    session_factory: SessionFactory,
) -> None:
    session, bus = session_factory()

    session[ExampleOutput].register(ExampleOutput, append_all)

    def faulty(
        view: SliceView[ExampleOutput],
        event: ReducerEvent,
        *,
        context: ReducerContextProtocol,
    ) -> Append[ExampleOutput]:
        del context, view
        raise RuntimeError("boom")

    session[ExampleOutput].register(ExampleOutput, faulty)

    first_result = bus.dispatch(make_prompt_event(ExampleOutput(text="first")))

    assert session[ExampleOutput].all() == (ExampleOutput(text="first"),)
    assert first_result.handled_count == 1

    # Second publish adds another entry (append_all doesn't dedupe)
    # The faulty reducer fails but append_all already ran successfully
    second_result = bus.dispatch(make_prompt_event(ExampleOutput(text="second")))

    assert second_result.handled_count == 1
    assert session[ExampleOutput].all() == (
        ExampleOutput(text="first"),
        ExampleOutput(text="second"),
    )


def test_dispatch_initialize_slice(session_factory: SessionFactory) -> None:
    """Test that dispatching InitializeSlice directly works."""
    from weakincentives.runtime.session import InitializeSlice

    session, _ = session_factory()

    # Dispatch InitializeSlice directly
    session.dispatch(
        InitializeSlice(
            slice_type=ExampleOutput,
            values=(ExampleOutput(text="first"), ExampleOutput(text="second")),
        )
    )

    assert session[ExampleOutput].all() == (
        ExampleOutput(text="first"),
        ExampleOutput(text="second"),
    )


def test_dispatch_clear_slice(session_factory: SessionFactory) -> None:
    """Test that dispatching ClearSlice directly works."""
    from weakincentives.runtime.session import ClearSlice

    session, bus = session_factory()

    # Add some initial data
    bus.dispatch(make_prompt_event(ExampleOutput(text="first")))
    bus.dispatch(make_prompt_event(ExampleOutput(text="second")))
    assert len(session[ExampleOutput].all()) == 2

    # Dispatch ClearSlice directly
    session.dispatch(ClearSlice(slice_type=ExampleOutput))

    assert session[ExampleOutput].all() == ()


def test_dispatch_clear_slice_with_predicate(session_factory: SessionFactory) -> None:
    """Test that dispatching ClearSlice with a predicate works."""
    from weakincentives.runtime.session import ClearSlice

    session, bus = session_factory()

    # Add some initial data
    bus.dispatch(make_prompt_event(ExampleOutput(text="apple")))
    bus.dispatch(make_prompt_event(ExampleOutput(text="banana")))
    bus.dispatch(make_prompt_event(ExampleOutput(text="apricot")))

    # Dispatch ClearSlice with predicate
    session.dispatch(
        ClearSlice(
            slice_type=ExampleOutput,
            predicate=lambda x: x.text.startswith("a"),
        )
    )

    assert session[ExampleOutput].all() == (ExampleOutput(text="banana"),)


def test_clear_slice_with_no_matching_predicate(
    session_factory: SessionFactory,
) -> None:
    """Test that ClearSlice with a predicate that matches nothing leaves values unchanged."""
    from weakincentives.runtime.session import ClearSlice

    session, bus = session_factory()

    # Add some initial data (none starts with 'z')
    bus.dispatch(make_prompt_event(ExampleOutput(text="apple")))
    bus.dispatch(make_prompt_event(ExampleOutput(text="banana")))

    # Dispatch ClearSlice with predicate that matches nothing
    session.dispatch(
        ClearSlice(
            slice_type=ExampleOutput,
            predicate=lambda x: x.text.startswith("z"),
        )
    )

    # Values should be unchanged
    assert session[ExampleOutput].all() == (
        ExampleOutput(text="apple"),
        ExampleOutput(text="banana"),
    )


def test_broadcast_triggers_registered_reducer(
    session_factory: SessionFactory,
) -> None:
    session, _ = session_factory()

    @dataclass(slots=True, frozen=True)
    class SetText:
        text: str

    def set_text_reducer(
        view: SliceView[ExampleOutput],
        event: ReducerEvent,
        *,
        context: ReducerContextProtocol,
    ) -> Replace[ExampleOutput]:
        del context, view
        value = cast(SetText, event)
        return Replace((ExampleOutput(text=value.text),))

    session[ExampleOutput].register(SetText, set_text_reducer)
    session.dispatch(SetText(text="dispatched"))

    assert session[ExampleOutput].all() == (ExampleOutput(text="dispatched"),)


def test_broadcast_dispatches_to_all_reducers(session_factory: SessionFactory) -> None:
    """session.dispatch() dispatches to ALL reducers registered for that event type."""

    @dataclass(slots=True, frozen=True)
    class AddItem:
        value: str

    @dataclass(slots=True, frozen=True)
    class SliceA:
        value: str

    @dataclass(slots=True, frozen=True)
    class SliceB:
        value: str

    session, _ = session_factory()

    call_order: list[str] = []

    def reducer_a(
        view: SliceView[SliceA],
        event: ReducerEvent,
        *,
        context: ReducerContextProtocol,
    ) -> Append[SliceA]:
        del context, view
        call_order.append("A")
        value = cast(AddItem, event)
        return Append(SliceA(value.value))

    def reducer_b(
        view: SliceView[SliceB],
        event: ReducerEvent,
        *,
        context: ReducerContextProtocol,
    ) -> Append[SliceB]:
        del context, view
        call_order.append("B")
        value = cast(AddItem, event)
        return Append(SliceB(value.value))

    # Register reducers for the same event type targeting different slices
    session[SliceA].register(AddItem, reducer_a)
    session[SliceB].register(AddItem, reducer_b)

    # Broadcast dispatch: runs ALL reducers for AddItem
    session.dispatch(AddItem(value="broadcast"))

    assert call_order == ["A", "B"]
    assert session[SliceA].all() == (SliceA("broadcast"),)
    assert session[SliceB].all() == (SliceB("broadcast"),)


def test_prompt_executed_with_mixed_iterable_output(
    session_factory: SessionFactory,
) -> None:
    """Test branch 759->758: non-dataclass items in iterable are skipped."""
    session, bus = session_factory()

    session[ExampleOutput].register(ExampleOutput, append_all)

    # Create a PromptExecuted event with iterable output containing mixed types
    mixed_output = [
        ExampleOutput(text="valid1"),  # Should be dispatched
        "not-a-dataclass",  # Should be skipped (branch 759->758)
        ExampleOutput(text="valid2"),  # Should be dispatched
        42,  # Should be skipped (branch 759->758)
    ]

    event = PromptExecuted(
        adapter=GENERIC_ADAPTER_NAME,
        prompt_name="test",
        result=PromptResponse(prompt_name="test", output=mixed_output, text=None),
        session_id=DEFAULT_SESSION_ID,
        created_at=datetime.now(UTC),
    )

    result = bus.dispatch(event)
    assert result.ok

    # Only dataclass instances should be in the session state
    assert session[ExampleOutput].all() == (
        ExampleOutput(text="valid1"),
        ExampleOutput(text="valid2"),
    )


def test_mutate_reset_clears_all_slices(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    session[ExampleOutput].register(ExampleOutput, append_all)
    bus.dispatch(make_prompt_event(ExampleOutput(text="first")))
    assert session[ExampleOutput].all()

    session.reset()

    assert session[ExampleOutput].all() == ()
