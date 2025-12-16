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

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast
from uuid import UUID, uuid4

import pytest

from tests.helpers.adapters import GENERIC_ADAPTER_NAME
from weakincentives.adapters.core import PromptResponse
from weakincentives.dbc import dbc_enabled
from weakincentives.prompt._types import SupportsDataclass
from weakincentives.prompt.tool import ToolResult
from weakincentives.runtime.events import (
    InProcessEventBus,
    PromptExecuted,
    PromptRendered,
    ToolInvoked,
)
from weakincentives.runtime.session import (
    QueryBuilder,
    ReducerContextProtocol,
    ReducerEvent,
    Session,
    SliceAccessor,
    SliceObserver,
    Snapshot,
    SnapshotRestoreError,
    SnapshotSerializationError,
    Subscription,
    append_all,
    replace_latest,
    upsert_by,
)

if TYPE_CHECKING:
    from tests.conftest import SessionFactory


DEFAULT_SESSION_ID = uuid4()


@dataclass(slots=True, frozen=True)
class ExampleParams:
    value: int


@dataclass(slots=True, frozen=True)
class ExamplePayload:
    value: int


@dataclass(slots=True, frozen=True)
class ExampleOutput:
    text: str


def make_tool_event(value: int) -> ToolInvoked:
    payload = ExamplePayload(value=value)
    tool_result = cast(
        ToolResult[object],
        ToolResult(message="ok", value=payload),
    )
    rendered_output = tool_result.render()
    return ToolInvoked(
        prompt_name="example",
        adapter=GENERIC_ADAPTER_NAME,
        name="tool",
        params=ExampleParams(value=value),
        result=tool_result,
        session_id=DEFAULT_SESSION_ID,
        created_at=datetime.now(UTC),
        rendered_output=rendered_output,
    )


def make_prompt_event(output: object) -> PromptExecuted:
    response = PromptResponse(
        prompt_name="example",
        text="done",
        output=output,
    )
    return PromptExecuted(
        prompt_name="example",
        adapter=GENERIC_ADAPTER_NAME,
        result=response,
        session_id=DEFAULT_SESSION_ID,
        created_at=datetime.now(UTC),
    )


def make_prompt_rendered(
    *,
    rendered_prompt: str,
    session_id: UUID | None = None,
    params_value: int = 1,
) -> PromptRendered:
    return PromptRendered(
        prompt_ns="example",
        prompt_key="example",
        prompt_name="Example",
        adapter=GENERIC_ADAPTER_NAME,
        session_id=session_id,
        render_inputs=(ExampleParams(value=params_value),),
        rendered_prompt=rendered_prompt,
        created_at=datetime.now(UTC),
    )


def test_tool_invoked_appends_payload_every_time(
    session_factory: SessionFactory,
) -> None:
    session, bus = session_factory()

    event = make_tool_event(1)
    first_result = bus.publish(event)
    second_result = bus.publish(event)

    assert first_result.ok
    assert second_result.ok
    # append_all uses ledger semantics - every publish appends
    assert session.query(ExamplePayload).all() == (
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

    bus.publish(event)

    # Session dispatches result.value to slice reducers
    assert session.query(ExamplePayload).all() == (payload,)


def test_prompt_rendered_appends_start_event(
    session_factory: SessionFactory,
) -> None:
    session, bus = session_factory()

    event = make_prompt_rendered(
        rendered_prompt="Rendered prompt text",
        session_id=uuid4(),
    )

    result = bus.publish(event)

    assert result.ok
    lifecycle = session.query(PromptRendered).all()
    assert lifecycle == (event,)
    assert lifecycle[0].rendered_prompt == "Rendered prompt text"
    assert isinstance(event.event_id, UUID)


def test_prompt_executed_emits_multiple_dataclasses(
    session_factory: SessionFactory,
) -> None:
    outputs = [ExampleOutput(text="first"), ExampleOutput(text="second")]

    session, bus = session_factory()

    result = bus.publish(make_prompt_event(outputs))

    assert result.ok
    assert session.query(ExampleOutput).all() == tuple(outputs)


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
        slice_values: tuple[FirstSlice, ...],
        event: ReducerEvent,
        *,
        context: ReducerContextProtocol,
    ) -> tuple[FirstSlice, ...]:
        del context
        call_order.append("first")
        value = cast(ExampleOutput, event)
        return (*slice_values, FirstSlice(value.text))

    def second(
        slice_values: tuple[SecondSlice, ...],
        event: ReducerEvent,
        *,
        context: ReducerContextProtocol,
    ) -> tuple[SecondSlice, ...]:
        del context
        call_order.append("second")
        value = cast(ExampleOutput, event)
        return (*slice_values, SecondSlice(value.text))

    session.mutate(FirstSlice).register(ExampleOutput, first)
    session.mutate(SecondSlice).register(ExampleOutput, second)

    result = bus.publish(make_prompt_event(ExampleOutput(text="hello")))

    assert call_order == ["first", "second"]
    assert session.query(FirstSlice).all() == (FirstSlice("hello"),)
    assert session.query(SecondSlice).all() == (SecondSlice("hello"),)
    assert result.ok


def test_default_append_used_when_no_custom_reducer(
    session_factory: SessionFactory,
) -> None:
    session, bus = session_factory()

    result = bus.publish(make_prompt_event(ExampleOutput(text="hello")))

    assert result.ok
    assert session.query(ExampleOutput).all() == (ExampleOutput(text="hello"),)


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

    bus.publish(event)

    # Session dispatches result.output to slice reducers
    assert session.query(ExampleOutput).all() == (output,)


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

    first_result = bus.publish(event)
    second_result = bus.publish(non_dataclass_event)

    assert first_result.ok
    assert second_result.ok
    assert session.query(ExamplePayload).all() == (ExamplePayload(value=1),)


def test_upsert_by_replaces_matching_keys(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    session.mutate(ExamplePayload).register(
        ExamplePayload, upsert_by(lambda payload: payload.value)
    )

    first_result = bus.publish(make_tool_event(1))
    second_result = bus.publish(make_tool_event(1))
    third_result = bus.publish(make_tool_event(2))

    assert first_result.ok
    assert second_result.ok
    assert third_result.ok
    assert session.query(ExamplePayload).all() == (
        ExamplePayload(value=1),
        ExamplePayload(value=2),
    )


def test_replace_latest_keeps_only_newest_value(
    session_factory: SessionFactory,
) -> None:
    session, bus = session_factory()

    session.mutate(ExamplePayload).register(ExamplePayload, replace_latest)

    first_result = bus.publish(make_tool_event(1))
    second_result = bus.publish(make_tool_event(2))

    assert first_result.ok
    assert second_result.ok
    assert session.query(ExamplePayload).all() == (ExamplePayload(value=2),)


def test_tool_data_slice_records_failures(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    bus.publish(make_tool_event(1))

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
    bus.publish(failure_event)

    tool_events = session.query(ToolInvoked).all()
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

    session.mutate(ExampleOutput).register(ExampleOutput, append_all)

    def faulty(
        slice_values: tuple[ExampleOutput, ...],
        event: ReducerEvent,
        *,
        context: ReducerContextProtocol,
    ) -> tuple[ExampleOutput, ...]:
        del context
        raise RuntimeError("boom")

    session.mutate(ExampleOutput).register(ExampleOutput, faulty)

    first_result = bus.publish(make_prompt_event(ExampleOutput(text="first")))

    assert session.query(ExampleOutput).all() == (ExampleOutput(text="first"),)
    assert first_result.handled_count == 1

    # Second publish adds another entry (append_all doesn't dedupe)
    # The faulty reducer fails but append_all already ran successfully
    second_result = bus.publish(make_prompt_event(ExampleOutput(text="second")))

    assert second_result.handled_count == 1
    assert session.query(ExampleOutput).all() == (
        ExampleOutput(text="first"),
        ExampleOutput(text="second"),
    )


def test_snapshot_round_trip_restores_state(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    first_result = bus.publish(make_prompt_event(ExampleOutput(text="first")))
    second_result = bus.publish(make_prompt_event(ExampleOutput(text="second")))

    assert first_result.ok
    assert second_result.ok
    original_state = session.query(ExampleOutput).all()

    snapshot = session.snapshot()
    raw = snapshot.to_json()
    restored = Snapshot.from_json(raw)

    third_result = bus.publish(make_prompt_event(ExampleOutput(text="third")))
    assert session.query(ExampleOutput).all() != original_state
    assert third_result.ok

    session.mutate().rollback(restored)

    assert session.query(ExampleOutput).all() == original_state


def test_snapshot_preserves_custom_reducer_behavior(
    session_factory: SessionFactory,
) -> None:
    @dataclass(slots=True, frozen=True)
    class Summary:
        entries: tuple[str, ...]

    session, bus = session_factory()

    def aggregate(
        slice_values: tuple[Summary, ...],
        event: ReducerEvent,
        *,
        context: ReducerContextProtocol,
    ) -> tuple[Summary, ...]:
        del context
        value = cast(ExampleOutput, event)
        entries = slice_values[-1].entries if slice_values else ()
        return (Summary((*entries, value.text)),)

    session.mutate(Summary).register(ExampleOutput, aggregate)

    first_result = bus.publish(make_prompt_event(ExampleOutput(text="start")))
    snapshot = session.snapshot()

    second_result = bus.publish(make_prompt_event(ExampleOutput(text="after")))
    assert session.query(Summary).all()[0].entries == ("start", "after")

    session.mutate().rollback(snapshot)

    assert session.query(Summary).all()[0].entries == ("start",)

    third_result = bus.publish(make_prompt_event(ExampleOutput(text="again")))

    assert session.query(Summary).all()[0].entries == ("start", "again")
    assert first_result.ok
    assert second_result.ok
    assert third_result.ok


def test_snapshot_includes_event_slices(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    rendered_event = make_prompt_rendered(rendered_prompt="Rendered prompt text")
    executed_event = make_prompt_event(ExampleOutput(text="complete"))
    tool_event = make_tool_event(4)

    bus.publish(rendered_event)
    bus.publish(executed_event)
    bus.publish(tool_event)

    snapshot = session.snapshot()
    serialized = snapshot.to_json()
    restored = Snapshot.from_json(serialized)

    rendered_snapshot = cast(
        tuple[PromptRendered, ...], snapshot.slices[PromptRendered]
    )
    executed_snapshot = cast(
        tuple[PromptExecuted, ...], snapshot.slices[PromptExecuted]
    )
    tools_snapshot = cast(tuple[ToolInvoked, ...], snapshot.slices[ToolInvoked])

    assert rendered_snapshot == (rendered_event,)
    assert executed_snapshot[0].result.output == ExampleOutput(text="complete")
    assert tools_snapshot[0].result.value == ExamplePayload(value=4)

    restored_rendered = cast(
        tuple[PromptRendered, ...], restored.slices[PromptRendered]
    )
    restored_executed_slice = cast(
        tuple[PromptExecuted, ...], restored.slices[PromptExecuted]
    )
    restored_tool_slice = cast(tuple[ToolInvoked, ...], restored.slices[ToolInvoked])

    assert restored_rendered[0].event_id == rendered_event.event_id
    restored_executed = restored_executed_slice[0]
    assert restored_executed.event_id == executed_event.event_id
    assert restored_executed.result["prompt_name"] == "example"
    assert restored_executed.result["output"] == {"text": "complete"}

    restored_tool = restored_tool_slice[0]
    assert restored_tool.event_id == tool_event.event_id
    assert restored_tool.name == "tool"
    assert restored_tool.params == {"value": 4}
    assert restored_tool.result["value"] == {"value": 4}


def test_snapshot_tracks_relationship_ids(session_factory: SessionFactory) -> None:
    parent, bus = session_factory()

    first_child = Session(bus=bus, parent=parent)
    parent_snapshot = parent.snapshot()

    assert parent_snapshot.parent_id is None
    assert parent_snapshot.children_ids == (first_child.session_id,)

    child_snapshot = first_child.snapshot()

    assert child_snapshot.parent_id == parent.session_id
    assert child_snapshot.children_ids == ()

    second_child = Session(bus=bus, parent=parent)

    parent.mutate().rollback(parent_snapshot)

    assert parent.children == (first_child, second_child)


def test_snapshot_rollback_requires_registered_slices(
    session_factory: SessionFactory,
) -> None:
    source, bus = session_factory()
    result = bus.publish(make_prompt_event(ExampleOutput(text="hello")))

    assert result.ok
    snapshot = source.snapshot()

    target = Session(bus=InProcessEventBus())

    with pytest.raises(SnapshotRestoreError):
        target.mutate().rollback(snapshot)

    assert target.query(ExampleOutput).all() == ()


def test_snapshot_rejects_non_dataclass_values(session_factory: SessionFactory) -> None:
    session, _ = session_factory()
    session.mutate(cast(type[SupportsDataclass], str)).seed(
        cast(tuple[SupportsDataclass, ...], ("value",)),
    )

    with pytest.raises(SnapshotSerializationError):
        session.snapshot()


def test_reset_clears_registered_slices(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    session.mutate(ExampleOutput).register(ExampleOutput, append_all)

    first_result = bus.publish(make_prompt_event(ExampleOutput(text="first")))
    assert first_result.ok
    assert session.query(ExampleOutput).all()

    session.mutate().reset()

    assert session.query(ExampleOutput).all() == ()

    second_result = bus.publish(make_prompt_event(ExampleOutput(text="second")))
    assert second_result.ok
    assert session.query(ExampleOutput).all() == (ExampleOutput(text="second"),)


def test_clone_preserves_state_and_reducer_registration(
    session_factory: SessionFactory,
) -> None:
    provided_session_id = uuid4()
    provided_created_at = datetime.now(UTC)
    session, bus = session_factory(
        session_id=provided_session_id, created_at=provided_created_at
    )

    session.mutate(ExampleOutput).register(ExampleOutput, replace_latest)

    result = bus.publish(make_prompt_event(ExampleOutput(text="first")))
    assert result.ok

    clone_bus = InProcessEventBus()
    clone = session.clone(bus=clone_bus)

    assert clone.session_id == provided_session_id
    assert clone.created_at == provided_created_at
    assert clone.query(ExampleOutput).all() == (ExampleOutput(text="first"),)
    assert session.query(ExampleOutput).all() == (ExampleOutput(text="first"),)
    assert clone._reducers.keys() == session._reducers.keys()

    clone_bus.publish(make_prompt_event(ExampleOutput(text="second")))

    assert clone.query(ExampleOutput).all()[-1] == ExampleOutput(text="second")
    assert session.query(ExampleOutput).all() == (ExampleOutput(text="first"),)

    bus.publish(make_prompt_event(ExampleOutput(text="third")))

    assert session.query(ExampleOutput).all()[-1] == ExampleOutput(text="third")
    assert clone.query(ExampleOutput).all()[-1] == ExampleOutput(text="second")


def test_clone_attaches_to_new_bus_when_provided(
    session_factory: SessionFactory,
) -> None:
    session, source_bus = session_factory()

    source_bus.publish(make_prompt_event(ExampleOutput(text="first")))

    target_bus = InProcessEventBus()
    clone_session_id = uuid4()
    clone_created_at = datetime.now(UTC)
    clone = session.clone(
        bus=target_bus, session_id=clone_session_id, created_at=clone_created_at
    )

    assert clone.session_id == clone_session_id
    assert clone.created_at == clone_created_at
    assert clone.query(ExampleOutput).all() == session.query(ExampleOutput).all()

    target_bus.publish(make_prompt_event(ExampleOutput(text="from clone")))

    assert clone.query(ExampleOutput).all()[-1] == ExampleOutput(text="from clone")
    assert session.query(ExampleOutput).all()[-1] == ExampleOutput(text="first")

    source_bus.publish(make_prompt_event(ExampleOutput(text="original")))

    assert session.query(ExampleOutput).all()[-1] == ExampleOutput(text="original")


def test_session_requires_timezone_aware_created_at() -> None:
    bus = InProcessEventBus()
    naive_timestamp = datetime.now()

    with pytest.raises(ValueError):
        Session(bus=bus, created_at=naive_timestamp)


def test_session_instantiates_default_bus_when_none_provided() -> None:
    session = Session()
    assert isinstance(session.event_bus, InProcessEventBus)


def test_query_returns_query_builder(session_factory: SessionFactory) -> None:
    session, _ = session_factory()

    builder = session.query(ExampleOutput)

    assert isinstance(builder, QueryBuilder)


def test_query_all_returns_empty_tuple_when_no_values(
    session_factory: SessionFactory,
) -> None:
    session, _ = session_factory()

    result = session.query(ExampleOutput).all()

    assert result == ()


def test_query_latest_returns_none_when_no_values(
    session_factory: SessionFactory,
) -> None:
    session, _ = session_factory()

    result = session.query(ExampleOutput).latest()

    assert result is None


def test_query_where_returns_empty_tuple_when_no_values(
    session_factory: SessionFactory,
) -> None:
    session, _ = session_factory()

    result = session.query(ExampleOutput).where(lambda x: True)

    assert result == ()


def test_query_all_returns_all_values(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    bus.publish(make_prompt_event(ExampleOutput(text="first")))
    bus.publish(make_prompt_event(ExampleOutput(text="second")))

    result = session.query(ExampleOutput).all()

    assert result == (ExampleOutput(text="first"), ExampleOutput(text="second"))


def test_query_latest_returns_most_recent_value(
    session_factory: SessionFactory,
) -> None:
    session, bus = session_factory()

    bus.publish(make_prompt_event(ExampleOutput(text="first")))
    bus.publish(make_prompt_event(ExampleOutput(text="second")))

    result = session.query(ExampleOutput).latest()

    assert result == ExampleOutput(text="second")


def test_query_where_filters_values(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    bus.publish(make_prompt_event(ExampleOutput(text="apple")))
    bus.publish(make_prompt_event(ExampleOutput(text="banana")))
    bus.publish(make_prompt_event(ExampleOutput(text="apricot")))

    result = session.query(ExampleOutput).where(lambda x: x.text.startswith("a"))

    assert result == (ExampleOutput(text="apple"), ExampleOutput(text="apricot"))


def test_query_respects_dbc_purity(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    bus.publish(make_prompt_event(ExampleOutput(text="first")))
    bus.publish(make_prompt_event(ExampleOutput(text="second")))

    with dbc_enabled():
        assert session.query(ExampleOutput).all() == (
            ExampleOutput(text="first"),
            ExampleOutput(text="second"),
        )
        assert session.query(ExampleOutput).latest() == ExampleOutput(text="second")
        assert session.query(ExampleOutput).where(lambda x: x.text.startswith("f")) == (
            ExampleOutput(text="first"),
        )


def test_query_where_logs_violate_purity_contract(
    session_factory: SessionFactory,
) -> None:
    session, bus = session_factory()

    bus.publish(make_prompt_event(ExampleOutput(text="first")))

    logger = logging.getLogger(__name__)

    def predicate(value: ExampleOutput) -> bool:
        logger.warning("Saw %s", value)
        return True

    with dbc_enabled(), pytest.raises(AssertionError):
        session.query(ExampleOutput).where(predicate)


# ──────────────────────────────────────────────────────────────────────
# Mutation API tests
# ──────────────────────────────────────────────────────────────────────


def test_mutate_returns_mutation_builder(session_factory: SessionFactory) -> None:
    from weakincentives.runtime.session import GlobalMutationBuilder, MutationBuilder

    session, _ = session_factory()

    slice_builder = session.mutate(ExampleOutput)
    global_builder = session.mutate()

    assert isinstance(slice_builder, MutationBuilder)
    assert isinstance(global_builder, GlobalMutationBuilder)


def test_mutate_seed_single_value(session_factory: SessionFactory) -> None:
    session, _ = session_factory()

    session.mutate(ExampleOutput).seed(ExampleOutput(text="seeded"))

    assert session.query(ExampleOutput).all() == (ExampleOutput(text="seeded"),)


def test_mutate_seed_iterable_values(session_factory: SessionFactory) -> None:
    session, _ = session_factory()

    session.mutate(ExampleOutput).seed(
        [
            ExampleOutput(text="first"),
            ExampleOutput(text="second"),
        ]
    )

    assert session.query(ExampleOutput).all() == (
        ExampleOutput(text="first"),
        ExampleOutput(text="second"),
    )


def test_mutate_clear_removes_all_values(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    bus.publish(make_prompt_event(ExampleOutput(text="first")))
    bus.publish(make_prompt_event(ExampleOutput(text="second")))
    assert session.query(ExampleOutput).all()

    session.mutate(ExampleOutput).clear()

    assert session.query(ExampleOutput).all() == ()


def test_mutate_clear_with_predicate(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    bus.publish(make_prompt_event(ExampleOutput(text="apple")))
    bus.publish(make_prompt_event(ExampleOutput(text="banana")))
    bus.publish(make_prompt_event(ExampleOutput(text="apricot")))

    session.mutate(ExampleOutput).clear(lambda x: x.text.startswith("a"))

    assert session.query(ExampleOutput).all() == (ExampleOutput(text="banana"),)


def test_mutate_dispatch_triggers_registered_reducer(
    session_factory: SessionFactory,
) -> None:
    session, _ = session_factory()

    @dataclass(slots=True, frozen=True)
    class SetText:
        text: str

    def set_text_reducer(
        slice_values: tuple[ExampleOutput, ...],
        event: ReducerEvent,
        *,
        context: ReducerContextProtocol,
    ) -> tuple[ExampleOutput, ...]:
        del context, slice_values
        value = cast(SetText, event)
        return (ExampleOutput(text=value.text),)

    session.mutate(ExampleOutput).register(SetText, set_text_reducer)
    session.mutate(ExampleOutput).dispatch(SetText(text="dispatched"))

    assert session.query(ExampleOutput).all() == (ExampleOutput(text="dispatched"),)


def test_mutate_append_uses_default_reducer(session_factory: SessionFactory) -> None:
    session, _ = session_factory()

    session.mutate(ExampleOutput).append(ExampleOutput(text="first"))
    session.mutate(ExampleOutput).append(ExampleOutput(text="second"))

    assert session.query(ExampleOutput).all() == (
        ExampleOutput(text="first"),
        ExampleOutput(text="second"),
    )


def test_mutate_register_adds_reducer(session_factory: SessionFactory) -> None:
    session, _ = session_factory()

    session.mutate(ExampleOutput).register(ExampleOutput, replace_latest)

    session.mutate(ExampleOutput).append(ExampleOutput(text="first"))
    session.mutate(ExampleOutput).append(ExampleOutput(text="second"))

    assert session.query(ExampleOutput).all() == (ExampleOutput(text="second"),)


def test_mutate_reset_clears_all_slices(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    session.mutate(ExampleOutput).register(ExampleOutput, append_all)
    bus.publish(make_prompt_event(ExampleOutput(text="first")))
    assert session.query(ExampleOutput).all()

    session.mutate().reset()

    assert session.query(ExampleOutput).all() == ()


def test_mutate_rollback_restores_snapshot(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    session.mutate(ExampleOutput).register(ExampleOutput, append_all)
    bus.publish(make_prompt_event(ExampleOutput(text="first")))

    snapshot = session.snapshot()

    bus.publish(make_prompt_event(ExampleOutput(text="second")))
    assert session.query(ExampleOutput).latest() == ExampleOutput(text="second")

    session.mutate().rollback(snapshot)

    assert session.query(ExampleOutput).all() == (ExampleOutput(text="first"),)


# ──────────────────────────────────────────────────────────────────────
# Observer API tests
# ──────────────────────────────────────────────────────────────────────


def test_observe_returns_subscription(session_factory: SessionFactory) -> None:
    session, _ = session_factory()

    def observer(
        old: tuple[ExampleOutput, ...], new: tuple[ExampleOutput, ...]
    ) -> None:
        pass

    subscription = session.observe(ExampleOutput, observer)

    assert isinstance(subscription, Subscription)


def test_observer_called_on_state_change(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    calls: list[tuple[tuple[ExampleOutput, ...], tuple[ExampleOutput, ...]]] = []

    def observer(
        old: tuple[ExampleOutput, ...], new: tuple[ExampleOutput, ...]
    ) -> None:
        calls.append((old, new))

    session.observe(ExampleOutput, observer)
    bus.publish(make_prompt_event(ExampleOutput(text="first")))

    assert len(calls) == 1
    assert calls[0] == ((), (ExampleOutput(text="first"),))


def test_observer_receives_correct_old_and_new_values(
    session_factory: SessionFactory,
) -> None:
    session, bus = session_factory()

    calls: list[tuple[tuple[ExampleOutput, ...], tuple[ExampleOutput, ...]]] = []

    def observer(
        old: tuple[ExampleOutput, ...], new: tuple[ExampleOutput, ...]
    ) -> None:
        calls.append((old, new))

    session.observe(ExampleOutput, observer)

    bus.publish(make_prompt_event(ExampleOutput(text="first")))
    bus.publish(make_prompt_event(ExampleOutput(text="second")))

    assert len(calls) == 2
    assert calls[0] == ((), (ExampleOutput(text="first"),))
    assert calls[1] == (
        (ExampleOutput(text="first"),),
        (ExampleOutput(text="first"), ExampleOutput(text="second")),
    )


def test_multiple_observers_all_called(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    first_calls: list[tuple[ExampleOutput, ...]] = []
    second_calls: list[tuple[ExampleOutput, ...]] = []

    def first_observer(
        old: tuple[ExampleOutput, ...], new: tuple[ExampleOutput, ...]
    ) -> None:
        first_calls.append(new)

    def second_observer(
        old: tuple[ExampleOutput, ...], new: tuple[ExampleOutput, ...]
    ) -> None:
        second_calls.append(new)

    session.observe(ExampleOutput, first_observer)
    session.observe(ExampleOutput, second_observer)

    bus.publish(make_prompt_event(ExampleOutput(text="value")))

    assert first_calls == [(ExampleOutput(text="value"),)]
    assert second_calls == [(ExampleOutput(text="value"),)]


def test_subscription_unsubscribe_removes_observer(
    session_factory: SessionFactory,
) -> None:
    session, bus = session_factory()

    calls: list[tuple[ExampleOutput, ...]] = []

    def observer(
        old: tuple[ExampleOutput, ...], new: tuple[ExampleOutput, ...]
    ) -> None:
        calls.append(new)

    subscription = session.observe(ExampleOutput, observer)

    bus.publish(make_prompt_event(ExampleOutput(text="first")))
    assert len(calls) == 1

    result = subscription.unsubscribe()
    assert result is True

    bus.publish(make_prompt_event(ExampleOutput(text="second")))
    assert len(calls) == 1  # Observer not called after unsubscribe


def test_subscription_unsubscribe_returns_false_when_already_unsubscribed(
    session_factory: SessionFactory,
) -> None:
    session, _ = session_factory()

    def observer(
        old: tuple[ExampleOutput, ...], new: tuple[ExampleOutput, ...]
    ) -> None:
        pass

    subscription = session.observe(ExampleOutput, observer)

    first_result = subscription.unsubscribe()
    second_result = subscription.unsubscribe()

    assert first_result is True
    assert second_result is False


def test_observer_exception_does_not_break_other_observers(
    session_factory: SessionFactory,
) -> None:
    session, bus = session_factory()

    calls: list[str] = []

    def failing_observer(
        old: tuple[ExampleOutput, ...], new: tuple[ExampleOutput, ...]
    ) -> None:
        raise RuntimeError("Observer failed")

    def working_observer(
        old: tuple[ExampleOutput, ...], new: tuple[ExampleOutput, ...]
    ) -> None:
        calls.append("called")

    session.observe(ExampleOutput, failing_observer)
    session.observe(ExampleOutput, working_observer)

    bus.publish(make_prompt_event(ExampleOutput(text="value")))

    assert calls == ["called"]
    assert session.query(ExampleOutput).all() == (ExampleOutput(text="value"),)


def test_observer_called_on_every_append(
    session_factory: SessionFactory,
) -> None:
    session, bus = session_factory()

    calls: list[tuple[ExampleOutput, ...]] = []

    def observer(
        old: tuple[ExampleOutput, ...], new: tuple[ExampleOutput, ...]
    ) -> None:
        calls.append(new)

    session.observe(ExampleOutput, observer)

    # First publish adds the item
    bus.publish(make_prompt_event(ExampleOutput(text="value")))
    assert len(calls) == 1

    # Second publish of same value - append_all always appends (ledger semantics)
    bus.publish(make_prompt_event(ExampleOutput(text="value")))
    assert len(calls) == 2  # Observer called since state changed


def test_observer_works_with_custom_reducer(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    calls: list[tuple[tuple[ExampleOutput, ...], tuple[ExampleOutput, ...]]] = []

    def observer(
        old: tuple[ExampleOutput, ...], new: tuple[ExampleOutput, ...]
    ) -> None:
        calls.append((old, new))

    session.mutate(ExampleOutput).register(ExampleOutput, replace_latest)
    session.observe(ExampleOutput, observer)

    bus.publish(make_prompt_event(ExampleOutput(text="first")))
    bus.publish(make_prompt_event(ExampleOutput(text="second")))

    assert len(calls) == 2
    assert calls[0] == ((), (ExampleOutput(text="first"),))
    assert calls[1] == (
        (ExampleOutput(text="first"),),
        (ExampleOutput(text="second"),),
    )


def test_observer_called_for_tool_invoked_slice(
    session_factory: SessionFactory,
) -> None:
    session, bus = session_factory()

    calls: list[tuple[ExamplePayload, ...]] = []

    def observer(
        old: tuple[ExamplePayload, ...], new: tuple[ExamplePayload, ...]
    ) -> None:
        calls.append(new)

    session.observe(ExamplePayload, observer)

    bus.publish(make_tool_event(42))

    assert len(calls) == 1
    assert calls[0] == (ExamplePayload(value=42),)


def test_observer_type_compatibility() -> None:
    """Verify SliceObserver type alias works correctly."""

    def typed_observer(
        old: tuple[ExampleOutput, ...], new: tuple[ExampleOutput, ...]
    ) -> None:
        pass

    observer: SliceObserver[ExampleOutput] = typed_observer
    assert callable(observer)


def test_subscription_repr(session_factory: SessionFactory) -> None:
    session, _ = session_factory()

    def observer(
        old: tuple[ExampleOutput, ...], new: tuple[ExampleOutput, ...]
    ) -> None:
        pass

    subscription = session.observe(ExampleOutput, observer)

    result = repr(subscription)

    assert "Subscription" in result
    assert "subscription_id" in result
    assert str(subscription.subscription_id) in result


# ──────────────────────────────────────────────────────────────────────
# Dispatch API tests (apply / SliceAccessor)
# ──────────────────────────────────────────────────────────────────────


def test_getitem_returns_slice_accessor(session_factory: SessionFactory) -> None:
    session, _ = session_factory()

    accessor = session[ExampleOutput]

    assert isinstance(accessor, SliceAccessor)


def test_slice_accessor_query_methods_work(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    bus.publish(make_prompt_event(ExampleOutput(text="first")))
    bus.publish(make_prompt_event(ExampleOutput(text="second")))

    # Test all(), latest(), and where() work via SliceAccessor
    assert session[ExampleOutput].all() == (
        ExampleOutput(text="first"),
        ExampleOutput(text="second"),
    )
    assert session[ExampleOutput].latest() == ExampleOutput(text="second")
    assert session[ExampleOutput].where(lambda x: x.text.startswith("f")) == (
        ExampleOutput(text="first"),
    )


def test_apply_broadcasts_to_all_reducers(session_factory: SessionFactory) -> None:
    """session.apply() broadcasts to ALL reducers registered for that event type."""

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
        slice_values: tuple[SliceA, ...],
        event: ReducerEvent,
        *,
        context: ReducerContextProtocol,
    ) -> tuple[SliceA, ...]:
        del context
        call_order.append("A")
        value = cast(AddItem, event)
        return (*slice_values, SliceA(value.value))

    def reducer_b(
        slice_values: tuple[SliceB, ...],
        event: ReducerEvent,
        *,
        context: ReducerContextProtocol,
    ) -> tuple[SliceB, ...]:
        del context
        call_order.append("B")
        value = cast(AddItem, event)
        return (*slice_values, SliceB(value.value))

    # Register reducers for the same event type targeting different slices
    session.mutate(SliceA).register(AddItem, reducer_a)
    session.mutate(SliceB).register(AddItem, reducer_b)

    # Broadcast dispatch: runs ALL reducers for AddItem
    session.apply(AddItem(value="broadcast"))

    assert call_order == ["A", "B"]
    assert session.query(SliceA).all() == (SliceA("broadcast"),)
    assert session.query(SliceB).all() == (SliceB("broadcast"),)


def test_slice_accessor_apply_targets_only_specific_slice(
    session_factory: SessionFactory,
) -> None:
    """session[SliceType].apply() only runs reducers targeting that slice."""

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
        slice_values: tuple[SliceA, ...],
        event: ReducerEvent,
        *,
        context: ReducerContextProtocol,
    ) -> tuple[SliceA, ...]:
        del context
        call_order.append("A")
        value = cast(AddItem, event)
        return (*slice_values, SliceA(value.value))

    def reducer_b(
        slice_values: tuple[SliceB, ...],
        event: ReducerEvent,
        *,
        context: ReducerContextProtocol,
    ) -> tuple[SliceB, ...]:
        del context
        call_order.append("B")
        value = cast(AddItem, event)
        return (*slice_values, SliceB(value.value))

    # Register reducers for the same event type targeting different slices
    session.mutate(SliceA).register(AddItem, reducer_a)
    session.mutate(SliceB).register(AddItem, reducer_b)

    # Targeted dispatch: only runs reducer for SliceA
    session[SliceA].apply(AddItem(value="targeted"))

    assert call_order == ["A"]  # Only A was called
    assert session.query(SliceA).all() == (SliceA("targeted"),)
    assert session.query(SliceB).all() == ()  # B was not updated


def test_slice_accessor_apply_uses_default_reducer_when_none_registered(
    session_factory: SessionFactory,
) -> None:
    """Targeted dispatch uses default append reducer when no custom reducer exists."""
    session, _ = session_factory()

    session[ExampleOutput].apply(ExampleOutput(text="default"))

    assert session[ExampleOutput].all() == (ExampleOutput(text="default"),)


def test_apply_vs_slice_accessor_apply_demonstrates_scope_difference(
    session_factory: SessionFactory,
) -> None:
    """Demonstrates the key mental model difference between broadcast and targeted."""

    @dataclass(slots=True, frozen=True)
    class IncrementCounter:
        amount: int

    @dataclass(slots=True, frozen=True)
    class CounterA:
        count: int

    @dataclass(slots=True, frozen=True)
    class CounterB:
        count: int

    session, _ = session_factory()

    def counter_reducer_a(
        slice_values: tuple[CounterA, ...],
        event: ReducerEvent,
        *,
        context: ReducerContextProtocol,
    ) -> tuple[CounterA, ...]:
        del context
        inc = cast(IncrementCounter, event)
        current = slice_values[-1].count if slice_values else 0
        return (CounterA(current + inc.amount),)

    def counter_reducer_b(
        slice_values: tuple[CounterB, ...],
        event: ReducerEvent,
        *,
        context: ReducerContextProtocol,
    ) -> tuple[CounterB, ...]:
        del context
        inc = cast(IncrementCounter, event)
        current = slice_values[-1].count if slice_values else 0
        return (CounterB(current + inc.amount),)

    session.mutate(CounterA).register(IncrementCounter, counter_reducer_a)
    session.mutate(CounterB).register(IncrementCounter, counter_reducer_b)

    # Broadcast: both counters get incremented
    session.apply(IncrementCounter(amount=10))
    assert session[CounterA].latest() == CounterA(10)
    assert session[CounterB].latest() == CounterB(10)

    # Targeted to CounterA: only A gets incremented
    session[CounterA].apply(IncrementCounter(amount=5))
    assert session[CounterA].latest() == CounterA(15)
    assert session[CounterB].latest() == CounterB(10)  # Unchanged

    # Targeted to CounterB: only B gets incremented
    session[CounterB].apply(IncrementCounter(amount=3))
    assert session[CounterA].latest() == CounterA(15)  # Unchanged
    assert session[CounterB].latest() == CounterB(13)


def test_mutate_dispatch_still_broadcasts_by_event_type(
    session_factory: SessionFactory,
) -> None:
    """Verifies that mutate().dispatch() retains broadcast behavior (for compatibility)."""

    @dataclass(slots=True, frozen=True)
    class MyEvent:
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
        slice_values: tuple[SliceA, ...],
        event: ReducerEvent,
        *,
        context: ReducerContextProtocol,
    ) -> tuple[SliceA, ...]:
        del context
        call_order.append("A")
        value = cast(MyEvent, event)
        return (*slice_values, SliceA(value.value))

    def reducer_b(
        slice_values: tuple[SliceB, ...],
        event: ReducerEvent,
        *,
        context: ReducerContextProtocol,
    ) -> tuple[SliceB, ...]:
        del context
        call_order.append("B")
        value = cast(MyEvent, event)
        return (*slice_values, SliceB(value.value))

    session.mutate(SliceA).register(MyEvent, reducer_a)
    session.mutate(SliceB).register(MyEvent, reducer_b)

    # The old API (mutate().dispatch()) still broadcasts to all reducers
    # This is by design - the slice_type in mutate() was always misleading
    session.mutate(SliceA).dispatch(MyEvent(value="via-mutate"))

    assert call_order == ["A", "B"]  # Both reducers were called
    assert session.query(SliceA).all() == (SliceA("via-mutate"),)
    assert session.query(SliceB).all() == (SliceB("via-mutate"),)
