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
from weakincentives.prompt.tool import ToolResult
from weakincentives.runtime.events import (
    InProcessDispatcher,
    PromptExecuted,
    PromptRendered,
    ToolInvoked,
)
from weakincentives.runtime.session import (
    Append,
    ReducerContextProtocol,
    ReducerEvent,
    Replace,
    Session,
    SliceAccessor,
    SlicePolicy,
    SliceView,
    Snapshot,
    SnapshotRestoreError,
    SnapshotSerializationError,
    append_all,
    replace_latest,
    upsert_by,
)
from weakincentives.types.dataclass import SupportsDataclass

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


def test_snapshot_round_trip_restores_state(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    first_result = bus.dispatch(make_prompt_event(ExampleOutput(text="first")))
    second_result = bus.dispatch(make_prompt_event(ExampleOutput(text="second")))

    assert first_result.ok
    assert second_result.ok
    original_state = session[ExampleOutput].all()

    snapshot = session.snapshot(include_all=True)
    raw = snapshot.to_json()
    restored = Snapshot.from_json(raw)

    third_result = bus.dispatch(make_prompt_event(ExampleOutput(text="third")))
    assert session[ExampleOutput].all() != original_state
    assert third_result.ok

    session.restore(restored)

    assert session[ExampleOutput].all() == original_state


def test_snapshot_preserves_custom_reducer_behavior(
    session_factory: SessionFactory,
) -> None:
    @dataclass(slots=True, frozen=True)
    class Summary:
        entries: tuple[str, ...]

    session, bus = session_factory()

    def aggregate(
        view: SliceView[Summary],
        event: ReducerEvent,
        *,
        context: ReducerContextProtocol,
    ) -> Replace[Summary]:
        del context
        value = cast(ExampleOutput, event)
        latest = view.latest()
        entries = latest.entries if latest else ()
        return Replace((Summary((*entries, value.text)),))

    session[Summary].register(ExampleOutput, aggregate)

    first_result = bus.dispatch(make_prompt_event(ExampleOutput(text="start")))
    snapshot = session.snapshot(include_all=True)

    second_result = bus.dispatch(make_prompt_event(ExampleOutput(text="after")))
    assert session[Summary].all()[0].entries == ("start", "after")

    session.restore(snapshot)

    assert session[Summary].all()[0].entries == ("start",)

    third_result = bus.dispatch(make_prompt_event(ExampleOutput(text="again")))

    assert session[Summary].all()[0].entries == ("start", "again")
    assert first_result.ok
    assert second_result.ok
    assert third_result.ok


def test_snapshot_includes_event_slices(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    rendered_event = make_prompt_rendered(rendered_prompt="Rendered prompt text")
    executed_event = make_prompt_event(ExampleOutput(text="complete"))
    tool_event = make_tool_event(4)

    bus.dispatch(rendered_event)
    bus.dispatch(executed_event)
    bus.dispatch(tool_event)

    snapshot = session.snapshot(
        policies=frozenset({SlicePolicy.STATE, SlicePolicy.LOG})
    )
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


def test_snapshot_filters_log_slices_by_default(
    session_factory: SessionFactory,
) -> None:
    session, bus = session_factory()

    bus.dispatch(make_tool_event(1))
    bus.dispatch(make_prompt_event(ExampleOutput(text="state")))

    snapshot = session.snapshot()

    assert ToolInvoked not in snapshot.slices
    assert snapshot.slices[ExampleOutput] == (ExampleOutput(text="state"),)


def test_snapshot_respects_registered_log_policy(
    session_factory: SessionFactory,
) -> None:
    @dataclass(slots=True, frozen=True)
    class LogEntry:
        message: str

    session, _ = session_factory()

    session[LogEntry].register(LogEntry, append_all, policy=SlicePolicy.LOG)
    session[LogEntry].seed((LogEntry(message="hello"),))

    snapshot = session.snapshot()
    assert LogEntry not in snapshot.slices

    snapshot_with_logs = session.snapshot(
        policies=frozenset({SlicePolicy.STATE, SlicePolicy.LOG})
    )
    assert snapshot_with_logs.slices[LogEntry] == (LogEntry(message="hello"),)


def test_rollback_preserves_log_slices(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    first_event = make_tool_event(1)
    second_event = make_tool_event(2)

    bus.dispatch(first_event)
    snapshot = session.snapshot()

    bus.dispatch(second_event)

    session.restore(snapshot)

    assert session[ToolInvoked].all() == (first_event, second_event)


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

    parent.restore(parent_snapshot)

    assert parent.children == (first_child, second_child)


def test_snapshot_rollback_requires_registered_slices(
    session_factory: SessionFactory,
) -> None:
    source, bus = session_factory()
    result = bus.dispatch(make_prompt_event(ExampleOutput(text="hello")))

    assert result.ok
    snapshot = source.snapshot()

    target = Session(bus=InProcessDispatcher())

    with pytest.raises(SnapshotRestoreError):
        target.restore(snapshot)

    assert target[ExampleOutput].all() == ()


def test_snapshot_rejects_non_dataclass_values(session_factory: SessionFactory) -> None:
    session, _ = session_factory()
    session[cast(type[SupportsDataclass], str)].seed(
        cast(tuple[SupportsDataclass, ...], ("value",)),
    )

    with pytest.raises(SnapshotSerializationError):
        session.snapshot()


def test_reset_clears_registered_slices(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    session[ExampleOutput].register(ExampleOutput, append_all)

    first_result = bus.dispatch(make_prompt_event(ExampleOutput(text="first")))
    assert first_result.ok
    assert session[ExampleOutput].all()

    session.reset()

    assert session[ExampleOutput].all() == ()

    second_result = bus.dispatch(make_prompt_event(ExampleOutput(text="second")))
    assert second_result.ok
    assert session[ExampleOutput].all() == (ExampleOutput(text="second"),)


def test_clone_preserves_state_and_reducer_registration(
    session_factory: SessionFactory,
) -> None:
    provided_session_id = uuid4()
    provided_created_at = datetime.now(UTC)
    session, bus = session_factory(
        session_id=provided_session_id, created_at=provided_created_at
    )

    session[ExampleOutput].register(ExampleOutput, replace_latest)

    result = bus.dispatch(make_prompt_event(ExampleOutput(text="first")))
    assert result.ok

    clone_bus = InProcessDispatcher()
    clone = session.clone(bus=clone_bus)

    assert clone.session_id == provided_session_id
    assert clone.created_at == provided_created_at
    assert clone[ExampleOutput].all() == (ExampleOutput(text="first"),)
    assert session[ExampleOutput].all() == (ExampleOutput(text="first"),)
    assert clone._reducers.keys() == session._reducers.keys()

    clone_bus.dispatch(make_prompt_event(ExampleOutput(text="second")))

    assert clone[ExampleOutput].all()[-1] == ExampleOutput(text="second")
    assert session[ExampleOutput].all() == (ExampleOutput(text="first"),)

    bus.dispatch(make_prompt_event(ExampleOutput(text="third")))

    assert session[ExampleOutput].all()[-1] == ExampleOutput(text="third")
    assert clone[ExampleOutput].all()[-1] == ExampleOutput(text="second")


def test_clone_attaches_to_new_bus_when_provided(
    session_factory: SessionFactory,
) -> None:
    session, source_bus = session_factory()

    source_bus.dispatch(make_prompt_event(ExampleOutput(text="first")))

    target_bus = InProcessDispatcher()
    clone_session_id = uuid4()
    clone_created_at = datetime.now(UTC)
    clone = session.clone(
        bus=target_bus, session_id=clone_session_id, created_at=clone_created_at
    )

    assert clone.session_id == clone_session_id
    assert clone.created_at == clone_created_at
    assert clone[ExampleOutput].all() == session[ExampleOutput].all()

    target_bus.dispatch(make_prompt_event(ExampleOutput(text="from clone")))

    assert clone[ExampleOutput].all()[-1] == ExampleOutput(text="from clone")
    assert session[ExampleOutput].all()[-1] == ExampleOutput(text="first")

    source_bus.dispatch(make_prompt_event(ExampleOutput(text="original")))

    assert session[ExampleOutput].all()[-1] == ExampleOutput(text="original")


def test_session_requires_timezone_aware_created_at() -> None:
    bus = InProcessDispatcher()
    naive_timestamp = datetime.now()

    with pytest.raises(ValueError):
        Session(bus=bus, created_at=naive_timestamp)


def test_session_instantiates_default_bus_when_none_provided() -> None:
    session = Session()
    assert isinstance(session.dispatcher, InProcessDispatcher)


def test_indexing_returns_slice_accessor(session_factory: SessionFactory) -> None:
    session, _ = session_factory()

    accessor = session[ExampleOutput]

    assert isinstance(accessor, SliceAccessor)


def test_query_all_returns_empty_tuple_when_no_values(
    session_factory: SessionFactory,
) -> None:
    session, _ = session_factory()

    result = session[ExampleOutput].all()

    assert result == ()


def test_query_latest_returns_none_when_no_values(
    session_factory: SessionFactory,
) -> None:
    session, _ = session_factory()

    result = session[ExampleOutput].latest()

    assert result is None


def test_query_where_returns_empty_tuple_when_no_values(
    session_factory: SessionFactory,
) -> None:
    session, _ = session_factory()

    result = session[ExampleOutput].where(lambda x: True)

    assert result == ()


def test_query_all_returns_all_values(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    bus.dispatch(make_prompt_event(ExampleOutput(text="first")))
    bus.dispatch(make_prompt_event(ExampleOutput(text="second")))

    result = session[ExampleOutput].all()

    assert result == (ExampleOutput(text="first"), ExampleOutput(text="second"))


def test_query_latest_returns_most_recent_value(
    session_factory: SessionFactory,
) -> None:
    session, bus = session_factory()

    bus.dispatch(make_prompt_event(ExampleOutput(text="first")))
    bus.dispatch(make_prompt_event(ExampleOutput(text="second")))

    result = session[ExampleOutput].latest()

    assert result == ExampleOutput(text="second")


def test_query_where_filters_values(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    bus.dispatch(make_prompt_event(ExampleOutput(text="apple")))
    bus.dispatch(make_prompt_event(ExampleOutput(text="banana")))
    bus.dispatch(make_prompt_event(ExampleOutput(text="apricot")))

    result = session[ExampleOutput].where(lambda x: x.text.startswith("a"))

    assert result == (ExampleOutput(text="apple"), ExampleOutput(text="apricot"))


def test_query_respects_dbc_purity(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    bus.dispatch(make_prompt_event(ExampleOutput(text="first")))
    bus.dispatch(make_prompt_event(ExampleOutput(text="second")))

    with dbc_enabled():
        assert session[ExampleOutput].all() == (
            ExampleOutput(text="first"),
            ExampleOutput(text="second"),
        )
        assert session[ExampleOutput].latest() == ExampleOutput(text="second")
        assert session[ExampleOutput].where(lambda x: x.text.startswith("f")) == (
            ExampleOutput(text="first"),
        )


def test_query_where_logs_violate_purity_contract(
    session_factory: SessionFactory,
) -> None:
    session, bus = session_factory()

    bus.dispatch(make_prompt_event(ExampleOutput(text="first")))

    logger = logging.getLogger(__name__)

    def predicate(value: ExampleOutput) -> bool:
        logger.warning("Saw %s", value)
        return True

    with dbc_enabled(), pytest.raises(AssertionError):
        session[ExampleOutput].where(predicate)


# ──────────────────────────────────────────────────────────────────────
# Mutation API tests
# ──────────────────────────────────────────────────────────────────────


def test_seed_single_value(session_factory: SessionFactory) -> None:
    session, _ = session_factory()

    session[ExampleOutput].seed(ExampleOutput(text="seeded"))

    assert session[ExampleOutput].all() == (ExampleOutput(text="seeded"),)


def test_mutate_seed_iterable_values(session_factory: SessionFactory) -> None:
    session, _ = session_factory()

    session[ExampleOutput].seed(
        [
            ExampleOutput(text="first"),
            ExampleOutput(text="second"),
        ]
    )

    assert session[ExampleOutput].all() == (
        ExampleOutput(text="first"),
        ExampleOutput(text="second"),
    )


def test_mutate_clear_removes_all_values(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    bus.dispatch(make_prompt_event(ExampleOutput(text="first")))
    bus.dispatch(make_prompt_event(ExampleOutput(text="second")))
    assert session[ExampleOutput].all()

    session[ExampleOutput].clear()

    assert session[ExampleOutput].all() == ()


def test_mutate_clear_with_predicate(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    bus.dispatch(make_prompt_event(ExampleOutput(text="apple")))
    bus.dispatch(make_prompt_event(ExampleOutput(text="banana")))
    bus.dispatch(make_prompt_event(ExampleOutput(text="apricot")))

    session[ExampleOutput].clear(lambda x: x.text.startswith("a"))

    assert session[ExampleOutput].all() == (ExampleOutput(text="banana"),)


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
    """Test that ClearSlice with a predicate that matches nothing doesn't notify observers."""
    from weakincentives.runtime.session import ClearSlice

    session, bus = session_factory()

    # Add some initial data (none starts with 'z')
    bus.dispatch(make_prompt_event(ExampleOutput(text="apple")))
    bus.dispatch(make_prompt_event(ExampleOutput(text="banana")))

    observer_calls: list[
        tuple[tuple[ExampleOutput, ...], tuple[ExampleOutput, ...]]
    ] = []

    def observer(
        old: tuple[ExampleOutput, ...], new: tuple[ExampleOutput, ...]
    ) -> None:
        observer_calls.append((old, new))

    session.observe(ExampleOutput, observer)

    # Dispatch ClearSlice with predicate that matches nothing
    session.dispatch(
        ClearSlice(
            slice_type=ExampleOutput,
            predicate=lambda x: x.text.startswith("z"),
        )
    )

    # Observer should NOT be called since state didn't change
    assert len(observer_calls) == 0
    # Values should be unchanged
    assert session[ExampleOutput].all() == (
        ExampleOutput(text="apple"),
        ExampleOutput(text="banana"),
    )


def test_initialize_slice_triggers_observer(session_factory: SessionFactory) -> None:
    """Test that InitializeSlice triggers observers."""
    from weakincentives.runtime.session import InitializeSlice

    session, _ = session_factory()

    observer_calls: list[
        tuple[tuple[ExampleOutput, ...], tuple[ExampleOutput, ...]]
    ] = []

    def observer(
        old: tuple[ExampleOutput, ...], new: tuple[ExampleOutput, ...]
    ) -> None:
        observer_calls.append((old, new))

    session.observe(ExampleOutput, observer)

    session.dispatch(
        InitializeSlice(
            slice_type=ExampleOutput,
            values=(ExampleOutput(text="init"),),
        )
    )

    assert len(observer_calls) == 1
    assert observer_calls[0][0] == ()
    assert observer_calls[0][1] == (ExampleOutput(text="init"),)


def test_clear_slice_triggers_observer(session_factory: SessionFactory) -> None:
    """Test that ClearSlice triggers observers."""
    from weakincentives.runtime.session import ClearSlice

    session, bus = session_factory()

    # Add initial data
    bus.dispatch(make_prompt_event(ExampleOutput(text="first")))

    observer_calls: list[
        tuple[tuple[ExampleOutput, ...], tuple[ExampleOutput, ...]]
    ] = []

    def observer(
        old: tuple[ExampleOutput, ...], new: tuple[ExampleOutput, ...]
    ) -> None:
        observer_calls.append((old, new))

    session.observe(ExampleOutput, observer)

    session.dispatch(ClearSlice(slice_type=ExampleOutput))

    assert len(observer_calls) == 1
    assert observer_calls[0][0] == (ExampleOutput(text="first"),)
    assert observer_calls[0][1] == ()


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


def test_mutate_append_uses_default_reducer(session_factory: SessionFactory) -> None:
    session, _ = session_factory()

    session[ExampleOutput].append(ExampleOutput(text="first"))
    session[ExampleOutput].append(ExampleOutput(text="second"))

    assert session[ExampleOutput].all() == (
        ExampleOutput(text="first"),
        ExampleOutput(text="second"),
    )


def test_mutate_register_adds_reducer(session_factory: SessionFactory) -> None:
    session, _ = session_factory()

    session[ExampleOutput].register(ExampleOutput, replace_latest)

    session[ExampleOutput].append(ExampleOutput(text="first"))
    session[ExampleOutput].append(ExampleOutput(text="second"))

    assert session[ExampleOutput].all() == (ExampleOutput(text="second"),)


def test_mutate_reset_clears_all_slices(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    session[ExampleOutput].register(ExampleOutput, append_all)
    bus.dispatch(make_prompt_event(ExampleOutput(text="first")))
    assert session[ExampleOutput].all()

    session.reset()

    assert session[ExampleOutput].all() == ()


def test_mutate_rollback_restores_snapshot(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    session[ExampleOutput].register(ExampleOutput, append_all)
    bus.dispatch(make_prompt_event(ExampleOutput(text="first")))

    snapshot = session.snapshot()

    bus.dispatch(make_prompt_event(ExampleOutput(text="second")))
    assert session[ExampleOutput].latest() == ExampleOutput(text="second")

    session.restore(snapshot)

    assert session[ExampleOutput].all() == (ExampleOutput(text="first"),)


# ──────────────────────────────────────────────────────────────────────
# Dispatch API tests (apply / SliceAccessor)
# ──────────────────────────────────────────────────────────────────────


def test_getitem_returns_slice_accessor(session_factory: SessionFactory) -> None:
    session, _ = session_factory()

    accessor = session[ExampleOutput]

    assert isinstance(accessor, SliceAccessor)


def test_slice_accessor_query_methods_work(session_factory: SessionFactory) -> None:
    session, bus = session_factory()

    bus.dispatch(make_prompt_event(ExampleOutput(text="first")))
    bus.dispatch(make_prompt_event(ExampleOutput(text="second")))

    # Test all(), latest(), and where() work via SliceAccessor
    assert session[ExampleOutput].all() == (
        ExampleOutput(text="first"),
        ExampleOutput(text="second"),
    )
    assert session[ExampleOutput].latest() == ExampleOutput(text="second")
    assert session[ExampleOutput].where(lambda x: x.text.startswith("f")) == (
        ExampleOutput(text="first"),
    )


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
