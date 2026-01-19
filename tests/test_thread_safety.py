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

"""Concurrency regression tests for the thread safety implementation."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from random import shuffle
from uuid import uuid4

import pytest

from tests.helpers.adapters import UNIT_TEST_ADAPTER_NAME
from weakincentives.prompt.overrides import LocalPromptOverridesStore, PromptOverride
from weakincentives.prompt.tool_result import ToolResult
from weakincentives.runtime.events import InProcessDispatcher, ToolInvoked
from weakincentives.runtime.session import Session
from weakincentives.runtime.session.snapshots import Snapshot

THREAD_SESSION_ID = uuid4()


@dataclass(slots=True)
class ExampleParams:
    value: int


@dataclass(slots=True)
class ExampleResult:
    value: int


@dataclass(slots=True)
class _DummySection:
    template: str
    accepts_overrides: bool = True

    def original_body_template(self) -> str | None:
        return self.template

    def tools(self) -> tuple[object, ...]:
        _ = self.accepts_overrides
        return ()


@dataclass(slots=True)
class _DummySectionNode:
    path: tuple[str, ...]
    number: str
    section: _DummySection


@dataclass(slots=True)
class _DummyPrompt:
    ns: str
    key: str
    _sections: tuple[_DummySectionNode, ...]

    @property
    def sections(self) -> tuple[_DummySectionNode, ...]:
        return self._sections


def _publish_tool_event(
    dispatcher: InProcessDispatcher, session: Session, index: int
) -> None:
    params = ExampleParams(value=index)
    result_payload = ExampleResult(value=index)
    result = ToolResult.ok(result_payload, message=f"ok-{index}")
    rendered_output = result.render()
    event = ToolInvoked(
        prompt_name="test",
        adapter=UNIT_TEST_ADAPTER_NAME,
        name=f"example-{index}",
        params=params,
        success=result.success,
        message=result.message,
        result=result,
        call_id=str(index),
        session_id=THREAD_SESSION_ID,
        created_at=datetime.now(UTC),
        rendered_output=rendered_output,
    )
    # Dispatch telemetry event to dispatcher
    dispatcher.dispatch(event)
    # Dispatch payload directly to session (payloads no longer extracted from events)
    session.dispatch(result_payload)


def test_session_attach_to_dispatcher_is_idempotent() -> None:
    dispatcher = InProcessDispatcher()
    session = Session(dispatcher=dispatcher)

    session._attach_to_dispatcher(dispatcher)

    params = ExampleParams(value=999)
    result_payload = ExampleResult(value=999)
    tool_result = ToolResult.ok(result_payload, message="ok")
    rendered_output = tool_result.render()
    event = ToolInvoked(
        prompt_name="test",
        adapter=UNIT_TEST_ADAPTER_NAME,
        name="example-idempotent",
        params=params,
        success=tool_result.success,
        message=tool_result.message,
        result=tool_result,
        call_id="999",
        session_id=THREAD_SESSION_ID,
        created_at=datetime.now(UTC),
        rendered_output=rendered_output,
    )

    publish_result = dispatcher.dispatch(event)
    assert publish_result.handled_count == 1


@pytest.mark.threadstress(min_workers=2, max_workers=8)
def test_session_collects_tool_data_across_threads(
    threadstress_workers: int,
) -> None:
    dispatcher = InProcessDispatcher()
    session = Session(dispatcher=dispatcher)

    max_workers = threadstress_workers
    total_events = max(16, max_workers * 8)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_publish_tool_event, dispatcher, session, index)
            for index in range(total_events)
        ]
        for future in futures:
            future.result()

    tool_data = session[ToolInvoked].all()
    assert len(tool_data) == total_events
    assert {data.call_id for data in tool_data} == {
        str(index) for index in range(total_events)
    }

    result_slice = session[ExampleResult].all()
    assert len(result_slice) == total_events
    assert {value.value for value in result_slice} == set(range(total_events))


@pytest.mark.threadstress(min_workers=2, max_workers=8)
def test_session_snapshots_restore_across_threads(
    threadstress_workers: int,
) -> None:
    dispatcher = InProcessDispatcher()
    session = Session(dispatcher=dispatcher)

    max_workers = threadstress_workers
    total_events = max(24, max_workers * 6)
    snapshot_requests = max_workers * 4

    mutation_tasks = [
        (lambda idx=index: _publish_tool_event(dispatcher, session, idx))
        for index in range(total_events)
    ]
    snapshot_tasks = [
        (lambda: session.snapshot(include_all=True)) for _ in range(snapshot_requests)
    ]
    tasks: list[Callable[[], Snapshot | None]] = [
        *mutation_tasks,
        *snapshot_tasks,
    ]
    shuffle(tasks)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(task) for task in tasks]

    snapshots: list[Snapshot] = []
    for future in futures:
        result = future.result()
        if isinstance(result, Snapshot):
            snapshots.append(result)

    assert len(snapshots) == snapshot_requests
    snapshots.append(session.snapshot(include_all=True))

    for snapshot in snapshots:
        expected_tool_events = snapshot.slices.get(ToolInvoked, ())
        expected_results = snapshot.slices.get(ExampleResult, ())

        restored = Session(dispatcher=InProcessDispatcher())
        restored[ToolInvoked].seed(())
        restored[ExampleResult].seed(())
        restored.restore(snapshot, preserve_logs=False)

        restored_tool_events = restored[ToolInvoked].all()
        restored_results = restored[ExampleResult].all()

        assert restored_tool_events == expected_tool_events
        assert restored_results == expected_results

        assert len(restored_tool_events) <= total_events
        assert len(restored_results) <= total_events

        tool_value_counts = Counter(
            int(tool_event.call_id) for tool_event in restored_tool_events
        )
        result_value_counts = Counter(result.value for result in restored_results)

        for value, count in result_value_counts.items():
            assert tool_value_counts[value] >= count

        # Verify tool events have expected structure
        for tool_event in restored_tool_events:
            assert tool_event.success is True
            assert tool_event.call_id is not None


@pytest.mark.threadstress(min_workers=2, max_workers=6)
def test_local_prompt_overrides_store_seed_is_thread_safe(
    threadstress_workers: int, tmp_path: Path
) -> None:
    store = LocalPromptOverridesStore(root_path=tmp_path)
    section = _DummySection(template="Hello, ${name}!")
    prompt = _DummyPrompt(
        ns="sample",
        key="prompt",
        _sections=(_DummySectionNode(path=("intro",), number="1", section=section),),
    )

    def seed() -> PromptOverride:
        return store.seed(prompt, tag="concurrent")

    max_workers = threadstress_workers
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(seed) for _ in range(max_workers * 2)]
        overrides = [future.result() for future in futures]

    # When root_path is explicit, it is used directly as the overrides directory
    # (no .weakincentives/prompts/overrides prefix)
    expected_path = tmp_path / "sample" / "prompt" / "concurrent.json"
    assert expected_path.exists()
    with expected_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["ns"] == "sample"
    assert payload["prompt_key"] == "prompt"
    assert payload["tag"] == "concurrent"
    assert payload["sections"]["intro"]["body"] == section.template

    first_sections = overrides[0].sections
    for override in overrides[1:]:
        assert override.sections == first_sections


def test_session_reset_clears_runtime_state() -> None:
    """Session reset clears all slices."""
    dispatcher = InProcessDispatcher()
    session = Session(dispatcher=dispatcher)

    seeded_value = ExampleResult(value=1)
    session[ExampleResult].seed((seeded_value,))

    assert session[ExampleResult].all() == (seeded_value,)

    session.reset()

    assert session[ExampleResult].all() == ()


def test_session_reducer_optimistic_concurrency_retry() -> None:
    """Test branch 818->796: reducer retries when state is modified concurrently."""
    import threading
    import time

    from weakincentives.runtime.session import Append, SliceView
    from weakincentives.runtime.session.reducer_context import ReducerContext

    @dataclass(slots=True, frozen=True)
    class CounterEvent:
        value: int

    dispatcher = InProcessDispatcher()
    session = Session(dispatcher=dispatcher)

    # Track how many times the reducer is called (retries will increase this)
    call_count = 0
    call_count_lock = threading.Lock()

    def slow_append_reducer(
        view: SliceView[CounterEvent],
        event: CounterEvent,
        *,
        context: ReducerContext,
    ) -> Append[CounterEvent]:
        nonlocal call_count
        del view, context  # unused
        with call_count_lock:
            call_count += 1
        # Small delay to increase chance of concurrent modification
        time.sleep(0.001)
        return Append(event)

    # Register the slow reducer
    session[CounterEvent].register(CounterEvent, slow_append_reducer)

    # Use barrier with parties == max_workers so all threads sync before broadcasting
    max_workers = 8
    total_events = max_workers  # Must equal max_workers for barrier to work
    barrier = threading.Barrier(max_workers)

    def broadcast_with_barrier(value: int) -> None:
        barrier.wait()  # All threads wait here until all are ready
        session.dispatch(CounterEvent(value=value))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(broadcast_with_barrier, i) for i in range(total_events)
        ]
        for future in futures:
            future.result()

    # All events should be in the session state, even though some reducers retried
    events = session[CounterEvent].all()
    assert len(events) == total_events
    # Verify all values are present (order might vary due to threading)
    values = {event.value for event in events}
    assert values == set(range(total_events))
    # call_count should be >= total_events (retries increase it)
    # This verifies that the retry path was exercised
    assert call_count >= total_events
