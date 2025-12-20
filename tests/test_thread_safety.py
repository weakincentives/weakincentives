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
from typing import cast
from uuid import uuid4

import pytest

from tests.helpers.adapters import UNIT_TEST_ADAPTER_NAME
from weakincentives.prompt.overrides import LocalPromptOverridesStore, PromptOverride
from weakincentives.prompt.tool_result import ToolResult
from weakincentives.runtime.events import InProcessEventBus, ToolInvoked
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


def _publish_tool_event(bus: InProcessEventBus, index: int) -> None:
    params = ExampleParams(value=index)
    result_payload = ExampleResult(value=index)
    result = ToolResult(message=f"ok-{index}", value=result_payload)
    rendered_output = result.render()
    event = ToolInvoked(
        prompt_name="test",
        adapter=UNIT_TEST_ADAPTER_NAME,
        name=f"example-{index}",
        params=params,
        result=cast(ToolResult[object], result),
        call_id=str(index),
        session_id=THREAD_SESSION_ID,
        created_at=datetime.now(UTC),
        rendered_output=rendered_output,
    )
    bus.publish(event)


def test_session_attach_to_bus_is_idempotent() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)

    session._attach_to_bus(bus)

    params = ExampleParams(value=999)
    result_payload = ExampleResult(value=999)
    tool_result = ToolResult(message="ok", value=result_payload)
    rendered_output = tool_result.render()
    event = ToolInvoked(
        prompt_name="test",
        adapter=UNIT_TEST_ADAPTER_NAME,
        name="example-idempotent",
        params=params,
        result=cast(ToolResult[object], tool_result),
        call_id="999",
        session_id=THREAD_SESSION_ID,
        created_at=datetime.now(UTC),
        rendered_output=rendered_output,
    )

    publish_result = bus.publish(event)
    assert publish_result.handled_count == 1


@pytest.mark.threadstress(min_workers=2, max_workers=8)
def test_session_collects_tool_data_across_threads(
    threadstress_workers: int,
) -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)

    max_workers = threadstress_workers
    total_events = max(16, max_workers * 8)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_publish_tool_event, bus, index)
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
    bus = InProcessEventBus()
    session = Session(bus=bus)

    max_workers = threadstress_workers
    total_events = max(24, max_workers * 6)
    snapshot_requests = max_workers * 4

    mutation_tasks = [
        (lambda idx=index: _publish_tool_event(bus, idx))
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

        restored = Session(bus=InProcessEventBus())
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

        for tool_event in restored_tool_events:
            assert isinstance(tool_event.result.value, ExampleResult)
            assert tool_event.result.value.value == int(tool_event.call_id)


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

    expected_path = (
        tmp_path
        / ".weakincentives"
        / "prompts"
        / "overrides"
        / "sample"
        / "prompt"
        / "concurrent.json"
    )
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
    bus = InProcessEventBus()
    session = Session(bus=bus)

    seeded_value = ExampleResult(value=1)
    session[ExampleResult].seed((seeded_value,))

    assert session[ExampleResult].all() == (seeded_value,)

    session.reset()

    assert session[ExampleResult].all() == ()
