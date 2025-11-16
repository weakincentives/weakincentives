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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import uuid4

import pytest

from code_reviewer_example import initialize_code_reviewer_runtime
from tests.helpers.adapters import UNIT_TEST_ADAPTER_NAME
from weakincentives.prompt._types import SupportsToolResult
from weakincentives.prompt.overrides import LocalPromptOverridesStore, PromptOverride
from weakincentives.prompt.tool_result import ToolResult
from weakincentives.runtime.events import InProcessEventBus, ToolInvoked
from weakincentives.runtime.session import Session

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
        return ()


@dataclass(slots=True)
class _DummySectionNode:
    path: tuple[str, ...]
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
    event = ToolInvoked(
        prompt_name="test",
        adapter=UNIT_TEST_ADAPTER_NAME,
        name=f"example-{index}",
        params=params,
        result=cast(ToolResult[SupportsToolResult], result),
        call_id=str(index),
        session_id=THREAD_SESSION_ID,
        created_at=datetime.now(UTC),
        value=result_payload,
    )
    bus.publish(event)


def test_session_attach_to_bus_is_idempotent() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)

    session._attach_to_bus(bus)

    params = ExampleParams(value=999)
    result_payload = ExampleResult(value=999)
    tool_result = ToolResult(message="ok", value=result_payload)
    event = ToolInvoked(
        prompt_name="test",
        adapter=UNIT_TEST_ADAPTER_NAME,
        name="example-idempotent",
        params=params,
        result=cast(ToolResult[SupportsToolResult], tool_result),
        call_id="999",
        session_id=THREAD_SESSION_ID,
        created_at=datetime.now(UTC),
        value=result_payload,
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

    tool_data = session.select_all(ToolInvoked)
    assert len(tool_data) == total_events
    assert {data.call_id for data in tool_data} == {
        str(index) for index in range(total_events)
    }

    result_slice = session.select_all(ExampleResult)
    assert len(result_slice) == total_events
    assert {value.value for value in result_slice} == set(range(total_events))


@pytest.mark.threadstress(min_workers=2, max_workers=6)
def test_local_prompt_overrides_store_seed_is_thread_safe(
    threadstress_workers: int, tmp_path: Path
) -> None:
    store = LocalPromptOverridesStore(root_path=tmp_path)
    section = _DummySection(template="Hello, ${name}!")
    prompt = _DummyPrompt(
        ns="sample",
        key="prompt",
        _sections=(_DummySectionNode(path=("intro",), section=section),),
    )

    def seed() -> PromptOverride:
        return store.seed_if_necessary(prompt, tag="concurrent")

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


def test_code_reviewer_session_reset_clears_runtime_state(tmp_path: Path) -> None:
    overrides_store = LocalPromptOverridesStore(root_path=tmp_path)
    (
        _prompt,
        session,
        _bus,
        _store,
        _tag,
    ) = initialize_code_reviewer_runtime(
        overrides_store=overrides_store,
        override_tag="reset",
    )

    seeded_value = ExampleResult(value=1)
    session.seed_slice(ExampleResult, (seeded_value,))

    assert session.select_all(ExampleResult) == (seeded_value,)

    session.reset()

    assert session.select_all(ExampleResult) == ()
