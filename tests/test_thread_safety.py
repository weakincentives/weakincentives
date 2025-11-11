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

from code_reviewer_example import ReviewResponse, SunfishReviewSession
from weakincentives.adapters import PromptResponse
from weakincentives.adapters.core import SessionProtocol
from weakincentives.prompt import Prompt, SupportsDataclass
from weakincentives.prompt.overrides import LocalPromptOverridesStore, PromptOverride
from weakincentives.prompt.tool_result import ToolResult
from weakincentives.runtime.events import EventBus, InProcessEventBus, ToolInvoked
from weakincentives.runtime.session import Session


@dataclass(slots=True)
class ExampleParams:
    value: int


@dataclass(slots=True)
class ExampleResult:
    value: int


class _StubAdapter:
    def evaluate(  # pragma: no cover - not exercised in tests
        self,
        prompt: Prompt[ReviewResponse],
        *params: SupportsDataclass,
        parse_output: bool = True,
        bus: EventBus,
        session: SessionProtocol,
    ) -> PromptResponse[ReviewResponse]:
        raise NotImplementedError


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
        adapter="unit",
        name=f"example-{index}",
        params=params,
        result=cast(ToolResult[object], result),
        call_id=str(index),
        session_id="threaded-session",
        created_at=datetime.now(UTC),
        duration_ms=float(index),
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
        adapter="unit",
        name="example-idempotent",
        params=params,
        result=cast(ToolResult[object], tool_result),
        call_id="999",
        session_id="threaded-session",
        created_at=datetime.now(UTC),
        duration_ms=99.9,
        value=result_payload,
    )

    publish_result = bus.publish(event)
    assert publish_result.handled_count == 1


def test_session_collects_tool_data_across_threads() -> None:
    bus = InProcessEventBus()
    session = Session(bus=bus)

    total_events = 64
    with ThreadPoolExecutor(max_workers=8) as executor:
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
    assert all(isinstance(data.duration_ms, float) for data in tool_data)

    result_slice = session.select_all(ExampleResult)
    assert len(result_slice) == total_events
    assert {value.value for value in result_slice} == set(range(total_events))


def test_local_prompt_overrides_store_seed_is_thread_safe(tmp_path: Path) -> None:
    store = LocalPromptOverridesStore(root_path=tmp_path)
    section = _DummySection(template="Hello, ${name}!")
    prompt = _DummyPrompt(
        ns="sample",
        key="prompt",
        _sections=(_DummySectionNode(path=("intro",), section=section),),
    )

    def seed() -> PromptOverride:
        return store.seed_if_necessary(prompt, tag="concurrent")

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(seed) for _ in range(12)]
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


def test_sunfish_review_session_records_history_concurrently(tmp_path: Path) -> None:
    overrides_store = LocalPromptOverridesStore(root_path=tmp_path)
    session = SunfishReviewSession(
        adapter=_StubAdapter(),
        overrides_store=overrides_store,
        override_tag="threads",
    )

    total_events = 20
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(_publish_tool_event, session._bus, index)
            for index in range(total_events)
        ]
        for future in futures:
            future.result()

    history = session.render_tool_history()
    entries = [
        line for line in history.splitlines() if line and not line.startswith(" ")
    ]
    assert len(entries) == total_events
    recorded_names = {entry.split(". ", 1)[1].split(" ", 1)[0] for entry in entries}
    assert recorded_names == {f"example-{index}" for index in range(total_events)}
    recorded_indexes = {int(entry.split(". ", 1)[0]) for entry in entries}
    assert recorded_indexes == set(range(1, total_events + 1))
