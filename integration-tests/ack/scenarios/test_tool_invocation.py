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

"""Tier 1 ACK scenarios for bridged tool execution."""

from __future__ import annotations

import pytest

from weakincentives.adapters.core import ProviderAdapter
from weakincentives.prompt import Prompt, ToolResult
from weakincentives.runtime.events.types import ToolInvoked
from weakincentives.runtime.session import Session

from ..adapters import AdapterFixture
from . import (
    TransformRequest,
    TransformResult,
    build_tool_prompt,
    build_uppercase_tool,
    make_adapter_ns,
)

pytestmark = pytest.mark.ack_capability("tool_invocation")


def test_bridged_tool_is_called(
    adapter: ProviderAdapter[object],
    session: Session,
    adapter_fixture: AdapterFixture,
) -> None:
    """ACK adapters emit ToolInvoked events for bridged custom tools."""
    calls: list[str] = []
    tool = build_uppercase_tool(calls)

    prompt = Prompt(
        build_tool_prompt(make_adapter_ns(adapter_fixture.adapter_name), tool)
    ).bind(TransformRequest(text="hello"))

    tool_events: list[ToolInvoked] = []
    session.dispatcher.subscribe(ToolInvoked, tool_events.append)

    response = adapter.evaluate(prompt, session=session)

    assert response.text is not None
    assert any(event.name == "uppercase_text" for event in tool_events)


def test_tool_result_is_correct(
    adapter: ProviderAdapter[object],
    session: Session,
    adapter_fixture: AdapterFixture,
) -> None:
    """Tool receives expected params and returns uppercase output."""
    calls: list[str] = []
    tool = build_uppercase_tool(calls)

    prompt = Prompt(
        build_tool_prompt(make_adapter_ns(adapter_fixture.adapter_name), tool)
    ).bind(TransformRequest(text="hello"))

    tool_events: list[ToolInvoked] = []
    session.dispatcher.subscribe(ToolInvoked, tool_events.append)

    _ = adapter.evaluate(prompt, session=session)

    assert "hello" in calls

    uppercase_events = [
        event for event in tool_events if event.name == "uppercase_text"
    ]
    assert uppercase_events

    successful_results: list[ToolResult[object]] = [
        result
        for result in (event.result for event in uppercase_events)
        if isinstance(result, ToolResult) and result.success
    ]
    assert successful_results

    assert any(
        isinstance(result.value, TransformResult) and result.value.text == "HELLO"
        for result in successful_results
    )
