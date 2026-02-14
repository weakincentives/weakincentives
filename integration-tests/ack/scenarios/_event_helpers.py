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

"""Shared event assertions for ACK scenarios."""

from __future__ import annotations

from typing import cast

from weakincentives.runtime.events import PromptExecuted, PromptRendered
from weakincentives.runtime.events.types import ToolInvoked
from weakincentives.runtime.session import Session


def capture_events(
    session: Session,
    *event_types: type[object],
) -> dict[type[object], list[object]]:
    """Subscribe to event types and return mutable capture lists."""
    captured: dict[type[object], list[object]] = {}
    for event_type in event_types:
        events: list[object] = []
        session.dispatcher.subscribe(event_type, events.append)
        captured[event_type] = events
    return captured


def assert_prompt_rendered(
    events: list[object],
    adapter_name: str,
    prompt_name: str,
) -> None:
    """Assert a single PromptRendered event with expected metadata."""
    assert len(events) == 1, (
        f"Expected exactly one PromptRendered event, got {len(events)}"
    )
    event = cast(PromptRendered, events[0])
    assert event.adapter == adapter_name
    assert event.prompt_name == prompt_name
    assert event.rendered_prompt.strip()


def assert_prompt_executed(
    events: list[object],
    adapter_name: str,
    prompt_name: str,
) -> None:
    """Assert a single PromptExecuted event with expected metadata."""
    assert len(events) == 1, (
        f"Expected exactly one PromptExecuted event, got {len(events)}"
    )
    event = cast(PromptExecuted, events[0])
    assert event.adapter == adapter_name
    assert event.prompt_name == prompt_name


def assert_tool_invoked(
    events: list[object],
    tool_name: str,
    adapter_name: str,
) -> None:
    """Assert at least one ToolInvoked event for the target tool."""
    tool_events = [cast(ToolInvoked, event) for event in events]
    matching = [event for event in tool_events if event.name == tool_name]
    assert matching, (
        f"Expected at least one ToolInvoked event for {tool_name}, "
        f"saw {[event.name for event in tool_events]}"
    )
    for event in matching:
        assert event.adapter == adapter_name
