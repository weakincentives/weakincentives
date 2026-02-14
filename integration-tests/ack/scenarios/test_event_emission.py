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

"""Tier 2 ACK scenarios for adapter telemetry event emission."""

from __future__ import annotations

import pytest

from weakincentives.adapters.core import ProviderAdapter
from weakincentives.prompt import Prompt
from weakincentives.runtime.events import PromptExecuted, PromptRendered
from weakincentives.runtime.events.types import ToolInvoked
from weakincentives.runtime.session import Session
from weakincentives.runtime.session.rendered_tools import RenderedTools

from ..adapters import AdapterFixture
from . import (
    GreetingParams,
    TransformRequest,
    build_greeting_prompt,
    build_tool_prompt,
    build_uppercase_tool,
    make_adapter_ns,
)
from ._event_helpers import (
    assert_prompt_executed,
    assert_prompt_rendered,
    assert_tool_invoked,
    capture_events,
)

pytestmark = pytest.mark.ack_capability("event_emission")


def test_prompt_rendered_event(
    adapter: ProviderAdapter[object],
    session: Session,
    adapter_fixture: AdapterFixture,
) -> None:
    """Adapters emit PromptRendered exactly once per evaluation."""
    prompt = Prompt(
        build_greeting_prompt(make_adapter_ns(adapter_fixture.adapter_name))
    ).bind(GreetingParams(audience="event emission"))

    events = capture_events(session, PromptRendered)
    _ = adapter.evaluate(prompt, session=session)

    rendered_events = events[PromptRendered]
    assert_prompt_rendered(
        rendered_events, adapter_fixture.adapter_name, "ack_greeting"
    )


def test_prompt_executed_event(
    adapter: ProviderAdapter[object],
    session: Session,
    adapter_fixture: AdapterFixture,
) -> None:
    """Adapters emit PromptExecuted after prompt completion."""
    prompt = Prompt(
        build_greeting_prompt(make_adapter_ns(adapter_fixture.adapter_name))
    ).bind(GreetingParams(audience="event executed"))

    events = capture_events(session, PromptExecuted)
    _ = adapter.evaluate(prompt, session=session)

    executed_events = events[PromptExecuted]
    assert_prompt_executed(
        executed_events, adapter_fixture.adapter_name, "ack_greeting"
    )


def test_tool_invoked_event(
    adapter: ProviderAdapter[object],
    session: Session,
    adapter_fixture: AdapterFixture,
) -> None:
    """Adapters emit ToolInvoked for bridged tool execution."""
    tool = build_uppercase_tool()
    prompt = Prompt(
        build_tool_prompt(make_adapter_ns(adapter_fixture.adapter_name), tool)
    ).bind(TransformRequest(text="hello"))

    events = capture_events(session, ToolInvoked)
    _ = adapter.evaluate(prompt, session=session)

    tool_events = events[ToolInvoked]
    assert_tool_invoked(tool_events, "uppercase_text", adapter_fixture.adapter_name)


@pytest.mark.ack_capability("rendered_tools_event")
def test_rendered_tools_event(
    adapter: ProviderAdapter[object],
    session: Session,
) -> None:
    """Adapters that support it emit RenderedTools correlated to PromptRendered."""
    tool = build_uppercase_tool()
    prompt = Prompt(build_tool_prompt("integration.ack.rendered-tools", tool)).bind(
        TransformRequest(text="hello")
    )

    events = capture_events(session, PromptRendered, RenderedTools)
    _ = adapter.evaluate(prompt, session=session)

    rendered_prompt_events = events[PromptRendered]
    rendered_tools_events = events[RenderedTools]

    assert len(rendered_prompt_events) == 1
    assert len(rendered_tools_events) == 1

    rendered_tools = rendered_tools_events[0]
    assert isinstance(rendered_tools, RenderedTools)
    assert rendered_tools.get_tool("uppercase_text") is not None
    assert rendered_tools.render_event_id == rendered_prompt_events[0].event_id
