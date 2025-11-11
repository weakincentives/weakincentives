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

"""Shared helpers for tool tests."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol, TypeVar, cast

from weakincentives.adapters.core import (
    PromptResponse,
    ProviderAdapter,
    SessionProtocol,
)
from weakincentives.prompt import Prompt, SupportsDataclass
from weakincentives.prompt.tool import Tool, ToolContext, ToolResult
from weakincentives.runtime.events import InProcessEventBus, ToolInvoked

ParamsT = TypeVar("ParamsT", bound=SupportsDataclass)
ResultT = TypeVar("ResultT", bound=SupportsDataclass)


class ToolSection(Protocol):
    """Protocol representing tool sections used in tests."""

    def tools(self) -> tuple[Tool[SupportsDataclass, SupportsDataclass], ...]:
        """Return the tools exposed by the section."""


class _DummyAdapter(ProviderAdapter[Any]):
    def evaluate(
        self,
        prompt: Prompt[Any],
        *params: SupportsDataclass,
        parse_output: bool = True,
        bus: InProcessEventBus,
        session: SessionProtocol,
    ) -> PromptResponse[Any]:
        raise NotImplementedError


def _build_context(bus: InProcessEventBus, session: SessionProtocol) -> ToolContext:
    prompt = Prompt(ns="tests", key="tool-context-helper")
    adapter = cast(ProviderAdapter[Any], _DummyAdapter())
    return ToolContext(
        prompt=prompt,
        rendered_prompt=None,
        adapter=adapter,
        session=session,
        event_bus=bus,
    )


def find_tool(
    section: ToolSection, name: str
) -> Tool[SupportsDataclass, SupportsDataclass]:
    """Return the tool with the provided name from ``section``."""

    for tool in section.tools():
        if tool.name == name:
            assert tool.handler is not None
            return tool
    message = f"Tool {name} not found"
    raise AssertionError(message)


def invoke_tool(
    bus: InProcessEventBus,
    tool: Tool[ParamsT, ResultT],
    params: ParamsT,
    *,
    session: SessionProtocol,
) -> ToolResult[ResultT]:
    """Execute ``tool`` with ``params`` and publish the invocation event."""

    handler = tool.handler
    assert handler is not None
    result = handler(params, context=_build_context(bus, session))
    event = ToolInvoked(
        prompt_name="test",
        adapter="adapter",
        name=tool.name,
        params=params,
        result=cast(ToolResult[object], result),
        session_id=getattr(session, "session_id", None),
        created_at=datetime.now(UTC),
        duration_ms=0.0,
        value=cast(SupportsDataclass | None, result.value),
    )
    publish_result = bus.publish(event)
    assert publish_result.ok
    return result
