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

from tests.helpers.adapters import GENERIC_ADAPTER_NAME
from weakincentives.adapters.core import (
    PromptResponse,
    ProviderAdapter,
)
from weakincentives.deadlines import Deadline
from weakincentives.filesystem import Filesystem
from weakincentives.prompt import (
    Prompt,
    PromptTemplate,
)
from weakincentives.prompt.protocols import PromptProtocol, ProviderAdapterProtocol
from weakincentives.prompt.tool import Tool, ToolContext, ToolResult
from weakincentives.runtime.events import ToolInvoked
from weakincentives.runtime.session import SessionProtocol
from weakincentives.types import SupportsDataclassOrNone, SupportsToolResult
from weakincentives.types.dataclass import is_dataclass_instance

ParamsT = TypeVar("ParamsT", bound=SupportsDataclassOrNone)
ResultT = TypeVar("ResultT", bound=SupportsToolResult)


class ToolSection(Protocol):
    """Protocol representing tool sections used in tests."""

    def tools(self) -> tuple[Tool[SupportsDataclassOrNone, SupportsToolResult], ...]:
        """Return the tools exposed by the section."""


class _DummyAdapter(ProviderAdapter[Any]):
    def evaluate(
        self,
        prompt: Prompt[Any],
        *,
        session: SessionProtocol,
        deadline: Deadline | None = None,
    ) -> PromptResponse[Any]:
        raise NotImplementedError


def build_tool_context(
    session: SessionProtocol, *, filesystem: Filesystem | None = None
) -> ToolContext:
    """Build a ToolContext for testing tool handlers.

    Note: The returned ToolContext has a prompt in active context mode.
    Resources are accessed via context.resources, which delegates to
    prompt.resources.
    """
    prompt = Prompt(PromptTemplate(ns="tests", key="tool-context-helper"))
    adapter = cast(ProviderAdapterProtocol[Any], _DummyAdapter())
    if filesystem is not None:
        prompt = prompt.bind(resources={Filesystem: filesystem})

    # Enter resource context for resource access
    prompt.resources.__enter__()

    return ToolContext(
        prompt=cast(PromptProtocol[Any], prompt),
        rendered_prompt=None,
        adapter=adapter,
        session=session,
    )


def find_tool(
    section: ToolSection, name: str
) -> Tool[SupportsDataclassOrNone, SupportsToolResult]:
    """Return the tool with the provided name from ``section``."""

    for tool in section.tools():
        if tool.name == name:
            assert tool.handler is not None
            return tool
    message = f"Tool {name} not found"
    raise AssertionError(message)


def invoke_tool(
    tool: Tool[ParamsT, ResultT],
    params: ParamsT,
    *,
    session: SessionProtocol,
    filesystem: Filesystem | None = None,
) -> ToolResult[ResultT]:
    """Execute ``tool`` with ``params`` and publish the invocation event."""

    handler = tool.handler
    assert handler is not None
    result = handler(params, context=build_tool_context(session, filesystem=filesystem))
    rendered_output = result.render()
    event = ToolInvoked(
        prompt_name="test",
        adapter=GENERIC_ADAPTER_NAME,
        name=tool.name,
        params=params,
        success=result.success,
        message=result.message,
        session_id=getattr(session, "session_id", None),
        created_at=datetime.now(UTC),
        rendered_output=rendered_output,
    )
    publish_result = session.dispatcher.dispatch(event)
    assert publish_result.ok

    # Dispatch payload directly to session reducers
    if is_dataclass_instance(result.value):
        session.dispatch(result.value)

    return result
