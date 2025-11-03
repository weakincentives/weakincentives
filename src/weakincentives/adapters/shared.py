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

"""Shared adapter helpers."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, cast

from ..events import EventBus, HandlerFailure, ToolInvoked
from ..prompt._types import SupportsDataclass
from ..prompt.tool import Tool, ToolResult
from ..serde import parse
from ..session import Session
from ..tools.errors import ToolValidationError
from .core import PromptEvaluationError

logger = logging.getLogger(__name__)


class ProviderFunctionCall(Protocol):
    name: str
    arguments: str | None


class ProviderToolCall(Protocol):
    @property
    def function(self) -> ProviderFunctionCall: ...


class ToolArgumentsParser(Protocol):
    def __call__(
        self,
        arguments_json: str | None,
        *,
        prompt_name: str,
        provider_payload: dict[str, Any] | None,
    ) -> dict[str, Any]: ...


def parse_tool_arguments(
    arguments_json: str | None,
    *,
    prompt_name: str,
    provider_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    """Parse tool call arguments from a provider payload."""

    if not arguments_json:
        return {}
    try:
        parsed = json.loads(arguments_json)
    except json.JSONDecodeError as error:
        raise PromptEvaluationError(
            "Failed to decode tool call arguments.",
            prompt_name=prompt_name,
            phase="tool",
            provider_payload=provider_payload,
        ) from error
    if not isinstance(parsed, Mapping):
        raise PromptEvaluationError(
            "Tool call arguments must be a JSON object.",
            prompt_name=prompt_name,
            phase="tool",
            provider_payload=provider_payload,
        )
    return dict(cast(Mapping[str, Any], parsed))


def execute_tool_call(
    *,
    adapter_name: str,
    tool_call: ProviderToolCall,
    tool_registry: Mapping[str, Tool[SupportsDataclass, SupportsDataclass]],
    bus: EventBus,
    session: Session | None,
    prompt_name: str,
    provider_payload: dict[str, Any] | None,
    format_publish_failures: Callable[[Sequence[HandlerFailure]], str],
    parse_arguments: ToolArgumentsParser,
    logger_override: logging.Logger | None = None,
) -> tuple[ToolInvoked, ToolResult[SupportsDataclass]]:
    """Execute a provider tool call and publish the resulting event."""

    function = tool_call.function
    tool_name = function.name
    tool = tool_registry.get(tool_name)
    if tool is None:
        raise PromptEvaluationError(
            f"Unknown tool '{tool_name}' requested by provider.",
            prompt_name=prompt_name,
            phase="tool",
            provider_payload=provider_payload,
        )
    handler = tool.handler
    if handler is None:
        raise PromptEvaluationError(
            f"Tool '{tool_name}' does not have a registered handler.",
            prompt_name=prompt_name,
            phase="tool",
            provider_payload=provider_payload,
        )

    arguments_mapping = parse_arguments(
        function.arguments,
        prompt_name=prompt_name,
        provider_payload=provider_payload,
    )

    try:
        parsed_params = parse(tool.params_type, arguments_mapping, extra="forbid")
    except (TypeError, ValueError) as error:
        raise PromptEvaluationError(
            f"Failed to parse params for tool '{tool_name}'.",
            prompt_name=prompt_name,
            phase="tool",
            provider_payload=provider_payload,
        ) from error
    tool_params = cast(SupportsDataclass, parsed_params)
    tool_result: ToolResult[SupportsDataclass]
    try:
        tool_result = handler(tool_params)
    except ToolValidationError as error:
        tool_result = ToolResult(
            message=f"Tool validation failed: {error}",
            value=None,
            success=False,
        )
    except Exception as error:  # noqa: BLE001 - propagate message via ToolResult
        log = logger_override or logger
        log.exception("Tool '%s' raised an unexpected exception.", tool_name)
        tool_result = ToolResult(
            message=f"Tool '{tool_name}' execution failed: {error}",
            value=None,
            success=False,
        )

    snapshot = session.snapshot() if session is not None else None
    invocation = ToolInvoked(
        prompt_name=prompt_name,
        adapter=adapter_name,
        name=tool_name,
        params=tool_params,
        result=cast(ToolResult[object], tool_result),
        call_id=getattr(tool_call, "id", None),
    )
    publish_result = bus.publish(invocation)
    if not publish_result.ok:
        if snapshot is not None and session is not None:
            session.rollback(snapshot)
        tool_result.message = format_publish_failures(publish_result.errors)
    return invocation, tool_result


__all__ = ["execute_tool_call", "parse_tool_arguments", "ProviderToolCall"]
