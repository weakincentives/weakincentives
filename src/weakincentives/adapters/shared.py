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

"""Shared helpers for provider adapters."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, Protocol, cast

from ..events import EventBus, HandlerFailure, ToolInvoked
from ..prompt._types import SupportsDataclass
from ..prompt.prompt import RenderedPrompt
from ..prompt.structured_output import ARRAY_WRAPPER_KEY
from ..prompt.tool import Tool, ToolResult
from ..serde import parse, schema
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


ToolChoice = Literal["auto"] | Mapping[str, Any] | None
"""Supported tool choice directives for provider APIs."""


def format_publish_failures(failures: Sequence[HandlerFailure]) -> str:
    """Summarize publish failures encountered while applying tool results."""

    messages: list[str] = []
    for failure in failures:
        error = failure.error
        message = str(error).strip()
        if not message:
            message = error.__class__.__name__
        messages.append(message)

    if not messages:
        return "Reducer errors prevented applying tool result."

    joined = "; ".join(messages)
    return f"Reducer errors prevented applying tool result: {joined}"


def tool_to_spec(tool: Tool[Any, Any]) -> dict[str, Any]:
    """Return a provider-agnostic tool specification payload."""

    parameters_schema = schema(tool.params_type, extra="forbid")
    parameters_schema.pop("title", None)
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": parameters_schema,
        },
    }


def extract_payload(response: object) -> dict[str, Any] | None:
    """Return a provider payload from an SDK response when available."""

    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        try:
            payload = model_dump()
        except Exception:  # pragma: no cover - defensive
            return None
        if isinstance(payload, Mapping):
            mapping_payload = _mapping_to_str_dict(cast(Mapping[Any, Any], payload))
            if mapping_payload is not None:
                return mapping_payload
        return None
    if isinstance(response, Mapping):  # pragma: no cover - defensive
        mapping_payload = _mapping_to_str_dict(cast(Mapping[Any, Any], response))
        if mapping_payload is not None:
            return mapping_payload
    return None


def first_choice(response: object, *, prompt_name: str) -> object:
    """Return the first choice in a provider response or raise consistently."""

    choices = getattr(response, "choices", None)
    if not isinstance(choices, Sequence):
        raise PromptEvaluationError(
            "Provider response did not include any choices.",
            prompt_name=prompt_name,
            phase="response",
        )
    sequence_choices = cast(Sequence[object], choices)
    try:
        return sequence_choices[0]
    except IndexError as error:  # pragma: no cover - defensive
        raise PromptEvaluationError(
            "Provider response did not include any choices.",
            prompt_name=prompt_name,
            phase="response",
        ) from error


def serialize_tool_call(tool_call: ProviderToolCall) -> dict[str, Any]:
    """Serialize a provider tool call into the assistant message payload."""

    function = tool_call.function
    return {
        "id": getattr(tool_call, "id", None),
        "type": "function",
        "function": {
            "name": function.name,
            "arguments": function.arguments or "{}",
        },
    }


def parse_tool_arguments(
    arguments_json: str | None,
    *,
    prompt_name: str,
    provider_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    """Decode tool call arguments from provider payloads."""

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
    parsed_mapping = cast(Mapping[Any, Any], parsed)
    arguments: dict[str, Any] = {}
    for key, value in parsed_mapping.items():
        if not isinstance(key, str):
            raise PromptEvaluationError(
                "Tool call arguments must use string keys.",
                prompt_name=prompt_name,
                phase="tool",
                provider_payload=provider_payload,
            )
        arguments[key] = value
    return arguments


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


def build_json_schema_response_format(
    rendered: RenderedPrompt[Any], prompt_name: str
) -> dict[str, Any] | None:
    """Construct a JSON schema response format for structured outputs."""

    output_type = rendered.output_type
    container = rendered.container
    allow_extra_keys = rendered.allow_extra_keys

    if output_type is None or container is None:
        return None

    extra_mode: Literal["ignore", "forbid"] = "ignore" if allow_extra_keys else "forbid"
    base_schema = schema(output_type, extra=extra_mode)
    base_schema.pop("title", None)

    if container == "array":
        schema_payload = cast(
            dict[str, Any],
            {
                "type": "object",
                "properties": {
                    ARRAY_WRAPPER_KEY: {
                        "type": "array",
                        "items": base_schema,
                    }
                },
                "required": [ARRAY_WRAPPER_KEY],
            },
        )
        if not allow_extra_keys:
            schema_payload["additionalProperties"] = False
    else:
        schema_payload = base_schema

    schema_name = _schema_name(prompt_name)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "schema": schema_payload,
        },
    }


def parse_schema_constrained_payload(
    payload: object, rendered: RenderedPrompt[Any]
) -> object:
    """Parse structured provider payloads constrained by prompt schema."""

    dataclass_type = rendered.output_type
    container = rendered.container
    allow_extra_keys = rendered.allow_extra_keys

    if dataclass_type is None or container is None:
        raise TypeError("Prompt does not declare structured output.")

    extra_mode: Literal["ignore", "forbid"] = "ignore" if allow_extra_keys else "forbid"

    if container == "object":
        if not isinstance(payload, Mapping):
            raise TypeError("Expected provider payload to be a JSON object.")
        return parse(
            dataclass_type, cast(Mapping[str, object], payload), extra=extra_mode
        )

    if container == "array":
        if isinstance(payload, Mapping):
            if ARRAY_WRAPPER_KEY not in payload:
                raise TypeError("Expected provider payload to be a JSON array.")
            payload = cast(Mapping[str, object], payload)[ARRAY_WRAPPER_KEY]
        if not isinstance(payload, Sequence) or isinstance(
            payload, (str, bytes, bytearray)
        ):
            raise TypeError("Expected provider payload to be a JSON array.")
        parsed_items: list[object] = []
        sequence_payload = cast(Sequence[object], payload)  # pyright: ignore[reportUnnecessaryCast]
        for index, item in enumerate(sequence_payload):
            if not isinstance(item, Mapping):
                raise TypeError(f"Array item at index {index} is not an object.")
            parsed_item = parse(
                dataclass_type,
                cast(Mapping[str, object], item),
                extra=extra_mode,
            )
            parsed_items.append(parsed_item)
        return parsed_items

    raise TypeError("Unknown output container declared.")


def message_text_content(content: object) -> str:
    """Extract text content from provider message payloads."""

    if isinstance(content, str) or content is None:
        return content or ""
    if isinstance(content, Sequence) and not isinstance(
        content, (str, bytes, bytearray)
    ):
        fragments: list[str] = []
        sequence_content = cast(Sequence[object], content)  # pyright: ignore[reportUnnecessaryCast]
        for part in sequence_content:
            fragments.append(_content_part_text(part))
        return "".join(fragments)
    return str(content)


def extract_parsed_content(message: object) -> object | None:
    """Extract structured payloads surfaced directly by the provider."""

    parsed = getattr(message, "parsed", None)
    if parsed is not None:
        return parsed

    content = getattr(message, "content", None)
    if isinstance(content, Sequence) and not isinstance(
        content, (str, bytes, bytearray)
    ):
        sequence_content = cast(Sequence[object], content)  # pyright: ignore[reportUnnecessaryCast]
        for part in sequence_content:
            payload = _parsed_payload_from_part(part)
            if payload is not None:
                return payload
    return None


def _schema_name(prompt_name: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_-]+", "_", prompt_name.strip())
    cleaned = sanitized.strip("_") or "prompt"
    return f"{cleaned}_schema"


def _content_part_text(part: object) -> str:
    if part is None:
        return ""
    if isinstance(part, Mapping):
        mapping_part = cast(Mapping[str, object], part)
        part_type = mapping_part.get("type")
        if part_type in {"output_text", "text"}:
            text_value = mapping_part.get("text")
            if isinstance(text_value, str):
                return text_value
        return ""
    part_type = getattr(part, "type", None)
    if part_type in {"output_text", "text"}:
        text_value = getattr(part, "text", None)
        if isinstance(text_value, str):
            return text_value
    return ""


def _parsed_payload_from_part(part: object) -> object | None:
    if isinstance(part, Mapping):
        mapping_part = cast(Mapping[str, object], part)
        if mapping_part.get("type") == "output_json":
            return mapping_part.get("json")
        return None
    part_type = getattr(part, "type", None)
    if part_type == "output_json":
        return getattr(part, "json", None)
    return None


def _mapping_to_str_dict(mapping: Mapping[Any, Any]) -> dict[str, Any] | None:
    str_mapping: dict[str, Any] = {}
    for key, value in mapping.items():
        if not isinstance(key, str):
            return None
        str_mapping[key] = value
    return str_mapping


__all__ = [
    "ProviderFunctionCall",
    "ProviderToolCall",
    "ToolArgumentsParser",
    "ToolChoice",
    "build_json_schema_response_format",
    "execute_tool_call",
    "extract_parsed_content",
    "extract_payload",
    "first_choice",
    "format_publish_failures",
    "message_text_content",
    "parse_schema_constrained_payload",
    "parse_tool_arguments",
    "serialize_tool_call",
    "tool_to_spec",
    "_content_part_text",
    "_parsed_payload_from_part",
]
