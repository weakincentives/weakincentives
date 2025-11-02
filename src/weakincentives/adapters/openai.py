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

"""Optional OpenAI adapter utilities."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from importlib import import_module
from typing import Any, Final, Literal, Protocol, cast

from ..events import EventBus, HandlerFailure, PromptExecuted, ToolInvoked
from ..prompt._types import SupportsDataclass
from ..prompt.prompt import Prompt, RenderedPrompt
from ..prompt.structured_output import (
    ARRAY_WRAPPER_KEY,
    OutputParseError,
    parse_structured_output,
)
from ..prompt.tool import Tool, ToolResult
from ..serde import parse, schema
from ..session import Session
from ..tools.errors import ToolValidationError
from ._tool_messages import serialize_tool_message
from .core import PromptEvaluationError, PromptResponse

_ERROR_MESSAGE: Final[str] = (
    "OpenAI support requires the optional 'openai' dependency. "
    "Install it with `uv sync --extra openai` or `pip install weakincentives[openai]`."
)


class _CompletionFunctionCall(Protocol):
    name: str
    arguments: str | None


class _ToolCall(Protocol):
    id: str
    function: _CompletionFunctionCall


class _Message(Protocol):
    content: str | Sequence[object] | None
    tool_calls: Sequence[_ToolCall] | None


class _CompletionChoice(Protocol):
    message: _Message


class _CompletionResponse(Protocol):
    choices: Sequence[_CompletionChoice]


class _CompletionsAPI(Protocol):
    def create(self, *args: object, **kwargs: object) -> _CompletionResponse: ...


class _ChatAPI(Protocol):
    completions: _CompletionsAPI


class _OpenAIProtocol(Protocol):
    """Structural type for the OpenAI client."""

    chat: _ChatAPI


class _OpenAIClientFactory(Protocol):
    def __call__(self, **kwargs: object) -> _OpenAIProtocol: ...


def _format_publish_failures(failures: Sequence[HandlerFailure]) -> str:
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


OpenAIProtocol = _OpenAIProtocol


class _OpenAIModule(Protocol):
    OpenAI: _OpenAIClientFactory


def _load_openai_module() -> _OpenAIModule:
    try:
        module = import_module("openai")
    except ModuleNotFoundError as exc:
        raise RuntimeError(_ERROR_MESSAGE) from exc
    return cast(_OpenAIModule, module)


def create_openai_client(**kwargs: object) -> _OpenAIProtocol:
    """Create an OpenAI client, raising a helpful error if the extra is missing."""

    openai_module = _load_openai_module()
    return openai_module.OpenAI(**kwargs)


logger = logging.getLogger(__name__)


ToolChoice = Literal["auto"] | Mapping[str, Any] | None
"""Supported tool choice directives for provider APIs."""


class OpenAIAdapter:
    """Adapter that evaluates prompts against OpenAI's Responses API."""

    def __init__(
        self,
        *,
        model: str,
        tool_choice: ToolChoice = "auto",
        use_native_response_format: bool = True,
        client: _OpenAIProtocol | None = None,
        client_factory: _OpenAIClientFactory | None = None,
        client_kwargs: Mapping[str, object] | None = None,
    ) -> None:
        if client is not None:
            if client_factory is not None:
                raise ValueError(
                    "client_factory cannot be provided when an explicit client is supplied.",
                )
            if client_kwargs:
                raise ValueError(
                    "client_kwargs cannot be provided when an explicit client is supplied.",
                )
        else:
            factory = client_factory or create_openai_client
            client = factory(**dict(client_kwargs or {}))

        self._client = client
        self._model = model
        self._tool_choice: ToolChoice = tool_choice
        self._use_native_response_format = use_native_response_format

    def evaluate[OutputT](
        self,
        prompt: Prompt[OutputT],
        *params: SupportsDataclass,
        parse_output: bool = True,
        bus: EventBus,
        session: Session | None = None,
    ) -> PromptResponse[OutputT]:
        prompt_name = prompt.name or prompt.__class__.__name__

        has_structured_output = (
            getattr(prompt, "_output_type", None) is not None
            and getattr(prompt, "_output_container", None) is not None
        )
        should_disable_instructions = (
            parse_output
            and has_structured_output
            and self._use_native_response_format
            and getattr(prompt, "inject_output_instructions", False)
        )

        if should_disable_instructions:
            rendered = prompt.render(
                *params,
                inject_output_instructions=False,
            )  # type: ignore[reportArgumentType]
        else:
            rendered = prompt.render(*params)  # type: ignore[reportArgumentType]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": rendered.text},
        ]

        should_parse_structured_output = (
            parse_output
            and rendered.output_type is not None
            and rendered.container is not None
        )
        response_format: dict[str, Any] | None = None
        if should_parse_structured_output and self._use_native_response_format:
            response_format = _build_json_schema_response_format(rendered, prompt_name)

        tools = list(rendered.tools)
        tool_specs = [_tool_to_openai_spec(tool) for tool in tools]
        tool_registry = {tool.name: tool for tool in tools}
        tool_events: list[ToolInvoked] = []
        provider_payload: dict[str, Any] | None = None
        # Allow forcing a specific tool once, then fall back to provider defaults.
        next_tool_choice: ToolChoice = self._tool_choice

        while True:
            request_payload: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
            }
            if tool_specs:
                request_payload["tools"] = tool_specs
                if next_tool_choice is not None:
                    request_payload["tool_choice"] = next_tool_choice
            if response_format is not None:
                request_payload["response_format"] = response_format

            try:
                response = self._client.chat.completions.create(**request_payload)
            except Exception as error:  # pragma: no cover - network/SDK failure
                raise PromptEvaluationError(
                    "OpenAI request failed.",
                    prompt_name=prompt_name,
                    phase="request",
                ) from error

            provider_payload = _extract_payload(response)
            choice = _first_choice(response, prompt_name=prompt_name)
            message = choice.message
            tool_calls = list(message.tool_calls or [])

            if not tool_calls:
                final_text = _message_text_content(message.content)
                output: OutputT | None = None
                text_value: str | None = final_text or None

                if should_parse_structured_output:
                    parsed_payload = _extract_parsed_content(message)
                    if parsed_payload is not None:
                        try:
                            output = cast(
                                OutputT,
                                _parse_schema_constrained_payload(
                                    parsed_payload, rendered
                                ),
                            )
                        except (TypeError, ValueError) as error:
                            raise PromptEvaluationError(
                                str(error),
                                prompt_name=prompt_name,
                                phase="response",
                                provider_payload=provider_payload,
                            ) from error
                    else:
                        try:
                            output = parse_structured_output(final_text, rendered)
                        except OutputParseError as error:
                            raise PromptEvaluationError(
                                error.message,
                                prompt_name=prompt_name,
                                phase="response",
                                provider_payload=provider_payload,
                            ) from error
                    if output is not None:
                        text_value = None

                response = PromptResponse(
                    prompt_name=prompt_name,
                    text=text_value,
                    output=output,
                    tool_results=tuple(tool_events),
                    provider_payload=provider_payload,
                )
                _ = bus.publish(
                    PromptExecuted(
                        prompt_name=prompt_name,
                        adapter="openai",
                        result=cast(PromptResponse[object], response),
                    )
                )
                return response

            assistant_tool_calls = [_serialize_tool_call(call) for call in tool_calls]
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": assistant_tool_calls,
                }
            )

            for tool_call in tool_calls:
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
                if tool.handler is None:
                    raise PromptEvaluationError(
                        f"Tool '{tool_name}' does not have a registered handler.",
                        prompt_name=prompt_name,
                        phase="tool",
                        provider_payload=provider_payload,
                    )

                arguments_mapping = _parse_tool_arguments(
                    function.arguments,
                    prompt_name=prompt_name,
                    provider_payload=provider_payload,
                )

                try:
                    tool_params = parse(
                        tool.params_type,
                        arguments_mapping,
                        extra="forbid",
                    )
                except (TypeError, ValueError) as error:
                    raise PromptEvaluationError(
                        f"Failed to parse params for tool '{tool_name}'.",
                        prompt_name=prompt_name,
                        phase="tool",
                        provider_payload=provider_payload,
                    ) from error

                handler = cast(
                    Callable[[SupportsDataclass], ToolResult[SupportsDataclass]],
                    tool.handler,
                )
                try:
                    tool_result = handler(tool_params)
                except ToolValidationError as error:
                    tool_result = ToolResult(
                        message=f"Tool validation failed: {error}",
                        value=None,
                        success=False,
                    )
                except Exception as error:  # pragma: no cover - handler bug
                    logger.exception(
                        "Tool '%s' raised an unexpected exception.", tool_name
                    )
                    tool_result = ToolResult(
                        message=f"Tool '{tool_name}' execution failed: {error}",
                        value=None,
                        success=False,
                    )

                snapshot = session.snapshot() if session is not None else None
                invocation = ToolInvoked(
                    prompt_name=prompt_name,
                    adapter="openai",
                    name=tool_name,
                    params=tool_params,
                    result=cast(ToolResult[object], tool_result),
                    call_id=getattr(tool_call, "id", None),
                )
                tool_events.append(invocation)
                publish_result = bus.publish(invocation)

                if not publish_result.ok:
                    if snapshot is not None and session is not None:
                        session.rollback(snapshot)
                    tool_result.message = _format_publish_failures(
                        publish_result.errors
                    )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": getattr(tool_call, "id", None),
                        "content": serialize_tool_message(tool_result),
                    }
                )

            if isinstance(next_tool_choice, Mapping):
                tool_choice_mapping = cast(Mapping[str, object], next_tool_choice)
                if tool_choice_mapping.get("type") == "function":
                    # Relax forced single-function choice after the first call.
                    next_tool_choice = "auto"


def _tool_to_openai_spec(tool: Tool[Any, Any]) -> dict[str, Any]:
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


def _extract_payload(response: _CompletionResponse) -> dict[str, Any] | None:
    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        try:
            payload = model_dump()
        except Exception:  # pragma: no cover - defensive
            return None
        if isinstance(payload, Mapping):
            mapping_payload = cast(Mapping[str, Any], payload)
            return dict(mapping_payload)
        return None
    if isinstance(response, Mapping):  # pragma: no cover - defensive
        return dict(response)
    return None


def _build_json_schema_response_format(
    rendered: RenderedPrompt[Any], prompt_name: str
) -> dict[str, Any] | None:
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


def _schema_name(prompt_name: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_-]+", "_", prompt_name.strip())
    cleaned = sanitized.strip("_") or "prompt"
    return f"{cleaned}_schema"


def _message_text_content(content: object) -> str:
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


def _extract_parsed_content(message: _Message) -> object | None:
    parsed = getattr(message, "parsed", None)
    if parsed is not None:
        return parsed

    content = message.content
    if isinstance(content, Sequence) and not isinstance(
        content, (str, bytes, bytearray)
    ):
        sequence_content = cast(Sequence[object], content)  # pyright: ignore[reportUnnecessaryCast]
        for part in sequence_content:
            payload = _parsed_payload_from_part(part)
            if payload is not None:
                return payload
    return None


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


def _parse_schema_constrained_payload(
    payload: object, rendered: RenderedPrompt[Any]
) -> object:
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


def _first_choice(
    response: _CompletionResponse, *, prompt_name: str
) -> _CompletionChoice:
    try:
        return response.choices[0]
    except (AttributeError, IndexError) as error:  # pragma: no cover - defensive
        raise PromptEvaluationError(
            "Provider response did not include any choices.",
            prompt_name=prompt_name,
            phase="response",
        ) from error


def _serialize_tool_call(tool_call: _ToolCall) -> dict[str, Any]:
    function = tool_call.function
    return {
        "id": getattr(tool_call, "id", None),
        "type": "function",
        "function": {
            "name": function.name,
            "arguments": function.arguments or "{}",
        },
    }


def _parse_tool_arguments(
    arguments_json: str | None,
    *,
    prompt_name: str,
    provider_payload: dict[str, Any] | None,
) -> dict[str, Any]:
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


__all__ = ["OpenAIAdapter", "OpenAIProtocol"]
