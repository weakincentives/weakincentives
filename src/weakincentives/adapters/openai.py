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

from collections.abc import Iterable, Mapping, Sequence
from importlib import import_module
from typing import TYPE_CHECKING, Any, Final, Protocol, cast

from ..events import EventBus
from ..logging import StructuredLogger, get_logger
from ..prompt._types import SupportsDataclass
from ..prompt.prompt import Prompt
from . import shared as _shared
from ._provider_protocols import ProviderChoice
from ._tool_messages import serialize_tool_message
from .core import PromptEvaluationError, PromptResponse, SessionProtocol
from .shared import (
    ToolChoice,
    build_json_schema_response_format,
    format_publish_failures,
    parse_tool_arguments,
    run_conversation,
)

if TYPE_CHECKING:
    from ..adapters.core import ProviderAdapter

_ERROR_MESSAGE: Final[str] = (
    "OpenAI support requires the optional 'openai' dependency. "
    "Install it with `uv sync --extra openai` or `pip install weakincentives[openai]`."
)


class _ResponsesAPI(Protocol):
    def create(self, *args: object, **kwargs: object) -> object: ...

    def parse(self, *args: object, **kwargs: object) -> object: ...


class _OpenAIProtocol(Protocol):
    """Structural type for the OpenAI client."""

    responses: _ResponsesAPI


class _OpenAIClientFactory(Protocol):
    def __call__(self, **kwargs: object) -> _OpenAIProtocol: ...


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


logger: StructuredLogger = get_logger(__name__, context={"component": "adapter.openai"})


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
        super().__init__()
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
        session: SessionProtocol,
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
            )
        else:
            rendered = prompt.render(*params)
        response_format: dict[str, Any] | None = None
        should_parse_structured_output = (
            parse_output
            and rendered.output_type is not None
            and rendered.container is not None
        )
        if should_parse_structured_output and self._use_native_response_format:
            response_format = build_json_schema_response_format(rendered, prompt_name)

        def _call_provider(
            messages: list[dict[str, Any]],
            tool_specs: Sequence[Mapping[str, Any]],
            tool_choice_directive: ToolChoice | None,
            response_format_payload: Mapping[str, Any] | None,
        ) -> object:
            request_payload: dict[str, Any] = {
                "model": self._model,
                "input": _convert_messages_to_input(messages),
            }
            if tool_specs:
                request_payload["tools"] = list(tool_specs)
                if tool_choice_directive is not None:
                    request_payload["tool_choice"] = tool_choice_directive
            if response_format_payload is not None:
                request_payload["text"] = response_format_payload

            try:
                if response_format_payload is not None:
                    return self._client.responses.parse(**request_payload)
                return self._client.responses.create(**request_payload)
            except Exception as error:  # pragma: no cover - network/SDK failure
                raise PromptEvaluationError(
                    "OpenAI request failed.",
                    prompt_name=prompt_name,
                    phase="request",
                ) from error

        def _select_choice(response: object) -> ProviderChoice:
            return cast(ProviderChoice, _wrap_response_choice(response, prompt_name))

        return run_conversation(
            adapter_name="openai",
            adapter=cast("ProviderAdapter[OutputT]", self),
            prompt=prompt,
            prompt_name=prompt_name,
            rendered=rendered,
            initial_messages=[{"role": "system", "content": rendered.text}],
            parse_output=parse_output,
            bus=bus,
            session=session,
            tool_choice=self._tool_choice,
            response_format=response_format,
            require_structured_output_text=False,
            call_provider=_call_provider,
            select_choice=_select_choice,
            serialize_tool_message_fn=serialize_tool_message,
            format_publish_failures=format_publish_failures,
            parse_arguments=parse_tool_arguments,
            logger_override=logger,
        )


__all__ = [
    "OpenAIAdapter",
    "OpenAIProtocol",
    "extract_parsed_content",
    "message_text_content",
    "parse_schema_constrained_payload",
]


message_text_content = _shared.message_text_content
extract_parsed_content = _shared.extract_parsed_content
parse_schema_constrained_payload = _shared.parse_schema_constrained_payload


def _convert_messages_to_input(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    input_items: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "user")
        content = message.get("content")

        if role == "tool":
            call_id = message.get("tool_call_id")
            if call_id is None:
                continue
            output_text = _shared.message_text_content(content)
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output_text,
                }
            )
            continue

        text_value = _shared.message_text_content(content)
        message_payload: dict[str, Any] = {
            "type": "message",
            "role": role,
            "content": [{"type": "input_text", "text": text_value}],
        }
        input_items.append(message_payload)

        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, Sequence):
            tool_call_sequence = cast(Sequence[object], tool_calls)
            for tool_call_obj in tool_call_sequence:
                if not isinstance(tool_call_obj, Mapping):
                    continue
                tool_call_mapping = cast(Mapping[str, Any], tool_call_obj)
                function_payload = tool_call_mapping.get("function")
                if not isinstance(function_payload, Mapping):
                    function_payload = {}
                function_mapping = cast(Mapping[str, Any], function_payload)
                call_identifier = tool_call_mapping.get("id") or tool_call_mapping.get(
                    "call_id"
                )
                call_id_value = (
                    str(call_identifier) if call_identifier is not None else None
                )
                arguments_value = function_mapping.get("arguments")
                if arguments_value is None:
                    arguments_value = "{}"
                input_items.append(
                    {
                        "type": "function_call",
                        "call_id": call_id_value,
                        "name": function_mapping.get("name"),
                        "arguments": arguments_value,
                    }
                )
    return input_items


def _wrap_response_choice(response: object, prompt_name: str) -> _ResponseChoice:
    output_items = getattr(response, "output", None)
    if not isinstance(output_items, Sequence):
        raise PromptEvaluationError(
            "Provider response did not include any output items.",
            prompt_name=prompt_name,
            phase="response",
            provider_payload=_shared.extract_payload(response),
        )

    content_parts: list[object] = []
    tool_calls: list[_ResponseToolCall] = []
    parsed_payload: object | None = getattr(response, "parsed", None)

    for item in cast(Sequence[object], output_items):
        item_type = getattr(item, "type", None)
        if item_type == "message":
            message_content = getattr(item, "content", ())
            if isinstance(message_content, Sequence):
                sequence_content = cast(Sequence[object], message_content)
                content_parts.extend(sequence_content)
                if parsed_payload is None:
                    for part in sequence_content:
                        parsed_candidate = getattr(part, "parsed", None)
                        if parsed_candidate is not None:
                            parsed_payload = parsed_candidate
            elif message_content is not None:
                content_parts.append(message_content)
        elif item_type == "function_call":
            tool_calls.append(_ResponseToolCall(item))
        else:
            content_parts.append(item)

    message = _ResponseMessage(content_parts, tool_calls, parsed_payload)
    return _ResponseChoice(message)


class _ResponseFunctionCall:
    def __init__(self, call: object) -> None:
        super().__init__()
        self._call = call

    @property
    def name(self) -> str | None:
        return getattr(self._call, "name", None)

    @property
    def arguments(self) -> str | None:
        return getattr(self._call, "arguments", None)


class _ResponseToolCall:
    def __init__(self, call: object) -> None:
        super().__init__()
        self._call = call
        self.id = getattr(call, "id", None) or getattr(call, "call_id", None)

    @property
    def function(self) -> _ResponseFunctionCall:
        return _ResponseFunctionCall(self._call)

    def model_dump(self) -> dict[str, Any]:
        return {
            "type": "function",
            "id": self.id,
            "function": {
                "name": getattr(self._call, "name", None),
                "arguments": getattr(self._call, "arguments", None),
            },
        }


class _ResponseMessage:
    def __init__(
        self,
        content: Sequence[object],
        tool_calls: Sequence[_ResponseToolCall],
        parsed: object | None,
    ) -> None:
        super().__init__()
        self.content: tuple[object, ...] = tuple(content)
        self.tool_calls: tuple[_ResponseToolCall, ...] | None = (
            tuple(tool_calls) if tool_calls else None
        )
        self.parsed = parsed

    def model_dump(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "content": [_model_dump_value(part) for part in self.content],
        }
        if self.tool_calls:
            payload["tool_calls"] = [call.model_dump() for call in self.tool_calls]
        if self.parsed is not None:
            payload["parsed"] = self.parsed
        return payload


class _ResponseChoice:
    def __init__(self, message: _ResponseMessage) -> None:
        super().__init__()
        self.message = message

    def model_dump(self) -> dict[str, Any]:
        return {"message": self.message.model_dump()}


def _model_dump_value(value: object) -> object:
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump()
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key_obj, value_obj in cast(Iterable[tuple[Any, Any]], value.items()):
            key_any: Any = key_obj
            value_any: Any = value_obj
            key_str = str(key_any)
            result[key_str] = value_any
        return result
    return value
