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

"""Optional Anthropic adapter utilities."""

# pyright: ignore[reportUnknownArgumentType,reportUnknownVariableType,reportUnknownMemberType,reportUnnecessaryIsInstance,reportUnnecessaryCast,reportRedundantCast]

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from importlib import import_module
from typing import TYPE_CHECKING, Any, Final, Protocol, cast

from ..events import EventBus
from ..logging import StructuredLogger, get_logger
from ..prompt._types import SupportsDataclass
from ..prompt.prompt import Prompt
from . import shared as _shared
from ._tool_messages import serialize_tool_message
from .core import PromptEvaluationError, PromptResponse
from .shared import (
    ProviderChoice,
    ProviderCompletionResponse,
    ToolChoice,
    format_publish_failures,
    parse_tool_arguments,
    run_conversation,
)

if TYPE_CHECKING:
    from ..session.session import Session

_ERROR_MESSAGE: Final[str] = (
    "Anthropic support requires the optional 'anthropic' dependency. "
    "Install it with `uv sync --extra anthropic` or `pip install weakincentives[anthropic]`."
)


class _MessagesAPI(Protocol):
    def create(self, *args: object, **kwargs: object) -> object: ...


class _AnthropicProtocol(Protocol):
    """Structural type for the Anthropic client."""

    messages: _MessagesAPI


class _AnthropicClientFactory(Protocol):
    def __call__(self, **kwargs: object) -> _AnthropicProtocol: ...


AnthropicProtocol = _AnthropicProtocol


class _AnthropicModule(Protocol):
    Anthropic: _AnthropicClientFactory


def _load_anthropic_module() -> _AnthropicModule:
    try:
        module = import_module("anthropic")
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(_ERROR_MESSAGE) from exc
    return cast(_AnthropicModule, module)


def create_anthropic_client(**kwargs: object) -> _AnthropicProtocol:
    """Create an Anthropic client, raising a helpful error if the extra is missing."""

    anthropic_module = _load_anthropic_module()
    return anthropic_module.Anthropic(**kwargs)


logger: StructuredLogger = get_logger(
    __name__, context={"component": "adapter.anthropic"}
)


class AnthropicAdapter:
    """Adapter that evaluates prompts against Anthropic's Messages API."""

    def __init__(
        self,
        *,
        model: str,
        tool_choice: ToolChoice = "auto",
        max_output_tokens: int = 1024,
        client: _AnthropicProtocol | None = None,
        client_factory: _AnthropicClientFactory | None = None,
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
            factory = client_factory or create_anthropic_client
            client = factory(**dict(client_kwargs or {}))

        self._client = client
        self._model = model
        self._tool_choice: ToolChoice = tool_choice
        self._max_output_tokens = max_output_tokens

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
            and getattr(prompt, "inject_output_instructions", False)
        )

        if should_disable_instructions:
            rendered = prompt.render(
                *params,
                inject_output_instructions=False,
            )
        else:
            rendered = prompt.render(*params)

        system_template_text: str | None = None
        if params:
            placeholder_params = tuple(
                _placeholder_dataclass(param) for param in params
            )
            if should_disable_instructions:
                template_rendered = prompt.render(
                    *placeholder_params,
                    inject_output_instructions=False,
                )
            else:
                template_rendered = prompt.render(*placeholder_params)
            system_template_text = _strip_markdown_headings(template_rendered.text)

        def _call_provider(
            messages: list[dict[str, Any]],
            tool_specs: Sequence[Mapping[str, Any]],
            tool_choice_directive: ToolChoice | None,
            response_format_payload: Mapping[str, Any] | None,
        ) -> object:
            del (
                response_format_payload
            )  # Anthropic messages API does not support this parameter.
            system, anthropic_messages = _convert_messages(
                messages,
                prompt_name,
                system_template_text=system_template_text,
            )
            tools_payload = _convert_tools(tool_specs, prompt_name)
            effective_tool_choice: ToolChoice | None = tool_choice_directive
            if effective_tool_choice is None:
                effective_tool_choice = self._tool_choice
            tool_choice_payload = None
            if effective_tool_choice is not None:
                tool_choice_payload = _convert_tool_choice(
                    effective_tool_choice, prompt_name
                )
                if not tools_payload and effective_tool_choice == "auto":
                    tool_choice_payload = None

            request_payload: dict[str, Any] = {
                "model": self._model,
                "messages": anthropic_messages,
                "max_tokens": self._max_output_tokens,
            }
            if system is not None:
                request_payload["system"] = system
            if tools_payload:
                request_payload["tools"] = tools_payload
            if tool_choice_payload is not None:
                request_payload["tool_choice"] = tool_choice_payload
            try:
                response = self._client.messages.create(**request_payload)
            except Exception as error:  # pragma: no cover - network/SDK failure
                raise PromptEvaluationError(
                    "Anthropic request failed.",
                    prompt_name=prompt_name,
                    phase="request",
                ) from error

            wrapper = _wrap_response(response, prompt_name)
            return cast(ProviderCompletionResponse, wrapper)

        def _select_choice(response: object) -> ProviderChoice:
            wrapper = cast(_AnthropicResponseWrapper, response)
            try:
                return cast(ProviderChoice, wrapper.choices[0])
            except IndexError as error:  # pragma: no cover - defensive
                raise PromptEvaluationError(
                    "Provider response did not include any choices.",
                    prompt_name=prompt_name,
                    phase="response",
                ) from error

        return run_conversation(
            adapter_name="anthropic",
            prompt_name=prompt_name,
            rendered=rendered,
            initial_messages=[{"role": "system", "content": rendered.text}],
            parse_output=parse_output,
            bus=bus,
            session=session,
            tool_choice=self._tool_choice,
            response_format=None,
            require_structured_output_text=True,
            call_provider=_call_provider,
            select_choice=_select_choice,
            serialize_tool_message_fn=serialize_tool_message,
            format_publish_failures=format_publish_failures,
            parse_arguments=parse_tool_arguments,
            logger_override=logger,
        )


@dataclass(slots=True)
class _AnthropicFunctionCall:
    name: str
    arguments: str | None


@dataclass(slots=True)
class _AnthropicToolCall:
    id: str | None
    function: _AnthropicFunctionCall


@dataclass(slots=True)
class _AnthropicMessage:
    content: tuple[object, ...]
    tool_calls: tuple[_AnthropicToolCall, ...] | None


@dataclass(slots=True)
class _AnthropicChoice:
    message: _AnthropicMessage


@dataclass(slots=True)
class _AnthropicResponseWrapper:
    _response: object
    choices: tuple[_AnthropicChoice, ...]

    def model_dump(self) -> object:
        model_dump = getattr(self._response, "model_dump", None)
        if callable(model_dump):
            return model_dump()
        return {}


def _strip_markdown_headings(text: str) -> str:
    lines = str(text).splitlines()
    filtered = [line for line in lines if not line.lstrip().startswith("#")]
    if not filtered:
        return ""
    return "\n".join(filtered).strip()


def _placeholder_dataclass(instance: SupportsDataclass) -> SupportsDataclass:
    if not is_dataclass(instance) or isinstance(instance, type):
        return instance
    placeholder_values: dict[str, object] = {}
    for field in fields(instance):
        placeholder_values[field.name] = f"${{{field.name}}}"
    try:
        return cast(SupportsDataclass, replace(instance, **placeholder_values))
    except (TypeError, ValueError):
        return instance


def _wrap_response(response: object, prompt_name: str) -> _AnthropicResponseWrapper:
    provider_payload = _shared.extract_payload(response)
    content = getattr(response, "content", None)
    if not isinstance(content, Sequence) or isinstance(
        content, (str, bytes, bytearray)
    ):
        raise PromptEvaluationError(
            "Anthropic response did not include content blocks.",
            prompt_name=prompt_name,
            phase="response",
            provider_payload=provider_payload,
        )
    sequence_content: list[object] = list(cast(Sequence[object], content))
    tool_calls: list[_AnthropicToolCall] = []
    for index, block in enumerate(sequence_content):
        block_type = cast(str | None, getattr(block, "type", None))
        if block_type != "tool_use":
            continue
        name = cast(str | None, getattr(block, "name", None))
        if not isinstance(name, str):
            raise PromptEvaluationError(
                "Anthropic tool invocation missing name.",
                prompt_name=prompt_name,
                phase="response",
                provider_payload=provider_payload,
            )
        input_payload = getattr(block, "input", None)
        try:
            arguments = json.dumps(input_payload or {})
        except TypeError as error:
            raise PromptEvaluationError(
                "Anthropic tool invocation payload is not JSON serialisable.",
                prompt_name=prompt_name,
                phase="response",
                provider_payload=provider_payload,
            ) from error
        call_id = getattr(block, "id", None)
        if call_id is None:
            call_id = f"tool_use_{index}"
        elif not isinstance(call_id, str):
            call_id = str(call_id)
        function = _AnthropicFunctionCall(name=name, arguments=arguments)
        tool_calls.append(_AnthropicToolCall(id=call_id, function=function))

    message = _AnthropicMessage(
        content=tuple(sequence_content),
        tool_calls=tuple(tool_calls) if tool_calls else None,
    )
    return _AnthropicResponseWrapper(response, (_AnthropicChoice(message),))


def _convert_messages(
    messages: Sequence[Any],
    prompt_name: str,
    *,
    system_template_text: str | None = None,
) -> tuple[str | None, list[dict[str, object]]]:
    system_parts: list[str] = []
    anthropic_messages: list[dict[str, object]] = []
    for raw_message in messages:
        if not isinstance(raw_message, Mapping):
            raise PromptEvaluationError(
                "Message payload must be a mapping.",
                prompt_name=prompt_name,
                phase="request",
            )
        message = cast(Mapping[str, Any], raw_message)
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            if system_template_text is not None and not system_parts:
                base_system = system_template_text
            else:
                base_system = _normalise_system_content(content)
            if base_system:
                system_parts.append(base_system)
            instruction_blocks = _normalise_text_content(content)
            if instruction_blocks:
                anthropic_messages.append(
                    {"role": "user", "content": instruction_blocks}
                )
            continue
        if role == "assistant":
            tool_calls = message.get("tool_calls")
            anthropic_messages.append(
                {
                    "role": "assistant",
                    "content": _normalise_assistant_content(
                        content, tool_calls, prompt_name
                    ),
                }
            )
            continue
        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            if tool_call_id is None:
                raise PromptEvaluationError(
                    "Tool result message missing tool_call_id.",
                    prompt_name=prompt_name,
                    phase="request",
                )
            anthropic_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_call_id,
                            "content": content or "",
                        }
                    ],
                }
            )
            continue
        if role == "user":
            anthropic_messages.append(
                {"role": "user", "content": _normalise_text_content(content)}
            )
            continue
        raise PromptEvaluationError(
            f"Unsupported message role '{role}' for Anthropic adapter.",
            prompt_name=prompt_name,
            phase="request",
        )

    if system_parts:
        filtered_parts = [part for part in system_parts if part]
        system_payload = "\n\n".join(filtered_parts)
    else:
        system_payload = None
    return system_payload, anthropic_messages


def _normalise_system_content(content: object) -> str:
    if isinstance(content, str):
        return _strip_markdown_headings(content)
    if isinstance(content, Sequence) and not isinstance(
        content, (str, bytes, bytearray)
    ):
        parts: list[str] = []
        sequence_content = cast(Sequence[object], content)
        for part in sequence_content:
            block = _normalise_content_block(part)
            text_value = block.get("text")
            if isinstance(text_value, str):
                parts.append(_strip_markdown_headings(text_value))
        return "".join(parts)
    return str(content or "")


def _normalise_text_content(content: object) -> list[dict[str, object]]:
    if isinstance(content, str):
        return [{"type": "text", "text": _strip_markdown_headings(content)}]
    blocks = _normalise_content(content)
    if not blocks:
        return [{"type": "text", "text": ""}]
    normalised: list[dict[str, object]] = []
    for block in blocks:
        block_copy = dict(block)
        text_value = block_copy.get("text")
        if isinstance(text_value, str):
            block_copy["text"] = _strip_markdown_headings(text_value)
        normalised.append(block_copy)
    return normalised


def _normalise_assistant_content(
    content: object,
    tool_calls: object,
    prompt_name: str,
) -> list[dict[str, object]]:
    blocks = _normalise_content(content)
    validated_tool_calls: tuple[Mapping[str, Any], ...]
    if isinstance(tool_calls, Sequence):
        collected: list[Mapping[str, Any]] = []
        sequence_calls = cast(Sequence[Mapping[object, object]], tool_calls)
        for call_obj in sequence_calls:
            if not isinstance(call_obj, Mapping):
                raise PromptEvaluationError(
                    "Tool call payload must be a mapping.",
                    prompt_name=prompt_name,
                    phase="request",
                )
            str_mapping: dict[str, Any] = {}
            for key, value in call_obj.items():
                str_mapping[str(key)] = value
            collected.append(str_mapping)
        validated_tool_calls = tuple(collected)
    else:
        validated_tool_calls = ()

    for index, call in enumerate(validated_tool_calls):
        function = call.get("function")
        if not isinstance(function, Mapping):
            raise PromptEvaluationError(
                "Tool call payload missing function descriptor.",
                prompt_name=prompt_name,
                phase="request",
            )
        function_mapping = function
        name = function_mapping.get("name")
        if not isinstance(name, str):
            raise PromptEvaluationError(
                "Tool call payload missing function name.",
                prompt_name=prompt_name,
                phase="request",
            )
        arguments_raw = function_mapping.get("arguments")
        if arguments_raw is None:
            arguments_payload: Mapping[str, Any] = {}
        else:
            if not isinstance(arguments_raw, str):
                raise PromptEvaluationError(
                    "Tool call arguments must be a JSON string.",
                    prompt_name=prompt_name,
                    phase="request",
                )
            try:
                decoded = json.loads(arguments_raw)
            except json.JSONDecodeError as error:
                raise PromptEvaluationError(
                    "Failed to decode tool call arguments for Anthropic payload.",
                    prompt_name=prompt_name,
                    phase="request",
                ) from error
            if not isinstance(decoded, Mapping):
                raise PromptEvaluationError(
                    "Decoded tool call arguments must be an object.",
                    prompt_name=prompt_name,
                    phase="request",
                )
            arguments_payload = cast(Mapping[str, Any], decoded)
        call_id = call.get("id")
        if call_id is None:
            call_id = f"tool_call_{index}"
        elif not isinstance(call_id, str):
            call_id = str(call_id)
        blocks.append(
            {
                "type": "tool_use",
                "id": call_id,
                "name": name,
                "input": dict(arguments_payload),
            }
        )
    if not blocks:
        return [{"type": "text", "text": ""}]
    return blocks


def _normalise_content(content: object) -> list[dict[str, object]]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, Sequence) and not isinstance(
        content, (str, bytes, bytearray)
    ):
        sequence_content = cast(Sequence[object], content)
        return [_normalise_content_block(part) for part in sequence_content]
    return [{"type": "text", "text": str(content)}]


def _normalise_content_block(part: object) -> dict[str, object]:
    if isinstance(part, Mapping):
        block = {str(key): value for key, value in part.items()}
        text_value = block.get("text")
        if isinstance(text_value, str):
            block["text"] = _strip_markdown_headings(text_value)
        return block
    block_type = getattr(part, "type", None)
    block: dict[str, object] = {}
    if isinstance(block_type, str):
        block["type"] = block_type
        if block_type in {"text", "output_text"}:
            text_value = getattr(part, "text", "")
            block["text"] = _strip_markdown_headings(str(text_value))
        elif block_type == "tool_use":
            block["id"] = getattr(part, "id", None)
            block["name"] = getattr(part, "name", None)
            block["input"] = getattr(part, "input", None)
        elif block_type == "tool_result":
            block["tool_use_id"] = getattr(part, "tool_use_id", None)
            block["content"] = getattr(part, "content", None)
        else:
            for attribute in ("text", "id", "name", "input", "content"):
                value = getattr(part, attribute, None)
                if value is not None:
                    block[attribute] = value
        return block
    return {"type": "text", "text": str(part)}


def _convert_tools(
    tool_specs: Sequence[Any],
    prompt_name: str,
) -> list[dict[str, object]]:
    converted: list[dict[str, object]] = []
    for raw_spec in tool_specs:
        if not isinstance(raw_spec, Mapping):
            raise PromptEvaluationError(
                "Tool specification must be a mapping.",
                prompt_name=prompt_name,
                phase="request",
            )
        spec = cast(Mapping[str, Any], raw_spec)
        spec_mapping: dict[str, Any] = {}
        for key, value in spec.items():
            spec_mapping[str(key)] = value
        if spec_mapping.get("type") != "function":
            raise PromptEvaluationError(
                "Anthropic adapter only supports function tools.",
                prompt_name=prompt_name,
                phase="request",
            )
        function_payload = spec_mapping.get("function")
        if not isinstance(function_payload, Mapping):
            raise PromptEvaluationError(
                "Tool specification missing function payload.",
                prompt_name=prompt_name,
                phase="request",
            )
        function_mapping: dict[str, Any] = {}
        for key, value in function_payload.items():
            function_mapping[str(key)] = value
        name = function_mapping.get("name")
        description = function_mapping.get("description")
        parameters = function_mapping.get("parameters")
        if not isinstance(name, str):
            raise PromptEvaluationError(
                "Tool specification missing function name.",
                prompt_name=prompt_name,
                phase="request",
            )
        if not isinstance(description, str):
            raise PromptEvaluationError(
                "Tool specification missing function description.",
                prompt_name=prompt_name,
                phase="request",
            )
        if not isinstance(parameters, Mapping):
            raise PromptEvaluationError(
                "Tool specification parameters must be a mapping.",
                prompt_name=prompt_name,
                phase="request",
            )
        parameters_mapping: dict[str, Any] = {}
        for key, value in parameters.items():
            parameters_mapping[str(key)] = value
        converted.append(
            {
                "name": name,
                "description": description,
                "input_schema": parameters_mapping,
            }
        )
    return converted


def _convert_tool_choice(
    tool_choice: ToolChoice | None,
    prompt_name: str,
) -> Mapping[str, object] | None:
    if tool_choice is None:
        return None
    if tool_choice == "auto":
        return {"type": "auto"}
    if isinstance(tool_choice, Mapping):
        choice_mapping: dict[str, Any] = {}
        for key, value in tool_choice.items():
            choice_mapping[str(key)] = value
        choice_type = choice_mapping.get("type")
        if choice_type == "function":
            function_payload = choice_mapping.get("function")
            if isinstance(function_payload, Mapping):
                function_mapping: dict[str, Any] = {}
                for key, value in function_payload.items():
                    function_mapping[str(key)] = value
                name = function_mapping.get("name")
                if isinstance(name, str):
                    disable_parallel = function_mapping.get("disable_parallel_tool_use")
                    payload: dict[str, object] = {"type": "tool", "name": name}
                    if isinstance(disable_parallel, bool):
                        payload["disable_parallel_tool_use"] = disable_parallel
                    return payload
        if choice_type in {"auto", "none", "any", "tool"}:
            return cast(Mapping[str, object], choice_mapping)
    raise PromptEvaluationError(
        "Unsupported tool_choice directive for Anthropic adapter.",
        prompt_name=prompt_name,
        phase="request",
    )


__all__ = [
    "AnthropicAdapter",
    "AnthropicProtocol",
    "_convert_messages",
    "_convert_tool_choice",
    "_convert_tools",
    "_wrap_response",
    "create_anthropic_client",
]


message_text_content = _shared.message_text_content
extract_parsed_content = _shared.extract_parsed_content
parse_schema_constrained_payload = _shared.parse_schema_constrained_payload
