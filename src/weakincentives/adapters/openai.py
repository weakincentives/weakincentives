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

import logging
from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import TYPE_CHECKING, Any, Final, Protocol, cast

from ..events import EventBus, PromptExecuted, ToolInvoked
from ..prompt._types import SupportsDataclass
from ..prompt.prompt import Prompt
from ..prompt.structured_output import OutputParseError, parse_structured_output
from ..prompt.tool import ToolResult
from ._tool_messages import serialize_tool_message
from .core import PromptEvaluationError, PromptResponse
from .shared import (
    ToolChoice,
    build_json_schema_response_format,
    execute_tool_call,
    extract_parsed_content,
    extract_payload,
    first_choice,
    format_publish_failures,
    message_text_content,
    parse_schema_constrained_payload,
    parse_tool_arguments,
    serialize_tool_call,
    tool_to_spec,
)

if TYPE_CHECKING:
    from ..session.session import Session

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
            )
        else:
            rendered = prompt.render(*params)
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
            response_format = build_json_schema_response_format(rendered, prompt_name)

        tools = list(rendered.tools)
        tool_specs = [tool_to_spec(tool) for tool in tools]
        tool_registry = {tool.name: tool for tool in tools}
        tool_events: list[ToolInvoked] = []
        tool_message_records: list[
            tuple[ToolResult[SupportsDataclass], dict[str, Any]]
        ] = []
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

            provider_payload = extract_payload(response)
            choice = cast(
                _CompletionChoice, first_choice(response, prompt_name=prompt_name)
            )
            message = choice.message
            tool_calls = list(message.tool_calls or [])

            if not tool_calls:
                final_text = message_text_content(message.content)
                output: OutputT | None = None
                text_value: str | None = final_text or None

                if should_parse_structured_output:
                    parsed_payload = extract_parsed_content(message)
                    if parsed_payload is not None:
                        try:
                            output = cast(
                                OutputT,
                                parse_schema_constrained_payload(
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

                if (
                    output is not None
                    and tool_message_records
                    and tool_message_records[-1][0].success
                ):
                    last_result, last_message = tool_message_records[-1]
                    last_message["content"] = serialize_tool_message(
                        last_result, payload=output
                    )

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

            assistant_tool_calls = [serialize_tool_call(call) for call in tool_calls]
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": assistant_tool_calls,
                }
            )

            for tool_call in tool_calls:
                invocation, tool_result = execute_tool_call(
                    adapter_name="openai",
                    tool_call=tool_call,
                    tool_registry=tool_registry,
                    bus=bus,
                    session=session,
                    prompt_name=prompt_name,
                    provider_payload=provider_payload,
                    format_publish_failures=format_publish_failures,
                    parse_arguments=parse_tool_arguments,
                    logger_override=logger,
                )
                tool_events.append(invocation)

                tool_message = {
                    "role": "tool",
                    "tool_call_id": getattr(tool_call, "id", None),
                    "content": serialize_tool_message(tool_result),
                }
                messages.append(tool_message)
                tool_message_records.append((tool_result, tool_message))

            if isinstance(next_tool_choice, Mapping):
                tool_choice_mapping = cast(Mapping[str, object], next_tool_choice)
                if tool_choice_mapping.get("type") == "function":
                    # Relax forced single-function choice after the first call.
                    next_tool_choice = "auto"


__all__ = ["OpenAIAdapter", "OpenAIProtocol"]
