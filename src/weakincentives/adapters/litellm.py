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

"""Optional LiteLLM adapter utilities."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol, cast

from ..events import EventBus, PromptExecuted, ToolInvoked
from ..prompts._types import SupportsDataclass
from ..prompts.prompt import Prompt
from ..prompts.structured import OutputParseError
from ..prompts.structured import parse_output as parse_structured_output
from ..prompts.tool import Tool, ToolResult
from ..serde import parse, schema
from .core import PromptEvaluationError, PromptResponse

_ERROR_MESSAGE: Final[str] = (
    "LiteLLM support requires the optional 'litellm' dependency. "
    "Install it with `uv sync --extra litellm` or `pip install weakincentives[litellm]`."
)

try:  # pragma: no cover - optional dependency import for tooling
    import litellm as _optional_litellm  # type: ignore[import]
except ModuleNotFoundError:  # pragma: no cover - handled lazily in loader
    _optional_litellm = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - optional dependency for typing only
    import litellm

    _ = litellm.__name__


class _CompletionFunctionCall(Protocol):
    name: str
    arguments: str | None


class _ToolCall(Protocol):
    id: str
    function: _CompletionFunctionCall


class _Message(Protocol):
    content: str | None
    tool_calls: Sequence[_ToolCall] | None


class _CompletionChoice(Protocol):
    message: _Message


class _CompletionResponse(Protocol):
    choices: Sequence[_CompletionChoice]


class _CompletionCallable(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> _CompletionResponse: ...


class _LiteLLMModule(Protocol):
    def completion(self, *args: object, **kwargs: object) -> _CompletionResponse: ...


class _LiteLLMCompletionFactory(Protocol):
    def __call__(self, **kwargs: object) -> _CompletionCallable: ...


LiteLLMCompletion = _CompletionCallable


def _load_litellm_module() -> _LiteLLMModule:
    try:
        module = import_module("litellm")
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(_ERROR_MESSAGE) from exc
    return cast(_LiteLLMModule, module)


def create_litellm_completion(**kwargs: object) -> LiteLLMCompletion:
    """Return a LiteLLM completion callable, guarding the optional dependency."""

    module = _load_litellm_module()
    if not kwargs:
        return module.completion

    def _wrapped_completion(
        *args: object, **request_kwargs: object
    ) -> _CompletionResponse:
        merged: dict[str, object] = dict(kwargs)
        merged.update(request_kwargs)
        return module.completion(*args, **merged)

    return _wrapped_completion


ToolChoice = Literal["auto"] | Mapping[str, Any] | None


class LiteLLMAdapter:
    """Adapter that evaluates prompts via LiteLLM's completion helper."""

    def __init__(
        self,
        *,
        model: str,
        tool_choice: ToolChoice = "auto",
        completion: LiteLLMCompletion | None = None,
        completion_factory: _LiteLLMCompletionFactory | None = None,
        completion_kwargs: Mapping[str, object] | None = None,
    ) -> None:
        if completion is not None:
            if completion_factory is not None:
                raise ValueError(
                    "completion_factory cannot be provided when an explicit completion is supplied.",
                )
            if completion_kwargs:
                raise ValueError(
                    "completion_kwargs cannot be provided when an explicit completion is supplied.",
                )
        else:
            factory = completion_factory or create_litellm_completion
            completion = factory(**dict(completion_kwargs or {}))

        self._completion = completion
        self._model = model
        self._tool_choice: ToolChoice = tool_choice

    def evaluate[OutputT](
        self,
        prompt: Prompt[OutputT],
        *params: SupportsDataclass,
        parse_output: bool = True,
        bus: EventBus,
    ) -> PromptResponse[OutputT]:
        prompt_name = prompt.name or prompt.__class__.__name__
        rendered = prompt.render(*params)  # type: ignore[reportArgumentType]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": rendered.text},
        ]

        tools = list(rendered.tools)
        tool_specs = [_tool_to_litellm_spec(tool) for tool in tools]
        tool_registry = {tool.name: tool for tool in tools}
        tool_events: list[ToolInvoked] = []
        provider_payload: dict[str, Any] | None = None
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

            try:
                response = self._completion(**request_payload)
            except Exception as error:  # pragma: no cover - network/SDK failure
                raise PromptEvaluationError(
                    "LiteLLM request failed.",
                    prompt_name=prompt_name,
                    stage="request",
                ) from error

            provider_payload = _extract_payload(response)
            choice = _first_choice(response, prompt_name=prompt_name)
            message = choice.message
            tool_calls = list(message.tool_calls or [])

            if not tool_calls:
                final_text = message.content or ""
                output: OutputT | None = None
                text_value: str | None = final_text or None

                if (
                    parse_output
                    and rendered.output_type is not None
                    and rendered.output_container is not None
                ):
                    try:
                        output = parse_structured_output(final_text, rendered)
                    except OutputParseError as error:
                        raise PromptEvaluationError(
                            error.message,
                            prompt_name=prompt_name,
                            stage="response",
                            provider_payload=provider_payload,
                        ) from error
                    text_value = None

                response = PromptResponse(
                    prompt_name=prompt_name,
                    text=text_value,
                    output=output,
                    tool_results=tuple(tool_events),
                    provider_payload=provider_payload,
                )
                bus.publish(
                    PromptExecuted(
                        prompt_name=prompt_name,
                        adapter="litellm",
                        response=cast(PromptResponse[object], response),
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
                        stage="tool",
                        provider_payload=provider_payload,
                    )
                if tool.handler is None:
                    raise PromptEvaluationError(
                        f"Tool '{tool_name}' does not have a registered handler.",
                        prompt_name=prompt_name,
                        stage="tool",
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
                        stage="tool",
                        provider_payload=provider_payload,
                    ) from error

                try:
                    tool_result = tool.handler(tool_params)
                except Exception as error:  # pragma: no cover - handler bug
                    raise PromptEvaluationError(
                        f"Tool '{tool_name}' raised an exception.",
                        prompt_name=prompt_name,
                        stage="tool",
                        provider_payload=provider_payload,
                    ) from error

                invocation = ToolInvoked(
                    prompt_name=prompt_name,
                    adapter="litellm",
                    name=tool_name,
                    params=tool_params,
                    result=cast(ToolResult[object], tool_result),
                    call_id=getattr(tool_call, "id", None),
                )
                tool_events.append(invocation)
                bus.publish(invocation)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": getattr(tool_call, "id", None),
                        "content": tool_result.message,
                    }
                )

            if isinstance(next_tool_choice, Mapping):
                tool_choice_mapping = cast(Mapping[str, object], next_tool_choice)
                if tool_choice_mapping.get("type") == "function":
                    next_tool_choice = "auto"


def _tool_to_litellm_spec(tool: Tool[Any, Any]) -> dict[str, Any]:
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


def _first_choice(
    response: _CompletionResponse, *, prompt_name: str
) -> _CompletionChoice:
    try:
        return response.choices[0]
    except (AttributeError, IndexError) as error:  # pragma: no cover - defensive
        raise PromptEvaluationError(
            "Provider response did not include any choices.",
            prompt_name=prompt_name,
            stage="response",
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
            stage="tool",
            provider_payload=provider_payload,
        ) from error
    if not isinstance(parsed, Mapping):
        raise PromptEvaluationError(
            "Tool call arguments must be a JSON object.",
            prompt_name=prompt_name,
            stage="tool",
            provider_payload=provider_payload,
        )
    return dict(cast(Mapping[str, Any], parsed))


__all__ = ["LiteLLMAdapter", "LiteLLMCompletion", "create_litellm_completion"]
