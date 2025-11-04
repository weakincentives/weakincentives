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

"""Optional Google Gemini adapter utilities."""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any, Final, Protocol, cast, no_type_check

from ..events import EventBus
from ..logging import StructuredLogger, get_logger
from ..prompt._types import SupportsDataclass
from ..prompt.prompt import Prompt
from ._provider_protocols import ProviderChoice
from ._tool_messages import serialize_tool_message
from .core import PromptEvaluationError, PromptResponse
from .shared import (
    ToolChoice,
    build_json_schema_response_format,
    format_publish_failures,
    parse_tool_arguments,
    run_conversation,
)

if TYPE_CHECKING:
    from ..session.session import Session

_ERROR_MESSAGE: Final[str] = (
    "Google Gemini support requires the optional 'google-genai' dependency. "
    "Install it with `uv sync --extra google-genai` or "
    "`pip install weakincentives[google-genai]`."
)


class _GeminiModelsAPI(Protocol):
    def generate_content(self, *args: object, **kwargs: object) -> object: ...


class _GeminiClientProtocol(Protocol):
    """Structural type for the Google GenAI client."""

    models: _GeminiModelsAPI


class _GeminiClientFactory(Protocol):
    def __call__(self, **kwargs: object) -> _GeminiClientProtocol: ...


GeminiClient = _GeminiClientProtocol


class _GeminiModule(Protocol):
    Client: _GeminiClientFactory


def _load_gemini_module() -> _GeminiModule:
    try:
        module = import_module("google.genai")
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(_ERROR_MESSAGE) from exc
    return cast(_GeminiModule, module)


def create_gemini_client(**kwargs: object) -> _GeminiClientProtocol:
    """Create a Google GenAI client, guarding the optional dependency."""

    module = _load_gemini_module()
    return module.Client(**kwargs)


logger: StructuredLogger = get_logger(__name__, context={"component": "adapter.google"})


@dataclass(slots=True)
class _GeminiFunction:
    name: str | None
    arguments: str | None


@dataclass(slots=True)
class _GeminiToolCall:
    id: str | None
    function: _GeminiFunction


@dataclass(slots=True)
class _GeminiMessage:
    content: str | None
    tool_calls: tuple[_GeminiToolCall, ...]
    parsed: object | None = None


@dataclass(slots=True)
class _GeminiChoice:
    message: _GeminiMessage


def _json_or_str(payload: object) -> object:
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError:  # pragma: no cover - defensive
            return payload
    return payload


@no_type_check
def _convert_messages(
    messages: Sequence[Mapping[str, Any]],
    tool_specs: Sequence[Mapping[str, Any]],
) -> list[dict[str, object]]:  # pragma: no cover - dynamic serialization helper
    contents: list[dict[str, object]] = []
    tool_name_by_id: dict[str, str] = {}
    pending_names: deque[str] = deque()
    available_tool_names: list[str] = []

    for spec in tool_specs:
        if spec.get("type") != "function":
            continue
        function_payload = spec.get("function")
        if isinstance(function_payload, Mapping):
            name = function_payload.get("name")
            if isinstance(name, str):
                available_tool_names.append(name)

    for message in messages:
        role = message.get("role")
        if role == "system":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                contents.append({"role": "user", "parts": [{"text": content}]})
            continue

        if role == "assistant":
            parts: list[dict[str, object]] = []
            content = message.get("content")
            if isinstance(content, str) and content:
                parts.append({"text": content})
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, Sequence):
                for item in tool_calls:
                    if not isinstance(item, Mapping):
                        continue
                    function_payload = item.get("function")
                    if not isinstance(function_payload, Mapping):
                        continue
                    name = function_payload.get("name")
                    if isinstance(name, str):
                        call_id = item.get("id")
                        if isinstance(call_id, str):
                            tool_name_by_id[call_id] = name
                        else:
                            pending_names.append(name)
                    arguments_payload = function_payload.get("arguments")
                    parsed_arguments = _json_or_str(arguments_payload)
                    function_call: dict[str, object] = {}
                    if isinstance(name, str):
                        function_call["name"] = name
                    if isinstance(parsed_arguments, Mapping):
                        function_call["args"] = dict(parsed_arguments)
                    elif parsed_arguments is not None:
                        function_call["args"] = parsed_arguments
                    call_id = item.get("id")
                    if isinstance(call_id, str):
                        function_call["id"] = call_id
                    parts.append({"functionCall": function_call})
            if parts:
                contents.append({"role": "model", "parts": parts})
            continue

        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            name: str | None = None
            if isinstance(tool_call_id, str):
                name = tool_name_by_id.get(tool_call_id)
            if name is None and pending_names:
                name = pending_names.popleft()
            if name is None and len(available_tool_names) == 1:
                name = available_tool_names[0]

            raw_content = message.get("content")
            parsed_content = _json_or_str(raw_content)
            response_payload: dict[str, object]
            if isinstance(parsed_content, Mapping):
                response_payload = dict(parsed_content)
            else:
                response_payload = {"output": parsed_content}

            function_response: dict[str, object] = {
                "response": response_payload,
            }
            if isinstance(name, str):
                function_response["name"] = name
            if isinstance(tool_call_id, str):
                function_response["id"] = tool_call_id

            contents.append(
                {"role": "user", "parts": [{"functionResponse": function_response}]}
            )
            continue

        content = message.get("content")
        parts: list[dict[str, object]] = []
        if isinstance(content, str) and content:
            parts.append({"text": content})
        if parts:
            contents.append({"role": "user", "parts": parts})

    return contents


def _build_function_calling_config(tool_choice: ToolChoice) -> dict[str, object] | None:
    if tool_choice is None:
        return {"mode": "NONE"}
    if isinstance(tool_choice, Mapping):
        choice_type = tool_choice.get("type")
        if choice_type == "none":
            return {"mode": "NONE"}
        if choice_type == "function":
            function = tool_choice.get("function")
            if isinstance(function, Mapping):
                function_mapping = cast(Mapping[str, object], function)
                name_obj = function_mapping.get("name")
                if isinstance(name_obj, str):
                    return {
                        "mode": "ANY",
                        "allowed_function_names": [name_obj],
                    }
    return None


@no_type_check
def _build_tools(
    tool_specs: Sequence[Mapping[str, Any]],
) -> list[dict[str, object]]:  # pragma: no cover - dynamic serialization helper
    if not tool_specs:
        return []
    function_declarations: list[dict[str, object]] = []
    for spec in tool_specs:
        if spec.get("type") != "function":
            continue
        function_payload = spec.get("function")
        if not isinstance(function_payload, Mapping):
            continue
        name = function_payload.get("name")
        if not isinstance(name, str):
            continue
        declaration: dict[str, object] = {"name": name}
        description = function_payload.get("description")
        if isinstance(description, str):
            declaration["description"] = description
        parameters = function_payload.get("parameters")
        if isinstance(parameters, Mapping):
            declaration["parameters_json_schema"] = dict(parameters)
        function_declarations.append(declaration)
    if not function_declarations:
        return []
    return [{"function_declarations": function_declarations}]


@no_type_check
def _select_choice(
    response: object, *, prompt_name: str
) -> _GeminiChoice:  # pragma: no cover - SDK interoperability helper
    candidates = getattr(response, "candidates", None)
    if not isinstance(candidates, Sequence):
        raise PromptEvaluationError(
            "Provider response did not include any candidates.",
            prompt_name=prompt_name,
            phase="response",
        )
    try:
        candidate = candidates[0]
    except IndexError as error:  # pragma: no cover - defensive
        raise PromptEvaluationError(
            "Provider response did not include any candidates.",
            prompt_name=prompt_name,
            phase="response",
        ) from error

    content = getattr(candidate, "content", None)
    parts = getattr(content, "parts", None) if content is not None else None

    text_fragments: list[str] = []
    tool_calls: list[_GeminiToolCall] = []

    if isinstance(parts, Sequence):
        for part in parts:
            text_value = getattr(part, "text", None)
            if isinstance(text_value, str):
                text_fragments.append(text_value)
            function_call = getattr(part, "function_call", None)
            if function_call is None:
                function_call = getattr(part, "functionCall", None)
            if function_call is not None:
                name = getattr(function_call, "name", None)
                arguments = getattr(function_call, "args", None)
                if isinstance(arguments, Mapping):
                    arguments_str = json.dumps(arguments)
                else:
                    arguments_str = cast(str | None, arguments)
                call_id = getattr(function_call, "id", None)
                tool_calls.append(
                    _GeminiToolCall(
                        id=call_id,
                        function=_GeminiFunction(
                            name=name,
                            arguments=arguments_str,
                        ),
                    )
                )

    parsed_payload = getattr(response, "parsed", None)
    message = _GeminiMessage(
        content="".join(text_fragments) or None,
        tool_calls=tuple(tool_calls),
        parsed=parsed_payload,
    )
    return _GeminiChoice(message=message)


class GoogleGeminiAdapter:
    """Adapter that evaluates prompts against Google Gemini models."""

    def __init__(
        self,
        *,
        model: str,
        tool_choice: ToolChoice = "auto",
        client: _GeminiClientProtocol | None = None,
        client_factory: _GeminiClientFactory | None = None,
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
            factory = client_factory or create_gemini_client
            client = factory(**dict(client_kwargs or {}))

        self._client = client
        self._model = model
        self._tool_choice: ToolChoice = tool_choice

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

        response_format: dict[str, Any] | None = None
        should_parse_structured_output = (
            parse_output
            and rendered.output_type is not None
            and rendered.container is not None
        )
        if should_parse_structured_output:
            json_schema_format = cast(
                dict[str, Any],
                build_json_schema_response_format(rendered, prompt_name),
            )
            schema_payload = json_schema_format.get("json_schema", {})
            response_format = {
                "response_mime_type": "application/json",
                "response_json_schema": schema_payload.get("schema", {}),
            }

        def _call_provider(
            messages: list[dict[str, Any]],
            tool_specs: Sequence[Mapping[str, Any]],
            tool_choice_directive: ToolChoice | None,
            response_format_payload: Mapping[str, Any] | None,
        ) -> object:
            contents = _convert_messages(messages, tool_specs)
            config: dict[str, object] = {}

            tools_payload = _build_tools(tool_specs)
            if tools_payload:
                config["tools"] = tools_payload
                function_config = _build_function_calling_config(
                    tool_choice_directive if tool_specs else "auto"
                )
                if function_config:
                    config["tool_config"] = {
                        "function_calling_config": function_config,
                    }

            if response_format_payload:
                schema_payload = response_format_payload.get("response_json_schema")
                if schema_payload:
                    config["response_mime_type"] = response_format_payload.get(
                        "response_mime_type", "application/json"
                    )
                    config["response_json_schema"] = schema_payload

            request_payload: dict[str, object] = {
                "model": self._model,
                "contents": contents,
            }
            if config:
                request_payload["config"] = config

            try:
                return self._client.models.generate_content(**request_payload)
            except Exception as error:  # pragma: no cover - network/SDK failure
                raise PromptEvaluationError(
                    "Google Gemini request failed.",
                    prompt_name=prompt_name,
                    phase="request",
                ) from error

        def _select(response: object) -> ProviderChoice:
            return cast(
                ProviderChoice,
                _select_choice(response, prompt_name=prompt_name),
            )

        return run_conversation(
            adapter_name="google-gemini",
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
            select_choice=_select,
            serialize_tool_message_fn=serialize_tool_message,
            format_publish_failures=format_publish_failures,
            parse_arguments=parse_tool_arguments,
            logger_override=logger,
        )


__all__ = ["GeminiClient", "GoogleGeminiAdapter", "create_gemini_client"]
