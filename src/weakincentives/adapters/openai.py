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
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import timedelta
from http import HTTPStatus
from importlib import import_module
from typing import TYPE_CHECKING, Any, Final, Protocol, TypeVar, cast, override

from ..deadlines import Deadline
from ..prompt._types import SupportsDataclass
from ..prompt.prompt import Prompt
from ..prompt.rendering import RenderedPrompt
from ..runtime.events import EventBus
from ..runtime.logging import StructuredLogger, get_logger
from . import shared as _shared
from ._provider_protocols import ProviderChoice, ProviderCompletionResponse
from ._tool_messages import serialize_tool_message
from .core import (
    PROMPT_EVALUATION_PHASE_REQUEST,
    PROMPT_EVALUATION_PHASE_RESPONSE,
    PromptEvaluationError,
    PromptResponse,
    ProviderAdapter,
    SessionProtocol,
)
from .shared import (
    OPENAI_ADAPTER_NAME,
    ConversationConfig,
    ThrottleError,
    ThrottleKind,
    ToolChoice,
    build_json_schema_response_format,
    deadline_provider_payload,
    format_publish_failures,
    parse_tool_arguments,
    run_conversation,
)

OutputT = TypeVar("OutputT")

if TYPE_CHECKING:
    from ..prompt.overrides import PromptOverridesStore

_ERROR_MESSAGE: Final[str] = (
    "OpenAI support requires the optional 'openai' dependency. "
    "Install it with `uv sync --extra openai` or `pip install weakincentives[openai]`."
)


class _ResponsesAPI(Protocol):
    def create(self, *args: object, **kwargs: object) -> ProviderCompletionResponse: ...


class _OpenAIProtocol(Protocol):
    """Structural type for the OpenAI client."""

    responses: _ResponsesAPI


class _OpenAIClientFactory(Protocol):
    def __call__(self, **kwargs: object) -> _OpenAIProtocol: ...


OpenAIProtocol = _OpenAIProtocol


class _OpenAIModule(Protocol):
    OpenAI: _OpenAIClientFactory


@dataclass(slots=True)
class _ResponseFunctionCall:
    name: str
    arguments: str | None


@dataclass(slots=True)
class _ResponseToolCall:
    id: str | None
    function: _ResponseFunctionCall


@dataclass(slots=True)
class _ResponseChoice:
    message: object


@dataclass(slots=True)
class _ResponseMessage:
    content: object
    tool_calls: Sequence[_ResponseToolCall] | None
    parsed: object | None = None


@dataclass(slots=True)
class _EvaluationContext[OutputT]:
    prompt_name: str
    render_inputs: tuple[SupportsDataclass, ...]
    rendered: RenderedPrompt[OutputT]
    response_format: dict[str, Any] | None


ProviderInvoker = Callable[
    [
        list[dict[str, Any]],
        Sequence[Mapping[str, Any]],
        ToolChoice | None,
        Mapping[str, Any] | None,
    ],
    object,
]


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


def _coerce_retry_after(value: object) -> timedelta | None:
    if value is None:
        return None
    if isinstance(value, timedelta):
        return value if value > timedelta(0) else None
    if isinstance(value, (int, float)):
        seconds = float(value)
        return timedelta(seconds=seconds) if seconds > 0 else None
    if isinstance(value, str) and value.isdigit():
        seconds = float(value)
        return timedelta(seconds=seconds)
    return None


def _retry_after_from_headers(headers: Mapping[str, Any] | None) -> timedelta | None:
    if headers is None:
        return None
    normalized = {str(key).lower(): val for key, val in headers.items()}
    value = normalized.get("retry-after")
    return _coerce_retry_after(value)


def _retry_after_from_error(error: object) -> timedelta | None:
    direct = _coerce_retry_after(getattr(error, "retry_after", None))
    if direct is not None:
        return direct
    headers = getattr(error, "headers", None)
    retry_after = _retry_after_from_headers(
        cast(Mapping[str, object], headers) if isinstance(headers, Mapping) else None
    )
    if retry_after is not None:
        return retry_after
    response = cast(object | None, getattr(error, "response", None))
    if isinstance(response, Mapping):
        response_mapping = cast(Mapping[str, object], response)
        retry_after = _retry_after_from_headers(
            cast(Mapping[str, object], response_mapping.get("headers"))
            if isinstance(response_mapping.get("headers"), Mapping)
            else None
        )
        if retry_after is not None:
            return retry_after
        retry_after = _coerce_retry_after(response_mapping.get("retry_after"))
        if retry_after is not None:
            return retry_after
        response_headers_obj: object | None = response_mapping.get("headers")
    else:
        response_headers_obj = (
            getattr(response, "headers", None) if response is not None else None
        )
    return _retry_after_from_headers(
        cast(Mapping[str, object], response_headers_obj)
        if isinstance(response_headers_obj, Mapping)
        else None
    )


def _error_payload(error: object) -> dict[str, Any] | None:
    payload_candidate = getattr(error, "response", None)
    if isinstance(payload_candidate, Mapping):
        payload_mapping = cast(Mapping[object, Any], payload_candidate)
        return {str(key): value for key, value in payload_mapping.items()}
    payload_candidate = getattr(error, "json_body", None)
    if isinstance(payload_candidate, Mapping):
        payload_mapping = cast(Mapping[object, Any], payload_candidate)
        return {str(key): value for key, value in payload_mapping.items()}
    return None


def _normalize_tool_arguments(arguments: object) -> str | None:
    if arguments is None:
        return None
    if isinstance(arguments, str):
        return arguments
    try:
        return json.dumps(arguments)
    except TypeError:
        return str(arguments)


def _normalize_tool_call(call: object) -> _ResponseToolCall:
    call_id = getattr(call, "id", None)
    function_obj = getattr(call, "function", None)
    if isinstance(call, Mapping):
        mapping_call = cast(Mapping[str, object], call)
        call_id = mapping_call.get("id", call_id)
        function_obj = mapping_call.get("function", function_obj)

    function_name = getattr(function_obj, "name", None)
    arguments_obj = getattr(function_obj, "arguments", None)
    if isinstance(function_obj, Mapping):
        function_mapping = cast(Mapping[str, object], function_obj)
        function_name = function_mapping.get("name", function_name)
        arguments_obj = function_mapping.get("arguments", arguments_obj)

    return _ResponseToolCall(
        id=str(call_id) if call_id is not None else None,
        function=_ResponseFunctionCall(
            name=function_name if isinstance(function_name, str) else "tool",
            arguments=_normalize_tool_arguments(arguments_obj),
        ),
    )


def _tool_call_from_output(output: object) -> _ResponseToolCall | None:
    name = getattr(output, "name", None)
    arguments_obj = getattr(output, "arguments", None)
    output_type = getattr(output, "type", None)

    if not isinstance(name, str) or arguments_obj is None:
        return None
    if output_type not in {None, "function_call"}:
        return None

    return _ResponseToolCall(
        id=str(getattr(output, "call_id", None) or getattr(output, "id", "")) or None,
        function=_ResponseFunctionCall(
            name=name,
            arguments=_normalize_tool_arguments(arguments_obj),
        ),
    )


def _tool_calls_from_content(parts: Sequence[object]) -> list[_ResponseToolCall]:
    for part in parts:
        tool_calls_obj = getattr(part, "tool_calls", None)
        if tool_calls_obj is None and isinstance(part, Mapping):
            mapping_part = cast(Mapping[str, object], part)
            tool_calls_obj = mapping_part.get("tool_calls")
        if tool_calls_obj:
            tool_calls = cast(Sequence[object], tool_calls_obj)
            return [_normalize_tool_call(call) for call in tool_calls]
    return []


def _parsed_from_content(parts: Sequence[object]) -> object | None:
    for part in parts:
        parsed = getattr(part, "parsed", None)
        if parsed is None and isinstance(part, Mapping):
            mapping_part = cast(Mapping[str, object], part)
            parsed = mapping_part.get("parsed")
        if parsed is not None:
            return parsed
    return None


def _select_output_with_content(response: object, *, prompt_name: str) -> object:
    outputs = getattr(response, "output", None)
    if not isinstance(outputs, Sequence):
        raise PromptEvaluationError(
            "Provider response did not include any output.",
            prompt_name=prompt_name,
            phase=PROMPT_EVALUATION_PHASE_RESPONSE,
        )

    usable_outputs: list[object] = []
    for output in cast(Sequence[object], outputs):
        if getattr(output, "type", None) == "reasoning":
            continue
        if _tool_call_from_output(output) is not None:
            return output
        content = getattr(output, "content", None)
        if isinstance(content, Sequence) and content:
            return output
        usable_outputs.append(output)

    if usable_outputs:
        return usable_outputs[0]

    raise PromptEvaluationError(
        "Provider response did not include any content.",
        prompt_name=prompt_name,
        phase=PROMPT_EVALUATION_PHASE_RESPONSE,
    )


def _content_from_output(output: object, *, prompt_name: str) -> Sequence[object]:
    content = getattr(output, "content", None)
    if not isinstance(content, Sequence) or not content:
        raise PromptEvaluationError(
            "Provider response did not include any content.",
            prompt_name=prompt_name,
            phase=PROMPT_EVALUATION_PHASE_RESPONSE,
        )
    return cast(Sequence[object], content)


def _normalize_content_parts(parts: Sequence[object]) -> list[object]:
    normalized: list[object] = []
    for part in parts:
        if isinstance(part, Mapping):
            mapping_part = cast(Mapping[str, object], part)
            normalized.append(dict(mapping_part))
        else:
            normalized.append(part)
    return normalized


def _text_config_from_response_format(
    response_format: Mapping[str, Any], *, prompt_name: str
) -> dict[str, Any]:
    """Translate a legacy response format payload into Responses text config."""

    response_type = response_format.get("type")
    if response_type != "json_schema":
        raise PromptEvaluationError(
            "Unsupported response format for OpenAI Responses API.",
            prompt_name=prompt_name,
            phase=PROMPT_EVALUATION_PHASE_REQUEST,
            provider_payload={"response_format": response_format},
        )

    json_schema = response_format.get("json_schema")
    if not isinstance(json_schema, Mapping):
        raise PromptEvaluationError(
            "OpenAI response format must include a JSON schema payload.",
            prompt_name=prompt_name,
            phase=PROMPT_EVALUATION_PHASE_REQUEST,
            provider_payload={"response_format": response_format},
        )

    json_schema_mapping = cast(Mapping[str, Any], json_schema)
    schema_name_obj = json_schema_mapping.get("name")
    schema_payload_obj = json_schema_mapping.get("schema")
    if not isinstance(schema_name_obj, str) or not isinstance(
        schema_payload_obj, Mapping
    ):
        raise PromptEvaluationError(
            "OpenAI response format schema is invalid.",
            prompt_name=prompt_name,
            phase=PROMPT_EVALUATION_PHASE_REQUEST,
            provider_payload={"response_format": response_format},
        )

    schema_name = schema_name_obj
    schema_payload = cast(Mapping[str, Any], schema_payload_obj)
    text_format: dict[str, Any] = {
        "type": "json_schema",
        "name": schema_name,
        "schema": dict(schema_payload),
    }

    description = json_schema_mapping.get("description")
    if isinstance(description, str):
        text_format["description"] = description

    strict = json_schema_mapping.get("strict")
    if isinstance(strict, bool):
        text_format["strict"] = strict

    return {"format": text_format}


def _responses_tool_spec(
    spec: Mapping[str, Any], *, prompt_name: str
) -> dict[str, Any]:
    """Normalize a provider-agnostic tool spec for the Responses API."""

    if spec.get("type") != "function":
        raise PromptEvaluationError(
            "OpenAI Responses only supports function tools.",
            prompt_name=prompt_name,
            phase=PROMPT_EVALUATION_PHASE_REQUEST,
            provider_payload={"tools": [spec]},
        )

    function_payload = spec.get("function")
    if not isinstance(function_payload, Mapping):
        raise PromptEvaluationError(
            "OpenAI tool specification is missing a function payload.",
            prompt_name=prompt_name,
            phase=PROMPT_EVALUATION_PHASE_REQUEST,
            provider_payload={"tools": [spec]},
        )

    function_mapping = cast(Mapping[str, Any], function_payload)
    name_obj = function_mapping.get("name")
    if not isinstance(name_obj, str) or not name_obj.strip():
        raise PromptEvaluationError(
            "OpenAI tool specification is missing a function name.",
            prompt_name=prompt_name,
            phase=PROMPT_EVALUATION_PHASE_REQUEST,
            provider_payload={"tools": [spec]},
        )

    name = name_obj
    normalized: dict[str, Any] = {
        "type": "function",
        "name": name,
    }

    description = function_mapping.get("description")
    if isinstance(description, str) and description.strip():
        normalized["description"] = description

    parameters = function_mapping.get("parameters")
    if parameters is not None:
        normalized["parameters"] = parameters

    strict = function_mapping.get("strict")
    if isinstance(strict, bool):
        normalized["strict"] = strict

    return normalized


def _responses_tool_choice(
    tool_choice: ToolChoice | None, *, prompt_name: str
) -> ToolChoice | dict[str, str] | None:
    """Normalize tool choice for the Responses API."""

    if tool_choice is None or isinstance(tool_choice, str):
        return tool_choice

    tool_choice_items = cast(Iterable[tuple[str, Any]], tool_choice.items())
    tool_choice_mapping: dict[str, Any] = dict(tool_choice_items)
    tool_type_obj = tool_choice_mapping.get("type")
    tool_type = tool_type_obj if isinstance(tool_type_obj, str) else None
    if tool_type == "function":
        function_payload = tool_choice_mapping.get("function")
        name_val: str | None = None
        if isinstance(function_payload, Mapping):
            function_items = cast(Iterable[tuple[str, Any]], function_payload.items())
            function_mapping: dict[str, Any] = dict(function_items)
            function_name = function_mapping.get("name")
            if isinstance(function_name, str):
                name_val = function_name
        else:
            alt_name = tool_choice_mapping.get("name")
            if isinstance(alt_name, str):
                name_val = alt_name

        if name_val and name_val.strip():
            return {"type": "function", "name": name_val}
        raise PromptEvaluationError(
            "OpenAI tool choice is missing a function name.",
            prompt_name=prompt_name,
            phase=PROMPT_EVALUATION_PHASE_REQUEST,
            provider_payload={"tool_choice": tool_choice},
        )

    raise PromptEvaluationError(
        "OpenAI tool choice is not supported by the Responses API.",
        prompt_name=prompt_name,
        phase=PROMPT_EVALUATION_PHASE_REQUEST,
        provider_payload={"tool_choice": tool_choice},
    )


def _normalize_input_messages(
    messages: Sequence[Mapping[str, Any]], *, prompt_name: str
) -> list[dict[str, Any]]:
    """Strip unsupported fields from request messages for the Responses API."""

    normalized: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        tool_calls = message.get("tool_calls")

        if role == "assistant" and tool_calls:
            if content:
                normalized.append(
                    {"type": "message", "role": "assistant", "content": content}
                )
            for tool_call in cast(Sequence[Mapping[str, Any]], tool_calls):
                function_payload_obj = tool_call.get("function")
                function_payload = (
                    cast(Mapping[str, Any], function_payload_obj)
                    if isinstance(function_payload_obj, Mapping)
                    else None
                )
                name = function_payload.get("name") if function_payload else None
                arguments = (
                    function_payload.get("arguments") if function_payload else None
                )
                call_id = tool_call.get("id") or tool_call.get("call_id")
                if not isinstance(name, str) or not isinstance(arguments, str):
                    raise PromptEvaluationError(
                        "OpenAI tool call is missing name or arguments.",
                        prompt_name=prompt_name,
                        phase=PROMPT_EVALUATION_PHASE_REQUEST,
                        provider_payload={"tool_call": tool_call},
                    )
                normalized.append(
                    {
                        "type": "function_call",
                        "call_id": str(call_id or ""),
                        "name": name,
                        "arguments": arguments,
                    }
                )
            continue

        if role == "tool":
            # Mutate the original message so downstream updates (final output rewrites)
            # are visible to recorded payloads.
            mutable_message = cast(dict[str, Any], message)
            call_id = mutable_message.get("tool_call_id")
            if call_id is None:
                raise PromptEvaluationError(
                    "OpenAI tool message is missing tool_call_id.",
                    prompt_name=prompt_name,
                    phase=PROMPT_EVALUATION_PHASE_REQUEST,
                    provider_payload={"message": mutable_message},
                )
            mutable_message.pop("role", None)
            mutable_message.pop("tool_call_id", None)
            mutable_message.pop("content", None)
            mutable_message["type"] = "function_call_output"
            mutable_message["call_id"] = str(call_id)
            mutable_message["output"] = content or ""
            normalized.append(mutable_message)
            continue

        if role in {"assistant", "system", "developer", "user"}:
            mutable_message = cast(dict[str, Any], message)
            mutable_message["type"] = "message"
            normalized.append(
                {
                    "type": "message",
                    "role": role,
                    "content": content or "",
                }
            )
            continue

        normalized.append(dict(message))
    return normalized


def _choice_from_response(response: object, *, prompt_name: str) -> _ResponseChoice:
    output = _select_output_with_content(response, prompt_name=prompt_name)
    tool_call_output = _tool_call_from_output(output)
    if tool_call_output is not None:
        message = _ResponseMessage(
            content=(),
            tool_calls=(tool_call_output,),
            parsed=None,
        )
        return _ResponseChoice(message=message)

    content_parts = _content_from_output(output, prompt_name=prompt_name)
    tool_calls = _tool_calls_from_content(content_parts)
    parsed = _parsed_from_content(content_parts)
    message = _ResponseMessage(
        content=_normalize_content_parts(content_parts),
        tool_calls=tool_calls or None,
        parsed=parsed,
    )
    return _ResponseChoice(message=message)


def _normalize_openai_throttle(
    error: Exception, *, prompt_name: str
) -> ThrottleError | None:
    message = str(error) or "OpenAI request failed."
    lower_message = message.lower()
    code = getattr(error, "code", None)
    status_code = getattr(error, "status_code", None)
    class_name = error.__class__.__name__.lower()
    kind: ThrottleKind | None = None

    if "insufficient_quota" in lower_message or code == "insufficient_quota":
        kind = "quota_exhausted"
    elif (
        status_code == HTTPStatus.TOO_MANY_REQUESTS
        or "ratelimit" in class_name
        or code
        in {
            "rate_limit_exceeded",
            "rate_limit",
        }
    ):
        kind = "rate_limit"
    elif "timeout" in class_name:
        kind = "timeout"

    if kind is None:
        return None

    retry_after = _retry_after_from_error(error)
    return ThrottleError(
        message,
        prompt_name=prompt_name,
        phase=PROMPT_EVALUATION_PHASE_REQUEST,
        kind=kind,
        retry_after=retry_after,
        provider_payload=_error_payload(error),
    )


logger: StructuredLogger = get_logger(__name__, context={"component": "adapter.openai"})


class OpenAIAdapter(ProviderAdapter[Any]):
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

    @override
    def evaluate(
        self,
        prompt: Prompt[OutputT],
        *params: SupportsDataclass,
        parse_output: bool = True,
        bus: EventBus,
        session: SessionProtocol,
        deadline: Deadline | None = None,
        overrides_store: PromptOverridesStore | None = None,
        overrides_tag: str = "latest",
    ) -> PromptResponse[OutputT]:
        context = self._setup_evaluation(
            prompt,
            params,
            parse_output=parse_output,
            deadline=deadline,
            overrides_store=overrides_store,
            overrides_tag=overrides_tag,
        )

        conversation_config = ConversationConfig(
            bus=bus,
            session=session,
            tool_choice=self._tool_choice,
            response_format=context.response_format,
            require_structured_output_text=False,
            call_provider=self._build_provider_invoker(context.prompt_name),
            select_choice=self._build_choice_selector(context.prompt_name),
            serialize_tool_message_fn=serialize_tool_message,
            parse_output=parse_output,
            format_publish_failures=format_publish_failures,
            parse_arguments=parse_tool_arguments,
            logger_override=self._conversation_logger(),
            deadline=deadline,
        )

        return run_conversation(
            adapter_name=OPENAI_ADAPTER_NAME,
            adapter=cast("ProviderAdapter[OutputT]", self),
            prompt=prompt,
            prompt_name=context.prompt_name,
            rendered=context.rendered,
            render_inputs=context.render_inputs,
            initial_messages=[{"role": "system", "content": context.rendered.text}],
            config=conversation_config,
        )

    def _setup_evaluation(
        self,
        prompt: Prompt[OutputT],
        params: tuple[SupportsDataclass, ...],
        *,
        parse_output: bool,
        deadline: Deadline | None,
        overrides_store: PromptOverridesStore | None,
        overrides_tag: str,
    ) -> _EvaluationContext[OutputT]:
        prompt_name = prompt.name or prompt.__class__.__name__
        render_inputs: tuple[SupportsDataclass, ...] = tuple(params)
        self._ensure_deadline_not_expired(deadline, prompt_name)
        rendered = self._render_prompt(
            prompt,
            render_inputs,
            parse_output=parse_output,
            deadline=deadline,
            overrides_store=overrides_store,
            overrides_tag=overrides_tag,
        )
        response_format = self._build_response_format(
            rendered,
            parse_output=parse_output,
            prompt_name=prompt_name,
        )
        return _EvaluationContext(
            prompt_name=prompt_name,
            render_inputs=render_inputs,
            rendered=rendered,
            response_format=response_format,
        )

    @staticmethod
    def _ensure_deadline_not_expired(
        deadline: Deadline | None, prompt_name: str
    ) -> None:
        if deadline is not None and deadline.remaining() <= timedelta(0):
            raise PromptEvaluationError(
                "Deadline expired before evaluation started.",
                prompt_name=prompt_name,
                phase=PROMPT_EVALUATION_PHASE_REQUEST,
                provider_payload=deadline_provider_payload(deadline),
            )

    def _render_prompt(
        self,
        prompt: Prompt[OutputT],
        render_inputs: tuple[SupportsDataclass, ...],
        *,
        parse_output: bool,
        deadline: Deadline | None,
        overrides_store: PromptOverridesStore | None,
        overrides_tag: str,
    ) -> RenderedPrompt[OutputT]:
        has_structured_output = prompt.structured_output is not None
        should_disable_instructions = (
            parse_output
            and has_structured_output
            and self._use_native_response_format
            and getattr(prompt, "inject_output_instructions", False)
        )

        if should_disable_instructions:
            rendered = prompt.render(
                *render_inputs,
                overrides_store=overrides_store,
                tag=overrides_tag,
                inject_output_instructions=False,
            )
        else:
            rendered = prompt.render(
                *render_inputs,
                overrides_store=overrides_store,
                tag=overrides_tag,
            )
        if deadline is not None:
            rendered = replace(rendered, deadline=deadline)
        return rendered

    def _build_response_format(
        self,
        rendered: RenderedPrompt[Any],
        *,
        parse_output: bool,
        prompt_name: str,
    ) -> dict[str, Any] | None:
        should_parse_structured_output = (
            parse_output
            and rendered.output_type is not None
            and rendered.container is not None
        )
        if should_parse_structured_output and self._use_native_response_format:
            return build_json_schema_response_format(rendered, prompt_name)
        return None

    def _build_provider_invoker(self, prompt_name: str) -> ProviderInvoker:
        def _call_provider(
            messages: list[dict[str, Any]],
            tool_specs: Sequence[Mapping[str, Any]],
            tool_choice_directive: ToolChoice | None,
            response_format_payload: Mapping[str, Any] | None,
        ) -> object:
            request_payload: dict[str, Any] = {
                "model": self._model,
                "input": _normalize_input_messages(messages, prompt_name=prompt_name),
            }
            if tool_specs:
                request_payload["tools"] = [
                    _responses_tool_spec(spec, prompt_name=prompt_name)
                    for spec in tool_specs
                ]
                if tool_choice_directive is not None:
                    request_payload["tool_choice"] = _responses_tool_choice(
                        tool_choice_directive, prompt_name=prompt_name
                    )
            if response_format_payload is not None:
                request_payload["text"] = _text_config_from_response_format(
                    response_format_payload, prompt_name=prompt_name
                )

            try:
                return self._client.responses.create(**request_payload)
            except Exception as error:  # pragma: no cover - network/SDK failure
                throttle_error = _normalize_openai_throttle(
                    error, prompt_name=prompt_name
                )
                if throttle_error is not None:
                    raise throttle_error from error
                raise PromptEvaluationError(
                    "OpenAI request failed.",
                    prompt_name=prompt_name,
                    phase=PROMPT_EVALUATION_PHASE_REQUEST,
                    provider_payload=_error_payload(error),
                ) from error

        return _call_provider

    @staticmethod
    def _build_choice_selector(
        prompt_name: str,
    ) -> Callable[[object], ProviderChoice]:
        def _select_choice(response: object) -> ProviderChoice:
            return cast(
                ProviderChoice, _choice_from_response(response, prompt_name=prompt_name)
            )

        return _select_choice

    @staticmethod
    def _conversation_logger() -> StructuredLogger:
        return logger


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
