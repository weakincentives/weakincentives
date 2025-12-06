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

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import timedelta
from http import HTTPStatus
from importlib import import_module
from typing import Any, Final, Protocol, TypeVar, cast, override

from ..budget import Budget, BudgetTracker
from ..deadlines import Deadline
from ..prompt._types import (
    SupportsDataclass,
    SupportsDataclassOrNone,
    SupportsToolResult,
)
from ..prompt._visibility import SectionVisibility
from ..prompt.errors import SectionPath
from ..prompt.prompt import Prompt
from ..prompt.rendering import RenderedPrompt
from ..prompt.tool import Tool
from ..runtime.events import EventBus
from ..runtime.logging import StructuredLogger, get_logger
from ..serde import schema
from . import shared as _shared
from ._provider_protocols import (
    ProviderChoice,
    ProviderChoiceData,
    ProviderFunctionCallData,
    ProviderMessageData,
    ProviderToolCallData,
)
from ._tool_messages import serialize_tool_message
from .config import AnthropicClientConfig, AnthropicModelConfig
from .core import (
    PROMPT_EVALUATION_PHASE_REQUEST,
    PROMPT_EVALUATION_PHASE_RESPONSE,
    PromptEvaluationError,
    PromptResponse,
    ProviderAdapter,
    SessionProtocol,
)
from .shared import (
    ANTHROPIC_ADAPTER_NAME,
    InnerLoopConfig,
    InnerLoopInputs,
    ThrottleError,
    ThrottleKind,
    ToolChoice,
    build_json_schema_response_format,
    deadline_provider_payload,
    format_publish_failures,
    parse_tool_arguments,
    run_inner_loop,
    throttle_details,
)

OutputT = TypeVar("OutputT")

_ERROR_MESSAGE: Final[str] = (
    "Anthropic support requires the optional 'anthropic' dependency. "
    "Install it with `uv sync --extra anthropic` or `pip install weakincentives[anthropic]`."
)

_STRUCTURED_OUTPUTS_BETA: Final[str] = "structured-outputs-2025-11-13"
"""Beta header value for native structured outputs."""

_DEFAULT_MODEL: Final[str] = "claude-opus-4-5-20250929"
"""Default model identifier (Claude Opus 4.5)."""

_ANTHROPIC_OVERLOADED_STATUS: Final[int] = 529
"""HTTP status code for Anthropic API overloaded."""


class _BetaMessagesAPI(Protocol):
    """Structural type for the Anthropic beta messages API."""

    def create(self, *args: object, **kwargs: object) -> object: ...


class _BetaAPI(Protocol):
    """Structural type for the Anthropic beta namespace."""

    @property
    def messages(self) -> _BetaMessagesAPI: ...


class _AnthropicProtocol(Protocol):
    """Structural type for the Anthropic client."""

    @property
    def beta(self) -> _BetaAPI: ...


class _AnthropicClientFactory(Protocol):
    def __call__(self, **kwargs: object) -> _AnthropicProtocol: ...


AnthropicProtocol = _AnthropicProtocol


class _AnthropicModule(Protocol):
    Anthropic: _AnthropicClientFactory


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


def _load_anthropic_module() -> _AnthropicModule:
    try:
        module = import_module("anthropic")
    except ModuleNotFoundError as exc:
        raise RuntimeError(_ERROR_MESSAGE) from exc
    return cast(_AnthropicModule, module)


def create_anthropic_client(**kwargs: object) -> _AnthropicProtocol:
    """Create an Anthropic client, raising a helpful error if the extra is missing."""

    anthropic_module = _load_anthropic_module()
    return anthropic_module.Anthropic(**kwargs)


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
    return None


def _error_payload(error: object) -> dict[str, Any] | None:
    payload_candidate = getattr(error, "response", None)
    if isinstance(payload_candidate, Mapping):
        payload_mapping = cast(Mapping[object, Any], payload_candidate)
        return {str(key): value for key, value in payload_mapping.items()}
    payload_candidate = getattr(error, "body", None)
    if isinstance(payload_candidate, Mapping):
        payload_mapping = cast(Mapping[object, Any], payload_candidate)
        return {str(key): value for key, value in payload_mapping.items()}
    return None


def _normalize_anthropic_throttle(
    error: Exception, *, prompt_name: str
) -> ThrottleError | None:
    """Detect and normalize Anthropic throttling errors."""

    message = str(error) or "Anthropic request failed."
    lower_message = message.lower()
    status_code = getattr(error, "status_code", None)
    class_name = error.__class__.__name__.lower()
    kind: ThrottleKind | None = None

    if status_code == HTTPStatus.TOO_MANY_REQUESTS or "rate" in lower_message:
        kind = "rate_limit"
    elif "overloaded" in lower_message or status_code == _ANTHROPIC_OVERLOADED_STATUS:
        kind = "rate_limit"  # API overloaded
    elif "timeout" in class_name:
        kind = "timeout"

    if kind is None:
        return None

    retry_after = _retry_after_from_error(error)
    return ThrottleError(
        message,
        prompt_name=prompt_name,
        phase=PROMPT_EVALUATION_PHASE_REQUEST,
        details=throttle_details(
            kind=kind,
            retry_after=retry_after,
            provider_payload=_error_payload(error),
        ),
    )


def tool_to_anthropic_spec(
    tool: Tool[SupportsDataclassOrNone, SupportsToolResult],
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Convert WINK tool to Anthropic tool specification."""

    if tool.params_type is type(None):
        input_schema: dict[str, Any] = {"type": "object", "properties": {}}
    else:
        input_schema = schema(tool.params_type, extra="forbid")
        _ = input_schema.pop("title", None)

    spec: dict[str, Any] = {
        "name": tool.name,
        "description": tool.description,
        "input_schema": input_schema,
    }
    if strict:
        spec["strict"] = True
    return spec


def _anthropic_tool_choice(
    tool_choice: ToolChoice,
) -> dict[str, Any] | None:
    """Convert WINK tool choice to Anthropic format."""

    if tool_choice is None:
        return None
    if tool_choice == "auto":
        return {"type": "auto"}
    # At this point tool_choice must be a Mapping
    mapping = tool_choice
    if mapping.get("type") == "function":
        function_payload = mapping.get("function")
        if isinstance(function_payload, Mapping):
            function_mapping = cast(Mapping[str, Any], function_payload)
            name = function_mapping.get("name")
            if isinstance(name, str):
                return {"type": "tool", "name": name}
        # Check for top-level name
        top_level_name = mapping.get("name")
        if isinstance(top_level_name, str):
            return {"type": "tool", "name": top_level_name}
    return {"type": "auto"}


def _normalize_messages_for_anthropic(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Extract system prompt and normalize messages for Anthropic API."""

    system_prompt = ""
    normalized: list[dict[str, Any]] = []

    for message in messages:
        role = message.get("role")
        content = message.get("content")

        if role == "system":
            system_prompt = content or ""
            continue

        if role == "tool":
            # Convert tool result message to Anthropic format
            normalized.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.get("tool_call_id"),
                            "content": content or "",
                        }
                    ],
                }
            )
            continue

        if role == "assistant" and message.get("tool_calls"):
            # Convert assistant message with tool calls
            content_blocks: list[dict[str, Any]] = []
            if content:
                content_blocks.append({"type": "text", "text": content})
            for tool_call in cast(Sequence[Mapping[str, Any]], message["tool_calls"]):
                func = tool_call.get("function", {})
                arguments_raw = func.get("arguments", "{}")
                if isinstance(arguments_raw, str):
                    try:
                        arguments = json.loads(arguments_raw)
                    except json.JSONDecodeError:
                        arguments = {}
                else:
                    arguments = arguments_raw
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": tool_call.get("id"),
                        "name": func.get("name"),
                        "input": arguments,
                    }
                )
            normalized.append({"role": "assistant", "content": content_blocks})
            continue

        if role in {"assistant", "user"}:
            normalized.append({"role": role, "content": content or ""})
            continue

        # Pass through unknown roles
        normalized.append(dict(message))

    return system_prompt, normalized


def _build_anthropic_output_format(
    rendered: RenderedPrompt[Any],
    prompt_name: str,
) -> dict[str, Any] | None:
    """Build Anthropic output_format from rendered prompt metadata."""

    response_format = build_json_schema_response_format(rendered, prompt_name)
    if response_format is None:
        return None

    json_schema = cast(Mapping[str, Any], response_format["json_schema"])
    return {
        "type": "json_schema",
        "schema": json_schema["schema"],
    }


def _extract_anthropic_content(response: object) -> tuple[str, list[Any]]:
    """Extract text content and tool use blocks from Anthropic response."""

    text_parts: list[str] = []
    tool_uses: list[Any] = []

    content = getattr(response, "content", None)
    if not isinstance(content, Sequence):
        return "", []

    for block in cast(Sequence[object], content):
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text = getattr(block, "text", None)
            if isinstance(text, str):
                text_parts.append(text)
        elif block_type == "tool_use":
            tool_uses.append(block)

    return "".join(text_parts), tool_uses


def _tool_calls_from_anthropic(
    tool_uses: Sequence[object],
) -> list[ProviderToolCallData]:
    """Convert Anthropic tool_use blocks to provider tool calls."""

    tool_calls: list[ProviderToolCallData] = []
    for tool_use in tool_uses:
        call_id = getattr(tool_use, "id", None)
        name = getattr(tool_use, "name", None)
        input_data = getattr(tool_use, "input", None)

        arguments: str | None = None
        if input_data is not None:
            if isinstance(input_data, str):
                arguments = input_data
            elif isinstance(input_data, Mapping):
                arguments = json.dumps(input_data)
            else:
                arguments = json.dumps(input_data)

        tool_calls.append(
            ProviderToolCallData(
                id=str(call_id) if call_id is not None else None,
                function=ProviderFunctionCallData(
                    name=name if isinstance(name, str) else "tool",
                    arguments=arguments,
                ),
            )
        )
    return tool_calls


def _choice_from_anthropic_response(
    response: object, *, prompt_name: str
) -> ProviderChoiceData:
    """Extract a ProviderChoice from an Anthropic response."""

    text_content, tool_uses = _extract_anthropic_content(response)
    tool_calls = _tool_calls_from_anthropic(tool_uses)

    # Check stop_reason for tool_use
    stop_reason = getattr(response, "stop_reason", None)
    if stop_reason == "tool_use" and not tool_calls:
        raise PromptEvaluationError(
            "Anthropic indicated tool_use but no tool calls found.",
            prompt_name=prompt_name,
            phase=PROMPT_EVALUATION_PHASE_RESPONSE,
        )

    # Build content list for message
    content_parts: list[dict[str, Any]] = []
    if text_content:
        content_parts.append({"type": "text", "text": text_content})

    message = ProviderMessageData(
        content=tuple(content_parts) if content_parts else (text_content,),
        tool_calls=tuple(tool_calls) if tool_calls else None,
        parsed=None,
    )
    return ProviderChoiceData(message=message)


def _convert_tool_specs(
    tool_specs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Convert provider-agnostic tool specs to Anthropic format."""

    anthropic_tools: list[dict[str, Any]] = []
    for spec in tool_specs:
        function_payload = spec.get("function", {})
        anthropic_spec: dict[str, Any] = {
            "name": function_payload.get("name"),
            "description": function_payload.get("description", ""),
            "input_schema": function_payload.get("parameters", {}),
        }
        anthropic_tools.append(anthropic_spec)
    return anthropic_tools


logger: StructuredLogger = get_logger(
    __name__, context={"component": "adapter.anthropic"}
)


class AnthropicAdapter(ProviderAdapter[Any]):
    """Adapter that evaluates prompts against Anthropic's Messages API.

    Args:
        model: Model identifier (e.g., "claude-opus-4-5-20250929").
        model_config: Typed configuration for model parameters like temperature,
            max_tokens, etc. When provided, these values are merged into each
            request payload.
        tool_choice: Tool selection directive. Defaults to "auto".
        use_native_structured_output: When True, uses Anthropic's beta structured
            outputs feature for JSON schema enforcement. Defaults to True.
        client: Pre-configured Anthropic client instance. Mutually exclusive with
            client_config.
        client_config: Typed configuration for client instantiation. Used when
            client is not provided.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        model: str = _DEFAULT_MODEL,
        model_config: AnthropicModelConfig | None = None,
        tool_choice: ToolChoice = "auto",
        use_native_structured_output: bool = True,
        client: _AnthropicProtocol | None = None,
        client_config: AnthropicClientConfig | None = None,
    ) -> None:
        super().__init__()
        if client is not None:
            if client_config is not None:
                raise ValueError(
                    "client_config cannot be provided when an explicit client is supplied.",
                )
        else:
            client_kwargs = client_config.to_client_kwargs() if client_config else {}
            client = create_anthropic_client(**client_kwargs)

        self._client = client
        self._model = model
        self._model_config = model_config
        self._tool_choice: ToolChoice = tool_choice
        self._use_native_structured_output = use_native_structured_output

    @override
    def evaluate(
        self,
        prompt: Prompt[OutputT],
        *,
        bus: EventBus,
        session: SessionProtocol,
        parse_output: bool = True,
        deadline: Deadline | None = None,
        visibility_overrides: Mapping[SectionPath, SectionVisibility] | None = None,
        budget: Budget | None = None,
        budget_tracker: BudgetTracker | None = None,
    ) -> PromptResponse[OutputT]:
        context = self._setup_evaluation(
            prompt,
            parse_output=parse_output,
            deadline=deadline,
            visibility_overrides=visibility_overrides,
        )

        # Create tracker if budget provided but tracker not supplied
        effective_tracker = budget_tracker
        if effective_tracker is None and budget is not None:
            effective_tracker = BudgetTracker(budget=budget)

        config = InnerLoopConfig(
            bus=bus,
            session=session,
            tool_choice=self._tool_choice,
            response_format=context.response_format,
            require_structured_output_text=not self._use_native_structured_output,
            call_provider=self._build_provider_invoker(context.prompt_name),
            select_choice=self._build_choice_selector(context.prompt_name),
            serialize_tool_message_fn=serialize_tool_message,
            parse_output=parse_output,
            format_publish_failures=format_publish_failures,
            parse_arguments=parse_tool_arguments,
            logger_override=self._conversation_logger(),
            deadline=deadline,
            budget_tracker=effective_tracker,
        )

        inputs = InnerLoopInputs[OutputT](
            adapter_name=ANTHROPIC_ADAPTER_NAME,
            adapter=cast("ProviderAdapter[OutputT]", self),
            prompt=prompt,
            prompt_name=context.prompt_name,
            rendered=context.rendered,
            render_inputs=context.render_inputs,
            initial_messages=[{"role": "system", "content": context.rendered.text}],
        )

        return run_inner_loop(inputs=inputs, config=config)

    def _setup_evaluation(
        self,
        prompt: Prompt[OutputT],
        *,
        parse_output: bool,
        deadline: Deadline | None,
        visibility_overrides: Mapping[SectionPath, SectionVisibility] | None = None,
    ) -> _EvaluationContext[OutputT]:
        prompt_name = prompt.name or prompt.template.__class__.__name__
        render_inputs = prompt.params
        self._ensure_deadline_not_expired(deadline, prompt_name)
        rendered = self._render_prompt(
            prompt,
            parse_output=parse_output,
            deadline=deadline,
            visibility_overrides=visibility_overrides,
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
        *,
        parse_output: bool,
        deadline: Deadline | None,
        visibility_overrides: Mapping[SectionPath, SectionVisibility] | None = None,
    ) -> RenderedPrompt[OutputT]:
        has_structured_output = prompt.structured_output is not None
        inject_instructions = (
            prompt.inject_output_instructions
            if prompt.inject_output_instructions is not None
            else prompt.template.inject_output_instructions
        )
        should_disable_instructions = (
            parse_output
            and has_structured_output
            and self._use_native_structured_output
            and inject_instructions
        )

        if should_disable_instructions:
            rendered = prompt.render(
                inject_output_instructions=False,
                visibility_overrides=visibility_overrides,
            )
        else:
            rendered = prompt.render(
                inject_output_instructions=inject_instructions,
                visibility_overrides=visibility_overrides,
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
        if should_parse_structured_output and self._use_native_structured_output:
            return _build_anthropic_output_format(rendered, prompt_name)
        return None

    def _build_request_payload(
        self,
        system_prompt: str,
        normalized_messages: list[dict[str, Any]],
        anthropic_tools: list[dict[str, Any]],
        tool_choice_directive: ToolChoice | None,
        response_format_payload: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        request_payload: dict[str, Any] = {
            "model": self._model,
            "messages": normalized_messages,
            "max_tokens": 8192,  # Anthropic requires max_tokens
        }

        if system_prompt:
            request_payload["system"] = system_prompt

        if self._model_config is not None:
            request_payload.update(self._model_config.to_request_params())

        if anthropic_tools:
            request_payload["tools"] = anthropic_tools
            anthropic_choice = _anthropic_tool_choice(tool_choice_directive)
            if anthropic_choice is not None:
                request_payload["tool_choice"] = anthropic_choice

        if response_format_payload is not None:
            request_payload["betas"] = [_STRUCTURED_OUTPUTS_BETA]
            request_payload["output_format"] = response_format_payload

        return request_payload

    def _build_provider_invoker(self, prompt_name: str) -> ProviderInvoker:
        def _call_provider(
            messages: list[dict[str, Any]],
            tool_specs: Sequence[Mapping[str, Any]],
            tool_choice_directive: ToolChoice | None,
            response_format_payload: Mapping[str, Any] | None,
        ) -> object:
            system_prompt, normalized_messages = _normalize_messages_for_anthropic(
                messages
            )

            anthropic_tools = _convert_tool_specs(tool_specs)
            request_payload = self._build_request_payload(
                system_prompt,
                normalized_messages,
                anthropic_tools,
                tool_choice_directive,
                response_format_payload,
            )

            try:
                return self._client.beta.messages.create(**request_payload)
            except Exception as error:  # pragma: no cover - network/SDK failure
                throttle_error = _normalize_anthropic_throttle(
                    error, prompt_name=prompt_name
                )
                if throttle_error is not None:
                    raise throttle_error from error
                raise PromptEvaluationError(
                    "Anthropic request failed.",
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
                ProviderChoice,
                _choice_from_anthropic_response(response, prompt_name=prompt_name),
            )

        return _select_choice

    @staticmethod
    def _conversation_logger() -> StructuredLogger:
        return logger


__all__ = [
    "AnthropicAdapter",
    "AnthropicClientConfig",
    "AnthropicModelConfig",
    "AnthropicProtocol",
    "create_anthropic_client",
    "extract_parsed_content",
    "message_text_content",
    "parse_schema_constrained_payload",
    "tool_to_anthropic_spec",
]


message_text_content = _shared.message_text_content
extract_parsed_content = _shared.extract_parsed_content
parse_schema_constrained_payload = _shared.parse_schema_constrained_payload
