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
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal, NoReturn, Protocol, TypeVar, cast
from uuid import uuid4

from ..deadlines import Deadline
from ..prompt._types import SupportsDataclass, SupportsToolResult
from ..prompt.prompt import Prompt, RenderedPrompt
from ..prompt.structured_output import (
    ARRAY_WRAPPER_KEY,
    OutputParseError,
    parse_dataclass_payload,
    parse_structured_output,
)
from ..prompt.tool import Tool, ToolContext, ToolResult
from ..runtime.events import (
    EventBus,
    HandlerFailure,
    PromptExecuted,
    PromptRendered,
    ToolInvoked,
)
from ..runtime.logging import StructuredLogger, StructuredLogPayload, get_logger
from ..runtime.session.dataclasses import is_dataclass_instance
from ..serde import parse, schema
from ..tools.errors import DeadlineExceededError, ToolValidationError
from ._names import LITELLM_ADAPTER_NAME, OPENAI_ADAPTER_NAME, AdapterName
from ._provider_protocols import (
    ProviderChoice,
    ProviderCompletionCallable,
    ProviderCompletionResponse,
    ProviderFunctionCall,
    ProviderMessage,
    ProviderToolCall,
)
from .core import (
    PROMPT_EVALUATION_PHASE_REQUEST,
    PROMPT_EVALUATION_PHASE_RESPONSE,
    PROMPT_EVALUATION_PHASE_TOOL,
    PromptEvaluationError,
    PromptEvaluationPhase,
    PromptResponse,
    SessionProtocol,
)

if TYPE_CHECKING:
    from ..adapters.core import ProviderAdapter


logger: StructuredLogger = get_logger(
    __name__, context={"component": "adapters.shared"}
)


@dataclass(slots=True)
class _RejectedToolParams:
    """Dataclass used when provider arguments fail validation."""

    raw_arguments: dict[str, Any]
    error: str


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


def deadline_provider_payload(deadline: Deadline | None) -> dict[str, Any] | None:
    """Return a provider payload snippet describing the active deadline."""

    if deadline is None:
        return None
    return {"deadline_expires_at": deadline.expires_at.isoformat()}


def _raise_tool_deadline_error(
    *, prompt_name: str, tool_name: str, deadline: Deadline
) -> NoReturn:
    raise PromptEvaluationError(
        f"Deadline expired before executing tool '{tool_name}'.",
        prompt_name=prompt_name,
        phase=PROMPT_EVALUATION_PHASE_TOOL,
        provider_payload=deadline_provider_payload(deadline),
    )


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


def tool_to_spec(tool: Tool[SupportsDataclass, SupportsToolResult]) -> dict[str, Any]:
    """Return a provider-agnostic tool specification payload."""

    parameters_schema = schema(tool.params_type, extra="forbid")
    _ = parameters_schema.pop("title", None)
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
            phase=PROMPT_EVALUATION_PHASE_RESPONSE,
        )
    sequence_choices = cast(Sequence[object], choices)
    try:
        return sequence_choices[0]
    except IndexError as error:  # pragma: no cover - defensive
        raise PromptEvaluationError(
            "Provider response did not include any choices.",
            prompt_name=prompt_name,
            phase=PROMPT_EVALUATION_PHASE_RESPONSE,
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
            phase=PROMPT_EVALUATION_PHASE_TOOL,
            provider_payload=provider_payload,
        ) from error
    if not isinstance(parsed, Mapping):
        raise PromptEvaluationError(
            "Tool call arguments must be a JSON object.",
            prompt_name=prompt_name,
            phase=PROMPT_EVALUATION_PHASE_TOOL,
            provider_payload=provider_payload,
        )
    parsed_mapping = cast(Mapping[Any, Any], parsed)
    arguments: dict[str, Any] = {}
    for key, value in parsed_mapping.items():
        if not isinstance(key, str):
            raise PromptEvaluationError(
                "Tool call arguments must use string keys.",
                prompt_name=prompt_name,
                phase=PROMPT_EVALUATION_PHASE_TOOL,
                provider_payload=provider_payload,
            )
        arguments[key] = value
    return arguments


def execute_tool_call(
    *,
    adapter_name: AdapterName,
    adapter: ProviderAdapter[Any],
    prompt: Prompt[Any],
    rendered_prompt: RenderedPrompt[Any] | None,
    tool_call: ProviderToolCall,
    tool_registry: Mapping[str, Tool[SupportsDataclass, SupportsToolResult]],
    bus: EventBus,
    session: SessionProtocol,
    prompt_name: str,
    provider_payload: dict[str, Any] | None,
    deadline: Deadline | None,
    format_publish_failures: Callable[[Sequence[HandlerFailure]], str],
    parse_arguments: ToolArgumentsParser,
    logger_override: StructuredLogger | None = None,
) -> tuple[ToolInvoked, ToolResult[SupportsToolResult]]:
    """Execute a provider tool call and publish the resulting event."""

    function = tool_call.function
    tool_name = function.name
    tool = tool_registry.get(tool_name)
    if tool is None:
        raise PromptEvaluationError(
            f"Unknown tool '{tool_name}' requested by provider.",
            prompt_name=prompt_name,
            phase=PROMPT_EVALUATION_PHASE_TOOL,
            provider_payload=provider_payload,
        )
    handler = tool.handler
    if handler is None:
        raise PromptEvaluationError(
            f"Tool '{tool_name}' does not have a registered handler.",
            prompt_name=prompt_name,
            phase=PROMPT_EVALUATION_PHASE_TOOL,
            provider_payload=provider_payload,
        )

    arguments_mapping = parse_arguments(
        function.arguments,
        prompt_name=prompt_name,
        provider_payload=provider_payload,
    )

    call_id = getattr(tool_call, "id", None)
    log = (logger_override or logger).bind(
        context={
            "adapter": adapter_name,
            "prompt": prompt_name,
            "tool": tool_name,
            "call_id": call_id,
        }
    )
    tool_params: SupportsDataclass | None = None
    tool_result: ToolResult[SupportsToolResult]
    try:
        try:
            parsed_params = parse(tool.params_type, arguments_mapping, extra="forbid")
        except (TypeError, ValueError) as error:
            tool_params = cast(
                SupportsDataclass,
                _RejectedToolParams(
                    raw_arguments=dict(arguments_mapping),
                    error=str(error),
                ),
            )
            raise ToolValidationError(str(error)) from error

        tool_params = parsed_params
        if deadline is not None and deadline.remaining() <= timedelta(0):
            _raise_tool_deadline_error(
                prompt_name=prompt_name, tool_name=tool_name, deadline=deadline
            )
        context = ToolContext(
            prompt=prompt,
            rendered_prompt=rendered_prompt,
            adapter=adapter,
            session=session,
            event_bus=bus,
            deadline=deadline,
        )
        tool_result = handler(tool_params, context=context)
    except ToolValidationError as error:
        if tool_params is None:  # pragma: no cover - defensive
            tool_params = cast(
                SupportsDataclass,
                _RejectedToolParams(
                    raw_arguments=dict(arguments_mapping),
                    error=str(error),
                ),
            )
        log.warning(
            "Tool validation failed.",
            payload=StructuredLogPayload(
                event="tool_validation_failed",
                context={"reason": str(error)},
            ),
        )
        tool_result = ToolResult(
            message=f"Tool validation failed: {error}",
            value=None,
            success=False,
        )
    except PromptEvaluationError:
        raise
    except DeadlineExceededError as error:
        raise PromptEvaluationError(
            str(error) or f"Tool '{tool_name}' exceeded the deadline.",
            prompt_name=prompt_name,
            phase=PROMPT_EVALUATION_PHASE_TOOL,
            provider_payload=deadline_provider_payload(deadline),
        ) from error
    except Exception as error:  # propagate message via ToolResult
        log.exception(
            "Tool handler raised an unexpected exception.",
            payload=StructuredLogPayload(
                event="tool_handler_exception",
                context={"provider_payload": provider_payload},
            ),
        )
        tool_result = ToolResult(
            message=f"Tool '{tool_name}' execution failed: {error}",
            value=None,
            success=False,
        )
    else:
        log.info(
            "Tool handler completed.",
            payload=StructuredLogPayload(
                event="tool_handler_completed",
                context={
                    "success": tool_result.success,
                    "has_value": tool_result.value is not None,
                },
            ),
        )

    if tool_params is None:  # pragma: no cover - defensive
        raise RuntimeError("Tool parameters were not parsed.")

    snapshot = session.snapshot()
    session_id = getattr(session, "session_id", None)
    tool_value = tool_result.value
    dataclass_value: SupportsDataclass | None = None
    if is_dataclass_instance(tool_value):
        dataclass_value = cast(SupportsDataclass, tool_value)  # pyright: ignore[reportUnnecessaryCast]

    invocation = ToolInvoked(
        prompt_name=prompt_name,
        adapter=adapter_name,
        name=tool_name,
        params=tool_params,
        result=cast(ToolResult[object], tool_result),
        session_id=session_id,
        created_at=datetime.now(UTC),
        value=dataclass_value,
        call_id=call_id,
        event_id=uuid4(),
    )
    publish_result = bus.publish(invocation)
    if not publish_result.ok:
        session.rollback(snapshot)
        log.warning(
            "Session rollback triggered after publish failure.",
            payload=StructuredLogPayload(
                event="session_rollback_due_to_publish_failure",
                context={},
            ),
        )
        failure_handlers = [
            getattr(failure.handler, "__qualname__", repr(failure.handler))
            for failure in publish_result.errors
        ]
        log.error(
            "Tool event publish failed.",
            payload=StructuredLogPayload(
                event="tool_event_publish_failed",
                context={
                    "failure_count": len(publish_result.errors),
                    "failed_handlers": failure_handlers,
                },
            ),
        )
        tool_result.message = format_publish_failures(publish_result.errors)
    else:
        log.debug(
            "Tool event published.",
            payload=StructuredLogPayload(
                event="tool_event_published",
                context={"handler_count": publish_result.handled_count},
            ),
        )
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
    _ = base_schema.pop("title", None)

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

    return parse_dataclass_payload(
        dataclass_type,
        container,
        payload,
        allow_extra_keys=allow_extra_keys,
        object_error="Expected provider payload to be a JSON object.",
        array_error="Expected provider payload to be a JSON array.",
        array_item_error="Array item at index {index} is not an object.",
    )


def message_text_content(content: object) -> str:
    """Extract text content from provider message payloads."""

    if isinstance(content, str) or content is None:
        return content or ""
    if isinstance(content, Sequence) and not isinstance(
        content, (str, bytes, bytearray)
    ):
        sequence_content = cast(
            Sequence[object],
            content,
        )
        fragments = [_content_part_text(part) for part in sequence_content]
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
        sequence_content = cast(
            Sequence[object],
            content,
        )
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
    "LITELLM_ADAPTER_NAME",
    "OPENAI_ADAPTER_NAME",
    "AdapterName",
    "ChoiceSelector",
    "ConversationRequest",
    "ConversationRunner",
    "ProviderChoice",
    "ProviderCompletionCallable",
    "ProviderCompletionResponse",
    "ProviderFunctionCall",
    "ProviderMessage",
    "ProviderToolCall",
    "ToolArgumentsParser",
    "ToolChoice",
    "ToolMessageSerializer",
    "_content_part_text",
    "_parsed_payload_from_part",
    "build_json_schema_response_format",
    "deadline_provider_payload",
    "execute_tool_call",
    "extract_parsed_content",
    "extract_payload",
    "first_choice",
    "format_publish_failures",
    "message_text_content",
    "parse_schema_constrained_payload",
    "parse_tool_arguments",
    "run_conversation",
    "serialize_tool_call",
    "tool_to_spec",
]


OutputT = TypeVar("OutputT")


ConversationRequest = Callable[
    [
        list[dict[str, Any]],
        Sequence[Mapping[str, Any]],
        ToolChoice | None,
        Mapping[str, Any] | None,
    ],
    object,
]
"""Callable responsible for invoking the provider with assembled payloads."""


ChoiceSelector = Callable[[object], ProviderChoice]
"""Callable that extracts the relevant choice from a provider response."""


class ToolMessageSerializer(Protocol):
    def __call__(
        self,
        result: ToolResult[SupportsToolResult],
        *,
        payload: object | None = ...,
    ) -> object: ...


@dataclass(slots=True)
class ConversationRunner[OutputT]:
    """Coordinate a conversational exchange with a provider."""

    adapter_name: AdapterName
    adapter: ProviderAdapter[OutputT]
    prompt: Prompt[OutputT]
    prompt_name: str
    rendered: RenderedPrompt[OutputT]
    render_inputs: tuple[SupportsDataclass, ...]
    initial_messages: list[dict[str, Any]]
    parse_output: bool
    bus: EventBus
    session: SessionProtocol
    tool_choice: ToolChoice
    response_format: Mapping[str, Any] | None
    require_structured_output_text: bool
    call_provider: ConversationRequest
    select_choice: ChoiceSelector
    serialize_tool_message_fn: ToolMessageSerializer
    format_publish_failures: Callable[[Sequence[HandlerFailure]], str] = (
        format_publish_failures
    )
    parse_arguments: ToolArgumentsParser = parse_tool_arguments
    logger_override: StructuredLogger | None = None
    deadline: Deadline | None = None
    _log: StructuredLogger = field(init=False)
    _messages: list[dict[str, Any]] = field(init=False)
    _tool_specs: list[dict[str, Any]] = field(init=False)
    _tool_registry: dict[str, Tool[SupportsDataclass, SupportsToolResult]] = field(
        init=False
    )
    _tool_events: list[ToolInvoked] = field(init=False)
    _tool_message_records: list[
        tuple[ToolResult[SupportsToolResult], dict[str, Any]]
    ] = field(init=False)
    _provider_payload: dict[str, Any] | None = field(init=False, default=None)
    _next_tool_choice: ToolChoice = field(init=False)
    _should_parse_structured_output: bool = field(init=False)

    def _raise_deadline_error(
        self, message: str, *, phase: PromptEvaluationPhase
    ) -> NoReturn:
        raise PromptEvaluationError(
            message,
            prompt_name=self.prompt_name,
            phase=phase,
            provider_payload=deadline_provider_payload(self.deadline),
        )

    def _ensure_deadline_remaining(
        self, message: str, *, phase: PromptEvaluationPhase
    ) -> None:
        if self.deadline is None:
            return
        if self.deadline.remaining() <= timedelta(0):
            self._raise_deadline_error(message, phase=phase)

    def run(self) -> PromptResponse[OutputT]:
        """Execute the conversation loop and return the final response."""

        self._prepare_payload()

        while True:
            self._ensure_deadline_remaining(
                "Deadline expired before provider request.",
                phase=PROMPT_EVALUATION_PHASE_REQUEST,
            )
            response = self.call_provider(
                self._messages,
                self._tool_specs,
                self._next_tool_choice if self._tool_specs else None,
                self.response_format,
            )

            self._provider_payload = extract_payload(response)
            choice = self.select_choice(response)
            message = getattr(choice, "message", None)
            if message is None:
                raise PromptEvaluationError(
                    "Provider response did not include a message payload.",
                    prompt_name=self.prompt_name,
                    phase=PROMPT_EVALUATION_PHASE_RESPONSE,
                    provider_payload=self._provider_payload,
                )

            tool_calls_sequence = getattr(message, "tool_calls", None)
            tool_calls = list(tool_calls_sequence or [])

            if not tool_calls:
                return self._finalize_response(message)

            self._handle_tool_calls(message, tool_calls)

    def _prepare_payload(self) -> None:
        """Initialize execution state prior to the provider loop."""

        self._messages = list(self.initial_messages)
        self._log = (self.logger_override or logger).bind(
            context={
                "adapter": self.adapter_name,
                "prompt": self.prompt_name,
            }
        )
        self._log.info(
            "Prompt execution started.",
            payload=StructuredLogPayload(
                event="prompt_execution_started",
                context={
                    "tool_count": len(self.rendered.tools),
                    "parse_output": self.parse_output,
                },
            ),
        )

        tools = list(self.rendered.tools)
        self._tool_specs = [tool_to_spec(tool) for tool in tools]
        self._tool_registry = {tool.name: tool for tool in tools}
        self._tool_events = []
        self._tool_message_records = []
        self._provider_payload = None
        self._next_tool_choice = self.tool_choice
        self._should_parse_structured_output = (
            self.parse_output
            and self.rendered.output_type is not None
            and self.rendered.container is not None
        )

        publish_result = self.bus.publish(
            PromptRendered(
                prompt_ns=self.prompt.ns,
                prompt_key=self.prompt.key,
                prompt_name=self.prompt.name,
                adapter=self.adapter_name,
                session_id=getattr(self.session, "session_id", None),
                render_inputs=self.render_inputs,
                rendered_prompt=self.rendered.text,
                created_at=datetime.now(UTC),
                event_id=uuid4(),
            )
        )
        if not publish_result.ok:
            failure_handlers = [
                getattr(failure.handler, "__qualname__", repr(failure.handler))
                for failure in publish_result.errors
            ]
            self._log.error(
                "Prompt rendered publish failed.",
                payload=StructuredLogPayload(
                    event="prompt_rendered_publish_failed",
                    context={
                        "failure_count": len(publish_result.errors),
                        "failed_handlers": failure_handlers,
                    },
                ),
            )
        else:
            self._log.debug(
                "Prompt rendered event published.",
                payload=StructuredLogPayload(
                    event="prompt_rendered_published",
                    context={"handler_count": publish_result.handled_count},
                ),
            )

    def _handle_tool_calls(
        self,
        message: object,
        tool_calls: Sequence[ProviderToolCall],
    ) -> None:
        """Execute provider tool calls and record emitted messages."""

        assistant_tool_calls = [serialize_tool_call(call) for call in tool_calls]
        self._messages.append(
            {
                "role": "assistant",
                "content": getattr(message, "content", None) or "",
                "tool_calls": assistant_tool_calls,
            }
        )

        self._log.debug(
            "Processing tool calls.",
            payload=StructuredLogPayload(
                event="prompt_tool_calls_detected",
                context={"count": len(tool_calls)},
            ),
        )

        for tool_call in tool_calls:
            tool_name = getattr(tool_call.function, "name", "tool")
            self._ensure_deadline_remaining(
                f"Deadline expired before executing tool '{tool_name}'.",
                phase=PROMPT_EVALUATION_PHASE_TOOL,
            )
            invocation, tool_result = execute_tool_call(
                adapter_name=self.adapter_name,
                adapter=self.adapter,
                prompt=self.prompt,
                rendered_prompt=self.rendered,
                tool_call=tool_call,
                tool_registry=self._tool_registry,
                bus=self.bus,
                session=self.session,
                prompt_name=self.prompt_name,
                provider_payload=self._provider_payload,
                deadline=self.deadline,
                format_publish_failures=self.format_publish_failures,
                parse_arguments=self.parse_arguments,
                logger_override=self.logger_override,
            )
            self._tool_events.append(invocation)

            tool_message = {
                "role": "tool",
                "tool_call_id": getattr(tool_call, "id", None),
                "content": self.serialize_tool_message_fn(tool_result),
            }
            self._messages.append(tool_message)
            self._tool_message_records.append((tool_result, tool_message))

        if isinstance(self._next_tool_choice, Mapping):
            tool_choice_mapping = cast(Mapping[str, object], self._next_tool_choice)
            if tool_choice_mapping.get("type") == "function":
                self._next_tool_choice = "auto"

    def _finalize_response(self, message: object) -> PromptResponse[OutputT]:
        """Assemble and publish the final prompt response."""

        self._ensure_deadline_remaining(
            "Deadline expired while finalizing provider response.",
            phase=PROMPT_EVALUATION_PHASE_RESPONSE,
        )
        final_text = message_text_content(getattr(message, "content", None))
        output: OutputT | None = None
        text_value: str | None = final_text or None

        if self._should_parse_structured_output:
            parsed_payload = extract_parsed_content(message)
            if parsed_payload is not None:
                try:
                    output = cast(
                        OutputT,
                        parse_schema_constrained_payload(parsed_payload, self.rendered),
                    )
                except (TypeError, ValueError) as error:
                    raise PromptEvaluationError(
                        str(error),
                        prompt_name=self.prompt_name,
                        phase=PROMPT_EVALUATION_PHASE_RESPONSE,
                        provider_payload=self._provider_payload,
                    ) from error
            else:
                if final_text or not self.require_structured_output_text:
                    try:
                        output = parse_structured_output(
                            final_text or "", self.rendered
                        )
                    except OutputParseError as error:
                        raise PromptEvaluationError(
                            error.message,
                            prompt_name=self.prompt_name,
                            phase=PROMPT_EVALUATION_PHASE_RESPONSE,
                            provider_payload=self._provider_payload,
                        ) from error
                else:
                    raise PromptEvaluationError(
                        "Provider response did not include structured output.",
                        prompt_name=self.prompt_name,
                        phase=PROMPT_EVALUATION_PHASE_RESPONSE,
                        provider_payload=self._provider_payload,
                    )
            if output is not None:
                text_value = None

        if (
            output is not None
            and self._tool_message_records
            and self._tool_message_records[-1][0].success
        ):
            last_result, last_message = self._tool_message_records[-1]
            last_message["content"] = self.serialize_tool_message_fn(
                last_result, payload=output
            )

        response_payload = PromptResponse(
            prompt_name=self.prompt_name,
            text=text_value,
            output=output,
            tool_results=tuple(self._tool_events),
            provider_payload=self._provider_payload,
        )
        prompt_value: SupportsDataclass | None = None
        if is_dataclass_instance(output):
            prompt_value = cast(SupportsDataclass, output)  # pyright: ignore[reportUnnecessaryCast]

        publish_result = self.bus.publish(
            PromptExecuted(
                prompt_name=self.prompt_name,
                adapter=self.adapter_name,
                result=cast(PromptResponse[object], response_payload),
                session_id=getattr(self.session, "session_id", None),
                created_at=datetime.now(UTC),
                value=prompt_value,
                event_id=uuid4(),
            )
        )
        if not publish_result.ok:
            failure_handlers = [
                getattr(failure.handler, "__qualname__", repr(failure.handler))
                for failure in publish_result.errors
            ]
            self._log.error(
                "Prompt execution publish failed.",
                payload=StructuredLogPayload(
                    event="prompt_execution_publish_failed",
                    context={
                        "failure_count": len(publish_result.errors),
                        "failed_handlers": failure_handlers,
                    },
                ),
            )
            publish_result.raise_if_errors()
        self._log.info(
            "Prompt execution completed.",
            payload=StructuredLogPayload(
                event="prompt_execution_succeeded",
                context={
                    "tool_count": len(self._tool_events),
                    "has_output": output is not None,
                    "text_length": len(text_value or "") if text_value else 0,
                    "structured_output": self._should_parse_structured_output,
                    "handler_count": publish_result.handled_count,
                },
            ),
        )
        return response_payload


def run_conversation[
    OutputT,
](
    *,
    adapter_name: AdapterName,
    adapter: ProviderAdapter[OutputT],
    prompt: Prompt[OutputT],
    prompt_name: str,
    rendered: RenderedPrompt[OutputT],
    render_inputs: tuple[SupportsDataclass, ...],
    initial_messages: list[dict[str, Any]],
    parse_output: bool,
    bus: EventBus,
    session: SessionProtocol,
    tool_choice: ToolChoice,
    response_format: Mapping[str, Any] | None,
    require_structured_output_text: bool,
    call_provider: ConversationRequest,
    select_choice: ChoiceSelector,
    serialize_tool_message_fn: ToolMessageSerializer,
    format_publish_failures: Callable[
        [Sequence[HandlerFailure]], str
    ] = format_publish_failures,
    parse_arguments: ToolArgumentsParser = parse_tool_arguments,
    logger_override: StructuredLogger | None = None,
    deadline: Deadline | None = None,
) -> PromptResponse[OutputT]:
    """Execute a conversational exchange with a provider and return the result."""

    effective_deadline = deadline or rendered.deadline
    rendered_with_deadline = rendered
    if effective_deadline is not None and rendered.deadline is not effective_deadline:
        rendered_with_deadline = replace(rendered, deadline=effective_deadline)

    runner = ConversationRunner[OutputT](
        adapter_name=adapter_name,
        adapter=adapter,
        prompt=prompt,
        prompt_name=prompt_name,
        rendered=rendered_with_deadline,
        render_inputs=render_inputs,
        initial_messages=initial_messages,
        parse_output=parse_output,
        bus=bus,
        session=session,
        tool_choice=tool_choice,
        response_format=response_format,
        require_structured_output_text=require_structured_output_text,
        call_provider=call_provider,
        select_choice=select_choice,
        serialize_tool_message_fn=serialize_tool_message_fn,
        format_publish_failures=format_publish_failures,
        parse_arguments=parse_arguments,
        logger_override=logger_override,
        deadline=effective_deadline,
    )
    return runner.run()
