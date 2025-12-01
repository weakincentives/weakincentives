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
import random
import re
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NoReturn,
    Protocol,
    TypeVar,
    cast,
)
from uuid import uuid4

from ..deadlines import Deadline
from ..prompt._types import SupportsDataclass, SupportsToolResult
from ..prompt.overrides import PromptOverridesStore
from ..prompt.prompt import Prompt, RenderedPrompt
from ..prompt.protocols import PromptProtocol, ProviderAdapterProtocol
from ..prompt.structured_output import (
    ARRAY_WRAPPER_KEY,
    OutputParseError,
    PayloadParsingConfig,
    parse_dataclass_payload,
    parse_structured_output,
)
from ..prompt.tool import NativeTool, Tool, ToolContext, ToolHandler, ToolResult
from ..prompt.tool_result import render_tool_payload
from ..runtime.events import (
    EventBus,
    HandlerFailure,
    PromptExecuted,
    PromptRendered,
    ToolInvoked,
)
from ..runtime.logging import StructuredLogger, get_logger
from ..runtime.session.dataclasses import is_dataclass_instance
from ..serde import parse, schema
from ..tools.errors import DeadlineExceededError, ToolValidationError
from ..tools.web_search import WebSearchTool
from ..types import JSONValue
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


_EXPORTED_ADAPTER_NAMES = (LITELLM_ADAPTER_NAME, OPENAI_ADAPTER_NAME)
_EXPORTED_PROVIDER_PROTOCOLS = (
    ProviderCompletionCallable,
    ProviderCompletionResponse,
    ProviderFunctionCall,
    ProviderMessage,
    ProviderToolCall,
)


OutputT = TypeVar("OutputT")


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


ThrottleKind = Literal["rate_limit", "quota_exhausted", "timeout", "unknown"]
"""Classification for throttling scenarios."""

_DEFAULT_MAX_ATTEMPTS = 5
_DEFAULT_BASE_DELAY = timedelta(milliseconds=500)
_DEFAULT_MAX_DELAY = timedelta(seconds=8)
_DEFAULT_MAX_TOTAL_DELAY = timedelta(seconds=30)


@dataclass(slots=True, frozen=True)
class ThrottlePolicy:
    """Configuration for throttle retry handling."""

    max_attempts: int = _DEFAULT_MAX_ATTEMPTS
    base_delay: timedelta = _DEFAULT_BASE_DELAY
    max_delay: timedelta = _DEFAULT_MAX_DELAY
    max_total_delay: timedelta = _DEFAULT_MAX_TOTAL_DELAY


def new_throttle_policy(
    *,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    base_delay: timedelta = _DEFAULT_BASE_DELAY,
    max_delay: timedelta = _DEFAULT_MAX_DELAY,
    max_total_delay: timedelta = _DEFAULT_MAX_TOTAL_DELAY,
) -> ThrottlePolicy:
    """Return a throttle policy instance with validation."""

    if max_attempts < 1:
        msg = "Throttle max_attempts must be at least 1."
        raise ValueError(msg)
    if base_delay <= timedelta(0):
        msg = "Throttle base_delay must be positive."
        raise ValueError(msg)
    if max_delay <= timedelta(0):
        msg = "Throttle max_delay must be positive."
        raise ValueError(msg)
    if max_total_delay <= timedelta(0):
        msg = "Throttle max_total_delay must be positive."
        raise ValueError(msg)
    return ThrottlePolicy(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        max_total_delay=max_total_delay,
    )


@dataclass(slots=True)
class AdapterRenderContext[OutputT]:
    """Rendering inputs and derived metadata for adapter evaluations."""

    prompt_name: str
    render_inputs: tuple[SupportsDataclass, ...]
    rendered: RenderedPrompt[OutputT]
    response_format: Mapping[str, Any] | None


@dataclass(frozen=True, slots=True)
class AdapterRenderOptions:
    """Configuration for rendering prompts ahead of provider evaluation."""

    parse_output: bool
    disable_output_instructions: bool
    enable_json_schema: bool
    deadline: Deadline | None
    overrides_store: PromptOverridesStore | None
    overrides_tag: str = "latest"


class ThrottleError(PromptEvaluationError):
    """Raised when a provider throttles a request."""

    def __init__(
        self,
        message: str,
        *,
        prompt_name: str,
        phase: PromptEvaluationPhase,
        kind: ThrottleKind,
        retry_after: timedelta | None = None,
        attempts: int = 1,
        retry_safe: bool = True,
        provider_payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            prompt_name=prompt_name,
            phase=phase,
            provider_payload=provider_payload,
        )
        self.kind: ThrottleKind = kind
        self.retry_after: timedelta | None = retry_after
        self.attempts: int = attempts
        self.retry_safe: bool = retry_safe


def _sleep_for(delay: timedelta) -> None:
    time.sleep(delay.total_seconds())


def _jittered_backoff(
    *,
    policy: ThrottlePolicy,
    attempt: int,
    retry_after: timedelta | None,
) -> timedelta:
    capped = min(policy.max_delay, policy.base_delay * 2 ** max(attempt - 1, 0))
    base = max(capped, retry_after or timedelta(0))
    if base <= timedelta(0):
        return policy.base_delay

    jitter_seconds = random.uniform(0, base.total_seconds())  # nosec B311
    delay = timedelta(seconds=jitter_seconds)
    if delay < policy.base_delay:
        delay = policy.base_delay
    if retry_after is not None and delay < retry_after:
        return retry_after
    return delay


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

    if isinstance(tool, WebSearchTool):
        payload: dict[str, Any] = {"type": tool.provider_type}
        web_search_payload: dict[str, object] = {}

        if tool.search_context_size is not None:
            web_search_payload["search_context_size"] = tool.search_context_size
        if tool.filters is not None:
            allowed = tool.filters.allowed_domains or ()
            if allowed:
                web_search_payload["filters"] = {"allowed_domains": list(allowed)}
        if tool.user_location is not None:
            location = {
                "city": tool.user_location.city,
                "country": tool.user_location.country,
                "region": tool.user_location.region,
                "timezone": tool.user_location.timezone,
                "type": tool.user_location.type,
            }
            location_payload = {k: v for k, v in location.items() if v is not None}
            if location_payload:
                web_search_payload["user_location"] = location_payload

        if web_search_payload:
            payload["web_search"] = web_search_payload

        return payload

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


@dataclass(slots=True)
class ToolExecutionOutcome:
    """Result of executing a tool handler."""

    tool: Tool[SupportsDataclass, SupportsToolResult]
    params: SupportsDataclass
    result: ToolResult[SupportsToolResult]
    call_id: str | None
    log: StructuredLogger


def _resolve_tool_and_handler(
    *,
    tool_call: ProviderToolCall,
    tool_registry: Mapping[str, Tool[SupportsDataclass, SupportsToolResult]],
    prompt_name: str,
    provider_payload: dict[str, Any] | None,
) -> tuple[
    Tool[SupportsDataclass, SupportsToolResult],
    ToolHandler[SupportsDataclass, SupportsToolResult],
]:
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

    return tool, handler


def _parse_tool_call_arguments(
    *,
    tool_call: ProviderToolCall,
    prompt_name: str,
    provider_payload: dict[str, Any] | None,
    parse_arguments: ToolArgumentsParser,
) -> dict[str, Any]:
    return parse_arguments(
        tool_call.function.arguments,
        prompt_name=prompt_name,
        provider_payload=provider_payload,
    )


def _build_tool_logger(
    *,
    adapter_name: AdapterName,
    prompt_name: str,
    tool_name: str,
    tool_call: ProviderToolCall,
    logger_override: StructuredLogger | None,
) -> tuple[str | None, StructuredLogger]:
    call_id = getattr(tool_call, "id", None)
    bound_log = (logger_override or logger).bind(
        adapter=adapter_name,
        prompt=prompt_name,
        tool=tool_name,
        call_id=call_id,
    )
    return call_id, bound_log


def _parse_tool_params(
    *,
    tool: Tool[SupportsDataclass, SupportsToolResult],
    arguments_mapping: Mapping[str, Any],
) -> SupportsDataclass:
    try:
        return parse(tool.params_type, arguments_mapping, extra="forbid")
    except (TypeError, ValueError) as error:
        raise ToolValidationError(str(error)) from error


def _rejected_params(
    *,
    arguments_mapping: Mapping[str, Any],
    error: Exception,
) -> _RejectedToolParams:
    return _RejectedToolParams(
        raw_arguments=dict(arguments_mapping),
        error=str(error),
    )


def _ensure_deadline_not_expired(
    *,
    deadline: Deadline | None,
    prompt_name: str,
    tool_name: str,
) -> None:
    if deadline is not None and deadline.remaining() <= timedelta(0):
        _raise_tool_deadline_error(
            prompt_name=prompt_name,
            tool_name=tool_name,
            deadline=deadline,
        )


def _invoke_tool_handler(
    *,
    handler: ToolHandler[SupportsDataclass, SupportsToolResult],
    tool_params: SupportsDataclass,
    context: ToolContext,
) -> ToolResult[SupportsToolResult]:
    return handler(tool_params, context=context)


def _log_tool_completion(
    log: StructuredLogger, tool_result: ToolResult[SupportsToolResult]
) -> None:
    log.info(
        "Tool handler completed.",
        event="tool_handler_completed",
        context={
            "success": tool_result.success,
            "has_value": tool_result.value is not None,
        },
    )


def _handle_tool_validation_error(
    *,
    log: StructuredLogger,
    error: Exception,
) -> ToolResult[SupportsToolResult]:
    log.warning(
        "Tool validation failed.",
        event="tool_validation_failed",
        context={"reason": str(error)},
    )
    return ToolResult(
        message=f"Tool validation failed: {error}",
        value=None,
        success=False,
    )


def _handle_tool_deadline_error(
    *,
    error: DeadlineExceededError,
    prompt_name: str,
    tool_name: str,
    deadline: Deadline | None,
) -> PromptEvaluationError:
    return PromptEvaluationError(
        str(error) or f"Tool '{tool_name}' exceeded the deadline.",
        prompt_name=prompt_name,
        phase=PROMPT_EVALUATION_PHASE_TOOL,
        provider_payload=deadline_provider_payload(deadline),
    )


def _handle_unexpected_tool_error(
    *,
    log: StructuredLogger,
    tool_name: str,
    provider_payload: dict[str, Any] | None,
    error: Exception,
) -> ToolResult[SupportsToolResult]:
    log.exception(
        "Tool handler raised an unexpected exception.",
        event="tool_handler_exception",
        context={"provider_payload": provider_payload},
    )
    return ToolResult(
        message=f"Tool '{tool_name}' execution failed: {error}",
        value=None,
        success=False,
    )


@contextmanager
def tool_execution(
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
    parse_arguments: ToolArgumentsParser,
    logger_override: StructuredLogger | None = None,
) -> Iterator[ToolExecutionOutcome]:
    """Context manager that executes a tool call and standardizes logging."""

    tool, handler = _resolve_tool_and_handler(
        tool_call=tool_call,
        tool_registry=tool_registry,
        prompt_name=prompt_name,
        provider_payload=provider_payload,
    )
    tool_name = tool.name
    arguments_mapping = _parse_tool_call_arguments(
        tool_call=tool_call,
        prompt_name=prompt_name,
        provider_payload=provider_payload,
        parse_arguments=parse_arguments,
    )

    call_id, log = _build_tool_logger(
        adapter_name=adapter_name,
        prompt_name=prompt_name,
        tool_name=tool_name,
        tool_call=tool_call,
        logger_override=logger_override,
    )

    tool_params: SupportsDataclass | None = None
    tool_result: ToolResult[SupportsToolResult]
    try:
        tool_params = _parse_tool_params(tool=tool, arguments_mapping=arguments_mapping)
        _ensure_deadline_not_expired(
            deadline=deadline,
            prompt_name=prompt_name,
            tool_name=tool_name,
        )
        context = ToolContext(
            prompt=cast(PromptProtocol[Any], prompt),
            rendered_prompt=rendered_prompt,
            adapter=cast(ProviderAdapterProtocol[Any], adapter),
            session=session,
            event_bus=bus,
            deadline=deadline,
        )
        tool_result = _invoke_tool_handler(
            handler=handler,
            tool_params=tool_params,
            context=context,
        )
    except ToolValidationError as error:
        if tool_params is None:
            tool_params = cast(
                SupportsDataclass,
                _rejected_params(arguments_mapping=arguments_mapping, error=error),
            )
        tool_result = _handle_tool_validation_error(log=log, error=error)
    except PromptEvaluationError:
        raise
    except DeadlineExceededError as error:
        raise _handle_tool_deadline_error(
            error=error,
            prompt_name=prompt_name,
            tool_name=tool_name,
            deadline=deadline,
        ) from error
    except Exception as error:  # propagate message via ToolResult
        tool_result = _handle_unexpected_tool_error(
            log=log,
            tool_name=tool_name,
            provider_payload=provider_payload,
            error=error,
        )
    else:
        _log_tool_completion(log, tool_result)

    if tool_params is None:  # pragma: no cover - defensive
        raise RuntimeError("Tool parameters were not parsed.")

    yield ToolExecutionOutcome(
        tool=tool,
        params=tool_params,
        result=tool_result,
        call_id=call_id,
        log=log,
    )


def _publish_tool_invocation(
    *,
    adapter_name: AdapterName,
    prompt_name: str,
    session: SessionProtocol,
    bus: EventBus,
    outcome: ToolExecutionOutcome,
    format_publish_failures: Callable[[Sequence[HandlerFailure]], str],
) -> ToolInvoked:
    snapshot = session.snapshot()
    session_id = getattr(session, "session_id", None)
    tool_value = outcome.result.value
    dataclass_value: SupportsDataclass | None = None
    if is_dataclass_instance(tool_value):
        dataclass_value = cast(SupportsDataclass, tool_value)  # pyright: ignore[reportUnnecessaryCast]

    rendered_output = outcome.result.render()

    invocation = ToolInvoked(
        prompt_name=prompt_name,
        adapter=adapter_name,
        name=outcome.tool.name,
        params=outcome.params,
        result=cast(ToolResult[object], outcome.result),
        session_id=session_id,
        created_at=datetime.now(UTC),
        value=dataclass_value,
        rendered_output=rendered_output,
        call_id=outcome.call_id,
        event_id=uuid4(),
    )
    publish_result = bus.publish(invocation)
    if not publish_result.ok:
        session.rollback(snapshot)
        outcome.log.warning(
            "Session rollback triggered after publish failure.",
            event="session_rollback_due_to_publish_failure",
        )
        failure_handlers = [
            getattr(failure.handler, "__qualname__", repr(failure.handler))
            for failure in publish_result.errors
        ]
        outcome.log.error(
            "Tool event publish failed.",
            event="tool_event_publish_failed",
            context={
                "failure_count": len(publish_result.errors),
                "failed_handlers": failure_handlers,
            },
        )
        outcome.result.message = format_publish_failures(publish_result.errors)
    else:
        outcome.log.debug(
            "Tool event published.",
            event="tool_event_published",
            context={"handler_count": publish_result.handled_count},
        )
    return invocation


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

    with tool_execution(
        adapter_name=adapter_name,
        adapter=adapter,
        prompt=prompt,
        rendered_prompt=rendered_prompt,
        tool_call=tool_call,
        tool_registry=tool_registry,
        bus=bus,
        session=session,
        prompt_name=prompt_name,
        provider_payload=provider_payload,
        deadline=deadline,
        parse_arguments=parse_arguments,
        logger_override=logger_override,
    ) as outcome:
        invocation = _publish_tool_invocation(
            adapter_name=adapter_name,
            prompt_name=prompt_name,
            session=session,
            bus=bus,
            outcome=outcome,
            format_publish_failures=format_publish_failures,
        )
    return invocation, outcome.result


def build_json_schema_response_format(
    rendered: RenderedPrompt[Any], prompt_name: str
) -> dict[str, JSONValue] | None:
    """Construct a JSON schema response format for structured outputs."""

    output_type = rendered.output_type
    container = rendered.container
    allow_extra_keys = bool(rendered.allow_extra_keys)

    if output_type is None or container is None:
        return None

    extra_mode: Literal["ignore", "forbid"] = "ignore" if allow_extra_keys else "forbid"
    base_schema = schema(output_type, extra=extra_mode)
    _ = base_schema.pop("title", None)

    if container == "array":
        schema_payload: dict[str, JSONValue] = {
            "type": "object",
            "properties": {
                ARRAY_WRAPPER_KEY: {
                    "type": "array",
                    "items": base_schema,
                }
            },
            "required": [ARRAY_WRAPPER_KEY],
        }
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
    payload: JSONValue, rendered: RenderedPrompt[Any]
) -> object:
    """Parse structured provider payloads constrained by prompt schema."""

    dataclass_type = rendered.output_type
    container = rendered.container
    allow_extra_keys = rendered.allow_extra_keys

    if dataclass_type is None or container is None:
        raise TypeError("Prompt does not declare structured output.")

    config = PayloadParsingConfig(
        container=container,
        allow_extra_keys=bool(allow_extra_keys),
        object_error="Expected provider payload to be a JSON object.",
        array_error="Expected provider payload to be a JSON array.",
        array_item_error="Array item at index {index} is not an object.",
    )
    return parse_dataclass_payload(dataclass_type, payload, config)


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


def call_provider_with_normalization(
    call_provider: Callable[[], object],
    *,
    prompt_name: str,
    normalize_throttle: Callable[[Exception], ThrottleError | None],
    provider_payload: Callable[[Exception], dict[str, Any] | None],
    request_error_message: str,
) -> object:
    """Invoke a provider callable and normalize errors into PromptEvaluationError."""

    try:
        return call_provider()
    except Exception as error:  # pragma: no cover - network/SDK failure
        throttle_error = normalize_throttle(error)
        if throttle_error is not None:
            raise throttle_error from error
        raise PromptEvaluationError(
            request_error_message,
            prompt_name=prompt_name,
            phase=PROMPT_EVALUATION_PHASE_REQUEST,
            provider_payload=provider_payload(error),
        ) from error


def prepare_adapter_conversation[
    OutputT,
](
    *,
    prompt: Prompt[OutputT],
    params: Sequence[SupportsDataclass],
    options: AdapterRenderOptions,
) -> AdapterRenderContext[OutputT]:
    """Render a prompt and compute adapter inputs shared across providers."""

    prompt_name = prompt.name or prompt.__class__.__name__
    render_inputs: tuple[SupportsDataclass, ...] = tuple(params)

    if options.deadline is not None and options.deadline.remaining() <= timedelta(0):
        raise PromptEvaluationError(
            "Deadline expired before evaluation started.",
            prompt_name=prompt_name,
            phase=PROMPT_EVALUATION_PHASE_REQUEST,
            provider_payload=deadline_provider_payload(options.deadline),
        )

    render_overrides_store = options.overrides_store
    render_tag = options.overrides_tag
    render_inject_output_instructions: bool | None = None
    if options.disable_output_instructions:
        render_inject_output_instructions = False

    rendered = prompt.render(
        *render_inputs,
        overrides_store=render_overrides_store,
        tag=render_tag,
        inject_output_instructions=render_inject_output_instructions,
    )
    if options.deadline is not None:
        rendered = replace(rendered, deadline=options.deadline)

    response_format: Mapping[str, Any] | None = None
    if (
        options.enable_json_schema
        and options.parse_output
        and rendered.output_type is not None
        and rendered.container is not None
    ):
        response_format = build_json_schema_response_format(rendered, prompt_name)

    return AdapterRenderContext(
        prompt_name=prompt_name,
        render_inputs=render_inputs,
        rendered=rendered,
        response_format=response_format,
    )


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


__all__ = (
    "LITELLM_ADAPTER_NAME",
    "OPENAI_ADAPTER_NAME",
    "AdapterName",
    "AdapterRenderContext",
    "AdapterRenderOptions",
    "ChoiceSelector",
    "ConversationConfig",
    "ConversationRequest",
    "ConversationRunner",
    "NativeToolCall",
    "ProviderChoice",
    "ProviderCompletionCallable",
    "ProviderCompletionResponse",
    "ProviderFunctionCall",
    "ProviderMessage",
    "ProviderToolCall",
    "ThrottleError",
    "ThrottleKind",
    "ThrottlePolicy",
    "ToolArgumentsParser",
    "ToolChoice",
    "ToolMessageSerializer",
    "_content_part_text",
    "_parsed_payload_from_part",
    "build_json_schema_response_format",
    "call_provider_with_normalization",
    "deadline_provider_payload",
    "execute_tool_call",
    "extract_parsed_content",
    "extract_payload",
    "first_choice",
    "format_publish_failures",
    "message_text_content",
    "new_throttle_policy",
    "parse_schema_constrained_payload",
    "parse_tool_arguments",
    "prepare_adapter_conversation",
    "run_conversation",
    "serialize_tool_call",
    "tool_execution",
    "tool_to_spec",
)


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


@dataclass(slots=True, frozen=True)
class NativeToolCall:
    """Provider-executed native tool invocation payload."""

    name: str
    arguments: Mapping[str, Any]
    call_id: str | None = None
    success: bool = True


class ToolMessageSerializer(Protocol):
    def __call__(
        self,
        result: ToolResult[SupportsToolResult],
        *,
        payload: object | None = ...,
    ) -> object: ...


@dataclass(slots=True)
class ConversationConfig:
    """Configuration and collaborators required to run a conversation."""

    bus: EventBus
    session: SessionProtocol
    tool_choice: ToolChoice
    response_format: Mapping[str, Any] | None
    require_structured_output_text: bool
    call_provider: ConversationRequest
    select_choice: ChoiceSelector
    serialize_tool_message_fn: ToolMessageSerializer
    parse_output: bool = True
    format_publish_failures: Callable[[Sequence[HandlerFailure]], str] = (
        format_publish_failures
    )
    parse_arguments: ToolArgumentsParser = parse_tool_arguments
    logger_override: StructuredLogger | None = None
    deadline: Deadline | None = None
    throttle_policy: ThrottlePolicy = field(default_factory=new_throttle_policy)
    native_tool_extractor: (
        Callable[[object, dict[str, Any] | None], Sequence[NativeToolCall]] | None
    ) = None

    def with_defaults(self, rendered: RenderedPrompt[object]) -> ConversationConfig:
        """Fill in optional settings using rendered prompt metadata."""

        return replace(self, deadline=self.deadline or rendered.deadline)


@dataclass(slots=True)
class ToolExecutor:
    """Handles execution of tool calls and event publishing."""

    adapter_name: AdapterName
    adapter: ProviderAdapter[Any]
    prompt: Prompt[Any]
    prompt_name: str
    rendered: RenderedPrompt[Any]
    bus: EventBus
    session: SessionProtocol
    tool_registry: Mapping[str, Tool[SupportsDataclass, SupportsToolResult]]
    serialize_tool_message_fn: ToolMessageSerializer
    format_publish_failures: Callable[[Sequence[HandlerFailure]], str]
    parse_arguments: ToolArgumentsParser
    logger_override: StructuredLogger | None = None
    deadline: Deadline | None = None
    _log: StructuredLogger = field(init=False)
    _tool_message_records: list[
        tuple[ToolResult[SupportsToolResult], dict[str, Any]]
    ] = field(init=False)

    def __post_init__(self) -> None:
        self._log = (self.logger_override or logger).bind(
            adapter=self.adapter_name,
            prompt=self.prompt_name,
        )
        self._tool_message_records = []

    def record_native_calls(
        self,
        calls: Sequence[NativeToolCall],
        provider_payload: dict[str, Any] | None,
    ) -> None:
        """Publish provider-managed native tool invocations as events."""

        for call in calls:
            tool = self.tool_registry.get(call.name)
            if tool is None:
                raise PromptEvaluationError(
                    f"Unknown tool '{call.name}' requested by provider.",
                    prompt_name=self.prompt_name,
                    phase=PROMPT_EVALUATION_PHASE_TOOL,
                    provider_payload=provider_payload,
                )
            if not isinstance(tool, NativeTool):
                raise PromptEvaluationError(
                    f"Tool '{call.name}' does not have a registered handler.",
                    prompt_name=self.prompt_name,
                    phase=PROMPT_EVALUATION_PHASE_TOOL,
                    provider_payload=provider_payload,
                )

            try:
                params = _parse_tool_params(tool=tool, arguments_mapping=call.arguments)
            except ToolValidationError as error:
                params = cast(
                    SupportsDataclass,
                    _rejected_params(
                        arguments_mapping=call.arguments,
                        error=error,
                    ),
                )
                result = _handle_tool_validation_error(log=self._log, error=error)
            else:
                value = cast(SupportsToolResult, params)
                result = ToolResult(
                    message=render_tool_payload(value),
                    value=value,
                    success=call.success,
                )

            outcome = ToolExecutionOutcome(
                tool=tool,
                params=params,
                result=result,
                call_id=call.call_id,
                log=self._log,
            )
            _ = _publish_tool_invocation(
                adapter_name=self.adapter_name,
                prompt_name=self.prompt_name,
                session=self.session,
                bus=self.bus,
                outcome=outcome,
                format_publish_failures=self.format_publish_failures,
            )

    def execute(
        self,
        tool_calls: Sequence[ProviderToolCall],
        provider_payload: dict[str, Any] | None,
    ) -> tuple[list[dict[str, Any]], ToolChoice]:
        """Execute tool calls and return resulting messages and next tool choice."""
        messages: list[dict[str, Any]] = []
        next_tool_choice: ToolChoice = "auto"

        self._log.debug(
            "Processing tool calls.",
            event="prompt_tool_calls_detected",
            context={"count": len(tool_calls)},
        )

        for tool_call in tool_calls:
            tool_name = getattr(tool_call.function, "name", "tool")
            if self.deadline is not None and self.deadline.remaining() <= timedelta(0):
                _raise_tool_deadline_error(
                    prompt_name=self.prompt_name,
                    tool_name=tool_name,
                    deadline=self.deadline,
                )
            with tool_execution(
                adapter_name=self.adapter_name,
                adapter=self.adapter,
                prompt=self.prompt,
                rendered_prompt=self.rendered,
                tool_call=tool_call,
                tool_registry=self.tool_registry,
                bus=self.bus,
                session=self.session,
                prompt_name=self.prompt_name,
                provider_payload=provider_payload,
                deadline=self.deadline,
                parse_arguments=self.parse_arguments,
                logger_override=self.logger_override,
            ) as outcome:
                _ = _publish_tool_invocation(
                    adapter_name=self.adapter_name,
                    prompt_name=self.prompt_name,
                    session=self.session,
                    bus=self.bus,
                    outcome=outcome,
                    format_publish_failures=self.format_publish_failures,
                )

            tool_message = {
                "role": "tool",
                "tool_call_id": getattr(tool_call, "id", None),
                "content": self.serialize_tool_message_fn(outcome.result),
            }
            messages.append(tool_message)
            self._tool_message_records.append((outcome.result, tool_message))

        return messages, next_tool_choice

    @property
    def tool_message_records(
        self,
    ) -> list[tuple[ToolResult[SupportsToolResult], dict[str, Any]]]:
        return self._tool_message_records


@dataclass(slots=True)
class ResponseParser[OutputT]:
    """Handles parsing of provider responses into structured output."""

    prompt_name: str
    rendered: RenderedPrompt[OutputT]
    parse_output: bool
    require_structured_output_text: bool
    _should_parse_structured_output: bool = field(init=False)

    def __post_init__(self) -> None:
        self._should_parse_structured_output = (
            self.parse_output
            and self.rendered.output_type is not None
            and self.rendered.container is not None
        )

    def parse(
        self, message: object, provider_payload: dict[str, Any] | None
    ) -> tuple[OutputT | None, str | None]:
        """Parse the provider message into output and text content."""
        final_text = message_text_content(getattr(message, "content", None))
        output: OutputT | None = None
        text_value: str | None = final_text or None

        if self._should_parse_structured_output:
            parsed_payload = extract_parsed_content(message)
            if parsed_payload is not None:
                try:
                    output = cast(
                        OutputT,
                        parse_schema_constrained_payload(
                            cast(JSONValue, parsed_payload), self.rendered
                        ),
                    )
                except (TypeError, ValueError) as error:
                    raise PromptEvaluationError(
                        str(error),
                        prompt_name=self.prompt_name,
                        phase=PROMPT_EVALUATION_PHASE_RESPONSE,
                        provider_payload=provider_payload,
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
                            provider_payload=provider_payload,
                        ) from error
                else:
                    raise PromptEvaluationError(
                        "Provider response did not include structured output.",
                        prompt_name=self.prompt_name,
                        phase=PROMPT_EVALUATION_PHASE_RESPONSE,
                        provider_payload=provider_payload,
                    )
            if output is not None:
                text_value = None

        return output, text_value

    @property
    def should_parse_structured_output(self) -> bool:
        return self._should_parse_structured_output


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
    throttle_policy: ThrottlePolicy = field(default_factory=new_throttle_policy)
    native_tool_extractor: (
        Callable[[object, dict[str, Any] | None], Sequence[NativeToolCall]] | None
    ) = None
    _log: StructuredLogger = field(init=False)
    _messages: list[dict[str, Any]] = field(init=False)
    _tool_specs: list[dict[str, Any]] = field(init=False)
    _provider_payload: dict[str, Any] | None = field(init=False, default=None)
    _next_tool_choice: ToolChoice = field(init=False)
    _tool_executor: ToolExecutor = field(init=False)
    _response_parser: ResponseParser[OutputT] = field(init=False)

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
            response = self._issue_provider_request()

            self._provider_payload = extract_payload(response)
            choice = self.select_choice(response)
            native_tool_calls: Sequence[NativeToolCall] = ()
            if self.native_tool_extractor is not None:
                native_tool_calls = self.native_tool_extractor(
                    response, self._provider_payload
                )
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

            if native_tool_calls:
                self._tool_executor.record_native_calls(
                    native_tool_calls, self._provider_payload
                )

            if not tool_calls:
                return self._finalize_response(message)

            self._handle_tool_calls(message, tool_calls)

    def _issue_provider_request(self) -> object:
        attempts = 0
        total_delay = timedelta(0)

        while True:
            attempts += 1
            self._ensure_deadline_remaining(
                "Deadline expired before provider request.",
                phase=PROMPT_EVALUATION_PHASE_REQUEST,
            )

            try:
                return self.call_provider(
                    self._messages,
                    self._tool_specs,
                    self._next_tool_choice if self._tool_specs else None,
                    self.response_format,
                )
            except ThrottleError as error:
                error.attempts = max(error.attempts, attempts)
                if not error.retry_safe:
                    raise

                if attempts >= self.throttle_policy.max_attempts:
                    raise ThrottleError(
                        "Throttle retry budget exhausted.",
                        prompt_name=self.prompt_name,
                        phase=PROMPT_EVALUATION_PHASE_REQUEST,
                        kind=error.kind,
                        retry_after=error.retry_after,
                        attempts=error.attempts,
                        retry_safe=False,
                        provider_payload=error.provider_payload,
                    ) from error

                delay = _jittered_backoff(
                    policy=self.throttle_policy,
                    attempt=attempts,
                    retry_after=error.retry_after,
                )

                if self.deadline is not None and self.deadline.remaining() <= delay:
                    raise ThrottleError(
                        "Deadline expired before retrying after throttling.",
                        prompt_name=self.prompt_name,
                        phase=PROMPT_EVALUATION_PHASE_REQUEST,
                        kind=error.kind,
                        retry_after=error.retry_after,
                        attempts=error.attempts,
                        retry_safe=False,
                        provider_payload=error.provider_payload,
                    ) from error

                total_delay += delay
                if total_delay > self.throttle_policy.max_total_delay:
                    raise ThrottleError(
                        "Throttle retry window exceeded configured budget.",
                        prompt_name=self.prompt_name,
                        phase=PROMPT_EVALUATION_PHASE_REQUEST,
                        kind=error.kind,
                        retry_after=error.retry_after,
                        attempts=error.attempts,
                        retry_safe=False,
                        provider_payload=error.provider_payload,
                    ) from error

                self._log.warning(
                    "Provider throttled request.",
                    event="prompt_throttled",
                    context={
                        "attempt": attempts,
                        "retry_after_seconds": error.retry_after.total_seconds()
                        if error.retry_after
                        else None,
                        "kind": error.kind,
                        "delay_seconds": delay.total_seconds(),
                    },
                )
                _sleep_for(delay)

    def _prepare_payload(self) -> None:
        """Initialize execution state prior to the provider loop."""

        self._messages = list(self.initial_messages)
        self._log = (self.logger_override or logger).bind(
            adapter=self.adapter_name,
            prompt=self.prompt_name,
        )
        self._log.info(
            "Prompt execution started.",
            event="prompt_execution_started",
            context={
                "tool_count": len(self.rendered.tools),
                "parse_output": self.parse_output,
            },
        )

        tools = list(self.rendered.tools)
        self._tool_specs = [tool_to_spec(tool) for tool in tools]
        tool_registry = {tool.name: tool for tool in tools}
        self._provider_payload = None
        self._next_tool_choice = self.tool_choice

        self._tool_executor = ToolExecutor(
            adapter_name=self.adapter_name,
            adapter=self.adapter,
            prompt=self.prompt,
            prompt_name=self.prompt_name,
            rendered=self.rendered,
            bus=self.bus,
            session=self.session,
            tool_registry=tool_registry,
            serialize_tool_message_fn=self.serialize_tool_message_fn,
            format_publish_failures=self.format_publish_failures,
            parse_arguments=self.parse_arguments,
            logger_override=self.logger_override,
            deadline=self.deadline,
        )
        self._response_parser = ResponseParser[OutputT](
            prompt_name=self.prompt_name,
            rendered=self.rendered,
            parse_output=self.parse_output,
            require_structured_output_text=self.require_structured_output_text,
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
                descriptor=self.rendered.descriptor,
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
                event="prompt_rendered_publish_failed",
                context={
                    "failure_count": len(publish_result.errors),
                    "failed_handlers": failure_handlers,
                },
            )
        else:
            self._log.debug(
                "Prompt rendered event published.",
                event="prompt_rendered_published",
                context={"handler_count": publish_result.handled_count},
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

        tool_messages, next_choice = self._tool_executor.execute(
            tool_calls, self._provider_payload
        )
        self._messages.extend(tool_messages)

        if isinstance(self._next_tool_choice, Mapping):
            tool_choice_mapping = cast(Mapping[str, object], self._next_tool_choice)
            if tool_choice_mapping.get("type") == "function":
                self._next_tool_choice = next_choice

    def _finalize_response(self, message: object) -> PromptResponse[OutputT]:
        """Assemble and publish the final prompt response."""

        self._ensure_deadline_remaining(
            "Deadline expired while finalizing provider response.",
            phase=PROMPT_EVALUATION_PHASE_RESPONSE,
        )

        output, text_value = self._response_parser.parse(
            message, self._provider_payload
        )
        tool_message_records = self._tool_executor.tool_message_records

        if (
            output is not None
            and tool_message_records
            and tool_message_records[-1][0].success
        ):
            last_result, last_message = tool_message_records[-1]
            serialized = self.serialize_tool_message_fn(last_result, payload=output)
            if "output" in last_message:
                last_message["output"] = serialized
            else:
                last_message["content"] = serialized

        response_payload = PromptResponse(
            prompt_name=self.prompt_name,
            text=text_value,
            output=output,
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
                event="prompt_execution_publish_failed",
                context={
                    "failure_count": len(publish_result.errors),
                    "failed_handlers": failure_handlers,
                },
            )
            publish_result.raise_if_errors()
        self._log.info(
            "Prompt execution completed.",
            event="prompt_execution_succeeded",
            context={
                "tool_count": len(tool_message_records),
                "has_output": output is not None,
                "text_length": len(text_value or "") if text_value else 0,
                "structured_output": self._response_parser.should_parse_structured_output,
                "handler_count": publish_result.handled_count,
            },
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
    config: ConversationConfig,
) -> PromptResponse[OutputT]:
    """Execute a conversational exchange with a provider and return the result."""

    normalized_config = config.with_defaults(rendered)
    rendered_with_deadline = rendered
    if normalized_config.deadline is not None and (
        rendered.deadline is not normalized_config.deadline
    ):
        rendered_with_deadline = replace(rendered, deadline=normalized_config.deadline)

    runner = ConversationRunner[OutputT](
        adapter_name=adapter_name,
        adapter=adapter,
        prompt=prompt,
        prompt_name=prompt_name,
        rendered=rendered_with_deadline,
        render_inputs=render_inputs,
        initial_messages=initial_messages,
        parse_output=normalized_config.parse_output,
        bus=normalized_config.bus,
        session=normalized_config.session,
        tool_choice=normalized_config.tool_choice,
        response_format=normalized_config.response_format,
        require_structured_output_text=normalized_config.require_structured_output_text,
        call_provider=normalized_config.call_provider,
        select_choice=normalized_config.select_choice,
        serialize_tool_message_fn=normalized_config.serialize_tool_message_fn,
        format_publish_failures=normalized_config.format_publish_failures,
        parse_arguments=normalized_config.parse_arguments,
        logger_override=normalized_config.logger_override,
        deadline=normalized_config.deadline,
        throttle_policy=normalized_config.throttle_policy,
        native_tool_extractor=normalized_config.native_tool_extractor,
    )
    return runner.run()
