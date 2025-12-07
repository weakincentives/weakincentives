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
from typing import TYPE_CHECKING, Any, Literal, NoReturn, Protocol, TypeVar, cast
from uuid import uuid4

from ..budget import BudgetExceededError, BudgetTracker
from ..dataclasses import FrozenDataclass
from ..deadlines import Deadline
from ..prompt._types import (
    SupportsDataclass,
    SupportsDataclassOrNone,
    SupportsToolResult,
)
from ..prompt._visibility import SectionVisibility
from ..prompt.errors import SectionPath, VisibilityExpansionRequired
from ..prompt.prompt import Prompt, RenderedPrompt
from ..prompt.protocols import PromptProtocol, ProviderAdapterProtocol
from ..prompt.structured_output import (
    ARRAY_WRAPPER_KEY,
    OutputParseError,
    PayloadParsingConfig,
    parse_dataclass_payload,
    parse_structured_output,
)
from ..prompt.tool import Tool, ToolContext, ToolHandler, ToolResult
from ..runtime.events import (
    EventBus,
    HandlerFailure,
    PromptExecuted,
    PromptRendered,
    TokenUsage,
    ToolInvoked,
)
from ..runtime.logging import StructuredLogger, get_logger
from ..runtime.session.dataclasses import is_dataclass_instance
from ..serde import parse, schema
from ..tools.errors import DeadlineExceededError, ToolValidationError
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
    PROMPT_EVALUATION_PHASE_BUDGET,
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


@FrozenDataclass()
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
_EMPTY_TOOL_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}
_DEFAULT_MAX_DELAY = timedelta(seconds=8)
_DEFAULT_MAX_TOTAL_DELAY = timedelta(seconds=30)


@FrozenDataclass()
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


@FrozenDataclass()
class AdapterRenderContext[OutputT]:
    """Rendering inputs and derived metadata for adapter evaluations."""

    prompt_name: str
    render_inputs: tuple[SupportsDataclass, ...]
    rendered: RenderedPrompt[OutputT]
    response_format: Mapping[str, Any] | None


@FrozenDataclass()
class AdapterRenderOptions:
    """Configuration for rendering prompts ahead of provider evaluation."""

    enable_json_schema: bool
    deadline: Deadline | None
    visibility_overrides: Mapping[SectionPath, SectionVisibility] | None = None


@FrozenDataclass()
class ThrottleDetails:
    """Provider throttle metadata tracked alongside PromptEvaluationError details."""

    kind: ThrottleKind
    retry_after: timedelta | None = None
    attempts: int = 1
    retry_safe: bool = True
    provider_payload: dict[str, Any] | None = None


class ThrottleError(PromptEvaluationError):
    """Raised when a provider throttles a request."""

    def __init__(
        self,
        message: str,
        *,
        prompt_name: str,
        phase: PromptEvaluationPhase,
        details: ThrottleDetails,
    ) -> None:
        super().__init__(
            message,
            prompt_name=prompt_name,
            phase=phase,
            provider_payload=details.provider_payload,
        )
        self.details = details

    @property
    def kind(self) -> ThrottleKind:
        return self.details.kind

    @property
    def retry_after(self) -> timedelta | None:
        return self.details.retry_after

    @property
    def attempts(self) -> int:
        return self.details.attempts

    @property
    def retry_safe(self) -> bool:
        return self.details.retry_safe


def throttle_details(
    *,
    kind: ThrottleKind,
    retry_after: timedelta | None = None,
    attempts: int = 1,
    retry_safe: bool = True,
    provider_payload: dict[str, Any] | None = None,
) -> ThrottleDetails:
    """Convenience wrapper for constructing throttle detail payloads."""

    return ThrottleDetails(
        kind=kind,
        retry_after=retry_after,
        attempts=attempts,
        retry_safe=retry_safe,
        provider_payload=provider_payload,
    )


def _details_from_error(
    error: ThrottleError, *, attempts: int, retry_safe: bool
) -> ThrottleDetails:
    return throttle_details(
        kind=error.kind,
        retry_after=error.retry_after,
        attempts=attempts,
        retry_safe=retry_safe,
        provider_payload=error.provider_payload,
    )


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
    delay = max(delay, policy.base_delay)
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

    def _message_for(failure: HandlerFailure) -> str:
        message = str(failure.error).strip()
        return message or failure.error.__class__.__name__

    messages = [_message_for(failure) for failure in failures]

    if not messages:
        return "Reducer errors prevented applying tool result."

    joined = "; ".join(messages)
    return f"Reducer errors prevented applying tool result: {joined}"


def tool_to_spec(
    tool: Tool[SupportsDataclassOrNone, SupportsToolResult],
) -> dict[str, Any]:
    """Return a provider-agnostic tool specification payload."""

    if tool.params_type is type(None):
        parameters_schema = dict(_EMPTY_TOOL_PARAMETERS_SCHEMA)
    else:
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


def _coerce_token_count(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        coerced = int(value)
        return coerced if coerced >= 0 else None
    return None


def token_usage_from_payload(payload: Mapping[str, Any] | None) -> TokenUsage | None:
    """Extract token usage metrics from a provider payload when present."""

    if not isinstance(payload, Mapping):
        return None
    usage_value = payload.get("usage")
    if not isinstance(usage_value, Mapping):
        return None
    usage_payload = cast(Mapping[str, object], usage_value)

    input_tokens = _coerce_token_count(
        usage_payload.get("input_tokens") or usage_payload.get("prompt_tokens")
    )
    output_tokens = _coerce_token_count(
        usage_payload.get("output_tokens") or usage_payload.get("completion_tokens")
    )
    cached_tokens = _coerce_token_count(usage_payload.get("cached_tokens"))

    if all(value is None for value in (input_tokens, output_tokens, cached_tokens)):
        return None

    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
    )


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
        "id": tool_call.id,
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


@FrozenDataclass()
class ToolExecutionOutcome:
    """Result of executing a tool handler."""

    tool: Tool[SupportsDataclassOrNone, SupportsToolResult]
    params: SupportsDataclass | None
    result: ToolResult[SupportsToolResult]
    call_id: str | None
    log: StructuredLogger


def _resolve_tool_and_handler(
    *,
    tool_call: ProviderToolCall,
    tool_registry: Mapping[str, Tool[SupportsDataclassOrNone, SupportsToolResult]],
    prompt_name: str,
    provider_payload: dict[str, Any] | None,
) -> tuple[
    Tool[SupportsDataclassOrNone, SupportsToolResult],
    ToolHandler[SupportsDataclassOrNone, SupportsToolResult],
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
    context: ToolExecutionContext,
    tool_name: str,
    tool_call: ProviderToolCall,
) -> tuple[str | None, StructuredLogger]:
    call_id = tool_call.id
    bound_log = (context.logger_override or logger).bind(
        adapter=context.adapter_name,
        prompt=context.prompt_name,
        tool=tool_name,
        call_id=call_id,
    )
    return call_id, bound_log


def _parse_tool_params(
    *,
    tool: Tool[SupportsDataclassOrNone, SupportsToolResult],
    arguments_mapping: Mapping[str, Any],
) -> SupportsDataclass | None:
    if tool.params_type is type(None):
        if arguments_mapping:
            raise ToolValidationError("Tool does not accept any arguments.")
        return None
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
    handler: ToolHandler[SupportsDataclassOrNone, SupportsToolResult],
    tool_params: SupportsDataclass | None,
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
    context: ToolExecutionContext,
    tool_call: ProviderToolCall,
) -> Iterator[ToolExecutionOutcome]:
    """Context manager that executes a tool call and standardizes logging."""

    tool, handler = _resolve_tool_and_handler(
        tool_call=tool_call,
        tool_registry=context.tool_registry,
        prompt_name=context.prompt_name,
        provider_payload=context.provider_payload,
    )
    tool_name = tool.name
    arguments_mapping = _parse_tool_call_arguments(
        tool_call=tool_call,
        prompt_name=context.prompt_name,
        provider_payload=context.provider_payload,
        parse_arguments=context.parse_arguments,
    )

    call_id, log = _build_tool_logger(
        context=context,
        tool_name=tool_name,
        tool_call=tool_call,
    )

    tool_params: SupportsDataclass | None = None
    tool_result: ToolResult[SupportsToolResult]
    try:
        tool_params = _parse_tool_params(tool=tool, arguments_mapping=arguments_mapping)
        _ensure_deadline_not_expired(
            deadline=context.deadline,
            prompt_name=context.prompt_name,
            tool_name=tool_name,
        )
        tool_context = ToolContext(
            prompt=cast(PromptProtocol[Any], context.prompt),
            rendered_prompt=context.rendered_prompt,
            adapter=cast(ProviderAdapterProtocol[Any], context.adapter),
            session=context.session,
            event_bus=context.bus,
            deadline=context.deadline,
            budget_tracker=context.budget_tracker,
        )
        tool_result = _invoke_tool_handler(
            handler=handler,
            tool_params=tool_params,
            context=tool_context,
        )
    except ToolValidationError as error:
        if tool_params is None:
            tool_params = cast(
                SupportsDataclass,
                _rejected_params(arguments_mapping=arguments_mapping, error=error),
            )
        tool_result = _handle_tool_validation_error(log=log, error=error)
    except VisibilityExpansionRequired:
        # Progressive disclosure: let this propagate to the caller
        raise
    except PromptEvaluationError:
        raise
    except DeadlineExceededError as error:
        raise _handle_tool_deadline_error(
            error=error,
            prompt_name=context.prompt_name,
            tool_name=tool_name,
            deadline=context.deadline,
        ) from error
    except Exception as error:  # propagate message via ToolResult
        tool_result = _handle_unexpected_tool_error(
            log=log,
            tool_name=tool_name,
            provider_payload=context.provider_payload,
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
    context: ToolExecutionContext,
    outcome: ToolExecutionOutcome,
) -> ToolInvoked:
    snapshot = context.session.snapshot()
    session_id = getattr(context.session, "session_id", None)
    tool_value = outcome.result.value
    dataclass_value: SupportsDataclass | None = None
    if is_dataclass_instance(tool_value):
        dataclass_value = cast(SupportsDataclass, tool_value)  # pyright: ignore[reportUnnecessaryCast]

    rendered_output = outcome.result.render()
    usage = token_usage_from_payload(context.provider_payload)

    invocation = ToolInvoked(
        prompt_name=context.prompt_name,
        adapter=context.adapter_name,
        name=outcome.tool.name,
        params=outcome.params,
        result=cast(ToolResult[object], outcome.result),
        session_id=session_id,
        created_at=datetime.now(UTC),
        usage=usage,
        value=dataclass_value,
        rendered_output=rendered_output,
        call_id=outcome.call_id,
        event_id=uuid4(),
    )
    publish_result = context.bus.publish(invocation)
    if not publish_result.ok:
        context.session.mutate().rollback(snapshot)
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
        outcome.result.message = context.format_publish_failures(publish_result.errors)
    else:
        outcome.log.debug(
            "Tool event published.",
            event="tool_event_published",
            context={"handler_count": publish_result.handled_count},
        )
    return invocation


def execute_tool_call(
    *,
    context: ToolExecutionContext,
    tool_call: ProviderToolCall,
) -> tuple[ToolInvoked, ToolResult[SupportsToolResult]]:
    """Execute a provider tool call and publish the resulting event."""

    with tool_execution(
        context=context,
        tool_call=tool_call,
    ) as outcome:
        invocation = _publish_tool_invocation(
            context=context,
            outcome=outcome,
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


def extract_parsed_content(message: ProviderMessage) -> object | None:
    """Extract structured payloads surfaced directly by the provider."""

    if message.parsed is not None:
        return message.parsed

    content = message.content
    if isinstance(content, Sequence) and not isinstance(
        content, (str, bytes, bytearray)
    ):
        for part in content:
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
    options: AdapterRenderOptions,
) -> AdapterRenderContext[OutputT]:
    """Render a prompt and compute adapter inputs shared across providers."""

    prompt_name = prompt.name or prompt.template.__class__.__name__
    render_inputs: tuple[SupportsDataclass, ...] = prompt.params

    if options.deadline is not None and options.deadline.remaining() <= timedelta(0):
        raise PromptEvaluationError(
            "Deadline expired before evaluation started.",
            prompt_name=prompt_name,
            phase=PROMPT_EVALUATION_PHASE_REQUEST,
            provider_payload=deadline_provider_payload(options.deadline),
        )

    rendered = prompt.render(
        visibility_overrides=options.visibility_overrides,
    )
    if options.deadline is not None:
        rendered = replace(rendered, deadline=options.deadline)

    response_format: Mapping[str, Any] | None = None
    if (
        options.enable_json_schema
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
    if any(not isinstance(key, str) for key in mapping):
        return None
    return {cast(str, key): value for key, value in mapping.items()}


__all__ = (  # noqa: RUF022
    "AdapterName",
    "AdapterRenderContext",
    "AdapterRenderOptions",
    "ChoiceSelector",
    "InnerLoop",
    "InnerLoopConfig",
    "InnerLoopInputs",
    "LITELLM_ADAPTER_NAME",
    "OPENAI_ADAPTER_NAME",
    "ProviderCall",
    "ProviderChoice",
    "ProviderCompletionCallable",
    "ProviderCompletionResponse",
    "ProviderFunctionCall",
    "ProviderMessage",
    "ProviderToolCall",
    "ThrottleDetails",
    "ThrottleError",
    "ThrottleKind",
    "ThrottlePolicy",
    "ToolArgumentsParser",
    "ToolChoice",
    "ToolExecutionContext",
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
    "run_inner_loop",
    "serialize_tool_call",
    "throttle_details",
    "token_usage_from_payload",
    "tool_execution",
    "tool_to_spec",
)


ProviderCall = Callable[
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


@FrozenDataclass()
class InnerLoopInputs[OutputT]:
    """Inputs required to start a conversation with a provider."""

    adapter_name: AdapterName
    adapter: ProviderAdapter[OutputT]
    prompt: Prompt[OutputT]
    prompt_name: str
    rendered: RenderedPrompt[OutputT]
    render_inputs: tuple[SupportsDataclass, ...]
    initial_messages: list[dict[str, Any]]


class ToolMessageSerializer(Protocol):
    def __call__(
        self,
        result: ToolResult[SupportsToolResult],
        *,
        payload: object | None = ...,
    ) -> object: ...


@FrozenDataclass()
class InnerLoopConfig:
    """Configuration and collaborators required to run the inner loop."""

    bus: EventBus
    session: SessionProtocol
    tool_choice: ToolChoice
    response_format: Mapping[str, Any] | None
    require_structured_output_text: bool
    call_provider: ProviderCall
    select_choice: ChoiceSelector
    serialize_tool_message_fn: ToolMessageSerializer
    format_publish_failures: Callable[[Sequence[HandlerFailure]], str] = (
        format_publish_failures
    )
    parse_arguments: ToolArgumentsParser = parse_tool_arguments
    logger_override: StructuredLogger | None = None
    deadline: Deadline | None = None
    throttle_policy: ThrottlePolicy = field(default_factory=new_throttle_policy)
    budget_tracker: BudgetTracker | None = None

    def with_defaults(self, rendered: RenderedPrompt[object]) -> InnerLoopConfig:
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
    tool_registry: Mapping[str, Tool[SupportsDataclassOrNone, SupportsToolResult]]
    serialize_tool_message_fn: ToolMessageSerializer
    format_publish_failures: Callable[[Sequence[HandlerFailure]], str]
    parse_arguments: ToolArgumentsParser
    logger_override: StructuredLogger | None = None
    deadline: Deadline | None = None
    budget_tracker: BudgetTracker | None = None
    _log: StructuredLogger = field(init=False)
    _context: ToolExecutionContext = field(init=False)
    _tool_message_records: list[
        tuple[ToolResult[SupportsToolResult], dict[str, Any]]
    ] = field(init=False)

    def __post_init__(self) -> None:
        self._log = (self.logger_override or logger).bind(
            adapter=self.adapter_name,
            prompt=self.prompt_name,
        )
        self._context = ToolExecutionContext(
            adapter_name=self.adapter_name,
            adapter=self.adapter,
            prompt=self.prompt,
            rendered_prompt=self.rendered,
            tool_registry=self.tool_registry,
            bus=self.bus,
            session=self.session,
            prompt_name=self.prompt_name,
            parse_arguments=self.parse_arguments,
            format_publish_failures=self.format_publish_failures,
            deadline=self.deadline,
            logger_override=self.logger_override,
            budget_tracker=self.budget_tracker,
        )
        self._tool_message_records = []

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

        execution_context = self._context.with_provider_payload(provider_payload)

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            if self.deadline is not None and self.deadline.remaining() <= timedelta(0):
                _raise_tool_deadline_error(
                    prompt_name=self.prompt_name,
                    tool_name=tool_name,
                    deadline=self.deadline,
                )
            with tool_execution(
                context=execution_context,
                tool_call=tool_call,
            ) as outcome:
                _ = _publish_tool_invocation(
                    context=execution_context,
                    outcome=outcome,
                )

            tool_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
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
    require_structured_output_text: bool
    _should_parse_structured_output: bool = field(init=False)

    def __post_init__(self) -> None:
        self._should_parse_structured_output = (
            self.rendered.output_type is not None
            and self.rendered.container is not None
        )

    def parse(
        self, message: ProviderMessage, provider_payload: dict[str, Any] | None
    ) -> tuple[OutputT | None, str | None]:
        """Parse the provider message into output and text content."""
        final_text = message_text_content(message.content)
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
            elif final_text or not self.require_structured_output_text:
                try:
                    output = parse_structured_output(final_text or "", self.rendered)
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
class InnerLoop[OutputT]:
    """Coordinate the inner loop of a conversational exchange with a provider.

    This class orchestrates the conversation lifecycle:
    1. Prepare initial messages and tool specifications
    2. Call the provider repeatedly until a final response is produced
    3. Execute tools as requested by the provider
    4. Parse and return structured output when configured

    The loop handles throttling, deadline enforcement, and budget tracking.
    """

    inputs: InnerLoopInputs[OutputT]
    config: InnerLoopConfig
    _evaluation_id: str = field(init=False)
    _log: StructuredLogger = field(init=False)
    _messages: list[dict[str, Any]] = field(init=False)
    _tool_specs: list[dict[str, Any]] = field(init=False)
    _provider_payload: dict[str, Any] | None = field(init=False, default=None)
    _next_tool_choice: ToolChoice = field(init=False)
    _tool_executor: ToolExecutor = field(init=False)
    _response_parser: ResponseParser[OutputT] = field(init=False)
    _rendered: RenderedPrompt[OutputT] = field(init=False)
    _deadline: Deadline | None = field(init=False)

    def __post_init__(self) -> None:
        normalized_config = self.config.with_defaults(self.inputs.rendered)
        self._deadline = normalized_config.deadline
        if self._deadline is not None and (
            self.inputs.rendered.deadline is not self._deadline
        ):
            self._rendered = replace(self.inputs.rendered, deadline=self._deadline)
        else:
            self._rendered = self.inputs.rendered

    def _raise_deadline_error(
        self, message: str, *, phase: PromptEvaluationPhase
    ) -> NoReturn:
        raise PromptEvaluationError(
            message,
            prompt_name=self.inputs.prompt_name,
            phase=phase,
            provider_payload=deadline_provider_payload(self._deadline),
        )

    def _ensure_deadline_remaining(
        self, message: str, *, phase: PromptEvaluationPhase
    ) -> None:
        if self._deadline is None:
            return
        if self._deadline.remaining() <= timedelta(0):
            self._raise_deadline_error(message, phase=phase)

    def _record_and_check_budget(self) -> None:
        """Record cumulative token usage and check budget limits."""
        if self.config.budget_tracker is None:
            return

        usage = token_usage_from_payload(self._provider_payload)
        if usage is not None:
            self.config.budget_tracker.record_cumulative(self._evaluation_id, usage)

        self._check_budget()

    def _check_budget(self) -> None:
        """Check budget limits and raise if exceeded."""
        if self.config.budget_tracker is None:
            return

        try:
            self.config.budget_tracker.check()
        except BudgetExceededError as error:
            raise PromptEvaluationError(
                str(error),
                prompt_name=self.inputs.prompt_name,
                phase=PROMPT_EVALUATION_PHASE_BUDGET,
                provider_payload=self._provider_payload,
            ) from error

    def run(self) -> PromptResponse[OutputT]:
        """Execute the inner loop and return the final response."""

        self._prepare()

        while True:
            response = self._issue_provider_request()

            self._provider_payload = extract_payload(response)
            self._record_and_check_budget()

            choice = self.config.select_choice(response)
            message = choice.message
            if message is None:
                raise PromptEvaluationError(
                    "Provider response did not include a message payload.",
                    prompt_name=self.inputs.prompt_name,
                    phase=PROMPT_EVALUATION_PHASE_RESPONSE,
                    provider_payload=self._provider_payload,
                )

            tool_calls = list(message.tool_calls or [])

            if not tool_calls:
                return self._finalize_response(message)

            self._handle_tool_calls(message, tool_calls)

    def _issue_provider_request(self) -> object:
        attempts = 0
        total_delay = timedelta(0)
        throttle_policy = self.config.throttle_policy

        while True:
            attempts += 1
            self._ensure_deadline_remaining(
                "Deadline expired before provider request.",
                phase=PROMPT_EVALUATION_PHASE_REQUEST,
            )

            try:
                return self.config.call_provider(
                    self._messages,
                    self._tool_specs,
                    self._next_tool_choice if self._tool_specs else None,
                    self.config.response_format,
                )
            except ThrottleError as error:
                attempts = max(error.attempts, attempts)
                if not error.retry_safe:
                    raise

                if attempts >= throttle_policy.max_attempts:
                    raise ThrottleError(
                        "Throttle retry budget exhausted.",
                        prompt_name=self.inputs.prompt_name,
                        phase=PROMPT_EVALUATION_PHASE_REQUEST,
                        details=_details_from_error(
                            error, attempts=attempts, retry_safe=False
                        ),
                    ) from error

                delay = _jittered_backoff(
                    policy=throttle_policy,
                    attempt=attempts,
                    retry_after=error.retry_after,
                )

                if self._deadline is not None and self._deadline.remaining() <= delay:
                    raise ThrottleError(
                        "Deadline expired before retrying after throttling.",
                        prompt_name=self.inputs.prompt_name,
                        phase=PROMPT_EVALUATION_PHASE_REQUEST,
                        details=_details_from_error(
                            error, attempts=attempts, retry_safe=False
                        ),
                    ) from error

                total_delay += delay
                if total_delay > throttle_policy.max_total_delay:
                    raise ThrottleError(
                        "Throttle retry window exceeded configured budget.",
                        prompt_name=self.inputs.prompt_name,
                        phase=PROMPT_EVALUATION_PHASE_REQUEST,
                        details=_details_from_error(
                            error, attempts=attempts, retry_safe=False
                        ),
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

    def _prepare(self) -> None:
        """Initialize execution state prior to the provider loop."""

        self._evaluation_id = str(uuid4())
        self._messages = list(self.inputs.initial_messages)
        self._log = (self.config.logger_override or logger).bind(
            adapter=self.inputs.adapter_name,
            prompt=self.inputs.prompt_name,
            evaluation_id=self._evaluation_id,
        )
        self._log.info(
            "Prompt execution started.",
            event="prompt_execution_started",
            context={
                "tool_count": len(self._rendered.tools),
            },
        )

        tools = list(self._rendered.tools)
        self._tool_specs = [tool_to_spec(tool) for tool in tools]
        tool_registry = {tool.name: tool for tool in tools}
        self._provider_payload = None
        self._next_tool_choice = self.config.tool_choice

        self._tool_executor = ToolExecutor(
            adapter_name=self.inputs.adapter_name,
            adapter=self.inputs.adapter,
            prompt=self.inputs.prompt,
            prompt_name=self.inputs.prompt_name,
            rendered=self._rendered,
            bus=self.config.bus,
            session=self.config.session,
            tool_registry=tool_registry,
            serialize_tool_message_fn=self.config.serialize_tool_message_fn,
            format_publish_failures=self.config.format_publish_failures,
            parse_arguments=self.config.parse_arguments,
            logger_override=self.config.logger_override,
            deadline=self._deadline,
            budget_tracker=self.config.budget_tracker,
        )
        self._response_parser = ResponseParser[OutputT](
            prompt_name=self.inputs.prompt_name,
            rendered=self._rendered,
            require_structured_output_text=self.config.require_structured_output_text,
        )

        self._publish_rendered_event()

    def _publish_rendered_event(self) -> None:
        """Publish the PromptRendered event."""

        publish_result = self.config.bus.publish(
            PromptRendered(
                prompt_ns=self.inputs.prompt.ns,
                prompt_key=self.inputs.prompt.key,
                prompt_name=self.inputs.prompt.name,
                adapter=self.inputs.adapter_name,
                session_id=getattr(self.config.session, "session_id", None),
                render_inputs=self.inputs.render_inputs,
                rendered_prompt=self._rendered.text,
                descriptor=self._rendered.descriptor,
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
        message: ProviderMessage,
        tool_calls: Sequence[ProviderToolCall],
    ) -> None:
        """Execute provider tool calls and record emitted messages."""

        assistant_tool_calls = [serialize_tool_call(call) for call in tool_calls]
        self._messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": assistant_tool_calls,
            }
        )

        tool_messages, next_choice = self._tool_executor.execute(
            tool_calls, self._provider_payload
        )
        self._messages.extend(tool_messages)

        self._check_budget()

        if isinstance(self._next_tool_choice, Mapping):
            tool_choice_mapping = cast(Mapping[str, object], self._next_tool_choice)
            if tool_choice_mapping.get("type") == "function":
                self._next_tool_choice = next_choice

    def _finalize_response(self, message: ProviderMessage) -> PromptResponse[OutputT]:
        """Assemble and publish the final prompt response."""

        self._ensure_deadline_remaining(
            "Deadline expired while finalizing provider response.",
            phase=PROMPT_EVALUATION_PHASE_RESPONSE,
        )

        self._check_budget()

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
            serialized = self.config.serialize_tool_message_fn(
                last_result, payload=output
            )
            if "output" in last_message:
                last_message["output"] = serialized
            else:
                last_message["content"] = serialized

        response_payload = PromptResponse(
            prompt_name=self.inputs.prompt_name,
            text=text_value,
            output=output,
        )

        self._publish_executed_event(response_payload, output)

        return response_payload

    def _publish_executed_event(
        self, response_payload: PromptResponse[OutputT], output: OutputT | None
    ) -> None:
        """Publish the PromptExecuted event."""

        usage = token_usage_from_payload(self._provider_payload)
        prompt_value: SupportsDataclass | None = None
        if is_dataclass_instance(output):
            prompt_value = cast(SupportsDataclass, output)  # pyright: ignore[reportUnnecessaryCast]

        publish_result = self.config.bus.publish(
            PromptExecuted(
                prompt_name=self.inputs.prompt_name,
                adapter=self.inputs.adapter_name,
                result=cast(PromptResponse[object], response_payload),
                session_id=getattr(self.config.session, "session_id", None),
                created_at=datetime.now(UTC),
                usage=usage,
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
                "tool_count": len(self._tool_executor.tool_message_records),
                "has_output": output is not None,
                "text_length": len(response_payload.text or "")
                if response_payload.text
                else 0,
                "structured_output": self._response_parser.should_parse_structured_output,
                "handler_count": publish_result.handled_count,
            },
        )


def run_inner_loop[
    OutputT,
](
    *,
    inputs: InnerLoopInputs[OutputT],
    config: InnerLoopConfig,
) -> PromptResponse[OutputT]:
    """Execute the inner loop of a conversation with a provider.

    This is the primary entry point for running a conversation. It creates
    an InnerLoop instance and executes it.
    """

    loop = InnerLoop[OutputT](inputs=inputs, config=config)
    return loop.run()


@dataclass(slots=True)
class ToolExecutionContext:
    """Inputs and collaborators required to execute a provider tool call."""

    adapter_name: AdapterName
    adapter: ProviderAdapter[Any]
    prompt: Prompt[Any]
    rendered_prompt: RenderedPrompt[Any] | None
    tool_registry: Mapping[str, Tool[SupportsDataclassOrNone, SupportsToolResult]]
    bus: EventBus
    session: SessionProtocol
    prompt_name: str
    parse_arguments: ToolArgumentsParser
    format_publish_failures: Callable[[Sequence[HandlerFailure]], str]
    deadline: Deadline | None
    provider_payload: dict[str, Any] | None = None
    logger_override: StructuredLogger | None = None
    budget_tracker: BudgetTracker | None = None

    def with_provider_payload(
        self, provider_payload: dict[str, Any] | None
    ) -> ToolExecutionContext:
        """Return a copy of the context with a new provider payload."""

        return replace(self, provider_payload=provider_payload)
