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

"""Shared utility functions for provider adapters."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import timedelta
from typing import TYPE_CHECKING, Any, NoReturn, Protocol, cast

from ..budget import BudgetTracker
from ..contrib.tools.filesystem import Filesystem
from ..dataclasses import FrozenDataclass
from ..deadlines import Deadline
from ..prompt.prompt import Prompt, RenderedPrompt
from ..prompt.tool import ResourceRegistry
from ..runtime.events import TokenUsage
from ..serde import schema
from ..types.dataclass import (
    SupportsDataclass,
    SupportsDataclassOrNone,
    SupportsToolResult,
)
from ._provider_protocols import ProviderToolCall
from .core import (
    PROMPT_EVALUATION_PHASE_REQUEST,
    PROMPT_EVALUATION_PHASE_RESPONSE,
    PROMPT_EVALUATION_PHASE_TOOL,
    PromptEvaluationError,
    SessionProtocol,
)
from .throttle import ThrottleError

if TYPE_CHECKING:
    from ..prompt.tool import Tool


ToolChoice = str | Mapping[str, Any] | None
"""Supported tool choice directives for provider APIs."""


class ToolArgumentsParser(Protocol):
    def __call__(
        self,
        arguments_json: str | None,
        *,
        prompt_name: str,
        provider_payload: dict[str, Any] | None,
    ) -> dict[str, Any]: ...


_EMPTY_TOOL_PARAMETERS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}


@FrozenDataclass()
class AdapterRenderContext[OutputT]:
    """Rendering inputs and derived metadata for adapter evaluations."""

    prompt_name: str
    render_inputs: tuple[SupportsDataclass, ...]
    rendered: RenderedPrompt[OutputT]
    response_format: Mapping[str, Any] | None


@FrozenDataclass()
class AdapterRenderOptions:
    """Configuration for rendering prompts ahead of provider evaluation.

    Visibility overrides are managed exclusively via Session state using the
    VisibilityOverrides state slice. Use session[VisibilityOverrides]
    to set visibility overrides before rendering.
    """

    enable_json_schema: bool
    deadline: Deadline | None
    session: SessionProtocol | None = None


def build_resources(
    *,
    filesystem: Filesystem | None,
    budget_tracker: BudgetTracker | None,
) -> ResourceRegistry:
    """Build a ResourceRegistry with the given resources.

    Resources are keyed by their protocol type (e.g., Filesystem) rather than
    their concrete type (e.g., InMemoryFilesystem) to enable protocol-based
    lookup in tool handlers.
    """
    entries: dict[type[object], object] = {}
    if filesystem is not None:
        entries[Filesystem] = filesystem
    if budget_tracker is not None:
        entries[BudgetTracker] = budget_tracker
    if not entries:
        return ResourceRegistry()
    return ResourceRegistry.build(entries)


def deadline_provider_payload(deadline: Deadline | None) -> dict[str, Any] | None:
    """Return a provider payload snippet describing the active deadline."""

    if deadline is None:
        return None
    return {"deadline_expires_at": deadline.expires_at.isoformat()}


def raise_tool_deadline_error(
    *, prompt_name: str, tool_name: str, deadline: Deadline
) -> NoReturn:
    """Raise a PromptEvaluationError for a deadline expired during tool execution."""
    raise PromptEvaluationError(
        f"Deadline expired before executing tool '{tool_name}'.",
        prompt_name=prompt_name,
        phase=PROMPT_EVALUATION_PHASE_TOOL,
        provider_payload=deadline_provider_payload(deadline),
    )


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
            mapping_payload = mapping_to_str_dict(cast(Mapping[Any, Any], payload))
            if mapping_payload is not None:
                return mapping_payload
        return None
    if isinstance(response, Mapping):  # pragma: no cover - defensive
        mapping_payload = mapping_to_str_dict(cast(Mapping[Any, Any], response))
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
    from .response_parser import build_json_schema_response_format

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
        session=options.session,
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


def mapping_to_str_dict(mapping: Mapping[Any, Any]) -> dict[str, Any] | None:
    if any(not isinstance(key, str) for key in mapping):
        return None
    return {cast(str, key): value for key, value in mapping.items()}


__all__ = [
    "AdapterRenderContext",
    "AdapterRenderOptions",
    "ToolArgumentsParser",
    "ToolChoice",
    "build_resources",
    "call_provider_with_normalization",
    "deadline_provider_payload",
    "extract_payload",
    "first_choice",
    "parse_tool_arguments",
    "prepare_adapter_conversation",
    "raise_tool_deadline_error",
    "serialize_tool_call",
    "token_usage_from_payload",
    "tool_to_spec",
]
