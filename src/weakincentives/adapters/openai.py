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

from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import timedelta
from importlib import import_module
from typing import TYPE_CHECKING, Any, Final, Mapping, Protocol, cast

from ..deadlines import Deadline
from ..prompt._types import SupportsDataclass
from ..prompt.prompt import Prompt
from ..runtime.events import EventBus
from ..runtime.logging import StructuredLogger, get_logger
from . import shared as _shared
from ._provider_protocols import ProviderChoice, ProviderCompletionResponse
from ._tool_messages import serialize_tool_message
from .core import (
    PROMPT_EVALUATION_PHASE_REQUEST,
    PromptEvaluationError,
    PromptResponse,
    SessionProtocol,
)
from .shared import (
    DEFAULT_THROTTLE_POLICY,
    OPENAI_ADAPTER_NAME,
    ThrottleError,
    ThrottlePolicy,
    ToolChoice,
    build_json_schema_response_format,
    extract_payload,
    deadline_provider_payload,
    first_choice,
    format_publish_failures,
    parse_tool_arguments,
    run_conversation,
)

if TYPE_CHECKING:
    from ..adapters.core import ProviderAdapter
    from ..prompt.overrides import PromptOverridesStore

_ERROR_MESSAGE: Final[str] = (
    "OpenAI support requires the optional 'openai' dependency. "
    "Install it with `uv sync --extra openai` or `pip install weakincentives[openai]`."
)


class _CompletionsAPI(Protocol):
    def create(self, *args: object, **kwargs: object) -> ProviderCompletionResponse: ...


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


def _extract_error_payload(error: Exception) -> Mapping[str, Any] | None:
    response = getattr(error, "response", None)
    if response is None:
        return None
    payload = extract_payload(response)
    if payload is not None:
        return payload
    json_method = getattr(response, "json", None)
    if callable(json_method):
        try:
            json_payload = json_method()
        except Exception:  # pragma: no cover - defensive
            return None
        if isinstance(json_payload, Mapping):
            return cast(Mapping[str, Any], json_payload)
    return None


def _retry_after_seconds(retry_after: object) -> float | None:
    if isinstance(retry_after, (int, float)):
        return float(retry_after)
    if isinstance(retry_after, str):
        try:
            return float(retry_after)
        except ValueError:  # pragma: no cover - defensive
            return None
    if isinstance(retry_after, timedelta):
        return retry_after.total_seconds()
    return None


def _maybe_throttle_error(error: Exception) -> ThrottleError | None:
    name = error.__class__.__name__
    status_code = getattr(error, "status_code", None) or getattr(error, "status", None)
    retry_after_hint = _retry_after_seconds(getattr(error, "retry_after", None))
    payload = _extract_error_payload(error)
    message = str(error)
    if status_code == 429 or name == "RateLimitError":
        return ThrottleError(
            kind="rate_limit",
            retry_after=retry_after_hint,
            provider_payload=payload,
            message=message,
        )
    error_code = getattr(error, "code", None)
    if error_code == "insufficient_quota" or "insufficient_quota" in message:
        return ThrottleError(
            kind="quota_exceeded",
            retry_after=retry_after_hint,
            provider_payload=payload,
            message=message,
        )
    if name == "APITimeoutError":
        return ThrottleError(
            kind="timeout",
            retry_after=retry_after_hint,
            provider_payload=payload,
            message=message,
        )
    return None


logger: StructuredLogger = get_logger(__name__, context={"component": "adapter.openai"})


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
        throttle_policy: ThrottlePolicy | None = None,
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
        self._throttle_policy = throttle_policy or DEFAULT_THROTTLE_POLICY

    def evaluate[OutputT](
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
        prompt_name = prompt.name or prompt.__class__.__name__
        render_inputs: tuple[SupportsDataclass, ...] = tuple(params)

        if deadline is not None and deadline.remaining() <= timedelta(0):
            raise PromptEvaluationError(
                "Deadline expired before evaluation started.",
                prompt_name=prompt_name,
                phase=PROMPT_EVALUATION_PHASE_REQUEST,
                provider_payload=deadline_provider_payload(deadline),
            )

        has_structured_output = prompt.structured_output is not None
        should_disable_instructions = (
            parse_output
            and has_structured_output
            and self._use_native_response_format
            and getattr(prompt, "inject_output_instructions", False)
        )

        if should_disable_instructions:
            rendered = prompt.render(
                *params,
                overrides_store=overrides_store,
                tag=overrides_tag,
                inject_output_instructions=False,
            )
        else:
            rendered = prompt.render(
                *params,
                overrides_store=overrides_store,
                tag=overrides_tag,
            )
        if deadline is not None:
            rendered = replace(rendered, deadline=deadline)
        response_format: dict[str, Any] | None = None
        should_parse_structured_output = (
            parse_output
            and rendered.output_type is not None
            and rendered.container is not None
        )
        if should_parse_structured_output and self._use_native_response_format:
            response_format = build_json_schema_response_format(rendered, prompt_name)

        def _call_provider(
            messages: list[dict[str, Any]],
            tool_specs: Sequence[Mapping[str, Any]],
            tool_choice_directive: ToolChoice | None,
            response_format_payload: Mapping[str, Any] | None,
        ) -> object:
            request_payload: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
            }
            if tool_specs:
                request_payload["tools"] = list(tool_specs)
                if tool_choice_directive is not None:
                    request_payload["tool_choice"] = tool_choice_directive
            if response_format_payload is not None:
                request_payload["response_format"] = response_format_payload

            try:
                return self._client.chat.completions.create(**request_payload)
            except Exception as error:  # pragma: no cover - network/SDK failure
                throttle_error = _maybe_throttle_error(error)
                if throttle_error is not None:
                    raise throttle_error
                raise PromptEvaluationError(
                    "OpenAI request failed.",
                    prompt_name=prompt_name,
                    phase=PROMPT_EVALUATION_PHASE_REQUEST,
                ) from error

        def _select_choice(response: object) -> ProviderChoice:
            return cast(
                ProviderChoice,
                first_choice(response, prompt_name=prompt_name),
            )

        return run_conversation(
            adapter_name=OPENAI_ADAPTER_NAME,
            adapter=cast("ProviderAdapter[OutputT]", self),
            prompt=prompt,
            prompt_name=prompt_name,
            rendered=rendered,
            render_inputs=render_inputs,
            initial_messages=[{"role": "system", "content": rendered.text}],
            parse_output=parse_output,
            bus=bus,
            session=session,
            tool_choice=self._tool_choice,
            response_format=response_format,
            require_structured_output_text=False,
            call_provider=_call_provider,
            select_choice=_select_choice,
            serialize_tool_message_fn=serialize_tool_message,
            format_publish_failures=format_publish_failures,
            parse_arguments=parse_tool_arguments,
            logger_override=logger,
            deadline=deadline,
            throttle_policy=self._throttle_policy,
        )


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
