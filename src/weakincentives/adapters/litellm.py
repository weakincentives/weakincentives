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

from collections.abc import Callable, Mapping, Sequence
from datetime import timedelta
from http import HTTPStatus
from importlib import import_module
from typing import Any, Final, Protocol, cast, override

from ..budget import Budget, BudgetTracker
from ..deadlines import Deadline
from ..prompt.prompt import Prompt
from ..runtime.logging import StructuredLogger, bind_run_context, get_logger
from ..runtime.run_context import RunContext
from ..runtime.watchdog import Heartbeat
from ..types import LITELLM_ADAPTER_NAME
from ._provider_protocols import (
    ProviderChoice,
    ProviderCompletionCallable,
    ProviderCompletionResponse,
)
from ._tool_messages import serialize_tool_message
from .config import LiteLLMClientConfig, LiteLLMModelConfig
from .core import (
    PROMPT_EVALUATION_PHASE_REQUEST,
    PromptResponse,
    ProviderAdapter,
    SessionProtocol,
)
from .inner_loop import InnerLoopConfig, InnerLoopInputs, run_inner_loop
from .throttle import ThrottleError, ThrottleKind, throttle_details
from .utilities import (
    AdapterRenderOptions,
    ToolChoice,
    call_provider_with_normalization,
    first_choice,
    format_dispatch_failures,
    parse_tool_arguments,
    prepare_adapter_conversation,
)

_ERROR_MESSAGE: Final[str] = (
    "LiteLLM support requires the optional 'litellm' dependency. "
    "Install it with `uv sync --extra litellm` or `pip install weakincentives[litellm]`."
)


class _LiteLLMModule(Protocol):
    def completion(
        self, *args: object, **kwargs: object
    ) -> ProviderCompletionResponse: ...


LiteLLMCompletion = ProviderCompletionCallable


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
    ) -> ProviderCompletionResponse:
        merged: dict[str, object] = dict(kwargs)
        merged.update(request_kwargs)
        return module.completion(*args, **merged)

    return _wrapped_completion


def _coerce_retry_after(value: object) -> timedelta | None:
    if value is None:
        return None
    if isinstance(value, timedelta):
        return value if value > timedelta(0) else None
    if isinstance(value, (int, float)):
        seconds = float(value)
        return timedelta(seconds=seconds) if seconds > 0 else None
    if isinstance(value, str) and value.isdigit():
        return timedelta(seconds=float(value))
    return None


def _retry_after_from_error(error: object) -> timedelta | None:
    direct = _coerce_retry_after(getattr(error, "retry_after", None))
    if direct is not None:
        return direct
    headers = getattr(error, "headers", None)
    if isinstance(headers, Mapping):
        header_mapping = cast(Mapping[str, object], headers)
        retry_after = header_mapping.get("retry-after") or header_mapping.get(
            "Retry-After"
        )
        coerced = _coerce_retry_after(retry_after)
        if coerced is not None:
            return coerced
    response = getattr(error, "response", None)
    if isinstance(response, Mapping):
        response_mapping = cast(Mapping[str, object], response)
        retry_after = response_mapping.get("retry_after")
        coerced = _coerce_retry_after(retry_after)
        if coerced is not None:
            return coerced
        headers = response_mapping.get("headers")
        if isinstance(headers, Mapping):
            header_mapping = cast(Mapping[str, object], headers)
            retry_after = header_mapping.get("retry-after") or header_mapping.get(
                "Retry-After"
            )
            coerced = _coerce_retry_after(retry_after)
            if coerced is not None:
                return coerced
    return None


def _error_payload(error: object) -> dict[str, Any] | None:
    response = getattr(error, "response", None)
    if isinstance(response, Mapping):
        response_mapping = cast(Mapping[object, Any], response)
        return {str(key): value for key, value in response_mapping.items()}
    return None


def _normalize_litellm_throttle(
    error: Exception, *, prompt_name: str
) -> ThrottleError | None:
    message = str(error) or "LiteLLM request failed."
    lower = message.lower()
    status_code = getattr(error, "status_code", None)
    code = getattr(error, "code", None)
    class_name = error.__class__.__name__.lower()
    kind: ThrottleKind | None = None

    logger.debug(
        "litellm.throttle.analyzing",
        event="throttle.analyzing",
        context={
            "prompt_name": prompt_name,
            "error_type": error.__class__.__name__,
            "error_message": message[:500],
            "status_code": status_code,
            "code": code,
        },
    )

    if "insufficient_quota" in lower or code == "insufficient_quota":
        kind = "quota_exhausted"
    elif (
        status_code == HTTPStatus.TOO_MANY_REQUESTS
        or "ratelimit" in class_name
        or code
        in {
            "rate_limit",
            "rate_limit_exceeded",
        }
    ):
        kind = "rate_limit"
    elif "timeout" in class_name:
        kind = "timeout"

    if kind is None:
        logger.debug(
            "litellm.throttle.not_throttle",
            event="throttle.not_throttle",
            context={
                "prompt_name": prompt_name,
                "error_type": error.__class__.__name__,
            },
        )
        return None

    retry_after = _retry_after_from_error(error)
    logger.debug(
        "litellm.throttle.detected",
        event="throttle.detected",
        context={
            "prompt_name": prompt_name,
            "throttle_kind": kind,
            "retry_after_seconds": (
                retry_after.total_seconds() if retry_after else None
            ),
        },
    )

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


logger: StructuredLogger = get_logger(
    __name__, context={"component": "adapter.litellm"}
)


def _prepare_budget_tracking[T](
    *,
    budget: Budget | None,
    budget_tracker: BudgetTracker | None,
    prompt: Prompt[T],
) -> tuple[BudgetTracker | None, Prompt[T]]:
    """Prepare budget tracking and bind tracker to prompt resources."""
    effective_tracker = budget_tracker
    if effective_tracker is None and budget is not None:
        effective_tracker = BudgetTracker(budget=budget)

    if effective_tracker is not None:
        prompt = prompt.bind(resources={BudgetTracker: effective_tracker})

    return effective_tracker, prompt


class LiteLLMAdapter(ProviderAdapter[Any]):
    """Adapter that evaluates prompts via LiteLLM's completion helper.

    Args:
        model: Model identifier in LiteLLM format (e.g., "gpt-4o", "claude-3-sonnet").
        model_config: Typed configuration for model parameters like temperature,
            max_tokens, etc. When provided, these values are merged into each
            request payload.
        tool_choice: Tool selection directive. Defaults to "auto".
        completion: Pre-configured LiteLLM completion callable. Mutually exclusive
            with factory inputs.
        completion_factory: Callable that returns a LiteLLM completion when
            invoked. Useful in tests to inject instrumented completions.
        completion_kwargs: Extra kwargs forwarded to ``completion_factory`` when
            it is used.
        completion_config: Typed configuration for completion instantiation. Used
            when neither ``completion`` nor ``completion_factory`` is provided.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        model: str,
        model_config: LiteLLMModelConfig | None = None,
        tool_choice: ToolChoice = "auto",
        completion: LiteLLMCompletion | None = None,
        completion_factory: Callable[..., LiteLLMCompletion] | None = None,
        completion_kwargs: Mapping[str, object] | None = None,
        completion_config: LiteLLMClientConfig | None = None,
    ) -> None:
        super().__init__()
        # Capture this before completion may be reassigned
        used_explicit_completion = completion is not None
        if completion is not None:
            if completion_factory is not None or completion_kwargs is not None:
                raise ValueError(
                    "completion_factory and completion_kwargs cannot be provided when an explicit completion is supplied.",
                )
            if completion_config is not None:
                raise ValueError(
                    "completion_config cannot be provided when an explicit completion is supplied.",
                )
        elif completion_factory is not None:
            factory_kwargs: dict[str, object] = {}
            if completion_config is not None:
                factory_kwargs.update(completion_config.to_completion_kwargs())
            if completion_kwargs is not None:
                factory_kwargs.update(completion_kwargs)
            completion = completion_factory(**factory_kwargs)
        else:
            completion_params = (
                completion_config.to_completion_kwargs() if completion_config else {}
            )
            completion = create_litellm_completion(**completion_params)

        self._completion = completion
        self._model = model
        self._model_config = model_config
        self._tool_choice: ToolChoice = tool_choice

        logger.debug(
            "litellm.adapter.init",
            event="adapter.init",
            context={
                "model": model,
                "tool_choice": tool_choice,
                "has_model_config": model_config is not None,
                "has_completion_config": completion_config is not None,
                "used_explicit_completion": used_explicit_completion,
                "used_completion_factory": completion_factory is not None,
                "temperature": model_config.temperature if model_config else None,
                "max_tokens": model_config.max_tokens if model_config else None,
            },
        )

    @override
    def evaluate[OutputT](
        self,
        prompt: Prompt[OutputT],
        *,
        session: SessionProtocol,
        deadline: Deadline | None = None,
        budget: Budget | None = None,
        budget_tracker: BudgetTracker | None = None,
        heartbeat: Heartbeat | None = None,
        run_context: RunContext | None = None,
    ) -> PromptResponse[OutputT]:
        prompt_name_for_log = prompt.name or prompt.template.__class__.__name__

        logger.debug(
            "litellm.evaluate.entry",
            event="evaluate.entry",
            context={
                "prompt_name": prompt_name_for_log,
                "has_deadline": deadline is not None,
                "deadline_remaining_seconds": (
                    deadline.remaining().total_seconds() if deadline else None
                ),
                "has_budget": budget is not None,
                "has_budget_tracker": budget_tracker is not None,
                "has_heartbeat": heartbeat is not None,
            },
        )

        render_options = AdapterRenderOptions(
            enable_json_schema=True,
            deadline=deadline,
            session=session,
        )

        render_context = prepare_adapter_conversation(
            prompt=prompt,
            options=render_options,
        )

        prompt_name = render_context.prompt_name
        rendered = render_context.rendered
        response_format = render_context.response_format
        render_inputs = render_context.render_inputs

        logger.debug(
            "litellm.evaluate.setup_complete",
            event="evaluate.setup_complete",
            context={
                "prompt_name": prompt_name,
                "has_response_format": response_format is not None,
                "tool_count": len(rendered.tools),
                "tool_names": [t.name for t in rendered.tools],
            },
        )

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
            if self._model_config is not None:
                request_payload.update(self._model_config.to_request_params())
            if tool_specs:
                request_payload["tools"] = list(tool_specs)
                if tool_choice_directive is not None:
                    request_payload["tool_choice"] = tool_choice_directive
            if response_format_payload is not None:
                request_payload["response_format"] = response_format_payload

            logger.debug(
                "litellm.provider.request",
                event="provider.request",
                context={
                    "prompt_name": prompt_name,
                    "model": self._model,
                    "message_count": len(messages),
                    "tool_count": len(tool_specs) if tool_specs else 0,
                    "tool_names": [
                        spec.get("function", {}).get("name") for spec in tool_specs
                    ]
                    if tool_specs
                    else [],
                    "tool_choice": tool_choice_directive,
                    "has_response_format": response_format_payload is not None,
                },
            )

            def _execute_completion() -> object:
                response = self._completion(**request_payload)
                logger.debug(
                    "litellm.provider.response",
                    event="provider.response",
                    context={
                        "prompt_name": prompt_name,
                        "response_type": type(response).__name__,
                        "has_choices": hasattr(response, "choices"),
                        "has_usage": hasattr(response, "usage"),
                    },
                )
                return response

            return call_provider_with_normalization(
                _execute_completion,
                prompt_name=prompt_name,
                normalize_throttle=lambda error: _normalize_litellm_throttle(
                    error, prompt_name=prompt_name
                ),
                provider_payload=_error_payload,
                request_error_message="LiteLLM request failed.",
            )

        def _select_choice(response: object) -> ProviderChoice:
            return cast(
                ProviderChoice,
                first_choice(response, prompt_name=prompt_name),
            )

        effective_tracker, prompt = _prepare_budget_tracking(
            budget=budget, budget_tracker=budget_tracker, prompt=prompt
        )

        # Enter resource context for lifecycle management
        with prompt.resources:
            config = InnerLoopConfig(
                session=session,
                tool_choice=self._tool_choice,
                response_format=response_format,
                require_structured_output_text=True,
                call_provider=_call_provider,
                select_choice=_select_choice,
                serialize_tool_message_fn=serialize_tool_message,
                format_dispatch_failures=format_dispatch_failures,
                parse_arguments=parse_tool_arguments,
                logger_override=bind_run_context(logger, run_context),
                deadline=deadline,
                budget_tracker=effective_tracker,
                heartbeat=heartbeat,
                run_context=run_context,
            )

            inputs = InnerLoopInputs[OutputT](
                adapter_name=LITELLM_ADAPTER_NAME,
                adapter=cast("ProviderAdapter[OutputT]", self),
                prompt=prompt,
                prompt_name=prompt_name,
                rendered=rendered,
                render_inputs=render_inputs,
                initial_messages=[{"role": "system", "content": rendered.text}],
            )

            return run_inner_loop(inputs=inputs, config=config)


__all__ = [
    "LiteLLMAdapter",
    "LiteLLMClientConfig",
    "LiteLLMCompletion",
    "LiteLLMModelConfig",
    "create_litellm_completion",
]
