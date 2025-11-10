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

from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import TYPE_CHECKING, Any, Final, Protocol, cast

from ..prompt._types import SupportsDataclass
from ..prompt.prompt import Prompt
from ..runtime.events import EventBus
from ..runtime.logging import StructuredLogger, get_logger
from . import shared as _shared
from ._provider_protocols import (
    ProviderChoice,
    ProviderCompletionCallable,
    ProviderCompletionResponse,
)
from ._tool_messages import serialize_tool_message
from .core import PromptEvaluationError, PromptResponse, SessionProtocol
from .shared import (
    ToolChoice,
    build_json_schema_response_format,
    first_choice,
    format_publish_failures,
    parse_tool_arguments,
    run_conversation,
)

if TYPE_CHECKING:
    from ..adapters.core import ProviderAdapter

_ERROR_MESSAGE: Final[str] = (
    "LiteLLM support requires the optional 'litellm' dependency. "
    "Install it with `uv sync --extra litellm` or `pip install weakincentives[litellm]`."
)


class _LiteLLMModule(Protocol):
    def completion(
        self, *args: object, **kwargs: object
    ) -> ProviderCompletionResponse: ...


class _LiteLLMCompletionFactory(Protocol):
    def __call__(self, **kwargs: object) -> ProviderCompletionCallable: ...


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


logger: StructuredLogger = get_logger(
    __name__, context={"component": "adapter.litellm"}
)


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
        super().__init__()
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
        session: SessionProtocol,
    ) -> PromptResponse[OutputT]:
        prompt_name = prompt.name or prompt.__class__.__name__
        render_inputs: tuple[SupportsDataclass, ...] = tuple(params)
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
                return self._completion(**request_payload)
            except Exception as error:  # pragma: no cover - network/SDK failure
                raise PromptEvaluationError(
                    "LiteLLM request failed.",
                    prompt_name=prompt_name,
                    phase="request",
                ) from error

        def _select_choice(response: object) -> ProviderChoice:
            return cast(
                ProviderChoice,
                first_choice(response, prompt_name=prompt_name),
            )

        return run_conversation(
            adapter_name="litellm",
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
            require_structured_output_text=True,
            call_provider=_call_provider,
            select_choice=_select_choice,
            serialize_tool_message_fn=serialize_tool_message,
            format_publish_failures=format_publish_failures,
            parse_arguments=parse_tool_arguments,
            logger_override=logger,
        )


__all__ = [
    "LiteLLMAdapter",
    "LiteLLMCompletion",
    "create_litellm_completion",
    "extract_parsed_content",
    "message_text_content",
    "parse_schema_constrained_payload",
]


message_text_content = _shared.message_text_content
extract_parsed_content = _shared.extract_parsed_content
parse_schema_constrained_payload = _shared.parse_schema_constrained_payload
