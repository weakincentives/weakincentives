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

"""Prompt rendering utilities for provider adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import timedelta
from typing import Any

from ..dataclasses import FrozenDataclass
from ..deadlines import Deadline
from ..prompt.prompt import Prompt, RenderedPrompt
from ..types.dataclass import SupportsDataclass
from .core import (
    PROMPT_EVALUATION_PHASE_REQUEST,
    PromptEvaluationError,
    SessionProtocol,
)
from .deadline_utils import deadline_provider_payload


@FrozenDataclass()
class AdapterRenderContext[OutputT]:
    """Rendering inputs and derived metadata for adapter evaluations.

    This immutable dataclass encapsulates all the information needed by provider
    adapters to make API calls after a prompt has been rendered. It separates
    prompt rendering (done once) from provider-specific API formatting (done
    per-provider).

    Attributes:
        prompt_name: Fully-qualified name of the prompt template
            (format: ``namespace.key``) or the template class name if unnamed.
            Used for error reporting and telemetry.
        render_inputs: The original dataclass parameters passed to the prompt,
            preserved for debugging and telemetry.
        rendered: The fully rendered prompt containing the text, tools,
            structured output configuration, and deadline constraints.
        response_format: JSON schema response format for providers that support
            structured output. ``None`` if the prompt has no output type or
            JSON schema is disabled.

    Example:
        >>> ctx = prepare_adapter_conversation(prompt=my_prompt, options=options)
        >>> print(ctx.prompt_name)  # "my_namespace.my_prompt"
        >>> print(ctx.rendered.text)  # The rendered prompt text
        >>> if ctx.response_format:
        ...     # Provider supports JSON schema output
        ...     pass
    """

    prompt_name: str
    render_inputs: tuple[SupportsDataclass, ...]
    rendered: RenderedPrompt[OutputT]
    response_format: Mapping[str, Any] | None


@FrozenDataclass()
class AdapterRenderOptions:
    """Configuration for rendering prompts ahead of provider evaluation.

    Controls how prompts are rendered before being sent to a provider adapter.
    This includes JSON schema generation, deadline enforcement, and session
    state access.

    Attributes:
        enable_json_schema: Whether to generate JSON schema response format
            for prompts with structured output types. Set to ``True`` for
            providers that support structured output (OpenAI, Anthropic).
            Set to ``False`` for providers without schema support or when
            you want text-only responses.
        deadline: Optional deadline constraint for the evaluation. If the
            deadline has already expired when rendering starts, a
            ``PromptEvaluationError`` is raised immediately. The deadline
            is passed through to the rendered prompt for provider-level
            timeout enforcement.
        session: Optional session providing state access during rendering.
            Required for prompts that use session-based visibility overrides
            or read session state in section rendering. Pass ``None`` for
            stateless prompt rendering.

    Note:
        Visibility overrides are managed exclusively via Session state using
        the ``VisibilityOverrides`` state slice. Use
        ``session[VisibilityOverrides]`` to set visibility overrides before
        rendering.

    Example:
        >>> options = AdapterRenderOptions(
        ...     enable_json_schema=True,
        ...     deadline=Deadline(expires_at=datetime.now(UTC) + timedelta(seconds=30)),
        ...     session=my_session,
        ... )
        >>> ctx = prepare_adapter_conversation(prompt=my_prompt, options=options)
    """

    enable_json_schema: bool
    deadline: Deadline | None
    session: SessionProtocol | None = None


def prepare_adapter_conversation[
    OutputT,
](
    *,
    prompt: Prompt[OutputT],
    options: AdapterRenderOptions,
) -> AdapterRenderContext[OutputT]:
    """Render a prompt and compute adapter inputs shared across providers.

    This function performs all provider-agnostic preparation work before a
    prompt is evaluated by a specific adapter. It handles:

    1. Deadline validation - fails fast if the deadline has already expired
    2. Prompt rendering - renders sections with session state and visibility
    3. Response format generation - builds JSON schema if enabled and applicable

    Args:
        prompt: The prompt to render, with parameters already bound. The prompt
            should be within a ``with prompt.resources:`` context if it uses
            resources.
        options: Configuration controlling rendering behavior including JSON
            schema generation, deadline constraints, and session state.

    Returns:
        An ``AdapterRenderContext`` containing the rendered prompt, response
        format, and metadata needed by provider adapters.

    Raises:
        PromptEvaluationError: If the deadline has already expired before
            rendering starts. The error includes the prompt name and a
            ``provider_payload`` with deadline details.

    Example:
        >>> prompt = MyPromptTemplate[OutputType]()(param=my_param)
        >>> options = AdapterRenderOptions(
        ...     enable_json_schema=True,
        ...     deadline=None,
        ...     session=session,
        ... )
        >>> with prompt.resources:
        ...     ctx = prepare_adapter_conversation(prompt=prompt, options=options)
        ...     # Use ctx.rendered, ctx.response_format with provider API
    """
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


__all__ = [
    "AdapterRenderContext",
    "AdapterRenderOptions",
    "prepare_adapter_conversation",
]
