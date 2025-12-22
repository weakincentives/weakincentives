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


__all__ = [
    "AdapterRenderContext",
    "AdapterRenderOptions",
    "prepare_adapter_conversation",
]
