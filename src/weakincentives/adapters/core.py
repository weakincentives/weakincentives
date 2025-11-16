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

"""Core adapter interfaces shared across provider integrations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar

from ..deadlines import Deadline
from ..prompt._types import SupportsDataclass
from ..runtime.session.protocols import SessionProtocol

if TYPE_CHECKING:
    from ..prompt.overrides import PromptOverridesStore
    from ..prompt.prompt import Prompt
    from ..runtime.events._types import EventBus, EventPayload, ToolInvoked

OutputT = TypeVar("OutputT")


class ProviderAdapter(Protocol[OutputT]):
    """Protocol describing the synchronous adapter contract."""

    def evaluate(
        self,
        prompt: Prompt[OutputT],
        *params: SupportsDataclass,
        parse_output: bool = True,
        bus: EventBus[EventPayload],
        session: SessionProtocol,
        deadline: Deadline | None = None,
        overrides_store: PromptOverridesStore | None = None,
        overrides_tag: str = "latest",
    ) -> PromptResponse[OutputT]:
        """Evaluate the prompt and return a structured response."""

        ...


@dataclass(slots=True)
class PromptResponse[OutputT]:
    """Structured result emitted by an adapter evaluation."""

    prompt_name: str
    text: str | None
    output: OutputT | None
    tool_results: tuple[ToolInvoked, ...]
    provider_payload: dict[str, Any] | None = None


class PromptEvaluationError(RuntimeError):
    """Raised when evaluation against a provider fails."""

    def __init__(
        self,
        message: str,
        *,
        prompt_name: str,
        phase: PromptEvaluationPhase,
        provider_payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.prompt_name = prompt_name
        self.phase: PromptEvaluationPhase = phase
        self.provider_payload = provider_payload


PromptEvaluationPhase = Literal["request", "response", "tool"]
"""Phases where a prompt evaluation error can occur."""

PROMPT_EVALUATION_PHASE_REQUEST: PromptEvaluationPhase = "request"
"""Prompt evaluation failed while issuing the provider request."""

PROMPT_EVALUATION_PHASE_RESPONSE: PromptEvaluationPhase = "response"
"""Prompt evaluation failed while handling the provider response."""

PROMPT_EVALUATION_PHASE_TOOL: PromptEvaluationPhase = "tool"
"""Prompt evaluation failed while handling a tool invocation."""


__all__ = [
    "PROMPT_EVALUATION_PHASE_REQUEST",
    "PROMPT_EVALUATION_PHASE_RESPONSE",
    "PROMPT_EVALUATION_PHASE_TOOL",
    "PromptEvaluationError",
    "PromptEvaluationPhase",
    "PromptResponse",
    "ProviderAdapter",
    "SessionProtocol",
]
