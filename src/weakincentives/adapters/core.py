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

from abc import ABC, abstractmethod
from typing import Any, Literal

from ..budget import Budget, BudgetTracker
from ..dataclasses import FrozenDataclass
from ..deadlines import Deadline
from ..errors import WinkError
from ..prompt import Prompt
from ..runtime.session.protocols import SessionProtocol


@FrozenDataclass()
class PromptResponse[OutputT]:
    """Structured result emitted by an adapter evaluation."""

    prompt_name: str
    text: str | None
    output: OutputT | None


class ProviderAdapter(ABC):
    """Abstract base class describing the synchronous adapter contract."""

    @classmethod
    def __class_getitem__(cls, _: object) -> type[ProviderAdapter[Any]]:
        return cls

    @abstractmethod
    def evaluate[OutputT](
        self,
        prompt: Prompt[OutputT],
        *,
        session: SessionProtocol,
        deadline: Deadline | None = None,
        budget: Budget | None = None,
        budget_tracker: BudgetTracker | None = None,
    ) -> PromptResponse[OutputT]:
        """Evaluate the prompt and return a structured response.

        The prompt must be within ``with prompt.resources:`` for resources to be
        available. Resources are accessed via ``prompt.resources.get()``.

        Visibility overrides are managed exclusively via Session state using the
        VisibilityOverrides state slice. Use session[VisibilityOverrides]
        to set visibility overrides before calling evaluate().

        When ``budget`` is provided and ``budget_tracker`` is not, a new tracker
        is created. When ``budget_tracker`` is supplied, it is used directly for
        shared limit enforcement.
        """

        ...


class PromptEvaluationError(WinkError, RuntimeError):
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


PromptEvaluationPhase = Literal["request", "response", "tool", "budget"]
"""Phases where a prompt evaluation error can occur."""

PROMPT_EVALUATION_PHASE_REQUEST: PromptEvaluationPhase = "request"
"""Prompt evaluation failed while issuing the provider request."""

PROMPT_EVALUATION_PHASE_RESPONSE: PromptEvaluationPhase = "response"
"""Prompt evaluation failed while handling the provider response."""

PROMPT_EVALUATION_PHASE_TOOL: PromptEvaluationPhase = "tool"
"""Prompt evaluation failed while handling a tool invocation."""

PROMPT_EVALUATION_PHASE_BUDGET: PromptEvaluationPhase = "budget"
"""Prompt evaluation failed due to budget limits being exceeded."""


__all__ = [
    "PROMPT_EVALUATION_PHASE_BUDGET",
    "PROMPT_EVALUATION_PHASE_REQUEST",
    "PROMPT_EVALUATION_PHASE_RESPONSE",
    "PROMPT_EVALUATION_PHASE_TOOL",
    "Budget",
    "BudgetTracker",
    "PromptEvaluationError",
    "PromptEvaluationPhase",
    "PromptResponse",
    "ProviderAdapter",
]
