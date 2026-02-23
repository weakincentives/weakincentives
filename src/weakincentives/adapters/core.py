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
from typing import TYPE_CHECKING, Any

from ..budget import Budget, BudgetTracker
from ..dataclasses import FrozenDataclass
from ..deadlines import Deadline
from ..errors import (
    PROMPT_EVALUATION_PHASE_BUDGET,
    PROMPT_EVALUATION_PHASE_REQUEST,
    PROMPT_EVALUATION_PHASE_RESPONSE,
    PROMPT_EVALUATION_PHASE_TOOL,
    PromptEvaluationError,
    PromptEvaluationPhase,
)
from ..prompt import Prompt
from ..runtime.session.protocols import SessionProtocol

if TYPE_CHECKING:
    from ..runtime.run_context import RunContext
    from ..runtime.watchdog import Heartbeat


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

    @property
    def adapter_name(self) -> str:
        """Canonical name for this adapter instance.

        Default implementation returns the class name.  Concrete adapters
        should override this to return a stable, well-known identifier
        (e.g. ``CLAUDE_AGENT_SDK_ADAPTER_NAME``).
        """
        return type(self).__name__

    @abstractmethod
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
        """Evaluate the prompt and return a structured response.

        The prompt must be within ``with prompt.resource_scope():`` for resources
        to be available. Resources are accessed via ``prompt.resources.get()``.

        Visibility overrides are managed exclusively via Session state using the
        VisibilityOverrides state slice. Use session[VisibilityOverrides]
        to set visibility overrides before calling evaluate().

        When ``budget`` is provided and ``budget_tracker`` is not, a new tracker
        is created. When ``budget_tracker`` is supplied, it is used directly for
        shared limit enforcement.

        When ``heartbeat`` is provided, the adapter will beat at key execution
        points (LLM calls, tool execution boundaries) to prove liveness. Tool
        handlers receive the heartbeat via ToolContext.beat() for additional
        beats during long-running operations.

        When ``run_context`` is provided, it is threaded through telemetry events
        (PromptRendered, PromptExecuted, ToolInvoked) for distributed tracing.
        """

        ...


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
